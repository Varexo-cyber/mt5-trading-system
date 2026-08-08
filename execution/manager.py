"""Open-position management and fail-closed restart reconciliation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from advisory.providers import Supervision
from analysis.position_health import PositionHealth, assess_position, drift_score
from config.schema import Settings, TradeManagementConfig
from core.broker import Broker
from core.data_manager import DataManager
from core.data_manager import atr as _atr_of
from core.types import Direction, Position, Timeframe
from filters.news_filter import NewsFilter
from filters.session_filter import SessionFilter
from infra.logging import get_logger
from journal.database import Journal

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ManagementEvent:
    ticket: int
    action: str
    detail: str
    exit_price: float | None = None
    pnl_money: float | None = None
    closed_at: datetime | None = None
    volume_closed: float | None = None
    remaining_volume: float | None = None
    r_at_action: float | None = None


#: How long a computed ATR stays usable, in seconds.
#:
#: The guard calls `manage` about once a second, and an H1 ATR recomputed at
#: that rate is sixty broker round-trips and sixty DataFrames an hour to watch
#: a number move in the fourth decimal. It is an average of fourteen hourly
#: bars: within a minute it cannot have meaningfully changed.
ATR_CACHE_SECONDS = 60.0

#: How long recent bars stay usable, in seconds.
#:
#: Much shorter than the ATR's, because shape is what the health readers work
#: on and a structure break is something we want to see within seconds of it
#: happening. The immediate price reaction never comes from here — that is the
#: tick, which is read fresh every pass.
BAR_CACHE_SECONDS = 3.0

#: Distinguishes "not built yet" from "built, and there is no window".
_UNSET = object()

#: How much a peak must improve to count as a new high, in R.
#:
#: Not zero. `peak_r` is recomputed from a live tick on every pass, so the last
#: decimal flickers constantly; treating a 0.0001R wobble as a new high would
#: reset the stall clock forever and the rule would never fire once.
_PEAK_EPSILON_R = 0.02

#: How much a stop must improve before it is worth moving, in R.
#:
#: A stop move is not free. It is a round-trip to the broker, and — because
#: every stop rule here ends in `continue` — it also costs the position its
#: turn at every rule below it. Break-even is recomputed from a live ATR, so
#: without a floor a fractionally rising ATR nudges the target up and re-fires
#: the rule on every pass, while the profit lock and the partial close sitting
#: beneath it never once get a look at the trade.
#:
#: Found by replaying the real rules over bar history rather than by reading
#: them: BREAK_EVEN fired on seven consecutive bars of a trade parked at 0.88R,
#: and the profit lock two rules below it never ran.
_STOP_IMPROVEMENT_R = 0.02


def _risk_money(row) -> float:  # type: ignore[no-untyped-def]
    """The money behind 1R on this trade, or 0.0 when the row cannot say.

    `trades.risk_money` is NOT NULL and written at open, so in production this
    always answers. It is read defensively anyway because it runs inside the
    guard loop: a recovered or hand-repaired row missing the column must cost
    the banking rule its money cap, not take down the loop that is watching
    every open position. Zero reads downstream as "no cap", which is exactly
    the behaviour this had before the cap existed.
    """
    try:
        return float(row["risk_money"] or 0.0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


class PositionManager:
    def __init__(
        self,
        broker: Broker,
        journal: Journal,
        settings: Settings,
        brain: object | None = None,
    ) -> None:
        self.broker = broker
        self.journal = journal
        self.settings = settings
        #: Optional, and only ever consulted to bank SOONER — see
        #: `_worth_taking`. None keeps every threshold exactly as configured.
        self.brain = brain
        self._learned_bank: object = _UNSET
        # symbol+period -> (monotonic deadline, value). Deliberately monotonic
        # rather than wall-clock: a simulated or rewound clock must not be able
        # to hold a stale ATR alive forever.
        self._atr_cache: dict[tuple[str, int], tuple[float, float]] = {}
        self._bar_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame | None]] = {}
        #: The most recent health reading per ticket. Read by the supervisor, so
        #: Claude sees what the fast layer has been watching rather than only a
        #: snapshot of the moment it happened to be asked, and by the deck.
        self.last_health: dict[int, PositionHealth] = {}
        self._session_filter: object = _UNSET
        #: Account equity, refreshed by the runner once a cycle. Zero switches
        #: the banking rule off, which is the safe default: a threshold that is
        #: a share of an equity nobody set would be a share of nothing.
        #:
        #: Not read from the broker on every guard pass — that is a round trip
        #: a second to watch a number that moves in cents, and the rule only
        #: needs it to the nearest euro.
        self.equity: float = 0.0
        #: ticket -> (highest R seen, when it was first reached). In memory on
        #: purpose: a restart resets the clock, which errs toward holding.
        self._peak_seen: dict[int, tuple[float, datetime]] = {}

    def reconcile(self, positions: list[Position]) -> list[ManagementEvent]:
        events: list[ManagementEvent] = []
        broker_tickets = {position.ticket for position in positions}
        for position in positions:
            if not position.has_stop:
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "EMERGENCY_CLOSE",
                        "position had no stop",
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
                continue
            if self.journal.open_trade_by_ticket(position.ticket) is None:
                adopted = self._adopt(position)
                if adopted is not None:
                    events.append(adopted)
                    continue
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "ORPHAN_CLOSE",
                        "strategy-magic position was absent from journal and matched no "
                        "recorded entry intent",
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
                continue
            row = self.journal.open_trade_by_ticket(position.ticket)
            assert row is not None
            journal_volume = float(row["volume"])
            tolerance = self.broker.spec(position.symbol).volume_step / 2
            partial_actions = ("PARTIAL_CLOSE", "PARTIAL_CLOSE_RECOVERED")
            if (
                position.volume < journal_volume - tolerance
                and not self.journal.management_action_exists(position.ticket, partial_actions)
            ):
                closed = self.broker.closed_position(position.ticket)
                if closed is None:
                    events.append(
                        ManagementEvent(
                            position.ticket,
                            "BROKER_CLOSED_PENDING_HISTORY",
                            "position volume shrank but no partial-close deal is available",
                        )
                    )
                else:
                    events.append(
                        ManagementEvent(
                            position.ticket,
                            "PARTIAL_CLOSE_RECOVERED",
                            f"recovered broker partial from deals {closed.deal_tickets}",
                            volume_closed=journal_volume - position.volume,
                            remaining_volume=position.volume,
                        )
                    )
        for row in self.journal.open_trades():
            ticket = int(row["ticket"] or 0)
            if ticket and ticket not in broker_tickets:
                closed = self.broker.closed_position(ticket)
                if closed is None:
                    events.append(
                        ManagementEvent(
                            ticket,
                            "BROKER_CLOSED_PENDING_HISTORY",
                            "journal trade absent at broker and no final deal is available; "
                            "block new risk",
                        )
                    )
                else:
                    events.append(
                        ManagementEvent(
                            ticket,
                            f"BROKER_{closed.reason}",
                            "recovered exact broker closure from deals "
                            + ",".join(str(item) for item in closed.deal_tickets),
                            closed.exit_price,
                            closed.pnl_money,
                            closed.closed_at,
                        )
                    )
        return events

    def manage(
        self,
        positions: list[Position],
        now: datetime,
        patience: float = 1.0,
    ) -> list[ManagementEvent]:
        """Apply the mechanical rules to every open position.

        `patience` scales only the stalled-trade timeout, and only downward —
        see `risk.posture`. After a run of losses a trade that is going nowhere
        gets less rope, because what recovers a drawdown is having the slot and
        the capital free for the next good setup, not sitting in a dead trade.
        Clamped here as well as at the source: a value above 1.0 would grant
        *extra* patience in a drawdown, which is the exact inversion this is
        meant to prevent.
        """
        patience = min(1.0, max(0.1, patience))
        events: list[ManagementEvent] = []
        config = self.settings.trade_management
        for position in positions:
            row = self.journal.open_trade_by_ticket(position.ticket)
            if row is None:
                self._unmanaged(
                    position,
                    "no open trade on record for this ticket, so there is no entry "
                    "intent to measure R against; the broker stop is the only thing "
                    "holding it",
                )
                continue
            original_sl = float(row["sl"])
            risk = abs(position.price_open - original_sl)
            if risk <= 0:
                self._unmanaged(
                    position,
                    f"journal records the stop at {original_sl:.5f}, the same as entry, "
                    f"so 1R is zero and every rule here divides by it",
                )
                continue
            tick = self.broker.tick(position.symbol)
            price = tick.bid if position.direction is Direction.LONG else tick.ask
            r_now = (price - position.price_open) * int(position.direction) / risk
            peak_r = self._record_excursion(int(row["id"]), r_now, float(row["mfe_r"] or 0.0))
            # Before anything else, because everything else assumes we intend to
            # still be in the trade. Nothing that happens after 20:15 UTC is
            # worth the spread it costs to be there.
            wind_down = self._evening_flatten(position, now, r_now)
            if wind_down is not None:
                events.append(wind_down)
                continue
            # Before the hour-before-the-close rule and before every other
            # profit rule, because it is the only one that acts while the money
            # is plainly on the table rather than after something has started to
            # go wrong.
            banked = self._bank_worthwhile_profit(position, r_now, risk, _risk_money(row))
            if banked is not None:
                events.append(banked)
                continue
            # Then the hour before that deadline. Placed above every other
            # profit rule because none of them know the session is ending: the
            # peak stall waits six minutes for a peak that will not come, and
            # the give-back waits for a drain the closing spread supplies for
            # free.
            decaying = self._session_decay_exit(position, now, r_now, risk, tick)
            if decaying is not None:
                events.append(decaying)
                continue
            # Next, because it is about to happen to us rather than about what
            # the trade is worth: a spread wide enough to trigger the stop on
            # its own.
            squeezed = self._spread_squeeze_exit(position, tick, r_now)
            if squeezed is not None:
                events.append(squeezed)
                continue
            age_hours = (now - position.opened_at).total_seconds() / 3600.0
            # How the trade is actually behaving, not just what R it is at.
            # Read before the give-back rule, because that rule now asks it
            # whether the move is still working rather than acting on a number
            # alone — see `_giveback_exit`.
            health = self._read_health(position, r_now, age_hours * 60.0, risk, tick)
            self.last_health[position.ticket] = health
            # Asked first, because it is the only exit here that can act while
            # the money is still on the table. The two rules cannot both apply:
            # this one needs price near the peak, the give-back needs it far
            # from the peak.
            stalled = self._peak_stall_exit(position, r_now, peak_r, now)
            if stalled is not None:
                events.append(stalled)
                continue
            giveback = self._giveback_exit(position, r_now, peak_r, health)
            if giveback is not None:
                events.append(giveback)
                continue
            reacted = self._act_on_health(position, health, r_now, risk, tick)
            if reacted is not None:
                events.append(reacted)
                if reacted.exit_price is not None:
                    continue
            deadline = (
                config.time_exit_hours * patience if config.time_exit_hours is not None else None
            )
            expired = self._time_exit_verdict(config, age_hours, deadline, r_now, peak_r)
            if expired is not None:
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "TIME_EXIT",
                        f"{age_hours:.1f}h, {r_now:.2f}R (peak {peak_r:.2f}R) — {expired}"
                        + (f"; drawdown posture: {deadline:.1f}h limit" if patience < 1.0 else ""),
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
                continue
            atr = self._atr(position.symbol)
            break_even = position.price_open + atr * config.break_even_offset_atr * int(
                position.direction
            )
            if r_now >= config.break_even_at_r and self._worth_moving(position, break_even, risk):
                result = self.broker.modify_stops(
                    position,
                    sl=self.broker.spec(position.symbol).normalize_price(break_even),
                    tp=position.tp,
                )
                if result.ok:
                    events.append(
                        ManagementEvent(
                            position.ticket, "BREAK_EVEN", f"stop protected at {r_now:.2f}R"
                        )
                    )
                    continue
            partial_actions = ("PARTIAL_CLOSE", "PARTIAL_CLOSE_RECOVERED")
            if r_now >= config.partial_close_at_r and not self.journal.management_action_exists(
                position.ticket, partial_actions
            ):
                spec = self.broker.spec(position.symbol)
                close_volume = spec.round_volume_down(
                    position.volume * config.partial_close_fraction
                )
                remaining = spec.round_volume_down(position.volume - close_volume)
                if close_volume >= spec.volume_min and remaining >= spec.volume_min:
                    result = self.broker.close_position(position, close_volume)
                    if result.ok:
                        events.append(
                            ManagementEvent(
                                position.ticket,
                                "PARTIAL_CLOSE",
                                f"closed {result.filled_volume:g} lots at {r_now:.2f}R",
                                exit_price=result.filled_price,
                                volume_closed=result.filled_volume,
                                remaining_volume=remaining,
                                r_at_action=r_now,
                            )
                        )
                        continue

            # Walk the stop up behind a trade that has proved itself, instead
            # of leaving it parked at break-even all the way to 1.5R. Placed
            # after the partial close deliberately: banking real money outranks
            # adjusting a stop, and the guard comes round again in a second.
            locked = self._profit_lock(position, r_now, peak_r, risk)
            if locked is not None:
                events.append(locked)
                continue

            if config.trailing_mode == "atr" and r_now >= config.partial_close_at_r:
                trailing = price - atr * config.trailing_atr_multiple * int(position.direction)
                if self._worth_moving(position, trailing, risk):
                    result = self.broker.modify_stops(
                        position,
                        sl=self.broker.spec(position.symbol).normalize_price(trailing),
                        tp=position.tp,
                    )
                    if result.ok:
                        events.append(
                            ManagementEvent(
                                position.ticket, "ATR_TRAIL", f"stop trailed at {r_now:.2f}R"
                            )
                        )
        return events

    def _bars(self, symbol: str, timeframe: Timeframe, count: int) -> pd.DataFrame | None:
        """Recent bars, cached briefly so a per-second pass stays cheap.

        The TTL is short rather than absent because the forming bar does move
        tick by tick — but the immediate price reaction comes from the tick,
        which is never cached. What is cached is shape, and shape does not
        change meaningfully inside a few seconds on any of these timeframes.
        """
        key = (symbol, timeframe.value, count)
        cached = self._bar_cache.get(key)
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1]
        frame: pd.DataFrame | None
        try:
            # The shared builder, not a local `pd.DataFrame(...)`: the structure
            # reader needs the time index, and a second almost-right conversion
            # here is how the two would drift apart.
            raw = self.broker.copy_rates(symbol, timeframe.mt5_value, count)
            frame = DataManager._to_frame(raw)
        except Exception:  # noqa: BLE001 - no bars is a reason to stay quiet, not to crash
            frame = None
        if frame is not None and frame.empty:
            frame = None
        self._bar_cache[key] = (now + BAR_CACHE_SECONDS, frame)
        return frame

    def _read_health(
        self, position: Position, r_now: float, age_minutes: float, risk: float, tick
    ) -> PositionHealth:  # type: ignore[no-untyped-def]
        config = self.settings.trade_management
        if not config.health_enabled:
            return PositionHealth("healthy", 0.0, "hold", (), "health reading disabled")
        return assess_position(
            sign=int(position.direction),
            r_now=r_now,
            age_minutes=age_minutes,
            fast=self._bars(position.symbol, Timeframe.M1, config.health_fast_bars),
            structure=self._bars(position.symbol, Timeframe.M5, config.health_structure_bars),
            spread=getattr(tick, "spread", 0.0),
            risk=risk,
            secure_at_r=config.health_secure_at_r,
            tighten_at_r=config.health_tighten_at_r,
            # So the drift readers can tell how much of their window is from
            # after this trade opened, rather than from the chart that produced
            # it. Taken from the timeframe rather than hardcoded: the two must
            # not be able to disagree.
            fast_bar_minutes=Timeframe.M1.duration.total_seconds() / 60.0,
        )

    def _act_on_health(
        self, position: Position, health: PositionHealth, r_now: float, risk: float, tick
    ) -> ManagementEvent | None:  # type: ignore[no-untyped-def]
        """Carry out the one thing the reading permits, and nothing more.

        `secure` and `exit` both close; they are kept apart because the journal
        reason is the only record of *why*, and "banked a profit as it turned"
        and "cut it before it got worse" are different lessons to learn from.

        Neither closes when the stop is nearer than the exit costs — see
        `_worth_paying_to_leave`. A reading is a reason to want out; it is not
        an argument that leaving is affordable, and on this account those come
        apart often.
        """
        if health.action == "hold":
            return None
        if health.action in ("secure", "exit") and not self._worth_paying_to_leave(
            position, risk, tick
        ):
            return None
        if health.action in ("secure", "exit"):
            result = self.broker.close_position(position)
            if not result.ok:
                return None
            return ManagementEvent(
                position.ticket,
                "HEALTH_SECURE" if health.action == "secure" else "HEALTH_EXIT",
                f"{health.reason} at {r_now:.2f}R",
                result.filled_price,
                position.profit + position.swap,
                r_at_action=r_now,
            )
        # tighten: pull the stop to just inside what the trade is currently
        # worth. Never past price, and never a widening — a reading this weak
        # has not earned the right to close anything, only to risk less.
        locked = position.price_open + risk * r_now * 0.5 * int(position.direction)
        improves = (position.direction is Direction.LONG and locked > position.sl) or (
            position.direction is Direction.SHORT and locked < position.sl
        )
        if not improves:
            return None
        spec = self.broker.spec(position.symbol)
        result = self.broker.modify_stops(position, sl=spec.normalize_price(locked), tp=position.tp)
        if not result.ok:
            return None
        return ManagementEvent(
            position.ticket,
            "HEALTH_TIGHTEN",
            f"{health.reason} at {r_now:.2f}R",
            r_at_action=r_now,
        )

    def _evening_flatten(
        self, position: Position, now: datetime, r_now: float
    ) -> ManagementEvent | None:
        """Go flat before the evening spread arrives.

        The rollover block stopped us *opening* into the worst half hour and
        said nothing about what was already on. So a position entered in a
        1-pip market was carried into a 6-pip one and charged the difference on
        the way out — the stop taken by the spread rather than by the market.
        On a small account with tight stops that is not a tail risk; it is what
        happened every evening.

        Deliberately unconditional on P&L. "Let this one run a bit longer, it
        is almost at target" is the reasoning that produces the loss, because
        the widening spread moves the market away from the target and toward
        the stop at the same time. Being flat before it widens is the point.

        Continuously traded markets are exempt: crypto has no FX rollover and
        no reason to be closed at 22:15 Amsterdam time.
        """
        asset_class = self.broker.spec(position.symbol).asset_class.value
        if asset_class in self.settings.filters.session.continuous_asset_classes:
            return None
        # Per asset class, because an index does not follow the FX rollover: its
        # cash session closes at 20:00 UTC and the quote widens from there, so
        # the forex wind-down at 20:15 leaves it in the widest spread of the day
        # for a final quarter of an hour.
        window = self._session_windows(asset_class)
        # `should_be_flat`, not `window.contains`. The window only knows about
        # the evening; the Friday cut-off is a deadline too, and asking the
        # window alone left positions open for the seventy-five minutes between
        # the Friday gate closing and the evening window opening.
        if window is None or not self._should_be_flat(now, asset_class):
            return None
        result = self.broker.close_position(position)
        if not result.ok:
            return None
        friday = now.weekday() == 4 and not window.contains(now)
        why = (
            "Friday cut-off; nothing opens after this and nothing should still be on"
            if friday
            else f"evening wind-down ({window.describe()} UTC)"
        )
        return ManagementEvent(
            position.ticket,
            "EVENING_FLAT",
            f"{why} at {r_now:.2f}R; spreads widen from here",
            result.filled_price,
            position.profit + position.swap,
            r_at_action=r_now,
        )

    def _bank_worthwhile_profit(
        self, position: Position, r_now: float, risk: float, risk_money: float
    ) -> ManagementEvent | None:
        """Take a sum worth taking, unless the move is clearly still running.

        Every other rule in this file holds by default and acts on evidence of
        trouble. This one is the other way round on purpose, and the operator
        put it plainly: on a hundred-euro account, sixty or eighty cents is a
        fine amount to bank, and a profit you can see beats a bigger one you
        are hoping for. So the default is to take it, and only a market still
        visibly running our way earns the right to keep going.

        The threshold is a share of equity rather than an amount, because "80
        cents on 100 euro" and "10 euro on 1000" are one rule said twice. It
        scales on its own after a deposit, with nobody editing anything.

        A live USDCHF long is the case: entered 0.81009, peaked 7.1 pips up —
        about 76 cents — never reached a target 14.5 pips away, and closed at
        +2.7 pips for 29 cents. It kept 38% of its best moment while a rule
        that simply took the 76 cents was sitting there unwritten.
        """
        config = self.settings.trade_management
        if not config.bank_enabled or self.equity <= 0 or risk <= 0:
            return None
        profit = position.profit + position.swap
        worth_taking = self._worth_taking(risk_money)
        if profit < worth_taking or profit <= 0:
            return None

        # THE SIGN HERE IS MEASURED, AND IT IS THE OPPOSITE OF WHAT IT WAS.
        #
        # This read `running >= threshold: return None` — hold while the move
        # is still going our way. It is the intuitive rule and it is wrong.
        # `backtest.cmd --exits --days 90` walked every position the theories
        # would have opened and reported what an extra minute of patience was
        # worth at each moment, split by pace:
        #
        #     in profit    running   stalled   against
        #     0.00-0.15R    -0.092    -0.019    +0.036
        #     0.15-0.30R    -0.096    -0.041    +0.023
        #     0.30-0.50R    -0.192    -0.137    -0.010
        #     0.50-0.75R    -0.136    -0.161    +0.078
        #     0.75-1.00R    -0.149    -0.146    +0.039
        #     1.00-1.50R    -0.133    -0.103    +0.121
        #     1.50+         -0.086    +0.137    +0.242
        #
        # Running is negative at every one of the seven levels, on 2,300 to
        # 6,300 observations each. A hard run is exhaustion, not confirmation,
        # and the old rule held on precisely when holding cost the most. Where
        # patience pays is when price is coming back against the position —
        # which is the reading that used to trigger an immediate exit.
        #
        # So the condition inverts: bank by default, and hold only while the
        # move is retracing.
        retracing = self._pace(position)
        if retracing is not None and retracing <= -config.bank_while_retracing_drift:
            return None

        result = self.broker.close_position(position)
        if not result.ok:
            return None
        pace = "the move is not retracing" if retracing is not None else "no read on the pace"
        return ManagementEvent(
            position.ticket,
            "PROFIT_BANKED",
            f"{profit:.2f} is {profit / self.equity * 100:.2f}% of the account at "
            f"{r_now:.2f}R and {pace}; a profit you can see beats one you are hoping for",
            result.filled_price,
            profit,
            r_at_action=r_now,
        )

    def _worth_taking(self, risk_money: float) -> float:
        """The sum this account considers worth banking, in account currency.

        Two ways of saying the same sentence, and the trade gets the smaller.

        The first is the operator's: on a hundred-euro account, sixty to ninety
        cents is a fine amount to take. That is `bank_at_equity_pct`, and it is
        the right shape because it rescales itself after a deposit.

        The second exists because the first one quietly broke. Its docstring
        claimed "at 2% risk per trade, 0.6% of equity is 0.3R" — true only if
        the account actually risks 2%. It does not. The sizer rounds the lot
        *down* to the broker's 0.01 step (`round_volume_down`, and rounding up
        is forbidden), so what is really at risk is whatever fits under the
        intention. Across the last twenty live trades the risk taken ran from
        EUR 1.00 to EUR 2.94 on a EUR 151 account: 0.66% to 1.95% against a
        2.0% intent.

        Divide 0.6% of equity by that and the threshold lands between 0.31R and
        0.91R depending on nothing but how the lot happened to round. On the
        AUDCAD long, 1R was EUR 1.00 and the rule wanted EUR 0.91 — it was not
        a banking rule on that trade, it was "hold to nearly the full target".
        It fired zero times in those twenty trades while ten of them kept under
        half of their best moment.

        So the threshold is also expressed against the risk the trade actually
        carries, which is the denominator of every other R in this system, and
        the lower of the two wins. `bank_at_r` is not a new invented number: it
        is the 0.3R the code already claimed to be doing.

        AND A THIRD, LEARNED FROM THE ACCOUNT'S OWN TRADES. `Brain.
        learned_bank_threshold` reads every closed trade in Postgres, compares
        the best each one reached against what it actually took home, and
        returns the lowest peak level where banking would have beaten holding.
        It needs forty closed trades before it says anything and returns None
        when the database is unreachable, so a system without it behaves
        exactly as it did before.

        THE MINIMUM IS WHAT MAKES THIS SAFE. The brain is otherwise forbidden
        from moving any threshold, because a learning system that can rewrite
        its own risk controls is how an account dies. This is the one exception
        and it is bounded by direction: taking the minimum means the learned
        value can only ever bank SOONER. Closing a position earlier is less
        exposure, never more. The worst case is money left on the table; the
        worst case of letting it raise the threshold would be the account.
        """
        config = self.settings.trade_management
        worth_taking = self.equity * config.bank_at_equity_pct / 100.0
        if risk_money > 0:
            worth_taking = min(worth_taking, risk_money * config.bank_at_r)
            learned = self._learned_bank_r()
            if learned is not None:
                worth_taking = min(worth_taking, risk_money * learned)
        return worth_taking

    def _learned_bank_r(self) -> float | None:
        """The account's own answer, cached for the life of the process.

        Cached because the guard runs once a second on every open position and
        this is a GROUP BY over the whole trade history. Recomputing it at that
        rate would be a query a second to watch a number that moves when a
        trade closes — and the cache clearing on restart is the right cadence
        anyway: a threshold that shifted mid-session would mean two positions
        opened minutes apart were managed by different rules.
        """
        if self._learned_bank is _UNSET:
            # `Brain` already swallows its own failures, and this does not rely
            # on that. A null object, a stub, or a future implementation could
            # all throw, and this runs inside the loop watching real money —
            # where the safe answer to any surprise is the configured value.
            try:
                self._learned_bank = self.brain.learned_bank_threshold() if self.brain else None
            except Exception:  # noqa: BLE001 - no opinion is not a crash
                log.warning(
                    "could not read the learned banking threshold; using the configured one",
                    extra={"event": "bank_threshold_unavailable"},
                )
                self._learned_bank = None
            if self._learned_bank is not None:
                log.info(
                    "banking threshold learned from this account's own trades",
                    extra={"event": "bank_threshold_learned", "take_at_r": self._learned_bank},
                )
        return self._learned_bank  # type: ignore[return-value]

    def _pace(self, position: Position) -> float | None:
        """How hard the market is moving, signed for this position.

        Positive is running our way, negative is coming back against us, None
        is no read. Named for what it returns rather than for what one caller
        does with it: it used to be `_still_running`, and that name quietly
        argued for the interpretation that turned out to be backwards.

        The same `drift_score` the health reader uses. Two separately derived
        answers to "is this still moving" would eventually disagree, and they
        would disagree while a position was open with money on it.

        A NOTE ON THE OTHER READER. `momentum_turned` treats a strongly
        negative reading as a reason to exit, which is the opposite of what
        the banking rule now does with it. That is not necessarily a
        contradiction — the health reader also judges losing positions, and
        the exit study only measured moments in profit — but it has not been
        measured, and it is the obvious next thing to point the study at.
        """
        frame = self._bars(
            position.symbol, Timeframe.M1, self.settings.trade_management.health_fast_bars
        )
        if frame is None:
            return None
        try:
            return drift_score(frame, int(position.direction))
        except Exception:  # noqa: BLE001 - no read is not a crash, see _minutes_to_target
            return None

    def _session_decay_exit(
        self, position: Position, now: datetime, r_now: float, risk: float, tick
    ) -> ManagementEvent | None:  # type: ignore[no-untyped-def]
        """Bank a profit whose target the session can no longer deliver.

        `_evening_flatten` closes everything at the wind-down whatever it is
        worth. That is the backstop. This is the judgement before it, and it is
        the one a person makes without calling it a rule: there are forty
        minutes of session left, the target is three hours away at the pace
        this market is actually moving, and you are up. The rest is not coming.
        Take the money.

        Deliberately *not* "close if R exceeds some number". A threshold like
        that is the behaviour this system keeps being criticised for and the
        criticism is right: it does not know how far the target is, how fast
        the market is going, or how much of the day is left, so it fires on a
        trade two pips from a target it will reach and holds one that has no
        chance. Both questions here are measured:

        * **Is the rest reachable?** The same estimate the entry gate uses, so
          a target ruled reachable at 14:00 and unreachable at 19:40 has moved
          for a reason rather than because two formulas disagree. Displacement
          over n bars grows with `sqrt(n) * ATR`, not with n, so covering d
          takes `(d / ATR)^2` bars — which is why a target that looks forty
          minutes away on a straight line is really three hours away.

        * **Is what is left worth collecting?** The floor is the measured cost
          of leaving — the spread you cross plus the commission on the exit —
          and not an invented R. Below that, banking hands the broker the
          profit and calls it discipline.

        Nothing about a widening spread here. `_spread_squeeze_exit` already
        owns that, and two rules watching one symptom is how they end up
        disagreeing.
        """
        if not self.settings.trade_management.session_decay_enabled or r_now <= 0:
            return None
        asset_class = self.broker.spec(position.symbol).asset_class.value
        runway = self._runway_minutes(now, asset_class)
        if runway is None:
            return None

        needed = self._minutes_to_target(position, tick)
        if needed is None or needed <= runway:
            return None

        cost_r = self._cost_of_leaving(position, risk, tick)
        if r_now <= cost_r:
            return None

        result = self.broker.close_position(position)
        if not result.ok:
            return None
        return ManagementEvent(
            position.ticket,
            "SESSION_DECAY",
            f"target is ~{needed:.0f} min away at the pace this market is actually moving "
            f"and {runway:.0f} min of session remain; {r_now:.2f}R on the table against "
            f"{cost_r:.2f}R to collect it",
            result.filled_price,
            position.profit + position.swap,
            r_at_action=r_now,
        )

    def _minutes_to_target(self, position: Position, tick) -> float | None:  # type: ignore[no-untyped-def]
        """How long the *remaining* distance would take at this market's pace.

        Measured from where price is now, not from the entry: half a target
        already covered is half a target that does not need covering again.
        """
        config = self.settings.filters.runway
        if position.tp == 0.0:
            return None
        try:
            timeframe = Timeframe.parse(config.speed_timeframe)
        except ValueError:
            return None
        frame = self._bars(position.symbol, timeframe, 60)
        if frame is None:
            return None
        try:
            speed = _atr_of(frame)
        except Exception:  # noqa: BLE001 - too little history is no estimate, not a crash
            # `atr` raises on a short frame, and this runs inside the guard loop
            # every second on every open position. No estimate must mean "do not
            # act", never "take the whole loop down" — the wind-down backstop is
            # still there and still unconditional.
            return None
        price = tick.bid if position.direction is Direction.LONG else tick.ask
        distance = abs(position.tp - price)
        if speed <= 0 or distance <= 0:
            return None
        atr_units = distance / speed / config.travel_efficiency
        return atr_units**2 * timeframe.duration.total_seconds() / 60.0

    def _cost_of_leaving(self, position: Position, risk: float, tick) -> float:  # type: ignore[no-untyped-def]
        """What it costs to take the money, in R.

        The spread crossed on the way out plus the commission on that side.
        Volume cancels out of the commission ratio, so this is a property of
        the trade's shape rather than of its size — which is why a tight stop
        makes leaving expensive however few lots are on.
        """
        if risk <= 0:
            return 0.0
        spec = self.broker.spec(position.symbol)
        cost = getattr(tick, "spread", 0.0) / risk
        per_side = self.settings.risk.commission_per_lot(spec.asset_class.value) / 2.0
        money = spec.money_per_lot(risk)
        if money > 0:
            cost += per_side / money
        return cost

    def _worth_paying_to_leave(self, position: Position, risk: float, tick) -> bool:  # type: ignore[no-untyped-def]
        """Is closing at market worth more than letting the stop do it?

        There is already a guaranteed exit on this position, sitting at the
        broker, costing nothing to keep. So a discretionary market close is not
        "get out" — it is "get out *here* instead of *there*", and the only
        thing it can win is the distance between the two. When that distance is
        smaller than the spread and commission it costs to collect, the rule is
        paying to lose more.

        The live AUDCAD long is the case. Stop 10.9 pips out, and it closed on
        a health reading for -1.61R against a plan of -1.00R. Whatever the last
        stretch of that gap was, most of it was bought rather than suffered:
        the stop was already there and already free.

        Running to the stop is not free either, so it is charged honestly.
        `stop_slippage_pips` is a measured figure — an AUDNZD stop at 1.19722
        filled at 1.19705 — and leaving it out would compare a certain market
        fill against a stop fill that never happens.

        This deliberately has no threshold. It is two costs compared, and it
        answers the same way on a trade at +0.1R above a break-even stop as on
        one at -0.9R above its original: in both, the stop is close and the
        crossing is not worth buying.
        """
        if risk <= 0:
            return True
        price = getattr(tick, "bid" if position.direction is Direction.LONG else "ask", 0.0)
        if not price or position.sl <= 0:
            # No read on where we are or where the stop is. The reading asked
            # for an exit and nothing here can argue it down, so it stands.
            return True
        spec = self.broker.spec(position.symbol)
        slip = self.settings.risk.stop_slippage_pips.get(spec.asset_class.value, 0.0)
        # Guarded rather than called flat, because this runs once a second on
        # every open position and an asset class with no measured figure must
        # cost the comparison its slippage term, not take down the loop.
        slip_price = spec.pips_to_price(slip) if slip > 0 else 0.0
        saved_r = (abs(price - position.sl) + slip_price) / risk
        return saved_r > self._cost_of_leaving(position, risk, tick)

    def _spread_squeeze_exit(
        self, position: Position, tick, r_now: float
    ) -> ManagementEvent | None:  # type: ignore[no-untyped-def]
        """Leave when the spread, not the market, is about to take the stop.

        A stop does not trigger on the price you watch. A short is closed at the
        ask, a long at the bid, and both of those move toward the stop when the
        book thins — with the market perfectly still. A live NZDJPY sell showed
        it exactly: bid 92.845, stop 92.904, five point nine pips of room, and
        six pips of spread would have taken it out having never gone wrong.

        Being stopped there is not the market disagreeing with the trade. It is
        the broker's book charging for the hour. Both exits pay the spread, but
        this one happens at a price we picked and turns a full stop-out into
        roughly break-even.

        The `r_now` floor is what keeps this from becoming "close every loser
        just before its stop". Once price has genuinely carried the trade most
        of the way there, the room is small because the trade is losing, not
        because the quote is wide, and the stop is doing precisely its job.
        """
        config = self.settings.trade_management
        share_limit = config.spread_squeeze_share
        if share_limit <= 0 or not position.sl:
            return None
        spread = getattr(tick, "spread", 0.0)
        if spread <= 0:
            return None
        # The side the stop actually triggers on: ask for a short, bid for a
        # long. Reading the wrong one hides the whole effect on the short side.
        trigger = tick.ask if position.direction is Direction.SHORT else tick.bid
        room = abs(position.sl - trigger)
        if room <= 0 or spread < room * share_limit:
            return None
        if r_now < config.spread_squeeze_min_r:
            return None
        result = self.broker.close_position(position)
        if not result.ok:
            return None
        return ManagementEvent(
            position.ticket,
            "SPREAD_SQUEEZE",
            f"spread {spread:.5g} is {spread / room:.0%} of the {room:.5g} left to the stop; "
            f"leaving at {r_now:.2f}R rather than being stopped by the quote",
            result.filled_price,
            position.profit + position.swap,
            r_at_action=r_now,
        )

    def _session_windows(self, asset_class: str):  # type: ignore[no-untyped-def]
        """The evening-flat window for this asset class, built once and reused.

        Built from the same filter the entry gate uses, so the moment entries
        stop is exactly the moment the flatten starts — for each class. Two
        independently derived times would eventually drift and leave a gap in
        which the loop closes a position and immediately re-opens it.
        """
        if self._session_filter is _UNSET:
            config = self.settings.filters.session
            self._session_filter = SessionFilter(config) if config.enabled else None
        if self._session_filter is None:
            return None
        return self._session_filter.evening_flat_window(asset_class)

    def _should_be_flat(self, now: datetime, asset_class: str) -> bool:
        """Whether we have already decided not to be in this market right now.

        Shares the `SessionFilter` instance with the entry gate for the same
        reason everything else here does: two independently derived answers to
        "is this a moment to be out" would drift apart, and the gap between
        them is where a position sits unattended.
        """
        self._session_windows(asset_class)  # builds the filter on first use
        if self._session_filter is None:
            return False
        return self._session_filter.should_be_flat(now, asset_class)

    def _runway_minutes(self, now: datetime, asset_class: str) -> float | None:
        """Minutes before we force this instrument flat, or None if never.

        The same `SessionFilter` instance the wind-down uses, for the same
        reason: two independently derived deadlines would drift, and the hour
        before the close is exactly where that gap would hurt.
        """
        self._session_windows(asset_class)  # builds the filter on first use
        if self._session_filter is None:
            return None
        return self._session_filter.minutes_of_runway(now, asset_class)

    def _worth_moving(self, position: Position, candidate: float, risk: float) -> bool:
        """Whether moving the stop to `candidate` is worth a broker round-trip.

        One test for all three rules that touch a stop, so a stop under any of
        them can only ever move in the direction of the trade, and only when
        the move is big enough to matter. See `_STOP_IMPROVEMENT_R` for why the
        floor is not zero.
        """
        if risk <= 0:
            return False
        return (candidate - position.sl) * int(position.direction) / risk >= _STOP_IMPROVEMENT_R

    def _record_excursion(self, trade_id: int, r_now: float, recorded_peak: float) -> float:
        """Ratchet how far the trade has run, and return the peak so far.

        The columns existed and nothing ever wrote to them, so every postmortem
        reported MAE and MFE as unknown and the give-back rule below had no
        memory to work from. Persisted rather than held in the process because
        a restart must not hand a trade a fresh peak — the whole point is that
        we remember it was up.
        """
        self.journal.update_excursions(trade_id, mae_r=min(0.0, r_now), mfe_r=max(0.0, r_now))
        return max(recorded_peak, r_now)

    def _unmanaged(self, position: Position, why: str) -> None:
        """Record that this position was skipped, and say why.

        The two paths into here both used to be a bare `continue`. A position
        that takes one of them gets nothing from this file: no health read, no
        give-back, no profit lock, no peak stall, no time exit. It is held by
        its broker stop and by nothing else — which is a legitimate state after
        a manual trade or a half-recovered restart, and a serious one to be in
        without knowing.

        It was invisible. The health map simply had no entry for the ticket,
        the deck rendered that absence as "geen live oordeel (draait Jarvis?)",
        and an operator looking at a running Jarvis reasonably concluded the
        deck was broken. Logged at warning because being unmanaged is not
        routine, and carried into the health map so the panel can say the true
        thing instead of the misleading one.
        """
        self.last_health[position.ticket] = PositionHealth("unmanaged", 0.0, "hold", (), why)
        log.warning(
            "position is not being managed: %s",
            why,
            extra={
                "event": "position_unmanaged",
                "ticket": position.ticket,
                "symbol": position.symbol,
                "reason": why,
            },
        )

    def _peak_stall_exit(
        self, position: Position, r_now: float, peak_r: float, now: datetime
    ) -> ManagementEvent | None:
        """Bank a profit whose move has stopped advancing, while it is still there.

        Every other exit in this file measures how much has been *given back*,
        which means every one of them can only act after the money has gone.
        This measures what a person actually watches: the trade has stopped
        making new highs. A move that is working prints a new high every few
        minutes. One that has sat at the same level for six is not pausing, it
        is over, and what comes next is the retrace that all the other rules
        are waiting for.

        A live NZDCAD long is the case. It peaked at 0.92R — EUR 1.60 on an
        EUR 87 account — and closed at 0.13R for 22 cents. Three separate rules
        were aimed at it: the profit lock armed at 1.0R and so never fired, the
        give-back allowed an 80% drain while the health read stayed healthy,
        and the break-even stop sat below both and took the trade. The crudest
        rule in the file decided the outcome, and the operator watching it had
        the right instinct forty minutes earlier: it has been sitting at the
        same high for a while, take it.

        Three conditions, all required. Enough profit to be worth protecting;
        still near the high, because past that the give-back owns the decision
        and this rule is specifically about leaving *at* the top; and a peak
        that has not moved for the configured wait.

        The peak's age is held in memory and is lost on restart, which resets
        the clock and errs toward holding. That is the safe direction for a
        rule that closes positions, and the alternative — persisting it — would
        make a restart able to close a trade on evidence gathered before the
        process that is now running ever existed.
        """
        config = self.settings.trade_management
        wait = config.peak_stall_minutes
        if wait <= 0 or peak_r < config.peak_stall_arm_r:
            self._peak_seen.pop(position.ticket, None)
            return None

        seen = self._peak_seen.get(position.ticket)
        if seen is None or peak_r > seen[0] + _PEAK_EPSILON_R:
            # A new high resets the clock. This is the whole mechanism: while
            # the trade keeps working it can never stall.
            self._peak_seen[position.ticket] = (peak_r, now)
            return None

        if r_now < peak_r * config.peak_stall_near_peak:
            return None
        standing_minutes = (now - seen[1]).total_seconds() / 60.0
        if standing_minutes < wait:
            return None

        result = self.broker.close_position(position)
        if not result.ok:
            return None
        self._peak_seen.pop(position.ticket, None)
        return ManagementEvent(
            position.ticket,
            "PEAK_STALL",
            f"{peak_r:.2f}R peak has not advanced in {standing_minutes:.0f} min and price "
            f"is still at {r_now:.2f}R; the move is done, banking it near the high",
            result.filled_price,
            position.profit + position.swap,
            r_at_action=r_now,
        )

    def _profit_lock(
        self, position: Position, r_now: float, peak_r: float, risk: float
    ) -> ManagementEvent | None:
        """Secure a share of the peak at the broker, once the trade has earned it.

        Break-even protects the entry; this protects the *move*. Between
        `break_even_at_r` and `partial_close_at_r` nothing touched the stop, so
        a position could run to 1.4R over several hours and hand back every
        cent of it to a stop still sitting at entry — right for hours, paid
        nothing.

        Measured from the peak rather than the current price, so the stop
        ratchets and never retreats: a trade that reached 2R keeps its 1R stop
        even after pulling back to 1.2R. Combined with the `improves` test
        below, a stop under this rule can only ever move in the direction of
        the trade.

        This overlaps the give-back exit on purpose, and the overlap is the
        point. The give-back rule lives inside our own loop; a stop lives at
        the broker and survives a VPS reboot, a dropped terminal connection,
        and this process dying at three in the morning. Protection that
        requires our code to still be running is absent exactly when it matters
        most, and this account is meant to be left alone overnight.
        """
        config = self.settings.trade_management
        if peak_r < config.profit_lock_from_r or risk <= 0:
            return None

        sign = int(position.direction)
        secured_r = peak_r * config.profit_lock_fraction
        target = position.price_open + secured_r * risk * sign
        if not self._worth_moving(position, target, risk):
            return None

        spec = self.broker.spec(position.symbol)
        result = self.broker.modify_stops(position, sl=spec.normalize_price(target), tp=position.tp)
        if not result.ok:
            return None
        return ManagementEvent(
            position.ticket,
            "PROFIT_LOCK",
            f"peak {peak_r:.2f}R, now {r_now:.2f}R; stop secures "
            f"{secured_r:.2f}R at the broker",
            r_at_action=r_now,
        )

    @staticmethod
    def _time_exit_verdict(
        config: TradeManagementConfig,
        age_hours: float,
        deadline: float | None,
        r_now: float,
        peak_r: float,
    ) -> str | None:
        """Why an aged position should be closed on the clock, or None to keep it.

        Two reasons, and the second one was missing.

        The first is the original rule: a trade that has gone nowhere. Dead
        capital still carries risk, still pays swap, and still occupies one of
        two slots on a small account.

        The second is the trade that got somewhere unremarkable and stopped. At
        `time_exit_min_abs_r` of 0.3 and a give-back that arms at 0.5R, a
        position sitting on +0.4R after a day and a half fell between the two:
        too profitable for the time exit, never good enough for the give-back.
        It stayed open indefinitely, paying swap for a slot it was not using —
        and the operator watching it asked exactly the right question, which is
        why is that still on.

        A person banks that. The test is the *peak*, not the current price:
        `peak_r` is ratcheted on every pass, so a trade that once ran to 2R and
        has come back to 0.4 has demonstrated something, and the give-back rule
        owns it. A trade whose best moment in a whole day was under
        `time_exit_stale_peak_r` has demonstrated the opposite. Being green is
        required — this rule banks a modest profit, it never realises a loss
        the old rule would have held.
        """
        if deadline is None or age_hours < deadline:
            return None
        if abs(r_now) < config.time_exit_min_abs_r:
            return "went nowhere"
        if r_now > 0 and peak_r < config.time_exit_stale_peak_r:
            return (
                f"in profit but never got going (best was {peak_r:.2f}R, under "
                f"{config.time_exit_stale_peak_r:.2f}R); banking it frees the slot"
            )
        return None

    def _giveback_exit(
        self, position: Position, r_now: float, peak_r: float, health: PositionHealth
    ) -> ManagementEvent | None:
        """Bank a profit that is being handed back — unless the move still works.

        The stop and the trail both measure from price and neither knows the
        trade was ever ahead, so a run that fully retraces looked exactly like a
        trade that never moved. This is the reflex a person has and the machine
        did not.

        Two things were wrong with the first version, and a live AUDJPY pair
        showed both. It peaked around EUR 1.00 against EUR 1.77 of risk — 0.57R
        — and ended at -0.28R. The rule armed at 1.0R, so it never engaged at
        all; neither did break-even, on the same threshold. Both sat 0.43R above
        anything that trade was ever going to reach, and watched the whole gain
        go.

        So the arming level now matches what counts as real profit on this
        account rather than a round number, and the decision itself is no longer
        a tripwire. At the give-back mark the question is the one a person
        actually asks: *is this still working?* If the health read says the move
        is intact, the position is held — that is the "it goes further" case,
        and closing there would be paying the spread to abandon a live trade on
        a wobble. If the read says otherwise, the profit is banked.

        The hard fraction is the part that is not a judgement. Whatever the
        read says, handing back nearly all of a gain is not a position worth
        holding, and no amount of confidence in the thesis makes it one.
        """
        config = self.settings.trade_management
        arm, fraction = config.giveback_arm_r, config.giveback_fraction
        if arm <= 0 or fraction <= 0 or peak_r < arm or r_now >= peak_r:
            return None

        given_back = (peak_r - r_now) / peak_r
        hard = config.giveback_hard_fraction
        if given_back < fraction:
            return None
        # Still working: give it the room. Only the hard backstop overrides.
        #
        # `healthy` and nothing looser. `watch` means a reader has already found
        # something off, and the give-back is itself independent evidence — the
        # trade's own P&L trajectory, which no health reader looks at. One of
        # each is the same corroboration standard the health engine applies to
        # itself, and "the drift is against us" is not a reason to keep holding
        # a gain that is draining away; it is the reason it is draining.
        conviction = health.verdict == "healthy"
        if conviction and given_back < hard:
            return None

        result = self.broker.close_position(position)
        if not result.ok:
            return None
        why = (
            f"gave back {given_back:.0%} of a {peak_r:.2f}R gain"
            if not conviction
            else f"gave back {given_back:.0%} of a {peak_r:.2f}R gain; too much to hold"
        )
        return ManagementEvent(
            position.ticket,
            "GIVEBACK_EXIT",
            f"{why} (now {r_now:.2f}R, read: {health.verdict})",
            result.filled_price,
            position.profit + position.swap,
            r_at_action=r_now,
        )

    def _adopt(self, position: Position) -> ManagementEvent | None:
        """Match an unexplained position to the intent that created it.

        The only positions eligible are ones this system planned: same magic
        (already filtered upstream), same symbol, direction and volume as an
        intent written shortly before, and no ticket recorded against it. That
        combination is a crash between sending and confirming, and closing it
        destroys a trade the system correctly decided to take.

        Anything else stays an orphan and is closed. A position opened by hand
        in the terminal, or by another strategy sharing the magic, has no stop
        the risk layer chose and no plan behind it, and adopting it would mean
        managing a trade nobody sized.

        `adoption_window_minutes` bounds how stale an intent may be. A position
        cannot predate its own intent, and one left over from days ago is far
        more likely to be a rejected order whose abandonment was itself lost
        than a match for what is on the screen now.
        """
        window = timedelta(minutes=self.settings.trade_management.adoption_window_minutes)
        spec = self.broker.spec(position.symbol)
        trade_id = self.journal.claim_pending_entry(
            symbol=position.symbol,
            direction=position.direction.name,
            volume=position.volume,
            ticket=position.ticket,
            entry_price=position.price_open,
            # Half a step: the broker fills in whole steps, so anything inside
            # this is the same lot size carrying float error.
            volume_tolerance=spec.volume_step / 2,
            since=position.opened_at - window,
            opened_at=position.opened_at,
        )
        if trade_id is None:
            return None
        log.warning(
            "adopted a position that was live at the broker but unrecorded",
            extra={
                "event": "position_adopted",
                "ticket": position.ticket,
                "trade_id": trade_id,
                "symbol": position.symbol,
            },
        )
        return ManagementEvent(
            position.ticket,
            "ADOPTED",
            f"matched to recorded entry intent #{trade_id}; the journal row lost its "
            f"ticket to a crash between sending the order and confirming it",
        )

    def apply_supervision(self, position: Position, verdict: Supervision) -> ManagementEvent | None:
        """Execute a supervisor's verdict, after re-proving it reduces risk.

        The verdict was formed against a snapshot; by the time it arrives the
        market has moved. So the guard runs here, against the position and the
        price as they are right now, and not where the question was asked. A
        verdict that no longer de-risks is dropped and logged rather than
        clamped into something the adviser did not say — silently "fixing" a
        stop into a valid one would execute a decision nobody made.

        The mechanical rules in `manage` have already run this cycle and their
        results stand. This layer can only move things further in the
        protective direction, never undo them.
        """
        if verdict.action == "hold":
            return None
        spec = self.broker.spec(position.symbol)
        tick = self.broker.tick(position.symbol)
        sign = int(position.direction)
        price = tick.bid if position.direction is Direction.LONG else tick.ask
        if not verdict.is_risk_reducing(
            direction_sign=sign,
            current_sl=position.sl,
            current_tp=position.tp,
            price_now=price,
        ):
            log.warning(
                "supervisor verdict refused: it would not reduce risk",
                extra={
                    "event": "supervision_refused",
                    "ticket": position.ticket,
                    "action": verdict.action,
                    "symbol": position.symbol,
                },
            )
            return ManagementEvent(
                position.ticket,
                "AI_SUPERVISION_REFUSED",
                f"{verdict.action} rejected: would not reduce risk ({verdict.reason})"[:400],
            )

        risk = abs(position.price_open - position.sl) if position.sl else 0.0
        r_now = ((price - position.price_open) * sign / risk) if risk else None

        if verdict.action == "close":
            result = self.broker.close_position(position)
            if not result.ok:
                return ManagementEvent(
                    position.ticket,
                    "AI_EXIT_REJECTED",
                    f"broker rejected AI exit: {result.retcode_name}",
                )
            closed = self.broker.closed_position(position.ticket)
            return ManagementEvent(
                position.ticket,
                "AI_CLOSE" if closed is not None else "AI_CLOSE_SENT",
                verdict.reason[:400],
                closed.exit_price if closed is not None else result.filled_price,
                closed.pnl_money if closed is not None else position.profit + position.swap,
                closed.closed_at if closed is not None else None,
                r_at_action=r_now,
            )

        if verdict.action == "partial_close":
            fraction = verdict.close_fraction or 0.0
            close_volume = spec.round_volume_down(position.volume * fraction)
            remaining = spec.round_volume_down(position.volume - close_volume)
            if close_volume < spec.volume_min or remaining < spec.volume_min:
                # At 0.01 lots there is no such thing as half a position. Saying
                # so is more useful than silently doing nothing, and closing the
                # whole thing instead would be a different decision.
                return ManagementEvent(
                    position.ticket,
                    "AI_PARTIAL_INFEASIBLE",
                    f"cannot split {position.volume:g} lots at the {spec.volume_min:g} minimum",
                )
            result = self.broker.close_position(position, close_volume)
            if not result.ok:
                return ManagementEvent(
                    position.ticket,
                    "AI_EXIT_REJECTED",
                    f"broker rejected AI partial: {result.retcode_name}",
                )
            return ManagementEvent(
                position.ticket,
                "AI_PARTIAL_CLOSE",
                verdict.reason[:400],
                exit_price=result.filled_price,
                volume_closed=result.filled_volume,
                remaining_volume=remaining,
                r_at_action=r_now,
            )

        if verdict.action == "tighten_stop":
            assert verdict.stop_loss is not None  # is_risk_reducing proved it
            result = self.broker.modify_stops(
                position,
                sl=spec.normalize_price(verdict.stop_loss),
                tp=position.tp,
            )
            if not result.ok:
                return ManagementEvent(
                    position.ticket,
                    "AI_MODIFY_REJECTED",
                    f"broker rejected AI stop: {result.retcode_name}",
                )
            return ManagementEvent(
                position.ticket,
                "AI_TIGHTEN_STOP",
                verdict.reason[:400],
                r_at_action=r_now,
            )

        if verdict.action == "pull_target_in":
            assert verdict.take_profit is not None  # is_risk_reducing proved it
            result = self.broker.modify_stops(
                position,
                sl=position.sl,
                tp=spec.normalize_price(verdict.take_profit),
            )
            if not result.ok:
                return ManagementEvent(
                    position.ticket,
                    "AI_MODIFY_REJECTED",
                    f"broker rejected AI target: {result.retcode_name}",
                )
            return ManagementEvent(
                position.ticket,
                "AI_PULL_TARGET_IN",
                verdict.reason[:400],
                r_at_action=r_now,
            )
        return None

    def manage_news(
        self, positions: list[Position], news_filter: NewsFilter
    ) -> list[ManagementEvent]:
        """De-risk open positions once their configured news window begins."""
        events: list[ManagementEvent] = []
        for position in positions:
            spec = self.broker.spec(position.symbol)
            action = news_filter.position_action(
                position,
                spec.currency_base,
                spec.currency_profit,
            )
            if action == "none":
                continue
            if action == "break_even":
                atr = self._atr(position.symbol)
                desired = spec.normalize_price(
                    position.price_open
                    + atr
                    * self.settings.trade_management.break_even_offset_atr
                    * int(position.direction)
                )
                tick = self.broker.tick(position.symbol)
                executable = tick.bid if position.direction is Direction.LONG else tick.ask
                valid_now = (position.direction is Direction.LONG and desired < executable) or (
                    position.direction is Direction.SHORT and desired > executable
                )
                improves = (position.direction is Direction.LONG and desired > position.sl) or (
                    position.direction is Direction.SHORT and desired < position.sl
                )
                if not improves:
                    continue
                if valid_now and improves:
                    result = self.broker.modify_stops(position, sl=desired, tp=position.tp)
                    if result.ok:
                        events.append(
                            ManagementEvent(
                                position.ticket,
                                "NEWS_BREAK_EVEN",
                                "news blackout started; stop moved beyond entry",
                            )
                        )
                        continue
                # A losing position cannot have a valid break-even stop. Keeping
                # it exposed would silently turn the configured protection off.
                action = "close"
            if action == "close":
                result = self.broker.close_position(position)
                if not result.ok:
                    events.append(
                        ManagementEvent(
                            position.ticket,
                            "NEWS_EXIT_REJECTED",
                            f"broker rejected protective exit: {result.retcode_name}",
                        )
                    )
                    continue
                closed = self.broker.closed_position(position.ticket)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "NEWS_EXIT" if closed is not None else "NEWS_EXIT_SENT",
                        "news blackout started; position closed",
                        closed.exit_price if closed is not None else None,
                        closed.pnl_money if closed is not None else None,
                        closed.closed_at if closed is not None else None,
                    )
                )
        return events

    def _atr(self, symbol: str, period: int = 14) -> float:
        key = (symbol, period)
        cached = self._atr_cache.get(key)
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1]
        value = self._compute_atr(symbol, period)
        self._atr_cache[key] = (now + ATR_CACHE_SECONDS, value)
        return value

    def _compute_atr(self, symbol: str, period: int = 14) -> float:
        raw = self.broker.copy_rates(symbol, Timeframe.H1.mt5_value, period + 2)
        frame = pd.DataFrame(raw)
        previous = frame["close"].shift(1)
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.tail(period).mean())
