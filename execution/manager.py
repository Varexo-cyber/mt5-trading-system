"""Open-position management and fail-closed restart reconciliation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from advisory.providers import Supervision
from config.schema import Settings
from core.broker import Broker
from core.types import Direction, Position, Timeframe
from filters.news_filter import NewsFilter
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


class PositionManager:
    def __init__(self, broker: Broker, journal: Journal, settings: Settings) -> None:
        self.broker = broker
        self.journal = journal
        self.settings = settings
        # symbol+period -> (monotonic deadline, value). Deliberately monotonic
        # rather than wall-clock: a simulated or rewound clock must not be able
        # to hold a stale ATR alive forever.
        self._atr_cache: dict[tuple[str, int], tuple[float, float]] = {}

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
                continue
            original_sl = float(row["sl"])
            risk = abs(position.price_open - original_sl)
            if risk <= 0:
                continue
            tick = self.broker.tick(position.symbol)
            price = tick.bid if position.direction is Direction.LONG else tick.ask
            r_now = (price - position.price_open) * int(position.direction) / risk
            peak_r = self._record_excursion(int(row["id"]), r_now, float(row["mfe_r"] or 0.0))
            giveback = self._giveback_exit(position, r_now, peak_r)
            if giveback is not None:
                events.append(giveback)
                continue
            age_hours = (now - position.opened_at).total_seconds() / 3600.0
            deadline = (
                config.time_exit_hours * patience if config.time_exit_hours is not None else None
            )
            if (
                deadline is not None
                and age_hours >= deadline
                and abs(r_now) < config.time_exit_min_abs_r
            ):
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "TIME_EXIT",
                        f"{age_hours:.1f}h, {r_now:.2f}R"
                        + (f" (drawdown posture: {deadline:.1f}h limit)" if patience < 1.0 else ""),
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
                continue
            atr = self._atr(position.symbol)
            break_even = position.price_open + atr * config.break_even_offset_atr * int(
                position.direction
            )
            stop_behind_break_even = (
                position.direction is Direction.LONG and position.sl < break_even
            ) or (position.direction is Direction.SHORT and position.sl > break_even)
            if r_now >= config.break_even_at_r and stop_behind_break_even:
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
            if config.trailing_mode == "atr" and r_now >= config.partial_close_at_r:
                trailing = price - atr * config.trailing_atr_multiple * int(position.direction)
                improves = (position.direction is Direction.LONG and trailing > position.sl) or (
                    position.direction is Direction.SHORT and trailing < position.sl
                )
                if improves:
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

    def _giveback_exit(
        self, position: Position, r_now: float, peak_r: float
    ) -> ManagementEvent | None:
        """Take what is left when a trade hands back most of its gain.

        The stop and the trail both measure from price and neither one knows
        the trade was ever ahead, so a run to 1.6R that fully retraces inside a
        single cycle looked exactly like a trade that never moved. This is the
        reflex a person has and the machine did not.
        """
        config = self.settings.trade_management
        arm, fraction = config.giveback_arm_r, config.giveback_fraction
        if arm <= 0 or fraction <= 0 or peak_r < arm:
            return None
        floor = peak_r * (1.0 - fraction)
        if r_now > floor:
            return None
        result = self.broker.close_position(position)
        if not result.ok:
            return None
        return ManagementEvent(
            position.ticket,
            "GIVEBACK_EXIT",
            f"peaked at {peak_r:.2f}R, back to {r_now:.2f}R",
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
