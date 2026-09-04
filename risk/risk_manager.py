"""Risk limits, circuit breakers, and the forbidden-practice assertions.

This is the safety net, and it is built before any strategy exists on purpose.
A system that can place orders before the net is in place will eventually place
a bad one, and the net is not something you can retrofit under a running
strategy without discovering which of its assumptions were load-bearing.

Three layers, in increasing severity:

1. **Per-trade gates** — this setup is refused, the system keeps running. Most
   rejections are here and they are entirely normal; no-trade is the default.
2. **Period stops** — the daily or weekly loss limit is hit. Trading pauses
   until the next period. Open positions are left alone; the limit governs new
   risk, and force-closing at a limit turns a paper loss into a real one.
3. **Circuit breaker** — drawdown from the equity peak. Everything closes, the
   kill switch trips, and a human has to restart. Deliberately the only
   condition that cannot clear itself with the passage of time.

Loss limits are measured on **equity**, not realised P/L, so open positions
count against the limit as they move. Measuring only realised P/L would let an
account sit 8% underwater with a "0% daily loss" and happily open more risk.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from config.schema import NO_DRAWDOWN_BREAKER, NO_LOSS_LIMIT, UNLIMITED_TRADES, Settings
from core.clock import Clock
from core.errors import ForbiddenStrategyError
from core.instrument import InstrumentSpec
from core.trade_origin import broker_comment, section_of_comment
from core.types import AccountSnapshot, Direction, Position
from infra.killswitch import KillSwitch
from infra.logging import get_logger
from journal.database import Journal
from risk.position_sizer import SizingResult
from risk.reasons import Reason, RiskDecision

log = get_logger(__name__)

#: Money callback: (symbol, direction, volume, price) -> margin required.
MarginEstimator = Callable[[str, Direction, float, float], float]

#: Market callback: symbol -> can an order or a stop change be sent right now?
#: False means the venue is shut, not that the trade is a bad idea.
ManageabilityProbe = Callable[[str], bool]
#: symbol -> InstrumentSpec, for valuing the risk on an open position.
SpecLookup = Callable[[str], object]

# `jarvis-addon` is retained for positions opened by the first deployed
# pyramiding build. New tickets use the operator-facing scalp name. MT5 stores
# this comment on the position, so the distinction survives a restart.
WINNER_SCALP_COMMENTS = frozenset({"jarvis-addon", "jarvis-scalp"})


@dataclass(frozen=True, slots=True)
class RiskState:
    """Everything the limits are evaluated against, at one instant.

    Assembled from the account snapshot, live positions, and the journal —
    never from in-memory counters. A restart must not hand the system a fresh
    daily budget, and the only way to guarantee that is to read the numbers
    back from disk every cycle.
    """

    now: datetime
    equity: float
    balance: float
    margin_free: float
    currency: str

    equity_peak: float
    day_start_equity: float
    week_start_equity: float
    day_start: datetime
    week_start: datetime

    trades_today: int
    trades_this_week: int
    consecutive_losses: int
    last_trade_risk_pct: float | None

    open_positions: tuple[Position, ...] = ()
    halted: bool = False
    halt_reason: str = ""

    # -- derived --------------------------------------------------------

    @property
    def day_pnl(self) -> float:
        return self.equity - self.day_start_equity

    @property
    def day_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return 100.0 * self.day_pnl / self.day_start_equity

    @property
    def week_pnl(self) -> float:
        return self.equity - self.week_start_equity

    @property
    def week_pnl_pct(self) -> float:
        if self.week_start_equity <= 0:
            return 0.0
        return 100.0 * self.week_pnl / self.week_start_equity

    @property
    def drawdown_pct(self) -> float:
        """Drawdown from the all-time equity peak, as a positive percentage."""
        if self.equity_peak <= 0:
            return 0.0
        return max(0.0, 100.0 * (self.equity_peak - self.equity) / self.equity_peak)

    def has_position_in(self, symbol: str) -> bool:
        return any(position.symbol == symbol for position in self.open_positions)

    def positions_in(self, symbol: str) -> tuple[Position, ...]:
        return tuple(position for position in self.open_positions if position.symbol == symbol)

    def position_in(self, symbol: str) -> Position | None:
        return next((p for p in self.open_positions if p.symbol == symbol), None)

    def summary(self) -> dict[str, object]:
        return {
            "equity": round(self.equity, 2),
            "day_pnl_pct": round(self.day_pnl_pct, 3),
            "week_pnl_pct": round(self.week_pnl_pct, 3),
            "drawdown_pct": round(self.drawdown_pct, 3),
            "trades_today": self.trades_today,
            "trades_this_week": self.trades_this_week,
            "consecutive_losses": self.consecutive_losses,
            "open_positions": len(self.open_positions),
            "halted": self.halted,
        }


@dataclass
class RiskManager:
    """Evaluates every limit before a trade, and trips the breaker after one."""

    settings: Settings
    journal: Journal
    clock: Clock
    kill_switch: KillSwitch | None = None
    margin_estimator: MarginEstimator | None = None
    #: Asks the broker whether a symbol's market is currently open. Optional:
    #: without it every position counts toward the limit, as it always did.
    manageability_probe: ManageabilityProbe | None = None
    #: Margin headroom to leave untouched. A position that consumes the last of
    #: free margin leaves nothing for the adverse excursion before the stop.
    margin_safety_factor: float = 2.0
    _halt_reason: str = field(default="", init=False)

    # -- state -------------------------------------------------------------

    def build_state(
        self, account: AccountSnapshot, positions: Sequence[Position] = ()
    ) -> RiskState:
        """Assemble the current risk picture from account, positions and journal.

        Side effect by design: the day and week equity anchors are written on
        first sight of a new period, and the equity peak ratchets. Both have to
        happen exactly once per period and survive restarts, so the journal is
        the only sensible place for them.
        """
        now = self.clock.now()
        day_start = self.journal.day_start(now)
        week_start = self.journal.week_start(now)

        self.journal.set_equity_mark("DAY", day_start, account.equity)
        self.journal.set_equity_mark("WEEK", week_start, account.equity)
        peak = self.journal.record_equity_peak(account.equity)
        participating = tuple(
            position
            for position in positions
            if not self.settings.instruments.is_hands_off(position.symbol)
        )
        ignored = tuple(self.settings.instruments.ignored_symbols)

        return RiskState(
            now=now,
            equity=account.equity,
            balance=account.balance,
            margin_free=account.margin_free,
            currency=account.currency,
            equity_peak=peak,
            day_start_equity=self.journal.equity_mark("DAY", day_start) or account.equity,
            week_start_equity=self.journal.equity_mark("WEEK", week_start) or account.equity,
            day_start=day_start,
            week_start=week_start,
            trades_today=self.journal.trades_since(day_start, excluded_symbols=ignored),
            trades_this_week=self.journal.trades_since(week_start, excluded_symbols=ignored),
            consecutive_losses=self.journal.consecutive_losses(excluded_symbols=ignored),
            last_trade_risk_pct=self.journal.last_trade_risk_pct(excluded_symbols=ignored),
            open_positions=participating,
            halted=bool(self._halt_reason),
            halt_reason=self._halt_reason,
        )

    def halt(self, reason: str) -> None:
        """Stop opening new trades until the process is restarted."""
        self._halt_reason = reason
        log.critical("trading halted", extra={"event": "risk_halt", "reason": reason})

    # -- anti-martingale ---------------------------------------------------

    def open_risk_pct(self, state: RiskState, spec_for: SpecLookup | None = None) -> float:
        """How much of the account is at risk across every open position, now.

        Measured on each position's CURRENT stop rather than the one it opened
        with, which is the honest reading and also the generous one: a winner
        whose stop has walked to break-even is risking nothing and stops
        consuming the budget, freeing room for the next trade.

        A position with no stop at the broker counts as its full notional
        against the cap. That is deliberately pessimistic — an unstopped
        position has unbounded downside and the whole point of the cap is to
        refuse the next trade when the book is already loaded.

        Returns 0.0 when nothing can be measured. The cap is a limit on known
        exposure, and it must never block trading because a spec lookup
        failed; every other gate is still standing behind it.
        """
        if state.equity <= 0 or not state.open_positions:
            return 0.0
        total = 0.0
        for position in state.open_positions:
            spec = None
            if spec_for is not None:
                try:
                    spec = spec_for(position.symbol)
                except Exception:  # noqa: BLE001 - an unreadable spec must not block
                    spec = None
            if spec is None:
                continue
            distance = (
                abs(position.price_open - position.sl) if position.sl else abs(position.price_open)
            )
            try:
                total += spec.money_per_lot(distance) * position.volume
            except Exception:  # noqa: BLE001 - as above
                continue
        return 100.0 * total / state.equity

    def room_for_more_risk(
        self, state: RiskState, wanted_pct: float, spec_for: SpecLookup | None = None
    ) -> float:
        """`wanted_pct`, trimmed to what the total-exposure cap still allows.

        Returns the stake this trade may actually take. Zero means the book is
        already full and the caller must not open anything.

        This is the guard that makes conviction-scaled staking survivable. Four
        slots at a fixed 2% is 8% of the account at risk at once; four slots at
        a conviction-scaled 6% is 24%, on an account of a hundred and forty
        euros with the daily and weekly loss limits switched off. Raising the
        per-trade ceiling without this is not the change that was asked for.
        """
        cap = self.settings.risk.max_total_open_risk_pct
        if cap <= 0:
            return wanted_pct
        used = self.open_risk_pct(state, spec_for)
        remaining = cap - used
        # Snapped, because `cap - used` on a book that exactly fills the cap
        # lands a few parts in 10^13 above zero, and the contract here is that
        # zero means full. A remainder that size is not room for anything.
        if remaining < 1e-9:
            return 0.0
        return min(wanted_pct, remaining)

    def risk_multiplier(self, state: RiskState) -> float:
        """Risk scaling for the next trade. Never above 1.0.

        After a losing streak the risk is *halved*, not raised. This is
        anti-martingale and it is the only streak-based sizing this system
        permits: it reduces exposure exactly when the evidence that the edge is
        working is weakest.
        """
        threshold = self.settings.risk.losing_streak_threshold
        if state.consecutive_losses < threshold:
            return 1.0
        multiplier = self.settings.risk.losing_streak_risk_multiplier
        detail = {
            "consecutive_losses": state.consecutive_losses,
            "threshold": threshold,
            "multiplier": multiplier,
        }
        if multiplier >= 1.0:
            # The halving is switched off on this account, on purpose: at EUR 88
            # a halved stake is below the smallest lot the broker sells, so the
            # effect is not "trade smaller" but "stop trading" — see the note on
            # `losing_streak_risk_multiplier` in config/eightcap.yaml.
            #
            # Counted and reported either way. But it must not claim to have
            # reduced anything, and it must not shout: an operator reading
            # "reducing risk after a losing streak" eight times in one cycle
            # concludes the account is protecting itself, and here it is not.
            log.info("losing streak noted, risk unchanged", extra={"event": "streak", **detail})
            return 1.0
        log.warning(
            "reducing risk after a losing streak", extra={"event": "risk_reduced", **detail}
        )
        return multiplier

    # -- the gates ---------------------------------------------------------

    def _is_winner_scalp(self, position: Position) -> bool:
        config = self.settings.trade_management.pyramiding
        return (
            config.enabled
            and not config.counts_toward_position_limit
            and position.comment in WINNER_SCALP_COMMENTS
        )

    def _is_unmanageable(self, position: Position) -> bool:
        """True when this position's market is shut and no action on it is possible.

        A slot bounds how many trades are being *run*. A share whose exchange
        closed cannot be closed, tightened, secured or reasoned about until the
        venue reopens, so holding a slot for it buys nothing and costs every
        opportunity the rest of the evening.

        Fail-safe direction is deliberate and it is *toward counting*. Anything
        unclear -- the probe is absent, the broker raised, the answer was not a
        bool -- leaves the position counted, which is the current behaviour and
        the tighter of the two. Only an unambiguous "this venue is not quoting"
        releases a slot.
        """
        if not self.settings.risk.release_slots_when_unmanageable:
            return False
        probe = self.manageability_probe
        if probe is None:
            return False
        try:
            manageable = probe(position.symbol)
        except Exception:  # noqa: BLE001 - an unreadable market keeps its slot
            log.warning(
                "cannot tell whether this market is open; position keeps its slot",
                extra={
                    "event": "manageability_unknown",
                    "symbol": position.symbol,
                    "ticket": position.ticket,
                },
            )
            return False
        if not isinstance(manageable, bool):
            return False
        return not manageable

    def unmanageable_positions(self, state: RiskState) -> tuple[Position, ...]:
        """Open tickets whose market is shut. They keep their risk, not their slot."""
        return tuple(
            position for position in state.open_positions if self._is_unmanageable(position)
        )

    def positions_counted_toward_limit(self, state: RiskState) -> tuple[Position, ...]:
        """Primary trade ideas the system can still act on.

        Excluded: bounded winner scalp tickets, which are tracked separately,
        and positions whose market is shut, which cannot be managed at all.
        """
        return tuple(
            position
            for position in state.open_positions
            if not self._is_winner_scalp(position) and not self._is_unmanageable(position)
        )

    def check_can_trade(
        self, state: RiskState, *, allow_pyramid_overflow: bool = False
    ) -> RiskDecision:
        """System-wide gates: is this system allowed to open anything at all?

        Evaluated most-severe first, so the reason recorded in the journal is
        the one that actually matters. A day on which both the daily stop and
        the position limit were hit should read as the daily stop.
        """
        if self.kill_switch is not None and self.kill_switch.is_engaged():
            return RiskDecision.block(
                Reason.KILL_SWITCH,
                f"STOP file present: {self.kill_switch.reason() or '(no reason given)'}",
            )

        if state.halted:
            return RiskDecision.block(Reason.SYSTEM_HALTED, state.halt_reason)

        # Routed through the same predicate as the post-trade trip, not a second
        # inline comparison. There were two of these, and switching the breaker
        # off fixed only the one named `circuit_breaker_tripped` — this copy
        # kept its bare `>=`, so a disabled breaker blocked every cycle at
        # 0.0% and said "manual restart required" while doing it. One rule,
        # one place to read it.
        if self.circuit_breaker_tripped(state):
            breaker = self.settings.risk.max_drawdown_circuit_breaker_pct
            return RiskDecision.block(
                Reason.CIRCUIT_BREAKER,
                f"drawdown {state.drawdown_pct:.2f}% from the {state.equity_peak:.2f} "
                f"{state.currency} peak has reached the {breaker:.1f}% circuit breaker; "
                f"manual restart required",
            )

        # Zero disables a pacing limit, the drawdown breaker included. With the
        # breaker on it is the backstop either way: it measures from the
        # all-time peak rather than a period start, so it never resets and
        # cannot be waited out. With it off, the fixed capital floor in
        # `promotion/experimental.py` is the last unconditional stop.
        weekly = self.settings.risk.weekly_loss_limit_pct
        if weekly != NO_LOSS_LIMIT and state.week_pnl_pct <= -weekly:
            return RiskDecision.block(
                Reason.WEEKLY_LOSS_LIMIT,
                f"week is {state.week_pnl_pct:.2f}% down against a {weekly:.1f}% limit; "
                f"paused until {state.week_start.isoformat()} + 7d",
            )

        daily = self.settings.effective_daily_loss_limit_pct()
        if daily != NO_LOSS_LIMIT and state.day_pnl_pct <= -daily:
            return RiskDecision.block(
                Reason.DAILY_LOSS_LIMIT,
                f"day is {state.day_pnl_pct:.2f}% down against a {daily:.1f}% limit; "
                f"paused until the next trading day",
            )

        # THE SAME LIMIT IN MONEY, and on this account it is the only one set.
        #
        # The percentage above is 0 here -- switched off in August, when 3% of
        # EUR 88 was less than two losing trades wide. A fixed figure does not
        # have that problem: it does not shrink with the account, and the owner
        # named it in euros because euros are what he is actually watching.
        # Both are checked, the first to bite wins, neither disables the other.
        #
        # `day_pnl` is equity minus the day's starting equity, so a position
        # that is underwater counts before it is closed. Stricter than
        # realised-only, for the reason at the top of this module.
        daily_money = self.settings.risk.daily_loss_limit_money
        if daily_money > 0.0 and state.day_pnl <= -daily_money:
            return RiskDecision.block(
                Reason.DAILY_LOSS_LIMIT,
                f"day is {state.day_pnl:.2f} {state.currency} down against a "
                f"{daily_money:.2f} {state.currency} limit; no further trades today, "
                f"resumes on the next trading day",
            )

        # Zero means no cap. The counter was always the crude proxy here: the
        # loss limits above halt a bad day long before it could bite, so the
        # only day a count cap ever stops is one that is going well. See
        # `UNLIMITED_TRADES` in config.schema for the full argument.
        max_week = self.settings.risk.max_trades_per_week
        if max_week != UNLIMITED_TRADES and state.trades_this_week >= max_week:
            return RiskDecision.block(
                Reason.MAX_TRADES_PER_WEEK,
                f"{state.trades_this_week} trades this week, limit {max_week}",
            )

        max_day = self.settings.effective_max_trades_per_day()
        if max_day != UNLIMITED_TRADES and state.trades_today >= max_day:
            return RiskDecision.block(
                Reason.MAX_TRADES_PER_DAY,
                f"{state.trades_today} trades today, limit {max_day}",
            )

        # Scaled by what the account actually holds, so a deposit widens the
        # book and a drawdown narrows it without anyone editing a config file.
        max_positions = self.settings.effective_max_positions(state.equity)
        counted_positions = self.positions_counted_toward_limit(state)
        frozen = self.unmanageable_positions(state)
        # Named in both the refusal and the approval. A slot count that silently
        # disagrees with the terminal's position list is the kind of thing an
        # operator debugs for an hour, so the arithmetic is always spelled out.
        frozen_note = (
            f"; {len(frozen)} ticket(s) excluded because their market is shut "
            f"({', '.join(sorted({p.symbol for p in frozen}))})"
            if frozen
            else ""
        )
        if len(counted_positions) >= max_positions and not allow_pyramid_overflow:
            return RiskDecision.block(
                Reason.MAX_POSITIONS_REACHED,
                f"{len(counted_positions)} primary trade ideas open, limit {max_positions} "
                f"at {state.equity:.2f} equity ({len(state.open_positions)} total tickets)"
                f"{frozen_note}",
            )

        return RiskDecision.allow(
            f"day {state.day_pnl_pct:+.2f}%, week {state.week_pnl_pct:+.2f}%, "
            f"dd {state.drawdown_pct:.2f}%, {state.trades_today}/{max_day} trades today, "
            f"{len(counted_positions)}/{max_positions} slots used{frozen_note}"
        )

    def check_symbol(
        self,
        symbol: str,
        state: RiskState,
        spec: InstrumentSpec,
        *,
        direction: Direction | None = None,
        entry: float | None = None,
        allow_pyramid: bool = False,
        setup_family: str = "",
    ) -> RiskDecision:
        """Per-symbol gates: whitelist, equity floor, and existing exposure."""
        allowed, reason_code = self.settings.symbol_allowed_at_equity(symbol, state.equity)
        if not allowed:
            mapped = (
                Reason.SYMBOL_NOT_WHITELISTED
                if reason_code.startswith("SYMBOL_NOT_WHITELISTED")
                else Reason.SYMBOL_BLOCKED_BY_EQUITY
            )
            return RiskDecision.block(
                mapped, f"{symbol}: {reason_code} at equity {state.equity:.2f} {state.currency}"
            )

        if not spec.is_tradable:
            return RiskDecision.block(
                Reason.SYMBOL_NOT_TRADABLE, f"{symbol}: broker trade_mode={spec.trade_mode}"
            )

        # Same-symbol exposure remains forbidden by default. The one explicit
        # exception is a smaller, freshly approved scalp after every recorded
        # leg is already winning. Opposite-direction hedges and additions to a
        # flat or losing idea remain blocked.
        if state.has_position_in(symbol):
            existing = state.positions_in(symbol)
            shared = self._another_section_may_join(existing, symbol, direction, setup_family)
            if shared is not None:
                return shared
            pyramid = self._pyramid_permission(
                existing,
                all_positions=state.open_positions,
                direction=direction,
                entry=entry,
                allow=allow_pyramid,
            )
            if pyramid.approved:
                return pyramid
            first = existing[0]
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                f"{symbol}: position #{first.ticket} ({first.direction.name} "
                f"{first.volume:g} lots) is already open; {pyramid.detail}",
            )

        return RiskDecision.allow(f"{symbol} clear")

    def _another_section_may_join(
        self,
        existing: tuple[Position, ...],
        symbol: str,
        direction: Direction | None,
        setup_family: str,
    ) -> RiskDecision | None:
        """A SECOND SECTION on a symbol another section already holds.

        Returns a decision when this rule has an opinion, and None when it does
        not -- in which case the pyramiding gate below decides, exactly as
        before.

        WHY THIS EXISTS. Same-symbol exposure was refused per SYMBOL, so
        whichever section reached gold first locked every other one out for the
        length of its trade. Section six and section ten both trade XAUUSD:
        over 180 days section six was offered 600 trades and took 356, and the
        244 refusals were worth 30.55 R. The owner asked for the limit to come
        off on 4 September.

        THIS IS NOT PYRAMIDING AND MUST NOT BECOME IT. A second leg from the
        SAME section is still refused here and still has to satisfy
        `_pyramid_permission`, which demands every existing leg already be
        winning. What this allows is two INDEPENDENT sections, each with its
        own plan and its own stop, holding one instrument. Adding to your own
        losing idea is the thing this account forbids outright, and nothing
        below weakens it.

        WHAT IT COSTS, said plainly: two positions on one instrument is twice
        the exposure to one thing going wrong. That is the trade being made
        deliberately, not a side effect.
        """
        if not self.settings.risk.sections_may_share_a_symbol:
            return None
        if not setup_family or not existing:
            return None

        mine = broker_comment(setup_family, is_addon=False, experimental_live=False)
        family = section_of_comment(mine)
        if not family:
            return None
        # EVERY open position has to be identifiable as a DIFFERENT section,
        # and an unlabelled one is not identifiable at all.
        #
        # The first version only looked for a position carrying THIS section's
        # comment and let everything else through. A ticket commented plain
        # `jarvis` -- an older entry, a manual trade, a scalp add-on -- names
        # no section, so it did not match, so it was treated as somebody
        # else's and waved past. It could just as easily be this section's own,
        # and then the exception for "two sections" has quietly authorised a
        # second leg of one. That is the thing this whole rule exists to keep
        # separate from pyramiding.
        #
        # Anything unrecognised means the old refusal stands.
        holders = [section_of_comment(position.comment) for position in existing]
        if any(not holder or holder == family for holder in holders):
            return None

        if self.settings.risk.refuse_opposite_direction_across_sections and direction is not None:
            against = [p for p in existing if p.direction is not direction]
            if against:
                other = against[0]
                return RiskDecision.block(
                    Reason.POSITION_ALREADY_OPEN,
                    f"{symbol}: {section_of_comment(other.comment) or 'another section'} is "
                    f"{other.direction.name} here and this is {direction.name}. Long and "
                    f"short on one instrument is flat exposure bought with two spreads; "
                    f"at most one of the two stops can be reached, so the pair cannot "
                    f"win. Set risk.refuse_opposite_direction_across_sections false to "
                    f"allow it anyway.",
                )

        held = ", ".join(sorted(set(holders)))
        return RiskDecision.allow(
            f"{symbol}: {family} joins {held}; separate sections, separate stops"
        )

    def _pyramid_permission(
        self,
        positions: tuple[Position, ...],
        *,
        all_positions: tuple[Position, ...] | None = None,
        direction: Direction | None,
        entry: float | None,
        allow: bool,
    ) -> RiskDecision:
        """Prove that another same-symbol leg adds to a winner, not a loser."""
        config = self.settings.trade_management.pyramiding
        if not allow or not config.enabled:
            return RiskDecision.block(Reason.POSITION_ALREADY_OPEN, "pyramiding is not enabled")
        if direction is None or entry is None:
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                "pyramiding needs the fresh direction and executable entry price",
            )
        if len(positions) >= config.max_legs_per_symbol:
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                f"{len(positions)} legs already open, per-symbol ceiling "
                f"{config.max_legs_per_symbol}",
            )
        active_symbols = {
            position.symbol
            for position in (all_positions or positions)
            if self._is_winner_scalp(position)
        }
        symbol = positions[0].symbol if positions else None
        if (
            symbol is not None
            and symbol not in active_symbols
            and len(active_symbols) >= config.max_active_symbols
        ):
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                f"winner scalp campaign already active on "
                f"{', '.join(sorted(active_symbols))}; maximum active symbols is "
                f"{config.max_active_symbols}",
            )
        if any(position.direction is not direction for position in positions):
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                "the fresh direction conflicts with an existing leg; hedging is forbidden",
            )
        readings: list[float] = []
        for position in positions:
            row = self.journal.open_trade_by_ticket(position.ticket)
            if row is None:
                return RiskDecision.block(
                    Reason.POSITION_ALREADY_OPEN,
                    f"position #{position.ticket} has no recorded plan to measure R against",
                )
            original_stop = float(row["sl"])
            risk = abs(position.price_open - original_stop)
            if risk <= 0:
                return RiskDecision.block(
                    Reason.POSITION_ALREADY_OPEN,
                    f"position #{position.ticket} has no measurable original risk",
                )
            readings.append((entry - position.price_open) * int(direction) / risk)
        # Asked before the stop is inspected, and the order is deliberate: a
        # leg that is losing should be reported as losing. "Its stop is behind
        # its entry" is true of every loser and says nothing about why this one
        # was refused.
        weakest = min(readings)
        if weakest < config.min_existing_r:
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                f"weakest existing leg is {weakest:+.2f}R; winner scalps require every leg "
                f"at or above +{config.min_existing_r:.2f}R",
            )
        # The broker's own stop, not the plan's. A leg in profit whose stop is
        # still behind its entry can give the whole gain back and finish at
        # -1R; stacking on it makes the same uncertainty bigger rather than
        # betting on further upside. Once the guard has walked the stop to
        # entry, the leg is closed to loss and the worst case of the whole
        # campaign is the add-on's own quarter-size risk.
        #
        # Read off the position because that is what the broker will honour if
        # this process dies. A journal claiming break-even while the broker
        # never received the modification is exactly the state this refuses.
        if config.require_stop_beyond_entry:
            for position in positions:
                secured = (position.sl - position.price_open) * int(direction)
                if position.sl <= 0 or secured < 0:
                    return RiskDecision.block(
                        Reason.POSITION_ALREADY_OPEN,
                        f"position #{position.ticket} is {weakest:+.2f}R up but its stop is "
                        f"still at {position.sl:g}, behind the {position.price_open:g} entry; "
                        f"a scalp may only be stacked on a leg that can no longer lose",
                    )
        return RiskDecision.allow(
            f"winner scalp allowed: {len(positions)} existing leg(s), weakest {weakest:+.2f}R, "
            f"every stop at or beyond entry"
        )

    def check_margin(
        self, state: RiskState, symbol: str, direction: Direction, volume: float, price: float
    ) -> RiskDecision:
        """Refuse a trade that would leave no margin headroom.

        Skipped when no estimator is wired in (backtests), because guessing at
        margin is worse than not checking: a wrong guess either blocks valid
        trades or gives false comfort.
        """
        if self.margin_estimator is None:
            return RiskDecision.allow("margin check skipped (no estimator)")

        try:
            required = self.margin_estimator(symbol, direction, volume, price)
        except Exception as exc:  # noqa: BLE001 - inability to price margin must fail closed
            return RiskDecision.block(
                Reason.MARGIN_ESTIMATE_FAILED,
                f"{symbol}: broker margin estimate failed: {exc}",
            )
        headroom = required * self.margin_safety_factor
        if headroom > state.margin_free:
            return RiskDecision.block(
                Reason.INSUFFICIENT_MARGIN,
                f"{volume:g} lots needs {required:.2f} {state.currency} margin; with a "
                f"{self.margin_safety_factor:g}x buffer that is {headroom:.2f} against "
                f"{state.margin_free:.2f} free",
            )
        return RiskDecision.allow(f"margin {required:.2f}/{state.margin_free:.2f} free")

    def largest_volume_within_margin(
        self,
        state: RiskState,
        symbol: str,
        direction: Direction,
        volume: float,
        price: float,
        *,
        volume_min: float,
        volume_step: float,
    ) -> float:
        """The biggest volume at or below `volume` whose margin actually fits.

        `check_margin` is a yes/no gate: it is handed a volume derived from the
        risk budget and refuses when the margin will not stretch. On a small
        account against single-share CFDs that refuses almost everything —
        0.15 lots of one German share wants 169 EUR of margin against 180 EUR
        of equity — and the refusal throws away a setup that had already
        cleared every analytical gate.

        Nothing about the trade needs to change to make it fit. Entry, stop and
        target are untouched, so every measurement that approved it stays true;
        only the position gets smaller, and a smaller position risks LESS. That
        is the opposite of rounding up to the broker minimum, which this system
        forbids outright, and it is why this is safe in a way that "just take
        the trade anyway" would not be.

        Returns 0.0 when not even `volume_min` fits, so the caller still
        refuses rather than sending an order that cannot be margined.

        The broker's own estimator has the last word. Margin is very nearly
        linear in volume, which makes it a good first guess and a bad final
        answer — tiered rates exist — so the guess is verified and walked down
        a step at a time until it passes or runs out of room.
        """
        if self.margin_estimator is None or volume <= 0 or volume_step <= 0:
            return volume

        def required_for(candidate: float) -> float | None:
            try:
                return self.margin_estimator(symbol, direction, candidate, price)
            except Exception:  # noqa: BLE001 - an unpriceable volume is not a usable one
                return None

        def fits(candidate: float) -> bool:
            required = required_for(candidate)
            return (
                required is not None and required * self.margin_safety_factor <= state.margin_free
            )

        # ONE QUESTION PER SIZE, AND NOT ONE MORE. Every estimate is an IPC
        # round trip to a terminal that this account shares with a full
        # catalogue scan on one vCPU, and this runs between a quote and an
        # order. Asking `check_margin` first and then re-pricing the same
        # volume cost a duplicate call on the hot path for nothing.
        required = required_for(volume)
        if required is None:
            return 0.0  # fail closed, exactly as check_margin does
        headroom = required * self.margin_safety_factor
        if headroom <= state.margin_free:
            return volume
        if headroom <= 0:
            return 0.0
        steps = int(volume * (state.margin_free / headroom) / volume_step)
        # Ten walks down at most. The linear guess is close enough that one or
        # two is normal, and an instrument needing more than ten is one this
        # account has no business sizing by trial and error.
        for _ in range(10):
            if steps < 1:
                return 0.0
            candidate = round(steps * volume_step, 8)
            if candidate < volume_min:
                return 0.0
            if fits(candidate):
                return candidate
            steps -= 1
        return 0.0

    # -- forbidden practices -----------------------------------------------

    def assert_not_forbidden(
        self, sizing: SizingResult, state: RiskState, *, allow_pyramid: bool = False
    ) -> None:
        """Crash rather than execute a martingale, grid, or unstopped trade.

        These are assertions, not gates, and they raise instead of returning a
        decision. A gate returning "no" is a normal outcome that gets recorded
        and moved past. Reaching this function at all means a strategy tried
        something the system forbids outright, and continuing would mean the
        prohibition is advisory. It is not.
        """
        practices = self.settings.risk.forbidden

        if not practices.trade_without_stop_loss and sizing.sl <= 0:
            raise ForbiddenStrategyError(f"{sizing.symbol}: order without a stop loss — forbidden")

        existing = state.position_in(sizing.symbol)
        if existing is not None and allow_pyramid:
            pyramid = self._pyramid_permission(
                state.positions_in(sizing.symbol),
                all_positions=state.open_positions,
                direction=sizing.direction,
                entry=sizing.entry,
                allow=True,
            )
            if pyramid.approved:
                existing = None
        if existing is not None:
            if existing.direction is sizing.direction:
                raise ForbiddenStrategyError(
                    f"{sizing.symbol}: adding to an existing {existing.direction.name} "
                    f"position (#{existing.ticket}) is averaging down / gridding — forbidden"
                )
            raise ForbiddenStrategyError(
                f"{sizing.symbol}: opening {sizing.direction.name} while "
                f"#{existing.ticket} is {existing.direction.name} would hedge the "
                f"position — forbidden; close it instead"
            )

        # Recovery sizing: the risk *decision* may never go up after a loss.
        #
        # This compared the new trade's intended risk against the previous
        # trade's ACTUAL risk, and those are not the same quantity. The sizer
        # rounds volume down to the broker's step, so on a EUR 100 account the
        # actual risk of any trade lands wherever 0.01 lots happens to put it —
        # 1.44% on one symbol, 1.9% on the next, purely from lot granularity.
        # Every candidate after a loss was then refused as martingale:
        # "risk would rise from 1.440% to 2.000%", when nothing had risen at all
        # and the configured risk had not moved.
        #
        # What the rule protects against is a deliberate increase after losing,
        # and that decision is `intended_risk_pct` — the configured risk with
        # the losing-streak multiplier already applied. Comparing it against
        # what this risk manager sanctioned catches exactly that, and is immune
        # to lot rounding because both sides are decisions rather than outcomes.
        #
        # THE CEILING, NOT THE ORDINARY STAKE, once conviction scaling is on,
        # and that is a genuine loss of resolution which should be stated
        # rather than buried. Before conviction staking existed there was one
        # sanctioned number and this line caught any deviation from it at all.
        # Now an approval may legitimately be sized anywhere between the floor
        # and the ceiling, so what this can still prove is that the stake never
        # left the sanctioned envelope — not that it was the exact right point
        # inside it.
        #
        # What keeps the actual rule intact is structural rather than numeric:
        # the stake is a pure function of the REVIEWER'S CONFIDENCE
        # (`ConvictionRiskConfig.stake_for`), and nothing on that path reads the
        # losing streak, the day's P&L, or the last trade's outcome. A stake
        # cannot rise because something lost, because the loss is not an input.
        # The multiplier below still shrinks the whole envelope after a streak,
        # and `PositionSizer.size` still crashes outright on any multiplier
        # above 1.0, which is the way recovery sizing would actually be
        # expressed.
        conviction = self.settings.risk.conviction_risk
        ceiling = (
            max(self.settings.effective_risk_pct(), conviction.ceiling_pct)
            if conviction.enabled
            else self.settings.effective_risk_pct()
        )
        sanctioned = min(ceiling, self.settings.effective_max_risk_pct()) * self.risk_multiplier(
            state
        )
        if sizing.intended_risk_pct > sanctioned + 1e-9:
            raise ForbiddenStrategyError(
                f"{sizing.symbol}: sizing intends {sizing.intended_risk_pct:.3f}% but "
                f"{sanctioned:.3f}% is sanctioned after {state.consecutive_losses} "
                f"consecutive losses — martingale / recovery sizing is forbidden"
            )
        # And an execution may never exceed the decision behind it.
        if sizing.actual_risk_pct > sizing.intended_risk_pct + 1e-9:
            raise ForbiddenStrategyError(
                f"{sizing.symbol}: {sizing.volume} lots risks "
                f"{sizing.actual_risk_pct:.3f}%, above the intended "
                f"{sizing.intended_risk_pct:.3f}% — rounding must never size up"
            )

    # -- circuit breaker ---------------------------------------------------

    def circuit_breaker_tripped(self, state: RiskState) -> bool:
        limit = self.settings.risk.max_drawdown_circuit_breaker_pct
        # Zero means off, not "trip at nought". Without this line a disabled
        # breaker would fire on the first cycle of every account, because any
        # drawdown at all — including exactly zero — clears a bar of zero.
        if limit == NO_DRAWDOWN_BREAKER:
            return False
        return state.drawdown_pct >= limit

    def trip_circuit_breaker(self, state: RiskState) -> str:
        """Halt, trip the kill switch, and return the message to alert on.

        Does not close positions itself — that needs the connector, and this
        module is deliberately free of it. The caller flattens, then calls
        this. The kill switch file is what makes the halt survive a restart:
        without it, a supervisor restarting the process would resume trading
        into the same drawdown.
        """
        message = (
            f"CIRCUIT BREAKER: equity {state.equity:.2f} {state.currency} is "
            f"{state.drawdown_pct:.2f}% below the {state.equity_peak:.2f} peak "
            f"(limit {self.settings.risk.max_drawdown_circuit_breaker_pct:.1f}%). "
            f"All positions closed. Manual restart required."
        )
        self.halt(message)
        if self.kill_switch is not None:
            self.kill_switch.engage(message)
        log.critical(
            "circuit breaker tripped",
            extra={
                "event": "circuit_breaker",
                "equity": state.equity,
                "peak": state.equity_peak,
                "drawdown_pct": round(state.drawdown_pct, 3),
            },
        )
        return message

    # -- convenience -------------------------------------------------------

    def evaluate(
        self,
        state: RiskState,
        symbol: str,
        spec: InstrumentSpec,
        *,
        direction: Direction | None = None,
        entry: float | None = None,
        allow_pyramid: bool = False,
        setup_family: str = "",
    ) -> RiskDecision:
        """Run every pre-sizing gate in order and return the first refusal."""
        for decision in (
            self.check_can_trade(state, allow_pyramid_overflow=allow_pyramid),
            self.check_symbol(
                symbol,
                state,
                spec,
                direction=direction,
                entry=entry,
                allow_pyramid=allow_pyramid,
                setup_family=setup_family,
            ),
        ):
            if not decision.approved:
                log.info(
                    "trade blocked",
                    extra={
                        "event": "risk_block",
                        "symbol": symbol,
                        "reason": str(decision.reason),
                        "detail": decision.detail,
                        **state.summary(),
                    },
                )
                return decision
        return RiskDecision.allow("all risk gates clear")
