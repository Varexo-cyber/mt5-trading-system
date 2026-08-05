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
from core.types import AccountSnapshot, Direction, Position
from infra.killswitch import KillSwitch
from infra.logging import get_logger
from journal.database import Journal
from risk.position_sizer import SizingResult
from risk.reasons import Reason, RiskDecision

log = get_logger(__name__)

#: Money callback: (symbol, direction, volume, price) -> margin required.
MarginEstimator = Callable[[str, Direction, float, float], float]


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
            trades_today=self.journal.trades_since(day_start),
            trades_this_week=self.journal.trades_since(week_start),
            consecutive_losses=self.journal.consecutive_losses(),
            last_trade_risk_pct=self.journal.last_trade_risk_pct(),
            open_positions=tuple(positions),
            halted=bool(self._halt_reason),
            halt_reason=self._halt_reason,
        )

    def halt(self, reason: str) -> None:
        """Stop opening new trades until the process is restarted."""
        self._halt_reason = reason
        log.critical("trading halted", extra={"event": "risk_halt", "reason": reason})

    # -- anti-martingale ---------------------------------------------------

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
        log.warning(
            "reducing risk after a losing streak",
            extra={
                "event": "risk_reduced",
                "consecutive_losses": state.consecutive_losses,
                "threshold": threshold,
                "multiplier": multiplier,
            },
        )
        return multiplier

    # -- the gates ---------------------------------------------------------

    def check_can_trade(self, state: RiskState) -> RiskDecision:
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

        breaker = self.settings.risk.max_drawdown_circuit_breaker_pct
        if state.drawdown_pct >= breaker:
            return RiskDecision.block(
                Reason.CIRCUIT_BREAKER,
                f"drawdown {state.drawdown_pct:.2f}% from the {state.equity_peak:.2f} "
                f"{state.currency} peak has reached the {breaker:.1f}% circuit breaker; "
                f"manual restart required",
            )

        # Zero disables a pacing limit. The drawdown breaker above is the
        # backstop either way: it measures from the all-time peak rather than a
        # period start, so it never resets and cannot be waited out.
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

        max_positions = self.settings.effective_max_positions()
        if len(state.open_positions) >= max_positions:
            return RiskDecision.block(
                Reason.MAX_POSITIONS_REACHED,
                f"{len(state.open_positions)} positions open, limit {max_positions}",
            )

        return RiskDecision.allow(
            f"day {state.day_pnl_pct:+.2f}%, week {state.week_pnl_pct:+.2f}%, "
            f"dd {state.drawdown_pct:.2f}%, {state.trades_today}/{max_day} trades today"
        )

    def check_symbol(self, symbol: str, state: RiskState, spec: InstrumentSpec) -> RiskDecision:
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

        # One position per symbol, full stop. A second one is either averaging
        # down (if same direction) or a hedge that nets to a worse spread paid
        # twice — both are on the forbidden list.
        if state.has_position_in(symbol):
            existing = state.position_in(symbol)
            assert existing is not None
            return RiskDecision.block(
                Reason.POSITION_ALREADY_OPEN,
                f"{symbol}: position #{existing.ticket} ({existing.direction.name} "
                f"{existing.volume:g} lots) is already open",
            )

        return RiskDecision.allow(f"{symbol} clear")

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

    # -- forbidden practices -----------------------------------------------

    def assert_not_forbidden(self, sizing: SizingResult, state: RiskState) -> None:
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
        sanctioned = self.settings.effective_risk_pct() * self.risk_multiplier(state)
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

    def evaluate(self, state: RiskState, symbol: str, spec: InstrumentSpec) -> RiskDecision:
        """Run every pre-sizing gate in order and return the first refusal."""
        for decision in (self.check_can_trade(state), self.check_symbol(symbol, state, spec)):
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
