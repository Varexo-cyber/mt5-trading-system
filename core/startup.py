"""Startup guard: refuse to run a configuration that cannot work.

Most of what this module does is arithmetic that anyone could do on paper, and
that almost nobody does before funding an account. It answers one question per
symbol:

    Given this equity, this risk percentage, and this broker's minimum lot,
    what is the widest stop I can actually afford?

On a EUR 100 account trading EURUSD at 0.01 lots, one pip is about EUR 0.92.
Risking 1% (EUR 1) therefore buys a stop of roughly **11 pips** — before spread.
There is no structural stop on a liquid major that is 11 pips wide often enough
to build a strategy on. That is not a bug in the system and not something a
better strategy fixes; it is the arithmetic of a EUR 100 account.

So this guard reports the number rather than hiding it, and the position sizer
(Phase 2) skips trades it cannot size instead of rounding up to the minimum lot.
Rounding up is the single most common way small accounts die: the trade that
was supposed to risk 1% quietly risks 4%, and five of those in a row is half
the account.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.schema import Settings
from core.broker import Broker
from core.errors import StartupGuardError
from core.instrument import InstrumentSpec
from core.types import AccountSnapshot, TradingMode
from infra.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolFeasibility:
    """What one symbol costs, in risk terms, at this account size."""

    symbol: str
    min_lot: float
    pip_value_at_min_lot: float
    #: Widest stop affordable at the configured risk, in pips.
    max_affordable_sl_pips: float
    #: Risk taken if the minimum lot is stopped at the mode's max SL distance.
    risk_pct_at_max_sl: float
    #: The mode's hard SL ceiling, for comparison.
    mode_max_sl_pips: float
    tradable: bool
    note: str = ""

    @property
    def is_expressible(self) -> bool:
        """True if at least some realistic stop distance fits the risk budget.

        Ten pips is the floor of realism: below that, spread and the broker's
        stop level eat the entire stop.
        """
        return self.tradable and self.max_affordable_sl_pips >= 10.0


@dataclass(frozen=True, slots=True)
class StartupReport:
    """Everything the guard checked, for the log and for the confirmation prompt."""

    mode: TradingMode
    account: AccountSnapshot
    risk_pct: float
    risk_money: float
    max_positions: int
    max_trades_per_day: int
    daily_loss_limit_money: float
    circuit_breaker_money: float
    symbols: tuple[SymbolFeasibility, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        """Human-readable summary. This is what you confirm before going live."""
        acct = self.account
        lines = [
            "=" * 72,
            f"  STARTUP CHECK — mode: {self.mode.value.upper()}",
            "=" * 72,
            f"  Account      {acct.login} @ {acct.server}   ({'DEMO' if acct.is_demo else 'LIVE'})",
            f"  Equity       {acct.equity:.2f} {acct.currency}"
            f"   (balance {acct.balance:.2f}, leverage 1:{acct.leverage})",
            "",
            f"  Risk/trade   {self.risk_pct:.2f}%  =  {self.risk_money:.2f} {acct.currency}",
            f"  Daily stop   {self.daily_loss_limit_money:.2f} {acct.currency}",
            f"  Breaker      {self.circuit_breaker_money:.2f} {acct.currency} from equity peak",
            f"  Limits       max {self.max_positions} positions, "
            f"{self.max_trades_per_day} trades/day",
            "",
            "  Per-symbol feasibility at minimum lot:",
            f"    {'symbol':<12}{'min lot':>9}{'pip val':>10}"
            f"{'max SL':>10}{'risk@maxSL':>12}   status",
        ]
        for s in self.symbols:
            status = "ok" if s.is_expressible else "NOT EXPRESSIBLE"
            if s.note:
                status = f"{status} — {s.note}"
            lines.append(
                f"    {s.symbol:<12}{s.min_lot:>9.2f}{s.pip_value_at_min_lot:>10.3f}"
                f"{s.max_affordable_sl_pips:>9.1f}p{s.risk_pct_at_max_sl:>11.2f}%   {status}"
            )

        if self.warnings:
            lines += ["", "  WARNINGS:"] + [f"    ! {w}" for w in self.warnings]
        if self.errors:
            lines += ["", "  BLOCKING:"] + [f"    x {e}" for e in self.errors]
        lines.append("=" * 72)
        return "\n".join(lines)


def run_startup_guard(
    settings: Settings, connector: Broker, account: AccountSnapshot
) -> StartupReport:
    """Validate that equity, mode, whitelist and broker specs are consistent.

    Returns the report. Raises `StartupGuardError` only via `enforce()`, so the
    caller can log and render the full picture before anything aborts.
    """
    limits = settings.active_limits
    errors: list[str] = []
    warnings: list[str] = []

    # -- mode vs. account type -------------------------------------------
    if settings.system.enforce_account_mode_match:
        if settings.mode.is_live and account.is_demo:
            warnings.append(
                f"mode is {settings.mode.value} but the terminal is on a DEMO account; "
                "execution findings from demo fills do not transfer to live"
            )
        if not settings.mode.is_live and not account.is_demo:
            errors.append(
                f"mode is {settings.mode.value} but the terminal is logged into a LIVE "
                "account. Refusing to start: a paper/backtest run must never be one "
                "code path away from a real order."
            )

    # -- equity band -------------------------------------------------------
    if account.equity < limits.min_equity:
        errors.append(
            f"equity {account.equity:.2f} {account.currency} is below the "
            f"{settings.mode.value} minimum of {limits.min_equity:.2f}"
        )
    if limits.max_equity is not None and account.equity > limits.max_equity:
        errors.append(
            f"equity {account.equity:.2f} {account.currency} exceeds the "
            f"{settings.mode.value} ceiling of {limits.max_equity:.2f} — "
            f"switch to a mode with limits appropriate to this account size"
        )

    # -- per-symbol feasibility -------------------------------------------
    risk_pct = settings.effective_risk_pct()
    risk_money = account.equity * risk_pct / 100.0

    feasibility: list[SymbolFeasibility] = []
    for symbol in settings.active_whitelist:
        allowed, reason = settings.symbol_allowed_at_equity(symbol, account.equity)
        if not allowed:
            errors.append(f"{symbol}: whitelisted for this mode but blocked — {reason}")
            continue
        try:
            spec = connector.spec(symbol)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the whole check
            errors.append(f"{symbol}: cannot read contract specification — {exc}")
            continue
        feasibility.append(_assess(spec, risk_money, account.equity, limits.max_sl_pips))

    if not feasibility:
        errors.append("no tradable symbol survived the startup check")

    for item in feasibility:
        if not item.tradable:
            errors.append(f"{item.symbol}: not tradable right now ({item.note})")
        elif not item.is_expressible:
            warnings.append(
                f"{item.symbol}: at {risk_pct:.2f}% risk the widest affordable stop is "
                f"{item.max_affordable_sl_pips:.1f} pips. Structural stops are rarely "
                f"that tight, so most setups on this symbol will be skipped as "
                f"TRADE_SKIPPED_UNDERCAPITALIZED."
            )
        elif item.risk_pct_at_max_sl > settings.effective_max_risk_pct():
            warnings.append(
                f"{item.symbol}: a stop at the mode's {item.mode_max_sl_pips:.0f}-pip ceiling "
                f"would risk {item.risk_pct_at_max_sl:.2f}% at the minimum lot, above the "
                f"{settings.effective_max_risk_pct():.2f}% cap. Setups needing a stop wider "
                f"than {item.max_affordable_sl_pips:.1f} pips will be skipped."
            )

    expressible = [f for f in feasibility if f.is_expressible]
    if feasibility and not expressible:
        warnings.append(
            "NOT ONE whitelisted symbol can express a trade at the configured risk. "
            "This account is too small for this risk setting; the system will run and "
            "analyse, but it will skip essentially every setup. Raising the risk "
            "percentage to make trades fit is the wrong fix."
        )

    report = StartupReport(
        mode=settings.mode,
        account=account,
        risk_pct=risk_pct,
        risk_money=risk_money,
        max_positions=settings.effective_max_positions(),
        max_trades_per_day=settings.effective_max_trades_per_day(),
        daily_loss_limit_money=account.equity * settings.effective_daily_loss_limit_pct() / 100.0,
        circuit_breaker_money=account.equity
        * settings.risk.max_drawdown_circuit_breaker_pct
        / 100.0,
        symbols=tuple(feasibility),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

    log.info(
        "startup check complete",
        extra={
            "event": "startup_check",
            "mode": settings.mode.value,
            "equity": account.equity,
            "currency": account.currency,
            "risk_pct": risk_pct,
            "risk_money": round(risk_money, 2),
            "symbols_ok": [f.symbol for f in expressible],
            "errors": list(errors),
            "warnings": list(warnings),
        },
    )
    return report


def _assess(
    spec: InstrumentSpec, risk_money: float, equity: float, mode_max_sl_pips: float
) -> SymbolFeasibility:
    pip_value_min_lot = spec.pip_value_per_lot() * spec.volume_min
    max_sl = spec.max_sl_pips_for_risk(risk_money)
    risk_at_max_sl = spec.min_risk_pct(spec.pips_to_price(mode_max_sl_pips), equity)

    note = ""
    if not spec.is_tradable:
        note = f"trade_mode={spec.trade_mode}"

    return SymbolFeasibility(
        symbol=spec.symbol,
        min_lot=spec.volume_min,
        pip_value_at_min_lot=pip_value_min_lot,
        max_affordable_sl_pips=max_sl,
        risk_pct_at_max_sl=risk_at_max_sl,
        mode_max_sl_pips=mode_max_sl_pips,
        tradable=spec.is_tradable,
        note=note,
    )


def enforce(report: StartupReport, *, require_confirmation: bool = True) -> None:
    """Abort on blocking errors; ask for confirmation before risking real money.

    The confirmation prompt is not ceremony. `micro_live` is the one mode where
    a config mistake costs money, and reading the equity, the risk in euros and
    the symbol list out loud once is cheap compared to discovering a wrong
    `symbol_suffix` after the fact.
    """
    print(report.render())

    if not report.ok:
        raise StartupGuardError("startup check failed:\n  - " + "\n  - ".join(report.errors))

    if report.mode is TradingMode.MICRO_LIVE and require_confirmation:
        answer = input("\nThis will trade REAL money. Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            raise StartupGuardError("operator did not confirm micro_live start")
        log.info("operator confirmed micro_live start", extra={"event": "startup_confirmed"})
