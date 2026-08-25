"""Read-only diagnostic entry point for the autonomous trading system.

python main.py --check-config      # offline: validate config only
python main.py --status            # connect, run the startup guard, report
python main.py --data EURUSD       # fetch and summarise the MTF view
python main.py --risk              # current risk state and every limit
python main.py --filters EURUSD    # run every filter and show each verdict
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from types import FrameType

from config.loader import PACKAGE_ROOT, load_credentials, load_settings, terminal_path_from_env
from config.schema import Settings
from core.broker import Broker
from core.clock import LiveClock
from core.data_manager import DataManager, atr
from core.errors import KillSwitchEngaged, TradingSystemError
from core.mt5_connector import MT5Connector
from core.startup import enforce, run_startup_guard
from core.types import AccountSnapshot, Direction, Timeframe
from filters.base import FilterChain, FilterContext
from filters.calendar.providers import build_providers
from filters.calendar.service import CalendarService
from filters.correlation_filter import CorrelationFilter
from filters.currency_exposure import CurrencyExposureFilter
from filters.headline_filter import HeadlineFilter
from filters.liveliness_filter import LivelinessFilter
from filters.loss_cooldown import LossCooldownFilter
from filters.news_filter import NewsFilter
from filters.newsfeed.providers import build_providers as build_headline_providers
from filters.newsfeed.service import HeadlineService
from filters.runway_filter import RunwayFilter
from filters.session_filter import SessionFilter
from filters.spread_filter import SpreadFilter
from infra.killswitch import KillSwitch
from infra.logging import get_logger, setup_logging
from journal.database import Journal
from journal.recorder import Recorder
from risk.risk_manager import RiskManager

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MT5 trading system (Phase 1)")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--overlay", type=Path, default=None, help="optional overlay config")
    parser.add_argument(
        "--check-config", action="store_true", help="validate configuration and exit"
    )
    parser.add_argument("--status", action="store_true", help="connect and run the startup guard")
    parser.add_argument(
        "--data", metavar="SYMBOL", help="fetch and summarise the multi-timeframe view"
    )
    parser.add_argument(
        "--risk", action="store_true", help="show the current risk state and all limits"
    )
    parser.add_argument(
        "--filters", metavar="SYMBOL", help="run every filter for a symbol and show each verdict"
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="skip the micro_live confirmation prompt (for supervised restarts)",
    )
    return parser


def configure(settings: Settings) -> None:
    setup_logging(
        level=settings.logging.level,
        log_dir=PACKAGE_ROOT / settings.logging.directory,
        filename=settings.logging.filename,
        console=settings.logging.console,
        console_level=settings.logging.console_level,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )


def install_signal_handlers(connector: Broker | None) -> None:
    """Ctrl+C must close the session cleanly and say so in the log.

    A half-open terminal session leaves positions unmanaged, which is the one
    state this system must never exit into silently.
    """

    def handle(signum: int, _frame: FrameType | None) -> None:
        log.warning(
            "shutdown signal received",
            extra={"event": "signal", "signal": signal.Signals(signum).name},
        )
        if connector is not None:
            connector.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config, overlay=args.overlay)
    except TradingSystemError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 2

    configure(settings)
    log.info(
        "configuration loaded",
        extra={
            "event": "config_loaded",
            "mode": settings.mode.value,
            "risk_pct": settings.effective_risk_pct(),
            "whitelist": list(settings.active_whitelist),
        },
    )

    if args.check_config:
        print(
            f"config OK — mode={settings.mode.value} "
            f"risk={settings.effective_risk_pct():.2f}% "
            f"symbols={', '.join(settings.active_whitelist)}"
        )
        return 0

    kill_switch = KillSwitch.in_dir(PACKAGE_ROOT, settings.system.kill_switch_file)
    if kill_switch.is_engaged():
        print(
            f"STOP file present ({kill_switch.path}) — refusing to start. "
            f"Reason: {kill_switch.reason() or '(none given)'}"
        )
        return 3

    def guard() -> None:
        if kill_switch.is_engaged():
            raise KillSwitchEngaged(f"STOP file present: {kill_switch.reason()}")

    credentials = load_credentials(required=settings.mode.is_live)
    connector = MT5Connector(
        settings.mt5,
        credentials,
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
        pre_send_guard=guard,
    )
    install_signal_handlers(connector)

    try:
        account = connector.connect()
        report = run_startup_guard(settings, connector, account)

        # `main.py` never opens a position — it reads. So the guard reports here
        # instead of aborting: a diagnostic that refuses to run exactly when the
        # configuration is wrong is useless at the one moment you need it. The
        # exit code still carries the verdict.
        #
        # This does not weaken the trading path, which never went through
        # `enforce()` in the first place. `JarvisRunner.connect` has its own hard
        # asserts for the cases that can lose money — `_assert_account_mode`,
        # the arming file, the experimental contract, the AI gate — and it now
        # logs this same feasibility report at startup.
        if not report.ok:
            print(report.render())
            print(
                "\nBLOCKED for trading — the checks above must pass before "
                "jarvis.py will start. Diagnostics below still ran.",
                file=sys.stderr,
            )
        else:
            enforce(report, require_confirmation=not args.no_confirm)

        if args.data:
            show_data(connector, settings, args.data)
        if args.risk:
            show_risk(connector, settings, account, kill_switch)
        if args.filters:
            show_filters(connector, settings, args.filters)
        if not (args.data or args.risk or args.filters or args.status):
            print(
                "\nDiagnostic mode only. Use --status, --risk, --filters SYMBOL or "
                "--data SYMBOL. Start autonomous modes with jarvis.py or the dashboard."
            )
        return 0 if report.ok else 1
    except TradingSystemError as exc:
        log.error("startup failed", exc_info=True)
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        connector.shutdown()


def show_data(connector: Broker, settings: Settings, symbol: str) -> None:
    """Print the multi-timeframe view, so it can be eyeballed against a chart."""
    clock = LiveClock(connector.server_offset)
    manager = DataManager(connector, settings.data, clock)
    ctx = manager.get_context(symbol)
    spec = connector.spec(symbol)

    print(f"\n{spec.describe()}")
    if ctx.tick is not None:
        print(
            f"  tick  bid={ctx.tick.bid} ask={ctx.tick.ask} "
            f"spread={spec.price_to_pips(ctx.tick.spread):.2f} pips"
        )
    print(f"  {'tf':<5}{'bars':>7}{'last closed bar (UTC)':>26}{'close':>12}{'ATR14':>12}")
    for tf in (Timeframe.parse(name) for name in settings.data.timeframes):
        series = ctx.series[tf]
        atr_pips = spec.price_to_pips(atr(series.df, settings.trade_management.sl_atr_period))
        print(
            f"  {tf.value:<5}{len(series):>7}"
            f"{series.last_bar_time.strftime('%Y-%m-%d %H:%M'):>26}"
            f"{series.last_close:>12.5f}{atr_pips:>11.1f}p"
        )


if __name__ == "__main__":
    raise SystemExit(main())


def show_risk(
    connector: Broker,
    settings: Settings,
    account: AccountSnapshot,
    kill_switch: KillSwitch,
) -> None:
    """Print the live risk picture: every limit, and how close we are to it.

    Reads the journal, so the numbers shown are the same ones the gates use —
    not a separate calculation that could drift from the real thing.
    """
    clock = LiveClock(connector.server_offset)
    journal = Journal(
        PACKAGE_ROOT / settings.journal.database_path,
        clock,
        day_boundary_utc=settings.risk.day_boundary_utc,
    ).open()
    try:
        Recorder(journal, clock, settings).record_config_snapshot()
        manager = RiskManager(
            settings=settings,
            journal=journal,
            clock=clock,
            kill_switch=kill_switch,
            margin_safety_factor=settings.risk.margin_safety_factor,
        )
        positions = connector.positions(magic=settings.system.magic_number)
        state = manager.build_state(account, positions)
        decision = manager.check_can_trade(state)

        ccy = state.currency
        print(f"\n  RISK STATE — {settings.mode.value}")
        print(f"    equity        {state.equity:.2f} {ccy}   (peak {state.equity_peak:.2f})")
        print(
            f"    day           {state.day_pnl_pct:+.2f}%  "
            f"limit -{settings.effective_daily_loss_limit_pct():.1f}%   "
            f"since {state.day_start:%Y-%m-%d %H:%M} UTC"
        )
        print(
            f"    week          {state.week_pnl_pct:+.2f}%  "
            f"limit -{settings.risk.weekly_loss_limit_pct:.1f}%   "
            f"since {state.week_start:%Y-%m-%d %H:%M} UTC"
        )
        print(
            f"    drawdown      {state.drawdown_pct:.2f}%  "
            f"breaker {settings.risk.max_drawdown_circuit_breaker_pct:.1f}%"
        )
        print(
            f"    trades        {state.trades_today}/"
            f"{settings.effective_max_trades_per_day()} today, "
            f"{state.trades_this_week}/{settings.risk.max_trades_per_week} this week"
        )
        print(
            f"    positions     {len(state.open_positions)}/"
            f"{settings.effective_max_positions()} open"
        )
        print(
            f"    streak        {state.consecutive_losses} consecutive losses  "
            f"-> risk x{manager.risk_multiplier(state):.2f}"
        )
        verdict = "CLEAR" if decision.approved else str(decision.reason)
        print(f"\n    verdict       {verdict}\n                  {decision.detail}")
    finally:
        journal.close()


def build_filter_chain(
    connector: Broker, settings: Settings, journal: Journal, clock: LiveClock
) -> FilterChain:
    """Assemble the filters in evaluation order.

    Order is not arbitrary: most absolute first, so the reason that reaches the
    journal is the most fundamental one. A cycle blocked by both a news
    blackout and a wide spread should read as the blackout.
    """
    news_config = settings.filters.news
    calendar_dir = PACKAGE_ROOT / Path(news_config.cache_path).parent
    calendar = CalendarService(
        build_providers(news_config.providers, calendar_dir=calendar_dir),
        clock,
        cache_path=PACKAGE_ROOT / news_config.cache_path,
        refresh_interval_minutes=news_config.refresh_interval_minutes,
        max_age_minutes=news_config.max_calendar_age_minutes,
    )
    data = DataManager(connector, settings.data, clock)
    # One instance, shared with the runway gate. The deadline the runway gate
    # protects is this object's wind-down; a second copy could drift.
    session = SessionFilter(settings.filters.session)

    headline_config = settings.filters.headlines
    headlines = HeadlineService(
        build_headline_providers(headline_config.feeds or None),
        clock,
        refresh_interval_seconds=headline_config.refresh_interval_seconds,
        window_minutes=headline_config.window_minutes,
        baseline_hours=headline_config.baseline_hours,
        max_age_minutes=headline_config.max_age_minutes,
    )

    return FilterChain(
        [
            NewsFilter(news_config, calendar, clock),
            # Directly under the calendar, because it answers the same question
            # about the part of it the calendar cannot see, and above
            # everything below: a war breaking out is a more fundamental
            # objection than a wide spread, and the reason that reaches the
            # journal should be the fundamental one.
            HeadlineFilter(headline_config, headlines),
            session,
            RunwayFilter(settings.filters.runway, session),
            # Early, and above everything that fetches bars: one indexed row
            # from the journal decides it, and no amount of market analysis
            # changes the answer. The reason recorded then names the real
            # objection rather than whichever measurement happened to fail
            # second.
            LossCooldownFilter(
                settings.filters.loss_cooldown,
                journal.last_loss_closed_at,
                journal.last_close_at,
            ),
            LivelinessFilter(
                settings.filters.liveliness,
                lambda symbol, timeframe: data.get_series(symbol, timeframe),
            ),
            SpreadFilter(
                settings.filters.spread,
                journal,
                clock,
                retention_days=settings.filters.spread.retention_days,
            ),
            # Before the correlation filter, because it is the cheaper and more
            # absolute of the two: a shared currency leg is an identity, while a
            # correlation is a measurement that needs 200 bars to make.
            CurrencyExposureFilter(settings.filters.currency_exposure, connector.spec),
            CorrelationFilter(
                settings.filters.correlation,
                lambda symbol, timeframe: data.get_series(symbol, timeframe),
            ),
        ]
    )


def show_filters(connector: Broker, settings: Settings, symbol: str) -> None:
    """Run every filter for one symbol and print each verdict individually.

    Runs them all rather than short-circuiting like the live chain does: when
    you are checking whether the filters behave, seeing only the first block
    hides the other three.
    """
    clock = LiveClock(connector.server_offset)
    journal = Journal(
        PACKAGE_ROOT / settings.journal.database_path,
        clock,
        day_boundary_utc=settings.risk.day_boundary_utc,
    ).open()
    try:
        chain = build_filter_chain(connector, settings, journal, clock)
        spec = connector.spec(symbol)
        ctx = FilterContext(
            symbol=symbol,
            spec=spec,
            now=clock.now(),
            direction=Direction.LONG,
            tick=connector.tick(symbol),
            open_positions=tuple(connector.positions(magic=settings.system.magic_number)),
        )

        print(f"\n  FILTERS — {symbol} at {ctx.now:%Y-%m-%d %H:%M} UTC")
        blocked = []
        for filter_ in chain.filters:
            verdict = filter_.check(ctx)
            mark = "pass" if verdict.passed else "BLOCK"
            print(f"    {mark:<6}{filter_.name:<13}{verdict.detail}")
            if not verdict.passed:
                blocked.append(f"{filter_.name} ({verdict.reason})")

        print(f"\n    entry {'ALLOWED' if not blocked else 'BLOCKED by ' + ', '.join(blocked)}\n")
    finally:
        journal.close()
