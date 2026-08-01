"""Phase 1 entry point.

What exists today: connect, validate, load data, report. There is no strategy
and no order loop yet, and that ordering is deliberate — the risk layer
(Phase 2) and the news filter (Phase 3) are the safety net, and a system that
can place orders before the net exists is a system that will place a bad one.

    python main.py --check-config      # offline: validate config only
    python main.py --status            # connect, run the startup guard, report
    python main.py --data EURUSD       # fetch and summarise the MTF view
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from types import FrameType

from config.loader import PACKAGE_ROOT, load_credentials, load_settings, terminal_path_from_env
from config.schema import Settings
from core.clock import LiveClock
from core.data_manager import DataManager, atr
from core.errors import KillSwitchEngaged, TradingSystemError
from core.mt5_connector import MT5Connector
from core.startup import enforce, run_startup_guard
from core.types import Timeframe
from infra.killswitch import KillSwitch
from infra.logging import get_logger, setup_logging

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


def install_signal_handlers(connector: MT5Connector | None) -> None:
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
        enforce(report, require_confirmation=not args.no_confirm)

        if args.data:
            show_data(connector, settings, args.data)
        elif not args.status:
            print(
                "\nPhase 1 only: no trading loop yet. "
                "Use --status or --data SYMBOL. See PLAN.md for the roadmap."
            )
        return 0
    except TradingSystemError as exc:
        log.error("startup failed", exc_info=True)
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        connector.shutdown()


def show_data(connector: MT5Connector, settings: Settings, symbol: str) -> None:
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
