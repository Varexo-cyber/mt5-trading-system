"""Phase 1 acceptance test — run this on Windows against a DEMO account.

It answers the only question Phase 1 has to answer:

    Can I reliably pull data, place an order, and close it, with every
    execution detail captured?

It places ONE minimum-lot order with a stop and a target, holds it briefly,
then closes it at market — and prints requested vs filled prices, slippage,
latency and the raw return codes. Nothing about profit; this measures the
plumbing.

    python scripts/phase1_acceptance.py --symbol EURUSD

Refuses to run against a live account. Phase 8 is where real money starts, and
it starts with a checklist, not with a script anyone can run by accident.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import PACKAGE_ROOT, load_credentials, load_settings
from core.clock import LiveClock
from core.data_manager import DataManager
from core.errors import TradingSystemError
from core.mt5_connector import MT5Connector
from core.types import Direction, OrderRequest, OrderResult, Timeframe
from infra.logging import get_logger, setup_logging

log = get_logger("phase1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--sl-pips", type=float, default=20.0)
    parser.add_argument("--tp-pips", type=float, default=40.0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(
        level="DEBUG",
        log_dir=PACKAGE_ROOT / settings.logging.directory,
        filename="phase1_acceptance.jsonl",
    )

    connector = MT5Connector(settings.mt5, load_credentials(required=True))
    try:
        account = connector.connect()
        if not account.is_demo:
            print("REFUSING: this is a LIVE account. Phase 1 acceptance runs on demo only.")
            return 2

        symbol = args.symbol + settings.instruments.symbol_suffix
        spec = connector.spec(symbol)
        print(f"\n{spec.describe()}\n")

        # 1. Data --------------------------------------------------------
        manager = DataManager(connector, settings.data, LiveClock(connector.server_offset))
        for tf in (Timeframe.parse(name) for name in settings.data.timeframes):
            series = manager.get_series(symbol, tf)
            print(
                f"  data  {tf.value:<4} {len(series):>5} closed bars, "
                f"last {series.last_bar_time:%Y-%m-%d %H:%M} @ {series.last_close}"
            )

        # 2. Order -------------------------------------------------------
        tick = connector.tick(symbol)
        entry = tick.ask
        request = OrderRequest(
            symbol=symbol,
            direction=Direction.LONG,
            volume=spec.volume_min,
            sl=spec.normalize_price(entry - spec.pips_to_price(args.sl_pips)),
            tp=spec.normalize_price(entry + spec.pips_to_price(args.tp_pips)),
            reference_price=entry,
            deviation_points=settings.mt5.deviation_points,
            magic=settings.system.magic_number,
            comment="phase1-acceptance",
        )
        result = connector.order_send(request, spec)
        _print_result("ENTRY", result)
        if not result.ok:
            return 1

        # 3. Verify the broker kept our stops ----------------------------
        time.sleep(args.hold_seconds)
        positions = connector.positions(symbol=symbol, magic=settings.system.magic_number)
        if not positions:
            print("  MISMATCH: order reported DONE but no position is open.")
            return 1

        position = positions[0]
        sl_drift = spec.price_to_pips(position.sl - request.sl)
        tp_drift = spec.price_to_pips(position.tp - request.tp)
        print(
            f"\n  position #{position.ticket}  volume {position.volume} "
            f"(requested {request.volume})\n"
            f"    SL requested {request.sl} -> broker {position.sl}  "
            f"drift {sl_drift:+.2f} pips\n"
            f"    TP requested {request.tp} -> broker {position.tp}  "
            f"drift {tp_drift:+.2f} pips"
        )
        if position.volume != request.volume:
            print("  MISMATCH: filled volume differs from the requested volume.")

        # 4. Close -------------------------------------------------------
        close = connector.close_position(position)
        _print_result("EXIT", close)

        remaining = connector.positions(symbol=symbol, magic=settings.system.magic_number)
        print(
            "\n  reconciliation: "
            + ("clean, no positions left." if not remaining else f"STILL OPEN: {remaining}")
        )
        return 0 if close.ok and not remaining else 1

    except TradingSystemError as exc:
        log.error("acceptance run failed", exc_info=True)
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        connector.shutdown()


def _print_result(label: str, r: OrderResult) -> None:
    """Every number here is one the Phase 8 execution report will need."""
    print(
        f"\n  {label}: {r.retcode_name} ({r.retcode})  attempts={r.attempts}\n"
        f"    requested {r.requested_price} -> filled {r.filled_price}  "
        f"slippage {r.slippage_pips:+.2f} pips\n"
        f"    volume {r.filled_volume} / {r.requested_volume}   "
        f"latency {r.latency_ms:.0f} ms   spread {r.spread_at_send}\n"
        f"    broker said: {r.comment!r}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
