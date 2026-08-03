"""Find out how this MetaTrader5 build wants a trade request passed.

Every order came back with no result at all and `[-2] Unnamed arguments not
allowed`. That is not the broker refusing a trade — it is the MetaTrader5 Python
extension refusing the *call*, before anything reaches Eightcap. The payload has
been checked and every value is already an exact `int`, `float` or `str`, so
what is left is the calling convention itself, and that varies between builds of
the package.

This asks the installed one directly. It uses `order_check`, which validates a
request against the account and returns margin and balance figures — it never
places, modifies or closes anything, so it is safe to run against the live
account.

    python scripts/probe_order_api.py
    python scripts/probe_order_api.py --symbol XAUUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_codes import OrderTime, OrderType, TradeAction
from core.mt5_connector import MT5Connector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD.i")
    parser.add_argument(
        "--send",
        action="store_true",
        help="actually place one minimum-size order after the checks pass. REAL MONEY.",
    )
    args = parser.parse_args(argv)

    load_dotenv(ROOT / "config" / ".env", override=False)
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        account = connector.connect()
        mt5 = connector.mt5
        print(f"\nMetaTrader5 package: {getattr(mt5, '__version__', 'unknown')}")
        print(f"account {account.login} @ {account.server}, equity {account.equity:.2f}\n")

        # order_check validates a request; order_send additionally requires
        # permission to trade programmatically. If these are not all True the
        # request is fine and the terminal still refuses to send it, which is
        # exactly the shape of the failure being chased: no result, no retcode,
        # nothing ever reaching the broker.
        terminal = mt5.terminal_info()
        info = mt5.account_info()
        print("permission to send orders")
        print(f"  terminal AutoTrading button   : {getattr(terminal, 'trade_allowed', '?')}")
        print(f"  account allows trading        : {getattr(info, 'trade_allowed', '?')}")
        print(f"  account allows expert/API     : {getattr(info, 'trade_expert', '?')}")
        print(f"  terminal connected            : {getattr(terminal, 'connected', '?')}")
        print()

        spec = connector.spec(args.symbol)
        tick = connector.tick(args.symbol)
        price = tick.ask
        # A stop far enough away that the request is valid on its own merits;
        # this probe is about the call, not about whether the setup is good.
        stop = spec.normalize_price(price - spec.pip_size * 50)

        payload = {
            "action": int(TradeAction.DEAL),
            "symbol": args.symbol,
            "volume": float(spec.volume_min),
            "type": int(OrderType.BUY),
            "price": float(price),
            "sl": float(stop),
            "tp": 0.0,
            "deviation": 10,
            "magic": int(settings.system.magic_number),
            "comment": "probe",
            "type_time": int(OrderTime.GTC),
            "type_filling": int(spec.preferred_filling()),
        }
        print(
            f"{args.symbol}: min volume {spec.volume_min}, step {spec.volume_step}, "
            f"digits {spec.digits}, stops_level {spec.stops_level}, "
            f"filling mask {spec.filling_mode_mask} -> {spec.preferred_filling().name}"
        )
        print(
            f"probe request: volume {payload['volume']}, price {payload['price']}, "
            f"sl {payload['sl']}\n"
        )

        # The runner does not build the dict by hand — it goes through
        # _build_deal_payload. Checking that exact object rules out a difference
        # between what this probe sends and what production sends.
        from core.types import Direction, OrderRequest

        real_request = OrderRequest(
            symbol=args.symbol,
            direction=Direction.LONG,
            volume=float(spec.volume_min),
            sl=float(stop),
            tp=0.0,
            reference_price=float(price),
            deviation_points=settings.mt5.deviation_points,
            magic=settings.system.magic_number,
            comment="jarvis-exp-live",
        )
        real_payload = connector._build_deal_payload(real_request, spec, price)
        differences = {
            key: (payload.get(key), value)
            for key, value in real_payload.items()
            if payload.get(key) != value or type(payload.get(key)) is not type(value)
        }
        print(f"production payload differs from the probe's in: {differences or 'nothing'}\n")

        for label, call in (
            ("order_check(probe payload)     ", lambda: mt5.order_check(payload)),
            ("order_check(production payload)", lambda: mt5.order_check(real_payload)),
        ):
            try:
                result = call()
            except Exception as exc:  # noqa: BLE001 - the point is to see what breaks
                print(f"  {label} raised {type(exc).__name__}: {exc}")
                continue
            if result is None:
                print(f"  {label} returned None, last_error={mt5.last_error()}")
            else:
                print(
                    f"  {label} OK  retcode={getattr(result, 'retcode', '?')} "
                    f"comment={getattr(result, 'comment', '')!r} "
                    f"margin={getattr(result, 'margin', '?')}"
                )
        if not args.send:
            print(
                "\nEverything above can pass while order_send still fails, because"
                "\norder_check only validates — it never asks to trade. Re-run with"
                "\n  --send   to place ONE order of the minimum size with real money"
                "\nand see exactly what order_send returns.\n"
            )
            return 0

        print("\n--send given: placing ONE order of the minimum size with REAL MONEY.")
        answer = input(f"Type the symbol '{args.symbol}' to confirm: ").strip()
        if answer != args.symbol:
            print("Not confirmed. Nothing was sent.\n")
            return 1

        before = mt5.last_error()
        result = mt5.order_send(real_payload)
        after = mt5.last_error()
        print(f"\n  last_error before : {before}")
        print(f"  order_send returned: {result}")
        print(f"  last_error after  : {after}")
        if result is None:
            print(
                "\nNothing came back. Since order_check passed on this exact request,"
                "\nthe request is not the problem and the two error readings above say"
                "\nwhether order_send failed or simply read back a stale message.\n"
            )
        else:
            print(f"\n  retcode {result.retcode}: {result.comment}")
            if result.retcode in (10008, 10009):
                print("  ORDER PLACED. Check the Positions tab.\n")
            else:
                print("  Refused by the broker, and now with a real code to act on.\n")
    finally:
        connector.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
