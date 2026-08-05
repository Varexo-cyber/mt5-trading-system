"""Bind explicit experimental-live approval to the connected MT5 account."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from promotion.experimental import (
    EXPERIMENTAL_LIVE_PHRASE,
    EXPERIMENTAL_MAX_DRAWDOWN_PCT,
    EXPERIMENTAL_RISK_PER_TRADE_PCT,
    ExperimentalLiveContract,
    contract_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--account",
        required=True,
        type=int,
        help="broker login this approval is bound to; must match the connected terminal",
    )
    # Defaulted, not required. These may only ever equal the build's own
    # constants — the check below rejects anything else — so demanding them on
    # the command line was pure ceremony that could only be got wrong. It was:
    # the documented invocation omitted two required flags and simply failed,
    # leaving the operator unable to re-arm at all.
    parser.add_argument("--risk-percent", type=float, default=EXPERIMENTAL_RISK_PER_TRADE_PCT)
    parser.add_argument("--drawdown-percent", type=float, default=EXPERIMENTAL_MAX_DRAWDOWN_PCT)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"must be exactly: {EXPERIMENTAL_LIVE_PHRASE}",
    )
    args = parser.parse_args()

    if args.confirm != EXPERIMENTAL_LIVE_PHRASE:
        raise RuntimeError(f"confirmation must be exactly: {EXPERIMENTAL_LIVE_PHRASE}")
    # Name both numbers. These said "must be exactly 1%" long after the constant
    # moved to 2%, so the message quoted a figure nothing in the system used and
    # gave no way to see which of the two values was actually wrong.
    if abs(args.risk_percent - EXPERIMENTAL_RISK_PER_TRADE_PCT) > 1e-9:
        raise RuntimeError(
            f"--risk-percent was {args.risk_percent:g} but this build requires "
            f"{EXPERIMENTAL_RISK_PER_TRADE_PCT:g}. Omit the flag to use the correct value."
        )
    if abs(args.drawdown_percent - EXPERIMENTAL_MAX_DRAWDOWN_PCT) > 1e-9:
        raise RuntimeError(
            f"--drawdown-percent was {args.drawdown_percent:g} but this build requires "
            f"{EXPERIMENTAL_MAX_DRAWDOWN_PCT:g}. Omit the flag to use the correct value."
        )

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        account = connector.connect()
        if account.login != args.account:
            raise RuntimeError(
                f"connected account {account.login} does not match requested {args.account}"
            )
        contract = ExperimentalLiveContract.create(account)
        contract.write(contract_path(ROOT))
        print(
            f"EXPERIMENTAL LIVE armed for {account.login} on {account.server}: "
            f"risk {contract.risk_per_trade_pct:.1f}%, drawdown {contract.max_drawdown_pct:.1f}%, "
            f"equity floor {contract.equity_floor:.2f} {contract.currency}."
        )
        # Which of the two backstops will actually stop you, at today's equity.
        #
        # The floor is the deeper one and is rarely what bites; the peak-to-
        # current breaker measures from the all-time high and normally halts
        # trading well before it is reached. Printing the floor alone let it be
        # read as "the point at which I stop losing money", which is wrong by
        # however far apart the two numbers happen to be.
        room = account.equity * contract.max_drawdown_pct / 100.0
        print(
            f"  absolute floor    {contract.equity_floor:.2f} {contract.currency} — fixed, "
            "re-arming never moves it\n"
            f"  drawdown breaker  {contract.max_drawdown_pct:.1f}% below the all-time equity "
            f"peak. About {room:.2f} {contract.currency} of room from {account.equity:.2f} "
            "if today is the peak, and less if the peak is higher.\n"
            "  Whichever is reached first halts trading and needs a manual restart."
        )
    finally:
        connector.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
