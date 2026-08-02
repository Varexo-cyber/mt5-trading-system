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
    parser.add_argument("--account", required=True, type=int)
    parser.add_argument("--risk-percent", required=True, type=float)
    parser.add_argument("--drawdown-percent", required=True, type=float)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    if args.confirm != EXPERIMENTAL_LIVE_PHRASE:
        raise RuntimeError(f"confirmation must be exactly: {EXPERIMENTAL_LIVE_PHRASE}")
    if abs(args.risk_percent - EXPERIMENTAL_RISK_PER_TRADE_PCT) > 1e-9:
        raise RuntimeError("experimental risk must be exactly 1%")
    if abs(args.drawdown_percent - EXPERIMENTAL_MAX_DRAWDOWN_PCT) > 1e-9:
        raise RuntimeError("experimental drawdown stop must be exactly 15%")

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
    finally:
        connector.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
