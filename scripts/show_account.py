"""Print what the connected terminal reports, and what this build requires.

    python scripts/show_account.py
    python scripts/show_account.py --login-only

Exists so re-arming does not require the operator to know or retype their own
account number, and so the two figures that have to agree — the risk the
contract was armed at and the risk this build enforces — are visible side by
side before anything is signed.

`--login-only` prints the bare login and nothing else, for scripting.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from promotion.experimental import (
    CONTRACT_VERSION,
    EXPERIMENTAL_MAX_DRAWDOWN_PCT,
    EXPERIMENTAL_MAX_STAKE_PCT,
    EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT,
    EXPERIMENTAL_RISK_PER_TRADE_PCT,
    ExperimentalLiveContract,
    contract_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-only", action="store_true", help="print just the login")
    args = parser.parse_args(argv)

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        account = connector.connect()
    except Exception as exc:  # noqa: BLE001 - the caller only needs the reason
        if not args.login_only:
            print(f"Could not connect to MT5: {type(exc).__name__}: {exc}")
        return 1
    finally:
        # A failing shutdown must not mask the account read that just succeeded.
        with contextlib.suppress(Exception):
            connector.shutdown()

    if args.login_only:
        print(account.login)
        return 0

    print(f"  account   {account.login} on {account.server}")
    print(f"  equity    {account.equity:.2f} {account.currency}")
    print(f"  type      {'DEMO' if account.is_demo else 'REAL MONEY'}")
    print()
    print(
        f"  this build requires  risk {EXPERIMENTAL_RISK_PER_TRADE_PCT:g}% ordinary up to "
        f"{EXPERIMENTAL_MAX_STAKE_PCT:g}% on conviction, "
        f"{EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT:g}% total open, "
        f"drawdown stop {EXPERIMENTAL_MAX_DRAWDOWN_PCT:g}%"
    )

    try:
        contract = ExperimentalLiveContract.load(contract_path(ROOT))
    except RuntimeError:
        print("  currently armed      nothing")
        return 0
    print(
        f"  currently armed      account {contract.login}, risk "
        f"{contract.risk_per_trade_pct:g}% up to {contract.max_stake_pct:g}%, "
        f"{contract.max_total_open_risk_pct:g}% total open, "
        f"drawdown stop {contract.max_drawdown_pct:g}%"
    )
    mismatched = (
        contract.login != account.login
        or contract.version != CONTRACT_VERSION
        or abs(contract.risk_per_trade_pct - EXPERIMENTAL_RISK_PER_TRADE_PCT) > 1e-9
        or abs(contract.max_stake_pct - EXPERIMENTAL_MAX_STAKE_PCT) > 1e-9
        or abs(contract.max_total_open_risk_pct - EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT) > 1e-9
        or abs(contract.max_drawdown_pct - EXPERIMENTAL_MAX_DRAWDOWN_PCT) > 1e-9
    )
    print()
    print(
        "  -> the armed contract does not match this build; re-arming is required"
        if mismatched
        else "  -> armed contract matches this build"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
