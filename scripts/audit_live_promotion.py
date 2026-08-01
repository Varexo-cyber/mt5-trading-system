"""Print live-promotion evidence and optionally arm the connected live account."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from promotion.audit import PromotionAudit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true")
    args = parser.parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    audit = PromotionAudit(ROOT, settings)
    checks = audit.run()
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    if not all(check.passed for check in checks):
        print("LIVE remains locked.")
        return 2
    if not args.arm:
        print("All gates pass. Re-run with --arm to bind approval to the current live account.")
        return 0

    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        account = connector.connect()
        if account.is_demo:
            raise RuntimeError("arming requires the intended live MT5 account")
        path = ROOT / "runtime" / "LIVE_ARMED.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "login": account.login,
                    "phrase": "I_ACCEPT_LIVE_RISK",
                    "armed_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Armed live account {account.login}.")
    finally:
        connector.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
