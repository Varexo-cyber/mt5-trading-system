"""Operator-approved config snapshots, shadow tests and rollback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.config_control import ConfigControl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", metavar="LABEL")
    group.add_argument("--start-shadow", type=Path, metavar="CANDIDATE_YAML")
    group.add_argument("--promote-shadow", action="store_true")
    group.add_argument("--restore", type=Path, metavar="SNAPSHOT_YAML")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    control = ConfigControl(ROOT)
    if args.snapshot:
        print(control.snapshot(args.snapshot))
    elif args.start_shadow:
        print(control.start_shadow(args.start_shadow))
    elif args.promote_shadow:
        if args.confirm != "APPROVE_SHADOW":
            raise SystemExit("--confirm APPROVE_SHADOW is required")
        print(f"Promoted; rollback snapshot: {control.promote_shadow()}")
    elif args.restore:
        if args.confirm != "RESTORE_CONFIG":
            raise SystemExit("--confirm RESTORE_CONFIG is required")
        control.restore(args.restore)
        print("Configuration restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
