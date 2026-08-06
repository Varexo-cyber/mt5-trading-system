"""Ask for a reading of the whole account.

    python scripts/analyst.py                 # the last 24 hours
    python scripts/analyst.py --hours 72
    python scripts/analyst.py --dry-run       # show the evidence, spend nothing

Advisory only. Nothing this prints is executed, and the trading service does
not read its output — see `analyst/__init__.py` for why that separation is the
point rather than a limitation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst import analyse, gather
from config.loader import load_settings
from infra.atomic import write_json_atomic
from promotion.experimental import apply_experimental_live_limits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0, help="window to review")
    parser.add_argument("--trades", type=int, default=30, help="most recent closed trades")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument("--model", default="", help="override the model")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the evidence that would be sent and stop; costs nothing",
    )
    args = parser.parse_args(argv)

    settings = apply_experimental_live_limits(
        load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    )
    evidence = gather(
        ROOT / args.db,
        settings,
        window_hours=args.hours,
        max_trades=args.trades,
        health_path=ROOT / "runtime" / "position_health.json",
    )

    if args.dry_run:
        print(json.dumps(evidence.as_payload(), indent=2))
        return 0

    kwargs = {"model": args.model} if args.model else {}
    assessment = analyse(evidence, **kwargs)  # type: ignore[arg-type]
    print(assessment.render())

    # Written so the deck can show the latest reading without paying again.
    with contextlib.suppress(OSError):
        write_json_atomic(ROOT / "runtime" / "analyst.json", assessment.as_dict())
    return 0 if assessment.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
