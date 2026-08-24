"""What the journal says each detector's weight should be.

Reads only. It prints what it would propose and, with --write, saves a
candidate config for `scripts/config_control.py --start-shadow` to run
alongside the live one. Nothing here changes what is trading.

    scripts/propose_weights.py                 what the record says
    scripts/propose_weights.py --days 14       a shorter memory
    scripts/propose_weights.py --write out.yaml    a candidate to shadow-test
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from learning.weight_proposal import proposals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument("--overlay", default="config/eightcap.yaml", help="live overlay")
    parser.add_argument("--minimum-trades", type=int, default=30, help="before judging a module")
    parser.add_argument("--window", type=int, default=400, help="most recent trades per module")
    parser.add_argument("--write", default="", help="save a candidate overlay here")
    args = parser.parse_args(argv)

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1

    settings = load_settings(DEFAULT_CONFIG_PATH, overlay=args.overlay, env_overrides=False)
    weights = dict(settings.analysis.confluence.weights)

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        report = proposals(conn, weights, minimum_trades=args.minimum_trades, window=args.window)

    width = max(len(item.module) for item in report)
    print(f"\n  WHAT EACH DETECTOR EARNED, AND WHAT THAT IS WORTH\n  {'-' * 74}")
    print(f"  {'module':<{width}}  {'trades':>6} {'R/trade':>9} {'now':>6} {'->':>6}   why")
    for item in report:
        arrow = f"{item.proposed:g}" if item.changed else ""
        mean = f"{item.evidence.mean_r:+.3f}" if item.evidence.trades else "-"
        print(
            f"  {item.module:<{width}}  {item.evidence.trades:>6} {mean:>9} "
            f"{item.current:>6g} {arrow:>6}   {item.why}"
        )

    movers = [item for item in report if item.changed]
    print()
    if not movers:
        print("  Nothing has a decided record yet. No proposal.\n")
        return 0
    print(f"  {len(movers)} of {len(report)} modules have a record decided enough to act on.\n")

    if args.write:
        overlay = yaml.safe_load((ROOT / args.overlay).read_text(encoding="utf-8"))
        for item in movers:
            overlay["analysis"]["confluence"]["weights"][item.module] = item.proposed
        Path(args.write).write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
        print(f"  Candidate written to {args.write}. Shadow-test it before it goes anywhere:")
        print(f"    scripts/config_control.py --start-shadow {args.write}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
