"""Print exactly why the broker refused each order.

Every rejection already records the broker's return code, its comment, the
volume and the stops. All of it goes to the JSON log, which nobody reads while
a console repeats "order rejected" and says nothing.

    python scripts/why_rejected.py
    python scripts/why_rejected.py --limit 30

The return code is the whole answer. INVALID_VOLUME means the size is not one
the broker accepts for that instrument; INVALID_STOPS means the stop sits closer
than it allows; MARKET_CLOSED means orders are not accepted right now even
though quotes still tick; TRADE_DISABLED means the instrument cannot be opened
at all on this account.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_INTERESTING = {"order_rejected", "order_attempt", "order_failed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=15, help="most recent rejections to show")
    parser.add_argument("--log", default="logs/trading.jsonl", help="log file to read")
    args = parser.parse_args(argv)

    path = ROOT / args.log
    if not path.exists():
        print(f"No log at {path}. Start jarvis.py first.")
        return 1

    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or '"order_' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written final line
            if row.get("event") in _INTERESTING:
                rows.append(row)

    rejections = [r for r in rows if r.get("event") == "order_rejected"]
    if not rejections:
        attempts = sum(1 for r in rows if r.get("event") == "order_attempt")
        print(f"\nNo rejections recorded. {attempts} order attempts in this log.\n")
        return 0

    counts = Counter(str(r.get("retcode_name", "?")) for r in rejections)
    print(f"\n{len(rejections)} rejected orders\n")
    print("by broker return code")
    for name, count in counts.most_common():
        print(f"  {count:5}  {name}")

    print(f"\nmost recent {min(args.limit, len(rejections))}")
    header = f"  {'time':21} {'symbol':12} {'code':22} {'volume':>8} {'sl':>10} {'tp':>10}"
    print(header)
    for row in rejections[-args.limit :]:
        volume = row.get("volume")
        stop = row.get("sl")
        target = row.get("tp")
        print(
            f"  {str(row.get('timestamp', row.get('time', '')))[:21]:21} "
            f"{row.get('symbol', '?')!s:12} "
            f"{row.get('retcode_name', '?')!s:22} "
            f"{('-' if volume is None else f'{float(volume):.2f}'):>8} "
            f"{('-' if stop is None else f'{float(stop):.5f}'):>10} "
            f"{('-' if target is None else f'{float(target):.5f}'):>10}"
        )

    comments = {str(r.get("broker_comment", "")).strip() for r in rejections}
    comments.discard("")
    if comments:
        print("\nbroker comments")
        for comment in sorted(comments):
            print(f"  {comment}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
