"""What execution actually costs on this account, and the stop floor it implies.

A live USDCHF short closed at -1.32R. A full stop-out is -1.00R by definition,
so that trade overshot its own risk model by a third — and the risk model is
what every other rule here is denominated in. The give-back arms at 0.5R, the
profit lock secures 0.46R, the health engine measures drift in ATR: all of it
is arithmetic on an R that the exit can overshoot by 32%.

The cause is not mysterious and it is already recorded. Every fill writes its
slippage to `order_attempts`, and on a 1.5-pip stop half a pip of slip is a
third of the risk. The same half pip on a 10-pip stop is five percent and
nobody would notice.

So the useful number is not slippage in pips. It is **slippage as a share of
the stop it happened on**, per instrument, measured rather than assumed. That
is what this prints, along with the stop floor implied by wanting execution to
stay a small part of the risk.

    python scripts/execution_noise.py
    python scripts/execution_noise.py --share 0.10   # stricter budget
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Share of the stop that execution may consume before R stops meaning
#: anything. At 15% a full stop-out lands between -1.0R and -1.15R, which the
#: statistics can carry; at a third it cannot, and every threshold expressed in
#: R is being applied to a number the market can move on its own.
DEFAULT_MAX_SHARE = 0.15


def rows(db: sqlite3.Connection, hours: float):  # type: ignore[no-untyped-def]
    """Accepted fills with the stop of the trade they belong to.

    Joined rather than read separately: slippage on its own says nothing, and
    the stop is the only thing that turns it into a number worth acting on.
    Rejected attempts are excluded — they have no fill and no slippage.
    """
    return db.execute(
        """
        SELECT a.symbol            AS symbol,
               a.kind              AS kind,
               a.slippage_pips     AS slip,
               t.sl_distance_pips  AS stop_pips
        FROM order_attempts a
        LEFT JOIN trades t ON t.id = a.trade_id
        WHERE a.ok = 1
          AND a.ts >= datetime('now', ?)
        ORDER BY a.id DESC
        """,
        (f"-{hours} hours",),
    ).fetchall()


def report(records, max_share: float) -> None:  # type: ignore[no-untyped-def]
    if not records:
        print("\nNo accepted fills recorded in this window.\n")
        return

    by_symbol: dict[str, list[tuple[float, float | None]]] = {}
    for row in records:
        slip = abs(float(row["slip"] or 0.0))
        stop = float(row["stop_pips"]) if row["stop_pips"] else None
        by_symbol.setdefault(str(row["symbol"]), []).append((slip, stop))

    print(f"\n{'=' * 78}")
    print(f"  EXECUTION NOISE — {len(records)} fills")
    print(f"{'=' * 78}\n")
    print(
        f"  {'market':<12}{'fills':>6}{'slip med':>10}{'slip max':>10}"
        f"{'stop':>8}{'worst as % of stop':>21}"
    )
    print(f"  {'-' * 74}")

    worst_overall = 0.0
    for symbol, entries in sorted(by_symbol.items()):
        slips = [slip for slip, _ in entries]
        stops = [stop for _, stop in entries if stop]
        worst = max(slips)
        worst_overall = max(worst_overall, worst)
        typical_stop = median(stops) if stops else None
        share = (worst / typical_stop) if typical_stop else None
        flag = "  <-- R is fiction here" if share and share > max_share else ""
        print(
            f"  {symbol:<12}{len(entries):>6}{median(slips):>10.2f}{worst:>10.2f}"
            + (f"{typical_stop:>8.1f}" if typical_stop else f"{'—':>8}")
            + (f"{share:>20.0%}" if share is not None else f"{'—':>20}")
            + flag
        )

    print(f"\n{'-' * 78}")
    print("  WHAT THIS IMPLIES FOR THE STOP FLOOR")
    print(f"{'-' * 78}\n")
    print(f"  Budget: execution may be at most {max_share:.0%} of the stop.\n")
    for symbol, entries in sorted(by_symbol.items()):
        worst = max(slip for slip, _ in entries)
        if worst <= 0:
            print(f"  {symbol:<12} no measurable slippage yet — leave the floor alone")
            continue
        floor = worst / max_share
        print(f"  {symbol:<12} worst slip {worst:.2f} pips  ->  stop floor {floor:.1f} pips")

    if worst_overall > 0:
        print(
            f"\n  Across everything: {worst_overall / max_share:.1f} pips is the floor that keeps "
            f"a stop-out\n  inside {1 + max_share:.2f}R instead of overshooting it."
        )
    print(
        "\n  Measured, not assumed. Every number here comes from fills this account\n"
        "  actually received. A floor set above these is caution; one set below is\n"
        "  a risk model that the exit can break on its own.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=168.0, help="how far back to look")
    parser.add_argument(
        "--share",
        type=float,
        default=DEFAULT_MAX_SHARE,
        help="share of the stop execution may consume",
    )
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    args = parser.parse_args(argv)

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1
    if not 0.0 < args.share < 1.0:
        print("--share must be between 0 and 1.")
        return 1

    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        report(rows(db, args.hours), args.share)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
