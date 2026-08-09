"""Where this account makes money and where it loses it, by bucket.

The journal records everything and nothing ever asks it the one question a
trader answers about themselves within a month: *what am I bad at?* A person
learns "I am hopeless on gold, I am fine on EURUSD in the London morning" and
stops doing the first thing. This system has all the data to know that and has
never once looked.

Two halves, and the second is the more valuable:

**What we took.** Closed trades bucketed by instrument, by asset class, by the
hour they were opened, and by what closed them. Net R and how much of the peak
survived, per bucket.

**What we refused.** Every gate that blocked a setup, against what that setup
went on to do — the journal shadows blocked trades and resolves them later, so
the counterfactual is recorded rather than imagined. This is the only honest
way to ask whether a gate earns its keep, including Claude's veto. A gate whose
blocked trades would have made money is costing you.

Read-only, no API calls, nothing written anywhere.

    python scripts/scorecard.py                # last 30 days
    python scripts/scorecard.py --days 7
    python scripts/scorecard.py --min-sample 5 # only buckets worth reading

A bucket with three trades in it is an anecdote. `--min-sample` exists because
the temptation to act on one is strong and the cost of doing so is a system
tuned to noise.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Sessions in UTC, matching filters.session so a bucket here means the same
#: thing as a window there. Overlaps resolve to the busier session.
SESSIONS = (("asia", 0, 7), ("london", 7, 12), ("overlap", 12, 16), ("newyork", 16, 21))


def session_of(hour: int) -> str:
    for name, start, end in SESSIONS:
        if start <= hour < end:
            return name
    return "rollover"


@dataclass
class Bucket:
    """One slice of the book, and what it did."""

    name: str
    trades: int = 0
    wins: int = 0
    net_r: float = 0.0
    money: float = 0.0
    kept: list[float] = field(default_factory=list)

    def add(self, pnl_r: float, pnl_money: float, peak_r: float | None) -> None:
        self.trades += 1
        self.wins += 1 if pnl_r > 0 else 0
        self.net_r += pnl_r
        self.money += pnl_money
        if peak_r and peak_r > 0:
            self.kept.append(pnl_r / peak_r)

    @property
    def median_kept(self) -> float | None:
        return sorted(self.kept)[len(self.kept) // 2] if self.kept else None

    def row(self) -> str:
        kept = self.median_kept
        return (
            f"  {self.name:<22}{self.trades:>7}{self.wins:>6}"
            f"{self.net_r:>+9.2f}R{self.money:>+9.2f}"
            + (f"{kept:>8.0%}" if kept is not None else f"{'—':>8}")
        )


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def taken(db: sqlite3.Connection, since: datetime) -> dict[str, dict[str, Bucket]]:
    """Closed trades, sliced every way the journal can support."""
    rows = db.execute(
        "SELECT symbol, direction, pnl_r, pnl_money, mfe_r, exit_reason, opened_at "
        "FROM trades WHERE closed_at IS NOT NULL AND closed_at >= ? AND pnl_r IS NOT NULL",
        (since.isoformat(),),
    ).fetchall()

    slices: dict[str, dict[str, Bucket]] = {
        "instrument": {},
        "session": {},
        "direction": {},
        "what closed it": {},
    }

    def into(slice_name: str, key: str, row: sqlite3.Row) -> None:
        bucket = slices[slice_name].setdefault(key, Bucket(key))
        bucket.add(float(row["pnl_r"]), float(row["pnl_money"] or 0.0), row["mfe_r"])

    for row in rows:
        into("instrument", str(row["symbol"]), row)
        into("direction", str(row["direction"]), row)
        into("what closed it", str(row["exit_reason"] or "unknown"), row)
        try:
            hour = datetime.fromisoformat(str(row["opened_at"])).astimezone(UTC).hour
        except ValueError:
            hour = -1
        into("session", session_of(hour) if hour >= 0 else "unknown", row)
    return slices


def refused(db: sqlite3.Connection, since: datetime) -> dict[str, Bucket]:
    """What each gate blocked, and what those setups went on to do.

    The journal shadows a blocked setup and resolves it against later price, so
    this is recorded rather than imagined. A gate whose blocked trades would
    have made money is costing you, and there is no other way to find out.
    """
    rows = db.execute(
        "SELECT blocked_by, outcome, pnl_r FROM shadow_trades "
        "WHERE opened_at >= ? AND pnl_r IS NOT NULL AND outcome IS NOT NULL",
        (since.isoformat(),),
    ).fetchall()
    gates: dict[str, Bucket] = {}
    for row in rows:
        gate = gates.setdefault(str(row["blocked_by"]), Bucket(str(row["blocked_by"])))
        gate.add(float(row["pnl_r"]), 0.0, None)
    return gates


def unresolved(db: sqlite3.Connection, since: datetime) -> int:
    row = db.execute(
        "SELECT COUNT(*) AS n FROM shadow_trades WHERE opened_at >= ? AND pnl_r IS NULL",
        (since.isoformat(),),
    ).fetchone()
    return int(row["n"]) if row else 0


def show(title: str, buckets: dict[str, Bucket], minimum: int) -> None:
    worth_reading = [b for b in buckets.values() if b.trades >= minimum]
    if not worth_reading:
        return
    print()
    print(f"  {title.upper()}")
    print(f"  {'':22}{'trades':>7}{'won':>6}{'net':>10}{'money':>9}{'kept':>8}")
    print("  " + "-" * 62)
    for bucket in sorted(worth_reading, key=lambda b: b.net_r):
        print(bucket.row())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30.0, help="window to report on")
    parser.add_argument("--min-sample", type=int, default=1, help="hide thinner buckets")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    args = parser.parse_args(argv)

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1

    since = datetime.now(UTC) - timedelta(days=args.days)
    db = connect(path)
    try:
        slices = taken(db, since)
        gates = refused(db, since)
        pending = unresolved(db, since)
    finally:
        db.close()

    total = sum(bucket.trades for bucket in slices["instrument"].values())
    print()
    print("=" * 72)
    print(f"  SCORECARD — last {args.days:g} days, {total} closed trades")
    print("=" * 72)

    if not total:
        print("\n  Nothing closed in this window.\n")
    for title, buckets in slices.items():
        show(title, buckets, args.min_sample)

    if gates:
        print()
        print("  WHAT THE GATES REFUSED, AND WHAT IT WOULD HAVE DONE")
        print("  A gate whose blocked trades would have made money is costing you.")
        print(f"  {'':22}{'blocked':>7}{'won':>6}{'net':>10}")
        print("  " + "-" * 45)
        for gate in sorted(gates.values(), key=lambda b: -b.net_r):
            if gate.trades < args.min_sample:
                continue
            verdict = "cost us" if gate.net_r > 0 else "saved us"
            print(
                f"  {gate.name:<22}{gate.trades:>7}{gate.wins:>6}{gate.net_r:>+9.2f}R"
                f"   {verdict} {abs(gate.net_r):.2f}R"
            )
    if pending:
        print(f"\n  {pending} blocked setup(s) not yet resolved and excluded.")

    print()
    if total < 30:
        print(f"  {total} trades. Not a sample — read this for shape, not for decisions.")
        print("  Thirty is where a bucket starts to mean something, and even then only")
        print("  the buckets with several trades of their own in them.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
