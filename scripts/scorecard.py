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


def conviction_band(score: float | None, threshold: float | None) -> str:
    """How far above its own bar the engine rated this setup.

    Measured against the threshold rather than in absolute points, because the
    threshold is a config value that has already moved once (55 to 40) and a
    fixed band would silently mean something different afterwards.
    """
    if score is None:
        return "unrecorded"
    bar = threshold if threshold and threshold > 0 else 40.0
    over = score - bar
    if over < 5:
        return "0-5 over the bar"
    if over < 10:
        return "5-10 over the bar"
    if over < 20:
        return "10-20 over the bar"
    return "20+ over the bar"


def score_band(score: float | None) -> str:
    """The raw confluence score, in fixed five-point bands.

    Deliberately absolute where `conviction_band` is deliberately relative, and
    the two answer opposite questions. Relative to the bar is right for "does
    being far above our own standard mean anything", and it becomes useless the
    moment the bar itself moves: drop the threshold from 40 to 35 and "0-5 over
    the bar" silently stops describing scores of 40-45 and starts describing
    35-40. Comparing before against after would then be comparing two different
    populations under one label.

    This is the slice that survives a threshold change, and the only one that
    can answer the question the change is being made to test: do the setups
    scoring 35-40 actually return less than the ones scoring 40-45?
    """
    if score is None:
        return "unrecorded"
    floor = int(score // 5 * 5)
    return f"score {floor}-{floor + 5}"


def taken(db: sqlite3.Connection, since: datetime) -> dict[str, dict[str, Bucket]]:
    """Closed trades, sliced every way the journal can support."""
    columns = (
        "SELECT t.symbol, t.direction, t.pnl_r, t.pnl_money, t.mfe_r, t.exit_reason, "
        "t.opened_at{extra} FROM trades t{join} "
        "WHERE t.closed_at IS NOT NULL AND t.closed_at >= ? AND t.pnl_r IS NOT NULL"
    )
    try:
        # The cycle that produced the trade carries the score the engine gave
        # it. Joined rather than assumed present: a journal predating the
        # analysis tables, or a hand-built one, must still produce a report.
        rows = db.execute(
            columns.format(
                extra=", c.total_score, c.score_threshold",
                join=" LEFT JOIN analysis_cycles c ON c.id = t.cycle_pk",
            ),
            (since.isoformat(),),
        ).fetchall()
        scored = True
    except sqlite3.OperationalError:
        rows = db.execute(columns.format(extra="", join=""), (since.isoformat(),)).fetchall()
        scored = False

    slices: dict[str, dict[str, Bucket]] = {
        "instrument": {},
        "session": {},
        "direction": {},
        "what closed it": {},
    }
    if scored:
        # Does the engine's own confidence predict anything? Nobody had asked.
        # A setup scoring 58.5 against a bar of 40 lost money on the same day a
        # 39.8 was refused, and "hold the ones we are sure about" is only a
        # strategy if being sure means something here.
        slices["how sure the engine was"] = {}
        # Absolute bands beside the relative ones, so lowering the threshold
        # can be judged instead of merely done. Without this the report
        # relabels itself the moment the bar moves and the comparison the move
        # exists to make becomes impossible to draw.
        slices["what the raw score was"] = {}

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
        if scored:
            into(
                "how sure the engine was",
                conviction_band(row["total_score"], row["score_threshold"]),
                row,
            )
            into("what the raw score was", score_band(row["total_score"]), row)
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


def intervened(db: sqlite3.Connection, since: datetime) -> list[sqlite3.Row]:
    """Every rule that closed a trade early, beside what holding would have paid.

    `management_baselines` has been filled since the resolver was written and
    read by nothing, which is how "AI_CLOSE is nought for eight" could sit in a
    report for a month with no way to tell whether those eight were rescues or
    mistakes. Nought for eight is not damning on its own: a rule that only ever
    fires on trades already going wrong shows a losing record while still
    losing less than doing nothing would have.

    The baseline settles it. It replays the untouched original stop and target
    over the same hours the trade really spanned, so `lift` is the honest
    question — did stepping in beat leaving it alone — and it is the only
    column here that can condemn or acquit an exit rule.
    """
    # A journal written before the baseline resolver existed simply has no such
    # table. This is a read-only report and must not die on an old database:
    # an absent measurement is a missing section, not a crash.
    try:
        return db.execute(
            """
        SELECT t.exit_reason AS name,
               COUNT(*) AS trades,
               AVG(b.actual_pnl_r) AS actual,
               AVG(b.baseline_pnl_r) AS baseline,
               AVG(b.lift_r) AS lift,
               SUM(CASE WHEN b.lift_r > 0 THEN 1 ELSE 0 END) AS better
        FROM management_baselines b
        JOIN trades t ON t.id = b.trade_id
        WHERE t.closed_at >= ? AND t.exit_reason IS NOT NULL
        GROUP BY t.exit_reason
        ORDER BY AVG(b.lift_r)
        """,
            (since.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


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
        interventions = intervened(db, since)
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
    if interventions:
        print()
        print("  DID STEPPING IN BEAT LEAVING IT ALONE")
        print("  Each early close replayed against its own untouched stop and target.")
        print(f"  {'':22}{'trades':>7}{'got':>9}{'holding':>9}{'lift':>9}{'better':>8}")
        print("  " + "-" * 66)
        for row in interventions:
            if int(row["trades"]) < args.min_sample:
                continue
            lift = float(row["lift"])
            print(
                f"  {row['name']!s:<22}{int(row['trades']):>7}"
                f"{float(row['actual']):>+8.2f}R{float(row['baseline']):>+8.2f}R"
                f"{lift:>+8.2f}R{int(row['better']):>5}/{int(row['trades'])}"
            )
        print()
        print("  A negative lift means the rule paid to do worse than nothing. A losing")
        print("  record with a positive lift is a rule earning its keep on bad trades.")

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
