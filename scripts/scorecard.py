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

    def row(self, width: int = 22) -> str:
        kept = self.median_kept
        return (
            f"  {self.name:<{width}}{self.trades:>7}{self.wins:>6}"
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


def taken(
    db: sqlite3.Connection,
    since: datetime,
    until: datetime | None = None,
    keep_regimes: frozenset[str] = frozenset(),
    drop_regimes: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Bucket]]:
    """Closed trades, sliced every way the journal can support.

    `keep_regimes` and `drop_regimes` restrict the book to the market shapes
    still being traded. Every per-detector number in this report was measured
    across the whole book, and when a regime is refused the book changes under
    it: a detector's average is the average of the markets it was allowed into.
    `--exclude-regime transition` re-reads the same four days as if the refusal
    had been in place all along, which is the only honest basis for deciding
    which detector to lean on next.
    """
    columns = (
        "SELECT t.symbol, t.direction, t.pnl_r, t.pnl_money, t.mfe_r, t.exit_reason, "
        "t.opened_at{extra} FROM trades t{join} "
        "WHERE t.closed_at IS NOT NULL AND t.closed_at >= ? AND t.pnl_r IS NOT NULL"
        "{bound}"
    )
    bound = " AND t.closed_at < ?" if until else ""
    window: tuple[str, ...] = (
        (since.isoformat(), until.isoformat()) if until else (since.isoformat(),)
    )
    try:
        # The cycle that produced the trade carries the score the engine gave
        # it. Joined rather than assumed present: a journal predating the
        # analysis tables, or a hand-built one, must still produce a report.
        rows = db.execute(
            columns.format(
                extra=", c.total_score, c.score_threshold, t.cycle_pk",
                join=" LEFT JOIN analysis_cycles c ON c.id = t.cycle_pk",
                bound=bound,
            ),
            window,
        ).fetchall()
        scored = True
    except sqlite3.OperationalError:
        rows = db.execute(columns.format(extra="", join="", bound=bound), window).fetchall()
        scored = False

    slices: dict[str, dict[str, Bucket]] = {
        # WHEN IT TURNED, before why.
        #
        # The report sliced the book six ways and never by the clock, so "it
        # worked Wednesday and Thursday morning and then stopped" could not be
        # checked — only felt. A day column and an hour column make the shape
        # visible in one run, and `--since` / `--until` then let the same
        # report be run over the good stretch and the bad one so every other
        # slice can be read side by side.
        "which day": {},
        "which hour it opened": {},
        "instrument": {},
        "session": {},
        "direction": {},
        "what closed it": {},
    }

    # WHICH DETECTOR WAS BEHIND THE MONEY, and under what regime.
    #
    # The journal has carried both since the beginning and nothing ever asked.
    # `module_scores` records every module's score, confidence and WEIGHT for
    # the cycle that produced the trade, and the cycle records the regime the
    # classifier read — so "where did the winners come from" is a join, not a
    # backtest.
    #
    # It matters more than the offline table does, because this is the account's
    # own money at its own sizes with its own costs, and because the offline
    # table cannot see how a detector behaved on the 845 markets this scans.
    modules_by_cycle: dict[int, list[tuple[str, float]]] = {}
    if scored:
        try:
            for row in db.execute(
                "SELECT m.cycle_pk, m.module, m.score FROM module_scores m "
                "JOIN trades t ON t.cycle_pk = m.cycle_pk "
                "WHERE m.weight > 0 AND m.score != 0 "
                "AND t.closed_at IS NOT NULL AND t.closed_at >= ?" + bound,
                window,
            ).fetchall():
                modules_by_cycle.setdefault(int(row["cycle_pk"]), []).append(
                    (str(row["module"]), float(row["score"]))
                )
            slices["which detector was behind it"] = {}
        except sqlite3.OperationalError:
            modules_by_cycle = {}
        # Fetched on its own rather than added to the join above, so a journal
        # without the column loses this one slice instead of every scored slice
        # in the report. An older journal must still produce a report.
        try:
            regimes = {
                int(row["id"]): str(row["volatility_regime"] or "unrecorded")
                for row in db.execute(
                    "SELECT id, volatility_regime FROM analysis_cycles"
                ).fetchall()
            }
            if regimes:
                slices["the regime at entry"] = {}
                # And the same regimes split by which way the trade faced.
                #
                # The regime slice alone can only recommend closing a regime
                # down, and closing `trend_down` down closes every short in a
                # falling market. A regime is not a verdict on a trade: it is
                # the market's shape, and a shape a LONG hates is often the
                # one a SHORT wants. Without this column the only available
                # action is a blanket refusal; with it a regime that loses on
                # one side and pays on the other can be halved instead.
                slices["the regime at entry, by side"] = {}
                if modules_by_cycle:
                    # And the detector crossed with the regime it fired in.
                    #
                    # "Which detector should I lean on" and "which market
                    # should I trade" are the same question asked twice, and
                    # neither slice on its own can answer it. A detector's
                    # average is the average of the markets it was let into: a
                    # module that pays in a trend and bleeds in a range shows
                    # up as mediocre in both single columns, and the action it
                    # deserves — let it fire only where it works — is invisible
                    # until the two are crossed.
                    slices["which detector, in which regime"] = {}
        except sqlite3.OperationalError:
            regimes = {}
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
        regime = regimes.get(int(row["cycle_pk"] or 0), "unrecorded") if scored else "unrecorded"
        # A trade whose regime was never recorded cannot be shown to be one of
        # the kept ones, so `--regime` drops it and `--exclude-regime` keeps
        # it. Both readings refuse to guess, in the direction the flag asked
        # for.
        if keep_regimes and regime not in keep_regimes:
            continue
        if regime in drop_regimes:
            continue
        into("instrument", str(row["symbol"]), row)
        into("direction", str(row["direction"]), row)
        into("what closed it", str(row["exit_reason"] or "unknown"), row)
        try:
            opened = datetime.fromisoformat(str(row["opened_at"])).astimezone(UTC)
            hour, day = opened.hour, opened.strftime("%a %d %b")
        except ValueError:
            hour, day = -1, "unknown"
        into("session", session_of(hour) if hour >= 0 else "unknown", row)
        into("which day", day, row)
        into("which hour it opened", f"{hour:02d}:00 UTC" if hour >= 0 else "unknown", row)
        if scored:
            into(
                "how sure the engine was",
                conviction_band(row["total_score"], row["score_threshold"]),
                row,
            )
            into("what the raw score was", score_band(row["total_score"]), row)
            if "the regime at entry" in slices:
                into("the regime at entry", regime, row)
                into("the regime at entry, by side", f"{regime} {row['direction']}", row)
            # A trade with three detectors behind it counts once in each of
            # their rows, so these columns do not add up to the book. That is
            # the same reading `backtest_modules.py` calls WHEN PRESENT, and it
            # is the honest one: a detector cannot be credited alone for a
            # decision three of them made.
            for module, score in modules_by_cycle.get(int(row["cycle_pk"] or 0), ()):
                if score * (1 if str(row["direction"]) == "LONG" else -1) > 0:
                    into("which detector was behind it", module, row)
                    if "which detector, in which regime" in slices:
                        into("which detector, in which regime", f"{module} in {regime}", row)
    return slices


def refused(
    db: sqlite3.Connection, since: datetime, until: datetime | None = None
) -> dict[str, Bucket]:
    """What each gate blocked, and what those setups went on to do.

    The journal shadows a blocked setup and resolves it against later price, so
    this is recorded rather than imagined. A gate whose blocked trades would
    have made money is costing you, and there is no other way to find out.
    """
    bound = " AND opened_at < ?" if until else ""
    window = (since.isoformat(), until.isoformat()) if until else (since.isoformat(),)
    rows = db.execute(
        "SELECT blocked_by, outcome, pnl_r FROM shadow_trades "
        "WHERE opened_at >= ? AND pnl_r IS NOT NULL AND outcome IS NOT NULL" + bound,
        window,
    ).fetchall()
    gates: dict[str, Bucket] = {}
    for row in rows:
        gate = gates.setdefault(str(row["blocked_by"]), Bucket(str(row["blocked_by"])))
        gate.add(float(row["pnl_r"]), 0.0, None)
    return gates


def intervened(
    db: sqlite3.Connection, since: datetime, until: datetime | None = None
) -> list[sqlite3.Row]:
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
        """
            + (" AND t.closed_at < ?" if until else "")
            + """
        GROUP BY t.exit_reason
        ORDER BY AVG(b.lift_r)
        """,
            (since.isoformat(), until.isoformat()) if until else (since.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def ratcheted(
    db: sqlite3.Connection, since: datetime, until: datetime | None = None
) -> list[sqlite3.Row]:
    """Did MOVING the stop pay, or did it only cut the plan short?

    `intervened` grades the rule that closed the trade. It cannot grade the
    rule that moved the stop, and that is the bigger question here: BROKER_SL
    was 49 of 90 closed trades over four days at a lift of -0.52R, better in
    15. But `BROKER_SL` names the broker filling a stop, not a decision — and
    the same label covers two entirely different trades:

      * the stop was never touched, so the plan simply failed. Baseline and
        actual are the same trade, lift is ~0 by construction, and the row is
        diluting the average toward nothing.
      * the stop was ratcheted up behind the price and THAT is what got hit.
        Here the baseline is a genuine counterfactual, and the lift is the
        price of securing early.

    Averaged together they answer nothing. `trades.sl` is written once at
    entry and never updated, so the baseline really is the plan as placed, and
    `management_actions` records every move with its old and new stop — the
    two halves can be told apart exactly.

    This is the account's stated priority measured against itself: "liever
    veilig stellen dan risico's" is a preference, and this is what it costs.
    """
    # 72 hours is the resolver's replay horizon, so a "would have recovered"
    # can be a hold this account would never have sat through, and holding a
    # loser also occupies risk budget this cannot see. The lift is evidence,
    # not a verdict.
    moved = (
        "SELECT trade_id, COUNT(*) AS moves, MIN(r_at_action) AS first_r "
        "FROM management_actions "
        "WHERE new_sl IS NOT NULL AND old_sl IS NOT NULL AND new_sl != old_sl "
        "GROUP BY trade_id"
    )
    try:
        return db.execute(
            "SELECT t.exit_reason AS name,"
            " CASE WHEN m.moves > 0 THEN 'stop moved' ELSE 'stop untouched' END AS ratchet,"
            " COUNT(*) AS trades, AVG(b.actual_pnl_r) AS actual,"
            " AVG(b.baseline_pnl_r) AS baseline, AVG(b.lift_r) AS lift,"
            " SUM(CASE WHEN b.lift_r > 0 THEN 1 ELSE 0 END) AS better,"
            " AVG(m.first_r) AS first_r"
            " FROM management_baselines b"
            " JOIN trades t ON t.id = b.trade_id"
            f" LEFT JOIN ({moved}) m ON m.trade_id = t.id"
            " WHERE t.closed_at >= ? AND t.exit_reason IS NOT NULL"
            + (" AND t.closed_at < ?" if until else "")
            + " GROUP BY name, ratchet ORDER BY AVG(b.lift_r)",
            (since.isoformat(), until.isoformat()) if until else (since.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def unresolved(db: sqlite3.Connection, since: datetime, until: datetime | None = None) -> int:
    bound = " AND opened_at < ?" if until else ""
    window = (since.isoformat(), until.isoformat()) if until else (since.isoformat(),)
    row = db.execute(
        "SELECT COUNT(*) AS n FROM shadow_trades WHERE opened_at >= ? AND pnl_r IS NULL" + bound,
        window,
    ).fetchone()
    return int(row["n"]) if row else 0


def show(title: str, buckets: dict[str, Bucket], minimum: int) -> None:
    worth_reading = [b for b in buckets.values() if b.trades >= minimum]
    if not worth_reading:
        return
    print()
    print(f"  {title.upper()}")
    if title == "which detector was behind it":
        print("  A trade with three detectors behind it counts once in each row,")
        print("  so these do not add up to the book. Read a row against the others.")
    # The crossed slices carry names like "ema_pullback_resume in trend_up",
    # which overran the fixed 22 and pushed every number out of its column —
    # so the one table built to be compared row against row was the one that
    # could not be. The width follows the longest name instead.
    width = max(22, max(len(b.name) for b in worth_reading))
    print(f"  {'':{width}}{'trades':>7}{'won':>6}{'net':>10}{'money':>9}{'kept':>8}")
    print("  " + "-" * (width + 40))
    # Time reads in order; everything else reads worst-first.
    chronological = title in ("which day", "which hour it opened")
    order = (lambda b: b.name) if chronological else (lambda b: b.net_r)
    for bucket in sorted(worth_reading, key=order):  # type: ignore[arg-type,return-value]
        print(bucket.row(width))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30.0, help="window to report on")
    parser.add_argument("--min-sample", type=int, default=1, help="hide thinner buckets")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument(
        "--since",
        default="",
        help="ISO instant to start from, e.g. 2026-08-21T00:00. Overrides --days, "
        "so the same report can be run over a good stretch and a bad one and "
        "every slice read side by side",
    )
    parser.add_argument("--until", default="", help="ISO instant to stop at. Open-ended without it")
    parser.add_argument(
        "--regime",
        action="append",
        default=[],
        metavar="NAME",
        help="report only trades opened in this regime; repeatable",
    )
    parser.add_argument(
        "--exclude-regime",
        action="append",
        default=[],
        metavar="NAME",
        help="drop trades opened in this regime, so a refusal added today can "
        "be read back over history as if it had always been there; repeatable",
    )
    args = parser.parse_args(argv)

    keep = frozenset(args.regime)
    drop = frozenset(args.exclude_regime)
    if keep & drop:
        print(f"a regime cannot be both kept and excluded: {', '.join(sorted(keep & drop))}")
        return 1

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1

    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"--since is not an ISO instant: {args.since!r}")
            return 1
        since = since if since.tzinfo else since.replace(tzinfo=UTC)
    else:
        since = datetime.now(UTC) - timedelta(days=args.days)
    until: datetime | None = None
    if args.until:
        try:
            until = datetime.fromisoformat(args.until)
        except ValueError:
            print(f"--until is not an ISO instant: {args.until!r}")
            return 1
        until = until if until.tzinfo else until.replace(tzinfo=UTC)
        if until <= since:
            print("--until must be after --since")
            return 1
    db = connect(path)
    try:
        slices = taken(db, since, until, keep_regimes=keep, drop_regimes=drop)
        gates = refused(db, since, until)
        pending = unresolved(db, since, until)
        interventions = intervened(db, since, until)
        stop_moves = ratcheted(db, since, until)
    finally:
        db.close()

    total = sum(bucket.trades for bucket in slices["instrument"].values())
    print()
    print("=" * 72)
    print(f"  SCORECARD — last {args.days:g} days, {total} closed trades")
    # Stated on the header rather than left to be remembered. A filtered report
    # is a different book from the unfiltered one, and the two are printed in
    # the same shape — the reader has to be told which one is on the screen.
    if keep:
        print(f"  only trades opened in: {', '.join(sorted(keep))}")
    if drop:
        print(f"  excluding trades opened in: {', '.join(sorted(drop))}")
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

    if stop_moves:
        print()
        print("  DID MOVING THE STOP PAY")
        print("  The same exit label covers two different trades: one where the stop was")
        print("  never touched and the plan simply failed, and one where the stop was")
        print("  ratcheted up behind price and THAT is what got hit. Only the second is a")
        print("  real counterfactual; averaged together they answer nothing.")
        head = f"{'':32}{'trades':>7}{'got':>9}{'holding':>9}{'lift':>9}"
        print(f"  {head}{'better':>8}{'moved at':>10}")
        print("  " + "-" * 86)
        for row in stop_moves:
            if int(row["trades"]) < args.min_sample:
                continue
            label = f"{row['name']!s} {row['ratchet']!s}"
            first = row["first_r"]
            when = f"{float(first):+.2f}R" if first is not None else "—"
            print(
                f"  {label:<32}{int(row['trades']):>7}"
                f"{float(row['actual']):>+8.2f}R{float(row['baseline']):>+8.2f}R"
                f"{float(row['lift']):>+8.2f}R{int(row['better']):>5}/{int(row['trades'])}"
                f"{when:>10}"
            )
        print()
        print("  `moved at` is the R the stop was FIRST moved at. A stop-moved row with a")
        print("  negative lift is what securing early costs; an untouched row near zero is")
        print("  the plan being graded against itself and carries no information.")
        print("  The replay runs 72h, so some recoveries are holds this account never had.")

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
