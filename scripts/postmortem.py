"""Everything the journal knows about one trade, in the order it happened.

Written because the question "what happened with that USDCHF trade" was being
answered by reading five tables by hand, and the answer that matters is almost
never in any single one of them. A trade that ends at +0.13R after peaking at
0.92R looks fine in `trades` and only makes sense once the management actions
sit next to the excursion: the peak, then a break-even stop, then nothing, then
the exit. Printed as a sequence, that story reads itself.

Everything here is already recorded. This adds no new data collection — it puts
what exists in one place and does the arithmetic the eye keeps having to do:
what the trade was worth at its best, what it actually returned, and how much
of the difference each rule is responsible for.

    python scripts/postmortem.py USDCHF
    python scripts/postmortem.py --ticket 123456
    python scripts/postmortem.py            # the last closed trade, whatever it was
    python scripts/postmortem.py --hours 24 # everything from the last day
    python scripts/postmortem.py --list 20  # the last twenty, whenever they were
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.trade_origin import origin_for_setup_family

#: What each management action meant, in the operator's terms. Anything absent
#: gets no gloss at all — see `describe`. An unexplained action beats a wrong
#: explanation, because the whole value of this report is that it is trusted.
_ACTIONS = {
    "BREAK_EVEN": "stop moved to entry",
    "PROFIT_LOCK": "stop walked up to secure part of the peak",
    "PEAK_STALL": "banked: the peak stopped advancing",
    "GIVEBACK_EXIT": "banked: the gain was draining away",
    "ATR_TRAIL": "stop trailed behind price",
    "PARTIAL_CLOSE": "part of the position banked",
    "PARTIAL_CLOSE_RECOVERED": "broker partial recovered into the journal",
    "TIME_EXIT": "closed on the clock",
    "SPREAD_SQUEEZE_EXIT": "left before the spread could take the stop",
    "EVENING_FLAT": "evening wind-down",
    "NEWS_BREAK_EVEN": "stop to entry ahead of a release",
    "HEALTH_EXIT": "the health read said the move had broken",
    "HEALTH_SECURE": "banked on a deteriorating read",
    "HEALTH_TIGHTEN": "stop tightened on a warning",
    "EMERGENCY_CLOSE": "closed: the position had no stop",
    "ORPHAN_CLOSE": "closed: unknown to the journal",
}

#: Closures the broker performed, recovered from deal history as `BROKER_<why>`.
#: Worth distinguishing from our own exits: these are the ones nothing in this
#: system chose, which on a trade that gave back its gain is the finding.
_BROKER = {
    "SL": "the stop was hit",
    "TP": "the target was hit",
    "SO": "stopped out by the broker (margin)",
    "MANUAL": "closed by hand in the terminal",
    "CLOSED": "closed at the broker",
}


def describe(action: str) -> str:
    if action in _ACTIONS:
        return _ACTIONS[action]
    if action.startswith("BROKER_"):
        return _BROKER.get(action.removeprefix("BROKER_"), "closed at the broker")
    return ""


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def find_trade(db: sqlite3.Connection, symbol: str, ticket: int) -> sqlite3.Row | None:
    """The requested trade, or the most recent one matching the filters.

    Matches the symbol loosely on purpose: the broker's name is `USDCHF.i` and
    nobody types the suffix.
    """
    if ticket:
        rows = db.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,)).fetchall()
        return rows[0] if rows else None
    where, params = "1=1", []
    if symbol:
        where = "UPPER(symbol) LIKE ?"
        params.append(f"%{symbol.upper()}%")
    rows = db.execute(
        f"SELECT * FROM trades WHERE {where} ORDER BY COALESCE(closed_at, opened_at) DESC LIMIT 1",
        params,
    ).fetchall()
    return rows[0] if rows else None


def moment(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def money(value: float | None, currency: str = "EUR") -> str:
    return "—" if value is None else f"{value:+.2f} {currency}"


def r_of(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}R"


def elapsed(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "?"
    minutes = (end - start).total_seconds() / 60.0
    if minutes < 90:
        return f"+{minutes:.0f}m"
    return f"+{minutes / 60:.1f}h"


def report(db: sqlite3.Connection, trade: sqlite3.Row) -> None:
    opened, closed = moment(trade["opened_at"]), moment(trade["closed_at"])
    risk_money = float(trade["risk_money"] or 0.0)
    peak_r = trade["mfe_r"]
    trough_r = trade["mae_r"]
    pnl_r = trade["pnl_r"]
    pnl_money = trade["pnl_money"]

    print(f"\n{'=' * 78}")
    print(f"  {trade['symbol']}  {trade['direction']}  {trade['volume']:g} lots")
    print(f"{'=' * 78}\n")

    print(f"  ticket        {trade['ticket']}")
    print(f"  opened        {trade['opened_at']}")
    print(f"  closed        {trade['closed_at'] or 'no closure recorded'}")
    if opened and closed:
        print(f"  held          {elapsed(opened, closed)}")
    if not closed:
        # "still open" is what this used to say, and it is a claim the journal
        # cannot actually make. All it knows is that no closure has been
        # written. An operator who has just watched the position disappear
        # from MT5 reads that as the report being wrong, when the real message
        # is that reconciliation has not run — it only runs inside a full
        # cycle, so a stopped Jarvis leaves every closure unrecorded.
        print("                ^ the journal has no closure for this ticket.")
        print("                  If it is gone from MT5, Jarvis has not")
        print("                  reconciled yet — reconciliation runs once per")
        print("                  cycle, so check that Jarvis is actually running.")
    print()
    print(f"  entry         {trade['entry_price']:.5f}")
    print(f"  stop          {trade['sl']:.5f}   ({trade['sl_distance_pips']:.1f} pips)")
    print(f"  target        {trade['tp']:.5f}   (planned RR {trade['planned_rr']:.2f})")
    if trade["exit_price"] is not None:
        print(f"  exit          {trade['exit_price']:.5f}   [{trade['exit_reason'] or '?'}]")
    print()
    print(f"  1R            {risk_money:.2f}  ({trade['risk_pct']:.2f}% of equity)")
    after = trade["equity_after"]
    print(f"  equity        {trade['equity_before']:.2f} -> " + (f"{after:.2f}" if after else "?"))

    # -- the part that is not in any single table --------------------------
    print(f"\n{'-' * 78}")
    print("  HOW IT WENT")
    print(f"{'-' * 78}\n")

    print(f"  best it reached      {r_of(peak_r):>8}   {money(_at(peak_r, risk_money))}")
    print(f"  worst it reached     {r_of(trough_r):>8}   {money(_at(trough_r, risk_money))}")
    print(f"  what it returned     {r_of(pnl_r):>8}   {money(pnl_money)}")

    if peak_r is not None and pnl_r is not None and float(peak_r) > 0:
        kept = float(pnl_r) / float(peak_r)
        given = _at(float(peak_r) - float(pnl_r), risk_money)
        print(f"\n  kept {kept:.0%} of the best moment; {money(given)} was left on the table")
        if kept < 0.5:
            print("  ^ over half the gain was handed back. Look at the timeline below for")
            print("    which rule was holding the position while that happened.")

    # -- what the management layer actually did ----------------------------
    actions = db.execute(
        "SELECT * FROM management_actions WHERE trade_id = ? ORDER BY ts",
        (trade["id"],),
    ).fetchall()

    print(f"\n{'-' * 78}")
    print("  WHAT THE SYSTEM DID, IN ORDER")
    print(f"{'-' * 78}\n")

    if not actions:
        print("  Nothing. The trade ran from entry to exit untouched — no break-even,")
        print("  no trail, no partial. On a trade that peaked well and closed poorly")
        print("  that is the finding, not a missing record.\n")
    for row in actions:
        when = elapsed(opened, moment(row["ts"]))
        label = describe(str(row["action"]))
        r_text = f"{row['r_at_action']:+.2f}R" if row["r_at_action"] is not None else "  ?  "
        print(f"  {when:>7}  {r_text:>7}  {row['action']:<22} {label}")
        if row["new_sl"] is not None and row["old_sl"] is not None:
            print(f"                            stop {row['old_sl']:.5f} -> {row['new_sl']:.5f}")
        if row["note"]:
            print(f"                            {row['note']}")

    # -- why it was taken at all -------------------------------------------
    cycle = db.execute(
        "SELECT * FROM analysis_cycles WHERE id = ?", (trade["cycle_pk"],)
    ).fetchone()
    if cycle is not None:
        print(f"\n{'-' * 78}")
        print("  WHY IT WAS TAKEN")
        print(f"{'-' * 78}\n")
        context = _json(cycle["context_json"])
        # SAY WHICH ROUTE TOOK IT, BEFORE ANY NUMBER.
        #
        # Section six has its own lane around the confluence vote, because its
        # module tops out at 33.75 against a bar of 45 and cannot clear it by
        # design. This report used to print "score 33.0 against a 35.0
        # threshold" over such a trade and stop there, which reads as the
        # account having traded through its own vote. It had not -- the vote
        # was never asked. Naming the lane is the difference between "the
        # system ignored its threshold" and "this threshold does not apply
        # here", and only one of those is worth investigating.
        section_six = str(context.get("section") or "") == "six" or str(
            cycle["detail"] or ""
        ).startswith("section six")
        origin = origin_for_setup_family(str(context.get("setup_family") or ""))
        score = cycle["total_score"]
        threshold = cycle["score_threshold"]
        if section_six:
            print("  route         section six's own lane - the confluence vote does not apply")
            if score is not None:
                print(f"  strength      {score:.1f}  (candle_momentum alone)")
        elif origin is not None:
            print(
                f"  route         SECTION {origin.section} / {origin.strategy} / "
                f"{origin.timeframe}"
            )
            print(f"  MT5 label     {origin.comment}")
            if score is not None and threshold is not None:
                print(f"  score         {score:.1f} against a {threshold:.1f} threshold")
            elif score is not None:
                print(f"  score         {score:.1f}")
        elif score is not None and threshold is not None:
            print(f"  score         {score:.1f} against a {threshold:.1f} threshold")
        elif score is not None:
            print(f"  score         {score:.1f}")
        print(f"  session       {cycle['session'] or '?'}")
        print(f"  regime        {cycle['volatility_regime'] or '?'}")
        if cycle["spread_pips"] is not None:
            print(f"  spread        {cycle['spread_pips']:.2f} pips at entry")
        if cycle["minutes_to_news"] is not None:
            print(f"  next news     {cycle['minutes_to_news']:.0f} min away")

        # WHAT THE TRADE HAD TO PAY BEFORE IT COULD BE RIGHT.
        #
        # On a trade measured in seconds this is usually the whole story, and
        # the report had no line for it. A plan is drawn from the entry side of
        # the book while both exits happen on the other side: a short is sold
        # at the bid and bought back at the ask, so the market has to travel
        # one spread LESS than the stop looks to lose, and one spread MORE than
        # the target looks to win. Printed side by side, those two distances
        # say whether a loser was a bad read or a trade that could not have
        # paid at any hit rate.
        stop_spreads = context.get("stop_in_spreads")
        target_spreads = context.get("target_in_spreads")
        if stop_spreads and target_spreads:
            adverse = float(stop_spreads) - 1.0
            favourable = float(target_spreads) + 1.0
            share = float(context.get("cost_share") or 0.0)
            print(f"\n  cost of entry {share:.1%} of R -- one spread, crossed once")
            print(f"    to lose     {adverse:.1f} spreads against you (stop at {stop_spreads})")
            print(f"    to win      {favourable:.1f} spreads for you  (target at {target_spreads})")
            if adverse > 0 and favourable > 0:
                flat = adverse / (adverse + favourable)
                print(f"    coin flip   {flat:.0%} - anything less than the payoff needs is a cost")

        modules = db.execute(
            "SELECT module, score, confidence, reasoning FROM module_scores "
            "WHERE cycle_pk = ? ORDER BY ABS(score) DESC",
            (cycle["id"],),
        ).fetchall()
        if modules:
            print("\n  what each module saw:")
            for row in modules:
                if abs(float(row["score"])) < 0.01:
                    continue
                print(
                    f"    {row['module']:<20} {row['score']:+6.1f}  "
                    f"conf {row['confidence']:.2f}  {(row['reasoning'] or '')[:80]}"
                )

        interesting = {
            key: context[key]
            for key in ("runway_minutes", "minutes_to_target", "activity_ratio", "ai_confidence")
            if key in context and context[key] is not None
        }
        if interesting:
            print("\n  gates recorded at entry:")
            for key, value in interesting.items():
                print(f"    {key:<20} {value}")

        # WHERE THE PRICE SAT WHEN IT WAS BOUGHT.
        #
        # The entry gate measures four things and the postmortem printed none
        # of them, so "why did it buy the top" could only be answered by
        # reading the code and guessing which sub-test had been closest. ETHUSD
        # LONG on 20 August went in within two points of a vertical M1 spike
        # and never printed a positive tick; the report offered `activity_ratio`
        # and `ai_confidence`.
        #
        # Each number is shown against the limit it was judged by, because the
        # gate only refuses when the price is at its range extreme AND one of
        # the other three is breached — so the interesting reading is always
        # which of them came closest and by how much.
        quality = context.get("entry_quality") or {}
        if isinstance(quality, dict) and quality:
            print("\n  where the price was when it was bought:")
            labels = (
                ("directional_range_location", "where in its own range", "%"),
                ("favourable_extension_atr", "already travelled", "ATR"),
                ("single_bar_body_atr", "last bar body", "ATR"),
                ("ema_distance_atr", "distance from EMA20", "ATR"),
                ("last_bar_adverse_atr", "last bar against it", "ATR"),
            )
            for key, label, unit in labels:
                value = quality.get(key)
                if value is None:
                    continue
                shown = f"{float(value):.0%}" if unit == "%" else f"{float(value):.2f} {unit}"
                print(f"    {label:<24} {shown}")
            if quality.get("reason_code"):
                print(f"    {'gate verdict':<24} {quality['reason_code']}")

    print()


def _at(r: float | None, risk_money: float) -> float | None:
    """Convert an R multiple into account currency."""
    return None if r is None else float(r) * risk_money


def _json(text: str | None) -> dict:
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


#: Exits this system chose for itself, as opposed to the market reaching a
#: level we left sitting at the broker.
#: Management actions that *close* a position. `exit_reason` is set from the
#: event action, and only a closing event carries an exit price, so this set is
#: exactly the closing half of what `execution/manager.py` emits.
#:
#: It has to be kept honest, because the "not one exit was chosen by this
#: system" line below is loud and is read as a finding. It previously omitted
#: PROFIT_BANKED — the banking rule's own exit — and listed SPREAD_SQUEEZE_EXIT,
#: a name the manager never emits (it says SPREAD_SQUEEZE). So the first trade
#: the banking rule ever closed would still have been reported as proof that no
#: rule had acted.
#:
#: Deliberately excluded: PROFIT_LOCK, BREAK_EVEN, HEALTH_SECURE,
#: HEALTH_TIGHTEN, ATR_TRAIL, NEWS_BREAK_EVEN, AI_TIGHTEN_STOP,
#: AI_PULL_TARGET_IN. Those move a stop or a target; they never close anything,
#: so they cannot be an exit reason. If one of them is what a trade died of,
#: the exit reason is BROKER_SL and the broker is correctly credited with it.
_OUR_EXITS = frozenset(
    {
        # banking and trailing rules
        "PROFIT_BANKED",
        "PEAK_STALL",
        "GIVEBACK_EXIT",
        "TIME_EXIT",
        "SESSION_DECAY",
        # health and market conditions
        "HEALTH_EXIT",
        "SPREAD_SQUEEZE",
        "EVENING_FLAT",
        "NEWS_EXIT",
        "NEWS_EXIT_SENT",
        # safety closes
        "EMERGENCY_CLOSE",
        "ORPHAN_CLOSE",
        # the paid adviser, when the manager actually carried it out
        "AI_CLOSE",
        "AI_CLOSE_SENT",
        "AI_PARTIAL_CLOSE",
        "PARTIAL_CLOSE",
    }
)


def overview(db: sqlite3.Connection, limit: int, hours: float = 0.0) -> None:
    """Every recent trade on one screen, with the two columns that matter.

    `hours` answers the question an operator actually has at the end of a
    session: "what did it do today". A count is the wrong unit for that —
    twenty trades can be two hours or two weeks — so the window is offered in
    the unit the question is asked in. Open positions are included, because
    "nothing closed since lunch" and "nothing was opened since lunch" are very
    different findings and a closed-only list cannot tell them apart.

    `kept` is what survived to the exit as a share of the trade's best moment.
    It is the number that separates a losing strategy from a strategy that
    wins and then hands it back, and neither `pnl_r` nor `mfe_r` says it alone.

    `by` says whether anything in this system chose the exit. A column of
    nothing but BROKER_SL means every trade ran to a level left sitting at the
    broker and not one rule ever acted — which was true here for a long time
    and invisible, because the guard loop that runs those rules was being
    starved by slow cycles.
    """
    columns = (
        "SELECT ticket, symbol, direction, sl_distance_pips, pnl_r, pnl_money, mae_r, mfe_r, "
        "exit_reason, closed_at FROM trades "
    )
    if hours > 0:
        # Sorted and cut off by the same expression, so a trade cannot be
        # inside the window by one clock and ordered by another. Timestamps are
        # stored as ISO-8601 UTC text, which sorts and compares correctly as
        # text without any parsing on the SQLite side.
        since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        rows = db.execute(
            columns + "WHERE COALESCE(closed_at, opened_at) >= ? "
            "ORDER BY COALESCE(closed_at, opened_at) DESC",
            (since,),
        ).fetchall()
        window = f"the last {hours:g}h"
    else:
        rows = db.execute(
            columns + "ORDER BY COALESCE(closed_at, opened_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        window = ""
    if not rows:
        # An empty window and an empty journal are different findings. "Nothing
        # in the last 12h" is a fact about the session; "no trades recorded" is
        # a fact about the whole account, and reading the first as the second
        # is how a quiet afternoon looks like a broken database.
        print(f"\n  No trades in {window}.\n" if window else "\n  No trades recorded yet.\n")
        return
    if window:
        print(f"\n  {len(rows)} trade(s) in {window}.")

    print(
        f"\n  {'ticket':>10}  {'symbol':<11}{'dir':<7}{'stop':>7}{'R':>9}{'money':>12}"
        f"{'peak':>9}{'kept':>7}   exit"
    )
    print(f"  {'-' * 84}")

    ours = closed = handed_back = 0
    for row in rows:
        peak, pnl = row["mfe_r"], row["pnl_r"]
        kept = "—"
        if peak is not None and pnl is not None and float(peak) > 0:
            share = float(pnl) / float(peak)
            kept = f"{share:.0%}"
            if share < 0.5:
                handed_back += 1
        reason = row["exit_reason"] or ("open" if not row["closed_at"] else "?")
        if row["closed_at"]:
            closed += 1
            if reason in _OUR_EXITS:
                ours += 1
        stop = row["sl_distance_pips"]
        print(
            f"  {row['ticket'] or 0:>10}  {row['symbol']:<11}{row['direction']:<7}"
            + (f"{stop:>6.1f}p" if stop else f"{'—':>7}")
            + f"{r_of(pnl):>9}{money(row['pnl_money']):>12}{r_of(peak):>9}{kept:>7}   {reason}"
        )

    print(f"\n  {closed} closed · {ours} exited by a rule of ours · {closed - ours} by the broker")
    if closed and not ours:
        print("  ^ not one exit was chosen by this system. Every trade ran to a level")
        print("    left sitting at the broker. If the guard loop is running, that is a")
        print("    finding; check the Positions tab for the age of the health reading.")
    if handed_back:
        print(f"  {handed_back} trade(s) kept under half of their best moment.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="", help="e.g. USDCHF (suffix optional)")
    parser.add_argument("--ticket", type=int, default=0, help="exact broker ticket")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument("--list", type=int, default=0, help="instead, list the last N trades")
    parser.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="instead, list every trade from the last N hours (e.g. 12, 24)",
    )
    args = parser.parse_args(argv)
    if args.hours < 0:
        parser.error("--hours must be positive")

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1

    db = connect(path)
    try:
        if args.hours:
            overview(db, 0, args.hours)
            return 0
        if args.list:
            overview(db, args.list)
            return 0

        trade = find_trade(db, args.symbol, args.ticket)
        if trade is None:
            target = args.symbol or f"ticket {args.ticket}"
            print(f"No trade found for {target}. Try --list 20 to see what is recorded.")
            return 1
        report(db, trade)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
