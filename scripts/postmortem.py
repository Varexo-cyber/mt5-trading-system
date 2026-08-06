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
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
        score = cycle["total_score"]
        threshold = cycle["score_threshold"]
        if score is not None and threshold is not None:
            print(f"  score         {score:.1f} against a {threshold:.1f} threshold")
        print(f"  session       {cycle['session'] or '?'}")
        print(f"  regime        {cycle['volatility_regime'] or '?'}")
        if cycle["spread_pips"] is not None:
            print(f"  spread        {cycle['spread_pips']:.2f} pips at entry")
        if cycle["minutes_to_news"] is not None:
            print(f"  next news     {cycle['minutes_to_news']:.0f} min away")

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

        context = _json(cycle["context_json"])
        interesting = {
            key: context[key]
            for key in ("runway_minutes", "minutes_to_target", "activity_ratio", "ai_confidence")
            if key in context and context[key] is not None
        }
        if interesting:
            print("\n  gates recorded at entry:")
            for key, value in interesting.items():
                print(f"    {key:<20} {value}")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="", help="e.g. USDCHF (suffix optional)")
    parser.add_argument("--ticket", type=int, default=0, help="exact broker ticket")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument("--list", type=int, default=0, help="instead, list the last N trades")
    args = parser.parse_args(argv)

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1

    db = connect(path)
    try:
        if args.list:
            rows = db.execute(
                "SELECT ticket, symbol, direction, opened_at, closed_at, pnl_r, pnl_money, "
                "mfe_r, exit_reason FROM trades ORDER BY COALESCE(closed_at, opened_at) DESC "
                "LIMIT ?",
                (args.list,),
            ).fetchall()
            print(
                f"\n  {'ticket':>10}  {'symbol':<12} {'dir':<6} {'R':>7} {'money':>9} "
                f"{'peak':>7}  exit"
            )
            for row in rows:
                print(
                    f"  {row['ticket'] or 0:>10}  {row['symbol']:<12} {row['direction']:<6} "
                    f"{r_of(row['pnl_r']):>7} {money(row['pnl_money']):>9} "
                    f"{r_of(row['mfe_r']):>7}  {row['exit_reason'] or 'open'}"
                )
            print()
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
