"""Answer "why has it not traded" from the journal, in one command.

    python scripts/why_no_trades.py
    python scripts/why_no_trades.py --hours 12
    python scripts/why_no_trades.py --symbol XAUUSD

The console line reports how many markets were scanned and how many were
analysed, and then stops. "62 analysed, 0 opened" invites exactly one question
and answers none of it, which is how five hours can pass with no trades and no
way to tell whether the system is working correctly or silently broken. Those
two look identical from the outside and need opposite responses.

Every one of those decisions is already in the journal with its reason and the
numbers behind it. This reads them back and says which gate is actually
stopping everything.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Reasons that mean "the system is working as designed", separated from the
#: ones that mean "something is misconfigured". The distinction is the whole
#: point: no-trade is the normal state, and only some causes of it are faults.
_EXPECTED = {
    "NO_SIGNAL",
    "AI_VETO",
    "MARKET_CLOSED",
    "OUTSIDE_TRADABLE_SESSION",
    "WEEKEND_EDGE",
    "ROLLOVER_WINDOW",
    "POSITION_ALREADY_OPEN",
    "MAX_POSITIONS_REACHED",
    "CORRELATED_EXPOSURE",
}

#: What to do about each, in the operator's terms.
_ADVICE = {
    "NO_SIGNAL": (
        "The confluence engine found no tradeable setup. Normal most of the time — "
        "but if this is 100% of decisions for hours across every market, the entry "
        "rules are too strict for current conditions. Check `analysis.confluence."
        "score_threshold` and `minimum_directional_modules`."
    ),
    "AI_VETO": "Claude declined. Look at the AI exchange tab for the reasoning.",
    "TRADE_SKIPPED_UNDERCAPITALIZED": (
        "The account cannot express the trade at the configured risk: the computed "
        "lot rounds below the broker minimum. Correct behaviour — rounding up would "
        "silently multiply your risk. Nothing to fix unless equity has grown."
    ),
    "SL_TOO_WIDE_FOR_ACCOUNT": (
        "The structural stop is wider than this account can carry. Raise "
        "`max_sl_pips` only if the arithmetic genuinely allows it."
    ),
    "SPREAD_TOO_WIDE": "Spread over the limit. Check `filters.spread.max_spread_bps`.",
    "STALE_QUOTE": "Quotes too old. Usually a shut market; check the terminal's connection.",
    "NEWS_CALENDAR_UNAVAILABLE": (
        "No economic calendar, so every entry is blocked by design. Run "
        "`python scripts/verify_calendar.py` — this one WILL stop all trading."
    ),
    "MAX_TRADES_PER_DAY": "Daily trade cap reached. Set `risk.max_trades_per_day: 0` to remove it.",
    "DAILY_LOSS_LIMIT_HIT": "Daily loss limit hit; paused until the next trading day.",
    "MAX_POSITIONS_REACHED": "Already holding the maximum number of positions.",
    "KILL_SWITCH_ENGAGED": "The STOP file is present. Clear it from the Control tab.",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0, help="how far back to look")
    parser.add_argument("--symbol", default="", help="restrict to one market")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument("--examples", type=int, default=3, help="sample details per reason")
    args = parser.parse_args(argv)

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}. Start jarvis.py first.")
        return 1

    since = (datetime.now(UTC) - timedelta(hours=args.hours)).isoformat()
    where = "ts >= ?"
    params: list[object] = [since]
    if args.symbol:
        where += " AND symbol = ?"
        params.append(args.symbol)

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT symbol, decision, reason, detail, total_score, score_threshold "
            f"FROM analysis_cycles WHERE {where} ORDER BY id DESC",
            params,
        ).fetchall()

    if not rows:
        print(f"No decisions recorded in the last {args.hours:g}h.")
        print("If the console shows cycles running, the journal path may differ from --db.")
        return 1

    traded = sum(1 for row in rows if str(row["decision"]) == "TRADE")
    counts = Counter(str(row["reason"]) for row in rows)
    print(f"\n{len(rows)} decisions in the last {args.hours:g}h · {traded} became trades\n")

    width = max(len(reason) for reason in counts)
    for reason, count in counts.most_common():
        share = 100.0 * count / len(rows)
        flag = " " if reason in _EXPECTED or reason == "OK" else "!"
        print(f"{flag} {reason:<{width}}  {count:>5}  {share:5.1f}%")

    print()
    top = counts.most_common(1)[0][0]
    if top in _ADVICE:
        print(f"Dominant reason — {top}:\n  {_ADVICE[top]}\n")

    # A NO_SIGNAL wall is the case worth interrogating: it is normal in small
    # doses and means the entry rules are unreachable when it is everything.
    if top == "NO_SIGNAL":
        details = Counter(
            _summarise(str(row["detail"])) for row in rows if str(row["reason"]) == "NO_SIGNAL"
        )
        print("What the engine actually said:")
        for detail, count in details.most_common(6):
            print(f"  {count:>5}x  {detail}")
        scored = [
            float(row["total_score"])
            for row in rows
            if row["total_score"] is not None and float(row["total_score"]) > 0
        ]
        if scored:
            threshold = next(
                (float(r["score_threshold"]) for r in rows if r["score_threshold"] is not None),
                0.0,
            )
            best = max(scored)
            print(
                f"\n  Best score reached: {best:.1f} against a {threshold:.1f} threshold "
                f"({len(scored)} setups scored above zero)."
            )
            if best < threshold:
                print(
                    "  Nothing came close. Either conditions genuinely offer nothing, or "
                    "the threshold is set above what this engine produces."
                )
        else:
            # Distinguish "the modules saw nothing" from "the column is empty".
            # Skip rows carried no score until recently, so this branch fired
            # and announced that nothing was firing while the detail text on
            # the very same rows read "confluence score 41.9 below threshold".
            # A diagnostic that contradicts itself is worse than none.
            scored_in_text = sum(1 for row in rows if "confluence score" in str(row["detail"]))
            if scored_in_text:
                print(
                    f"\n  {scored_in_text} decisions name a score in their detail text but the "
                    "score column is empty.\n  Those rows predate the score being recorded on "
                    "skips; read the detail lines above instead."
                )
            else:
                print("\n  No setup scored above zero at all — no module fired on any market.")

    for reason in counts:
        if reason not in _EXPECTED and reason != "OK" and reason in _ADVICE:
            print(f"\n! {reason}: {_ADVICE[reason]}")

    if args.examples:
        print("\nMost recent decisions:")
        for row in rows[: args.examples * 3]:
            print(f"  {row['symbol']:<12} {row['reason']:<28} {_summarise(str(row['detail']))}")
    print()
    return 0


def _summarise(detail: str) -> str:
    """Collapse a detail string so near-identical reasons group together."""
    text = " ".join(detail.split())
    return text[:110] + ("…" if len(text) > 110 else "")


if __name__ == "__main__":
    raise SystemExit(main())
