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
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: The entry path in the order the runner walks it, so the report can say where
#: candidates are actually lost rather than only which reason is commonest.
#:
#: A flat league table of reasons cannot answer the question people actually
#: ask, which is "does anything reach the reviewer at all". NO_SIGNAL is nearly
#: always the top row and nearly always should be — most markets offer nothing
#: most of the time — and it says nothing about whether the eleven gates behind
#: it are passable. Twelve serial gates each rejecting a modest share leaves
#: very little at the end, and only a stage-by-stage count shows which one is
#: doing it.
#:
#: Every row in the journal died at exactly one gate, so the arithmetic is a
#: real funnel: what enters a stage is everything that survived the stages
#: above it. Reasons are grouped by *where in the code* they fire, which is why
#: INSUFFICIENT_RUNWAY sits with the cost gates — the sharper of its two checks
#: runs after sizing, not with the session filters.
_STAGES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "usable market data",
        frozenset(
            {
                "DATA_UNAVAILABLE",
                "DATA_QUARANTINED",
                "MARKET_CLOSED",
                "STALE_QUOTE",
                "SYMBOL_NOT_TRADABLE",
                "SYMBOL_NOT_WHITELISTED",
                "SYMBOL_BLOCKED_BY_EQUITY",
            }
        ),
    ),
    ("a setup on the chart at all", frozenset({"NO_SIGNAL", "METHODS_DISAGREE"})),
    (
        "room in the book",
        frozenset(
            {
                "MAX_POSITIONS_REACHED",
                "MAX_TRADES_PER_DAY",
                "MAX_TRADES_PER_WEEK",
                "POSITION_ALREADY_OPEN",
                "DAILY_LOSS_LIMIT_HIT",
                "WEEKLY_LOSS_LIMIT_HIT",
                "MAX_DRAWDOWN_CIRCUIT_BREAKER",
                "KILL_SWITCH_ENGAGED",
                "SYSTEM_HALTED",
                "LIVE_NOT_ARMED",
            }
        ),
    ),
    (
        "session, news and exposure filters",
        frozenset(
            {
                "OUTSIDE_TRADABLE_SESSION",
                "ROLLOVER_WINDOW",
                "EVENING_WIND_DOWN",
                "WEEKEND_EDGE",
                "MARKET_TOO_QUIET",
                "SPREAD_TOO_WIDE",
                "NEWS_BLACKOUT",
                "NEWS_CALENDAR_UNAVAILABLE",
                "HEADLINE_PRESSURE",
                "HEADLINES_UNAVAILABLE",
                "CORRELATED_EXPOSURE",
                "CURRENCY_CONCENTRATION",
                "SECTOR_CONCENTRATION",
                "LOSS_COOLDOWN",
            }
        ),
    ),
    ("entry timing", frozenset({"AWAITING_CONFIRMATION", "ENTRY_OVEREXTENDED"})),
    (
        "can the trade pay its own costs",
        frozenset(
            {
                "SPREAD_EATS_THE_STOP",
                "SL_TOO_TIGHT_FOR_COSTS",
                "SL_TOO_TIGHT_FOR_BROKER",
                "SL_TOO_WIDE_FOR_ACCOUNT",
                "RR_BELOW_MINIMUM",
                "INVALID_STOP",
                "RISK_EXCEEDS_CAP",
                "TRADE_SKIPPED_UNDERCAPITALIZED",
                "INSUFFICIENT_MARGIN",
                "MARGIN_ESTIMATE_FAILED",
                "INSUFFICIENT_RUNWAY",
            }
        ),
    ),
    (
        "worth buying an opinion about",
        frozenset({"AI_VETO_PATTERN_KNOWN", "AI_REVIEW_BUDGET_SPENT"}),
    ),
)

#: What Claude did with the ones that got there, kept apart from the stages
#: above because these are the only rows that cost money.
_REVIEWED = frozenset({"AI_VETO", "AI_WAIT_RETEST"})

#: Approved and then thrown away by the recheck on a fresh quote.
_AFTER_REVIEW = frozenset({"ENTRY_MOVED_DURING_REVIEW", "ENTRY_STATE_CHANGED_DURING_REVIEW"})

#: Reasons that mean "the system is working as designed", separated from the
#: ones that mean "something is misconfigured". The distinction is the whole
#: point: no-trade is the normal state, and only some causes of it are faults.
#:
#: The list started small and every gate added since has defaulted to "!", so a
#: healthy session now prints an exclamation mark against nearly every line and
#: the mark stopped carrying information. A flag on everything is a flag on
#: nothing. What belongs here is any gate whose firing is the system doing its
#: job — a concentration limit refusing a doubled bet, a cost gate refusing a
#: stop the spread would eat, a quiet market. What stays out is the short list
#: below of things that should not be happening at all.
_EXPECTED = {
    "NO_SIGNAL",
    "METHODS_DISAGREE",
    "AI_VETO",
    "AI_VETO_PATTERN_KNOWN",
    "AI_REVIEW_BUDGET_SPENT",
    "MARKET_CLOSED",
    "DATA_QUARANTINED",
    "OUTSIDE_TRADABLE_SESSION",
    "WEEKEND_EDGE",
    "ROLLOVER_WINDOW",
    "EVENING_WIND_DOWN",
    "NEWS_BLACKOUT",
    "HEADLINE_PRESSURE",
    "MARKET_TOO_QUIET",
    "INSUFFICIENT_RUNWAY",
    "SPREAD_TOO_WIDE",
    "SPREAD_EATS_THE_STOP",
    "SL_TOO_TIGHT_FOR_COSTS",
    "SL_TOO_TIGHT_FOR_BROKER",
    "SL_TOO_WIDE_FOR_ACCOUNT",
    "RR_BELOW_MINIMUM",
    "TRADE_SKIPPED_UNDERCAPITALIZED",
    "POSITION_ALREADY_OPEN",
    "MAX_POSITIONS_REACHED",
    "MAX_TRADES_PER_DAY",
    "MAX_TRADES_PER_WEEK",
    "CORRELATED_EXPOSURE",
    "CURRENCY_CONCENTRATION",
    "SECTOR_CONCENTRATION",
    "LOSS_COOLDOWN",
    "AWAITING_CONFIRMATION",
    "ENTRY_OVEREXTENDED",
    "AI_WAIT_RETEST",
    "ENTRY_MOVED_DURING_REVIEW",
    "ENTRY_STATE_CHANGED_DURING_REVIEW",
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
    "SL_TOO_TIGHT_FOR_COSTS": (
        "The stop is legal at the broker and still too narrow to be worth taking: "
        "commission plus the slippage a stop-out actually suffers would be a large "
        "share of the risk. Expected on scalp-width setups. It is the gate that "
        "stops a -1.00R plan returning -1.48R; loosen `risk.max_cost_share_of_risk` "
        "only if you are willing to have the cost of trading decide the outcome."
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
    "AWAITING_CONFIRMATION": (
        "Price is running against the trade at the moment of entry — a short into a "
        "rising market, or a long into a falling one. Not a judgement on the setup: "
        "it may be right and simply early, and it is re-checked every cycle and taken "
        "as soon as the adverse move stops. If it dominates all day, either the "
        "analysis is consistently calling turns too soon or "
        "`analysis.confluence.confirmation_max_adverse_atr` is too tight."
    ),
    "ENTRY_OVEREXTENDED": (
        "The direction passed, but entering now would chase an ATR-extended move at "
        "the edge of its recent range. This is temporary: the same market is checked "
        "again next cycle and may enter after a non-extended retest. If it dominates "
        "normal liquid sessions, inspect `analysis.entry_quality` rather than lowering "
        "directional signal thresholds."
    ),
    "AI_WAIT_RETEST": (
        "Claude accepted the directional thesis but would not place a market order at "
        "the current price. It asked for a retest; this is not stored as a veto and the "
        "changed setup can be reviewed again."
    ),
    "ENTRY_MOVED_DURING_REVIEW": (
        "The price or review age moved beyond the snapshot Claude judged. The order was "
        "not chased with stale SL/TP sizing; fresh market data is analysed next cycle."
    ),
    "ENTRY_STATE_CHANGED_DURING_REVIEW": (
        "The account's positions changed while Claude was reviewing the proposal. "
        "Its slot, correlation or add-on context was stale, so no order used that "
        "approval; the next cycle rebuilds it from the current account."
    ),
    "CURRENCY_CONCENTRATION": (
        "A second position leaning the same way on a currency already in the book. "
        "GBPAUD short and GBPJPY short are not two trades, they are one GBP short "
        "with a second lot on it. Expected when the whole market is moving on one "
        "currency; raise `filters.currency_exposure.max_positions_per_currency` only "
        "if you actually want that bet doubled."
    ),
    "AI_VETO_PATTERN_KNOWN": (
        "Claude has refused this symbol and direction several times for the same "
        "underlying reason, so the question was not bought again. Not a judgement on "
        "this setup — it was recognised, not reviewed, and an approval on the same "
        "pair clears the pattern immediately. If it dominates a whole day, the "
        "instrument selection is the problem rather than this gate."
    ),
    "AI_REVIEW_BUDGET_SPENT": (
        "The cycle's paid reviews went to higher-conviction setups. Not a judgement "
        "on these — they were never asked about, and they are first in line next "
        "cycle. Only worth changing if the top-ranked candidates are consistently "
        "rejected by a later gate: raise `ai.max_reviews_per_cycle`, or 0 for no cap."
    ),
    "INSUFFICIENT_RUNWAY": (
        "Too close to the evening wind-down for a trade to finish — either under "
        "`filters.runway.min_runway_minutes` of clear market left, or a target that "
        "needs longer than the time remaining at the market's current pace. Expected "
        "in the last hour of the session; if it dominates all day, the floor or "
        "`filters.runway.travel_efficiency` is set too conservatively."
    ),
    "MARKET_TOO_QUIET": (
        "The market is ranging well below its own recent normal, so a target priced "
        "in ATR cannot be reached in any reasonable time. Expected during lulls and "
        "holidays; if it dominates for hours across every market, lower "
        "`filters.liveliness.min_activity_ratio`."
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
            f"SELECT symbol, decision, reason, detail, total_score, score_threshold, "
            f"context_json FROM analysis_cycles WHERE {where} ORDER BY id DESC",
            params,
        ).fetchall()

    if not rows:
        print(f"No decisions recorded in the last {args.hours:g}h.")
        print("If the console shows cycles running, the journal path may differ from --db.")
        return 1

    traded = sum(1 for row in rows if str(row["decision"]) == "TRADE")
    counts = Counter(str(row["reason"]) for row in rows)
    print(f"\n{len(rows)} decisions in the last {args.hours:g}h · {traded} became trades\n")

    # A refusal replayed from the veto memory lands in the journal as AI_VETO,
    # exactly like one that was paid for, and there is no way to tell them
    # apart from the reason column. The runner writes `ai_veto_remembered` on
    # precisely those rows.
    remembered = sum(
        1
        for row in rows
        if str(row["reason"]) == "AI_VETO"
        and "ai_veto_remembered" in str(row["context_json"] or "")
    )
    _print_funnel(counts, len(rows), traded, remembered)

    # Refused before the review, or after paying for it?
    #
    # Both look identical here -- ENTRY_OVEREXTENDED either way -- and they are
    # completely different findings. Before the review is the system working:
    # the gate cost nothing and saved a paid opinion. After the review is money
    # spent on an answer that was thrown away, because the entry gates all run
    # again on a fresh tick once Claude replies, and a forty-second reply is
    # long enough for a marginal setup to cross its own limit.
    #
    # The runner already records `post_review_revalidation` on exactly those
    # rows; nothing ever read it, so the waste was invisible.
    paid_then_refused = Counter(
        str(row["reason"])
        for row in rows
        if "post_review_revalidation" in str(row["context_json"] or "")
    )
    if paid_then_refused:
        spent = sum(paid_then_refused.values())
        print(f"{spent} of these were refused AFTER a paid review had already approved them:")
        for reason, count in paid_then_refused.most_common():
            print(f"    {reason:<34}{count:>5}")
        print(
            "  The review is the slowest step in the entry path, so every gate runs\n"
            "  again on a fresh quote once it answers. A setup sitting on its limit\n"
            "  when the question is asked is a coin flip by the time it is answered.\n"
        )

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
            _group(str(row["detail"])) for row in rows if str(row["reason"]) == "NO_SIGNAL"
        )
        print("What the engine actually said:")
        shown = 0
        for detail, count in details.most_common(8):
            print(f"  {count:>6}x  {detail}")
            shown += count
        rest = sum(details.values()) - shown
        if rest:
            print(f"  {rest:>6}x  everything else")
        _print_score_reach(rows)

    for reason in counts:
        if reason not in _EXPECTED and reason != "OK" and reason in _ADVICE:
            print(f"\n! {reason}: {_ADVICE[reason]}")

    if args.examples:
        print("\nMost recent decisions:")
        for row in rows[: args.examples * 3]:
            print(f"  {row['symbol']:<12} {row['reason']:<28} {_summarise(str(row['detail']))}")
    print()
    return 0


#: "confluence score 37.5 below threshold", as the engine writes it on the skip.
_SCORE_IN_TEXT = re.compile(r"confluence score (\d+(?:\.\d+)?)")


def _print_score_reach(rows: Sequence[sqlite3.Row]) -> None:
    """How close the engine actually came to its own threshold.

    Two sources say the same thing and only one of them was read. The
    `total_score` column is populated on a small minority of rows, and the
    detail text names the number on every scored skip. Reading the column alone
    produced the line "Best score reached: 58.5 ... (3 setups scored above
    zero)" on a night with thirty-five thousand NO_SIGNAL rows, six of which
    were visibly quoting scores in the mid-thirties two lines further up. Three
    is the number of rows with a populated column, not the number of setups
    that scored, and printing it as the latter is a diagnostic contradicting
    itself on the same screen.

    Both sources are merged, and what matters is not the single best score but
    how much of the distribution is within reach: a threshold nothing comes
    within twenty points of is a different problem from one being missed by two.
    """
    scores = [
        float(row["total_score"])
        for row in rows
        if row["total_score"] is not None and float(row["total_score"]) > 0
    ]
    from_column = len(scores)
    scores += [
        float(match.group(1))
        for row in rows
        if (match := _SCORE_IN_TEXT.search(str(row["detail"] or "")))
    ]
    if not scores:
        print("\n  No setup scored above zero at all — no module fired on any market.")
        return

    threshold = next(
        (float(row["score_threshold"]) for row in rows if row["score_threshold"] is not None),
        0.0,
    )
    best = max(scores)
    print(f"\n  {len(scores)} setups scored above zero, best {best:.1f} against {threshold:.1f}.")
    if from_column < len(scores):
        print(
            f"  ({from_column} of those come from the score column and the rest were read "
            "out of\n  the detail text, which is where older skip rows recorded it.)"
        )
    if threshold > 0:
        near = sum(1 for score in scores if threshold - 5.0 <= score < threshold)
        over = sum(1 for score in scores if score >= threshold)
        print(f"  {over} cleared it; {near} came within 5 points and did not.")
    if best < threshold:
        print(
            "  Nothing cleared it. Either conditions genuinely offer nothing, or the "
            "threshold\n  is set above what this engine produces."
        )


def _print_funnel(counts: Counter[str], total: int, traded: int, remembered: int = 0) -> None:
    """Where the candidates went, in the order the gates run.

    Reads top to bottom: everything scanned, then what each stage took out of
    it, then what survived to be paid for.

    `remembered` is the correction that makes the bottom line mean what it
    says. A refusal replayed from the veto memory is written to the journal as
    AI_VETO, indistinguishable in the reason column from one the account paid
    for, and counting those below the line inflated "reached the paid reviewer"
    by an order of magnitude -- 751 against 84 real calls on a live twelve
    hours, which turned a $4.79 daily API bill into a reported $40 one. They
    belong with the other gate that suppresses a call before it is made.
    """
    print("FROM SCAN TO A PAID REVIEW")
    print(f"  {total:>6}  decisions recorded")
    remaining = total
    for label, reasons in _STAGES:
        lost = sum(count for reason, count in counts.items() if reason in reasons)
        if "opinion" in label:
            lost += remembered
        if lost:
            print(f"  {-lost:>6}  {label}")
        remaining -= lost

    # Anything the stage table does not know about. Printed rather than folded
    # into a bucket: a reason added to the enum and not added here would
    # otherwise silently inflate the survivor count, which is the one number
    # this whole section exists to get right.
    staged = {reason for _, group in _STAGES for reason in group}
    known = staged | _REVIEWED | _AFTER_REVIEW | {"OK"}
    unmapped = {reason: count for reason, count in counts.items() if reason not in known}
    if unmapped:
        lost = sum(unmapped.values())
        print(f"  {-lost:>6}  not yet classified: {', '.join(sorted(unmapped))}")
        remaining -= lost

    share = 100.0 * remaining / total if total else 0.0
    print(f"  {'':>6}  {'-' * 46}")
    print(f"  {remaining:>6}  reached the paid reviewer   ({share:.1f}% of everything scanned)")
    if remembered:
        print(f"  {'':>6}  ({remembered} of the refusals above were replayed free from memory)")

    refused = sum(count for reason, count in counts.items() if reason in _REVIEWED) - remembered
    discarded = sum(count for reason, count in counts.items() if reason in _AFTER_REVIEW)
    if refused:
        print(f"  {-refused:>6}  Claude declined or asked for a retest")
    if discarded:
        print(f"  {-discarded:>6}  approved, then the price moved before the order went out")
    print(f"  {traded:>6}  became trades")

    if remaining == 0 and total:
        print(
            "\n  Nothing was sent to Claude at all. Every candidate died at a free gate,\n"
            "  so the review budget is irrelevant right now — the stage taking the\n"
            "  largest bite above is the only thing worth changing."
        )
    print()


def _summarise(detail: str) -> str:
    """Collapse a detail string so near-identical reasons group together."""
    text = " ".join(detail.split())
    return text[:110] + ("…" if len(text) > 110 else "")


def _group(detail: str) -> str:
    """`_summarise`, with the measurements taken out of the grouping key.

    The engine writes the number into the sentence: "confluence score 37.5
    below threshold". Grouped on the raw text, 37.5 and 37.4 are two different
    causes, so the single biggest reachable gate shatters into several hundred
    rows of two hundred each and never appears in a top-six list at all. A live
    night read "17432x no weighted directional evidence" at the top and three
    rows of ~250 near the bottom, and the ~14,000 decisions refused by the
    score threshold — the one number in this file an operator can actually
    move — were nowhere on the screen.
    """
    return _NUMBER.sub("N", _summarise(detail))


#: A measurement inside a detail sentence, replaced before counting so one gate
#: reads as one gate.
#:
#: The lookbehind is doing real work: without it, M5 and M15 both collapse to
#: "MN" and two different timeframes are reported as one cause. A timeframe is
#: part of the name of the thing being described, not a measurement of it.
#: Digits are excluded from the lookbehind as well as letters, or "M15" matches
#: at its second digit and becomes "M1N" — the engine simply advances one
#: character and tries again once the first position is rejected.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?")


if __name__ == "__main__":
    raise SystemExit(main())
