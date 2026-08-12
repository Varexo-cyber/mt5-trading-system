"""What the paid reviews cost, and whether anything predicts their answer.

    python scripts/review_value.py
    python scripts/review_value.py --hours 12

Twelve hours of live running produced 44,061 decisions, 359 of which reached
Claude. Of those 359, 348 came back refused. Three became trades.

Two questions follow from that and neither had an answer anywhere. The first is
what it costs: 359 calls at the measured per-call rate is roughly forty dollars
a day, against an account holding a hundred and fifty euros. Whether the
reviewer is right on every one of them stops mattering at that ratio — the fee
alone empties the account inside a week.

The second is whether the waste is avoidable, and that turns on something
nobody has measured: does anything visible *before* the call predict what comes
back? If the setups Claude approves score differently from the ones it refuses,
the engine can stop paying for the answer it already knows. If they score
identically then no cheap filter exists, and the only honest levers left are
the payload and the call rate.

This reads the audit ledger and answers both from calls already made. It
changes nothing and recommends nothing it has not measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.ai_exchange import call_cost

#: Conviction is score times confidence, so it lands on a 0-100-ish scale. Five
#: points is fine enough to see a boundary and coarse enough that a few hundred
#: calls put something in most occupied bands.
BAND = 5.0


@dataclass(frozen=True, slots=True)
class Review:
    """One paid review, joined from its request and its response."""

    when: datetime
    symbol: str
    direction: str
    score: float
    confidence: float
    conviction: float
    setup_family: str
    thesis: str
    #: The ENGINE's confidence rides in `confidence`; this is the reviewer's own
    #: number. Reading the first where the second is meant reports the same
    #: value for every call in a cycle and hides exactly the clustering this is
    #: here to expose.
    verdict_confidence: float
    approved: bool
    said_yes: bool
    waiting: bool
    cost: float | None
    replayed: bool

    @property
    def useful(self) -> bool:
        """Did this call produce anything other than "no"?

        `said_yes` rather than `approved`, deliberately. A setup the reviewer
        endorsed and the confidence threshold then refused is not evidence that
        asking was pointless — it is evidence the threshold may be wrong, and a
        filter trained to skip those would be training on the wrong label.
        """
        return self.said_yes or self.waiting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0, help="how far back to look")
    parser.add_argument("--ledger", default="runtime/ai_reviews.jsonl", help="audit ledger path")
    args = parser.parse_args(argv)

    path = ROOT / args.ledger
    if not path.exists():
        print(f"No AI ledger at {path}.")
        return 1

    since = datetime.now(UTC) - timedelta(hours=args.hours)
    reviews = _load(path, since)
    if not reviews:
        print(f"No paid reviews in the last {args.hours:g}h.")
        return 1

    _report_spend(reviews, args.hours)
    _report_separation(reviews)
    _report_floor(reviews)
    _report_refusals(reviews)
    _report_repeats(reviews)
    return 0


# --------------------------------------------------------------------- report


def _report_spend(reviews: Sequence[Review], hours: float) -> None:
    paid = [review for review in reviews if not review.replayed]
    replayed = len(reviews) - len(paid)
    priced = [review.cost for review in paid if review.cost is not None]
    spent = sum(priced)
    approvals = sum(1 for review in reviews if review.approved)
    said_yes = sum(1 for review in reviews if review.said_yes)
    waiting = sum(1 for review in reviews if review.waiting)

    print(f"\nWHAT THE REVIEWS COST — last {hours:g}h\n")
    print(f"  {len(reviews):>6}  reviews on record")
    print(f"  {replayed:>6}  served from cache, free")
    print(f"  {len(paid):>6}  paid calls")
    if priced:
        print(f"  {f'${spent:.2f}':>6}  spent  (${spent / len(priced):.4f} per priced call)")
        print(f"  {f'${spent * 24.0 / hours:.2f}':>6}  at this rate per day")
    else:
        print("     n/a  no priced calls — the model is not in the dashboard price table")
    print()
    print(f"  {approvals:>6}  approved and executable")
    print(f"  {said_yes - approvals:>6}  endorsed but under the confidence threshold")
    print(f"  {waiting:>6}  asked for a retest")
    print(f"  {len(reviews) - said_yes - waiting:>6}  refused outright")
    if priced and approvals:
        print(f"\n  ${spent / approvals:.2f} of review spend per approval.")
    elif priced:
        print("\n  Nothing was approved, so every dollar above bought a refusal.")


def _report_separation(reviews: Sequence[Review]) -> None:
    """Does conviction predict the verdict, or is the engine's score blind here?

    This is the question the whole idea of a pre-review filter rests on, and
    the answer decides whether the next section means anything. Bands rather
    than a correlation because the shape matters more than a coefficient: a
    filter needs a *place to cut*, and that is only visible if the useful calls
    cluster somewhere the useless ones do not.
    """
    buckets: dict[float, list[Review]] = defaultdict(list)
    for review in reviews:
        buckets[review.conviction // BAND * BAND].append(review)

    print("\nDOES THE ENGINE'S SCORE PREDICT THE ANSWER?\n")
    print(f"  {'conviction':<14}{'calls':>7}{'useful':>8}{'rate':>8}")
    for floor in sorted(buckets):
        rows = buckets[floor]
        useful = sum(1 for review in rows if review.useful)
        print(
            f"  {f'{floor:.0f}-{floor + BAND:.0f}':<14}{len(rows):>7}{useful:>8}"
            f"{useful / len(rows):>7.0%}"
        )

    useful_rows = [review for review in reviews if review.useful]
    if not useful_rows:
        print(
            "\n  Nothing was useful at any score, so there is no boundary to find here.\n"
            "  A filter fitted to this would be fitted to a period with no approvals\n"
            "  in it, and would keep refusing after conditions changed."
        )
        return
    lowest = min(review.conviction for review in useful_rows)
    below = sum(1 for review in reviews if review.conviction < lowest)
    print(
        f"\n  Lowest conviction that produced anything useful: {lowest:.1f}\n"
        f"  Calls made below that line: {below} of {len(reviews)} "
        f"({below / len(reviews):.0%})"
    )


def _report_floor(reviews: Sequence[Review]) -> None:
    """What a pre-review conviction floor would have saved, and cost.

    Both columns matter and only one is ever quoted. A floor that saves 60% of
    the spend by discarding the single approval of the day is not a saving, it
    is the off switch with a percentage attached.

    In-sample by construction: every floor here is fitted to the same calls it
    is scored against, so the "lost" column is the optimistic case and the real
    one is worse. That is why the recommendation below sits a full band under
    the lowest useful score rather than on it.
    """
    useful_rows = [review for review in reviews if review.useful]
    if not useful_rows:
        return
    paid = [review for review in reviews if not review.replayed and review.cost is not None]
    per_call = sum(review.cost or 0.0 for review in paid) / len(paid) if paid else 0.0

    print("\nWHAT A PRE-REVIEW FLOOR WOULD HAVE DONE\n")
    print(f"  {'floor':<8}{'calls skipped':>15}{'saved':>10}{'useful lost':>14}")
    lowest = min(review.conviction for review in useful_rows)
    for floor in _candidate_floors(reviews):
        skipped = [review for review in reviews if review.conviction < floor]
        if not skipped:
            # A floor below every call made. True, safe and meaningless: it
            # would pad the top of the table with rows saving nothing and
            # marked safe, which reads as evidence and is not.
            continue
        lost = sum(1 for review in skipped if review.useful)
        saved = len(skipped) * per_call
        mark = "  <-- safe" if floor <= lowest else ""
        print(f"  {floor:<8.0f}{len(skipped):>15}{saved:>9.2f}{lost:>14}{mark}")

    safe = max((floor for floor in _candidate_floors(reviews) if floor <= lowest), default=0.0)
    skipped = sum(1 for review in reviews if review.conviction < safe)
    if safe > 0 and skipped:
        print(
            f"\n  A floor at {safe:.0f} would have skipped {skipped} calls "
            f"(~${skipped * per_call:.2f}) without\n"
            f"  losing a single useful answer, in this sample. Out of sample it will "
            "eventually\n  cost one, which is the argument for setting it a band lower "
            "than the best fit."
        )
    else:
        print(
            "\n  No floor skips anything without cost here: the useful answers reach\n"
            "  down into the lowest band. The engine's score does not separate them,\n"
            "  and a cheaper filter has to come from somewhere else."
        )


def _report_refusals(reviews: Sequence[Review]) -> None:
    """What the reviewer actually keeps saying, grouped.

    "Claude vetoes everything" is a feeling until the refusals are counted by
    what they object to. Three possibilities and they call for opposite
    responses: it is repeating one objection, in which case the engine is
    producing one kind of bad setup and that is fixable upstream; it is
    objecting to something different every time, in which case the setups are
    genuinely varied and genuinely weak; or the confidences are all clustered
    at one value, which is not judgement at all but a prompt or a threshold
    doing the deciding.
    """
    refused = [review for review in reviews if not review.useful]
    if not refused:
        return

    print("\nWHAT THE REFUSALS ACTUALLY SAY\n")
    themes: dict[str, int] = defaultdict(int)
    for review in refused:
        themes[_theme(review.thesis)] += 1
    for theme, count in sorted(themes.items(), key=lambda item: -item[1])[:8]:
        print(f"  {count:>5}x  {theme}")

    # A cluster is the tell. Four independent reads of four different charts
    # landing within a few hundredths of each other is not four judgements.
    spread = {round(review.verdict_confidence, 1) for review in refused}
    lowest = min(review.verdict_confidence for review in refused)
    highest = max(review.verdict_confidence for review in refused)
    print(
        f"\n  the reviewer's own confidence on refusals: {lowest:.2f} to "
        f"{highest:.2f}, {len(spread)} distinct bands"
    )
    if len(spread) <= 2:
        print(
            "  Almost every refusal came back at the same confidence. That is not\n"
            "  the shape of independent judgement on different charts — look at the\n"
            "  prompt and at `ai.minimum_confidence` before blaming the setups."
        )


def _theme(thesis: str) -> str:
    """Collapse a refusal to its first clause, so repeats group together."""
    text = " ".join(thesis.split())
    if not text:
        return "(no reasoning recorded)"
    for stop in (". ", "; "):
        if stop in text:
            text = text.split(stop)[0]
            break
    return text[:96] + ("…" if len(text) > 96 else "")


def _report_repeats(reviews: Sequence[Review]) -> None:
    """How often the same market and side was bought twice within the hour.

    The open question behind restructuring the payload for prompt caching. A
    cached prefix only pays for itself if the same question comes back while
    the cache is still warm; a 25% write premium against a 90% read discount
    needs roughly one repeat in three to break even.
    """
    last: dict[tuple[str, str], datetime] = {}
    repeats = 0
    for review in sorted(reviews, key=lambda item: item.when):
        key = (review.symbol, review.direction)
        previous = last.get(key)
        if previous is not None and review.when - previous <= timedelta(hours=1):
            repeats += 1
        last[key] = review.when

    share = repeats / len(reviews) if reviews else 0.0
    print("\nREPEAT RATE\n")
    print(
        f"  {repeats} of {len(reviews)} calls ({share:.0%}) asked about a market and side\n"
        "  already reviewed within the hour."
    )
    print(
        "  Below the ~33% break-even for caching the payload prefix: the write\n"
        "  premium would cost more than the read discount returns."
        if share < 0.33
        else "  Above the ~33% break-even, so caching the payload prefix would pay."
    )
    print()


# ----------------------------------------------------------------------- load


def _candidate_floors(reviews: Sequence[Review]) -> list[float]:
    top = max((review.conviction for review in reviews), default=0.0)
    return [floor * BAND for floor in range(1, int(top // BAND) + 1)]


def _load(path: Path, since: datetime) -> list[Review]:
    """Join each request to its response on cycle_id.

    Requests without a response are dropped rather than counted as refusals: a
    call that crashed or timed out is a different event from one that came back
    negative, and folding them together would inflate the veto rate with the
    system's own outages.
    """
    requests: dict[str, Mapping[str, object]] = {}
    reviews: list[Review] = []
    for row in _rows(path):
        event = row.get("event")
        if event not in {"pretrade_request", "pretrade_response"}:
            continue
        key = str(row.get("cycle_id", "")) + str(row.get("symbol", ""))
        if event == "pretrade_request":
            requests[key] = row
            continue
        request = requests.pop(key, None)
        if request is None:
            continue
        when = _moment(row.get("timestamp"))
        if when is None or when < since:
            continue
        review = _join(request, row, when)
        if review is not None:
            reviews.append(review)
    return reviews


def _join(
    request: Mapping[str, object], response: Mapping[str, object], when: datetime
) -> Review | None:
    payload = request.get("request")
    decision = response.get("decision")
    if not isinstance(payload, Mapping) or not isinstance(decision, Mapping):
        return None
    score = _number(payload.get("score"))
    confidence = _number(payload.get("confidence"))
    proposal = payload.get("executable_proposal")
    family = ""
    if isinstance(proposal, Mapping):
        family = str(proposal.get("setup_family", ""))
    usage = decision.get("usage")
    cost = (
        call_cost(str(decision.get("model", "")), usage)
        if isinstance(usage, Mapping) and usage
        else None
    )
    return Review(
        when=when,
        symbol=str(response.get("symbol", "")),
        direction=str(response.get("direction", "")),
        score=score,
        confidence=confidence,
        # Recomputed rather than read from the briefing. `standing_this_cycle`
        # is only present once the local memory has evidence, so reading it
        # would silently drop every call made before the memory filled -- which
        # is most of them, and exactly the period worth measuring.
        conviction=score * confidence,
        setup_family=family,
        thesis=str(decision.get("thesis", "") or ""),
        verdict_confidence=_number(decision.get("confidence")),
        approved=bool(decision.get("approved")),
        said_yes=bool(decision.get("said_yes") or decision.get("approved")),
        waiting=str(decision.get("entry_timing", "")) == "WAIT_RETEST",
        cost=cost,
        replayed=bool(decision.get("replayed")),
    )


def _rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _moment(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
