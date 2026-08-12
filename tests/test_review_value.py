"""Is the review budget buying anything, and could a cheap filter replace it?

Twelve hours of live running: 44,061 decisions, 359 reached Claude, 348 came
back refused, 3 became trades. At the measured per-call rate that is roughly
forty dollars a day against an account holding a hundred and fifty euros, so
whether the reviewer is right on every one of them stops being the question.

The tests here hold the two properties that make the report worth acting on.
The first is that "what a floor would save" is never reported without what it
would cost — a floor that saves most of the spend by discarding the day's only
approval is the off switch with a percentage attached. The second is that a
period containing no approvals produces no recommendation at all, because a
filter fitted to it would be fitted to the absence of evidence and would go on
refusing after conditions changed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.review_value import main

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)

USAGE = {
    "input_tokens": 20_000,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 400,
}


def review(
    handle,  # type: ignore[no-untyped-def]
    *,
    cycle: str,
    symbol: str = "EURUSD.i",
    direction: str = "LONG",
    score: float = 50.0,
    confidence: float = 0.8,
    approved: bool = False,
    waiting: bool = False,
    minutes_ago: float = 0.0,
    replayed: bool = False,
) -> None:
    when = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    handle.write(
        json.dumps(
            {
                "timestamp": when,
                "event": "pretrade_request",
                "cycle_id": cycle,
                "symbol": symbol,
                "request": {
                    "score": score,
                    "confidence": confidence,
                    "executable_proposal": {"setup_family": "trend"},
                },
            }
        )
        + "\n"
    )
    handle.write(
        json.dumps(
            {
                "timestamp": when,
                "event": "pretrade_response",
                "cycle_id": cycle,
                "symbol": symbol,
                "direction": direction,
                "decision": {
                    "approved": approved,
                    "said_yes": approved,
                    "entry_timing": "WAIT_RETEST" if waiting else "ENTER_NOW",
                    "model": "claude-sonnet-5",
                    "usage": USAGE,
                    "replayed": replayed,
                },
            }
        )
        + "\n"
    )


@pytest.fixture
def ledger(tmp_path: Path):  # type: ignore[no-untyped-def]
    path = tmp_path / "ai_reviews.jsonl"

    def write(*, vetoes: int, approvals: int, approval_score: float = 70.0) -> Path:
        with path.open("w", encoding="utf-8") as handle:
            for i in range(vetoes):
                review(handle, cycle=f"v{i}", score=30.0, minutes_ago=i)
            for i in range(approvals):
                review(
                    handle,
                    cycle=f"a{i}",
                    score=approval_score,
                    approved=True,
                    minutes_ago=100 + i,
                )
        return path

    return write


class TestTheCostIsStatedInMoney:
    def test_it_reports_spend_per_approval_not_only_per_call(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """Per-call is the number that looks affordable. Per-approval is the
        one that says whether the budget is buying anything."""
        path = ledger(vetoes=9, approvals=1)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "10  reviews on record" in out
        assert "of review spend per approval" in out

    def test_a_replayed_verdict_is_not_billed_again(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """A cached answer keeps the original call's token counts so the audit
        trail can point at the call it came from. Billing them twice reports a
        cost nobody is incurring."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            review(handle, cycle="paid", score=30.0)
            review(handle, cycle="free", score=30.0, replayed=True, minutes_ago=1)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "1  served from cache, free" in out
        assert "1  paid calls" in out


class TestAFloorIsNeverQuotedWithoutItsCost:
    def test_the_useful_answers_lost_are_shown_beside_the_saving(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        path = ledger(vetoes=20, approvals=2, approval_score=70.0)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "useful lost" in out
        assert "WHAT A PRE-REVIEW FLOOR WOULD HAVE DONE" in out

    def test_a_floor_under_every_approval_is_marked_safe(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """Vetoes at conviction 24, approvals at 56: a floor between them skips
        every wasted call and loses nothing."""
        path = ledger(vetoes=20, approvals=2, approval_score=70.0)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "<-- safe" in out
        assert "without\n  losing a single useful answer" in out

    def test_no_floor_is_recommended_when_the_score_does_not_separate(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """Approvals down in the same band as the refusals means no cheap
        filter exists, and saying so is the finding."""
        path = ledger(vetoes=20, approvals=2, approval_score=30.0)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "No floor skips anything without cost here" in out

    def test_a_period_with_no_approvals_recommends_nothing(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """The dangerous case. Every call refused looks like the strongest
        possible argument for a filter, and it is the weakest: there is no
        evidence of where the line is, so any floor fitted here is fitted to
        the absence of approvals and keeps refusing once that changes."""
        path = ledger(vetoes=30, approvals=0)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "no boundary to find here" in out
        assert "<-- safe" not in out
        assert "WHAT A PRE-REVIEW FLOOR WOULD HAVE DONE" not in out


class TestWhatCountsAsUseful:
    def test_a_retest_request_is_not_waste(self, ledger, capsys) -> None:  # type: ignore[no-untyped-def]
        """WAIT_RETEST means the thesis held and the price did not. A filter
        trained to skip those would be skipping setups the reviewer wanted."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            for i in range(9):
                review(handle, cycle=f"v{i}", score=30.0, minutes_ago=i)
            review(handle, cycle="w", score=70.0, waiting=True, minutes_ago=100)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "1  asked for a retest" in out
        assert "Lowest conviction that produced anything useful: 56.0" in out


class TestTheRepeatRate:
    def test_the_same_market_asked_twice_within_the_hour_is_counted(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """The open question behind caching the payload prefix: a warm cache
        only pays if the same question comes back while it is still warm."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            review(handle, cycle="a", symbol="EURUSD.i", minutes_ago=10)
            review(handle, cycle="b", symbol="EURUSD.i", minutes_ago=5)
            review(handle, cycle="c", symbol="XAUUSD.i", minutes_ago=4)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "1 of 3 calls (33%)" in out

    def test_the_same_market_a_day_later_is_not_a_repeat(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            review(handle, cycle="a", symbol="EURUSD.i", minutes_ago=1500)
            review(handle, cycle="b", symbol="EURUSD.i", minutes_ago=5)

        main(["--ledger", str(path), "--hours", "48"])

        assert "0 of 2 calls (0%)" in capsys.readouterr().out


class TestItSurvivesAnImperfectLedger:
    def test_a_request_with_no_response_is_not_counted_as_a_refusal(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """A call that timed out is an outage, not a veto, and folding the two
        together inflates the refusal rate with the system's own failures."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            review(handle, cycle="ok", score=30.0)
            handle.write(
                json.dumps(
                    {
                        "timestamp": NOW.isoformat(),
                        "event": "pretrade_request",
                        "cycle_id": "lost",
                        "symbol": "GBPUSD.i",
                        "request": {"score": 40.0, "confidence": 0.9},
                    }
                )
                + "\n"
            )

        main(["--ledger", str(path), "--hours", "48"])

        assert "1  reviews on record" in capsys.readouterr().out

    def test_a_missing_ledger_says_so_rather_than_raising(self, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        assert main(["--ledger", str(tmp_path / "nothing.jsonl")]) == 1
        assert "No AI ledger" in capsys.readouterr().out


class TestWhatTheRefusalsSay:
    """ "Claude vetoes everything" is a feeling until the refusals are counted.

    Three shapes, three opposite responses: one objection repeated means the
    engine produces one kind of bad setup and it is fixable upstream; a
    different objection each time means the setups are varied and genuinely
    weak; and confidences all landing on one value is not judgement at all.
    """

    def refusal(self, handle, *, cycle: str, thesis: str, confidence: float) -> None:  # type: ignore[no-untyped-def]
        when = NOW.isoformat()
        handle.write(
            json.dumps(
                {
                    "timestamp": when,
                    "event": "pretrade_request",
                    "cycle_id": cycle,
                    "symbol": "EURCAD.i",
                    "request": {"score": 45.0, "confidence": 0.7},
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": when,
                    "event": "pretrade_response",
                    "cycle_id": cycle,
                    "symbol": "EURCAD.i",
                    "direction": "SHORT",
                    "decision": {
                        "approved": False,
                        "said_yes": False,
                        "confidence": confidence,
                        "thesis": thesis,
                        "entry_timing": "ENTER_NOW",
                        "model": "claude-sonnet-5",
                        "usage": USAGE,
                    },
                }
            )
            + "\n"
        )

    def test_one_objection_repeated_is_counted_as_one(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """The fixable case: the same complaint every time names something
        upstream to change."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            for i in range(6):
                self.refusal(
                    handle,
                    cycle=f"r{i}",
                    thesis=f"Counter-trend against H4. Price at {1.6 + i / 1000:.4f}.",
                    confidence=0.30 + i / 100,
                )

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "6x  Counter-trend against H4" in out

    def test_a_cluster_of_identical_confidences_is_called_out(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """Four reads of four charts landing within a hundredth of each other
        is not four judgements, and blaming the setups for it wastes days."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            for i, confidence in enumerate([0.32, 0.32, 0.32, 0.28]):
                self.refusal(handle, cycle=f"r{i}", thesis="Weak structure.", confidence=confidence)

        main(["--ledger", str(path), "--hours", "48"])
        out = capsys.readouterr().out

        assert "distinct bands" in out
        assert "not\n  the shape of independent judgement" in out

    def test_varied_confidences_are_not_flagged(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            for i, confidence in enumerate([0.10, 0.35, 0.52, 0.71]):
                self.refusal(handle, cycle=f"r{i}", thesis="Weak structure.", confidence=confidence)

        main(["--ledger", str(path), "--hours", "48"])

        assert "not\n  the shape of independent judgement" not in capsys.readouterr().out

    def test_a_refusal_with_no_reasoning_is_named_rather_than_dropped(
        self, ledger, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """Silently dropping them would report "no refusals" on a run where
        every single call failed to record its reasoning."""
        path = ledger(vetoes=0, approvals=0)
        with path.open("w", encoding="utf-8") as handle:
            self.refusal(handle, cycle="r0", thesis="", confidence=0.3)

        main(["--ledger", str(path), "--hours", "48"])

        assert "(no reasoning recorded)" in capsys.readouterr().out
