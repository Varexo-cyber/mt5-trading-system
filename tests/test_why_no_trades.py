"""Refused before the review, or after paying for it?

Both land in the journal as the same reason -- ENTRY_OVEREXTENDED either way --
and they are opposite findings. Refused before is the system working: the gate
cost nothing and saved a paid opinion. Refused after is money spent on an answer
that was discarded, because every entry gate runs again on a fresh quote once
Claude replies, and a forty-second reply is long enough for a setup sitting on
its limit to cross it.

Live case behind this. GBPCAD produced 129 decisions in two hours and no trade:
39 ENTRY_OVEREXTENDED, one paid approval at 16:31 that never opened. Whether
that approval was thrown away by the recheck was unanswerable from the report,
so the cost of the review budget could not be seen at all. The runner has
recorded `post_review_revalidation` on exactly those rows the whole time and
nothing read it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.why_no_trades import main

BEFORE = "{}"
AFTER = json.dumps({"post_review_revalidation": {"entry_quality": {"decision": "WAIT_RETEST"}}})


@pytest.fixture
def journal(tmp_path: Path) -> Path:
    path = tmp_path / "trading.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE analysis_cycles (id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, "
        "decision TEXT, reason TEXT, detail TEXT, total_score REAL, score_threshold REAL, "
        "context_json TEXT DEFAULT '{}')"
    )
    db.commit()
    db.close()
    return path


def add(journal: Path, reason: str, context: str, count: int = 1) -> None:
    db = sqlite3.connect(journal)
    now = datetime.now(UTC)
    for i in range(count):
        db.execute(
            "INSERT INTO analysis_cycles (ts, symbol, decision, reason, detail, context_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                (now - timedelta(minutes=i)).isoformat(),
                "GBPCAD.i",
                "SKIP",
                reason,
                "SHORT price is at 96% of its directional 12-bar range",
                context,
            ),
        )
    db.commit()
    db.close()


class TestTheCostOfADiscardedApproval:
    def test_a_refusal_after_payment_is_called_out(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        add(journal, "ENTRY_OVEREXTENDED", BEFORE, count=39)
        add(journal, "ENTRY_OVEREXTENDED", AFTER, count=2)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "2 of these were refused AFTER a paid review" in out

    def test_gate_refusals_alone_say_nothing_about_paid_reviews(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """The quiet case is the good one and must not be dressed up as waste."""
        add(journal, "ENTRY_OVEREXTENDED", BEFORE, count=39)

        main(["--db", str(journal), "--hours", "4"])

        assert "refused AFTER a paid review" not in capsys.readouterr().out

    def test_each_reason_is_counted_separately(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Which gate discards the approvals decides what there is to fix."""
        add(journal, "ENTRY_OVEREXTENDED", AFTER, count=3)
        add(journal, "ENTRY_MOVED_DURING_REVIEW", AFTER, count=1)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "4 of these were refused AFTER a paid review" in out
        assert "ENTRY_OVEREXTENDED" in out
        assert "ENTRY_MOVED_DURING_REVIEW" in out

    def test_an_older_journal_without_the_column_is_not_required(self, journal: Path) -> None:
        """The marker is absent on every row written before the runner added it,
        which reads as "none were discarded" rather than as an error."""
        add(journal, "NO_SIGNAL", BEFORE, count=5)
        assert main(["--db", str(journal), "--hours", "4"]) == 0
