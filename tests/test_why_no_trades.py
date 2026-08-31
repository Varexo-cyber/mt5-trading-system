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


class TestWhereTheCandidatesActuallyGo:
    """ "Does anything reach the reviewer at all" was unanswerable from this report.

    A league table of reasons cannot answer it. NO_SIGNAL is nearly always the
    top row and nearly always should be, and it says nothing about whether the
    eleven gates behind it are passable. Twelve serial gates each taking a
    modest share leave very little at the end, and only a stage-by-stage count
    shows which one is doing it.
    """

    def test_it_prints_how_many_got_as_far_as_being_paid_for(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        add(journal, "NO_SIGNAL", BEFORE, count=90)
        add(journal, "ENTRY_OVEREXTENDED", BEFORE, count=6)
        add(journal, "AI_VETO", BEFORE, count=4)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "4  reached the paid reviewer   (4.0% of everything scanned)" in out

    def test_a_gate_that_stops_everything_is_named_before_the_reviewer(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """The case the operator keeps hitting: a whole night, no trades, and
        the review budget blamed for something no candidate ever got near."""
        add(journal, "SPREAD_EATS_THE_STOP", BEFORE, count=50)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "-50  can the trade pay its own costs" in out
        assert "50  SPREAD_EATS_THE_STOP" in out
        assert "Nothing was sent to Claude at all" in out

    def test_cost_stage_separates_target_math_from_account_size(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        add(journal, "TARGET_RARELY_REACHED", BEFORE, count=40)
        add(journal, "SL_TOO_TIGHT_FOR_COSTS", BEFORE, count=9)
        add(journal, "TRADE_SKIPPED_UNDERCAPITALIZED", BEFORE, count=2)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "-51  can the trade pay its own costs" in out
        assert "40  TARGET_RARELY_REACHED" in out
        assert "9  SL_TOO_TIGHT_FOR_COSTS" in out
        assert "2  TRADE_SKIPPED_UNDERCAPITALIZED" in out

    def test_a_reason_missing_from_the_stage_table_is_shown_not_swallowed(
        self, journal: Path, capsys
    ) -> None:
        """A reason added to the enum and not added here would otherwise
        silently inflate the survivor count -- the one number this section
        exists to get right."""
        add(journal, "NO_SIGNAL", BEFORE, count=10)
        add(journal, "SOME_FUTURE_GATE", BEFORE, count=3)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "not yet classified: SOME_FUTURE_GATE" in out
        assert "0  reached the paid reviewer" in out

    def test_a_discarded_approval_is_counted_below_the_review_not_above_it(
        self, journal: Path, capsys
    ) -> None:
        """It survived every free gate and was paid for. Counting it as a
        pre-review loss would hide exactly the waste worth seeing."""
        add(journal, "NO_SIGNAL", BEFORE, count=8)
        add(journal, "ENTRY_MOVED_DURING_REVIEW", AFTER, count=2)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "2  reached the paid reviewer" in out
        assert "-2  approved, then the price moved before the order went out" in out


def test_directional_module_firings_are_reported_before_later_gates(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    db = sqlite3.connect(journal)
    db.execute(
        "CREATE TABLE module_scores (cycle_pk INTEGER, module TEXT, score REAL, "
        "confidence REAL, weight REAL)"
    )
    now = datetime.now(UTC).isoformat()
    db.execute(
        "INSERT INTO analysis_cycles (id, ts, symbol, decision, reason, detail) "
        "VALUES (1, ?, 'EURUSD.i', 'SKIP', 'NO_SIGNAL', 'later gate')",
        (now,),
    )
    db.executemany(
        "INSERT INTO module_scores VALUES (?,?,?,?,?)",
        [
            (1, "fast_ema_cross", 50.0, 0.6, 0.5),
            (1, "m1_micro_breakout", -62.0, 0.7, 0.55),
            (1, "market_regime", -1.0, 1.0, 0.0),
        ],
    )
    db.commit()
    db.close()

    main(["--db", str(journal), "--hours", "4"])
    out = capsys.readouterr().out

    assert "DIRECTIONAL DETECTION BEFORE LATER GATES" in out
    assert "1 LONG / 1 SHORT" in out
    assert "fast_ema_cross" in out
    assert "m1_micro_breakout" in out


class TestTheWarningFlagMeansSomething:
    def test_a_designed_gate_is_not_flagged_as_a_fault(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """A concentration limit refusing a doubled bet is the system working.
        Flagging it alongside a missing calendar taught the operator to ignore
        the flag, which is worse than not having one."""
        add(journal, "CURRENCY_CONCENTRATION", BEFORE, count=5)
        add(journal, "SPREAD_EATS_THE_STOP", BEFORE, count=5)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "! CURRENCY_CONCENTRATION" not in out
        assert "! SPREAD_EATS_THE_STOP" not in out

    def test_a_real_fault_still_is(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """No calendar stops every symbol, all day, silently."""
        add(journal, "NEWS_CALENDAR_UNAVAILABLE", BEFORE, count=5)

        main(["--db", str(journal), "--hours", "4"])

        assert "! NEWS_CALENDAR_UNAVAILABLE" in capsys.readouterr().out


class TestTheBiggestReachableGateIsVisible:
    """The engine writes the measurement into the sentence, so grouping on raw
    text shatters one gate into hundreds of rows.

    A live night read "17432x no weighted directional evidence" at the top and
    three rows of ~250 near the bottom. The ~14,000 decisions refused by the
    score threshold -- the one number in the config an operator can actually
    move -- were nowhere on the screen.
    """

    def test_the_same_gate_at_different_scores_is_one_row(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        db = sqlite3.connect(journal)
        now = datetime.now(UTC)
        for i in range(300):
            db.execute(
                "INSERT INTO analysis_cycles (ts, symbol, decision, reason, detail, context_json) "
                "VALUES (?,?,?,?,?,?)",
                (
                    (now - timedelta(seconds=i)).isoformat(),
                    "EURUSD.i",
                    "SKIP",
                    "NO_SIGNAL",
                    f"confluence score {30 + i * 0.03:.1f} below threshold",
                    BEFORE,
                ),
            )
        db.commit()
        db.close()

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "300x  confluence score N below threshold" in out

    def test_distinct_causes_are_still_distinct(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Collapsing numbers must not collapse the sentences around them."""
        add(journal, "NO_SIGNAL", BEFORE, count=5)
        db = sqlite3.connect(journal)
        db.execute(
            "INSERT INTO analysis_cycles (ts, symbol, decision, reason, detail, context_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                datetime.now(UTC).isoformat(),
                "EURUSD.i",
                "SKIP",
                "NO_SIGNAL",
                "extreme volatility regime",
                BEFORE,
            ),
        )
        db.commit()
        db.close()

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "extreme volatility regime" in out
        assert "1x  extreme volatility regime" in out

    def test_positive_scores_are_not_described_as_funnel_survivors(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        db = sqlite3.connect(journal)
        db.execute(
            "INSERT INTO analysis_cycles "
            "(ts, symbol, decision, reason, detail, total_score, score_threshold, context_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now(UTC).isoformat(),
                "EURUSD.i",
                "SKIP",
                "NO_SIGNAL",
                "reachable target rejected after scoring",
                42.0,
                26.0,
                BEFORE,
            ),
        )
        db.commit()
        db.close()

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "decision rows expose a positive confluence score" in out
        assert "cleared the score threshold" in out
        assert "score diagnostic, not a survivor count" in out


REMEMBERED = json.dumps({"ai_veto_remembered": True, "ai_veto_repeats": 3})


class TestAReplayedRefusalIsNotAPaidOne:
    """The correction that makes the bottom line mean what it says.

    A refusal replayed from the veto memory is journalled as AI_VETO, identical
    in the reason column to one the account paid for. Counting those below the
    line reported 751 paid reviews on a live twelve hours where 84 calls were
    actually made, and turned a $4.79 daily API bill into a claimed $40 one.
    """

    def test_a_remembered_veto_is_counted_before_the_line(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        add(journal, "NO_SIGNAL", BEFORE, count=90)
        add(journal, "AI_VETO", REMEMBERED, count=8)
        add(journal, "AI_VETO", BEFORE, count=2)

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "2  reached the paid reviewer" in out
        assert "-2  Claude declined or asked for a retest" in out

    def test_the_free_replays_are_named_so_the_saving_is_visible(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """The memory absorbing 89% of the refusals is the system working, and
        it was invisible in every report."""
        add(journal, "AI_VETO", REMEMBERED, count=8)
        add(journal, "AI_VETO", BEFORE, count=2)

        main(["--db", str(journal), "--hours", "4"])

        assert "(8 of the refusals above were replayed free from memory)" in capsys.readouterr().out


class TestATimeframeIsNotAMeasurement:
    def test_m5_and_m15_stay_apart(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Stripping digits to group "score 37.4" with "score 37.5" must not
        also fold two different timeframes into one cause."""
        db = sqlite3.connect(journal)
        now = datetime.now(UTC)
        rows = [
            ("M5 price is moving against the short: 1.05 ATR adverse", 4),
            ("M15 price is moving against the long: 2.18 ATR adverse", 6),
        ]
        for detail, count in rows:
            for i in range(count):
                db.execute(
                    "INSERT INTO analysis_cycles (ts, symbol, decision, reason, detail, "
                    "context_json) VALUES (?,?,?,?,?,?)",
                    (
                        (now - timedelta(seconds=i)).isoformat(),
                        "EURUSD.i",
                        "SKIP",
                        "NO_SIGNAL",
                        detail,
                        BEFORE,
                    ),
                )
        db.commit()
        db.close()

        main(["--db", str(journal), "--hours", "4"])
        out = capsys.readouterr().out

        assert "6x  M15 price is moving against the long" in out
        assert "4x  M5 price is moving against the short" in out


class TestSilenceAndAbsenceAreDifferentFindings:
    """A live section that produced nothing, and no way to say which nothing.

    24 hours on the account: 414 setups formed, 444 module firings, and every
    one of them `impulse_retest`. `order_block` -- the other section trading
    real money -- did not appear in the detection table at all.

    That table has `HAVING longs > 0 OR shorts > 0`, so it cannot distinguish
    a section that ran on every cycle and scored zero from a section that
    never ran. Those need opposite responses and they print identically: as
    nothing.
    """

    def _with_modules(self, journal: Path, rows: list[tuple[str, float, float]]) -> None:
        db = sqlite3.connect(journal)
        db.execute(
            "CREATE TABLE IF NOT EXISTS module_scores (id INTEGER PRIMARY KEY, cycle_pk INTEGER, "
            "module TEXT, score REAL, confidence REAL, weight REAL, reasoning TEXT, "
            "details_json TEXT)"
        )
        cycle = db.execute("SELECT MIN(id) FROM analysis_cycles").fetchone()[0] or 1
        for module, score, weight in rows:
            db.execute(
                "INSERT INTO module_scores (cycle_pk, module, score, confidence, weight) "
                "VALUES (?,?,?,?,?)",
                (cycle, module, score, 0.7, weight),
            )
        db.commit()
        db.close()

    def test_it_separates_never_ran_from_found_nothing_from_weighted_away(self) -> None:
        from scripts.why_no_trades import _print_live_section_rollcall

        presence = {
            "impulse_retest": (444, 444, 444),
            "order_block": (2100, 0, 2100),
            "shadowed": (50, 50, 0),
        }

        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_live_section_rollcall(
                presence, ("impulse_retest", "order_block", "shadowed", "absent")
            )
        out = buffer.getvalue()

        assert "impulse_retest" in out and "444 with a direction" in out
        assert "scored 0 every time" in out, "it ran and found nothing"
        assert "WEIGHT 0" in out, "recorded and then multiplied away"
        assert "NO ROWS AT ALL" in out, "never ran"
        assert "absent produced NOTHING" in out

    def test_the_rollcall_prints_even_when_every_section_is_healthy(self) -> None:
        """The hole one level up: a warning that only appears when something is
        wrong cannot be told apart from a check that did not run."""
        import contextlib
        import io

        from scripts.why_no_trades import _print_live_section_rollcall

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_live_section_rollcall({"a": (10, 10, 10)}, ("a",))
        out = buffer.getvalue()

        assert "LIVE SECTIONS" in out
        assert "!" not in out, "nothing is wrong, so nothing may be flagged"

    def test_the_query_keeps_the_zeros(self, journal: Path) -> None:
        """`_directional_modules` drops them, which is the whole defect."""
        add(journal, "NO_SIGNAL", BEFORE, count=1)
        self._with_modules(journal, [("impulse_retest", 60.0, 1.2), ("order_block", 0.0, 1.0)])
        db = sqlite3.connect(journal)
        db.row_factory = sqlite3.Row

        from scripts.why_no_trades import _directional_modules, _module_presence

        seen = _module_presence(db, "1=1", [])
        directional = {str(row["module"]) for row in _directional_modules(db, "1=1", [])}
        db.close()

        assert "order_block" not in directional, "the old query cannot see it"
        assert seen["order_block"] == (1, 0, 1), "the new one says it ran and scored nothing"
        assert seen["impulse_retest"] == (1, 1, 1)

    def test_the_live_list_is_read_from_config_not_restated(self) -> None:
        from scripts.why_no_trades import _live_modules

        assert set(_live_modules()) == {
            "impulse_retest",
            "impulse_retest_m30",
            "order_block_m15",
            "order_block",
            "order_block_fast",
            "order_block_h1",
        }
