"""Learning why setups are refused, not just which ones were.

`VetoMemory` remembers the shape of a proposal and stays quiet while it holds.
That is the right memory for "you already asked me this", and the wrong one for
what a live account produced: five GBPCAD longs at five different entries, in
five different cycles, every one refused as a counter-trend long.

The shape moved each time, so the shape memory forgot. The flaw never moved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from advisory.veto_patterns import VetoPatterns, classify, readable

NOW = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)


def memory(tmp_path: Path, **kwargs) -> VetoPatterns:  # type: ignore[no-untyped-def]
    return VetoPatterns(tmp_path / "veto_patterns.json", **kwargs)


class TestClassify:
    """Claude writes prose; three sentences can be one observation."""

    def test_the_live_countertrend_wordings_are_one_tag(self) -> None:
        for phrase in (
            "Counter-trend long",
            "Countertrend short against D1",
            "counter trend entry",
            "fighting against the higher timeframe",
        ):
            assert "countertrend" in classify([phrase]), phrase

    def test_the_live_stop_wording_is_recognised(self) -> None:
        assert "stop-too-tight" in classify(["Stop is only 0.8 ATR"])

    def test_the_live_conviction_wordings_are_recognised(self) -> None:
        assert "low-conviction" in classify(["This is the weakest setup of the 10 candidates"])
        assert "low-conviction" in classify(["ranked dead last (11 of 11)"])

    def test_one_risk_can_carry_two_tags(self) -> None:
        tags = classify(["Counter-trend long and the stop is only 0.6 ATR"])
        assert set(tags) == {"countertrend", "stop-too-tight"}

    def test_an_unrecognised_objection_teaches_nothing(self) -> None:
        """The safe answer. Inventing a tag would silence future questions on
        evidence that was never understood."""
        assert classify(["the moon is in the wrong house"]) == ()

    def test_it_falls_back_to_the_thesis_only_when_risks_are_empty(self) -> None:
        assert classify([], "This is a counter-trend long") == ("countertrend",)
        # With risks present the thesis is ignored: a full paragraph matches
        # half the vocabulary and dilutes everything.
        assert classify(["spread"], "counter-trend and momentum and range") == ("spread",)

    def test_empty_input_is_empty_output(self) -> None:
        assert classify([], "") == ()


class TestEstablishedPattern:
    def test_one_refusal_is_an_anecdote(self, tmp_path: Path) -> None:
        patterns = memory(tmp_path)
        patterns.remember("GBPCAD.i", "LONG", risks=["Counter-trend long"], thesis="", now=NOW)
        assert patterns.established("GBPCAD.i", "LONG", NOW) is None

    def test_two_is_still_not_enough(self, tmp_path: Path) -> None:
        """The same market at two nearby moments is one observation twice."""
        patterns = memory(tmp_path)
        for minute in (0, 3):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(minutes=3)) is None

    def test_the_live_gbpcad_case_is_caught_on_the_third(self, tmp_path: Path) -> None:
        """Five cycles, five entries, one flaw. The fifth is never bought."""
        patterns = memory(tmp_path)
        for minute in (0, 3, 6):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long, the bars support the opposite"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        found = patterns.established("GBPCAD.i", "LONG", NOW + timedelta(minutes=6))
        assert found is not None
        assert found.tag == "countertrend"
        assert found.occurrences == 3
        assert "3 refusals" in found.describe()

    def test_the_other_direction_is_a_different_question(self, tmp_path: Path) -> None:
        """If longs are counter-trend, shorts are exactly what to look at."""
        patterns = memory(tmp_path)
        for minute in (0, 3, 6):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        assert patterns.established("GBPCAD.i", "SHORT", NOW + timedelta(minutes=6)) is None

    def test_a_different_symbol_is_untouched(self, tmp_path: Path) -> None:
        patterns = memory(tmp_path)
        for minute in (0, 3, 6):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        assert patterns.established("EURUSD.i", "LONG", NOW + timedelta(minutes=6)) is None

    def test_three_different_objections_establish_nothing(self, tmp_path: Path) -> None:
        """Three refusals for three unrelated reasons is not a pattern.

        It is an instrument having a bad morning, and the next proposal may be
        fine. Only a repeated *same* objection predicts the next answer.
        """
        patterns = memory(tmp_path)
        for minute, risk in ((0, "Counter-trend long"), (3, "spread too wide"), (6, "stale quote")):
            patterns.remember(
                "GBPCAD.i", "LONG", risks=[risk], thesis="", now=NOW + timedelta(minutes=minute)
            )
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(minutes=6)) is None


class TestItExpires:
    def test_evidence_ages_out_of_the_window(self, tmp_path: Path) -> None:
        patterns = memory(tmp_path, window_hours=6.0)
        for minute in (0, 3, 6):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(hours=5)) is not None
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(hours=7)) is None

    def test_a_pattern_that_keeps_recurring_keeps_applying(self, tmp_path: Path) -> None:
        """The window rolls; it does not expire a live pattern."""
        patterns = memory(tmp_path, window_hours=6.0)
        for hour in range(0, 12, 2):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(hours=hour),
            )
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(hours=10)) is not None


class TestAnApprovalWipesIt:
    def test_clear_removes_the_pattern_outright(self, tmp_path: Path) -> None:
        """Not weakened — wrong by demonstration.

        The reviewer has just approved exactly the pair the pattern called
        hopeless. Anything softer lets a bad early run silence an instrument
        long after the market has moved on.
        """
        patterns = memory(tmp_path)
        for minute in (0, 3, 6):
            patterns.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(minutes=6)) is not None

        patterns.clear("GBPCAD.i", "LONG")
        assert patterns.established("GBPCAD.i", "LONG", NOW + timedelta(minutes=6)) is None

    def test_clearing_one_pair_leaves_the_other_direction_alone(self, tmp_path: Path) -> None:
        patterns = memory(tmp_path)
        for direction in ("LONG", "SHORT"):
            for minute in (0, 3, 6):
                patterns.remember(
                    "GBPCAD.i",
                    direction,
                    risks=["Counter-trend"],
                    thesis="",
                    now=NOW + timedelta(minutes=minute),
                )
        patterns.clear("GBPCAD.i", "LONG")
        assert patterns.established("GBPCAD.i", "SHORT", NOW + timedelta(minutes=6)) is not None


class TestItSurvivesARestart:
    def test_patterns_are_written_and_read_back(self, tmp_path: Path) -> None:
        """Stopping the service must not be a way to re-buy known refusals."""
        first = memory(tmp_path)
        for minute in (0, 3, 6):
            first.remember(
                "GBPCAD.i",
                "LONG",
                risks=["Counter-trend long"],
                thesis="",
                now=NOW + timedelta(minutes=minute),
            )
        restarted = memory(tmp_path)
        found = restarted.established("GBPCAD.i", "LONG", NOW + timedelta(minutes=6))
        assert found is not None
        assert found.occurrences == 3

    def test_a_corrupt_file_starts_empty_rather_than_failing(self, tmp_path: Path) -> None:
        """Not worth ending a trading session over; it rebuilds in three refusals."""
        path = tmp_path / "veto_patterns.json"
        path.write_text("{ not json", encoding="utf-8")
        assert VetoPatterns(path).established("GBPCAD.i", "LONG", NOW) is None


def test_every_tag_has_an_operator_sentence() -> None:
    """The deck shows these. A raw slug on screen explains nothing."""
    from advisory.veto_patterns import _TAGS

    for tag, _ in _TAGS:
        assert readable(tag) != tag, tag


def test_an_unknown_tag_degrades_to_itself() -> None:
    assert readable("something-new") == "something-new"


def test_the_live_session_in_full(tmp_path: Path) -> None:
    """What one live hour on GBPCAD would now cost, end to end.

    The recorded session: five GBPCAD LONG proposals reaching the reviewer at
    08:31, 08:34, 08:36, 08:39 and 08:41, each refused as a counter-trend long.
    Four of those were already free — the review cache replayed the verdict for
    an unchanged setup — but the shape memory forgets the moment an entry
    moves, so on a day when it moves the account buys all five.

    With the pattern memory the third refusal establishes the reason and the
    fourth and fifth are never asked. Two paid calls instead of five.
    """
    patterns = VetoPatterns(tmp_path / "veto_patterns.json")
    start = datetime(2026, 8, 6, 8, 31, tzinfo=UTC)
    offsets = (0, 3, 5, 8, 10)

    paid = 0
    for minutes in offsets:
        now = start + timedelta(minutes=minutes)
        if patterns.established("GBPCAD.i", "LONG", now) is not None:
            continue
        paid += 1
        patterns.remember(
            "GBPCAD.i",
            "LONG",
            risks=["Counter-trend long, the bars support the opposite direction"],
            thesis="This is the weakest setup of the 10 tradeable candidates this cycle",
            now=now,
        )

    assert paid == 3
    established = patterns.established("GBPCAD.i", "LONG", start + timedelta(minutes=10))
    assert established is not None
    assert established.tag == "countertrend"

    # And the moment the reviewer approves a GBPCAD long, the silence lifts —
    # the pattern claimed that trade was hopeless and it was just taken.
    patterns.clear("GBPCAD.i", "LONG")
    assert patterns.established("GBPCAD.i", "LONG", start + timedelta(minutes=11)) is None
