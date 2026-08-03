"""The learning loop: lessons must accumulate, decay, and stay bounded."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from learning.memory import MAX_LESSONS, TradingMemory

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def store(tmp_path: Path, **kwargs: object) -> TradingMemory:
    return TradingMemory(tmp_path / "memory.json", **kwargs)  # type: ignore[arg-type]


def test_a_fresh_memory_has_nothing_to_say(tmp_path: Path) -> None:
    memory = store(tmp_path)
    assert not memory.has_evidence()


def test_a_lesson_is_retained(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.record_reflection(
        {"symbol": "SPX500"},
        ("The stop sat inside ordinary M15 noise and was clipped",),
        NOW,
    )
    assert memory.has_evidence()
    assert "inside ordinary M15 noise" in memory.briefing()["lessons"][0]


def test_the_same_lesson_reworded_is_counted_once(tmp_path: Path) -> None:
    """A pattern is a repeated lesson, so counting must survive filler and order.

    Synonyms are explicitly out of scope — see `_normalise`. This covers what it
    does claim to absorb: case, punctuation, word order and filler words.
    """
    memory = store(tmp_path)
    memory.record_reflection({"symbol": "A"}, ("The stop was in the noise band",), NOW)
    memory.record_reflection({"symbol": "B"}, ("Noise band: stop in it.",), NOW)
    lessons = memory.briefing()["lessons"]
    assert len(lessons) == 1
    assert "seen 2x" in lessons[0]
    # And the first wording is kept, so the text does not drift under the count.
    assert lessons[0].startswith("The stop was in the noise band")


def test_genuinely_different_lessons_stay_separate(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.record_reflection({"symbol": "A"}, ("The stop was in the noise band",), NOW)
    memory.record_reflection({"symbol": "B"}, ("The target was never reachable in a day",), NOW)
    assert len(memory.briefing()["lessons"]) == 2


def test_a_repeated_lesson_outranks_a_newer_single_one(tmp_path: Path) -> None:
    memory = store(tmp_path)
    for index in range(3):
        memory.record_reflection(
            {"symbol": f"S{index}"},
            ("Afternoon entries on indices reverse before the close",),
            NOW,
        )
    memory.record_reflection(
        {"symbol": "Z"}, ("A one-off observation about something else",), NOW + timedelta(hours=1)
    )
    assert "Afternoon entries" in memory.briefing()["lessons"][0]


def test_trivially_short_lessons_are_dropped(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.record_reflection({"symbol": "A"}, ("Be careful", "ok"), NOW)
    assert memory.briefing()["lessons"] == []


def test_realised_results_build_a_per_symbol_record(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.record_outcome("SPX500", "LONG", -1.0, NOW)
    memory.record_outcome("SPX500", "LONG", -1.0, NOW)
    memory.record_outcome("SPX500", "LONG", 2.0, NOW)
    brief = memory.briefing("SPX500", "LONG")
    assert brief["closed_trades_recorded"] == 3
    assert brief["cumulative_r"] == 0.0
    assert "3 trades" in brief["this_instrument"]
    assert "33% won" in brief["this_instrument"]


def test_an_instrument_with_no_history_says_so(tmp_path: Path) -> None:
    """Neutral, not suspect — the reviewer must not read absence as evidence."""
    memory = store(tmp_path)
    memory.record_outcome("SPX500", "LONG", 1.0, NOW)
    assert memory.briefing("EURUSD", "SHORT")["this_instrument"] == "no history on record"


def test_vetoes_are_counted_against_the_instrument(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.record_veto("SPX500", "LONG", NOW)
    memory.record_veto("SPX500", "LONG", NOW)
    assert "refused 2x" in memory.briefing("SPX500", "LONG")["this_instrument"]


def test_the_worst_performers_surface(tmp_path: Path) -> None:
    memory = store(tmp_path)
    for _ in range(2):
        memory.record_outcome("BAD", "LONG", -1.5, NOW)
        memory.record_outcome("GOOD", "LONG", 2.0, NOW)
    assert "BAD" in memory.briefing()["worst_performing"][0]


def test_old_observations_decay(tmp_path: Path) -> None:
    memory = store(tmp_path, retention_days=30)
    memory.record_reflection({"symbol": "A"}, ("An observation from long ago about stops",), NOW)
    # Any later write triggers the prune.
    memory.record_outcome("B", "LONG", 1.0, NOW + timedelta(days=40))
    assert memory.briefing()["lessons"] == []


def test_the_lesson_store_stays_bounded(tmp_path: Path) -> None:
    memory = store(tmp_path)
    for index in range(MAX_LESSONS + 25):
        memory.record_reflection(
            {"symbol": "A"}, (f"Distinct observation number {index} about market behaviour",), NOW
        )
    assert len(memory.briefing()["lessons"]) <= 8


def test_it_survives_a_restart(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.record_outcome("SPX500", "LONG", -1.0, NOW)
    memory.record_reflection(
        {"symbol": "SPX500"}, ("A lesson worth keeping across a restart",), NOW
    )
    reopened = TradingMemory(tmp_path / "memory.json")
    assert reopened.briefing()["closed_trades_recorded"] == 1
    assert reopened.briefing()["lessons"]


def test_unreadable_file_starts_empty_rather_than_crashing(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("}}not json", encoding="utf-8")
    assert not TradingMemory(path).has_evidence()
