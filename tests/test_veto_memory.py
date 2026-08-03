"""The refusal memory: it must forget on change and escalate on repetition."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from advisory.veto_memory import VetoMemory

NOW = datetime(2026, 8, 3, 11, 0, tzinfo=UTC)
ATR = 10.0


def memory(tmp_path: Path, **kwargs: object) -> VetoMemory:
    return VetoMemory(tmp_path / "veto.json", **kwargs)  # type: ignore[arg-type]


def refuse(store: VetoMemory, *, entry: float = 7500.0, stop: float = 7450.0, at=NOW):  # type: ignore[no-untyped-def]
    return store.remember(
        "SPX500",
        "LONG",
        entry=entry,
        stop=stop,
        atr=ATR,
        thesis="rallied into H4 resistance",
        confidence=0.32,
        now=at,
    )


def test_the_same_setup_is_not_asked_again(tmp_path: Path) -> None:
    store = memory(tmp_path)
    refuse(store)
    assert store.recall("SPX500", "LONG", 7500.0, 7450.0, NOW) is not None


def test_a_new_bar_alone_does_not_reopen_the_question(tmp_path: Path) -> None:
    """The bug this exists to fix: three identical vetoes two minutes apart."""
    store = memory(tmp_path)
    refuse(store)
    later = NOW + timedelta(hours=1, minutes=5)
    assert store.recall("SPX500", "LONG", 7500.0, 7450.0, later) is not None


def test_a_materially_different_setup_is_asked_afresh(tmp_path: Path) -> None:
    store = memory(tmp_path)
    refuse(store, entry=7500.0, stop=7450.0)
    # Entry has moved a full ATR — four times the tolerance. Different setup.
    assert store.recall("SPX500", "LONG", 7510.0, 7450.0, NOW) is None


def test_drift_inside_the_tolerance_is_the_same_setup(tmp_path: Path) -> None:
    store = memory(tmp_path)
    refuse(store, entry=7500.0, stop=7450.0)
    # Two points on a 10-point ATR is 0.2 ATR, inside the 0.25 default.
    assert store.recall("SPX500", "LONG", 7502.0, 7451.0, NOW) is not None


def test_the_other_direction_is_a_different_question(tmp_path: Path) -> None:
    store = memory(tmp_path)
    refuse(store)
    assert store.recall("SPX500", "SHORT", 7500.0, 7550.0, NOW) is None


def test_the_refusal_expires(tmp_path: Path) -> None:
    store = memory(tmp_path, base_minutes=90.0)
    refuse(store)
    assert store.recall("SPX500", "LONG", 7500.0, 7450.0, NOW + timedelta(minutes=91)) is None


def test_repetition_buys_a_longer_silence(tmp_path: Path) -> None:
    store = memory(tmp_path, base_minutes=60.0)
    first = refuse(store)
    assert first.repeats == 1
    second = refuse(store, at=NOW + timedelta(minutes=30))
    assert second.repeats == 2
    # 60 minutes doubled, measured from the second refusal.
    assert second.expires_at == NOW + timedelta(minutes=30) + timedelta(minutes=120)


def test_escalation_is_capped(tmp_path: Path) -> None:
    store = memory(tmp_path, base_minutes=60.0, max_minutes=180.0)
    at = NOW
    for _ in range(6):
        record = refuse(store, at=at)
        at += timedelta(minutes=5)
    assert record.expires_at - at + timedelta(minutes=5) == timedelta(minutes=180)


def test_a_gap_resets_the_escalation(tmp_path: Path) -> None:
    store = memory(tmp_path, base_minutes=60.0)
    refuse(store)
    # Long after the refusal expired and past the grace window: a fresh argument.
    later = refuse(store, at=NOW + timedelta(hours=5))
    assert later.repeats == 1


def test_an_approval_clears_the_refusal(tmp_path: Path) -> None:
    store = memory(tmp_path)
    refuse(store)
    store.clear("SPX500", "LONG")
    assert store.recall("SPX500", "LONG", 7500.0, 7450.0, NOW) is None


def test_it_survives_a_restart(tmp_path: Path) -> None:
    refuse(memory(tmp_path))
    reopened = VetoMemory(tmp_path / "veto.json")
    assert reopened.recall("SPX500", "LONG", 7500.0, 7450.0, NOW) is not None


def test_a_zero_atr_falls_back_to_exact_comparison(tmp_path: Path) -> None:
    """No scale means no tolerance, which errs toward asking again."""
    store = memory(tmp_path)
    store.remember(
        "WHEAT",
        "LONG",
        entry=550.0,
        stop=540.0,
        atr=0.0,
        thesis="no history",
        confidence=0.3,
        now=NOW,
    )
    assert store.recall("WHEAT", "LONG", 550.0, 540.0, NOW) is not None
    assert store.recall("WHEAT", "LONG", 550.01, 540.0, NOW) is None


def test_active_lists_only_live_refusals(tmp_path: Path) -> None:
    store = memory(tmp_path, base_minutes=60.0)
    refuse(store)
    assert len(store.active(NOW)) == 1
    assert store.active(NOW + timedelta(minutes=61)) == []


def test_unreadable_file_starts_empty_rather_than_crashing(tmp_path: Path) -> None:
    path = tmp_path / "veto.json"
    path.write_text("{ not json", encoding="utf-8")
    assert VetoMemory(path).recall("SPX500", "LONG", 1.0, 0.9, NOW) is None
