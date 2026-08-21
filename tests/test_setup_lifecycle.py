from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from analysis.confluence import TradeIdea
from analysis.entry_quality import EntryTimingAssessment, EntryTimingDecision
from analysis.setup_lifecycle import SetupLifecycleBook, SetupState
from core.types import Direction


def idea() -> TradeIdea:
    return TradeIdea(
        symbol="EURUSD.i",
        approved=True,
        direction=Direction.LONG,
        score=45.0,
        confidence=0.7,
        entry=1.10,
        stop_loss=1.09,
        take_profit=1.12,
        reason="test",
        signals=(),
        setup_family="trend_momentum",
        horizon="quick",
        planning_timeframe="M5",
        expected_horizon_minutes=30,
    )


def timing(
    decision: EntryTimingDecision,
    reason: str,
    *,
    directional: float = 0.0,
) -> EntryTimingAssessment:
    return EntryTimingAssessment(
        decision,
        reason,
        reason,
        "M5",
        last_bar_directional_atr=directional,
        reference_atr=0.01,
    )


def book(path: Path) -> SetupLifecycleBook:
    return SetupLifecycleBook(
        path,
        pullback_atr={"quick": 0.2, "intraday": 0.3, "swing": 0.4},
        resumption_atr={"quick": 0.05, "intraday": 0.08, "swing": 0.1},
        expiry_minutes={"quick": 30, "intraday": 240, "swing": 1440},
    )


def test_timely_new_setup_enters_without_artificial_wait(tmp_path: Path) -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    verdict = book(tmp_path / "setups.json").observe(
        idea(),
        timing(EntryTimingDecision.ENTER_NOW, "TIMELY", directional=0.1),
        executable_price=1.10,
        now=now,
        bar_time=now,
    )
    assert verdict.state is SetupState.ENTER_NOW
    assert not verdict.tracked


def test_overextended_setup_enters_when_native_timing_recovers(tmp_path: Path) -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    store = tmp_path / "setups.json"
    lifecycle = book(store)

    late = lifecycle.observe(
        idea(),
        timing(EntryTimingDecision.WAIT_RETEST, "DIRECTIONAL_MOVE_OVEREXTENDED"),
        executable_price=1.110,
        now=now,
        bar_time=now,
    )
    assert late.state is SetupState.WAIT_PULLBACK

    recovered = lifecycle.observe(
        idea(),
        timing(EntryTimingDecision.ENTER_NOW, "TIMELY", directional=0.1),
        executable_price=1.107,
        now=now + timedelta(minutes=1),
        bar_time=now,
    )
    assert recovered.state is SetupState.ENTER_NOW
    assert recovered.tracked
    assert "entry timing is executable again" in recovered.reason


def test_active_pullback_enters_on_first_closed_bar_that_clears_native_gate(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    lifecycle = book(tmp_path / "setups.json")
    waiting = lifecycle.observe(
        idea(),
        timing(EntryTimingDecision.WAIT_RETEST, "PULLBACK_STILL_ACTIVE", directional=-0.8),
        executable_price=1.10,
        now=now,
        bar_time=now,
    )
    assert waiting.state is SetupState.WAIT_RESUMPTION

    recovered = lifecycle.observe(
        idea(),
        timing(EntryTimingDecision.ENTER_NOW, "TIMELY", directional=0.01),
        executable_price=1.101,
        now=now + timedelta(minutes=5),
        bar_time=now + timedelta(minutes=5),
    )
    assert recovered.state is SetupState.ENTER_NOW


def test_waiting_setup_survives_restart_and_expires(tmp_path: Path) -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    store = tmp_path / "setups.json"
    first = book(store)
    first.observe(
        idea(),
        timing(EntryTimingDecision.WAIT_RETEST, "DIRECTIONAL_MOVE_OVEREXTENDED"),
        executable_price=1.11,
        now=now,
        bar_time=now,
    )

    restarted = book(store)
    expired = restarted.observe(
        idea(),
        timing(EntryTimingDecision.ENTER_NOW, "TIMELY", directional=0.1),
        executable_price=1.10,
        now=now + timedelta(minutes=31),
        bar_time=now + timedelta(minutes=30),
    )
    assert expired.state is SetupState.EXPIRED
