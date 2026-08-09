"""Event-driven AI management reacts to evidence, not polling noise."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from analysis.position_health import PositionHealth
from core.types import Direction, Position, Tick
from runner.service import JarvisRunner, _SupervisionSnapshot

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


class _Journal:
    def open_trade_by_ticket(self, _ticket: int) -> dict[str, float]:
        return {"sl": 90.0, "mfe_r": 0.2}


class _Broker:
    bid = 101.0

    def tick(self, _symbol: str) -> Tick:
        return Tick("TEST", NOW, self.bid, self.bid + 0.1)


def _runner() -> JarvisRunner:
    runner = object.__new__(JarvisRunner)
    runner.journal = _Journal()  # type: ignore[assignment]
    runner.broker = _Broker()  # type: ignore[assignment]
    runner.manager = SimpleNamespace(last_health={})  # type: ignore[assignment]
    runner.settings = SimpleNamespace(  # type: ignore[assignment]
        trade_management=SimpleNamespace(
            supervision_interval_minutes=15.0,
            supervision_event_driven=True,
            supervision_min_interval_minutes=2.0,
            supervision_profit_step_r=0.25,
            supervision_giveback_trigger_fraction=0.25,
            giveback_arm_r=0.5,
        )
    )
    runner._supervised_at = {}
    runner._supervision_due_at = {}
    runner._supervision_snapshots = {}
    return runner


def _position() -> Position:
    return Position(
        ticket=1,
        symbol="TEST",
        direction=Direction.LONG,
        volume=0.01,
        price_open=100.0,
        sl=95.0,
        tp=120.0,
        profit=1.0,
        swap=0.0,
        opened_at=NOW - timedelta(hours=1),
    )


def test_a_new_position_is_reviewed_immediately() -> None:
    triggered = _runner()._supervision_trigger(_position(), NOW)

    assert triggered is not None
    assert triggered[0] == "position_opened"


def test_worsening_health_brings_the_review_forward() -> None:
    runner = _runner()
    runner._supervised_at[1] = NOW - timedelta(minutes=3)
    runner._supervision_due_at[1] = NOW + timedelta(minutes=12)
    runner._supervision_snapshots[1] = _SupervisionSnapshot(0.1, 0.2, 0.5, "healthy", 0.0)
    runner.manager.last_health[1] = PositionHealth(
        "deteriorating", 0.6, "tighten", (), "structure weakened"
    )

    triggered = runner._supervision_trigger(_position(), NOW)

    assert triggered is not None
    assert triggered[0] == "health_worsened:healthy->deteriorating"


def test_the_cost_cooldown_blocks_repeated_reconsideration() -> None:
    runner = _runner()
    runner._supervised_at[1] = NOW - timedelta(seconds=30)
    runner._supervision_due_at[1] = NOW + timedelta(minutes=14)
    runner._supervision_snapshots[1] = _SupervisionSnapshot(0.1, 0.2, 0.5, "healthy", 0.0)
    runner.manager.last_health[1] = PositionHealth(
        "broken", 1.0, "exit", (), "structure invalidated"
    )

    assert runner._supervision_trigger(_position(), NOW) is None
