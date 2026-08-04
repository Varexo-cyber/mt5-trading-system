"""The loop that keeps the guard running between full cycles.

`run_forever` used to spend the gap between cycles asleep. That gap is not idle
time — it is the time open money spends unwatched, and on a slow cycle it is
most of a minute. These tests pin the two properties that make filling it safe:
the guard never opens anything and never calls the adviser, and a failure in it
cannot take the service down.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest

from analysis.position_health import PositionHealth
from execution.manager import ManagementEvent
from runner.service import JarvisRunner

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


def open_position(ticket: int = 555) -> SimpleNamespace:
    """The three fields the guard path reads off a position."""
    return SimpleNamespace(ticket=ticket, symbol="EURUSD", direction=SimpleNamespace(name="LONG"))


@dataclass
class BrokerSpy:
    positions_list: list[object] = field(default_factory=list)
    connects: int = 0
    reads: int = 0
    raises: Exception | None = None

    def ensure_connected(self) -> None:
        self.connects += 1
        if self.raises is not None:
            raise self.raises

    def positions(self, magic: int | None = None):  # type: ignore[no-untyped-def]
        del magic
        self.reads += 1
        return list(self.positions_list)


@dataclass
class ManagerSpy:
    calls: int = 0
    events: list[ManagementEvent] = field(default_factory=list)

    def manage(self, positions, now, patience):  # type: ignore[no-untyped-def]
        del positions, now, patience
        self.calls += 1
        return list(self.events)


def runner(
    *,
    guard_interval: float = 0.01,
    loop_interval: float = 1.0,
    positions: list[object] | None = None,
    engaged: bool = False,
    broker: BrokerSpy | None = None,
    tmp_path: Path | None = None,
) -> tuple[JarvisRunner, BrokerSpy, ManagerSpy, list[list[ManagementEvent]]]:
    """A runner with only the pieces the guard path touches.

    Built by hand rather than through `JarvisRunner(...)`, which would need a
    terminal, a catalogue and an API key to construct. What is under test here
    is the loop, and the loop reaches exactly the attributes set below.
    """
    spy_broker = broker if broker is not None else BrokerSpy(positions_list=positions or [])
    manager = ManagerSpy()
    recorded: list[list[ManagementEvent]] = []

    instance = JarvisRunner.__new__(JarvisRunner)
    instance.health_file = (tmp_path or Path(mkdtemp())) / "position_health.json"  # type: ignore[assignment]
    instance.broker = spy_broker  # type: ignore[assignment]
    instance.manager = manager  # type: ignore[assignment]
    manager.last_health = {}  # type: ignore[attr-defined]
    instance.clock = SimpleNamespace(now=lambda: NOW)  # type: ignore[assignment]
    instance.posture = SimpleNamespace(patience_multiplier=1.0)  # type: ignore[assignment]
    instance.kill_switch = SimpleNamespace(is_engaged=lambda: engaged)  # type: ignore[assignment]
    instance.settings = SimpleNamespace(  # type: ignore[assignment]
        system=SimpleNamespace(
            magic_number=777,
            guard_interval_seconds=guard_interval,
            loop_interval_seconds=loop_interval,
        )
    )
    instance._record_management = recorded.append  # type: ignore[assignment]
    return instance, spy_broker, manager, recorded


# ------------------------------------------------------------------- tick ---


def test_a_tick_reconnects_before_reading() -> None:
    """The same dropped-IPC failure that crashed the deck would otherwise stop
    the guard the first time the terminal blinked."""
    jarvis, broker, _, _ = runner(positions=[open_position()])
    jarvis.guard_tick()
    assert broker.connects == 1


def test_no_positions_means_no_management_call() -> None:
    """With nothing open there is nothing to manage, and a pass a second that
    still queried prices would be pure waste."""
    jarvis, _, manager, _ = runner(positions=[])
    assert jarvis.guard_tick() == []
    assert manager.calls == 0


def test_events_are_recorded() -> None:
    jarvis, _, manager, recorded = runner(positions=[open_position()])
    manager.events = [ManagementEvent(1, "BREAK_EVEN", "protected")]
    events = jarvis.guard_tick()
    assert [event.action for event in events] == ["BREAK_EVEN"]
    assert recorded == [[ManagementEvent(1, "BREAK_EVEN", "protected")]]


def test_a_failing_tick_does_not_take_the_service_down() -> None:
    """This runs between cycles on a best-effort basis. A momentary IPC hiccup
    must not end a session that is holding real positions — the next full cycle
    redoes all of this with its own error handling."""
    broker = BrokerSpy(raises=RuntimeError("[-10004] No IPC connection"))
    jarvis, _, manager, _ = runner(broker=broker)
    assert jarvis.guard_tick() == []
    assert manager.calls == 0


# ------------------------------------------------------------------- loop ---


def test_the_gap_between_cycles_is_spent_watching() -> None:
    jarvis, _, manager, _ = runner(guard_interval=0.01, positions=[open_position()])
    jarvis._guard_until(time.monotonic() + 0.09)
    assert manager.calls >= 4, f"only {manager.calls} passes in 90ms at a 10ms interval"


def test_it_returns_at_the_deadline() -> None:
    """Overrunning would delay the scan cycle indefinitely."""
    jarvis, _, _, _ = runner(guard_interval=0.01, positions=[open_position()])
    started = time.monotonic()
    jarvis._guard_until(started + 0.05)
    assert time.monotonic() - started < 0.5


def test_a_deadline_already_past_returns_at_once() -> None:
    """A cycle slower than its own interval must not then be charged a full
    guard interval of sleep on top."""
    jarvis, _, manager, _ = runner(guard_interval=0.01, positions=[open_position()])
    started = time.monotonic()
    jarvis._guard_until(started - 5.0)
    assert time.monotonic() - started < 0.05
    assert manager.calls == 0


def test_zero_disables_the_guard_but_still_waits() -> None:
    """Switching the guard off must not turn the loop into a busy spin."""
    jarvis, _, manager, _ = runner(guard_interval=0.0, positions=[open_position()])
    started = time.monotonic()
    jarvis._guard_until(started + 0.05)
    elapsed = time.monotonic() - started
    assert manager.calls == 0
    assert elapsed == pytest.approx(0.05, abs=0.05)


def test_the_kill_switch_ends_the_wait_immediately() -> None:
    """STOP means flatten now. The flattening lives in the full cycle, so the
    guard's job is to stop holding it up."""
    jarvis, _, manager, _ = runner(guard_interval=0.01, engaged=True, positions=[open_position()])
    started = time.monotonic()
    jarvis._guard_until(started + 10.0)
    assert time.monotonic() - started < 1.0
    assert manager.calls == 0, "no management pass after STOP is seen"


def test_the_guard_never_sleeps_past_the_deadline() -> None:
    """With an interval longer than the remaining gap, sleeping a full interval
    would push the next cycle out by nearly that much."""
    jarvis, _, _, _ = runner(guard_interval=5.0, positions=[open_position()])
    started = time.monotonic()
    jarvis._guard_until(started + 0.05)
    assert time.monotonic() - started < 1.0


# --------------------------------------------------------------- published ---


def test_the_read_is_published_for_the_deck(tmp_path: Path) -> None:
    """The manager holds the reading in memory and the dashboard is a different
    process. Without the file, "what does the system think of my open trade
    right now" is answered by a fifteen-minute-old supervisor entry."""
    jarvis, _, manager, _ = runner(positions=[open_position(4242)], tmp_path=tmp_path)
    manager.last_health = {  # type: ignore[attr-defined]
        4242: PositionHealth("deteriorating", 0.6, "tighten", (), "structure gone")
    }

    jarvis.guard_tick()

    published = json.loads(jarvis.health_file.read_text())
    (entry,) = published["positions"]
    assert entry["ticket"] == 4242
    assert entry["verdict"] == "deteriorating"
    assert entry["action"] == "tighten"
    assert published["recorded_at"] == NOW.isoformat()


def test_an_unread_position_is_published_as_unknown(tmp_path: Path) -> None:
    """ "We looked and it is fine" and "we have not looked" are different claims,
    and reporting the second as `healthy` would be the wrong one."""
    jarvis, _, _, _ = runner(positions=[open_position(1)], tmp_path=tmp_path)
    jarvis.guard_tick()
    (entry,) = json.loads(jarvis.health_file.read_text())["positions"]
    assert entry["verdict"] == "unknown"
