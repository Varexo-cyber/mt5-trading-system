"""Time abstraction.

Nothing in this system calls `datetime.now()` directly. Everything asks a
`Clock`. That is what lets the backtester replay a news filter, a session
filter or a time-based exit with the exact same code path as live trading —
and what makes "the filter accidentally used tomorrow's data" a testable bug
rather than a mystery.

All times are tz-aware UTC. Broker server time is treated as a separate,
explicitly converted concept (`server_offset`), because brokers run on
GMT+2/+3 with their own DST rules and confusing the two shifts every session
boundary by hours.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta


class Clock(ABC):
    """Source of 'now' for the whole system."""

    @abstractmethod
    def now(self) -> datetime:
        """Current time, tz-aware UTC."""

    def today(self) -> datetime:
        """Midnight UTC of the current day."""
        return self.now().replace(hour=0, minute=0, second=0, microsecond=0)


class LiveClock(Clock):
    """Wall-clock time, with a measured offset to broker server time.

    `server_offset` is the difference (server - UTC), measured once at connect
    by comparing the terminal's last tick time to local UTC. It is used only to
    interpret broker-supplied timestamps, never to shift our own decisions.
    """

    def __init__(self, server_offset: timedelta = timedelta(0)) -> None:
        self.server_offset = server_offset

    def now(self) -> datetime:
        return datetime.now(UTC)

    def server_now(self) -> datetime:
        return self.now() + self.server_offset

    def to_utc(self, server_time: datetime) -> datetime:
        """Convert a naive broker timestamp to tz-aware UTC."""
        if server_time.tzinfo is None:
            server_time = server_time.replace(tzinfo=UTC)
        return server_time - self.server_offset


class SimulatedClock(Clock):
    """Deterministic clock for backtests and unit tests."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("SimulatedClock requires a tz-aware start time")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def set(self, moment: datetime) -> None:
        """Jump to `moment`. Refuses to move backwards.

        Backwards time travel in a backtest means the driver has a bug, and the
        symptom would otherwise be look-ahead bias that looks like alpha.
        """
        if moment.tzinfo is None:
            raise ValueError("SimulatedClock.set requires a tz-aware time")
        moment = moment.astimezone(UTC)
        if moment < self._now:
            raise ValueError(f"clock cannot move backwards: {self._now} -> {moment}")
        self._now = moment

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ValueError("clock cannot advance by a negative delta")
        self._now += delta
