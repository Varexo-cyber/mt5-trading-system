"""The calendar the news filter reads, with fallback and a disk cache.

Fail-closed is the whole contract: if no provider answers and the cache is too
old, this raises, and the news filter turns that into "no new trades". No data
is never treated as no news. Getting this backwards once, during an FOMC
release, costs more than every skipped setup the filter will ever cause.

The cache is not a third source of truth. It is the last good fetch, and it
expires — `max_age_minutes` from config. Past that it is refused, because a
calendar from four hours ago cannot tell you about a release that was added or
rescheduled since.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from core.clock import Clock
from filters.calendar.events import EconomicEvent, deduplicate
from filters.calendar.providers import (
    CalendarProvider,
    CalendarUnavailableError,
    FileCalendarProvider,
    default_window,
)
from infra.logging import get_logger

log = get_logger(__name__)


class CalendarService:
    """Fetches from providers in order, caches the result, expires the cache."""

    def __init__(
        self,
        providers: list[CalendarProvider],
        clock: Clock,
        *,
        cache_path: Path | str,
        refresh_interval_minutes: int = 30,
        max_age_minutes: int = 180,
        require_multiple_providers: bool = True,
    ) -> None:
        if require_multiple_providers and len(providers) < 2:
            raise ValueError(
                "configure at least two calendar providers; with one source, that "
                "source going down stops all trading"
            )
        self.providers = providers
        self.clock = clock
        self.cache_path = Path(cache_path)
        self.refresh_interval = timedelta(minutes=refresh_interval_minutes)
        self.max_age = timedelta(minutes=max_age_minutes)

        self._events: list[EconomicEvent] = []
        self._fetched_at: datetime | None = None
        self._source: str = ""

    # -- state -------------------------------------------------------------

    @property
    def last_fetch(self) -> datetime | None:
        return self._fetched_at

    @property
    def source(self) -> str:
        """Which provider (or the cache) supplied the events currently held."""
        return self._source

    @property
    def age(self) -> timedelta | None:
        if self._fetched_at is None:
            return None
        return self.clock.now() - self._fetched_at

    def is_usable(self) -> bool:
        age = self.age
        return age is not None and age <= self.max_age

    # -- fetching ----------------------------------------------------------

    def events(self, *, force_refresh: bool = False) -> list[EconomicEvent]:
        """Current calendar. Raises `CalendarUnavailableError` if there is none.

        The raise is the point. Callers must not paper over it.
        """
        if force_refresh or self._needs_refresh():
            self._refresh()

        if not self.is_usable():
            age = self.age
            raise CalendarUnavailableError(
                "no usable economic calendar: "
                + (
                    "never fetched successfully"
                    if age is None
                    else f"last good fetch was {age} ago, budget {self.max_age}"
                )
            )
        return self._events

    def _needs_refresh(self) -> bool:
        if self._fetched_at is None:
            return True
        return self.clock.now() - self._fetched_at >= self.refresh_interval

    def _refresh(self) -> None:
        """Try each provider in order; the first success wins and is cached.

        Providers are tried in sequence rather than merged, because a merge
        would make every fetch as slow as the slowest source and as fragile as
        the flakiest. Merging happens only when a fetch is partial — see
        `fetch_all` used by the verification script.
        """
        now = self.clock.now()
        start, end = default_window(now)
        failures: list[str] = []

        for provider in self.providers:
            try:
                events = deduplicate(provider.fetch(start, end))
            except CalendarUnavailableError as exc:
                failures.append(f"{provider.name}: {exc}")
                log.warning(
                    "calendar provider failed",
                    extra={
                        "event": "calendar_provider_failed",
                        "provider": provider.name,
                        "reason": str(exc),
                    },
                )
                continue

            self._events = events
            self._fetched_at = now
            self._source = provider.name
            self._write_cache(events)
            log.info(
                "calendar refreshed",
                extra={
                    "event": "calendar_refreshed",
                    "provider": provider.name,
                    "events": len(events),
                    "high_impact": sum(1 for e in events if e.impact >= 3),
                    "window_end": end.isoformat(),
                },
            )
            return

        self._load_cache(failures)

    def _write_cache(self, events: list[EconomicEvent]) -> None:
        try:
            FileCalendarProvider.write(self.cache_path, events)
        except OSError as exc:  # pragma: no cover - disk failure
            log.warning(
                "could not write calendar cache",
                extra={"event": "calendar_cache_write_failed", "reason": str(exc)},
            )

    def _load_cache(self, failures: list[str]) -> None:
        """Fall back to the last good fetch, if it is still young enough.

        Deliberately does not refresh `_fetched_at`: the cache's age is the age
        of the data in it, not of the moment we re-read the file. Otherwise a
        stale calendar would look fresh forever simply by being loaded again.
        """
        if not self.cache_path.exists():
            log.error(
                "no calendar and no cache",
                extra={"event": "calendar_unavailable", "failures": failures},
            )
            return

        cache_written = datetime.fromtimestamp(
            self.cache_path.stat().st_mtime, tz=self.clock.now().tzinfo
        )
        age = self.clock.now() - cache_written
        if age > self.max_age:
            log.error(
                "calendar cache is too old to use",
                extra={
                    "event": "calendar_cache_expired",
                    "age_minutes": round(age.total_seconds() / 60, 1),
                    "max_age_minutes": round(self.max_age.total_seconds() / 60, 1),
                    "failures": failures,
                },
            )
            return

        now = self.clock.now()
        start, end = default_window(now)
        try:
            self._events = deduplicate(FileCalendarProvider(self.cache_path).fetch(start, end))
        except CalendarUnavailableError:  # pragma: no cover - corrupt cache
            log.exception("calendar cache is unreadable", extra={"event": "calendar_cache_corrupt"})
            return

        self._fetched_at = cache_written
        self._source = f"cache({self.cache_path.name})"
        log.warning(
            "serving calendar from cache; every live provider failed",
            extra={
                "event": "calendar_from_cache",
                "age_minutes": round(age.total_seconds() / 60, 1),
                "events": len(self._events),
                "failures": failures,
            },
        )

    # -- diagnostics -------------------------------------------------------

    def fetch_all(self, start: datetime, end: datetime) -> dict[str, object]:
        """Query every provider and report each one's result separately.

        Used by `scripts/verify_calendar.py`. Comparing sources side by side is
        the only way to notice that one of them quietly stopped publishing
        high-impact flags.
        """
        report: dict[str, object] = {}
        for provider in self.providers:
            try:
                events = provider.fetch(start, end)
            except CalendarUnavailableError as exc:
                report[provider.name] = {"ok": False, "error": str(exc)}
            else:
                report[provider.name] = {
                    "ok": True,
                    "events": len(events),
                    "high_impact": sum(1 for e in events if e.impact >= 3),
                    "sample": [e.describe() for e in events[:5]],
                }
        return report
