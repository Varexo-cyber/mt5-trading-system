"""Calendar sources.

Two independent remote feeds plus a local file source, so one host going down
does not stop all trading — which is what would otherwise happen, because the
news filter fails closed by design.

**Verification status.** The parsers below are written against each feed's
documented/observed JSON shape, but the build environment this code was written
in blocks outbound HTTPS to third-party hosts, so they have **not** been run
against live responses. Run `scripts/verify_calendar.py` on a machine with
open internet before trusting them; it dumps the raw payload next to the
parsed events so any mismatch is obvious. Until that has been done once, treat
the remote providers as unverified.

A parser that silently returns 3 of 40 events is more dangerous than one that
raises: the missing 37 are invisible, and the filter reports "all clear" for a
window it should have blocked. So each parser tracks how many records it could
not read and fails the whole fetch past a threshold.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from core.errors import TradingSystemError
from filters.calendar.events import EconomicEvent, Impact
from infra.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "mt5-trading-system/0.1 (+calendar filter)"

#: Fail the fetch if more than this fraction of records will not parse.
MAX_UNPARSEABLE_FRACTION = 0.10


class CalendarUnavailableError(TradingSystemError):
    """No provider could supply a usable calendar. Means: do not trade."""


class CalendarProvider(ABC):
    """One source of economic events."""

    name: str = "provider"

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        """Events in [start, end]. Raises `CalendarUnavailableError` on failure.

        Returning an empty list means "this source is up and there is genuinely
        nothing scheduled". Never return empty to signal a failure — the caller
        cannot tell the difference, and one of the two readings lets a trade
        through during an FOMC release.
        """

    # -- shared plumbing ---------------------------------------------------

    def _get_json(self, url: str, timeout: float) -> Any:
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise CalendarUnavailableError(
                        f"{self.name}: HTTP {response.status} from {url}"
                    )
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CalendarUnavailableError(f"{self.name}: {url} unreachable — {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CalendarUnavailableError(f"{self.name}: {url} returned invalid JSON") from exc

    def _finish(self, events: list[EconomicEvent], failed: int, total: int) -> list[EconomicEvent]:
        """Reject a fetch that lost too many records to parse errors."""
        if total and failed / total > MAX_UNPARSEABLE_FRACTION:
            raise CalendarUnavailableError(
                f"{self.name}: could not parse {failed}/{total} records. The feed's "
                f"format has probably changed; refusing to return a partial calendar."
            )
        if failed:
            log.warning(
                "calendar records skipped",
                extra={"event": "calendar_parse_skip", "provider": self.name, "skipped": failed},
            )
        return events


class FairEconomyProvider(CalendarProvider):
    """ForexFactory's weekly JSON, mirrored by FairEconomy.

    Expected record shape::

        {"title": "Non-Farm Employment Change", "country": "USD",
         "date": "2026-03-06T13:30:00-05:00", "impact": "High",
         "forecast": "170K", "previous": "143K"}

    `country` is already a currency code, which is why no country-to-currency
    table is needed here.

    Only "this week" and "next week" are published, so this provider cannot
    answer questions about last month. That is fine for live trading and
    useless for backtesting — the backtest uses `FileCalendarProvider` over an
    archive built by running `scripts/verify_calendar.py --archive` weekly.
    """

    name = "faireconomy"

    THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    NEXT_WEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

    def __init__(self, timeout: float = 15.0, *, include_next_week: bool = True) -> None:
        self.timeout = timeout
        self.include_next_week = include_next_week

    def fetch(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        urls = [self.THIS_WEEK]
        if self.include_next_week:
            urls.append(self.NEXT_WEEK)

        events: list[EconomicEvent] = []
        failed = total = 0
        reachable = False

        for url in urls:
            try:
                payload = self._get_json(url, self.timeout)
            except CalendarUnavailableError:
                # Next week's file is frequently absent early in the week; only
                # the current week is load-bearing.
                if url is self.THIS_WEEK:
                    raise
                log.debug("next-week calendar unavailable", extra={"provider": self.name})
                continue
            reachable = True

            if not isinstance(payload, list):
                raise CalendarUnavailableError(
                    f"{self.name}: expected a JSON array, got {type(payload).__name__}"
                )
            for record in payload:
                total += 1
                try:
                    event = self._parse(record)
                except (KeyError, TypeError, ValueError):
                    failed += 1
                    continue
                if start <= event.when <= end:
                    events.append(event)

        if not reachable:
            raise CalendarUnavailableError(f"{self.name}: no calendar file could be fetched")
        return self._finish(events, failed, total)

    def _parse(self, record: dict[str, Any]) -> EconomicEvent:
        when = datetime.fromisoformat(str(record["date"])).astimezone(UTC)
        return EconomicEvent(
            when=when,
            currency=str(record["country"]).upper(),
            title=str(record["title"]),
            impact=Impact.parse(str(record.get("impact", ""))),
            source=self.name,
            forecast=str(record.get("forecast") or ""),
            previous=str(record.get("previous") or ""),
        )


class TradingViewProvider(CalendarProvider):
    """TradingView's public economic-calendar endpoint.

    Expected response shape::

        {"status": "ok", "result": [
            {"title": "Nonfarm Payrolls", "country": "US", "importance": 1,
             "date": "2026-03-06T13:30:00.000Z", "currency": "USD", ...}]}

    `importance` is -1 (low), 0 (medium), 1 (high). Some records carry
    `currency`, others only a two-letter `country`, so both are handled.
    """

    name = "tradingview"

    URL = "https://economic-calendar.tradingview.com/events"

    #: Only what we actually trade. Keeping the request narrow keeps the
    #: response small and the parse surface honest.
    COUNTRY_TO_CURRENCY: ClassVar[dict[str, str]] = {
        "US": "USD",
        "EU": "EUR",
        "DE": "EUR",
        "FR": "EUR",
        "IT": "EUR",
        "ES": "EUR",
        "GB": "GBP",
        "JP": "JPY",
        "AU": "AUD",
        "NZ": "NZD",
        "CA": "CAD",
        "CH": "CHF",
        "CN": "CNY",
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def fetch(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        countries = ",".join(sorted(self.COUNTRY_TO_CURRENCY))
        url = f"{self.URL}?from={_z(start)}&to={_z(end)}&countries={countries}"
        payload = self._get_json(url, self.timeout)

        if not isinstance(payload, dict) or "result" not in payload:
            raise CalendarUnavailableError(f"{self.name}: unexpected response envelope")
        records = payload["result"]
        if not isinstance(records, list):
            raise CalendarUnavailableError(f"{self.name}: `result` is not an array")

        events: list[EconomicEvent] = []
        failed = 0
        for record in records:
            try:
                event = self._parse(record)
            except (KeyError, TypeError, ValueError):
                failed += 1
                continue
            if event is not None and start <= event.when <= end:
                events.append(event)
        return self._finish(events, failed, len(records))

    def _parse(self, record: dict[str, Any]) -> EconomicEvent | None:
        currency = str(record.get("currency") or "").upper()
        if not currency:
            currency = self.COUNTRY_TO_CURRENCY.get(str(record.get("country", "")).upper(), "")
        if not currency:
            return None  # a market we do not trade; not a parse failure

        when = datetime.fromisoformat(str(record["date"]).replace("Z", "+00:00")).astimezone(UTC)
        importance = int(record.get("importance", 1))
        impact = {1: Impact.HIGH, 0: Impact.MEDIUM, -1: Impact.LOW}.get(importance, Impact.HIGH)

        return EconomicEvent(
            when=when,
            currency=currency,
            title=str(record.get("title") or record.get("indicator") or "unnamed event"),
            impact=impact,
            source=self.name,
            forecast=str(record.get("forecast") or ""),
            previous=str(record.get("previous") or ""),
            actual=str(record.get("actual") or ""),
        )


class FileCalendarProvider(CalendarProvider):
    """Events from a local JSON file.

    Three jobs: the backtester's calendar, the archive built from live fetches,
    and the manual escape hatch when both feeds are down and you would rather
    hand-enter next week's FOMC than not trade at all.

    File format is our own, not a provider's::

        [{"when": "2026-03-06T13:30:00+00:00", "currency": "USD",
          "title": "Non-Farm Payrolls", "impact": "High"}]
    """

    name = "file"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def fetch(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        if not self.path.exists():
            raise CalendarUnavailableError(f"{self.name}: {self.path} does not exist")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalendarUnavailableError(f"{self.name}: cannot read {self.path} — {exc}") from exc
        if not isinstance(payload, list):
            raise CalendarUnavailableError(f"{self.name}: {self.path} must contain a JSON array")

        events: list[EconomicEvent] = []
        failed = 0
        for record in payload:
            try:
                event = EconomicEvent(
                    when=datetime.fromisoformat(str(record["when"])).astimezone(UTC),
                    currency=str(record["currency"]).upper(),
                    title=str(record["title"]),
                    impact=Impact.parse(str(record.get("impact", "High"))),
                    source=f"{self.name}:{self.path.name}",
                    forecast=str(record.get("forecast") or ""),
                    previous=str(record.get("previous") or ""),
                )
            except (KeyError, TypeError, ValueError):
                failed += 1
                continue
            if start <= event.when <= end:
                events.append(event)
        return self._finish(events, failed, len(payload))

    @staticmethod
    def write(path: Path | str, events: list[EconomicEvent]) -> None:
        """Persist events in the file format above (used by the disk cache)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                [
                    {
                        "when": event.when.isoformat(),
                        "currency": event.currency,
                        "title": event.title,
                        "impact": event.impact.name.title(),
                        "forecast": event.forecast,
                        "previous": event.previous,
                        "source": event.source,
                    }
                    for event in events
                ],
                indent=1,
            ),
            encoding="utf-8",
        )


def _z(moment: datetime) -> str:
    """RFC-3339 with a literal Z, which is what the endpoint expects."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def build_providers(
    names: tuple[str, ...], *, calendar_dir: Path, timeout: float = 15.0
) -> list[CalendarProvider]:
    """Instantiate providers by config name, in the order given."""
    registry: dict[str, Any] = {
        "faireconomy": lambda: FairEconomyProvider(timeout),
        "tradingview": lambda: TradingViewProvider(timeout),
        "file": lambda: FileCalendarProvider(calendar_dir / "manual.json"),
    }
    providers: list[CalendarProvider] = []
    for name in names:
        factory = registry.get(name)
        if factory is None:
            raise ValueError(f"unknown calendar provider {name!r}; known: {sorted(registry)}")
        providers.append(factory())
    return providers


def default_window(now: datetime) -> tuple[datetime, datetime]:
    """The span the filter needs: a little behind, a few days ahead.

    Behind, because the post-event blackout extends after a release; ahead,
    because the pre-event window has to be visible before we reach it.
    """
    return now - timedelta(days=1), now + timedelta(days=7)
