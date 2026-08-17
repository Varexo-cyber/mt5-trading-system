"""News filter: blackout windows, currency matching, and the fail-closed path.

The fail-closed tests are the important ones. A filter that blocks correctly
when the calendar is present but quietly passes when it is absent provides no
protection at all on the one day it matters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.schema import NewsFilterConfig, NewsWindow
from core.clock import SimulatedClock
from core.instrument import InstrumentSpec
from core.types import Direction, Position
from filters.base import FilterContext
from filters.calendar.events import EconomicEvent, Impact, deduplicate, symbol_currencies
from filters.calendar.providers import (
    TRADINGVIEW_HEADERS,
    CalendarProvider,
    CalendarUnavailableError,
    FairEconomyProvider,
    FileCalendarProvider,
    TradingViewProvider,
)
from filters.calendar.service import CalendarService
from filters.news_filter import NewsFilter
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec, usdjpy_spec, xauusd_spec

NOW = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)


def event(
    minutes_from_now: float,
    currency: str = "USD",
    title: str = "Retail Sales",
    impact: Impact = Impact.HIGH,
) -> EconomicEvent:
    return EconomicEvent(
        when=NOW + timedelta(minutes=minutes_from_now),
        currency=currency,
        title=title,
        impact=impact,
        source="test",
    )


class StubProvider(CalendarProvider):
    """A provider that serves a fixed list, or fails on command."""

    def __init__(self, events: list[EconomicEvent], name: str = "stub", fail: bool = False):
        self.name = name
        self._events = events
        self.fail = fail
        self.calls = 0

    def fetch(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        self.calls += 1
        if self.fail:
            raise CalendarUnavailableError(f"{self.name} is down")
        return [e for e in self._events if start <= e.when <= end]


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def config() -> NewsFilterConfig:
    return NewsFilterConfig()


def make_filter(
    events: list[EconomicEvent],
    clock: SimulatedClock,
    tmp_path: Path,
    config: NewsFilterConfig | None = None,
    fail: bool = False,
) -> NewsFilter:
    service = CalendarService(
        [StubProvider(events, "primary", fail), StubProvider(events, "backup", fail)],
        clock,
        cache_path=tmp_path / "cache.json",
    )
    return NewsFilter(config or NewsFilterConfig(), service, clock)


def context(spec_factory=eurusd_spec, now: datetime = NOW) -> FilterContext:  # type: ignore[no-untyped-def]
    spec = InstrumentSpec.from_mt5(spec_factory())
    return FilterContext(symbol=spec.symbol, spec=spec, now=now, direction=Direction.LONG)


class TestBlackoutWindows:
    def test_clear_well_before_an_event(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(180)], clock, tmp_path)
        verdict = f.check(context())
        assert verdict.passed
        assert verdict.data["minutes_to_news"] == pytest.approx(120.0)
        assert verdict.data["next_news_event"] == "Retail Sales"
        assert verdict.data["next_news_currency"] == "USD"
        assert verdict.data["next_news_at"] == event(180).when.isoformat()
        assert verdict.data["next_news_blackout_start"] == event(120).when.isoformat()

    def test_blocked_sixty_minutes_before(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(45)], clock, tmp_path)
        verdict = f.check(context())
        assert not verdict.passed
        assert verdict.reason is Reason.NEWS_BLACKOUT

    def test_exactly_at_the_window_edge_is_blocked(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        """60 minutes before means 60 counts, not 59."""
        f = make_filter([event(60)], clock, tmp_path)
        assert not f.check(context()).passed

    def test_one_minute_outside_the_window_is_clear(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        f = make_filter([event(61)], clock, tmp_path)
        assert f.check(context()).passed

    def test_blocked_thirty_minutes_after(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(-20)], clock, tmp_path)
        verdict = f.check(context())
        assert not verdict.passed
        assert verdict.data["minutes_to_news_clear"] == pytest.approx(10.0)

    def test_clear_after_the_window_closes(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(-31)], clock, tmp_path)
        assert f.check(context()).passed

    def test_medium_impact_does_not_block(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(10, impact=Impact.MEDIUM)], clock, tmp_path)
        assert f.check(context()).passed

    def test_low_impact_does_not_block(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(10, impact=Impact.LOW)], clock, tmp_path)
        assert f.check(context()).passed


class TestExtremeEvents:
    def test_nfp_gets_the_wide_window(self, clock: SimulatedClock, tmp_path: Path) -> None:
        """110 minutes out: inside the 120-minute extreme window, not the 60."""
        f = make_filter([event(110, title="Non-Farm Employment Change")], clock, tmp_path)
        verdict = f.check(context())
        assert not verdict.passed
        assert "extreme" in verdict.detail

    def test_an_ordinary_release_at_the_same_distance_is_clear(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        f = make_filter([event(110, title="Retail Sales")], clock, tmp_path)
        assert f.check(context()).passed

    def test_fomc_stays_blocked_an_hour_after(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(-50, title="FOMC Statement")], clock, tmp_path)
        assert not f.check(context()).passed

    @pytest.mark.parametrize(
        "title",
        [
            "FOMC Statement",
            "CPI m/m",
            "Non-Farm Employment Change",
            "ECB Press Conference",
            "Interest Rate Decision",
        ],
    )
    def test_every_configured_keyword_escalates(
        self, clock: SimulatedClock, tmp_path: Path, title: str
    ) -> None:
        f = make_filter([event(110, title=title)], clock, tmp_path)
        assert not f.check(context()).passed


class TestCurrencyMatching:
    def test_usd_event_blocks_eurusd(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(30, currency="USD")], clock, tmp_path)
        assert not f.check(context()).passed

    def test_eur_event_blocks_eurusd(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(30, currency="EUR")], clock, tmp_path)
        assert not f.check(context()).passed

    def test_unrelated_currency_does_not_block(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(30, currency="AUD")], clock, tmp_path)
        assert f.check(context()).passed

    def test_usd_event_blocks_gold(self, clock: SimulatedClock, tmp_path: Path) -> None:
        """XAUUSD is a USD instrument; its base currency simply never matches."""
        f = make_filter([event(30, currency="USD")], clock, tmp_path)
        assert not f.check(context(xauusd_spec)).passed

    def test_jpy_event_blocks_usdjpy(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(30, currency="JPY")], clock, tmp_path)
        assert not f.check(context(usdjpy_spec)).passed

    def test_currencies_come_from_the_spec_not_the_name(self) -> None:
        spec = InstrumentSpec.from_mt5(eurusd_spec(name="EURUSD.pro"))
        assert symbol_currencies(spec.currency_base, spec.currency_profit) == {"EUR", "USD"}


class TestFailClosed:
    """No calendar means no trade. This is the rule the filter exists for."""

    def test_every_provider_down_blocks(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([], clock, tmp_path, fail=True)
        verdict = f.check(context())
        assert not verdict.passed
        assert verdict.reason is Reason.NEWS_CALENDAR_UNAVAILABLE
        assert verdict.reason.is_halt

    def test_an_empty_calendar_is_not_a_failure(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        """A quiet week is a real answer, and different from no answer."""
        f = make_filter([], clock, tmp_path)
        verdict = f.check(context())
        assert verdict.passed
        assert verdict.data["minutes_to_news"] is None

    def test_fallback_to_the_second_provider(self, clock: SimulatedClock, tmp_path: Path) -> None:
        primary = StubProvider([], "primary", fail=True)
        backup = StubProvider([event(30)], "backup")
        service = CalendarService([primary, backup], clock, cache_path=tmp_path / "c.json")
        f = NewsFilter(NewsFilterConfig(), service, clock)

        assert not f.check(context()).passed  # backup supplied the blocking event
        assert primary.calls == 1
        assert backup.calls == 1
        assert service.source == "backup"

    def test_a_single_provider_is_refused_at_construction(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="at least two"):
            CalendarService([StubProvider([])], clock, cache_path=tmp_path / "c.json")


class TestCacheFallback:
    def test_cache_serves_when_providers_die(self, clock: SimulatedClock, tmp_path: Path) -> None:
        cache = tmp_path / "cache.json"
        live = StubProvider([event(30)], "live")
        service = CalendarService([live, StubProvider([], "other")], clock, cache_path=cache)
        service.events()  # warm the cache
        assert cache.exists()

        live.fail = True
        clock.advance(timedelta(minutes=45))
        events = CalendarService(
            [StubProvider([], "a", fail=True), StubProvider([], "b", fail=True)],
            clock,
            cache_path=cache,
        ).events()
        assert len(events) == 1

    def test_expired_cache_is_refused(self, clock: SimulatedClock, tmp_path: Path) -> None:
        """A four-hour-old calendar cannot know about a rescheduled release."""
        cache = tmp_path / "cache.json"
        FileCalendarProvider.write(cache, [event(600)])
        import os

        stale = (NOW - timedelta(hours=6)).timestamp()
        os.utime(cache, (stale, stale))

        service = CalendarService(
            [StubProvider([], "a", fail=True), StubProvider([], "b", fail=True)],
            clock,
            cache_path=cache,
            max_age_minutes=180,
        )
        with pytest.raises(CalendarUnavailableError, match=r"never fetched|last good fetch"):
            service.events()


class TestOpenPositions:
    def _position(self) -> Position:
        return Position(
            ticket=1,
            symbol="EURUSD",
            direction=Direction.LONG,
            volume=0.1,
            price_open=1.085,
            sl=1.083,
            tp=1.091,
            profit=0.0,
            swap=0.0,
            opened_at=NOW,
        )

    def test_no_action_when_news_is_far_away(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(300)], clock, tmp_path)
        assert f.position_action(self._position(), "EUR", "USD") == "none"

    def test_break_even_inside_the_window(self, clock: SimulatedClock, tmp_path: Path) -> None:
        f = make_filter([event(30)], clock, tmp_path)
        assert f.position_action(self._position(), "EUR", "USD") == "break_even"

    def test_configured_close_action_is_honoured(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        config = NewsFilterConfig(open_position_action="close")
        f = make_filter([event(30)], clock, tmp_path, config)
        assert f.position_action(self._position(), "EUR", "USD") == "close"

    def test_missing_calendar_de_risks_open_positions(
        self, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        """Unknown news is a reason to protect what is open, not to ignore it."""
        f = make_filter([], clock, tmp_path, fail=True)
        assert f.position_action(self._position(), "EUR", "USD") == "break_even"


class TestConfigWindows:
    def test_widened_windows_are_respected(self, clock: SimulatedClock, tmp_path: Path) -> None:
        config = NewsFilterConfig(high_impact=NewsWindow(minutes_before=120, minutes_after=90))
        f = make_filter([event(100)], clock, tmp_path, config)
        assert not f.check(context()).passed


class TestEventParsing:
    def test_unknown_impact_labels_become_high(self) -> None:
        """Cautious by default: an unrecognised label is not evidence of safety."""
        assert Impact.parse("catastrophic") is Impact.HIGH
        assert Impact.parse("") is Impact.HOLIDAY
        assert Impact.parse("High") is Impact.HIGH
        assert Impact.parse("  medium ") is Impact.MEDIUM

    def test_naive_event_times_are_refused(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            EconomicEvent(
                when=datetime(2026, 3, 11, 12, 0),
                currency="USD",
                title="x",
                impact=Impact.HIGH,
            )

    def test_duplicates_merge_keeping_the_higher_impact(self) -> None:
        low = EconomicEvent(when=NOW, currency="USD", title="CPI", impact=Impact.MEDIUM)
        high = EconomicEvent(when=NOW, currency="USD", title="CPI", impact=Impact.HIGH)
        merged = deduplicate([low, high])
        assert len(merged) == 1
        assert merged[0].impact is Impact.HIGH

    def test_seconds_level_disagreement_still_deduplicates(self) -> None:
        a = EconomicEvent(when=NOW, currency="USD", title="CPI", impact=Impact.HIGH)
        b = EconomicEvent(
            when=NOW + timedelta(seconds=20), currency="USD", title="CPI", impact=Impact.HIGH
        )
        assert len(deduplicate([a, b])) == 1


class TestFileProvider:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "cal.json"
        FileCalendarProvider.write(path, [event(60, title="CPI m/m")])
        loaded = FileCalendarProvider(path).fetch(NOW - timedelta(days=1), NOW + timedelta(days=1))
        assert len(loaded) == 1
        assert loaded[0].title == "CPI m/m"
        assert loaded[0].impact is Impact.HIGH

    def test_missing_file_raises_unavailable(self, tmp_path: Path) -> None:
        with pytest.raises(CalendarUnavailableError, match="does not exist"):
            FileCalendarProvider(tmp_path / "nope.json").fetch(NOW, NOW)

    def test_mostly_unparseable_file_is_refused(self, tmp_path: Path) -> None:
        """A partial calendar is more dangerous than an absent one."""
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                [{"nope": 1}] * 9
                + [{"when": NOW.isoformat(), "currency": "USD", "title": "ok", "impact": "High"}]
            ),
            encoding="utf-8",
        )
        with pytest.raises(CalendarUnavailableError, match="could not parse"):
            FileCalendarProvider(path).fetch(NOW - timedelta(days=1), NOW + timedelta(days=1))


class TestRemoteProviders:
    def test_faireconomy_refuses_a_missing_next_week(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = FairEconomyProvider(retries=0, spacing_seconds=0.0)
        current = [
            {
                "date": NOW.isoformat(),
                "country": "USD",
                "title": "CPI",
                "impact": "High",
            }
        ]

        def fake_get(url: str, timeout: float, headers=None):  # type: ignore[no-untyped-def]
            del timeout, headers
            if url == provider.THIS_WEEK:
                return current
            raise CalendarUnavailableError("HTTP Error 429: Too Many Requests")

        monkeypatch.setattr(provider, "_get_json", fake_get)

        with pytest.raises(CalendarUnavailableError, match="refusing partial data") as caught:
            provider.fetch(NOW - timedelta(days=1), NOW + timedelta(days=7))

        # The reason has to survive the wrapping. A 429 means back off, a 404
        # means the feed moved, a timeout means the network — different fixes
        # that were previously all reported with the same word, "failed".
        assert "429" in str(caught.value)

    def test_faireconomy_retries_before_declaring_the_feed_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rate-limited second file must not be promoted into "do not trade"."""
        provider = FairEconomyProvider(retries=2, spacing_seconds=0.0)
        attempts: list[str] = []
        record = {"date": NOW.isoformat(), "country": "USD", "title": "CPI", "impact": "High"}

        def fake_get(url: str, timeout: float, headers=None):  # type: ignore[no-untyped-def]
            del timeout, headers
            attempts.append(url)
            if url == provider.NEXT_WEEK and attempts.count(url) < 3:
                raise CalendarUnavailableError("HTTP Error 429: Too Many Requests")
            return [record]

        monkeypatch.setattr(provider, "_get_json", fake_get)

        events = provider.fetch(NOW - timedelta(days=1), NOW + timedelta(days=7))

        assert len(events) == 2
        assert attempts.count(provider.NEXT_WEEK) == 3

    def test_faireconomy_still_gives_up_when_every_attempt_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying must not soften fail-closed: a real outage still stops trading."""
        provider = FairEconomyProvider(retries=2, spacing_seconds=0.0)

        def fake_get(url: str, timeout: float, headers=None):  # type: ignore[no-untyped-def]
            del url, timeout, headers
            raise CalendarUnavailableError("connection refused")

        monkeypatch.setattr(provider, "_get_json", fake_get)

        with pytest.raises(CalendarUnavailableError):
            provider.fetch(NOW - timedelta(days=1), NOW + timedelta(days=7))

    def test_tradingview_sends_required_browser_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = TradingViewProvider()
        captured: dict[str, str] = {}

        def fake_get(url: str, timeout: float, headers: dict[str, str] | None = None):
            del url, timeout
            captured.update(headers or {})
            return {"status": "ok", "result": []}

        monkeypatch.setattr(provider, "_get_json", fake_get)

        assert provider.fetch(NOW, NOW + timedelta(days=7)) == []
        assert captured == TRADINGVIEW_HEADERS


class TestGoldIsWatchedInDollars:
    """The currency that moves gold is not one of gold's two legs.

    `symbol_currencies` returned exactly the pair, and its own docstring said
    so: "metals and crypto carry a pseudo-currency base that no calendar
    publishes events for... it simply never matches". That describes a hole, not
    a design.

    A live XAUAUD short is the case. It was working, a red-folder release moved
    gold, and the stop was taken for -1.01R and EUR 6.82 on a EUR 172 account —
    the biggest single loss of the day. The calendar HAD the event. Nothing
    asked it, because USD is neither leg of XAUAUD. The same day, an XAUJPY
    trade reported its next news as 3,509 minutes away: fifty-eight hours of
    clear sky, counted over Japanese releases only.
    """

    def test_gold_against_the_aussie_still_watches_america(self) -> None:
        from filters.calendar.events import symbol_currencies

        assert symbol_currencies("XAU", "AUD") == frozenset({"XAU", "AUD", "USD"})

    def test_gold_against_the_yen_too(self) -> None:
        from filters.calendar.events import symbol_currencies

        assert "USD" in symbol_currencies("XAU", "JPY")

    def test_crypto_is_dollar_priced_as_well(self) -> None:
        from filters.calendar.events import symbol_currencies

        assert "USD" in symbol_currencies("BTC", "EUR")

    def test_an_ordinary_cross_gains_nothing(self) -> None:
        """EURJPY has no dollar leg and must not start blacking out on US data."""
        from filters.calendar.events import symbol_currencies

        assert symbol_currencies("EUR", "JPY") == frozenset({"EUR", "JPY"})

    def test_xauusd_is_unchanged(self) -> None:
        from filters.calendar.events import symbol_currencies

        assert symbol_currencies("XAU", "USD") == frozenset({"XAU", "USD"})

    def test_the_base_is_still_carried(self) -> None:
        """It never matches today. Dropping it would be deciding which side of
        a pair counts, on the day a calendar finally publishes for it."""
        from filters.calendar.events import symbol_currencies

        assert "XAU" in symbol_currencies("XAU", "AUD")
