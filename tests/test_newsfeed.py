"""The headline layer, and the traps it is built to avoid.

Most of this file is about the tagger, because a tagger is the part of a news
layer that fails without anybody noticing. If "Goldman Sachs" tags XAU, gold
trading stops on bank stories and every log, chart and report shows a market
that was simply busy. There is no downstream symptom. So the substring traps
are enumerated here by name.

The rest holds the two properties the layer's honesty rests on: a feed that
half-parses is refused rather than reported as quiet, and an outage never
empties the window into "nothing is happening".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.schema import HeadlineFilterConfig
from filters.base import FilterContext
from filters.headline_filter import HeadlineFilter
from filters.newsfeed.items import Headline, NewsPressure
from filters.newsfeed.providers import (
    DEFAULT_FEEDS,
    FeedUnavailableError,
    HeadlineProvider,
    RssHeadlineProvider,
    parse_feed,
    parse_published,
)
from filters.newsfeed.service import HeadlineService
from filters.newsfeed.tagging import currencies_in, is_systemic

NOW = datetime(2026, 8, 7, 14, 0, tzinfo=UTC)


class FrozenClock:
    def __init__(self, at: datetime = NOW) -> None:
        self.at = at

    def now(self) -> datetime:
        return self.at


class StubProvider(HeadlineProvider):
    def __init__(self, name: str, items: list[Headline] | None = None, fails: bool = False) -> None:
        self.name = name
        self.items = items or []
        self.fails = fails
        self.calls = 0

    def fetch(self, now: datetime) -> list[Headline]:
        self.calls += 1
        if self.fails:
            raise FeedUnavailableError(f"{self.name}: down")
        return list(self.items)


def headline(title: str, *, minutes_ago: float = 1.0, source: str = "wire") -> Headline:
    return Headline(published=NOW - timedelta(minutes=minutes_ago), title=title, source=source)


# ------------------------------------------------------------- tagging ---


class TestTheSubstringTraps:
    """Each of these is a real headline shape that a naive `in` test mis-tags,
    and each one would block an instrument for a reason nobody could see."""

    @pytest.mark.parametrize(
        "title",
        [
            "Goldman Sachs downgrades Ford to neutral",
            "Goldcorp announces merger",
            "Marigold Resources files for bankruptcy",
        ],
    )
    def test_goldman_is_not_gold(self, title: str) -> None:
        assert "XAU" not in currencies_in(title)

    def test_gold_itself_still_tags(self) -> None:
        assert "XAU" in currencies_in("Gold hits record high on haven demand")
        assert "XAU" in currencies_in("Bullion rallies as yields fall")

    @pytest.mark.parametrize(
        "title",
        [
            "Neural networks reshape trading desks",
            "Amateur investors pile into meme stocks",
        ],
    )
    def test_eur_does_not_match_inside_a_word(self, title: str) -> None:
        assert "EUR" not in currencies_in(title)

    def test_euro_and_its_institutions_tag(self) -> None:
        assert "EUR" in currencies_in("Euro slips ahead of the ECB decision")
        assert "EUR" in currencies_in("Lagarde signals patience on cuts")
        assert "EUR" in currencies_in("Eurozone inflation cools to 2.1%")

    def test_a_bare_dollar_does_not_tag_usd(self) -> None:
        """Australian, Canadian, New Zealand and Singapore dollars all answer
        to it. Tagging USD on 'dollar' puts a blackout on every major at once,
        which is worse than missing the story."""
        assert "USD" not in currencies_in("Dollar edges higher in thin trade")

    def test_the_unambiguous_usd_terms_do_tag(self) -> None:
        for title in ("US dollar firms after Powell", "Greenback rallies", "FOMC holds rates"):
            assert "USD" in currencies_in(title), title

    def test_a_headline_about_nothing_tradeable_tags_nothing(self) -> None:
        assert currencies_in("Local council approves new cycle lane") == frozenset()

    def test_one_headline_can_touch_two_currencies(self) -> None:
        tags = currencies_in("Yen slides to a 30-year low against the US dollar")
        assert tags == frozenset({"JPY", "USD"})

    def test_case_and_punctuation_do_not_matter(self) -> None:
        assert "GBP" in currencies_in("STERLING JUMPS; BoE hints at a hold.")


class TestSystemicStories:
    def test_a_market_wide_story_is_recognised(self) -> None:
        assert is_systemic("Risk-off grips markets as invasion escalates")
        assert is_systemic("Exchange halts trading after circuit breaker")

    def test_an_ordinary_story_is_not(self) -> None:
        assert not is_systemic("Euro slips ahead of the ECB decision")

    def test_a_systemic_headline_touches_an_instrument_it_never_names(self) -> None:
        """The tagger's most expensive miss would be here. A story naming no
        currency and moving all of them is exactly when nothing should open."""
        item = headline("Flight to safety as sanctions widen")

        assert item.currencies == frozenset()
        assert item.touches(frozenset({"NZD", "CAD"}))


# ------------------------------------------------------------ parsing ---


class TestReadingAFeed:
    RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><title>Euro slips ahead of the ECB</title>
        <pubDate>Fri, 07 Aug 2026 13:55:00 GMT</pubDate>
        <link>https://example.test/1</link></item>
      <item><title>Yen firms after BoJ comments</title>
        <pubDate>Fri, 07 Aug 2026 13:40:00 +0000</pubDate></item>
    </channel></rss>"""

    ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>Gold hits record high</title>
        <updated>2026-08-07T13:50:00Z</updated>
        <link href="https://example.test/2"/></entry>
    </feed>"""

    def test_rss_parses(self) -> None:
        items = parse_feed(self.RSS.encode(), "wire", now=NOW)

        assert [i.title for i in items] == [
            "Euro slips ahead of the ECB",
            "Yen firms after BoJ comments",
        ]
        assert items[0].link == "https://example.test/1"
        assert items[0].published == datetime(2026, 8, 7, 13, 55, tzinfo=UTC)

    def test_atom_parses_including_the_link_attribute(self) -> None:
        items = parse_feed(self.ATOM.encode(), "wire", now=NOW)

        assert items[0].title == "Gold hits record high"
        assert items[0].link == "https://example.test/2"

    def test_a_mostly_unreadable_feed_is_refused(self) -> None:
        """The failure this exists to prevent: three items out of forty parse,
        the other thirty-seven are invisible, and the layer reports calm."""
        broken = (
            "<rss><channel>"
            + "<item><title>x</title></item>" * 9
            + ("<item><title>ok</title><pubDate>Fri, 07 Aug 2026 13:55:00 GMT</pubDate></item>")
            + "</channel></rss>"
        )

        with pytest.raises(FeedUnavailableError, match="unreadable"):
            parse_feed(broken.encode(), "wire", now=NOW)

    def test_a_feed_with_no_items_is_refused(self) -> None:
        with pytest.raises(FeedUnavailableError, match="no <item>"):
            parse_feed(b"<rss><channel></channel></rss>", "wire", now=NOW)

    def test_something_that_is_not_xml_is_refused(self) -> None:
        with pytest.raises(FeedUnavailableError, match="not XML"):
            parse_feed(b"403 Forbidden <<< not markup at all", "wire", now=NOW)

    def test_an_error_page_that_happens_to_parse_is_still_refused(self) -> None:
        """A CDN block page is often well-formed XML. It parses cleanly and
        contains no items, and the only safe reading of that is a failure —
        returning zero headlines would report the calmest hour on record."""
        with pytest.raises(FeedUnavailableError, match="no <item>"):
            parse_feed(b"<html><body>403 Forbidden</body></html>", "wire", now=NOW)

    def test_stale_items_are_dropped_not_counted(self) -> None:
        old = """<rss><channel><item><title>Ancient</title>
          <pubDate>Mon, 03 Aug 2026 09:00:00 GMT</pubDate></item>
          <item><title>Euro slips</title>
          <pubDate>Fri, 07 Aug 2026 13:55:00 GMT</pubDate></item></channel></rss>"""

        assert [i.title for i in parse_feed(old.encode(), "wire", now=NOW)] == ["Euro slips"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Fri, 07 Aug 2026 13:55:00 GMT", datetime(2026, 8, 7, 13, 55, tzinfo=UTC)),
            ("2026-08-07T13:55:00Z", datetime(2026, 8, 7, 13, 55, tzinfo=UTC)),
            ("2026-08-07T15:55:00+02:00", datetime(2026, 8, 7, 13, 55, tzinfo=UTC)),
            ("2026-08-07 13:55:00", datetime(2026, 8, 7, 13, 55, tzinfo=UTC)),
        ],
    )
    def test_the_dialects_of_a_timestamp(self, raw: str, expected: datetime) -> None:
        assert parse_published(raw) == expected

    def test_an_unreadable_timestamp_is_not_stamped_as_now(self) -> None:
        """An item dated 'now' because nobody could read its date is a false
        spike this layer would then act on."""
        assert parse_published("last Tuesday-ish") is None
        assert parse_published("") is None

    def test_a_non_https_feed_is_refused_before_it_is_fetched(self) -> None:
        provider = RssHeadlineProvider("insecure", "http://example.test/feed")

        with pytest.raises(FeedUnavailableError, match="non-HTTPS"):
            provider.fetch(NOW)


# ------------------------------------------------------------ service ---


class TestTheWindow:
    def service(self, providers: list[HeadlineProvider], **kwargs) -> HeadlineService:  # type: ignore[no-untyped-def]
        defaults = {"window_minutes": 20.0, "baseline_hours": 12.0, "max_age_minutes": 30.0}
        return HeadlineService(providers, FrozenClock(), **{**defaults, **kwargs})

    def test_it_counts_what_touches_the_instrument(self) -> None:
        feed = StubProvider(
            "wire",
            [
                headline("Euro slips ahead of the ECB", minutes_ago=2),
                headline("Lagarde signals patience", minutes_ago=5),
                headline("Gold hits record high", minutes_ago=3),
            ],
        )
        service = self.service([feed])
        service.refresh()

        assert service.pressure("EURUSD", frozenset({"EUR", "USD"})).recent == 2
        assert service.pressure("XAUUSD", frozenset({"XAU", "USD"})).recent == 1
        assert service.pressure("AUDNZD", frozenset({"AUD", "NZD"})).recent == 0

    def test_the_same_story_from_two_wires_counts_once(self) -> None:
        """This layer measures how many things are happening. A story carried
        by three wires is one thing, and counting it three times manufactures
        exactly the spike the filter acts on."""
        title = "ECB cuts rates by 25bp"
        one = StubProvider("a", [headline(title, minutes_ago=2, source="a")])
        two = StubProvider("b", [headline(title.upper(), minutes_ago=1, source="b")])
        service = self.service([one, two])
        service.refresh()

        assert service.pressure("EURUSD", frozenset({"EUR"})).recent == 1

    def test_the_first_sighting_keeps_the_timestamp(self) -> None:
        """A re-publication is not a fresh event, and letting the later stamp
        win would keep resetting the story's age."""
        service = self.service(
            [StubProvider("a", [headline("ECB cuts rates", minutes_ago=30)])],
            window_minutes=10.0,
        )
        service.refresh()
        service.providers.append(StubProvider("b", [headline("ECB cuts rates", minutes_ago=1)]))
        service.refresh(force=True)

        assert service.pressure("EURUSD", frozenset({"EUR"})).recent == 0

    def test_the_baseline_is_the_instruments_own_rate(self) -> None:
        """Twelve EUR stories over twelve hours is one per hour, which over a
        twenty-minute window is a third of a story."""
        items = [headline(f"Euro note {n}", minutes_ago=60 * n + 30) for n in range(12)]
        service = self.service([StubProvider("wire", items)])
        service.refresh()

        pressure = service.pressure("EURUSD", frozenset({"EUR"}))

        assert pressure.recent == 0
        assert pressure.baseline == pytest.approx(12 / 36)

    def test_a_quiet_instrument_is_not_permanently_spiking(self) -> None:
        """Without the floor on the baseline, one routine mention of a pair
        nobody writes about is a fiftyfold spike and blocks it for good."""
        service = self.service([StubProvider("wire", [headline("Kiwi steadies", minutes_ago=2)])])
        service.refresh()

        assert service.pressure("NZDUSD", frozenset({"NZD"})).multiple == pytest.approx(1.0)

    def test_an_outage_holds_the_window_rather_than_emptying_it(self) -> None:
        """An empty window reads as 'nothing is happening'. That is the one
        conclusion an outage must never produce."""
        feed = StubProvider("wire", [headline("Euro slips", minutes_ago=2)])
        service = self.service([feed])
        service.refresh()
        feed.fails = True

        assert service.refresh(force=True) is False
        assert service.count == 1

    def test_one_feed_down_does_not_stop_the_others(self) -> None:
        """`force` polls the whole list at once. Without it the feeds are
        staggered and only the first is due on the opening call, which is the
        subject of `TestStaggeringTheFeeds` below rather than of this test."""
        service = self.service(
            [
                StubProvider("down", fails=True),
                StubProvider("up", [headline("Euro slips", minutes_ago=2)]),
            ]
        )

        assert service.refresh(force=True) is True
        assert service.sources == ("up",)

    def test_it_does_not_refetch_inside_the_interval(self) -> None:
        """Eighty-eight symbols asking would be eighty-eight fetches for one
        answer, on a machine with one core and a one-second guard loop."""
        feed = StubProvider("wire", [headline("Euro slips")])
        service = self.service([feed], refresh_interval_seconds=600.0)
        for _ in range(20):
            service.refresh()

        assert feed.calls == 1

    def test_a_window_that_has_never_filled_is_not_usable(self) -> None:
        assert self.service([StubProvider("wire")]).is_usable() is False

    def test_a_stale_window_stops_being_evidence(self) -> None:
        clock = FrozenClock()
        service = HeadlineService(
            [StubProvider("wire", [headline("Euro slips")])],
            clock,
            max_age_minutes=30.0,
        )
        service.refresh()
        assert service.is_usable()

        clock.at = NOW + timedelta(minutes=31)
        assert service.is_usable() is False

    def test_a_baseline_shorter_than_the_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="compared against itself"):
            HeadlineService(
                [StubProvider("wire")], FrozenClock(), window_minutes=120.0, baseline_hours=1.0
            )


class TestNewsPressureArithmetic:
    def test_the_multiple_floors_the_baseline_at_one(self) -> None:
        quiet = NewsPressure("NZDUSD", frozenset({"NZD"}), 1, 0.02, False, 20.0)
        assert quiet.multiple == pytest.approx(1.0)

    def test_a_real_spike_reads_as_one(self) -> None:
        busy = NewsPressure("EURUSD", frozenset({"EUR"}), 9, 3.0, False, 20.0)
        assert busy.multiple == pytest.approx(3.0)


# ------------------------------------------------------------- filter ---


class Spec:
    currency_base = "EUR"
    currency_profit = "USD"


class TestTheFilter:
    def context(self) -> FilterContext:
        return FilterContext(symbol="EURUSD.i", spec=Spec(), now=NOW)  # type: ignore[arg-type]

    def build(self, items: list[Headline], **config):  # type: ignore[no-untyped-def]
        service = HeadlineService(
            [StubProvider("wire", items)], FrozenClock(), window_minutes=20.0, baseline_hours=12.0
        )
        settings = HeadlineFilterConfig(enabled=True, **config)
        return HeadlineFilter(settings, service)

    def test_a_quiet_market_passes_and_still_reports_its_numbers(self) -> None:
        verdict = self.build([headline("Euro slips", minutes_ago=2)]).check(self.context())

        assert verdict.passed
        assert verdict.data["headline_count"] == 1
        assert "headline_baseline" in verdict.data

    def test_a_burst_on_this_instrument_blocks(self) -> None:
        items = [headline(f"Euro headline {n}", minutes_ago=n) for n in (1, 2, 3, 4, 5)]
        items += [headline(f"ECB story {n}", minutes_ago=60 * n) for n in range(1, 4)]

        verdict = self.build(items).check(self.context())

        assert not verdict.passed
        assert str(verdict.reason) == "HEADLINE_PRESSURE"
        assert "unusual news flow" in verdict.detail

    def test_a_burst_on_another_instrument_does_not(self) -> None:
        items = [headline(f"Kiwi tumbles {n}", minutes_ago=n) for n in (1, 2, 3, 4, 5)]

        assert self.build(items).check(self.context()).passed

    def test_a_market_wide_story_blocks_on_its_own(self) -> None:
        """It touches every pair equally, so it never looks like a spike on any
        one of them and the spike test would wave all of them through at the
        worst possible moment."""
        verdict = self.build([headline("Flight to safety as invasion widens")]).check(
            self.context()
        )

        assert not verdict.passed
        assert "market-wide" in verdict.detail

    def test_three_headlines_that_are_normal_for_this_pair_pass(self) -> None:
        """Both conditions have to hold. Enough stories to mean anything, and
        enough above normal to be unusual."""
        items = [headline(f"Euro headline {n}", minutes_ago=n) for n in (1, 2, 3)]
        # Forty more across the twelve hours, all outside the recent window, so
        # this pair's normal traffic works out above one story per window and
        # three of them is a slow afternoon rather than an event.
        items += [headline(f"Euro note {n}", minutes_ago=25 + 15 * n) for n in range(1, 41)]

        verdict = self.build(items).check(self.context())

        assert verdict.passed
        assert verdict.data["headline_count"] == 3
        assert verdict.data["headline_multiple"] < 3.0

    def test_disabled_is_a_pass_without_a_fetch(self) -> None:
        feed = StubProvider("wire", [headline("Flight to safety as invasion widens")])
        service = HeadlineService([feed], FrozenClock())
        verdict = HeadlineFilter(HeadlineFilterConfig(enabled=False), service).check(self.context())

        assert verdict.passed and feed.calls == 0

    def test_a_dark_feed_does_not_block_by_default(self) -> None:
        """The departure from 'no data, no trade', and the narrow one: the
        calendar still fails closed, so this leaves the system at the safety
        level it ran at before the layer existed."""
        service = HeadlineService([StubProvider("wire", fails=True)], FrozenClock())
        verdict = HeadlineFilter(HeadlineFilterConfig(enabled=True), service).check(self.context())

        assert verdict.passed
        assert "not blocking on it" in verdict.detail

    def test_a_dark_feed_blocks_when_the_operator_asks_for_it(self) -> None:
        service = HeadlineService([StubProvider("wire", fails=True)], FrozenClock())
        config = HeadlineFilterConfig(enabled=True, block_when_unavailable=True)

        verdict = HeadlineFilter(config, service).check(self.context())

        assert not verdict.passed
        assert str(verdict.reason) == "HEADLINES_UNAVAILABLE"

    def test_it_is_off_until_the_feeds_have_been_verified(self) -> None:
        """Not one default URL could be reached from the environment this was
        written in, so no response shape is confirmed. A layer that silently
        returns nothing reports a quiet market."""
        assert HeadlineFilterConfig().enabled is False


class TestBeyondCurrencies:
    """The operator asked whether it was only currency news. It was."""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Nasdaq tumbles as chip stocks slide", "US100"),
            ("S&P 500 closes at a record", "US500"),
            ("FTSE lags European peers", "UK100"),
            ("DAX rallies on factory orders", "DE40"),
            ("Nikkei jumps after the BoJ holds", "JP225"),
            ("Brent crude slips below $70", "OIL"),
            ("OPEC extends production cuts", "OIL"),
            ("Natural gas spikes on a cold snap", "NGAS"),
        ],
    )
    def test_indices_and_commodities_tag(self, title: str, expected: str) -> None:
        assert expected in currencies_in(title)

    def test_a_story_can_reach_an_index_and_a_currency_at_once(self) -> None:
        tags = currencies_in("Nikkei jumps as the yen weakens")

        assert tags == frozenset({"JP225", "JPY"})

    def test_individual_equities_are_deliberately_absent(self) -> None:
        """Half the ticker space is ordinary English -- Gap, Ford, Target,
        Shell, Visa -- and the substring traps get very much worse. Index-level
        terms cover the same risk for what this account actually trades."""
        assert currencies_in("Ford beats on revenue") == frozenset()
        assert currencies_in("Shell announces a buyback") == frozenset()

    def test_the_word_boundary_rule_still_holds_for_the_new_terms(self) -> None:
        assert "OIL" not in currencies_in("The spoiled batch was recalled")
        assert "OIL" not in currencies_in("Turmoil in the bond market")


class TestPollingPolitely:
    """Fifteen-second polling is only defensible because of the conditional
    GET. Without it this is hammering somebody's CDN until the VPS address is
    blocked, and a blocked address reports a permanently quiet market."""

    @staticmethod
    def install(monkeypatch: pytest.MonkeyPatch, responses: list[object]):  # type: ignore[no-untyped-def]
        """Queue up what the network will answer, and record what was asked.

        Through `monkeypatch` rather than by assigning to the module, so the
        real `urlopen` is restored even when a test fails. A leaked patch here
        would make some later test's network call return a fixture.
        """
        import urllib.error

        import filters.newsfeed.providers as module

        queue = list(responses)
        sent: list[dict[str, str]] = []

        class Response:
            def __init__(self, body: bytes, headers: dict[str, str]) -> None:
                self.body, self.headers = body, headers

            def read(self) -> bytes:
                return self.body

            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *_: object) -> bool:
                return False

        def fake_urlopen(request, timeout=None, context=None):  # type: ignore[no-untyped-def]
            del context
            sent.append({key.lower(): value for key, value in request.headers.items()})
            nxt = queue.pop(0)
            if isinstance(nxt, urllib.error.HTTPError):
                raise nxt
            return Response(*nxt)  # type: ignore[misc]

        monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
        return RssHeadlineProvider("wire", "https://example.test/feed"), sent

    def test_the_etag_comes_back_on_the_next_poll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = TestReadingAFeed.RSS.encode()
        provider, sent = self.install(monkeypatch, [(body, {"ETag": '"abc123"'}), (body, {})])

        provider.fetch(NOW)
        provider.fetch(NOW)

        assert "if-none-match" not in sent[0], "nothing to send on the first poll"
        assert sent[1]["if-none-match"] == '"abc123"'

    def test_last_modified_is_used_when_there_is_no_etag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = TestReadingAFeed.RSS.encode()
        stamp = "Fri, 07 Aug 2026 13:55:00 GMT"
        provider, sent = self.install(monkeypatch, [(body, {"Last-Modified": stamp}), (body, {})])

        provider.fetch(NOW)
        provider.fetch(NOW)

        assert sent[1]["if-modified-since"] == stamp

    def test_a_304_returns_the_previous_parse_not_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty would read downstream as 'nothing is being published', which
        is the one thing this layer must never invent."""
        import urllib.error

        not_modified = urllib.error.HTTPError(
            "https://example.test/feed", 304, "Not Modified", {}, None
        )
        provider, _ = self.install(monkeypatch, [(TestReadingAFeed.RSS.encode(), {}), not_modified])

        first = provider.fetch(NOW)
        second = provider.fetch(NOW)

        assert [i.title for i in second] == [i.title for i in first]
        assert provider.not_modified == 1

    def test_a_real_http_error_is_still_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.error

        forbidden = urllib.error.HTTPError("https://example.test/feed", 403, "Forbidden", {}, None)
        provider, _ = self.install(monkeypatch, [forbidden])

        with pytest.raises(FeedUnavailableError, match="HTTP 403"):
            provider.fetch(NOW)

    def test_the_default_cadence_is_per_feed_not_per_check(self) -> None:
        """Twenty seconds each, across twenty feeds on a rotation: something is
        fetched roughly every second and no host sees more than three requests
        a minute. Hitting one URL every second instead gets the address
        blocked, and a blocked address reports a permanently quiet market."""
        assert HeadlineFilterConfig().refresh_interval_seconds == 20.0
        assert len(DEFAULT_FEEDS) >= 18


class TestStaggeringTheFeeds:
    """How "scrape every second" is delivered without any host being hit every
    second. Twenty feeds on a twenty-second rotation: something is fetched
    roughly every second, each host sees three requests a minute.

    Hitting one URL every second instead gets the address rate-limited and then
    blocked, and a blocked address reports a permanently quiet market — the
    worst failure available here, because nothing downstream can detect it.
    """

    def service(self, count: int, clock: FrozenClock, interval: float = 20.0) -> HeadlineService:
        providers = [
            StubProvider(f"feed{n}", [headline(f"Euro note {n}", minutes_ago=1)])
            for n in range(count)
        ]
        return HeadlineService(providers, clock, refresh_interval_seconds=interval)  # type: ignore[arg-type]

    def test_the_first_call_does_not_poll_every_feed_at_once(self) -> None:
        """Twenty simultaneous HTTPS requests on a one-core VPS is a thundering
        herd, and it repeats every interval forever."""
        clock = FrozenClock()
        service = self.service(20, clock)

        service.refresh()

        assert sum(p.calls for p in service.providers) == 1  # type: ignore[attr-defined]

    def test_the_whole_list_is_covered_within_one_interval(self) -> None:
        clock = FrozenClock()
        service = self.service(20, clock)

        for second in range(21):
            clock.at = NOW + timedelta(seconds=second)
            service.refresh()

        assert all(p.calls >= 1 for p in service.providers), "every feed had its turn"  # type: ignore[attr-defined]

    def test_no_single_feed_is_polled_faster_than_its_interval(self) -> None:
        """The property that keeps the address unblocked."""
        clock = FrozenClock()
        service = self.service(20, clock)

        for second in range(60):
            clock.at = NOW + timedelta(seconds=second)
            service.refresh()

        for provider in service.providers:
            assert provider.calls <= 4, f"{provider.name} polled {provider.calls} times a minute"  # type: ignore[attr-defined]

    def test_something_is_fetched_almost_every_second(self) -> None:
        """The freshness the operator asked for, as a property of the batch."""
        clock = FrozenClock()
        service = self.service(20, clock)
        busy = 0

        for second in range(40):
            clock.at = NOW + timedelta(seconds=second)
            before = sum(p.calls for p in service.providers)  # type: ignore[attr-defined]
            service.refresh()
            if sum(p.calls for p in service.providers) > before:  # type: ignore[attr-defined]
                busy += 1

        assert busy >= 30, f"only {busy} of 40 seconds fetched anything"

    def test_force_still_polls_everything(self) -> None:
        """`verify_newsfeed.py` needs one complete sweep, not a slice of one."""
        clock = FrozenClock()
        service = self.service(20, clock)

        service.refresh(force=True)

        assert all(p.calls == 1 for p in service.providers)  # type: ignore[attr-defined]

    def test_a_single_feed_still_polls_on_the_interval(self) -> None:
        """With one feed there is nothing to stagger against, and the rotation
        must not accidentally make it slower than configured."""
        clock = FrozenClock()
        service = self.service(1, clock, interval=20.0)

        service.refresh()
        clock.at = NOW + timedelta(seconds=19)
        service.refresh()
        clock.at = NOW + timedelta(seconds=21)
        service.refresh()

        assert service.providers[0].calls == 2  # type: ignore[attr-defined]


class TestVerifyingCertificates:
    """Seventeen of twenty-one feeds failed on the VPS with
    CERTIFICATE_VERIFY_FAILED. Python on Windows does not read the OS
    certificate store for `ssl`, so without a CA bundle on disk it cannot
    verify most hosts — and the few that worked were the ones whose chain
    reached a root it happened to have. That pattern reads as "the feeds are
    down" rather than "this client cannot verify anybody", which is why it
    needs a test rather than a comment.
    """

    def test_the_context_uses_certifis_bundle(self) -> None:
        import certifi

        from filters.newsfeed.providers import ssl_context

        context = ssl_context()

        assert context.verify_mode.name == "CERT_REQUIRED"
        assert context.get_ca_certs(), "the bundle has to actually be loaded"
        assert certifi.where()  # the dependency is declared and importable

    def test_verification_is_never_switched_off(self) -> None:
        """The one-line fix for this error is an unverified context, and it
        would make every feed work immediately. It would also let anything on
        the path feed this system invented headlines — a strange thing to
        accept for a layer whose job is deciding when not to trade."""
        from filters.newsfeed.providers import ssl_context

        context = ssl_context()

        assert context.check_hostname is True
        assert context.verify_mode == __import__("ssl").CERT_REQUIRED

    def test_the_source_never_reaches_for_the_escape_hatch(self) -> None:
        from filters.newsfeed.providers import __file__ as source_path

        source = Path(source_path).read_text(encoding="utf-8")

        assert "_create_unverified_context" not in source
        assert "CERT_NONE" not in source

    def test_the_context_is_built_once(self) -> None:
        """It is reached on every poll of every feed, and building one parses
        the whole bundle."""
        from filters.newsfeed.providers import ssl_context

        assert ssl_context() is ssl_context()

    def test_the_fetch_passes_it_to_urlopen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A context nobody hands to `urlopen` verifies nothing."""
        import filters.newsfeed.providers as module

        seen: dict[str, object] = {}

        class Response:
            def __init__(self) -> None:
                self.headers: dict[str, str] = {}

            def read(self) -> bytes:
                return TestReadingAFeed.RSS.encode()

            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *_: object) -> bool:
                return False

        def fake_urlopen(request, timeout=None, context=None):  # type: ignore[no-untyped-def]
            seen["context"] = context
            return Response()

        monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
        RssHeadlineProvider("wire", "https://example.test/feed").fetch(NOW)

        assert seen["context"] is module.ssl_context()

    def test_the_blocked_feeds_were_removed(self) -> None:
        """fxstreet answered HTTP 403, which is a deliberate block and not a
        transport problem. A feed that will never answer costs a slot in the
        rotation every twenty seconds forever."""
        assert not any("fxstreet" in url for _, url in DEFAULT_FEEDS)
