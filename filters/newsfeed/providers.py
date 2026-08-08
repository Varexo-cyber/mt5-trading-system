"""Where the headlines come from.

RSS and Atom only, and that is a deliberate limit rather than a first step.
Scraping a news site's HTML breaks on the next redesign, violates most terms of
service, and gets a VPS address blocked — at which point the layer is silently
dead and everything downstream reports "quiet market". A feed is published to
be read by machines, it is stable, and when it stops working it stops loudly.

**Verification status: UNVERIFIED FROM THE BUILD ENVIRONMENT.** The sandbox
this was written in refuses outbound connections to every one of these hosts,
so unlike `filters.calendar.providers` — whose response shapes were checked
against live feeds on 2026-08-01 — no URL below has been confirmed to answer or
to parse. `scripts/verify_newsfeed.py` exists to do that on the VPS, and it
should be run before this layer is trusted to block anything. Until it has
been, the honest reading of an empty result is "not wired up yet", which is why
`block_when_unavailable` defaults to false.

The parser is deliberately tolerant of dialect and strict about volume. Feeds
disagree on date format, on whether items are `<item>` or `<entry>`, and on
which namespace they declare. What they must not do is hand back three items
out of forty and have that pass for a quiet hour, so each fetch tracks what it
could not read and fails past a threshold — the same rule, and for the same
reason, as the calendar's.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

from core.errors import TradingSystemError
from filters.newsfeed.items import Headline
from infra.logging import get_logger

log = get_logger(__name__)

#: Browsers get served; bespoke agents get filtered by the CDNs in front of
#: most of these. The calendar providers learned the same thing.
FEED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
}

#: Fail the fetch if more than this fraction of items will not parse. A feed
#: half of which is unreadable is not a quiet feed.
MAX_UNPARSEABLE_FRACTION = 0.25

#: Ignore anything older than this on a fetch. Feeds carry a week of history
#: and this layer only ever asks about the last hour or so; parsing the rest is
#: work whose answer is discarded.
MAX_ITEM_AGE = timedelta(hours=12)

#: Candidate feeds. Every one of these needs `verify_newsfeed.py` run against
#: it on the VPS before it is worth anything — see the module docstring.
DEFAULT_FEEDS: tuple[tuple[str, str], ...] = (
    ("forexlive", "https://www.forexlive.com/feed/news"),
    ("fxstreet", "https://www.fxstreet.com/rss/news"),
    ("investing-forex", "https://www.investing.com/rss/news_1.rss"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
)


class FeedUnavailableError(TradingSystemError):
    """No feed answered, or what answered could not be believed."""


def _text(element: ElementTree.Element, *names: str) -> str:
    """First non-empty child matching any of `names`, namespace ignored.

    Namespace-blind because feeds declare Atom, Dublin Core and RSS extensions
    in whatever combination their generator felt like, and matching on the
    local tag name is the only thing that works across all of them without a
    table of namespace URIs that goes stale.
    """
    for child in element.iter():
        tag = child.tag.rsplit("}", 1)[-1].lower()
        if tag in names:
            if child.text and child.text.strip():
                return child.text.strip()
            # Atom puts the URL in an attribute rather than in the body.
            href = child.attrib.get("href", "").strip()
            if href:
                return href
    return ""


_ISO_TRAILING_Z = re.compile(r"Z$")


def parse_published(raw: str) -> datetime | None:
    """A feed timestamp, as tz-aware UTC, or None when it cannot be read.

    Two formats cover everything in practice: RFC 822 as RSS specifies, and
    ISO 8601 as Atom does. A feed offering neither has its item counted as
    unparseable rather than stamped with the time of the fetch — an item dated
    "now" because nobody could read its date is exactly the false spike this
    layer must not manufacture.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(_ISO_TRAILING_Z.sub("+00:00", text))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        # A naive stamp is almost always UTC on these feeds, and the
        # alternative — discarding it — loses the item entirely.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_feed(body: bytes, source: str, *, now: datetime) -> list[Headline]:
    """Every readable item in one feed document.

    Raises `FeedUnavailableError` when too much of it will not parse, because a
    partially-read feed reports a quiet market that is indistinguishable, from
    every log downstream, from an actually quiet market.
    """
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise FeedUnavailableError(f"{source}: not XML ({exc})") from exc

    items = [
        element
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1].lower() in ("item", "entry")
    ]
    if not items:
        raise FeedUnavailableError(f"{source}: no <item> or <entry> elements")

    headlines: list[Headline] = []
    unreadable = 0
    for element in items:
        title = _text(element, "title")
        stamp = parse_published(_text(element, "pubdate", "published", "updated", "date"))
        if not title or stamp is None:
            unreadable += 1
            continue
        if now - stamp > MAX_ITEM_AGE:
            continue
        headlines.append(
            Headline(
                published=stamp,
                title=" ".join(title.split()),
                source=source,
                link=_text(element, "link"),
            )
        )

    share = unreadable / len(items)
    if share > MAX_UNPARSEABLE_FRACTION:
        raise FeedUnavailableError(
            f"{source}: {unreadable} of {len(items)} items unreadable ({share:.0%}); "
            "refusing a partial answer"
        )
    return headlines


class HeadlineProvider(ABC):
    """One source of headlines."""

    name: str = "provider"

    @abstractmethod
    def fetch(self, now: datetime) -> list[Headline]:
        """Recent headlines, or raise `FeedUnavailableError`."""


class RssHeadlineProvider(HeadlineProvider):
    """A single RSS or Atom URL, polled with a conditional GET.

    The conditional GET is what makes fast polling defensible. Every response
    carries an `ETag` or a `Last-Modified`; sending it back means the server
    answers `304 Not Modified` in a couple of hundred bytes and does no work
    when nothing has changed, which on a wire is most of the time. That is the
    difference between checking often and hammering somebody's CDN until the
    VPS address is blocked — and a blocked address is the worst outcome
    available here, because the layer then reports a permanently quiet market.

    A 304 returns the previous parse rather than an empty list. Empty would
    read downstream as "nothing is being published", which is the one thing
    this must never invent.
    """

    def __init__(self, name: str, url: str, *, timeout_seconds: float = 10.0) -> None:
        self.name = name
        self.url = url
        self.timeout = timeout_seconds
        self._etag = ""
        self._modified = ""
        self._last: list[Headline] = []
        #: How many polls came back 304. Worth seeing: a feed that is always
        #: 304 is either genuinely quiet or quietly broken, and the count next
        #: to the fetch count is what tells them apart.
        self.not_modified = 0

    def fetch(self, now: datetime) -> list[Headline]:
        if not self.url.lower().startswith("https://"):
            raise FeedUnavailableError(f"{self.name}: refusing a non-HTTPS feed URL")
        headers = dict(FEED_HEADERS)
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._modified:
            headers["If-Modified-Since"] = self._modified

        request = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                self._etag = response.headers.get("ETag", "") or self._etag
                self._modified = response.headers.get("Last-Modified", "") or self._modified
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                self.not_modified += 1
                return list(self._last)
            raise FeedUnavailableError(f"{self.name}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise FeedUnavailableError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

        self._last = parse_feed(body, self.name, now=now)
        return list(self._last)


def build_providers(
    feeds: dict[str, str] | None = None, *, timeout_seconds: float = 10.0
) -> list[HeadlineProvider]:
    """Providers for the configured feeds, or the unverified defaults."""
    chosen = feeds if feeds else dict(DEFAULT_FEEDS)
    return [
        RssHeadlineProvider(name, url, timeout_seconds=timeout_seconds)
        for name, url in chosen.items()
    ]
