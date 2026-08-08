"""Keeps a rolling window of headlines and answers questions about one symbol.

Every feed is read on one schedule, not per symbol. Eighty-eight instruments
asking eighty-eight times would be eighty-eight fetches for one answer, and on
a one-core VPS running a one-second guard loop that is not a rounding error.
The wires publish for everyone; the split by instrument happens locally and
costs a regex.

Unlike `CalendarService` this does not fail closed on its own. That is a
departure from the rule the operator set — no data means no trade — and it is
narrower than it looks. The calendar's guarantee covers scheduled high-impact
releases and is untouched; it still stops all trading when it goes dark. This
layer adds coverage of *unscheduled* events, and when it is unavailable the
system falls back to exactly the safety level it ran at before this package
existed. Failing closed here would mean one flaky RSS host stopping a day of
trading, and RSS hosts are considerably flakier than the two calendar feeds.

`block_when_unavailable` in config flips it, and the reasoning above is the
whole argument for the default rather than a preference.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.clock import Clock
from filters.newsfeed.items import Headline, NewsPressure
from filters.newsfeed.providers import FeedUnavailableError, HeadlineProvider
from infra.logging import get_logger

log = get_logger(__name__)


class HeadlineService:
    """Fetches every feed on a schedule and holds a de-duplicated window."""

    def __init__(
        self,
        providers: list[HeadlineProvider],
        clock: Clock,
        *,
        refresh_interval_seconds: float = 120.0,
        window_minutes: float = 20.0,
        baseline_hours: float = 12.0,
        max_age_minutes: float = 30.0,
    ) -> None:
        if window_minutes <= 0 or baseline_hours <= 0:
            raise ValueError("window and baseline must be positive")
        if baseline_hours * 60.0 <= window_minutes:
            raise ValueError(
                "the baseline window must be longer than the recent window, or an "
                "instrument is being compared against itself"
            )
        self.providers = providers
        self.clock = clock
        self.refresh_interval = timedelta(seconds=refresh_interval_seconds)
        self.window = timedelta(minutes=window_minutes)
        self.baseline = timedelta(hours=baseline_hours)
        self.max_age = timedelta(minutes=max_age_minutes)

        self._headlines: dict[str, Headline] = {}
        self._fetched_at: datetime | None = None
        self._sources: tuple[str, ...] = ()
        # Per feed, so each is polled on its own rotation rather than the whole
        # list firing together. See `_is_due`.
        self._polled_at: dict[str, datetime] = {}
        self._started = clock.now()

    # -- state -------------------------------------------------------------

    @property
    def last_fetch(self) -> datetime | None:
        return self._fetched_at

    @property
    def sources(self) -> tuple[str, ...]:
        """Which feeds answered on the last refresh."""
        return self._sources

    @property
    def age(self) -> timedelta | None:
        if self._fetched_at is None:
            return None
        return self.clock.now() - self._fetched_at

    def is_usable(self) -> bool:
        """Is what is held recent enough to draw a conclusion from?

        A window of headlines from forty minutes ago cannot tell you the market
        is quiet now. It can only tell you it was quiet then, and the two are
        the same sentence right up until they are not.
        """
        age = self.age
        return age is not None and age <= self.max_age

    @property
    def count(self) -> int:
        return len(self._headlines)

    # -- refresh -----------------------------------------------------------

    def refresh(self, *, force: bool = False) -> bool:
        """Pull every feed if the interval has elapsed. True when it did.

        One provider failing is normal and is logged at debug: wires go down,
        and the others still answer. Every provider failing leaves what is held
        in place to age out through `is_usable`, rather than emptying the
        window — an empty window reads as "nothing is happening", which is the
        one conclusion an outage must never produce.
        """
        now = self.clock.now()
        due = [p for p in self.providers if force or self._is_due(p, now)]
        if not due:
            return False

        collected: list[Headline] = []
        answered: list[str] = []
        for provider in due:
            self._polled_at[provider.name] = now
            try:
                items = provider.fetch(now)
            except FeedUnavailableError as exc:
                log.debug(
                    "headline feed unavailable",
                    extra={"event": "headline_feed_down", "feed": provider.name, "why": str(exc)},
                )
                continue
            except Exception as exc:  # noqa: BLE001 - a wire must not end the cycle
                log.warning(
                    "headline feed raised",
                    extra={
                        "event": "headline_feed_error",
                        "feed": provider.name,
                        "why": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            collected.extend(items)
            answered.append(provider.name)

        if not answered:
            log.warning(
                "no headline feed answered; the layer holds its last window",
                extra={"event": "headline_all_feeds_down", "held": len(self._headlines)},
            )
            return False

        for headline in collected:
            # First sighting wins the timestamp. A story re-published by a
            # second wire ten minutes later is not a second event, and letting
            # the later stamp win would keep resetting its age.
            self._headlines.setdefault(headline.key, headline)
        self._prune(now)
        self._fetched_at = now
        self._sources = tuple(answered)
        log.info(
            "headlines refreshed",
            extra={
                "event": "headlines_refreshed",
                "feeds": answered,
                "fetched": len(collected),
                "held": len(self._headlines),
            },
        )
        return True

    def _is_due(self, provider: HeadlineProvider, now: datetime) -> bool:
        """Is this one feed's own turn yet?

        Per feed rather than for the batch, and staggered across the interval,
        which is how "check every second" is delivered without any single host
        being polled every second.

        The operator asked for one-second scraping. Hitting one URL every
        second gets the VPS rate-limited and then blocked, and a blocked
        address reports a permanently quiet market — the worst failure this
        layer has, because nothing downstream can tell. But polling twenty
        feeds on a rotation, each one every twenty seconds, means something is
        being fetched every second and no host sees more than three requests a
        minute. That is what a wire service expects, and the freshness the
        operator actually wanted is a property of the batch, not of any one
        source.

        Each feed's slot is its own position in the list, spread evenly across
        the interval. Without the offset all twenty fire together every twenty
        seconds, which is the same total traffic arriving as a thundering herd
        on a one-core VPS.
        """
        interval = self.refresh_interval.total_seconds()
        if interval <= 0:
            return True
        last = self._polled_at.get(provider.name)
        if last is None:
            # Stagger the very first poll too, or the whole list still starts
            # together on the first cycle after a restart.
            slot = self.providers.index(provider) / max(len(self.providers), 1)
            return (now - self._started).total_seconds() >= slot * interval
        return (now - last).total_seconds() >= interval

    def _prune(self, now: datetime) -> None:
        cutoff = now - self.baseline
        self._headlines = {
            key: item for key, item in self._headlines.items() if item.published >= cutoff
        }

    # -- questions ---------------------------------------------------------

    def pressure(self, symbol: str, currencies: frozenset[str]) -> NewsPressure:
        """What is being written about this instrument, against its own normal.

        The baseline is this instrument's own rate over the long window scaled
        to the short one, not a constant and not an average across symbols. EUR
        and NZD do not carry comparable traffic on any wire, and one threshold
        over both would mean blocking EUR permanently or NZD never.
        """
        now = self.clock.now()
        recent_cutoff = now - self.window
        touching = [item for item in self._headlines.values() if item.touches(currencies)]
        recent = [item for item in touching if item.published >= recent_cutoff]

        windows = self.baseline / self.window
        baseline = len(touching) / windows if windows > 0 else 0.0

        return NewsPressure(
            symbol=symbol,
            currencies=currencies,
            recent=len(recent),
            baseline=baseline,
            systemic=any(item.systemic for item in recent),
            window_minutes=self.window.total_seconds() / 60.0,
            latest=tuple(sorted(recent, key=lambda item: item.published, reverse=True)[:5]),
        )

    def newest(self, *, limit: int = 200) -> tuple[Headline, ...]:
        """Everything currently held, newest first.

        For the archive rather than for a decision. A feed carries a few hours;
        keeping the copy is the only way "what was being written when this
        trade opened" is still answerable next year, and that link is the one
        worth learning from.
        """
        ordered = sorted(self._headlines.values(), key=lambda item: item.published, reverse=True)
        return tuple(ordered[:limit])

    def recent_for(self, currencies: frozenset[str], *, limit: int = 8) -> tuple[Headline, ...]:
        """The newest headlines touching these currencies, for a reader.

        Used by the supervision payload. This is the one place the *text* is
        worth carrying: a language model reads a headline far better than any
        regex in this package, and it is already being asked about the position
        anyway, so the marginal cost is a few hundred tokens.
        """
        touching = [item for item in self._headlines.values() if item.touches(currencies)]
        return tuple(sorted(touching, key=lambda item: item.published, reverse=True)[:limit])
