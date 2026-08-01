"""The news filter. Mandatory, and it fails closed.

Two rules, neither negotiable:

1. **No entry inside a blackout window** around a high-impact release that
   touches either of the instrument's currencies. 60 minutes before to 30 after
   by default; 120/60 for the releases that move everything (FOMC, NFP, CPI,
   rate decisions).
2. **No calendar means no trade.** If every provider is down and the cache has
   expired, this filter blocks. It never assumes an empty calendar means a
   quiet day — those two states are indistinguishable from the outside, and
   guessing wrong once during an NFP print costs more than every skipped setup
   the filter will ever cause.

Open positions are handled separately from entries. Approaching news does not
close a winner by default; it moves the stop to break-even, because a spread
spike through a structural stop during a release is a specific, avoidable way
to lose. `open_position_action` in config chooses between break-even, closing,
and leaving it alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from config.schema import NewsFilterConfig
from core.clock import Clock
from core.types import Position
from filters.base import Filter, FilterContext, FilterVerdict
from filters.calendar.events import EconomicEvent, Impact, symbol_currencies
from filters.calendar.providers import CalendarUnavailableError
from filters.calendar.service import CalendarService
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Blackout:
    """A window during which one event forbids entry."""

    event: EconomicEvent
    start: datetime
    end: datetime
    extreme: bool

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end

    def minutes_until_start(self, moment: datetime) -> float:
        return (self.start - moment).total_seconds() / 60.0

    def minutes_until_end(self, moment: datetime) -> float:
        return (self.end - moment).total_seconds() / 60.0


class NewsFilter(Filter):
    """Blocks entries around high-impact releases, and says so loudly."""

    name = "news"

    def __init__(self, config: NewsFilterConfig, calendar: CalendarService, clock: Clock) -> None:
        self.config = config
        self.calendar = calendar
        self.clock = clock

    # -- entry gate --------------------------------------------------------

    def check(self, ctx: FilterContext) -> FilterVerdict:
        currencies = symbol_currencies(ctx.spec.currency_base, ctx.spec.currency_profit)

        try:
            blackouts = self._blackouts(currencies)
        except CalendarUnavailableError as exc:
            # The fail-safe. Not a warning, not a degraded mode: a block.
            log.error(  # noqa: TRY400 - the traceback adds nothing; the reason is the point
                "no calendar — blocking all entries",
                extra={
                    "event": "news_fail_closed",
                    "symbol": ctx.symbol,
                    "reason": str(exc),
                },
            )
            return FilterVerdict.block(
                self.name,
                Reason.NEWS_CALENDAR_UNAVAILABLE,
                f"economic calendar unavailable ({exc}); no data means no trade",
                calendar_source="none",
            )

        active = next((b for b in blackouts if b.contains(ctx.now)), None)
        if active is not None:
            minutes = active.minutes_until_end(ctx.now)
            return FilterVerdict.block(
                self.name,
                Reason.NEWS_BLACKOUT,
                f"{'extreme' if active.extreme else 'high'}-impact "
                f"{active.event.currency} event in the blackout window: "
                f"{active.event.title} at {active.event.when:%H:%M} UTC; "
                f"clear in {minutes:.0f} min",
                calendar_source=self.calendar.source,
                news_event=active.event.title,
                news_currency=active.event.currency,
                news_at=active.event.when.isoformat(),
                minutes_to_news_clear=round(minutes, 1),
            )

        upcoming = self.next_blackout(currencies, ctx.now)
        minutes_to_news = (
            round(upcoming.minutes_until_start(ctx.now), 1) if upcoming is not None else None
        )
        return FilterVerdict.allow(
            self.name,
            (
                f"clear; next blackout in {minutes_to_news:.0f} min ({upcoming.event.title})"
                if upcoming is not None
                else "clear; no high-impact events in the visible window"
            ),
            calendar_source=self.calendar.source,
            minutes_to_news=minutes_to_news,
        )

    # -- open positions ----------------------------------------------------

    def position_action(self, position: Position, currency_base: str, currency_profit: str) -> str:
        """What to do with an open position as news approaches.

        Returns one of `none`, `break_even`, `close` — the configured action if
        a blackout is imminent or active, `none` otherwise. Returns the
        configured action on calendar failure too: an unknown calendar is a
        reason to de-risk what is already open, not to leave it exposed.
        """
        now = self.clock.now()
        currencies = symbol_currencies(currency_base, currency_profit)

        try:
            blackouts = self._blackouts(currencies)
        except CalendarUnavailableError:
            log.warning(
                "no calendar; de-risking open position",
                extra={
                    "event": "news_position_fail_closed",
                    "ticket": position.ticket,
                    "action": self.config.open_position_action,
                },
            )
            return self.config.open_position_action

        imminent = next((b for b in blackouts if b.contains(now) or b.start >= now), None)
        if imminent is None:
            return "none"
        if not (imminent.contains(now) or imminent.minutes_until_start(now) <= 0):
            # Only act once we are actually inside the pre-event window. The
            # window start already carries the 60/120-minute lead time.
            return "none"

        log.info(
            "news approaching an open position",
            extra={
                "event": "news_position_action",
                "ticket": position.ticket,
                "symbol": position.symbol,
                "action": self.config.open_position_action,
                "news_event": imminent.event.title,
                "news_at": imminent.event.when.isoformat(),
            },
        )
        return self.config.open_position_action

    # -- introspection -----------------------------------------------------

    def next_blackout(self, currencies: frozenset[str], now: datetime) -> Blackout | None:
        """The next blackout window that has not yet started."""
        try:
            blackouts = self._blackouts(currencies)
        except CalendarUnavailableError:
            return None
        future = [b for b in blackouts if b.start > now]
        return min(future, key=lambda b: b.start) if future else None

    def blackouts_for(self, currency_base: str, currency_profit: str) -> list[Blackout]:
        """Every window that applies to an instrument. Used by the verifier."""
        return self._blackouts(symbol_currencies(currency_base, currency_profit))

    # -- internals ---------------------------------------------------------

    def _blackouts(self, currencies: frozenset[str]) -> list[Blackout]:
        """Build the windows for the relevant currencies.

        Only HIGH impact produces a window. Medium and low releases are noisy
        but not dangerous, and blocking on them would leave almost no tradable
        time in a normal week.
        """
        windows: list[Blackout] = []
        for event in self.calendar.events():
            if event.impact < Impact.HIGH or not event.affects(currencies):
                continue
            extreme = event.is_extreme(self.config.extreme_keywords)
            window = self.config.extreme_impact if extreme else self.config.high_impact
            windows.append(
                Blackout(
                    event=event,
                    start=event.when - timedelta(minutes=window.minutes_before),
                    end=event.when + timedelta(minutes=window.minutes_after),
                    extreme=extreme,
                )
            )
        return sorted(windows, key=lambda b: b.start)
