"""Stand aside while something is happening that nobody scheduled.

`NewsFilter` above this one knows what is coming: it reads a calendar and
refuses to enter around a release whose time was published in advance. It is
blind to everything else. A central bank moving between meetings, a
geopolitical shock, a story that breaks at 03:00 — all of it arrives with the
calendar showing a clear afternoon, and the account walks straight into it.

This closes that. It asks one question and it is not "is this news good or bad
for the trade": it is *is an unusual amount being written about this instrument
right now*. That is a fact available at retail latency. Direction is not — by
the time a story reaches a public feed the move it describes has happened, and
`filters.newsfeed.items.NewsPressure` sets out at length why no sentiment score
appears anywhere in this package.

So the rule is: an instrument running well above its own normal news rate is
not an instrument to open a new position on. Positions already open are not
touched here — the guard reads the same pressure and the reviewer is handed the
actual headlines, both of which are better placed to judge a live trade than a
gate that only knows how to say no.
"""

from __future__ import annotations

from typing import Any

from config.schema import HeadlineFilterConfig
from filters.base import Filter, FilterContext, FilterVerdict
from filters.calendar.events import symbol_currencies
from filters.newsfeed.service import HeadlineService
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)


class HeadlineFilter(Filter):
    """Refuses entry into an instrument that is unusually busy in the news."""

    name = "headlines"

    def __init__(
        self,
        config: HeadlineFilterConfig,
        service: HeadlineService,
        brain: Any | None = None,
    ) -> None:
        self.config = config
        self.service = service
        # Optional and fail-soft, like everything else that touches it. Wire
        # copy is kept past the few hours a feed carries so that "what was
        # being written when this trade opened" stays answerable a year later
        # — which is the only way the link between news and outcome can ever
        # be measured.
        self.brain = brain

    def check(self, ctx: FilterContext) -> FilterVerdict:
        if not self.config.enabled:
            return FilterVerdict.allow(self.name, "headline filter disabled")

        # Cheap and idempotent: returns immediately unless the interval has
        # elapsed. Driving it from the filter rather than from a separate task
        # keeps the fetch on the same thread as the decision that needs it, so
        # there is no window where the two disagree about what is held.
        if self.service.refresh() and self.brain is not None:
            self.brain.record_headlines(self.service.newest())

        if not self.service.is_usable():
            return self._unavailable()

        currencies = symbol_currencies(
            ctx.spec.currency_base,
            ctx.spec.currency_profit,
            getattr(getattr(ctx.spec, "asset_class", None), "value", None),
        )
        pressure = self.service.pressure(ctx.symbol, currencies)
        data = {
            "headline_count": pressure.recent,
            "headline_baseline": round(pressure.baseline, 2),
            "headline_multiple": round(pressure.multiple, 2),
            "headline_systemic": pressure.systemic,
            "headline_feeds": list(self.service.sources),
        }

        # A market-wide story is its own reason and does not need to clear the
        # per-instrument spike test. "Risk assets sell off as war escalates"
        # touches every pair equally, so it never looks like a spike on any one
        # of them, and the spike test would let all of them through.
        if pressure.systemic and self.config.block_on_systemic:
            return FilterVerdict.block(
                self.name,
                Reason.HEADLINE_PRESSURE,
                f"a market-wide story is running: {pressure.describe()}",
                **data,
            )

        loud = pressure.recent >= self.config.min_headlines
        spiking = pressure.multiple >= self.config.spike_multiple
        if loud and spiking:
            return FilterVerdict.block(
                self.name,
                Reason.HEADLINE_PRESSURE,
                f"unusual news flow: {pressure.describe()}",
                **data,
            )

        return FilterVerdict.allow(self.name, pressure.describe(), **data)

    def _unavailable(self) -> FilterVerdict:
        """No usable window. Whether that stops trading is a config decision.

        Defaulting to open is a departure from the operator's "no data, no
        trade" rule and a deliberate one. That rule protects against a missing
        *calendar*, and the calendar still fails closed on its own — nothing
        here weakens it. With this layer dark the system is back to the safety
        level it ran at before the layer existed, which is the level it has run
        at all along. Failing closed here would let one flaky RSS host stop a
        day of trading.
        """
        age = self.service.age
        detail = (
            "no headline feed has answered yet"
            if age is None
            else f"headlines are {age.total_seconds() / 60.0:.0f} min old"
        )
        if self.config.block_when_unavailable:
            return FilterVerdict.block(
                self.name,
                Reason.HEADLINES_UNAVAILABLE,
                f"{detail}; configured to stand down without them",
                headline_feeds=list(self.service.sources),
            )
        log.info(
            "headline layer dark; the calendar still governs",
            extra={"event": "headlines_unavailable", "detail": detail},
        )
        return FilterVerdict.allow(self.name, f"{detail}; not blocking on it")
