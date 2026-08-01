"""Spread gate, with a baseline the system learns for itself.

A fixed "block above 2 pips" threshold is wrong twice over: too loose for
EURUSD at 09:00, far too tight for GBPJPY at 22:00. So the filter builds a
per-instrument, per-hour median from its own observations and blocks when the
current spread exceeds a multiple of that.

Learning here is safe under the Layer-1 rule from `PLAN.md`: the baseline
adapts to *context* (which instrument, which hour), never to *performance*.
It cannot overfit to recent wins, because it never sees them.

Until enough observations exist for an hour, the configured absolute ceiling in
pips applies. That fallback is deliberately conservative — a wrong baseline
from twelve samples is worse than a blunt constant.

The median is used rather than the mean because spread distributions have a
long right tail: one 40-pip rollover print drags a mean far enough to let
genuinely bad spreads through for the rest of the day.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta

from config.schema import SpreadFilterConfig
from core.clock import Clock
from filters.base import Filter, FilterContext, FilterVerdict
from infra.logging import get_logger
from journal.database import Journal
from risk.reasons import Reason

log = get_logger(__name__)

#: Do not store more than one observation per symbol per this interval. At a
#: 30-second loop that would otherwise be 2 880 rows per symbol per day, almost
#: all of them redundant.
MIN_OBSERVATION_INTERVAL = timedelta(minutes=1)


class SpreadFilter(Filter):
    """Blocks entry when the cost of crossing the spread is abnormal."""

    name = "spread"

    def __init__(
        self,
        config: SpreadFilterConfig,
        journal: Journal,
        clock: Clock,
        *,
        retention_days: int = 60,
    ) -> None:
        self.config = config
        self.journal = journal
        self.clock = clock
        self.retention = timedelta(days=retention_days)
        self._last_recorded: dict[str, datetime] = {}
        self._last_pruned: datetime | None = None

    # -- observation -------------------------------------------------------

    def observe(self, symbol: str, spread_pips: float, moment: datetime | None = None) -> bool:
        """Record a spread sample. Returns True if it was stored.

        Called every cycle regardless of whether we intend to trade — the
        baseline has to describe the whole day, not only the hours in which a
        setup happened to appear.
        """
        now = moment or self.clock.now()
        last = self._last_recorded.get(symbol)
        if last is not None and now - last < MIN_OBSERVATION_INTERVAL:
            return False
        if spread_pips < 0:
            return False

        self.journal.record_spread(symbol, spread_pips, now)
        self._last_recorded[symbol] = now
        self._maybe_prune(now)
        return True

    def _maybe_prune(self, now: datetime) -> None:
        """Drop observations past the retention window, at most once a day."""
        if self._last_pruned is not None and now - self._last_pruned < timedelta(days=1):
            return
        removed = self.journal.prune_spread_observations(now - self.retention)
        self._last_pruned = now
        if removed:
            log.debug(
                "pruned spread observations",
                extra={"event": "spread_prune", "removed": removed},
            )

    # -- baseline ----------------------------------------------------------

    def baseline(self, symbol: str, hour_utc: int) -> tuple[float | None, int]:
        """Median spread for this symbol at this hour, and the sample count.

        Returns `(None, n)` while `n` is below `min_observations`, which is the
        caller's signal to use the absolute fallback instead.
        """
        samples = self.journal.spread_samples(symbol, hour_utc)
        if len(samples) < self.config.min_observations:
            return None, len(samples)
        return statistics.median(samples), len(samples)

    def ceiling(self, symbol: str, hour_utc: int) -> tuple[float, str, int]:
        """Maximum acceptable spread in pips, plus how it was derived."""
        median, count = self.baseline(symbol, hour_utc)
        if median is not None:
            return median * self.config.max_spread_multiple, "learned", count

        fallback = self.config.absolute_max_pips.get(_bare(symbol))
        if fallback is None:
            # No baseline and no configured ceiling. Refuse rather than invent
            # a number: an unknown instrument's normal spread is unknowable.
            return 0.0, "unknown", count
        return fallback, "fallback", count

    # -- gate --------------------------------------------------------------

    def check(self, ctx: FilterContext) -> FilterVerdict:
        if ctx.tick is None:
            return FilterVerdict.block(
                self.name,
                Reason.SPREAD_TOO_WIDE,
                "no tick available; cannot measure the spread, so cannot clear it",
            )

        spread_pips = ctx.spec.price_to_pips(ctx.tick.spread)
        hour = ctx.now.hour
        self.observe(ctx.symbol, spread_pips, ctx.now)

        if not self.config.enabled:
            return FilterVerdict.allow(
                self.name, "spread filter disabled", spread_pips=round(spread_pips, 3)
            )

        limit, source, count = self.ceiling(ctx.symbol, hour)
        data = {
            "spread_pips": round(spread_pips, 3),
            "spread_limit_pips": round(limit, 3),
            "spread_baseline_source": source,
            "spread_samples": count,
        }

        if source == "unknown":
            return FilterVerdict.block(
                self.name,
                Reason.SPREAD_TOO_WIDE,
                f"{ctx.symbol}: only {count} spread observations for hour {hour:02d} "
                f"(need {self.config.min_observations}) and no entry in "
                f"filters.spread.absolute_max_pips. Refusing rather than guessing.",
                **data,
            )

        if spread_pips > limit:
            return FilterVerdict.block(
                self.name,
                Reason.SPREAD_TOO_WIDE,
                f"spread {spread_pips:.2f} pips exceeds the {limit:.2f} pip limit for "
                f"hour {hour:02d} ({source} baseline from {count} samples)",
                **data,
            )

        return FilterVerdict.allow(
            self.name,
            f"spread {spread_pips:.2f} of {limit:.2f} pips allowed ({source})",
            **data,
        )


def _bare(symbol: str) -> str:
    """Strip a broker suffix so `EURUSD.pro` finds the `EURUSD` fallback."""
    for separator in (".", "_", "-"):
        if separator in symbol:
            return symbol.split(separator, 1)[0]
    # Trailing single-letter suffixes such as `EURUSDm`.
    if len(symbol) > 6 and symbol[:6].isalpha() and symbol[6:].isalpha():
        return symbol[:6]
    return symbol
