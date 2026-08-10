"""Liveliness gate: refuse to trade a market that has stopped moving.

Every target in this system is priced in ATR, and ATR is a statement about the
recent past. When a market goes quiet — the lunch lull, a holiday session, the
hour before a central bank speaks and nobody wants to be the one holding the
bag — the target stays where the maths put it while the market stops producing
the movement needed to reach it. The setup still looks valid on the chart. The
stop is still the same distance away. Only one thing changed, and it is the
one thing that decides whether the trade can win.

What that produces is not a loss, it is worse: a position that sits there
paying the spread and the swap, drifting inside the noise band until either
the time exit closes it flat-minus-costs or the wind-down does. That is a
trade that was never a trade.

The measurement is deliberately relative to the instrument's *own* recent
behaviour rather than an absolute pip figure. Gold moves twenty times as much
as EURCHF in a normal hour; the only meaningful question is whether this
market is moving like itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from config.schema import LivelinessFilterConfig
from core.types import Series, Timeframe
from filters.base import Filter, FilterContext, FilterVerdict
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)

#: (symbol, timeframe) -> Series. Injected for the same reason the correlation
#: filter injects it: this gate never touches the connector, so it replays.
SeriesProvider = Callable[[str, Timeframe], Series]


class LivelinessFilter(Filter):
    """Blocks entries into a market that has gone to sleep."""

    name = "liveliness"

    def __init__(self, config: LivelinessFilterConfig, series_provider: SeriesProvider) -> None:
        self.config = config
        self.series_provider = series_provider
        self.timeframe = Timeframe.parse(config.timeframe)

    # -- maths -------------------------------------------------------------

    @staticmethod
    def _true_ranges(series: Series) -> np.ndarray:
        df = series.df
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        prev_close = df["close"].shift(1).to_numpy()
        return np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )[1:]

    def activity(self, symbol: str) -> float | None:
        """How fast this market is moving, as a fraction of its own normal.

        1.0 means the recent bars are ranging exactly like the typical bar of
        the baseline window; 0.4 means the market is running at 40% speed.

        The baseline is a *median*, not a mean. One news spike inside the
        lookback would drag a mean up far enough to make every subsequent
        normal hour read as quiet, and the gate would then block hardest right
        after the most tradeable moment of the day.

        Returns None when it cannot be measured — too little history, or a flat
        baseline. The caller decides what to do with that.
        """
        series = self.series_provider(symbol, self.timeframe)
        ranges = self._true_ranges(series)
        if len(ranges) < self.config.min_bars:
            return None

        recent = float(np.mean(ranges[-self.config.recent_bars :]))
        baseline = float(np.median(ranges[-self.config.baseline_bars :]))
        if baseline <= 0.0:
            return None
        return recent / baseline

    def execution_quality(self, series: Series) -> tuple[float, float] | None:
        """Return recent sparse-gap and flat-bar fractions.

        ATR cannot distinguish a liquid trend from one discontinuous jump. The
        timestamp index and the bar ranges can: a tradeable tape produces bars
        at roughly its nominal cadence and most of those bars contain an actual
        auction between more than one price.
        """
        df = series.df.tail(self.config.quality_bars)
        if len(df) < self.config.quality_bars:
            return None
        expected_seconds = self.timeframe.duration.total_seconds()
        gaps = df.index.to_series().diff().dropna().dt.total_seconds()
        sparse = float((gaps > expected_seconds * 1.5).mean()) if len(gaps) else 0.0
        flat = float(((df["high"] - df["low"]).abs() <= 0.0).mean())
        return sparse, flat

    # -- gate --------------------------------------------------------------

    def check(self, ctx: FilterContext) -> FilterVerdict:
        if not self.config.enabled:
            return FilterVerdict.allow(self.name, "liveliness filter disabled")

        try:
            series = self.series_provider(ctx.symbol, self.timeframe)
            measured = self.activity(ctx.symbol)
            quality = self.execution_quality(series)
        except Exception as exc:  # noqa: BLE001 - any data failure means unknown
            log.warning(
                "cannot measure liveliness",
                extra={
                    "event": "liveliness_data_missing",
                    "symbol": ctx.symbol,
                    "reason": str(exc),
                },
            )
            return FilterVerdict.block(
                self.name,
                Reason.DATA_UNAVAILABLE,
                f"cannot read {self.timeframe.value} bars for {ctx.symbol} to tell "
                f"whether the market is moving; unknown is not the same as fine",
                activity_ratio=None,
            )

        if measured is None:
            # Not enough history yet, or a baseline of exactly zero. This is a
            # gap in what we know rather than evidence of the pathology, and
            # the freshness and spread gates already stand between us and bad
            # data. Blocking here instead would turn a cold start into a silent
            # no-trade day, which is the failure mode this system has already
            # paid for once.
            return FilterVerdict.allow(
                self.name,
                f"not enough {self.timeframe.value} history to judge liveliness; "
                f"deferring to the freshness and spread gates",
                activity_ratio=None,
            )

        if quality is not None:
            sparse, flat = quality
            if sparse > self.config.max_sparse_gap_fraction:
                return FilterVerdict.block(
                    self.name,
                    Reason.MARKET_TOO_QUIET,
                    f"{ctx.symbol} has missing intervals in {sparse:.0%} of its recent "
                    f"{self.timeframe.value} tape (limit "
                    f"{self.config.max_sparse_gap_fraction:.0%}); higher-timeframe "
                    "structure is not reliable enough to execute on this stuttering feed",
                    activity_ratio=round(measured, 3),
                    sparse_gap_fraction=round(sparse, 3),
                    flat_bar_fraction=round(flat, 3),
                )
            if flat > self.config.max_flat_bar_fraction:
                return FilterVerdict.block(
                    self.name,
                    Reason.MARKET_TOO_QUIET,
                    f"{ctx.symbol} printed no intrabar range in {flat:.0%} of its recent "
                    f"{self.timeframe.value} bars (limit "
                    f"{self.config.max_flat_bar_fraction:.0%}); the chart is too thin "
                    "for a dependable entry and exit",
                    activity_ratio=round(measured, 3),
                    sparse_gap_fraction=round(sparse, 3),
                    flat_bar_fraction=round(flat, 3),
                )

        floor = self.config.min_activity_ratio
        if measured < floor:
            return FilterVerdict.block(
                self.name,
                Reason.MARKET_TOO_QUIET,
                f"{ctx.symbol} is ranging at {measured:.0%} of its own recent normal "
                f"(floor {floor:.0%}); the target is priced off a market that has "
                f"stopped producing the movement needed to reach it",
                activity_ratio=round(measured, 3),
                activity_floor=floor,
            )

        return FilterVerdict.allow(
            self.name,
            f"moving at {measured:.0%} of its recent normal",
            activity_ratio=round(measured, 3),
            activity_floor=floor,
            sparse_gap_fraction=(round(quality[0], 3) if quality is not None else None),
            flat_bar_fraction=(round(quality[1], 3) if quality is not None else None),
        )
