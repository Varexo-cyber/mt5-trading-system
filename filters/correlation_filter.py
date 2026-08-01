"""Correlation gate: stop two positions from being one bet in disguise.

Long EURUSD and long GBPUSD is not two trades. It is one short-USD position at
double size, and if the dollar rallies both stops go together. The account's
real risk is roughly 2R, not the 1R each position was sized for.

Correlation is computed on a rolling window of log returns rather than taken
from a static table, because currency correlations break down precisely when
they matter — during a risk event, pairs that normally move together decouple,
and pairs that normally do not suddenly move as one. A number from 2019 is
worse than useless.

Direction matters as much as the coefficient:

* positively correlated + same direction  -> doubled exposure  -> block
* positively correlated + opposite direction -> hedged, cancels out -> allow
* negatively correlated + opposite direction -> doubled exposure  -> block
* negatively correlated + same direction -> hedged -> allow

Which reduces to one test: `correlation * direction_a * direction_b > threshold`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from config.schema import CorrelationFilterConfig
from core.types import Direction, Series, Timeframe
from filters.base import Filter, FilterContext, FilterVerdict
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)

#: (symbol, timeframe) -> Series. Injected so this filter never touches the
#: connector and stays replayable in a backtest.
SeriesProvider = Callable[[str, Timeframe], Series]


class CorrelationFilter(Filter):
    """Blocks a second position that would double an existing exposure."""

    name = "correlation"

    def __init__(self, config: CorrelationFilterConfig, series_provider: SeriesProvider) -> None:
        self.config = config
        self.series_provider = series_provider
        self.timeframe = Timeframe.parse(config.timeframe)

    # -- maths -------------------------------------------------------------

    def correlation(self, symbol_a: str, symbol_b: str) -> float | None:
        """Pearson correlation of log returns over the configured window.

        Returns None when the two series cannot be compared — too few bars, or
        no overlapping timestamps. None means "unknown", and the caller treats
        unknown as *blocking*: an unmeasurable correlation between two open
        positions is not evidence that they are independent.
        """
        try:
            series_a = self.series_provider(symbol_a, self.timeframe)
            series_b = self.series_provider(symbol_b, self.timeframe)
        except Exception as exc:  # noqa: BLE001 - any data failure means unknown
            log.warning(
                "cannot load series for correlation",
                extra={
                    "event": "correlation_data_missing",
                    "symbols": [symbol_a, symbol_b],
                    "reason": str(exc),
                },
            )
            return None

        lookback = self.config.lookback_bars
        closes_a = series_a.df["close"].tail(lookback)
        closes_b = series_b.df["close"].tail(lookback)

        # Align on bar time. Two symbols can have different holiday gaps, and
        # correlating misaligned rows produces a confident, meaningless number.
        aligned = closes_a.to_frame("a").join(closes_b.to_frame("b"), how="inner")
        if len(aligned) < max(30, lookback // 4):
            log.debug(
                "not enough overlapping bars for correlation",
                extra={"symbols": [symbol_a, symbol_b], "overlap": len(aligned)},
            )
            return None

        returns = np.diff(np.log(aligned.to_numpy()), axis=0)
        if returns.shape[0] < 2:
            return None
        # A flat series has zero variance; np.corrcoef would emit a warning and
        # return nan. Catch it here so the caller sees a clean "unknown".
        if float(returns[:, 0].std()) == 0.0 or float(returns[:, 1].std()) == 0.0:
            return None

        coefficient = float(np.corrcoef(returns[:, 0], returns[:, 1])[0, 1])
        return None if np.isnan(coefficient) else coefficient

    @staticmethod
    def doubles_exposure(correlation: float, a: Direction, b: Direction) -> float:
        """Signed exposure overlap. Positive means the two bets reinforce."""
        return correlation * int(a) * int(b)

    # -- gate --------------------------------------------------------------

    def check(self, ctx: FilterContext) -> FilterVerdict:
        if not self.config.enabled or not ctx.open_positions:
            return FilterVerdict.allow(
                self.name,
                (
                    "no open positions to correlate against"
                    if self.config.enabled
                    else "correlation filter disabled"
                ),
            )
        if ctx.direction is None:
            raise ValueError(
                "correlation filter needs the intended direction; doubled exposure "
                "is not defined without it"
            )

        threshold = self.config.max_abs_correlation
        measured: dict[str, float] = {}

        for position in ctx.open_positions:
            if position.symbol == ctx.symbol:
                # Same instrument twice is the risk manager's problem
                # (POSITION_ALREADY_OPEN); nothing to correlate.
                continue

            coefficient = self.correlation(ctx.symbol, position.symbol)
            if coefficient is None:
                return FilterVerdict.block(
                    self.name,
                    Reason.CORRELATED_EXPOSURE,
                    f"cannot measure correlation between {ctx.symbol} and open "
                    f"{position.symbol} (#{position.ticket}); unknown is not "
                    f"the same as independent",
                    correlations=measured,
                )

            measured[position.symbol] = round(coefficient, 3)
            overlap = self.doubles_exposure(coefficient, ctx.direction, position.direction)
            if overlap > threshold:
                return FilterVerdict.block(
                    self.name,
                    Reason.CORRELATED_EXPOSURE,
                    f"{ctx.direction.name} {ctx.symbol} against open "
                    f"{position.direction.name} {position.symbol} has an exposure "
                    f"overlap of {overlap:+.2f} (correlation {coefficient:+.2f}), above "
                    f"{threshold:.2f}. Taken together that is one position at roughly "
                    f"double size.",
                    correlations=measured,
                    exposure_overlap=round(overlap, 3),
                )

        return FilterVerdict.allow(
            self.name,
            f"exposure overlap acceptable against {len(measured)} open position(s)",
            correlations=measured,
        )
