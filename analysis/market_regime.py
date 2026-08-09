"""Non-directional closed-bar regime classification for conditional scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import MarketRegimeConfig
from core.types import MarketContext, Signal, Timeframe


class MarketRegime:
    """Classify persistence and volatility without inventing a trade signal."""

    name = "market_regime"

    def __init__(self, config: MarketRegimeConfig) -> None:
        self.config = config
        self.fast_timeframe = Timeframe.parse(config.fast_timeframe)
        self.slow_timeframe = Timeframe.parse(config.slow_timeframe)

    def analyze(self, ctx: MarketContext) -> Signal:
        if not self.config.enabled:
            return Signal.neutral(self.name, "market-regime classifier disabled")
        fast = ctx.series.get(self.fast_timeframe)
        slow = ctx.series.get(self.slow_timeframe)
        if fast is None or slow is None:
            return Signal.neutral(
                self.name,
                f"needs {self.fast_timeframe.value} and {self.slow_timeframe.value} history",
            )
        fast_required = max(
            self.config.efficiency_lookback + 1,
            self.config.atr_period + self.config.atr_percentile_lookback + 1,
        )
        slow_required = self.config.efficiency_lookback + 1
        if len(fast.df) < fast_required or len(slow.df) < slow_required:
            return Signal.neutral(
                self.name,
                f"needs {fast_required} {self.fast_timeframe.value} and "
                f"{slow_required} {self.slow_timeframe.value} closed bars",
            )

        fast_er, fast_direction = self._efficiency(fast.df["close"])
        slow_er, slow_direction = self._efficiency(slow.df["close"])
        atr, atr_percentile = self._atr_percentile(fast.df)

        if atr_percentile >= self.config.extreme_atr_percentile:
            regime = "extreme"
        elif (
            fast_er >= self.config.trend_efficiency_min
            and slow_er >= self.config.trend_efficiency_min
            and fast_direction != 0
            and fast_direction == slow_direction
        ):
            regime = "trend_up" if fast_direction > 0 else "trend_down"
        elif (
            fast_er <= self.config.range_fast_efficiency_max
            and slow_er <= self.config.range_slow_efficiency_max
        ):
            regime = "range"
        else:
            regime = "transition"

        return Signal(
            module=self.name,
            score=0.0,
            confidence=1.0,
            reasoning=f"closed-bar market regime is {regime}",
            details={
                "regime": regime,
                "fast_timeframe": self.fast_timeframe.value,
                "slow_timeframe": self.slow_timeframe.value,
                "fast_efficiency": fast_er,
                "slow_efficiency": slow_er,
                "fast_direction": fast_direction,
                "slow_direction": slow_direction,
                "atr": atr,
                "atr_percentile": atr_percentile,
            },
        )

    def _efficiency(self, close: pd.Series) -> tuple[float, int]:
        window = close.iloc[-(self.config.efficiency_lookback + 1) :].astype(float)
        change = float(window.iloc[-1] - window.iloc[0])
        path = float(window.diff().abs().sum())
        if not np.isfinite(path) or path <= 0.0:
            return 0.0, 0
        ratio = min(1.0, max(0.0, abs(change) / path))
        return ratio, 1 if change > 0 else -1 if change < 0 else 0

    def _atr_percentile(self, frame: pd.DataFrame) -> tuple[float, float]:
        previous = frame["close"].shift(1)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atrs = true_range.rolling(self.config.atr_period).mean().dropna()
        current = float(atrs.iloc[-1])
        reference = atrs.iloc[-(self.config.atr_percentile_lookback + 1) : -1]
        # Midrank prevents a flat volatility series from being labelled as the
        # 100th percentile merely because every observation ties the current.
        below = float((reference < current).mean())
        tied = float(np.isclose(reference.to_numpy(dtype=float), current).mean())
        return current, min(1.0, max(0.0, below + 0.5 * tied))
