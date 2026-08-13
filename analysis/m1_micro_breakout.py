"""Detect a closed-M1 range break aligned with measured M5 structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import M1MicroBreakoutConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    previous = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(ranges.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class M1MicroBreakout:
    """A fresh one-minute break that borrows direction, not permission, from M5."""

    name = "m1_micro_breakout"

    def __init__(self, config: M1MicroBreakoutConfig | None = None) -> None:
        self.config = config or M1MicroBreakoutConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "M1 micro breakout disabled")
        m1 = ctx.series.get(Timeframe.M1)
        m5 = ctx.series.get(Timeframe.M5)
        needed_m1 = max(config.atr_period + config.base_bars + 2, config.volume_lookback + 2)
        needed_m5 = max(config.m5_slow_ema * 3, config.atr_period + config.m5_slope_bars + 1)
        if m1 is None or len(m1.df) < needed_m1:
            return Signal.neutral(self.name, f"needs {needed_m1} closed M1 bars")
        if m5 is None or len(m5.df) < needed_m5:
            return Signal.neutral(self.name, f"needs {needed_m5} closed M5 bars")

        m5_frame = m5.df
        m5_close = m5_frame["close"].astype(float)
        fast = m5_close.ewm(span=config.m5_fast_ema, adjust=False).mean()
        slow = m5_close.ewm(span=config.m5_slow_ema, adjust=False).mean()
        m5_atr = _atr(m5_frame, config.atr_period)
        if m5_atr <= 0:
            return Signal.neutral(self.name, "no usable M5 ATR")
        gap = float(fast.iloc[-1] - slow.iloc[-1])
        direction = 1 if gap > 0 else -1 if gap < 0 else 0
        separation = abs(gap) / m5_atr
        slope = float(slow.iloc[-1] - slow.iloc[-1 - config.m5_slope_bars])
        aligned_slope = slope * direction / m5_atr if direction else 0.0
        if (
            direction == 0
            or separation < config.minimum_m5_separation_atr
            or aligned_slope < config.minimum_m5_slope_atr
        ):
            return Signal.neutral(
                self.name,
                f"M5 structure is not directional enough: separation {separation:.2f} ATR, "
                f"slope {aligned_slope:.2f} ATR",
            )

        frame = m1.df
        m1_atr = _atr(frame, config.atr_period)
        if m1_atr <= 0:
            return Signal.neutral(self.name, "no usable M1 ATR")
        base = frame.iloc[-(config.base_bars + 1) : -1]
        base_high = float(base["high"].max())
        base_low = float(base["low"].min())
        base_width = (base_high - base_low) / m1_atr
        if base_width > config.maximum_base_width_atr:
            return Signal.neutral(
                self.name,
                f"M1 base width {base_width:.2f} ATR exceeds {config.maximum_base_width_atr:.2f}",
            )

        latest = frame.iloc[-1]
        previous_close = float(frame["close"].iloc[-2])
        close = float(latest["close"])
        open_ = float(latest["open"])
        high = float(latest["high"])
        low = float(latest["low"])
        threshold = base_high if direction > 0 else base_low
        broke = (close - threshold) * direction > config.minimum_break_atr * m1_atr
        was_inside = (previous_close - threshold) * direction <= 0
        if not (broke and was_inside):
            return Signal.neutral(self.name, "no fresh closed M1 break from the micro-range")

        candle_range = high - low
        body = abs(close - open_) / m1_atr
        close_location = (close - low) / candle_range if candle_range > 0 else 0.5
        directional_location = close_location if direction > 0 else 1.0 - close_location
        if (close - open_) * direction <= 0 or body < config.minimum_body_atr:
            return Signal.neutral(self.name, f"M1 breakout body {body:.2f} ATR is too weak")
        if directional_location < config.minimum_close_location:
            return Signal.neutral(
                self.name,
                f"M1 breakout closed only {directional_location:.0%} into its direction",
            )
        median_volume = float(
            frame["tick_volume"].iloc[-(config.volume_lookback + 1) : -1].median()
        )
        volume_ratio = float(latest["tick_volume"]) / max(1.0, median_volume)
        if volume_ratio < config.minimum_volume_ratio:
            return Signal.neutral(
                self.name,
                f"M1 breakout volume {volume_ratio:.2f}x is below participation floor",
            )

        buffer = config.stop_buffer_atr * m1_atr
        invalidation = base_low - buffer if direction > 0 else base_high + buffer
        confidence = min(
            config.maximum_confidence,
            config.base_confidence
            + min(body, 1.5) * config.body_confidence_scale
            + min(max(0.0, volume_ratio - 1.0), 2.0) * config.volume_confidence_scale,
        )
        return Signal(
            module=self.name,
            score=config.score * direction,
            confidence=confidence,
            reasoning=(
                f"closed M1 micro-range break aligned with M5 EMA trend; "
                f"body {body:.2f} ATR, volume {volume_ratio:.2f}x, "
                f"M5 separation {separation:.2f} ATR"
            ),
            invalidation_price=invalidation,
            details={
                "timeframe": "M1",
                "setup": self.name,
                "m1_body_atr": round(body, 2),
                "m1_volume_ratio": round(volume_ratio, 2),
                "m1_base_width_atr": round(base_width, 2),
                "m5_separation_atr": round(separation, 2),
                "atr": m1_atr,
            },
        )
