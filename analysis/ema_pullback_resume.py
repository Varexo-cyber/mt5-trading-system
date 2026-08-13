"""Re-enter an established M5 EMA trend after a closed-bar pullback and reclaim."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import EmaPullbackResumeConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class EmaPullbackResume:
    """A discrete M5 reclaim after a shallow pullback inside an EMA trend."""

    name = "ema_pullback_resume"

    def __init__(self, config: EmaPullbackResumeConfig | None = None) -> None:
        self.config = config or EmaPullbackResumeConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "EMA pullback resume disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = max(config.slow_ema * 3, config.atr_period + config.pullback_bars + 2)
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        close = frame["close"].astype(float)
        fast = close.ewm(span=config.fast_ema, adjust=False).mean()
        slow = close.ewm(span=config.slow_ema, adjust=False).mean()
        atr = _atr(frame, config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")

        gap = float(fast.iloc[-1] - slow.iloc[-1])
        direction = 1 if gap > 0 else -1 if gap < 0 else 0
        separation = abs(gap) / atr
        if direction == 0 or separation < config.minimum_separation_atr:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} EMA separation {separation:.2f} ATR is below "
                f"the {config.minimum_separation_atr:.2f} trend floor",
            )

        slope = float(slow.iloc[-1] - slow.iloc[-1 - config.slope_bars]) * direction / atr
        if slope < config.minimum_slope_atr:
            return Signal.neutral(
                self.name,
                f"EMA{config.slow_ema} slope {slope:.2f} ATR does not confirm the trend",
            )

        latest = frame.iloc[-1]
        previous_close = float(close.iloc[-2])
        previous_fast = float(fast.iloc[-2])
        latest_close = float(latest["close"])
        latest_open = float(latest["open"])
        latest_fast = float(fast.iloc[-1])

        # This makes the signal an event. Once price remains beyond EMA9, the
        # module goes silent until a new pullback and reclaim occur.
        was_in_pullback = (previous_close - previous_fast) * direction <= 0
        resumed = (latest_close - latest_fast) * direction > 0
        body_confirms = (latest_close - latest_open) * direction > 0
        if not (was_in_pullback and resumed and body_confirms):
            return Signal.neutral(
                self.name,
                f"{timeframe.value} trend exists but no fresh EMA reclaim closed on this bar",
            )

        pullback = frame.iloc[-(config.pullback_bars + 1) : -1]
        pullback_fast = fast.iloc[-(config.pullback_bars + 1) : -1]
        pullback_slow = slow.iloc[-(config.pullback_bars + 1) : -1]
        if direction > 0:
            touched = bool((pullback["low"].to_numpy() <= pullback_fast.to_numpy()).any())
            deepest = float((pullback_slow - pullback["close"]).max()) / atr
            extreme = float(pullback["low"].min())
            invalidation = min(extreme, float(slow.iloc[-1]) - atr * config.stop_buffer_atr)
        else:
            touched = bool((pullback["high"].to_numpy() >= pullback_fast.to_numpy()).any())
            deepest = float((pullback["close"] - pullback_slow).max()) / atr
            extreme = float(pullback["high"].max())
            invalidation = max(extreme, float(slow.iloc[-1]) + atr * config.stop_buffer_atr)
        if not touched:
            return Signal.neutral(self.name, "no pullback touched the fast EMA before the reclaim")
        if deepest > config.maximum_slow_ema_breach_atr:
            return Signal.neutral(
                self.name,
                f"pullback closed {deepest:.2f} ATR through EMA{config.slow_ema}; "
                "this is a possible reversal, not a shallow pullback",
            )

        m1_drift = self._m1_drift(ctx, direction, config)
        if m1_drift is not None and m1_drift < -config.maximum_m1_adverse_atr:
            return Signal.neutral(
                self.name,
                f"M1 drift {m1_drift:.2f} ATR still opposes the M5 reclaim",
            )

        reclaim_body = abs(latest_close - latest_open) / atr
        confidence = min(
            config.maximum_confidence,
            config.base_confidence
            + separation * config.separation_confidence_scale
            + min(reclaim_body, 1.0) * config.reclaim_confidence_scale,
        )
        return Signal(
            module=self.name,
            score=config.score * direction,
            confidence=confidence,
            reasoning=(
                f"{timeframe.value} EMA{config.fast_ema}/{config.slow_ema} trend resumed "
                f"after a shallow pullback; separation {separation:.2f} ATR, "
                f"reclaim body {reclaim_body:.2f} ATR"
            ),
            invalidation_price=invalidation,
            details={
                "timeframe": timeframe.value,
                "setup": "ema_pullback_resume",
                "separation_atr": round(separation, 2),
                "slow_ema_slope_atr": round(slope, 2),
                "pullback_slow_breach_atr": round(max(0.0, deepest), 2),
                "reclaim_body_atr": round(reclaim_body, 2),
                "m1_drift_atr": None if m1_drift is None else round(m1_drift, 2),
                "atr": atr,
            },
        )

    @staticmethod
    def _m1_drift(
        ctx: MarketContext, direction: int, config: EmaPullbackResumeConfig
    ) -> float | None:
        series = ctx.series.get(Timeframe.M1)
        if series is None or len(series.df) < max(config.m1_confirmation_bars + 1, 16):
            return None
        frame = series.df
        atr = _atr(frame, config.atr_period)
        if atr <= 0:
            return None
        start = frame["close"].iloc[-1 - config.m1_confirmation_bars]
        drift = float(frame["close"].iloc[-1] - start)
        return drift * direction / atr
