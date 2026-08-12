"""The fast chart: a 9/20 EMA cross on M5, for entries measured in minutes.

Every directional module here reads slow charts. `trend_momentum` runs 20/50
EMAs on H4 and H1, `market_structure` wants a break of structure, and even
`liquidity_sweep` — the fastest of them — needs a specific wick on M15. Nothing
looks at the timeframe a day trade is actually taken on.

WHAT THIS ADDS AND WHY IT IS DANGEROUS. A 9/20 cross on a five-minute chart is
the classic quick-entry signal and it is also the classic way to be sawn to
pieces: the two averages brush against each other all day in a quiet market,
and each brush is a trade paying the spread. So the cross alone is not the
signal here. Three things have to hold together:

  1. **The cross is fresh.** Within the last few bars. A cross that happened
     forty minutes ago is a state, not an entry — and the state is what
     `trend_momentum` is already for.
  2. **The averages have actually separated.** Measured in ATR, so it means the
     same thing on gold as on EURUSD. Two EMAs sitting on top of each other
     crossing back and forth is one market making no decision, and reading each
     touch as a signal is reading noise at high frequency.
  3. **Price is on the right side of both.** A cross with price already back
     through the slow average is a cross that has failed in the time it took
     to print.

WHAT STILL SITS BEHIND IT. Registered in `trend_continuation_modules`, so the
confluence engine refuses it outright when the regime classifier measures a
range — the condition under which a fast cross is at its very worst. Classified
as an intraday setup, so the plan it produces is an M15 one with a target
measured over twelve bars, rather than a 24-hour swing target on a five-minute
signal. And `entry_quality` still refuses an entry sitting at the extreme of
its own recent range.

Enabled live at the owner's explicit request, ahead of the promotion protocol,
alongside the modules already in that position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import FastEmaCrossConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    high = frame["high"]
    low = frame["low"]
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class FastEmaCross:
    """A recent, separated 9/20 EMA cross on the fast chart."""

    name = "fast_ema_cross"

    def __init__(self, config: FastEmaCrossConfig | None = None) -> None:
        self.config = config or FastEmaCrossConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        if not self.config.enabled:
            return Signal.neutral(self.name, "fast EMA cross disabled")
        timeframe = Timeframe.parse(self.config.timeframe)
        series = ctx.series.get(timeframe)
        needed = max(self.config.slow_ema * 3, self.config.atr_period + 1)
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        close = frame["close"]
        fast = close.ewm(span=self.config.fast_ema, adjust=False).mean()
        slow = close.ewm(span=self.config.slow_ema, adjust=False).mean()
        gap = fast - slow
        side = np.sign(gap.to_numpy())
        now_side = int(side[-1])
        if now_side == 0:
            return Signal.neutral(self.name, f"{timeframe.value} EMAs are exactly level")

        # How long ago did the fast average take this side? Counting back
        # rather than testing the last bar, so a cross two bars old is still an
        # entry and a cross forty minutes old is not.
        bars_since = 0
        for value in reversed(side[:-1]):
            if int(value) == now_side:
                bars_since += 1
                continue
            break
        if bars_since > self.config.max_bars_since_cross:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} EMA{self.config.fast_ema}/{self.config.slow_ema} crossed "
                f"{bars_since} bars ago, past the {self.config.max_bars_since_cross}-bar "
                f"window — that is a state, not an entry",
            )

        atr = _atr(frame, self.config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")
        separation = abs(float(gap.iloc[-1])) / atr
        if separation < self.config.minimum_separation_atr:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} EMAs are {separation:.2f} ATR apart, below the "
                f"{self.config.minimum_separation_atr:.2f} floor — they are brushing, "
                f"not crossing",
            )

        # A cross with price already back through the slow average is a cross
        # that failed in the time it took to print.
        last = float(close.iloc[-1])
        if (last - float(slow.iloc[-1])) * now_side <= 0:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} crossed but price closed back through the "
                f"EMA{self.config.slow_ema}",
            )

        confidence = min(
            self.config.maximum_confidence,
            self.config.base_confidence + separation * self.config.separation_confidence_scale,
        )
        recent = frame.iloc[-self.config.invalidation_lookback :]
        invalidation = float(recent["low"].min() if now_side > 0 else recent["high"].max())
        return Signal(
            module=self.name,
            score=self.config.score * now_side,
            confidence=confidence,
            reasoning=(
                f"{timeframe.value} EMA{self.config.fast_ema} crossed "
                f"{'above' if now_side > 0 else 'below'} EMA{self.config.slow_ema} "
                f"{bars_since} bars ago and they are {separation:.2f} ATR apart"
            ),
            invalidation_price=invalidation,
            details={
                "timeframe": timeframe.value,
                "bars_since_cross": bars_since,
                "separation_atr": round(separation, 2),
                f"ema{self.config.fast_ema}": float(fast.iloc[-1]),
                f"ema{self.config.slow_ema}": float(slow.iloc[-1]),
                "atr": atr,
            },
        )
