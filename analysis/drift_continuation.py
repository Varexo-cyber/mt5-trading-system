"""Join a move that is already happening, in the direction it is happening.

WHAT WAS MISSING, and it took a live day to see it. GBPUSD fell steadily for
most of 12 August. The engine spent that day trying to buy it — 233 M15 and 111
M5 refusals reading "price is moving against the long" — and never once
proposed a short. Not because a gate refused one, but because no module was
looking for one:

    trend_momentum   20/50 EMAs on H4 and H1. After a market tops out those
                     stay crossed the old way for hours, so it goes quiet long
                     before it turns, and it turns long after the move began.
    liquidity_sweep  needs a specific wick through a 20-bar extreme on the very
                     last candle. Rare by construction, and it is a reversal
                     pattern, not a continuation one.
    market_structure needs a break of structure.

Between the slow module and the rare one there is a hole, and an hour of clean
one-way drift sits exactly in it. The account watched a move it had measured
344 times and had no way to express.

WHY THIS IS NOT "THE REFUSED LONG, FLIPPED". That would be inventing a signal
out of a gate, and it is how a system gets sawn apart in a range: refuse the
buy, sell, refuse the sell, buy, paying the spread each way. This measures
something positive instead — that price has actually travelled, and travelled
consistently — and it says nothing at all when it has not.

THREE CONDITIONS, and the second and third are what keep it out of chop:

  1. Distance. Net close-to-close movement over the window, in ATR. A market
     that has drifted a tenth of an ATR has not moved, it has breathed.
  2. Consistency. Most bars in the window must close in the same direction. A
     market that ends the hour lower having gone up, down, up and down has the
     same net drift as one that ground steadily lower, and they are not the
     same thing. The live exit note put it as "4 of 5 closing adverse".
  3. Not a range. This is a continuation module, so it is registered as one in
     `trend_continuation_modules` and the confluence engine refuses it outright
     when the regime classifier measures a range.

Everything downstream still applies, and one gate in particular matters here:
`entry_quality` refuses an entry sitting at the extreme of its recent range.
That is what stops this module selling the low of the move it just noticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import DriftContinuationConfig
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


class DriftContinuation:
    """Score a sustained, consistent one-way move on the signal timeframe."""

    name = "drift_continuation"

    def __init__(self, config: DriftContinuationConfig | None = None) -> None:
        self.config = config or DriftContinuationConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        if not self.config.enabled:
            return Signal.neutral(self.name, "drift continuation disabled")
        timeframe = Timeframe.parse(self.config.timeframe)
        series = ctx.series.get(timeframe)
        needed = max(self.config.lookback_bars + 1, self.config.atr_period + 1)
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        window = frame.iloc[-self.config.lookback_bars :]
        atr = _atr(frame, self.config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")

        drift = float(window["close"].iloc[-1]) - float(frame["close"].iloc[-1 - len(window)])
        travelled = abs(drift) / atr
        if travelled < self.config.minimum_drift_atr:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} drifted {travelled:.2f} ATR over {len(window)} bars, "
                f"below the {self.config.minimum_drift_atr:.2f} floor",
            )

        direction = 1 if drift > 0 else -1
        # Consistency, and this is the condition that separates a trend from a
        # market that ended the hour lower having gone up, down, up and down.
        # Both have the same net drift; only one of them is going somewhere.
        steps = window["close"].diff().dropna()
        agreeing = int((steps * direction > 0).sum())
        consistency = agreeing / len(steps) if len(steps) else 0.0
        if consistency < self.config.minimum_consistency:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} moved {travelled:.2f} ATR but only {consistency:.0%} of "
                f"bars closed with it, below the {self.config.minimum_consistency:.0%} floor "
                f"— this is chop with a net, not a trend",
            )

        # Confidence rides on both terms, because they fail in different ways:
        # a big move nobody sustained is a spike, and a perfectly consistent
        # move of no size is noise with a tidy shape.
        reach = min(1.0, travelled / max(self.config.confident_drift_atr, 1e-9))
        confidence = min(
            self.config.maximum_confidence,
            self.config.base_confidence + (reach * consistency) * self.config.scale_confidence_by,
        )
        # The move is invalidated where it started: price returning through the
        # far end of the window means the drift this is built on is gone.
        invalidation = float(window["low"].min() if direction > 0 else window["high"].max())
        return Signal(
            module=self.name,
            score=self.config.score * direction,
            confidence=confidence,
            reasoning=(
                f"{timeframe.value} has travelled {travelled:.2f} ATR "
                f"{'up' if direction > 0 else 'down'} over {len(window)} bars with "
                f"{consistency:.0%} of them closing that way"
            ),
            invalidation_price=invalidation,
            details={
                "timeframe": timeframe.value,
                "lookback_bars": len(window),
                "drift_atr": round(travelled, 2),
                "consistency": round(consistency, 2),
                "atr": atr,
            },
        )
