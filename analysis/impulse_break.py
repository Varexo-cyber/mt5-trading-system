"""One violent bar with follow-through: the move no other module can see.

WHY THIS EXISTS, and it is a hole found in live data rather than reasoned into
being. A GBPCAD short on 13 August: the reviewer described the market as "a
large, fresh down-impulse — M15 last candle body -1.34 ATR, M5 3-bar drift
-2.3 ATR". A move that size and that recent, and not one directional module
fired on it. The only thing that spoke was `trend_momentum` reading an EMA
alignment, which is why the trade went out as a 24-hour swing off a slow
indicator instead of a three-hour intraday trade off the move that was actually
happening — and the reviewer refused it, correctly, for chasing.

Nothing was looking:

  - `drift_continuation` measures eight M15 bars and requires 65% of them to
    close with the move. One or two violent bars inside six flat ones gives
    about 25% consistency, so a genuine impulse fails its floor by design. It
    is built for a grind, not a jump.
  - `fast_ema_cross` needs the 9 and 20 EMAs to cross inside the last six M5
    bars. In a vertical move the cross is usually already behind you.
  - `market_structure` wants a confirmed break of a swing level, which arrives
    later and often not at all.
  - `liquidity_sweep` is a reversal pattern and points the other way.

THE MECHANISM, stated so it can be wrong. A bar that closes near its extreme
having travelled more than an ATR is a repricing that ran out of resting
liquidity — the participants who wanted to trade the old level are gone, and
the next marginal order has to reach further. That is a claim about the minutes
after the bar, not the day, which is why this is an intraday module with an
intraday plan.

HOW IT AVOIDS BUYING THE TOP, which is the obvious way this loses money. Four
floors, and each is checked below:

  1. **Size.** The body, not the range, above `minimum_body_atr`. A bar with a
     huge range and a small body is indecision wearing a big candle.
  2. **Conviction in the close.** The bar must close in the far third of its own
     range. A large body with a large opposing wick is a rejection, and buying
     a rejection is buying the reversal of the move you think you are joining.
  3. **Freshness.** Within `max_bars_since`. The mechanism is about the minutes
     after the repricing; an hour later the liquidity has come back.
  4. **Not already given back.** Price must still hold most of the impulse. Half
     of it retraced means the move has been rejected by the market itself, and
     what looked like a break was a spike.

Registered as an intraday module and deliberately NOT as a trend-continuation
one. It measures a move rather than inferring a trend, so an H1 range does not
contradict it — see the note on `trend_continuation_modules`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import ImpulseBreakConfig
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


class ImpulseBreak:
    """A recent, decisive, unretraced impulse bar."""

    name = "impulse_break"

    def __init__(self, config: ImpulseBreakConfig | None = None) -> None:
        self.config = config or ImpulseBreakConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "impulse break disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.atr_period + config.max_bars_since + 2
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        atr = _atr(frame, config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")

        # The largest body inside the freshness window. Searching rather than
        # testing only the last bar, so an impulse that printed two bars ago is
        # still tradeable — that is the whole point of a window.
        window = frame.iloc[-(config.max_bars_since + 1) :]
        bodies = (window["close"] - window["open"]).to_numpy()
        pick = int(np.argmax(np.abs(bodies)))
        body = float(bodies[pick])
        body_atr = abs(body) / atr
        if body_atr < config.minimum_body_atr:
            return Signal.neutral(
                self.name,
                f"largest {timeframe.value} body in the last {config.max_bars_since + 1} bars is "
                f"{body_atr:.2f} ATR, below the {config.minimum_body_atr:.2f} floor",
            )
        side = 1 if body > 0 else -1

        bar = window.iloc[pick]
        bar_range = float(bar["high"]) - float(bar["low"])
        if bar_range <= 0:
            return Signal.neutral(self.name, "impulse bar has no range")
        # Where in its own range did it close? A big body that closed mid-range
        # is a bar that gave half of itself back before it even finished.
        close_location = (float(bar["close"]) - float(bar["low"])) / bar_range
        if side < 0:
            close_location = 1.0 - close_location
        if close_location < config.minimum_close_location:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} impulse closed at {close_location:.0%} of its own range, "
                f"below the {config.minimum_close_location:.0%} floor — that is a rejection, "
                f"not a break",
            )

        # How much of the move has already been handed back?
        after = window.iloc[pick + 1 :]
        last = float(frame["close"].iloc[-1])
        origin = float(bar["open"])
        extreme = float(bar["close"])
        if len(after):
            extreme = float(after["close"].max() if side > 0 else after["close"].min())
            extreme = (
                max(extreme, float(bar["close"])) if side > 0 else min(extreme, float(bar["close"]))
            )
        travelled = (extreme - origin) * side
        if travelled <= 0:
            return Signal.neutral(self.name, f"{timeframe.value} impulse fully unwound")
        given_back = (extreme - last) * side / travelled
        if given_back > config.maximum_retracement:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} impulse has given back {given_back:.0%} of itself, above "
                f"the {config.maximum_retracement:.0%} limit — the market rejected it",
            )

        bars_since = len(window) - 1 - pick
        confidence = min(
            config.maximum_confidence,
            config.base_confidence
            + (body_atr - config.minimum_body_atr) * config.body_confidence_scale,
        )
        return Signal(
            module=self.name,
            score=config.score * side,
            confidence=confidence,
            reasoning=(
                f"{timeframe.value} printed a {body_atr:.2f} ATR "
                f"{'up' if side > 0 else 'down'} impulse {bars_since} bars ago, closing at "
                f"{close_location:.0%} of its range with {given_back:.0%} given back since"
            ),
            # The origin of the impulse. Price back through where the bar opened
            # means the repricing did not hold, which is precisely the claim.
            invalidation_price=origin,
            details={
                "timeframe": timeframe.value,
                "body_atr": round(body_atr, 2),
                "bars_since_impulse": bars_since,
                "close_location": round(close_location, 2),
                "given_back": round(given_back, 2),
                "impulse_origin": origin,
                "atr": atr,
            },
        )
