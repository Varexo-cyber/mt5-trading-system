"""The candle just moved. Is everything else already pointing that way?

SECTION SIX. This is the shape of the fast bots — watch M1, jump in when a
candle goes, jump straight back out. What makes those lose money is that the
candle is the WHOLE thesis: a green minute is a reason to buy, and a green
minute happens roughly half the time.

So the candle here is only the trigger. The thesis is everything above it
agreeing first, and the candle is the moment that agreement becomes actionable.

WHAT IT LOOKS AT, all of it before a single trade is considered:

    M15   the direction the session is going          (slow, decides the side)
    M5    the direction the last half hour is going   (must not contradict)
    M1    the candle that just closed                 (the trigger, and only that)

All three must point the same way. Two out of three is not a majority here, it
is a disagreement, and a disagreement on a trade that lives for minutes is a
coin flip with costs attached.

THE THREE REFUSALS, and each exists because of a specific way this shape dies:

*Exhaustion.* A candle that closes on its extreme after the move has already run
is the last buyer, not the first. Measured against the recent range rather than
against a fixed pip count, because "already run" means something different on
gold and on EURUSD.

*Extreme volume.* A minute carrying many times its own normal activity is not
momentum, it is an event — a release, a headline, a stop cascade. The bots that
blow up are the ones that read the first candle of a red-folder release as the
strongest signal they have ever seen. It is the strongest signal they have ever
seen, and it is followed by a spread that eats them.

*Cost.* A scalp's edge is measured in a handful of pips and the spread is
subtracted from every one of them. This refuses when the candle's own body
cannot pay for the round trip several times over.

WHAT IS NOT HERE. The news blackout is not in this module, deliberately. This
reads bars; it has no calendar, and inventing one here would put a second,
divergent copy of the news rules next to the real one. `news_filter`,
`headline_filter` and the runner's observer hold that, and the observer refuses
to record a scalp inside a blackout window. Same rule, one place.

It does not trade. It carries a weight so the module backtest measures it, and
it is absent from `live_enabled_modules`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from config.schema import MomentumScalpConfig
from core.types import MarketContext, Signal, Timeframe


@dataclass(frozen=True, slots=True)
class CandleRead:
    """The trigger candle, measured against its own recent history."""

    #: +1 up, -1 down, 0 for a candle with no body worth the name.
    direction: int
    #: Body as a fraction of the whole candle. A doji is near zero.
    body_share: float
    #: Body in multiples of the recent average body. This is "it moved", scaled.
    body_multiple: float
    #: Where the close sits inside the candle. 1.0 is closing on the high.
    close_position: float
    #: Tick volume against its own recent median.
    volume_multiple: float


def read_candle(frame: pd.DataFrame, lookback: int) -> CandleRead | None:
    """Measure the last closed candle against the `lookback` before it."""
    if len(frame) < lookback + 2:
        return None
    last = frame.iloc[-1]
    prior = frame.iloc[-(lookback + 1) : -1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    open_ = float(last["open"])
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return None

    body = close - open_
    bodies = (prior["close"] - prior["open"]).abs()
    average_body = float(bodies.mean())
    if not math.isfinite(average_body) or average_body <= 0:
        return None

    volumes = prior["tick_volume"].astype(float)
    median_volume = float(volumes.median())
    volume = float(last["tick_volume"])
    volume_multiple = volume / median_volume if median_volume > 0 and math.isfinite(volume) else 1.0

    return CandleRead(
        direction=1 if body > 0 else -1 if body < 0 else 0,
        body_share=abs(body) / span,
        body_multiple=abs(body) / average_body,
        close_position=(close - low) / span,
        volume_multiple=volume_multiple,
    )


def slope_direction(closes: pd.Series, bars: int) -> int:
    """Which way a timeframe is pointing, as a sign and nothing more.

    Deliberately crude. This is not a trend module — it is the question "does
    the slower chart contradict the candle", and a contradiction is a sign
    disagreement. Anything finer would be a second trend reader competing with
    the ones that already exist.
    """
    if len(closes) < bars + 1:
        return 0
    start = float(closes.iloc[-(bars + 1)])
    end = float(closes.iloc[-1])
    if start <= 0 or not math.isfinite(end):
        return 0
    change = end / start - 1.0
    return 1 if change > 0 else -1 if change < 0 else 0


class MomentumScalp:
    """A closed M1 candle, taken only when M5 and M15 already agree."""

    name = "momentum_scalp"

    def __init__(self, config: MomentumScalpConfig | None = None) -> None:
        self.config = config or MomentumScalpConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "momentum scalp disabled")

        fast = ctx.series.get(Timeframe.parse(config.trigger_timeframe))
        middle = ctx.series.get(Timeframe.parse(config.confirm_timeframe))
        slow = ctx.series.get(Timeframe.parse(config.bias_timeframe))
        if fast is None or middle is None or slow is None:
            return Signal.neutral(self.name, "needs M1, M5 and M15 history")

        candle = read_candle(fast.df, config.candle_lookback)
        if candle is None:
            return Signal.neutral(self.name, f"needs {config.candle_lookback + 2} closed bars")

        details = {
            "body_share": candle.body_share,
            "body_multiple": candle.body_multiple,
            "close_position": candle.close_position,
            "volume_multiple": candle.volume_multiple,
        }

        # AN EVENT IS NOT MOMENTUM, and this is the refusal that keeps the whole
        # idea out of the ditch the fast bots fall into. A minute carrying many
        # times its own normal activity is a release, a headline or a stop
        # cascade. It is the strongest-looking candle such a bot will ever
        # print, and what follows it is a spread that takes the account apart.
        if candle.volume_multiple >= config.extreme_volume_multiple:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"this minute carried {candle.volume_multiple:.1f}x its normal activity — "
                    f"that is an event, not momentum, and the spread after it is the risk"
                ),
                details=details,
            )

        if candle.direction == 0 or candle.body_share < config.minimum_body_share:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"the candle is {candle.body_share:.0%} body — mostly wick, so nobody "
                    f"finished the minute in control"
                ),
                details=details,
            )
        if candle.body_multiple < config.minimum_body_multiple:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"body {candle.body_multiple:.1f}x the recent average, under the "
                    f"{config.minimum_body_multiple:.1f}x that makes it a move rather than a minute"
                ),
                details=details,
            )

        confirm = slope_direction(middle.df["close"], config.confirm_bars)
        bias = slope_direction(slow.df["close"], config.bias_bars)
        details |= {"confirm_direction": confirm, "bias_direction": bias}
        if confirm != candle.direction or bias != candle.direction:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"candle {candle.direction:+d} against confirm {confirm:+d} and bias "
                    f"{bias:+d} — two out of three is a disagreement, not a majority"
                ),
                details=details,
            )

        # THE LAST BUYER PROBLEM. A candle closing hard on its own extreme after
        # the move has already travelled is the end of the move, not the start.
        # Measured against this instrument's own recent range, because "already
        # run" means one thing on gold and another on EURUSD.
        reached = candle.close_position if candle.direction > 0 else 1.0 - candle.close_position
        details["closed_at_extreme"] = reached
        if reached >= config.exhaustion_close_position:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"closed at {reached:.0%} of its own range — that is the last buyer "
                    f"finishing the move, not the first one starting it"
                ),
                details=details,
            )

        span = max(1e-9, config.body_saturation_multiple - config.minimum_body_multiple)
        strength = min(1.0, (candle.body_multiple - config.minimum_body_multiple) / span)
        score = candle.direction * (config.base_score + strength * (100.0 - config.base_score))
        room = config.maximum_confidence - config.base_confidence
        confidence = min(config.maximum_confidence, config.base_confidence + strength * room)
        way = "up" if candle.direction > 0 else "down"
        return Signal(
            module=self.name,
            score=score,
            confidence=confidence,
            reasoning=(
                f"M1 closed {way} on a body {candle.body_multiple:.1f}x its recent average "
                f"with M5 and M15 already pointing the same way, and {candle.volume_multiple:.1f}x "
                f"normal activity — busy, not an event"
            ),
            details=details,
        )
