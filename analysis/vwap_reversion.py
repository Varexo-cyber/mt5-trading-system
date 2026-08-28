"""How far is price from what the day actually paid?

SECTION TWO, replacing `drift_burst`, and the replacement is on the owner's
instruction after that section finished 89 observed setups at 19% hit and
-64.07R. Same slot, same paper-first discipline, a different question.

THE QUESTION. VWAP is the volume-weighted average price since the session
opened: every trade of the day, weighted by how much went through at it. It is
not an indicator laid over the chart, it is a summary of the chart -- the
average price at which the day's business was actually done, and the benchmark
an execution desk is measured against.

WHY THAT MATTERS HERE AND NOT AS FOLKLORE. The usual telling is that price
"snaps back to VWAP like an elastic band", which explains nothing and predicts
nothing. What is defensible is narrower: a desk with a large order to work is
judged on whether it beat VWAP, so it becomes a buyer when price is well below
it and a seller when price is well above. That is a real, dated, one-directional
pressure and it decays as the session ends and the benchmark stops mattering.

WHY THIS ACCOUNT NEEDS IT, from its own scorecard rather than from theory:

    range        16 trades   -2.04R
    transition   35 trades   -1.76R
    trend_up      6 trades   +0.20R
    trend_down    2 trades   +0.18R

Every loss is in a sideways or undecided market, which is exactly where a trend
follower should lose and exactly where mean reversion should work. And the only
mean-reversion reader on the book, `basket_divergence`, fired nine times in
twenty-four hours against `trend_momentum`'s 58,612. The account is nine
detectors reading the same shape and one that does not, and the one that does
not barely speaks.

WHAT IT WILL NOT DO. Fade a move that has a reason. A price two standard
deviations below VWAP during a release is not mispriced, it is repriced, and
the difference is invisible in the distance alone. So the module requires the
stretch to be STALLING before it acts -- the last bars must stop extending --
and the news filter still owns the calendar side of that question.

AND IT STARTS ON PAPER. `drift_burst` traded nothing for its whole life and was
graded on observation; this inherits that. A tenth detector that loses is worth
less than nothing, and the only way to know which it is, is to make it commit
in writing before it commits money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from config.schema import VwapReversionConfig
from core.types import MarketContext, Signal, Timeframe


@dataclass(frozen=True, slots=True)
class VwapReading:
    """What the session has done so far, and where price sits in it."""

    vwap: float
    #: Distance from VWAP in standard deviations of the session's own spread
    #: around it. Signed: positive means price is above.
    sigma: float
    #: How far the last `stall_bars` have extended the stretch, in the same
    #: units. Near zero means the move away from VWAP has stopped.
    extension: float
    bars: int

    @property
    def side(self) -> int:
        return 1 if self.sigma > 0 else -1


def session_vwap(frame: pd.DataFrame, *, stall_bars: int) -> VwapReading | None:
    """VWAP and the spread around it, over the bars supplied.

    TYPICAL PRICE AND NOT CLOSE. VWAP is defined on (high + low + close) / 3
    because a bar is a range of trades, not one trade at its end. Using the
    close is the common shortcut and it biases the average toward whichever end
    of each bar the market happened to finish, which is precisely the quantity
    this module then measures a deviation from.

    The dispersion is the volume-weighted standard deviation around that same
    VWAP, so "two sigma" means two of the day's own sigmas rather than a fixed
    number of pips that means something different on gold and on EURUSD.

    None when the session is too young to have a shape yet. A VWAP over four
    bars is the price, and a deviation from it is zero by construction.
    """
    if frame.empty or len(frame) < max(stall_bars + 1, 10):
        return None
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    # `tick_volume` is what MT5 supplies for FX and CFDs: the number of price
    # changes in the bar. It is not contract volume and does not pretend to be
    # -- it is a proxy for activity, and it is the same proxy every serious
    # retail VWAP uses because real volume does not exist off-exchange.
    weight = frame.get("tick_volume")
    if weight is None or float(weight.sum()) <= 0:
        # Equal weights degrade this to a simple average, which is a different
        # statistic wearing the same name. Refuse rather than rename it.
        return None
    weight = weight.astype(float)
    total = float(weight.sum())
    vwap = float((typical * weight).sum() / total)
    variance = float((weight * (typical - vwap) ** 2).sum() / total)
    dispersion = math.sqrt(variance)
    if dispersion <= 0:
        return None

    price = float(frame["close"].iloc[-1])
    sigma = (price - vwap) / dispersion
    # How much of the current stretch was added by the last few bars. A move
    # still extending is not a stretch to fade, it is a trend in progress.
    earlier = float(frame["close"].iloc[-(stall_bars + 1)])
    extension = (price - earlier) / dispersion
    return VwapReading(vwap=vwap, sigma=sigma, extension=extension, bars=len(frame))


class VwapReversion:
    """Fade a session stretch that has stopped stretching."""

    name = "vwap_reversion"

    def __init__(self, config: VwapReversionConfig) -> None:
        self.config = config

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "vwap reversion disabled")

        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        if series is None or len(series.df) < config.session_bars:
            return Signal.neutral(
                self.name, f"needs {config.session_bars} closed {timeframe.value} bars"
            )

        frame = series.df.iloc[-config.session_bars :]
        reading = session_vwap(frame, stall_bars=config.stall_bars)
        if reading is None:
            return Signal.neutral(self.name, "no usable session VWAP yet")

        stretch = abs(reading.sigma)
        if stretch < config.minimum_sigma:
            return Signal.neutral(
                self.name,
                f"{stretch:.2f} sigma from VWAP, under the {config.minimum_sigma:.2f} "
                f"this fades",
            )

        # THE HALF THAT STOPS IT STANDING IN FRONT OF A TRAIN.
        #
        # Distance alone cannot tell a mispricing from a repricing. Two sigma
        # below VWAP on a quiet drift and two sigma below on a release look
        # identical in `sigma` and have opposite futures. So the stretch has to
        # have STOPPED: the last `stall_bars` may not have added more than
        # `maximum_extension` of it, in the session's own units.
        #
        # Sign matters. A stretch above VWAP that is still rising extends
        # positively; below VWAP it extends negatively. Comparing the absolute
        # value would refuse a stretch that is coming BACK, which is the one
        # this module wants.
        still_going = reading.extension * reading.side
        if still_going > config.maximum_extension:
            return Signal.neutral(
                self.name,
                f"{stretch:.2f} sigma from VWAP but still extending "
                f"({still_going:+.2f} over the last {config.stall_bars} bars); "
                f"a move in progress is not a stretch to fade",
            )

        # Toward VWAP: short a price above it, long a price below it.
        direction = -reading.side
        # Scaled on how far past the floor the stretch is, capped so a five
        # sigma outlier -- which is usually a data fault or a halt -- cannot
        # score higher than the population this was built for.
        beyond = min(stretch - config.minimum_sigma, config.saturation_sigma)
        span = max(config.saturation_sigma, 1e-9)
        score = config.base_score + (config.maximum_score - config.base_score) * (beyond / span)
        confidence = config.base_confidence + (
            config.maximum_confidence - config.base_confidence
        ) * (beyond / span)

        way = "above" if reading.side > 0 else "below"
        return Signal(
            self.name,
            direction * score,
            round(confidence, 3),
            reasoning=(
                f"{stretch:.2f} sigma {way} the session VWAP and no longer extending "
                f"({still_going:+.2f} over {config.stall_bars} bars); the day's own "
                f"average is {reading.vwap:.5f}"
            ),
            details={
                "vwap": round(reading.vwap, 6),
                "sigma_from_vwap": round(reading.sigma, 3),
                "extension": round(still_going, 3),
                "session_bars": reading.bars,
                "timeframe": timeframe.value,
            },
        )
