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

import numpy as np
import pandas as pd

from config.schema import CandleMomentumConfig
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


def slope_direction(frame: pd.DataFrame, bars: int, minimum_range: float = 0.0) -> int:
    """Which way a timeframe is pointing — or 0 when it is not pointing.

    THIS IS WHERE SECTION SIX TOOK THE WRONG SIDE. The first version read two
    closes and nothing else:

        change = closes.iloc[-1] / closes.iloc[-(bars + 1)] - 1.0
        return 1 if change > 0 else -1 if change < 0 else 0

    Two faults, and together they are the whole "it goes long where it should
    have gone short" complaint.

    NO MINIMUM. A change of one hundred-thousandth of a percent returned +1,
    carrying exactly the same authority as a real trend. On a market going
    nowhere that sign is a coin flip, so M5 and M15 "agreeing" with the candle
    is a one-in-four COINCIDENCE rather than evidence -- and what follows is a
    lone M1 candle traded with no context behind it at all. This module's whole
    thesis is "everything above it is already pointing that way". On a flat
    chart that thesis was being satisfied by noise.

    TWO POINTS. Whichever bar happens to sit at `-(bars + 1)` decides the
    answer by itself, so one spike in that reference bar inverts the read of
    the entire window. With `bias_bars` at 4, a quarter of the evidence rested
    on a single close.

    So: a least-squares fit across every bar in the window, which no single bar
    can flip, and a floor measured in that timeframe's OWN average bar range. A
    move smaller than one bar's normal travel is not a direction, it is the
    market standing still, and the honest answer there is 0 -- which reads as a
    disagreement upstream and refuses the trade.

    Still deliberately not a trend module. It answers "is the slower chart
    actually going somewhere, and is it the same way", and nothing else.
    """
    if len(frame) < bars + 1:
        return 0
    window = frame.iloc[-(bars + 1) :]
    closes = window["close"].astype(float).to_numpy()
    if not np.isfinite(closes).all():
        return 0

    # Degree-1 polyfit returns (slope, intercept). The slope is price per bar,
    # so times the span is the move the fit describes across the whole window.
    x = np.arange(len(closes), dtype=float)
    slope = float(np.polyfit(x, closes, 1)[0])
    if not math.isfinite(slope):
        return 0
    travel = slope * (len(closes) - 1)

    # A DEAD-FLAT SERIES IS NOT A DIRECTION, and least squares does not say so
    # by itself: on twenty identical closes `polyfit` returns a slope of about
    # 1e-15 rather than exactly zero, and `travel > 0` then reports a confident
    # +1. That is this function's original defect reappearing in the repair --
    # a meaningless move carrying a full sign. Scaled to the price level, so it
    # means the same thing on gold at 4,000 and on EURUSD at 1.08.
    if abs(travel) <= abs(float(closes[-1])) * 1e-12:
        return 0

    if minimum_range > 0.0:
        # The instrument's own yardstick rather than a pip count: "meaningful"
        # is a different number on gold and on EURUSD, and a fixed threshold
        # would silently switch this off on one of them.
        highs = window["high"].astype(float).to_numpy()
        lows = window["low"].astype(float).to_numpy()
        average_range = float(np.mean(highs - lows))
        if not math.isfinite(average_range) or average_range <= 0:
            return 0
        if abs(travel) < minimum_range * average_range:
            return 0

    return 1 if travel > 0 else -1 if travel < 0 else 0


class CandleMomentum:
    """A closed M1 candle, taken only when M5 and M15 already agree."""

    name = "candle_momentum"

    def __init__(self, config: CandleMomentumConfig | None = None) -> None:
        self.config = config or CandleMomentumConfig()

    @staticmethod
    def _range_location(frame: pd.DataFrame, bars: int, direction: int) -> float | None:
        """Where the last close sits in the recent range, in the trade's favour.

        1.0 means the close is at the extreme the trade is betting on -- the
        high for a long, the low for a short -- so there is nothing left in
        front of it. 0.0 means the whole range is still ahead.

        Returned rather than judged so the number lands in `details` even when
        it passes: "how close to the top was it" is the first question about
        any entry that went straight into the red, and a gate that only records
        its refusals cannot answer it.
        """
        if bars < 2 or len(frame) < bars:
            return None
        window = frame.iloc[-bars:]
        high = float(window["high"].astype(float).max())
        low = float(window["low"].astype(float).min())
        close = float(window["close"].astype(float).iloc[-1])
        span = high - low
        if not math.isfinite(span) or span <= 0:
            return None
        position = (close - low) / span
        return position if direction > 0 else 1.0 - position

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
        # A CANDLE CAN ALSO BE TOO BIG, and nothing here said so.
        #
        # There is a ceiling on VOLUME -- "this minute carried 8.4x its normal
        # activity, that is an event, not momentum" -- and the argument is
        # entirely about the size of the move relative to normal. It was never
        # applied to the move itself. `minimum_body_multiple` is a floor and
        # `body_saturation_multiple` only caps the SCORE, so a bar ten times
        # its recent average body sailed through with maximum confidence.
        #
        # The strongest-looking bar such a system will ever see is the one it
        # should not trade. That sentence is already written in
        # `volume_spike_filter`; this is the same sentence about price.
        #
        # WHAT IT LOOKED LIKE LIVE. Four NDX100 scalps overnight:
        #
        #     +0.14   stop  2.92 points
        #     +0.40   stop  5.16 points
        #     -1.96   stop 21.12 points
        #     -1.36   stop 15.00 points
        #
        # The stop is one candle span, so the column is the candle. The two
        # winners triggered on ordinary minutes and the two losers on minutes
        # three to seven times larger. Four trades prove nothing on their own,
        # which is why the reason to refuse these is the argument above and not
        # the table -- but the table is what made anyone look.
        #
        # A wide bar also carries a second cost that is not obvious: the stop
        # is a candle span, so an outsized candle produces an outsized R, and
        # on an account that cannot express it the position lands on the broker
        # minimum and the money risked stops matching the plan.
        if candle.body_multiple >= config.maximum_body_multiple:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"body {candle.body_multiple:.1f}x the recent average, at or above the "
                    f"{config.maximum_body_multiple:.1f}x that makes it an event rather than "
                    f"momentum — the stop would be a candle this wide and the move is already "
                    f"spent"
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

        # WAS M1 ITSELF ALREADY GOING THIS WAY, BEFORE THE CANDLE FIRED?
        #
        # Nothing asked. M1 was read as a single candle -- its body, its wick,
        # its volume -- and never as a direction, so a lone green minute inside
        # a falling M1 sequence was a buy as long as M5 and M15 happened to
        # point up. That is the owner's complaint in its purest form: the side
        # agrees with the slower charts and disagrees with the chart the trade
        # is actually being taken on.
        #
        # MEASURED WITHOUT THE TRIGGER CANDLE, which is the whole trick. The
        # trigger is by construction a big bar -- 1.75x the recent average body
        # -- and it is the last bar in the window, so including it would drag
        # the fit its own way and the test would confirm itself. Dropping it
        # asks the honest question instead: was this move under way before this
        # minute, or is this minute the entire evidence?
        #
        # NOT SYMMETRIC WITH THE OTHER TWO, deliberately. M5 and M15 must AGREE
        # -- they are the thesis. M1 must merely not CONTRADICT: flat is
        # allowed, because the first minute of a real push often follows a
        # quiet stretch and refusing that would remove the setups this module
        # exists for. Only an M1 chart actively running the other way is
        # refused, and that is the case that was losing money.
        prior = fast.df.iloc[:-1]
        m1_trend = slope_direction(prior, config.trigger_bars, config.minimum_slope_range)
        details["m1_direction"] = m1_trend
        if m1_trend != 0 and m1_trend != candle.direction:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"the {config.trigger_timeframe} bars before this candle were going "
                    f"{m1_trend:+d} and the candle is {candle.direction:+d} — this is one "
                    f"minute against its own chart, not a continuation of it"
                ),
                details=details,
            )

        confirm = slope_direction(middle.df, config.confirm_bars, config.minimum_slope_range)
        bias = slope_direction(slow.df, config.bias_bars, config.minimum_slope_range)
        details |= {"confirm_direction": confirm, "bias_direction": bias}
        if confirm != candle.direction or bias != candle.direction:
            # 0 lands here too, and that is the point. A timeframe that is not
            # going anywhere is not agreement, and it used to read as agreement
            # half the time by coin flip.
            flat = " (0 means that chart is not going anywhere)" if 0 in (confirm, bias) else ""
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"candle {candle.direction:+d} against confirm {confirm:+d} and bias "
                    f"{bias:+d} — two out of three is a disagreement, not a majority{flat}"
                ),
                details=details,
            )

        # WHERE IN THE MOVE THIS IS JOINING -- MEASURED, NOT YET JUDGED.
        #
        # The suspicion is real: the exhaustion refusal below asks "is there
        # room left" of the TRIGGER CANDLE, which is sixty seconds of evidence,
        # and never asks it of the move the trade is actually joining.
        #
        # It is not gated, and the reason is a measurement rather than caution.
        # Refusing above 80% of the recent range kills the module outright: in
        # any clean trend the newest close IS near the recent high, which is
        # what a trend is. Gating on that would refuse exactly the setups this
        # module exists to take, and "it stopped losing because it stopped
        # trading" is not a fix.
        #
        # So it is recorded on every reading, pass or refuse. After a day of
        # live scalps the journal can say what this number was on the winners
        # and on the losers, and then a threshold is a measurement instead of a
        # guess. A gate invented today would be the third knob turned on a
        # hunch on this module, and the first two both had to come back out.
        location = self._range_location(middle.df, config.confirm_range_bars, candle.direction)
        if location is not None:
            details["confirm_range_location"] = round(location, 3)

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

        # THE CANDLE HAS TO PAY FOR ITSELF, and this refusal was documented
        # above and never built until the arithmetic was actually run.
        #
        # The target here is deliberately SMALLER than the stop, so the
        # break-even hit rate starts at 58% before any cost. On a quiet minute
        # the spread is most of what the target is reaching for and the
        # required hit rate goes past 80%, which no filter delivers. That is
        # not a trade with a thin edge, it is a trade with no edge available at
        # any hit rate, and the only correct response is to refuse the market
        # rather than to try harder on the entry.
        spread = float(getattr(ctx.tick, "spread", 0.0) or 0.0) if ctx.tick else 0.0
        if spread > 0 and config.minimum_target_spreads > 0:
            last = fast.df.iloc[-1]
            candle_span = float(last["high"]) - float(last["low"])
            target = candle_span * config.target_candle_spans
            details |= {"spread": spread, "target_in_spreads": target / spread}
            if target < spread * config.minimum_target_spreads:
                return Signal(
                    module=self.name,
                    score=0.0,
                    confidence=0.0,
                    reasoning=(
                        f"the target is {target / spread:.1f} spreads wide, under the "
                        f"{config.minimum_target_spreads:.0f} it needs — at this range the "
                        f"spread is most of the move and no hit rate pays for it"
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
