"""Buy the level a break cleared, not the break.

MEASURED, and this is the first module on this account whose numbers came from
data rather than from an argument. Ten years of M15 bars, eight instruments
(XAUUSD and seven majors), 1.84 million bars, first-touch resolution with the
stop checked inside the fill bar and unresolved trades excluded:

    buying the channel break     190,505 signals   E -0.067R   -29 sigma
    buying its retest, shipped    55,582 trades    E +0.134R   +22 sigma

Same detector, same breaks, entered in two different places.

TWO CONFIGURATIONS, AND THE SHIPPED ONE IS NOT THE BEST ONE. The strongest
cell in the sweep puts the stop 0.35 ATR beyond the level:

    stop 0.35 ATR   train  49,700 trades   E +0.340R   +54 sigma
                    holdout 30,507 trades  E +0.347R   +43 sigma
    stop 0.75 ATR   train  55,582 trades   E +0.134R   +22 sigma   <- shipped

The holdout was read once, for the best cell, and reproduced its training half
to within 0.007R. The shipped cell is the same family with a wider stop,
because `ConfluenceConfig.min_stop_atr` floors the stop at 0.8 ATR and a floor
that silently widens it would leave this measuring one trade and sending
another. 0.15 + 0.75 = 0.90 ATR passes through untouched. It has no holdout of
its own: what is out-of-sample is the family, not this cell.

Buying the break is not a neutral entry that costs a spread -- it is
measurably, overwhelmingly negative, which is what the account's own scorecard
said in miniature:

    bought at 95-100% of its own range    8 trades   -0.99R
    bought at 60-80%                     21 trades   +0.66R

HOW CLOSE TO THE LEVEL, and this is the whole parameter. The edge is monotone
in it, over 27 configurations and three channel lengths:

    within 0.15 ATR of the level   +11.3 points over chance   +54 sigma
    within 0.35 ATR               ~ +5.0 points               +20 sigma
    within 0.60 ATR                 -3.4 points               NEGATIVE

0.60 ATR from the level is already a losing trade. "Retest" is not a loose
description of a pullback; the distance IS the strategy, and every ATR of
slack thrown at it gives back a measurable share of the edge.

WHAT WAS MEASURED AND THROWN AWAY, because a module list that only records the
survivors is how this account ended up with nine detectors reading one shape:

    bollinger 2.5s + RSI(7) extreme   16,391 signals   +0.5 sigma   nothing
    far-break continuation             train +0.103R   holdout -0.012R
    trend_momentum (EMA20/50)         176,341 signals   E +0.02R, dies at cost

The Bollinger/RSI exhaustion scalp is the strategy the owner was quoted for
gold. It has no edge at any payoff between 1:1 and 5:1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import LevelRetestConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(spans.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class LevelRetest:
    """A channel break that came back to the edge it cleared."""

    name = "level_retest"

    def __init__(self, config: LevelRetestConfig | None = None) -> None:
        self.config = config or LevelRetestConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "level retest disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.channel_period + config.atr_period + config.lookback_bars + 2
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"{config.timeframe} needs {needed} closed bars")
        frame = series.df
        unit = _atr(frame, config.atr_period)
        if unit <= 0.0:
            return Signal.neutral(self.name, "ATR unavailable")

        price_now = ctx.tick.mid if ctx.tick is not None else float(frame["close"].iloc[-1])
        found = self._recent_break(frame, unit, price_now)
        if found is None:
            return Signal.neutral(
                self.name,
                f"no unspent {config.channel_period}-bar break in the last "
                f"{config.lookback_bars} {config.timeframe} bars",
            )
        direction, level, age = found

        price = price_now
        # How far above the broken level the price still is, in ATR. Negative
        # means it has traded back through, which is still a retest -- the
        # break has not failed until it CLOSES beyond the stop.
        above = direction * (price - level) / unit
        if above > config.tolerance_atr:
            return Signal.neutral(
                self.name,
                f"{above:.2f} ATR above the level it broke; the retest wants "
                f"{config.tolerance_atr:.2f}",
            )

        # Closer to the level is a better trade, measured, so the score says so
        # rather than treating every retest inside the window as identical.
        closeness = 1.0 - max(0.0, above) / config.tolerance_atr
        score = config.base_score + closeness * config.closeness_score_bonus
        confidence = min(
            config.maximum_confidence,
            config.base_confidence + closeness * config.closeness_confidence_bonus,
        )
        invalidation = level - direction * config.stop_beyond_atr * unit
        return Signal(
            module=self.name,
            score=min(score, 100.0) * direction,
            confidence=confidence,
            reasoning=(
                f"price returned to within {above:.2f} ATR of the "
                f"{config.channel_period}-bar level it broke {age} bars ago "
                f"({config.timeframe}); stop {config.stop_beyond_atr:.2f} ATR beyond it"
            ),
            key_levels=(level,),
            invalidation_price=invalidation,
            details={
                "timeframe": config.timeframe,
                "setup": self.name,
                "level": round(level, 6),
                "above_atr": round(above, 3),
                "break_age_bars": age,
            },
        )

    def _recent_break(
        self, frame: pd.DataFrame, unit: float, price: float
    ) -> tuple[int, float, int] | None:
        """The unspent channel break whose level price is nearest, or None.

        NOT simply the newest one, and the difference is the whole behaviour.
        A move that runs makes a new N-bar extreme on every bar of the run, so
        a leg from 100 to 104 is not one break -- it is twenty, each with a
        higher level. Taking the newest hands back the level at 103.8, which
        the pullback has already closed through, and the module then reports
        that nothing broke. The level being retested is the one the price is
        AT: the bottom of that chain, which is the edge of the range that
        actually gave way.

        A break is dropped once something after it CLOSED through where its
        stop would sit. That level has been given up, and re-offering it is
        how a retest becomes a knife catch.
        """
        config = self.config
        high = frame["high"].to_numpy()
        low = frame["low"].to_numpy()
        close = frame["close"].to_numpy()
        upper = pd.Series(high).shift(1).rolling(config.channel_period).max().to_numpy()
        lower = pd.Series(low).shift(1).rolling(config.channel_period).min().to_numpy()
        last = len(close) - 1
        oldest = max(config.channel_period + 1, last - config.lookback_bars)
        best: tuple[float, int, float, int] | None = None
        for i in range(last, oldest - 1, -1):
            if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
                continue
            if close[i] > upper[i]:
                direction, level = 1, float(upper[i])
            elif close[i] < lower[i]:
                direction, level = -1, float(lower[i])
            else:
                continue
            beyond = level - direction * config.stop_beyond_atr * unit
            after = close[i + 1 :]
            if len(after) and (
                (direction > 0 and float(after.min()) < beyond)
                or (direction < 0 and float(after.max()) > beyond)
            ):
                continue
            # Only a level price sits on the right side of can be retested:
            # a long retests a level BELOW it, having broken up through it.
            above = direction * (price - level) / unit
            if above < 0.0:
                continue
            if best is None or above < best[0]:
                best = (above, direction, level, last - i)
        if best is None:
            return None
        return best[1], best[2], best[3]
