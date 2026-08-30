"""SECTION TWO. A decisive break, bought back at the level it cleared.

THE ONLY STRATEGY ON THIS ACCOUNT WITH A NUMBER BEHIND IT, and the number came
from ninety-four detectors measured over sixteen instruments and six
timeframes rather than from an argument.

    24,063 resolved trades, 11 FX majors, M15 and H1, 2012-2022
    hit 48.19%   gross +0.446R   net of harness bias and spread  +0.237R

At the ratio actually shipped -- 1:1 against a one-ATR stop -- it is better
still, and the two timeframes agree to within a hundredth of an R:

    M15   18,828 trades   hit 67.9%   net +0.279R   (train +0.272 / test +0.285)
    H1     5,235 trades   hit 67.9%   net +0.281R   (train +0.276 / test +0.284)

Positive in all eleven years, from +0.181R to +0.394R. No losing year.

THE RULE, and every clause of it was measured separately:

    1. price closes beyond a 20-bar channel edge          the break
    2. that closing price is >= 1.0 ATR past the edge     the IMPULSE filter
    3. the level is frozen; the break itself is not bought
    4. a limit order rests within 0.15 ATR of the level   the retest
    5. the stop sits 0.85 ATR beyond the level            R = 1.00 ATR
    6. the target is 1R

WHY THE BREAK IS NOT BOUGHT. Buying it is not neutral-minus-a-spread, it is
-0.067R over 190,505 signals at -29 sigma. The break and its retest are two
different trades on the same signal, about 0.35R apart.

WHY THE IMPULSE FILTER IS THE WHOLE THING. Without it the same retest nets
roughly zero; with it, +0.28R. A break that closes a full ATR beyond the level
has taken every resting offer on the way and left a queue behind it. A break
that merely pokes through has not.

WHY 1:1 AND NOT 3:1. Measured, at a one-ATR stop on FX:

    0.75R  +0.245      1.5R  +0.176
    1.00R  +0.279      2.0R  +0.081
    1.25R  +0.235      3.0R  +0.016

The edge is in a high hit rate near the level, not in a long run. Reaching for
payoff trades it at nearly nothing. `target_r_multiple_by_family` exists so
this family can carry 1.0 while the rest of the account keeps 3.0.

WHY THE STOP IS 1.00 ATR AND NOT THE 0.50 THAT MEASURED BEST. 0.50 ATR nets
+0.242R at 2R, better than any wider stop -- but `ConfluenceConfig.min_stop_atr`
floors the stop at 0.8, and a floor that silently widens it would leave this
measuring one trade and sending another. 1.00 ATR passes through untouched.

GOLD, and a correction I owe the record. This file first said gold could not
be afforded, quoting -0.099R. That number was measured at a 0.50 ATR stop with
a 2R target -- a configuration that was then NOT shipped. At the configuration
that IS shipped, gold is positive:

    XAUUSD M15, stop 1.00 ATR, 1R   1,945 trades   +0.144R   (tr +0.104/te +0.184)

The conclusion had been carried over from one table to a different setting.

Gold still needs a wider stop than a major, and not for profitability. Its
spread is ~0.10 ATR against 0.04, so at the family's one-ATR stop it spends 20%
of R on execution and `max_spread_share_of_stop: 0.08` refuses it -- correctly.
`stop_beyond_atr_by_symbol` gives it 1.50 ATR, where the same spread is 6.7%
and the gate passes it untouched. The gate is not loosened for gold; gold is
given a stop it fits inside. Measured there: +0.083R, train +0.065 / test
+0.100.

WHAT DOES NOT HELP IS A SMALLER LOT. Cost in R is spread/R, a ratio: halving
the lot halves the win, the loss and the spread together and leaves the
R-multiple exactly where it was. Position size decides what a trade pays in
euros, never whether it pays.

IT MUST BE ABLE TO TRADE ALONE. All 18,828 trades were taken on this signal
by itself. `lone_module_minimum_confidence` is 0.65 on this account, the fill
sits at `level + tolerance` so the closeness term is near zero on almost every
real signal, and the module first shipped at 0.58 base confidence -- meaning
the typical setup was silently refused and would only ever have traded when
some other detector happened to agree. That is not the measured strategy.
Confidence is therefore anchored at 0.68, which is what it hits.

WHAT THE MEASUREMENT SUBTRACTS, because without it this file would be wrong.
The harness was checked with random entries, and random entries are not free:
at a 3:1 target a coin flip reads +0.073R at +13.8 sigma, because a bar
registers a barrier when its extreme crosses it and that overshoot is
proportionally larger on the nearer one. Every figure above has that bias
removed. At 1:1 the bias is -0.002R, which is the other reason this ratio is
the one to trade: it is the least contaminated point on the curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import ImpulseRetestConfig
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


class ImpulseRetest:
    """A break that closed a full ATR clear, offered back at its level."""

    name = "impulse_retest"

    def __init__(self, config: ImpulseRetestConfig | None = None) -> None:
        self.config = config or ImpulseRetestConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "impulse retest disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.channel_period + config.atr_period + config.lookback_bars + 2
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"{config.timeframe} needs {needed} closed bars")
        frame = series.df
        unit = _atr(frame, config.atr_period)
        if unit <= 0.0:
            return Signal.neutral(self.name, "ATR unavailable")

        # A wider stop where the spread is wider. Gold spends 20% of a
        # one-ATR stop on execution and `max_spread_share_of_stop` refuses it;
        # at 1.50 ATR the same spread is 6.7% and the gate passes it untouched.
        stop_beyond = config.stop_beyond_atr_by_symbol.get(ctx.symbol, config.stop_beyond_atr)
        price = ctx.tick.mid if ctx.tick is not None else float(frame["close"].iloc[-1])
        found = self._live_break(frame, unit, price, stop_beyond)
        if found is None:
            return Signal.neutral(
                self.name,
                f"no unspent {config.channel_period}-bar break of at least "
                f"{config.minimum_impulse_atr:.2f} ATR in the last "
                f"{config.lookback_bars} {config.timeframe} bars",
            )
        direction, level, impulse, age = found

        above = direction * (price - level) / unit
        if above > config.tolerance_atr:
            return Signal.neutral(
                self.name,
                f"{above:.2f} ATR above the level it broke; the retest wants "
                f"{config.tolerance_atr:.2f}",
            )

        # A bigger impulse left a bigger queue, and closer to the level is a
        # better fill. Both were measured monotone, so both move the score
        # rather than every qualifying setup scoring the same.
        closeness = 1.0 - max(0.0, above) / config.tolerance_atr
        conviction = min(1.0, (impulse - config.minimum_impulse_atr) / config.impulse_span_atr)
        score = config.base_score + (closeness + conviction) / 2.0 * config.quality_score_bonus
        confidence = min(
            config.maximum_confidence,
            config.base_confidence
            + (closeness + conviction) / 2.0 * config.quality_confidence_bonus,
        )
        return Signal(
            module=self.name,
            score=min(score, 100.0) * direction,
            confidence=confidence,
            reasoning=(
                f"{config.channel_period}-bar level broken by {impulse:.2f} ATR "
                f"{age} bars ago and retested to within {above:.2f} ATR "
                f"({config.timeframe}); stop {stop_beyond:.2f} ATR beyond it"
            ),
            key_levels=(level,),
            invalidation_price=level - direction * stop_beyond * unit,
            details={
                "timeframe": config.timeframe,
                "setup": self.name,
                "level": round(level, 6),
                "impulse_atr": round(impulse, 3),
                "above_atr": round(above, 3),
                "break_age_bars": age,
            },
        )

    def _live_break(
        self, frame: pd.DataFrame, unit: float, price: float, stop_beyond: float
    ) -> tuple[int, float, float, int] | None:
        """The unspent decisive break whose level price is nearest, or None.

        NOT the newest one. A leg that runs makes a new N-bar extreme on every
        bar of the run, so a move from 100 to 104 is twenty breaks with twenty
        rising levels. Taking the newest hands back the level at 103.8, which
        the pullback has already gone through, and the module reports that
        nothing broke. The level being retested is the one price is AT.
        """
        config = self.config
        high = frame["high"].to_numpy()
        low = frame["low"].to_numpy()
        close = frame["close"].to_numpy()
        upper = pd.Series(high).shift(1).rolling(config.channel_period).max().to_numpy()
        lower = pd.Series(low).shift(1).rolling(config.channel_period).min().to_numpy()
        last = len(close) - 1
        oldest = max(config.channel_period + 1, last - config.lookback_bars)
        best: tuple[float, int, float, float, int] | None = None
        for i in range(last, oldest - 1, -1):
            if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
                continue
            if close[i] > upper[i]:
                direction, level = 1, float(upper[i])
            elif close[i] < lower[i]:
                direction, level = -1, float(lower[i])
            else:
                continue
            # THE IMPULSE FILTER. Measured on the close, not the high: a bar
            # that spiked through and closed back is a different event, and
            # `false_break` measured it separately at nothing.
            impulse = direction * (close[i] - level) / unit
            if impulse < config.minimum_impulse_atr:
                continue
            # Given up? Then it is not a level any more.
            beyond = level - direction * stop_beyond * unit
            after = close[i + 1 :]
            if len(after) and (
                (direction > 0 and float(after.min()) < beyond)
                or (direction < 0 and float(after.max()) > beyond)
            ):
                continue
            above = direction * (price - level) / unit
            if above < 0.0:
                continue  # price is through the level; wrong side to retest
            if best is None or above < best[0]:
                best = (above, direction, level, impulse, last - i)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]
