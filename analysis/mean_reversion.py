"""A statistical extreme that has stopped extending — faded, not followed.

Every directional module in this system is a continuation reader in some form:
a trend, a cross, a break of structure, a drift, a resumed pullback, a
micro-break. The single exception is `liquidity_sweep`, and over 180 days on
five symbols it is the only one that is not negative — +0.119R a trade against
`trend_momentum` at -0.382R over 62 trades, t = -3.26.

That is a hint, not a proof: liquidity_sweep's own t is 0.59, which is noise,
and it needs around 268 trades to establish itself at that expectancy. The way
to find out whether the reversion family is the one that works here is to have
a SECOND reversion reader and measure them together, rather than to add a ninth
continuation reader to seven that lose.

DISTINCT FROM `range_fade`. That playbook needs an identified range with
touched edges, and it was measured at -0.174R over 898 trades. This asks a
purely statistical question instead: how far is price from its own mean, in its
own standard deviations, and has the move stopped going.

THE STALL REQUIREMENT IS THE WHOLE MODULE. An extreme that is still extending
is a trend, and fading a trend because it has gone far is the most expensive
mistake available in this family — it is the shape of every account that has
ever been destroyed by averaging into a loser. So the extreme has to be recent
AND the market has to have already given some of it back before this says
anything.

Not enabled live. It goes on the allowlist when the module backtest says it
earns its place, and not before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import MeanReversionConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    high, low = frame["high"], frame["low"]
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class MeanReversion:
    """Fade a stretched, stalling market back toward its own mean."""

    name = "mean_reversion"

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "mean reversion disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.lookback + config.max_bars_since_extreme + config.atr_period + 1
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        closes = frame["close"]
        window = closes.iloc[-config.lookback :]
        mean = float(window.mean())
        sd = float(window.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return Signal.neutral(self.name, f"{timeframe.value} has no dispersion to measure")

        # Which recent bar was the extreme, and how stretched was it. Measured
        # on the extreme rather than on the last close, because by the time a
        # stall is visible the last close is no longer the stretched one — and
        # requiring the LAST bar to be extreme is requiring the fade not to
        # have started, which is the knife-catch this module refuses to be.
        recent = frame.iloc[-(config.max_bars_since_extreme + 1) :]
        high_z = (float(recent["high"].max()) - mean) / sd
        low_z = (mean - float(recent["low"].min())) / sd
        if max(high_z, low_z) < config.entry_z:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} is {max(high_z, low_z):.2f} SD from its "
                f"{config.lookback}-bar mean, under the {config.entry_z:.2f} an extreme needs",
            )

        if high_z >= low_z:
            direction, stretch = -1, high_z
            extreme = float(recent["high"].max())
        else:
            direction, stretch = 1, low_z
            extreme = float(recent["low"].min())

        atr = _atr(frame, config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")

        # How far price has come off the extreme. This is the stall: still
        # extending means still a trend, and a trend is not this module's trade
        # at any distance from the mean.
        #
        # Measured in ATR rather than as a share of the extreme bar's own
        # range, which is how this was first written and which a test caught. A
        # thin final bar makes that share meaningless — a bar spanning a tenth
        # of an ATR whose close sits halfway down it reads as "50% retraced"
        # while price has come off the high by a twentieth of an ATR. The
        # module would have faded markets that were still running, which is the
        # one mistake this family cannot afford.
        last_close = float(closes.iloc[-1])
        retraced = (extreme - last_close) * -direction / atr
        if retraced < config.stall_retrace_atr:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} is {stretch:.2f} SD stretched but has come off the "
                f"extreme by only {retraced:.2f} ATR, under the "
                f"{config.stall_retrace_atr:.2f} a stall needs — this is still going, and "
                f"going is not this module's trade",
            )

        confidence = min(
            config.maximum_confidence,
            config.base_confidence + (stretch - config.entry_z) * config.stretch_confidence_scale,
        )
        # Wrong the moment the extreme is exceeded: the stretch was not the
        # end of the move after all.
        invalidation = extreme + atr * config.invalidation_buffer_atr * -direction
        return Signal(
            module=self.name,
            score=config.score * direction,
            confidence=confidence,
            reasoning=(
                f"{timeframe.value} reached {stretch:.2f} SD "
                f"{'above' if direction < 0 else 'below'} its {config.lookback}-bar mean and "
                f"has come off it by {retraced:.2f} ATR without extending"
            ),
            key_levels=(extreme, mean),
            invalidation_price=invalidation,
            details={
                "timeframe": timeframe.value,
                "stretch_sd": round(stretch, 2),
                "retraced_atr": round(retraced, 2),
                "mean": mean,
                "standard_deviation": sd,
                "extreme": extreme,
                "atr": atr,
            },
        )
