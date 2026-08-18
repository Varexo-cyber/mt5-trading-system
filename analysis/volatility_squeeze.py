"""Range contracts to an extreme, then breaks out of its own compression.

`volatility_regime` already measures compression and only ever reports it as
CONTEXT — it scores no direction, so it can never produce a setup. The one
thing in this system that knows a market is coiled has no way to say "and it
has just uncoiled, that way".

WHY THIS FAMILY AND NOT ANOTHER TREND READER. All eight existing directional
modules read the same price series in a different way: a trend, a cross, a
break of structure, a drift, a resumed pullback, a micro-break. They fire
together and, measured over 180 days, they lose together — `trend_momentum`
significantly so at -0.382R a trade over 62 of them. Compression is a different
observation about the same chart: it says nothing about direction until the
break, which is precisely why it does not simply agree with the others.

WHAT MAKES IT HARD TO SATISFY, deliberately. The compression must be a genuine
extreme against this instrument's own history rather than "quiet today", and
the expansion bar must be large against the COMPRESSION rather than against the
ordinary range. Without the second test, the first bar of every London session
qualifies on every symbol, which is a schedule and not a signal.

Not enabled live. It goes on the allowlist when `scripts/backtest_modules.py`
says it earns its place and not before — that discipline is the only thing that
has produced a real finding on this account.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import VolatilitySqueezeConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    high, low = frame["high"], frame["low"]
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class VolatilitySqueeze:
    """A break out of a range that had contracted to a historic extreme."""

    name = "volatility_squeeze"

    def __init__(self, config: VolatilitySqueezeConfig | None = None) -> None:
        self.config = config or VolatilitySqueezeConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "volatility squeeze disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.percentile_lookback + config.compression_bars + 1
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        highs, lows = frame["high"], frame["low"]

        # The compression is measured on the bars BEFORE the breakout bar.
        # Including the breakout in its own compression window is the classic
        # way to make this signal look better than it is: the expansion widens
        # the range it is being compared against, and the test grades itself.
        window = frame.iloc[-(config.compression_bars + 1) : -1]
        top = float(window["high"].max())
        bottom = float(window["low"].min())
        compressed = top - bottom
        if compressed <= 0:
            return Signal.neutral(self.name, f"{timeframe.value} compression window has no range")

        # The same measurement rolled over history, so "narrow" means narrow
        # FOR THIS INSTRUMENT rather than narrow in pips.
        rolling = (
            highs.rolling(config.compression_bars).max()
            - lows.rolling(config.compression_bars).min()
        ).dropna()
        history = rolling.iloc[-config.percentile_lookback :]
        if len(history) < 20:
            return Signal.neutral(self.name, "not enough history to judge the compression")
        rank = float((history <= compressed).mean())
        if rank > config.compression_percentile:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} range sits at the {rank:.0%} percentile of its own "
                f"history, above the {config.compression_percentile:.0%} a squeeze needs — "
                f"this market is not coiled, it is merely calm",
            )

        last = frame.iloc[-1]
        bar_range = float(last["high"]) - float(last["low"])
        expansion = bar_range / compressed
        if expansion < config.expansion_multiple:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} is compressed at the {rank:.0%} percentile but the last "
                f"bar spans {expansion:.2f}x that range, under the "
                f"{config.expansion_multiple:.2f}x an expansion needs — still coiled",
            )

        close = float(last["close"])
        margin = compressed * config.breakout_close_share
        if close > top + margin:
            direction = 1
            edge, opposite = top, bottom
        elif close < bottom - margin:
            direction = -1
            edge, opposite = bottom, top
        else:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} expanded {expansion:.2f}x but closed back inside the "
                f"{compressed:.5g} compression — an expansion that resolved nothing",
            )

        atr = _atr(frame, config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")

        confidence = min(
            config.maximum_confidence,
            config.base_confidence
            + (expansion - config.expansion_multiple) * config.expansion_confidence_scale,
        )
        # The thesis is that the coil has released. It is dead the moment price
        # is back on the far side of the coil, so that is where the stop goes —
        # not at the near edge, which the breakout bar has already traded
        # through and which sits inside the noise the compression is made of.
        buffer = atr * config.invalidation_buffer_atr
        invalidation = opposite - buffer * direction
        return Signal(
            module=self.name,
            score=config.score * direction,
            confidence=confidence,
            reasoning=(
                f"{timeframe.value} compressed to the {rank:.0%} percentile of its own "
                f"{config.compression_bars}-bar range and broke "
                f"{'up' if direction > 0 else 'down'} through {edge:.5g} on a bar spanning "
                f"{expansion:.2f}x the compression"
            ),
            key_levels=(top, bottom),
            invalidation_price=invalidation,
            details={
                "timeframe": timeframe.value,
                "compression_percentile": round(rank, 3),
                "compressed_range": compressed,
                "expansion_multiple": round(expansion, 2),
                "range_top": top,
                "range_bottom": bottom,
                "atr": atr,
            },
        )
