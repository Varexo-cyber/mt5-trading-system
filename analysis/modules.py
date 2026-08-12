"""Small, replayable analysis modules operating only on closed OHLCV bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import (
    LevelReactionConfig,
    LiquiditySweepConfig,
    TrendMomentumConfig,
    VolatilityRegimeConfig,
)
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    high = frame["high"]
    low = frame["low"]
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class TrendMomentum:
    """EMA alignment plus normalised momentum on H4 and H1."""

    name = "trend_momentum"

    def __init__(self, config: TrendMomentumConfig | None = None) -> None:
        self.config = config or TrendMomentumConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        reads: list[int] = []
        details: dict[str, object] = {}
        invalidation: float | None = None
        timeframes = (
            Timeframe.parse(self.config.bias_timeframe),
            Timeframe.parse(self.config.signal_timeframe),
        )
        minimum = max(
            self.config.slow_ema + self.config.slope_lookback,
            self.config.atr_period + 1,
            self.config.invalidation_lookback,
        )
        for timeframe in timeframes:
            series = ctx.series.get(timeframe)
            if series is None or len(series.df) < minimum:
                return Signal.neutral(
                    self.name,
                    f"{timeframe.value} needs {minimum} closed bars",
                )
            frame = series.df
            close = frame["close"]
            fast_ema = close.ewm(span=self.config.fast_ema, adjust=False).mean()
            fast = float(fast_ema.iloc[-1])
            slow = float(close.ewm(span=self.config.slow_ema, adjust=False).mean().iloc[-1])
            atr = _atr(frame, self.config.atr_period)
            slope = float(fast_ema.diff(self.config.slope_lookback).iloc[-1])
            direction = 1 if fast > slow and slope > 0 else -1 if fast < slow and slope < 0 else 0
            reads.append(direction)
            details[timeframe.value] = {
                f"ema{self.config.fast_ema}": fast,
                f"ema{self.config.slow_ema}": slow,
                f"ema{self.config.fast_ema}_slope_{self.config.slope_lookback}": slope,
                f"atr{self.config.atr_period}": atr,
            }
            if timeframe is timeframes[1] and direction:
                recent = frame.iloc[-self.config.invalidation_lookback :]
                invalidation = float(recent["low"].min() if direction > 0 else recent["high"].max())

        # Three situations, and two of them used to give the same answer.
        #
        #   signal timeframe flat   -> nothing to trade. Silence is correct.
        #   bias against the signal -> a real conflict. Refusing is correct.
        #   bias flat, signal trends-> NOT a conflict. It is the absence of a
        #                              headwind, and it was returning the same
        #                              hard zero as an outright disagreement.
        #
        # That third branch is expensive. Over a live twelve hours "no weighted
        # directional evidence" was 18,150 of 36,331 no-signals — half of every
        # refusal the system made — and a bias timeframe with no opinion is the
        # commonest way to land in it, because H4 spends most of its life
        # somewhere between a crossover and a slope.
        #
        # Conflating "I have no view" with "no" is the same error as reading an
        # unmapped instrument as a closed market. Discounted rather than waved
        # through: an unconfirmed trend is genuinely worth less than a
        # confirmed one, so it takes a fraction of its usual confidence and
        # still has to clear the confidence floor and the score threshold.
        bias, signal_read = reads[0], reads[1]
        unconfirmed = not bias
        blocked = (
            not signal_read
            or (bias and bias != signal_read)
            or (unconfirmed and self.config.neutral_bias_confidence_scale <= 0)
        )
        if blocked:
            if not signal_read:
                why = f"{timeframes[1].value} momentum is flat"
            elif bias:
                why = f"{timeframes[0].value} momentum opposes {timeframes[1].value}"
            else:
                why = f"{timeframes[0].value} momentum is neutral"
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=why,
                details=details,
            )
        signal_frame = ctx.series[timeframes[1]].df
        atr = max(_atr(signal_frame, self.config.atr_period), 1e-12)
        separation = (
            abs(
                float(
                    signal_frame["close"]
                    .ewm(span=self.config.fast_ema, adjust=False)
                    .mean()
                    .iloc[-1]
                )
                - float(
                    signal_frame["close"]
                    .ewm(span=self.config.slow_ema, adjust=False)
                    .mean()
                    .iloc[-1]
                )
            )
            / atr
        )
        confidence = min(
            self.config.maximum_confidence,
            self.config.base_confidence + separation * self.config.separation_confidence_scale,
        )
        if unconfirmed:
            confidence *= self.config.neutral_bias_confidence_scale
        direction = signal_read
        side = "bullish" if direction > 0 else "bearish"
        return Signal(
            module=self.name,
            score=self.config.score * direction,
            confidence=confidence,
            reasoning=(
                f"{timeframes[1].value} EMA/momentum {side} with "
                f"{timeframes[0].value} neutral — unconfirmed by the bias timeframe"
                if unconfirmed
                else f"{timeframes[0].value} and {timeframes[1].value} "
                f"EMA/momentum aligned {side}"
            ),
            invalidation_price=invalidation,
            # Named in the signal itself, not only in the confidence number.
            # It reaches the journal and the review payload from here, so
            # "was this trend confirmed by the higher timeframe" becomes a
            # column that can be grouped by once there are trades to group.
            details={**details, "bias_confirmed": not unconfirmed},
        )


class LiquiditySweep:
    """Detect a wick through a 20-bar extreme that closes back inside."""

    name = "liquidity_sweep"

    def __init__(self, config: LiquiditySweepConfig | None = None) -> None:
        self.config = config or LiquiditySweepConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        primary = Timeframe.parse(self.config.primary_timeframe)
        fallback = Timeframe.parse(self.config.fallback_timeframe)
        series = ctx.series.get(primary) or ctx.series.get(fallback)
        minimum = max(
            self.config.minimum_bars,
            self.config.range_lookback + 1,
            self.config.atr_period + 1,
        )
        if series is None or len(series.df) < minimum:
            return Signal.neutral(
                self.name,
                f"needs {minimum} {primary.value} or {fallback.value} bars",
            )
        frame = series.df
        candle = frame.iloc[-1]
        prior = frame.iloc[-1 - self.config.range_lookback : -1]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        atr = max(_atr(frame, self.config.atr_period), 1e-12)
        if float(candle["low"]) < prior_low and float(candle["close"]) > prior_low:
            depth = (prior_low - float(candle["low"])) / atr
            if depth < self.config.minimum_depth_atr:
                return Signal.neutral(
                    self.name,
                    f"sell-side sweep depth {depth:.2f} ATR below minimum",
                )
            return Signal(
                module=self.name,
                score=self.config.score,
                confidence=min(
                    self.config.maximum_confidence,
                    self.config.base_confidence + depth * self.config.depth_confidence_scale,
                ),
                reasoning="sell-side liquidity swept and candle closed back above the range",
                key_levels=(prior_low, prior_high),
                invalidation_price=float(candle["low"]),
                details={"timeframe": series.timeframe.value, "sweep_depth_atr": depth},
            )
        if float(candle["high"]) > prior_high and float(candle["close"]) < prior_high:
            depth = (float(candle["high"]) - prior_high) / atr
            if depth < self.config.minimum_depth_atr:
                return Signal.neutral(
                    self.name,
                    f"buy-side sweep depth {depth:.2f} ATR below minimum",
                )
            return Signal(
                module=self.name,
                score=-self.config.score,
                confidence=min(
                    self.config.maximum_confidence,
                    self.config.base_confidence + depth * self.config.depth_confidence_scale,
                ),
                reasoning="buy-side liquidity swept and candle closed back below the range",
                key_levels=(prior_low, prior_high),
                invalidation_price=float(candle["high"]),
                details={"timeframe": series.timeframe.value, "sweep_depth_atr": depth},
            )
        return Signal.neutral(self.name, "no confirmed liquidity sweep")


class LevelReaction:
    """Score rejection from a rolling support/resistance zone."""

    name = "level_reaction"

    def __init__(self, config: LevelReactionConfig | None = None) -> None:
        self.config = config or LevelReactionConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        timeframe = Timeframe.parse(self.config.timeframe)
        series = ctx.series.get(timeframe)
        minimum = max(
            self.config.minimum_bars,
            self.config.history_lookback + 1,
            self.config.atr_period + 1,
        )
        if series is None or len(series.df) < minimum:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} needs {minimum} closed bars",
            )
        frame = series.df
        candle = frame.iloc[-1]
        history = frame.iloc[-1 - self.config.history_lookback : -1]
        support = float(history["low"].quantile(self.config.support_quantile))
        resistance = float(history["high"].quantile(self.config.resistance_quantile))
        atr = max(_atr(frame, self.config.atr_period), 1e-12)
        lower_wick = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
        upper_wick = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
        near_support = abs(float(candle["low"]) - support) <= atr * self.config.proximity_atr
        near_resistance = abs(float(candle["high"]) - resistance) <= atr * self.config.proximity_atr
        if (
            near_support
            and lower_wick > upper_wick * self.config.wick_ratio
            and float(candle["close"]) > float(candle["open"])
        ):
            return Signal(
                module=self.name,
                score=self.config.score,
                confidence=min(
                    self.config.maximum_confidence,
                    self.config.base_confidence
                    + lower_wick / atr * self.config.wick_confidence_scale,
                ),
                reasoning="bullish H1 rejection from rolling support",
                key_levels=(support, resistance),
                invalidation_price=float(candle["low"]),
            )
        if (
            near_resistance
            and upper_wick > lower_wick * self.config.wick_ratio
            and float(candle["close"]) < float(candle["open"])
        ):
            return Signal(
                module=self.name,
                score=-self.config.score,
                confidence=min(
                    self.config.maximum_confidence,
                    self.config.base_confidence
                    + upper_wick / atr * self.config.wick_confidence_scale,
                ),
                reasoning="bearish H1 rejection from rolling resistance",
                key_levels=(support, resistance),
                invalidation_price=float(candle["high"]),
            )
        return Signal.neutral(self.name, "no qualified H1 level rejection")


class VolatilityRegime:
    """Non-directional ATR percentile regime used as a veto/diagnostic."""

    name = "volatility_regime"

    def __init__(self, config: VolatilityRegimeConfig | None = None) -> None:
        self.config = config or VolatilityRegimeConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        timeframe = Timeframe.parse(self.config.timeframe)
        series = ctx.series.get(timeframe)
        minimum = max(
            self.config.minimum_bars,
            self.config.atr_period + self.config.percentile_lookback - 1,
        )
        if series is None or len(series.df) < minimum:
            return Signal.neutral(
                self.name,
                f"{timeframe.value} needs {minimum} closed bars",
            )
        frame = series.df
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - frame["close"].shift(1)).abs(),
                (frame["low"] - frame["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atrs = tr.rolling(self.config.atr_period).mean().dropna()
        current = float(atrs.iloc[-1])
        percentile = float((atrs.iloc[-self.config.percentile_lookback :] <= current).mean())
        regime = (
            "compressed"
            if percentile < self.config.compressed_percentile
            else "extreme"
            if percentile > self.config.extreme_percentile
            else "normal"
        )
        return Signal(
            module=self.name,
            score=0.0,
            confidence=1.0,
            reasoning=f"H1 volatility regime is {regime}",
            details={
                "regime": regime,
                f"atr{self.config.atr_period}": current,
                "percentile": percentile,
            },
        )
