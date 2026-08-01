"""Small, replayable analysis modules operating only on closed OHLCV bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    def analyze(self, ctx: MarketContext) -> Signal:
        reads: list[int] = []
        details: dict[str, object] = {}
        invalidation: float | None = None
        for timeframe in (Timeframe.H4, Timeframe.H1):
            series = ctx.series.get(timeframe)
            if series is None or len(series.df) < 55:
                return Signal.neutral(self.name, f"{timeframe.value} needs 55 closed bars")
            frame = series.df
            close = frame["close"]
            fast = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            slow = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            atr = _atr(frame)
            slope = float(close.ewm(span=20, adjust=False).mean().diff(5).iloc[-1])
            direction = 1 if fast > slow and slope > 0 else -1 if fast < slow and slope < 0 else 0
            reads.append(direction)
            details[timeframe.value] = {
                "ema20": fast,
                "ema50": slow,
                "ema20_slope_5": slope,
                "atr14": atr,
            }
            if timeframe is Timeframe.H1 and direction:
                recent = frame.iloc[-12:]
                invalidation = float(recent["low"].min() if direction > 0 else recent["high"].max())

        if not reads[0] or reads[0] != reads[1]:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning="H4/H1 momentum is neutral or disagrees",
                details=details,
            )
        h1 = ctx.series[Timeframe.H1].df
        atr = max(_atr(h1), 1e-12)
        separation = (
            abs(
                float(h1["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                - float(h1["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            )
            / atr
        )
        confidence = min(0.9, 0.5 + separation * 0.2)
        direction = reads[0]
        return Signal(
            module=self.name,
            score=65.0 * direction,
            confidence=confidence,
            reasoning=f"H4 and H1 EMA/momentum aligned {'bullish' if direction > 0 else 'bearish'}",
            invalidation_price=invalidation,
            details=details,
        )


class LiquiditySweep:
    """Detect a wick through a 20-bar extreme that closes back inside."""

    name = "liquidity_sweep"

    def analyze(self, ctx: MarketContext) -> Signal:
        series = ctx.series.get(Timeframe.M15) or ctx.series.get(Timeframe.H1)
        if series is None or len(series.df) < 25:
            return Signal.neutral(self.name, "needs M15 or H1 history")
        frame = series.df
        candle = frame.iloc[-1]
        prior = frame.iloc[-21:-1]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        atr = max(_atr(frame), 1e-12)
        if float(candle["low"]) < prior_low and float(candle["close"]) > prior_low:
            depth = (prior_low - float(candle["low"])) / atr
            return Signal(
                module=self.name,
                score=75.0,
                confidence=min(0.9, 0.55 + depth * 0.25),
                reasoning="sell-side liquidity swept and candle closed back above the range",
                key_levels=(prior_low, prior_high),
                invalidation_price=float(candle["low"]),
                details={"timeframe": series.timeframe.value, "sweep_depth_atr": depth},
            )
        if float(candle["high"]) > prior_high and float(candle["close"]) < prior_high:
            depth = (float(candle["high"]) - prior_high) / atr
            return Signal(
                module=self.name,
                score=-75.0,
                confidence=min(0.9, 0.55 + depth * 0.25),
                reasoning="buy-side liquidity swept and candle closed back below the range",
                key_levels=(prior_low, prior_high),
                invalidation_price=float(candle["high"]),
                details={"timeframe": series.timeframe.value, "sweep_depth_atr": depth},
            )
        return Signal.neutral(self.name, "no confirmed liquidity sweep")


class LevelReaction:
    """Score rejection from a rolling support/resistance zone."""

    name = "level_reaction"

    def analyze(self, ctx: MarketContext) -> Signal:
        series = ctx.series.get(Timeframe.H1)
        if series is None or len(series.df) < 60:
            return Signal.neutral(self.name, "H1 needs 60 closed bars")
        frame = series.df
        candle = frame.iloc[-1]
        history = frame.iloc[-51:-1]
        support = float(history["low"].quantile(0.05))
        resistance = float(history["high"].quantile(0.95))
        atr = max(_atr(frame), 1e-12)
        lower_wick = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
        upper_wick = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
        near_support = abs(float(candle["low"]) - support) <= atr * 0.35
        near_resistance = abs(float(candle["high"]) - resistance) <= atr * 0.35
        if (
            near_support
            and lower_wick > upper_wick * 1.5
            and float(candle["close"]) > float(candle["open"])
        ):
            return Signal(
                module=self.name,
                score=55.0,
                confidence=min(0.8, 0.5 + lower_wick / atr * 0.2),
                reasoning="bullish H1 rejection from rolling support",
                key_levels=(support, resistance),
                invalidation_price=float(candle["low"]),
            )
        if (
            near_resistance
            and upper_wick > lower_wick * 1.5
            and float(candle["close"]) < float(candle["open"])
        ):
            return Signal(
                module=self.name,
                score=-55.0,
                confidence=min(0.8, 0.5 + upper_wick / atr * 0.2),
                reasoning="bearish H1 rejection from rolling resistance",
                key_levels=(support, resistance),
                invalidation_price=float(candle["high"]),
            )
        return Signal.neutral(self.name, "no qualified H1 level rejection")


class VolatilityRegime:
    """Non-directional ATR percentile regime used as a veto/diagnostic."""

    name = "volatility_regime"

    def analyze(self, ctx: MarketContext) -> Signal:
        series = ctx.series.get(Timeframe.H1)
        if series is None or len(series.df) < 120:
            return Signal.neutral(self.name, "H1 needs 120 closed bars")
        frame = series.df
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - frame["close"].shift(1)).abs(),
                (frame["low"] - frame["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atrs = tr.rolling(14).mean().dropna()
        current = float(atrs.iloc[-1])
        percentile = float((atrs.iloc[-100:] <= current).mean())
        regime = "compressed" if percentile < 0.2 else "extreme" if percentile > 0.95 else "normal"
        return Signal(
            module=self.name,
            score=0.0,
            confidence=1.0,
            reasoning=f"H1 volatility regime is {regime}",
            details={"regime": regime, "atr14": current, "percentile": percentile},
        )
