"""Combine independent analysis modules into one auditable trade idea."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.schema import ConfluenceConfig
from core.types import AnalysisModule, Direction, MarketContext, Signal, Timeframe, TradingMode


@dataclass(frozen=True, slots=True)
class TradeIdea:
    symbol: str
    approved: bool
    direction: Direction | None
    score: float
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reason: str
    signals: tuple[Signal, ...]


class ConfluenceEngine:
    """Weighted agreement with live-module allowlisting and structural stops."""

    def __init__(self, modules: list[AnalysisModule], config: ConfluenceConfig) -> None:
        self.modules = modules
        self.config = config

    def evaluate(self, ctx: MarketContext, mode: TradingMode) -> TradeIdea:
        signals = tuple(module.analyze(ctx) for module in self.modules)
        if ctx.tick is None:
            return self._reject(ctx, signals, "no executable quote")

        regime = next(
            (s.details.get("regime") for s in signals if s.module == "volatility_regime"), None
        )
        if regime == "extreme":
            return self._reject(ctx, signals, "extreme volatility regime")

        allowed_live = set(self.config.live_enabled_modules)
        weighted: list[tuple[Signal, float]] = []
        for signal in signals:
            weight = self.config.weights.get(signal.module, 0.0)
            if mode.is_live and signal.module not in allowed_live:
                weight = 0.0
            if weight > 0 and signal.score and signal.confidence >= self.config.minimum_confidence:
                weighted.append((signal, weight))
        if not weighted:
            suffix = "; no modules validated for live" if mode.is_live and not allowed_live else ""
            return self._reject(ctx, signals, f"no weighted directional evidence{suffix}")

        positive = sum(weight for signal, weight in weighted if signal.score > 0)
        negative = sum(weight for signal, weight in weighted if signal.score < 0)
        direction = Direction.LONG if positive > negative else Direction.SHORT
        agreeing = [
            (signal, weight) for signal, weight in weighted if signal.score * int(direction) > 0
        ]
        agreement = sum(weight for _, weight in agreeing) / sum(weight for _, weight in weighted)
        if len(agreeing) < self.config.minimum_directional_modules:
            return self._reject(ctx, signals, "too few independent directional modules")
        if agreement < self.config.minimum_agreement_ratio:
            return self._reject(
                ctx, signals, f"directional agreement {agreement:.1%} below threshold"
            )

        denominator = sum(weight for _, weight in agreeing)
        score = (
            sum(abs(signal.score) * signal.confidence * weight for signal, weight in agreeing)
            / denominator
        )
        confidence = sum(signal.confidence * weight for signal, weight in agreeing) / denominator
        if score < self.config.score_threshold:
            return self._reject(ctx, signals, f"confluence score {score:.1f} below threshold")

        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid
        frame = ctx.series[Timeframe.H1].df
        atr = self._atr(frame)
        candidates = [
            signal.invalidation_price
            for signal, _ in agreeing
            if signal.invalidation_price is not None
            and (
                (direction is Direction.LONG and signal.invalidation_price < entry)
                or (direction is Direction.SHORT and signal.invalidation_price > entry)
            )
        ]
        if candidates:
            structural = min(candidates) if direction is Direction.LONG else max(candidates)
            stop = (
                structural - atr * 0.25 if direction is Direction.LONG else structural + atr * 0.25
            )
        else:
            stop = entry - atr * self.config.atr_stop_multiple * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return self._reject(ctx, signals, "could not construct a positive stop distance")
        target = entry + risk * self.config.target_r_multiple * int(direction)
        return TradeIdea(
            symbol=ctx.symbol,
            approved=True,
            direction=direction,
            score=score,
            confidence=confidence,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            reason=f"{len(agreeing)} modules agree ({agreement:.0%})",
            signals=signals,
        )

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> float:
        previous = frame["close"].shift(1)
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def _reject(ctx: MarketContext, signals: tuple[Signal, ...], reason: str) -> TradeIdea:
        return TradeIdea(ctx.symbol, False, None, 0.0, 0.0, 0.0, 0.0, 0.0, reason, signals)
