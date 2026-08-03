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

        adverse = self._entry_timing_conflict(ctx, direction)
        if adverse is not None:
            return self._reject(ctx, signals, adverse)

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

    def _entry_timing_conflict(self, ctx: MarketContext, direction: Direction) -> str | None:
        """Refuse an entry the immediate price action is moving against.

        The engine went straight from an H4/H1 bias to an entry at the current
        ask, with nothing between. The plan was always "higher timeframe bias,
        middle timeframe zone, lower timeframe timing" and the timing step did
        not exist, so a long was proposed at whatever price happened to be
        printing — including while the last hour was selling into it.

        Claude caught exactly this and nothing else. Eleven of the first twelve
        reviews were vetoes, and the recurring sentence was "lower-timeframe
        (M15/M5) price action is falling into the entry, directly opposing the
        long". Encoding that here is cheaper than paying for the same finding
        once per candidate, and it is a real gate rather than a stricter
        threshold: it rejects on evidence that contradicts the setup, not on the
        setup being merely unremarkable.

        Deliberately one-sided. It never *creates* a signal, and a flat lower
        timeframe is not an objection — only a move materially against the
        proposed direction is.
        """
        for timeframe in self.config.entry_timing_timeframes:
            series = ctx.series.get(Timeframe(timeframe))
            if series is None or len(series.df) < 20:
                continue
            frame = series.df
            atr = self._atr(frame)
            if atr <= 0:
                continue
            bars = self.config.entry_timing_lookback
            move = float(frame["close"].iloc[-1]) - float(frame["close"].iloc[-1 - bars])
            adverse_atr = -(move * int(direction)) / atr
            if adverse_atr > self.config.entry_timing_max_adverse_atr:
                return (
                    f"{timeframe} price is moving against the {direction.name.lower()}: "
                    f"{adverse_atr:.2f} ATR adverse over the last {bars} closed bars, "
                    f"limit {self.config.entry_timing_max_adverse_atr:.2f}"
                )
        return None

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
