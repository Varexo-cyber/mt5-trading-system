"""Combine independent analysis modules into one auditable trade idea."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
            return self._reject(
                ctx, signals, f"confluence score {score:.1f} below threshold", score, confidence
            )

        adverse = self._entry_timing_conflict(ctx, direction)
        if adverse is not None:
            return self._reject(ctx, signals, adverse, score, confidence)

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
            return self._reject(
                ctx, signals, "could not construct a positive stop distance", score, confidence
            )
        target, target_note = self._reachable_target(ctx, entry, risk, direction)
        if target is None:
            return self._reject(ctx, signals, target_note, score, confidence)
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

    def _reachable_target(
        self, ctx: MarketContext, entry: float, risk: float, direction: Direction
    ) -> tuple[float | None, str]:
        """Place the target where this market actually goes, not where R says.

        `entry + 2R` is arithmetic. It never asks whether the instrument travels
        that far, so a slow market gets a target it reaches once a month and the
        trade becomes, in practice, a bet on the stop not being hit — the reward
        half of the reward-to-risk never arrives.

        So the distance is also measured against the instrument's own history:
        how far it has moved in the proposed direction within `horizon_bars`,
        taken at a percentile rather than a maximum so one violent week does not
        set the expectation. The target is the smaller of the two.

        A floor protects the other side. Shrinking the target indefinitely would
        buy a high hit rate with trades that cannot pay for their own spread, so
        below `minimum_r` the setup is rejected outright rather than sized down
        into something not worth taking.
        """
        config = self.config
        planned = risk * config.target_r_multiple
        signal = ctx.series.get(Timeframe.H1)
        if signal is None or len(signal.df) < config.target_horizon_bars * 3:
            return entry + planned * int(direction), "no history to bound the target"

        frame = signal.df.tail(400)
        closes = frame["close"].to_numpy()
        extremes = (frame["high"] if direction is Direction.LONG else frame["low"]).to_numpy()
        horizon = config.target_horizon_bars
        windows = len(closes) - horizon
        if windows <= 0:
            return entry + planned * int(direction), "no history to bound the target"

        # Favourable excursion: how far price ran our way from each starting bar.
        runs = [
            (
                extremes[start + 1 : start + 1 + horizon].max() - closes[start]
                if direction is Direction.LONG
                else closes[start] - extremes[start + 1 : start + 1 + horizon].min()
            )
            for start in range(windows)
        ]
        typical = float(np.quantile(runs, config.target_reach_quantile))

        if typical <= 0:
            return None, "this market has not moved in this direction over the horizon"

        distance = min(planned, typical)
        achieved_r = distance / risk
        if achieved_r < config.minimum_r_multiple:
            return None, (
                f"a reachable target is only {achieved_r:.2f}R — this market travels "
                f"{typical:.5f} in {horizon} bars against a {risk:.5f} stop, below the "
                f"{config.minimum_r_multiple:.2f}R minimum"
            )
        note = "planned" if distance >= planned else f"trimmed to {achieved_r:.2f}R"
        return entry + distance * int(direction), note

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
    def _reject(
        ctx: MarketContext,
        signals: tuple[Signal, ...],
        reason: str,
        score: float = 0.0,
        confidence: float = 0.0,
    ) -> TradeIdea:
        """A rejected idea, carrying the score it reached where one was computed.

        Returning a flat zero threw away the only number that distinguishes
        "the modules saw nothing" from "the modules saw something and the
        threshold is out of reach". Both land in the journal as NO_SIGNAL, and
        with the score blanked the two are indistinguishable — which is exactly
        the question an operator asks after a day with no trades.
        """
        return TradeIdea(ctx.symbol, False, None, score, confidence, 0.0, 0.0, 0.0, reason, signals)
