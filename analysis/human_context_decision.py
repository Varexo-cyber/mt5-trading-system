"""One explainable top-down decision process, not another isolated indicator."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.schema import HumanContextDecisionConfig
from core.data_manager import atr
from core.types import MarketContext, Signal, Timeframe


@dataclass(frozen=True, slots=True)
class _Story:
    name: str
    direction: int
    invalidation: float
    level: float
    objective: float
    explanation: str


class HumanContextDecision:
    """Require context, event, closed-bar confirmation and structural invalidation."""

    name = "human_context_decision"

    def __init__(self, config: HumanContextDecisionConfig) -> None:
        self.config = config
        self.entry_timeframe = Timeframe.parse(config.entry_timeframe)
        self.context_timeframes = tuple(Timeframe.parse(v) for v in config.context_timeframes)
        self._last_signal_bar: dict[str, object] = {}

    def analyze(self, ctx: MarketContext) -> Signal:
        c = self.config
        if not c.enabled or ctx.symbol not in c.allowed_symbols:
            return Signal.neutral(self.name, "human context decision disabled for this market")
        entry_series = ctx.series.get(self.entry_timeframe)
        context = [ctx.series.get(tf) for tf in self.context_timeframes]
        if entry_series is None or any(series is None for series in context):
            return Signal.neutral(self.name, "needs closed M5, M15, H1 and H4 candles")
        frame = entry_series.df
        needed = max(c.atr_period + 2, c.level_lookback + 3, c.slope_bars + 2)
        if len(frame) < needed:
            return Signal.neutral(self.name, "insufficient closed candles for a market story")
        unit = atr(frame.iloc[:-1], c.atr_period)
        if not pd.notna(unit) or unit <= 0:
            return Signal.neutral(self.name, "entry ATR is unavailable")

        biases = tuple(
            self._bias(series.df, c.slope_bars) for series in context if series is not None
        )
        stories = self._stories(frame, float(unit), biases)
        if not stories:
            return self._neutral(
                "no coherent continuation, failed-auction or break-retest story", biases
            )
        directions = {story.direction for story in stories}
        if len(directions) != 1:
            return self._neutral("supported market stories disagree on direction", biases)
        priority = {name: index for index, name in enumerate(c.story_priority)}
        story = max(
            stories,
            key=lambda item: (
                self._context_votes(item.direction, biases),
                -priority.get(item.name, len(priority)),
            ),
        )
        votes = self._context_votes(story.direction, biases)
        if votes < c.minimum_context_agreement:
            return self._neutral(
                "higher timeframes do not provide enough directional control", biases
            )

        entry = ctx.tick.mid if ctx.tick is not None else float(frame.close.iloc[-1])
        risk = (entry - story.invalidation) * story.direction
        if risk <= 0 or not c.minimum_stop_atr * unit <= risk <= c.maximum_stop_atr * unit:
            return self._neutral(
                "structural invalidation is not executable at this volatility", biases
            )
        reward_r = (story.objective - entry) * story.direction / risk
        if reward_r < c.minimum_reward_r:
            return self._neutral(
                "opposing liquidity does not leave the required reward after the stop",
                biases,
            )
        stamp = frame.index[-1]
        if self._last_signal_bar.get(ctx.symbol) == stamp:
            return self._neutral("this completed M5 decision was already emitted", biases)
        self._last_signal_bar[ctx.symbol] = stamp
        direction_name = "long" if story.direction > 0 else "short"
        return Signal(
            module=self.name,
            score=story.direction * c.score,
            confidence=c.confidence,
            reasoning=(
                f"{story.name} {direction_name}: {story.explanation}; "
                f"{votes}/{len(biases)} higher timeframes agree; M5 closed confirmation"
            ),
            invalidation_price=story.invalidation,
            key_levels=(story.level, story.invalidation, story.objective),
            details={
                "timeframe": c.entry_timeframe,
                "story": story.name,
                "supported_stories": tuple(item.name for item in stories),
                "context_biases": biases,
                "context_votes": votes,
                "confirmation": "closed_m5",
                "risk_atr": risk / unit,
                "objective": story.objective,
                "available_reward_r": reward_r,
            },
        )

    def _stories(self, frame: pd.DataFrame, unit: float, biases: tuple[int, ...]) -> list[_Story]:
        c = self.config
        bar = frame.iloc[-1]
        previous = frame.iloc[-2]
        history = frame.iloc[-(c.level_lookback + 1) : -1]
        high = float(history.high.max())
        low = float(history.low.min())
        open_, close = float(bar.open), float(bar.close)
        bar_high, bar_low = float(bar.high), float(bar.low)
        span = max(bar_high - bar_low, unit * 0.01)
        body = abs(close - open_)
        location = (close - bar_low) / span
        bullish = (
            close > open_ and body >= c.minimum_body_atr * unit and location >= c.close_location
        )
        bearish = (
            close < open_
            and body >= c.minimum_body_atr * unit
            and location <= 1.0 - c.close_location
        )
        stories: list[_Story] = []

        dominant = 1 if sum(biases) > 0 else -1 if sum(biases) < 0 else 0
        ema = frame.close.astype(float).ewm(span=c.context_bars, adjust=False).mean()
        touched = (
            float(bar.low) <= float(ema.iloc[-1]) + c.pullback_tolerance_atr * unit
            and float(bar.high) >= float(ema.iloc[-1]) - c.pullback_tolerance_atr * unit
        )
        if dominant > 0 and touched and bullish and close > float(previous.close):
            stop = min(float(bar.low), float(frame.low.iloc[-3:].min())) - c.stop_buffer_atr * unit
            stories.append(
                _Story(
                    "trend_continuation",
                    1,
                    stop,
                    float(ema.iloc[-1]),
                    high,
                    "pullback rejected above the local value line",
                )
            )
        if dominant < 0 and touched and bearish and close < float(previous.close):
            stop = (
                max(float(bar.high), float(frame.high.iloc[-3:].max())) + c.stop_buffer_atr * unit
            )
            stories.append(
                _Story(
                    "trend_continuation",
                    -1,
                    stop,
                    float(ema.iloc[-1]),
                    low,
                    "pullback rejected below the local value line",
                )
            )

        if bar_low <= low - c.sweep_excursion_atr * unit and close > low and bullish:
            stories.append(
                _Story(
                    "failed_auction",
                    1,
                    bar_low - c.stop_buffer_atr * unit,
                    low,
                    high,
                    "sell-side liquidity was swept and the candle reclaimed the range",
                )
            )
        if bar_high >= high + c.sweep_excursion_atr * unit and close < high and bearish:
            stories.append(
                _Story(
                    "failed_auction",
                    -1,
                    bar_high + c.stop_buffer_atr * unit,
                    high,
                    low,
                    "buy-side liquidity was swept and the candle reclaimed the range",
                )
            )

        prior = frame.iloc[-(c.level_lookback + 2) : -2]
        prior_high, prior_low = float(prior.high.max()), float(prior.low.min())
        broke_up = float(previous.close) > prior_high + c.break_buffer_atr * unit
        broke_down = float(previous.close) < prior_low - c.break_buffer_atr * unit
        if (
            broke_up
            and bar_low <= prior_high + c.pullback_tolerance_atr * unit
            and close > prior_high
            and bullish
        ):
            stories.append(
                _Story(
                    "break_retest",
                    1,
                    bar_low - c.stop_buffer_atr * unit,
                    prior_high,
                    prior_high + (prior_high - prior_low),
                    "broken resistance was retested and defended on close",
                )
            )
        if (
            broke_down
            and bar_high >= prior_low - c.pullback_tolerance_atr * unit
            and close < prior_low
            and bearish
        ):
            stories.append(
                _Story(
                    "break_retest",
                    -1,
                    bar_high + c.stop_buffer_atr * unit,
                    prior_low,
                    prior_low - (prior_high - prior_low),
                    "broken support was retested and defended on close",
                )
            )
        return stories

    @staticmethod
    def _bias(frame: pd.DataFrame, bars: int) -> int:
        close = frame.close.astype(float)
        if len(close) < bars + 1:
            return 0
        delta = float(close.iloc[-1] - close.iloc[-bars])
        return 1 if delta > 0 else -1 if delta < 0 else 0

    @staticmethod
    def _context_votes(direction: int, biases: tuple[int, ...]) -> int:
        return sum(value == direction for value in biases)

    def _neutral(self, reason: str, biases: tuple[int, ...]) -> Signal:
        return Signal(
            module=self.name,
            score=0.0,
            confidence=0.0,
            reasoning=reason,
            details={"timeframe": self.config.entry_timeframe, "context_biases": biases},
        )
