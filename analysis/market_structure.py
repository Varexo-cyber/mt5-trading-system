"""Confirmed swing structure without look-ahead.

A fractal pivot is plotted on the pivot candle but becomes usable only after
its right-hand confirmation window has closed. Keeping both indices explicit is
what prevents a backtest from acting on a swing before it was knowable.

The module emits a directional signal only for an external break of structure
aligned with the higher-timeframe structure. CHoCH and internal breaks remain
diagnostics: promoting either to an entry would be a different hypothesis and
must be pre-registered separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd

from config.schema import MarketStructureConfig
from core.data_manager import atr
from core.types import MarketContext, Signal, Timeframe

SwingKind = Literal["high", "low"]
SwingScope = Literal["internal", "external"]
StructureDirection = Literal["bullish", "bearish", "neutral"]
EventType = Literal["BOS", "CHoCH", "break"]


@dataclass(frozen=True, slots=True)
class SwingPoint:
    kind: SwingKind
    scope: SwingScope
    index: int
    confirmed_index: int
    when: datetime
    confirmed_at: datetime
    price: float


@dataclass(frozen=True, slots=True)
class EqualLevel:
    kind: SwingKind
    price: float
    first: datetime
    second: datetime


@dataclass(frozen=True, slots=True)
class StructureEvent:
    kind: EventType
    direction: StructureDirection
    level: float
    when: datetime
    break_distance_atr: float


@dataclass(frozen=True, slots=True)
class StructureSnapshot:
    timeframe: Timeframe
    atr: float
    internal_direction: StructureDirection
    external_direction: StructureDirection
    internal_swings: tuple[SwingPoint, ...]
    external_swings: tuple[SwingPoint, ...]
    equal_levels: tuple[EqualLevel, ...]
    event: StructureEvent | None
    invalidation_price: float | None


class MarketStructure:
    """Identify confirmed structure and score only HTF-aligned external BOS."""

    name = "market_structure"

    def __init__(self, config: MarketStructureConfig) -> None:
        self.config = config
        self.signal_timeframe = Timeframe.parse(config.signal_timeframe)
        self.bias_timeframe = Timeframe.parse(config.bias_timeframe)

    def analyze(self, ctx: MarketContext) -> Signal:
        if not self.config.enabled:
            return Signal.neutral(self.name, "module disabled")

        signal_snapshot = self.inspect(ctx, self.signal_timeframe)
        bias_snapshot = self.inspect(ctx, self.bias_timeframe)
        if signal_snapshot is None or bias_snapshot is None:
            return Signal.neutral(self.name, "insufficient confirmed bars for structure")

        event = signal_snapshot.event
        details = self._details(signal_snapshot, bias_snapshot)
        if event is None:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning="no new external structure break on the latest closed bar",
                key_levels=self._key_levels(signal_snapshot),
                invalidation_price=signal_snapshot.invalidation_price,
                details=details,
            )
        if event.kind != "BOS":
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=f"{event.kind} is diagnostic only; it is not a registered trigger",
                key_levels=self._key_levels(signal_snapshot),
                invalidation_price=signal_snapshot.invalidation_price,
                details=details,
            )
        if event.direction != bias_snapshot.external_direction:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"{event.direction} H1 BOS conflicts with "
                    f"{bias_snapshot.external_direction} H4 structure"
                ),
                key_levels=self._key_levels(signal_snapshot),
                invalidation_price=signal_snapshot.invalidation_price,
                details=details,
            )

        sign = 1.0 if event.direction == "bullish" else -1.0
        confidence_range = 1.0 - self.config.minimum_confidence
        scaled_break = min(
            event.break_distance_atr / self.config.full_confidence_break_atr,
            1.0,
        )
        confidence = self.config.minimum_confidence + confidence_range * scaled_break
        return Signal(
            module=self.name,
            score=sign * self.config.bos_score,
            confidence=confidence,
            reasoning=(
                f"{event.direction} external BOS aligned with "
                f"{self.bias_timeframe.value} structure; close cleared the swing by "
                f"{event.break_distance_atr:.2f} ATR"
            ),
            key_levels=self._key_levels(signal_snapshot),
            invalidation_price=signal_snapshot.invalidation_price,
            details=details,
        )

    def inspect(self, ctx: MarketContext, timeframe: Timeframe) -> StructureSnapshot | None:
        """Return the chart annotations separately from the trading opinion."""
        frame = ctx.bars(timeframe).df
        needed = max(
            self.config.atr_period + 1,
            self.config.external_swing_lookback * 2 + 2,
        )
        if len(frame) < needed:
            return None

        current_atr = atr(frame, self.config.atr_period)
        internal = find_swings(frame, self.config.internal_swing_lookback, "internal")
        external = find_swings(frame, self.config.external_swing_lookback, "external")
        internal_direction = structure_direction(internal)
        external_direction = structure_direction(external)
        event = detect_break(
            frame,
            external,
            external_direction,
            current_atr * self.config.bos_close_buffer_atr,
            current_atr,
        )
        invalidation = opposing_swing_price(external, event.direction) if event else None
        equals = find_equal_levels(
            external,
            current_atr * self.config.equal_level_tolerance_atr,
        )
        return StructureSnapshot(
            timeframe=timeframe,
            atr=current_atr,
            internal_direction=internal_direction,
            external_direction=external_direction,
            internal_swings=tuple(internal),
            external_swings=tuple(external),
            equal_levels=tuple(equals),
            event=event,
            invalidation_price=invalidation,
        )

    def _key_levels(self, snapshot: StructureSnapshot) -> tuple[float, ...]:
        levels: list[float] = []
        if snapshot.event is not None:
            levels.append(snapshot.event.level)
        if snapshot.invalidation_price is not None:
            levels.append(snapshot.invalidation_price)
        levels.extend(level.price for level in snapshot.equal_levels)
        return tuple(dict.fromkeys(levels))

    def _details(
        self,
        signal: StructureSnapshot,
        bias: StructureSnapshot,
    ) -> dict[str, object]:
        return {
            "signal_timeframe": signal.timeframe.value,
            "bias_timeframe": bias.timeframe.value,
            "internal_direction": signal.internal_direction,
            "external_direction": signal.external_direction,
            "bias_direction": bias.external_direction,
            "event": signal.event.kind if signal.event else None,
            "event_direction": signal.event.direction if signal.event else None,
            "atr": signal.atr,
            "external_swings": len(signal.external_swings),
            "internal_swings": len(signal.internal_swings),
            "equal_levels": len(signal.equal_levels),
        }


def find_swings(df: pd.DataFrame, lookback: int, scope: SwingScope) -> list[SwingPoint]:
    """Find pivots whose full left and right windows are closed.

    A pivot high is the strict maximum of the window centred on it — strict
    meaning it appears exactly once, so a flat double top is not two pivots.

    VECTORISED, AND THAT IS NOT COSMETIC. Written as a Python loop this asked
    numpy for four small reductions per bar: about 4,800 per decision, and at
    two hundred H1 bars across five timeframes it was a third of the entire
    module backtest. Eighty-four minutes for twenty symbols over 180 days, and
    a measurement nobody will wait for is a measurement that does not get run —
    which is how this project spent a day changing a live account on two or
    three trades at a time.

    A centred rolling max is one strided view and one reduction over an axis.
    The uniqueness test rides on the same view. `test_market_structure` holds
    the result identical to the loop it replaces, because a faster answer that
    is a different answer is worse than the slow one.
    """
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    span = 2 * lookback + 1
    if len(df) <= 2 * lookback:
        return []

    # Pivot p is the centre of window p - lookback, and there are exactly as
    # many windows as pivots: len - span + 1 == len - 2 * lookback. Trimming a
    # window here dropped the only pivot of a three-bar frame, which is the
    # case `test_pivot_is_visible_only_after_right_hand_confirmation` exists for.
    windows_high = np.lib.stride_tricks.sliding_window_view(highs, span)
    windows_low = np.lib.stride_tricks.sliding_window_view(lows, span)
    centres = np.arange(lookback, len(df) - lookback)
    centre_high = highs[centres]
    centre_low = lows[centres]

    is_high = (windows_high.max(axis=1) == centre_high) & (
        (windows_high == centre_high[:, None]).sum(axis=1) == 1
    )
    is_low = (windows_low.min(axis=1) == centre_low) & (
        (windows_low == centre_low[:, None]).sum(axis=1) == 1
    )

    # One vectorised conversion instead of boxing a pandas Timestamp per pivot
    # — 126,000 of those calls in a 150-decision profile.
    #
    # A caller may hand this a frame with no dates at all: the position guard
    # builds one on a RangeIndex, where the old per-element access was tolerated
    # and this is not. Falling back rather than demanding a DatetimeIndex,
    # because refusing a frame the previous implementation accepted would be a
    # behaviour change smuggled in with a speedup.
    index = df.index
    moments = index.to_pydatetime() if hasattr(index, "to_pydatetime") else list(index)
    swings: list[SwingPoint] = []
    for offset, pivot in enumerate(centres):
        confirmed = int(pivot) + lookback
        if is_high[offset]:
            swings.append(
                SwingPoint(
                    kind="high",
                    scope=scope,
                    index=int(pivot),
                    confirmed_index=confirmed,
                    when=moments[pivot],
                    confirmed_at=moments[confirmed],
                    price=float(centre_high[offset]),
                )
            )
        if is_low[offset]:
            swings.append(
                SwingPoint(
                    kind="low",
                    scope=scope,
                    index=int(pivot),
                    confirmed_index=confirmed,
                    when=moments[pivot],
                    confirmed_at=moments[confirmed],
                    price=float(centre_low[offset]),
                )
            )
    return sorted(swings, key=lambda swing: (swing.index, swing.kind))


def structure_direction(swings: list[SwingPoint]) -> StructureDirection:
    highs = [swing for swing in swings if swing.kind == "high"]
    lows = [swing for swing in swings if swing.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "neutral"
    higher_high = highs[-1].price > highs[-2].price
    higher_low = lows[-1].price > lows[-2].price
    lower_high = highs[-1].price < highs[-2].price
    lower_low = lows[-1].price < lows[-2].price
    if higher_high and higher_low:
        return "bullish"
    if lower_high and lower_low:
        return "bearish"
    return "neutral"


def detect_break(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    prior_direction: StructureDirection,
    buffer: float,
    current_atr: float,
) -> StructureEvent | None:
    if len(df) < 2 or not swings or current_atr <= 0:
        return None
    previous_close = float(df["close"].iloc[-2])
    current_close = float(df["close"].iloc[-1])
    latest_high = next((s for s in reversed(swings) if s.kind == "high"), None)
    latest_low = next((s for s in reversed(swings) if s.kind == "low"), None)
    when = df.index[-1].to_pydatetime()

    if latest_high is not None:
        threshold = latest_high.price + buffer
        if previous_close <= threshold < current_close:
            event_type: EventType = "BOS" if prior_direction == "bullish" else "break"
            if prior_direction == "bearish":
                event_type = "CHoCH"
            return StructureEvent(
                kind=event_type,
                direction="bullish",
                level=latest_high.price,
                when=when,
                break_distance_atr=(current_close - latest_high.price) / current_atr,
            )

    if latest_low is not None:
        threshold = latest_low.price - buffer
        if previous_close >= threshold > current_close:
            event_type = "BOS" if prior_direction == "bearish" else "break"
            if prior_direction == "bullish":
                event_type = "CHoCH"
            return StructureEvent(
                kind=event_type,
                direction="bearish",
                level=latest_low.price,
                when=when,
                break_distance_atr=(latest_low.price - current_close) / current_atr,
            )
    return None


def opposing_swing_price(swings: list[SwingPoint], direction: StructureDirection) -> float | None:
    opposing: SwingKind = "low" if direction == "bullish" else "high"
    match = next((swing for swing in reversed(swings) if swing.kind == opposing), None)
    return match.price if match else None


def find_equal_levels(swings: list[SwingPoint], tolerance: float) -> list[EqualLevel]:
    """Pair adjacent same-kind pivots without turning them into extra votes."""
    levels: list[EqualLevel] = []
    for kind in ("high", "low"):
        same_kind = [swing for swing in swings if swing.kind == kind]
        for first, second in pairwise(same_kind):
            if abs(first.price - second.price) <= tolerance:
                levels.append(
                    EqualLevel(
                        kind=kind,
                        price=(first.price + second.price) / 2.0,
                        first=first.when,
                        second=second.when,
                    )
                )
    return sorted(levels, key=lambda level: level.second)
