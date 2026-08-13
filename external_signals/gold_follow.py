"""Deterministic protection for authenticated Rio Gold directions.

Rio remains the directional source. These calculations do not predict direction;
they only turn an incomplete instruction into a broker-valid, bounded-risk plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.data_manager import atr
from core.types import Direction
from external_signals.models import ExternalSignalEvent


@dataclass(frozen=True, slots=True)
class GoldFollowPlan:
    stop_loss: float
    take_profit: float
    measured_atr: float
    used_fallback_stop: bool
    used_fallback_target: bool
    provider_entry: float | None
    maximum_entry_deviation: float


def build_gold_follow_plan(
    event: ExternalSignalEvent,
    *,
    live_entry: float,
    bars: pd.DataFrame,
    spread: float,
    minimum_stop_distance: float,
    atr_period: int,
    structure_bars: int,
    stop_atr_multiple: float,
    structure_buffer_atr: float,
    target_reward_risk: float,
    max_entry_deviation_atr: float,
    max_entry_deviation_bps: float,
) -> GoldFollowPlan:
    if event.direction is None:
        raise ValueError("Gold follow signal has no direction")
    measured_atr = atr(bars, atr_period)
    sign = int(event.direction)
    minimum_distance = max(
        minimum_stop_distance * 2.0,
        spread * 3.0,
        measured_atr * stop_atr_multiple,
    )
    provider_entry = event.provider_entry
    maximum_deviation = max(
        measured_atr * max_entry_deviation_atr,
        live_entry * max_entry_deviation_bps / 10_000.0,
    )
    if provider_entry is not None and abs(live_entry - provider_entry) > maximum_deviation:
        raise ValueError(
            f"live price {live_entry:g} moved {abs(live_entry - provider_entry):g} from "
            f"provider entry {provider_entry:g}; maximum fresh-follow distance is "
            f"{maximum_deviation:g}"
        )

    stop = event.stop_loss
    stop_valid = bool(stop and (live_entry - float(stop)) * sign > minimum_stop_distance)
    if not stop_valid:
        recent = bars.tail(structure_bars)
        buffer = measured_atr * structure_buffer_atr
        volatility_stop = live_entry - sign * minimum_distance
        if event.direction is Direction.LONG:
            structure_stop = float(recent["low"].min()) - buffer
            stop = min(volatility_stop, structure_stop)
        else:
            structure_stop = float(recent["high"].max()) + buffer
            stop = max(volatility_stop, structure_stop)
    assert stop is not None
    risk_distance = abs(live_entry - stop)

    target = event.take_profits[0] if event.take_profits else None
    target_valid = bool(target and (float(target) - live_entry) * sign > minimum_stop_distance)
    if not target_valid:
        target = live_entry + sign * risk_distance * target_reward_risk
    assert target is not None
    return GoldFollowPlan(
        stop_loss=float(stop),
        take_profit=float(target),
        measured_atr=measured_atr,
        used_fallback_stop=not stop_valid,
        used_fallback_target=not target_valid,
        provider_entry=provider_entry,
        maximum_entry_deviation=maximum_deviation,
    )
