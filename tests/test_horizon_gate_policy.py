"""Quick trades are judged on quick evidence without turning safety gates off."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.confluence import TradeIdea
from analysis.target_reach import ReachVerdict
from config.loader import load_settings
from core.types import Direction, MarketContext, Series, Signal, Timeframe
from runner.service import JarvisRunner


def runner() -> JarvisRunner:
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = load_settings(env_overrides=False)  # type: ignore[assignment]
    return instance


def idea(*, horizon: str, family: str, signals: tuple[Signal, ...] = ()) -> TradeIdea:
    return TradeIdea(
        symbol="EURUSD.i",
        approved=True,
        direction=Direction.LONG,
        score=70.0,
        confidence=0.7,
        entry=1.10,
        stop_loss=1.099,
        take_profit=1.102,
        reason="test",
        signals=signals,
        setup_family=family,
        horizon=horizon,
        planning_timeframe="M5" if horizon == "quick" else "H1",
        expected_horizon_minutes=30 if horizon == "quick" else 1440,
    )


def context() -> MarketContext:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    series: dict[Timeframe, Series] = {}
    for timeframe, frequency in ((Timeframe.M1, "1min"), (Timeframe.M5, "5min")):
        index = pd.date_range(end=now, periods=80, freq=frequency)
        close = 1.10 + np.arange(80) * 0.00001
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
            },
            index=index,
        )
        series[timeframe] = Series("EURUSD.i", timeframe, frame, now)
    return MarketContext("EURUSD.i", now, series, None)


def test_quick_only_softens_plausible_base_rate_failures() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")

    plausible = ReachVerdict(200, 20.0, 27.0, 33.3)
    fantasy = ReachVerdict(200, 10.0, 27.0, 33.3)

    assert service._reach_failure_is_advisory(quick, plausible)
    assert not service._reach_failure_is_advisory(quick, fantasy)
    assert not service._reach_failure_is_advisory(swing, plausible)


def test_large_direction_disadvantage_remains_hard() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")

    assert service._direction_failure_is_advisory(quick, ReachVerdict(200, 30.0, 42.0, 33.3))
    assert not service._direction_failure_is_advisory(quick, ReachVerdict(200, 20.0, 45.0, 33.3))


def test_quick_relaxation_does_not_relax_adverse_reversal_limit() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")
    quick_config = service._entry_quality_config_for(quick)
    swing_config = service._entry_quality_config_for(swing)

    assert (
        quick_config.max_favourable_extension_atr["forex"]
        > (swing_config.max_favourable_extension_atr["forex"])
    )
    assert quick_config.max_last_bar_adverse_atr == swing_config.max_last_bar_adverse_atr


def test_new_m1_event_gets_its_own_review_identity() -> None:
    service = runner()
    signal = Signal(
        "m1_micro_breakout",
        75.0,
        0.8,
        details={"timeframe": "M1"},
    )
    quick = idea(
        horizon="quick",
        family="m1_micro_breakout_m1",
        signals=(signal,),
    )

    key = service._review_key(quick, context())

    assert key is not None
    assert key[4] == "M1"


def test_broad_old_veto_does_not_silence_a_new_quick_event() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")

    assert not service._broad_veto_memory_applies(quick)
    assert service._broad_veto_memory_applies(swing)
