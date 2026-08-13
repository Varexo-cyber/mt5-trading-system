"""A short plan must not inherit a swing plan's chart authority."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.confluence import ConfluenceEngine
from config.loader import load_settings
from core.types import Direction, MarketContext, Series, Signal, Tick, Timeframe, TradingMode

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def bars(timeframe: Timeframe, *, drift: float = 0.0, count: int = 500) -> Series:
    step = timeframe.duration
    index = pd.DatetimeIndex([NOW - step * (count - i) for i in range(count)])
    wave = np.sin(np.arange(count) / 4.0) * 0.35
    close = 100.0 + np.arange(count) * drift + wave
    frame = pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "tick_volume": np.full(count, 100),
            "spread": np.zeros(count),
            "real_volume": np.zeros(count),
        },
        index=index,
    )
    return Series("TEST", timeframe, frame, NOW)


class SweepOnly:
    name = "liquidity_sweep"

    def analyze(self, ctx: MarketContext) -> Signal:
        entry = ctx.tick.ask if ctx.tick else 100.0
        return Signal(
            module=self.name,
            score=75.0,
            confidence=0.9,
            reasoning="confirmed M15 sell-side sweep",
            invalidation_price=entry - 0.8,
            details={"timeframe": "M15"},
        )


class PullbackOnly:
    name = "ema_pullback_resume"

    def analyze(self, ctx: MarketContext) -> Signal:
        entry = ctx.tick.ask if ctx.tick else 100.0
        return Signal(
            module=self.name,
            score=58.0,
            confidence=0.8,
            reasoning="M5 EMA pullback reclaimed",
            invalidation_price=entry - 0.8,
            details={"timeframe": "M5"},
        )


def context(*, h4_drift: float = 0.0, d1_drift: float = 0.0) -> MarketContext:
    def drift_for(timeframe: Timeframe) -> float:
        if timeframe is Timeframe.H4:
            return h4_drift
        if timeframe is Timeframe.D1:
            return d1_drift
        if timeframe is Timeframe.M15:
            return 0.1
        return 0.0

    series = {
        timeframe: bars(timeframe, drift=drift_for(timeframe))
        for timeframe in (
            Timeframe.W1,
            Timeframe.D1,
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        )
    }
    last = series[Timeframe.M15].last_close
    return MarketContext(
        "TEST",
        NOW,
        series,
        Tick("TEST", NOW, bid=last - 0.01, ask=last + 0.01),
    )


def engine() -> ConfluenceEngine:
    base = load_settings(env_overrides=False).analysis.confluence
    config = base.model_copy(
        update={
            "weights": {"liquidity_sweep": 1.0},
            "minimum_directional_modules": 1,
            "minimum_agreement_ratio": 0.5,
            "minimum_confidence": 0.1,
            "score_threshold": 1.0,
        }
    )
    return ConfluenceEngine([SweepOnly()], config)


def quick_engine() -> ConfluenceEngine:
    base = load_settings(env_overrides=False).analysis.confluence
    config = base.model_copy(
        update={
            "weights": {"ema_pullback_resume": 1.0},
            "minimum_directional_modules": 1,
            "minimum_agreement_ratio": 0.5,
            "minimum_confidence": 0.1,
            "score_threshold": 1.0,
            "minimum_r_multiple": 0.5,
        }
    )
    return ConfluenceEngine([PullbackOnly()], config)


def test_standalone_m15_sweep_gets_an_intraday_plan() -> None:
    idea = engine().evaluate(context(), TradingMode.PAPER)

    assert idea.approved, idea.reason
    assert idea.horizon == "intraday"
    assert idea.setup_family == "liquidity_sweep_m15"
    assert idea.planning_timeframe == "M15"
    assert idea.expected_horizon_minutes == 180


def test_m5_pullback_gets_a_genuinely_quick_plan() -> None:
    idea = quick_engine().evaluate(context(), TradingMode.PAPER)

    assert idea.approved, idea.reason
    assert idea.horizon == "quick"
    assert idea.setup_family == "ema_pullback_resume_m5"
    assert idea.planning_timeframe == "M5"
    assert idea.expected_horizon_minutes == 30


def test_one_slow_countertrend_warns_but_does_not_veto_intraday() -> None:
    profile = engine().config.horizon_profiles["intraday"]
    said = engine().higher_timeframe_conflict(
        context(h4_drift=0.0, d1_drift=0.5),
        Direction.SHORT,
        timeframes=profile.htf_trend_timeframes,
        threshold=profile.htf_trend_veto,
        minimum_conflicts=profile.minimum_htf_conflicts,
    )

    assert said is None


def test_two_slow_countertrends_still_veto_intraday() -> None:
    profile = engine().config.horizon_profiles["intraday"]
    said = engine().higher_timeframe_conflict(
        context(h4_drift=0.5, d1_drift=0.5),
        Direction.SHORT,
        timeframes=profile.htf_trend_timeframes,
        threshold=profile.htf_trend_veto,
        minimum_conflicts=profile.minimum_htf_conflicts,
    )

    assert said is not None
    assert "H4" in said and "D1" in said
