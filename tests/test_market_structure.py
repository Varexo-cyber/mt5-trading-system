"""Market structure: confirmed pivots, BOS/CHoCH, and HTF alignment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from analysis.market_structure import (
    MarketStructure,
    SwingPoint,
    find_equal_levels,
    find_swings,
)
from config.schema import MarketStructureConfig
from core.types import MarketContext, Series, Timeframe

START = datetime(2026, 1, 5, tzinfo=UTC)
BULLISH = [
    1.00,
    1.20,
    1.40,
    1.60,
    1.40,
    1.20,
    1.10,
    1.40,
    1.70,
    2.00,
    1.80,
    1.60,
    1.50,
    1.80,
    2.10,
    2.40,
    2.20,
    2.00,
    1.90,
    2.20,
    2.50,
    2.80,
    2.60,
    2.40,
    2.30,
    2.50,
    2.60,
    2.70,
    2.75,
    3.00,
]


def frame(values: list[float], timeframe: Timeframe) -> pd.DataFrame:
    index = pd.date_range(START, periods=len(values), freq=timeframe.duration, tz=UTC)
    return pd.DataFrame(
        {
            "open": [value - 0.01 for value in values],
            "high": [value + 0.05 for value in values],
            "low": [value - 0.05 for value in values],
            "close": values,
            "tick_volume": [100] * len(values),
            "spread": [10] * len(values),
            "real_volume": [0] * len(values),
        },
        index=index,
    )


def context(signal_values: list[float], bias_values: list[float]) -> MarketContext:
    h1 = frame(signal_values, Timeframe.H1)
    h4 = frame(bias_values, Timeframe.H4)
    now = h1.index[-1].to_pydatetime() + Timeframe.H1.duration
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={
            Timeframe.H1: Series("EURUSD", Timeframe.H1, h1, now),
            Timeframe.H4: Series("EURUSD", Timeframe.H4, h4, now),
        },
    )


@pytest.fixture
def module() -> MarketStructure:
    return MarketStructure(
        MarketStructureConfig(
            atr_period=3,
            bos_close_buffer_atr=0.0,
            external_swing_lookback=3,
            internal_swing_lookback=1,
        )
    )


class TestSwingConfirmation:
    def test_pivot_is_visible_only_after_right_hand_confirmation(self) -> None:
        incomplete = frame([1.0, 2.0], Timeframe.H1)
        complete = frame([1.0, 2.0, 1.0], Timeframe.H1)

        assert find_swings(incomplete, 1, "external") == []
        swings = find_swings(complete, 1, "external")
        high = next(swing for swing in swings if swing.kind == "high")
        assert high.index == 1
        assert high.confirmed_index == 2
        assert high.confirmed_at == complete.index[2].to_pydatetime()

    def test_equal_levels_are_descriptive_pairs(self) -> None:
        swings = [
            SwingPoint("high", "external", 1, 2, START, START, 1.2000),
            SwingPoint("high", "external", 4, 5, START, START + timedelta(hours=3), 1.2005),
        ]
        levels = find_equal_levels(swings, tolerance=0.001)
        assert len(levels) == 1
        assert levels[0].price == pytest.approx(1.20025)


class TestSignals:
    def test_aligned_external_bos_emits_directional_signal(self, module: MarketStructure) -> None:
        signal = module.analyze(context(BULLISH, BULLISH))

        assert signal.score == 70.0
        assert signal.confidence >= 0.5
        assert signal.details["event"] == "BOS"
        assert signal.details["bias_direction"] == "bullish"
        assert signal.invalidation_price == pytest.approx(2.25)

    def test_higher_timeframe_conflict_blocks_the_signal(self, module: MarketStructure) -> None:
        bearish = [4.0 - value for value in BULLISH]
        signal = module.analyze(context(BULLISH, bearish))

        assert signal.score == 0.0
        assert signal.confidence == 0.0
        assert "conflicts" in signal.reasoning

    def test_choch_is_diagnostic_not_a_trigger(self, module: MarketStructure) -> None:
        bearish = [4.0 - value for value in BULLISH]
        bearish[-1] = 4.0
        signal = module.analyze(context(bearish, BULLISH))

        assert signal.score == 0.0
        assert signal.details["event"] == "CHoCH"
        assert "diagnostic only" in signal.reasoning

    def test_break_is_not_repeated_after_the_cross(self, module: MarketStructure) -> None:
        already_broken = BULLISH.copy()
        already_broken[-2] = 3.0
        already_broken[-1] = 3.1
        signal = module.analyze(context(already_broken, BULLISH))

        assert signal.score == 0.0
        assert signal.details["event"] is None

    def test_insufficient_bars_returns_neutral(self, module: MarketStructure) -> None:
        short = BULLISH[:5]
        signal = module.analyze(context(short, short))
        assert signal.score == 0.0
        assert signal.confidence == 0.0
        assert "insufficient" in signal.reasoning


def test_internal_lookback_must_be_smaller() -> None:
    with pytest.raises(ValueError, match="internal swing lookback"):
        MarketStructureConfig(internal_swing_lookback=3, external_swing_lookback=3)
