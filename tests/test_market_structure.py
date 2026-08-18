"""Market structure: confirmed pivots, BOS/CHoCH, and HTF alignment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
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


class TestTheVectorisedSwingSearchIsTheSameSearch:
    """A third of the module backtest was four small numpy reductions per bar.

    `find_swings` asked numpy for a max, a min and two equality counts on a
    fresh slice for every pivot — around 4,800 reductions per decision across
    five timeframes, and eighty-four minutes for twenty symbols over 180 days.
    A measurement nobody will wait for does not get run, and this project spent
    a day changing a live account on two or three trades at a time for exactly
    that reason.

    A centred rolling max is one strided view and one reduction. Faster is
    worthless if it is also different, so the loop it replaced is kept here as
    the reference and the two are compared field by field — including on the
    shapes that break naive vectorisation: a frame exactly long enough for one
    pivot, a flat series with no strict extreme, and a plateau where the
    uniqueness test is the only thing separating a pivot from a shelf.
    """

    @staticmethod
    def reference(df, lookback, scope):  # type: ignore[no-untyped-def]
        """The implementation this replaced, kept verbatim."""
        swings = []
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        index = df.index
        for pivot in range(lookback, len(df) - lookback):
            left, right = pivot - lookback, pivot + lookback + 1
            high_window, low_window = highs[left:right], lows[left:right]
            confirmed = pivot + lookback
            if highs[pivot] == high_window.max() and (high_window == highs[pivot]).sum() == 1:
                swings.append(
                    SwingPoint(
                        kind="high",
                        scope=scope,
                        index=pivot,
                        confirmed_index=confirmed,
                        when=index[pivot].to_pydatetime(),
                        confirmed_at=index[confirmed].to_pydatetime(),
                        price=float(highs[pivot]),
                    )
                )
            if lows[pivot] == low_window.min() and (low_window == lows[pivot]).sum() == 1:
                swings.append(
                    SwingPoint(
                        kind="low",
                        scope=scope,
                        index=pivot,
                        confirmed_index=confirmed,
                        when=index[pivot].to_pydatetime(),
                        confirmed_at=index[confirmed].to_pydatetime(),
                        price=float(lows[pivot]),
                    )
                )
        return sorted(swings, key=lambda swing: (swing.index, swing.kind))

    @pytest.mark.parametrize("lookback", [1, 2, 3, 5])
    @pytest.mark.parametrize("seed", [1, 7, 19])
    def test_a_random_walk_gives_identical_swings(self, lookback: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        closes = list(1.10 + np.cumsum(rng.normal(0, 0.0004, 260)))

        data = frame(closes, Timeframe.H1)

        assert find_swings(data, lookback, "external") == self.reference(data, lookback, "external")

    def test_a_plateau_is_not_a_pivot(self) -> None:
        """The uniqueness test is the whole difference between a pivot and a
        shelf, and it is the part a rolling max alone would get wrong."""
        data = frame([1.0, 2.0, 2.0, 1.0, 0.5], Timeframe.H1)

        assert find_swings(data, 1, "external") == self.reference(data, 1, "external")

    def test_a_flat_series_has_no_strict_extreme(self) -> None:
        data = frame([1.0] * 20, Timeframe.H1)

        assert find_swings(data, 2, "external") == []

    @pytest.mark.parametrize("bars", [0, 1, 2, 3, 4, 5])
    def test_the_short_frames_agree_too(self, bars: int) -> None:
        """Exactly at the boundary is where the trim was wrong: a three-bar
        frame with lookback 1 holds precisely one pivot, and dropping a window
        threw it away."""
        data = frame([1.0, 2.0, 1.0, 3.0, 1.0][:bars] or [1.0], Timeframe.H1)

        assert find_swings(data, 1, "external") == self.reference(data, 1, "external")

    def test_a_frame_without_dates_is_still_accepted(self) -> None:
        """The position guard builds frames on a RangeIndex. Per-element access
        tolerated that; a single vectorised `to_pydatetime` does not, and
        refusing a frame the previous implementation accepted would be a
        behaviour change smuggled in with a speedup."""
        data = pd.DataFrame({"high": [1.0, 3.0, 1.0, 2.0, 1.0], "low": [0.5] * 5})

        swings = find_swings(data, 1, "external")

        # Two strict local highs at 3 and 2, both with a lower bar either side.
        assert [swing.index for swing in swings if swing.kind == "high"] == [1, 3]
        assert swings[0].when == 1
