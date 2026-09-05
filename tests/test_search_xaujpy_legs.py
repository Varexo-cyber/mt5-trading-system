"""XAUJPY against its own two legs: the first mechanism here with a counterparty.

Everything section eleven ran was a generic bar pattern written for index CFDs.
`streak_reversal` -- "four closes the same way, then trade against it" -- fires
on 10.3% of all bars and lost 0.18 R a trade, because nobody loses money when
four one-minute candles go up. There was no mechanism to find.

    XAUJPY = XAUUSD x USDJPY

is different in kind: two legs in two books, and a cross quote that follows them
with a lag. The counterparty is the maker whose cross has not caught up yet.

The two tests that matter here are the two ways this can be wrong: finding a gap
that is really a rounding error, and missing one that is really there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.search_xaujpy_legs import (
    DEAD_GAP_ATR,
    NORMAL_BARS,
    gap_reading,
    implied_cross,
    signals_from_gap,
)


def _legs(count: int = 4_000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=count, freq="15min", tz="UTC")

    def walk(start: float, scale: float) -> pd.DataFrame:
        close = start + rng.normal(0.0, scale, size=count).cumsum()
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(rng.normal(0.0, scale, size=count)),
                "low": close - np.abs(rng.normal(0.0, scale, size=count)),
                "close": close,
            },
            index=index,
        )
        return frame

    return walk(4_700.0, 3.0), walk(154.0, 0.08)


def _product(gold: pd.DataFrame, yen: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {column: gold[column] * yen[column] for column in ("open", "high", "low", "close")},
        index=gold.index,
    )


class TestACrossComputedFromItsLegsHasNothingToTrade:
    """THE QUESTION THAT HAS TO BE ASKED BEFORE ANY BACKTEST.

    Many brokers derive a cross from its legs. Then the gap is zero by
    construction, and a search over it finds its own rounding error and reports
    it as an edge. The script says so and stops, in a minute, instead of after a
    ninety-day replay.
    """

    def test_a_perfectly_derived_cross_shows_no_gap(self) -> None:
        gold, yen = _legs()
        cross = _product(gold, yen)

        _frame, reading, _raw = gap_reading(cross, implied_cross(gold, yen))
        finite = reading[np.isfinite(reading)]

        assert len(finite) > NORMAL_BARS
        assert float(np.percentile(np.abs(finite), 90)) < DEAD_GAP_ATR

    def test_no_threshold_produces_a_trade_out_of_nothing(self) -> None:
        gold, yen = _legs()
        _frame, reading, _raw = gap_reading(_product(gold, yen), implied_cross(gold, yen))

        for threshold in (0.25, 0.5, 1.0):
            assert not (signals_from_gap(reading, threshold) != 0).any(), threshold


class TestALaggingCrossIsFound:
    """The other way to be wrong: a real lag that the reading cannot see."""

    def test_a_one_bar_lag_shows_up_well_clear_of_the_dead_band(self) -> None:
        gold, yen = _legs()
        lagging = _product(gold, yen).shift(1).dropna()

        _frame, reading, _raw = gap_reading(lagging, implied_cross(gold, yen))
        finite = reading[np.isfinite(reading)]

        assert float(np.median(np.abs(finite))) > DEAD_GAP_ATR * 5

    def test_the_direction_sells_a_rich_cross_and_buys_a_cheap_one(self) -> None:
        """A quote above its legs has not come down yet, so it is sold. Getting
        this backwards would produce a mirror-image edge and look identical in
        every summary statistic."""
        reading = np.array([2.0, -2.0, 0.0])

        assert list(signals_from_gap(reading, 1.0)) == [-1, 1, 0]

    def test_a_constant_offset_is_not_a_signal(self) -> None:
        """Contract size, a markup and the financing leg all sit between the
        cross and the product of its legs. None of it is tradeable, so a fixed
        offset must read as zero rather than as a permanent one-way trade."""
        gold, yen = _legs()
        offset = _product(gold, yen) + 5_000.0

        _frame, reading, _raw = gap_reading(offset, implied_cross(gold, yen))
        finite = reading[np.isfinite(reading)]

        assert float(np.percentile(np.abs(finite), 90)) < DEAD_GAP_ATR


class TestTheAlignmentHappensOnce:
    def test_gap_reading_returns_the_frame_it_aligned(self) -> None:
        """The first version returned only the readings and left the caller to
        rebuild the shared index with a second intersection. Two alignments of
        one thing is how a reading ends up a bar out of step with the bar it
        labels -- silently, and in the flattering direction."""
        gold, yen = _legs()
        frame, reading, raw = gap_reading(_product(gold, yen), implied_cross(gold, yen))

        assert len(frame) == len(reading) == len(raw)
        assert list(frame.columns) == ["open", "high", "low", "close"]

    def test_it_keeps_only_bars_all_three_traded(self) -> None:
        """A timestamp one leg is missing is a bar it did not trade. Filling it
        would invent the very gap this is looking for."""
        gold, yen = _legs()
        thin = yen.iloc[::2]

        frame, _reading, _raw = gap_reading(_product(gold, yen), implied_cross(gold, thin))

        assert len(frame) == len(thin)
        assert frame.index.isin(thin.index).all()
