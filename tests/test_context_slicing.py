"""The visible-history slice: same answer, without the quadratic.

`_context` used to read

    visible = frame[frame.index + timeframe.duration <= upto]
    ... visible.tail(WARMUP)

which is correct and quadratic. Every call shifted the WHOLE index, compared
the whole thing, copied every matching row, and then discarded all but the last
260. Over a 180-day window that is a 52,000-row M5 frame walked in full for
each of ~17,000 M15 bars on each of sixteen markets -- roughly 1.4e10 row
operations. Six-month runs took hours, and the MT5 fetch got the blame.

Measured on this machine, 400 bars of work:

     30 days   1.027s -> 0.097s    10.6x
    180 days   2.842s -> 0.131s    21.7x

and the ratio grows with the window, which is the signature of removing an
O(n) factor rather than tuning a constant.

A REWRITE FOR SPEED IS ONLY SAFE IF IT IS THE SAME FUNCTION, so the first test
here runs both implementations over the same bars and compares them frame for
frame. The old one is kept in the test as the reference; it is the definition
this file is defending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.types import MarketContext, Series, Tick, Timeframe
from scripts.dry_run_sections import WARMUP, _context


def _frame(n: int, freq: str) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    close = pd.Series(100.0 + np.arange(n) * 0.01, index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "spread": 1},
        index=index,
    )


def _frames(days: int = 60) -> dict:
    """Every clock covering the SAME span, which the first draft did not.

    Deriving each frame's length from a fixed M5 bar count gave H4 only
    eighty-odd bars over the period M15 covered, so `_context` correctly
    returned None and two tests failed on the fixture rather than the code.
    """
    per_day = {"5min": 288, "15min": 96, "30min": 48, "1h": 24, "4h": 6}
    clocks = {
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
    }
    return {tf: _frame(days * per_day[freq], freq) for tf, freq in clocks.items()}


def _reference(symbol: str, frames: dict, upto, spread: float):
    """THE OLD IMPLEMENTATION, kept as the definition of correct."""
    series: dict = {}
    for timeframe, frame in frames.items():
        visible = frame[frame.index + timeframe.duration <= upto]
        if len(visible) < WARMUP:
            return None
        series[timeframe] = Series(symbol, timeframe, visible.tail(WARMUP), upto)
    price = float(series[Timeframe.M5].df["close"].iloc[-1])
    half = spread / 2.0
    return MarketContext(symbol, upto, series, Tick(symbol, upto, price - half, price + half))


class TestTheFastSliceIsTheSameSlice:
    def test_it_matches_the_old_implementation_bar_for_bar(self) -> None:
        frames = _frames()
        stamps = list(frames[Timeframe.M15].index[-120:])

        for upto in stamps:
            fast = _context("X", frames, upto, 0.0002)
            slow = _reference("X", frames, upto, 0.0002)

            assert (fast is None) == (slow is None), upto
            if fast is None:
                continue
            for timeframe in frames:
                left, right = fast.series[timeframe].df, slow.series[timeframe].df
                assert left.index.equals(right.index), (timeframe, upto)
                assert np.allclose(left["close"].to_numpy(), right["close"].to_numpy())
            assert fast.tick.mid == pytest.approx(slow.tick.mid)

    def test_it_agrees_on_the_boundary_where_a_bar_has_just_closed(self) -> None:
        """The off-by-one that a searchsorted rewrite invites. A bar is visible
        once `open + duration <= upto`, so at exactly that instant it counts."""
        frames = _frames()
        m15 = frames[Timeframe.M15]
        boundary = m15.index[-30] + Timeframe.M15.duration

        for offset in (-1, 0, 1):
            upto = boundary + pd.Timedelta(seconds=offset)
            fast = _context("X", frames, upto, 0.0)
            slow = _reference("X", frames, upto, 0.0)

            assert fast.series[Timeframe.M15].df.index.equals(
                slow.series[Timeframe.M15].df.index
            ), offset

    def test_too_little_history_still_returns_nothing(self) -> None:
        frames = _frames()
        early = frames[Timeframe.M15].index[WARMUP // 2]

        assert _context("X", frames, early, 0.0) is None
        assert _reference("X", frames, early, 0.0) is None

    def test_it_always_hands_over_exactly_the_warmup(self) -> None:
        frames = _frames()

        ctx = _context("X", frames, frames[Timeframe.M15].index[-1], 0.0)

        for timeframe in frames:
            assert len(ctx.series[timeframe].df) == WARMUP, timeframe


class TestItIsNoLongerQuadratic:
    def test_a_longer_history_does_not_cost_proportionally_more(self) -> None:
        """The property, stated so it cannot be satisfied by a faster constant.

        Ten times the history for the same 200 bars of work. The old code paid
        ten times as much; this must pay barely more. Generous bound because a
        timing test on shared hardware has to be, and it still separates
        O(log n) from O(n) by a wide margin.
        """
        import time

        def cost(days: int) -> float:
            frames = _frames(days)
            stamps = list(frames[Timeframe.M15].index[-200:])
            start = time.perf_counter()
            for upto in stamps:
                _context("X", frames, upto, 0.0002)
            return time.perf_counter() - start

        small = cost(60)
        large = cost(600)

        assert large < small * 3.0, f"{large:.3f}s against {small:.3f}s looks linear in history"

    def test_it_does_not_build_a_mask_over_the_whole_frame(self) -> None:
        import inspect

        from scripts import dry_run_sections

        source = inspect.getsource(dry_run_sections._context)
        # Strip the docstring: it QUOTES the old line as the thing being
        # replaced, and a first draft of this test matched its own explanation.
        body = source.split('"""')[-1]

        assert "searchsorted" in body
        assert "frame[frame.index" not in body, "the quadratic mask is back"


class TestTheSliceCacheChangesNothing:
    """Walking an M30 pass, the H4 window changes once every eight bars and
    the H1 window once every two, yet every bar re-sliced all five frames.
    Memoising the DataFrame per (timeframe, cut) turns most of that into a
    dict lookup -- but a cache that returns a different answer is not an
    optimisation, it is a bug with a speedup attached."""

    def test_cached_and_uncached_agree_bar_for_bar(self) -> None:
        frames = _frames()
        stamps = list(frames[Timeframe.M30].index[-150:])
        cache: dict = {}

        for upto in stamps:
            cold = _context("X", frames, upto, 0.0002)
            warm = _context("X", frames, upto, 0.0002, cache)

            assert (cold is None) == (warm is None)
            if cold is None:
                continue
            for timeframe in frames:
                assert cold.series[timeframe].df.index.equals(warm.series[timeframe].df.index)
            assert cold.tick.mid == pytest.approx(warm.tick.mid)

    def test_it_actually_hits(self) -> None:
        """A cache that never hits is pure overhead. On an M30 walk the slower
        frames must be reused."""
        frames = _frames()
        cache: dict = {}
        misses = 0

        class Counting(dict):
            def __setitem__(self, key, value):
                nonlocal misses
                misses += 1
                super().__setitem__(key, value)

        counting = Counting()
        for upto in frames[Timeframe.M30].index[-40:]:
            _context("X", frames, upto, 0.0002, counting)

        # 40 bars x 5 frames = 200 slices without a cache.
        # 25%, and the arithmetic says why: M5, M15 and M30 advance every bar
        # and can never hit; only H1 (every 2nd) and H4 (every 8th) are reused.
        # A first draft asserted < 140 on a guess of "about sixty percent".
        assert misses < 160, f"{misses} misses of 200 -- the cache is not helping"
        assert cache == {}

    def test_it_stays_bounded(self) -> None:
        """An unbounded memo would hold the whole history a second time."""
        frames = _frames()
        cache: dict = {}

        for upto in frames[Timeframe.M30].index[-400:]:
            _context("X", frames, upto, 0.0002, cache)

        assert len(cache) <= 70
