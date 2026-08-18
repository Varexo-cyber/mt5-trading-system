"""The replay window is where look-ahead bias would enter, so it is pinned twice.

The slice was `frame[frame.index + duration <= decided_at].tail(history_bars)`:
a boolean mask over the whole frame, built per decision per timeframe. Over 90
days the M5 frame is around 26,000 rows, so a five-symbol run spent hundreds of
millions of row comparisons finding a cut point in a sorted index, and the tool
built to make measurement cheap took long enough that nobody would run it twice.

It is a binary search now. Making it faster is worthless if it also makes it
different, and "different" here means a bar the analysis could not have seen —
which does not fail loudly, it produces a backtest that looks better than
reality. So this compares the two implementations directly, on an index with
the awkward shapes: weekend gaps, a decision landing exactly on a close, and a
decision before enough history exists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from backtesting.replay import REPLAY_TIMEFRAMES, HistoricalContextReplay
from core.types import Timeframe

START = datetime(2026, 3, 2, tzinfo=UTC)


def frame(timeframe: Timeframe, count: int) -> pd.DataFrame:
    """Bars on a real calendar: weekdays only, so weekends leave gaps."""
    stamps: list[pd.Timestamp] = []
    cursor = pd.Timestamp(START)
    while len(stamps) < count:
        if cursor.weekday() < 5:
            stamps.append(cursor)
        cursor += timeframe.duration
    index = pd.DatetimeIndex(stamps)
    values = [100.0 + i * 0.01 for i in range(count)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 0.05 for v in values],
            "low": [v - 0.05 for v in values],
            "close": values,
            "spread": 2,
        },
        index=index,
    )


def old_window(data: pd.DataFrame, timeframe: Timeframe, decided_at, history_bars: int):
    """The implementation this replaced, kept here as the reference."""
    return data[data.index + timeframe.duration <= decided_at].tail(history_bars)


def new_window(data: pd.DataFrame, timeframe: Timeframe, decided_at, history_bars: int):
    close_times = data.index + timeframe.duration
    cut = int(close_times.searchsorted(pd.Timestamp(decided_at), side="right"))
    return data.iloc[max(0, cut - history_bars) : cut]


class TestTheFastWindowIsTheSameWindow:
    @pytest.mark.parametrize("timeframe", list(REPLAY_TIMEFRAMES))
    def test_every_decision_selects_identical_bars(self, timeframe: Timeframe) -> None:
        data = frame(timeframe, 600)
        for offset in range(0, 600, 37):
            decided_at = (data.index[min(offset, 599)] + timeframe.duration).to_pydatetime()
            assert new_window(data, timeframe, decided_at, 300).equals(
                old_window(data, timeframe, decided_at, 300)
            )

    def test_a_decision_exactly_on_a_close_includes_that_bar(self) -> None:
        """`side="right"` and `<=` have to agree on the boundary. Off by one
        here is either a bar thrown away or a bar the analysis could not have
        seen, and the second one silently improves every backtest."""
        data = frame(Timeframe.H1, 400)
        decided_at = (data.index[200] + Timeframe.H1.duration).to_pydatetime()

        window = new_window(data, Timeframe.H1, decided_at, 300)

        assert window.index[-1] == data.index[200]
        assert window.equals(old_window(data, Timeframe.H1, decided_at, 300))

    def test_a_decision_one_tick_before_a_close_excludes_it(self) -> None:
        data = frame(Timeframe.H1, 400)
        decided_at = (
            data.index[200] + Timeframe.H1.duration - timedelta(seconds=1)
        ).to_pydatetime()

        window = new_window(data, Timeframe.H1, decided_at, 300)

        assert window.index[-1] == data.index[199]
        assert window.equals(old_window(data, Timeframe.H1, decided_at, 300))

    def test_before_any_bar_has_closed_both_return_nothing(self) -> None:
        data = frame(Timeframe.H1, 400)
        decided_at = (data.index[0] - timedelta(days=1)).to_pydatetime()

        assert len(new_window(data, Timeframe.H1, decided_at, 300)) == 0
        assert len(old_window(data, Timeframe.H1, decided_at, 300)) == 0


class TestTheReplayStillRefusesThinHistory:
    def test_a_decision_without_120_bars_is_skipped(self) -> None:
        """The floor that keeps an indicator from being computed on nothing."""

        class _Engine:
            config = None

            def evaluate(self, *_args, **_kwargs):  # pragma: no cover - never reached
                raise AssertionError("evaluated a decision with too little history")

        frames = {tf: frame(tf, 130) for tf in REPLAY_TIMEFRAMES}
        replay = HistoricalContextReplay(_Engine())  # type: ignore[arg-type]

        produced = list(
            replay.ideas(
                "TEST",
                frames,
                point=0.00001,
                start=frames[Timeframe.H1].index[0].to_pydatetime(),
                end=frames[Timeframe.H1].index[5].to_pydatetime(),
            )
        )

        assert produced == []
