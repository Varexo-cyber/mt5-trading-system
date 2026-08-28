"""Turning M1 on cost the entire report, and the report said nothing useful.

`--with-m1` became the default on 27 August so the three live detectors that
trigger on M1 could be graded at all. The very next run:

    replaying EURUSD.i ...
      skipped: SymbolNotAvailableError: copy_rates_range(timeframe=1, ...)
               failed ([-2] Terminal: Invalid params)
    ... the same for all five markets ...
    No proposals in the window. Nothing to measure.

TWO FAULTS, ONE OUTPUT. Ninety days of M1 is about 130,000 bars and the
terminal refuses that request outright instead of truncating it -- a limit
nothing had hit before, because D1 through M5 over the same window are all
comfortably under it. And when the request failed, the exception escaped
`history()` and the caller's per-symbol `except` dropped the whole market:
every timeframe, not just the one that could not be had.

So an OPTIONAL extra chart deleted the required five. That is the same shape as
the replay's completeness check, and it is fixed the same way: the required set
is fatal, an extra is offered when it can be had and left out when it cannot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest

from core.errors import SymbolNotAvailableError
from core.types import Timeframe
from scripts.backtest_modules import _SLICE_BARS, _fetch_in_slices, history

END = datetime(2026, 8, 28, tzinfo=UTC)


def _bars(count: int) -> np.ndarray:
    return np.array(
        [(0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0) for _ in range(count)],
        dtype=[
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
            ("real_volume", "i8"),
        ],
    )


class FakeConnector:
    """Refuses any window wider than `limit` bars, which is what MT5 does."""

    def __init__(self, limit: int | None = 100_000, refuse: set[int] | None = None) -> None:
        self.limit = limit
        self.refuse = refuse or set()
        self.calls: list[tuple[int, datetime, datetime]] = []

    def copy_rates_range(self, symbol: str, timeframe: int, start: datetime, end: datetime):  # type: ignore[no-untyped-def]
        del symbol
        self.calls.append((timeframe, start, end))
        if timeframe in self.refuse:
            raise SymbolNotAvailableError(f"timeframe={timeframe} failed ([-2] Invalid params)")
        minutes = (end - start).total_seconds() / 60.0
        bars = int(minutes / max(1, timeframe))
        if self.limit is not None and bars > self.limit:
            raise SymbolNotAvailableError(f"timeframe={timeframe} failed ([-2] Invalid params)")
        return _bars(min(bars, 10))


class TestTheWindowIsCutIntoPiecesTheTerminalWillAnswer:
    def test_ninety_days_of_m1_no_longer_asks_for_it_all_at_once(self) -> None:
        connector = FakeConnector()
        start = END - timedelta(days=90)

        _fetch_in_slices(connector, "EURUSD.i", Timeframe.M1, start, END)

        assert len(connector.calls) > 1
        # Every piece inside the limit, which is the whole point.
        for _tf, first, last in connector.calls:
            assert (last - first).total_seconds() / 60.0 <= _SLICE_BARS

    def test_the_pieces_cover_the_window_exactly_once(self) -> None:
        """No gap and no overlap. A gap silently deletes decisions; an overlap
        would duplicate bars and every ATR computed from them would be wrong."""
        connector = FakeConnector()
        start = END - timedelta(days=90)

        _fetch_in_slices(connector, "EURUSD.i", Timeframe.M1, start, END)

        edges = [(first, last) for _tf, first, last in connector.calls]
        assert edges[0][0] == start
        assert edges[-1][1] == END
        for (_a, previous_end), (next_start, _b) in pairwise(edges):
            assert previous_end == next_start

    def test_a_short_window_is_still_a_single_call(self) -> None:
        """Everything that worked before must take exactly the path it took
        before. D1 through M5 over ninety days are all well inside one slice."""
        connector = FakeConnector()

        _fetch_in_slices(connector, "EURUSD.i", Timeframe.H1, END - timedelta(days=90), END)

        assert len(connector.calls) == 1

    def test_an_empty_window_is_reported_and_not_returned_as_nothing(self) -> None:
        """Zero bars from every slice is a missing symbol, not a symbol with no
        price action. Returning an empty array would be graded as "this market
        produced no setups", which is a different and wrong conclusion."""
        connector = FakeConnector(limit=0)

        with pytest.raises(SymbolNotAvailableError):
            _fetch_in_slices(connector, "EURUSD.i", Timeframe.M1, END - timedelta(days=90), END)


class TestAnOptionalChartCannotDeleteTheSymbol:
    def test_m1_failing_still_returns_the_required_five(self) -> None:
        """The defect exactly: M1 was unavailable and all five markets vanished
        from the report."""
        connector = FakeConnector(refuse={Timeframe.M1.mt5_value})

        frames = history(
            connector,
            "EURUSD.i",
            END - timedelta(days=90),
            END,
            timeframes=(Timeframe.H1, Timeframe.M5, Timeframe.M1),
        )

        assert Timeframe.M1 not in frames
        assert {Timeframe.H1, Timeframe.M5} <= set(frames)

    def test_a_required_chart_failing_is_still_fatal(self) -> None:
        """The other direction, or the fix would have swapped one silent
        failure for another: without M5 there is no execution frame and any
        result computed from what is left would be a fiction."""
        connector = FakeConnector(refuse={Timeframe.M5.mt5_value})

        with pytest.raises(SymbolNotAvailableError):
            history(
                connector,
                "EURUSD.i",
                END - timedelta(days=90),
                END,
                timeframes=(Timeframe.H1, Timeframe.M5, Timeframe.M1),
            )
