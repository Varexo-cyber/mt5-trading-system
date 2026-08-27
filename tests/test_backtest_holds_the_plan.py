"""The grader held every order for forty-one hours, whatever its plan said.

`max_holding_bars` is 500 and the replay runs on M5, so every proposal this
tool has ever graded was carried for 500 x 5 = 2,500 minutes — forty-one and a
half hours — regardless of whether the engine had planned it as a thirty-minute
quick trade, a three-hour intraday one or a twenty-four hour swing.

THAT MATTERS MORE HERE THAN IN THE LIVE EXIT, because this is the instrument
the rest of the system was tuned against. Three conclusions came out of it and
all three were measured under that hold:

  * the target-distance table that moved `minimum_r_multiple` from 0.75 to 0.35
  * the cost-band table that cleared the cost gate of blame
  * "not one detector beat a coin flip taking the same moments"

None of those is wrong about a forty-one-hour hold. None of them was ever about
the horizon the module in question claims to read. A detector whose whole
statement is about the next thirty minutes, graded over forty-one hours, is
being scored almost entirely on movement it never spoke about — and it will
come out looking like a coin flip whether or not it is one.

This does not assert that any detector has an edge. It asserts that the
question can now be asked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from backtesting.engine import BacktestAssumptions, BacktestOrder, PessimisticBacktester
from core.types import Direction

START = datetime(2026, 8, 1, tzinfo=UTC)


def _frame(bars: int = 600, freq: str = "5min") -> pd.DataFrame:
    """A flat market, so nothing but the clock can end a trade.

    Deliberately flat: with no stop or target ever touched, the number of bars
    the trade survives IS the holding rule, and the test cannot pass by
    accident because price happened to resolve it.
    """
    index = pd.date_range(start=START, periods=bars, freq=freq, tz=UTC)
    return pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "spread": 0},
        index=index,
    )


def _order(horizon: int | None) -> BacktestOrder:
    return BacktestOrder(
        symbol="TEST",
        decided_at=START,
        direction=Direction.LONG,
        entry=100.0,
        # Far enough out that a flat market touches neither.
        stop_loss=90.0,
        take_profit=110.0,
        horizon_minutes=horizon,
    )


def _bars_held(horizon: int | None, **assumptions) -> int:  # type: ignore[no-untyped-def]
    result = PessimisticBacktester(BacktestAssumptions(**assumptions)).run(
        _frame(), [_order(horizon)]
    )
    return result.trades[0].holding_bars


class TestEachOrderIsHeldForItsOwnPlan:
    def test_a_quick_plan_is_no_longer_carried_for_forty_one_hours(self) -> None:
        """Thirty minutes is six M5 bars. It was 500."""
        assert _bars_held(30) == 6

    def test_an_intraday_plan_gets_its_three_hours(self) -> None:
        assert _bars_held(180) == 36

    def test_a_swing_plan_gets_its_twenty_four(self) -> None:
        """288 M5 bars, and still inside the 500-bar cap — so this one was
        being over-held by seventeen hours rather than by forty."""
        assert _bars_held(24 * 60) == 288

    def test_the_cap_still_wins_when_a_plan_asks_for_more(self) -> None:
        assert _bars_held(10_000) == 500


class TestNothingWithoutAPlanChanges:
    def test_an_order_built_by_hand_is_replayed_exactly_as_before(self) -> None:
        """`scripts/backtest_section_six.py` builds its own orders and has its
        own lane. It must keep the behaviour it was measured under."""
        assert _bars_held(None) == 500

    def test_a_nonsense_horizon_falls_back_rather_than_closing_instantly(self) -> None:
        """Zero bars is not a trade. The dangerous failure here is the mirror
        of the live one: a bad number read literally would end every trade on
        its entry bar and report a flat, confident, meaningless result."""
        assert _bars_held(0) == 500
        assert _bars_held(-5) == 500

    def test_a_plan_shorter_than_the_replay_resolution_still_gets_one_bar(self) -> None:
        """A four-minute plan on an M5 replay. One bar, not zero — the frame
        cannot express the plan, and rounding it away would silently delete the
        trade instead of grading it coarsely."""
        assert _bars_held(4) == 1


class TestTheBarLengthIsReadAndNotAssumed:
    def test_an_m1_replay_gets_m1_bars_for_the_same_plan(self) -> None:
        """The caller picks the timeframe. 500 bars means forty-one hours on M5
        and eight on M1, so a hard-coded assumption about which one it is would
        be wrong half the time."""
        result = PessimisticBacktester(BacktestAssumptions()).run(_frame(freq="1min"), [_order(30)])

        assert result.trades[0].holding_bars == 30


class TestTheReplayActuallySetsIt:
    """The half that is easy to leave out, and leaving it out is the defect
    class this whole day has been about: the engine reads a field nothing
    writes, every test above still passes, and every replay silently keeps the
    forty-one-hour hold."""

    def test_the_order_the_replay_builds_carries_the_plan_length(self) -> None:
        import inspect

        from backtesting import replay

        source = " ".join(inspect.getsource(replay).split())

        assert "horizon_minutes=idea.expected_horizon_minutes" in source

    def test_and_it_is_not_read_through_a_forgiving_getattr(self) -> None:
        """A `getattr(idea, "expected_horizon_minutes", None)` would make a
        production regression — the grader falling back to the flat cap on
        every order — look exactly like a passing suite."""
        import inspect

        from backtesting import replay

        source = " ".join(inspect.getsource(replay).split())

        assert 'getattr(idea, "expected_horizon_minutes"' not in source


class TestFlatMarketSanity:
    """Guarding the fixture itself. If the flat frame ever resolved a trade on
    price, every number above would be measuring the wrong thing."""

    def test_nothing_above_was_decided_by_a_stop_or_a_target(self) -> None:
        result = PessimisticBacktester(BacktestAssumptions()).run(_frame(), [_order(180)])

        assert result.trades[0].outcome == "TIME"


def test_the_cap_is_still_what_it_was() -> None:
    """The 500 that produced every earlier conclusion. Written down so the
    forty-one hours in the comments stays checkable."""
    assert BacktestAssumptions().max_holding_bars == 500
    assert pytest.approx(41.67, abs=0.01) == 500 * 5 / 60
