"""Can this specific target be reached before we force the position flat?

The runway filter enforces a flat floor because it never sees the setup. This
is the same question with the setup in hand, and it is the one that matters:
forty-five minutes is generous for a target one ATR away and hopeless for one
eight ATR away, and a rule that cannot tell them apart is simultaneously too
strict and too loose.

The estimate uses the same normalisation as the health reader and the
higher-timeframe conflict check. Net displacement over n bars scales with
`sqrt(n) x ATR`, so covering d ATR needs `d^2` bars — not `d`. Getting that
wrong understates a distant target by a factor of five, which is precisely how
an entry looks fine at 19:50 and is in fact unfinishable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from analysis.confluence import TradeIdea
from config.schema import RunwayFilterConfig, SessionFilterConfig
from core.types import Direction, MarketContext, Series, Tick, Timeframe
from filters.base import FilterChain
from filters.runway_filter import RunwayFilter
from filters.session_filter import SessionFilter
from runner.service import JarvisRunner

#: Wednesday 18:45 UTC — 90 minutes before the forex wind-down at 20:15.
NOW = datetime(2026, 3, 11, 18, 45, tzinfo=UTC)
#: One ATR, in price. Every bar below moves exactly this much.
STEP = 0.0010


def runner(
    now: datetime = NOW,
    *,
    config: RunwayFilterConfig | None = None,
    session_config: SessionFilterConfig | None = None,
):  # type: ignore[no-untyped-def]
    """The method under test, on a runner carrying only what it reads."""
    session = SessionFilter(session_config or SessionFilterConfig())
    runway = config or RunwayFilterConfig()

    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = SimpleNamespace(filters=SimpleNamespace(runway=runway))  # type: ignore[assignment]
    instance.filters = FilterChain([RunwayFilter(runway, session)])  # type: ignore[assignment]
    instance.clock = SimpleNamespace(now=lambda: now)  # type: ignore[assignment]
    return instance


def context(*, bars: int = 200, step: float = STEP) -> MarketContext:
    """M5 bars each ranging exactly `step`, so ATR(14) is `step`."""
    index = pd.date_range("2026-03-11", periods=bars, freq="5min", tz=UTC)
    mid = 1.08500
    frame = pd.DataFrame(
        {
            "open": mid,
            "high": mid + step / 2,
            "low": mid - step / 2,
            "close": mid,
            "tick_volume": 100,
            "spread": 12,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol="EURUSD",
        now=NOW,
        series={
            Timeframe.M5: Series(
                symbol="EURUSD",
                timeframe=Timeframe.M5,
                df=frame,
                fetched_at=NOW,
            )
        },
        tick=Tick(symbol="EURUSD", time=NOW, bid=mid, ask=mid + 0.00004),
    )


def idea(atr_units: float) -> TradeIdea:
    """A long whose target sits `atr_units` ATR above entry."""
    entry = 1.08500
    return TradeIdea(
        symbol="EURUSD",
        approved=True,
        direction=Direction.LONG,
        score=70.0,
        confidence=0.7,
        entry=entry,
        stop_loss=entry - STEP,
        take_profit=entry + atr_units * STEP,
        reason="test",
        signals=(),
    )


class TestReachability:
    def test_a_near_target_clears_easily(self) -> None:
        """2 ATR is ~9 minutes away against 90 minutes of runway."""
        ok, needed, runway = runner()._target_is_reachable_in_time(context(), idea(2.0), "forex")
        assert ok
        assert needed == pytest.approx(8.9, abs=0.3)
        assert runway == 90.0

    def test_a_distant_target_is_refused_with_time_still_on_the_clock(self) -> None:
        """8 ATR needs over two hours. There are ninety minutes left.

        Nothing else in the system objects to this trade: the session is open,
        the runway floor of 45 minutes is comfortably met, the spread is fine.
        It is still a trade that cannot finish.
        """
        ok, needed, runway = runner()._target_is_reachable_in_time(context(), idea(8.0), "forex")
        assert not ok
        assert needed == pytest.approx(142.2, abs=1.0)
        assert runway == 90.0

    def test_the_same_target_is_fine_earlier_in_the_day(self) -> None:
        """Identical setup at 14:30, when there are 345 minutes left."""
        morning = datetime(2026, 3, 11, 14, 30, tzinfo=UTC)
        ok, _, runway = runner(morning)._target_is_reachable_in_time(context(), idea(8.0), "forex")
        assert ok
        assert runway == 345.0

    def test_a_quiet_market_pushes_the_same_target_out_of_reach(self) -> None:
        """Same distance in price, half the ATR — so four times the bars.

        This is the case that makes the whole check worth having. The chart
        looks identical. Only the speed changed, and the target went from
        thirty-five minutes away to well past the wind-down.
        """
        fast = runner()._target_is_reachable_in_time(context(step=STEP), idea(4.0), "forex")
        slow = runner()._target_is_reachable_in_time(
            context(step=STEP / 2),
            idea(4.0),
            "forex",
        )
        assert fast[0]
        assert not slow[0]
        assert slow[1] == pytest.approx(fast[1] * 4, rel=0.02)

    def test_a_continuous_market_has_nothing_to_be_late_for(self) -> None:
        ok, needed, runway = runner()._target_is_reachable_in_time(context(), idea(20.0), "crypto")
        assert ok
        assert needed is None and runway is None

    def test_switching_the_check_off_leaves_the_filter_floor_alone(self) -> None:
        config = RunwayFilterConfig(require_reachable_target=False)
        ok, needed, _ = runner(config=config)._target_is_reachable_in_time(
            context(), idea(20.0), "forex"
        )
        assert ok
        assert needed is None

    def test_missing_speed_data_does_not_block(self) -> None:
        """No ATR means no estimate — and the filter floor already cleared it.

        Failing closed here would be the wrong instinct: this check is a
        refinement on top of a gate that has already said yes, so an
        unanswerable refinement leaves the answer where the gate put it.
        """
        bare = MarketContext(symbol="EURUSD", now=NOW, series={}, tick=None)
        ok, needed, runway = runner()._target_is_reachable_in_time(bare, idea(8.0), "forex")
        assert ok
        assert needed is None
        assert runway == 90.0

    def test_travel_efficiency_moves_the_line(self) -> None:
        """The optimism is one number in one place, and it is load-bearing."""
        pessimist = RunwayFilterConfig(travel_efficiency=1.0)
        optimist = RunwayFilterConfig(travel_efficiency=2.5)

        assert not runner(config=pessimist)._target_is_reachable_in_time(
            context(), idea(5.0), "forex"
        )[0]
        assert runner(config=optimist)._target_is_reachable_in_time(context(), idea(5.0), "forex")[
            0
        ]

    def test_an_index_loses_the_last_fifteen_minutes(self) -> None:
        """At 18:45 the index has 75 minutes, forex has 90.

        A target needing ~80 minutes is reachable on one and not the other,
        from the same chart at the same instant.
        """
        six_atr = idea(6.0)
        assert runner()._target_is_reachable_in_time(context(), six_atr, "forex")[0]
        assert not runner()._target_is_reachable_in_time(context(), six_atr, "index")[0]
