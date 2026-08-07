"""The five theories, finally driven over price history.

They were argued into existence, reviewed, unit-tested against synthetic bars
— and never once run against a market. This harness is what changes that, so
the tests here are mostly about the harness being honest rather than about the
theories being right. A replay that can see the future will happily report
that everything works.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.playbooks import FadeConfig, PlaybookEngine, RangeFade
from backtesting.engine import BacktestOrder, PessimisticBacktester
from backtesting.playbook_replay import (
    DECISION_TIMEFRAME,
    REPLAY_TIMEFRAMES,
    PlaybookEvidence,
    PlaybookReplay,
    evidence_by_playbook,
    render,
)
from config.loader import load_settings
from core.types import Direction, Timeframe

START = datetime(2026, 6, 1, tzinfo=UTC)


def frame(timeframe: Timeframe, bars: int, *, seed: int = 7) -> pd.DataFrame:
    """A random-walk series with plausible wicks and a recorded spread."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.0004, bars).cumsum()
    closes = 1.1000 + steps
    opens = np.concatenate([[closes[0]], closes[:-1]])
    wick = np.abs(closes - opens) * 0.3 + 0.00002
    index = pd.date_range(start=START, periods=bars, freq=timeframe.duration, tz=UTC)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + wick,
            "low": np.minimum(opens, closes) - wick,
            "close": closes,
            "tick_volume": np.full(bars, 500),
            "spread": np.full(bars, 10),
            "real_volume": np.zeros(bars),
        },
        index=index,
    )


#: Every timeframe has to span the same stretch of calendar, or the slowest
#: one runs out of history first and every context is silently skipped. Twenty
#: five days: H1 needs 120 closed bars before a decision can be made at all.
SPAN_DAYS = 25
BARS = {Timeframe.H1: 24, Timeframe.M15: 96, Timeframe.M5: 288}

#: Far enough in that all three have their history, and only an hour wide so
#: the tests stay quick.
DECIDE_FROM = START + timedelta(days=20)
DECIDE_TO = DECIDE_FROM + timedelta(hours=1)


@pytest.fixture
def frames() -> dict[Timeframe, pd.DataFrame]:
    return {tf: frame(tf, BARS[tf] * SPAN_DAYS) for tf in REPLAY_TIMEFRAMES}


@pytest.fixture
def engine() -> PlaybookEngine:
    settings = load_settings(env_overrides=False)
    return PlaybookEngine([RangeFade(FadeConfig())], settings.analysis.confluence)


def order(name: str, decided_at: datetime, *, up: bool = True) -> BacktestOrder:
    entry = 1.1000
    return BacktestOrder(
        symbol="EURUSD",
        decided_at=decided_at,
        direction=Direction.LONG if up else Direction.SHORT,
        entry=entry,
        stop_loss=entry - 0.0010 if up else entry + 0.0010,
        take_profit=entry + 0.0020 if up else entry - 0.0020,
        modules=(name,),
    )


# ------------------------------------------------------------- the harness ---


def test_no_decision_can_see_a_bar_that_had_not_closed(engine, frames) -> None:  # type: ignore[no-untyped-def]
    """The one bug a replay must be structurally unable to have. A decision at
    10:05 sees the M15 bar that closed at 10:00, never the one still forming."""
    replay = PlaybookReplay(engine)
    seen: list[tuple[datetime, datetime]] = []

    original = replay._context

    def watched(symbol, frames_, decided_at, point):  # type: ignore[no-untyped-def]
        context = original(symbol, frames_, decided_at, point)
        if context is not None:
            newest = context.series[Timeframe.M15].df.index[-1].to_pydatetime()
            seen.append((decided_at, newest + Timeframe.M15.duration))
        return context

    replay._context = watched  # type: ignore[assignment]
    replay.orders("EURUSD", frames, point=0.00001, start=DECIDE_FROM, end=DECIDE_TO)

    assert seen, "the replay never built a context"
    assert all(closed_at <= decided_at for decided_at, closed_at in seen)


def test_a_window_with_too_little_history_produces_nothing(engine, frames) -> None:  # type: ignore[no-untyped-def]
    """Rather than a context padded with whatever happens to be there."""
    replay = PlaybookReplay(engine)

    orders = replay.orders(
        "EURUSD", frames, point=0.00001, start=START, end=START + timedelta(hours=1)
    )

    assert orders == []


def test_a_missing_timeframe_is_refused_loudly(engine, frames) -> None:  # type: ignore[no-untyped-def]
    """Silently replaying without H1 would measure a different system."""
    del frames[Timeframe.H1]
    replay = PlaybookReplay(engine)

    with pytest.raises(ValueError, match="missing timeframes"):
        replay.orders("EURUSD", frames, point=0.00001, start=START, end=START)


def test_the_stride_must_be_positive(engine) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        PlaybookReplay(engine, decision_stride_bars=0)


def test_a_short_history_window_is_refused(engine) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        PlaybookReplay(engine, history_bars=10)


def test_the_recorded_spread_reaches_the_theories(engine, frames) -> None:  # type: ignore[no-untyped-def]
    """On a ten-pip stop the spread decides whether the trade was ever worth
    taking, so inventing one would quietly answer that question."""
    replay = PlaybookReplay(engine)
    context = replay._context("EURUSD", frames, DECIDE_FROM, 0.00001)

    assert context is not None and context.tick is not None
    assert context.tick.spread == pytest.approx(10 * 0.00001)


# ------------------------------------------------------------- the report ---


def test_each_theory_is_reported_on_its_own(frames) -> None:  # type: ignore[no-untyped-def]
    """A blended number hides the finding: four theories that break even and
    one that bleeds average out to "needs tuning" when the answer is "delete
    the fifth"."""
    decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()
    orders = [order("range_fade", decided), order("momentum_scalp", decided, up=False)]

    evidence = evidence_by_playbook(orders, frames[DECISION_TIMEFRAME])

    assert {item.playbook for item in evidence} == {"range_fade", "momentum_scalp"}


def test_a_losing_theory_is_named_for_deletion(frames) -> None:  # type: ignore[no-untyped-def]
    losing = PlaybookEvidence("range_fade", 900, 200, -40.0, 0.3, -0.2, 12.0)

    printed = render([losing])

    assert "Negative over a real sample: range_fade" in printed
    assert "Switching one of those off" in printed


def test_a_thin_sample_is_not_read_as_a_verdict(frames) -> None:  # type: ignore[no-untyped-def]
    thin = PlaybookEvidence("failed_break", 12, 4, -1.0, 0.25, -0.25, 1.0)

    printed = render([thin])

    assert "Too few trades to read: failed_break" in printed
    assert "Negative over a real sample" not in printed


def test_the_missing_commission_is_stated(frames) -> None:  # type: ignore[no-untyped-def]
    """A thin positive that ignores commission is not a positive."""
    printed = render([PlaybookEvidence("range_fade", 900, 200, 4.0, 0.5, 0.02, 3.0)])

    assert "commission is not" in printed


def test_an_empty_run_says_so() -> None:
    assert "No theory proposed anything" in render([])


def test_the_backtester_is_the_shared_pessimistic_one(frames) -> None:  # type: ignore[no-untyped-def]
    """Not a second copy with friendlier assumptions. Stop before target on a
    bar that touches both, gaps worsen fills, and none of that is re-derived
    here."""
    decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()

    evidence = evidence_by_playbook(
        [order("range_fade", decided)], frames[DECISION_TIMEFRAME], PessimisticBacktester()
    )

    assert len(evidence) == 1


def test_the_binary_search_slice_matches_the_mask_it_replaced(engine, frames) -> None:  # type: ignore[no-untyped-def]
    """The optimisation must be a pure speedup, not a change of meaning.

    The original walked every row testing `index + duration <= decided_at`.
    This asserts the binary search selects exactly the same bars, because an
    off-by-one here would hand a theory one bar of the future and nothing
    about the output would look wrong.
    """
    replay = PlaybookReplay(engine)
    decided_at = DECIDE_FROM

    context = replay._context("EURUSD", frames, decided_at, 0.00001)

    assert context is not None
    for timeframe in REPLAY_TIMEFRAMES:
        expected = frames[timeframe][
            frames[timeframe].index + timeframe.duration <= decided_at
        ].tail(300)
        assert context.series[timeframe].df.equals(expected)
