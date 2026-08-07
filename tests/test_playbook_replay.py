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
    Comparison,
    PlaybookEvidence,
    PlaybookReplay,
    coin_flip,
    compare_to_chance,
    evidence_by_playbook,
    render,
    render_comparison,
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


# ------------------------------------------------------------- the control ---


class TestTheCoinFlip:
    """The control the whole exercise needed.

    Five theories all landing between 24% and 33% wins, when a 1R stop against
    a 2R target wins 33% by chance alone, is either a coincidence or the
    answer. Only a matched control can say which, and matched means everything
    held constant except the one thing the theory claims to know.
    """

    def test_only_the_direction_changes(self, frames) -> None:  # type: ignore[no-untyped-def]
        decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()
        original = [order("range_fade", decided), order("range_fade", decided, up=False)]

        flipped = coin_flip(original, seed=1)

        for before, after in zip(original, flipped, strict=True):
            assert after.decided_at == before.decided_at
            assert after.symbol == before.symbol
            assert after.entry == before.entry
            assert after.modules == before.modules

    def test_the_geometry_is_mirrored_not_merely_relabelled(self, frames) -> None:  # type: ignore[no-untyped-def]
        """A flipped long has to be a real short. Keeping the levels and
        changing the label would leave the stop below the entry on a short,
        which is not a trade — it is a target wearing the wrong name."""
        decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()
        original = [order("range_fade", decided) for _ in range(40)]

        flipped = coin_flip(original, seed=4)

        for before, after in zip(original, flipped, strict=True):
            sign = int(after.direction)
            assert (after.stop_loss - after.entry) * sign < 0, "stop on the wrong side"
            assert (after.take_profit - after.entry) * sign > 0, "target on the wrong side"
            assert abs(after.entry - after.stop_loss) == pytest.approx(
                abs(before.entry - before.stop_loss)
            )
            assert abs(after.take_profit - after.entry) == pytest.approx(
                abs(before.take_profit - before.entry)
            )

    def test_it_actually_flips_some_of_them(self, frames) -> None:  # type: ignore[no-untyped-def]
        """A control that returned the input unchanged would report a perfect
        edge of zero and look entirely plausible."""
        decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()
        original = [order("range_fade", decided) for _ in range(60)]

        flipped = coin_flip(original, seed=11)
        shorts = sum(1 for item in flipped if item.direction is Direction.SHORT)

        assert 15 < shorts < 45, f"{shorts}/60 shorts is not a coin"

    def test_the_same_seed_gives_the_same_coin(self, frames) -> None:  # type: ignore[no-untyped-def]
        """A result nobody can reproduce is not evidence."""
        decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()
        original = [order("range_fade", decided) for _ in range(30)]

        first = [item.direction for item in coin_flip(original, seed=7)]
        second = [item.direction for item in coin_flip(original, seed=7)]

        assert first == second

    def test_a_theory_inside_chance_is_marked_as_such(self) -> None:
        real = PlaybookEvidence("range_fade", 1666, 445, -151.0, 0.29, -0.340, 68.0)
        inside = Comparison(
            real=real,
            flip_win_rate=0.30,
            flip_expectancy_r=-0.31,
            flip_best_r=-0.28,
            flip_worst_r=-0.35,
        )

        assert not inside.beats_the_coin
        printed = render_comparison([inside])
        assert "inside chance" in printed
        assert "Not one theory beat guessing" in printed

    def test_beating_the_average_coin_is_not_enough(self) -> None:
        """It has to beat every shuffle. Landing inside the range chance
        produces has not shown it knows anything."""
        real = PlaybookEvidence("range_break", 259, 93, -16.0, 0.31, -0.178, 18.0)
        lucky = Comparison(
            real=real,
            flip_win_rate=0.30,
            flip_expectancy_r=-0.20,
            flip_best_r=-0.15,
            flip_worst_r=-0.26,
        )

        assert not lucky.beats_the_coin

    def test_a_real_edge_is_reported_with_its_caveat(self) -> None:
        real = PlaybookEvidence("trend_pullback", 500, 300, 30.0, 0.42, 0.100, 8.0)
        good = Comparison(
            real=real,
            flip_win_rate=0.31,
            flip_expectancy_r=-0.30,
            flip_best_r=-0.25,
            flip_worst_r=-0.34,
        )

        assert good.beats_the_coin
        assert good.edge_r == pytest.approx(0.40)
        printed = render_comparison([good])
        assert "Outside what chance produced: trend_pullback" in printed
        assert "nowhere near sufficient" in printed

    def test_several_seeds_are_used(self, frames) -> None:  # type: ignore[no-untyped-def]
        """One shuffle is a sample of one, and the question is whether the
        theory sits outside what chance produces — not whether it beat one
        particular coin."""
        decided = frames[DECISION_TIMEFRAME].index[3000].to_pydatetime()
        orders = [order("range_fade", decided)]
        evidence = evidence_by_playbook(orders, frames[DECISION_TIMEFRAME])

        comparisons = compare_to_chance(orders, frames[DECISION_TIMEFRAME], evidence, seeds=5)

        assert len(comparisons) == 1
        assert comparisons[0].flip_worst_r <= comparisons[0].flip_expectancy_r
        assert comparisons[0].flip_expectancy_r <= comparisons[0].flip_best_r


def test_a_coin_on_a_random_walk_lands_where_theory_says_it_should() -> None:
    """The control has to be right before it can judge anything.

    A 1R stop against a 2R target wins one time in three by chance alone —
    that is arithmetic, not an opinion, and it is the whole reason the live
    theories' 24% to 33% is damning. If the harness produced 60% here it would
    be flattering every result it ever printed, and nothing about the output
    would look wrong.
    """
    rng = np.random.default_rng(0)
    bars = 20000
    closes = 1.1000 + rng.normal(0, 0.00015, bars).cumsum()
    opens = np.concatenate([[closes[0]], closes[:-1]])
    wick = np.abs(closes - opens) * 0.4 + 3e-6
    index = pd.date_range(start=START, periods=bars, freq=timedelta(minutes=5), tz=UTC)
    walk = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + wick,
            "low": np.minimum(opens, closes) - wick,
            "close": closes,
            "tick_volume": np.full(bars, 500),
            "spread": np.zeros(bars),
            "real_volume": np.zeros(bars),
        },
        index=index,
    )
    risk = 0.0010
    orders = [
        BacktestOrder(
            symbol="X",
            decided_at=index[i].to_pydatetime(),
            direction=Direction.LONG,
            entry=float(closes[i]),
            stop_loss=float(closes[i]) - risk,
            take_profit=float(closes[i]) + 2 * risk,
            modules=("coin",),
        )
        for i in range(300, bars - 600, 60)
    ]

    result = PessimisticBacktester().run_non_overlapping(walk, coin_flip(orders, 3))

    assert result.sample_size > 100
    assert 0.25 < result.win_rate < 0.45, f"a coin won {result.win_rate:.0%}, theory says ~33%"
    # Slightly negative: the same slippage and commission every trade pays.
    assert -0.2 < result.expectancy_r < 0.05
