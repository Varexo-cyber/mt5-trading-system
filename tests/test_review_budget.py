"""Paying Claude for the best few ideas, not for all of them.

A live cycle sent five candidates and was told, in the reviewer's own words,
that one was "the weakest setup of the 10 tradeable candidates this cycle" and
another was "ranked dead last (11 of 11)". Both answers were correct and both
cost real money to obtain. The engine had already ranked those candidates last
itself — nothing was learned.

The budget is deliberately not a rank cut-off. Candidates are processed in rank
order and a low rank is only *reached* because the better ones were rejected by
some earlier gate, so "never review below rank 3" would silently stop trading
on any day the top three keep failing. A budget cannot do that: whatever
reaches the reviewer first is by construction the best thing still standing.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from advisory.providers import Advice
from analysis.confluence import TradeIdea
from core.types import Direction, MarketContext, Series, Tick, Timeframe
from runner.service import JarvisRunner

NOW = datetime(2026, 8, 6, 14, 30, tzinfo=UTC)


def runner(budget: int = 3):  # type: ignore[no-untyped-def]
    """A runner carrying only what the budget path reads."""
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = SimpleNamespace(  # type: ignore[assignment]
        ai=SimpleNamespace(max_reviews_per_cycle=budget)
    )
    instance._review_cache = {}  # type: ignore[assignment]
    instance._reviews_this_cycle = 0  # type: ignore[assignment]
    instance.advisor = SimpleNamespace(review=_counting_advisor(instance))  # type: ignore[assignment]
    return instance


def _counting_advisor(instance):  # type: ignore[no-untyped-def]
    def review(idea, context, proposal, memory):  # type: ignore[no-untyped-def]
        instance.calls_made = getattr(instance, "calls_made", 0) + 1
        return Advice(True, 0.8, "fine", provider="anthropic")

    return review


def context(symbol: str = "EURUSD.i", *, last_bar: int = 0) -> MarketContext:
    """M5 is the fastest frame, so its last close keys the review cache."""
    index = pd.date_range(
        NOW.replace(minute=0) - pd.Timedelta(minutes=5 * (60 - last_bar)),
        periods=60,
        freq="5min",
        tz=UTC,
    )
    base = 1.085 + np.cumsum(np.full(60, 0.0001))
    frame = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.0006,
            "low": base - 0.0006,
            "close": base + 0.0001,
            "tick_volume": 100,
            "spread": 12,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol=symbol,
        now=NOW,
        series={
            Timeframe.M5: Series(symbol=symbol, timeframe=Timeframe.M5, df=frame, fetched_at=NOW)
        },
        tick=Tick(symbol=symbol, time=NOW, bid=1.0850, ask=1.08512),
    )


def swing_context(*, last_m5_bar: int = 0) -> MarketContext:
    fast = context(last_bar=last_m5_bar)
    h1_index = pd.date_range(NOW - pd.Timedelta(hours=59), periods=60, freq="h", tz=UTC)
    base = 1.08 + np.cumsum(np.full(60, 0.0005))
    h1 = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.001,
            "low": base - 0.001,
            "close": base + 0.0002,
            "tick_volume": 100,
            "spread": 12,
            "real_volume": 0,
        },
        index=h1_index,
    )
    return MarketContext(
        symbol=fast.symbol,
        now=NOW,
        series={
            **fast.series,
            Timeframe.H1: Series(fast.symbol, Timeframe.H1, h1, NOW),
        },
        tick=fast.tick,
    )


def idea(symbol: str = "EURUSD.i") -> TradeIdea:
    return TradeIdea(
        symbol=symbol,
        approved=True,
        direction=Direction.LONG,
        score=70.0,
        confidence=0.7,
        entry=1.08512,
        stop_loss=1.0840,
        take_profit=1.0875,
        reason="test",
        signals=(),
    )


class TestBudget:
    def test_a_fresh_cycle_has_its_full_budget(self) -> None:
        assert runner(3)._review_budget_left() == 3

    def test_zero_means_no_budget_at_all(self) -> None:
        """Not "never review" — the setting is an off switch for the cap."""
        service = runner(0)
        for _ in range(20):
            service._reviews_this_cycle += 1
        assert service._review_budget_left() is None

    def test_each_paid_review_spends_one(self) -> None:
        """Three *different* candidates, as a cycle actually presents them."""
        service = runner(3)
        for symbol, expected in (("EURUSD.i", 2), ("GBPUSD.i", 1), ("AUDJPY.i", 0)):
            service._reviewed(idea(symbol), context(symbol), {}, None)
            assert service._review_budget_left() == expected

    def test_it_never_goes_negative(self) -> None:
        service = runner(2)
        for _ in range(5):
            service._reviewed(idea(f"SYM{_}"), context(f"SYM{_}"), {}, None)
        assert service._review_budget_left() == 0

    def test_a_replayed_verdict_is_free(self) -> None:
        """The cache existed before the budget and must not be rationed.

        Charging budget for a replay would make a cheap cycle look expensive
        and starve the candidates that genuinely need asking — the exact
        opposite of what the budget is for.
        """
        service = runner(3)
        first = service._reviewed(idea(), context(), {}, None)
        assert service._review_budget_left() == 2

        again = service._reviewed(idea(), context(), {}, None)
        assert service._review_budget_left() == 2
        assert service.calls_made == 1
        # Same verdict, marked as a replay so the spend report does not bill
        # the original call's tokens a second time.
        assert not first.replayed
        assert again.replayed
        assert (again.approved, again.confidence, again.thesis) == (
            first.approved,
            first.confidence,
            first.thesis,
        )

    def test_a_new_bar_is_a_new_question_and_costs_again(self) -> None:
        service = runner(3)
        service._reviewed(idea(), context(last_bar=0), {}, None)
        service._reviewed(idea(), context(last_bar=1), {}, None)
        assert service.calls_made == 2
        assert service._review_budget_left() == 1

    def test_a_swing_wait_is_reconsidered_on_a_new_entry_timing_bar(self) -> None:
        service = runner(3)
        swing = idea()
        assert swing.planning_timeframe == "H1"

        service._reviewed(swing, swing_context(last_m5_bar=0), {}, None)
        service._reviewed(swing, swing_context(last_m5_bar=1), {}, None)

        assert service.calls_made == 2

    def test_a_materially_different_plan_on_the_same_bar_is_a_new_question(self) -> None:
        service = runner(3)
        original = idea()
        relocated = replace(
            original,
            entry=original.entry + 0.0010,
            stop_loss=original.stop_loss + 0.0010,
            take_profit=original.take_profit + 0.0010,
        )

        service._reviewed(original, context(), {}, None)
        service._reviewed(relocated, context(), {}, None)

        assert service.calls_made == 2

    def test_an_intraday_plan_expires_on_its_planning_bar_not_h1(self) -> None:
        service = runner(3)
        intraday = replace(
            idea(),
            setup_family="momentum_scalp",
            horizon="intraday",
            planning_timeframe="M5",
            expected_horizon_minutes=60,
        )

        service._reviewed(intraday, context(last_bar=0), {}, None)
        service._reviewed(intraday, context(last_bar=1), {}, None)

        assert service.calls_made == 2


class TestCachedReviewLookup:
    def test_nothing_on_file_reads_as_none(self) -> None:
        assert runner()._cached_review(idea(), context()) is None

    def test_the_lookup_alone_never_spends_budget(self) -> None:
        """The gate asks "would this cost anything" before committing."""
        service = runner(3)
        service._reviewed(idea(), context(), {}, None)
        spent = service._reviews_this_cycle

        for _ in range(5):
            assert service._cached_review(idea(), context()) is not None
        assert service._reviews_this_cycle == spent

    def test_a_different_symbol_is_a_different_question(self) -> None:
        service = runner(3)
        service._reviewed(idea("EURUSD.i"), context("EURUSD.i"), {}, None)
        assert service._cached_review(idea("GBPUSD.i"), context("GBPUSD.i")) is None
