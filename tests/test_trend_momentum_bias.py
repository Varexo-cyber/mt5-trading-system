"""A bias timeframe with no opinion is not a bias timeframe voting against.

`trend_momentum` folded three situations into two answers. H1 flat is nothing
to trade. H4 trending against H1 is a real conflict. H4 *flat* while H1 trends
is neither — it is the absence of a headwind — and it was returning the same
hard zero as an outright disagreement.

That branch is the expensive one. Over a live twelve hours "no weighted
directional evidence" accounted for 18,150 of 36,331 no-signals, half of every
refusal the system made, and a bias timeframe with no opinion is the commonest
way to land there: H4 spends most of its life somewhere between a crossover and
a slope.

The fix has to stay a discount rather than a pass, which is what most of this
file is about. An unconfirmed trend is genuinely worth less than a confirmed
one, and if the discount ever stopped applying the module would be quietly
promoting half-evidence to full.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.modules import TrendMomentum
from config.schema import TrendMomentumConfig
from core.types import MarketContext, Series, Tick, Timeframe

BARS = 120


def _frame(step: float, start: float = 1.10) -> pd.DataFrame:
    """A series whose EMAs separate at `step` per bar. 0.0 is flat."""
    index = pd.date_range("2026-01-01", periods=BARS, freq="1h", tz=UTC)
    close = pd.Series([start + i * step for i in range(BARS)], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )


def context(h4_step: float, h1_step: float) -> MarketContext:
    now = datetime(2026, 1, 6, 10, tzinfo=UTC)
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={
            Timeframe.H4: Series("EURUSD", Timeframe.H4, _frame(h4_step), now),
            Timeframe.H1: Series("EURUSD", Timeframe.H1, _frame(h1_step), now),
        },
        tick=Tick("EURUSD", now, 1.1098, 1.1100),
    )


def analyse(h4_step: float, h1_step: float, **overrides):  # type: ignore[no-untyped-def]
    return TrendMomentum(TrendMomentumConfig(**overrides)).analyze(context(h4_step, h1_step))


class TestTheThreeSituationsAreThree:
    def test_both_timeframes_rising_is_a_confirmed_signal(self) -> None:
        signal = analyse(0.0008, 0.0006)

        assert signal.score > 0
        assert signal.details["bias_confirmed"] is True

    def test_the_bias_opposing_the_signal_is_still_refused(self) -> None:
        """The case the old branch existed for, and it must not change."""
        signal = analyse(-0.0008, 0.0006)

        assert signal.score == 0
        assert "opposes" in signal.reasoning

    def test_a_flat_signal_timeframe_is_still_refused(self) -> None:
        """Nothing to trade is nothing to trade, whatever H4 is doing."""
        signal = analyse(0.0008, 0.0)

        assert signal.score == 0
        assert "flat" in signal.reasoning

    def test_a_flat_bias_no_longer_kills_a_trending_signal(self) -> None:
        """The 18,150 rows. H4 has no view; that is not a vote against."""
        signal = analyse(0.0, 0.0006)

        assert signal.score > 0
        assert signal.details["bias_confirmed"] is False
        assert "unconfirmed" in signal.reasoning

    def test_direction_follows_the_signal_timeframe_when_the_bias_is_flat(self) -> None:
        """Reading direction off the bias would return zero for exactly the
        case this branch exists to serve."""
        assert analyse(0.0, -0.0006).score < 0


class TestItIsADiscountAndNotAPass:
    def test_an_unconfirmed_trend_is_worth_less_than_a_confirmed_one(self) -> None:
        confirmed = analyse(0.0008, 0.0006)
        unconfirmed = analyse(0.0, 0.0006)

        assert unconfirmed.confidence < confirmed.confidence

    def test_the_discount_is_the_configured_fraction(self) -> None:
        confirmed = analyse(0.0008, 0.0006)
        unconfirmed = analyse(0.0, 0.0006, neutral_bias_confidence_scale=0.5)

        assert unconfirmed.confidence == confirmed.confidence * 0.5

    def test_zero_restores_the_old_behaviour_exactly(self) -> None:
        """The escape hatch has to actually close the door, not merely narrow
        it — otherwise there is no way to undo this if it turns out badly."""
        signal = analyse(0.0, 0.0006, neutral_bias_confidence_scale=0.0)

        assert signal.score == 0
        assert "neutral" in signal.reasoning

    def test_a_steep_discount_can_drop_it_under_the_confluence_floor(self) -> None:
        """What keeps this honest downstream: the discounted confidence is
        compared against `minimum_confidence` like any other, so the engine can
        still throw the signal away without this module knowing about it."""
        signal = analyse(0.0, 0.0006, neutral_bias_confidence_scale=0.1)

        assert signal.score > 0
        assert signal.confidence < 0.45


class TestTheStopStillComesFromTheSignalTimeframe:
    def test_an_unconfirmed_long_still_carries_an_invalidation_level(self) -> None:
        """A signal without one cannot be sized, and this branch is new — if
        the invalidation were only set on the confirmed path, every trade it
        creates would be refused later for a missing stop."""
        signal = analyse(0.0, 0.0006)
        last_close = float(_frame(0.0006)["close"].iloc[-1])

        assert signal.invalidation_price is not None
        assert signal.invalidation_price < last_close
