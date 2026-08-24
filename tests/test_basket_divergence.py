"""Section five: one index stepped out of line with the others.

The first reader on this account that is MEANINGLESS on a single chart, which
is why it can corroborate the nine that all read one. It does not forecast the
market — it bets a gap closes — and that is a different and easier claim.

Two tests here matter more than the rest: the stale-peer guard, because
comparing a closed market to an open one manufactures gaps that never close;
and the basket-must-have-moved guard, because the same arithmetic that finds a
laggard also finds a market decoupling on its own, and those are opposite
trades.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.basket_divergence import (
    BASKET_META_KEY,
    BasketDivergence,
    PeerMove,
    divergence,
)
from config.schema import BasketDivergenceConfig
from core.types import MarketContext, Series, Timeframe

BARS = 15


def context(move_bp: float, peers: list[PeerMove] | None = None) -> MarketContext:
    """A market that moved `move_bp` over the comparison window."""
    total = 1.0 + move_bp / 10_000.0
    steps = np.linspace(1.0, total, BARS + 1)
    closes = 7600.0 * np.r_[np.ones(40), steps]
    index = pd.date_range("2026-08-24T09:00", periods=len(closes), freq="1min", tz=UTC)
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.00002,
            "low": closes * 0.99998,
            "close": closes,
            "tick_volume": 100,
            "spread": 1,
            "real_volume": 0,
        },
        index=index,
    )
    now = index[-1].to_pydatetime()
    ctx = MarketContext(
        symbol="SPX500",
        now=now,
        series={Timeframe.M1: Series("SPX500", Timeframe.M1, frame, now)},
        tick=None,
    )
    if peers is not None:
        ctx.meta[BASKET_META_KEY] = peers
    return ctx


def peer(symbol: str, move_bp: float, age: float = 10.0) -> PeerMove:
    return PeerMove(symbol=symbol, move_bp=move_bp, age_seconds=age)


class TestTheBasket:
    def test_the_basket_is_a_median_not_a_mean(self) -> None:
        """One index halted, gapping on its own news, or mispriced for a minute
        would drag a mean far enough to invent a divergence in every other
        member of the group at once. The median ignores it."""
        peers = [peer("A", 40.0), peer("B", 42.0), peer("C", 4000.0)]

        found = divergence(41.0, peers, max_age_seconds=180.0, minimum_peers=2)

        assert found is not None
        _, basket, used = found
        assert used == 3
        assert basket == pytest.approx(42.0)  # not the ~1360 a mean would give

    def test_too_few_peers_is_no_reading(self) -> None:
        assert divergence(10.0, [peer("A", 40.0)], max_age_seconds=180.0, minimum_peers=2) is None


class TestTheSessionTrap:
    """FRA40 does not keep SPX500's hours. Comparing a closed market to an open
    one compares a STALE price to a live one and manufactures gaps that will
    never close, because one side stopped printing hours ago.

    The guard is a last-bar timestamp rather than a session table on purpose: a
    calendar can be wrong about a holiday or a half-day, a timestamp cannot.
    """

    def test_a_stale_peer_is_not_counted(self) -> None:
        fresh = [peer("A", 40.0, age=10.0), peer("B", 41.0, age=10.0)]
        stale = [peer("A", 40.0, age=4000.0), peer("B", 41.0, age=4000.0)]

        assert divergence(0.0, fresh, max_age_seconds=180.0, minimum_peers=2) is not None
        assert divergence(0.0, stale, max_age_seconds=180.0, minimum_peers=2) is None

    def test_a_closed_market_cannot_make_a_setup_on_its_own(self) -> None:
        """The whole trap in one assertion: two closed indices showing a big
        gap must produce nothing, not the largest divergence of the day."""
        signal = BasketDivergence().analyze(
            context(0.0, [peer("FRA40", 80.0, age=7200.0), peer("UK100", 82.0, age=7200.0)])
        )

        assert signal.score == 0.0
        assert "stale peer is a closed market" in signal.reasoning


class TestTheSignal:
    def test_a_laggard_is_bought_toward_its_basket(self) -> None:
        signal = BasketDivergence().analyze(
            context(5.0, [peer("NDX100", 45.0), peer("US30", 42.0), peer("UK100", 44.0)])
        )

        assert signal.score > 0
        assert signal.details["gap_bp"] > 20.0

    def test_a_laggard_on_the_way_down_is_sold(self) -> None:
        signal = BasketDivergence().analyze(
            context(-5.0, [peer("NDX100", -45.0), peer("US30", -42.0), peer("UK100", -44.0)])
        )

        assert signal.score < 0

    def test_a_market_in_line_with_its_basket_says_nothing(self) -> None:
        signal = BasketDivergence().analyze(
            context(40.0, [peer("NDX100", 42.0), peer("US30", 41.0)])
        )

        assert signal.score == 0.0
        assert "in line with its basket" in signal.reasoning

    def test_a_market_decoupling_on_its_own_is_refused(self) -> None:
        """THE OTHER GUARD THAT MATTERS. The gap is a difference, so it can be
        large because the basket ran OR because this market ran while the
        basket sat still. The second is the decoupling this module is most
        wrong about, and it is the opposite trade."""
        signal = BasketDivergence().analyze(context(60.0, [peer("NDX100", 1.0), peer("US30", 2.0)]))

        assert signal.score == 0.0
        assert "decoupling" in signal.reasoning

    def test_a_market_that_has_overshot_its_peers_is_not_a_laggard(self) -> None:
        """Ran further than a basket that genuinely moved. That is not a gap to
        close in this direction, and trading it would be chasing."""
        signal = BasketDivergence().analyze(
            context(90.0, [peer("NDX100", 40.0), peer("US30", 41.0)])
        )

        assert signal.score == 0.0
        assert "overshot" in signal.reasoning

    def test_a_bigger_gap_scores_higher(self) -> None:
        small = BasketDivergence().analyze(context(15.0, [peer("A", 40.0), peer("B", 41.0)]))
        large = BasketDivergence().analyze(context(-20.0, [peer("A", 40.0), peer("B", 41.0)]))

        assert abs(large.score) > abs(small.score)
        assert large.confidence >= small.confidence

    def test_no_peers_means_no_opinion(self) -> None:
        signal = BasketDivergence().analyze(context(40.0))

        assert signal.score == 0.0
        assert "no basket peers" in signal.reasoning

    def test_the_reading_shows_its_working(self) -> None:
        signal = BasketDivergence().analyze(context(5.0, [peer("A", 45.0), peer("B", 44.0)]))

        for key in ("gap_bp", "basket_bp", "own_bp", "peers"):
            assert key in signal.details, key

    def test_disabling_it_silences_it(self) -> None:
        module = BasketDivergence(BasketDivergenceConfig(enabled=False))

        assert module.analyze(context(5.0, [peer("A", 45.0), peer("B", 44.0)])).score == 0.0


class TestItIsWiredToTradeAndToStandApart:
    def _confluence(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_it_is_live_unlike_sections_two_and_four(self) -> None:
        """Those rest on a statistic measured on TICK data, so whether it
        survives M1 bars is an open question and they wait on paper. A move
        between two M1 closes has no such question."""
        from core.types import TradingMode

        confluence = self._confluence()

        assert "basket_divergence" in confluence.live_enabled_modules
        assert confluence.effective_weights(TradingMode.MICRO_LIVE)["basket_divergence"] > 0
        assert confluence.effective_weights(TradingMode.MICRO_LIVE)["drift_burst"] == 0.0

    def test_it_is_its_own_evidence_family(self) -> None:
        """It is the only reader whose evidence is a RELATION between two
        instruments rather than a property of one. Filing it beside a chart
        family would let it corroborate exactly the reading it exists to
        check."""
        from analysis.evidence_families import family_for

        assert family_for("basket_divergence") == "relative"
        for other in ("trend_momentum", "impulse_break", "market_structure", "drift_burst"):
            assert family_for(other) != "relative"

    def test_it_can_contradict_a_chart_reader_when_it_matters(self) -> None:
        """The reason for the weight. Four indices up and a detector wanting
        short on the fifth is exactly the case this exists for, so its vote has
        to be heavy enough to be heard against one."""
        confluence = self._confluence()

        assert confluence.weights["basket_divergence"] >= confluence.weights["drift_continuation"]


class TestTheConfigCannotBeIncoherent:
    def test_saturation_must_exceed_the_minimum_gap(self) -> None:
        with pytest.raises(ValueError, match="must exceed minimum_gap_bp"):
            BasketDivergenceConfig(minimum_gap_bp=50.0, gap_saturation_bp=40.0)

    def test_confidence_may_not_narrow_to_nothing(self) -> None:
        with pytest.raises(ValueError, match="below base_confidence"):
            BasketDivergenceConfig(base_confidence=0.9, maximum_confidence=0.5)

    def test_a_basket_of_one_is_a_pair_and_is_refused(self) -> None:
        with pytest.raises(ValueError):
            BasketDivergenceConfig(minimum_peers=1)


class TestPeerAgeIsMeasuredNotAssumed:
    def test_the_age_travels_with_the_reading(self) -> None:
        """845 markets cannot be read simultaneously by anything, so peers are
        always a few seconds old. That is fine and it is why the age is carried
        rather than assumed to be zero."""
        now = datetime(2026, 8, 24, 9, 55, tzinfo=UTC)
        stamp = now - timedelta(seconds=45)
        reading = PeerMove("NDX100", 40.0, (now - stamp).total_seconds())

        assert reading.age_seconds == pytest.approx(45.0)
