"""Rank the setup that keeps what it wins, not the one the engine likes most.

Two readings on this account say conviction does not predict the outcome. The
"20+ over the bar" bucket was the worst of them all at -4.92R over 23 trades,
and among 84 paid reviews the 40-45 conviction band produced nothing useful
while 20-25 produced 33%. Conviction was nonetheless worth 16 to 76 points in
the selection score, so the queue was ordered almost entirely by noise.

The toll is not noise. Measured on real fills, every live trade spent over a
quarter of its risk on commission and slippage, and that entered the ordering
only as a spread preference that never once looked at the target — so a two-pip
spread scored identically against a forty-pip target and a six-pip one, when it
is a rounding error against the first and half the winnings against the second.

What is held here is that the new term is a subtraction and not a forecast, and
that it stays a preference: it reorders, it never approves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from runner.service import AnalysedCandidate, JarvisRunner


class Spec:
    """Enough of an InstrumentSpec for the sizer's cost model."""

    def __init__(self, asset_class: str = "forex") -> None:
        self.symbol = "EURUSD.i"
        self.asset_class = SimpleNamespace(value=asset_class)
        self.point = 0.00001
        self.digits = 5

    def money_per_lot(self, distance: float) -> float:
        return distance * 100_000.0

    def pips_to_price(self, pips: float) -> float:
        return pips * 0.0001


def candidate(*, stop: float, target: float, spread: float = 0.00002) -> AnalysedCandidate:
    entry = 1.10000
    idea = SimpleNamespace(
        entry=entry,
        stop_loss=entry - stop,
        take_profit=entry + target,
        score=50.0,
        confidence=0.7,
        direction=SimpleNamespace(name="LONG"),
        signals=(),
    )
    context = SimpleNamespace(tick=SimpleNamespace(spread=spread))
    return AnalysedCandidate(
        symbol="EURUSD.i", cycle_id="c", idea=idea, context=context  # type: ignore[arg-type]
    )


def service(weight: float = 12.0, commission: float = 3.5, slippage: float = 0.3):  # type: ignore[no-untyped-def]
    """A bare service object: only the settings and the broker are touched."""
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = SimpleNamespace(  # type: ignore[attr-defined]
        scanner=SimpleNamespace(after_cost_priority_weight=weight),
        risk=SimpleNamespace(
            commission_per_lot=lambda _cls: commission,
            stop_slippage_pips={"forex": slippage},
        ),
    )
    instance.broker = SimpleNamespace(spec=lambda _symbol: Spec())  # type: ignore[attr-defined]
    return instance


class TestItPrefersTheSetupThatKeepsMore:
    def test_a_distant_target_outranks_a_near_one_at_the_same_risk(self) -> None:
        """The case the spread preference could never see: identical spread,
        identical stop, and one of them hands most of its winnings back."""
        wide = service()._after_cost_priority(candidate(stop=0.0010, target=0.0030))
        narrow = service()._after_cost_priority(candidate(stop=0.0010, target=0.0006))

        assert wide > narrow

    def test_the_same_target_is_worth_less_on_a_wider_spread(self) -> None:
        cheap = service()._after_cost_priority(candidate(stop=0.0010, target=0.0030))
        dear = service()._after_cost_priority(candidate(stop=0.0010, target=0.0030, spread=0.00020))

        assert cheap > dear

    def test_a_tiny_stop_is_penalised_because_the_toll_dominates_it(self) -> None:
        """The measured failure on this account: 1.8 to 6.3 pip stops, each
        spending over a quarter of its risk on costs."""
        roomy = service()._after_cost_priority(candidate(stop=0.0020, target=0.0040))
        scalp = service()._after_cost_priority(candidate(stop=0.00018, target=0.00036))

        assert roomy > scalp


class TestItStaysAPreference:
    def test_it_is_bounded_by_the_configured_weight(self) -> None:
        """An unbounded ratio would let one spectacular setup vault a fallback
        instrument over a core market, and the lane order is deliberate."""
        absurd = service()._after_cost_priority(candidate(stop=0.0010, target=1.0))

        assert absurd <= 12.0

    def test_it_never_goes_negative(self) -> None:
        """A target that does not clear its own toll is already refused by the
        cost gate. Scoring it negative here would count one rejection twice."""
        hopeless = service()._after_cost_priority(
            candidate(stop=0.0010, target=0.00005, spread=0.00050)
        )

        assert hopeless == 0.0

    def test_zero_weight_switches_it_off_entirely(self) -> None:
        assert service(weight=0.0)._after_cost_priority(candidate(stop=0.001, target=0.003)) == 0.0

    def test_it_adds_into_the_selection_score_and_nothing_else(self) -> None:
        """Ordering only. `conviction` is untouched, so no gate anywhere sees a
        different number because of this."""
        item = candidate(stop=0.0010, target=0.0030)
        ranked = AnalysedCandidate(
            symbol=item.symbol,
            cycle_id=item.cycle_id,
            idea=item.idea,
            context=item.context,
            after_cost_priority=6.0,
        )

        assert ranked.conviction == pytest.approx(35.0)
        assert ranked.selection_score == pytest.approx(ranked.ranking_score + 6.0)


class TestItCannotBreakACycle:
    def test_a_broker_that_cannot_describe_the_symbol_scores_zero(self) -> None:
        """Ordering is a preference. A broker hiccup must leave the queue in
        its previous order, never empty it."""
        instance = service()
        instance.broker = SimpleNamespace(  # type: ignore[attr-defined]
            spec=lambda _s: (_ for _ in ()).throw(RuntimeError("terminal busy"))
        )

        assert instance._after_cost_priority(candidate(stop=0.001, target=0.003)) == 0.0

    def test_a_setup_without_a_target_scores_zero(self) -> None:
        item = candidate(stop=0.0010, target=0.0030)
        item.idea.take_profit = 0.0

        assert service()._after_cost_priority(item) == 0.0

    def test_a_setup_without_a_stop_scores_zero(self) -> None:
        item = candidate(stop=0.0010, target=0.0030)
        item.idea.stop_loss = 0.0

        assert service()._after_cost_priority(item) == 0.0
