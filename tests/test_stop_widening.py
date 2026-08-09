"""Widen the stop instead of refusing the trade.

The cost gate alone is an off switch on this account. Measured against real
fills, all four live trades had stops between 1.8 and 6.3 pips, and every one
of them spends over a quarter of its risk on commission and slippage. A rule
that says no to all of them is not a risk control.

Widening is what a person does. The invalidation level does not move — the
trade is still wrong in the same place — it simply stops being sized as though
the market cannot breathe. The price is reward-to-risk, and that is already
measured elsewhere: if the target no longer justifies the wider stop, the RR
gate refuses it on the merits.
"""

from __future__ import annotations

import pytest

from analysis.confluence import TradeIdea
from config.loader import load_settings
from core.instrument import InstrumentSpec
from core.types import Direction
from risk.position_sizer import PositionSizer
from runner.service import JarvisRunner
from tests.fakes.fake_mt5 import eurusd_spec

ENTRY = 1.08500


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


def runner(limit: float = 0.25, slip: float = 1.7):  # type: ignore[no-untyped-def]
    base = load_settings(env_overrides=False)
    risk = base.risk.model_copy(
        update={
            "commission_per_lot_per_side": 2.75,
            "stop_slippage_pips": {"forex": slip},
            "max_cost_share_of_risk": limit,
        }
    )
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = base.model_copy(update={"risk": risk})  # type: ignore[assignment]
    return instance


def idea(stop_pips: float, direction: Direction = Direction.LONG) -> TradeIdea:
    sign = int(direction)
    return TradeIdea(
        symbol="EURUSD",
        approved=True,
        direction=direction,
        score=70.0,
        confidence=0.7,
        entry=ENTRY,
        stop_loss=ENTRY - stop_pips * 0.0001 * sign,
        take_profit=ENTRY + stop_pips * 0.0004 * sign,
        reason="test",
        signals=(),
    )


def stop_pips_of(result: TradeIdea, spec: InstrumentSpec) -> float:
    return spec.price_to_pips(abs(result.entry - result.stop_loss))


class TestWidening:
    def test_a_scalp_stop_is_pushed_out_to_where_it_can_pay(self, spec: InstrumentSpec) -> None:
        """The AUDNZD shape: 5 pips in, viable width out."""
        widened = runner()._widen_stop_for_costs(idea(5.0), spec)
        assert stop_pips_of(widened, spec) > 5.0

    def test_it_lands_just_inside_the_limit_not_on_it(self, spec: InstrumentSpec) -> None:
        """Solved, not stepped outward — every extra pip is real money at risk.

        Commission EUR 5.50/lot plus 1.7 pips of slip at EUR 10/pip is EUR
        22.50 of cost; at a 25% budget that needs EUR 90 of price risk, or 9
        pips. It lands at 9.5, five percent inside, because aiming at exactly
        the limit loses to the float: the stop is normalised to the tick and
        comes back as 8.99999, which the gate then refuses.
        """
        widened = runner()._widen_stop_for_costs(idea(5.0), spec)
        assert stop_pips_of(widened, spec) == pytest.approx(9.5, abs=0.1)

    def test_the_widened_stop_passes_the_gate_it_used_to_fail(self, spec: InstrumentSpec) -> None:
        service = runner()
        widened = service._widen_stop_for_costs(idea(5.0), spec)
        result = PositionSizer(service.settings).size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=widened.entry,
            sl=widened.stop_loss,
            tp=widened.take_profit,
        )
        assert str(result.reason) != "SL_TOO_TIGHT_FOR_COSTS"

    def test_a_short_widens_upward(self, spec: InstrumentSpec) -> None:
        """Sign errors here would put the stop on the wrong side of entry."""
        widened = runner()._widen_stop_for_costs(idea(5.0, Direction.SHORT), spec)
        assert widened.stop_loss > widened.entry
        assert stop_pips_of(widened, spec) == pytest.approx(9.5, abs=0.1)


class TestItLeavesGoodTradesAlone:
    def test_a_stop_that_already_pays_is_untouched(self, spec: InstrumentSpec) -> None:
        original = idea(25.0)
        assert runner()._widen_stop_for_costs(original, spec) is original

    def test_the_target_is_never_moved(self, spec: InstrumentSpec) -> None:
        """Preserving the ratio would mean inventing a level the chart never
        offered. The RR gate is allowed to refuse the result instead."""
        original = idea(5.0)
        widened = runner()._widen_stop_for_costs(original, spec)
        assert widened.take_profit == original.take_profit
        assert widened.entry == original.entry

    def test_widening_costs_reward_to_risk_and_that_is_visible(self, spec: InstrumentSpec) -> None:
        """A 5-pip stop with a 20-pip target is 1:4. At 9 pips it is 1:2.2.

        Still tradeable here, and the point is that the loss is real and lands
        where the system already measures it rather than being hidden.
        """
        widened = runner()._widen_stop_for_costs(idea(5.0), spec)
        rr = abs(widened.take_profit - widened.entry) / abs(widened.entry - widened.stop_loss)
        assert rr == pytest.approx(2.1, abs=0.1)

    def test_switching_the_check_off_disables_widening_too(self, spec: InstrumentSpec) -> None:
        original = idea(2.0)
        assert runner(limit=0.0)._widen_stop_for_costs(original, spec) is original

    def test_a_missing_direction_is_left_alone(self, spec: InstrumentSpec) -> None:
        original = idea(2.0)
        blank = TradeIdea(
            symbol="EURUSD",
            approved=False,
            direction=None,
            score=0.0,
            confidence=0.0,
            entry=original.entry,
            stop_loss=original.stop_loss,
            take_profit=original.take_profit,
            reason="no direction",
            signals=(),
        )
        assert runner()._widen_stop_for_costs(blank, spec) is blank

    def test_a_zero_width_stop_is_not_the_place_to_fix_that(self, spec: InstrumentSpec) -> None:
        """`_validate_stop` owns that refusal; this must not divide by zero."""
        broken = TradeIdea(
            symbol="EURUSD",
            approved=True,
            direction=Direction.LONG,
            score=70.0,
            confidence=0.7,
            entry=ENTRY,
            stop_loss=ENTRY,
            take_profit=ENTRY + 0.0020,
            reason="test",
            signals=(),
        )
        assert runner()._widen_stop_for_costs(broken, spec) is broken


def test_no_slippage_measurement_still_widens_for_commission(spec: InstrumentSpec) -> None:
    """Commission alone is enough to make a 1-pip stop unpayable.

    EUR 5.50 of cost at a 25% budget needs EUR 22 of price risk: 2.2 pips,
    2.3 with the margin. Modest, and correct — the gate only ever claims what
    it can measure.
    """
    widened = runner(slip=0.0)._widen_stop_for_costs(idea(1.0), spec)
    assert stop_pips_of(widened, spec) == pytest.approx(2.3, abs=0.1)
