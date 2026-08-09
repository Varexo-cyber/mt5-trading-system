"""A stop can be legal at the broker and still not worth taking.

A live AUDNZD long had a 5-pip stop. The broker accepted it, every gate passed
it, and it returned -1.48R on a -1.00R plan: commission was 0.22R and the fill
came 1.7 pips *through* the stop for another 0.34R. Over half the risk was the
cost of trading.

At that ratio the strategy is not what is being tested. The gate here is not
about setup quality at all — it asks whether the trade, rather than the cost of
taking it, will decide the outcome.
"""

from __future__ import annotations

import pytest

from config.loader import load_settings
from config.schema import Settings
from core.instrument import InstrumentSpec
from core.types import Direction
from risk.position_sizer import PositionSizer
from tests.fakes.fake_mt5 import eurusd_spec


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


def settings_with(**overrides: object) -> Settings:
    base = load_settings(env_overrides=False)
    risk = base.risk.model_copy(
        update={
            "commission_per_lot_per_side": 2.75,
            "stop_slippage_pips": {"forex": 1.7},
            "max_cost_share_of_risk": 0.25,
            **overrides,
        }
    )
    return base.model_copy(update={"risk": risk})


def size(  # type: ignore[no-untyped-def]
    settings: Settings, spec: InstrumentSpec, stop_pips: float, spread_pips: float = 0.0
):
    entry = 1.08500
    return PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=entry,
        sl=entry - stop_pips * 0.0001,
        tp=entry + stop_pips * 0.0003,
        spread_price=spread_pips * 0.0001,
    )


class TestTheGate:
    def test_the_audnzd_shape_is_refused(self, spec: InstrumentSpec) -> None:
        """5 pips: commission and slippage together are most of the risk."""
        result = size(settings_with(), spec, 5.0)
        assert not result.approved
        assert str(result.reason) == "SL_TOO_TIGHT_FOR_COSTS"
        assert "of the risk" in result.decision.detail

    def test_the_refusal_states_what_a_stop_out_would_really_cost(
        self, spec: InstrumentSpec
    ) -> None:
        """The number the operator needs, not a threshold name."""
        detail = size(settings_with(), spec, 5.0).decision.detail
        assert "rather than 1.00R" in detail

    def test_a_normal_stop_passes(self, spec: InstrumentSpec) -> None:
        assert size(settings_with(), spec, 25.0).approved

    def test_the_boundary_is_where_the_arithmetic_puts_it(self, spec: InstrumentSpec) -> None:
        """Commission EUR 5.50/lot round trip plus 1.7 pips of slip at EUR
        10/pip is EUR 22.50 of cost per lot.

        Against EUR 10 of risk per pip that is a quarter of the risk at exactly
        nine pips — too close to assert on, since the stop distance arrives as
        0.00089999. Ten passes at 22.5%, eight fails at 28.1%.
        """
        assert size(settings_with(), spec, 10.0).approved
        assert not size(settings_with(), spec, 8.0).approved

    def test_switching_it_off_restores_the_old_behaviour(self, spec: InstrumentSpec) -> None:
        """Default is 0, so no profile changes without being told to."""
        assert size(settings_with(max_cost_share_of_risk=0.0), spec, 5.0).approved
        assert load_settings(env_overrides=False).risk.max_cost_share_of_risk == 0.0

    def test_an_unmeasured_asset_class_rests_on_commission_alone(
        self, spec: InstrumentSpec
    ) -> None:
        """No slippage figure is not an excuse to invent one.

        Without a measurement the gate still catches the worst stops on
        commission, and says nothing it cannot support.
        """
        settings = settings_with(stop_slippage_pips={})
        assert not size(settings, spec, 2.0).approved
        assert size(settings, spec, 5.0).approved


class TestItIsAboutCostNotQuality:
    def test_a_wide_stop_is_never_touched_by_this_gate(self, spec: InstrumentSpec) -> None:
        for stop_pips in (15.0, 25.0, 30.0):
            result = size(settings_with(), spec, stop_pips)
            assert str(result.reason) != "SL_TOO_TIGHT_FOR_COSTS", stop_pips

    def test_the_share_falls_monotonically_as_the_stop_widens(self, spec: InstrumentSpec) -> None:
        sizer = PositionSizer(settings_with())
        shares = [sizer._cost_share(spec, pips * 0.0001, 5.50) for pips in (2.0, 5.0, 10.0, 30.0)]
        assert shares == sorted(shares, reverse=True)
        assert shares[0] > 1.0  # a 2-pip stop costs more than it risks
        assert shares[-1] < 0.1

    def test_a_zero_width_stop_reads_as_all_cost(self, spec: InstrumentSpec) -> None:
        """Guarded rather than dividing by zero; `_validate_stop` rejects it
        first in practice, but this must not be the thing that raises."""
        assert PositionSizer(settings_with())._cost_share(spec, 0.0, 5.50) == 1.0


class TestTheSpreadIsPartOfTheCost:
    """The term the gate used to leave out, and the largest one on this account.

    Two gates existed and neither saw the whole bill. The playbooks refuse a
    setup whose spread is more than 12-15% of the stop; this one refused
    commission plus slippage over 25%. Nobody added them together, so a trade
    could satisfy both while costing nearly 40% of its own risk to place. That
    is the shape of every tight-stop loss in the last twenty trades.
    """

    def test_a_stop_that_passes_without_the_spread_fails_with_it(
        self, spec: InstrumentSpec
    ) -> None:
        """Ten pips: 22.5% on commission and slippage alone, under the 25%
        limit. Add the 1.5-pip spread the playbooks would happily allow at that
        width and the real bill is 37.5%."""
        assert size(settings_with(), spec, 10.0).approved
        refused = size(settings_with(), spec, 10.0, spread_pips=1.5)

        assert not refused.approved
        assert str(refused.reason) == "SL_TOO_TIGHT_FOR_COSTS"

    def test_the_refusal_names_the_spread(self, spec: InstrumentSpec) -> None:
        detail = size(settings_with(), spec, 10.0, spread_pips=1.5).decision.detail
        assert "spread" in detail
        assert "1.38R rather than 1.00R" in detail

    def test_a_wide_stop_still_absorbs_a_normal_spread(self, spec: InstrumentSpec) -> None:
        """The point is not to refuse everything. A 30-pip stop pays the same
        spread and barely notices it — which is the argument for wider stops
        rather than for a stricter gate."""
        assert size(settings_with(), spec, 30.0, spread_pips=1.5).approved

    def test_the_spread_counts_once_not_twice(self, spec: InstrumentSpec) -> None:
        """Filled at the ask, closed at the bid: one crossing per round trip.
        `analysis.playbooks._spread_is_affordable` measures it the same way."""
        sizer = PositionSizer(settings_with())
        without = sizer._cost_share(spec, 0.0010, 5.50, 0.0)
        with_spread = sizer._cost_share(spec, 0.0010, 5.50, 0.00015)

        assert with_spread - without == pytest.approx(0.00015 / 0.0010, abs=1e-9)

    def test_not_supplying_it_leaves_the_old_answer(self, spec: InstrumentSpec) -> None:
        """Backtests and older callers pass nothing. That understates the cost
        and is documented as doing so; it must not change what they measured."""
        sizer = PositionSizer(settings_with())

        assert sizer._cost_share(spec, 0.0010, 5.50) == sizer._cost_share(spec, 0.0010, 5.50, 0.0)

    def test_the_share_still_falls_as_the_stop_widens(self, spec: InstrumentSpec) -> None:
        sizer = PositionSizer(settings_with())
        shares = [
            sizer._cost_share(spec, pips * 0.0001, 5.50, 0.00015) for pips in (2.0, 5.0, 10.0, 30.0)
        ]

        assert shares == sorted(shares, reverse=True)
        assert shares[-1] < 0.15


class TestTheLiveCallerHandsOverTheRealSpread:
    """A gate that reads the spread is worthless if the runner passes zero.

    This is the wiring, checked by reading the call rather than by running a
    cycle: `JarvisRunner` is the only production caller and it has to be
    passing `spread_price`, or the whole change is decoration.
    """

    def test_the_runner_passes_the_ticks_spread(self) -> None:
        import inspect

        from runner import service

        source = inspect.getsource(service.JarvisRunner)
        call = source[source.index("PositionSizer(self.settings).size(") :]
        call = call[: call.index(")\n")]

        assert "spread_price=" in call, "the sizer is being called without the spread"
        assert "context.tick.spread" in call, "it must be the live tick, not a constant"
