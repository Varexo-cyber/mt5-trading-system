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


def size(settings: Settings, spec: InstrumentSpec, stop_pips: float):  # type: ignore[no-untyped-def]
    entry = 1.08500
    return PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=entry,
        sl=entry - stop_pips * 0.0001,
        tp=entry + stop_pips * 0.0003,
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
