"""Commission belongs in the risk model, not in the bookkeeping.

A live AUDNZD stop-out cost EUR 1.93 against a modelled 1R of EUR 1.53. The
operator checked the deal in the terminal: EUR 0.33 of the EUR 0.40 gap was
commission. On the USDCHF short before it the gap was EUR 0.53 on a EUR 1.57 R.

Every threshold in this system is written in R — the give-back arms at 0.5R,
the profit lock secures half the peak, the stall rule needs 0.6R. An R that
omits a fifth of what a loss actually costs makes all of them wrong in the same
direction, and makes every expectancy figure the account produces flatter than
the truth.
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


def settings_with(commission: float, **overrides: object) -> Settings:
    base = load_settings(env_overrides=False)
    risk = base.risk.model_copy(update={"commission_per_lot_per_side": commission, **overrides})
    return base.model_copy(update={"risk": risk})


def size(settings: Settings, spec: InstrumentSpec, *, stop_pips: float, equity: float = 10_000.0):  # type: ignore[no-untyped-def]
    entry = 1.08500
    return PositionSizer(settings).size(
        spec=spec,
        equity=equity,
        direction=Direction.LONG,
        entry=entry,
        sl=entry - stop_pips * 0.0001,
        tp=entry + stop_pips * 0.0002,
    )


class TestRoundTrip:
    def test_both_sides_are_charged(self) -> None:
        """A stop-out pays to get in and to get out."""
        assert settings_with(2.75).risk.commission_per_lot("forex") == pytest.approx(5.50)

    def test_zero_is_respected_for_spread_only_accounts(self) -> None:
        assert settings_with(0.0).risk.commission_per_lot("forex") == 0.0

    def test_an_asset_class_override_wins(self) -> None:
        tuned = settings_with(2.75, commission_by_asset_class={"index": 0.0})
        assert tuned.risk.commission_per_lot("index") == 0.0
        assert tuned.risk.commission_per_lot("forex") == pytest.approx(5.50)


class TestSizing:
    def test_commission_shrinks_the_position(self, spec: InstrumentSpec) -> None:
        """The same budget buys fewer lots once the true cost is counted."""
        free = size(settings_with(0.0), spec, stop_pips=20.0)
        charged = size(settings_with(2.75), spec, stop_pips=20.0)

        assert free.approved and charged.approved
        assert charged.volume < free.volume

    def test_the_recorded_risk_is_what_a_stop_out_really_costs(self, spec: InstrumentSpec) -> None:
        """`actual_risk_money` becomes `risk_money` in the journal, and every R
        the account ever reports is divided by it."""
        result = size(settings_with(2.75), spec, stop_pips=20.0)
        price_risk = spec.money_per_lot(20.0 * 0.0001) * result.volume
        commission = 5.50 * result.volume

        assert result.actual_risk_money == pytest.approx(price_risk + commission)
        assert result.actual_risk_money > price_risk

    def test_a_full_stop_out_now_lands_at_minus_one_r(self, spec: InstrumentSpec) -> None:
        """The whole point, stated as the arithmetic that was failing.

        Losing the price distance plus the round-trip commission is exactly
        `actual_risk_money`, so pnl_r comes out at -1.00 rather than -1.26.
        """
        result = size(settings_with(2.75), spec, stop_pips=20.0)
        realised_loss = spec.money_per_lot(20.0 * 0.0001) * result.volume + 5.50 * result.volume
        assert -realised_loss / result.actual_risk_money == pytest.approx(-1.0)

    def test_without_commission_the_same_loss_reads_worse_than_minus_one(
        self, spec: InstrumentSpec
    ) -> None:
        """The bug, pinned. Same trade, same costs, a lie in the denominator."""
        result = size(settings_with(0.0), spec, stop_pips=20.0)
        realised_loss = spec.money_per_lot(20.0 * 0.0001) * result.volume + 5.50 * result.volume
        assert -realised_loss / result.actual_risk_money < -1.0


class TestTightStopsArePricedHonestly:
    def test_the_cost_share_falls_away_as_the_stop_widens(self, spec: InstrumentSpec) -> None:
        """Measured on EURUSD at EUR 2.75 a side:

            2 pips  -> 21.6% of the risk is commission
            5 pips  ->  9.9%
           20 pips  ->  2.7%
           40 pips  ->  1.4%

        A fifth of the risk on a scalp stop and a rounding error on a swing
        stop. That is the second effect of counting it, and the one that
        changes behaviour: narrow stops get less attractive by arithmetic
        rather than by a threshold someone picked.
        """
        share = {
            pips: 5.50 / (spec.money_per_lot(pips * 0.0001) + 5.50) for pips in (2.0, 20.0, 40.0)
        }
        assert share[2.0] == pytest.approx(0.216, abs=0.01)
        assert share[20.0] == pytest.approx(0.027, abs=0.005)
        assert share[40.0] < 0.02
        assert share[2.0] > 8 * share[20.0]

    def test_a_tight_stop_on_a_small_account_is_sized_down_hardest(
        self, spec: InstrumentSpec
    ) -> None:
        """No new gate, and no dramatic refusal either — just honest sizing.

        Counting commission does not make a 2-pip stop illegal; it makes the
        position it buys smaller, by the same 21.6% the cost represents. On a
        small account that is what eventually pushes such setups under the
        broker minimum, through the undercapitalised rule that already exists
        rather than through anything added here.
        """
        free = size(settings_with(0.0), spec, stop_pips=2.0, equity=85.0)
        charged = size(settings_with(2.75), spec, stop_pips=2.0, equity=85.0)

        assert free.approved and charged.approved
        assert charged.raw_volume == pytest.approx(free.raw_volume * (1 - 0.216), rel=0.02)

    def test_a_wide_stop_is_barely_affected(self, spec: InstrumentSpec) -> None:
        free = size(settings_with(0.0), spec, stop_pips=40.0)
        charged = size(settings_with(2.75), spec, stop_pips=40.0)

        assert charged.approved
        assert charged.volume >= free.volume * 0.95
