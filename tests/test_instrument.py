"""Instrument maths. A bug here is a wrong position size on every trade."""

from __future__ import annotations

import math

import pytest

from core.instrument import AssetClass, InstrumentSpec
from core.mt5_codes import OrderFilling
from tests.fakes.fake_mt5 import eurusd_spec, usdjpy_spec, xauusd_spec


@pytest.fixture
def eurusd() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


@pytest.fixture
def usdjpy() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(usdjpy_spec())


@pytest.fixture
def xauusd() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(xauusd_spec())


class TestPipMaths:
    def test_five_digit_fx_pip_is_ten_points(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.pip_size == pytest.approx(0.0001)
        assert eurusd.points_per_pip == pytest.approx(10.0)

    def test_three_digit_jpy_pip_is_ten_points(self, usdjpy: InstrumentSpec) -> None:
        assert usdjpy.pip_size == pytest.approx(0.01)

    def test_four_digit_fx_pip_is_one_point(self) -> None:
        spec = InstrumentSpec.from_mt5(
            eurusd_spec(digits=4, point=0.0001, trade_tick_size=0.0001, trade_tick_value=10.0)
        )
        assert spec.pip_size == pytest.approx(0.0001)

    def test_gold_is_not_treated_as_forex(self, xauusd: InstrumentSpec) -> None:
        # Two digits and a non-currency base: pip must be one point, not ten.
        assert xauusd.is_forex is False
        assert xauusd.pip_size == pytest.approx(0.01)

    def test_mt5_catalogue_path_classifies_asset_family(self, xauusd: InstrumentSpec) -> None:
        crypto = InstrumentSpec.from_mt5(
            xauusd_spec(
                name="BTCUSD",
                path="Cryptos\\High Cap\\BTCUSD",
                currency_base="USD",
                trade_contract_size=1.0,
            )
        )
        stock = InstrumentSpec.from_mt5(
            xauusd_spec(name="AAPL", path="Stock\\NAS\\AAPL", currency_base="USD")
        )

        assert crypto.asset_class is AssetClass.CRYPTO
        assert stock.asset_class is AssetClass.STOCK
        assert xauusd.asset_class is AssetClass.METAL

    def test_pip_value_per_lot_eurusd(self, eurusd: InstrumentSpec) -> None:
        # tick_value 1.0 per 0.00001 -> 10.0 per 0.0001 on 1.00 lot
        assert eurusd.pip_value_per_lot() == pytest.approx(10.0)

    def test_pip_value_per_lot_usdjpy(self, usdjpy: InstrumentSpec) -> None:
        # 0.0067 per 0.001 -> 0.067 per 0.01... times 10 points per pip
        assert usdjpy.pip_value_per_lot() == pytest.approx(0.067)

    def test_roundtrip_pips_to_price(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.price_to_pips(eurusd.pips_to_price(17.5)) == pytest.approx(17.5)

    def test_price_to_pips_preserves_sign(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.price_to_pips(-0.0025) == pytest.approx(-25.0)

    def test_money_per_lot_uses_absolute_distance(self, eurusd: InstrumentSpec) -> None:
        # 20 pips at USD 10/pip on 1.00 lot, sign-independent.
        assert eurusd.money_per_lot(-0.0020) == pytest.approx(200.0)
        assert eurusd.money_per_lot(0.0020) == pytest.approx(200.0)


class TestVolumeNormalisation:
    def test_rounds_down_never_up(self, eurusd: InstrumentSpec) -> None:
        # 0.0199 lots must become 0.01, not 0.02. Rounding up would silently
        # double the risk the sizer computed on a minimum-lot account.
        assert eurusd.round_volume_down(0.0199) == 0.01

    def test_exact_multiples_survive(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.round_volume_down(0.07) == 0.07
        assert eurusd.round_volume_down(1.23) == 1.23

    def test_no_float_dust(self, eurusd: InstrumentSpec) -> None:
        result = eurusd.round_volume_down(0.3)
        assert result == 0.3
        assert repr(result) == "0.3"

    def test_below_minimum_rounds_to_zero(self, eurusd: InstrumentSpec) -> None:
        # The sizer must see 0.0 and skip, rather than receive a usable lot.
        assert eurusd.round_volume_down(0.004) == 0.0

    def test_negative_volume_is_zero(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.round_volume_down(-1.0) == 0.0

    def test_tradability_checks_step_and_bounds(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.is_volume_tradable(0.01)
        assert not eurusd.is_volume_tradable(0.005)  # below min
        assert not eurusd.is_volume_tradable(0.015)  # off-step
        assert not eurusd.is_volume_tradable(200.0)  # above max

    def test_finer_lot_step_keeps_precision(self) -> None:
        spec = InstrumentSpec.from_mt5(eurusd_spec(volume_min=0.001, volume_step=0.001))
        assert spec.round_volume_down(0.0047) == 0.004


class TestRiskFeasibility:
    """The arithmetic that decides whether a small account can trade at all."""

    def test_min_risk_at_minimum_lot(self, eurusd: InstrumentSpec) -> None:
        # 20-pip stop, 0.01 lot -> 20 * 0.10 = USD 2.00
        assert eurusd.min_risk_money(eurusd.pips_to_price(20)) == pytest.approx(2.0)

    def test_hundred_euro_account_cannot_afford_a_normal_stop(self, eurusd: InstrumentSpec) -> None:
        # This is the headline number from the startup guard: 1% of 100 buys
        # roughly 10 pips of stop on EURUSD at the minimum lot.
        assert eurusd.max_sl_pips_for_risk(risk_money=1.0) == pytest.approx(10.0)
        # A realistic 30-pip structural stop costs 3% of a 100 account.
        assert eurusd.min_risk_pct(eurusd.pips_to_price(30), equity=100.0) == pytest.approx(3.0)

    def test_gold_is_unaffordable_on_a_small_account(self, xauusd: InstrumentSpec) -> None:
        # A modest USD 8 stop on gold at the minimum lot risks 8% of EUR 100.
        assert xauusd.min_risk_pct(price_distance := 8.0, equity=100.0) == pytest.approx(8.0)
        assert price_distance == 8.0

    def test_min_risk_pct_is_infinite_without_equity(self, eurusd: InstrumentSpec) -> None:
        assert math.isinf(eurusd.min_risk_pct(0.0020, equity=0.0))


class TestBrokerConstraints:
    def test_stop_level_zero_never_blocks(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.stops_level == 0
        assert not eurusd.violates_stop_level(1.08500, 1.08499)

    def test_stop_inside_stop_level_is_rejected(self, usdjpy: InstrumentSpec) -> None:
        # 20 points = 0.020 on a 3-digit JPY pair.
        assert usdjpy.violates_stop_level(150.100, 150.090)
        assert not usdjpy.violates_stop_level(150.100, 150.050)

    def test_filling_mode_prefers_ioc(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.preferred_filling() is OrderFilling.IOC

    def test_filling_mode_falls_back_to_fok(self, usdjpy: InstrumentSpec) -> None:
        assert usdjpy.preferred_filling() is OrderFilling.FOK

    def test_filling_mode_defaults_to_return(self) -> None:
        spec = InstrumentSpec.from_mt5(eurusd_spec(filling_mode=0))
        assert spec.preferred_filling() is OrderFilling.RETURN

    def test_normalize_price_snaps_to_digits(self, eurusd: InstrumentSpec) -> None:
        assert eurusd.normalize_price(1.0850123456) == pytest.approx(1.08501)


class TestConstructionGuards:
    """Bad broker data must fail loudly, not divide by zero later."""

    def test_zero_tick_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="trade_tick_value"):
            InstrumentSpec.from_mt5(eurusd_spec(trade_tick_value=0.0))

    def test_zero_volume_step_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="volume_step"):
            InstrumentSpec.from_mt5(eurusd_spec(volume_step=0.0))

    def test_zero_point_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="point"):
            InstrumentSpec.from_mt5(eurusd_spec(point=0.0, trade_tick_size=0.0))

    def test_missing_tick_size_falls_back_to_point(self) -> None:
        spec = InstrumentSpec.from_mt5(eurusd_spec(trade_tick_size=0.0))
        assert spec.tick_size == spec.point
