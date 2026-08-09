"""Position sizing. Every assertion here is a euro that does not get lost."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import Settings
from core.instrument import InstrumentSpec
from core.types import Direction
from risk.position_sizer import PositionSizer
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec, usdjpy_spec, xauusd_spec


@pytest.fixture
def raw() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def settings_for(tmp_path: Path, raw: dict[str, Any], **overrides: Any) -> Settings:
    """Load the shipped config with targeted overrides."""
    data = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node: Any = data
        *path, leaf = dotted.split(".")
        for part in path:
            node = node[part]
        node[leaf] = value
    path_ = tmp_path / "config.yaml"
    path_.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_settings(path_, env_overrides=False)


@pytest.fixture
def settings(tmp_path: Path, raw: dict[str, Any]) -> Settings:
    return settings_for(tmp_path, raw, **{"system.mode": "scaling"})


@pytest.fixture
def sizer(settings: Settings) -> PositionSizer:
    return PositionSizer(settings)


@pytest.fixture
def eurusd() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


class TestHappyPath:
    def test_basic_sizing(self, sizer: PositionSizer, eurusd: InstrumentSpec) -> None:
        # 10 000 equity, 1% = 100. A 20-pip stop at 10/pip/lot -> 0.50 lots.
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
        )
        assert result.approved
        assert result.volume == pytest.approx(0.50)
        assert result.actual_risk_money == pytest.approx(100.0)
        assert result.actual_risk_pct == pytest.approx(1.0)
        assert result.sl_distance_pips == pytest.approx(20.0)
        assert result.reward_risk == pytest.approx(3.0)

    def test_short_side_is_symmetric(self, sizer: PositionSizer, eurusd: InstrumentSpec) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.SHORT,
            entry=1.08500,
            sl=1.08700,
            tp=1.07900,
        )
        assert result.approved
        assert result.volume == pytest.approx(0.50)
        assert result.reward_risk == pytest.approx(3.0)

    def test_jpy_pair_uses_its_own_pip(self, sizer: PositionSizer) -> None:
        spec = InstrumentSpec.from_mt5(usdjpy_spec())
        result = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=150.100,
            sl=149.900,
            tp=150.700,
        )
        assert result.approved
        assert result.sl_distance_pips == pytest.approx(20.0)
        # 0.067 per pip per lot -> 100 / (20 * 0.067) = 74.6 lots, floored.
        assert result.volume == pytest.approx(74.62, abs=0.01)

    def test_rounding_is_always_down(self, sizer: PositionSizer, eurusd: InstrumentSpec) -> None:
        """The rounded position must never risk more than intended."""
        result = sizer.size(
            spec=eurusd,
            equity=1_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08270,
            tp=1.09190,  # 23-pip stop
        )
        assert result.approved
        # EUR 10 risk / (23 pips x EUR 10 per pip per lot) = 0.0434 lots.
        # Rounds to 0.04, never up to 0.05.
        assert result.volume == pytest.approx(0.04)
        assert result.actual_risk_money == pytest.approx(9.20)
        assert result.actual_risk_money <= result.intended_risk_money
        assert result.actual_risk_pct <= 1.0


class TestUndercapitalized:
    """The EUR 100 reality check — the whole reason this module is careful."""

    def test_hundred_euro_account_cannot_take_a_thirty_pip_stop(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=100.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08200,
            tp=1.09400,  # 30-pip stop
        )
        assert not result.approved
        assert result.reason is Reason.UNDERCAPITALIZED
        assert result.volume == 0.0
        # The refusal must state what the account *can* do.
        assert "10.0 pips" in result.decision.detail
        assert "3.00%" in result.decision.detail

    def test_never_rounds_up_to_the_minimum_lot(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        """The single most expensive shortcut in retail trading, refused."""
        result = sizer.size(
            spec=eurusd,
            equity=100.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08200,
            tp=1.09400,
        )
        assert result.volume == 0.0
        assert result.volume != eurusd.volume_min
        # 0.01 lots would have risked 3% instead of the intended 1%.
        assert eurusd.min_risk_pct(0.0030, 100.0) == pytest.approx(3.0)

    def test_a_tight_enough_stop_does_fit(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        # 1% of 100 = EUR 1; at 0.10/pip that is 10 pips of stop.
        result = sizer.size(
            spec=eurusd,
            equity=100.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08400,
            tp=1.08800,  # 10-pip stop, 1:3
        )
        assert result.approved
        assert result.volume == pytest.approx(0.01)
        assert result.actual_risk_money == pytest.approx(1.0)

    def test_gold_is_unaffordable_on_a_small_account(self, sizer: PositionSizer) -> None:
        """Gold skips the pip ceiling and is caught by the money check instead."""
        spec = InstrumentSpec.from_mt5(xauusd_spec())
        result = sizer.size(
            spec=spec,
            equity=100.0,
            direction=Direction.LONG,
            entry=2400.00,
            sl=2392.00,
            tp=2424.00,  # USD 8 stop
        )
        assert not result.approved
        # 800 "pips" is way past the 60-pip mode ceiling, but that ceiling is
        # an FX rule and does not apply here. The binding constraint is money:
        # 0.01 lots stopped at USD 8 risks 8% of a EUR 100 account.
        assert result.reason is Reason.UNDERCAPITALIZED
        assert spec.min_risk_pct(8.0, 100.0) == pytest.approx(8.0)

    def test_a_wide_gold_stop_is_fine_on_a_big_account(self, sizer: PositionSizer) -> None:
        spec = InstrumentSpec.from_mt5(xauusd_spec())
        result = sizer.size(
            spec=spec,
            equity=50_000.0,
            direction=Direction.LONG,
            entry=2400.00,
            sl=2392.00,
            tp=2424.00,
        )
        assert result.approved
        assert result.actual_risk_pct <= 1.0

    def test_finer_lot_step_rescues_a_small_account(self, sizer: PositionSizer) -> None:
        """A broker offering 0.001 lots changes the answer, which is the point."""
        spec = InstrumentSpec.from_mt5(eurusd_spec(volume_min=0.001, volume_step=0.001))
        result = sizer.size(
            spec=spec,
            equity=100.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08200,
            tp=1.09400,  # the same 30-pip stop
        )
        assert result.approved
        assert result.volume == pytest.approx(0.003)
        assert result.actual_risk_pct <= 1.0


class TestStopValidation:
    def test_missing_stop_is_refused(self, sizer: PositionSizer, eurusd: InstrumentSpec) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=0.0,
            tp=1.09100,
        )
        assert result.reason is Reason.INVALID_STOP
        assert "forbidden" in result.decision.detail

    def test_stop_at_entry_is_refused(self, sizer: PositionSizer, eurusd: InstrumentSpec) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08500,
            tp=1.09100,
        )
        assert result.reason is Reason.INVALID_STOP

    def test_long_stop_above_entry_is_refused(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08900,
            tp=1.09100,
        )
        assert result.reason is Reason.INVALID_STOP

    def test_short_stop_below_entry_is_refused(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.SHORT,
            entry=1.08500,
            sl=1.08100,
            tp=1.07900,
        )
        assert result.reason is Reason.INVALID_STOP

    def test_stop_inside_the_broker_stop_level_is_refused(self, sizer: PositionSizer) -> None:
        spec = InstrumentSpec.from_mt5(usdjpy_spec(trade_stops_level=200))  # 20 pips
        result = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=150.100,
            sl=150.000,
            tp=150.400,  # only 10 pips
        )
        assert result.reason is Reason.SL_TOO_TIGHT_FOR_BROKER
        assert "stop level" in result.decision.detail

    def test_untradable_symbol_is_refused(self, sizer: PositionSizer) -> None:
        spec = InstrumentSpec.from_mt5(eurusd_spec(trade_mode=3))  # close-only
        result = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
        )
        assert result.reason is Reason.SYMBOL_NOT_TRADABLE


class TestModeCeilings:
    def test_stop_wider_than_the_mode_allows(
        self, tmp_path: Path, raw: dict[str, Any], eurusd: InstrumentSpec
    ) -> None:
        settings = settings_for(tmp_path, raw, **{"system.mode": "micro_live"})
        sizer = PositionSizer(settings)
        result = sizer.size(
            spec=eurusd,
            equity=200.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08100,
            tp=1.09700,  # 40 pips, ceiling is 30
        )
        assert result.reason is Reason.SL_TOO_WIDE_FOR_ACCOUNT
        assert "40.0 pips" in result.decision.detail

    def test_reward_risk_below_minimum_is_refused(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.08700,  # 1:1
        )
        assert result.reason is Reason.RR_BELOW_MINIMUM
        assert "1:1.00" in result.decision.detail

    def test_one_broker_point_of_rr_rounding_does_not_reject_exact_boundary(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.08899,  # one 0.00001 point below exact 2R
        )

        assert result.approved

    def test_more_than_one_broker_point_below_rr_boundary_is_refused(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.08898,
        )

        assert result.reason is Reason.RR_BELOW_MINIMUM

    def test_target_on_the_wrong_side_is_refused(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.08100,  # target below the stop
        )
        assert result.reason is Reason.RR_BELOW_MINIMUM
        assert result.reward_risk == 0.0

    def test_no_target_skips_the_rr_gate(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        """Exits driven purely by structure or time are legitimate."""
        result = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=0.0,
        )
        assert result.approved

    def test_volume_is_capped_at_the_broker_maximum(self, sizer: PositionSizer) -> None:
        spec = InstrumentSpec.from_mt5(eurusd_spec(volume_max=1.0))
        result = sizer.size(
            spec=spec,
            equity=1_000_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
        )
        assert result.approved
        assert result.volume == pytest.approx(1.0)
        # Capping can only reduce risk, so it must never breach the ceiling.
        assert result.actual_risk_pct < 1.0
        assert "maximum" in result.decision.detail


class TestAntiMartingale:
    def test_multiplier_scales_risk_down(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        full = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
        )
        halved = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
            risk_multiplier=0.5,
        )
        assert halved.volume == pytest.approx(full.volume / 2)
        assert halved.actual_risk_money == pytest.approx(50.0)

    def test_multiplier_above_one_is_rejected_outright(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        """Scaling risk UP is martingale. It raises, it does not warn."""
        with pytest.raises(ValueError, match="forbidden"):
            sizer.size(
                spec=eurusd,
                equity=10_000.0,
                direction=Direction.LONG,
                entry=1.08500,
                sl=1.08300,
                tp=1.09100,
                risk_multiplier=2.0,
            )


class TestJournalRow:
    def test_row_is_flat_and_complete(self, sizer: PositionSizer, eurusd: InstrumentSpec) -> None:
        row = sizer.size(
            spec=eurusd,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
        ).journal_row()
        assert row["reason"] == "OK"
        assert row["direction"] == "LONG"
        assert row["volume"] == pytest.approx(0.50)
        assert all(not isinstance(value, dict | list) for value in row.values())

    def test_rejections_are_journalled_with_their_numbers(
        self, sizer: PositionSizer, eurusd: InstrumentSpec
    ) -> None:
        row = sizer.size(
            spec=eurusd,
            equity=100.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08200,
            tp=1.09400,
        ).journal_row()
        assert row["approved"] is False
        assert row["reason"] == "TRADE_SKIPPED_UNDERCAPITALIZED"
        assert row["sl_distance_pips"] == pytest.approx(30.0)
        assert row["intended_risk_money"] == pytest.approx(1.0)
