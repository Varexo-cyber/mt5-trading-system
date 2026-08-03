"""Operator control over open positions, and the risk ceiling it may not breach."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_connector import MT5Connector
from core.types import Direction, Position
from dashboard.position_control import PositionControl
from tests.fakes.fake_mt5 import FakeMT5

EQUITY = 100.0


@pytest.fixture
def fake() -> FakeMT5:
    return FakeMT5()


@pytest.fixture
def control(fake: FakeMT5) -> PositionControl:
    connector = MT5Connector(MT5Config(), mt5_module=fake)
    connector.connect()
    return PositionControl(connector, load_settings(env_overrides=False))


def long_position(sl: float = 1.09000, tp: float = 1.11000, volume: float = 0.01) -> Position:
    return Position(
        ticket=1,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=volume,
        price_open=1.10000,
        sl=sl,
        tp=tp,
        profit=0.0,
        swap=0.0,
        opened_at=datetime.now(UTC),
    )


def short_position(sl: float = 1.11000, tp: float = 1.09000) -> Position:
    return Position(
        ticket=2,
        symbol="EURUSD",
        direction=Direction.SHORT,
        volume=0.01,
        price_open=1.10000,
        sl=sl,
        tp=tp,
        profit=0.0,
        swap=0.0,
        opened_at=datetime.now(UTC),
    )


# ------------------------------------------------------------- stop checks ---


def test_removing_the_stop_is_refused(control: PositionControl) -> None:
    preview = control.preview_stop(long_position(), 0.0, EQUITY)
    assert not preview.valid
    assert "must keep a stop" in preview.detail


def test_a_long_stop_above_entry_is_refused(control: PositionControl) -> None:
    preview = control.preview_stop(long_position(), 1.10500, EQUITY)
    assert not preview.valid
    assert "below entry" in preview.detail


def test_a_short_stop_below_entry_is_refused(control: PositionControl) -> None:
    preview = control.preview_stop(short_position(), 1.09500, EQUITY)
    assert not preview.valid
    assert "above entry" in preview.detail


def test_tightening_is_always_permitted(control: PositionControl) -> None:
    preview = control.preview_stop(long_position(sl=1.09000), 1.09500, EQUITY)
    assert preview.valid and preview.permitted
    assert "Tightens" in preview.detail


def test_the_preview_prices_the_stop_in_money(control: PositionControl) -> None:
    """The operator should see the euros while typing, not after submitting."""
    preview = control.preview_stop(long_position(), 1.09000, EQUITY)
    assert preview.risk_money > 0
    assert preview.risk_pct == pytest.approx(preview.risk_money / EQUITY * 100.0)


def test_widening_past_the_ceiling_is_refused(control: PositionControl) -> None:
    """The dashboard must not become a way to hold a trade the sizer would not open."""
    preview = control.preview_stop(long_position(sl=1.09900), 1.00000, EQUITY)
    assert preview.valid
    assert not preview.permitted
    assert "Refused" in preview.detail
    assert "ceiling" in preview.detail


def test_a_modest_widening_inside_the_ceiling_is_allowed(control: PositionControl) -> None:
    preview = control.preview_stop(long_position(sl=1.09990), 1.09900, EQUITY)
    assert preview.valid and preview.permitted
    assert "Widens" in preview.detail


def test_modify_refuses_what_the_preview_refuses(control: PositionControl) -> None:
    outcome = control.modify(long_position(sl=1.09900), sl=1.00000, tp=1.11000, equity=EQUITY)
    assert not outcome.ok
    assert "ceiling" in outcome.message


# ----------------------------------------------------------- target checks ---


def test_a_long_target_below_entry_is_refused(control: PositionControl) -> None:
    outcome = control.modify(long_position(), sl=1.09500, tp=1.09000, equity=EQUITY)
    assert not outcome.ok
    assert "above entry" in outcome.message


def test_a_zero_target_is_a_legitimate_choice(control: PositionControl) -> None:
    """MT5's "no target" means managed to an exit, not an invalid order."""
    outcome = control.modify(long_position(), sl=1.09500, tp=0.0, equity=EQUITY)
    assert outcome.ok


def test_a_valid_modification_reaches_the_broker(control: PositionControl) -> None:
    outcome = control.modify(long_position(), sl=1.09500, tp=1.12000, equity=EQUITY)
    assert outcome.ok
    assert "#1 updated" in outcome.message


# ------------------------------------------------------------------ closing ---


def test_closing_a_position_reaches_the_broker(control: PositionControl) -> None:
    outcome = control.close(long_position())
    assert outcome.ok
    assert "Closed all of #1" in outcome.message


def test_a_valid_partial_reaches_the_broker(control: PositionControl) -> None:
    outcome = control.close(long_position(volume=0.02), 0.01)
    assert outcome.ok
    assert "Closed 0.01 lots of #1" in outcome.message


def test_a_partial_below_the_minimum_lot_is_refused(control: PositionControl) -> None:
    outcome = control.close(long_position(volume=0.01), 0.005)
    assert not outcome.ok
    assert "rounds below" in outcome.message


def test_a_partial_leaving_an_unclosable_stub_is_refused(control: PositionControl) -> None:
    """The remainder must not be rounded away — 0.005 lots is a real leftover."""
    outcome = control.close(long_position(volume=0.015), 0.01)
    assert not outcome.ok
    assert "would leave 0.005 lots" in outcome.message


def test_close_all_flattens_every_position(control: PositionControl) -> None:
    results = control.close_all([long_position(), short_position()])
    assert len(results) == 2
    assert all(item.ok for item in results)
