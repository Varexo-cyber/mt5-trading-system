"""Leaving before the quote takes the stop.

A stop does not trigger on the price you watch. A short is closed at the ask, a
long at the bid, and both move toward the stop when the book thins — with the
market perfectly still.

The live case: NZDJPY sell, bid 92.845, stop 92.904. Five point nine pips of
room, and six pips of spread would have taken it out having never once gone
wrong. That is not the market disagreeing with the trade; it is the broker's
book charging for the hour.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.types import Direction, Position, Tick
from tests.test_position_guard import (
    ENTRY,
    NOW,
    STOP,
    BrokerStub,
    JournalStub,
    manager_for,
)

# The real numbers, in the instrument's own scale.
NZD_ENTRY, NZD_STOP, NZD_BID = 92.848, 92.904, 92.845


def yen_position(direction: Direction = Direction.SHORT) -> Position:
    return Position(
        ticket=901,
        symbol="NZDJPY",
        direction=direction,
        volume=0.05,
        price_open=NZD_ENTRY,
        sl=NZD_STOP,
        tp=92.656,
        profit=0.08,
        swap=0.0,
        opened_at=NOW,
    )


def squeeze(spread_pips: float, *, r_now: float = 0.05, direction=Direction.SHORT, **overrides):  # type: ignore[no-untyped-def]
    """Run the rule at a given spread and return the event, if any."""
    manager = manager_for(BrokerStub(), JournalStub(), **overrides)
    bid = NZD_BID
    tick = Tick(symbol="NZDJPY", time=NOW, bid=bid, ask=bid + spread_pips * 0.01)
    return manager._spread_squeeze_exit(yen_position(direction), tick, r_now)


def test_a_normal_spread_leaves_the_trade_alone() -> None:
    assert squeeze(1.0) is None


def test_the_quote_owning_the_room_is_an_exit() -> None:
    """At three pips the ask sits at 92.875 against a stop at 92.904 — under
    three pips of room left, all of it one more widening away."""
    event = squeeze(3.0)
    assert event is not None
    assert event.action == "SPREAD_SQUEEZE"
    assert "of the" in event.detail


def test_the_exit_happens_before_the_stop_would() -> None:
    """The whole point: leave at roughly break-even rather than at a full stop.

    Six pips is where the ask reaches the stop. The rule has to act earlier
    than that or it has changed nothing.
    """
    assert squeeze(6.0) is not None
    assert squeeze(4.0) is not None, "must fire before the ask reaches the stop"


def test_the_side_the_stop_triggers_on_is_the_one_read() -> None:
    """A short is stopped on the ask. Reading the bid instead hides the entire
    effect, because the bid is the side that is not moving."""
    wide = Tick(symbol="NZDJPY", time=NOW, bid=NZD_BID, ask=NZD_BID + 0.04)
    manager = manager_for(BrokerStub(), JournalStub())
    # Ask is 4 pips from the bid and within 2 pips of the stop.
    assert manager._spread_squeeze_exit(yen_position(), wide, 0.05) is not None


def test_a_long_is_measured_on_the_bid() -> None:
    """Mirrored: a long is stopped on the bid, which falls as the spread widens
    while the ask stands still."""
    manager = manager_for(BrokerStub(), JournalStub())
    long_position = Position(
        ticket=902,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=0.01,
        price_open=1.1000,
        sl=1.0994,
        tp=1.1020,
        profit=0.0,
        swap=0.0,
        opened_at=NOW,
    )
    # Bid 1.09975, stop 1.0994: 3.5 points of room against a 4-point spread.
    tick = Tick(symbol="EURUSD", time=NOW, bid=1.09975, ask=1.09975 + 0.00040)
    assert manager._spread_squeeze_exit(long_position, tick, 0.05) is not None


def test_a_trade_already_most_of_the_way_to_its_stop_is_left_to_the_stop() -> None:
    """Otherwise this degenerates into closing every loser a moment early. Past
    the floor the room is small because the trade is losing, not because the
    quote is wide."""
    assert squeeze(3.0, r_now=-0.8) is None


def test_the_floor_is_where_the_change_of_meaning_is() -> None:
    assert squeeze(3.0, r_now=-0.4) is not None
    assert squeeze(3.0, r_now=-0.6) is None


def test_zero_disables_it() -> None:
    assert squeeze(6.0, spread_squeeze_share=0.0) is None


def test_no_spread_is_not_a_squeeze() -> None:
    assert squeeze(0.0) is None


def test_a_position_without_a_stop_is_not_measured() -> None:
    """Reconciliation closes those outright; dividing by their absent room here
    would raise instead."""
    manager = manager_for(BrokerStub(), JournalStub())
    naked = replace(yen_position(), sl=0.0)
    tick = Tick(symbol="NZDJPY", time=NOW, bid=NZD_BID, ask=NZD_BID + 0.06)
    assert manager._spread_squeeze_exit(naked, tick, 0.05) is None


def test_a_refused_close_is_not_reported_as_an_exit() -> None:
    broker = BrokerStub()
    broker.close_position = lambda _p, volume=None: type(  # type: ignore[assignment]
        "R", (), {"ok": False, "filled_price": None, "filled_volume": None}
    )()
    manager = manager_for(broker, JournalStub())
    tick = Tick(symbol="NZDJPY", time=NOW, bid=NZD_BID, ask=NZD_BID + 0.06)
    assert manager._spread_squeeze_exit(yen_position(), tick, 0.05) is None


def test_the_reason_carries_the_numbers() -> None:
    """The journal line is read back weeks later, and "spread" alone does not
    say whether it was 2 pips or 20."""
    event = squeeze(4.0)
    assert event is not None
    assert "%" in event.detail and "stop" in event.detail
    assert event.r_at_action == pytest.approx(0.05)


def test_the_configured_share_is_what_decides() -> None:
    assert squeeze(2.0, spread_squeeze_share=0.3) is not None
    assert squeeze(2.0, spread_squeeze_share=2.0) is None


del ENTRY, STOP  # imported only to keep the shared fixtures importable
