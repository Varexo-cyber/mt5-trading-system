"""An unknown ATR must mean "do not act", never "act with an offset of zero".

Three rules in `execution/manager.py` build a stop out of `atr * multiple`,
and all three used to read a missing measurement as a legitimate zero:

    break_even = price_open + atr * break_even_offset_atr * direction
    trailing   = price      - atr * trailing_atr_multiple * direction

Zero is not a neutral offset in either line. The trail collapses to
`trailing = price` -- a stop at the current price, which is an immediate
stop-out, and `_worth_moving` waves it through because moving the stop from
-1R to 0R is a large improvement. Break-even collapses to the entry price,
which is not break-even at all: a long is closed on the bid, so a stop sitting
exactly on the entry is a guaranteed loss of the spread. Covering that cost is
the entire job of `break_even_offset_atr`.

And the ATR could go missing far more easily than it looks. `_compute_atr`
built a DataFrame straight from `copy_rates` and indexed `frame["close"]`, so
an empty return -- one symbol, one unlucky moment -- was a KeyError raised from
inside the per-second guard loop. `guard_tick` caught it as a single warning
line and discarded the whole pass: no break-even, no profit lock, no peak
stall, no health read, for EVERY open position, once a second, for as long as
that one symbol's H1 history stayed unavailable. `_travel_time` guards its own
ATR against exactly this and says so in a comment; this one did not.

The last test here is the one that would have caught the ATR trail's other
half. It never had the stop-room floor that `_health_tighten` was given after
the EURAUD ratchet, and the two rules differ only in how often their offset
happens to be wide enough to hide the exposure.
"""

from __future__ import annotations

from dataclasses import replace

from core.types import Direction
from tests.test_position_guard import (
    ENTRY,
    NOW,
    STOP,
    BrokerStub,
    JournalStub,
    at,
    manager_for,
    position,
)


def test_no_history_is_not_an_atr_of_zero() -> None:
    """The empty-rates case, which used to raise a KeyError."""
    broker = BrokerStub()
    broker.copy_rates = lambda *_args, **_kwargs: []  # type: ignore[assignment]
    manager = manager_for(broker, JournalStub())

    assert manager._compute_atr("EURUSD") == 0.0


def test_a_symbol_with_no_history_does_not_take_the_whole_pass_down() -> None:
    """The reason the line above matters. This call used to raise, and every
    open position went unmanaged for that second as a result."""
    broker = BrokerStub()
    broker.copy_rates = lambda *_args, **_kwargs: []  # type: ignore[assignment]
    manager = manager_for(broker, JournalStub())

    manager.manage([position()], NOW)  # must not raise


def test_an_unknown_atr_yields_no_offset_rather_than_a_zero_one() -> None:
    manager = manager_for(BrokerStub(atr=0.0), JournalStub())

    assert manager._atr_offset("EURUSD", 2.0, Direction.LONG) is None


def test_a_known_atr_still_yields_one_signed_by_direction() -> None:
    manager = manager_for(BrokerStub(atr=1.0), JournalStub())

    assert manager._atr_offset("EURUSD", 2.0, Direction.LONG) == 2.0
    assert manager._atr_offset("EURUSD", 2.0, Direction.SHORT) == -2.0


#: Reaching the ATR trail through the whole chain is a fight, and the fight is
#: itself worth recording. Break-even re-fires and `continue`s on every pass
#: unless the stop is already above entry; the partial close `continue`s unless
#: it has been taken; the profit lock arms at 0.7R and the trail cannot run
#: until 1.5R, so on a price that only rises the lock always has a better level
#: to offer. All three are switched off or satisfied here so that what is being
#: measured is the trail and not the rule that pre-empted it.
def _trail_fixture(atr: float, spread: float):  # type: ignore[no-untyped-def]
    broker = BrokerStub(atr=atr, spread=spread)
    manager = manager_for(broker, JournalStub(partial_taken=True), profit_lock_from_r=99.0)
    held = replace(position(), sl=ENTRY + 0.5)
    at(broker, 2.0)
    return broker, manager, held


def test_the_trail_does_not_run_on_an_atr_that_does_not_exist() -> None:
    """`trailing = price - 0` is the current price. `_worth_moving` approves it
    -- moving a stop from above entry to the market is a large improvement --
    and the position is stopped out on the next tick that goes against it."""
    broker, manager, held = _trail_fixture(atr=0.0, spread=0.02)

    events = manager.manage([held], NOW)

    assert [event.action for event in events] == []
    assert broker.modified == []


def test_the_trail_still_runs_on_an_atr_that_does() -> None:
    """The other half, so the test above cannot be satisfied by a trail that
    never fires at all."""
    broker, manager, held = _trail_fixture(atr=0.5, spread=0.02)

    events = manager.manage([held], NOW)

    assert [event.action for event in events] == ["ATR_TRAIL"]
    assert broker.modified == [broker.price - 2.0 * 0.5]


def test_the_trail_is_pushed_back_to_the_room_a_stop_needs() -> None:
    """A wide spread against a modest ATR: two ATRs of trail is less room than
    a stop needs to be a stop, and the naive level would sit inside its own
    cost of exit."""
    broker, manager, held = _trail_fixture(atr=0.5, spread=0.6)
    config = manager.settings.trade_management
    floor = max(
        broker.spread * config.min_stop_room_spreads, (ENTRY - STOP) * config.min_stop_room_r
    )

    manager.manage([held], NOW)

    naive = broker.price - 2.0 * 0.5
    assert floor > 2.0 * 0.5, "fixture no longer exercises the floor"
    assert broker.modified == [broker.price - floor]
    assert broker.modified[0] < naive


def test_break_even_does_not_park_the_stop_on_the_entry_price() -> None:
    """With no ATR the offset is zero and break-even becomes the entry itself.
    A long is closed on the bid, so that stop is not break-even -- it is a
    guaranteed loss of the spread, which is the exact cost the offset exists to
    clear."""
    broker = BrokerStub(atr=0.0, spread=0.02)
    manager = manager_for(broker, JournalStub())
    held = position()
    at(broker, 1.0)

    events = manager.manage([held], NOW)

    assert "BREAK_EVEN" not in [event.action for event in events]
    assert ENTRY not in broker.modified


def test_the_trail_keeps_the_same_room_the_tighten_rule_keeps() -> None:
    """One definition of "a stop must not sit inside its own costs", asked of
    the rule that never had it.

    The floor is the larger of `min_stop_room_spreads` spreads and
    `min_stop_room_r` of the trade's own risk -- two units because neither is a
    floor on its own, which is what the EURAUD ratchet demonstrated.
    """
    broker = BrokerStub(spread=0.02)
    manager = manager_for(broker, JournalStub())
    config = manager.settings.trade_management
    risk = ENTRY - STOP
    broker.price = ENTRY + 2.0 * risk
    tick = broker.tick("EURUSD")
    expected_floor = max(
        broker.spread * config.min_stop_room_spreads, risk * config.min_stop_room_r
    )

    # A candidate right on top of the trigger price, which is what a vanishing
    # ATR produces.
    roomed = manager._with_stop_room(position(), tick.bid, risk, tick)

    assert roomed == tick.bid - expected_floor
    # And a candidate already further away is left exactly where it is.
    far = tick.bid - expected_floor * 5
    assert manager._with_stop_room(position(), far, risk, tick) == far


def test_the_room_floor_reads_the_short_side_of_the_book() -> None:
    """A short is closed at the ask. Measuring its room from the bid would
    leave it one spread short of the room it was promised."""
    broker = BrokerStub(spread=0.02)
    manager = manager_for(broker, JournalStub())
    config = manager.settings.trade_management
    risk = ENTRY - STOP
    tick = broker.tick("EURUSD")
    short = replace(position(), direction=Direction.SHORT)
    expected_floor = max(
        broker.spread * config.min_stop_room_spreads, risk * config.min_stop_room_r
    )

    roomed = manager._with_stop_room(short, tick.ask, risk, tick)

    assert roomed == tick.ask + expected_floor
