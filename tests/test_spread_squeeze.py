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
from datetime import timedelta

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


def _tick(spread_pips: float, at=NOW):  # type: ignore[no-untyped-def]
    return Tick(symbol="NZDJPY", time=at, bid=NZD_BID, ask=NZD_BID + spread_pips * 0.01)


def squeeze(  # type: ignore[no-untyped-def]
    spread_pips: float,
    *,
    r_now: float = -0.05,
    direction=Direction.SHORT,
    calm_pips: float = 0.5,
    manager=None,
    **overrides,
):
    """Run the rule at a given spread and return the event, if any.

    `r_now` defaults just below break-even, which is the situation the rule
    describes: the live NZDJPY sell had bid 92.845 against a stop at 92.904 and
    was leaning on it. The squeeze no longer reaches into profit — crossing a
    blown-out spread there pays a certain cost to avoid a possible one — so a
    fixture in profit would be testing a case the rule declines by design.

    Two things now have to be supplied that the rule did not used to need, and
    both are the point of it. It compares against what THIS position has been
    living with, so `calm_pips` seeds a quiet baseline first; and it will not
    act on a single reading, so the wide quote is presented twice with the
    persistence window elapsed between. A helper that skipped either would be
    testing a rule the account does not run.
    """
    position = yen_position(direction)
    manager = manager or manager_for(BrokerStub(), JournalStub(), **overrides)
    config = manager.settings.trade_management
    for _ in range(config.spread_squeeze_min_samples):
        manager._spread_squeeze_exit(position, _tick(calm_pips), r_now, NOW)
    manager._spread_squeeze_exit(position, _tick(spread_pips), r_now, NOW)
    later = NOW + timedelta(seconds=config.spread_squeeze_persist_seconds + 1)
    return manager._spread_squeeze_exit(position, _tick(spread_pips, later), r_now, later)


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
    # Ask is 4 pips from the bid and within 2 pips of the stop.
    assert squeeze(4.0) is not None


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

    def euro(spread: float, at=NOW):  # type: ignore[no-untyped-def]
        return Tick(symbol="EURUSD", time=at, bid=1.09975, ask=1.09975 + spread)

    config = manager.settings.trade_management
    for _ in range(config.spread_squeeze_min_samples):
        manager._spread_squeeze_exit(long_position, euro(0.00005), -0.05, NOW)
    # Bid 1.09975, stop 1.0994: 3.5 points of room against a 4-point spread.
    manager._spread_squeeze_exit(long_position, euro(0.00040), -0.05, NOW)
    later = NOW + timedelta(seconds=config.spread_squeeze_persist_seconds + 1)
    assert manager._spread_squeeze_exit(long_position, euro(0.00040), -0.05, later) is not None


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
    assert manager._spread_squeeze_exit(naked, tick, 0.05, NOW) is None


def test_a_refused_close_is_not_reported_as_an_exit() -> None:
    broker = BrokerStub()
    broker.close_position = lambda _p, volume=None: type(  # type: ignore[assignment]
        "R", (), {"ok": False, "filled_price": None, "filled_volume": None}
    )()
    manager = manager_for(broker, JournalStub())
    assert squeeze(6.0, r_now=0.05, manager=manager) is None


def test_the_reason_carries_the_numbers() -> None:
    """The journal line is read back weeks later, and "spread" alone does not
    say whether it was 2 pips or 20."""
    event = squeeze(4.0)
    assert event is not None
    assert "%" in event.detail and "stop" in event.detail
    assert event.r_at_action == pytest.approx(-0.05)


def test_the_configured_share_is_what_decides() -> None:
    assert squeeze(2.0, spread_squeeze_share=0.3) is not None
    assert squeeze(2.0, spread_squeeze_share=2.0) is None


del ENTRY, STOP  # imported only to keep the shared fixtures importable


class TestItDoesNotCashOutAWinner:
    """The rule was paying the very cost it exists to avoid.

    Closing at market during a blown-out quote crosses that spread with
    certainty. Being stopped by it costs the same spread only IF price actually
    travels to the stop. On a losing trade already leaning on its stop that swap
    is worth making — a probable cost for a certain, smaller one, which is the
    NZDJPY case the rule was written for. On a winner with room to spare it is
    the reverse: a certain cost paid to avoid a possible one, on a position that
    is working.

    The account's own replay says exactly that. SPREAD_SQUEEZE closed eight of
    twenty-two trades on 17 August and every one scored a NEGATIVE lift against
    its own untouched stop and target: +0.38R banked where leaving it alone
    returned +1.02R. The blowout passed, as blowouts do, and the trade carried
    on without us.
    """

    def test_a_position_in_profit_is_left_to_its_stop(self) -> None:
        assert squeeze(3.0, r_now=0.40) is None

    def test_a_position_leaning_on_its_stop_is_still_rescued(self) -> None:
        """The half that earns its keep, and the reason this is a ceiling rather
        than a removal."""
        event = squeeze(3.0, r_now=-0.30)

        assert event is not None
        assert event.action == "SPREAD_SQUEEZE"

    def test_break_even_is_the_boundary(self) -> None:
        assert squeeze(3.0, r_now=0.0) is not None
        assert squeeze(3.0, r_now=0.01) is None

    def test_the_old_reach_into_profit_can_be_restored(self) -> None:
        """One config away, so the change is reversible on evidence rather than
        on argument."""
        assert squeeze(3.0, r_now=0.40, spread_squeeze_max_r=5.0) is not None


class TestItWaitsToSeeWhetherTheBlowoutIsReal:
    """The three tests the rule did not have, and why it has them.

    It was measured negative eleven times out of eleven — eight in the 17
    August replay, every one worse than leaving the position alone, and three
    live afterwards. What was wrong was not what it looked at but how fast it
    believed it: every one of those blowouts passed and the trade carried on
    without us, so acting on the first reading turned a temporary quote into a
    certain loss.
    """

    @staticmethod
    def _live():  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH,
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml",
            env_overrides=False,
        ).trade_management

    def test_one_reading_of_a_wide_quote_is_not_an_exit(self) -> None:
        """The whole difference. Waiting costs nothing — the stop is there
        throughout, and a quote that really will take it is still wide in half
        a minute."""
        manager = manager_for(BrokerStub(), JournalStub())
        position = yen_position()
        config = manager.settings.trade_management
        for _ in range(config.spread_squeeze_min_samples):
            manager._spread_squeeze_exit(position, _tick(0.5), -0.05, NOW)

        assert manager._spread_squeeze_exit(position, _tick(6.0), -0.05, NOW) is None

    def test_a_blowout_that_passes_leaves_the_trade_alone(self) -> None:
        """One wide reading, then calm again, then wide once more later. The
        clock has run but the condition has not held, so nothing fires."""
        manager = manager_for(BrokerStub(), JournalStub())
        position = yen_position()
        config = manager.settings.trade_management
        for _ in range(config.spread_squeeze_min_samples):
            manager._spread_squeeze_exit(position, _tick(0.5), -0.05, NOW)
        later = NOW + timedelta(seconds=config.spread_squeeze_persist_seconds + 1)

        manager._spread_squeeze_exit(position, _tick(6.0), -0.05, NOW)
        manager._spread_squeeze_exit(position, _tick(0.5), -0.05, NOW)  # it passed

        assert manager._spread_squeeze_exit(position, _tick(6.0), -0.05, later) is None

    def test_a_blowout_that_holds_is_still_an_exit(self) -> None:
        assert squeeze(6.0) is not None

    def test_a_market_that_is_simply_always_this_wide_is_not_a_blowout(self) -> None:
        """The second complaint, and it is a different one. A wide quote argues
        for leaving; a near stop does not. FRA40 tripped this every tick once
        `HEALTH_TIGHTEN` had pulled the stop in three times — the spread never
        changed, the room did."""
        assert squeeze(6.0, calm_pips=6.0) is None

    def test_it_says_nothing_until_it_has_seen_enough(self) -> None:
        """Below the sample floor nothing can be called abnormal, and holding
        is the safe answer: the stop is still there and it is what the size was
        set against."""
        manager = manager_for(BrokerStub(), JournalStub())
        position = yen_position()
        later = NOW + timedelta(seconds=120)

        assert manager._spread_squeeze_exit(position, _tick(6.0), -0.05, NOW) is None
        assert manager._spread_squeeze_exit(position, _tick(6.0), -0.05, later) is None

    def test_the_live_overlay_runs_it_with_all_three(self) -> None:
        config = self._live()

        assert config.spread_squeeze_share > 0.0
        assert config.spread_squeeze_persist_seconds > 0.0
        assert config.spread_squeeze_abnormal_multiple > 1.0
        assert config.spread_squeeze_min_samples >= 2

    def test_the_reason_records_what_convinced_it(self) -> None:
        """A rule with this record has to say why it acted, not just that it
        did — the next review of it starts from these numbers."""
        event = squeeze(6.0)

        assert event is not None
        assert "usual" in event.detail
        assert "held for" in event.detail
        assert "readings" in event.detail
