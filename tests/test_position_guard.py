"""The fast layer: watching open positions between cycles.

Management used to run once per full cycle. A cycle scans the catalogue and can
take most of a minute, so a trade could run to 1.6R and hand every bit of it
back between two consecutive looks — the rules were correct and simply were not
being asked often enough.

Three things make the one-second cadence work, and each is asserted here: the
ATR is cached so the pass stays cheap, the peak R is persisted so a restart
cannot forget the trade was ahead, and the guard never calls the adviser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import pytest

from config.loader import load_settings
from core.types import Direction, Position, Tick, Timeframe
from execution.manager import ATR_CACHE_SECONDS, ManagementEvent, PositionManager

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
ENTRY = 100.0
STOP = 98.0  # risk of 2.0 price units, so 1R = 2.0


# ----------------------------------------------------------------- doubles ---


@dataclass
class OrderResult:
    ok: bool = True
    filled_price: float | None = 100.0
    filled_volume: float | None = 0.01


@dataclass
class BrokerStub:
    """Just enough broker to drive `manage`, and it counts its own calls.

    The call counts are the point of half these tests: the guard's whole
    licence to run every second is that a pass is cheap.
    """

    price: float = ENTRY
    atr: float = 1.0
    spread: float = 0.0
    rate_calls: dict[int, int] = field(default_factory=dict)
    closed: list[tuple[int, float | None]] = field(default_factory=list)
    modified: list[float] = field(default_factory=list)

    @property
    def atr_calls(self) -> int:
        """Only the H1 fetches. The health readers pull M1 and M5 on their own
        much shorter TTL, so a single total would conflate two caches."""
        return self.rate_calls.get(Timeframe.H1.mt5_value, 0)

    def tick(self, symbol: str) -> Tick:
        return Tick(symbol=symbol, bid=self.price, ask=self.price, time=NOW)

    def spec(self, _symbol: str):  # type: ignore[no-untyped-def]
        broker = self

        class Spec:
            volume_min = 0.01

            @staticmethod
            def normalize_price(price: float) -> float:
                return round(price, 5)

            @staticmethod
            def round_volume_down(volume: float) -> float:
                return round(volume, 2)

        del broker
        return Spec()

    def copy_rates(self, _symbol, timeframe, count):  # type: ignore[no-untyped-def]
        self.rate_calls[timeframe] = self.rate_calls.get(timeframe, 0) + 1
        # A flat series whose true range is exactly `atr` on every bar, and
        # which is deliberately featureless: no swings, no drift, so the health
        # readers stay silent and these tests keep measuring what they name.
        base = int(datetime(2026, 8, 4, 0, 0, tzinfo=UTC).timestamp())
        return [
            {
                "time": base + index * 60,
                "high": 100.0 + self.atr,
                "low": 100.0,
                "close": 100.0,
                "open": 100.0,
                "tick_volume": 1,
                "spread": 0,
                "real_volume": 0,
            }
            for index in range(count)
        ]

    def close_position(self, position: Position, volume: float | None = None) -> OrderResult:
        self.closed.append((position.ticket, volume))
        return OrderResult(filled_price=self.price, filled_volume=volume or position.volume)

    def modify_stops(self, _position, sl, tp) -> OrderResult:  # type: ignore[no-untyped-def]
        del tp
        self.modified.append(sl)
        return OrderResult(filled_price=sl)


@dataclass
class JournalStub:
    """Holds the excursion ratchet the way the real journal does."""

    peak_r: float = 0.0
    trough_r: float = 0.0
    partial_taken: bool = False

    def open_trade_by_ticket(self, ticket: int):  # type: ignore[no-untyped-def]
        return {"id": 1, "ticket": ticket, "sl": STOP, "volume": 0.02, "mfe_r": self.peak_r}

    def management_action_exists(self, _ticket, _actions) -> bool:  # type: ignore[no-untyped-def]
        return self.partial_taken

    def update_excursions(self, _trade_id, *, mae_r: float, mfe_r: float) -> None:
        self.peak_r = max(self.peak_r, mfe_r)
        self.trough_r = min(self.trough_r, mae_r)


def position(volume: float = 0.02, opened_at: datetime = NOW) -> Position:
    return Position(
        ticket=555,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=volume,
        price_open=ENTRY,
        sl=STOP,
        tp=ENTRY + 10.0,
        profit=1.0,
        swap=0.0,
        opened_at=opened_at,
    )


def manager_for(broker: BrokerStub, journal: JournalStub, **overrides) -> PositionManager:  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    if overrides:
        management = settings.trade_management.model_copy(update=overrides)
        settings = settings.model_copy(update={"trade_management": management})
    return PositionManager(broker, journal, settings)  # type: ignore[arg-type]


def at(broker: BrokerStub, r: float) -> None:
    """Move the market to a given R for the fixture's position."""
    broker.price = ENTRY + r * (ENTRY - STOP)


# ------------------------------------------------------------- excursions ---


def test_the_peak_is_recorded_as_the_trade_runs() -> None:
    """`mfe_r` and `mae_r` existed as columns and nothing ever wrote them, so
    every postmortem reported both as unknown."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 1.2)
    manager.manage([position()], NOW)
    at(broker, 0.9)
    manager.manage([position()], NOW)

    assert journal.peak_r == pytest.approx(1.2)


def test_the_trough_is_recorded_too() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, -0.4)
    manager.manage([position()], NOW)

    assert journal.trough_r == pytest.approx(-0.4)


def test_the_peak_only_ratchets_upward() -> None:
    """A late retrace must not erase the fact that the trade was once ahead —
    that memory is the entire input to the give-back rule."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 2.0)
    manager.manage([position()], NOW)
    at(broker, 0.1)
    manager.manage([position()], NOW)

    assert journal.peak_r == pytest.approx(2.0)


# ---------------------------------------------------------------- giveback ---


def test_a_trade_that_hands_back_half_its_peak_is_closed() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=0.5)

    at(broker, 1.6)
    manager.manage([position()], NOW)
    at(broker, 0.7)  # floor is 0.8
    events = manager.manage([position()], NOW)

    assert [event.action for event in events] == ["GIVEBACK_EXIT"]
    assert "1.60R" in events[0].detail and "0.70R" in events[0].detail
    assert broker.closed == [(555, None)], "the whole position, not a partial"


def test_a_trade_still_above_the_floor_is_left_alone() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=0.5)

    at(broker, 1.6)
    manager.manage([position()], NOW)
    at(broker, 0.9)  # floor is 0.8
    events = manager.manage([position()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)
    assert broker.closed == []


def test_the_rule_does_not_arm_below_the_threshold() -> None:
    """Around entry a half-give-back is noise, and acting on it would close
    every trade that breathed."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=0.5)

    at(broker, 0.8)
    manager.manage([position()], NOW)
    at(broker, 0.1)
    events = manager.manage([position()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


def test_a_fresh_high_is_never_a_giveback() -> None:
    """The first observation of a new peak has r_now == peak; a rule that
    compared with `>=` against the floor would close the trade at its best
    price for a fraction of 1.0."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=1.0)

    at(broker, 2.5)
    events = manager.manage([position()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


def test_the_peak_survives_a_restart() -> None:
    """A new manager reads the peak from the journal, so a crash mid-trade
    cannot hand the position a clean slate and let it give back everything."""
    broker, journal = BrokerStub(), JournalStub()
    at(broker, 1.8)
    manager_for(broker, journal).manage([position()], NOW)

    reborn = manager_for(BrokerStub(price=ENTRY + 0.4 * (ENTRY - STOP)), journal)
    events = reborn.manage([position()], NOW)

    assert journal.peak_r == pytest.approx(1.8)
    assert [event.action for event in events] == ["GIVEBACK_EXIT"]


def test_zero_disables_the_rule() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 2.0)
    manager.manage([position()], NOW)
    at(broker, 0.0)
    events = manager.manage([position()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


def test_a_refused_close_is_not_reported_as_an_exit() -> None:
    """Recording a close the broker rejected would leave the journal believing
    the position is flat while the money is still at risk."""
    broker, journal = BrokerStub(), JournalStub()

    def refuse(_position, volume=None):  # type: ignore[no-untyped-def]
        return OrderResult(ok=False, filled_price=None)

    broker.close_position = refuse  # type: ignore[assignment]
    manager = manager_for(broker, journal)

    at(broker, 1.6)
    manager.manage([position()], NOW)
    at(broker, 0.2)
    events = manager.manage([position()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


# --------------------------------------------------------------- the cost ---


def test_the_atr_is_not_refetched_on_every_pass() -> None:
    """Sixty round-trips an hour to watch an average of fourteen hourly bars
    move in the fourth decimal is what makes a one-second cadence unaffordable."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    for _ in range(20):
        manager.manage([position()], NOW)

    assert broker.atr_calls == 1


def test_the_cache_expires() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    manager.manage([position()], NOW)
    # Reach into the deadline rather than sleeping a minute in a unit test.
    manager._atr_cache = {
        key: (deadline - ATR_CACHE_SECONDS - 1.0, value)
        for key, (deadline, value) in manager._atr_cache.items()
    }
    manager.manage([position()], NOW)

    assert broker.atr_calls == 2


def test_the_cache_is_keyed_per_symbol() -> None:
    """One cache entry for everything would price a gold position off an
    EURUSD ATR — a stop dozens of times too tight or too wide."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    manager.manage([position()], NOW)
    manager.manage([replace(position(), symbol="XAUUSD")], NOW)

    assert broker.atr_calls == 2


def test_the_cache_uses_a_monotonic_deadline() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    manager.manage([position()], NOW)
    ((deadline, _),) = manager._atr_cache.values()

    assert time.monotonic() < deadline <= time.monotonic() + ATR_CACHE_SECONDS


# ------------------------------------------------------------- ordering ---


def test_the_giveback_pre_empts_the_partial_and_the_trail() -> None:
    """Half-closing a position we have already decided to leave would pay two
    spreads to exit and carry the rest through the reversal."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=0.4)

    at(broker, 2.0)
    manager.manage([position()], NOW)  # break-even moves the stop here
    before = len(broker.modified)
    at(broker, 1.1)  # above partial_close_at_r, but 45% given back
    events = manager.manage([position()], NOW)

    assert [event.action for event in events] == ["GIVEBACK_EXIT"]
    assert broker.closed == [(555, None)], "closed whole, not halved"
    assert len(broker.modified) == before, "no stop moves on the way out"


def test_a_stalled_trade_still_times_out() -> None:
    """The give-back rule must not shadow the time exit: a trade that never
    ran has no peak, so the earlier `continue` cannot swallow it."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, time_exit_hours=4.0)

    at(broker, 0.05)
    events = manager.manage([position(opened_at=NOW - timedelta(hours=9))], NOW)

    assert [event.action for event in events] == ["TIME_EXIT"]


def test_management_events_are_still_typed() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)
    at(broker, 1.6)
    manager.manage([position()], NOW)
    at(broker, 0.1)
    (event,) = manager.manage([position()], NOW)
    assert isinstance(event, ManagementEvent)
    assert event.r_at_action == pytest.approx(0.1)
