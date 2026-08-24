"""A locked stop that sits inside the spread is a fee, not protection.

24 AUGUST, 25 CLOSED TRADES. Fifteen won and the book was still negative in R.
Every loss came from HEALTH_EXIT; every win came from a stop at the broker.
Eleven of those wins were BROKER_SL, +1.22R between them — 0.11R each, which is
the profit lock's arming minimum to the cent. Eleven trades reached the lock and
not one of them survived it. Exactly one trade in 25 ever reached a take-profit.

`profit_lock_fraction` is a share of the PEAK, so the room it leaves shrinks
with the peak. At 0.6 and an arming peak of 0.20R the stop lands 0.08R under
the high, and on an index with a nine-point stop that is less than a point —
inside the spread. The next tick takes it whatever the market is doing.

So the gap gets a floor in the instrument's own cost. These tests hold the
strategy still and check only that: the lock keeps securing a share of the
peak, and never places that share where a tick can reach it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from config.schema import TradeManagementConfig
from core.types import Direction, Position
from execution.manager import PositionManager

ENTRY = 1.0800
RISK = 0.0010  # ten pips


def lock(
    *,
    peak_r: float,
    spread: float,
    minimum_spreads: float = 2.0,
    fraction: float = 0.5,
    from_r: float = 0.7,
    current_sl: float = 1.0800,
):
    """One profit-lock evaluation. Returns (event, requested stop)."""
    sent: dict[str, float] = {}

    def modify_stops(position, sl, tp):  # type: ignore[no-untyped-def]
        sent["sl"] = sl
        return SimpleNamespace(ok=True)

    manager = PositionManager.__new__(PositionManager)
    manager.settings = SimpleNamespace(
        trade_management=TradeManagementConfig(
            profit_lock_from_r=from_r,
            profit_lock_fraction=fraction,
            profit_lock_minimum_spreads=minimum_spreads,
        )
    )
    manager.equity = 0.0
    manager.broker = SimpleNamespace(
        modify_stops=modify_stops,
        spec=lambda symbol: SimpleNamespace(normalize_price=lambda price: round(price, 6)),
    )
    position = Position(
        ticket=1,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=0.01,
        price_open=ENTRY,
        sl=current_sl,
        tp=ENTRY + 0.0040,
        profit=1.0,
        swap=0.0,
        opened_at=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
        magic=1,
    )
    tick = SimpleNamespace(bid=ENTRY, ask=ENTRY + spread, spread=spread)
    event = manager._profit_lock(position, peak_r, peak_r, RISK, tick=tick)
    return event, sent.get("sl")


def secured_r(stop: float) -> float:
    return (stop - ENTRY) / RISK


class TestTheGapIsFlooredInTheSpread:
    def test_a_wide_spread_pushes_the_stop_further_from_the_peak(self) -> None:
        """Two pips of spread on a ten-pip risk needs 0.40R of room. The
        fraction wanted 0.35R of it, so the lock takes 0.30R instead — less
        secured, and what is secured can actually be held."""
        event, stop = lock(peak_r=0.7, spread=0.0002)

        assert event is not None
        assert secured_r(stop) == pytest.approx(0.30, abs=1e-6)

    def test_a_narrow_spread_leaves_the_rule_exactly_as_it_was(self) -> None:
        """Half a pip needs 0.10R, the fraction already leaves 0.35R, so
        nothing about the old behaviour changes. The guard must bite only
        where it is the binding constraint."""
        event, stop = lock(peak_r=0.7, spread=0.00005)

        assert event is not None
        assert secured_r(stop) == pytest.approx(0.35, abs=1e-6)

    def test_it_refuses_rather_than_locking_somewhere_unholdable(self) -> None:
        """Four pips of spread needs 0.80R of room and the peak is 0.7R. There
        is no stop above entry that survives a tick, so the trade keeps the
        one it has. The rule runs again every second and arms the moment the
        peak has the room."""
        event, stop = lock(peak_r=0.7, spread=0.0004)

        assert event is None
        assert stop is None

    def test_the_index_case_from_24_august(self) -> None:
        """The shape that produced eleven identical wins. Arming at 0.20R with
        a 0.6 fraction leaves 0.08R; a spread worth 0.10R swallows it whole.
        Before this the lock armed anyway and the tick collected."""
        event, stop = lock(peak_r=0.20, spread=0.0001, fraction=0.6, from_r=0.20)

        assert event is None
        assert stop is None

    def test_the_same_trade_arms_once_it_has_actually_run(self) -> None:
        """Not a refusal to protect winners — a refusal to protect them at a
        price that cannot hold. At 0.6R the same spread leaves room."""
        event, stop = lock(peak_r=0.6, spread=0.0001, fraction=0.6, from_r=0.20)

        assert event is not None
        assert secured_r(stop) == pytest.approx(0.36, abs=1e-6)


class TestItStillOnlyEverImprovesTheStop:
    def test_it_will_not_move_a_stop_backwards(self) -> None:
        """The floor lowers what gets secured, and a lower stop than the one
        already at the broker must still be refused. Securing less than last
        time is not a reason to give room back."""
        event, stop = lock(peak_r=0.7, spread=0.0002, current_sl=ENTRY + 0.00034)

        assert event is None
        assert stop is None


class TestTheGuardCanBeSwitchedOff:
    def test_zero_spreads_restores_the_old_placement(self) -> None:
        event, stop = lock(peak_r=0.7, spread=0.0004, minimum_spreads=0.0)

        assert event is not None
        assert secured_r(stop) == pytest.approx(0.35, abs=1e-6)

    def test_no_tick_leaves_the_rule_alone(self) -> None:
        """Callers without a quote — replays, older paths — must get the
        behaviour they had rather than a silently different one."""
        manager = PositionManager.__new__(PositionManager)
        manager.settings = SimpleNamespace(trade_management=TradeManagementConfig())
        manager.equity = 0.0
        sent: dict[str, float] = {}
        manager.broker = SimpleNamespace(
            modify_stops=lambda position, sl, tp: (
                sent.__setitem__("sl", sl),
                SimpleNamespace(ok=True),
            )[1],
            spec=lambda symbol: SimpleNamespace(normalize_price=lambda price: round(price, 6)),
        )
        position = Position(
            ticket=1,
            symbol="EURUSD",
            direction=Direction.LONG,
            volume=0.01,
            price_open=ENTRY,
            sl=ENTRY,
            tp=ENTRY + 0.0040,
            profit=1.0,
            swap=0.0,
            opened_at=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
            magic=1,
        )

        event = manager._profit_lock(position, 0.7, 0.7, RISK)

        assert event is not None
        assert secured_r(sent["sl"]) == pytest.approx(0.35, abs=1e-6)
