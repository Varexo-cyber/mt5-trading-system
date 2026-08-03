"""The crash window between sending an order and recording it.

Before the intent log, a crash in that window left a real position at the broker
that the journal had never heard of. On restart reconciliation could only read
it as an orphan and close it — a correctly sized, AI-approved trade destroyed by
a power cut, with the loss booked and no record of why the position had existed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from config.loader import load_settings
from config.schema import MT5Config
from core.clock import SimulatedClock
from core.mt5_connector import MT5Connector
from core.types import Direction, Position
from execution.manager import PositionManager
from journal.database import SCHEMA_VERSION, Journal
from journal.recorder import Recorder
from risk.position_sizer import PositionSizer
from tests.fakes.fake_mt5 import FakeMT5

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def journal(tmp_path, clock) -> Journal:  # type: ignore[no-untyped-def]
    return Journal(tmp_path / "trading.db", clock).open()


@pytest.fixture
def broker() -> MT5Connector:
    connector = MT5Connector(MT5Config(), mt5_module=FakeMT5())
    connector.connect()
    return connector


def sizing(broker: MT5Connector, volume: float = 0.01):  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    spec = broker.spec("EURUSD")
    tick = broker.tick("EURUSD")
    result = PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=tick.ask,
        sl=spec.normalize_price(tick.ask - 0.0020),
        tp=spec.normalize_price(tick.ask + 0.0040),
    )
    assert result.approved, result.decision.detail
    return result


def intent(journal: Journal, broker: MT5Connector, clock: SimulatedClock) -> int:
    recorder = Recorder(journal, clock, load_settings(env_overrides=False))
    return recorder.record_entry_intent(
        cycle_pk=None, sizing=sizing(broker), equity_before=10_000.0
    )


def position_from(journal: Journal, ticket: int, opened_at: datetime) -> Position:
    row = journal.pending_entries()[0]
    return Position(
        ticket=ticket,
        symbol=str(row["symbol"]),
        direction=Direction[str(row["direction"])],
        volume=float(row["volume"]),
        price_open=float(row["entry_price"]),
        sl=float(row["sl"]),
        tp=float(row["tp"]),
        profit=0.0,
        swap=0.0,
        opened_at=opened_at,
    )


# ------------------------------------------------------------------ schema ---


def test_the_migration_lands(journal: Journal) -> None:
    version = journal.scalar("SELECT MAX(version) FROM schema_version")
    assert version == SCHEMA_VERSION == 3


def test_existing_trades_default_to_open(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    """A row written the old way is a real trade, so OPEN is the truthful value."""
    recorder = Recorder(journal, clock, load_settings(env_overrides=False))
    recorder.record_trade_open(
        cycle_pk=None, sizing=sizing(broker), ticket=99, entry_price=1.1, equity_before=100.0
    )
    row = journal.open_trade_by_ticket(99)
    assert row is not None
    assert str(row["entry_state"]) == "OPEN"


# ------------------------------------------------------------------ intent ---


def test_an_intent_is_written_before_the_order(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    trade_id = intent(journal, broker, clock)
    pending = journal.pending_entries()
    assert len(pending) == 1
    assert int(pending[0]["id"]) == trade_id
    assert pending[0]["ticket"] is None


def test_a_pending_intent_is_not_reported_as_a_lost_position(
    journal: Journal, broker, clock
) -> None:  # type: ignore[no-untyped-def]
    """`open_trades` drives "the broker lost a position", which halts new risk.

    An intent whose order may never have been sent must not trigger that.
    """
    intent(journal, broker, clock)
    assert journal.open_trades() == []


def test_promotion_makes_it_a_live_trade(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    trade_id = intent(journal, broker, clock)
    journal.promote_pending_entry(trade_id, ticket=4242, entry_price=1.10055)
    assert journal.pending_entries() == []
    row = journal.open_trade_by_ticket(4242)
    assert row is not None
    assert float(row["entry_price"]) == pytest.approx(1.10055)
    assert len(journal.open_trades()) == 1


def test_abandonment_retires_the_intent_without_deleting_it(
    journal: Journal, broker, clock
) -> None:  # type: ignore[no-untyped-def]
    """A rejected entry is evidence; a run of them is a broker-side problem."""
    trade_id = intent(journal, broker, clock)
    journal.abandon_pending_entry(trade_id, "entry rejected: NO_MONEY")

    assert journal.pending_entries() == []
    assert journal.open_trades() == []
    row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
    assert str(row["entry_state"]) == "ABANDONED"
    assert "NO_MONEY" in str(row["exit_reason"])
    assert float(row["pnl_money"]) == 0.0


# ---------------------------------------------------------------- adoption ---


def test_a_crashed_entry_is_adopted_not_closed(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    """The whole point: the position survives, attached to its own plan."""
    trade_id = intent(journal, broker, clock)
    manager = PositionManager(broker, journal, load_settings(env_overrides=False))
    position = position_from(journal, ticket=7001, opened_at=NOW)

    events = manager.reconcile([position])

    assert [event.action for event in events] == ["ADOPTED"]
    assert f"#{trade_id}" in events[0].detail
    row = journal.open_trade_by_ticket(7001)
    assert row is not None
    assert int(row["id"]) == trade_id
    assert str(row["entry_state"]) == "OPEN"


def test_an_unrelated_position_is_still_closed(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    """A hand-opened position has no sized plan; adopting it would manage a
    trade nobody sized."""
    intent(journal, broker, clock)
    manager = PositionManager(broker, journal, load_settings(env_overrides=False))
    stranger = position_from(journal, ticket=7002, opened_at=NOW)
    # Same symbol and direction, ten times the size. Not our order.
    stranger = replace(stranger, volume=0.10)

    events = manager.reconcile([stranger])

    assert [event.action for event in events] == ["ORPHAN_CLOSE"]
    assert journal.pending_entries(), "the intent must survive an unrelated orphan"


def test_a_stale_intent_cannot_adopt(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    """A week-old intent must not claim a position opened by hand today."""
    intent(journal, broker, clock)
    manager = PositionManager(broker, journal, load_settings(env_overrides=False))
    much_later = position_from(journal, ticket=7003, opened_at=NOW + timedelta(days=7))

    events = manager.reconcile([much_later])

    assert [event.action for event in events] == ["ORPHAN_CLOSE"]


def test_the_opposite_direction_is_not_adopted(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    intent(journal, broker, clock)
    manager = PositionManager(broker, journal, load_settings(env_overrides=False))
    base = position_from(journal, ticket=7004, opened_at=NOW)
    flipped = replace(base, direction=Direction.SHORT)

    events = manager.reconcile([flipped])

    assert [event.action for event in events] == ["ORPHAN_CLOSE"]


def test_only_one_position_can_claim_an_intent(journal: Journal, broker, clock) -> None:  # type: ignore[no-untyped-def]
    """Two identical positions must not both adopt the same plan."""
    intent(journal, broker, clock)
    manager = PositionManager(broker, journal, load_settings(env_overrides=False))
    first = position_from(journal, ticket=7005, opened_at=NOW)
    second = replace(first, ticket=7006)

    events = manager.reconcile([first, second])

    assert [event.action for event in events] == ["ADOPTED", "ORPHAN_CLOSE"]
