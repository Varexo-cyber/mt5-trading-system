from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_connector import MT5Connector
from core.types import Direction, OrderRequest
from execution.manager import PositionManager
from execution.paper_broker import PaperBroker
from runner.service import JarvisRunner, OperationMode
from scanner.universe import UniverseScanner
from tests.fakes.fake_mt5 import FakeMT5, eurusd_spec


def connector(fake: FakeMT5) -> MT5Connector:
    return MT5Connector(MT5Config(), mt5_module=fake)


def test_paper_position_survives_restart(tmp_path: Path) -> None:
    fake = FakeMT5()
    market = connector(fake)
    state = tmp_path / "paper.json"
    paper = PaperBroker(market, state)
    paper.connect()
    spec = paper.spec("EURUSD")
    result = paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512), spec
    )
    paper.shutdown()

    restored = PaperBroker(connector(FakeMT5()), state)
    restored.connect()
    positions = restored.positions()

    assert result.ok
    assert len(positions) == 1
    assert positions[0].ticket == result.position_ticket
    restored.shutdown()


def test_paper_stop_closes_and_changes_balance(tmp_path: Path) -> None:
    fake = FakeMT5()
    paper = PaperBroker(connector(fake), tmp_path / "paper.json")
    paper.connect()
    spec = paper.spec("EURUSD")
    paper.order_send(OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512), spec)
    fake.quotes["EURUSD"] = (1.079, 1.07912)

    events = paper.mark_to_market()

    assert events[0][1] == "SL"
    assert not paper.positions()
    assert paper.account().balance < 100
    paper.shutdown()


def test_paper_closed_position_survives_crash_window(tmp_path: Path) -> None:
    fake = FakeMT5()
    state = tmp_path / "paper.json"
    paper = PaperBroker(connector(fake), state)
    paper.connect()
    spec = paper.spec("EURUSD")
    result = paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512), spec
    )
    fake.quotes["EURUSD"] = (1.079, 1.07912)
    paper.mark_to_market()
    paper.shutdown()

    restored = PaperBroker(connector(FakeMT5()), state)
    restored.connect()
    closed = restored.closed_position(int(result.position_ticket or 0))

    assert closed is not None
    assert closed.reason == "SL"
    assert closed.exit_price == 1.079
    assert closed.pnl_money < 0
    restored.shutdown()


def test_partial_close_is_persistent_and_recoverable(tmp_path: Path) -> None:
    class JournalStub:
        def open_trade_by_ticket(self, ticket: int):  # type: ignore[no-untyped-def]
            return {"ticket": ticket, "sl": 1.083, "volume": 0.10}

        def open_trades(self):  # type: ignore[no-untyped-def]
            return []

        def management_action_exists(self, _ticket, _actions):  # type: ignore[no-untyped-def]
            return False

    fake = FakeMT5(now=datetime.now(UTC))
    paper = PaperBroker(connector(fake), tmp_path / "paper.json")
    paper.connect()
    spec = paper.spec("EURUSD")
    paper.order_send(OrderRequest("EURUSD", Direction.LONG, 0.10, 1.083, 1.10, 1.08512), spec)
    fake.quotes["EURUSD"] = (1.09, 1.09012)
    settings = load_settings(env_overrides=False)
    manager = PositionManager(paper, JournalStub(), settings)  # type: ignore[arg-type]

    manager.manage(paper.positions(), datetime.now(UTC))  # break-even first
    events = manager.manage(paper.positions(), datetime.now(UTC))

    assert events[0].action == "PARTIAL_CLOSE"
    assert paper.positions()[0].volume == pytest.approx(0.05)
    recovered = manager.reconcile(paper.positions())
    assert any(event.action == "PARTIAL_CLOSE_RECOVERED" for event in recovered)
    remaining = paper.positions()[0]
    paper.close_position(remaining)
    closed = paper.closed_position(remaining.ticket)
    assert closed is not None
    assert closed.volume == pytest.approx(0.10)
    assert len(closed.deal_tickets) == 2
    paper.shutdown()


def test_news_break_even_does_not_close_an_already_protected_position(tmp_path: Path) -> None:
    class NewsStub:
        @staticmethod
        def position_action(*_args):  # type: ignore[no-untyped-def]
            return "break_even"

    fake = FakeMT5(now=datetime.now(UTC))
    paper = PaperBroker(connector(fake), tmp_path / "paper.json")
    paper.connect()
    paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.10, 1.083, 1.10, 1.08512),
        paper.spec("EURUSD"),
    )
    fake.quotes["EURUSD"] = (1.09, 1.09012)
    manager = PositionManager(
        paper,
        object(),
        load_settings(env_overrides=False),  # type: ignore[arg-type]
    )

    first = manager.manage_news(paper.positions(), NewsStub())  # type: ignore[arg-type]
    second = manager.manage_news(paper.positions(), NewsStub())  # type: ignore[arg-type]

    assert first[0].action == "NEWS_BREAK_EVEN"
    assert second == []
    assert len(paper.positions()) == 1
    paper.shutdown()


def test_scanner_scans_and_ranks_full_available_catalogue_by_default() -> None:
    fake = FakeMT5()
    market = connector(fake)
    market.connect()
    settings = load_settings()
    scanner = UniverseScanner(market, settings)

    batch = scanner.scan(cursor=0, keep=2)

    assert batch.inspected == 2
    assert batch.universe_size == 2
    assert batch.next_cursor == 0
    assert {item.symbol for item in batch.candidates} == {"EURUSD", "USDJPY"}
    assert len(batch.inspections) == 2
    assert {item.status for item in batch.inspections} == {"SHORTLISTED"}
    assert all(item.reason for item in batch.inspections)
    market.shutdown()


def test_default_scanner_does_not_stop_after_25_markets() -> None:
    symbols = [f"EURUSD{number:02d}" for number in range(30)]
    fake = FakeMT5(
        specs={
            symbol: eurusd_spec(name=symbol, path=f"Forex\\Test\\{symbol}") for symbol in symbols
        },
        quotes=dict.fromkeys(symbols, (1.08500, 1.08512)),
    )
    market = connector(fake)
    market.connect()
    scanner = UniverseScanner(market, load_settings(env_overrides=False))

    batch = scanner.scan(keep=5)

    assert batch.universe_size == 30
    assert batch.inspected == 30
    assert len(batch.inspections) == 30
    assert len(batch.candidates) == 5
    market.shutdown()


def test_scanner_explains_a_closed_or_stale_market() -> None:
    # Older than any plausible broker-server timezone offset, so the connector
    # correctly treats this as a closed-market quote rather than UTC normalization.
    fake = FakeMT5(now=datetime.now(UTC) - timedelta(days=2))
    market = connector(fake)
    market.connect()
    scanner = UniverseScanner(market, load_settings(env_overrides=False))

    batch = scanner.scan(cursor=0, batch_size=1, keep=1)

    assert batch.candidates == ()
    assert batch.rejected == 1
    assert batch.inspections[0].status == "REJECTED"
    assert batch.inspections[0].stage == "quote"
    assert "stale" in batch.inspections[0].reason.casefold()
    market.shutdown()


def test_demo_mode_hard_refuses_a_live_account(tmp_path: Path) -> None:
    fake = FakeMT5(is_demo=False, server="FakeBroker-Live")
    runner = JarvisRunner(
        connector(fake),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.DEMO,
    )

    with pytest.raises(RuntimeError, match="DEMO_ACCOUNT_REQUIRED"):
        runner.connect()

    assert not fake.orders_sent
    assert not runner.broker.is_connected
