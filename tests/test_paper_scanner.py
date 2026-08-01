from __future__ import annotations

from pathlib import Path

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_connector import MT5Connector
from core.types import Direction, OrderRequest
from execution.paper_broker import PaperBroker
from scanner.universe import UniverseScanner
from tests.fakes.fake_mt5 import FakeMT5


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


def test_scanner_rotates_and_ranks_available_catalogue() -> None:
    fake = FakeMT5()
    market = connector(fake)
    market.connect()
    settings = load_settings()
    scanner = UniverseScanner(market, settings)

    batch = scanner.scan(cursor=0, batch_size=2, keep=2)

    assert batch.inspected == 2
    assert batch.universe_size == 2
    assert batch.next_cursor == 0
    assert {item.symbol for item in batch.candidates} == {"EURUSD", "USDJPY"}
    market.shutdown()
