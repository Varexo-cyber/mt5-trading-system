"""Owner-opened MT5 positions become first-class managed trades."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import MT5Config
from core.clock import SimulatedClock
from core.data_manager import DataManager
from core.mt5_codes import PositionType
from core.mt5_connector import MT5Connector
from journal.database import Journal
from journal.recorder import Recorder
from runner.service import JarvisRunner, OperationMode
from tests.fakes.fake_mt5 import FakeMT5, eurusd_spec

NOW = datetime(2026, 8, 10, 10, 30, tzinfo=UTC)


class _Brain:
    def record_decision(self, **_kwargs):  # type: ignore[no-untyped-def]
        return 1

    def record_trade_opened(self, **_kwargs):  # type: ignore[no-untyped-def]
        return 2


def _manual(*, symbol: str = "EURUSD", sl: float = 0.0, tp: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        ticket=9001,
        symbol=symbol,
        type=int(PositionType.BUY),
        volume=0.01,
        price_open=1.08500,
        sl=sl,
        tp=tp,
        profit=0.0,
        swap=0.0,
        time=int(NOW.timestamp()),
        magic=0,
        comment="manual",
    )


def _runner(tmp_path: Path, fake: FakeMT5) -> JarvisRunner:
    base = load_settings(DEFAULT_CONFIG_PATH, env_overrides=False)
    manual = base.trade_management.manual_positions.model_copy(update={"enabled": True})
    management = base.trade_management.model_copy(update={"manual_positions": manual})
    settings = base.model_copy(
        update={
            "trade_management": management,
            "journal": base.journal.model_copy(
                update={"database_path": str(tmp_path / "journal.db")}
            ),
        },
        deep=True,
    )
    connector = MT5Connector(MT5Config(), mt5_module=fake)
    connector.connect()
    service = object.__new__(JarvisRunner)
    service.settings = settings
    service.operation = OperationMode.EXPERIMENTAL_LIVE
    service.clock = SimulatedClock(NOW)
    service.broker = connector
    service.data = DataManager(connector, settings.data, service.clock)
    service.journal = Journal(tmp_path / "journal.db", service.clock).open()
    service.recorder = Recorder(service.journal, service.clock, settings)
    service.brain = _Brain()
    service._brain_trades = {}
    service.alerts = SimpleNamespace(send=lambda _message: None)
    return service


def test_a_manual_trade_without_protection_gets_sl_tp_and_a_durable_plan(
    tmp_path: Path,
) -> None:
    fake = FakeMT5(
        now=NOW,
        specs={"EURUSD": eurusd_spec()},
        quotes={"EURUSD": (1.08500, 1.08512)},
        positions=[_manual()],
    )
    runner = _runner(tmp_path, fake)

    events = runner._adopt_manual_positions(runner.broker.account())

    assert [event.action for event in events] == ["MANUAL_ADOPTED"]
    row = runner.journal.open_trade_by_ticket(9001)
    assert row is not None
    assert int(row["magic"]) == 0
    assert float(row["sl"]) > 0
    assert float(row["tp"]) > float(row["entry_price"])
    assert any(request.get("action") == 6 for request in fake.orders_sent)
    runner.journal.close()


def test_an_already_protected_manual_trade_is_adopted_without_rewriting_stops(
    tmp_path: Path,
) -> None:
    fake = FakeMT5(
        now=NOW,
        specs={"EURUSD": eurusd_spec()},
        quotes={"EURUSD": (1.08500, 1.08512)},
        positions=[_manual(sl=1.08000, tp=1.09500)],
    )
    runner = _runner(tmp_path, fake)

    events = runner._adopt_manual_positions(runner.broker.account())

    assert [event.action for event in events] == ["MANUAL_ADOPTED"]
    assert fake.orders_sent == []
    runner.journal.close()


def test_ignored_gold_is_neither_adopted_nor_returned_for_management(tmp_path: Path) -> None:
    fake = FakeMT5(
        now=NOW,
        specs={"XAUUSD": eurusd_spec(name="XAUUSD")},
        quotes={"XAUUSD": (2400.0, 2400.2)},
        positions=[_manual(symbol="XAUUSD")],
    )
    runner = _runner(tmp_path, fake)
    instruments = runner.settings.instruments.model_copy(update={"ignored_symbols": ("XAUUSD",)})
    runner.settings = runner.settings.model_copy(update={"instruments": instruments})

    assert runner._adopt_manual_positions(runner.broker.account()) == []
    assert runner._managed_positions() == []
    assert runner._flatten_owned_positions("test hard stop") == ()
    assert fake.orders_sent == []
    assert runner.journal.open_trade_by_ticket(9001) is None
    runner.journal.close()
