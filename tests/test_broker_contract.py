"""The MT5 adapter satisfies the broker-neutral domain contract."""

from __future__ import annotations

from config.schema import MT5Config
from core.broker import Broker, MarketDataProvider
from core.mt5_connector import MT5Connector
from tests.fakes.fake_mt5 import FakeMT5


def test_mt5_connector_is_a_broker() -> None:
    connector = MT5Connector(MT5Config(), mt5_module=FakeMT5())
    assert isinstance(connector, MarketDataProvider)
    assert isinstance(connector, Broker)
