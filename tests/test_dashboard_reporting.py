"""Read-only dashboard service and PDF report smoke tests."""

from __future__ import annotations

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_codes import TIMEFRAME_VALUES
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from dashboard.service import DashboardService, catalogue_asset_class
from reporting.pdf_report import build_pdf_report
from tests.fakes.fake_mt5 import FakeMT5


def test_every_mt5_timeframe_is_selectable() -> None:
    assert {timeframe.value for timeframe in Timeframe} == set(TIMEFRAME_VALUES)


def test_dashboard_reads_catalogue_bars_and_builds_pdf() -> None:
    connector = MT5Connector(MT5Config(), mt5_module=FakeMT5())
    service = DashboardService(connector, load_settings(env_overrides=False))
    try:
        account = service.connect()
        symbols = service.symbols()
        spec = service.spec("EURUSD")
        tick = service.tick("EURUSD")
        frame = service.bars("EURUSD", Timeframe.H1, count=50)

        assert any(item.name == "EURUSD" for item in symbols)
        assert catalogue_asset_class("Cryptos\\High Cap\\BTCUSD").value == "crypto"
        assert len(frame) == 50

        pdf = build_pdf_report(account, [], "EURUSD", spec, tick, {"H1": frame})
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 10_000
    finally:
        service.close()
