"""Read-only dashboard service and PDF report smoke tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_codes import TIMEFRAME_VALUES
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from dashboard.service import (
    DashboardService,
    catalogue_asset_class,
    load_market_intelligence,
    load_paper_snapshot,
    stop_confirmation_matches,
)
from reporting.pdf_report import build_pdf_report
from tests.fakes.fake_mt5 import FakeMT5


def test_every_mt5_timeframe_is_selectable() -> None:
    assert {timeframe.value for timeframe in Timeframe} == set(TIMEFRAME_VALUES)


def test_clear_stop_confirmation_ignores_case_and_extra_spaces() -> None:
    assert stop_confirmation_matches("CLEAR STOP")
    assert stop_confirmation_matches("clear stop")
    assert stop_confirmation_matches("  Clear    Stop  ")
    assert not stop_confirmation_matches("stop")


#: A Wednesday, 14:00 UTC. Pinned, and the pin is the point.
#:
#: This test used to seed the fake with `datetime.now(UTC) + 3h` and then
#: assert the newest bar was within two hours of the real clock. That holds
#: from Monday to Friday and fails every Saturday and Sunday: `_bar_times`
#: correctly skips weekends, so on a Saturday morning the newest bar it can
#: produce is Friday evening — twelve hours away, and growing until Monday.
#: A suite that goes red on its own at the weekend teaches everyone to ignore
#: a red suite, which is the expensive part.
MIDWEEK = datetime(2026, 8, 5, 14, 0, tzinfo=UTC)


def test_dashboard_reads_catalogue_bars_and_builds_pdf() -> None:
    connector = MT5Connector(MT5Config(), mt5_module=FakeMT5(now=MIDWEEK + timedelta(hours=3)))
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
        # Still the real assertion — the connector must serve bars that reach
        # up to the present, not stale history — measured against the clock the
        # fake was actually given.
        newest = frame.index[-1].to_pydatetime()
        assert abs((newest - (MIDWEEK + timedelta(hours=3))).total_seconds()) < 7200

        pdf = build_pdf_report(account, [], "EURUSD", spec, tick, {"H1": frame})
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 10_000
    finally:
        service.close()


def test_dashboard_reads_persistent_paper_positions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "paper.json"
    path.write_text('{"balance":99.5,"currency":"EUR","positions":[]}', encoding="utf-8")

    snapshot = load_paper_snapshot(path)

    assert snapshot is not None
    assert snapshot.equity == 99.5
    assert snapshot.currency == "EUR"


def test_dashboard_reads_market_brain_snapshot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "market_intelligence.json"
    path.write_text('{"world":{"risk_tone":"mixed"}}', encoding="utf-8")

    snapshot = load_market_intelligence(path)

    assert snapshot is not None
    assert snapshot["world"] == {"risk_tone": "mixed"}
