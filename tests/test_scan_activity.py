from __future__ import annotations

from datetime import UTC, datetime

from core.instrument import AssetClass
from monitoring.scan_activity import ScanActivityLedger, read_scan_activity
from scanner.universe import ScanBatch, ScanInspection


def test_scan_activity_keeps_batch_and_deep_reason(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scan_activity.json"
    now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    inspection = ScanInspection(
        inspected_at=now,
        symbol="EURUSD.i",
        path="Forex\\Majors\\EURUSD.i",
        asset_class=AssetClass.FOREX,
        status="SHORTLISTED",
        stage="deep_analysis_queued",
        reason="Queued",
        rank=1.25,
        spread_bps=0.8,
        quote_age_seconds=0.5,
    )
    batch = ScanBatch(
        candidates=(),
        inspections=(inspection,),
        inspected=1,
        rejected=0,
        next_cursor=1,
        universe_size=847,
    )
    ledger = ScanActivityLedger(path)

    ledger.record_batch(batch, now, "experimental_live")
    ledger.record_deep_decision(
        "EURUSD.i",
        "DEEP_REJECTED",
        "NO_SIGNAL",
        "Confluence score below threshold",
        now,
    )
    state = read_scan_activity(path)

    assert state["total_inspections"] == 1
    assert state["universe_size"] == 847
    assert state["symbols"]["EURUSD.i"]["deep_status"] == "DEEP_REJECTED"
    assert state["recent"][-1]["deep_detail"] == "Confluence score below threshold"


def test_scan_activity_recent_log_is_bounded(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scan_activity.json"
    monkeypatch.setattr("monitoring.scan_activity.MAX_RECENT_INSPECTIONS", 2)
    ledger = ScanActivityLedger(path)
    now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    for number in range(3):
        inspection = ScanInspection(
            now,
            f"TEST{number}",
            "Test",
            AssetClass.FOREX,
            "REJECTED",
            "quote",
            "Market closed",
        )
        ledger.record_batch(ScanBatch((), (inspection,), 1, 1, number, 3), now, "monitor")

    state = read_scan_activity(path)
    assert [row["symbol"] for row in state["recent"]] == ["TEST1", "TEST2"]
