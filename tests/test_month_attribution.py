import csv
from pathlib import Path

from scripts.month_attribution import load_trades, render

FIELDS = [
    "when",
    "symbol",
    "module",
    "outcome",
    "result_r_fixed_stop",
    "pnl_money_fixed_stop",
    "managed_r_LIVE",
    "managed_money_LIVE",
]


def _write(path: Path) -> None:
    rows = [
        ["2026-01-02T10:00:00", "X", "goud", "TRADE", "9", "90", "1", "10"],
        ["2026-01-03T10:00:00", "X", "slecht", "TRADE", "-3", "-30", "-2", "-20"],
        ["2026-02-02T10:00:00", "X", "goud", "TRADE", "1", "10", "", ""],
        ["2026-02-03T10:00:00", "X", "slecht", "TRADE", "-4", "-40", "", ""],
        ["2026-02-04T10:00:00", "X", "slecht", "REFUSED", "-99", "-990", "", ""],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def test_report_uses_managed_result_and_ignores_refusals(tmp_path: Path) -> None:
    path = tmp_path / "hoeveel.csv"
    _write(path)

    trades = load_trades(path)
    report = render(path, trades)

    assert len(trades) == 4
    assert "2/2 rode maanden" in report
    assert "2026-01   ROOD       -1.00" in report
    assert "2026-02   ROOD       -3.00" in report
    assert "zonder slecht" in report
    assert "+2.00" in report
