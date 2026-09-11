from datetime import datetime

from scripts.trade_outcome_analysis import Row, render


def test_s6s10_launcher_is_read_only_and_runs_all_measurements() -> None:
    from pathlib import Path

    launcher = Path("s6s10-meting.cmd").read_text(encoding="utf-8")
    assert "--days 360" in launcher
    assert "--only section_six_gold_m5,section_ten_gold_m1" in launcher
    assert "--exit-grid kern" in launcher
    assert "--trend-grid" in launcher
    assert "--fault-exit-grid" in launcher
    assert "scripts.trade_outcome_analysis" in launcher


def test_only_hours_negative_in_both_halves_are_candidates(tmp_path) -> None:
    rows = []
    for day in range(1, 12):
        rows.append(Row(datetime(2025, 1, day, 5), "s10", "LONG", -1.0))
        rows.append(Row(datetime(2025, 7, day, 5), "s10", "LONG", -1.0))
        rows.append(Row(datetime(2025, 1, day, 6), "s10", "LONG", -1.0))
        rows.append(Row(datetime(2025, 7, day, 6), "s10", "LONG", 2.0))

    report = render(tmp_path / "gold.csv", rows)

    candidate_block = report.split("HERHAALDE VERLIESUREN", 1)[1].split("RODE DAGEN", 1)[0]
    assert "s10|05:00" in candidate_block
    assert "s10|06:00" not in candidate_block
