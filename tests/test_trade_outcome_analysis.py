from datetime import datetime

from scripts.trade_outcome_analysis import Row, _tertiles, render


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


def test_diagnostics_use_the_distribution_of_accepted_trades() -> None:
    rows = [
        Row(datetime(2025, 1, day + 1), "section_ten_gold_m1", "LONG", 1.0, breakout_atr=value)
        for day, value in enumerate((0.41, 0.45, 0.52, 0.60, 0.75, 1.10))
    ]

    groups = _tertiles(rows, lambda row: row.breakout_atr, ("laag", "midden", "hoog"))

    assert len(groups) == 3
    assert sum(map(len, groups.values())) == len(rows)


def test_constant_diagnostic_is_called_out_instead_of_looking_useful(tmp_path) -> None:
    rows = [
        Row(datetime(2025, 1, day + 1), "section_ten_gold_m1", "LONG", 1.0, breakout_atr=0.4)
        for day in range(6)
    ]

    report = render(tmp_path / "gold.csv", rows)

    assert "GEEN ONDERSCHEID (0.400)" in report
    assert "trek hier geen filterconclusie uit" in report


def test_human_context_report_splits_the_preregistered_market_stories(tmp_path) -> None:
    rows = [
        Row(
            datetime(2025, 1, 1),
            "human_context_decision",
            "LONG",
            2.0,
            story="trend_continuation",
        ),
        Row(
            datetime(2025, 1, 2),
            "human_context_decision",
            "SHORT",
            -1.0,
            story="failed_auction",
        ),
    ]

    report = render(tmp_path / "human.csv", rows)

    story_block = report.split("PER MARKTVERHAAL", 1)[1].split("PER SECTIE + UTC-UUR", 1)[0]
    assert "trend_continuation" in story_block
    assert "failed_auction" in story_block
