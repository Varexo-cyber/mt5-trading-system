from pathlib import Path

from scripts.dry_run_sections import build_parser


def test_calendar_window_flags_parse() -> None:
    args = build_parser().parse_args(
        ["--start-date", "2025-09-01", "--end-date", "2025-09-30"]
    )

    assert args.start_date == "2025-09-01"
    assert args.end_date == "2025-09-30"


def test_september_launcher_is_gold_only_and_uses_the_complete_month() -> None:
    launcher = (Path(__file__).parents[1] / "september2025-goud.cmd").read_text()

    assert "--start-date 2025-09-01" in launcher
    assert "--end-date 2025-09-30" in launcher
    assert "--only section_six_gold_m5,section_ten_gold_m1" in launcher
    assert "--jarvis-replay" in launcher
