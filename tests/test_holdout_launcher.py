from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_holdout_launcher_uses_prior_non_overlapping_year() -> None:
    launcher = (ROOT / "jarvis-holdout-2024-2025.cmd").read_text(encoding="utf-8")

    assert "--start-date 2024-09-01" in launcher
    assert "--end-date 2025-08-31" in launcher
    assert "--only section_six_gold_m5,section_ten_gold_m1" in launcher
    assert "--jarvis-replay" in launcher
    assert "--exit-grid kern" in launcher
    assert "runtime\\jarvis-holdout-2024-2025.csv" in launcher
