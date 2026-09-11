from pathlib import Path


def test_goud360_is_a_shadow_replay_of_only_the_two_gold_sections() -> None:
    launcher = (Path(__file__).parents[1] / "goud360.cmd").read_text()

    assert "--days 360" in launcher
    assert "--only section_six_gold_m5,section_ten_gold_m1" in launcher
    assert "--section-markets" in launcher
    assert "--jarvis-replay" in launcher
    assert "--csv runtime\\hoeveel-goud-360.csv" in launcher
    assert "section_five_ndx100_m5" not in launcher
    assert "--live-only" not in launcher
