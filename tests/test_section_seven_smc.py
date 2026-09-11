from pathlib import Path

from analysis.evidence_families import family_for
from analysis.section_seven_smc import SectionSevenGoldSmc
from config.loader import load_settings
from runner.service import build_analysis_modules


def _settings():
    return load_settings(
        "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
    )


def test_section_seven_is_built_but_cannot_trade_live() -> None:
    settings = _settings()
    names = {module.name for module in build_analysis_modules(settings)}

    assert SectionSevenGoldSmc.name in names
    assert settings.analysis.section_seven_gold_smc.enabled
    assert SectionSevenGoldSmc.name not in settings.analysis.confluence.live_enabled_modules


def test_section_seven_has_all_requested_context_and_one_daily_setup_contract() -> None:
    source = Path("analysis/section_seven_smc.py").read_text(encoding="utf-8")

    for timeframe in ("M15", "M30", "H1", "H4"):
        assert f"Timeframe.{timeframe}" in source
    assert "_last_trade_day" in source
    assert "liquidity sweep" in source
    assert "displacement/BOS" in source
    assert "FVG retest" in source
    assert "liquidity_target" in source
    assert family_for(SectionSevenGoldSmc.name) == "liquidity_structure"


def test_research_launcher_compares_m1_and_slower_execution_clocks() -> None:
    launcher = Path("sectie7.cmd").read_text(encoding="utf-8")

    assert "--days 180" in launcher
    assert "--only section_seven_gold_smc" in launcher
    assert "--sweep M1 M5 M15 M30 H1" in launcher
    assert "--no-m1" not in launcher
    assert "--fixed-exits" in launcher
    assert "--strict-risk" in launcher
    assert "M1, M5, M15, M30 en H1" in launcher
