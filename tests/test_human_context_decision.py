from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.human_context_decision import HumanContextDecision
from config.loader import load_settings
from config.schema import HumanContextDecisionConfig
from core.types import MarketContext, Series, Timeframe
from runner.service import build_analysis_modules
from scripts.dry_run_sections import _frames_read


def _frame(freq: str, *, rising: bool = True, periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=periods, freq=freq, tz=UTC)
    close = np.linspace(100.0, 104.0 if rising else 96.0, periods)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100.0,
            "spread": 1.0,
        },
        index=index,
    )


def _context(*, rising_context: bool = True) -> MarketContext:
    m5 = _frame("5min")
    m5.loc[:, "open"] = 100.0
    m5.loc[:, "high"] = 101.0
    m5.loc[:, "low"] = 99.0
    m5.loc[:, "close"] = 100.0
    m5.iloc[-1, m5.columns.get_loc("open")] = 98.5
    m5.iloc[-1, m5.columns.get_loc("low")] = 98.5
    m5.iloc[-1, m5.columns.get_loc("high")] = 99.3
    m5.iloc[-1, m5.columns.get_loc("close")] = 99.25
    now = m5.index[-1].to_pydatetime()
    frames = {
        Timeframe.M5: m5,
        Timeframe.M15: _frame("15min", rising=rising_context),
        Timeframe.H1: _frame("1h", rising=rising_context),
        Timeframe.H4: _frame("4h", rising=rising_context),
    }
    return MarketContext(
        "XAUUSD",
        now,
        {tf: Series("XAUUSD", tf, frame, now) for tf, frame in frames.items()},
    )


def test_failed_auction_needs_a_closed_reclaim_and_context() -> None:
    decision = HumanContextDecision(HumanContextDecisionConfig(enabled=True, stop_buffer_atr=0.05))

    signal = decision.analyze(_context())

    assert signal.score > 0
    assert signal.invalidation_price is not None
    assert signal.details["story"] == "failed_auction"
    assert signal.details["confirmation"] == "closed_m5"
    assert "higher timeframes agree" in signal.reasoning


def test_same_price_event_is_neutral_when_higher_timeframes_oppose_it() -> None:
    decision = HumanContextDecision(HumanContextDecisionConfig(enabled=True, stop_buffer_atr=0.05))

    signal = decision.analyze(_context(rising_context=False))

    assert signal.score == 0
    assert "directional control" in signal.reasoning


def test_human_decision_is_built_for_replay_but_has_no_live_permission() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)
    built = {module.name for module in build_analysis_modules(settings)}

    assert "human_context_decision" in built
    assert settings.analysis.human_context_decision.enabled
    assert "human_context_decision" not in settings.analysis.confluence.live_enabled_modules
    assert set(settings.analysis.confluence.live_enabled_modules) == {
        "section_six_gold_m5",
        "section_ten_gold_m1",
    }


def test_human30_launcher_is_one_frozen_shadow_measurement() -> None:
    launcher = (Path(__file__).parents[1] / "human30.cmd").read_text(encoding="utf-8")

    assert "--days 30" in launcher
    assert "--only human_context_decision" in launcher
    assert "--jarvis-replay" in launcher
    assert "--fixed-exits" in launcher
    assert "--exit-grid" not in launcher
    assert "runtime\\human-context-30.csv" in launcher


def test_replay_fetches_every_context_timeframe_the_decision_reads() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)

    frames = set(
        _frames_read(
            settings,
            Timeframe.M5,
            Timeframe.M1,
            ("human_context_decision",),
        )
    )

    assert {Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4} <= frames
