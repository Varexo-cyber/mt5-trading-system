from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from analysis.gold_cross_discoveries import GoldCrossDiscovery
from config.loader import load_settings
from config.schema import GoldCrossDiscoveryConfig
from core.types import MarketContext, Series, Timeframe
from scripts.dry_run_sections import _retimed


def _context(symbol: str, timeframe: Timeframe, frame: pd.DataFrame) -> MarketContext:
    now = frame.index[-1].to_pydatetime()
    return MarketContext(symbol, now, {timeframe: Series(symbol, timeframe, frame, now)})


def _rising_frame() -> pd.DataFrame:
    index = pd.date_range("2026-09-04 10:40", periods=80, freq="1min", tz=UTC)
    close = np.linspace(100.0, 110.0, len(index))
    close[-1] = 112.0
    return pd.DataFrame(
        {"open": close - 0.05, "high": close + 0.10, "low": close - 0.10, "close": close},
        index=index,
    )


def test_xaueur_fades_a_closed_bar_channel_breakout() -> None:
    config = GoldCrossDiscoveryConfig(
        enabled=True,
        allowed_symbols=("XAUEUR",),
        timeframe="M1",
        mechanism="channel_breakout",
        polarity=-1,
        session_start_hour_utc=7,
        session_end_hour_utc=13,
    )
    signal = GoldCrossDiscovery(config, name="section_eleven_xaueur_m1").analyze(
        _context("XAUEUR", Timeframe.M1, _rising_frame())
    )

    assert signal.score < 0.0
    assert signal.invalidation_price is not None
    assert signal.invalidation_price > _rising_frame()["close"].iloc[-1]


def test_gold_cross_discovery_is_exact_market_and_session_only() -> None:
    config = GoldCrossDiscoveryConfig(
        enabled=True,
        allowed_symbols=("XAUEUR",),
        timeframe="M1",
        session_start_hour_utc=13,
        session_end_hour_utc=20,
    )
    module = GoldCrossDiscovery(config, name="section_eleven_xaueur_m1")

    assert module.analyze(_context("XAUEUR", Timeframe.M1, _rising_frame())).score == 0.0
    assert module.analyze(_context("XAUGBP", Timeframe.M1, _rising_frame())).score == 0.0


def test_four_discoveries_are_shadowed_after_full_broker_replay() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)
    names = (
        "section_eleven_xaueur_m1",
        "section_twelve_xaugbp_m1",
        "section_thirteen_xauaud_m5",
        "section_fourteen_xaujpy_m1",
    )

    assert [settings.analysis.confluence.target_r_multiple_by_family[name] for name in names] == [
        1.5,
        1.0,
        0.75,
        1.5,
    ]
    assert all(settings.analysis.confluence.weights[name] == 0.0 for name in names)
    assert all(name not in settings.analysis.confluence.live_enabled_modules for name in names)
    assert all(settings.analysis.confluence.lone_floor_for(name) == 0.55 for name in names)

    measured = _retimed(settings, names[0], "M1")
    assert measured.analysis.confluence.weights[names[0]] == 1.0
    assert names[0] in measured.analysis.confluence.live_enabled_modules
    assert settings.analysis.confluence.weights[names[0]] == 0.0
