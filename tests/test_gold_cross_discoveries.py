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


def test_the_shadow_wiring_holds_for_every_discovery_section() -> None:
    """THE FOUR GOLD-CROSS DISCOVERIES ARE GONE, on the owner's instruction of
    5 September: sections eleven, twelve and thirteen are XAUJPY now, and
    fourteen went with them because it was a second XAUJPY M1.

    What that measurement said, kept here so the removal is not mistaken for
    an accident: `sectie10.cmd 180` put the four crosses at -110,22 R over
    3.217 trades against XAUUSD's +79,12 over 595, and the position cap cost
    section six a further 260 trades worth +40,48 R.

    The PROPERTY the deleted test was really about survives and is asserted
    here instead: every discovery section this config carries is weighted zero,
    is off the live allowlist, and gets its weight back only on the settings
    COPY a measurement pass uses. An earlier version pinned the four names, so
    removing them failed here with nothing wrong.
    """
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)
    confluence = settings.analysis.confluence
    names = [
        name
        for name in dir(settings.analysis)
        if name.startswith("section_")
        and isinstance(getattr(settings.analysis, name, None), GoldCrossDiscoveryConfig)
    ]

    assert names, "no discovery section left; if that was deliberate, delete this test"
    for name in names:
        assert confluence.weights[name] == 0.0, name
        assert name not in confluence.live_enabled_modules, name
        assert confluence.lone_floor_for(name) == 0.55, name
        assert name in confluence.target_r_multiple_by_family, name

    measured = _retimed(settings, names[0], settings.analysis.__getattribute__(names[0]).timeframe)
    assert measured.analysis.confluence.weights[names[0]] == 1.0
    assert names[0] in measured.analysis.confluence.live_enabled_modules
    assert confluence.weights[names[0]] == 0.0, "the measurement pass leaked into the account"


def test_three_btc_discoveries_are_separate_and_shadow_only() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)
    sections = {
        "section_fifteen_btc_m1": ("M1", "adaptive_channel_100", 3.0, 2.0),
        "section_sixteen_btc_m5": ("M5", "adaptive_channel_25", 6.0, 2.0),
        "section_seventeen_btc_m15": ("M15", "trend_pullback", 4.0, 4.0),
    }

    for name, expected in sections.items():
        config = getattr(settings.analysis, name)
        assert (config.timeframe, config.mechanism, config.stop_atr, config.target_r) == expected
        assert config.allowed_symbols == ("BTCUSD",)
        assert settings.analysis.confluence.weights[name] == 0.0
        assert name not in settings.analysis.confluence.live_enabled_modules

    assert not settings.analysis.section_seventeen_btc_m15.weekday_only
    assert settings.analysis.section_fifteen_btc_m1.shadow_break_even_at_r == 1.5
    assert settings.analysis.section_sixteen_btc_m5.shadow_break_even_at_r == 0.0
    assert settings.analysis.section_seventeen_btc_m15.shadow_break_even_at_r == 1.0
