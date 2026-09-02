from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from analysis.sections_eight_nine import SectionEightTrendDayH1, SectionNineSessionVwapM30
from config.loader import load_settings
from config.schema import SectionEightTrendDayConfig, SectionNineSessionVwapConfig
from core.types import MarketContext, Series, Timeframe


def _context(symbol: str, timeframe: Timeframe, frame: pd.DataFrame) -> MarketContext:
    now = frame.index[-1].to_pydatetime()
    return MarketContext(symbol, now, {timeframe: Series(symbol, timeframe, frame, now)})


def test_section_eight_follows_an_extreme_prior_spx_close() -> None:
    index = pd.date_range("2026-08-29 00:00", periods=49, freq="1h", tz=UTC)
    close = np.full(len(index), 5000.0)
    yesterday = index.normalize() == pd.Timestamp("2026-08-30", tz=UTC)
    close[yesterday] = np.linspace(4991.0, 5009.5, yesterday.sum())
    frame = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": np.where(yesterday, 5010.0, close + 2.0),
            "low": np.where(yesterday, 4990.0, close - 2.0),
            "close": close,
            "spread": 1.0,
        },
        index=index,
    )
    # The production module needs 60 warm-up bars; prepend a quiet day.
    prefix_index = pd.date_range(
        end=index[0] - pd.Timedelta(hours=1), periods=20, freq="1h", tz=UTC
    )
    prefix = pd.DataFrame(
        {"open": 5000.0, "high": 5002.0, "low": 4998.0, "close": 5000.0, "spread": 1.0},
        index=prefix_index,
    )
    frame = pd.concat([prefix, frame])
    signal = SectionEightTrendDayH1(SectionEightTrendDayConfig(enabled=True)).analyze(
        _context("SPX500", Timeframe.H1, frame)
    )

    assert signal.score > 0.0
    assert signal.invalidation_price is not None


def test_section_nine_fades_a_two_atr_vwap_displacement() -> None:
    index = pd.date_range("2026-08-29 00:00", periods=100, freq="30min", tz=UTC)
    close = np.full(len(index), 150.0)
    close[-1] = 160.0
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 100.0,
            "spread": 1.0,
        },
        index=index,
    )
    config = SectionNineSessionVwapConfig(enabled=True, minimum_displacement_atr=1.0)
    signal = SectionNineSessionVwapM30(config).analyze(
        _context("USDJPY.i", Timeframe.M30, frame)
    )

    assert signal.score < 0.0
    assert signal.invalidation_price is not None


def test_new_sections_are_measured_but_not_live_promoted() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)

    assert settings.analysis.section_eight_trend_day_h1.enabled is True
    assert settings.analysis.section_nine_vwap_m30.enabled is True
    assert "section_eight_trend_day_h1" not in settings.analysis.confluence.live_enabled_modules
    assert "section_nine_vwap_m30" not in settings.analysis.confluence.live_enabled_modules
    assert settings.analysis.confluence.target_r_multiple_by_family[
        "section_eight_trend_day_h1"
    ] == 1.0
    assert settings.analysis.confluence.target_r_multiple_by_family[
        "section_nine_vwap_m30"
    ] == 1.5
