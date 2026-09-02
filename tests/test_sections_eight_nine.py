from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from analysis.sections_eight_nine import (
    SectionEightTrendDayH1,
    SectionNineSessionVwapM30,
    SectionTenGoldM1,
)
from config.loader import load_settings
from config.schema import (
    SectionEightTrendDayConfig,
    SectionNineSessionVwapConfig,
    SectionTenGoldM1Config,
)
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
    signal = SectionNineSessionVwapM30(config).analyze(_context("USDJPY.i", Timeframe.M30, frame))

    assert signal.score < 0.0
    assert signal.invalidation_price is not None


def test_new_sections_are_live_promoted_with_measured_targets() -> None:
    """Section nine is DELIBERATELY not on the live list any more.

    SECTION 5 AND SECTION 9 CAME OFF ON 2 SEPTEMBER, on the owner's dry-run:

        section_five_m5        170 trades   -1,09 R   EUR  -1,11
        section_nine_vwap_m30    6 trades   -0,02 R   EUR  -0,08

    Section five is the clear one -- 170 trades is not noise any more and the
    number is under zero. Section nine on six trades is not proven bad; six
    observations prove nothing either way. It is off because it is not proven
    GOOD while spending real money. Both stay enabled and weighted, so they
    are still measured in the shadow.
    """
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)

    assert settings.analysis.section_eight_trend_day_h1.enabled is True
    assert settings.analysis.section_ten_gold_m1.enabled is True
    assert settings.analysis.section_ten_gold_m1.minimum_break_atr == 0.75
    assert "section_eight_trend_day_h1" in settings.analysis.confluence.live_enabled_modules
    assert "section_ten_gold_m1" in settings.analysis.confluence.live_enabled_modules

    live = settings.analysis.confluence.live_enabled_modules
    assert "section_nine_vwap_m30" not in live
    assert "section_five_m5" not in live
    # Off is not deleted. A module with no weight cannot be measured, and that
    # is how three M1 detectors went unjudged for months.
    assert settings.analysis.section_nine_vwap_m30.enabled is True
    assert settings.analysis.confluence.weights["section_nine_vwap_m30"] > 0
    assert settings.analysis.confluence.weights["section_five_m5"] > 0
    assert (
        settings.analysis.confluence.target_r_multiple_by_family["section_eight_trend_day_h1"]
        == 1.0
    )
    assert settings.analysis.confluence.target_r_multiple_by_family["section_nine_vwap_m30"] == 1.5
    assert settings.analysis.confluence.target_r_multiple_by_family["section_ten_gold_m1"] == 1.5


def test_section_ten_enters_first_closed_bar_retest_after_large_gold_break() -> None:
    index = pd.date_range("2026-08-01", periods=250, freq="1min", tz=UTC)
    close = np.full(len(index), 100.0)
    high = np.full(len(index), 101.0)
    low = np.full(len(index), 99.0)
    close[-2], high[-2], low[-2] = 104.0, 104.5, 100.0
    close[-1], high[-1], low[-1] = 102.0, 103.0, 101.0
    frame = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
            "spread": 1.0,
        },
        index=index,
    )
    signal = SectionTenGoldM1(SectionTenGoldM1Config(enabled=True)).analyze(
        _context("XAUUSD", Timeframe.M1, frame)
    )

    assert signal.score > 0.0
    assert signal.invalidation_price is not None
    assert signal.details["wait_bars"] == 1
