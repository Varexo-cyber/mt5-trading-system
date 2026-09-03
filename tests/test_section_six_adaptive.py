from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.section_six_adaptive import SectionSixGoldM5, SectionSixSpxH1
from config.loader import load_settings
from config.schema import SectionSixModelConfig
from core.types import MarketContext, Series, Tick, Timeframe

NOW = datetime(2026, 8, 31, 20, 0, tzinfo=UTC)


def _context(symbol: str, timeframe: Timeframe) -> MarketContext:
    index = pd.date_range(end=NOW, periods=120, freq=timeframe.duration)
    drift = np.linspace(0.0, 8.0, len(index))
    wave = np.sin(np.arange(len(index)) / 3.0)
    close = 1000.0 + drift + wave
    frame = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "tick_volume": np.linspace(100.0, 200.0, len(index)),
            "spread": 10,
        },
        index=index,
    )
    return MarketContext(
        symbol,
        NOW,
        {timeframe: Series(symbol, timeframe, frame, NOW)},
        Tick(symbol, NOW, bid=float(close[-1] - 0.1), ask=float(close[-1] + 0.1)),
    )


def test_gold_model_only_reads_xauusd_m5() -> None:
    config = SectionSixModelConfig(
        enabled=True, timeframe="M5", polarity=-1, threshold=0.0001, stop_atr=1.0
    )
    module = SectionSixGoldM5(config)

    signal = module.analyze(_context("XAUUSD", Timeframe.M5))
    wrong_market = module.analyze(_context("SPX500", Timeframe.M5))

    assert signal.score != 0.0
    assert signal.invalidation_price is not None
    assert wrong_market.score == 0.0


def test_failed_spx_variant_stays_disabled_by_default() -> None:
    module = SectionSixSpxH1(SectionSixModelConfig(timeframe="H1"))

    assert module.analyze(_context("SPX500", Timeframe.H1)).score == 0.0


def test_gold_route_does_not_emit_outside_its_measured_session() -> None:
    config = SectionSixModelConfig(
        enabled=True,
        timeframe="M5",
        polarity=-1,
        threshold=0.0001,
        stop_atr=0.8,
        session_start_hour_utc=0,
        session_end_hour_utc=1,
    )

    signal = SectionSixGoldM5(config).analyze(_context("XAUUSD", Timeframe.M5))

    assert signal.score == 0.0
    assert "outside measured" in signal.reasoning


def test_live_overlay_keeps_the_measured_gold_exit_and_rejects_spx() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)

    assert settings.analysis.section_six_gold_m5.enabled is True
    assert settings.analysis.section_six_spx_h1.enabled is False
    assert settings.analysis.section_six_gold_m5.threshold == 0.15
    assert settings.analysis.section_six_gold_m5.stop_atr == 0.8
    assert settings.analysis.section_six_gold_m5.long_only is True
    assert settings.analysis.section_six_gold_m5.session_start_hour_utc == 20
    assert settings.analysis.section_six_gold_m5.session_end_hour_utc == 2
    assert settings.analysis.confluence.target_r_multiple_by_family["section_six_gold_m5"] == 3.0
    assert settings.analysis.confluence.strategy_owned_entry_families == ("section_six_gold_m5",)
    assert "section_six_gold_m5" in settings.analysis.confluence.target_reach_advisory_families
    # OFF THE FIXED-EXIT LIST ON 3 SEPTEMBER, and that is the whole change the
    # owner asked for. While the label sat there the manager returned early and
    # this route received NO position management -- not live and not in the dry
    # run -- so "dry run with position management" could not be done at all.
    # On the same 180-day entries, break-even alone moved the outcome from
    # -53.23R to +36.42R, and that figure still stood on the broken offset.
    assert "JARVIS-S6-AU-M5" not in settings.trade_management.fixed_exit_comments
    assert settings.analysis.section_six_gold_m5.confirmation_bars == 12
