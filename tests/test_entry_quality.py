"""A valid direction is not permission to chase the price that confirms it."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.entry_quality import (
    EntryTimingDecision,
    assess_entry_quality,
    assess_review_drift,
)
from config.loader import load_settings
from core.instrument import AssetClass
from core.types import Direction, MarketContext, Series, Tick, Timeframe

NOW = datetime(2026, 8, 10, 13, 30, tzinfo=UTC)


def config():  # type: ignore[no-untyped-def]
    return load_settings(env_overrides=False).analysis.entry_quality


def context(closes: np.ndarray) -> MarketContext:
    index = pd.date_range("2026-08-10 06:00", periods=len(closes), freq="5min", tz=UTC)
    opens = np.r_[closes[0], closes[:-1]]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.25,
            "low": np.minimum(opens, closes) - 0.25,
            "close": closes,
            "tick_volume": 100,
            "spread": 2,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol="TEST",
        now=NOW,
        series={Timeframe.M5: Series("TEST", Timeframe.M5, frame, NOW)},
        tick=Tick("TEST", NOW, float(closes[-1]), float(closes[-1] + 0.02)),
    )


def quiet_history() -> np.ndarray:
    return 100.0 + np.sin(np.arange(80) / 4.0) * 0.10


def test_a_long_does_not_buy_the_end_of_a_large_m5_impulse() -> None:
    closes = quiet_history()
    closes[-4:] = [100.0, 100.9, 101.8, 102.7]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.STOCK, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED"
    assert verdict.favourable_extension_atr is not None
    assert verdict.favourable_extension_atr > 1.25


def test_a_short_does_not_sell_the_end_of_a_large_m5_impulse() -> None:
    closes = quiet_history()
    closes[-4:] = [100.0, 99.1, 98.2, 97.3]

    verdict = assess_entry_quality(context(closes), Direction.SHORT, AssetClass.STOCK, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED"


def test_an_ordinary_move_is_still_tradeable() -> None:
    closes = quiet_history()
    closes[-4:] = [100.0, 100.08, 100.16, 100.24]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.FOREX, config())

    assert verdict.decision is EntryTimingDecision.ENTER_NOW


def test_a_pullback_that_is_still_falling_waits_for_the_turn() -> None:
    closes = quiet_history()
    closes[-4:] = [100.5, 100.45, 100.35, 100.0]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.FOREX, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "PULLBACK_STILL_ACTIVE"


def test_a_retest_after_an_older_impulse_can_clear_next_cycle() -> None:
    closes = quiet_history()
    closes[-8:] = [100.0, 100.8, 101.6, 102.0, 101.6, 101.3, 101.35, 101.45]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.STOCK, config())

    assert verdict.decision is EntryTimingDecision.ENTER_NOW


def test_review_price_is_bound_in_atr_not_raw_dollars() -> None:
    closes = quiet_history()
    market = context(closes)
    allowed = assess_review_drift(market, Direction.LONG, 100.0, 100.05, 20.0, config())
    stale = assess_review_drift(market, Direction.LONG, 100.0, 100.5, 20.0, config())

    assert allowed.decision is EntryTimingDecision.ENTER_NOW
    assert stale.decision is EntryTimingDecision.WAIT_RETEST


def test_slow_ai_response_never_becomes_permission_to_trade_old_analysis() -> None:
    closes = quiet_history()
    verdict = assess_review_drift(context(closes), Direction.LONG, 100.0, 100.0, 46.0, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert "review-age" in verdict.detail


def test_missing_entry_chart_fails_closed() -> None:
    bare = MarketContext(symbol="TEST", now=NOW, series={}, tick=None)

    verdict = assess_entry_quality(bare, Direction.LONG, AssetClass.FOREX, config())

    assert verdict.decision is EntryTimingDecision.DATA_UNAVAILABLE
