"""Market-regime classification is closed-bar, deterministic and non-directional."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.market_regime import MarketRegime
from config.schema import MarketRegimeConfig
from core.types import MarketContext, Series, Timeframe


def _frame(close: np.ndarray, *, range_size: float = 0.002) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(close), freq="1h", tz=UTC)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + range_size / 2,
            "low": close - range_size / 2,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )


def _context(fast: pd.DataFrame, slow: pd.DataFrame) -> MarketContext:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={
            Timeframe.H1: Series("EURUSD", Timeframe.H1, fast, now),
            Timeframe.H4: Series("EURUSD", Timeframe.H4, slow, now),
        },
    )


def test_linear_prices_are_trending_not_extreme_when_atr_is_tied() -> None:
    fast = _frame(np.linspace(1.0, 1.2, 150))
    slow = _frame(np.linspace(1.0, 1.4, 150))

    signal = MarketRegime(MarketRegimeConfig()).analyze(_context(fast, slow))

    assert signal.score == 0.0
    assert signal.details["regime"] == "trend_up"
    assert signal.details["atr_percentile"] == 0.5


def test_oscillation_is_a_range() -> None:
    values = 1.1 + np.tile(np.array([-0.001, 0.001]), 75)
    signal = MarketRegime(MarketRegimeConfig()).analyze(_context(_frame(values), _frame(values)))

    assert signal.details["regime"] == "range"
    assert signal.confidence == 1.0


def test_recent_volatility_shock_has_precedence() -> None:
    values = np.linspace(1.0, 1.1, 150)
    fast = _frame(values)
    fast.loc[fast.index[-14:], "high"] += 0.05
    fast.loc[fast.index[-14:], "low"] -= 0.05

    signal = MarketRegime(MarketRegimeConfig()).analyze(
        _context(fast, _frame(np.linspace(1.0, 1.2, 150)))
    )

    assert signal.details["regime"] == "extreme"


def test_missing_history_fails_neutral() -> None:
    short = _frame(np.linspace(1.0, 1.1, 30))

    signal = MarketRegime(MarketRegimeConfig()).analyze(_context(short, short))

    assert signal.score == 0.0
    assert signal.confidence == 0.0
    assert "needs" in signal.reasoning
