from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.failed_session_breakout import FailedSessionBreakout
from config.schema import FailedSessionBreakoutConfig
from core.types import MarketContext, Series, Timeframe


def _context(symbol: str = "SPX500", *, break_down: bool = False) -> MarketContext:
    index = pd.date_range("2026-08-31 06:00", periods=38, freq="5min", tz=UTC)
    close = np.full(len(index), 5000.0)
    close[-1] = 4988.0 if break_down else 5012.0
    frame = pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(close, 5005.0),
            "low": np.minimum(close, 4995.0),
            "close": close,
            "spread": 20.0,
        },
        index=index,
    )
    now = datetime(2026, 8, 31, 9, 5, tzinfo=UTC)
    return MarketContext(
        symbol=symbol,
        now=now,
        series={Timeframe.M5: Series(symbol, Timeframe.M5, frame, now)},
    )


def test_break_above_is_faded_short_with_half_range_stop() -> None:
    signal = FailedSessionBreakout(FailedSessionBreakoutConfig(enabled=True)).analyze(_context())

    assert signal.score < 0
    assert signal.invalidation_price == 5017.0


def test_break_below_is_faded_long() -> None:
    module = FailedSessionBreakout(FailedSessionBreakoutConfig(enabled=True))
    assert module.analyze(_context(break_down=True)).score > 0


def test_other_symbols_cannot_inherit_spx500_result() -> None:
    module = FailedSessionBreakout(FailedSessionBreakoutConfig(enabled=True))
    assert module.analyze(_context("US30")).score == 0
