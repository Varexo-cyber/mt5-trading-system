"""A repeatable M5 re-entry that remains a discrete closed-bar event."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.ema_pullback_resume import EmaPullbackResume
from config.schema import EmaPullbackResumeConfig
from core.types import MarketContext, Series, Tick, Timeframe


def _frame(closes: list[float], *, minutes: int = 5) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(closes), freq=f"{minutes}min", tz=UTC)
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "open": open_,
            "high": pd.concat([open_, close], axis=1).max(axis=1) + 0.00008,
            "low": pd.concat([open_, close], axis=1).min(axis=1) - 0.00008,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )


def _context(m5: list[float], m1: list[float] | None = None) -> MarketContext:
    now = datetime(2026, 1, 6, 10, tzinfo=UTC)
    series = {Timeframe.M5: Series("EURUSD", Timeframe.M5, _frame(m5), now)}
    if m1 is not None:
        series[Timeframe.M1] = Series("EURUSD", Timeframe.M1, _frame(m1, minutes=1), now)
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series=series,
        tick=Tick("EURUSD", now, m5[-1], m5[-1] + 0.0001),
    )


def _long_setup() -> list[float]:
    trend = [1.1000 + i * 0.00020 for i in range(80)]
    top = trend[-1]
    return [*trend, top - 0.00025, top - 0.00050, top - 0.00075, top + 0.00005]


def _analyse(ctx: MarketContext, **overrides):  # type: ignore[no-untyped-def]
    defaults = {
        "minimum_separation_atr": 0.05,
        "minimum_slope_atr": 0.01,
        "maximum_slow_ema_breach_atr": 1.0,
    }
    defaults.update(overrides)
    return EmaPullbackResume(EmaPullbackResumeConfig(**defaults)).analyze(ctx)


def test_a_pullback_reclaim_in_an_uptrend_is_long() -> None:
    signal = _analyse(_context(_long_setup()))

    assert signal.score > 0, signal.reasoning
    assert signal.details["setup"] == "ema_pullback_resume"
    assert signal.invalidation_price is not None
    assert signal.invalidation_price < _long_setup()[-1]


def test_the_mirrored_setup_is_short() -> None:
    long = _long_setup()
    pivot = long[0]
    short = [pivot - (price - pivot) for price in long]

    signal = _analyse(_context(short))

    assert signal.score < 0, signal.reasoning
    assert signal.invalidation_price is not None
    assert signal.invalidation_price > short[-1]


def test_remaining_above_the_ema_is_not_another_entry() -> None:
    setup = _long_setup()
    closes = [*setup, setup[-1] + 0.00020]

    signal = _analyse(_context(closes))

    assert signal.score == 0
    assert "no fresh EMA reclaim" in signal.reasoning


def test_a_deep_break_through_the_slow_ema_is_not_called_a_pullback() -> None:
    closes = _long_setup()
    closes[-2] -= 0.00120
    signal = _analyse(
        _context(closes),
        maximum_slow_ema_breach_atr=0.01,
    )

    assert signal.score == 0
    assert "possible reversal" in signal.reasoning


def test_materially_opposing_m1_flow_blocks_the_entry() -> None:
    m1 = [1.1200 - i * 0.00020 for i in range(30)]

    signal = _analyse(
        _context(_long_setup(), m1),
        maximum_m1_adverse_atr=0.10,
    )

    assert signal.score == 0
    assert "M1 drift" in signal.reasoning
