"""Closed-M1 breakout entries must be fresh, directional and symmetric."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.m1_micro_breakout import M1MicroBreakout
from config.schema import M1MicroBreakoutConfig
from core.types import MarketContext, Series, Tick, Timeframe

NOW = datetime(2026, 8, 13, 10, tzinfo=UTC)


def _frame(closes: list[float], *, minutes: int, latest_volume: int = 100) -> pd.DataFrame:
    index = pd.date_range(end=NOW, periods=len(closes), freq=f"{minutes}min", tz=UTC)
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": pd.concat([open_, close], axis=1).max(axis=1) + 0.00004,
            "low": pd.concat([open_, close], axis=1).min(axis=1) - 0.00004,
            "close": close,
            "tick_volume": 100,
            "spread": 8,
            "real_volume": 0,
        },
        index=index,
    )
    frame.iloc[-1, frame.columns.get_loc("tick_volume")] = latest_volume
    return frame


def _long_context(*, breakout: float = 1.1080, volume: int = 180) -> MarketContext:
    m5 = [1.0900 + index * 0.00020 for index in range(80)]
    lead = [1.1050 + index * 0.00003 for index in range(40)]
    base = [1.10615, 1.10618, 1.10614, 1.10620, 1.10616, 1.10619]
    m1 = [*lead, *base, breakout]
    return MarketContext(
        symbol="EURUSD.i",
        now=NOW,
        series={
            Timeframe.M1: Series(
                "EURUSD.i", Timeframe.M1, _frame(m1, minutes=1, latest_volume=volume), NOW
            ),
            Timeframe.M5: Series("EURUSD.i", Timeframe.M5, _frame(m5, minutes=5), NOW),
        },
        tick=Tick("EURUSD.i", NOW, breakout, breakout + 0.00005),
    )


def _analyse(context: MarketContext, **overrides):  # type: ignore[no-untyped-def]
    values = {
        "minimum_m5_separation_atr": 0.02,
        "minimum_m5_slope_atr": 0.01,
        "maximum_base_width_atr": 4.0,
        "minimum_body_atr": 0.20,
        "minimum_volume_ratio": 1.10,
    }
    values.update(overrides)
    return M1MicroBreakout(M1MicroBreakoutConfig(**values)).analyze(context)


def _mirror(context: MarketContext) -> MarketContext:
    pivot = 1.10
    series = {}
    for timeframe, item in context.series.items():
        frame = item.df.copy()
        old_open = frame["open"].copy()
        old_high = frame["high"].copy()
        old_low = frame["low"].copy()
        old_close = frame["close"].copy()
        frame["open"] = pivot - (old_open - pivot)
        frame["high"] = pivot - (old_low - pivot)
        frame["low"] = pivot - (old_high - pivot)
        frame["close"] = pivot - (old_close - pivot)
        series[timeframe] = Series(item.symbol, timeframe, frame, NOW)
    price = pivot - (context.tick.bid - pivot)
    return MarketContext(
        symbol=context.symbol,
        now=NOW,
        series=series,
        tick=Tick(context.symbol, NOW, price - 0.00005, price),
    )


def test_fresh_m1_break_aligned_with_m5_is_long() -> None:
    signal = _analyse(_long_context())

    assert signal.score > 0, signal.reasoning
    assert signal.details["setup"] == "m1_micro_breakout"
    assert signal.invalidation_price is not None
    assert signal.invalidation_price < 1.1080


def test_mirrored_setup_is_short() -> None:
    context = _mirror(_long_context())
    signal = _analyse(context)

    assert signal.score < 0, signal.reasoning
    assert signal.invalidation_price is not None
    assert signal.invalidation_price > context.tick.ask


def test_remaining_outside_the_range_is_not_a_second_signal() -> None:
    context = _long_context()
    frame = context.series[Timeframe.M1].df.copy()
    extra = frame.iloc[[-1]].copy()
    extra.index = [frame.index[-1] + pd.Timedelta(minutes=1)]
    extra.loc[:, "open"] = 1.1080
    extra.loc[:, "close"] = 1.1082
    extra.loc[:, "high"] = 1.10824
    extra.loc[:, "low"] = 1.10796
    extended = pd.concat([frame, extra])
    context = MarketContext(
        context.symbol,
        context.now,
        {**context.series, Timeframe.M1: Series(context.symbol, Timeframe.M1, extended, NOW)},
        context.tick,
    )

    signal = _analyse(context)

    assert signal.score == 0


def test_weak_participation_does_not_create_a_trade() -> None:
    signal = _analyse(_long_context(volume=80), minimum_volume_ratio=1.10)

    assert signal.score == 0
    assert "volume" in signal.reasoning
