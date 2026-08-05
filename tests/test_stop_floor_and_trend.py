"""Two gates the adviser's rejections pointed straight at.

A day of vetoes said almost the same thing every time. Ten of them named the
stop — "sits inside recent M1/M5 noise", "only 0.41 ATR on H1", "3.9 pips
against a market whose bars routinely swing 5-15" — and the rest named the
direction: "countertrend short against a clear, accelerating multi-timeframe
uptrend". Both were correct, and both are cheaper to encode than to pay for
once per candidate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.confluence import ConfluenceEngine
from config.loader import load_settings
from core.types import Direction, MarketContext, Series, Signal, Tick, Timeframe, TradingMode

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def series(
    closes: list[float], *, minutes: int, wick: float = 0.2, timeframe: Timeframe = Timeframe.D1
) -> Series:
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c + wick for c in closes],
            "low": [c - wick for c in closes],
            "close": closes,
            "tick_volume": [1] * len(closes),
        },
        index=pd.DatetimeIndex(
            [NOW - timedelta(minutes=minutes * (len(closes) - i)) for i in range(len(closes))]
        ),
    )
    return Series(symbol="SPX500", timeframe=timeframe, df=frame, fetched_at=NOW)


def engine(**overrides):  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    config = settings.analysis.confluence
    if overrides:
        config = config.model_copy(update=overrides)
    return ConfluenceEngine([], config)


def context(frames: dict[Timeframe, list[float]], *, minutes: int = 60) -> MarketContext:
    return MarketContext(
        symbol="SPX500",
        now=NOW,
        series={tf: series(closes, minutes=minutes, timeframe=tf) for tf, closes in frames.items()},
        tick=None,
    )


# ------------------------------------------------------------- stop floor ---


def test_the_stop_floor_is_configured_above_the_noise_band() -> None:
    """The adviser rejected stops between 0.13 and 0.85 ATR as noise. A floor
    inside that band would leave the problem exactly where it was."""
    assert engine().config.min_stop_atr >= 0.8


def test_a_zero_floor_leaves_the_structural_stop_untouched() -> None:
    """Switching it off has to be a real off, not a smaller floor."""
    assert engine(min_stop_atr=0.0).config.min_stop_atr == 0.0


# ---------------------------------------------------- higher-timeframe trend ---


def rising(bars: int = 40, step: float = 3.0) -> list[float]:
    return [4000.0 + i * step for i in range(bars)]


def falling(bars: int = 40, step: float = 3.0) -> list[float]:
    return [4000.0 - i * step for i in range(bars)]


def flat(bars: int = 40) -> list[float]:
    return [4000.0 + (1.0 if i % 2 else -1.0) for i in range(bars)]


def conflict(engine_, frames, direction: Direction) -> str | None:  # type: ignore[no-untyped-def]
    return engine_.higher_timeframe_conflict(context(frames), direction)


def test_a_short_into_a_multi_week_uptrend_is_refused() -> None:
    """Exactly the trade the adviser kept rejecting on SPX500 and ASX200."""
    said = conflict(engine(), {Timeframe.D1: rising()}, Direction.SHORT)
    assert said is not None
    assert "uptrend" in said and "short" in said


def test_a_long_with_that_uptrend_is_fine() -> None:
    assert conflict(engine(), {Timeframe.D1: rising()}, Direction.LONG) is None


def test_a_long_into_a_downtrend_is_refused() -> None:
    said = conflict(engine(), {Timeframe.D1: falling()}, Direction.LONG)
    assert said is not None
    assert "downtrend" in said


def test_a_flat_higher_timeframe_is_not_an_objection() -> None:
    """The gate is one-sided by design: it never creates a signal, and a market
    going nowhere on D1 says nothing about an H1 setup."""
    assert conflict(engine(), {Timeframe.D1: flat()}, Direction.SHORT) is None
    assert conflict(engine(), {Timeframe.D1: flat()}, Direction.LONG) is None


def test_a_mild_drift_does_not_block() -> None:
    """Counter-trend trades are not banned. Trading into a strong, still
    accelerating trend is."""
    assert conflict(engine(), {Timeframe.D1: rising(step=0.05)}, Direction.SHORT) is None


def test_a_missing_timeframe_is_skipped_rather_than_assumed() -> None:
    assert conflict(engine(), {}, Direction.SHORT) is None


def test_too_little_history_is_skipped() -> None:
    """A newly listed instrument has no multi-week trend to be wrong about."""
    assert conflict(engine(), {Timeframe.D1: rising(bars=5)}, Direction.SHORT) is None


def test_the_weekly_is_checked_too() -> None:
    """D1 flat while W1 climbs is still a trade taken into the tide."""
    said = conflict(engine(), {Timeframe.D1: flat(), Timeframe.W1: rising()}, Direction.SHORT)
    assert said is not None
    assert "W1" in said


def test_an_empty_timeframe_list_disables_the_gate() -> None:
    assert (
        conflict(engine(htf_trend_timeframes=()), {Timeframe.D1: rising()}, Direction.SHORT) is None
    )


def test_the_threshold_is_what_decides() -> None:
    frames = {Timeframe.D1: rising(step=1.0)}
    assert conflict(engine(htf_trend_veto=0.1), frames, Direction.SHORT) is not None
    assert conflict(engine(htf_trend_veto=5.0), frames, Direction.SHORT) is None


def test_the_measure_is_scale_free() -> None:
    """A dimensionless reading is the whole reason one threshold can serve gold,
    an index and a currency pair. Ten times the price with ten times the range
    has to read the same."""
    small = engine().higher_timeframe_conflict(
        context({Timeframe.D1: [100.0 + i * 0.3 for i in range(40)]}), Direction.SHORT
    )
    large = engine().higher_timeframe_conflict(
        MarketContext(
            symbol="XAUUSD",
            now=NOW,
            series={
                Timeframe.D1: series([1000.0 + i * 3.0 for i in range(40)], minutes=60, wick=2.0)
            },
            tick=None,
        ),
        Direction.SHORT,
    )
    assert (small is None) == (large is None)


def test_the_normalisation_matches_the_health_reader() -> None:
    """Both express drift in `sqrt(bars) * ATR`, so "a strong trend" means one
    thing in this codebase rather than two."""
    closes = np.array(rising(), dtype=float)
    bars = engine().config.htf_trend_lookback
    window = closes[-bars:]
    slope = float(np.polyfit(np.arange(bars, dtype=float), window, 1)[0])
    assert slope > 0
    # The engine's own reading on the same data must have the same sign.
    said = conflict(engine(), {Timeframe.D1: list(closes)}, Direction.SHORT)
    assert said is not None


# ------------------------------------------- the floor, through the engine ---


class FixedModule:
    """A module that always returns the same read, with a chosen invalidation."""

    def __init__(self, name: str, score: float, invalidation: float | None) -> None:
        self.name = name
        self._signal = Signal(
            module=name,
            score=score,
            confidence=0.9,
            reasoning="fixture",
            invalidation_price=invalidation,
        )

    def analyze(self, ctx: MarketContext) -> Signal:
        return self._signal


#: A market that actually travels. The reachable-target gate measures how far
#: this instrument moves in `target_horizon_bars`, and refuses a target the
#: market does not reach — so a flat fixture is rejected before the stop is ever
#: looked at, and would have tested nothing.
BASE = 4000.0
DRIFT = 1.2


def travelling(bars: int = 120) -> list[float]:
    return [BASE + i * DRIFT + (0.6 if i % 2 else -0.6) for i in range(bars)]


def last_price() -> float:
    return travelling()[-1]


def priced_context() -> MarketContext:
    closes = travelling()
    price = closes[-1]
    return MarketContext(
        symbol="EURUSD",
        now=NOW,
        series={
            tf: series(closes, minutes=60, wick=1.5, timeframe=tf)
            for tf in (Timeframe.H1, Timeframe.M15, Timeframe.M5)
        },
        tick=Tick(symbol="EURUSD", time=NOW, bid=price - 0.05, ask=price + 0.05),
    )


def h1_atr() -> float:
    engine_ = ConfluenceEngine([], load_settings(env_overrides=False).analysis.confluence)
    return engine_._atr(priced_context().series[Timeframe.H1].df)


def built(invalidation: float | None, **overrides):  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    config = settings.analysis.confluence.model_copy(
        update={
            "weights": {"a": 1.0, "b": 1.0},
            "minimum_directional_modules": 2,
            "score_threshold": 1.0,
            "minimum_confidence": 0.1,
            # Off for these: the fixture trends up hard on purpose, and this
            # gate would refuse the long before the stop logic is reached.
            "htf_trend_timeframes": (),
            "entry_timing_timeframes": (),
            **overrides,
        }
    )
    modules = [FixedModule("a", 80.0, invalidation), FixedModule("b", 80.0, invalidation)]
    return ConfluenceEngine(modules, config).evaluate(priced_context(), TradingMode.PAPER)


def test_a_structural_stop_inside_the_noise_is_pushed_out() -> None:
    """The bug in one line: an invalidation price a hair below entry produced a
    stop at a fraction of an ATR, taken out by ordinary chop rather than by the
    trade being wrong."""
    idea = built(last_price() - 0.05)
    assert idea.approved, idea.reason
    assert abs(idea.entry - idea.stop_loss) == pytest.approx(h1_atr() * 0.8, rel=0.05)


def test_a_stop_already_wide_enough_is_left_where_it_is() -> None:
    """The floor is a floor, not a replacement. A genuine structural level far
    enough out keeps its own distance."""
    far = last_price() - h1_atr() * 4
    idea = built(far)
    assert idea.approved, idea.reason
    assert abs(idea.entry - idea.stop_loss) > h1_atr() * 3


def test_the_floor_can_be_switched_off() -> None:
    idea = built(last_price() - 0.05, min_stop_atr=0.0)
    assert idea.approved, idea.reason
    assert abs(idea.entry - idea.stop_loss) < h1_atr() * 0.5
