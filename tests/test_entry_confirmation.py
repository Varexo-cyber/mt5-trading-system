"""Do not sell into a market that is going up.

The engine finds a level, forms a view, and the order goes out on the same
tick. Nothing in between ever looked at which way price was moving right then
— so a short was sent into a market climbing through the very level it was
short against. The operator watched GBPJPY break resistance upward and the
system sold the break.

This is the cheapest form of waiting for confirmation and the only one needing
no state. It does not ask whether the market has proved the thesis right; it
asks whether the market has already started proving it wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from analysis.confluence import TradeIdea
from config.loader import load_settings
from core.types import Direction, MarketContext, Series, Tick, Timeframe
from runner.service import JarvisRunner

NOW = datetime(2026, 8, 6, 12, 30, tzinfo=UTC)
#: One ATR per bar, so "0.5 ATR against" is half a bar's range.
STEP = 0.0010


def runner(**overrides):  # type: ignore[no-untyped-def]
    base = load_settings(env_overrides=False)
    confluence = base.analysis.confluence.model_copy(
        update={"require_entry_confirmation": True, **overrides}
    )
    analysis = base.analysis.model_copy(update={"confluence": confluence})
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = base.model_copy(update={"analysis": analysis})  # type: ignore[assignment]
    return instance


def context(drift_per_bar: float, bars: int = 80) -> MarketContext:
    """M5 bars each ranging one STEP, closing `drift_per_bar` higher each time."""
    index = pd.date_range("2026-08-06", periods=bars, freq="5min", tz=UTC)
    closes = 1.08500 + np.arange(bars) * drift_per_bar
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + STEP / 2,
            "low": closes - STEP / 2,
            "close": closes,
            "tick_volume": 100,
            "spread": 12,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol="GBPJPY.i",
        now=NOW,
        series={
            Timeframe.M5: Series(
                symbol="GBPJPY.i", timeframe=Timeframe.M5, df=frame, fetched_at=NOW
            )
        },
        tick=Tick(symbol="GBPJPY.i", time=NOW, bid=closes[-1], ask=closes[-1] + 0.00004),
    )


def idea(direction: Direction) -> TradeIdea:
    entry = 1.08500
    sign = int(direction)
    return TradeIdea(
        symbol="GBPJPY.i",
        approved=True,
        direction=direction,
        score=70.0,
        confidence=0.7,
        entry=entry,
        stop_loss=entry - 0.0012 * sign,
        take_profit=entry + 0.0024 * sign,
        reason="test",
        signals=(),
    )


class TestTheLiveCase:
    def test_a_short_into_a_rising_market_waits(self) -> None:
        """GBPJPY: resistance breaking upward, and the system wanted to sell it."""
        rising = context(drift_per_bar=STEP * 0.4)  # 1.2 ATR over three bars
        ok, adverse = runner()._entry_is_confirmed(rising, idea(Direction.SHORT))

        assert not ok
        assert adverse == pytest.approx(1.2, abs=0.05)

    def test_a_long_into_the_same_rise_is_fine(self) -> None:
        """The move is *for* the trade. Nothing to wait for."""
        rising = context(drift_per_bar=STEP * 0.4)
        ok, adverse = runner()._entry_is_confirmed(rising, idea(Direction.LONG))

        assert ok
        assert adverse == pytest.approx(-1.2, abs=0.05)

    def test_a_short_into_a_falling_market_is_fine(self) -> None:
        falling = context(drift_per_bar=-STEP * 0.4)
        assert runner()._entry_is_confirmed(falling, idea(Direction.SHORT))[0]


class TestItIsNotAWall:
    def test_ordinary_wobble_does_not_block(self) -> None:
        """A tenth of a bar per bar is noise, not the level failing."""
        drifting = context(drift_per_bar=STEP * 0.05)
        assert runner()._entry_is_confirmed(drifting, idea(Direction.SHORT))[0]

    def test_a_flat_market_never_blocks(self) -> None:
        assert runner()._entry_is_confirmed(context(0.0), idea(Direction.SHORT))[0]

    def test_the_boundary_is_where_the_setting_puts_it(self) -> None:
        """0.5 ATR over three bars: 0.16 per bar passes, 0.20 does not."""
        service = runner()
        assert service._entry_is_confirmed(context(STEP * 0.16), idea(Direction.SHORT))[0]
        assert not service._entry_is_confirmed(context(STEP * 0.20), idea(Direction.SHORT))[0]

    def test_a_looser_limit_lets_more_through(self) -> None:
        rising = context(drift_per_bar=STEP * 0.4)
        assert not runner()._entry_is_confirmed(rising, idea(Direction.SHORT))[0]
        assert runner(confirmation_max_adverse_atr=2.0)._entry_is_confirmed(
            rising, idea(Direction.SHORT)
        )[0]

    def test_switching_it_off_restores_immediate_entry(self) -> None:
        rising = context(drift_per_bar=STEP * 0.4)
        ok, adverse = runner(require_entry_confirmation=False)._entry_is_confirmed(
            rising, idea(Direction.SHORT)
        )
        assert ok
        assert adverse is None


class TestItFailsOpen:
    def test_a_missing_timeframe_does_not_block(self) -> None:
        """Not evidence that price is running against the trade, and every
        analysis gate that judged this setup has already passed."""
        bare = MarketContext(symbol="GBPJPY.i", now=NOW, series={}, tick=None)
        ok, adverse = runner()._entry_is_confirmed(bare, idea(Direction.SHORT))
        assert ok
        assert adverse is None

    def test_too_few_bars_does_not_block(self) -> None:
        ok, adverse = runner()._entry_is_confirmed(context(0.0, bars=3), idea(Direction.SHORT))
        assert ok
        assert adverse is None

    def test_a_missing_direction_does_not_block(self) -> None:
        blank = TradeIdea(
            symbol="GBPJPY.i",
            approved=False,
            direction=None,
            score=0.0,
            confidence=0.0,
            entry=1.085,
            stop_loss=1.084,
            take_profit=1.087,
            reason="none",
            signals=(),
        )
        assert runner()._entry_is_confirmed(context(STEP), blank)[0]


def test_a_refusal_is_temporary_by_construction() -> None:
    """The same setup passes as soon as the adverse move stops.

    Nothing is remembered and nothing is discarded — which is the point. A
    setup that is right but early is taken on the cycle after the market stops
    arguing with it.
    """
    service = runner()
    short = idea(Direction.SHORT)

    assert not service._entry_is_confirmed(context(STEP * 0.4), short)[0]
    assert service._entry_is_confirmed(context(0.0), short)[0]
