"""Module analysis is recomputed only when a new bar has closed.

The live journal showed one symbol reporting confluence score 38.8 eighty-two
times in twelve hours, 39.0 fifty-eight times, and so on: 308 identical
evaluations of five numbers. The score can only move when a bar closes, and the
fastest frame in the ladder is M5, so eleven of every twelve of those recomputed
something that could not have changed — for every symbol in an 800-instrument
catalogue, on one vCPU shared with the one-second position guard.

THE CLAIM THESE TESTS HAVE TO SETTLE is not "this is faster". It is that the
cached answer is the *same* answer. That holds only because every module reads
closed bars and nothing else: no `ctx.tick`, no `ctx.now`. The last test in this
file checks that property against the source, because it is the assumption the
whole optimisation rests on and it would break silently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from analysis.confluence import ConfluenceEngine
from config.schema import ConfluenceConfig
from core.types import MarketContext, Series, Signal, Tick, Timeframe, TradingMode


class CountingModule:
    """A module that reports how often it was actually asked to think."""

    def __init__(self, name: str = "market_structure", score: float = 70.0) -> None:
        self.name = name
        self.score = score
        self.calls = 0

    def analyze(self, ctx: MarketContext) -> Signal:
        self.calls += 1
        return Signal(
            module=self.name,
            score=self.score,
            confidence=0.8,
            reasoning=f"call {self.calls}",
            details={},
        )


def frame_of(bars: int, *, step: float = 0.0006) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=bars, freq="1h", tz=UTC)
    close = pd.Series([1.10 + i * step for i in range(bars)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        }
    )


def context_of(bars: int = 100, *, symbol: str = "EURUSD", bid: float = 1.1098) -> MarketContext:
    now = datetime(2026, 1, 5, 5, tzinfo=UTC)
    return MarketContext(
        symbol=symbol,
        now=now,
        series={Timeframe.H1: Series(symbol, Timeframe.H1, frame_of(bars), now)},
        tick=Tick(symbol, now, bid, bid + 0.0002),
    )


def engine_with(module: CountingModule, **overrides: object) -> ConfluenceEngine:
    config = ConfluenceConfig(
        weights={module.name: 1.0},
        live_enabled_modules=(module.name,),
        minimum_directional_modules=1,
        **overrides,  # type: ignore[arg-type]
    )
    return ConfluenceEngine([module], config)  # type: ignore[list-item]


class TestReuse:
    def test_the_same_bars_are_analysed_once(self) -> None:
        module = CountingModule()
        engine = engine_with(module)

        for _ in range(20):
            engine.evaluate(context_of(), TradingMode.PAPER)

        assert module.calls == 1, "twenty cycles over one closed bar is one analysis"

    def test_the_verdict_is_identical_not_merely_cheap(self) -> None:
        """The whole point. A cheaper different answer would be a bug."""
        module = CountingModule()
        engine = engine_with(module)

        first = engine.evaluate(context_of(), TradingMode.PAPER)
        again = engine.evaluate(context_of(), TradingMode.PAPER)

        assert again.approved == first.approved
        assert again.direction == first.direction
        assert again.score == first.score
        assert again.signals == first.signals

    def test_a_new_closed_bar_forces_a_rethink(self) -> None:
        module = CountingModule()
        engine = engine_with(module)

        engine.evaluate(context_of(100), TradingMode.PAPER)
        engine.evaluate(context_of(101), TradingMode.PAPER)

        assert module.calls == 2

    def test_each_symbol_keeps_its_own_answer(self) -> None:
        module = CountingModule()
        engine = engine_with(module)

        engine.evaluate(context_of(symbol="EURUSD"), TradingMode.PAPER)
        engine.evaluate(context_of(symbol="GBPUSD"), TradingMode.PAPER)
        engine.evaluate(context_of(symbol="EURUSD"), TradingMode.PAPER)

        assert module.calls == 2, "two symbols, one analysis each, no cross-talk"


class TestLivePriceStillCounts:
    """Only the bar-derived half is reused. The quote is read every time."""

    def test_a_moved_quote_repricess_the_entry_without_reanalysing(self) -> None:
        module = CountingModule()
        engine = engine_with(module)

        first = engine.evaluate(context_of(bid=1.1098), TradingMode.PAPER)
        moved = engine.evaluate(context_of(bid=1.1150), TradingMode.PAPER)

        assert module.calls == 1, "the bars did not change"
        assert moved.entry != first.entry, "but the executable price did"

    def test_a_missing_quote_is_still_refused(self) -> None:
        module = CountingModule()
        engine = engine_with(module)
        engine.evaluate(context_of(), TradingMode.PAPER)

        blind = MarketContext(
            symbol="EURUSD",
            now=datetime(2026, 1, 5, 5, tzinfo=UTC),
            series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame_of(100), None)},
            tick=None,
        )
        assert engine.evaluate(blind, TradingMode.PAPER).approved is False


class TestSafety:
    def test_an_empty_frame_is_never_cached(self) -> None:
        """A transient absence must not freeze a verdict that outlives it.

        Checked on the fingerprint rather than through `evaluate`: an empty
        frame cannot reach the engine in production — `DataManager` raises
        InsufficientDataError long before — and driving one through it here
        would only prove that an unsupported path stays unsupported.
        """
        engine = engine_with(CountingModule())
        now = datetime(2026, 1, 5, 5, tzinfo=UTC)
        empty = MarketContext(
            symbol="EURUSD",
            now=now,
            series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame_of(0), now)},
            tick=Tick("EURUSD", now, 1.1098, 1.1100),
        )

        assert engine._bar_fingerprint(empty) is None

    def test_the_fingerprint_covers_every_timeframe(self) -> None:
        """A new M5 bar must invalidate, even when H1 has not moved."""
        engine = engine_with(CountingModule())
        now = datetime(2026, 1, 5, 5, tzinfo=UTC)

        def ladder(m5_bars: int) -> MarketContext:
            return MarketContext(
                symbol="EURUSD",
                now=now,
                series={
                    Timeframe.H1: Series("EURUSD", Timeframe.H1, frame_of(100), now),
                    Timeframe.M5: Series("EURUSD", Timeframe.M5, frame_of(m5_bars), now),
                },
                tick=Tick("EURUSD", now, 1.1098, 1.1100),
            )

        assert engine._bar_fingerprint(ladder(50)) != engine._bar_fingerprint(ladder(51))
        assert engine._bar_fingerprint(ladder(50)) == engine._bar_fingerprint(ladder(50))

    def test_the_escape_hatch_disables_it(self) -> None:
        module = CountingModule()
        engine = engine_with(module, cache_signals_per_bar=False)

        for _ in range(5):
            engine.evaluate(context_of(), TradingMode.PAPER)

        assert module.calls == 5

    def test_it_is_on_by_default(self) -> None:
        assert ConfluenceConfig().cache_signals_per_bar is True

    def test_no_module_reads_the_tick_or_the_wall_clock(self) -> None:
        """The assumption the whole optimisation rests on.

        If a future module starts reading `ctx.tick` or `ctx.now`, its signal
        stops being a function of the closed bars and reusing it becomes a real
        bug rather than a saving. Pinned against the source because that break
        would otherwise be invisible: everything would still pass, and the
        system would quietly act on a stale reading of a live price.
        """
        source = (Path(__file__).resolve().parent.parent / "analysis" / "modules.py").read_text(
            encoding="utf-8"
        )

        assert "ctx.tick" not in source, "a module reads the live quote; the cache is now unsafe"
        assert "ctx.now" not in source, "a module reads the wall clock; the cache is now unsafe"
