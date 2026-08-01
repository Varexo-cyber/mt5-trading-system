"""Confluence is deterministic, auditable, and live fails closed."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.confluence import ConfluenceEngine
from config.schema import ConfluenceConfig
from core.types import MarketContext, Series, Signal, Tick, Timeframe, TradingMode


class StubModule:
    def __init__(self, signal: Signal) -> None:
        self.name = signal.module
        self.signal = signal

    def analyze(self, ctx: MarketContext) -> Signal:
        return self.signal


def context() -> MarketContext:
    index = pd.date_range("2026-01-01", periods=100, freq="1h", tz=UTC)
    close = pd.Series([1.10 + index * 0.0001 for index in range(100)], index=index)
    frame = pd.DataFrame(
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
    now = datetime(2026, 1, 5, 5, tzinfo=UTC)
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
        tick=Tick("EURUSD", now, 1.1098, 1.1100),
    )


def modules() -> list[StubModule]:
    return [
        StubModule(Signal("one", 70, 0.8, invalidation_price=1.1050)),
        StubModule(Signal("two", 60, 0.7, invalidation_price=1.1060)),
        StubModule(Signal.neutral("volatility_regime")),
    ]


def config(**overrides: object) -> ConfluenceConfig:
    values = {
        "score_threshold": 40,
        "minimum_directional_modules": 2,
        "weights": {"one": 1.0, "two": 1.0, "volatility_regime": 0.0},
    }
    values.update(overrides)
    return ConfluenceConfig(**values)


def test_paper_builds_structural_trade_idea() -> None:
    idea = ConfluenceEngine(modules(), config()).evaluate(context(), TradingMode.PAPER)

    assert idea.approved
    assert idea.direction is not None and idea.direction.name == "LONG"
    assert idea.stop_loss < idea.entry < idea.take_profit
    assert (idea.take_profit - idea.entry) / (idea.entry - idea.stop_loss) == 2.0


def test_live_blocks_when_no_module_is_validated() -> None:
    idea = ConfluenceEngine(modules(), config()).evaluate(context(), TradingMode.MICRO_LIVE)

    assert not idea.approved
    assert "no modules validated for live" in idea.reason


def test_live_uses_only_explicitly_validated_modules() -> None:
    cfg = config(live_enabled_modules=("one", "two"))
    idea = ConfluenceEngine(modules(), cfg).evaluate(context(), TradingMode.MICRO_LIVE)

    assert idea.approved


def test_disagreement_blocks() -> None:
    disagreeing = modules()
    disagreeing[1] = StubModule(Signal("two", -80, 0.9, invalidation_price=1.1140))
    idea = ConfluenceEngine(disagreeing, config()).evaluate(context(), TradingMode.PAPER)

    assert not idea.approved
