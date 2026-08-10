"""The best setups must reach Claude first, not merely the earliest ones.

With ~200 markets analysed per cycle and two position slots, "acceptable and
scanned early" is a far weaker filter than "best available". The loop used to
take candidates in scanner order and stop when the slots filled, so the scarce
slots went to whatever the scanner's cheap trend/activity heuristic happened to
rank first — a number that knows nothing about whether the setup is any good.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from analysis.confluence import TradeIdea
from core.types import Direction, MarketContext
from runner.service import AnalysedCandidate

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def candidate(
    symbol: str,
    score: float,
    confidence: float,
    *,
    tier: int = 0,
    cost_priority: float = 0.0,
) -> AnalysedCandidate:
    idea = TradeIdea(
        symbol=symbol,
        approved=True,
        direction=Direction.LONG,
        score=score,
        confidence=confidence,
        entry=1.10,
        stop_loss=1.09,
        take_profit=1.12,
        reason="test",
        signals=(),
    )
    return AnalysedCandidate(
        symbol,
        "cycle",
        idea,
        MarketContext(symbol=symbol, now=NOW, series={}, tick=None),
        market_priority_tier=tier,
        cost_priority=cost_priority,
    )


def ranked(items: list[AnalysedCandidate]) -> list[str]:
    return [item.symbol for item in sorted(items, key=lambda x: x.conviction, reverse=True)]


def test_conviction_combines_score_and_confidence() -> None:
    """Either one being weak has to weaken the whole, so they multiply."""
    assert candidate("A", 80.0, 0.5).conviction == pytest.approx(40.0)
    assert candidate("B", 60.0, 0.9).conviction == pytest.approx(54.0)


def test_the_strongest_setup_goes_first() -> None:
    items = [
        candidate("EARLY_BUT_WEAK", 62.0, 0.55),
        candidate("BEST", 85.0, 0.9),
        candidate("MIDDLING", 70.0, 0.7),
    ]
    assert ranked(items) == ["BEST", "MIDDLING", "EARLY_BUT_WEAK"]


def test_a_confident_moderate_setup_beats_an_unsure_strong_one() -> None:
    """A high score from modules that are individually unsure is a weaker claim."""
    items = [candidate("UNSURE_STRONG", 90.0, 0.5), candidate("CONFIDENT", 70.0, 0.85)]
    assert ranked(items) == ["CONFIDENT", "UNSURE_STRONG"]


def test_scanner_order_does_not_survive_ranking() -> None:
    """The regression this exists to prevent."""
    items = [candidate(f"S{index}", 60.0 + index, 0.6 + index / 100) for index in range(5)]
    assert ranked(items) == ["S4", "S3", "S2", "S1", "S0"]


def test_ranking_is_not_a_gate() -> None:
    """Low conviction still gets its turn; `score_threshold` decides tradeability."""
    items = [candidate("STRONG", 90.0, 0.9), candidate("WEAK", 61.0, 0.56)]
    assert len(ranked(items)) == 2
    assert "WEAK" in ranked(items)


def test_preferred_market_lane_precedes_a_stronger_fallback_setup() -> None:
    preferred = candidate("EURUSD.i", 55.0, 0.70, tier=1)
    fallback = candidate("DB1", 90.0, 0.90, tier=0)

    ordered = sorted([fallback, preferred], key=lambda item: item.selection_key, reverse=True)

    assert [item.symbol for item in ordered] == ["EURUSD.i", "DB1"]


def test_lower_spread_can_break_a_close_comparison_inside_one_lane() -> None:
    cheaper = candidate("EURUSD.i", 70.0, 0.70, tier=1, cost_priority=9.0)
    expensive = candidate("GBPUSD.i", 75.0, 0.70, tier=1, cost_priority=1.0)

    ordered = sorted([expensive, cheaper], key=lambda item: item.selection_key, reverse=True)

    assert [item.symbol for item in ordered] == ["EURUSD.i", "GBPUSD.i"]
