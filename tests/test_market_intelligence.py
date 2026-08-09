from __future__ import annotations

from analysis.confluence import TradeIdea
from analysis.market_intelligence import MarketObservation, assess_opportunity, world_state
from core.instrument import AssetClass
from core.types import Direction, Signal


def _idea(direction: Direction = Direction.LONG) -> TradeIdea:
    return TradeIdea(
        "EURUSD.i",
        True,
        direction,
        70.0,
        0.8,
        1.1,
        1.09 if direction is Direction.LONG else 1.11,
        1.12 if direction is Direction.LONG else 1.08,
        "structural continuation",
        (Signal("market_structure", 70.0 * int(direction), 0.8, "break"),),
    )


def test_regime_context_only_ranks_and_never_changes_idea_eligibility() -> None:
    observation = MarketObservation(
        "EURUSD.i", "forex", "trend_up", 1.0, 1.4, 0.8, 3, 0.6, "2026-08-07"
    )
    idea = _idea()

    intelligence = assess_opportunity(idea, observation, AssetClass.FOREX, cap=12.0)

    assert intelligence.modifier > 0
    assert idea.approved is True
    assert idea.score == 70.0
    assert "risk" not in intelligence.asset_context


def test_countertrend_is_demoted_but_not_vetoed() -> None:
    observation = MarketObservation(
        "EURUSD.i", "forex", "trend_up", 1.0, 1.4, 0.8, 3, 0.6, "2026-08-07"
    )
    idea = _idea(Direction.SHORT)

    intelligence = assess_opportunity(idea, observation, AssetClass.FOREX, cap=12.0)

    assert intelligence.modifier < 0
    assert idea.approved is True


def test_world_state_compares_assets_and_currency_sides() -> None:
    rows = [
        MarketObservation("EURUSD.i", "forex", "trend_up", 1.0, 1.0, 0.5, 3, 0.5, "x"),
        MarketObservation("GBPUSD.i", "forex", "trend_up", 0.8, 0.8, 0.4, 3, 0.5, "x"),
        MarketObservation("US500", "index", "trend_up", 0.7, 0.5, 0.2, 2, 0.5, "x"),
        MarketObservation("BTCUSD", "crypto", "trend_up", 0.9, 0.6, 0.4, 2, 0.5, "x"),
        MarketObservation("NVDA", "stock", "trend_up", 1.1, 0.7, 0.4, 3, 0.5, "x"),
    ]

    state = world_state(rows)

    assert state["markets_observed"] == 5
    assert state["risk_tone"] == "risk_on"
    assert "USD" in state["weakest_currencies"]
