"""Market-specific ordering and its evidence floor."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from analysis.confluence import TradeIdea
from analysis.market_intelligence import (
    MarketObservation,
    apply_cross_market_context,
    assess_opportunity,
)
from brain.store import Brain
from config.schema import AssetClassRoutingConfig
from core.instrument import AssetClass
from core.types import Direction, Signal

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def idea(direction: Direction = Direction.LONG) -> TradeIdea:
    return TradeIdea(
        "EURUSD.i",
        True,
        direction,
        70.0,
        0.8,
        1.1,
        1.09 if direction is Direction.LONG else 1.11,
        1.12 if direction is Direction.LONG else 1.08,
        "trend",
        (Signal("trend_momentum", 70.0 * int(direction), 0.8, "aligned"),),
        setup_family="trend_momentum_swing",
        horizon="swing",
    )


def observation() -> MarketObservation:
    return MarketObservation(
        "EURUSD.i", "forex", "trend_up", 1.0, 1.0, 0.4, 3, 0.5, NOW.isoformat()
    )


def test_forex_module_affinity_changes_ordering_not_eligibility() -> None:
    policy = AssetClassRoutingConfig(module_affinity={"trend_momentum": 3.0})
    intelligence = assess_opportunity(
        idea(), observation(), AssetClass.FOREX, cap=20.0, routing=policy
    )

    assert intelligence.modifier > 3.0
    assert any("forex routing" in reason for reason in intelligence.reasons)
    assert idea().approved


def test_relative_currency_strength_is_symmetric_for_shorts() -> None:
    policy = AssetClassRoutingConfig(cross_market_bonus=4.0)
    short = idea(Direction.SHORT)
    initial = assess_opportunity(short, observation(), AssetClass.FOREX, cap=20.0, routing=policy)
    world = {"strongest_currencies": ["USD"], "weakest_currencies": ["EUR"]}

    routed = apply_cross_market_context(
        initial,
        short,
        observation(),
        AssetClass.FOREX,
        world,
        routing=policy,
        cap=20.0,
    )

    assert routed.modifier == pytest.approx(initial.modifier + 4.0)
    assert any("currency strength confirms" in reason for reason in routed.reasons)


def test_realised_calibration_is_shrunk_and_bounded() -> None:
    brain = Brain("", account="test")
    brain._run = lambda *_args, **_kwargs: [  # type: ignore[method-assign]
        ("forex", "trend_momentum_swing", "swing", "SHORT", "trend_down", 40, 1.0, 3)
    ]

    estimates = brain.edge_calibrations(
        minimum_trades=40,
        shrinkage_trades=80,
        points_per_r=6.0,
        modifier_cap=4.0,
    )

    assert len(estimates) == 1
    assert estimates[0].modifier == pytest.approx(2.0)
    assert estimates[0].trades == 40


def test_thin_calibration_returns_no_authority() -> None:
    brain = Brain("", account="test")
    brain._run = lambda *_args, **_kwargs: []  # type: ignore[method-assign]

    assert (
        brain.edge_calibrations(
            minimum_trades=40,
            shrinkage_trades=80,
            points_per_r=6.0,
            modifier_cap=4.0,
        )
        == []
    )


def test_counterfactual_history_is_sent_to_neon_in_one_batch() -> None:
    brain = Brain("", account="5049535")
    captured: dict[str, object] = {}

    def run(sql, params, **_kwargs):  # type: ignore[no-untyped-def]
        captured.update({"sql": sql, "params": params})

    brain._run = run  # type: ignore[method-assign]
    rows = [
        {
            "symbol": symbol,
            "direction": "SHORT",
            "blocked_by": "AI_VETO",
            "opened_at": NOW,
            "entry": 100.0,
            "stop_loss": 102.0,
            "take_profit": 96.0,
            "resolved_at": NOW,
            "outcome": "TP",
            "pnl_r": 2.0,
        }
        for symbol in ("EURUSD.i", "GBPUSD.i")
    ]

    brain.record_counterfactuals(rows)

    assert "jsonb_to_recordset" in str(captured["sql"])
    payload = json.loads(captured["params"][0])  # type: ignore[index]
    assert len(payload) == 2
    assert {row["account"] for row in payload} == {"5049535"}
    assert len({row["fingerprint"] for row in payload}) == 2


def test_decision_writes_typed_market_segment_for_future_calibration() -> None:
    brain = Brain("", account="5049535")
    captured: dict[str, object] = {}

    def run(sql, params, **_kwargs):  # type: ignore[no-untyped-def]
        captured.update({"sql": sql, "params": params})
        return (17,)

    brain._run = run  # type: ignore[method-assign]
    result = brain.record_decision(
        decided_at=NOW,
        symbol="EURUSD.i",
        direction="SHORT",
        reason="OK",
        mode="experimental_live",
        playbook="liquidity_sweep_m15",
        filters={
            "asset_class": "forex",
            "volatility_regime": "range",
            "session": "london",
            "trade_horizon": "intraday",
            "planning_timeframe": "M15",
        },
    )

    assert result == 17
    assert "asset_class, regime" in str(captured["sql"])
    params = captured["params"]
    assert params[12:17] == ("forex", "range", "london", "intraday", "M15")  # type: ignore[index]
