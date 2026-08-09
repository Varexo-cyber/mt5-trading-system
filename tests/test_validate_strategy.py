"""Unit coverage for the isolated strategy-validation entry point."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backtesting.engine import BacktestOrder
from config.loader import load_settings
from core.types import Direction
from scripts.validate_strategy import directional_overlap, engine, nominal_parameters, variants


@pytest.mark.parametrize(
    ("module", "variant_count"),
    [
        ("market_structure", 9),
        ("trend_momentum", 27),
        ("liquidity_sweep", 9),
    ],
)
def test_each_live_directional_module_has_a_declared_stability_sweep(
    module: str, variant_count: int
) -> None:
    neighbourhood = variants(module)

    assert len(neighbourhood) == variant_count
    assert nominal_parameters(module) in neighbourhood


@pytest.mark.parametrize("module", ["market_structure", "trend_momentum", "liquidity_sweep"])
def test_validation_engine_isolates_exactly_one_directional_module(module: str) -> None:
    settings = load_settings(env_overrides=False)

    isolated = engine(settings, module=module)

    positive_weights = {name for name, weight in isolated.config.weights.items() if weight > 0}
    assert positive_weights == {module}
    assert [candidate.name for candidate in isolated.modules] == [
        "market_structure",
        "trend_momentum",
        "liquidity_sweep",
        "level_reaction",
        "volatility_regime",
    ]


def test_unknown_validation_module_fails_closed() -> None:
    settings = load_settings(env_overrides=False)

    with pytest.raises(ValueError, match="unsupported validation module"):
        engine(settings, module="unknown")


def test_sweeps_match_the_pre_registered_parameter_boundaries() -> None:
    trend = variants("trend_momentum")
    liquidity = variants("liquidity_sweep")

    assert {row["fast_ema"] for row in trend} == {15, 20, 25}
    assert {row["slow_ema"] for row in trend} == {40, 50, 60}
    assert {row["slope_lookback"] for row in trend} == {3, 5, 8}
    assert {row["range_lookback"] for row in liquidity} == {15, 20, 30}
    assert {row["minimum_depth_atr"] for row in liquidity} == {0.0, 0.3, 0.5}


def _order(at: datetime, direction: Direction = Direction.LONG) -> BacktestOrder:
    return BacktestOrder("EURUSD.i", at, direction, 1.1, 1.0, 1.3)


def test_directional_overlap_counts_only_same_time_and_direction() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    left = [_order(now), _order(now + timedelta(hours=1))]
    right = [_order(now), _order(now + timedelta(hours=1), Direction.SHORT)]

    assert directional_overlap(left, right) == 0.5
    assert directional_overlap([], right) is None
