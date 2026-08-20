"""Market-specific ordering and its evidence floor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from analysis.confluence import TradeIdea
from analysis.market_intelligence import (
    MarketObservation,
    apply_cross_market_context,
    assess_opportunity,
)
from brain.store import Brain
from config.loader import DEFAULT_CONFIG_PATH, load_settings
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


class TestTheCalibrationCanActuallyReachThisAccount:
    """A learner that runs and changes no ordering has not learned anything.

    The shipped defaults are calibrated for a system trading hundreds of times
    a month. On forty-seven trades they failed twice over: no segment ever met
    the forty-trade floor, and had one met it the modifier would have been
    -0.39 points against confluence scores spanning 38.8 to 58.8 — one and a
    half percent of the range, invisible.

    The account's loudest fact is that it is worse at longs than shorts
    (-6.95R over 27 against -2.02R over 20). These pin that the machinery can
    now express it.
    """

    def modifier(self, settings, trades: int, mean_r: float) -> float:
        learning = settings.learning
        shrink = trades / (trades + learning.selection_shrinkage_trades)
        raw = mean_r * shrink * learning.selection_points_per_r
        cap = learning.selection_modifier_cap
        return max(-cap, min(cap, raw))

    @pytest.fixture
    def live(self):  # type: ignore[no-untyped-def]
        return load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

    def test_the_floor_is_reachable_at_this_volume(self, live) -> None:  # type: ignore[no-untyped-def]
        assert live.learning.selection_min_trades <= 20

    def test_the_direction_split_this_account_shows_is_visible(self, live) -> None:  # type: ignore[no-untyped-def]
        """The gap has to be worth more than rounding against a 20-point spread."""
        longs = self.modifier(live, 27, -6.95 / 27)
        shorts = self.modifier(live, 20, -2.02 / 20)

        assert longs < shorts, "the worse direction must rank below the better one"
        assert shorts - longs > 1.0, f"gap of {shorts - longs:.2f} points is still noise"

    def test_it_stays_a_nudge_and_never_a_takeover(self, live) -> None:
        """Even a segment losing a full R per trade is bounded by the cap."""
        ruinous = self.modifier(live, 500, -1.0)

        assert ruinous >= -live.learning.selection_modifier_cap
        assert live.learning.selection_modifier_cap <= 5.0

    def test_a_thin_segment_still_barely_whispers(self, live) -> None:
        """Shrinkage is the protection, not the floor.

        Ten trades earns a proportionally tiny voice rather than nothing
        followed, one trade later, by a full one.
        """
        thin = self.modifier(live, 10, -6.95 / 27)
        thick = self.modifier(live, 200, -6.95 / 27)

        assert abs(thin) < abs(thick) / 2

    def test_the_ladder_ends_at_direction_alone(self) -> None:
        """The only rung a backfilled trade can match.

        A trade copied from the local journal has no decision behind it, so its
        asset class reads 'unknown' and it can never match a live 'forex'
        candidate in the finer buckets. Without this rung the ladder stops
        above the only evidence a small account has.
        """
        store = (Path(__file__).resolve().parent.parent / "brain" / "store.py").read_text(
            encoding="utf-8"
        )
        assert "SELECT 'direction', '*', direction" in store
        selector = (Path(__file__).resolve().parent.parent / "brain" / "selection.py").read_text(
            encoding="utf-8"
        )
        assert '"direction": "*"' in selector, "the broad direction facet must be matched"
