"""The second brain improves ordering without becoming another trade gate."""

from __future__ import annotations

import pytest

from brain.selection import combine_selection_evidence, score_band
from brain.store import Brain, SelectionEvidence


def row(
    dimension: str,
    value: str,
    modifier: float,
    *,
    direction: str = "LONG",
    trades: int = 30,
) -> SelectionEvidence:
    return SelectionEvidence(dimension, value, direction, trades, 20, 0.2, modifier)


WEIGHTS = {
    "setup_horizon": 1.5,
    "setup_family": 1.0,
    "detector": 1.0,
    "regime": 1.0,
    "session": 0.8,
    "score_band": 0.8,
    "asset_class": 0.6,
    "horizon": 0.6,
    "direction": 0.5,
}


def verdict(evidence: list[SelectionEvidence], detectors: tuple[str, ...] = ("fast",)):
    return combine_selection_evidence(
        evidence,
        asset_class="forex",
        setup_family="pullback",
        horizon="quick",
        direction="LONG",
        regime="trend_up",
        session="london",
        score=47.0,
        detectors=detectors,
        weights=WEIGHTS,
        strength=1.5,
        cap=6.0,
    )


def test_no_history_is_exactly_neutral_and_never_a_gate() -> None:
    result = verdict([])

    assert result.modifier == 0.0
    assert result.reasons == ()
    assert "neutral" in result.summary()


def test_matching_context_combines_into_a_bounded_explained_nudge() -> None:
    result = verdict(
        [
            row("setup_horizon", "pullback|quick", 3.0),
            row("regime", "trend_up", 2.0),
            row("session", "london", 1.0),
            row("score_band", "40", -1.0),
            row("direction", "*", 0.5),
        ]
    )

    assert 0.0 < result.modifier <= 6.0
    assert "setup_horizon" in result.matched_dimensions
    assert any("setup_horizon=pullback|quick" in reason for reason in result.reasons)


def test_multiple_correlated_detectors_get_one_averaged_voice() -> None:
    result = verdict(
        [row("detector", "fast", 4.0), row("detector", "trend", 2.0)],
        detectors=("fast", "trend"),
    )

    # Average 3.0, then configured ensemble strength 1.5.  Summing both would
    # hit the 6 point cap and would incorrectly treat one trade as two samples.
    assert result.modifier == pytest.approx(4.5)
    assert result.matched_dimensions == ("detector",)


def test_long_history_never_leaks_into_a_short_candidate() -> None:
    result = combine_selection_evidence(
        [row("regime", "trend_up", 4.0, direction="LONG")],
        asset_class="forex",
        setup_family="pullback",
        horizon="quick",
        direction="SHORT",
        regime="trend_up",
        session="london",
        score=47.0,
        detectors=(),
        weights=WEIGHTS,
        strength=1.5,
        cap=6.0,
    )

    assert result.modifier == 0.0


def test_score_band_matches_the_database_bucket() -> None:
    assert score_band(0.0) == "0"
    assert score_band(39.9) == "30"
    assert score_band(90.0) == "90"


def test_database_query_uses_only_closed_broker_trades_and_supporting_detectors() -> None:
    brain = Brain("", account="5049535")
    captured: dict[str, object] = {}

    def run(sql, params, **_kwargs):  # type: ignore[no-untyped-def]
        captured.update({"sql": sql, "params": params})
        return [("session", "london", "LONG", 15, 11, 0.2)]

    brain._run = run  # type: ignore[method-assign]
    result = brain.selection_evidence(
        minimum_trades=15,
        shrinkage_trades=25,
        points_per_r=15.0,
        modifier_cap=4.0,
        outcome_floor_r=-1.0,
        outcome_cap_r=2.0,
    )

    sql = str(captured["sql"])
    assert "t.closed_at IS NOT NULL" in sql
    assert "t.pnl_r IS NOT NULL" in sql
    assert "signal->>'direction'" in sql
    assert "NULLIF(d.setup_family, '')" in sql
    assert captured["params"] == (-1.0, 2.0, "5049535", 15)
    assert result[0].modifier == pytest.approx(1.125)
