from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from advisory.local_history import LocalHistoryAdvisor
from advisory.providers import build_advisor, build_review_payload
from analysis.confluence import TradeIdea
from config.schema import AIConfig
from core.types import Direction, MarketContext, Signal


def _idea() -> TradeIdea:
    signal = Signal("fast_ema_cross", 50.0, 0.7, "M5 cross")
    return TradeIdea(
        "EURUSD.i",
        True,
        Direction.SHORT,
        35.0,
        0.7,
        1.1,
        1.102,
        1.097,
        "quick short",
        (signal,),
    )


def _context() -> MarketContext:
    return MarketContext("EURUSD.i", datetime(2026, 8, 14, tzinfo=UTC), {})


def _ledger(
    path: Path,
    count: int,
    *,
    approved: bool = False,
    provider: str = "anthropic",
) -> None:
    payload = build_review_payload(_idea(), _context(), None)
    rows: list[dict[str, object]] = []
    for index in range(count):
        common = {
            "cycle_id": f"cycle-{index}",
            "symbol": "EURUSD.i",
            "direction": "SHORT",
        }
        rows.append({**common, "event": "pretrade_request", "request": payload})
        rows.append(
            {
                **common,
                "event": "pretrade_response",
                "decision": {
                    "approved": approved,
                    "said_yes": approved,
                    "confidence": 0.8,
                    "thesis": "same historical chart pattern",
                    "provider": provider,
                    "model": "claude-sonnet-5",
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _position_state() -> dict[str, object]:
    return {
        "symbol": "EURUSD.i",
        "direction": "SHORT",
        "unrealised_r": -0.30,
        "peak_unrealised_r": 0.05,
        "profit_given_back_fraction": 1.0,
        "age_hours": 0.75,
        "spread_as_fraction_of_initial_risk": 0.04,
        "distance_to_stop_in_atr": 0.42,
        "distance_to_target_in_atr": 2.1,
        "unrealised_pct_of_account": -0.5,
        "context": {
            "mechanical_health": {
                "verdict": "broken",
                "severity": 0.9,
                "action": "exit",
                "signals": ["structure_broken", "momentum_turned"],
            }
        },
    }


def _supervision_ledger(
    path: Path,
    count: int,
    *,
    action: str = "close",
    provider: str = "anthropic",
) -> None:
    rows = [
        {
            "event": "position_supervision",
            "ticket": index,
            "symbol": "EURUSD.i",
            "direction": "SHORT",
            "request": _position_state(),
            "decision": {
                "action": action,
                "confidence": 0.8,
                "reason": "structure broke and momentum turned against the position",
                "provider": provider,
                "model": "claude-sonnet-5",
            },
        }
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with sqlite3.connect(path.parent / "trading.db") as connection:
        connection.execute("CREATE TABLE trades (ticket INTEGER, closed_at TEXT, pnl_r REAL)")
        connection.executemany(
            "INSERT INTO trades (ticket, closed_at, pnl_r) VALUES (?, ?, ?)",
            [
                (
                    index,
                    "2026-08-14T12:00:00+00:00",
                    -0.8 if action == "close" else 0.4,
                )
                for index in range(count)
            ],
        )


def _config(**changes: object) -> AIConfig:
    values: dict[str, object] = {
        "enabled": True,
        "provider": "local_history",
        "minimum_confidence": 0.45,
        "local_history_min_neighbors": 5,
        "local_history_max_distance": 0.55,
        "local_history_veto_rate": 0.8,
    }
    values.update(changes)
    return AIConfig(**values)


def test_local_history_repeats_only_a_supported_historical_veto(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _ledger(history, 5)
    adviser = LocalHistoryAdvisor(_config(), history)

    advice = adviser.review(_idea(), _context())

    assert not advice.approved
    assert advice.provider == "local_history"
    assert advice.model == "jarvis_outcome_memory"
    assert "5 comparable reviewer opinions" in advice.thesis
    assert "Jarvis independently formed" in advice.thesis
    assert not advice.usage


def test_unknown_setup_passes_to_deterministic_jarvis(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _ledger(history, 4)
    adviser = LocalHistoryAdvisor(_config(), history)

    advice = adviser.review(_idea(), _context())

    assert advice.approved
    assert advice.said_yes
    assert "5 are required" in advice.thesis
    assert "Jarvis independently formed" in advice.thesis


def test_entry_explanation_hot_loads_jarvis_own_realised_record(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _ledger(history, 5, approved=True)
    adviser = LocalHistoryAdvisor(_config(), history)
    assert "no matching symbol/direction trade" in adviser.review(_idea(), _context()).thesis

    with sqlite3.connect(tmp_path / "trading.db") as connection:
        connection.execute(
            "CREATE TABLE trades (symbol TEXT, direction TEXT, closed_at TEXT, pnl_r REAL)"
        )
        connection.executemany(
            "INSERT INTO trades VALUES (?, ?, ?, ?)",
            [
                ("EURUSD.i", "SHORT", "2026-08-14T12:00:00+00:00", result)
                for result in (0.5, 0.3, -0.2)
            ],
        )

    advice = adviser.review(_idea(), _context())

    assert "Its own EURUSD.i SHORT record is 3 trades" in advice.thesis
    assert "67% wins" in advice.thesis
    assert "+0.60R total" in advice.thesis


def test_factory_builds_local_adviser_without_api_credentials(tmp_path: Path) -> None:
    history = tmp_path / "missing.jsonl"

    adviser = build_advisor(_config(), history_path=history)

    assert isinstance(adviser, LocalHistoryAdvisor)
    assert adviser.examples == ()
    assert adviser.supervision_examples == ()
    assert not adviser.uses_paid_api
    assert adviser.supports_dynamic_management


def test_local_verdicts_never_reinforce_their_own_archive(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _ledger(history, 20, provider="local_history")

    adviser = LocalHistoryAdvisor(_config(), history)

    assert adviser.examples == ()
    assert adviser.review(_idea(), _context()).approved


def test_local_position_ai_replays_a_strong_historical_close_pattern(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _supervision_ledger(history, 5)
    adviser = LocalHistoryAdvisor(_config(), history)

    verdict = adviser.supervise(_position_state())

    assert verdict.action == "close"
    assert verdict.confidence == 1.0
    assert verdict.provider == "local_history"
    assert "5 closest outcome-graded close states" in verdict.reason
    assert not verdict.usage


def test_unknown_position_state_is_held_instead_of_guessed(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _supervision_ledger(history, 4)
    adviser = LocalHistoryAdvisor(_config(), history)

    verdict = adviser.supervise(_position_state())

    assert verdict.action == "hold"
    assert "5 are required" in verdict.reason


def test_local_supervision_cannot_train_on_its_own_answers(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _supervision_ledger(history, 20, provider="local_history")
    adviser = LocalHistoryAdvisor(_config(), history)

    assert adviser.supervision_examples == ()
    assert adviser.supervise(_position_state()).action == "hold"


def test_a_historical_ai_answer_that_hurt_is_not_a_training_example(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _supervision_ledger(history, 5, action="hold")
    # Every hold started around -0.30R and ultimately lost a full R. Replaying
    # that advice would teach confidence, not intelligence.
    with sqlite3.connect(tmp_path / "trading.db") as connection:
        connection.execute("UPDATE trades SET pnl_r = -1.0")

    adviser = LocalHistoryAdvisor(_config(), history)

    assert adviser.supervision_examples == ()
    assert adviser.supervise(_position_state()).action == "hold"
    assert "0 comparable Claude states" in adviser.supervise(_position_state()).reason


def test_position_ai_learns_a_newly_completed_outcome_without_restart(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _supervision_ledger(history, 5, action="close")
    with sqlite3.connect(tmp_path / "trading.db") as connection:
        connection.execute("UPDATE trades SET pnl_r = 0.4")
    adviser = LocalHistoryAdvisor(_config(), history)
    assert adviser.supervision_examples == ()

    with sqlite3.connect(tmp_path / "trading.db") as connection:
        connection.execute("UPDATE trades SET pnl_r = -0.8")

    assert adviser.supervise(_position_state()).action == "close"
    assert len(adviser.supervision_examples) == 5
