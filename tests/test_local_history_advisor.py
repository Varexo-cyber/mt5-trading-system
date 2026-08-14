from __future__ import annotations

import json
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
    assert advice.model == "claude_archive"
    assert "5 comparable reviews" in advice.thesis
    assert not advice.usage


def test_unknown_setup_passes_to_deterministic_jarvis(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _ledger(history, 4)
    adviser = LocalHistoryAdvisor(_config(), history)

    advice = adviser.review(_idea(), _context())

    assert advice.approved
    assert advice.said_yes
    assert "5 are required" in advice.thesis


def test_factory_builds_local_adviser_without_api_credentials(tmp_path: Path) -> None:
    history = tmp_path / "missing.jsonl"

    adviser = build_advisor(_config(), history_path=history)

    assert isinstance(adviser, LocalHistoryAdvisor)
    assert adviser.examples == ()
    assert not adviser.uses_paid_api


def test_local_verdicts_never_reinforce_their_own_archive(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    _ledger(history, 20, provider="local_history")

    adviser = LocalHistoryAdvisor(_config(), history)

    assert adviser.examples == ()
    assert adviser.review(_idea(), _context()).approved
