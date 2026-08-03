"""Pure helpers for presenting the request/response AI audit stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def pair_ai_reviews(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    """Pair durable request/response rows by cycle, newest exchange first."""
    exchanges: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        cycle_id = str(row.get("cycle_id", "")).strip()
        if not cycle_id:
            continue
        if cycle_id not in exchanges:
            exchanges[cycle_id] = {
                "cycle_id": cycle_id,
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "requested_at": None,
                "responded_at": None,
                "latency_ms": None,
                "request": {},
                "decision": {},
                "status": "PENDING",
            }
            order.append(cycle_id)
        exchange = exchanges[cycle_id]
        event = row.get("event")
        if event == "pretrade_request":
            exchange["requested_at"] = row.get("timestamp")
            exchange["request"] = row.get("request") or {}
        elif event in {"pretrade_response", "pretrade_review"}:
            exchange["responded_at"] = row.get("timestamp")
            exchange["latency_ms"] = row.get("latency_ms")
            exchange["decision"] = row.get("decision") or {}
            if event == "pretrade_review" and not exchange["request"]:
                exchange["request"] = {"executable_proposal": row.get("proposal") or {}}

    paired = []
    for cycle_id in reversed(order):
        exchange = exchanges[cycle_id]
        decision = exchange["decision"] if isinstance(exchange["decision"], Mapping) else {}
        if decision:
            if decision.get("error"):
                exchange["status"] = "ERROR / FAIL CLOSED"
            elif decision.get("approved"):
                exchange["status"] = "APPROVED"
            else:
                exchange["status"] = "VETO"
        paired.append(exchange)
    return paired


def supervision_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    """Extract the open-position management stream, newest first.

    Kept separate from `pair_ai_reviews` because it is a different question with
    a different shape: those rows are a request paired with a verdict on whether
    to open, these are one self-contained record of what was decided about a
    position that is already running.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event") != "position_supervision":
            continue
        decision = row.get("decision")
        decision = decision if isinstance(decision, Mapping) else {}
        out.append(
            {
                "at": row.get("timestamp"),
                "ticket": row.get("ticket"),
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "action": decision.get("action", "?"),
                "confidence": decision.get("confidence"),
                "reason": decision.get("reason") or decision.get("error"),
                "latency_ms": row.get("latency_ms"),
                "error": decision.get("error", ""),
            }
        )
    out.reverse()
    return out
