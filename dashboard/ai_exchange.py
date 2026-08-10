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
            elif decision.get("entry_timing") == "WAIT_RETEST":
                # Directional agreement is not a confidence failure. Claude
                # liked the thesis but refused this exact market price, and
                # Jarvis will offer the changed setup again after a retest.
                exchange["status"] = "WAITING FOR RETEST"
            elif decision.get("said_yes"):
                # Approved on the merits and refused for want of conviction.
                # Reporting these as VETO made a screen of them read as "Claude
                # rejects every setup", when the real question is whether the
                # confidence threshold sits where it should. Different problem,
                # different fix, so it gets its own label.
                exchange["status"] = "UNDER THRESHOLD"
            else:
                exchange["status"] = "VETO"
        paired.append(exchange)
    return paired


def read_posture(path: Any) -> dict[str, Any]:
    """The trading posture the last cycle recorded, or empty if unknown.

    Read from the heartbeat rather than recomputed, so the dashboard reports
    what the trader actually acted on rather than a second opinion that could
    disagree with it.
    """
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return {}
    stance = payload.get("posture") if isinstance(payload, Mapping) else None
    return dict(stance) if isinstance(stance, Mapping) else {}


def read_block_reason(path: Any) -> dict[str, str]:
    """Why new risk is refused, or empty when trading is permitted.

    Read from the heartbeat rather than recomputed, so the deck reports the
    decision the trader actually acted on instead of a second opinion that
    could disagree with it.
    """
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    reason = str(payload.get("blocked_reason", "") or "")
    if not reason:
        return {}
    return {"reason": reason, "detail": str(payload.get("blocked_detail", "") or "")}


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


#: USD per million tokens, by model. Input, output, and the two cache rates
#: (write is 1.25x input for the 5-minute TTL, read is 0.1x).
#:
#: Hardcoded rather than fetched: this is a dashboard estimate, and a pricing
#: table that silently fails to load would show a plausible wrong number, which
#: is worse than a stale one. An unknown model reports no cost rather than
#: guessing at somebody else's rate.
_PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def call_cost(model: str, usage: Mapping[str, Any]) -> float | None:
    """Estimated USD for one call, or None when the model is not priced here."""
    price = _PRICES.get(str(model))
    if price is None or not usage:
        return None
    per_in, per_out = price
    fresh = float(usage.get("input_tokens", 0) or 0)
    written = float(usage.get("cache_creation_input_tokens", 0) or 0)
    read = float(usage.get("cache_read_input_tokens", 0) or 0)
    out = float(usage.get("output_tokens", 0) or 0)
    return (
        fresh * per_in + written * per_in * 1.25 + read * per_in * 0.1 + out * per_out
    ) / 1_000_000


def spend_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """What the adviser has cost, and how much of it the cache absorbed.

    "Am I burning credit" had no answer anywhere in the interface. The cache
    hit rate is the part worth watching: if it collapses, something has started
    varying the system prompt and every call is paying full price for a prefix
    that used to be free.
    """
    calls = priced = replayed = 0
    cost = 0.0
    fresh = written = read = out = 0
    for row in rows:
        decision = row.get("decision")
        if not isinstance(decision, Mapping):
            continue
        usage = decision.get("usage")
        if not isinstance(usage, Mapping) or not usage:
            continue
        # A replayed verdict keeps the original call's token counts, because
        # the audit trail should still show which call it came from. Billing
        # them again is the bug this skips. A live deck read nine calls and
        # $0.41 where five were paid, and averaged the per-call figure over
        # four rows that never touched the network — their latencies gave them
        # away at 1 to 3 milliseconds against 12 to 34 seconds for a real one.
        if decision.get("replayed"):
            replayed += 1
            continue
        calls += 1
        fresh += int(usage.get("input_tokens", 0) or 0)
        written += int(usage.get("cache_creation_input_tokens", 0) or 0)
        read += int(usage.get("cache_read_input_tokens", 0) or 0)
        out += int(usage.get("output_tokens", 0) or 0)
        amount = call_cost(str(decision.get("model", "")), usage)
        if amount is not None:
            cost += amount
            priced += 1
    total_in = fresh + written + read
    return {
        "calls": calls,
        "priced_calls": priced,
        #: Verdicts served from the review cache: real decisions, no API call.
        "replayed_calls": replayed,
        "usd": round(cost, 4),
        "usd_per_call": round(cost / priced, 4) if priced else 0.0,
        "input_tokens": total_in,
        "output_tokens": out,
        # Share of input served from cache at a tenth of the price.
        "cache_hit_rate": (read / total_in) if total_in else 0.0,
    }


#: Operation modes that analyse and advise but can never send an order.
#:
#: `monitor` is the default, which is what makes this worth naming: a session
#: started without an explicit mode looks completely healthy — markets scanned,
#: setups analysed, verdicts returned — and cannot place a trade no matter what
#: any of it says.
NON_TRADING_OPERATIONS = {"monitor"}


def read_operation(path: Any) -> str:
    """The mode the last cycle actually ran in, from the heartbeat."""
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return ""
    return str(payload.get("operation", "") or "") if isinstance(payload, Mapping) else ""


def cannot_trade(operation: str) -> bool:
    return operation in NON_TRADING_OPERATIONS
