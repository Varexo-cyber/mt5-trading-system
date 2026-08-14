"""Cost-free pre-trade vetoes distilled from completed Claude reviews.

This is deliberately not presented as a replacement language model. Jarvis'
deterministic analysis still creates the setup. The archive contributes only
one bounded lesson: when several genuinely similar historical questions were
all refused by Claude, do not pay to rediscover or blindly ignore that pattern.
Unknown shapes pass back to the ordinary deterministic gates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from analysis.confluence import TradeIdea
from config.schema import AIConfig
from core.types import MarketContext

from .providers import (
    Advice,
    Reflection,
    ScoutDecision,
    Supervision,
    build_review_payload,
)

_MAX_NEIGHBORS = 15


@dataclass(frozen=True, slots=True)
class _Example:
    symbol: str
    direction: str
    setup_family: str
    horizon: str
    features: dict[str, float]
    useful: bool
    confidence: float
    thesis: str
    risks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Shape:
    setup_family: str
    horizon: str
    features: dict[str, float]


class LocalHistoryAdvisor:
    """Replay broad Claude judgement without making an external API call."""

    supports_dynamic_management = False
    uses_paid_api = False

    def __init__(self, config: AIConfig, ledger_path: Path) -> None:
        self.minimum = config.minimum_confidence
        self.min_neighbors = config.local_history_min_neighbors
        self.max_distance = config.local_history_max_distance
        self.veto_rate = config.local_history_veto_rate
        self.ledger_path = ledger_path
        self.examples = _load_examples(ledger_path)

    def scout(self, _market_state: Mapping[str, object]) -> ScoutDecision:
        return ScoutDecision(
            thesis="Paid market scouting is disabled in local-history mode",
            provider="local_history",
            model="claude_archive",
        )

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> Advice:
        payload = build_review_payload(idea, context, proposal, memory)
        current = _shape(payload)
        direction = idea.direction.name if idea.direction is not None else ""
        ranked = sorted(
            (
                (_distance(current, example), example)
                for example in self.examples
                if example.direction == direction
            ),
            key=lambda item: item[0],
        )
        neighbors = [item for item in ranked[:_MAX_NEIGHBORS] if item[0] <= self.max_distance]
        if len(neighbors) < self.min_neighbors:
            return Advice(
                True,
                max(self.minimum, min(1.0, idea.confidence)),
                f"Local history found {len(neighbors)} sufficiently similar Claude reviews; "
                f"{self.min_neighbors} are required for a learned veto. Deterministic Jarvis "
                "analysis remains authoritative.",
                provider="local_history",
                model="claude_archive",
                said_yes=True,
                threshold=self.minimum,
            )

        total_weight = sum(1.0 / (0.05 + distance) for distance, _ in neighbors)
        veto_weight = sum(
            1.0 / (0.05 + distance) for distance, example in neighbors if not example.useful
        )
        learned_veto_rate = veto_weight / total_weight if total_weight else 0.0
        closest = neighbors[0][1]
        if learned_veto_rate >= self.veto_rate:
            return Advice(
                False,
                min(1.0, learned_veto_rate),
                f"Local Claude history veto: {len(neighbors)} comparable reviews produced "
                f"a weighted {learned_veto_rate:.0%} refusal rate. Closest recorded reason: "
                f"{closest.thesis or 'no thesis recorded'}",
                risks=closest.risks,
                provider="local_history",
                model="claude_archive",
                said_yes=False,
                threshold=self.minimum,
            )

        useful_rate = 1.0 - learned_veto_rate
        return Advice(
            True,
            max(self.minimum, min(1.0, useful_rate)),
            f"Local Claude history permits this setup: {len(neighbors)} comparable reviews "
            f"did not reach the {self.veto_rate:.0%} learned-veto threshold. Deterministic "
            "price, spread, sizing and broker checks still run.",
            provider="local_history",
            model="claude_archive",
            said_yes=True,
            threshold=self.minimum,
        )

    def reflect(self, _outcome: Mapping[str, object]) -> Reflection:
        return Reflection(
            "Paid post-trade reflection is disabled; realised outcome remains in Jarvis memory",
            provider="local_history",
            model="claude_archive",
        )

    def supervise(self, _position_state: Mapping[str, object]) -> Supervision:
        return Supervision(
            "hold",
            "Paid AI supervision is disabled; mechanical position management remains active",
            provider="local_history",
            model="claude_archive",
        )


def _load_examples(path: Path) -> tuple[_Example, ...]:
    requests: dict[str, Mapping[str, object]] = {}
    examples: list[_Example] = []
    if not path.exists():
        return ()
    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return ()
    with lines:
        for line in lines:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(row, Mapping):
                continue
            event = row.get("event")
            key = f"{row.get('cycle_id', '')}|{row.get('symbol', '')}"
            if event == "pretrade_request":
                request = row.get("request")
                if isinstance(request, Mapping):
                    requests[key] = request
                continue
            if event not in {"pretrade_response", "pretrade_review"}:
                continue
            request = requests.pop(key, None)
            decision = row.get("decision")
            if request is None or not isinstance(decision, Mapping):
                continue
            if decision.get("error") or decision.get("replayed"):
                continue
            # Never train the archive on its own output. Doing that would turn
            # one historical veto into five local vetoes and eventually make
            # the synthetic echo look like independent Claude agreement.
            if str(decision.get("provider") or "") not in {"anthropic", "consensus"}:
                continue
            direction = str(row.get("direction") or request.get("direction") or "").upper()
            if direction not in {"LONG", "SHORT"}:
                continue
            shape = _shape(request)
            useful = (
                bool(decision.get("said_yes") or decision.get("approved"))
                or str(decision.get("entry_timing", "")) == "WAIT_RETEST"
            )
            risks = decision.get("risks")
            examples.append(
                _Example(
                    symbol=str(row.get("symbol") or request.get("symbol") or ""),
                    direction=direction,
                    setup_family=shape.setup_family,
                    horizon=shape.horizon,
                    features=shape.features,
                    useful=useful,
                    confidence=_number(decision.get("confidence")),
                    thesis=str(decision.get("thesis") or ""),
                    risks=tuple(str(item) for item in risks) if isinstance(risks, list) else (),
                )
            )
    return tuple(examples)


def _question(payload: Mapping[str, object]) -> Mapping[str, object]:
    evidence = payload.get("evidence_brief")
    if not isinstance(evidence, Mapping):
        return {}
    question = evidence.get("trade_question")
    return question if isinstance(question, Mapping) else {}


def _shape(payload: Mapping[str, object]) -> _Shape:
    target = payload.get("target_realism")
    target = target if isinstance(target, Mapping) else {}
    range_position = target.get("range_position")
    range_position = range_position if isinstance(range_position, Mapping) else {}
    balance = payload.get("evidence_brief")
    balance = balance if isinstance(balance, Mapping) else {}
    directional = balance.get("directional_balance")
    directional = directional if isinstance(directional, Mapping) else {}
    supporting = directional.get("supports_proposal")
    opposing = directional.get("opposes_proposal")
    support_count = len(supporting) if isinstance(supporting, list) else 0
    oppose_count = len(opposing) if isinstance(opposing, list) else 0
    directional_total = support_count + oppose_count
    history = target.get("history")
    history = history if isinstance(history, Mapping) else {}
    direction = str(payload.get("direction") or "").upper()
    historical_reach = history.get(
        "moved_up_that_far_pct" if direction == "LONG" else "moved_down_that_far_pct"
    )
    reach = target.get("proposed_direction_reach_pct", historical_reach)
    stop_distance = _number(target.get("stop_distance"))
    target_distance = _number(target.get("target_distance"))
    implied_break_even = (
        100.0 * stop_distance / (stop_distance + target_distance)
        if stop_distance > 0 and target_distance > 0
        else 0.0
    )

    raw = {
        "conviction": _number(payload.get("score")) * _number(payload.get("confidence")) / 100.0,
        "stop_atr": _number(target.get("stop_in_atr")) / 5.0,
        "target_atr": _number(target.get("target_in_atr")) / 10.0,
        "spread_share": _number(target.get("spread_as_pct_of_stop")) / 100.0,
        "reach": _number(reach) / 100.0,
        "break_even": _number(target.get("break_even_reach_pct", implied_break_even)) / 100.0,
        "entry_location": _number(range_position.get("entry_location_pct")) / 100.0,
        "target_beyond": 1.0 if range_position.get("target_beyond_range") else 0.0,
        "alignment": (
            (support_count - oppose_count) / directional_total if directional_total else 0.0
        ),
    }
    question = _question(payload)
    modules = payload.get("modules")
    modules = modules if isinstance(modules, list) else []
    active_modules = sorted(
        str(module.get("module") or "")
        for module in modules
        if isinstance(module, Mapping) and _number(module.get("score")) > 0
    )
    setup_family = str(question.get("setup_family") or "+".join(active_modules))
    horizon = str(question.get("horizon") or target.get("timeframe") or "")
    raw["horizon_minutes"] = min(1.0, _number(question.get("expected_horizon_minutes")) / 1_440.0)
    return _Shape(
        setup_family=setup_family,
        horizon=horizon,
        features=raw,
    )


def _distance(current: _Shape, example: _Example) -> float:
    keys = current.features.keys() & example.features.keys()
    numerical = (
        sum(abs(current.features[key] - example.features[key]) for key in keys) / len(keys)
        if keys
        else 1.0
    )
    family_penalty = (
        0.0 if not current.setup_family or current.setup_family == example.setup_family else 0.25
    )
    horizon_penalty = 0.0 if not current.horizon or current.horizon == example.horizon else 0.15
    return numerical + family_penalty + horizon_penalty


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
