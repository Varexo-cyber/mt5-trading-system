"""Cost-free decisions distilled from completed Claude reviews.

This is deliberately not presented as a replacement language model. Jarvis'
deterministic analysis still creates the setup. The archive contributes two
bounded lessons: repeat a well-supported pre-trade refusal, and manage an open
position like comparable Claude-supervised states only when their realised
outcomes prove that advice helped. Unknown shapes pass back to the ordinary
deterministic and broker-native safety layers.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True, slots=True)
class _SupervisionExample:
    direction: str
    features: dict[str, float]
    action: str
    confidence: float
    reason: str
    r_at_the_time: float
    result_r: float


@dataclass(frozen=True, slots=True)
class _SupervisionShape:
    direction: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class _OutcomeRecord:
    symbol: str
    direction: str
    trades: int
    wins: int
    total_r: float

    @property
    def mean_r(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0


class LocalHistoryAdvisor:
    """Replay broad Claude judgement without making an external API call.

    This is a small nearest-neighbour model, not a renamed threshold. Entry
    review compares the complete proposal shape with prior Claude reviews.
    Position supervision separately compares the live trade state -- R, peak,
    giveback, age, costs, target/stop distance and mechanical candle health --
    with prior Claude supervision states. It acts only when several nearby
    examples strongly agree; unfamiliar states remain a hold.
    """

    supports_dynamic_management = True
    uses_paid_api = False

    def __init__(self, config: AIConfig, ledger_path: Path, brain: Any | None = None) -> None:
        self.minimum = config.minimum_confidence
        self.min_neighbors = config.local_history_min_neighbors
        self.max_distance = config.local_history_max_distance
        self.veto_rate = config.local_history_veto_rate
        self.ledger_path = ledger_path
        #: The durable archive, when one is configured. The local JSONL is a
        #: file on a rented VPS that starts empty after a fresh clone, and this
        #: model needs five comparable past states before it may act on a
        #: position — so on an empty file it holds everything, forever, while
        #: the account's whole history sits in Postgres unread. Both are used:
        #: the file is the most recent and the fastest, the brain is the one
        #: that outlives the machine.
        self.brain = brain
        self.examples = _load_examples(ledger_path)
        self.supervision_examples = _merge_supervision_examples(
            _load_supervision_examples(ledger_path), brain
        )
        self._supervision_signature = _supervision_sources_signature(ledger_path)
        self.outcomes = _load_outcome_records(ledger_path)
        self._entry_signature = self._supervision_signature

    def attach_brain(self, brain: Any) -> None:
        """Adopt the durable archive after construction.

        The runner builds the adviser before it builds the brain, so the
        constructor cannot be given one. Attaching afterwards and re-merging is
        less invasive than reordering startup, and it is idempotent: the merge
        de-duplicates on (direction, feature vector), so calling it twice adds
        nothing.
        """
        self.brain = brain
        self.supervision_examples = _merge_supervision_examples(
            _load_supervision_examples(self.ledger_path), brain
        )

    def scout(self, _market_state: Mapping[str, object]) -> ScoutDecision:
        return ScoutDecision(
            thesis="Paid market scouting is disabled in local-history mode",
            provider="local_history",
            model="jarvis_outcome_memory",
        )

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> Advice:
        self._refresh_entry_evidence()
        payload = build_review_payload(idea, context, proposal, memory)
        current = _shape(payload)
        direction = idea.direction.name if idea.direction is not None else ""
        engine_basis = _engine_basis(idea, self.outcomes.get((idea.symbol.upper(), direction)))
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
                f"{engine_basis} The secondary reviewer archive found {len(neighbors)} "
                f"sufficiently similar opinions; {self.min_neighbors} are required before "
                "that archive may veto Jarvis.",
                provider="local_history",
                model="jarvis_outcome_memory",
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
                f"{engine_basis} Secondary archive veto: {len(neighbors)} comparable reviewer "
                f"opinions produced a weighted {learned_veto_rate:.0%} refusal rate. Closest "
                f"recorded reason: {closest.thesis or 'no thesis recorded'}",
                risks=closest.risks,
                provider="local_history",
                model="jarvis_outcome_memory",
                said_yes=False,
                threshold=self.minimum,
            )

        useful_rate = 1.0 - learned_veto_rate
        return Advice(
            True,
            max(self.minimum, min(1.0, useful_rate)),
            f"{engine_basis} The secondary archive contains {len(neighbors)} comparable "
            f"reviewer opinions and did not reach its {self.veto_rate:.0%} veto threshold. "
            "Price, spread, sizing and broker checks still run.",
            provider="local_history",
            model="jarvis_outcome_memory",
            said_yes=True,
            threshold=self.minimum,
        )

    def reflect(self, _outcome: Mapping[str, object]) -> Reflection:
        return Reflection(
            "Paid post-trade reflection is disabled; realised outcome remains in Jarvis memory",
            provider="local_history",
            model="jarvis_outcome_memory",
        )

    def supervise(self, position_state: Mapping[str, object]) -> Supervision:
        self._refresh_supervision_examples()
        current = _supervision_shape(position_state)
        ranked = sorted(
            (
                (_supervision_distance(current, example), example)
                for example in self.supervision_examples
                if example.direction == current.direction
            ),
            key=lambda item: item[0],
        )
        neighbors = [item for item in ranked[:_MAX_NEIGHBORS] if item[0] <= self.max_distance]
        if len(neighbors) < self.min_neighbors:
            return Supervision(
                "hold",
                f"Local position AI found {len(neighbors)} comparable Claude states; "
                f"{self.min_neighbors} are required before it may intervene.",
                confidence=0.0,
                provider="local_history",
                model="jarvis_outcome_memory",
                review_after_minutes=2.0,
            )

        # Class-balanced nearest neighbours. HOLD naturally outnumbers CLOSE by
        # a wide margin because a healthy trade is reviewed repeatedly. A raw
        # majority vote would therefore answer HOLD forever, even when all of
        # the closest broken-thesis examples are closes. Compare equal-sized
        # neighbourhoods per action instead; frequency cannot drown proximity.
        by_action: dict[str, list[tuple[float, _SupervisionExample]]] = {}
        for distance, example in ranked:
            if distance <= self.max_distance:
                by_action.setdefault(example.action, []).append((distance, example))
        action_distance = {
            action: sum(distance for distance, _ in items[: self.min_neighbors])
            / self.min_neighbors
            for action, items in by_action.items()
            if len(items) >= self.min_neighbors
        }
        if not action_distance:
            return Supervision(
                "hold",
                "Local position AI has no outcome-graded action with enough comparable states.",
                confidence=0.0,
                provider="local_history",
                model="jarvis_outcome_memory",
                review_after_minutes=2.0,
            )

        ordered = sorted(action_distance.items(), key=lambda item: item[1])
        action, distance = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else None
        confidence = max(0.0, min(1.0, 1.0 - distance / self.max_distance))
        separated = runner_up is None or distance <= runner_up * 0.75
        closest = by_action[action][0][1]

        if action not in {"hold", "close"} or confidence < self.veto_rate or not separated:
            comparison = (
                "no competing action has enough evidence"
                if runner_up is None
                else f"nearest competing class is only {runner_up / max(distance, 1e-9):.2f}x away"
            )
            return Supervision(
                "hold",
                f"Local position AI found {len(neighbors)} nearby states, but the learned "
                f"{action} class is not decisive ({confidence:.0%}; {comparison}).",
                confidence=confidence,
                provider="local_history",
                model="jarvis_outcome_memory",
                review_after_minutes=2.0,
            )

        return Supervision(
            action,
            f"Local position AI: the {self.min_neighbors} closest outcome-graded {action} "
            f"states match at {confidence:.0%} confidence. Closest recorded reasoning: "
            f"{closest.reason or 'no reason recorded'}",
            confidence=max(self.minimum, confidence),
            provider="local_history",
            model="jarvis_outcome_memory",
            thesis_state="broken" if action == "close" else "intact",
            urgency="soon" if action == "close" else "routine",
            review_after_minutes=2.0,
        )

    def _refresh_supervision_examples(self) -> None:
        """Learn newly completed outcomes without requiring a Jarvis restart."""
        signature = _supervision_sources_signature(self.ledger_path)
        if signature == self._supervision_signature:
            return
        self.supervision_examples = _merge_supervision_examples(
            _load_supervision_examples(self.ledger_path), self.brain
        )
        self._supervision_signature = signature

    def _refresh_entry_evidence(self) -> None:
        """Pick up new reviews and realised trades without a Jarvis restart."""
        signature = _supervision_sources_signature(self.ledger_path)
        if signature == self._entry_signature:
            return
        self.examples = _load_examples(self.ledger_path)
        self.outcomes = _load_outcome_records(self.ledger_path)
        self._entry_signature = signature


def _engine_basis(idea: TradeIdea, outcome: _OutcomeRecord | None) -> str:
    modules = [signal.module for signal in idea.signals if signal.score > 0]
    module_text = ", ".join(modules) if modules else "the active price model"
    direction = idea.direction.name if idea.direction is not None else "unknown"
    basis = (
        f"Jarvis independently formed this {direction} setup from {module_text}, "
        f"scoring {idea.score:.1f} with {idea.confidence:.0%} engine confidence."
    )
    if outcome is None or outcome.trades == 0:
        return basis + " Its own realised record has no matching symbol/direction trade yet."
    evidence = "anecdotal" if outcome.trades < 30 else "developing"
    if outcome.trades >= 100:
        evidence = "minimum sample reached"
    return (
        basis + f" Its own {idea.symbol} {outcome.direction} record is {outcome.trades} trades, "
        f"{outcome.wins / outcome.trades:.0%} wins and {outcome.total_r:+.2f}R total "
        f"({outcome.mean_r:+.2f}R/trade; {evidence})."
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



def _merge_supervision_examples(
    local: tuple[_SupervisionExample, ...], brain: Any | None
) -> tuple[_SupervisionExample, ...]:
    """Local file plus durable archive, the file winning on overlap.

    Both, deliberately, and not one or the other. The JSONL is the freshest
    and needs no network; the brain is the only copy that survives a fresh
    clone of the repository onto a new VPS, which is where the local one has
    been starting from zero. This model refuses to act until it has five
    comparable states, so an empty archive is not a degraded model — it is a
    model that holds every position indefinitely.

    Brain rows carry no `result_r`: the outcome grading lives with the closed
    trade, not with the moment the question was asked. They are matched on and
    counted, and the outcome-weighted paths simply see a zero, which is the
    same treatment an ungraded local row already receives.
    """
    if brain is None:
        return local
    try:
        rows = brain.supervision_examples()
    except Exception:  # noqa: BLE001 - memory must never break the adviser
        return local
    if not rows:
        return local
    seen = {(example.direction, tuple(sorted(example.features.items()))) for example in local}
    merged = list(local)
    for row in rows:
        try:
            features = {str(k): float(v) for k, v in dict(row["features"]).items()}
            key = (str(row["direction"]), tuple(sorted(features.items())))
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                _SupervisionExample(
                    direction=str(row["direction"]),
                    features=features,
                    action=str(row["action"]),
                    confidence=float(row.get("confidence") or 0.0),
                    reason="",
                    r_at_the_time=float(row.get("r_at_the_time") or 0.0),
                    result_r=0.0,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(merged)


def _load_supervision_examples(path: Path) -> tuple[_SupervisionExample, ...]:
    examples: list[_SupervisionExample] = []
    if not path.exists():
        return ()
    outcomes = _closed_trade_outcomes(path)
    if not outcomes:
        # Ungraded language-model answers are opinions, not learning data.
        # Waiting for the trade to close costs coverage; treating every eloquent
        # answer as truth permanently teaches mistakes.
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
            if not isinstance(row, Mapping) or row.get("event") != "position_supervision":
                continue
            request = row.get("request")
            decision = row.get("decision")
            if not isinstance(request, Mapping) or not isinstance(decision, Mapping):
                continue
            if decision.get("error") or str(decision.get("provider") or "") not in {
                "anthropic",
                "consensus",
            }:
                continue
            action = str(decision.get("action") or "").lower()
            if action not in {"hold", "close"}:
                continue
            try:
                ticket = int(row.get("ticket") or request.get("ticket") or 0)
            except (TypeError, ValueError):
                continue
            if ticket not in outcomes:
                continue
            r_at_the_time = _number(request.get("unrealised_r"))
            result_r = outcomes[ticket]
            # A close was useful when waiting did not improve the realised R;
            # a hold was useful when the eventual result did not deteriorate.
            # Five hundredths of R absorbs spread and fill noise around equality.
            useful = (
                result_r <= r_at_the_time + 0.05
                if action == "close"
                else result_r >= r_at_the_time - 0.05
            )
            if not useful:
                continue
            shape = _supervision_shape(request)
            if shape.direction not in {"LONG", "SHORT"}:
                continue
            examples.append(
                _SupervisionExample(
                    direction=shape.direction,
                    features=shape.features,
                    action=action,
                    confidence=_number(decision.get("confidence")),
                    reason=str(decision.get("reason") or ""),
                    r_at_the_time=r_at_the_time,
                    result_r=result_r,
                )
            )
    return tuple(examples)


def _closed_trade_outcomes(ledger_path: Path) -> dict[int, float]:
    database = _outcome_database(ledger_path)
    if database is None:
        return {}
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT ticket, pnl_r FROM trades "
                "WHERE closed_at IS NOT NULL AND ticket IS NOT NULL AND pnl_r IS NOT NULL"
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}
    return {int(ticket): float(pnl_r) for ticket, pnl_r in rows}


def _load_outcome_records(ledger_path: Path) -> dict[tuple[str, str], _OutcomeRecord]:
    """Jarvis' own realised scoreboard, independent of reviewer wording."""
    database = _outcome_database(ledger_path)
    if database is None:
        return {}
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT UPPER(symbol), UPPER(direction), COUNT(*), "
                "SUM(CASE WHEN pnl_r > 0 THEN 1 ELSE 0 END), SUM(pnl_r) "
                "FROM trades WHERE closed_at IS NOT NULL AND pnl_r IS NOT NULL "
                "GROUP BY UPPER(symbol), UPPER(direction)"
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}
    return {
        (str(symbol), str(direction)): _OutcomeRecord(
            str(symbol), str(direction), int(trades), int(wins or 0), float(total_r or 0.0)
        )
        for symbol, direction, trades, wins, total_r in rows
    }


def _outcome_database(ledger_path: Path) -> Path | None:
    candidates = (
        ledger_path.parent.parent / "journal" / "trading.db",
        ledger_path.parent / "trading.db",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _supervision_sources_signature(ledger_path: Path) -> tuple[tuple[object, ...], ...]:
    sources = [ledger_path]
    database = _outcome_database(ledger_path)
    if database is not None:
        sources.append(database)
    signature: list[tuple[object, ...]] = []
    for source in sources:
        try:
            stat = source.stat()
        except OSError:
            signature.append((0, 0))
        else:
            signature.append((stat.st_mtime_ns, stat.st_size))
    # SQLite may update values without changing file size, and on some Windows
    # volumes two writes inside one clock tick share an mtime. Include a cheap
    # outcome aggregate so a just-closed trade becomes learning evidence on
    # the very next review instead of waiting for a restart or another write.
    if database is not None:
        try:
            with sqlite3.connect(database) as connection:
                row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(pnl_r), 0.0), "
                    "COALESCE(MAX(closed_at), '') FROM trades "
                    "WHERE closed_at IS NOT NULL AND pnl_r IS NOT NULL"
                ).fetchone()
        except (OSError, sqlite3.Error):
            signature.append(("outcomes", "unavailable"))
        else:
            signature.append(("outcomes", int(row[0]), float(row[1]), str(row[2])))
    return tuple(signature)


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


def supervision_features(payload: Mapping[str, object]) -> dict[str, float]:
    """The feature vector for one supervision payload, for durable storage.

    Public, and a thin wrapper on purpose. The archive is matched on these
    numbers, so whoever writes a row and whoever compares against it must use
    one definition — two extractors that drift apart produce a matcher that
    confidently compares different things, which is worse than no matcher.
    """
    return dict(_supervision_shape(payload).features)


def _supervision_shape(payload: Mapping[str, object]) -> _SupervisionShape:
    context = payload.get("context")
    context = context if isinstance(context, Mapping) else {}
    health = context.get("mechanical_health")
    health = health if isinstance(health, Mapping) else {}
    health_action = str(health.get("action") or "").lower()
    raw_signals = health.get("signals")
    signals: dict[str, float] = {}
    if isinstance(raw_signals, list):
        for item in raw_signals:
            if isinstance(item, str):
                signals[item.lower()] = 1.0
            elif isinstance(item, Mapping):
                name = str(item.get("name") or "").lower()
                if name:
                    signals[name] = _clip(_number(item.get("severity")), 0.0, 1.0)
    health_verdict = str(health.get("verdict") or "").lower()
    features = {
        "unrealised_r": _clip(_number(payload.get("unrealised_r")), -3.0, 3.0) / 3.0,
        "peak_r": _clip(_number(payload.get("peak_unrealised_r")), 0.0, 3.0) / 3.0,
        "giveback": _clip(_number(payload.get("profit_given_back_fraction")), 0.0, 1.0),
        "age": _clip(_number(payload.get("age_hours")), 0.0, 24.0) / 24.0,
        "spread": _clip(_number(payload.get("spread_as_fraction_of_initial_risk")), 0.0, 0.25)
        / 0.25,
        "stop_atr": _clip(_number(payload.get("distance_to_stop_in_atr")), 0.0, 5.0) / 5.0,
        "target_atr": _clip(_number(payload.get("distance_to_target_in_atr")), 0.0, 8.0) / 8.0,
        "account_profit": _clip(_number(payload.get("unrealised_pct_of_account")), -2.0, 2.0) / 2.0,
        "health": _clip(_number(health.get("severity")), 0.0, 1.0),
        "health_exit": 1.0 if health_action == "exit" else 0.0,
        "health_tighten": 1.0 if health_action == "tighten" else 0.0,
        "health_broken": 1.0 if health_verdict == "broken" else 0.0,
        "health_deteriorating": 1.0 if health_verdict == "deteriorating" else 0.0,
        "structure_broken": signals.get("structure_broken", 0.0),
        "momentum_turned": signals.get("momentum_turned", 0.0),
        "profit_giveback": signals.get("profit_giveback", 0.0),
        "peak_stall": signals.get("peak_stall", 0.0),
    }
    return _SupervisionShape(str(payload.get("direction") or "").upper(), features)


def _supervision_distance(current: _SupervisionShape, example: _SupervisionExample) -> float:
    keys = current.features.keys() & example.features.keys()
    if not keys:
        return 1.0
    return sum(abs(current.features[key] - example.features[key]) for key in keys) / len(keys)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
