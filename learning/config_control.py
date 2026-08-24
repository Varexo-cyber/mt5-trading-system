"""Version, constrain and shadow-test learning proposals."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
import yaml

from config.loader import load_settings
from config.schema import LearningConfig
from core.broker import Broker
from core.types import Direction, Timeframe
from infra.atomic import write_json_atomic
from learning.counterfactual import classify_path, future_bars


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConfigControl:
    def __init__(self, root: Path, live_path: Path | None = None) -> None:
        """`live_path` is the file whose weights actually decide.

        IT IS NOT ALWAYS `config/config.yaml`, and assuming it was made this
        whole pipeline a no-op on the live account. Weights are set in the
        base config AND in `config/eightcap.yaml`, and the overlay wins. A
        promotion that rewrote the base file would version it, shadow-test it,
        pass, promote — and change nothing at all about what trades, because
        the overlay still says what it always said.

        Left defaulting to the base config so existing callers and their tests
        are unchanged; the operator scripts pass the overlay they are running.
        """
        self.root = root
        self.live_path = live_path or root / "config" / "config.yaml"
        self.shadow_dir = root / "runtime" / "shadow"
        self.versions = root / "runtime" / "config_versions"

    def snapshot(self, label: str) -> Path:
        safe = "".join(char for char in label if char.isalnum() or char in "-_") or "snapshot"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.versions / f"{stamp}-{safe}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.live_path, path)
        return path

    def start_shadow(self, candidate: Path) -> Path:
        load_settings(candidate, env_overrides=False)
        self._assert_weight_only_change(candidate)
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        target = self.shadow_dir / "candidate.yaml"
        shutil.copy2(candidate, target)
        now = datetime.now(UTC).isoformat()
        (self.shadow_dir / "state.json").write_text(
            json.dumps(
                {
                    "candidate_hash": file_hash(target),
                    "started_at": now,
                    "last_seen_at": now,
                    "decisions": 0,
                    "different_decisions": 0,
                    "resolved_differences": 0,
                    "champion_total_r": 0.0,
                    "challenger_total_r": 0.0,
                    "lift_sum_r": 0.0,
                    "lift_sum_squares_r": 0.0,
                    "last_pending_fingerprint": {},
                    "resolved_days": [],
                    "resolved_symbols": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    def promote_shadow(self) -> Path:
        candidate = self.shadow_dir / "candidate.yaml"
        state_path = self.shadow_dir / "state.json"
        if not candidate.exists() or not state_path.exists():
            raise RuntimeError("no active shadow candidate")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        started = datetime.fromisoformat(state["started_at"])
        last_seen = datetime.fromisoformat(state["last_seen_at"])
        learning = load_settings(self.live_path, env_overrides=False).learning
        days = (last_seen - started).days
        resolved = int(state.get("resolved_differences", 0))
        if days < learning.shadow_min_days or resolved < learning.shadow_min_paired_outcomes:
            raise RuntimeError(
                "anti-recency gate: shadow needs "
                f"{learning.shadow_min_days} days and "
                f"{learning.shadow_min_paired_outcomes} paired outcomes; "
                f"has {days} days and {resolved}"
            )
        unique_days = len(set(state.get("resolved_days", [])))
        unique_symbols = len(set(state.get("resolved_symbols", [])))
        if (
            unique_days < learning.shadow_min_unique_days
            or unique_symbols < learning.shadow_min_symbols
        ):
            raise RuntimeError(
                "shadow evidence lacks breadth: "
                f"days={unique_days}/{learning.shadow_min_unique_days}, "
                f"symbols={unique_symbols}/{learning.shadow_min_symbols}"
            )
        mean, lower = _paired_lift(state, learning.shadow_confidence_level)
        if mean < learning.shadow_min_expectancy_lift_r or lower <= 0.0:
            raise RuntimeError(
                "shadow challenger has not proven positive lift: "
                f"mean={mean:.3f}R, lower {learning.shadow_confidence_level:.1%} "
                f"bound={lower:.3f}R, required mean="
                f"{learning.shadow_min_expectancy_lift_r:.3f}R"
            )
        if file_hash(candidate) != state.get("candidate_hash"):
            raise RuntimeError("shadow candidate changed during its test")
        self._assert_weight_only_change(candidate)
        backup = self.snapshot("before-shadow-promotion")
        shutil.copy2(candidate, self.live_path)
        return backup

    def restore(self, version: Path) -> None:
        resolved = version.resolve()
        if self.versions.resolve() not in resolved.parents:
            raise RuntimeError("restore target must be a snapshot in runtime/config_versions")
        load_settings(resolved, env_overrides=False)
        self.snapshot("before-rollback")
        shutil.copy2(resolved, self.live_path)

    def _assert_weight_only_change(self, candidate: Path) -> None:
        current = yaml.safe_load(self.live_path.read_text(encoding="utf-8"))
        proposed = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        current_weights = current["analysis"]["confluence"]["weights"]
        proposed_weights = proposed["analysis"]["confluence"]["weights"]
        current_without = deepcopy(current)
        proposed_without = deepcopy(proposed)
        current_without["analysis"]["confluence"]["weights"] = {}
        proposed_without["analysis"]["confluence"]["weights"] = {}
        if current_without != proposed_without:
            raise RuntimeError("automatic shadow proposals may change module weights only")
        if set(current_weights) != set(proposed_weights):
            raise RuntimeError("a shadow proposal may not add or remove modules")
        for module, old_raw in current_weights.items():
            old, new = float(old_raw), float(proposed_weights[module])
            if old == 0 and new != 0:
                raise RuntimeError(f"{module}: zero-weight modules require full revalidation")
            if old and abs(new / old - 1.0) > 0.15 + 1e-9:
                raise RuntimeError(f"{module}: weight shift exceeds the quarterly +/-15% cap")


def _paired_lift(state: dict[str, Any], confidence: float) -> tuple[float, float]:
    """Mean challenger-minus-champion R and its two-sided lower bound."""
    count = int(state.get("resolved_differences", 0))
    if count < 2:
        return 0.0, float("-inf")
    total = float(state.get("lift_sum_r", 0.0))
    squares = float(state.get("lift_sum_squares_r", 0.0))
    mean = total / count
    variance = max(0.0, (squares - total * total / count) / (count - 1))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return mean, mean - z * sqrt(variance / count)


def _plan(idea: Any) -> dict[str, Any]:
    approved = (
        bool(getattr(idea, "approved", False)) and getattr(idea, "direction", None) is not None
    )
    if not approved:
        return {"approved": False}
    return {
        "approved": True,
        "direction": idea.direction.name,
        "entry": float(idea.entry),
        "stop_loss": float(idea.stop_loss),
        "take_profit": float(idea.take_profit),
        "score": float(idea.score),
        "confidence": float(idea.confidence),
        "reason": str(idea.reason),
    }


class ShadowRecorder:
    def __init__(self, root: Path, config: LearningConfig | None = None) -> None:
        self.directory = root / "runtime" / "shadow"
        self.config = config or LearningConfig()

    @property
    def active(self) -> bool:
        return (self.directory / "candidate.yaml").exists() and (
            self.directory / "state.json"
        ).exists()

    def record(self, symbol: str, current, candidate, now: datetime) -> None:  # type: ignore[no-untyped-def]
        if not self.active:
            return
        state_path = self.directory / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        champion = _plan(current)
        challenger = _plan(candidate)
        different = (champion.get("approved"), champion.get("direction")) != (
            challenger.get("approved"),
            challenger.get("direction"),
        )
        state["last_seen_at"] = now.astimezone(UTC).isoformat()
        state["decisions"] = int(state.get("decisions", 0)) + 1
        state["different_decisions"] = int(state.get("different_decisions", 0)) + int(different)
        decision = {
            "at": now.astimezone(UTC).isoformat(),
            "symbol": symbol,
            "champion": champion,
            "challenger": challenger,
            "different": different,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {"symbol": symbol, "champion": champion, "challenger": challenger},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        seen = state.setdefault("last_pending_fingerprint", {})
        should_queue = bool(different and (champion["approved"] or challenger["approved"]))
        should_queue = should_queue and seen.get(symbol) != fingerprint
        if should_queue:
            decision_id = hashlib.sha256(
                f"{decision['at']}|{symbol}|{fingerprint}".encode()
            ).hexdigest()
            decision["decision_id"] = decision_id
            seen[symbol] = fingerprint
            with (self.directory / "pending.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision, sort_keys=True) + "\n")
        write_json_atomic(state_path, state)
        with (self.directory / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")

    def resolve(self, broker: Broker, now: datetime, *, limit: int = 50) -> int:
        """Passively score paired champion/challenger decisions on later bars."""
        if not self.active:
            return 0
        pending_path = self.directory / "pending.jsonl"
        outcomes_path = self.directory / "outcomes.jsonl"
        if not pending_path.exists():
            return 0
        resolved_ids = {
            str(row.get("decision_id", ""))
            for row in _jsonl(outcomes_path)
            if row.get("decision_id")
        }
        state_path = self.directory / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        resolved = 0
        horizon = timedelta(hours=self.config.shadow_resolution_hours)
        for decision in _jsonl(pending_path):
            decision_id = str(decision.get("decision_id", ""))
            if not decision_id or decision_id in resolved_ids:
                continue
            opened = datetime.fromisoformat(str(decision["at"]))
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=UTC)
            symbol = str(decision["symbol"])
            try:
                raw = broker.copy_rates_range(
                    symbol,
                    Timeframe.M15.mt5_value,
                    opened,
                    now,
                )
                frame = future_bars(raw, opened)
            except Exception:  # noqa: BLE001 - passive research may never stop live management
                continue
            if frame.empty:
                continue
            timed_out = now - opened >= horizon
            champion = _resolve_plan(decision["champion"], frame, timed_out)
            challenger = _resolve_plan(decision["challenger"], frame, timed_out)
            if champion is None or challenger is None:
                continue
            lift = challenger[1] - champion[1]
            outcome = {
                "decision_id": decision_id,
                "resolved_at": now.astimezone(UTC).isoformat(),
                "symbol": symbol,
                "champion_outcome": champion[0],
                "champion_r": champion[1],
                "challenger_outcome": challenger[0],
                "challenger_r": challenger[1],
                "lift_r": lift,
            }
            with outcomes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(outcome, sort_keys=True) + "\n")
            state["resolved_differences"] = int(state.get("resolved_differences", 0)) + 1
            state["champion_total_r"] = float(state.get("champion_total_r", 0.0)) + champion[1]
            state["challenger_total_r"] = (
                float(state.get("challenger_total_r", 0.0)) + challenger[1]
            )
            state["lift_sum_r"] = float(state.get("lift_sum_r", 0.0)) + lift
            state["lift_sum_squares_r"] = float(state.get("lift_sum_squares_r", 0.0)) + lift * lift
            state["last_resolved_at"] = now.astimezone(UTC).isoformat()
            state.setdefault("resolved_days", []).append(opened.date().isoformat())
            state["resolved_days"] = sorted(set(state["resolved_days"]))
            state.setdefault("resolved_symbols", []).append(symbol)
            state["resolved_symbols"] = sorted(set(state["resolved_symbols"]))
            resolved_ids.add(decision_id)
            resolved += 1
            if resolved >= limit:
                break
        if resolved:
            mean, lower = _paired_lift(state, self.config.shadow_confidence_level)
            state["paired_mean_lift_r"] = mean
            state["paired_lower_confidence_bound_r"] = lower
            write_json_atomic(state_path, state)
        return resolved


def _resolve_plan(
    plan: dict[str, Any], frame: pd.DataFrame, timed_out: bool
) -> tuple[str, float] | None:
    if not bool(plan.get("approved", False)):
        return "NO_TRADE", 0.0
    direction = Direction[str(plan["direction"])]
    outcome, result = classify_path(
        frame,
        direction,
        float(plan["entry"]),
        float(plan["stop_loss"]),
        float(plan["take_profit"]),
        timed_out=timed_out,
    )
    return None if outcome is None else (outcome, result)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        return []
    return rows
