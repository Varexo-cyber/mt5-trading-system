"""Version, constrain and shadow-test learning proposals."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import yaml

from config.loader import load_settings


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConfigControl:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.live_path = root / "config" / "config.yaml"
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
        if (last_seen - started).days < 30 or int(state.get("decisions", 0)) < 30:
            raise RuntimeError("anti-recency gate: shadow needs 30 days and 30 decisions")
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


class ShadowRecorder:
    def __init__(self, root: Path) -> None:
        self.directory = root / "runtime" / "shadow"

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
        different = (current.approved, current.direction) != (
            candidate.approved,
            candidate.direction,
        )
        state["last_seen_at"] = now.astimezone(UTC).isoformat()
        state["decisions"] = int(state.get("decisions", 0)) + 1
        state["different_decisions"] = int(state.get("different_decisions", 0)) + int(different)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(state_path)
        with (self.directory / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": now.astimezone(UTC).isoformat(),
                        "symbol": symbol,
                        "current_approved": current.approved,
                        "current_direction": (
                            current.direction.name if current.direction is not None else None
                        ),
                        "candidate_approved": candidate.approved,
                        "candidate_direction": (
                            candidate.direction.name if candidate.direction is not None else None
                        ),
                        "different": different,
                    }
                )
                + "\n"
            )
