"""Durable rate and duplicate control for paid market-scout calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class ScoutThrottle:
    """Reserve a paid call before dispatch so restarts cannot create a burst."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def reserve(
        self,
        signature: str,
        now: datetime,
        *,
        cooldown_minutes: int,
        max_calls_per_day: int,
    ) -> tuple[bool, str]:
        current = now.astimezone(UTC)
        state = self._load()
        day = current.date().isoformat()
        if state.get("utc_day") != day:
            state = {"utc_day": day, "calls": 0, "signatures": []}
        signatures = [str(item) for item in state.get("signatures", [])]
        if signature in signatures:
            return False, "same closed-bar snapshot already scouted"
        calls = int(state.get("calls", 0))
        if calls >= max_calls_per_day:
            return False, "daily Claude scout call budget reached"
        last_value = state.get("last_call_at")
        if last_value:
            try:
                last_call = datetime.fromisoformat(str(last_value)).astimezone(UTC)
            except ValueError:
                last_call = None
            if last_call is not None and current - last_call < timedelta(minutes=cooldown_minutes):
                return False, "Claude scout global cooldown active"
        state.update(
            {
                "last_call_at": current.isoformat(),
                "calls": calls + 1,
                "signatures": [*signatures, signature][-500:],
            }
        )
        self._save(state)
        return True, "reserved"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
