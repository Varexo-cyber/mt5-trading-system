"""Bounded durable state for explaining whole-catalogue scanner activity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scanner.universe import ScanBatch

MAX_RECENT_INSPECTIONS = 500


class ScanActivityLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record_batch(self, batch: ScanBatch, now: datetime, operation: str) -> None:
        state = self._load()
        cycle = int(state.get("cycles", 0)) + 1
        rows = [{**item.safe_dict(), "cycle": cycle} for item in batch.inspections]
        recent = [*state.get("recent", []), *rows][-MAX_RECENT_INSPECTIONS:]
        symbols = dict(state.get("symbols", {}))
        for row in rows:
            symbols[str(row["symbol"])] = row
        state.update(
            {
                "version": 1,
                "operation": operation,
                "updated_at": now.astimezone(UTC).isoformat(),
                "cycles": cycle,
                "total_inspections": int(state.get("total_inspections", 0)) + batch.inspected,
                "universe_size": batch.universe_size,
                "next_cursor": batch.next_cursor,
                "last_batch": {
                    "inspected": batch.inspected,
                    "rejected": batch.rejected,
                    "shortlisted": len(batch.candidates),
                },
                "recent": recent,
                "symbols": symbols,
            }
        )
        self._save(state)

    def record_deep_decision(
        self,
        symbol: str,
        status: str,
        reason: str,
        detail: str,
        now: datetime,
    ) -> None:
        state = self._load()
        update = {
            "deep_status": status,
            "deep_reason": reason,
            "deep_detail": detail,
            "deep_at": now.astimezone(UTC).isoformat(),
        }
        symbols = dict(state.get("symbols", {}))
        if symbol in symbols:
            symbols[symbol] = {**symbols[symbol], **update}
        recent = list(state.get("recent", []))
        for index in range(len(recent) - 1, -1, -1):
            if recent[index].get("symbol") == symbol:
                recent[index] = {**recent[index], **update}
                break
        state["symbols"] = symbols
        state["recent"] = recent
        state["updated_at"] = now.astimezone(UTC).isoformat()
        self._save(state)

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


def read_scan_activity(path: Path) -> dict[str, Any]:
    return ScanActivityLedger(path)._load()
