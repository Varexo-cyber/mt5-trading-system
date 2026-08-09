"""Durable, secret-free audit trail for pre-trade reviews and post-trade reflections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from core.clock import Clock, LiveClock

if TYPE_CHECKING:
    from journal.database import Journal


class AIReviewLedger:
    def __init__(
        self,
        path: Path,
        database: Journal | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self.database = database
        self.clock = clock or LiveClock()

    def append(self, event: str, payload: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": self.clock.now().isoformat(),
            "event": event,
            **dict(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        if self.database is not None:
            self.database.record_ai_event(row)

    def backfill_database(self) -> int:
        """Idempotently mirror historical JSONL events into the journal."""
        if self.database is None:
            return 0
        rows = _read_rows(self.path)
        for row in rows:
            self.database.record_ai_event(row)
        return len(rows)


def read_recent_reviews(path: Path, limit: int = 25) -> list[dict[str, object]]:
    return _read_rows(path)[-limit:]


def read_trade_reflections(path: Path) -> list[dict[str, object]]:
    """Every successful structured post-trade reflection, oldest first."""
    return [
        row
        for row in _read_rows(path)
        if row.get("event") == "posttrade_reflection"
        and isinstance(row.get("outcome"), dict)
        and isinstance(row.get("reflection"), dict)
        and not row["reflection"].get("error")  # type: ignore[union-attr]
    ]


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows
