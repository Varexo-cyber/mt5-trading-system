"""Durable, secret-free audit trail for pre-trade reviews and post-trade reflections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path


class AIReviewLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: str, payload: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **dict(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def read_recent_reviews(path: Path, limit: int = 25) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError, TypeError):
        return []
