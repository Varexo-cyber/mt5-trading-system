"""Durable acceptance evidence for autonomous paper/demo operation."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OperationLedger:
    """Append-only-ish session ledger that survives orderly exits and crashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.session_id: str | None = None

    def start(self, operation: str, account_login: int, now: datetime) -> None:
        payload = self._load()
        self.session_id = str(uuid.uuid4())
        payload.setdefault("sessions", []).append(
            {
                "id": self.session_id,
                "operation": operation,
                "account_login": account_login,
                "started_at": now.astimezone(UTC).isoformat(),
                "last_seen_at": now.astimezone(UTC).isoformat(),
                "ended_at": None,
                "cycles": 0,
                "trades_opened": 0,
                "unexplained_reconciliations": 0,
            }
        )
        self._save(payload)

    def cycle(self, now: datetime, *, trades_opened: int = 0) -> None:
        session = self._session()
        if session is None:
            return
        session["last_seen_at"] = now.astimezone(UTC).isoformat()
        session["cycles"] = int(session["cycles"]) + 1
        session["trades_opened"] = int(session["trades_opened"]) + trades_opened
        self._replace(session)

    def reconciliation_failure(self, now: datetime) -> None:
        session = self._session()
        if session is None:
            return
        session["last_seen_at"] = now.astimezone(UTC).isoformat()
        session["unexplained_reconciliations"] = int(session["unexplained_reconciliations"]) + 1
        self._replace(session)

    def finish(self, now: datetime) -> None:
        session = self._session()
        if session is None:
            return
        value = now.astimezone(UTC).isoformat()
        session["last_seen_at"] = value
        session["ended_at"] = value
        self._replace(session)

    def _session(self) -> dict[str, Any] | None:
        if self.session_id is None:
            return None
        return next(
            (row for row in self._load().get("sessions", []) if row.get("id") == self.session_id),
            None,
        )

    def _replace(self, session: dict[str, Any]) -> None:
        payload = self._load()
        rows = payload.setdefault("sessions", [])
        payload["sessions"] = [
            session if row.get("id") == session.get("id") else row for row in rows
        ]
        self._save(payload)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"sessions": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"sessions": []}

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
