"""Crash-safe spool and audit ledger shared by the HTTP thread and Jarvis."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from external_signals.models import NotificationEnvelope
from infra.atomic import write_json_atomic


class ExternalSignalInbox:
    """One JSON file per notification avoids cross-thread SQLite ownership."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending_dir = root / "pending"
        self.done_dir = root / "done"
        self.ledger_path = root / "events.jsonl"
        self.status_path = root / "status.json"
        self.campaigns_path = root / "campaigns.json"
        self._lock = Lock()
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, raw: dict[str, Any], *, now: datetime | None = None) -> NotificationEnvelope:
        received = (now or datetime.now(UTC)).astimezone(UTC)
        envelope = NotificationEnvelope.from_dict(
            {
                **raw,
                "event_id": str(raw.get("event_id") or uuid.uuid4()),
                "received_at": received.isoformat(),
            }
        )
        if self._duplicate_recent(envelope):
            self.record(envelope.event_id, "DUPLICATE", "identical notification already queued")
            return envelope
        target = self.pending_dir / f"{received:%Y%m%dT%H%M%S%f}-{envelope.event_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(envelope.safe_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        self.record(envelope.event_id, "RECEIVED", "authenticated notification accepted")
        return envelope

    def pending(self) -> list[tuple[Path, NotificationEnvelope]]:
        rows: list[tuple[Path, NotificationEnvelope]] = []
        for path in sorted(self.pending_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                rows.append((path, NotificationEnvelope.from_dict(raw)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.finish(path, "INVALID_FILE", "notification spool file could not be decoded")
        return rows

    def finish(self, path: Path, status: str, detail: str, **extra: object) -> None:
        event_id = path.stem.split("-", 1)[-1]
        self.record(event_id, status, detail, **extra)
        target = self.done_dir / path.name
        try:
            path.replace(target)
        except FileNotFoundError:
            return

    def record(self, event_id: str, status: str, detail: str, **extra: object) -> None:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_id": event_id,
            "status": status,
            "detail": detail,
            **extra,
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            write_json_atomic(self.status_path, row)

    def recent(self, limit: int = 200) -> list[dict[str, object]]:
        try:
            lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows: list[dict[str, object]] = []
        for line in lines[-limit:]:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                rows.append(raw)
        return rows

    def campaigns(self) -> dict[str, dict[str, object]]:
        try:
            raw = json.loads(self.campaigns_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            str(key): dict(value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def open_campaign(
        self,
        *,
        source: str,
        symbol_alias: str,
        broker_symbol: str,
        ticket: int,
        direction: str,
        event_id: str,
    ) -> None:
        campaigns = self.campaigns()
        key = f"{source.casefold()}:{symbol_alias.upper()}"
        campaigns[key] = {
            "source": source,
            "symbol_alias": symbol_alias.upper(),
            "broker_symbol": broker_symbol,
            "ticket": ticket,
            "direction": direction,
            "event_id": event_id,
            "opened_at": datetime.now(UTC).isoformat(),
            "status": "OPEN",
        }
        write_json_atomic(self.campaigns_path, campaigns)

    def active_campaign(
        self, source: str, symbol_alias: str | None
    ) -> tuple[str, dict[str, object]] | None:
        campaigns = self.campaigns()
        active = [
            (key, row)
            for key, row in campaigns.items()
            if row.get("source") == source and row.get("status") == "OPEN"
        ]
        if symbol_alias:
            wanted = symbol_alias.upper()
            active = [row for row in active if row[1].get("symbol_alias") == wanted]
        return active[0] if len(active) == 1 else None

    def close_campaign(self, key: str, status: str) -> None:
        campaigns = self.campaigns()
        row = campaigns.get(key)
        if row is None:
            return
        row["status"] = status
        row["closed_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(self.campaigns_path, campaigns)

    def _duplicate_recent(self, envelope: NotificationEnvelope) -> bool:
        fingerprint = envelope.content_fingerprint
        for path in (*self.pending_dir.glob("*.json"), *self.done_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                other = NotificationEnvelope.from_dict(raw)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            age = abs((envelope.received_at - other.received_at).total_seconds())
            if age <= 900 and other.content_fingerprint == fingerprint:
                return True
        return False
