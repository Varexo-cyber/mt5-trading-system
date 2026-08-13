"""Immutable messages crossing the phone-to-Jarvis boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from core.types import Direction


class ExternalSignalKind(StrEnum):
    NEW = "NEW"
    MOVE_STOP = "MOVE_STOP"
    BOOK_PROFIT = "BOOK_PROFIT"
    CLOSE = "CLOSE"
    TP_HIT = "TP_HIT"
    CANCEL = "CANCEL"
    INFO = "INFO"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class NotificationEnvelope:
    """Exactly what MacroDroid captured, before interpretation."""

    event_id: str
    received_at: datetime
    app: str = ""
    title: str = ""
    text: str = ""
    big_text: str = ""
    notification_timestamp: str = ""
    source: str = "rio"

    @property
    def combined_text(self) -> str:
        parts: list[str] = []
        for value in (self.title, self.text, self.big_text):
            clean = value.strip()
            if clean and clean not in parts:
                parts.append(clean)
        return "\n".join(parts)

    @property
    def content_fingerprint(self) -> str:
        canonical = "|".join(
            (
                self.source.casefold().strip(),
                self.app.casefold().strip(),
                " ".join(self.combined_text.casefold().split()),
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def safe_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["received_at"] = self.received_at.astimezone(UTC).isoformat()
        return row

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NotificationEnvelope:
        timestamp = raw.get("received_at")
        received = (
            datetime.fromisoformat(str(timestamp))
            if timestamp
            else datetime.now(UTC)
        )
        if received.tzinfo is None:
            received = received.replace(tzinfo=UTC)
        return cls(
            event_id=str(raw.get("event_id") or ""),
            received_at=received.astimezone(UTC),
            app=str(raw.get("app") or ""),
            title=str(raw.get("title") or ""),
            text=str(raw.get("text") or ""),
            big_text=str(raw.get("big_text") or raw.get("big") or ""),
            notification_timestamp=str(raw.get("notification_timestamp") or raw.get("time") or ""),
            source=str(raw.get("source") or "rio"),
        )


@dataclass(frozen=True, slots=True)
class ExternalSignalEvent:
    """One parsed instruction; prices remain provider prices until broker validation."""

    envelope: NotificationEnvelope
    kind: ExternalSignalKind
    symbol_alias: str | None = None
    direction: Direction | None = None
    entry_low: float | None = None
    entry_high: float | None = None
    stop_loss: float | None = None
    take_profits: tuple[float, ...] = ()
    move_stop_to: float | None = None
    reason: str = ""

    @property
    def complete_entry(self) -> bool:
        return bool(
            self.kind is ExternalSignalKind.NEW
            and self.symbol_alias
            and self.direction is not None
            and self.stop_loss
            and self.take_profits
        )

    @property
    def provider_entry(self) -> float | None:
        if self.entry_low is None:
            return None
        if self.entry_high is None:
            return self.entry_low
        return (self.entry_low + self.entry_high) / 2.0

    def safe_dict(self) -> dict[str, object]:
        return {
            "event_id": self.envelope.event_id,
            "received_at": self.envelope.received_at.isoformat(),
            "source": self.envelope.source,
            "kind": self.kind.value,
            "symbol_alias": self.symbol_alias,
            "direction": self.direction.name if self.direction else None,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop_loss": self.stop_loss,
            "take_profits": list(self.take_profits),
            "move_stop_to": self.move_stop_to,
            "reason": self.reason,
            "raw": self.envelope.safe_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.safe_dict(), ensure_ascii=False, separators=(",", ":"))
