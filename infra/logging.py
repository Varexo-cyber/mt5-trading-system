"""Structured JSON logging.

Two sinks, deliberately different:

* **file** — one JSON object per line, machine-readable, never rotated away
  faster than the journal retention. This is the audit trail: every decision
  the system made must be reconstructable from it.
* **console** — human-readable, for watching a live session.

Every record carries the contextual fields bound via `bind_context()`, so a
whole analysis cycle can be filtered out of the log by its `cycle_id`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import sys
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Ambient key/values merged into every log record on this task/thread.
_log_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "log_context", default=None
)


def _context() -> dict[str, Any]:
    return _log_context.get() or {}


_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """Attach fields to every log record emitted inside this block."""
    current = _context()
    token = _log_context.set({**current, **fields})
    try:
        yield
    finally:
        _log_context.reset(token)


def new_cycle_id() -> str:
    """Short correlation id for one analysis cycle."""
    return uuid.uuid4().hex[:12]


class JsonFormatter(logging.Formatter):
    """Renders records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_context())
        # Anything passed as logger.info("...", extra={...}) lands here.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["error"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_value),
                "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            }
        return json.dumps(payload, default=_fallback, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable format, with bound context appended."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        base = f"{ts} {record.levelname:<8} {record.name:<28} {record.getMessage()}"
        ctx = _context()
        if ctx:
            base += "  " + " ".join(f"{k}={v}" for k, v in ctx.items())
        if record.exc_info:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return base


def _fallback(obj: Any) -> str:
    """JSON encoder of last resort — never let logging raise."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return repr(obj)


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: Path | str = "logs",
    filename: str = "trading.jsonl",
    console: bool = True,
    console_level: str | None = None,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 20,
) -> logging.Logger:
    """Configure the root logger. Idempotent — safe to call twice."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers do the filtering
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.handlers.RotatingFileHandler(
        log_path / filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(getattr(logging, (console_level or level).upper()))
        stream.setFormatter(ConsoleFormatter())
        root.addHandler(stream)

    # These are chatty and tell us nothing we act on.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
