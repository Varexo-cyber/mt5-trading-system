"""Writing a file without losing the old one, on Windows too.

Write-temp-then-rename is the standard way to avoid a half-written file. On
POSIX the rename is atomic and always succeeds. On Windows it raises
``PermissionError`` if *anything* else has the destination open — and this
project runs a Streamlit dashboard that reads exactly these files while the
trader writes them.

That is what happened in production: the dashboard had ``scan_activity.json``
open, ``os.replace`` was denied, and because the write sat on the path that
records a skipped trade, the exception propagated out of ``run_once`` and killed
the whole service. A telemetry file lost a reader's race and the account stopped
being managed.

So two rules here, and the second matters more than the first:

1. Retry briefly. The reader holds the file for microseconds; a few attempts
   spread over a moment clear essentially every collision.
2. Never raise. These are monitoring artefacts. A dashboard that misses one
   update is a cosmetic problem; a trading loop that exits because it could not
   write one is a real one. Failure is logged and swallowed.

Anything that must not be lost — the trade journal, the experimental contract —
belongs in SQLite or must call `write_json_atomic(..., required=True)` and
handle the exception deliberately.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from infra.logging import get_logger

log = get_logger(__name__)

#: Attempts, and the pause before each retry.
#:
#: The original five at a flat 50ms assumed "the contending reader holds the
#: handle only for the length of one read". True of one reader. The deck polls
#: several of these files on one- and two-second fragments from a separate
#: process, so on Windows the rename lands in a contended window often enough
#: to lose updates outright — a live session filled with `this update is lost`
#: once or twice per cycle.
#:
#: Exponential now, capped so one slow write cannot stall a caller for long:
#: 0.05, 0.1, 0.2, then 0.4 repeated. Eight attempts is about 2.3s worst case
#: and almost always returns on the first or second.
_ATTEMPTS = 8
_BACKOFF_SECONDS = 0.05
_BACKOFF_CAP_SECONDS = 0.4


def write_json_atomic(path: Path, payload: Any, *, required: bool = False) -> bool:
    """Serialise `payload` to `path`. Returns whether it landed.

    With `required=True` the final failure is raised instead of swallowed. Use
    that only where losing the write would corrupt state rather than a display.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        log.warning(
            "could not write temporary file",
            extra={"event": "atomic_write_failed", "path": str(path), "stage": "write"},
        )
        if required:
            raise
        return False

    last: OSError | None = None
    for attempt in range(_ATTEMPTS):
        try:
            temporary.replace(path)
        except PermissionError as exc:
            # Windows only: a reader has the destination open. Wait it out.
            last = exc
            time.sleep(min(_BACKOFF_SECONDS * 2**attempt, _BACKOFF_CAP_SECONDS))
        except OSError as exc:
            last = exc
            break
        else:
            return True

    log.warning(
        "could not replace file after retries; this update is lost",
        extra={
            "event": "atomic_write_failed",
            "path": str(path),
            "stage": "replace",
            "attempts": _ATTEMPTS,
            "reason": type(last).__name__ if last else "unknown",
        },
    )
    temporary.unlink(missing_ok=True)
    if required and last is not None:
        raise last
    return False
