"""OS-backed single-instance lock for the real-money service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class AlreadyRunningError(RuntimeError):
    """Raised when another process already owns the Jarvis lock."""


class ProcessLock:
    """Hold an advisory byte lock for the complete process lifetime.

    A PID file is telemetry and can be overwritten. This lock is enforced by
    Windows/the OS, is released automatically after a crash, and therefore
    closes the race that allowed two scanners to open the same symbol at once.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b", buffering=0)
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - production runs on Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise AlreadyRunningError(
                f"another Jarvis process already owns {self.path.name}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - production runs on Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> ProcessLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
