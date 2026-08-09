from __future__ import annotations

from pathlib import Path

import pytest

from infra.process_lock import AlreadyRunningError, ProcessLock


def test_second_jarvis_instance_cannot_take_the_same_lock(tmp_path: Path) -> None:
    first = ProcessLock(tmp_path / "jarvis.lock")
    second = ProcessLock(tmp_path / "jarvis.lock")
    first.acquire()
    try:
        with pytest.raises(AlreadyRunningError, match="another Jarvis"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()
