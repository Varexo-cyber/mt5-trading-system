"""Manual kill switch.

Dropping a file named `STOP` in the project root makes the system flatten
everything and halt. It is intentionally the dumbest possible mechanism: no
API, no auth, no network. If the process is alive at all, this works — and it
keeps working when Telegram is down, the broker's web terminal is unreachable,
and you are on a phone with one bar of signal.

The switch is checked at the top of every loop iteration AND immediately before
any order is sent, so a flip during a slow analysis cycle still takes effect
before money moves.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from infra.logging import get_logger

log = get_logger(__name__)

DEFAULT_FILENAME = "STOP"


class KillSwitch:
    """Filesystem-backed halt flag."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._announced = False

    @classmethod
    def in_dir(cls, directory: Path | str, filename: str = DEFAULT_FILENAME) -> KillSwitch:
        return cls(Path(directory) / filename)

    def is_engaged(self) -> bool:
        """True if the STOP file exists. Logs the transition exactly once."""
        engaged = self.path.exists()
        if engaged and not self._announced:
            log.critical(
                "kill switch engaged", extra={"event": "kill_switch", "path": str(self.path)}
            )
            self._announced = True
        elif not engaged and self._announced:
            log.warning("kill switch cleared", extra={"event": "kill_switch_cleared"})
            self._announced = False
        return engaged

    def reason(self) -> str:
        """Contents of the STOP file, if whoever tripped it left a note."""
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def engage(self, reason: str = "") -> None:
        """Trip the switch from inside the system (e.g. circuit breaker)."""
        stamp = datetime.now(UTC).isoformat()
        self.path.write_text(f"{stamp} {reason}\n".strip() + "\n", encoding="utf-8")
        log.critical(
            "kill switch tripped by system",
            extra={"event": "kill_switch_tripped", "reason": reason},
        )

    def clear(self) -> None:
        """Remove the STOP file. Only ever called by a human, never on a timer."""
        self.path.unlink(missing_ok=True)
        self._announced = False
