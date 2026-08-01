"""Launch the local Streamlit dashboard on loopback only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "false",
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
