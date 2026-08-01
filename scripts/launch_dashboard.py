"""Launch the local Streamlit dashboard on loopback only."""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"
DASHBOARD_URL = "http://127.0.0.1:8501"
HEALTH_URL = f"{DASHBOARD_URL}/_stcore/health"


def dashboard_is_running() -> bool:
    """Return whether the local Streamlit dashboard is already healthy."""

    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def main() -> int:
    if dashboard_is_running():
        webbrowser.open(DASHBOARD_URL)
        return 0

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
