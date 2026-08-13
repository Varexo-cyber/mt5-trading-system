"""Create the phone bridge secret locally without committing or logging it."""

from __future__ import annotations

import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "config" / ".env"
KEY = "RIO_SIGNAL_TOKEN"


def main() -> int:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    token = ""
    lines = existing.splitlines()
    for line in lines:
        if line.strip().startswith(f"{KEY}="):
            token = line.split("=", 1)[1].strip()
            break
    if not token:
        token = secrets.token_urlsafe(36)
        with ENV_FILE.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(f"{KEY}={token}\n")
    print("Rio bridge secret (keep private; paste only into MacroDroid):")
    print(token)
    print()
    print("Stored in config\\.env. Restart Jarvis after setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
