"""Put a secret into config/.env without it passing through shell history.

The obvious way to add a connection string is to echo it into the file. That
works and it leaves the secret in three places nobody thinks about: the shell's
history file, the terminal scrollback, and — on Windows — the console host's
buffer, which survives the window being closed. A database password is then
recoverable by anyone who later gets at that machine, long after the operator
believes they only put it in one place.

This prompts for the value instead. Nothing is echoed, nothing is passed as an
argument, and the file is created with owner-only permissions.

    python scripts/set_secret.py NEON_DATABASE_URL
    python scripts/set_secret.py ANTHROPIC_API_KEY

Replaces the key if it is already there rather than appending a second copy —
`python-dotenv` takes the last occurrence, so a duplicate is a configuration
that changes meaning depending on where in the file somebody looks.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import stat
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "config" / ".env"

#: Keys this is willing to write. A typo in a key name produces a variable
#: nothing reads, and the resulting failure ("the brain is not connected") does
#: not point anywhere near the cause.
KNOWN = {
    "NEON_DATABASE_URL": "Postgres connection string for the long-term memory",
    "ANTHROPIC_API_KEY": "Claude API key",
    "OPENAI_API_KEY": "OpenAI API key",
    "MT5_LOGIN": "MetaTrader account number",
    "MT5_PASSWORD": "MetaTrader password",
    "MT5_SERVER": "MetaTrader server name",
    "TELEGRAM_BOT_TOKEN": "Telegram bot token for alerts",
    "TELEGRAM_CHAT_ID": "Telegram chat id for alerts",
    "DISCORD_WEBHOOK_URL": "Discord webhook for alerts",
}


def write_secret(key: str, value: str, path: Path | None = None) -> None:
    """Set `key` in the env file, replacing any existing line for it.

    `path` is resolved at call time, not bound as a default. A default argument
    captures `ENV_PATH` once at import, so this would keep writing to the
    original location after anything reassigned it — and would then report
    success against a path it had not touched.
    """
    path = path or ENV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [line for line in existing if not line.startswith(f"{key}=")]
    kept.append(f"{key}={value}")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    # Owner read/write only. A no-op on Windows, where the file inherits the
    # directory's ACL, which is why this is not the only protection here.
    with_permissions(path)


def with_permissions(path: Path) -> None:
    """Owner read/write, where the filesystem has an opinion about it.

    Suppressed rather than checked: this is a hardening step, and a filesystem
    that will not take the mode is not a reason to refuse to store the secret
    the operator asked to store.
    """
    with contextlib.suppress(OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", nargs="?", default="", help=f"one of: {', '.join(sorted(KNOWN))}")
    args = parser.parse_args(argv)

    if not args.key:
        print()
        print("  Which secret? Pass one of:")
        for name, what in sorted(KNOWN.items()):
            print(f"    {name:<22}{what}")
        print()
        return 1

    key = args.key.strip().upper()
    if key not in KNOWN:
        print(f"\n  Unknown key {key!r}. Pass one of: {', '.join(sorted(KNOWN))}\n")
        return 1

    print()
    print(f"  {key} — {KNOWN[key]}")
    print(f"  Writing to {ENV_PATH}")
    print("  Nothing is shown as you type, and nothing reaches your shell history.")
    print()
    value = getpass("  Paste the value: ").strip()
    if not value:
        print("\n  Nothing entered; the file is unchanged.\n")
        return 1

    write_secret(key, value)

    # Enough to prove the right thing landed, not enough to be worth a
    # screenshot: the first few characters and the length.
    shape = f"{value[:12]}…{len(value)} characters" if len(value) > 12 else f"{len(value)} chars"
    print()
    print(f"  Saved. {key} = {shape}")
    if os.name != "nt":
        print(f"  Permissions: {stat.filemode(ENV_PATH.stat().st_mode)}")
    print()
    print("  config/.env is gitignored and stays that way. Never commit it,")
    print("  never paste it into a chat, and rotate anything that has been.")
    print()
    if key == "NEON_DATABASE_URL":
        print("  Next:  python scripts/verify_brain.py --stats")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
