"""Update the system unattended, and put it back if the update is bad.

An automatic update on a machine trading real money is a genuinely dangerous
thing to build. It takes code nobody has looked at on this machine and puts it
in charge of an account, on a schedule, while the operator is asleep. So the
question is not "can this pull" — `update_repo.py` already does — it is what
has to be true before and after for that to be safe.

Four rules, and each one exists because skipping it has an obvious failure:

1. **Never while a position is open.** Swapping the management rules underneath
   a live trade means the position was opened by one set of rules and will be
   closed by another. Whatever that trade then does teaches nothing, and if it
   goes wrong there is no way to say which version did it.

2. **Never inside a trading session.** Even flat, a restart in the middle of
   the London open is minutes of scanning lost at the busiest hour. The default
   window is the weekend, which is when this account is flat by design anyway.

3. **Verify before trusting.** After the pull the configuration must load and
   the test suite must pass ON THIS MACHINE. A commit that is green in CI can
   still be wrong here — a missing dependency, a Windows path, an MT5 build.

4. **Roll back on failure.** If either check fails, the working tree goes back
   to the commit it was on and the operator is told. An unattended update that
   can only go forward is a way to wake up to a system that will not start.

    python scripts/auto_update.py --check          # what would it do
    python scripts/auto_update.py                  # apply, with every guard
    python scripts/auto_update.py --force          # ignore the timing guards

Exit codes: 0 applied or nothing to do, 1 refused by a guard, 2 rolled back.
The distinction matters for a scheduled task — 1 is the system working, 2 is
something to look at.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Days and UTC hours during which an unattended update may run. Saturday and
#: Sunday: FX is closed, the account is flat because `_evening_flatten` closed
#: everything on Friday, and a restart costs nothing.
SAFE_DAYS = (5, 6)  # Saturday, Sunday
SAFE_HOURS = range(6, 20)

#: How long the suite may take before this gives up and rolls back. It runs in
#: about a minute; ten minutes means something is hanging.
TEST_TIMEOUT = 600


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why an update did not happen. Not an error — usually the guards working."""

    reason: str


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False, timeout=120
    )


def current_commit() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def open_positions() -> int | None:
    """How many positions the journal thinks are live. None when unreadable.

    From the journal rather than from MT5, deliberately: this must work with
    the terminal closed, and a journal that says a position is open when the
    broker disagrees is itself a reason not to update anything.
    """
    import sqlite3

    path = ROOT / "journal" / "trading.db"
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE closed_at IS NULL AND ticket IS NOT NULL"
                ).fetchone()[0]
            )
    except sqlite3.Error:
        return None


def timing_is_safe(now: datetime) -> bool:
    return now.weekday() in SAFE_DAYS and now.hour in SAFE_HOURS


def guards(now: datetime) -> Refusal | None:
    """Everything that must be true before an unattended pull."""
    if (ROOT / "STOP").exists():
        return Refusal("the STOP file is present; the operator has halted the system")

    live = open_positions()
    if live is None:
        return Refusal("the journal cannot be read, so it is unknown whether anything is open")
    if live:
        return Refusal(f"{live} position(s) open; rules must not change under a live trade")

    dirty = _git("status", "--porcelain").stdout.strip()
    if dirty:
        return Refusal(f"the working tree has local changes:\n{dirty}")

    if not timing_is_safe(now):
        return Refusal(f"{now:%A %H:%M} UTC is outside the weekend maintenance window")
    return None


def verify() -> tuple[bool, str]:
    """Does this machine agree the new code is good?

    Config first because it is two seconds and catches the common case, then
    the suite. Both ON THIS MACHINE: a commit green in CI can still be wrong
    here, and that is the whole reason for verifying rather than trusting.
    """
    python = ROOT / ".venv-live" / "Scripts" / "python.exe"
    if not python.exists():  # a developer checkout, or POSIX
        python = Path(sys.executable)

    config = subprocess.run(
        [str(python), "main.py", "--overlay", "config/eightcap.yaml", "--check-config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if config.returncode != 0:
        return False, f"the configuration does not load:\n{config.stdout}\n{config.stderr}"

    tests = subprocess.run(
        [str(python), "-m", "pytest", "-q", "--timeout", "120"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=TEST_TIMEOUT,
    )
    if tests.returncode != 0:
        tail = "\n".join(tests.stdout.strip().splitlines()[-25:])
        return False, f"the test suite failed:\n{tail}"
    return True, tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else "tests passed"


def record(payload: dict[str, object]) -> None:
    path = ROOT / "runtime" / "auto_update.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        history.append(payload)
        path.write_text(json.dumps(history[-50:], indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass  # the console already said it; failing to file it is not a finding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report, change nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="skip the timing window. Never skips the open-position or STOP guards",
    )
    args = parser.parse_args(argv)
    now = datetime.now(UTC)

    print()
    print(f"  AUTO-UPDATE — {now:%Y-%m-%d %H:%M} UTC")
    print("  " + "-" * 70)

    before = current_commit()
    print(f"  on        {before[:8]}")

    refusal = guards(now)
    # --force waives the calendar only. An open position or a STOP file is
    # never waived: those are about safety, not about convenience.
    if refusal and args.force and "maintenance window" in refusal.reason:
        print(f"  forced    ignoring: {refusal.reason}")
        refusal = None
    if refusal:
        print(f"  refused   {refusal.reason}")
        print()
        return 1

    fetched = _git("fetch", "--quiet", "origin")
    if fetched.returncode != 0:
        print(f"  refused   git fetch failed: {fetched.stderr.strip()}")
        print()
        return 1

    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    behind = _git("rev-list", "--count", f"HEAD..origin/{branch}").stdout.strip()
    if behind in ("", "0"):
        print("  nothing   already up to date")
        print()
        return 0
    print(f"  behind    {behind} commit(s) on {branch}")

    if args.check:
        print(_git("log", "--oneline", f"HEAD..origin/{branch}").stdout.rstrip())
        print("  check     --check given, nothing was changed")
        print()
        return 0

    pull = _git("merge", "--ff-only", f"origin/{branch}")
    if pull.returncode != 0:
        print(f"  refused   not a fast-forward: {pull.stderr.strip()}")
        print()
        return 1
    after = current_commit()
    print(f"  updated   {before[:8]} -> {after[:8]}")

    print("  verifying this machine agrees the new code is good ...")
    good, detail = verify()
    if good:
        print(f"  verified  {detail}")
        print()
        record({"at": now.isoformat(), "from": before, "to": after, "result": "applied"})
        print("  Applied. Restart the runner to pick it up.")
        print()
        return 0

    print(f"  FAILED    {detail}")
    reset = _git("reset", "--hard", before)
    rolled = reset.returncode == 0 and current_commit() == before
    print(f"  rollback  {'back on ' + before[:8] if rolled else 'ROLLBACK FAILED — do not start'}")
    print()
    record(
        {
            "at": now.isoformat(),
            "from": before,
            "to": after,
            "result": "rolled_back" if rolled else "rollback_failed",
            "why": detail[:2000],
        }
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
