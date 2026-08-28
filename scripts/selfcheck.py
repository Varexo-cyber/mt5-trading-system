"""Is this thing actually running, and is it healthy?

The failure that costs the most is not a crash. A crash is loud: the window
closes, the dashboard goes blank, somebody notices within the hour. The
expensive failure is the quiet one — the runner still up, the log still
scrolling, and one layer underneath it dead. A calendar that stopped
refreshing, a database that stopped accepting writes, an MT5 terminal that
logged itself out, a feed set that has answered nothing since Tuesday. From
the outside all of those look exactly like a slow market.

So this asks each layer directly and reports one line per answer. It reads
only; it sends no orders, spends nothing on the Claude API, and writes nothing
except its own status file.

    python scripts/selfcheck.py               # print the report
    python scripts/selfcheck.py --quiet       # only print when something is wrong
    python scripts/selfcheck.py --alert       # and send it, if alerts are set up

Exit code is 0 when everything passed, 1 when anything is WARN or FAIL. That is
what makes it useful as a scheduled task: Windows records the code, and a task
that has been returning 1 for two days is visible in Task Scheduler without
anybody reading a log.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import Brain, build_brain
from config.loader import load_credentials, load_settings
from monitoring.operation_ledger import LEDGER_FILENAME

#: How long the runner may go without touching the ledger before that is a
#: problem. A scan cycle takes under a minute and the guard runs every second,
#: so ten minutes of silence is not a slow market — it is a stopped process.
HEARTBEAT_WARN = timedelta(minutes=10)
HEARTBEAT_FAIL = timedelta(minutes=45)

OK, WARN, FAIL = "ok", "WARN", "FAIL"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    state: str
    detail: str

    @property
    def bad(self) -> bool:
        return self.state != OK


def _age(moment: datetime | None, now: datetime) -> timedelta | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return now - moment


def _minutes(delta: timedelta) -> str:
    total = delta.total_seconds() / 60.0
    if total < 90:
        return f"{total:.0f} min"
    if total < 60 * 48:
        return f"{total / 60:.1f} h"
    return f"{total / 1440:.1f} days"


def check_kill_switch(now: datetime) -> Check:
    """The operator's own stop file. Present means the system is deliberately
    off, and reporting that as healthy would be worse than reporting nothing."""
    del now
    stop = ROOT / "STOP"
    if stop.exists():
        return Check("kill switch", WARN, "STOP file present; the system is halted on purpose")
    return Check("kill switch", OK, "no STOP file")


def check_heartbeat(now: datetime) -> Check:
    """When did the runner last complete a cycle?

    Read from the operation ledger rather than from a log file, because a log
    keeps growing whether or not anything useful happened — a process stuck in
    a retry loop writes more, not less.
    """
    path = ROOT / "runtime" / LEDGER_FILENAME
    if not path.exists():
        return Check("runner", WARN, "no operations ledger yet; has it ever started?")
    try:
        sessions = json.loads(path.read_text(encoding="utf-8")).get("sessions", [])
    except (OSError, ValueError) as exc:
        return Check("runner", FAIL, f"operations ledger unreadable: {exc}")
    if not sessions:
        return Check("runner", WARN, "the ledger has no sessions in it")

    latest = sessions[-1]
    seen = _age(datetime.fromisoformat(str(latest["last_seen_at"])), now)
    cycles = latest.get("cycles", 0)
    mode = latest.get("operation", "?")
    if latest.get("ended_at"):
        return Check("runner", WARN, f"last {mode} session ended cleanly after {cycles} cycles")
    if seen is None:
        return Check("runner", WARN, "no heartbeat recorded")
    if seen > HEARTBEAT_FAIL:
        return Check("runner", FAIL, f"{mode}: nothing for {_minutes(seen)} — it is not running")
    if seen > HEARTBEAT_WARN:
        return Check("runner", WARN, f"{mode}: quiet for {_minutes(seen)} after {cycles} cycles")
    return Check("runner", OK, f"{mode}: {cycles} cycles, last seen {_minutes(seen)} ago")


def check_journal(now: datetime) -> Check:
    """Is the journal there, readable, and has anything happened in it?"""
    path = ROOT / "journal" / "trading.db"
    if not path.exists():
        return Check("journal", WARN, "no journal yet")
    try:
        # Read-only, so a running system's writer is never blocked by this.
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            # A file with no schema is a different situation from a corrupt
            # one, and "no such table: trades" is a poor way to say "this has
            # never been initialised". The operator's next step differs.
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades'"
            ).fetchone():
                return Check("journal", WARN, "the journal file has no tables yet; never started")
            open_trades = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE closed_at IS NULL AND ticket IS NOT NULL"
            ).fetchone()[0]
            last = conn.execute("SELECT MAX(opened_at) FROM trades").fetchone()[0]
            # `analysis_cycles`, not `cycles`. The first run of this check
            # reported "journal FAIL: no such table: cycles" on a perfectly
            # healthy journal — a self-check that invents faults is worse than
            # no self-check, because the next real one gets ignored with it.
            cycles = conn.execute(
                "SELECT COUNT(*) FROM analysis_cycles WHERE ts > ?",
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()[0]
    except sqlite3.Error as exc:
        return Check("journal", FAIL, f"cannot read: {exc}")

    since = "never" if not last else f"{_minutes(_age(datetime.fromisoformat(last), now))} ago"
    return Check(
        "journal",
        OK,
        f"{open_trades} open, last entry {since}, {cycles} decisions in 24h",
    )


def runner_is_alive(now: datetime) -> bool:
    """Is anything actually driving the system right now?

    Several checks below only mean something in the light of this. A stale
    calendar with the runner up is an emergency — every entry is blocked and
    nobody knows. The same staleness with the runner down is arithmetic: the
    thing that refreshes it is not running.
    """
    return check_heartbeat(now).state == OK


def check_calendar(now: datetime) -> Check:
    """The one layer that stops all trading when it goes dark, by design.

    Judged against whether the runner is up. The first run of this check
    reported FAIL on a Saturday with the system deliberately switched off —
    true in the letter ("entries would be blocked") and useless, because
    nothing was trying to enter and the refresher was not running. A check
    that cries wolf on a quiet weekend is a check that gets ignored on the
    Tuesday it matters.
    """
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    path = ROOT / settings.filters.news.cache_path
    if not path.exists():
        return Check("calendar", FAIL, "no cached calendar; the news filter blocks every entry")
    age = _age(datetime.fromtimestamp(path.stat().st_mtime, UTC), now)
    limit = timedelta(minutes=settings.filters.news.max_calendar_age_minutes)
    if age is not None and age > limit:
        if not runner_is_alive(now):
            return Check(
                "calendar",
                OK,
                f"{_minutes(age)} old, which follows from the runner being down",
            )
        return Check(
            "calendar",
            FAIL,
            f"{_minutes(age)} old against a {_minutes(limit)} limit "
            f"while the runner is up — every entry is being blocked",
        )
    return Check("calendar", OK, f"refreshed {_minutes(age)} ago" if age else "fresh")


def check_brain(now: datetime) -> Check:
    """Is the long-term memory reachable, and is anything landing in it?"""
    del now
    load_credentials(required=False)
    brain = build_brain(account="selfcheck")
    if not isinstance(brain, Brain):
        return Check("brain", WARN, "no NEON_DATABASE_URL or no psycopg; memory is local only")
    try:
        row = brain._run("SELECT COUNT(*), MAX(decided_at) FROM decisions", fetch="one")
    finally:
        brain.close()
    if row is None:
        return Check("brain", FAIL, f"unreachable: {brain.status.last_error}")
    count, newest = row
    if not count:
        return Check("brain", WARN, "connected, but nothing has been written yet")
    return Check("brain", OK, f"{count} decisions stored, newest {newest:%Y-%m-%d %H:%M} UTC")


def check_headlines(now: datetime) -> Check:
    """Are the wires answering, and is anything being kept?"""
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    if not settings.filters.headlines.enabled:
        return Check("headlines", WARN, "layer disabled; run verify_newsfeed.py and enable it")
    load_credentials(required=False)
    brain = build_brain(account="selfcheck")
    if not isinstance(brain, Brain):
        return Check("headlines", WARN, "enabled, but no database to check what it stored")
    try:
        row = brain._run(
            "SELECT COUNT(*) FROM headlines WHERE seen_at > %s",
            (now - timedelta(hours=6),),
            fetch="one",
        )
    finally:
        brain.close()
    if row is None:
        return Check("headlines", WARN, "cannot read the headline table")
    if not row[0]:
        # Same reasoning as the calendar: the runner is what polls the feeds,
        # so with it down an empty table is arithmetic rather than a fault.
        if not runner_is_alive(now):
            return Check(
                "headlines", OK, "nothing stored, which follows from the runner being down"
            )
        return Check("headlines", FAIL, "no headline stored in six hours; the feeds are dark")
    return Check("headlines", OK, f"{row[0]} headlines in the last six hours")


def check_disk(now: datetime) -> Check:
    """A full disk stops the journal writing, which stops everything safely but
    silently."""
    del now
    import shutil

    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 0.5:
        return Check("disk", FAIL, f"{free_gb:.2f} GB free; the journal cannot write")
    if free_gb < 2.0:
        return Check("disk", WARN, f"{free_gb:.1f} GB free")
    return Check("disk", OK, f"{free_gb:.0f} GB free")


def check_breakers(now: datetime) -> Check:
    """Has a detector switched ITSELF off while nobody was looking?

    THE ONE STATE CHANGE NOTHING ANNOUNCED. `section_breakers` exists so a
    module that starts losing stops on its own -- eight losers in a row for the
    swing pair, ten for the fast ones. That is the entire reason it is
    acceptable to run detectors whose record is thin or unmeasured.

    And a tripped breaker looks, from outside, exactly like a quiet market. The
    operator's own question was "how do I know everything is working", and
    "four of your five detectors switched themselves off overnight" is the
    single most important answer nothing was able to give him.

    A module with no verdict yet is not a finding: below `minimum_trades` the
    breaker deliberately says nothing, and reporting that silence as either
    health or failure would be a lie.
    """
    # The only check here that needs no clock: a breaker's verdict is over the
    # last N closed trades, not over a window of time.
    del now
    path = ROOT / "journal" / "trading.db"
    if not path.exists():
        return Check("breakers", WARN, "no journal yet; nothing to judge")
    try:
        from risk.section_breaker import tripped_modules

        # THE OVERLAY, like every other check in this file. Without it this
        # reads the base config, where `section_breakers` is empty -- so the
        # check reported "no section breakers configured" on an account that
        # has five of them armed. A health check that describes a different
        # config than the one running is worse than no check.
        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        breakers = settings.risk.section_breakers
        if not breakers:
            return Check("breakers", WARN, "no section breakers configured")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            tripped = tripped_modules(conn, breakers)
    except Exception as exc:  # noqa: BLE001 - a check that cannot read must say so
        return Check("breakers", WARN, f"cannot judge: {type(exc).__name__}: {exc}")

    if not tripped:
        return Check("breakers", OK, f"{len(breakers)} armed, none tripped")
    # A tripped breaker on a LIVE module is the loud case: that detector is not
    # trading, and the account is quieter than the operator thinks it is.
    live = set(settings.analysis.confluence.live_enabled_modules)
    names = ", ".join(sorted(tripped))
    state = FAIL if set(tripped) & live else WARN
    return Check("breakers", state, f"switched themselves off: {names}")


def check_trading(now: datetime) -> Check:
    """Is the funnel producing trades, or only decisions?

    `check_journal` counts decisions, and decisions are the thing that keeps
    happening whether or not anything works: 66,140 of them in twelve hours
    produced two trades, and every layer reported healthy throughout. That is
    the failure this check exists for -- a system that is up, busy, logging,
    and not trading.

    It states the ratio rather than judging it, because there is no correct
    number of trades per day and inventing one would make this alarm on a
    legitimately quiet Sunday. WARN only when a full day of decisions produced
    nothing at all, which is the case worth walking over to look at.
    """
    path = ROOT / "journal" / "trading.db"
    if not path.exists():
        return Check("trading", WARN, "no journal yet")
    since = (now - timedelta(hours=24)).isoformat()
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades'"
            ).fetchone():
                return Check("trading", WARN, "journal has no tables yet")
            decisions = conn.execute(
                "SELECT COUNT(*) FROM analysis_cycles WHERE ts > ?", (since,)
            ).fetchone()[0]
            opened = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE opened_at > ?", (since,)
            ).fetchone()[0]
    except sqlite3.Error as exc:
        return Check("trading", FAIL, f"cannot read: {exc}")

    if not decisions:
        return Check("trading", WARN, "no decisions in 24h; is the runner scanning?")
    if not opened:
        return Check("trading", WARN, f"{decisions} decisions in 24h and 0 trades - run whynot.cmd")
    return Check("trading", OK, f"{opened} trades from {decisions} decisions in 24h")


CHECKS = (
    check_kill_switch,
    check_heartbeat,
    check_journal,
    check_trading,
    check_breakers,
    check_calendar,
    check_brain,
    check_headlines,
    check_disk,
)


def run_checks(now: datetime | None = None) -> list[Check]:
    """Every check, with a failure in one never stopping the rest.

    A self-check that dies on its first surprise reports nothing about the six
    layers behind it, which is the opposite of the job.
    """
    moment = now or datetime.now(UTC)
    results = []
    for check in CHECKS:
        try:
            results.append(check(moment))
        except Exception as exc:  # noqa: BLE001 - a broken check is a finding
            results.append(
                Check(check.__name__, FAIL, f"check raised: {type(exc).__name__}: {exc}")
            )
    return results


def render(checks: list[Check], now: datetime) -> str:
    worst = (
        FAIL
        if any(c.state == FAIL for c in checks)
        else (WARN if any(c.bad for c in checks) else OK)
    )
    lines = [
        "",
        f"  SELF-CHECK — {now:%Y-%m-%d %H:%M} UTC — {worst.upper()}",
        "  " + "-" * 72,
    ]
    lines.extend(f"  {c.name:<14}{c.state:<7}{c.detail}" for c in checks)
    lines.append("  " + "-" * 72)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print only when something is wrong")
    parser.add_argument("--alert", action="store_true", help="send failures to the alert channel")
    args = parser.parse_args(argv)

    now = datetime.now(UTC)
    checks = run_checks(now)
    bad = [c for c in checks if c.bad]
    report = render(checks, now)

    if not args.quiet or bad:
        print(report)

    # A status file so the dashboard and the next run can see the last result
    # without re-running every check.
    status = ROOT / "runtime" / "selfcheck.json"
    try:
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(
            json.dumps(
                {
                    "checked_at": now.isoformat(),
                    "worst": (
                        FAIL if any(c.state == FAIL for c in checks) else (WARN if bad else OK)
                    ),
                    "checks": [
                        {"name": c.name, "state": c.state, "detail": c.detail} for c in checks
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # the report already printed; failing to cache it is not a finding

    if args.alert and bad:
        try:
            from monitoring.alerts import AlertSender

            settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
            AlertSender(settings.monitoring).send(
                "Jarvis self-check: " + "; ".join(f"{c.name} {c.state} — {c.detail}" for c in bad)
            )
        except Exception as exc:  # noqa: BLE001 - alerting is best effort
            print(f"  (could not send the alert: {type(exc).__name__}: {exc})")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
