"""A self-check that invents faults is worse than none at all.

Both of the bugs pinned here were found by running it on the VPS, and both had
the same shape: a check reporting FAIL on a perfectly healthy system. That is
the failure mode that matters for this script — not missing a problem, but
crying wolf until the operator stops reading it, and then missing a problem.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


@pytest.fixture
def module(monkeypatch, tmp_path):  # type: ignore[no-untyped-def]
    """Loaded with ROOT pointed at a scratch directory, so no check reads the
    developer's own journal, calendar or STOP file."""
    spec = importlib.util.spec_from_file_location("selfcheck", ROOT / "scripts" / "selfcheck.py")
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "selfcheck", loaded)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(loaded, "ROOT", tmp_path)
    # The calendar check loads the real configuration to read the cache path
    # and the staleness limit, so the scratch root needs those two files. Real
    # copies rather than stubs: the limit under test is the one the account
    # actually runs, and a stub would let it drift.
    import shutil

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "eightcap.yaml"):
        shutil.copy(ROOT / "config" / name, tmp_path / "config" / name)
    return loaded


def heartbeat(tmp_path: Path, *, minutes_ago: float, ended: bool = False) -> None:
    from monitoring.operation_ledger import LEDGER_FILENAME

    path = tmp_path / "runtime" / LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    path.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "id": "x",
                        "operation": "experimental_live",
                        "started_at": seen,
                        "last_seen_at": seen,
                        "ended_at": seen if ended else None,
                        "cycles": 120,
                    }
                ]
            }
        )
    )


def journal(tmp_path: Path, *, with_schema: bool = True, open_trades: int = 0) -> None:
    path = tmp_path / "journal" / "trading.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        if not with_schema:
            return
        conn.execute("CREATE TABLE trades (closed_at TEXT, ticket INTEGER, opened_at TEXT)")
        conn.execute("CREATE TABLE analysis_cycles (ts TEXT)")
        for n in range(open_trades):
            conn.execute(
                "INSERT INTO trades (closed_at, ticket, opened_at) VALUES (NULL, ?, ?)",
                (n + 1, NOW.isoformat()),
            )
        conn.commit()


def calendar(tmp_path: Path, *, hours_old: float) -> None:
    import os

    path = tmp_path / "data" / "calendar" / "cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    stamp = (NOW - timedelta(hours=hours_old)).timestamp()
    os.utime(path, (stamp, stamp))


class TestTheJournalCheck:
    def test_the_table_it_counts_is_the_one_that_exists(self, module, tmp_path) -> None:
        """It queried `cycles`. The table is `analysis_cycles`, so the first
        real run reported 'journal FAIL: no such table: cycles' against a
        perfectly healthy journal."""
        journal(tmp_path, open_trades=2)

        check = module.check_journal(NOW)

        assert check.state == module.OK
        assert "2 open" in check.detail

    def test_a_file_with_no_schema_reads_as_never_started(self, module, tmp_path) -> None:
        """Different from corrupt, and the operator's next step differs."""
        journal(tmp_path, with_schema=False)

        check = module.check_journal(NOW)

        assert check.state == module.WARN
        assert "never started" in check.detail

    def test_no_journal_at_all_is_a_warning_not_a_failure(self, module) -> None:
        assert module.check_journal(NOW).state == module.WARN

    def test_something_that_is_not_a_database_fails(self, module, tmp_path) -> None:
        path = tmp_path / "journal" / "trading.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not a database")

        assert module.check_journal(NOW).state == module.FAIL


class TestTheCalendarCheckReadsTheRunnerFirst:
    """The second cry-wolf bug. A stale calendar with the runner up is an
    emergency; the same staleness with the runner down is arithmetic, because
    the thing that refreshes it is not running."""

    def test_stale_with_the_runner_down_is_not_a_failure(self, module, tmp_path) -> None:
        calendar(tmp_path, hours_old=12.8)  # no heartbeat file at all

        check = module.check_calendar(NOW)

        assert check.state == module.OK
        assert "runner being down" in check.detail

    def test_stale_with_the_runner_up_is_a_failure(self, module, tmp_path) -> None:
        calendar(tmp_path, hours_old=12.8)
        heartbeat(tmp_path, minutes_ago=1)

        check = module.check_calendar(NOW)

        assert check.state == module.FAIL
        assert "every entry is being blocked" in check.detail

    def test_a_fresh_calendar_passes_either_way(self, module, tmp_path) -> None:
        calendar(tmp_path, hours_old=0.5)

        assert module.check_calendar(NOW).state == module.OK

    def test_a_missing_calendar_always_fails(self, module) -> None:
        """No cache at all is not explained by the runner being down: the file
        should still be there from the last time it ran."""
        assert module.check_calendar(NOW).state == module.FAIL


class TestTheHeartbeat:
    def test_a_live_runner_is_ok(self, module, tmp_path) -> None:
        heartbeat(tmp_path, minutes_ago=2)

        assert module.check_heartbeat(NOW).state == module.OK

    def test_quiet_for_a_while_warns(self, module, tmp_path) -> None:
        heartbeat(tmp_path, minutes_ago=20)

        assert module.check_heartbeat(NOW).state == module.WARN

    def test_long_silence_fails(self, module, tmp_path) -> None:
        """A scan cycle takes under a minute and the guard runs every second.
        An hour of nothing is not a slow market, it is a stopped process."""
        heartbeat(tmp_path, minutes_ago=90)

        check = module.check_heartbeat(NOW)

        assert check.state == module.FAIL
        assert "not running" in check.detail

    def test_a_clean_shutdown_warns_rather_than_fails(self, module, tmp_path) -> None:
        heartbeat(tmp_path, minutes_ago=300, ended=True)

        check = module.check_heartbeat(NOW)

        assert check.state == module.WARN
        assert "ended cleanly" in check.detail


class TestTheKillSwitch:
    def test_a_stop_file_is_reported(self, module, tmp_path) -> None:
        """Deliberately off is not healthy, and reporting it as healthy would
        hide the reason nothing is trading."""
        (tmp_path / "STOP").write_text("")

        assert module.check_kill_switch(NOW).state == module.WARN


class TestRunningAllOfThem:
    def test_a_check_that_raises_becomes_a_finding(self, module, monkeypatch) -> None:
        """A self-check that dies on its first surprise reports nothing about
        the six layers behind it, which is the opposite of the job."""

        def explode(_now):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        monkeypatch.setattr(module, "CHECKS", (explode, module.check_kill_switch))

        results = module.run_checks(NOW)

        assert len(results) == 2
        assert results[0].state == module.FAIL
        assert "boom" in results[0].detail
        assert results[1].state == module.OK

    def test_the_exit_code_distinguishes_healthy_from_not(self, module, monkeypatch) -> None:
        """It is what Task Scheduler records, and a task sitting on 0x1 for two
        days is visible without anybody reading a log."""
        monkeypatch.setattr(module, "CHECKS", (module.check_kill_switch,))
        assert module.main(["--quiet"]) == 0

        monkeypatch.setattr(module, "CHECKS", (lambda _now: module.Check("x", module.FAIL, "bad"),))
        assert module.main(["--quiet"]) == 1

    def test_quiet_prints_nothing_when_everything_passes(self, module, monkeypatch, capsys) -> None:
        monkeypatch.setattr(module, "CHECKS", (module.check_kill_switch,))

        module.main(["--quiet"])

        assert capsys.readouterr().out == ""


class TestTheRuntimeCalendarCacheIsNotTracked:
    """`data/calendar/cache.json` is rewritten every thirty minutes by the
    running system. Tracked in git it left the working tree permanently dirty,
    which is cosmetic right up until `auto_update.py` refuses to touch a dirty
    tree — and then it blocks every unattended update forever."""

    def test_it_is_gitignored(self) -> None:
        assert "data/calendar/cache.json" in (ROOT / ".gitignore").read_text()

    def test_the_archive_is_still_committed(self) -> None:
        """Deliberately kept: it is the historical record the replay reads, it
        grows only when the weekly task appends, and it should survive a
        rebuilt VPS."""
        assert (ROOT / "data" / "calendar" / "archive.json").exists()


class TestTheHeartbeatFileIsNamedInOnePlace:
    """Three of the four bugs in this script were a name guessed wrong — a
    table, a config attribute, and this file. Two modules writing and reading
    the same path from two string literals is how the heartbeat check spent its
    first day reporting "has it ever started?" at a system with thirty thousand
    decisions in its journal."""

    def test_the_runner_and_the_check_use_the_same_constant(self) -> None:
        runner = (ROOT / "runner" / "service.py").read_text(encoding="utf-8")
        check = (ROOT / "scripts" / "selfcheck.py").read_text(encoding="utf-8")

        assert 'runtime" / LEDGER_FILENAME' in runner
        assert 'runtime" / LEDGER_FILENAME' in check
        assert "operations.json" not in check, "the wrong name must not come back"

    def test_the_constant_matches_what_is_on_disk_in_practice(self) -> None:
        from monitoring.operation_ledger import LEDGER_FILENAME

        assert LEDGER_FILENAME == "operation_history.json"
