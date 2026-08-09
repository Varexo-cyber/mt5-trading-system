"""Unattended updates on a machine that trades real money.

The dangerous part of this script is not the pull. It is that it runs while
nobody is watching, on a system that has an open account, and it can leave
that system in a state that will not start. So the tests here are almost
entirely about the guards refusing, and about `--force` refusing to waive the
two that exist for safety rather than for convenience.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: A Saturday inside the maintenance window, and a Wednesday afternoon.
WEEKEND = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
MIDWEEK = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


@pytest.fixture
def module(monkeypatch):  # type: ignore[no-untyped-def]
    """Load the script as a module.

    Registered in `sys.modules` before it executes, and that is not optional:
    the script uses `from __future__ import annotations`, so `@dataclass` has
    to look its field types up by name at class-creation time and does that
    through `sys.modules[cls.__module__]`. Without the registration it finds
    `None` there and raises inside the decorator.
    """
    spec = importlib.util.spec_from_file_location(
        "auto_update", ROOT / "scripts" / "auto_update.py"
    )
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "auto_update", loaded)
    spec.loader.exec_module(loaded)
    return loaded


def journal_with(path: Path, open_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE trades (closed_at TEXT, ticket INTEGER)")
        for n in range(open_trades):
            conn.execute("INSERT INTO trades (closed_at, ticket) VALUES (NULL, ?)", (n + 1,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def clean(module, monkeypatch, tmp_path):  # type: ignore[no-untyped-def]
    """A repository where every guard passes, so each test breaks exactly one."""
    monkeypatch.setattr(module, "ROOT", tmp_path)
    journal_with(tmp_path / "journal" / "trading.db", 0)
    monkeypatch.setattr(module, "_git", fake_git())
    return module


class _Result:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def fake_git(*, behind: str = "0", dirty: str = "", branch: str = "main", record=None):  # type: ignore[no-untyped-def]
    """A git that answers per subcommand.

    One stub returning the same string for everything is how an earlier version
    of these tests had `status --porcelain` answering "main" and the guard
    correctly reporting uncommitted changes. Dispatching keeps each answer
    meaning what git means by it.
    """

    def run(*args: str) -> _Result:
        if record is not None:
            record.append(args)
        if args[0] == "status":
            return _Result(dirty)
        if args[0] == "rev-list":
            return _Result(behind)
        if args[0] == "rev-parse" and "--abbrev-ref" in args:
            return _Result(branch)
        return _Result("")

    return run


class TestTheTimingWindow:
    def test_the_weekend_is_open(self, module) -> None:
        assert module.timing_is_safe(WEEKEND)

    def test_a_weekday_is_not(self, module) -> None:
        """Even flat, a restart during the London open costs minutes of
        scanning at the busiest hour of the day."""
        assert not module.timing_is_safe(MIDWEEK)

    def test_the_small_hours_are_not(self, module) -> None:
        """Nobody is awake to read a rollback message at 03:00."""
        assert not module.timing_is_safe(WEEKEND.replace(hour=3))


class TestTheGuards:
    def test_a_clean_flat_weekend_passes(self, clean) -> None:
        assert clean.guards(WEEKEND) is None

    def test_an_open_position_refuses(self, clean, tmp_path) -> None:
        """Swapping the management rules under a live trade means the position
        was opened by one set and will be closed by another. Whatever it then
        does teaches nothing."""
        journal_with(tmp_path / "journal" / "trading2.db", 1)
        (tmp_path / "journal" / "trading.db").unlink()
        (tmp_path / "journal" / "trading2.db").rename(tmp_path / "journal" / "trading.db")

        refusal = clean.guards(WEEKEND)

        assert refusal is not None and "1 position(s) open" in refusal.reason

    def test_an_unreadable_journal_refuses(self, clean, tmp_path) -> None:
        """Unknown is not the same as zero. A journal that cannot be read is
        itself a reason to change nothing."""
        (tmp_path / "journal" / "trading.db").write_text("not a database")

        refusal = clean.guards(WEEKEND)

        assert refusal is not None and "cannot be read" in refusal.reason

    def test_the_stop_file_refuses(self, clean, tmp_path) -> None:
        (tmp_path / "STOP").write_text("")

        refusal = clean.guards(WEEKEND)

        assert refusal is not None and "STOP file" in refusal.reason

    def test_local_changes_refuse(self, clean, monkeypatch) -> None:
        """A hard reset on rollback would destroy them."""
        monkeypatch.setattr(clean, "_git", fake_git(dirty=" M config/eightcap.yaml"))

        refusal = clean.guards(WEEKEND)

        assert refusal is not None and "local changes" in refusal.reason

    def test_a_weekday_refuses(self, clean) -> None:
        refusal = clean.guards(MIDWEEK)

        assert refusal is not None and "maintenance window" in refusal.reason

    def test_a_missing_journal_counts_as_flat(self, clean, tmp_path) -> None:
        """A machine that has never traded has nothing open, and refusing to
        update it forever would be absurd."""
        (tmp_path / "journal" / "trading.db").unlink()

        assert clean.guards(WEEKEND) is None


class TestForceWaivesOnlyTheCalendar:
    """`--force` is for the operator standing at the machine on a Tuesday. It
    must not become a way to update on top of a live position."""

    def run(self, module, monkeypatch, now, refusal_reason):  # type: ignore[no-untyped-def]
        monkeypatch.setattr(module, "guards", lambda _now: module.Refusal(refusal_reason))
        monkeypatch.setattr(module, "current_commit", lambda: "abc1234")
        monkeypatch.setattr(module, "datetime", _FrozenDatetime(now))
        return module.main(["--force"])

    def test_the_timing_guard_is_waived(self, module, monkeypatch, capsys) -> None:
        def refuse_fetch(*args: str) -> _Result:
            return _Result("", 1, "no network") if args[0] == "fetch" else _Result("")

        monkeypatch.setattr(module, "_git", refuse_fetch)

        assert self.run(module, monkeypatch, MIDWEEK, "outside the weekend maintenance window") == 1

        assert "forced" in capsys.readouterr().out, "the waiver has to be visible"

    def test_an_open_position_is_never_waived(self, module, monkeypatch, capsys) -> None:
        assert self.run(module, monkeypatch, MIDWEEK, "2 position(s) open; rules must not") == 1

        out = capsys.readouterr().out
        assert "refused" in out
        assert "forced" not in out

    def test_the_stop_file_is_never_waived(self, module, monkeypatch, capsys) -> None:
        assert self.run(module, monkeypatch, WEEKEND, "the STOP file is present") == 1

        assert "forced" not in capsys.readouterr().out


class _FrozenDatetime:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self, tz=None):  # type: ignore[no-untyped-def]
        del tz
        return self._moment


class TestItReportsTheRightExitCode:
    """A scheduled task records the code, and the three outcomes need to be
    distinguishable without reading a log: 0 is fine, 1 is the guards working,
    2 is something to look at."""

    def test_up_to_date_is_zero(self, clean, monkeypatch, capsys) -> None:
        monkeypatch.setattr(clean, "current_commit", lambda: "abc1234")
        monkeypatch.setattr(clean, "datetime", _FrozenDatetime(WEEKEND))
        monkeypatch.setattr(clean, "_git", fake_git(behind="0"))

        assert clean.main([]) == 0
        assert "already up to date" in capsys.readouterr().out

    def test_a_guard_refusal_is_one(self, clean, monkeypatch) -> None:
        monkeypatch.setattr(clean, "current_commit", lambda: "abc1234")
        monkeypatch.setattr(clean, "datetime", _FrozenDatetime(MIDWEEK))

        assert clean.main([]) == 1

    def test_a_failed_verification_rolls_back_and_returns_two(
        self, clean, monkeypatch, capsys
    ) -> None:
        """The point of the whole script. An unattended update that can only go
        forward is a way to wake up to a system that will not start."""
        commits = iter(["before01", "before01", "after999", "before01"])
        monkeypatch.setattr(clean, "current_commit", lambda: next(commits))
        monkeypatch.setattr(clean, "datetime", _FrozenDatetime(WEEKEND))
        monkeypatch.setattr(clean, "verify", lambda: (False, "the test suite failed"))
        seen: list[tuple[str, ...]] = []
        monkeypatch.setattr(clean, "_git", fake_git(behind="3", record=seen))

        assert clean.main([]) == 2

        assert any(args[:2] == ("reset", "--hard") for args in seen), "it must reset"
        assert "rollback" in capsys.readouterr().out
