"""Updating must never report success while changing nothing.

Against real git repositories rather than mocks, because the failure being
prevented is a property of git's own behaviour: `git pull` answers about the
current branch's upstream, which can be a branch nobody involved cares about.
A mock would have happily reproduced whatever I believed git does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "update_repo.py"
BRANCH = "test-development-branch"


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the copy that lives *inside* the repo under test.

    The script derives the repository root from its own location, so invoking
    the original from the real checkout would have it update this project
    rather than the fixture — and the tests would pass while proving nothing.
    """
    return subprocess.run(
        [sys.executable, str(cwd / "scripts" / "update_repo.py"), "--branch", BRANCH, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """An origin with two commits on the development branch."""
    path = tmp_path / "origin"
    path.mkdir()
    git("init", "-q", "-b", BRANCH, cwd=path)
    git("config", "user.email", "t@t.t", cwd=path)
    git("config", "user.name", "t", cwd=path)
    (path / "first.txt").write_text("one", encoding="utf-8")
    # Tracked, not dropped into the clone afterwards: an untracked copy makes
    # every working tree dirty and the dirty-tree guard then fires everywhere.
    (path / "scripts").mkdir()
    (path / "scripts" / "update_repo.py").write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    git("add", "-A", cwd=path)
    git("commit", "-qm", "first", cwd=path)
    return path


def clone(remote: Path, tmp_path: Path) -> Path:
    path = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(remote), str(path)], capture_output=True, check=True)
    git("config", "user.email", "t@t.t", cwd=path)
    git("config", "user.name", "t", cwd=path)
    return path


def advance(remote: Path, name: str = "second.txt") -> str:
    (remote / name).write_text("two", encoding="utf-8")
    git("add", "-A", cwd=remote)
    git("commit", "-qm", f"add {name}", cwd=remote)
    return git("rev-parse", "--short", "HEAD", cwd=remote).stdout.strip()


def head(path: Path) -> str:
    return git("rev-parse", "--short", "HEAD", cwd=path).stdout.strip()


def test_nothing_to_do_is_reported_honestly(remote: Path, tmp_path: Path) -> None:
    work = clone(remote, tmp_path)
    result = run(work)
    assert result.returncode == 0
    assert "Already at the tip" in result.stdout


def test_new_commits_are_applied(remote: Path, tmp_path: Path) -> None:
    work = clone(remote, tmp_path)
    expected = advance(remote)

    result = run(work)

    assert result.returncode == 0, result.stdout
    assert head(work) == expected
    assert "1 new commit" in result.stdout


def test_the_wrong_branch_is_named_and_corrected(remote: Path, tmp_path: Path) -> None:
    """The exact failure: "Already up to date." under a line showing new commits.

    Caused by the local branch tracking something other than the branch the
    work is on. `git pull` is telling the truth about a question nobody asked.
    """
    work = clone(remote, tmp_path)
    expected = advance(remote)
    git("checkout", "-q", "-b", "main", cwd=work)

    # Confirm git really does mislead here before asserting we handle it.
    subprocess.run(["git", "pull"], cwd=work, capture_output=True, text=True, check=False)
    assert head(work) != expected, "precondition: the plain pull must not have updated"

    result = run(work)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "is on 'main'" in result.stdout
    assert "why 'git pull' said it was up to date" in result.stdout
    assert head(work) == expected


def test_check_mode_changes_nothing(remote: Path, tmp_path: Path) -> None:
    work = clone(remote, tmp_path)
    before = head(work)
    advance(remote)

    result = run(work, "--check")

    assert result.returncode == 1, "pending work must be a non-zero status"
    assert head(work) == before


def test_local_commits_are_not_silently_merged(remote: Path, tmp_path: Path) -> None:
    """An unexpected merge commit is worse than a refused update."""
    work = clone(remote, tmp_path)
    (work / "mine.txt").write_text("local", encoding="utf-8")
    git("add", "-A", cwd=work)
    git("commit", "-qm", "local work", cwd=work)
    mine = head(work)
    advance(remote)

    result = run(work)

    assert result.returncode == 1
    assert "local commit" in result.stdout
    assert head(work) == mine


def test_an_unreachable_remote_fails_loudly(remote: Path, tmp_path: Path) -> None:
    work = clone(remote, tmp_path)
    subprocess.run(
        ["git", "remote", "set-url", "origin", str(tmp_path / "gone")],
        cwd=work,
        capture_output=True,
        check=True,
    )
    result = run(work)
    assert result.returncode == 1
    assert "Could not reach" in result.stdout
    assert "Nothing was changed" in result.stdout


def test_a_missing_branch_is_named(remote: Path, tmp_path: Path) -> None:
    work = clone(remote, tmp_path)
    result = subprocess.run(
        [sys.executable, str(work / "scripts" / "update_repo.py"), "--branch", "no-such-branch"],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def test_uncommitted_edits_stop_the_update_and_say_so(remote: Path, tmp_path: Path) -> None:
    """git's own wording for this scrolls past in a window full of pip output.

    Nothing is lost when a fast-forward is refused — the edits stay exactly as
    they are — but the update silently does not happen, which is the failure
    this script exists to make impossible.
    """
    work = clone(remote, tmp_path)
    (work / "first.txt").write_text("edited by hand", encoding="utf-8")
    before = head(work)
    advance(remote)

    result = run(work)

    assert result.returncode == 1
    assert "uncommitted local changes" in result.stdout
    assert "git stash" in result.stdout
    assert head(work) == before
    assert (work / "first.txt").read_text(encoding="utf-8") == "edited by hand"
