"""Pull the development branch, and say plainly when it did not happen.

    python scripts/update_repo.py
    python scripts/update_repo.py --check

`git pull` reports "Already up to date." when the *current branch's upstream*
has nothing new — which is true, and deeply misleading, when the fetch it just
performed did update some other branch. That exact output appeared directly
under a line reading `3c3ad50..ca47281 claude/... -> origin/claude/...`: new
commits arrived, none of them were merged, and the operator ran an update that
changed nothing while announcing success.

Everything downstream then looks broken for the wrong reason. Scripts that were
supposedly just added are missing, fixes that were supposedly applied are not
there, and the natural conclusion is that the fix did not work rather than that
it never arrived.

So this checks the thing that actually matters — is the working tree at the tip
of the development branch — and when it is not, says which branch it is on,
which one it should be on, and the command that moves it.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The branch this system is developed on. Hardcoded deliberately: the whole
#: failure being prevented is the working tree quietly sitting somewhere else.
DEVELOPMENT_BRANCH = "claude/mt5-autonomous-trading-system-ujd1sk"


def git(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def short(ref: str) -> str:
    result = git("rev-parse", "--short", ref)
    return result.stdout.strip() if result.returncode == 0 else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only; change nothing")
    parser.add_argument("--branch", default=DEVELOPMENT_BRANCH)
    args = parser.parse_args(argv)

    if git("rev-parse", "--git-dir").returncode != 0:
        print("  Not a git repository. Nothing to update.")
        return 1

    branch = args.branch
    here = current_branch()
    print(f"  on branch   {here} ({short('HEAD')})")

    # Uncommitted edits stop a fast-forward dead, and git's own wording for it
    # ("Please commit your changes or stash them before you merge. Aborting")
    # scrolls past in a window full of pip output. Nothing is lost when this
    # happens — the edits stay exactly as they are — but the update silently
    # does not occur, which is the failure mode this whole script exists for.
    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        print("\n  There are uncommitted local changes:")
        for line in dirty.splitlines()[:10]:
            print(f"    {line}")
        if len(dirty.splitlines()) > 10:
            print(f"    ... and {len(dirty.splitlines()) - 10} more")
        print("\n  Nothing was changed, and these edits are safe. To continue, either")
        print("  keep them:      git stash    (then run this again, then: git stash pop)")
        print("  or discard:     git checkout -- .")
        return 1

    fetch = git("fetch", "origin", branch, "--prune")
    if fetch.returncode != 0:
        print("\n  Could not reach GitHub:")
        print("   ", fetch.stderr.strip().splitlines()[-1] if fetch.stderr.strip() else "unknown")
        print("\n  Nothing was changed. Check the connection and run this again.")
        return 1

    remote = f"origin/{branch}"
    if not short(remote):
        print(f"\n  The remote has no branch called {branch}.")
        return 1
    print(f"  remote      {remote} ({short(remote)})")

    if short("HEAD") == short(remote):
        print("\n  Already at the tip of the development branch. Nothing to do.")
        return 0

    behind = git("rev-list", "--count", f"HEAD..{remote}").stdout.strip() or "?"
    ahead = git("rev-list", "--count", f"{remote}..HEAD").stdout.strip() or "?"

    # On the wrong branch entirely. This is the case `git pull` hides, because
    # it answers about a branch nobody asked about.
    if here != branch:
        print(f"\n  This working tree is on '{here}', not '{branch}'.")
        print(f"  That is why 'git pull' said it was up to date: it checked '{here}'.")
        print(f"  There are {behind} commit(s) waiting on {remote}.")
        if args.check:
            return 1
        print(f"\n  Switching to {branch}...")
        switch = git("checkout", "-B", branch, remote)
        if switch.returncode != 0:
            print("\n  Could not switch branches:")
            print("   ", switch.stderr.strip())
            print("\n  Most likely cause: local edits that would be overwritten.")
            print("  Run 'git status' to see them, then either commit or discard them.")
            return 1
        print(f"  Now on {branch} ({short('HEAD')}).")
        return 0

    if ahead != "0":
        print(f"\n  This branch has {ahead} local commit(s) not on the remote.")
        print("  Not merging automatically — that could produce a merge you did not ask for.")
        print("  Run 'git status' and 'git log --oneline -5' to see what they are.")
        return 1

    print(f"\n  {behind} new commit(s) to apply.")
    if args.check:
        return 1
    merge = git("merge", "--ff-only", remote)
    if merge.returncode != 0:
        print("\n  Could not fast-forward:")
        print("   ", merge.stderr.strip())
        print("\n  Run 'git status' to see what is in the way.")
        return 1
    print(f"  Updated to {short('HEAD')}.")
    for line in git("log", "--oneline", "HEAD@{1}..HEAD").stdout.strip().splitlines()[:10]:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
