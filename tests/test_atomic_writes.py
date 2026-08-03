"""Telemetry writes must never be able to stop the trading loop."""

from __future__ import annotations

import json
from pathlib import Path

from infra.atomic import write_json_atomic


def test_a_normal_write_lands(tmp_path: Path) -> None:
    path = tmp_path / "state.json"

    assert write_json_atomic(path, {"a": 1}) is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_a_denied_replace_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Windows denies the rename while a reader holds the file open.

    This is the production failure: the dashboard had scan_activity.json open,
    the replace was denied, and the PermissionError came out of the path that
    records a skipped trade — killing the service and leaving open positions
    unmanaged. A monitoring file losing a race must cost nothing.
    """
    path = tmp_path / "state.json"

    def always_denied(self: Path, target: object) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_denied)

    assert write_json_atomic(path, {"a": 1}) is False
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp")), "the temporary file must be cleaned up"


def test_a_denied_replace_still_raises_when_the_write_matters(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Paper-account state and the experimental contract are not display data."""
    path = tmp_path / "state.json"

    def always_denied(self: Path, target: object) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_denied)

    try:
        write_json_atomic(path, {"a": 1}, required=True)
    except PermissionError:
        return
    raise AssertionError("required=True must propagate the failure")


def test_a_transient_denial_succeeds_on_retry(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The contending reader holds the handle for microseconds, not forever."""
    path = tmp_path / "state.json"
    real = Path.replace
    attempts = {"n": 0}

    def flaky(self: Path, target: object):  # type: ignore[no-untyped-def]
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError(5, "Access is denied")
        return real(self, target)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "replace", flaky)

    assert write_json_atomic(path, {"a": 1}) is True
    assert attempts["n"] == 3
