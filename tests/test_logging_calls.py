"""Every log call in this repository must survive being switched on.

WHAT THIS COST. `_run_scalp_lane` logged the opened trade like structlog:

    log.info("section six scalp opened", symbol=symbol, direction=..., ...)

`infra.logging.get_logger` returns a plain `logging.Logger`, which takes
`extra=` and nothing else. So that call is a TypeError -- and the whole suite
was green, because `Logger.info` checks `isEnabledFor` BEFORE unpacking its
keywords. With logging off in tests the broken line is never executed.

On the VPS it is on, and Python is 3.14 rather than the 3.11 here. Every scalp
opened a real position and then threw:

    TypeError: Logger._log() got an unexpected keyword argument 'symbol'

which surfaced as "candidate analysis failed; continuing with the rest of the
batch" -- the same message a genuinely broken detector produces, on an account
whose trades were going through fine.

Two tests, because the failure has two halves. The first reads the source and
refuses the shape outright, which catches it without needing the line to run.
The second runs the logger at DEBUG so a call that only misbehaves when
switched on cannot hide behind a default level again.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pytest

#: What `logging.Logger._log` actually accepts. Anything else is a TypeError
#: the moment the level is enabled.
ALLOWED = {"exc_info", "stack_info", "stacklevel", "extra"}
METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
LOGGERS = {"log", "logger", "LOG"}
ROOT = pathlib.Path(__file__).resolve().parent.parent


def offending_calls() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*.py"):
        text = str(path)
        relative_parts = path.relative_to(ROOT).parts
        # Dependency scratch directories are ignored source-wise and are not
        # part of this repository. Some package wheels also carry deliberately
        # unreadable ACLs on Windows, so never traverse them as application
        # logging code.
        if (
            ".venv" in text
            or "tests" in relative_parts
            or any(part in {".local-deps", "research-deps-local"} for part in path.parts)
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a file that cannot parse fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in METHODS:
                continue
            base = node.func.value
            if (getattr(base, "id", None) or getattr(base, "attr", None)) not in LOGGERS:
                continue
            stray = [k.arg for k in node.keywords if k.arg is not None and k.arg not in ALLOWED]
            if stray:
                found.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} "
                    f"log.{node.func.attr}(... {', '.join(stray)})"
                )
    return found


def test_no_log_call_passes_keywords_the_stdlib_logger_rejects() -> None:
    """Structured fields go in `extra=`. This is not structlog."""
    offenders = offending_calls()

    assert offenders == [], "\n" + "\n".join(offenders)


def test_a_structlog_shaped_call_would_actually_raise(caplog: pytest.LogCaptureFixture) -> None:
    """The mechanism, pinned so the test above is not merely stylistic.

    It raises only when the level is enabled, which is exactly why a green
    suite said nothing. `caplog` turns the level on, and that is the condition
    the VPS runs in every second of the day.
    """
    logger = logging.getLogger("test_logging_calls")

    with caplog.at_level(logging.DEBUG, logger="test_logging_calls"):
        with pytest.raises(TypeError):
            logger.info("opened", symbol="XAUUSD")  # type: ignore[call-arg]

        # And the correct shape does not.
        logger.info("opened", extra={"symbol": "XAUUSD"})

    assert "opened" in caplog.text
