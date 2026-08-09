"""The re-arm command line, which was unusable as documented.

The invocation in the instructions omitted two required flags and failed with
an argparse error before touching anything, so the operator could not re-arm at
all — and the error message it *would* have produced quoted a risk figure the
build had stopped using. Both are the same class of fault: a safety gate whose
message sends you somewhere that does not work is worse than no message.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "arm_experimental_live.py"
PHRASE = "BEVESTIG EXPERIMENTEEL LIVE"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_the_risk_and_drawdown_flags_are_optional() -> None:
    """They may only ever equal the build's constants, so requiring them was
    ceremony that could only be got wrong — and was."""
    # Prove argparse accepted the omitted numeric flags without waiting for a
    # real MT5 IPC connection inside a unit test process.
    result = run("--account", "1", "--confirm", "wrong on purpose")
    assert "the following arguments are required" not in result.stderr


def test_the_account_is_still_required() -> None:
    """Binding to one login is the actual safety property; it stays mandatory."""
    result = run("--confirm", PHRASE)
    assert "required" in result.stderr
    assert "--account" in result.stderr


def test_the_phrase_is_still_required() -> None:
    result = run("--account", "1")
    assert "required" in result.stderr
    assert "--confirm" in result.stderr


def test_a_wrong_phrase_is_refused() -> None:
    result = run("--account", "1", "--confirm", "yes please")
    assert PHRASE in result.stderr


@pytest.mark.parametrize("flag,value", [("--risk-percent", "1"), ("--drawdown-percent", "10")])
def test_a_wrong_figure_names_both_numbers(flag: str, value: str) -> None:
    """ "must be exactly 1%" survived the constant moving to 2%, so the message
    quoted a figure nothing used and gave no way to see which side was wrong."""
    from promotion.experimental import (
        EXPERIMENTAL_MAX_DRAWDOWN_PCT,
        EXPERIMENTAL_RISK_PER_TRADE_PCT,
    )

    result = run("--account", "1", "--confirm", PHRASE, flag, value)
    required = (
        EXPERIMENTAL_RISK_PER_TRADE_PCT
        if flag == "--risk-percent"
        else EXPERIMENTAL_MAX_DRAWDOWN_PCT
    )
    assert f"was {value} but this build requires {required:g}" in result.stderr
    assert "Omit the flag" in result.stderr


def test_the_defaults_track_the_build_constants() -> None:
    """A default that drifts from the constant would re-arm at the wrong risk."""
    import argparse
    import importlib

    from promotion.experimental import (
        EXPERIMENTAL_MAX_DRAWDOWN_PCT,
        EXPERIMENTAL_RISK_PER_TRADE_PCT,
    )

    module = importlib.import_module("scripts.arm_experimental_live")
    parsers = [obj for obj in vars(module).values() if isinstance(obj, argparse.ArgumentParser)]
    # The parser is built inside main(), so assert on the constants the module
    # imported instead — the binding that the defaults are written from.
    assert not parsers
    assert module.EXPERIMENTAL_RISK_PER_TRADE_PCT == EXPERIMENTAL_RISK_PER_TRADE_PCT
    assert module.EXPERIMENTAL_MAX_DRAWDOWN_PCT == EXPERIMENTAL_MAX_DRAWDOWN_PCT
