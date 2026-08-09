"""Slippage as a share of the stop it happened on.

A live USDCHF short closed at -1.32R. A full stop-out is -1.00R by definition,
so that exit overshot the risk model by a third — and R is the unit every other
rule here is written in. The give-back arms at 0.5R, the profit lock secures
0.46R, the health engine measures drift in ATR.

Slippage in pips does not show this. Half a pip is a third of a 1.5-pip stop
and five percent of a ten-pip one, and only the second number is survivable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.execution_noise import main, report, rows


@pytest.fixture
def journal(tmp_path: Path) -> Path:
    path = tmp_path / "trading.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE trades (id INTEGER PRIMARY KEY, sl_distance_pips REAL);
        CREATE TABLE order_attempts (
            id INTEGER PRIMARY KEY, trade_id INTEGER, ts TEXT, kind TEXT,
            symbol TEXT, ok INTEGER, slippage_pips REAL, retcode_name TEXT);
        """)
    db.execute("INSERT INTO trades (id, sl_distance_pips) VALUES (1, 1.5)")
    db.execute("INSERT INTO trades (id, sl_distance_pips) VALUES (2, 12.0)")
    for trade_id, symbol, slip, ok in (
        (1, "USDCHF.i", 0.50, 1),
        (1, "USDCHF.i", 0.20, 1),
        (2, "XAUUSD", 0.50, 1),
        (2, "XAUUSD", 0.30, 1),
        (1, "USDCHF.i", 9.90, 0),  # rejected: no fill, no slippage
    ):
        db.execute(
            "INSERT INTO order_attempts (trade_id, ts, kind, symbol, ok, slippage_pips, "
            "retcode_name) VALUES (?, datetime('now'), 'ENTRY', ?, ?, ?, 'DONE')",
            (trade_id, symbol, ok, slip),
        )
    db.commit()
    db.close()
    return path


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def test_rejected_attempts_are_excluded(journal: Path) -> None:
    """A rejection has no fill, so it has no slippage to report."""
    records = rows(connect(journal), 168.0)
    assert len(records) == 4
    assert max(abs(float(r["slip"])) for r in records) == pytest.approx(0.5)


def test_the_same_slip_is_reported_very_differently_per_stop(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """Half a pip on both instruments. Only one of them is a problem.

    This is the whole point of the report: the pips are identical and the
    consequence is not.
    """
    report(rows(connect(journal), 168.0), 0.15)
    out = capsys.readouterr().out

    assert "USDCHF.i" in out and "XAUUSD" in out
    assert "33%" in out  # 0.50 on a 1.5-pip stop
    assert "4%" in out  # 0.50 on a 12-pip stop


def test_only_the_offender_is_flagged(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    report(rows(connect(journal), 168.0), 0.15)
    lines = [line for line in capsys.readouterr().out.splitlines() if "R is fiction" in line]

    assert len(lines) == 1
    assert "USDCHF.i" in lines[0]


def test_it_states_the_floor_each_market_needs(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """0.5 pips of slip at a 15% budget means a 3.3-pip stop, not 1.5."""
    report(rows(connect(journal), 168.0), 0.15)
    out = capsys.readouterr().out
    assert "stop floor 3.3 pips" in out


def test_a_stricter_budget_demands_a_wider_floor(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    report(rows(connect(journal), 168.0), 0.10)
    assert "stop floor 5.0 pips" in capsys.readouterr().out


def test_an_empty_window_says_so_rather_than_printing_a_table(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    report([], 0.15)
    assert "No accepted fills" in capsys.readouterr().out


def test_a_fill_with_no_trade_row_does_not_crash(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """An entry rejected before the trade row existed joins to nothing.

    It still carries a slippage figure worth seeing; it simply has no stop to
    measure it against, and the column says so instead of inventing one.
    """
    path = tmp_path / "t.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE trades (id INTEGER PRIMARY KEY, sl_distance_pips REAL);
        CREATE TABLE order_attempts (
            id INTEGER PRIMARY KEY, trade_id INTEGER, ts TEXT, kind TEXT,
            symbol TEXT, ok INTEGER, slippage_pips REAL, retcode_name TEXT);
        INSERT INTO order_attempts (trade_id, ts, kind, symbol, ok, slippage_pips, retcode_name)
        VALUES (NULL, datetime('now'), 'ENTRY', 'EURUSD.i', 1, 0.4, 'DONE');
        """)
    db.commit()
    db.close()

    report(rows(connect(path), 168.0), 0.15)
    out = capsys.readouterr().out
    assert "EURUSD.i" in out
    assert "R is fiction" not in out


def test_a_missing_journal_is_not_a_traceback(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--db", str(tmp_path / "nope.db")]) == 1
    assert "No journal" in capsys.readouterr().out


def test_a_nonsensical_share_is_refused(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--share", "1.5", "--db", str(journal)]) == 1
    assert "between 0 and 1" in capsys.readouterr().out
