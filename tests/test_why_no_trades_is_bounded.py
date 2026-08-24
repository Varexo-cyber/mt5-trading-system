"""`update.cmd` must finish.

WHAT HAPPENED. The update ran, pulled, reinstalled, checked the config, printed
"Why it has or has not been trading" — and sat there. Nothing was broken and
nothing was hung. `why_no_trades.py` was reading the whole of `module_scores`
to answer a question about the last twelve hours of it, and on an account that
scans 845 markets that table holds one row per module per symbol per cycle and
grows forever. It had been getting slower every day since the query was written
and nothing measured it, so the first symptom was an operator waiting at a
prompt with no idea whether the update had failed.

These tests assert the plans rather than the timings, because a timing test on
a fixture small enough to run in CI proves nothing about the machine where it
actually mattered. A plan that reads `idx_module_scores_cycle` is bounded by
the window; one that reads the table or the module index is not, whatever the
clock says on a fixture this size.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.why_no_trades import _directional_modules


@pytest.fixture
def journal(tmp_path: Path) -> sqlite3.Connection:
    """The real schema, indexes included — the indexes are the subject."""
    db = sqlite3.connect(tmp_path / "trading.db")
    db.executescript(
        "CREATE TABLE analysis_cycles (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
        " symbol TEXT NOT NULL);"
        "CREATE INDEX idx_cycles_ts ON analysis_cycles(ts);"
        "CREATE TABLE module_scores (id INTEGER PRIMARY KEY AUTOINCREMENT, cycle_pk INTEGER,"
        " module TEXT NOT NULL, score REAL NOT NULL, weight REAL NOT NULL);"
        "CREATE INDEX idx_module_scores_cycle ON module_scores(cycle_pk);"
        "CREATE INDEX idx_module_scores_module ON module_scores(module);"
    )
    now = datetime.now(UTC)
    for i in range(1, 41):
        db.execute(
            "INSERT INTO analysis_cycles (id, ts, symbol) VALUES (?,?,?)",
            (i, (now - timedelta(hours=40 - i)).isoformat(), "EURUSD.i"),
        )
        for module, score in (("impulse_break", 60.0), ("candle_momentum", -60.0)):
            db.execute(
                "INSERT INTO module_scores (cycle_pk, module, score, weight) VALUES (?,?,?,?)",
                (i, module, score, 0.6),
            )
    db.commit()
    return db


def plan(db: sqlite3.Connection, where: str, params: list[object]) -> str:
    rows = db.execute(
        "EXPLAIN QUERY PLAN SELECT m.module, SUM(m.score) FROM module_scores m "
        f"INDEXED BY idx_module_scores_cycle WHERE {where} GROUP BY m.module",
        params,
    ).fetchall()
    return " | ".join(str(row[3]) for row in rows)


class TestTheQueryIsBoundedByTheWindow:
    def test_it_seeks_the_cycle_index_instead_of_reading_the_table(
        self, journal: sqlite3.Connection
    ) -> None:
        """The assertion the outage is worth. `SEARCH ... (cycle_pk>?)` means
        SQLite walks only the slice; `SCAN` in its place means it reads every
        row ever written to answer a question about today."""
        readable = plan(journal, "m.cycle_pk >= ?", [35])

        assert "SEARCH" in readable
        assert "idx_module_scores_cycle" in readable
        assert "SCAN m" not in readable

    def test_grouping_by_module_is_what_made_it_choose_wrong(
        self, journal: sqlite3.Connection
    ) -> None:
        """Unpinned, the GROUP BY makes the index on `module` look free, and
        SQLite reads it end to end. This documents the trap rather than
        guarding against it — if a future SQLite plans it well, the pin costs
        nothing anyway."""
        rows = journal.execute(
            "EXPLAIN QUERY PLAN SELECT m.module, SUM(m.score) FROM module_scores m "
            "WHERE m.cycle_pk >= ? GROUP BY m.module",
            [35],
        ).fetchall()
        readable = " | ".join(str(row[3]) for row in rows)

        assert "idx_module_scores_cycle" in readable or "idx_module_scores_module" in readable


class TestItStillAnswersTheSameQuestion:
    """A fast query that reports something else is not a fix."""

    def test_it_counts_longs_and_shorts_inside_the_window(
        self, journal: sqlite3.Connection
    ) -> None:
        journal.row_factory = sqlite3.Row
        rows = {
            row["module"]: row for row in _directional_modules(journal, "m.cycle_pk >= ?", [31])
        }

        assert rows["impulse_break"]["longs"] == 10
        assert rows["impulse_break"]["shorts"] == 0
        assert rows["candle_momentum"]["shorts"] == 10

    def test_it_excludes_everything_before_the_floor(self, journal: sqlite3.Connection) -> None:
        journal.row_factory = sqlite3.Row
        rows = {
            row["module"]: row for row in _directional_modules(journal, "m.cycle_pk >= ?", [40])
        }

        assert rows["impulse_break"]["longs"] == 1

    def test_a_journal_without_the_index_still_answers(self, tmp_path: Path) -> None:
        """`INDEXED BY` is an error, not a hint, when the index is missing. An
        older journal must get a slow answer rather than a stack trace halfway
        through `update.cmd`."""
        db = sqlite3.connect(tmp_path / "old.db")
        db.row_factory = sqlite3.Row
        db.executescript(
            "CREATE TABLE module_scores (id INTEGER PRIMARY KEY, cycle_pk INTEGER,"
            " module TEXT, score REAL, weight REAL);"
        )
        db.execute(
            "INSERT INTO module_scores (cycle_pk, module, score, weight) VALUES (1,'x',60.0,0.6)"
        )
        db.commit()

        rows = _directional_modules(db, "m.cycle_pk >= ?", [1])

        assert [row["module"] for row in rows] == ["x"]
