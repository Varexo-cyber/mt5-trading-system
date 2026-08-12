"""What this account is bad at, asked of data it has always had.

The journal records everything and nothing ever asked it the question a trader
answers about themselves within a month. The half that matters most is the
second one: every gate against what the setups it blocked went on to do, which
is the only honest way to find out whether a gate — including Claude's veto —
earns its keep.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.scorecard import conviction_band, main, score_band, session_of


@pytest.fixture
def journal(tmp_path: Path) -> Path:
    path = tmp_path / "trading.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, pnl_r REAL,
            pnl_money REAL, mfe_r REAL, exit_reason TEXT, opened_at TEXT, closed_at TEXT);
        CREATE TABLE shadow_trades (
            id INTEGER PRIMARY KEY, blocked_by TEXT, outcome TEXT, pnl_r REAL,
            opened_at TEXT);
    """)
    now = "2026-08-07T09:30:00+00:00"
    db.executemany(
        "INSERT INTO trades (symbol, direction, pnl_r, pnl_money, mfe_r, exit_reason, "
        "opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?)",
        [
            ("FRA40", "LONG", 0.6, 0.68, 1.2, "SESSION_DECAY", now, now),
            ("UK100", "LONG", -1.0, -1.55, 0.1, "BROKER_SL", now, now),
            ("UK100", "LONG", -0.8, -1.10, 0.2, "BROKER_SL", now, now),
            ("EURUSD.i", "SHORT", 0.4, 0.50, 0.5, "PROFIT_LOCK", now, now),
        ],
    )
    db.executemany(
        "INSERT INTO shadow_trades (blocked_by, outcome, pnl_r, opened_at) VALUES (?,?,?,?)",
        [
            ("AI_VETO", "SL", -1.0, now),
            ("AI_VETO", "SL", -1.0, now),
            ("LOSS_COOLDOWN", "TP", 1.8, now),
            ("SPREAD_TOO_WIDE", None, None, now),
        ],
    )
    db.commit()
    db.close()
    return path


def run(journal: Path, *args: str, capsys) -> str:  # type: ignore[no-untyped-def]
    assert main(["--db", str(journal), *args]) == 0
    return capsys.readouterr().out


def test_the_instrument_that_loses_is_named(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    printed = run(journal, capsys=capsys)

    assert "UK100" in printed
    assert "-1.80R" in printed


def test_a_gate_that_saved_money_is_credited(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """Two setups Claude refused both went on to hit their stop."""
    printed = run(journal, capsys=capsys)

    assert "AI_VETO" in printed
    assert "saved us 2.00R" in printed


def test_a_gate_that_cost_money_is_named_as_such(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The finding this half exists for. A gate is not automatically right
    because it is a gate."""
    printed = run(journal, capsys=capsys)

    assert "cost us 1.80R" in printed


def test_an_unresolved_shadow_is_excluded_and_counted(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """Counting a setup whose outcome is still unknown as a zero would flatter
    every gate toward neutral."""
    printed = run(journal, capsys=capsys)

    assert "1 blocked setup(s) not yet resolved" in printed


def test_it_refuses_to_let_four_trades_read_as_a_result(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    printed = run(journal, capsys=capsys)

    assert "Not a sample" in printed


def test_thin_buckets_can_be_hidden(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A bucket with one trade in it is an anecdote, and the temptation to act
    on one is exactly what this flag is for."""
    printed = run(journal, "--min-sample", "2", capsys=capsys)

    assert "UK100" in printed  # two trades
    assert "FRA40" not in printed  # one


def test_an_empty_window_says_so(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    printed = run(journal, "--days", "0.001", capsys=capsys)

    assert "Nothing closed in this window" in printed


def test_a_missing_journal_is_reported_not_raised() -> None:
    assert main(["--db", "journal/nowhere.db"]) == 1


class TestSessionBuckets:
    """A bucket here must mean the same window as the session filter, or the
    report says "london" about hours london does not cover."""

    def test_the_night_is_asia(self) -> None:
        assert session_of(3) == "asia"

    def test_the_morning_is_london(self) -> None:
        assert session_of(9) == "london"

    def test_the_afternoon_is_the_overlap(self) -> None:
        assert session_of(14) == "overlap"

    def test_the_evening_is_new_york(self) -> None:
        assert session_of(19) == "newyork"

    def test_the_dead_hours_are_named_rollover(self) -> None:
        assert session_of(22) == "rollover"


class TestInterventionsAgainstHolding:
    """ "Nought for eight" is not a verdict until you know what holding paid.

    A rule that only ever fires on trades already going wrong shows a losing
    record while still losing less than doing nothing would have. The baseline
    replays each trade's untouched original stop and target over the same hours
    it really spanned, and `lift` is the only column that can condemn or
    acquit an exit rule.

    `management_baselines` was written by the resolver and read by nothing,
    which is how AI_CLOSE could sit at 0-for-8 in a report for a month with no
    way to tell rescues from mistakes.
    """

    def measured(self, journal: Path, rows: list[tuple]) -> Path:
        db = sqlite3.connect(journal)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS management_baselines (
                trade_id INTEGER PRIMARY KEY, observed_at TEXT, resolved_at TEXT,
                outcome TEXT, baseline_pnl_r REAL, actual_pnl_r REAL, lift_r REAL);
        """)
        now = datetime.now(UTC)
        for trade_id, reason, actual, baseline in rows:
            db.execute(
                "INSERT INTO trades (id, symbol, direction, pnl_r, pnl_money, mfe_r, "
                "exit_reason, opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    trade_id,
                    "EURUSD",
                    "LONG",
                    actual,
                    actual,
                    0.2,
                    reason,
                    (now - timedelta(hours=6)).isoformat(),
                    (now - timedelta(hours=2)).isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO management_baselines (trade_id, observed_at, resolved_at, "
                "outcome, baseline_pnl_r, actual_pnl_r, lift_r) VALUES (?,?,?,?,?,?,?)",
                (
                    trade_id,
                    now.isoformat(),
                    now.isoformat(),
                    "SL",
                    baseline,
                    actual,
                    actual - baseline,
                ),
            )
        db.commit()
        db.close()
        return journal

    def test_a_rule_that_lost_less_than_holding_is_credited(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Nought for two, and still the right call both times."""
        self.measured(
            journal, [(90, "HEALTH_EXIT", -0.45, -1.00), (91, "HEALTH_EXIT", -0.40, -1.00)]
        )
        main(["--db", str(journal), "--days", "30", "--min-sample", "1"])
        out = capsys.readouterr().out

        assert "DID STEPPING IN BEAT LEAVING IT ALONE" in out
        assert "HEALTH_EXIT" in out
        assert "+0.57R" in out, "average lift of +0.575R over the untouched stop"
        assert "2/2" in out

    def test_a_rule_that_paid_to_do_worse_is_exposed(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """The finding worth having: an exit that beat nothing."""
        self.measured(journal, [(92, "AI_CLOSE", -0.60, -0.10), (93, "AI_CLOSE", -0.80, +0.20)])
        main(["--db", str(journal), "--days", "30", "--min-sample", "1"])
        out = capsys.readouterr().out

        assert "AI_CLOSE" in out
        assert "-0.75R" in out, "it cost 0.75R a trade against leaving the position alone"
        assert "0/2" in out

    def test_an_old_journal_without_the_table_still_reports(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """A missing measurement is a missing section, never a crash."""
        assert main(["--db", str(journal), "--days", "30"]) == 0
        assert "DID STEPPING IN BEAT LEAVING IT ALONE" not in capsys.readouterr().out


class TestDoesBeingSureMeanAnything:
    """ "Hold the ones we are sure about" is only a strategy if sure means something.

    Nobody had ever asked. A setup scoring 58.5 against a bar of 40 lost money
    on the same day a 39.8 was refused, and the engine's own confidence was
    never once compared against what the trades did. Conviction-scaled exits
    would be fitting noise until this bucket says otherwise.
    """

    def scored(self, journal: Path, rows: list[tuple[float, float]]) -> Path:
        db = sqlite3.connect(journal)
        db.executescript(
            "CREATE TABLE IF NOT EXISTS analysis_cycles "
            "(id INTEGER PRIMARY KEY, total_score REAL, score_threshold REAL);"
            "ALTER TABLE trades ADD COLUMN cycle_pk INTEGER;"
        )
        now = datetime.now(UTC)
        for i, (score, pnl) in enumerate(rows, start=500):
            db.execute("INSERT INTO analysis_cycles VALUES (?,?,?)", (i, score, 40.0))
            db.execute(
                "INSERT INTO trades (id, cycle_pk, symbol, direction, pnl_r, pnl_money, "
                "mfe_r, exit_reason, opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    i,
                    i,
                    "EURUSD",
                    "LONG",
                    pnl,
                    pnl * 3,
                    0.3,
                    "BROKER_SL",
                    (now - timedelta(hours=8)).isoformat(),
                    (now - timedelta(hours=3)).isoformat(),
                ),
            )
        db.commit()
        db.close()
        return journal

    def test_the_bands_are_measured_against_the_bar_not_in_raw_points(self) -> None:
        """The threshold has already moved once, 55 to 40. A fixed band would
        have quietly changed meaning underneath a month of history."""
        assert conviction_band(44.0, 40.0) == conviction_band(59.0, 55.0)
        assert conviction_band(58.5, 40.0) == "10-20 over the bar"
        assert conviction_band(41.0, 40.0) == "0-5 over the bar"
        assert conviction_band(70.0, 40.0) == "20+ over the bar"

    def test_a_missing_score_is_named_rather_than_guessed(self) -> None:
        assert conviction_band(None, 40.0) == "unrecorded"

    def test_the_report_splits_trades_by_how_sure_it_was(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        self.scored(journal, [(42.0, -0.5), (43.0, -0.4), (58.0, 0.8), (59.0, 0.6)])
        main(["--db", str(journal), "--days", "30", "--min-sample", "1"])
        out = capsys.readouterr().out

        assert "HOW SURE THE ENGINE WAS" in out
        assert "0-5 over the bar" in out
        assert "10-20 over the bar" in out

    def test_a_journal_without_the_analysis_table_still_reports(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """An older journal loses the section, never the whole report."""
        assert main(["--db", str(journal), "--days", "30"]) == 0
        assert "HOW SURE THE ENGINE WAS" not in capsys.readouterr().out


class TestAThresholdChangeCanBeJudged:
    """`conviction_band` is measured against the bar, which is right for "does
    being sure mean anything" and useless for judging a move of the bar itself.

    Drop the threshold from 40 to 35 and "0-5 over the bar" stops describing
    scores of 40-45 and starts describing 35-40, under the same label. The
    before-and-after comparison the move exists to support would be comparing
    two different populations.
    """

    def test_the_raw_band_does_not_move_when_the_bar_does(self) -> None:
        assert score_band(37.5) == "score 35-40"
        assert score_band(42.0) == "score 40-45"

    def test_the_relative_band_does_move_and_that_is_the_problem(self) -> None:
        """Same setup, two thresholds, two different labels — which is exactly
        why the absolute slice had to be added rather than reusing this one."""
        assert conviction_band(37.5, 35.0) == "0-5 over the bar"
        assert conviction_band(42.0, 40.0) == "0-5 over the bar"

    def test_an_unscored_trade_is_named_rather_than_bucketed_at_zero(self) -> None:
        """A missing score put in the lowest band would invent evidence that
        low scores lose, on rows that never carried a score at all."""
        assert score_band(None) == "unrecorded"
