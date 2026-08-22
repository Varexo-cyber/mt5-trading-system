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


class TestWhereTheMoneyActuallyCameFrom:
    """ "Which detector produced the winners" was a join, not a backtest.

    `module_scores` has recorded every module's score, confidence and weight
    for the cycle behind each trade since the journal was written, and
    `analysis_cycles` records the regime the classifier read. Nothing ever
    asked either of them, so the only way to judge a detector was an offline
    replay on five symbols — which cannot see how it behaves on the 845 markets
    this account scans, at its own sizes, paying its own costs.

    The regime slice answers a live question directly: `drift_continuation`
    fires in `transition` and the guard that should discount it protects only
    `trend_momentum`, which carries no live weight. Whether that costs money is
    measurable here rather than arguable.
    """

    def _journal(self, journal: Path, rows) -> Path:  # type: ignore[no-untyped-def]
        db = sqlite3.connect(journal)
        db.executescript(
            "CREATE TABLE IF NOT EXISTS analysis_cycles "
            "(id INTEGER PRIMARY KEY, total_score REAL, score_threshold REAL,"
            " volatility_regime TEXT);"
            "CREATE TABLE IF NOT EXISTS module_scores "
            "(id INTEGER PRIMARY KEY, cycle_pk INTEGER, module TEXT, score REAL,"
            " confidence REAL, weight REAL);"
            "ALTER TABLE trades ADD COLUMN cycle_pk INTEGER;"
        )
        now = datetime.now(UTC)
        for i, (pnl, regime, modules) in enumerate(rows, start=900):
            db.execute("INSERT INTO analysis_cycles VALUES (?,?,?,?)", (i, 55.0, 26.0, regime))
            for module, score, weight in modules:
                db.execute(
                    "INSERT INTO module_scores (cycle_pk, module, score, confidence, weight)"
                    " VALUES (?,?,?,?,?)",
                    (i, module, score, 0.8, weight),
                )
            db.execute(
                "INSERT INTO trades (id, cycle_pk, symbol, direction, pnl_r, pnl_money, "
                "mfe_r, exit_reason, opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    i,
                    i,
                    "EURUSD",
                    "LONG",
                    pnl,
                    pnl * 10,
                    0.4,
                    "BROKER_TP",
                    (now - timedelta(hours=8)).isoformat(),
                    (now - timedelta(hours=3)).isoformat(),
                ),
            )
        db.commit()
        db.close()
        return journal

    def test_each_detector_gets_its_own_line(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        path = self._journal(
            journal,
            [
                (+0.5, "trend_up", [("impulse_break", 60.0, 0.6)]),
                (+0.5, "trend_up", [("impulse_break", 60.0, 0.6)]),
                (-0.4, "transition", [("drift_continuation", 55.0, 0.7)]),
                (-0.4, "transition", [("drift_continuation", 55.0, 0.7)]),
            ],
        )

        main(["--db", str(path), "--days", "2"])
        out = capsys.readouterr().out

        assert "WHICH DETECTOR WAS BEHIND IT" in out
        assert "impulse_break" in out
        assert "drift_continuation" in out
        assert "THE REGIME AT ENTRY" in out
        assert "transition" in out

    def test_a_detector_pointing_the_other_way_is_not_credited(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """A module scoring negative on a LONG did not produce that trade, and
        crediting it would put the blame — or the credit — on the wrong row."""
        path = self._journal(
            journal,
            [(+0.5, "trend_up", [("impulse_break", 60.0, 0.6), ("mean_reversion", -40.0, 0.5)])],
        )

        main(["--db", str(path), "--days", "2"])
        out = capsys.readouterr().out

        assert "impulse_break" in out
        assert "mean_reversion" not in out

    def test_a_module_carrying_no_weight_is_not_credited_either(
        self, journal: Path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        """`trend_momentum` is computed and logged on this account and votes on
        nothing. It did not produce the trade and must not appear to have."""
        path = self._journal(
            journal,
            [(+0.5, "trend_up", [("impulse_break", 60.0, 0.6), ("trend_momentum", 65.0, 0.0)])],
        )

        main(["--db", str(path), "--days", "2"])
        out = capsys.readouterr().out

        assert "impulse_break" in out
        assert "trend_momentum" not in out

    def test_the_double_counting_is_stated_rather_than_hidden(self, journal: Path, capsys) -> None:
        """Three detectors behind one trade puts it in three rows. Without the
        note the columns look like a book that does not balance."""
        path = self._journal(
            journal,
            [
                (
                    +0.5,
                    "trend_up",
                    [("impulse_break", 60.0, 0.6), ("liquidity_sweep", 50.0, 0.8)],
                )
            ],
        )

        main(["--db", str(path), "--days", "2"])
        out = capsys.readouterr().out

        assert "counts once in each row" in out


class TestWhenItTurned:
    """ "It worked Wednesday and Thursday morning and then stopped" could be
    felt and not checked.

    The report sliced the book six ways and never by the clock. A day column
    and an hour column make the shape visible in one run; `--since` / `--until`
    then let the same report be run over the good stretch and the bad one, so
    every other slice — detector, regime, instrument, direction — can be read
    side by side instead of guessed at.
    """

    def _two_days(self, journal: Path) -> Path:
        db = sqlite3.connect(journal)
        now = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
        rows = [
            (now - timedelta(days=1, hours=4), +0.5),  # the good stretch
            (now - timedelta(days=1, hours=3), +0.4),
            (now - timedelta(hours=6), -0.6),  # and the bad one
            (now - timedelta(hours=5), -0.7),
        ]
        for i, (opened, pnl) in enumerate(rows, start=700):
            db.execute(
                "INSERT INTO trades (id, symbol, direction, pnl_r, pnl_money, mfe_r, "
                "exit_reason, opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    i,
                    "EURUSD",
                    "LONG",
                    pnl,
                    pnl * 10,
                    0.5,
                    "BROKER_SL",
                    opened.isoformat(),
                    (opened + timedelta(minutes=30)).isoformat(),
                ),
            )
        db.commit()
        db.close()
        return journal

    def test_the_day_and_the_hour_each_get_a_column(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        path = self._two_days(journal)

        main(["--db", str(path), "--days", "3"])
        out = capsys.readouterr().out

        assert "WHICH DAY" in out
        assert "WHICH HOUR IT OPENED" in out
        assert "UTC" in out

    def test_time_columns_read_in_order_not_by_profit(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """A day column sorted worst-first would hide the one thing it exists
        to show: which way the book moved as the clock ran."""
        path = self._two_days(journal)

        main(["--db", str(path), "--days", "3"])
        block = capsys.readouterr().out.split("WHICH HOUR IT OPENED")[1].split("INSTRUMENT")[0]
        hours = [line.strip().split()[0] for line in block.splitlines() if ":00 UTC" in line]

        assert hours == sorted(hours)

    def test_a_window_can_be_bounded_at_both_ends(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """The point of the whole thing: run the good stretch alone."""
        path = self._two_days(journal)
        cut = (datetime.now(UTC) - timedelta(hours=12)).isoformat()

        main(
            [
                "--db",
                str(path),
                "--since",
                (datetime.now(UTC) - timedelta(days=2)).isoformat(),
                "--until",
                cut,
            ]
        )
        out = capsys.readouterr().out

        assert "2 closed trades" in out

    def test_a_bad_instant_is_refused_rather_than_ignored(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Silently falling back to --days would report a different window than
        the one asked for, which is the failure every tool here guards."""
        assert main(["--db", str(journal), "--since", "not-a-date"]) == 1
        assert "not an ISO instant" in capsys.readouterr().out

    def test_an_until_before_the_since_is_refused(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        assert (
            main(
                [
                    "--db",
                    str(journal),
                    "--since",
                    "2026-08-21T00:00",
                    "--until",
                    "2026-08-20T00:00",
                ]
            )
            == 1
        )
        assert "must be after" in capsys.readouterr().out
