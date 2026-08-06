"""One trade, told as the sequence it actually was.

The question "what happened with that USDCHF trade" was being answered by
reading five tables by hand, and the answer is almost never in any one of
them. A trade that ends at +0.13R after peaking at 0.92R looks unremarkable in
`trades` and only makes sense once the management actions sit next to the
excursion: peak, break-even stop, nothing, exit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.postmortem import connect, find_trade, main, overview, report


@pytest.fixture
def journal(tmp_path: Path) -> Path:
    """A journal holding the NZDCAD shape: good peak, poor exit."""
    path = tmp_path / "trading.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE analysis_cycles (
            id INTEGER PRIMARY KEY, cycle_id TEXT, ts TEXT, symbol TEXT, mode TEXT,
            decision TEXT, reason TEXT, detail TEXT, direction TEXT,
            total_score REAL, score_threshold REAL, equity REAL, atr REAL,
            spread_pips REAL, session TEXT, volatility_regime TEXT,
            minutes_to_news REAL, context_json TEXT DEFAULT '{}');
        CREATE TABLE module_scores (
            id INTEGER PRIMARY KEY, cycle_pk INTEGER, module TEXT, score REAL,
            confidence REAL, reasoning TEXT);
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, cycle_pk INTEGER, ticket INTEGER, magic INTEGER,
            symbol TEXT, direction TEXT, volume REAL, entry_price REAL, sl REAL, tp REAL,
            risk_money REAL, risk_pct REAL, sl_distance_pips REAL, planned_rr REAL,
            opened_at TEXT, closed_at TEXT, exit_price REAL, exit_reason TEXT,
            pnl_money REAL, pnl_r REAL, mae_r REAL, mfe_r REAL, duration_seconds INTEGER,
            equity_before REAL, equity_after REAL);
        CREATE TABLE management_actions (
            id INTEGER PRIMARY KEY, trade_id INTEGER, ts TEXT, action TEXT,
            old_sl REAL, new_sl REAL, old_tp REAL, new_tp REAL, volume_closed REAL,
            r_at_action REAL, note TEXT DEFAULT '');
        """)
    db.execute(
        "INSERT INTO analysis_cycles (id, cycle_id, ts, symbol, mode, decision, reason, "
        "detail, direction, total_score, score_threshold, equity, session, "
        "volatility_regime, spread_pips, context_json) VALUES "
        "(1,'c1','2026-08-06T08:24:11Z','USDCHF.i','micro_live','TRADE','OK','', 'LONG',"
        "87.0, 40.0, 87.17, 'london+newyork', 'normal', 1.2, "
        '\'{"runway_minutes": 420.0, "activity_ratio": 1.1}\')'
    )
    db.execute(
        "INSERT INTO module_scores (cycle_pk, module, score, confidence, reasoning) "
        "VALUES (1,'market_structure', 72.0, 0.7, 'break of structure held')"
    )
    db.execute(
        "INSERT INTO module_scores (cycle_pk, module, score, confidence, reasoning) "
        "VALUES (1,'liquidity_sweep', 0.0, 0.0, 'not my setup')"
    )
    db.execute(
        "INSERT INTO trades (id, cycle_pk, ticket, magic, symbol, direction, volume, "
        "entry_price, sl, tp, risk_money, risk_pct, sl_distance_pips, planned_rr, "
        "opened_at, closed_at, exit_price, exit_reason, pnl_money, pnl_r, mae_r, mfe_r, "
        "duration_seconds, equity_before, equity_after) VALUES "
        "(1,1,777777,4242,'USDCHF.i','LONG',0.02,0.88000,0.87900,0.88200,1.74,2.0,10.0,2.0,"
        "'2026-08-06T08:24:45+00:00','2026-08-06T10:11:45+00:00',0.88013,'SL',"
        "0.22,0.13,-0.10,0.92,6420,87.17,87.39)"
    )
    for ts, action, r, old_sl, new_sl in (
        ("2026-08-06T09:02:00+00:00", "BREAK_EVEN", 0.61, 0.87900, 0.88010),
        ("2026-08-06T10:11:45+00:00", "BROKER_SL", 0.13, None, None),
    ):
        db.execute(
            "INSERT INTO management_actions (trade_id, ts, action, old_sl, new_sl, "
            "r_at_action, note) VALUES (1,?,?,?,?,?,'')",
            (ts, action, old_sl, new_sl, r),
        )
    db.commit()
    db.close()
    return path


def test_it_finds_the_trade_without_the_broker_suffix(journal: Path) -> None:
    """Nobody types `.i`."""
    db = connect(journal)
    trade = find_trade(db, "USDCHF", 0)
    assert trade is not None
    assert trade["symbol"] == "USDCHF.i"


def test_an_exact_ticket_wins(journal: Path) -> None:
    db = connect(journal)
    assert find_trade(db, "", 777777)["ticket"] == 777777
    assert find_trade(db, "", 111111) is None


def test_no_arguments_returns_the_most_recent(journal: Path) -> None:
    db = connect(journal)
    assert find_trade(db, "", 0)["ticket"] == 777777


def test_the_report_puts_the_peak_next_to_the_result(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The whole point. Neither number means much without the other."""
    db = connect(journal)
    report(db, find_trade(db, "USDCHF", 0))
    out = capsys.readouterr().out

    assert "+0.92R" in out  # what it was worth
    assert "+0.13R" in out  # what it returned
    assert "+1.60 EUR" in out  # the peak in money, computed from 1R
    assert "kept 14% of the best moment" in out
    assert "1.37 EUR was left on the table" in out


def test_it_says_so_when_most_of_the_gain_went_back(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    db = connect(journal)
    report(db, find_trade(db, "USDCHF", 0))
    assert "over half the gain was handed back" in capsys.readouterr().out


def test_the_timeline_names_the_rule_that_acted(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """BREAK_EVEN at +0.61R, then the stop it created taking the trade."""
    db = connect(journal)
    report(db, find_trade(db, "USDCHF", 0))
    out = capsys.readouterr().out

    assert "BREAK_EVEN" in out
    assert "stop moved to entry" in out
    assert "0.87900 -> 0.88010" in out


def test_a_silent_trade_is_reported_as_a_finding(tmp_path: Path, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """No management actions on a trade that peaked well is the answer.

    An empty section reads as missing data; it is not, and saying so is the
    difference between "the log is broken" and "no rule ever engaged".
    """
    db = sqlite3.connect(journal)
    db.execute("DELETE FROM management_actions")
    db.commit()
    db.close()

    connection = connect(journal)
    report(connection, find_trade(connection, "USDCHF", 0))
    assert "ran from entry to exit untouched" in capsys.readouterr().out


def test_it_shows_why_the_trade_was_taken(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    db = connect(journal)
    report(db, find_trade(db, "USDCHF", 0))
    out = capsys.readouterr().out

    assert "87.0 against a 40.0 threshold" in out
    assert "market_structure" in out
    # A module scoring zero is saying "not my setup", not voting against. It is
    # noise in a postmortem and is left out.
    assert "liquidity_sweep" not in out
    assert "runway_minutes" in out


def test_listing_recent_trades_works(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--list", "5", "--db", str(journal)]) == 0
    assert "USDCHF.i" in capsys.readouterr().out


def test_an_unknown_symbol_says_so_and_offers_the_list(journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["GBPJPY", "--db", str(journal)]) == 1
    assert "--list" in capsys.readouterr().out


def test_a_missing_journal_is_not_a_traceback(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--db", str(tmp_path / "nope.db")]) == 1
    assert "No journal" in capsys.readouterr().out


class TestActionLabels:
    """A broker closure is not one of our exits, and should not read like one."""

    def test_our_own_exits_are_explained(self) -> None:
        from scripts.postmortem import describe

        assert describe("PEAK_STALL") == "banked: the peak stopped advancing"
        assert describe("PROFIT_LOCK").startswith("stop walked up")

    def test_broker_closures_are_named_by_their_cause(self) -> None:
        from scripts.postmortem import describe

        assert describe("BROKER_SL") == "the stop was hit"
        assert describe("BROKER_TP") == "the target was hit"

    def test_an_unknown_broker_reason_still_reads_as_a_closure(self) -> None:
        from scripts.postmortem import describe

        assert describe("BROKER_WHATEVER") == "closed at the broker"

    def test_an_action_with_no_gloss_prints_nothing_rather_than_a_guess(self) -> None:
        """An unexplained action beats a wrong explanation."""
        from scripts.postmortem import describe

        assert describe("SOME_FUTURE_RULE") == ""


class TestUnclosedTrades:
    """ "Still open" is a claim the journal cannot make.

    All it knows is that no closure has been written. Reconciliation only runs
    inside a full cycle, so a stopped Jarvis leaves every broker closure
    unrecorded — and an operator who has just watched the position vanish from
    MT5 reads "still open" as the report being broken.
    """

    @staticmethod
    def without_closure(path: Path) -> None:
        db = sqlite3.connect(path)
        db.execute("UPDATE trades SET closed_at=NULL, exit_price=NULL, pnl_money=NULL, pnl_r=NULL")
        db.commit()
        db.close()

    def test_it_does_not_claim_the_position_is_open(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        self.without_closure(journal)
        db = connect(journal)
        report(db, find_trade(db, "USDCHF", 0))
        out = capsys.readouterr().out

        assert "still open" not in out
        assert "no closure recorded" in out

    def test_it_names_reconciliation_as_the_thing_to_check(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        self.without_closure(journal)
        db = connect(journal)
        report(db, find_trade(db, "USDCHF", 0))
        out = capsys.readouterr().out

        assert "reconcile" in out.lower()
        assert "Jarvis is actually running" in out

    def test_a_closed_trade_says_nothing_about_reconciliation(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        db = connect(journal)
        report(db, find_trade(db, "USDCHF", 0))
        out = capsys.readouterr().out

        assert "no closure recorded" not in out
        assert "reconciled yet" not in out


class TestOverview:
    """Every recent trade on one screen, with the two columns that matter.

    `kept` separates a losing strategy from one that wins and hands it back;
    neither `pnl_r` nor `mfe_r` says that alone. `by` says whether anything in
    this system chose the exit — a column of nothing but BROKER_SL means no
    rule ever acted, which was true here for a long time and invisible.
    """

    def test_it_reports_what_survived_to_the_exit(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Peak 0.92R, returned 0.13R: 14% kept."""
        overview(connect(journal), 10)
        out = capsys.readouterr().out
        assert "14%" in out

    def test_it_counts_who_chose_the_exit(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        overview(connect(journal), 10)
        out = capsys.readouterr().out
        assert "1 closed · 0 exited by a rule of ours · 1 by the broker" in out

    def test_it_says_so_when_no_rule_ever_acted(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """The finding that took a guard-loop bug and a live account to notice."""
        overview(connect(journal), 10)
        assert "not one exit was chosen by this system" in capsys.readouterr().out

    def test_a_rule_exit_is_credited(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        db = sqlite3.connect(journal)
        db.execute("UPDATE trades SET exit_reason = 'PEAK_STALL'")
        db.commit()
        db.close()

        overview(connect(journal), 10)
        out = capsys.readouterr().out
        assert "1 exited by a rule of ours · 0 by the broker" in out
        assert "not one exit was chosen" not in out

    def test_it_flags_a_trade_that_handed_its_gain_back(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        overview(connect(journal), 10)
        assert "kept under half of their best moment" in capsys.readouterr().out

    def test_an_open_trade_is_not_counted_as_closed(self, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        db = sqlite3.connect(journal)
        db.execute("UPDATE trades SET closed_at = NULL, exit_reason = NULL")
        db.commit()
        db.close()

        overview(connect(journal), 10)
        out = capsys.readouterr().out
        assert "0 closed" in out
        assert "open" in out

    def test_an_empty_journal_says_so(self, tmp_path: Path, journal: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        db = sqlite3.connect(journal)
        db.execute("DELETE FROM trades")
        db.commit()
        db.close()

        overview(connect(journal), 10)
        assert "No trades recorded yet" in capsys.readouterr().out
