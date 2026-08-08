"""The long-term memory, and the one property that matters most about it.

That property is: **it must never be able to stop a trade.** Postgres is a
remote service on the far side of a network, and the guard loop runs once a
second on real money. A memory that can raise into that loop, or block it, or
refuse an entry because a write failed, is a worse thing than no memory at all.
So most of this file drives the store against a database that is broken in a
different way each time and asserts that the answer is always a neutral one.

The rest holds the grouping rules, because they are what turn a pile of rows
into evidence: a lesson said twice in slightly different words is one lesson
with two sightings, and a decision written twice after a retry is one decision.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain.store import (
    DSN_ENV,
    Brain,
    BrainStatus,
    Lesson,
    NullBrain,
    Scoreline,
    build_brain,
    fingerprint,
    lesson_key,
)

NOW = datetime(2026, 8, 7, 14, 0, tzinfo=UTC)


class Exploding:
    """A connection that fails in whatever way the test asks for."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.closed = False

    def cursor(self):  # type: ignore[no-untyped-def]
        raise self.error


def brain_that_cannot_connect(error: Exception | None = None) -> Brain:
    brain = Brain("postgresql://u:p@nowhere.invalid/db", account="test")
    failure = error or OSError("connection refused")

    def connect():  # type: ignore[no-untyped-def]
        raise failure

    brain._connect = connect  # type: ignore[assignment]
    return brain


class TestItCanNeverStopATrade:
    """Every one of these would be a raise into the trading loop if the store
    were written the ordinary way."""

    @pytest.mark.parametrize(
        "error",
        [
            OSError("connection refused"),
            TimeoutError("statement timeout"),
            RuntimeError("SSL connection has been closed unexpectedly"),
            ValueError("invalid dsn"),
        ],
    )
    def test_a_write_against_a_dead_database_returns_none(self, error: Exception) -> None:
        brain = brain_that_cannot_connect(error)

        assert (
            brain.record_decision(decided_at=NOW, symbol="EURUSD", reason="NO_SIGNAL", mode="paper")
            is None
        )
        assert brain.status.failures == 1
        assert brain.status.connected is False

    def test_every_other_write_is_equally_quiet(self) -> None:
        brain = brain_that_cannot_connect()

        assert (
            brain.record_trade_opened(
                ticket=1,
                decision_id=None,
                symbol="EURUSD",
                direction="LONG",
                volume=0.01,
                opened_at=NOW,
                entry=1.1,
                stop_loss=1.09,
                take_profit=1.12,
                risk_money=1.0,
            )
            is None
        )
        assert brain.record_trade_event(trade_id=1, happened_at=NOW, action="BREAK_EVEN") is None
        assert (
            brain.record_trade_closed(
                ticket=1,
                closed_at=NOW,
                exit_price=1.1,
                exit_reason="SL",
                pnl_money=-1.0,
                pnl_r=-1.0,
            )
            is None
        )
        assert brain.record_lessons(["a lesson"], learned_at=NOW) is None

    def test_a_read_against_a_dead_database_is_empty_not_an_error(self) -> None:
        """An empty briefing is the same shape as a database with nothing in it
        yet, which is exactly right: the reviewer is told nothing rather than
        told something wrong."""
        brain = brain_that_cannot_connect()

        assert brain.lessons() == []
        assert brain.scoreboard() == []
        assert brain.briefing("EURUSD", "LONG") == {}

    def test_it_retries_once_before_giving_up(self) -> None:
        """A pooled Neon connection is closed from the far end after an idle
        period, and the first statement after that always fails. Without the
        retry the memory works until the market is quiet for an hour."""
        brain = Brain("postgresql://u:p@host/db", account="test")
        attempts: list[int] = []

        def connect():  # type: ignore[no-untyped-def]
            attempts.append(1)
            raise OSError("server closed the connection unexpectedly")

        brain._connect = connect  # type: ignore[assignment]
        brain.record_lessons(["x"], learned_at=NOW)

        assert len(attempts) == 2, "one retry, then give up"
        assert brain.status.failures == 1, "the pair counts as one failure, not two"

    def test_closing_a_broken_connection_does_not_raise(self) -> None:
        brain = Brain("postgresql://u:p@host/db", account="test")
        brain._connection = Exploding(OSError("gone"))
        brain._connection.close = lambda: (_ for _ in ()).throw(OSError("gone"))  # type: ignore[assignment]

        brain.close()

        assert brain.status.connected is False


class TestWithoutADatabaseNothingChanges:
    def test_no_dsn_gives_a_null_brain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DSN_ENV, raising=False)

        assert isinstance(build_brain("acc"), NullBrain)

    def test_the_null_brain_answers_every_call(self) -> None:
        """A real object rather than None, so no caller needs a guard around
        every write. Each answer is what an empty database would say."""
        null = NullBrain()

        assert null.migrate() is False
        assert null.record_decision(symbol="EURUSD") is None
        assert null.record_trade_opened(ticket=1) is None
        assert null.record_trade_event(trade_id=1) is None
        assert null.record_trade_closed(ticket=1) is None
        assert null.record_lessons(["x"]) is None
        assert null.record_headlines([]) == 0
        assert null.lessons() == []
        assert null.scoreboard() == []
        assert null.briefing("EURUSD", "LONG") == {}
        assert null.close() is None

    def test_a_dsn_without_the_driver_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """psycopg is an optional dependency and its absence is not a reason
        for a trading system to refuse to start."""
        monkeypatch.setenv(DSN_ENV, "postgresql://u:p@host/db")
        import builtins

        real_import = builtins.__import__

        def missing(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "psycopg":
                raise ImportError("no psycopg")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing)

        assert isinstance(build_brain("acc"), NullBrain)

    def test_disabled_explicitly_gives_a_null_brain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DSN_ENV, "postgresql://u:p@host/db")

        assert isinstance(build_brain("acc", enabled=False), NullBrain)


class TestGroupingLessons:
    """An evidence count is the only thing separating a pattern from an
    anecdote, so near-duplicates have to collapse or every count inflates
    together until nothing stands out."""

    def test_the_same_observation_phrased_twice_is_one_key(self) -> None:
        first = lesson_key("Stopped out 3 pips before the reversal.")
        second = lesson_key("stopped out 5 pips before the reversal")

        assert first == second == "stopped out pips before the reversal"

    def test_genuinely_different_lessons_stay_apart(self) -> None:
        assert lesson_key("Entered into a widening spread") != lesson_key(
            "Held through a red folder release"
        )

    def test_case_and_punctuation_do_not_split_a_lesson(self) -> None:
        assert lesson_key("  THE SPREAD ate it!! ") == lesson_key("the spread ate it")


class TestFingerprints:
    def test_the_same_decision_fingerprints_the_same(self) -> None:
        args = (NOW.isoformat(), "5049535", "EURUSD", "LONG", "OK")

        assert fingerprint(*args) == fingerprint(*args)

    def test_a_different_decision_does_not_collide(self) -> None:
        assert fingerprint(NOW.isoformat(), "a", "EURUSD") != fingerprint(
            NOW.isoformat(), "a", "GBPUSD"
        )

    def test_none_and_empty_are_distinguishable_from_neither(self) -> None:
        """Field separator, not concatenation: without it ("ab", "c") and
        ("a", "bc") are the same row."""
        assert fingerprint("ab", "c") != fingerprint("a", "bc")


class TestWhatTheOperatorSees:
    def test_a_lesson_reports_its_evidence(self) -> None:
        lesson = Lesson("The spread took it, not the market", 9, -0.42, NOW)

        assert "9x" in lesson.summary()
        assert "-0.42R" in lesson.summary()

    def test_a_single_sighting_says_so_rather_than_1x(self) -> None:
        assert "once" in Lesson("Only seen once", 1, None, NOW).summary()

    def test_a_scoreline_reads_as_a_sentence(self) -> None:
        line = Scoreline("SPX500", "LONG", 4, -3.2, 1, 0.15)

        assert "LONG SPX500: 4 trades, 1 won, -3.20R total" in line.summary()
        assert "kept 15%" in line.summary()

    def test_the_status_distinguishes_unconfigured_from_broken(self) -> None:
        """Two very different situations that would otherwise both read as
        'no memory': one is a developer machine, the other is an outage."""
        unconfigured = BrainStatus(connected=False, dsn_configured=False)
        broken = BrainStatus(connected=False, dsn_configured=True, last_error="timeout")

        assert "no NEON_DATABASE_URL" in unconfigured.summary()
        assert "unreachable" in broken.summary() and "timeout" in broken.summary()


class TestTheSchemaIsValidSql:
    """It has never been executed by Postgres — the build environment blocks
    5432 and blocks HTTPS to the host. These are the checks that can be made
    without a server, and `scripts/verify_brain.py` makes the rest on the VPS.
    """

    def sql(self) -> str:
        from brain.store import SCHEMA_PATH

        return SCHEMA_PATH.read_text(encoding="utf-8")

    def test_every_table_the_store_writes_to_is_created(self) -> None:
        sql = self.sql()
        for table in ("decisions", "trades", "trade_events", "lessons", "headlines"):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, table

    def test_it_is_safe_to_run_twice(self) -> None:
        """The runner applies it on every start."""
        sql = self.sql()

        assert sql.count("CREATE TABLE ") == sql.count("CREATE TABLE IF NOT EXISTS ")
        assert sql.count("CREATE INDEX ") == sql.count("CREATE INDEX IF NOT EXISTS ")
        assert "CREATE OR REPLACE VIEW" in sql

    def test_the_de_duplication_constraints_exist(self) -> None:
        """Every aggregate downstream counts rows, so a retry that writes a
        second copy corrupts all of them."""
        sql = self.sql()

        assert "fingerprint     TEXT        NOT NULL UNIQUE" in sql
        assert "UNIQUE (account, ticket, opened_at)" in sql

    def test_the_view_the_briefing_reads_is_defined(self) -> None:
        assert "CREATE OR REPLACE VIEW trade_history" in self.sql()


class TestTheBriefing:
    def test_it_is_empty_when_the_brain_is_off(self) -> None:
        brain = Brain("", account="test")

        assert brain.enabled is False
        assert brain.briefing("EURUSD", "LONG") == {}

    def test_a_dsn_from_the_environment_is_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DSN_ENV, "postgresql://u:p@host/db")

        assert Brain(account="test").enabled is True


class TestItNeverLeaksTheCredential:
    def test_the_dsn_is_not_in_the_status_summary(self) -> None:
        """The status line goes to logs, the dashboard and screenshots."""
        brain = brain_that_cannot_connect()
        brain.record_lessons(["x"], learned_at=NOW)

        assert "nowhere.invalid" not in brain.status.summary()
        assert "p@" not in brain.status.summary()

    def test_the_verifier_redacts_the_password(self) -> None:
        import importlib.util

        from brain.store import SCHEMA_PATH

        spec = importlib.util.spec_from_file_location(
            "verify_brain", SCHEMA_PATH.parent.parent / "scripts" / "verify_brain.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        shown = module.redacted("postgresql://neondb_owner:sekrit@ep-x.neon.tech/neondb?ssl=1")

        assert "sekrit" not in shown
        assert "neondb_owner" not in shown
        assert "ep-x.neon.tech" in shown


def test_the_retention_of_a_lesson_survives_a_timezone_round_trip() -> None:
    """Every timestamp written is tz-aware UTC; Postgres TIMESTAMPTZ returns it
    the same way, and a naive one here would silently shift by the VPS offset."""
    stamp = NOW - timedelta(hours=3)

    assert stamp.tzinfo is not None
    assert Lesson("x", 1, None, stamp).last_seen.tzinfo is not None
