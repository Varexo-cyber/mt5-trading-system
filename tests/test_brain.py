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
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.store import (
    DSN_ENV,
    MIN_TRADES_TO_GRADE_A_MODULE,
    MIN_TRADES_TO_SPLIT_SIDES,
    Brain,
    BrainStatus,
    Lesson,
    NullBrain,
    Scoreline,
    _signal_rows,
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
        assert null.record_counterfactual(symbol="EURUSD") is None
        assert null.record_counterfactuals([]) is None
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
        for table in (
            "decisions",
            "counterfactuals",
            "trades",
            "trade_events",
            "lessons",
            "headlines",
        ):
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


class TestNothingIsEverDeleted:
    """The operator's requirement, stated plainly: this data must never be
    thrown away. The local JSON memory prunes on a retention window, which is
    the behaviour the database exists to replace — a market regime from three
    months ago is weak evidence about this morning, but it is still evidence,
    and it is the operator's to discard rather than the system's.
    """

    def source(self) -> str:
        from brain.store import SCHEMA_PATH

        return (SCHEMA_PATH.parent / "store.py").read_text(encoding="utf-8")

    def test_the_store_issues_no_delete_of_any_kind(self) -> None:
        text = self.source().upper()

        for destructive in ("DELETE FROM", "TRUNCATE", "DROP TABLE", "DROP VIEW"):
            assert destructive not in text, destructive

    def test_there_is_no_retention_window(self) -> None:
        """`learning/memory.py` has RETENTION_DAYS and prunes against it. This
        deliberately has no equivalent, and adding one would be a decision to
        make the account forget."""
        text = self.source().lower()

        assert "retention" not in text
        assert "_prune" not in text

    def test_the_schema_creates_but_never_drops(self) -> None:
        from brain.store import SCHEMA_PATH

        sql = SCHEMA_PATH.read_text(encoding="utf-8").upper()

        assert "DROP " not in sql
        assert "TRUNCATE" not in sql

    def test_a_repeated_decision_updates_rather_than_replaces(self) -> None:
        """`ON CONFLICT DO UPDATE` on the detail only. A retry must not be able
        to overwrite the equity, the verdict or the headlines that were true at
        the moment the decision was first made."""
        from brain.store import SCHEMA_PATH

        sql = (SCHEMA_PATH.parent / "store.py").read_text(encoding="utf-8")

        assert "ON CONFLICT (fingerprint) DO UPDATE SET detail = EXCLUDED.detail" in sql

    def test_a_repeated_headline_is_ignored_rather_than_rewritten(self) -> None:
        """First sighting keeps the timestamp. A story re-published by a fourth
        wire is not a fresh event."""
        assert "ON CONFLICT (fingerprint) DO NOTHING" in self.source()


class TestLearningWhenToTakeIt:
    """The account reading its own history to decide when profit is enough.

    The query needs a real Postgres to run, so what is held here is the
    contract around it: how little evidence is too little, what None means,
    and that the floor cannot be argued below the cost of collecting.
    """

    def brain_answering(self, rows) -> Brain:  # type: ignore[no-untyped-def]
        made = Brain("postgresql://u:p@host/db", account="test")
        made._run = lambda *_args, **_kwargs: rows  # type: ignore[assignment]
        return made

    def test_it_says_nothing_until_there_are_enough_trades(self) -> None:
        """A threshold learned from fifteen trades is a threshold learned from
        one week's weather, and it would then govern real exits."""
        thin = [(1, 12, 0.40, -0.30), (2, 9, 0.80, -0.10)]

        assert self.brain_answering(thin).learned_bank_threshold() is None

    def test_it_answers_once_the_evidence_is_there(self) -> None:
        rows = [(1, 25, 0.40, -0.30), (2, 20, 0.80, 0.10)]

        assert self.brain_answering(rows).learned_bank_threshold() == pytest.approx(0.40)

    def test_it_takes_the_lowest_band_that_pays_not_the_best_one(self) -> None:
        """The rule fires on the way up, so it only ever acts at the first
        level price crosses. Choosing the band with the biggest gap would be
        choosing a level most trades never reach."""
        rows = [(1, 20, 0.35, 0.10), (2, 20, 1.60, -0.90)]  # both pay, 1.60 pays more

        assert self.brain_answering(rows).learned_bank_threshold() == pytest.approx(0.35)

    def test_a_band_where_holding_won_is_skipped(self) -> None:
        rows = [(1, 20, 0.30, 0.55), (2, 25, 0.90, 0.20)]  # first band did better held

        assert self.brain_answering(rows).learned_bank_threshold() == pytest.approx(0.90)

    def test_it_says_nothing_when_holding_won_everywhere(self) -> None:
        """None leaves the configured value in charge, which is the same
        behaviour as before this existed."""
        rows = [(1, 25, 0.30, 0.55), (2, 25, 0.90, 1.20)]

        assert self.brain_answering(rows).learned_bank_threshold() is None

    def test_it_never_goes_below_what_the_round_trip_costs(self) -> None:
        """Banking under the spread and commission converts a small win into a
        small loss, however confidently the history points there."""
        from brain.store import MIN_LEARNED_BANK_R

        rows = [(1, 50, 0.02, -0.90)]

        answer = self.brain_answering(rows).learned_bank_threshold()

        assert answer == pytest.approx(MIN_LEARNED_BANK_R)

    def test_an_unreachable_database_has_no_opinion(self) -> None:
        assert brain_that_cannot_connect().learned_bank_threshold() is None

    def test_the_null_brain_has_no_opinion_either(self) -> None:
        assert NullBrain().learned_bank_threshold() is None

    def test_no_rows_at_all_is_no_opinion(self) -> None:
        assert self.brain_answering([]).learned_bank_threshold() is None


class TestGradingTheAdviser:
    """The loop nothing could see. Every other table records the account
    grading its own rules; this records it grading its own adviser."""

    def test_the_schema_keeps_supervisions_apart_from_decisions(self) -> None:
        """A decision is asked once, before anything exists. A supervision is
        asked repeatedly of a position already running. Folding them together
        would make "how often was the reviewer right" a query nobody could
        write correctly."""
        from brain.store import SCHEMA_PATH

        sql = SCHEMA_PATH.read_text(encoding="utf-8")

        assert "CREATE TABLE IF NOT EXISTS supervisions" in sql
        assert "CREATE OR REPLACE VIEW supervision_outcomes" in sql

    def test_the_view_joins_the_verdict_to_what_the_trade_did(self) -> None:
        from brain.store import SCHEMA_PATH

        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        view = sql[sql.index("CREATE OR REPLACE VIEW supervision_outcomes") :]

        assert "trade_ended_at_r" in view
        assert "JOIN trades" in view

    def test_it_records_whether_the_verdict_was_carried_out(self) -> None:
        """A verdict the risk layer refused is still evidence about the
        adviser. Counting it as acted-upon would credit or blame it for
        something that never happened."""
        source = (
            __import__("pathlib").Path(__file__).resolve().parent.parent / "brain" / "store.py"
        ).read_text(encoding="utf-8")

        assert "applied: bool = False" in source

    def test_a_hold_is_written_too(self) -> None:
        """The most informative row this table can carry is a hold that
        preceded a full stop-out, and the old code returned before recording
        anything on a hold."""
        source = (
            __import__("pathlib").Path(__file__).resolve().parent.parent / "runner" / "service.py"
        ).read_text(encoding="utf-8")
        call = source[source.index("self.brain.record_supervision(") :]

        # The early `continue` on hold is gone: the verdict is recorded first
        # and the action decides only whether an event was applied.
        assert 'if verdict.action == "hold"' in source
        assert "applied=event is not None" in call[:600]

    def test_it_stays_quiet_against_a_dead_database(self) -> None:
        brain = brain_that_cannot_connect()

        assert (
            brain.record_supervision(trade_id=1, asked_at=NOW, symbol="EURUSD", action="hold")
            is None
        )

    def test_the_null_brain_accepts_it(self) -> None:
        assert NullBrain().record_supervision(trade_id=1, action="hold") is None


class TestTheBrainCatchesUpOnRealisedTrades:
    """The account could not read its own history, and it cost money.

    The brain only ever received trades opened after it was armed, so a journal
    holding forty-seven closed trades faced a Neon table holding twenty-two.
    `learned_bank_threshold` needs forty before it will speak, so it stayed
    silent, `bank_at_r` stayed at its configured 0.30, and a USDCAD short that
    peaked at +0.23R -- seven hundredths under the line -- was closed at -0.10R
    instead of banked. The floor the learned threshold would have returned is
    0.15R, which takes that trade.

    Counterfactuals already had this catch-up. Realised trades, which are what
    every learned threshold is actually built on, did not.
    """

    def captured(self) -> tuple[Brain, list[tuple]]:
        brain = Brain("postgresql://u:p@host/db", account="test")
        seen: list[tuple] = []

        def run(sql, params=None, fetch=None):  # type: ignore[no-untyped-def]
            seen.append((sql, params))
            return

        brain._run = run  # type: ignore[assignment]
        return brain, seen

    def row(self, ticket: int) -> dict[str, object]:
        return {
            "ticket": ticket,
            "symbol": "USDCAD.i",
            "direction": "SHORT",
            "volume": 0.03,
            "opened_at": NOW,
            "entry": 1.39333,
            "stop_loss": 1.39475,
            "take_profit": 1.39040,
            "risk_money": 2.82,
            "closed_at": NOW,
            "exit_price": 1.39348,
            "exit_reason": "AI_CLOSE_SENT",
            "pnl_money": -0.28,
            "pnl_r": -0.10,
            "mfe_r": 0.23,
            "mae_r": -0.17,
        }

    def test_a_batch_goes_in_one_round_trip(self) -> None:
        """Forty-seven single statements against a remote database at startup
        would delay the first scan; Postgres expands one JSON payload."""
        brain, seen = self.captured()

        sent = brain.record_trade_history([self.row(i) for i in range(47)])

        assert sent == 47
        assert len(seen) == 1, "one statement, not forty-seven"

    def test_it_never_overwrites_what_the_runner_already_wrote(self) -> None:
        """A live row carries its decision_id. A backfilled copy knows less."""
        brain, seen = self.captured()
        brain.record_trade_history([self.row(1)])
        sql = seen[0][0]

        assert "ON CONFLICT (account, ticket, opened_at) DO NOTHING" in sql
        assert "DO UPDATE" not in sql

    def test_the_excursions_the_thresholds_are_built_on_are_carried(self) -> None:
        """`learned_bank_threshold` reads mfe_r against pnl_r and nothing else.

        A backfill that dropped those columns would fill the table and still
        leave the account unable to learn, which is the failure this fixes
        wearing a different face.
        """
        brain, seen = self.captured()
        brain.record_trade_history([self.row(1)])
        sql = seen[0][0]

        for column in ("mfe_r", "mae_r", "pnl_r", "exit_reason"):
            assert column in sql

    def test_an_empty_journal_costs_no_round_trip(self) -> None:
        brain, seen = self.captured()

        assert brain.record_trade_history([]) == 0
        assert seen == []

    def test_the_null_brain_answers_it_too(self) -> None:
        assert NullBrain().record_trade_history([{"ticket": 1}]) == 0


class TestTheOperatorCanSeeWhatItLearned:
    """Row counts prove the memory fills. They do not prove anything reads it.

    A database nothing reads is decoration, and until now the two places a
    stored trade actually changes behaviour were invisible even when they
    worked: nothing printed the banking threshold the history produced, and
    nothing printed the ranking adjustments. "Is it learning" had no answer
    short of reading the source.
    """

    def test_the_report_shows_both_learners(self) -> None:
        source = (Path(__file__).resolve().parent.parent / "scripts" / "verify_brain.py").read_text(
            encoding="utf-8"
        )

        assert "learned_bank_threshold()" in source, "the threshold that changes when it banks"
        assert "edge_calibrations(" in source, "the adjustment that changes what ranks first"

    def test_it_says_what_the_threshold_will_actually_do(self) -> None:
        """A learned number nobody can compare to the configured one is noise.

        `_worth_taking` takes the minimum of the two, so the operator needs
        both and the result, not one figure in isolation.
        """
        source = (Path(__file__).resolve().parent.parent / "scripts" / "verify_brain.py").read_text(
            encoding="utf-8"
        )

        assert "bank_at_r" in source
        assert "min(configured, learned)" in source

    def test_it_says_plainly_when_there_is_not_enough_evidence_yet(self) -> None:
        """Silence reads as breakage. "Not yet, and here is what it needs" does not."""
        source = (Path(__file__).resolve().parent.parent / "scripts" / "verify_brain.py").read_text(
            encoding="utf-8"
        )

        assert "not yet" in source
        assert "MIN_TRADES_TO_LEARN" in source


class TestWhichSideTheAccountLosesOn:
    """The largest measured asymmetry in the record, and it was never said.

    `scoreboard` groups by symbol *and* direction. Sixty-four trades spread
    across thirty instruments come back as thirty lines of one and two trades,
    and the reviewer was handed the top twelve of them. Pooled by side the same
    trades carry a real finding — one book losing several times what the other
    lost over a comparable number of trades — and no chart shows it.
    """

    def brain_answering(self, rows) -> Brain:  # type: ignore[no-untyped-def]
        made = Brain("postgresql://u:p@host/db", account="test")
        made._run = lambda *_args, **_kwargs: rows  # type: ignore[assignment]
        return made

    def test_it_reports_per_trade_expectancy_not_just_a_total(self) -> None:
        """A total is a function of how many trades were taken. The per-trade
        figure is the one that compares the two sides."""
        brain = self.brain_answering([("LONG", 27, -6.95, 7), ("SHORT", 20, -2.02, 8)])

        long_side, short_side = brain.side_records()

        assert long_side.mean_r == pytest.approx(-0.257, abs=0.001)
        assert short_side.mean_r == pytest.approx(-0.101, abs=0.001)
        assert "-0.26R per trade" in long_side.summary()
        assert "26% won" in long_side.summary()

    def test_a_side_with_too_few_trades_is_not_reported_at_all(self) -> None:
        """The floor is enforced in SQL, so what is asserted here is that the
        configured minimum is what actually reaches the database. Reporting
        "LONG: 3 trades, -1.20R per trade" is not a weaker finding, it is a
        false one, and a reviewer handed a number will use it."""
        seen: list[object] = []
        brain = Brain("postgresql://u:p@host/db", account="test")

        def run(_sql: str, params=(), **_kwargs):  # type: ignore[no-untyped-def]
            seen.append(params)
            return []

        brain._run = run  # type: ignore[assignment]

        assert brain.side_records() == []
        assert seen[0][-1] == MIN_TRADES_TO_SPLIT_SIDES

    def brain_briefing_on(self, sides) -> Brain:  # type: ignore[no-untyped-def]
        """A store where only the per-side query answers.

        `briefing` fans out over four queries and they return different row
        shapes, so a stub that answers them all identically crashes on whichever
        one it does not fit rather than testing anything.
        """
        made = Brain("postgresql://u:p@host/db", account="test")

        def run(sql: str, _params=(), **_kwargs):  # type: ignore[no-untyped-def]
            return sides if "GROUP BY direction" in sql else []

        made._run = run  # type: ignore[assignment]
        return made

    def test_the_briefing_carries_it_and_says_it_is_not_a_veto(self) -> None:
        """A losing side must raise the bar for the setup in front of the
        reviewer, never close that direction outright. One month of weather is
        not a reason to stop trading half the market."""
        brain = self.brain_briefing_on([("LONG", 27, -6.95, 7)])

        section = brain.briefing("EURUSD", "LONG")["how_each_side_has_actually_done"]

        assert "LONG: 27 trades" in section["records"][0]
        assert "not as a standing" in section["weight"]

    def test_an_empty_record_leaves_the_payload_untouched(self) -> None:
        """An empty section is tokens spent telling the reviewer nothing, and
        worse, it makes a memory with no evidence look like one with some."""
        brain = self.brain_briefing_on([])

        assert "how_each_side_has_actually_done" not in brain.briefing("EURUSD", "LONG")

    def test_no_database_answers_the_way_an_empty_one_would(self) -> None:
        assert NullBrain().side_records() == []


class TestWhichDetectorActuallyEarnsItsKeep:
    """The largest hole in the record, and the one that decides whether this
    system can improve at all.

    `conviction` stored the blended score and nothing stored what produced it.
    Sixty-four closed trades were therefore sixty-four undifferentiated data
    points, and the only lesson available from them was "we are down" -- which
    names nothing to stop doing. With attribution the same trades become votes
    on each detector.
    """

    def brain_answering(self, rows) -> Brain:  # type: ignore[no-untyped-def]
        made = Brain("postgresql://u:p@host/db", account="test")
        made._run = lambda *_args, **_kwargs: rows  # type: ignore[assignment]
        return made

    def test_a_losing_detector_is_named_with_its_per_trade_cost(self) -> None:
        brain = self.brain_answering([("trend_momentum", 31, -4.20, 9)])

        record = brain.module_records()[0]

        assert record.mean_r == pytest.approx(-0.135, abs=0.001)
        assert "trend_momentum: 31 trades found" in record.summary()
        assert "-0.14R each" in record.summary()

    def test_a_detector_with_too_few_trades_is_not_graded(self) -> None:
        """Shown at four trades it reads as a verdict on the detector, and it
        is a verdict on four trades. Switching off the wrong one removes setups
        permanently, so the floor is higher than for the side split."""
        seen: list[object] = []
        brain = Brain("postgresql://u:p@host/db", account="test")

        def run(_sql: str, params=(), **_kwargs):  # type: ignore[no-untyped-def]
            seen.append(params)
            return []

        brain._run = run  # type: ignore[assignment]

        assert brain.module_records() == []
        assert seen[0][-1] == MIN_TRADES_TO_GRADE_A_MODULE
        assert MIN_TRADES_TO_GRADE_A_MODULE > MIN_TRADES_TO_SPLIT_SIDES

    def test_no_database_answers_the_way_an_empty_one_would(self) -> None:
        assert NullBrain().module_records() == []


class TestOnlyModulesThatSpokeAreRecorded:
    """A neutral module is the absence of evidence. Storing forty thousand rows
    of "trend_momentum saw nothing" would turn every per-module average into a
    measure of how often the detector is quiet rather than whether it is right.
    """

    def signal(self, module: str, score: float, confidence: float = 0.7):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            module=module, score=score, confidence=confidence, details={"bias_confirmed": False}
        )

    def test_a_neutral_module_is_dropped(self) -> None:
        rows = _signal_rows(
            [self.signal("trend_momentum", 65.0), self.signal("level_reaction", 0.0)]
        )

        assert [row["module"] for row in rows] == ["trend_momentum"]

    def test_the_side_the_module_argued_for_is_kept(self) -> None:
        """Grouping a detector's wins without its direction would average a
        module that is right on shorts and wrong on longs into "neutral"."""
        rows = _signal_rows([self.signal("liquidity_sweep", -75.0)])

        assert rows[0]["direction"] == "SHORT"

    def test_whatever_the_module_recorded_about_itself_survives(self) -> None:
        """`bias_confirmed` rides here. It is the whole point of storing the
        details rather than only the score."""
        rows = _signal_rows([self.signal("trend_momentum", 65.0)])

        assert rows[0]["details"]["bias_confirmed"] is False

    def test_a_signal_object_missing_fields_does_not_break_the_write(self) -> None:
        """This runs inside the trade-recording path. A malformed signal must
        cost a row of analytics, never the record of the trade."""
        assert _signal_rows([SimpleNamespace()]) == []
