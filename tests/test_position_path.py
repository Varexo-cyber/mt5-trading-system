"""Every second of a position, kept somewhere that outlives the VPS.

Everything the brain stored recorded a MOMENT a decision was taken: opened,
banked, closed, reviewed. Nothing recorded what a trade was doing in between,
so every question about management was answered from its endpoints. "Should
the stop have gone to entry sooner" and "how long did it sit at its high before
giving up" are questions about the path.

The live case: a CADCHF long showing EUR 2.82 on a EUR 130 account with the
broker stop still twelve pips below entry. Whether holding that was right is
answerable only against what price did next, second by second, and the only
copy of that lived in SQLite on a rented machine.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from brain.store import Brain, BrainStatus, NullBrain

NOW = datetime(2026, 8, 14, 19, 40, tzinfo=UTC)


class _Cursor:
    def __init__(self, sink: dict) -> None:
        self.sink = sink

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def executemany(self, sql: str, rows: list) -> None:
        self.sink["sql"] = sql
        self.sink["rows"] = rows


class _Connection:
    closed = False

    def __init__(self, sink: dict, fail_times: int = 0) -> None:
        self.sink = sink
        self.fail_times = fail_times

    def cursor(self) -> _Cursor:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("connection reset by peer")
        return _Cursor(self.sink)


def brain(fail_times: int = 0) -> tuple[Brain, dict]:
    sink: dict = {}
    instance = Brain.__new__(Brain)
    instance._lock = threading.Lock()
    instance._connection = _Connection(sink, fail_times)
    instance.enabled = True
    instance.status = BrainStatus(connected=True, dsn_configured=True)
    instance.account = "5049535"
    return instance, sink


def sample(**overrides: object) -> dict:
    row = {
        "trade_id": 7,
        "sampled_at": NOW,
        "price": 0.58595,
        "r_now": 0.44,
        "peak_r": 0.44,
        "money": 2.82,
        "stop_price": 0.58422,
        "stop_r": -1.0,
        "protected": False,
        "health": "healthy",
    }
    row.update(overrides)
    return row


class TestItKeepsWhatMattersForManagement:
    def test_the_cadchf_sample_survives_the_round_trip(self) -> None:
        store, sink = brain()

        assert store.record_position_path([sample()]) == 1
        assert sink["rows"][0] == (
            7, NOW, 0.58595, 0.44, 0.44, 2.82, 0.58422, -1.0, False, "healthy"
        )

    def test_it_records_whether_the_money_was_safe(self) -> None:
        """`protected` and `stop_r` are what make this a management record
        rather than a price series. Without them the question the whole table
        exists for — was holding this right — has no data behind it."""
        store, sink = brain()
        safe = sample(stop_price=0.58600, stop_r=0.5, protected=True)
        store.record_position_path([sample(), safe])

        assert [row[-2] for row in sink["rows"]] == [False, True]
        assert [row[-3] for row in sink["rows"]] == [-1.0, 0.5]

    def test_a_whole_minute_goes_in_one_statement(self) -> None:
        """The guard runs about once a second on one vCPU. A round trip per
        pass would make the thing watching the money the slowest part of the
        system, so the batch is the point, not an optimisation."""
        store, sink = brain()
        batch = [sample(sampled_at=NOW) for _ in range(60)]

        assert store.record_position_path(batch) == 60
        assert len(sink["rows"]) == 60
        assert sink["sql"].count("INSERT") == 1


class TestItNeverDisturbsTheGuard:
    def test_a_row_without_a_trade_is_dropped_not_raised(self) -> None:
        store, _ = brain()

        assert store.record_position_path([sample(trade_id=None)]) == 0

    def test_an_empty_batch_costs_nothing(self) -> None:
        store, sink = brain()

        assert store.record_position_path([]) == 0
        assert "rows" not in sink

    def test_a_dropped_connection_is_retried_once(self) -> None:
        """A pooled Neon connection is closed from the far end after an idle
        period and the first statement after that always fails."""
        store, sink = brain(fail_times=1)
        store._connect = lambda: _Connection(sink)  # type: ignore[method-assign]

        assert store.record_position_path([sample()]) == 1

    def test_a_dead_brain_reports_zero_and_does_not_raise(self) -> None:
        """Memory is not allowed to take down the loop watching real money."""
        store, sink = brain(fail_times=9)
        store._connect = lambda: _Connection(sink, fail_times=9)  # type: ignore[method-assign]

        assert store.record_position_path([sample()]) == 0
        assert store.status.failures == 1

    def test_without_a_database_it_is_a_no_op(self) -> None:
        assert NullBrain().record_position_path([sample()]) == 0


class TestWhatSteppingInEarned:
    """The comparison existed, had its answer, and nobody who needed it could read it.

    Every closed trade is replayed against its own untouched stop and target,
    and the difference says whether the rule that closed it beat leaving the
    trade alone. That ran for weeks and wrote to local SQLite on a rented VPS —
    while the layer deciding hold-versus-close about once a second had no
    record of what its own interventions had earned.
    """

    def test_it_groups_by_the_rule_that_closed_the_trade(self) -> None:
        """"Our exits cost us" names nothing to stop doing. AI_CLOSE and
        PEAK_STALL are different decisions and on this account they have
        pointed in opposite directions."""
        from brain.store import ManagementRecord

        ai = ManagementRecord(action="AI_CLOSE", trades=3, total_lift_r=1.92, better=3)
        stall = ManagementRecord(action="PEAK_STALL", trades=10, total_lift_r=-14.9, better=1)

        assert ai.mean_lift_r == pytest.approx(0.64, abs=0.01)
        assert "beat holding" in ai.summary()
        assert stall.mean_lift_r == pytest.approx(-1.49, abs=0.01)
        assert "cost us against holding" in stall.summary()

    def test_the_briefing_says_which_way_to_read_it(self) -> None:
        """A number handed to an adviser is used however much hedging surrounds
        it, so the weight line has to state the direction explicitly."""
        store, _ = brain()
        store.management_records = lambda *_, **__: [  # type: ignore[method-assign]
            ManagementRecordStub("AI_CLOSE", 3, 1.92, 3)
        ]
        store.lessons = lambda *_, **__: []  # type: ignore[method-assign]
        store.scoreboard = lambda *_, **__: []  # type: ignore[method-assign]
        store.gate_scoreboard = lambda *_, **__: []  # type: ignore[method-assign]
        store.side_records = lambda *_, **__: []  # type: ignore[method-assign]
        store.module_records = lambda *_, **__: []  # type: ignore[method-assign]

        brief = store.briefing()

        assert "what_stepping_in_has_earned" in brief
        weight = brief["what_stepping_in_has_earned"]["weight"]
        assert "MANAGEMENT rather than" in weight
        assert "negative record is a reason to hold" in weight

    def test_a_thin_sample_is_withheld_rather_than_hedged(self) -> None:
        """One trade is not a weaker version of the finding, it is a false one."""
        store, sink = brain()
        sink["rows"] = []
        store._run = lambda *_, **__: []  # type: ignore[method-assign]

        assert store.management_records(minimum_trades=3) == []

    def test_without_a_database_there_is_no_record(self) -> None:
        assert NullBrain().management_records() == []
        assert NullBrain().record_management_outcome(local_trade_id=1) is None


class ManagementRecordStub:
    def __init__(self, action: str, trades: int, lift: float, better: int) -> None:
        self.action, self.trades, self.total_lift_r, self.better = action, trades, lift, better

    def summary(self) -> str:
        return f"{self.action}: {self.trades} trades, {self.total_lift_r:+.2f}R"


class TestTheArchiveLivesInTheBrainNotOnTheVPS:
    """The nearest-neighbour adviser was learning from a file on a rented box.

    It reads `runtime/ai_reviews.jsonl` and requires five comparable past
    states before it may act on a position. A fresh clone starts that file at
    zero, so the model holds every position indefinitely — which is exactly
    what the CADCHF long showed — while the account's whole history of
    supervisions sits in Postgres unread.
    """

    @staticmethod
    def _advisor(brain: object | None = None):  # type: ignore[no-untyped-def]
        from pathlib import Path

        from advisory.local_history import LocalHistoryAdvisor
        from config.loader import load_settings

        root = Path(__file__).resolve().parent.parent
        config = load_settings(overlay=root / "config" / "eightcap.yaml", env_overrides=False).ai
        advisor = LocalHistoryAdvisor(config, root / "runtime" / "no-such-ledger.jsonl")
        if brain is not None:
            advisor.attach_brain(brain)
        return advisor

    @staticmethod
    def _rows(count: int) -> list[dict]:
        """Distinct states. Identical feature vectors are one observation and
        are de-duplicated, which is correct and would hide a broken merge."""
        return [
            {
                "symbol": "CADCHF.i",
                "direction": "LONG",
                "action": "close",
                "confidence": 0.7,
                "r_at_the_time": 0.40 + index * 0.01,
                "features": {"unrealised_r": 0.14 + index * 0.001, "peak_r": 0.15},
            }
            for index in range(count)
        ]

    def test_without_a_brain_an_empty_file_is_an_empty_archive(self) -> None:
        assert self._advisor().supervision_examples == ()

    def test_the_brain_fills_it(self) -> None:
        class Stocked:
            def supervision_examples(self) -> list[dict]:
                return TestTheArchiveLivesInTheBrainNotOnTheVPS._rows(8)

        assert len(self._advisor(Stocked()).supervision_examples) == 8

    def test_identical_states_count_once(self) -> None:
        """Six samples of one unchanged position is one observation. Counting
        it six times would let a single state satisfy the five-neighbour floor
        on its own."""

        class Repeating:
            def supervision_examples(self) -> list[dict]:
                row = TestTheArchiveLivesInTheBrainNotOnTheVPS._rows(1)[0]
                return [dict(row) for _ in range(6)]

        assert len(self._advisor(Repeating()).supervision_examples) == 1

    def test_an_unreachable_brain_falls_back_rather_than_raising(self) -> None:
        """Memory is not a risk control. Neon being down must cost the model
        its archive, never the guard loop that is watching real money."""

        class Sick:
            def supervision_examples(self) -> list[dict]:
                raise RuntimeError("neon unreachable")

        assert self._advisor(Sick()).supervision_examples == ()

    def test_attaching_twice_adds_nothing(self) -> None:
        """The runner builds the adviser before the brain, so attachment
        happens after construction and must be idempotent."""

        class Stocked:
            def supervision_examples(self) -> list[dict]:
                return TestTheArchiveLivesInTheBrainNotOnTheVPS._rows(8)

        advisor = self._advisor(Stocked())
        advisor.attach_brain(Stocked())

        assert len(advisor.supervision_examples) == 8

    def test_the_stored_features_use_the_matcher_s_own_definition(self) -> None:
        """Two extractors that drift apart produce a matcher confidently
        comparing different things, which is worse than no matcher."""
        from advisory.local_history import _supervision_shape, supervision_features

        payload = {
            "unrealised_r": 0.44,
            "peak_unrealised_r": 0.44,
            "unrealised_pct_of_account": 2.17,
            "context": {"mechanical_health": {"verdict": "healthy", "severity": 0.0}},
        }

        assert supervision_features(payload) == _supervision_shape(payload).features
