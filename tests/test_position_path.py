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
