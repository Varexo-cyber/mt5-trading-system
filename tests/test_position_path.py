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


class TestTheAdviserMayActuallyMoveTheStop:
    """Three independent reasons the CADCHF stop never went to break-even.

    The mechanical rule read only R, and 0.44R is under its 0.6 floor even
    though the trade held two percent of the account. That was one.

    The other two are here, and both made the judgement layer unable to protect
    money no matter what it concluded:

      - `action not in {"hold", "close"}` excluded `tighten_stop` outright, so
        the model could never return a stop move however strong the evidence;
      - and had it returned one, `Supervision.is_risk_reducing` refuses a stop
        move carrying no level, so the verdict would have been built, logged
        and thrown away.
    """

    @staticmethod
    def _state() -> dict:
        from advisory.local_history import supervision_features

        state = {
            "direction": "LONG",
            "symbol": "CADCHF.i",
            "entry_price": 0.58542,
            "price_now": 0.58595,
            "unrealised_r": 0.44,
            "peak_unrealised_r": 0.44,
            "profit_given_back_fraction": 0.0,
            "age_hours": 0.1,
            "unrealised_pct_of_account": 2.17,
            "context": {"mechanical_health": {"verdict": "healthy", "severity": 0.0}},
        }
        state["_features"] = supervision_features(state)
        return state

    @classmethod
    def _advisor(cls, *, with_level: bool):  # type: ignore[no-untyped-def]
        from pathlib import Path

        from advisory.local_history import LocalHistoryAdvisor
        from config.loader import load_settings

        root = Path(__file__).resolve().parent.parent
        config = load_settings(overlay=root / "config" / "eightcap.yaml", env_overrides=False).ai
        base = cls._state()["_features"]

        class Stocked:
            def supervision_examples(self) -> list[dict]:
                out = []
                for index in range(8):
                    features = dict(base)
                    features["unrealised_r"] = base["unrealised_r"] + index * 0.0005
                    out.append(
                        {
                            "symbol": "CADCHF.i",
                            "direction": "LONG",
                            "action": "tighten_stop",
                            "confidence": 0.8,
                            "r_at_the_time": 0.44,
                            "features": features,
                            "stop_fraction": 0.35 if with_level else None,
                        }
                    )
                return out

        advisor = LocalHistoryAdvisor(config, root / "runtime" / "no-such-ledger.jsonl")
        advisor.attach_brain(Stocked())
        return advisor

    def test_it_now_returns_a_stop_move_with_a_level(self) -> None:
        verdict = self._advisor(with_level=True).supervise(self._state())

        assert verdict.action == "tighten_stop"
        assert verdict.stop_loss is not None

    def test_the_level_lands_above_the_entry(self) -> None:
        """Which is the entire point: below entry the stop caps a loss, it does
        not protect a gain."""
        verdict = self._advisor(with_level=True).supervise(self._state())

        assert verdict.stop_loss > 0.58542

    def test_the_risk_layer_now_accepts_it(self) -> None:
        """It used to refuse every one of these for carrying no level."""
        verdict = self._advisor(with_level=True).supervise(self._state())

        assert verdict.is_risk_reducing(
            direction_sign=1, current_sl=0.58422, current_tp=0.58691, price_now=0.58595
        )

    def test_without_a_learned_placement_it_holds_instead(self) -> None:
        """Honest rather than unexpressable. A verdict that will certainly be
        refused looks like a decision and is worse than a hold."""
        verdict = self._advisor(with_level=False).supervise(self._state())

        assert verdict.action == "hold"
        assert verdict.stop_loss is None


class TestTheColdStartDeadlock:
    """The model needed an example of the thing it could not yet do.

    `_learned_stop` returns a level only when a comparable past state recorded
    one. `stop_fraction` is written only when a verdict IS a stop move. And the
    model emits a stop move only when `_learned_stop` returns a level. Three
    conditions in a circle, so it could never produce its first one.

    Both keys were shut at once: the brain column is new and therefore empty,
    and the JSONL loader dropped every `tighten_stop` row before it could be
    read. Claude's historical stop moves are what breaks the circle.
    """

    @staticmethod
    def _fraction(action: str, stop: float | None):  # type: ignore[no-untyped-def]
        from advisory.local_history import _recorded_stop_fraction

        return _recorded_stop_fraction(
            {"entry_price": 0.58542, "price_now": 0.58595},
            {"action": action, "stop_loss": stop},
        )

    def test_a_past_stop_move_yields_a_reusable_placement(self) -> None:
        """Halfway between entry and price reads as 0.5, whatever the pair."""
        assert self._fraction("tighten_stop", 0.585685) == pytest.approx(0.5, abs=0.01)

    def test_a_hold_carries_no_placement(self) -> None:
        assert self._fraction("hold", None) is None

    def test_a_stop_beyond_the_price_is_not_copied(self) -> None:
        """That is an instant market exit dressed as a stop, not a placement
        anything should learn from."""
        assert self._fraction("tighten_stop", 0.58700) is None

    def test_a_stop_behind_the_entry_is_not_copied(self) -> None:
        """It protects nothing, which is the state this whole change exists to
        get out of."""
        assert self._fraction("tighten_stop", 0.58400) is None

    def test_the_loader_no_longer_discards_stop_moves(self) -> None:
        """The line that kept the circle closed."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "advisory" / "local_history.py"
        ).read_text(encoding="utf-8")

        assert 'if action not in {"hold", "close", "tighten_stop"}:' in source


class TestItLearnsFromWhatHappenedAfterItLetGo:
    """"Je safede 90 cent maar het ging verder omhoog" — that lesson.

    Every other column about an exit reduces to one binary answer: would the
    untouched plan have reached its stop or its target. Which means "closed at
    +0.1R and it ran to +2R" and "closed at +0.1R and it collapsed" are the
    same row, and the system could never learn the difference between banking
    too early and banking exactly right.
    """

    @staticmethod
    def _frame(after_close: list[tuple[float, float]]):  # type: ignore[no-untyped-def]
        import pandas as pd

        index = pd.date_range("2026-08-14T19:00", periods=len(after_close) + 2, freq="15min",
                              tz="UTC")
        highs = [1.1000, 1.1000, *[high for high, _ in after_close]]
        lows = [1.1000, 1.1000, *[low for _, low in after_close]]
        return pd.DataFrame({"high": highs, "low": lows}, index=index)

    @staticmethod
    def _row(direction: str = "LONG"):  # type: ignore[no-untyped-def]
        return {
            "direction": direction,
            "closed_at": "2026-08-14T19:15:00+00:00",
        }

    def _measure(self, after, direction="LONG"):  # type: ignore[no-untyped-def]
        from learning.counterfactual import _post_exit_excursion

        return _post_exit_excursion(
            self._frame(after), self._row(direction), entry=1.1000, risk=0.0010
        )

    def test_it_reports_what_was_left_on_the_table(self) -> None:
        """Banked early and the market ran two more R. That is the finding."""
        best, _ = self._measure([(1.1020, 1.1000), (1.1010, 1.1005)])

        assert best == pytest.approx(2.0, abs=0.01)

    def test_it_reports_what_getting_out_avoided(self) -> None:
        """The same measurement, the other way. An exit before a collapse is
        the rule earning its keep, and it deserves to be visible too."""
        _, worst = self._measure([(1.1000, 1.0985), (1.0990, 1.0980)])

        assert worst == pytest.approx(-2.0, abs=0.01)

    def test_a_short_is_measured_in_its_own_direction(self) -> None:
        """Price falling after a short exit is profit forgone, not a loss."""
        best, _ = self._measure([(1.1000, 1.0980)], direction="SHORT")

        assert best == pytest.approx(2.0, abs=0.01)

    def test_unmeasured_is_not_zero(self) -> None:
        """No bar after the exit means no answer. Recording it as zero would
        report every unresolvable trade as a perfect exit."""
        assert self._measure([]) == (None, None)

    def test_a_missing_exit_time_says_nothing(self) -> None:
        from learning.counterfactual import _post_exit_excursion

        assert _post_exit_excursion(
            self._frame([(1.1020, 1.1000)]),
            {"direction": "LONG", "closed_at": None},
            entry=1.1000,
            risk=0.0010,
        ) == (None, None)

    def test_the_briefing_line_states_it_plainly(self) -> None:
        from brain.store import ManagementRecord

        record = ManagementRecord(
            action="PROFIT_BANKED",
            trades=6,
            total_lift_r=-3.0,
            better=1,
            left_on_the_table_r=1.8,
        )

        assert "went on to reach +1.80R on average after it acted" in record.summary()
        assert "cost us against holding" in record.summary()

    def test_without_the_measurement_the_line_stays_quiet(self) -> None:
        from brain.store import ManagementRecord

        record = ManagementRecord(action="AI_CLOSE", trades=3, total_lift_r=1.9, better=3)

        assert "went on to reach" not in record.summary()


class TestItLearnsWhenItShouldHaveGotOutEarlier:
    """The losing side of the exit question, and the side nothing measured.

    HK50 short, 14 August: peak +0.00R, took -0.89R, ran to the broker stop
    with no exit chosen by anything. UKOUSD the same shape. Every column about
    those trades says "-0.89R, hit the stop" — the moment it was only 0.2R down
    and drifting leaves no trace, so the decision to keep holding is invisible
    and unlearnable.

    `position_path` is the only record of what was on offer between the open
    and the close. This is the first thing that reads it.
    """

    def test_the_missed_exit_is_reported_when_there_was_one(self) -> None:
        from brain.store import ManagementRecord

        record = ManagementRecord(
            action="BROKER_SL",
            trades=4,
            total_lift_r=0.0,
            better=0,
            missed_r=0.71,
        )

        assert "a better exit was available and missed by 0.71R on average" in record.summary()

    def test_a_perfect_exit_says_nothing(self) -> None:
        """Nothing was left behind, so there is nothing to report. A zero line
        in a briefing is a sentence the reader has to discard."""
        from brain.store import ManagementRecord

        record = ManagementRecord(
            action="BROKER_TP", trades=3, total_lift_r=0.2, better=2, missed_r=0.0
        )

        assert "better exit was available" not in record.summary()

    def test_both_sides_can_appear_together(self) -> None:
        """One rule can be late out of a loser and early out of a winner, and
        those are different corrections."""
        from brain.store import ManagementRecord

        record = ManagementRecord(
            action="AI_CLOSE",
            trades=6,
            total_lift_r=-1.2,
            better=2,
            left_on_the_table_r=1.4,
            missed_r=0.3,
        )
        summary = record.summary()

        assert "went on to reach +1.40R" in summary
        assert "missed by 0.30R" in summary

    def test_an_unrecorded_path_yields_nothing_rather_than_zero(self) -> None:
        """A position opened before the table existed has an UNKNOWN best
        moment. Zero would claim it was held to the perfect second."""
        from brain.store import NullBrain

        assert NullBrain().best_exit_available(1) is None
