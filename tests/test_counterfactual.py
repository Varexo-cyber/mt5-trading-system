from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from core.types import Direction
from learning.counterfactual import (
    classify_path,
    future_bars,
    resolve_counterfactuals,
    resolve_management_baselines,
)


def _bars(*rows: tuple[float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["high", "low", "close"])


def test_long_target_first_is_measured_in_original_r() -> None:
    outcome, pnl_r = classify_path(
        _bars((101.0, 99.5, 100.5), (104.5, 100.0, 104.0)),
        Direction.LONG,
        100.0,
        98.0,
        104.0,
        timed_out=False,
    )
    assert outcome == "TP"
    assert pnl_r == 2.0


def test_same_bar_stop_and_target_is_scored_as_a_pessimistic_loss() -> None:
    outcome, pnl_r = classify_path(
        _bars((105.0, 97.0, 101.0)),
        Direction.LONG,
        100.0,
        98.0,
        104.0,
        timed_out=False,
    )
    assert outcome == "SL_FIRST_AMBIGUOUS"
    assert pnl_r == -1.0


def test_unfinished_path_remains_unresolved_until_timeout() -> None:
    bars = _bars((101.0, 99.0, 100.5))
    assert classify_path(bars, Direction.SHORT, 100.0, 102.0, 96.0, timed_out=False) == (None, 0.0)
    outcome, pnl_r = classify_path(bars, Direction.SHORT, 100.0, 102.0, 96.0, timed_out=True)
    assert outcome == "TIMEOUT"
    assert pnl_r == -0.25


def test_the_decision_bar_is_excluded_from_counterfactual_evidence() -> None:
    opened = datetime(2026, 8, 1, 10, 7, tzinfo=UTC)
    raw = [
        {"time": int(datetime(2026, 8, 1, 10, 0, tzinfo=UTC).timestamp()), "close": 99.0},
        {"time": int(datetime(2026, 8, 1, 10, 15, tzinfo=UTC).timestamp()), "close": 101.0},
    ]

    result = future_bars(raw, opened)

    assert result["close"].tolist() == [101.0]


def test_counterfactual_evidence_without_timestamps_fails_closed() -> None:
    opened = datetime(2026, 8, 1, 10, 7, tzinfo=UTC)

    assert future_bars([{"high": 2.0, "low": 1.0, "close": 1.5}], opened).empty


def test_resolved_refusal_is_forwarded_to_durable_memory() -> None:
    opened = datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
    now = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    class Recorder:
        def __init__(self) -> None:
            self.forwarded: list[dict[str, object]] = []

        @staticmethod
        def unresolved_shadow_trades(_limit):  # type: ignore[no-untyped-def]
            return [
                {
                    "id": 9,
                    "symbol": "EURUSD.i",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl": 98.0,
                    "tp": 104.0,
                    "opened_at": opened.isoformat(),
                    "blocked_by": "AI_VETO",
                }
            ]

        def resolve_shadow_trade(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    class Broker:
        @staticmethod
        def copy_rates_range(*_args):  # type: ignore[no-untyped-def]
            return [
                {
                    "time": int((opened + pd.Timedelta(minutes=15)).timestamp()),
                    "high": 104.5,
                    "low": 99.5,
                    "close": 104.0,
                }
            ]

    recorder = Recorder()
    assert (
        resolve_counterfactuals(
            recorder,  # type: ignore[arg-type]
            Broker(),  # type: ignore[arg-type]
            now,
            on_resolved=recorder.forwarded.append,
        )
        == 1
    )
    assert recorder.forwarded[0]["outcome"] == "TP"
    assert recorder.forwarded[0]["pnl_r"] == 2.0


def test_closed_trade_is_compared_with_untouched_original_plan() -> None:
    opened = datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
    now = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    class Journal:
        @staticmethod
        def unresolved_management_baselines(_limit):  # type: ignore[no-untyped-def]
            return [
                {
                    "id": 7,
                    "symbol": "EURUSD",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl": 98.0,
                    "tp": 104.0,
                    "opened_at": opened.isoformat(),
                    "pnl_r": 0.5,
                    "pnl_money": 1.0,
                    "risk_money": 2.0,
                }
            ]

    class Recorder:
        def __init__(self) -> None:
            self.journal = Journal()
            self.recorded: dict[str, object] = {}

        def record_management_baseline(self, **kwargs):  # type: ignore[no-untyped-def]
            self.recorded = kwargs

    class Broker:
        @staticmethod
        def copy_rates_range(*_args):  # type: ignore[no-untyped-def]
            return [
                {
                    "time": int((opened + pd.Timedelta(minutes=15)).timestamp()),
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                },
                {
                    "time": int((opened + pd.Timedelta(minutes=30)).timestamp()),
                    "high": 104.5,
                    "low": 100.0,
                    "close": 104.0,
                },
            ]

    recorder = Recorder()
    assert resolve_management_baselines(recorder, Broker(), now) == 1  # type: ignore[arg-type]
    assert recorder.recorded["outcome"] == "TP"
    assert recorder.recorded["baseline_pnl_r"] == 2.0
    assert recorder.recorded["actual_pnl_r"] == 0.5
