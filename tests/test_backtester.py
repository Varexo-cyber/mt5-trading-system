from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from backtesting.engine import (
    BacktestAssumptions,
    BacktestOrder,
    PessimisticBacktester,
    deflated_sharpe_probability,
    longest_losing_streak,
    max_drawdown_duration,
    monte_carlo_drawdown_probability,
    walk_forward_split,
)
from backtesting.replay import (
    REPLAY_TIMEFRAMES,
    HistoricalContextReplay,
    evidence_digest,
    frame_digest,
    implementation_digest,
    write_evidence_report,
)
from core.types import Direction, Signal, Timeframe


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz=UTC),
    )


def order(direction: Direction = Direction.LONG) -> BacktestOrder:
    return BacktestOrder(
        "TEST",
        datetime(2026, 1, 1, tzinfo=UTC),
        direction,
        100.0,
        99.0 if direction is Direction.LONG else 101.0,
        102.0 if direction is Direction.LONG else 98.0,
    )


def test_same_bar_stop_and_target_uses_stop() -> None:
    bars = frame([(100, 100, 100, 100), (100, 103, 98, 101)])
    result = PessimisticBacktester(BacktestAssumptions(0, 0, 0)).run(bars, [order()])

    assert result.trades[0].outcome == "SL_FIRST_AMBIGUOUS"
    assert result.trades[0].net_r == -1.0


def test_gap_through_stop_gets_worse_fill() -> None:
    bars = frame([(100, 100, 100, 100), (98, 99, 97, 98.5)])
    result = PessimisticBacktester(BacktestAssumptions(0, 0, 0)).run(bars, [order()])

    assert result.trades[0].exit_price == 98
    assert result.trades[0].net_r == -2


def test_costs_reduce_result() -> None:
    bars = frame([(100, 100, 100, 100), (100, 102.1, 99.5, 102)])
    result = PessimisticBacktester(BacktestAssumptions(0, 0, 10)).run(bars, [order()])

    assert result.trades[0].gross_r == 2
    assert result.trades[0].costs_r == 0.1
    assert result.trades[0].net_r == pytest.approx(1.9)


def test_walk_forward_is_chronological_60_20_20() -> None:
    bars = frame([(1, 1, 1, 1)] * 10)
    split = walk_forward_split(bars)

    assert [len(split.train), len(split.validation), len(split.holdout)] == [6, 2, 2]
    assert split.train.index[-1] < split.validation.index[0] < split.holdout.index[0]


def test_deflated_sharpe_penalises_more_trials() -> None:
    returns = [1, 1, 0.5, -0.2, 1.2, 0.4, 0.8, -0.1] * 5

    assert deflated_sharpe_probability(returns, 2) > deflated_sharpe_probability(returns, 100)


def test_historical_context_never_contains_an_unclosed_bar() -> None:
    end = datetime(2026, 8, 1, tzinfo=UTC)
    frames = {}
    for timeframe in REPLAY_TIMEFRAMES:
        index = pd.date_range(end=end, periods=150, freq=timeframe.duration, tz=UTC)
        frames[timeframe] = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "spread": 2,
            },
            index=index,
        )

    class SpyEngine:
        calls = 0

        def evaluate(self, context, _mode):  # type: ignore[no-untyped-def]
            self.calls += 1
            for timeframe, series in context.series.items():
                assert series.df.index[-1] + timeframe.duration <= context.now
            return SimpleNamespace(approved=False, direction=None)

    spy = SpyEngine()
    replay = HistoricalContextReplay(spy, history_bars=120)  # type: ignore[arg-type]

    orders = replay.orders(
        "TEST",
        frames,
        point=0.01,
        start=end - Timeframe.H1.duration * 10,
        end=end + Timeframe.H1.duration,
    )

    assert orders == []
    assert spy.calls > 0


def test_replay_attributes_orders_only_to_modules_that_carried_weight() -> None:
    end = datetime(2026, 8, 1, tzinfo=UTC)
    frames = {}
    for timeframe in REPLAY_TIMEFRAMES:
        index = pd.date_range(end=end, periods=150, freq=timeframe.duration, tz=UTC)
        frames[timeframe] = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "spread": 2},
            index=index,
        )

    class AttributionEngine:
        config = SimpleNamespace(weights={"causal": 1.0, "passive": 0.0}, minimum_confidence=0.5)

        def evaluate(self, context, _mode):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                approved=True,
                direction=Direction.LONG,
                entry=context.tick.ask,
                stop_loss=99.0,
                take_profit=102.0,
                score=70.0,
                confidence=0.8,
                # A real `TradeIdea` always carries this. The double is given
                # it rather than the replay being made tolerant of its absence:
                # a `getattr(..., None)` there would turn a production
                # regression — the grader silently falling back to a
                # forty-one-hour hold — into a passing test.
                expected_horizon_minutes=180,
                signals=(
                    Signal("causal", 70.0, 0.8),
                    Signal("passive", 90.0, 0.9),
                ),
            )

    replay = HistoricalContextReplay(AttributionEngine(), history_bars=120)  # type: ignore[arg-type]
    orders = replay.orders(
        "TEST",
        frames,
        point=0.01,
        start=end - Timeframe.H1.duration * 2,
        end=end + Timeframe.H1.duration,
    )

    assert orders
    assert all(item.modules == ("causal",) for item in orders)


def test_risk_diagnostics_include_streak_duration_and_monte_carlo() -> None:
    returns = [1.0, -1.0, -1.0, -1.0, 2.0]

    assert longest_losing_streak(returns) == 3
    assert max_drawdown_duration(returns) == 4
    probability = monte_carlo_drawdown_probability(
        returns,
        simulations=100,
        risk_pct_per_r=10.0,
        drawdown_threshold_pct=15.0,
    )
    assert 0.0 <= probability <= 1.0


def test_historical_data_and_report_contents_are_content_addressed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bars = frame([(100, 101, 99, 100), (100, 102, 99, 101)])
    changed = bars.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] = 102
    assert frame_digest(bars) != frame_digest(changed)

    path = tmp_path / "evidence.json"
    write_evidence_report(path, metadata={"source": "test"}, segments=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evidence_digest"] == evidence_digest(payload)


def test_implementation_digest_changes_with_production_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "analysis"
    source.mkdir()
    module = source / "strategy.py"
    module.write_text("EDGE = 1\n", encoding="utf-8")
    before = implementation_digest(tmp_path)

    module.write_text("EDGE = 2\n", encoding="utf-8")

    assert implementation_digest(tmp_path) != before
