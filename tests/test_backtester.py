from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from backtesting.engine import (
    BacktestAssumptions,
    BacktestOrder,
    PessimisticBacktester,
    deflated_sharpe_probability,
    walk_forward_split,
)
from core.types import Direction


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
