"""Pessimistic, event-driven trade replay without look-ahead shortcuts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import e, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

from core.types import Direction


@dataclass(frozen=True, slots=True)
class BacktestAssumptions:
    entry_slippage_bps: float = 0.5
    exit_slippage_bps: float = 0.5
    round_trip_commission_bps: float = 0.5
    max_holding_bars: int = 500


@dataclass(frozen=True, slots=True)
class BacktestOrder:
    symbol: str
    decided_at: datetime
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    score: float = 0.0
    confidence: float = 0.0

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    order: BacktestOrder
    entered_at: datetime
    exited_at: datetime
    exit_price: float
    outcome: str
    gross_r: float
    costs_r: float
    net_r: float
    holding_bars: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: tuple[BacktestTrade, ...]
    total_r: float
    expectancy_r: float
    win_rate: float
    profit_factor: float
    max_drawdown_r: float
    sharpe: float

    @property
    def sample_size(self) -> int:
        return len(self.trades)


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame


def walk_forward_split(frame: pd.DataFrame) -> WalkForwardSplit:
    """Chronological 60/20/20 split; never shuffle time-series data."""
    if len(frame) < 5:
        raise ValueError("walk-forward split needs at least five rows")
    train_end = max(1, int(len(frame) * 0.60))
    validation_end = max(train_end + 1, int(len(frame) * 0.80))
    return WalkForwardSplit(
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


class PessimisticBacktester:
    """Replay pre-registered orders against later bars.

    Orders decided on a bar may enter no earlier than the next bar. If one bar
    touches both stop and target, the stop wins. Positive gaps through a target
    do not improve the fill; negative gaps through a stop do worsen it.
    """

    def __init__(self, assumptions: BacktestAssumptions | None = None) -> None:
        self.assumptions = assumptions or BacktestAssumptions()

    def run(self, frame: pd.DataFrame, orders: list[BacktestOrder]) -> BacktestResult:
        self._validate(frame)
        trades: list[BacktestTrade] = []
        for order in sorted(orders, key=lambda item: item.decided_at):
            if order.risk <= 0:
                continue
            future = frame[frame.index > order.decided_at]
            if future.empty:
                continue
            future = future.iloc[: self.assumptions.max_holding_bars]
            trades.append(self._replay(order, future))
        return self._summarise(trades)

    def _replay(self, order: BacktestOrder, future: pd.DataFrame) -> BacktestTrade:
        entered_at = future.index[0].to_pydatetime()
        entry = self._worse_entry(order.direction, order.entry)
        exit_price = float(future.iloc[-1]["close"])
        exited_at = future.index[-1].to_pydatetime()
        outcome = "TIME"
        holding = len(future)

        for index, (_, bar) in enumerate(future.iterrows(), start=1):
            low, high, opening = float(bar["low"]), float(bar["high"]), float(bar["open"])
            if order.direction is Direction.LONG:
                stop_hit = low <= order.stop_loss
                target_hit = high >= order.take_profit
                if stop_hit:
                    exit_price = min(opening, order.stop_loss)
                    outcome = "SL" if not target_hit else "SL_FIRST_AMBIGUOUS"
                elif target_hit:
                    exit_price = order.take_profit
                    outcome = "TP"
                else:
                    continue
            else:
                stop_hit = high >= order.stop_loss
                target_hit = low <= order.take_profit
                if stop_hit:
                    exit_price = max(opening, order.stop_loss)
                    outcome = "SL" if not target_hit else "SL_FIRST_AMBIGUOUS"
                elif target_hit:
                    exit_price = order.take_profit
                    outcome = "TP"
                else:
                    continue
            exited_at = future.index[index - 1].to_pydatetime()
            holding = index
            break

        exit_price = self._worse_exit(order.direction, exit_price)
        gross_r = (exit_price - entry) * int(order.direction) / order.risk
        notional_cost = order.entry * (self.assumptions.round_trip_commission_bps / 10_000.0)
        costs_r = notional_cost / order.risk
        return BacktestTrade(
            order,
            entered_at,
            exited_at,
            exit_price,
            outcome,
            gross_r,
            costs_r,
            gross_r - costs_r,
            holding,
        )

    def _worse_entry(self, direction: Direction, price: float) -> float:
        fraction = self.assumptions.entry_slippage_bps / 10_000.0
        return price * (1 + fraction * int(direction))

    def _worse_exit(self, direction: Direction, price: float) -> float:
        fraction = self.assumptions.exit_slippage_bps / 10_000.0
        return price * (1 - fraction * int(direction))

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"backtest frame missing columns: {sorted(required - set(frame.columns))}"
            )
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("backtest frame needs a timezone-aware DatetimeIndex")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("backtest timestamps must be unique and increasing")

    @staticmethod
    def _summarise(trades: list[BacktestTrade]) -> BacktestResult:
        if not trades:
            return BacktestResult((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        returns = np.asarray([trade.net_r for trade in trades], dtype=float)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        curve = np.cumsum(returns)
        drawdown = np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:] - curve
        std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        sharpe = float(returns.mean() / std * sqrt(len(returns))) if std > 0 else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size else float("inf")
        return BacktestResult(
            tuple(trades),
            float(returns.sum()),
            float(returns.mean()),
            float((returns > 0).mean()),
            profit_factor,
            float(drawdown.max(initial=0.0)),
            sharpe,
        )


def deflated_sharpe_probability(returns: list[float], configurations_tested: int) -> float:
    """Probability Sharpe exceeds the expected best of many null trials.

    Implements the Bailey/Lopez de Prado deflated-Sharpe construction using
    trade returns. It is evidence, not a substitute for untouched holdout data.
    """
    if configurations_tested < 1:
        raise ValueError("configurations_tested must be positive")
    values = np.asarray(returns, dtype=float)
    if len(values) < 3 or float(values.std(ddof=1)) == 0.0:
        return 0.0
    sharpe = float(values.mean() / values.std(ddof=1))
    trials = max(configurations_tested, 2)
    normal = NormalDist()
    gamma = 0.5772156649
    expected_max = (1 - gamma) * normal.inv_cdf(1 - 1 / trials) + gamma * normal.inv_cdf(
        1 - 1 / (trials * e)
    )
    centered = values - values.mean()
    std = values.std(ddof=0)
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4))
    denominator = sqrt(max(1e-12, 1 - skew * sharpe + ((kurtosis - 1) / 4) * sharpe**2))
    statistic = (sharpe - expected_max) * sqrt(len(values) - 1) / denominator
    return float(normal.cdf(statistic))
