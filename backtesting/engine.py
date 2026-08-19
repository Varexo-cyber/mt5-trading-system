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
    #: Measured, not assumed. `scripts/execution_noise.py` read every live
    #: entry fill on this account and found slippage of exactly 0.00 pips on
    #: all of them — market orders on FX majors at 0.01 lots simply do not slip
    #: at a retail broker outside news.
    #:
    #: It was 0.5 bps, which is 0.55 pips on EURUSD, and on an eight-pip scalp
    #: stop that is 7% of the risk charged for something that never happened.
    #: Small in isolation, and this is the term a backtest of tight-stop
    #: strategies is most sensitive to.
    entry_slippage_bps: float = 0.0
    #: Still charged, because it has never been measured. `record_order_attempt`
    #: only runs on the entry path, so a broker-initiated exit leaves no
    #: telemetry and the honest position is that we do not know. Assuming zero
    #: for something unmeasured would be the mistake this docstring is about,
    #: in the other direction.
    exit_slippage_bps: float = 0.5
    #: Verified against the account: EUR 5.50 per lot round trip is EUR 0.055
    #: on 0.01 lots, which is 0.55 pips on EURUSD at 1.10 — 0.5 bps exactly.
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
    modules: tuple[str, ...] = ()
    #: The broker's own recorded spread at the deciding bar, in price units.
    #:
    #: Both replays already computed this to build a tick and then threw it
    #: away, so every backtest this project has ever run was filled at the mid
    #: on both sides — a round trip that crosses no spread at all. On the stop
    #: widths this account trades that is not a rounding error: at 7 pips on
    #: EURUSD a 0.8-pip spread is 0.11R, against the 0.16R the backtester was
    #: charging in total.
    #:
    #: Zero for an order built by hand, which is the old behaviour and honest
    #: about being a lower bound.
    spread: float = 0.0
    #: What `market_regime` read at the deciding bar — trend_up, trend_down,
    #: range, transition or extreme.
    #:
    #: It was computed on every decision and thrown away here, so the module
    #: table could say a detector loses on average and never whether it loses
    #: EVERYWHERE. Those are different findings with different answers: the
    #: first says switch it off, the second says only let it fire where it
    #: works. Empty when the replay did not record one.
    regime: str = ""

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
    longest_losing_streak: int = 0
    max_drawdown_duration_trades: int = 0
    monte_carlo_20pct_drawdown_probability: float = 0.0

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

    def run_non_overlapping(
        self, frame: pd.DataFrame, orders: list[BacktestOrder]
    ) -> BacktestResult:
        """Replay one position at a time, matching the small-account risk limit."""
        self._validate(frame)
        trades: list[BacktestTrade] = []
        unavailable_until: datetime | None = None
        for order in sorted(orders, key=lambda item: item.decided_at):
            if unavailable_until is not None and order.decided_at <= unavailable_until:
                continue
            if order.risk <= 0:
                continue
            future = frame[frame.index > order.decided_at]
            if future.empty:
                continue
            trade = self._replay(order, future.iloc[: self.assumptions.max_holding_bars])
            trades.append(trade)
            unavailable_until = trade.exited_at
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
        # Commission, and the spread the round trip actually crosses. The
        # spread is counted ONCE: a long is filled at the ask and closed at the
        # bid, which is one crossing, and `risk.PositionSizer._cost_share`
        # measures it the same way.
        notional_cost = order.entry * (self.assumptions.round_trip_commission_bps / 10_000.0)
        costs_r = (notional_cost + max(order.spread, 0.0)) / order.risk
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
        losing_streak = longest_losing_streak(returns.tolist())
        duration = max_drawdown_duration(returns.tolist())
        return BacktestResult(
            tuple(trades),
            float(returns.sum()),
            float(returns.mean()),
            float((returns > 0).mean()),
            profit_factor,
            float(drawdown.max(initial=0.0)),
            sharpe,
            losing_streak,
            duration,
            monte_carlo_drawdown_probability(returns.tolist()),
        )


def longest_losing_streak(returns: list[float]) -> int:
    longest = current = 0
    for value in returns:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def max_drawdown_duration(returns: list[float]) -> int:
    equity = peak = 0.0
    current = longest = 0
    for value in returns:
        equity += value
        if equity >= peak:
            peak = equity
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def monte_carlo_drawdown_probability(
    returns: list[float],
    *,
    simulations: int = 1_000,
    risk_pct_per_r: float = 1.0,
    drawdown_threshold_pct: float = 20.0,
    seed: int = 770101,
) -> float:
    """Probability a shuffled trade path breaches the configured drawdown."""
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if not returns:
        return 0.0
    rng = np.random.default_rng(seed)
    values = np.asarray(returns, dtype=float)
    breaches = 0
    for _ in range(simulations):
        equity = peak = 1.0
        for value in rng.permutation(values):
            equity *= max(0.0, 1.0 + value * risk_pct_per_r / 100.0)
            peak = max(peak, equity)
            if peak > 0 and (peak - equity) / peak * 100.0 >= drawdown_threshold_pct:
                breaches += 1
                break
    return breaches / simulations


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
