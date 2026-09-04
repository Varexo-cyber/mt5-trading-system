"""Bounded chronological search for a new section eleven on synthetic gold crosses."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

COMPONENTS = {
    "XAUEUR": ("EURUSD.i", "divide"),
    "XAUGBP": ("GBPUSD.i", "divide"),
    "XAUAUD": ("AUDUSD.i", "divide"),
    "XAUJPY": ("USDJPY.i", "multiply"),
}
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1")
HORIZONS = {"M1": 120, "M5": 48, "M15": 24, "M30": 12, "H1": 8}
MECHANISMS = (
    "trend_pullback",
    "channel_breakout",
    "aligned_momentum",
    "trend_rejection",
    "contraction_breakout",
    "two_leg_trend",
    "two_leg_impulse",
)
STOP_ATR = 1.0
TARGET_R = 1.5
COST_R = 0.02
WARMUP = 80


@dataclass(frozen=True)
class Result:
    cell: str
    split: str
    trades: int
    total_r: float
    per_trade: float
    sigma: float
    wins: float
    by_market: dict[str, float]
    first_half_r: float
    second_half_r: float


def load_bars(connection: sqlite3.Connection, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = connection.execute(
        "SELECT time_utc, open, high, low, close, tick_volume FROM bars "
        "WHERE broker_symbol=? AND timeframe=? ORDER BY time_utc",
        (symbol, timeframe),
    ).fetchall()
    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    frame.index = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    return frame.astype(float)


def has_bars(connection: sqlite3.Connection, symbol: str, timeframe: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM bars WHERE broker_symbol=? AND timeframe=? LIMIT 1",
            (symbol, timeframe),
        ).fetchone()
        is not None
    )


def synthetic_cross(gold: pd.DataFrame, fx: pd.DataFrame, operation: str) -> pd.DataFrame:
    joined = gold.add_prefix("g_").join(fx.add_prefix("f_"), how="inner")
    if operation == "divide":
        data = {
            "open": joined.g_open / joined.f_open,
            "high": joined.g_high / joined.f_low,
            "low": joined.g_low / joined.f_high,
            "close": joined.g_close / joined.f_close,
        }
    else:
        data = {
            "open": joined.g_open * joined.f_open,
            "high": joined.g_high * joined.f_high,
            "low": joined.g_low * joined.f_low,
            "close": joined.g_close * joined.f_close,
        }
    frame = pd.DataFrame(data, index=joined.index)
    frame["volume"] = np.minimum(joined.g_volume, joined.f_volume)
    frame["gold_close"] = joined.g_close
    frame["fx_close"] = joined.f_close
    frame["fx_sign"] = -1.0 if operation == "divide" else 1.0
    return frame


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame.close.shift(1)
    ranges = pd.concat(
        [frame.high - frame.low, (frame.high - previous).abs(), (frame.low - previous).abs()],
        axis=1,
    ).max(axis=1)
    return ranges.rolling(period).mean()


def signals(frame: pd.DataFrame, mechanism: str) -> np.ndarray:
    close, open_, high, low = frame.close, frame.open, frame.high, frame.low
    unit = atr(frame)
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    trend = np.sign(fast - slow)
    out = pd.Series(0, index=frame.index, dtype=int)
    if mechanism == "trend_pullback":
        long = (trend > 0) & (low <= fast) & (close > fast) & (close > open_)
        short = (trend < 0) & (high >= fast) & (close < fast) & (close < open_)
    elif mechanism == "channel_breakout":
        ceiling = high.shift(1).rolling(20).max()
        floor = low.shift(1).rolling(20).min()
        long, short = (trend > 0) & (close > ceiling), (trend < 0) & (close < floor)
    elif mechanism == "aligned_momentum":
        impulse = close.diff(6) / unit
        long = (trend > 0) & (impulse > 0.75) & (close > open_)
        short = (trend < 0) & (impulse < -0.75) & (close < open_)
    elif mechanism == "trend_rejection":
        close_position = (close - low) / (high - low).replace(0.0, np.nan)
        long = (trend > 0) & (low < fast) & (close_position > 0.8)
        short = (trend < 0) & (high > fast) & (close_position < 0.2)
    elif mechanism == "contraction_breakout":
        quiet = unit < 0.75 * unit.rolling(48).mean()
        ceiling = high.shift(1).rolling(12).max()
        floor = low.shift(1).rolling(12).min()
        long = quiet & (trend > 0) & (close > ceiling)
        short = quiet & (trend < 0) & (close < floor)
    elif mechanism == "two_leg_trend":
        gold_trend = np.sign(
            frame.gold_close.ewm(span=20, adjust=False).mean()
            - frame.gold_close.ewm(span=50, adjust=False).mean()
        )
        fx_trend = np.sign(
            frame.fx_close.ewm(span=20, adjust=False).mean()
            - frame.fx_close.ewm(span=50, adjust=False).mean()
        ) * frame.fx_sign
        long = (gold_trend > 0) & (fx_trend > 0) & (close > open_)
        short = (gold_trend < 0) & (fx_trend < 0) & (close < open_)
    elif mechanism == "two_leg_impulse":
        gold_move = frame.gold_close.pct_change(6)
        fx_move = frame.fx_close.pct_change(6) * frame.fx_sign
        gold_large = gold_move.abs() > gold_move.abs().rolling(48).median()
        fx_large = fx_move.abs() > fx_move.abs().rolling(48).median()
        long = gold_large & fx_large & (gold_move > 0) & (fx_move > 0)
        short = gold_large & fx_large & (gold_move < 0) & (fx_move < 0)
    else:
        raise ValueError(mechanism)
    out[long.fillna(False)] = 1
    out[short.fillna(False)] = -1
    out.iloc[:WARMUP] = 0
    return out.to_numpy()


def replay(frame: pd.DataFrame, directions: np.ndarray, horizon: int) -> pd.DataFrame:
    unit = atr(frame).to_numpy()
    open_, high, low, close = (frame[name].to_numpy() for name in ("open", "high", "low", "close"))
    records: list[tuple[pd.Timestamp, float]] = []
    available = WARMUP
    for signal_bar in np.flatnonzero(directions):
        entry_bar = signal_bar + 1
        if entry_bar < available or entry_bar >= len(frame) or not np.isfinite(unit[signal_bar]):
            continue
        direction = directions[signal_bar]
        entry = open_[entry_bar]
        risk = STOP_ATR * unit[signal_bar]
        if risk <= 0:
            continue
        stop = entry - direction * risk
        target = entry + direction * TARGET_R * risk
        final_bar = min(entry_bar + horizon - 1, len(frame) - 1)
        result: float | None = None
        exit_bar = final_bar
        for bar in range(entry_bar, final_bar + 1):
            stop_hit = low[bar] <= stop if direction > 0 else high[bar] >= stop
            target_hit = high[bar] >= target if direction > 0 else low[bar] <= target
            if stop_hit:  # pessimistic when both barriers occur in one bar
                result, exit_bar = -1.0, bar
                break
            if target_hit:
                result, exit_bar = TARGET_R, bar
                break
        if result is None:
            result = float(np.clip(direction * (close[final_bar] - entry) / risk, -1.0, TARGET_R))
        records.append((frame.index[entry_bar], result - COST_R))
        available = exit_bar + 1
    return pd.DataFrame(records, columns=["time", "r"])


def summarise(cell: str, split: str, trades: pd.DataFrame) -> Result:
    if trades.empty:
        return Result(cell, split, 0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0)
    daily = trades.assign(day=trades.time.dt.floor("D")).groupby("day").r.sum()
    deviation = daily.std(ddof=1)
    sigma = float(daily.mean() / deviation * np.sqrt(len(daily))) if deviation else 0.0
    midpoint = trades.time.min() + (trades.time.max() - trades.time.min()) / 2
    return Result(
        cell,
        split,
        len(trades),
        float(trades.r.sum()),
        float(trades.r.mean()),
        sigma,
        float((trades.r > 0).mean()),
        {str(key): float(value) for key, value in trades.groupby("market").r.sum().items()},
        float(trades.loc[trades.time < midpoint, "r"].sum()),
        float(trades.loc[trades.time >= midpoint, "r"].sum()),
    )


def print_result(result: Result) -> None:
    markets = " ".join(f"{key}={value:+.2f}" for key, value in result.by_market.items())
    print(
        f"{result.split:10} {result.cell:32} n={result.trades:4} "
        f"win={result.wins:5.1%} R={result.total_r:+8.2f} "
        f"R/tr={result.per_trade:+.3f} sigma={result.sigma:+.2f} "
        f"halves={result.first_half_r:+.2f}/{result.second_half_r:+.2f} {markets}"
    )


def split_trades(
    trades: pd.DataFrame, frame: pd.DataFrame, low_fraction: float, high_fraction: float
) -> pd.DataFrame:
    span = frame.index.max() - frame.index.min()
    low_time = frame.index.min() + span * low_fraction
    high_time = frame.index.min() + span * high_fraction
    return trades.loc[(trades.time >= low_time) & (trades.time < high_time)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for timeframe in TIMEFRAMES:
        gold = load_bars(connection, "XAUUSD", timeframe)
        for market, (leg, operation) in COMPONENTS.items():
            fx = load_bars(connection, leg, timeframe)
            if has_bars(connection, market, timeframe):
                actual = load_bars(connection, market, timeframe)
                actual["gold_close"] = gold.close.reindex(actual.index)
                actual["fx_close"] = fx.close.reindex(actual.index)
                actual["fx_sign"] = -1.0 if operation == "divide" else 1.0
                frames[(market, timeframe)] = actual.dropna()
            else:
                frames[(market, timeframe)] = synthetic_cross(gold, fx, operation)
    connection.close()

    cells = [(mechanism, timeframe) for mechanism in MECHANISMS for timeframe in TIMEFRAMES]
    bar = NormalDist().inv_cdf(1.0 - 0.05 / len(cells))
    print(f"SEARCH cells={len(cells)} one-sided Bonferroni bar={bar:.2f}")
    cached: dict[tuple[str, str, str], pd.DataFrame] = {}
    train_results: list[Result] = []
    for mechanism, timeframe in cells:
        pooled = []
        for market in COMPONENTS:
            frame = frames[(market, timeframe)]
            trades = replay(frame, signals(frame, mechanism), HORIZONS[timeframe])
            trades["market"] = market
            cached[(mechanism, timeframe, market)] = trades
            pooled.append(split_trades(trades, frame, 0.0, 0.5))
        train_results.append(summarise(f"{mechanism}/{timeframe}", "train", pd.concat(pooled)))
    train_results.sort(key=lambda item: item.sigma, reverse=True)
    for result in train_results[:10]:
        print_result(result)

    selected = train_results[0]
    mechanism, timeframe = selected.cell.split("/")
    print(f"\nFROZEN WINNER {selected.cell}; later periods were not used for selection")
    later: list[Result] = []
    for name, low, high in (("validation", 0.5, 0.75), ("holdout", 0.75, 1.01)):
        pooled = [
            split_trades(
                cached[(mechanism, timeframe, market)],
                frames[(market, timeframe)],
                low,
                high,
            )
            for market in COMPONENTS
        ]
        result = summarise(selected.cell, name, pd.concat(pooled))
        later.append(result)
        print_result(result)

    validation, holdout = later
    checks = {
        "train cleared multiplicity bar": selected.sigma >= bar,
        "validation positive": validation.total_r > 0,
        "holdout positive": holdout.total_r > 0,
        "every holdout market positive": all(value > 0 for value in holdout.by_market.values()),
        "both holdout halves positive": holdout.first_half_r > 0 and holdout.second_half_r > 0,
        "at least 200 holdout trades": holdout.trades >= 200,
        "holdout at least +2 sigma": holdout.sigma >= 2.0,
    }
    print("\nPROMOTION CHECKS")
    for label, passed in checks.items():
        print(f"[{'x' if passed else ' '}] {label}")
    print("SYNTHETIC_CANDIDATE" if all(checks.values()) else "REJECTED")
    print("Actual broker-bar replay remains mandatory even when every box passes.")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
