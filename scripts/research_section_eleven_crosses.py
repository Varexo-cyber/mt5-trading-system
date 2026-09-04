"""Bounded chronological search for gold-cross sections on broker bars."""

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
    "fair_value_gap",
)
STOP_ATRS = (1.0, 1.5, 2.0)
TARGETS_R = (0.5, 0.75, 1.0, 1.5, 2.0)
SESSIONS = {
    "all": tuple(range(24)),
    "asia": tuple(range(0, 7)),
    "london": tuple(range(7, 13)),
    "new_york": tuple(range(13, 20)),
    "late": tuple(range(20, 24)),
}
# Extra round-trip allowance for commission/slippage. Historical broker spread is
# charged separately at each actual entry, relative to that trade's stop.
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
        "SELECT time_utc, open, high, low, close, tick_volume, spread FROM bars "
        "WHERE broker_symbol=? AND timeframe=? ORDER BY time_utc",
        (symbol, timeframe),
    ).fetchall()
    frame = pd.DataFrame(
        rows, columns=["time", "open", "high", "low", "close", "volume", "spread"]
    )
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
    frame["spread"] = np.nan
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
    elif mechanism == "liquidity_sweep":
        ceiling = high.shift(1).rolling(20).max()
        floor = low.shift(1).rolling(20).min()
        long = (low < floor) & (close > floor) & (close > open_)
        short = (high > ceiling) & (close < ceiling) & (close < open_)
    elif mechanism == "bollinger_reversal":
        mean = close.rolling(20).mean()
        deviation = close.rolling(20).std(ddof=0)
        lower, upper = mean - 2.0 * deviation, mean + 2.0 * deviation
        long = (low < lower) & (close > lower) & (close > open_)
        short = (high > upper) & (close < upper) & (close < open_)
    elif mechanism == "volatility_breakout":
        active = unit > 1.25 * unit.rolling(48).mean()
        ceiling = high.shift(1).rolling(20).max()
        floor = low.shift(1).rolling(20).min()
        long = active & (trend > 0) & (close > ceiling)
        short = active & (trend < 0) & (close < floor)
    elif mechanism == "jump_impulse":
        move = close.diff(6) / unit
        unusual_volume = frame.volume > 1.5 * frame.volume.rolling(48).median()
        long = unusual_volume & (move > 1.5) & (close > open_)
        short = unusual_volume & (move < -1.5) & (close < open_)
    elif mechanism == "volume_breakout":
        unusual_volume = frame.volume > 1.5 * frame.volume.rolling(48).median()
        ceiling = high.shift(1).rolling(20).max()
        floor = low.shift(1).rolling(20).min()
        long = unusual_volume & (trend > 0) & (close > ceiling)
        short = unusual_volume & (trend < 0) & (close < floor)
    elif mechanism == "vwap_reversal":
        typical = (high + low + close) / 3.0
        rolling_volume = frame.volume.rolling(48).sum().replace(0.0, np.nan)
        vwap = (typical * frame.volume).rolling(48).sum() / rolling_volume
        long = (close < vwap - unit) & (close > open_)
        short = (close > vwap + unit) & (close < open_)
    elif mechanism.startswith("adaptive_channel_"):
        threshold = float(mechanism.rsplit("_", 1)[1]) / 100.0
        ceiling = high.shift(1).rolling(20).max()
        floor = low.shift(1).rolling(20).min()
        breakout_long, breakout_short = close > ceiling, close < floor
        strong = (fast - slow).abs() / unit > threshold
        long = (strong & breakout_long) | (~strong & breakout_short)
        short = (strong & breakout_short) | (~strong & breakout_long)
    elif mechanism == "jump_regime":
        move = close.diff(6) / unit
        volatility = unit / unit.rolling(48).mean()
        unusual_volume = frame.volume > 1.25 * frame.volume.rolling(48).median()
        # Moderate, liquid moves continue; extreme high-volatility jumps mean-revert.
        continue_long = unusual_volume & (move > 0.75) & (move <= 2.0) & (volatility <= 1.5)
        continue_short = unusual_volume & (move < -0.75) & (move >= -2.0) & (volatility <= 1.5)
        reverse_long = (move < -2.0) & (volatility > 1.5) & (close > open_)
        reverse_short = (move > 2.0) & (volatility > 1.5) & (close < open_)
        long = continue_long | reverse_long
        short = continue_short | reverse_short
    elif mechanism == "vwap_trend_resume":
        typical = (high + low + close) / 3.0
        rolling_volume = frame.volume.rolling(48).sum().replace(0.0, np.nan)
        vwap = (typical * frame.volume).rolling(48).sum() / rolling_volume
        long = (trend > 0) & (low <= vwap) & (close > vwap) & (close > open_)
        short = (trend < 0) & (high >= vwap) & (close < vwap) & (close < open_)
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
    elif mechanism == "fair_value_gap":
        # M5 only. Spans 240 and 600 are the M5 equivalents of H1 EMA20/50.
        h1_fast = close.ewm(span=240, adjust=False).mean()
        h1_slow = close.ewm(span=600, adjust=False).mean()
        long = pd.Series(False, index=frame.index)
        short = pd.Series(False, index=frame.index)
        active: tuple[int, float, int] | None = None
        for index in range(2, len(frame)):
            if active is not None:
                formed, midpoint, direction = active
                if index - formed > 24:
                    active = None
                elif direction > 0 and low.iloc[index] <= midpoint:
                    if close.iloc[index] > midpoint and h1_fast.iloc[index] > h1_slow.iloc[index]:
                        long.iloc[index] = True
                    active = None
                elif direction < 0 and high.iloc[index] >= midpoint:
                    if close.iloc[index] < midpoint and h1_fast.iloc[index] < h1_slow.iloc[index]:
                        short.iloc[index] = True
                    active = None
            body_ratio = abs(close.iloc[index - 1] - open_.iloc[index - 1]) / max(
                high.iloc[index - 1] - low.iloc[index - 1], 1e-12
            )
            impulse = high.iloc[index - 1] - low.iloc[index - 1]
            if body_ratio >= 0.7 and impulse >= 1.5 * unit.iloc[index - 1]:
                if low.iloc[index] > high.iloc[index - 2]:
                    active = (index, (low.iloc[index] + high.iloc[index - 2]) / 2.0, 1)
                elif high.iloc[index] < low.iloc[index - 2]:
                    active = (index, (high.iloc[index] + low.iloc[index - 2]) / 2.0, -1)
    else:
        raise ValueError(mechanism)
    out[long.fillna(False)] = 1
    out[short.fillna(False)] = -1
    out.iloc[:WARMUP] = 0
    return out.to_numpy()


def replay(
    frame: pd.DataFrame,
    directions: np.ndarray,
    horizon: int,
    stop_atr: float,
    target_r: float,
    max_spread_r: float | None = None,
    max_stop_price: float | None = None,
) -> pd.DataFrame:
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
        risk = stop_atr * unit[signal_bar]
        if risk <= 0:
            continue
        if max_stop_price is not None and risk > max_stop_price:
            # At the broker's minimum lot this stop would risk more than the
            # account permits. Research rows that live sizing must refuse are
            # not trading opportunities for this account.
            continue
        spread_cost_r = 0.0
        if "spread_price" in frame:
            spread_price = float(frame.spread_price.iloc[entry_bar])
            if np.isfinite(spread_price):
                spread_cost_r = spread_price / risk
        if max_spread_r is not None and spread_cost_r > max_spread_r:
            continue
        stop = entry - direction * risk
        target = entry + direction * target_r * risk
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
                result, exit_bar = target_r, bar
                break
        if result is None:
            result = float(np.clip(direction * (close[final_bar] - entry) / risk, -1.0, target_r))
        records.append((frame.index[entry_bar], result - COST_R - spread_cost_r))
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


def promotion_checks(
    train: Result, validation: Result, holdout: Result, bar: float
) -> dict[str, bool]:
    return {
        "train cleared multiplicity bar": train.sigma >= bar,
        "validation positive": validation.total_r > 0,
        "holdout positive": holdout.total_r > 0,
        "both holdout halves positive": holdout.first_half_r > 0 and holdout.second_half_r > 0,
        "at least 50 holdout trades": holdout.trades >= 50,
        "holdout at least +2 sigma": holdout.sigma >= 2.0,
    }


def print_checks(checks: dict[str, bool]) -> None:
    for label, passed in checks.items():
        print(f"[{'x' if passed else ' '}] {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    points = {
        symbol: float(__import__("json").loads(spec)["point"])
        for symbol, spec in connection.execute(
            "SELECT broker_symbol, spec_json FROM instruments"
        )
    }
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    actual_markets: set[tuple[str, str]] = set()
    for timeframe in TIMEFRAMES:
        gold = load_bars(connection, "XAUUSD", timeframe)
        for market, (leg, operation) in COMPONENTS.items():
            fx = load_bars(connection, leg, timeframe)
            if has_bars(connection, market, timeframe):
                actual = load_bars(connection, market, timeframe)
                actual["spread_price"] = actual.spread * points[market]
                actual["gold_close"] = gold.close.reindex(actual.index)
                actual["fx_close"] = fx.close.reindex(actual.index)
                actual["fx_sign"] = -1.0 if operation == "divide" else 1.0
                frames[(market, timeframe)] = actual.dropna()
                actual_markets.add((market, timeframe))
            else:
                frames[(market, timeframe)] = synthetic_cross(gold, fx, operation)
    connection.close()

    print("DATA PROVENANCE")
    for market in COMPONENTS:
        actual = [tf for tf in TIMEFRAMES if (market, tf) in actual_markets]
        synthetic = [tf for tf in TIMEFRAMES if (market, tf) not in actual_markets]
        print(
            f"  {market}: actual={','.join(actual) or '-'} "
            f"synthetic={','.join(synthetic) or '-'}"
        )
    if len(actual_markets) != len(COMPONENTS) * len(TIMEFRAMES):
        print("  SYNTHETIC ROWS MAY GENERATE HYPOTHESES BUT MAY NOT PROMOTE LIVE.\n")

    cells = [
        (mechanism, timeframe, polarity, stop_atr, target_r, session)
        for mechanism in MECHANISMS
        for timeframe in TIMEFRAMES
        for polarity in (1, -1)
        for stop_atr in STOP_ATRS
        for target_r in TARGETS_R
        for session in SESSIONS
        if mechanism != "fair_value_gap" or timeframe == "M5"
    ]
    bar = NormalDist().inv_cdf(1.0 - 0.05 / len(cells))
    print(f"SEARCH cells={len(cells)} one-sided Bonferroni bar={bar:.2f}")
    raw_cache: dict[tuple[str, str, int, float, float, str], pd.DataFrame] = {}
    cached: dict[tuple[str, str, int, float, float, str, str], pd.DataFrame] = {}
    signal_cache: dict[tuple[str, str, str], np.ndarray] = {}
    train_results: list[Result] = []
    for mechanism, timeframe, polarity, stop_atr, target_r, session in cells:
        pooled = []
        for market in COMPONENTS:
            frame = frames[(market, timeframe)]
            raw_key = (mechanism, timeframe, polarity, stop_atr, target_r, market)
            if raw_key not in raw_cache:
                signal_key = (mechanism, timeframe, market)
                if signal_key not in signal_cache:
                    signal_cache[signal_key] = signals(frame, mechanism)
                raw = replay(
                    frame,
                    polarity * signal_cache[signal_key],
                    HORIZONS[timeframe],
                    stop_atr,
                    target_r,
                )
                raw["market"] = market
                raw_cache[raw_key] = raw
            raw = raw_cache[raw_key]
            trades = raw.loc[raw.time.dt.hour.isin(SESSIONS[session])].copy()
            cached[(mechanism, timeframe, polarity, stop_atr, target_r, session, market)] = trades
            pooled.append(split_trades(trades, frame, 0.0, 0.5))
        direction = "follow" if polarity > 0 else "fade"
        train_results.append(
            summarise(
                f"{mechanism}/{timeframe}/{direction}/{stop_atr:g}ATR/{target_r:g}R/{session}",
                "train",
                pd.concat(pooled),
            )
        )
    # Never let a handful of lucky trades win a large parameter search.
    eligible_train = [item for item in train_results if item.trades >= 200]
    eligible_train.sort(key=lambda item: item.sigma, reverse=True)
    train_results.sort(key=lambda item: item.sigma, reverse=True)
    for result in train_results[:10]:
        print_result(result)

    selected = eligible_train[0]
    mechanism, timeframe, direction, stop_label, target_label, session = selected.cell.split("/")
    polarity = 1 if direction == "follow" else -1
    stop_atr = float(stop_label.removesuffix("ATR"))
    target_r = float(target_label.removesuffix("R"))
    print(f"\nFROZEN WINNER {selected.cell}; later periods were not used for selection")
    later: list[Result] = []
    for name, low, high in (("validation", 0.5, 0.75), ("holdout", 0.75, 1.01)):
        pooled = [
            split_trades(
                cached[(mechanism, timeframe, polarity, stop_atr, target_r, session, market)],
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
    checks = promotion_checks(selected, validation, holdout, bar)
    checks["every holdout market positive"] = all(value > 0 for value in holdout.by_market.values())
    checks["at least 200 holdout trades"] = holdout.trades >= 200
    checks.pop("at least 50 holdout trades")
    print("\nPROMOTION CHECKS")
    print_checks(checks)
    print("SYNTHETIC_CANDIDATE" if all(checks.values()) else "REJECTED")

    print(f"\nPER-MARKET SEARCH — each market pays all {len(cells)} trials")
    for market in COMPONENTS:
        market_train: list[Result] = []
        for (
            candidate_mechanism,
            candidate_timeframe,
            candidate_polarity,
            candidate_stop,
            candidate_target,
            candidate_session,
        ) in cells:
            candidate_frame = frames[(market, candidate_timeframe)]
            candidate_trades = cached[
                (
                    candidate_mechanism,
                    candidate_timeframe,
                    candidate_polarity,
                    candidate_stop,
                    candidate_target,
                    candidate_session,
                    market,
                )
            ]
            train = split_trades(candidate_trades, candidate_frame, 0.0, 0.5)
            candidate_direction = "follow" if candidate_polarity > 0 else "fade"
            market_train.append(
                summarise(
                    f"{candidate_mechanism}/{candidate_timeframe}/{candidate_direction}/"
                    f"{candidate_stop:g}ATR/{candidate_target:g}R/{candidate_session}",
                    "train",
                    train,
                )
            )
        eligible_market_train = [item for item in market_train if item.trades >= 100]
        eligible_market_train.sort(key=lambda item: item.sigma, reverse=True)
        winner = eligible_market_train[0]
        (
            winner_mechanism,
            winner_timeframe,
            winner_direction,
            winner_stop_label,
            winner_target_label,
            winner_session,
        ) = winner.cell.split("/")
        winner_polarity = 1 if winner_direction == "follow" else -1
        winner_stop = float(winner_stop_label.removesuffix("ATR"))
        winner_target = float(winner_target_label.removesuffix("R"))
        winner_frame = frames[(market, winner_timeframe)]
        winner_trades = cached[
            (
                winner_mechanism,
                winner_timeframe,
                winner_polarity,
                winner_stop,
                winner_target,
                winner_session,
                market,
            )
        ]
        market_validation = summarise(
            winner.cell,
            "validation",
            split_trades(winner_trades, winner_frame, 0.5, 0.75),
        )
        market_holdout = summarise(
            winner.cell,
            "holdout",
            split_trades(winner_trades, winner_frame, 0.75, 1.01),
        )
        print(f"\n{market} FROZEN WINNER")
        print_result(winner)
        print_result(market_validation)
        print_result(market_holdout)
        market_checks = promotion_checks(winner, market_validation, market_holdout, bar)
        print_checks(market_checks)
        print("SYNTHETIC_CANDIDATE" if all(market_checks.values()) else "REJECTED")

        # The holdout has already been inspected above. This screen therefore
        # generates the next hypothesis; it does not validate one. Any row it
        # finds still needs fresh actual-broker data before implementation.
        stable: list[tuple[float, Result, Result, Result]] = []
        for (
            candidate_mechanism,
            candidate_timeframe,
            candidate_polarity,
            candidate_stop,
            candidate_target,
            candidate_session,
        ) in cells:
            candidate_frame = frames[(market, candidate_timeframe)]
            candidate_trades = cached[
                (
                    candidate_mechanism,
                    candidate_timeframe,
                    candidate_polarity,
                    candidate_stop,
                    candidate_target,
                    candidate_session,
                    market,
                )
            ]
            candidate_direction = "follow" if candidate_polarity > 0 else "fade"
            label = (
                f"{candidate_mechanism}/{candidate_timeframe}/{candidate_direction}/"
                f"{candidate_stop:g}ATR/{candidate_target:g}R/{candidate_session}"
            )
            discovery_train = summarise(
                label,
                "train",
                split_trades(candidate_trades, candidate_frame, 0.0, 0.5),
            )
            discovery_validation = summarise(
                label,
                "validation",
                split_trades(candidate_trades, candidate_frame, 0.5, 0.75),
            )
            discovery_holdout = summarise(
                label,
                "holdout",
                split_trades(candidate_trades, candidate_frame, 0.75, 1.01),
            )
            splits = (discovery_train, discovery_validation, discovery_holdout)
            if (
                all(result.trades >= 50 and result.total_r > 0 for result in splits)
                and discovery_holdout.first_half_r > 0
                and discovery_holdout.second_half_r > 0
            ):
                floor = min(result.per_trade for result in splits)
                stable.append(
                    (floor, discovery_train, discovery_validation, discovery_holdout)
                )
        stable.sort(key=lambda item: item[0], reverse=True)
        print(f"{market} STABILITY DISCOVERY (not independent evidence)")
        if not stable:
            print("  none")
        for floor, discovery_train, discovery_validation, discovery_holdout in stable[:3]:
            print(f"  floor R/trade {floor:+.3f}")
            print_result(discovery_train)
            print_result(discovery_validation)
            print_result(discovery_holdout)
        own_chart = [
            item
            for item in stable
            if not item[1].cell.startswith(("two_leg_trend", "two_leg_impulse"))
        ]
        print(f"{market} TOP OWN-CHART DISCOVERIES")
        if not own_chart:
            print("  none")
        else:
            for floor, own_train, own_validation, own_holdout in own_chart[:3]:
                print(f"  floor R/trade {floor:+.3f}")
                print_result(own_train)
                print_result(own_validation)
                print_result(own_holdout)

    print("Actual broker-bar replay remains mandatory even when every box passes.")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
