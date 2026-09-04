"""Chronological, broker-costed strategy search for BTCUSD."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from scripts.research_section_eleven_crosses import (
    COST_R,
    HORIZONS,
    STOP_ATRS,
    TARGETS_R,
    load_bars,
    print_checks,
    print_result,
    promotion_checks,
    replay,
    signals,
    split_trades,
    summarise,
)

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1")
MECHANISMS = (
    "trend_pullback",
    "channel_breakout",
    "aligned_momentum",
    "trend_rejection",
    "contraction_breakout",
    "liquidity_sweep",
    "bollinger_reversal",
    "volatility_breakout",
)
SESSIONS = {
    "all": tuple(range(24)),
    "asia": tuple(range(0, 7)),
    "london": tuple(range(7, 13)),
    "new_york": tuple(range(13, 20)),
    "late": tuple(range(20, 24)),
    "weekday": tuple(range(24)),
    "weekend": tuple(range(24)),
}


def clock_filter(trades: pd.DataFrame, session: str) -> pd.Series:
    mask = trades.time.dt.hour.isin(SESSIONS[session])
    if session == "weekday":
        mask &= trades.time.dt.dayofweek < 5
    elif session == "weekend":
        mask &= trades.time.dt.dayofweek >= 5
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    row = connection.execute(
        "SELECT broker_symbol, spec_json FROM instruments "
        "WHERE canonical_symbol='BTCUSD'"
    ).fetchone()
    if row is None:
        raise SystemExit("database contains no actual BTCUSD broker instrument")
    broker_symbol, spec_json = row
    point = float(json.loads(spec_json)["point"])
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in TIMEFRAMES:
        frame = load_bars(connection, broker_symbol, timeframe)
        if frame.empty:
            raise SystemExit(f"database contains no BTCUSD {timeframe} bars")
        frame["spread_price"] = frame.spread * point
        frames[timeframe] = frame
    connection.close()

    cells = [
        (mechanism, timeframe, polarity, stop_atr, target_r, session)
        for mechanism in MECHANISMS
        for timeframe in TIMEFRAMES
        for polarity in (1, -1)
        for stop_atr in STOP_ATRS
        for target_r in TARGETS_R
        for session in SESSIONS
    ]
    multiplicity_bar = NormalDist().inv_cdf(1.0 - 0.05 / len(cells))
    print(
        f"BTCUSD ACTUAL BROKER SEARCH: {len(cells)} cells, "
        f"spread + {COST_R:.2f}R execution allowance, bar={multiplicity_bar:.2f}"
    )

    signal_cache: dict[tuple[str, str], object] = {}
    trade_cache: dict[tuple[object, ...], pd.DataFrame] = {}
    train_results = []
    for mechanism, timeframe, polarity, stop_atr, target_r, session in cells:
        frame = frames[timeframe]
        signal_key = (mechanism, timeframe)
        if signal_key not in signal_cache:
            signal_cache[signal_key] = signals(frame, mechanism)
        replay_key = (mechanism, timeframe, polarity, stop_atr, target_r)
        if replay_key not in trade_cache:
            replayed = replay(
                frame,
                polarity * signal_cache[signal_key],
                HORIZONS[timeframe],
                stop_atr,
                target_r,
            )
            replayed["market"] = "BTCUSD"
            trade_cache[replay_key] = replayed
        trades = trade_cache[replay_key]
        trades = trades.loc[clock_filter(trades, session)].copy()
        direction = "follow" if polarity > 0 else "fade"
        label = (
            f"{mechanism}/{timeframe}/{direction}/{stop_atr:g}ATR/"
            f"{target_r:g}R/{session}"
        )
        train_results.append(
            summarise(label, "train", split_trades(trades, frame, 0.0, 0.5))
        )

    eligible = [result for result in train_results if result.trades >= 100]
    eligible.sort(key=lambda result: result.sigma, reverse=True)
    if not eligible:
        raise SystemExit("no BTCUSD training candidate produced 100 trades")
    for result in eligible[:10]:
        print_result(result)

    winner = eligible[0]
    mechanism, timeframe, direction, stop_label, target_label, session = winner.cell.split("/")
    polarity = 1 if direction == "follow" else -1
    stop_atr = float(stop_label.removesuffix("ATR"))
    target_r = float(target_label.removesuffix("R"))
    frame = frames[timeframe]
    trades = trade_cache[(mechanism, timeframe, polarity, stop_atr, target_r)]
    trades = trades.loc[clock_filter(trades, session)].copy()
    validation = summarise(
        winner.cell, "validation", split_trades(trades, frame, 0.5, 0.75)
    )
    holdout = summarise(
        winner.cell, "holdout", split_trades(trades, frame, 0.75, 1.01)
    )
    print(f"\nFROZEN WINNER {winner.cell}")
    print_result(winner)
    print_result(validation)
    print_result(holdout)
    checks = promotion_checks(winner, validation, holdout, multiplicity_bar)
    print("\nPROMOTION CHECKS")
    print_checks(checks)
    passed = all(checks.values())
    print("PROMOTION_CANDIDATE" if passed else "REJECTED — REMAINS SHADOW")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
