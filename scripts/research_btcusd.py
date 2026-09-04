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
    load_bars,
    print_checks,
    print_result,
    promotion_checks,
    replay,
    signals,
    split_trades,
    summarise,
)

TIMEFRAMES = ("M1", "M5", "M15")
STOP_ATRS = (1.0, 2.0, 3.0, 4.0, 6.0)
TARGETS_R = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
MAX_SPREAD_R = {"M1": 0.15, "M5": 0.12, "M15": 0.10}
MECHANISMS = (
    "trend_pullback",
    "channel_breakout",
    "aligned_momentum",
    "trend_rejection",
    "contraction_breakout",
    "liquidity_sweep",
    "bollinger_reversal",
    "volatility_breakout",
    "jump_impulse",
    "volume_breakout",
    "vwap_reversal",
    "adaptive_channel_25",
    "adaptive_channel_50",
    "adaptive_channel_100",
    "jump_regime",
    "vwap_trend_resume",
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
    if trades.empty:
        return pd.Series(False, index=trades.index, dtype=bool)
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
                MAX_SPREAD_R[timeframe],
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

    all_passed = True
    for wanted_timeframe in TIMEFRAMES:
        eligible = [
            result
            for result in train_results
            if f"/{wanted_timeframe}/" in result.cell and result.trades >= 100
        ]
        eligible.sort(key=lambda result: result.sigma, reverse=True)
        if not eligible:
            print(f"\n{wanted_timeframe}: no candidate produced 100 training trades")
            all_passed = False
            continue
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
        print(f"\nSECTION BTC-{timeframe} FROZEN WINNER {winner.cell}")
        print_result(winner)
        print_result(validation)
        print_result(holdout)
        checks = promotion_checks(winner, validation, holdout, multiplicity_bar)
        print_checks(checks)
        passed = all(checks.values())
        all_passed &= passed
        print("PROMOTION_CANDIDATE" if passed else "REJECTED — REMAINS SHADOW")

        discoveries = []
        for candidate in cells:
            candidate_mechanism, candidate_timeframe, candidate_polarity, candidate_stop, candidate_target, candidate_session = candidate
            if candidate_timeframe != wanted_timeframe:
                continue
            candidate_direction = "follow" if candidate_polarity > 0 else "fade"
            candidate_label = (
                f"{candidate_mechanism}/{candidate_timeframe}/{candidate_direction}/"
                f"{candidate_stop:g}ATR/{candidate_target:g}R/{candidate_session}"
            )
            candidate_trades = trade_cache[
                (candidate_mechanism, candidate_timeframe, candidate_polarity, candidate_stop, candidate_target)
            ]
            candidate_trades = candidate_trades.loc[
                clock_filter(candidate_trades, candidate_session)
            ].copy()
            parts = tuple(
                summarise(
                    candidate_label,
                    name,
                    split_trades(candidate_trades, frame, low, high),
                )
                for name, low, high in (
                    ("train", 0.0, 0.5),
                    ("validation", 0.5, 0.75),
                    ("holdout", 0.75, 1.01),
                )
            )
            if all(part.trades >= 50 and part.total_r > 0 for part in parts):
                discoveries.append((min(part.per_trade for part in parts), parts))
        discoveries.sort(key=lambda item: item[0], reverse=True)
        print(f"{wanted_timeframe} STABLE DISCOVERIES — NOT INDEPENDENT EVIDENCE")
        if not discoveries:
            print("  none")
        for floor, parts in discoveries[:3]:
            print(f"  floor R/trade {floor:+.3f}")
            for part in parts:
                print_result(part)
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
