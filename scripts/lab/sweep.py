"""Run every detector over every market and every timeframe, and be honest.

    python -m lab.sweep [--timeframes M5,M15] [--strategies level_retest,...]

Output is one row per (strategy, timeframe, asset class, ratio), split into a
training half and a holdout half by DATE, with the holdout printed alongside
rather than after -- the holdout here is not selecting anything, it is being
shown, and a number that only survives in one column says so on its own line.

Bonferroni runs over the whole grid, which is the point of running it as one
grid: 10 detectors x 6 timeframes x 4 ratios is 240 comparisons, and at 240
comparisons a three-sigma result happens by chance most runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from scripts.lab import data
from scripts.lab.resolve import chance, expectancy, resolve, sigmas
from scripts.lab.strategies import CATALOGUE, HORIZON_BARS
from scripts.lab.zoo2 import ALL as _ZOO_ALL
from scripts.lab.zoo3 import ALL3 as _ZOO3

CATALOGUE = {**CATALOGUE, **_ZOO_ALL, **_ZOO3}

RATIOS = (1.0, 1.5, 2.0, 3.0)
SPLIT = pd.Timestamp("2017-01-01", tz="UTC")
#: Spread as a share of one ATR, per asset class, for the cost column. Rough
#: and deliberately pessimistic -- it is there to sort survivors from
#: gross-only curiosities, not to price a broker.
SPREAD_ATR = {"fx": 0.04, "metal": 0.10, "index": 0.08}


def measure(symbol: str, timeframe: str, name: str) -> dict | None:
    frame = data.load(symbol, timeframe)
    if len(frame) < 2000:
        return None
    a = data.atr(frame)
    batch = CATALOGUE[name](frame, a)
    if len(batch) == 0:
        return None
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    train = np.asarray(frame.index[batch.index] < SPLIT)

    # One R in ATR, needed to turn a spread into a cost share.
    scale = np.where(a[batch.index] > 0, a[batch.index], np.nan)
    unit_atr = float(np.nanmedian(batch.unit / scale))

    # HOW MANY INDEPENDENT DAYS, not how many trades. Eleven currency pairs
    # gap on the same Monday morning; counting that as eleven observations
    # overstates significance by about sqrt(11). Distinct signal DATES is the
    # conservative denominator, and for detectors that fire all day it barely
    # differs from the trade count.
    days = frame.index[batch.index].normalize()

    out: dict = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": name,
        "asset": data.asset_class(symbol),
        "signals": len(batch),
        "unit_atr": unit_atr,
        "ratios": {},
    }
    for k in RATIOS:
        outcome = resolve(high, low, batch, k, HORIZON_BARS)
        row = {}
        for half, mask in (("train", train), ("test", ~train)):
            sub = outcome[mask]
            resolved = int((sub >= 0).sum())
            wins = int((sub == 1).sum())
            row[half] = {
                "days": int(days[mask][sub >= 0].nunique()) if resolved else 0,
                "n": resolved,
                "unresolved": int((sub == -1).sum()),
                "wins": wins,
                "hit": wins / resolved if resolved else 0.0,
                "sigma": sigmas(wins, resolved, chance(k)),
                "E": expectancy(wins, resolved, k),
            }
        out["ratios"][str(k)] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", default=",".join(data.TIMEFRAMES))
    parser.add_argument("--strategies", default=",".join(CATALOGUE))
    parser.add_argument("--symbols", default=",".join(data.every_symbol()))
    parser.add_argument("--out", default="sweep.json")
    args = parser.parse_args()

    timeframes = args.timeframes.split(",")
    strategies = args.strategies.split(",")
    symbols = args.symbols.split(",")

    cells = len(strategies) * len(timeframes) * len(RATIOS)
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * max(cells, 1)))
    print(
        f"{len(symbols)} symbols x {len(timeframes)} timeframes " f"x {len(strategies)} strategies"
    )
    print(f"Bonferroni over {cells} comparisons: needs {bar:.2f} sigma")
    print(
        f"train < {SPLIT.date()}   holdout >= {SPLIT.date()}   " f"horizon {HORIZON_BARS} bars\n",
        flush=True,
    )

    results = []
    started = time.time()
    for timeframe in timeframes:
        for name in strategies:
            for symbol in symbols:
                if timeframe not in data.available(symbol):
                    continue
                try:
                    row = measure(symbol, timeframe, name)
                except FileNotFoundError:
                    continue
                if row is not None:
                    results.append(row)
            print(f"  [{time.time() - started:7.0f}s] {timeframe:>4} {name}", flush=True)
        Path(args.out).write_text(json.dumps(results))
    Path(args.out).write_text(json.dumps(results))
    print(f"\n{len(results)} cells -> {args.out}")


if __name__ == "__main__":
    main()
