"""Tune the retest -- on 2012-2018 only -- then read 2018-2022 ONCE.

The retest measured +0.099R at 3:1 over 107,660 signals and +19 sigma. That is
the strongest thing in this account's history and it is exactly why it must
now be attacked rather than celebrated: a sweep over tolerance, stop distance
and payoff is 60-odd cells, and the best cell of 60 is a number about the
sweep, not about the market.

So: every parameter is chosen on the first six years. The last four are read
once, at the end, with the winner already fixed. A holdout that gets peeked at
twice is not a holdout, so this script prints the test column only for the
single configuration the training half selected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_edges import HORIZON, SYMBOLS, Signal, atr, load, resolve, sigmas

SPLIT = pd.Timestamp("2018-01-01")


def retest_signals(
    frame: pd.DataFrame,
    a: np.ndarray,
    period: int,
    tolerance: float,
    stop_beyond: float,
) -> list[tuple[int, Signal]]:
    """(break index, entry signal) for every break that came back to its level.

    `tolerance`  how close to the level counts as the retest, in ATR
    `stop_beyond` how far past the level the stop sits, in ATR

    One R is the entry-to-stop distance, so a tighter retest and a tighter stop
    both shrink R -- which lifts the payoff on the same objective and raises
    cost as a share of R at the same time. Those two pull in opposite
    directions and that is the whole point of sweeping them.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper = pd.Series(high).shift(1).rolling(period).max().to_numpy()
    lower = pd.Series(low).shift(1).rolling(period).min().to_numpy()
    out: list[tuple[int, Signal]] = []
    i = period + 15
    end = len(close) - HORIZON - 1
    while i < end:
        if close[i] > upper[i]:
            direction, level = 1, upper[i]
        elif close[i] < lower[i]:
            direction, level = -1, lower[i]
        else:
            i += 1
            continue
        unit0 = a[i]
        if not np.isfinite(unit0) or unit0 <= 0:
            i += 1
            continue
        for j in range(i + 1, min(i + HORIZON, end)):
            if direction > 0:
                failed = close[j] < level - stop_beyond * unit0
                touched = low[j] <= level + tolerance * unit0
            else:
                failed = close[j] > level + stop_beyond * unit0
                touched = high[j] >= level - tolerance * unit0
            # TOUCHED BEFORE FAILED, and this ordering was wrong until
            # 30 August. The entry sits between the level and the stop, so
            # price cannot reach the stop without passing through the limit
            # order first. Testing the failure first DISCARDED every bar that
            # swept through both at once -- 6% to 16% of the sample, all of
            # them losses -- and that alone accounted for three quarters of
            # this strategy's apparent edge (+0.336R -> +0.063R).
            if touched:
                entry = level + direction * tolerance * unit0
                unit = (stop_beyond + tolerance) * unit0
                out.append((i, Signal(j, direction, entry, unit)))
                i = j
                break
            if failed:
                break
        i += 1
    return out


def main() -> None:
    frames = {s: load(s) for s in SYMBOLS}
    atrs = {s: atr(f) for s, f in frames.items()}
    cuts = {s: int((frames[s].index < SPLIT).sum()) for s in SYMBOLS}
    print(f"train < {SPLIT.date()}   test >= {SPLIT.date()}")
    print(
        f"train bars {sum(cuts.values())}   test bars "
        f"{sum(len(frames[s]) - cuts[s] for s in SYMBOLS)}\n"
    )

    periods = (10, 20, 40)
    tolerances = (0.15, 0.35, 0.60)
    stops = (0.35, 0.50, 0.75)
    ratios = (2.0, 3.0, 4.0)
    cells = len(periods) * len(tolerances) * len(stops) * len(ratios)
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * cells))
    print(f"{cells} cells swept on TRAIN only; needs {bar:.2f} sigma\n")
    print(
        f"{'per':>4} {'tol':>5} {'stop':>5} {'R:R':>4} {'n':>7} {'hit':>7} "
        f"{'edge':>6} {'sigma':>7} {'E':>8} {'E@c':>8}"
    )

    results = []
    cache: dict[tuple[int, float, float], dict[str, list]] = {}
    for period in periods:
        for tolerance in tolerances:
            for stop_beyond in stops:
                key = (period, tolerance, stop_beyond)
                cache[key] = {
                    s: retest_signals(frames[s], atrs[s], period, tolerance, stop_beyond)
                    for s in SYMBOLS
                }
                for k in ratios:
                    wins = resolved = 0
                    for symbol in SYMBOLS:
                        frame = frames[symbol]
                        high, low = frame["high"].to_numpy(), frame["low"].to_numpy()
                        for break_index, signal in cache[key][symbol]:
                            if break_index >= cuts[symbol]:
                                continue  # test half: not looked at
                            outcome = resolve(high, low, signal, k, same_bar=True)
                            if outcome is None:
                                continue
                            resolved += 1
                            wins += outcome
                    if resolved < 200:
                        continue
                    hit = wins / resolved
                    chance = 1.0 / (1.0 + k)
                    s = sigmas(wins, resolved, chance)
                    expectancy = hit * k - (1 - hit)
                    # Cost in R rises as the stop shrinks: a fixed 0.04 ATR of
                    # spread against a stop of (stop+tol) ATR.
                    cost = 0.04 / (stop_beyond + tolerance)
                    net = expectancy - 2 * cost
                    results.append((net, expectancy, s, resolved, key, k, cost))
                    flag = "  <--" if s >= bar else ""
                    print(
                        f"{period:>4} {tolerance:>5.2f} {stop_beyond:>5.2f} {k:>4.1f} "
                        f"{resolved:>7} {100 * hit:>6.1f}% {100 * (hit - chance):>+5.1f} "
                        f"{s:>+7.2f} {expectancy:>+8.3f} {net:>+8.3f}{flag}"
                    )

    results.sort(reverse=True)
    print("\n=== best five on TRAIN, by expectancy net of modelled cost")
    for net, expectancy, s, n, key, k, cost in results[:5]:
        print(
            f"    period {key[0]}  tol {key[1]}  stop {key[2]}  R:R {k}  "
            f"n={n}  E {expectancy:+.3f}  cost {2 * cost:.3f}  net {net:+.3f}  {s:+.2f}s"
        )

    winner = results[0]
    _, _, _, _, key, k, cost = winner
    print(f"\n=== HOLDOUT, read once, for period {key[0]} tol {key[1]} " f"stop {key[2]} R:R {k}")
    wins = resolved = 0
    for symbol in SYMBOLS:
        frame = frames[symbol]
        high, low = frame["high"].to_numpy(), frame["low"].to_numpy()
        for break_index, signal in cache[key][symbol]:
            if break_index < cuts[symbol]:
                continue
            outcome = resolve(high, low, signal, k, same_bar=True)
            if outcome is None:
                continue
            resolved += 1
            wins += outcome
    hit = wins / resolved
    chance = 1.0 / (1.0 + k)
    expectancy = hit * k - (1 - hit)
    print(
        f"    n={resolved}  hit {100 * hit:.1f}%  chance {100 * chance:.1f}%  "
        f"edge {100 * (hit - chance):+.1f}  sigma {sigmas(wins, resolved, chance):+.2f}"
    )
    print(f"    E {expectancy:+.3f}R   net of cost {expectancy - 2 * cost:+.3f}R")


if __name__ == "__main__":
    main()
