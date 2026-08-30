"""Turn the sweep into a decision: which six, at what timeframe, and why not the rest.

Pooling rule: cells are summed across symbols WITHIN an asset class, never
across. An edge that exists on four equity indices and not on eleven currency
pairs is a real finding about equity indices; averaging the two into one row
hides both.

A detector is reported as surviving only if all four hold:

    1. train sigma clears Bonferroni over the whole grid
    2. the holdout agrees in SIGN and reaches at least 2.0 sigma
    3. expectancy net of a pessimistic spread is positive in BOTH halves
    4. it produced enough trades to matter

Rule 2 is the one that does the work. Every dead idea so far -- the far-break
continuation, the payoff sweep, the section six own lane -- passed rule 1.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from scripts.lab.resolve import sigmas

SPREAD_ATR = {"fx": 0.04, "metal": 0.10, "index": 0.08}
CONTROL = "coin_flip_control"
MIN_TRADES = 400
HOLDOUT_SIGMA = 2.0


def pooled(rows: list[dict]) -> dict:
    """Sum wins and resolved counts over symbols, keeping the ratio split."""
    out: dict = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "n": 0, "unresolved": 0, "days": 0})
    )
    units = []
    for row in rows:
        units.append(row["unit_atr"])
        for k, halves in row["ratios"].items():
            for half, cell in halves.items():
                bucket = out[k][half]
                bucket["wins"] += cell["wins"]
                bucket["n"] += cell["n"]
                bucket["unresolved"] += cell["unresolved"]
                bucket["days"] = max(bucket["days"], cell.get("days", 0))
    finite = [u for u in units if u == u]
    return {
        "ratios": out,
        "unit_atr": sum(finite) / len(finite) if finite else float("nan"),
        "symbols": len({r["symbol"] for r in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="sweep.json")
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    args = parser.parse_args()

    rows = json.loads(Path(args.file).read_text())
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["timeframe"], row["asset"])].append(row)

    # THE BASELINE IS THE COIN FLIP, NOT THE ARITHMETIC 1/(1+k).
    #
    # A driftless martingale touches a stop of 1 before a target of k with
    # probability 1/(1+k) exactly -- but that is a statement about a continuous
    # process, and these are OHLC bars. A bar registers a barrier when its
    # extreme crosses it, so both barriers get overshot, and the overshoot is
    # proportionally larger on the NEARER one. The stop is therefore
    # effectively further away than nominal, which favours the target, and the
    # bias grows with the payoff ratio exactly as observed:
    #
    #     random entries, R:R 1.0    E -0.002    (nothing)
    #     random entries, R:R 2.0    E +0.049
    #     random entries, R:R 3.0    E +0.073    +13.8 sigma
    #
    # Thirteen sigma of edge from entering at random. Every expectancy in this
    # study carries that, so it is subtracted per (timeframe, asset, ratio)
    # rather than assumed away. This is what the control was for.
    control: dict[tuple[str, str, str], float] = {}
    for (strategy, timeframe, asset), group in groups.items():
        if strategy != CONTROL:
            continue
        summary = pooled(group)
        for k, halves in summary["ratios"].items():
            wins = halves["train"]["wins"] + halves["test"]["wins"]
            n = halves["train"]["n"] + halves["test"]["n"]
            if n:
                hit = wins / n
                control[(timeframe, asset, k)] = hit * float(k) - (1 - hit)
    if not control:
        print("!! no coin-flip control in this file: expectancies are NOT bias-adjusted\n")

    ratios = sorted({k for row in rows for k in row["ratios"]}, key=float)
    cells = len(groups) * len(ratios)
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * max(cells, 1)))
    print(f"{len(rows)} measured cells, {len(groups)} groups, {cells} comparisons")
    print(f"Bonferroni bar {bar:.2f} sigma on TRAIN; holdout must reach {HOLDOUT_SIGMA:.1f}\n")

    survivors = []
    everything = []
    for key, group in sorted(groups.items()):
        strategy, timeframe, asset = key
        if strategy == CONTROL:
            continue
        summary = pooled(group)
        spread = SPREAD_ATR[asset]
        unit = summary["unit_atr"]
        # Cost per trade in R: one spread crossed each way, over a stop that is
        # `unit` ATR wide.
        cost = 2.0 * spread / unit if unit and unit == unit else float("inf")
        for k in ratios:
            kf = float(k)
            train = summary["ratios"][k]["train"]
            test = summary["ratios"][k]["test"]
            if train["n"] < args.min_trades or test["n"] < args.min_trades // 4:
                continue
            # The sigma baseline moves with the control too: the hit rate a
            # coin flip achieves here, not the one theory says it should.
            drift0 = control.get((timeframe, asset, k), 0.0)
            base = (1.0 + drift0) / (1.0 + kf)
            ts = sigmas(train["wins"], train["n"], base)
            hs = sigmas(test["wins"], test["n"], base)
            # Clustered by day. `days` is the max over symbols, i.e. the count
            # of distinct dates one instrument contributed, so the ratio
            # days/n is how much of the sample is genuinely independent when
            # every symbol fires together.
            ts *= (min(train.get("days") or train["n"], train["n"]) / train["n"]) ** 0.5
            hs *= (min(test.get("days") or test["n"], test["n"]) / test["n"]) ** 0.5
            drift = control.get((timeframe, asset, k), 0.0)
            te = train["wins"] / train["n"] * kf - (1 - train["wins"] / train["n"]) - drift
            he = test["wins"] / test["n"] * kf - (1 - test["wins"] / test["n"]) - drift
            record = {
                "strategy": strategy,
                "timeframe": timeframe,
                "asset": asset,
                "ratio": kf,
                "train_n": train["n"],
                "test_n": test["n"],
                "train_sigma": ts,
                "test_sigma": hs,
                "train_E": te,
                "test_E": he,
                "cost": cost,
                "net_train": te - cost,
                "net_test": he - cost,
                "unit_atr": unit,
            }
            everything.append(record)
            if (
                ts >= bar
                and hs >= HOLDOUT_SIGMA
                and record["net_train"] > 0
                and record["net_test"] > 0
            ):
                survivors.append(record)

    def show(records, title):
        print(f"=== {title}  ({len(records)})")
        if not records:
            print("    nothing\n")
            return
        print(
            f"    {'strategy':<17}{'tf':>4}{'asset':>7}{'R:R':>5}{'trainN':>8}{'testN':>7}"
            f"{'trS':>7}{'teS':>7}{'trE':>8}{'teE':>8}{'cost':>7}{'net_te':>8}"
        )
        for r in records:
            print(
                f"    {r['strategy']:<17}{r['timeframe']:>4}{r['asset']:>7}{r['ratio']:>5.1f}"
                f"{r['train_n']:>8}{r['test_n']:>7}{r['train_sigma']:>+7.1f}"
                f"{r['test_sigma']:>+7.1f}{r['train_E']:>+8.3f}{r['test_E']:>+8.3f}"
                f"{r['cost']:>7.3f}{r['net_test']:>+8.3f}"
            )
        print()

    survivors.sort(key=lambda r: -r["net_test"])
    show(survivors, "SURVIVED ALL FOUR RULES")

    # The near misses are worth printing: they say WHICH rule killed each idea.
    killed = defaultdict(list)
    for r in everything:
        if r in survivors:
            continue
        if r["train_sigma"] < bar:
            killed["no edge on train"].append(r)
        elif r["test_sigma"] < HOLDOUT_SIGMA:
            killed["train only -- holdout disagreed"].append(r)
        else:
            killed["real but eaten by cost"].append(r)
    for reason, records in killed.items():
        records.sort(key=lambda r: -r["train_sigma"])
        show(records[:12], f"REJECTED: {reason} (top 12 by train sigma)")

    print("=== best ratio per surviving (strategy, timeframe, asset)")
    best: dict[tuple, dict] = {}
    for r in survivors:
        key = (r["strategy"], r["timeframe"], r["asset"])
        if key not in best or r["net_test"] > best[key]["net_test"]:
            best[key] = r
    for key, r in sorted(best.items(), key=lambda kv: -kv[1]["net_test"]):
        print(
            f"    {key[0]:<17}{key[1]:>4}{key[2]:>7}  R:R {r['ratio']:.1f}  "
            f"net holdout {r['net_test']:+.3f}R  over {r['test_n']} trades"
        )


if __name__ == "__main__":
    main()
