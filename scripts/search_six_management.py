"""Screen causal S6 trend filters on an existing managed dry-run.

The first 90 days select, the next 45 validate, and only surviving rules may
see the newest 45-day holdout. This is a cheap screen: removing trades can
change later availability, so every survivor must still be replayed through
``dry_run_sections`` before promotion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.research_dataset import ResearchDataset


def _stats(frame: pd.DataFrame) -> tuple[int, float, float]:
    values = frame["managed_r_LIVE"].astype(float)
    return len(values), float(values.mean()) if len(values) else -999.0, float(values.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    trades = pd.read_csv(args.csv, parse_dates=["when"], encoding="cp1252")
    trades = trades.loc[trades.outcome.eq("TRADE")].copy()
    store = ResearchDataset(args.database, read_only=True)
    try:
        m5 = store.frame(
            "XAUUSD",
            "M5",
            trades.when.min() - pd.Timedelta(days=10),
            trades.when.max(),
        )
    finally:
        store.close()

    close = m5["close"].astype(float)
    features: dict[str, pd.Series] = {}
    for fast, slow in ((8, 32), (20, 50), (50, 200), (288, 1152)):
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        features[f"ema_{fast}_{slow}"] = fast_ema - slow_ema
    for period in (20, 50, 100, 288):
        ema = close.ewm(span=period, adjust=False).mean()
        for bars in (3, 12, 48):
            features[f"slope_{period}_{bars}"] = ema - ema.shift(bars)
    for bars in (12, 48, 288):
        features[f"return_{bars}"] = close - close.shift(bars)

    feature_frame = pd.DataFrame(features, index=m5.index + pd.Timedelta(minutes=5))
    feature_frame.index.name = "when"
    joined = pd.merge_asof(
        trades.sort_values("when"),
        feature_frame.reset_index().sort_values("when"),
        on="when",
        direction="backward",
    )
    start, end = joined.when.min(), joined.when.max()
    selection_end = start + (end - start) / 2
    validation_end = selection_end + (end - start) / 4

    survivors = []
    for name in features:
        for sign in (1, -1):
            kept = joined.loc[joined[name].astype(float) * sign > 0.0]
            selection = kept.loc[kept.when < selection_end]
            validation = kept.loc[(kept.when >= selection_end) & (kept.when < validation_end)]
            holdout = kept.loc[kept.when >= validation_end]
            sn, sm, st = _stats(selection)
            vn, vm, vt = _stats(validation)
            hn, hm, ht = _stats(holdout)
            if sn >= 150 and vn >= 60 and sm > 0.01 and vm > 0.01:
                survivors.append((min(sm, vm), name, sign, sn, sm, st, vn, vm, vt, hn, hm, ht))

    survivors.sort(reverse=True)
    print("RANKED ON FIRST 90D + NEXT 45D; NEWEST 45D IS UNTOUCHED HOLDOUT")
    for row in survivors[: args.top]:
        _rank, name, sign, sn, sm, st, vn, vm, vt, hn, hm, ht = row
        side = "> 0" if sign > 0 else "< 0"
        print(
            f"{name:18s} {side:3s} | SELECT n={sn:4d} mean={sm:+.3f} total={st:+.2f}R | "
            f"VALID n={vn:3d} mean={vm:+.3f} total={vt:+.2f}R | "
            f"HOLDOUT n={hn:3d} mean={hm:+.3f} total={ht:+.2f}R"
        )
    if not survivors:
        print("NO causal trend filter survived selection and validation")


if __name__ == "__main__":
    main()
