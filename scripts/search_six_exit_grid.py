"""Screen S6 stop/target choices on its already-generated causal entries.

The oldest 90 days select, the next 45 validate, and only survivors see the
newest 45. This is a shortlist generator: the winner must be rerun through the
complete dry-run because a different exit changes when the next entry is free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.research_dataset import ResearchDataset


def _stats(rows: list[tuple[pd.Timestamp, float]]) -> tuple[int, float, float]:
    values = np.asarray([value for _when, value in rows], dtype=float)
    if not len(values):
        return 0, -999.0, 0.0
    return len(values), float(values.mean()), float(values.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    entries = pd.read_csv(args.csv, parse_dates=["when"], encoding="cp1252")
    entries = entries.loc[entries.outcome.eq("TRADE")].sort_values("when")
    store = ResearchDataset(args.database, read_only=True)
    try:
        first = entries.when.min()
        last = entries.when.max() + pd.Timedelta(hours=8)
        m1 = store.frame("XAUUSD", "M1", first - pd.Timedelta(hours=1), last)
        h1 = store.frame("XAUUSD", "H1", first - pd.Timedelta(days=3), last)
    finally:
        store.close()

    previous = h1.close.shift(1)
    true_range = pd.concat(
        [h1.high - h1.low, (h1.high - previous).abs(), (h1.low - previous).abs()], axis=1
    ).max(axis=1)
    hourly_atr = true_range.rolling(14).mean()
    start, end = entries.when.min(), entries.when.max()
    selection_end = start + (end - start) / 2
    validation_end = selection_end + (end - start) / 4
    candidates = []

    for stop_scale in (0.75, 1.0, 1.25, 1.5):
        for target_r in (1.5, 2.0, 2.5, 3.0):
            outcomes: list[tuple[pd.Timestamp, float]] = []
            busy_until = None
            for row in entries.itertuples():
                opened = row.when
                if busy_until is not None and opened <= busy_until:
                    continue
                sign = 1.0 if row.direction == "LONG" else -1.0
                entry = float(row.entry)
                base_risk = abs(entry - float(row.stop))
                risk = base_risk * stop_scale
                stop = entry - sign * risk
                target = entry + sign * risk * target_r
                atr_cut = hourly_atr.loc[hourly_atr.index <= opened]
                if atr_cut.empty or not np.isfinite(float(atr_cut.iloc[-1])):
                    continue
                offset = 0.10 * float(atr_cut.iloc[-1])
                managed_stop = stop
                armed = False
                first_bar = int(m1.index.searchsorted(opened, side="left"))
                last_bar = min(first_bar + 480, len(m1))
                result = None
                for position in range(first_bar, last_bar):
                    bar = m1.iloc[position]
                    hit_stop = bar.low <= managed_stop if sign > 0 else bar.high >= managed_stop
                    hit_target = bar.high >= target if sign > 0 else bar.low <= target
                    if hit_stop:
                        result = (managed_stop - entry) / risk * sign
                    elif hit_target:
                        result = target_r
                    if result is not None:
                        busy_until = m1.index[position]
                        break
                    excursion = (bar.high - entry) if sign > 0 else (entry - bar.low)
                    if not armed and excursion >= max(0.25 * risk, offset):
                        armed = True
                        managed_stop = entry + sign * offset
                if result is None:
                    busy_until = last
                    continue
                # Scale the measured round-trip cost with the changed stop.
                cost = float(row.cost_r_charged) / stop_scale
                outcomes.append((opened, result - cost))

            selection = [row for row in outcomes if row[0] < selection_end]
            validation = [row for row in outcomes if selection_end <= row[0] < validation_end]
            holdout = [row for row in outcomes if row[0] >= validation_end]
            sn, sm, st = _stats(selection)
            vn, vm, vt = _stats(validation)
            hn, hm, ht = _stats(holdout)
            if sn >= 100 and vn >= 40 and sm > 0.0 and vm > 0.0:
                candidates.append(
                    (min(sm, vm), stop_scale, target_r, sn, sm, st, vn, vm, vt, hn, hm, ht)
                )

    for row in sorted(candidates, reverse=True):
        _rank, scale, target, sn, sm, st, vn, vm, vt, hn, hm, ht = row
        print(
            f"stop={scale:.2f}x target={target:.1f}R | "
            f"SELECT n={sn} mean={sm:+.3f} total={st:+.2f}R | "
            f"VALID n={vn} mean={vm:+.3f} total={vt:+.2f}R | "
            f"HOLDOUT n={hn} mean={hm:+.3f} total={ht:+.2f}R"
        )
    if not candidates:
        print("NO stop/target candidate survived selection and validation")


if __name__ == "__main__":
    main()
