"""Section 6 search using Jarvis' exact trade semantics.

The frozen production model is evaluated once per M5 close. Candidate exits
use the real bid/ask entry, one-position-per-symbol rule, 480 M1-bar horizon,
pessimistic same-bar resolution, recorded spread and live sizer cost.
Candidates rank on an old training block, must survive a later validation
block, and only then disclose the newest holdout block.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.section_six_adaptive import _GOLD, model_reading
from backtesting.research_dataset import ResearchDataset
from config.loader import load_settings
from core.types import Direction, TradingMode
from risk.position_sizer import PositionSizer

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Candidate:
    threshold: float
    stop_atr: float
    target_r: float


@dataclass(frozen=True, slots=True)
class Trade:
    opened: pd.Timestamp
    result_r: float
    direction: int
    regime: int


def _model_series(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Production hands every module the newest 260 closed bars.  EMA state
    # depends on where its frame begins, so evaluating one full month at once
    # is not equivalent and previously produced a flattering, false ranking.
    # Recompute the frozen reader on the exact rolling frame Jarvis receives.
    score = np.full(len(frame), np.nan)
    exact_atr = np.full(len(frame), np.nan)
    for end in range(80, len(frame) + 1):
        found = model_reading(frame.iloc[max(0, end - 260) : end], _GOLD)
        if found is not None:
            score[end - 1], exact_atr[end - 1] = found[0] * -1.0, found[1]
    return score, exact_atr


def _simulate(candidate, m5, m1, score, atr, spec, sizer, start, end) -> list[Trade]:
    m1_index = m1.index
    m1_high = m1["high"].to_numpy(float)
    m1_low = m1["low"].to_numpy(float)
    m1_close = m1["close"].to_numpy(float)
    spread = m5["spread"].to_numpy(float) if "spread" in m5 else np.zeros(len(m5))
    close = m5["close"].astype(float)
    regime_values = (
        close.ewm(span=288, adjust=False).mean() - close.ewm(span=1152, adjust=False).mean()
    ).to_numpy(float)
    closes = m5.index + pd.Timedelta(minutes=5)
    chosen = np.flatnonzero(
        (closes >= start)
        & (closes <= end)
        & np.isfinite(score)
        & np.isfinite(atr)
        & (atr > 0.0)
        & (np.abs(score) >= candidate.threshold)
    )
    out: list[Trade] = []
    busy_until = None
    for at in chosen:
        opened = closes[at]
        if busy_until is not None and opened <= busy_until:
            continue
        sign = 1 if score[at] > 0.0 else -1
        direction = Direction(sign)
        quote_at = int(m1_index.searchsorted(opened - pd.Timedelta(minutes=1), side="right")) - 1
        if quote_at < 0:
            continue
        spread_price = float(spread[at]) * spec.point
        mid = float(m1_close[quote_at])
        entry = mid + sign * spread_price / 2.0
        risk = float(atr[at]) * candidate.stop_atr
        # The module anchors invalidation on tick.mid; Confluence executes at
        # ask/bid. Preserve that half-spread difference exactly.
        stop = mid - sign * risk
        risk = abs(entry - stop)
        target = entry + sign * risk * candidate.target_r
        sized = sizer.size(
            spec=spec,
            equity=203.0,
            direction=direction,
            entry=entry,
            sl=stop,
            tp=target,
            spread_price=spread_price,
        )
        if not sized.approved:
            continue
        first = int(m1_index.searchsorted(opened, side="left"))
        last = min(first + 480, len(m1_index))
        result = None
        exit_at = None
        for position in range(first, last):
            hit_stop = m1_low[position] <= stop if sign > 0 else m1_high[position] >= stop
            hit_target = m1_high[position] >= target if sign > 0 else m1_low[position] <= target
            if hit_stop:
                result, exit_at = -1.0, m1_index[position]
                break
            if hit_target:
                result, exit_at = candidate.target_r, m1_index[position]
                break
        if result is None or exit_at is None:
            busy_until = end + pd.Timedelta(minutes=5)
            continue
        busy_until = exit_at
        regime = 1 if regime_values[at] >= 0.0 else -1
        out.append(Trade(opened, result - sizer.cost_share(spec, risk, spread_price), sign, regime))
    return out


def _stats(rows: list[Trade]) -> tuple[int, float, float, float]:
    if not rows:
        return 0, -999.0, 0.0, 0.0
    values = np.asarray([row.result_r for row in rows])
    return len(rows), float(values.mean()), float(values.sum()), float((values > 0).mean())


def _positive_blocks(rows: list[Trade], start: pd.Timestamp, days: int, count: int) -> int:
    totals = [0.0] * count
    for row in rows:
        bucket = min(max(int((row.opened - start).days // days), 0), count - 1)
        totals[bucket] += row.result_r
    return sum(total > 0.0 for total in totals)


def _in_session(row: Trade, first_hour: int, hours: int) -> bool:
    return (row.opened.hour - first_hour) % 24 < hours


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=30)
    args = parser.parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    settings = settings.model_copy(
        update={"system": settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})}
    )
    sizer = PositionSizer(settings)
    store = ResearchDataset(args.database, read_only=True)
    try:
        window = store.window()
        if window is None:
            raise SystemExit("database has no evaluation window metadata")
        _days, end_text = window
        end = pd.Timestamp(end_text)
        start = end - pd.Timedelta(days=args.days)
        holdout_start = end - pd.Timedelta(days=args.holdout_days)
        validation_start = holdout_start - pd.Timedelta(days=args.validation_days)
        if validation_start <= start:
            raise SystemExit("--days must exceed validation-days + holdout-days")
        m5 = store.frame("XAUUSD", "M5", start - pd.Timedelta(days=7), end)
        m1 = store.frame("XAUUSD", "M1", start - pd.Timedelta(days=1), end)
        spec = store.spec("XAUUSD")
        score, atr = _model_series(m5)
        ranked = []
        candidates = (
            Candidate(threshold, stop, target)
            for threshold in (0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
            for stop in (0.625, 0.675, 0.70, 0.725, 0.75)
            for target in (2.0, 2.25, 2.5, 2.75, 3.0)
        )
        for candidate in candidates:
            rows = _simulate(candidate, m5, m1, score, atr, spec, sizer, start, end)
            selection = [row for row in rows if row.opened < validation_start]
            validation = [row for row in rows if validation_start <= row.opened < holdout_start]
            holdout = [row for row in rows if row.opened >= holdout_start]
            sn, sm, st, sw = _stats(selection)
            vn, vm, vt, vw = _stats(validation)
            hn, hm, ht, hw = _stats(holdout)
            green = _positive_blocks(selection, start, 30, 4)
            if sn >= 300 and sm > 0.02 and green >= 3 and vn >= 60 and vm > 0.02:
                ranked.append(
                    (
                        min(sm, vm),
                        green,
                        sn,
                        candidate,
                        sn,
                        sm,
                        st,
                        sw,
                        vn,
                        vm,
                        vt,
                        vw,
                        hn,
                        hm,
                        ht,
                        hw,
                    )
                )
        ranked.sort(reverse=True, key=lambda row: row[:3])
        print("RANKED ON OLD TRAIN + VALIDATION; NEWEST BLOCK IS UNTOUCHED HOLDOUT")
        for (
            _rank,
            green,
            _count,
            candidate,
            sn,
            sm,
            st,
            sw,
            vn,
            vm,
            vt,
            vw,
            hn,
            hm,
            ht,
            hw,
        ) in ranked[: args.top]:
            print(
                f"{candidate} | SELECT n={sn:4d} win={sw:.1%} mean={sm:+.3f} "
                f"total={st:+.2f}R green_months={green}/4 | "
                f"VALID n={vn:3d} win={vw:.1%} mean={vm:+.3f} total={vt:+.2f}R | "
                f"HOLDOUT n={hn:3d} "
                f"win={hw:.1%} mean={hm:+.3f} total={ht:+.2f}R"
            )
        if not ranked:
            print("NO candidate cleared the predeclared selection requirements")

        # A simple session/routing search, ranked on the selection period only.
        # Sessions are contiguous clock windows; no individual hour picking.
        # This is deliberately printed separately because it is an additional
        # hypothesis search, not free confirmation.
        routed = []
        route_candidates = (
            Candidate(threshold, stop, target)
            for threshold in (0.10, 0.15, 0.20)
            for stop in (0.675, 0.70, 0.75, 0.80, 0.90, 1.0)
            for target in (1.5, 2.0, 2.25, 2.5, 2.75, 3.0)
        )
        for candidate in route_candidates:
            rows = _simulate(candidate, m5, m1, score, atr, spec, sizer, start, end)
            for first_hour in range(24):
                for hours in (4, 6, 8, 10, 12):
                    for direction in (-1, 0, 1, 2):
                        selected = [
                            row
                            for row in rows
                            if row.opened < validation_start
                            and _in_session(row, first_hour, hours)
                            and (
                                direction == 0
                                or row.direction == direction
                                or (direction == 2 and row.direction == row.regime)
                            )
                        ]
                        validated = [
                            row
                            for row in rows
                            if validation_start <= row.opened < holdout_start
                            and _in_session(row, first_hour, hours)
                            and (
                                direction == 0
                                or row.direction == direction
                                or (direction == 2 and row.direction == row.regime)
                            )
                        ]
                        held = [
                            row
                            for row in rows
                            if row.opened >= holdout_start
                            and _in_session(row, first_hour, hours)
                            and (
                                direction == 0
                                or row.direction == direction
                                or (direction == 2 and row.direction == row.regime)
                            )
                        ]
                        sn, sm, st, sw = _stats(selected)
                        vn, vm, vt, vw = _stats(validated)
                        hn, hm, ht, hw = _stats(held)
                        green = _positive_blocks(selected, start, 30, 4)
                        if sn >= 200 and sm >= 0.05 and green >= 3 and vn >= 40 and vm >= 0.05:
                            routed.append(
                                (
                                    min(sm, vm),
                                    sn,
                                    candidate,
                                    first_hour,
                                    hours,
                                    direction,
                                    sn,
                                    sm,
                                    st,
                                    sw,
                                    vn,
                                    vm,
                                    vt,
                                    vw,
                                    hn,
                                    hm,
                                    ht,
                                    hw,
                                )
                            )
        routed.sort(reverse=True, key=lambda row: row[:2])
        print("\nSESSION ROUTES RANKED ON TRAIN + VALIDATION; NEWEST BLOCK UNTOUCHED")
        for (
            _rank,
            _count,
            candidate,
            first,
            hours,
            direction,
            sn,
            sm,
            st,
            sw,
            vn,
            vm,
            vt,
            vw,
            hn,
            hm,
            ht,
            hw,
        ) in routed[: args.top]:
            route = {
                -1: "SHORT",
                0: "BOTH",
                1: "LONG",
                2: "TREND",
            }[direction]
            print(
                f"{candidate} UTC={first:02d}-{(first + hours) % 24:02d} {route:5s} | "
                f"SELECT n={sn:3d} win={sw:.1%} mean={sm:+.3f} total={st:+.2f}R | "
                f"VALID n={vn:3d} win={vw:.1%} mean={vm:+.3f} total={vt:+.2f}R | "
                f"HOLDOUT n={hn:3d} win={hw:.1%} mean={hm:+.3f} total={ht:+.2f}R"
            )
        if not routed:
            print("NO session route cleared the predeclared selection requirements")
    finally:
        store.close()


if __name__ == "__main__":
    main()
