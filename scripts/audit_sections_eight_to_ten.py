"""Exact broker-cost audit for the frozen section 8-10 candidates.

The broad zoo is only a shortlist: it uses a conservative asset-class cost
estimate and permits overlapping signals.  This audit charges the spread
stored on every candidate bar, asks the real EUR-203 position sizer whether
the order can exist, and permits only one open trade per symbol.  Parameters
are fixed before this file is run; both chronological halves are printed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from core.types import Direction
from risk.position_sizer import PositionSizer
from scripts.audit_section_four_candidates import _resolve_one
from scripts.lab import data
from scripts.lab.resolve import Batch
from scripts.lab.strategies import CATALOGUE as BASE
from scripts.lab.strategies import HORIZON_BARS, _channel
from scripts.lab.zoo2 import ALL as ZOO2
from scripts.lab.zoo3 import ALL3, _last_swing_prices

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = {**BASE, **ZOO2, **ALL3}


def _closed_bar_swing_retest(frame: pd.DataFrame, atr: np.ndarray) -> Batch:
    """Executable version: enter at the close of the completed retest bar."""

    high, low, close = (frame[column].to_numpy(float) for column in ("high", "low", "close"))
    swing_high, swing_low = _last_swing_prices(high, low, 3)
    rows: list[tuple[int, int, float, float]] = []
    end = len(close) - HORIZON_BARS - 1
    i = 43
    while i < end:
        unit0 = atr[i]
        if not (np.isfinite(unit0) and unit0 > 0.0):
            i += 1
            continue
        if np.isfinite(swing_high[i]) and close[i] > swing_high[i]:
            direction, level = 1, float(swing_high[i])
        elif np.isfinite(swing_low[i]) and close[i] < swing_low[i]:
            direction, level = -1, float(swing_low[i])
        else:
            i += 1
            continue
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            touched = (
                low[j] <= level + 0.15 * unit0 if direction > 0 else high[j] >= level - 0.15 * unit0
            )
            failed = (
                close[j] < level - 0.85 * unit0
                if direction > 0
                else close[j] > level + 0.85 * unit0
            )
            if touched:
                entry = float(close[j])
                stop = level - direction * 0.85 * unit0
                unit = direction * (entry - stop)
                if unit > 0.0:
                    rows.append((j, direction, entry, unit))
                i = j
                break
            if failed:
                break
        i += 1
    if not rows:
        empty = np.empty(0)
        return Batch(empty.astype(int), empty, empty, empty, False)
    values = np.asarray(rows, dtype=float)
    return Batch(values[:, 0].astype(int), values[:, 1], values[:, 2], values[:, 3], False)


CATALOGUE["closed_bar_swing_retest"] = _closed_bar_swing_retest


def _closed_bar_order_block(frame: pd.DataFrame, atr: np.ndarray) -> Batch:
    """Executable order block: wait for a revisit bar to close, then enter."""

    open_, high, low, close = (
        frame[column].to_numpy(float) for column in ("open", "high", "low", "close")
    )
    rows: list[tuple[int, int, float, float]] = []
    end = len(close) - HORIZON_BARS - 1
    i = 50
    while i < end:
        unit0 = atr[i]
        if not (np.isfinite(unit0) and unit0 > 0.0):
            i += 1
            continue
        move = (close[i] - open_[i]) / unit0
        if abs(move) < 1.5:
            i += 1
            continue
        direction = 1 if move > 0 else -1
        block = None
        for k in range(i - 1, max(i - 6, 0), -1):
            if (close[k] < open_[k]) == (direction > 0):
                block = (min(open_[k], close[k]), max(open_[k], close[k]))
                break
        if block is None:
            i += 1
            continue
        edge = block[1] if direction > 0 else block[0]
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            touched = (
                low[j] <= edge + 0.5 * unit0 if direction > 0 else high[j] >= edge - 0.5 * unit0
            )
            if touched:
                entry = float(close[j])
                stop = edge - direction * unit0
                unit = direction * (entry - stop)
                if unit > 0.0:
                    rows.append((j, direction, entry, unit))
                i = j
                break
        i += 1
    if not rows:
        empty = np.empty(0)
        return Batch(empty.astype(int), empty, empty, empty, False)
    values = np.asarray(rows, dtype=float)
    return Batch(values[:, 0].astype(int), values[:, 1], values[:, 2], values[:, 3], False)


CATALOGUE["closed_bar_order_block"] = _closed_bar_order_block


def _market_big_impulse_retest(
    frame: pd.DataFrame,
    atr: np.ndarray,
    *,
    stop_beyond: float,
    confirmation: bool,
    entry_hours: tuple[int, int] | None = None,
) -> Batch:
    """Large channel break, then an executable close/confirmation entry."""

    high, low, close = (frame[column].to_numpy(float) for column in ("high", "low", "close"))
    upper, lower = _channel(high, low, 20)
    rows: list[tuple[int, int, float, float]] = []
    end = len(close) - HORIZON_BARS - 1
    i = 220
    while i < end:
        unit0 = atr[i]
        if not (np.isfinite(unit0) and unit0 > 0.0):
            i += 1
            continue
        if close[i] > upper[i]:
            direction, level = 1, float(upper[i])
        elif close[i] < lower[i]:
            direction, level = -1, float(lower[i])
        else:
            i += 1
            continue
        if direction * (close[i] - level) < unit0:
            i += 1
            continue
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            failed = (
                close[j] < level - stop_beyond * unit0
                if direction > 0
                else close[j] > level + stop_beyond * unit0
            )
            touched = (
                low[j] <= level + 0.15 * unit0 if direction > 0 else high[j] >= level - 0.15 * unit0
            )
            if failed:
                break
            if not touched:
                continue
            entry_at = j
            if confirmation:
                entry_at = -1
                for k in range(j, min(j + 4, end)):
                    confirms = (
                        close[k] > level and close[k] > close[max(k - 1, 0)]
                        if direction > 0
                        else close[k] < level and close[k] < close[max(k - 1, 0)]
                    )
                    invalid = (
                        close[k] < level - stop_beyond * unit0
                        if direction > 0
                        else close[k] > level + stop_beyond * unit0
                    )
                    if invalid:
                        break
                    if confirms:
                        entry_at = k
                        break
                if entry_at < 0:
                    break
            entry = float(close[entry_at])
            stop = level - direction * stop_beyond * unit0
            risk = direction * (entry - stop)
            if risk > 0.0:
                hour = int(frame.index[entry_at].hour)
                if entry_hours is None or entry_hours[0] <= hour < entry_hours[1]:
                    rows.append((entry_at, direction, entry, risk))
            i = entry_at
            break
        i += 1
    if not rows:
        empty = np.empty(0)
        return Batch(empty.astype(int), empty, empty, empty, False)
    values = np.asarray(rows, dtype=float)
    return Batch(values[:, 0].astype(int), values[:, 1], values[:, 2], values[:, 3], False)


for _stop in (0.35, 0.75, 1.0):
    CATALOGUE[f"market_big_retest_s{_stop}"] = lambda frame, atr, stop=_stop: (
        _market_big_impulse_retest(frame, atr, stop_beyond=stop, confirmation=False)
    )
    CATALOGUE[f"confirmed_big_retest_s{_stop}"] = lambda frame, atr, stop=_stop: (
        _market_big_impulse_retest(frame, atr, stop_beyond=stop, confirmation=True)
    )

for _start, _end in ((3, 19), (6, 19), (13, 19)):
    CATALOGUE[f"market_big_retest_s0.75_h{_start}_{_end}"] = (
        lambda frame, atr, start=_start, end=_end: _market_big_impulse_retest(
            frame,
            atr,
            stop_beyond=0.75,
            confirmation=False,
            entry_hours=(start, end),
        )
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    section: int
    strategy: str
    symbol: str
    timeframe: str
    target_r: float


CANDIDATES = (
    Candidate(8, "trend_day_continuation", "SPX500", "H1", 1.0),
    Candidate(9, "session_vwap_reversion", "USDJPY.i", "M30", 1.5),
    Candidate(10, "market_big_retest_s0.75_h3_19", "XAUUSD", "M1", 1.5),
    Candidate(0, "swing_break_retest", "EURCHF.i", "M5", 0.75),
    Candidate(0, "trend_day_continuation", "GBPJPY.i", "M5", 1.5),
    Candidate(0, "rsi_divergence", "AUDUSD.i", "M5", 1.5),
    Candidate(0, "session_vwap_reversion", "USDJPY.i", "M30", 1.5),
    Candidate(0, "range_fade", "EURCHF.i", "H1", 1.5),
    Candidate(0, "false_break", "EURGBP.i", "M30", 1.5),
    Candidate(0, "session_vwap_reversion", "USDJPY.i", "M15", 1.5),
    Candidate(0, "range_fade", "EURUSD.i", "M15", 1.5),
    Candidate(0, "range_fade", "EURJPY.i", "M15", 1.5),
    Candidate(0, "range_fade", "EURCHF.i", "M30", 1.5),
    Candidate(0, "rsi_divergence", "USDCHF.i", "M15", 1.5),
    Candidate(0, "rsi_divergence", "EURCHF.i", "M15", 1.0),
    Candidate(0, "swing_break_retest", "EURUSD.i", "M30", 0.75),
    Candidate(0, "swing_break_retest", "GBPUSD.i", "M30", 0.5),
    Candidate(0, "swing_break_retest", "USDCAD.i", "M30", 0.75),
    Candidate(0, "swing_break_retest", "NZDUSD.i", "H1", 0.5),
    Candidate(0, "swing_break_retest", "US30", "H1", 1.0),
    Candidate(0, "swing_break_retest", "NDX100", "M30", 1.0),
    Candidate(0, "swing_break_retest", "GER40", "M5", 0.75),
    Candidate(0, "swing_break_retest", "SPX500", "M5", 0.75),
    Candidate(0, "swing_break_retest", "XAUUSD", "M5", 0.5),
    Candidate(0, "inside_day_squeeze", "GER40", "M15", 1.5),
    Candidate(0, "trend_day_continuation", "SPX500", "H1", 1.0),
    Candidate(0, "closed_bar_order_block", "XAUUSD", "M1", 0.75),
    Candidate(0, "closed_bar_order_block", "XAUUSD", "M5", 0.75),
    Candidate(0, "closed_bar_order_block", "SPX500", "M1", 0.75),
    Candidate(0, "closed_bar_order_block", "SPX500", "M5", 0.75),
    Candidate(0, "closed_bar_order_block", "NDX100", "M1", 0.75),
    Candidate(0, "closed_bar_order_block", "NDX100", "M5", 0.75),
    Candidate(0, "closed_bar_order_block", "US30", "M1", 0.75),
    Candidate(0, "closed_bar_order_block", "US30", "M5", 0.75),
    Candidate(0, "closed_bar_order_block", "GER40", "M1", 0.75),
    Candidate(0, "closed_bar_order_block", "GER40", "M5", 0.75),
    Candidate(0, "retest_big_impulse", "XAUUSD", "M1", 1.5),
    Candidate(0, "first_hour_projection", "NDX100", "M5", 1.5),
    Candidate(0, "ema_stretch_fade", "USDJPY.i", "M1", 1.5),
    Candidate(0, "midpoint_reversion", "USDJPY.i", "M1", 1.5),
    Candidate(0, "session_vwap_reversion", "USDJPY.i", "M1", 1.5),
    *(
        Candidate(0, strategy, "XAUUSD", "M1", target)
        for strategy in (
            "market_big_retest_s0.35",
            "market_big_retest_s0.75",
            "market_big_retest_s1.0",
            "confirmed_big_retest_s0.35",
            "confirmed_big_retest_s0.75",
            "confirmed_big_retest_s1.0",
        )
        for target in (0.5, 0.75, 1.0, 1.5)
    ),
)


def _audit(candidate: Candidate, sizer: PositionSizer) -> pd.DataFrame:
    frame = data.load(candidate.symbol, candidate.timeframe)
    spec = data.instrument_spec(candidate.symbol)
    batch = CATALOGUE[candidate.strategy](frame, data.atr(frame))
    rows: list[dict[str, object]] = []
    last_exit = -1
    for at, direction, entry, unit in zip(
        batch.index, batch.direction, batch.entry, batch.unit, strict=True
    ):
        at = int(at)
        if at <= last_exit:
            continue
        result = _resolve_one(
            frame,
            index=at,
            direction=int(direction),
            entry=float(entry),
            unit=float(unit),
            ratio=candidate.target_r,
            same_bar=batch.same_bar,
        )
        if result is None:
            continue
        won, exit_at = result
        spread_price = float(frame["spread"].iloc[at]) * spec.point
        sized = sizer.size(
            spec=spec,
            equity=203.0,
            direction=Direction(int(direction)),
            entry=float(entry),
            sl=float(entry) - int(direction) * float(unit),
            tp=float(entry) + int(direction) * float(unit) * candidate.target_r,
            spread_price=spread_price,
            risk_pct=2.0,
            enforce_minimum_rr=False,
        )
        if not sized.approved:
            continue
        last_exit = int(exit_at)
        cost = sizer.cost_share(spec, float(unit), spread_price)
        net_r = (candidate.target_r if won else -1.0) - cost
        rows.append(
            {
                "stamp": frame.index[at],
                "exit_stamp": frame.index[exit_at],
                "won": bool(won),
                "net_r": net_r,
                "money": net_r * sized.actual_risk_money,
            }
        )
    return pd.DataFrame(rows)


def _resolve_managed(
    frame: pd.DataFrame,
    *,
    index: int,
    direction: int,
    entry: float,
    unit: float,
    ratio: float,
    trigger_r: float,
    offset_price: float,
) -> tuple[float, int] | None:
    """Conservative break-even walk: an excursion only arms for the next bar."""

    stop = entry - direction * unit
    target = entry + direction * unit * ratio
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    end = min(index + HORIZON_BARS, len(frame) - 1)
    armed = False
    for at in range(index + 1, end + 1):
        hit_stop = low[at] <= stop if direction > 0 else high[at] >= stop
        hit_target = high[at] >= target if direction > 0 else low[at] <= target
        if hit_stop:
            return direction * (stop - entry) / unit, at
        if hit_target:
            return ratio, at
        excursion = high[at] - entry if direction > 0 else entry - low[at]
        if not armed and excursion / unit >= trigger_r:
            armed = True
            stop = entry + direction * offset_price
    return None


def audit_management(
    candidate: Candidate,
    sizer: PositionSizer,
    *,
    trigger_r: float,
    offset_atr: float,
) -> pd.DataFrame:
    """Audit one moving-stop rule with its own chronological position occupancy."""

    frame = data.load(candidate.symbol, candidate.timeframe)
    spec = data.instrument_spec(candidate.symbol)
    atr_values = data.atr(frame)
    batch = CATALOGUE[candidate.strategy](frame, atr_values)
    rows: list[dict[str, object]] = []
    last_exit = -1
    for at, direction, entry, unit in zip(
        batch.index, batch.direction, batch.entry, batch.unit, strict=True
    ):
        at = int(at)
        direction = int(direction)
        entry = float(entry)
        unit = float(unit)
        if at <= last_exit:
            continue
        spread_price = float(frame["spread"].iloc[at]) * spec.point
        sized = sizer.size(
            spec=spec,
            equity=203.0,
            direction=Direction(direction),
            entry=entry,
            sl=entry - direction * unit,
            tp=entry + direction * unit * candidate.target_r,
            spread_price=spread_price,
            risk_pct=2.0,
            enforce_minimum_rr=False,
        )
        if not sized.approved:
            continue
        result = _resolve_managed(
            frame,
            index=at,
            direction=direction,
            entry=entry,
            unit=unit,
            ratio=candidate.target_r,
            trigger_r=trigger_r,
            offset_price=offset_atr * float(atr_values[at]),
        )
        if result is None:
            continue
        gross_r, last_exit = result
        cost = sizer.cost_share(spec, unit, spread_price)
        net_r = gross_r - cost
        rows.append(
            {
                "stamp": frame.index[at],
                "exit_stamp": frame.index[last_exit],
                "won": gross_r > 0.0,
                "net_r": net_r,
                "money": net_r * sized.actual_risk_money,
            }
        )
    return pd.DataFrame(rows)


def _print(label: str, rows: pd.DataFrame) -> None:
    if rows.empty:
        print(f"    {label:<8} no executable resolved trades")
        return
    curve = rows.sort_values("stamp")["net_r"].cumsum()
    drawdown = float((curve - curve.cummax()).min())
    print(
        f"    {label:<8} n={len(rows):4d} win={rows['won'].mean():6.1%} "
        f"net={rows['net_r'].sum():+8.2f}R avg={rows['net_r'].mean():+.3f}R "
        f"money={rows['money'].sum():+8.2f} maxDD={drawdown:+.2f}R"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    sizer = PositionSizer(settings)
    data.configure_database(args.database)
    try:
        days, end_text = data.database_window() or (180, "")
        end = pd.Timestamp(end_text)
        start = end - pd.Timedelta(days=days)
        split = start + (end - start) * 0.60
        for candidate in CANDIDATES:
            rows = _audit(candidate, sizer)
            print(
                f"S{candidate.section} {candidate.strategy} "
                f"{candidate.symbol}/{candidate.timeframe} target={candidate.target_r}R"
            )
            if rows.empty:
                _print("EARLY", rows)
                _print("LATE", rows)
                _print("ALL", rows)
                continue
            _print("EARLY", rows[rows["stamp"] < split])
            _print("LATE", rows[rows["stamp"] >= split])
            _print("ALL", rows)
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
