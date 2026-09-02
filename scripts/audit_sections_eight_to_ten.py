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
from scripts.lab.strategies import HORIZON_BARS
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
                low[j] <= level + 0.15 * unit0
                if direction > 0
                else high[j] >= level - 0.15 * unit0
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
    Candidate(10, "closed_bar_swing_retest", "SPX500", "M5", 0.75),
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
