"""Audit the few near-misses from the broad Section Four search honestly.

The detector zoo is intentionally permissive: it finds hypotheses.  This
second pass asks whether a hypothesis can actually be expressed by the live
account.  It uses the stored Eightcap contract, historical spread, configured
commission/slippage, the EUR 203 lot-size constraint, and prevents overlapping
positions on one instrument.  A pretty gross hit rate that fails here is not a
live candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.research_dataset import ResearchDataset
from config.loader import load_settings
from core.types import Direction
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.lab.resolve import Batch
from scripts.lab.strategies import HORIZON_BARS
from scripts.lab.sweep import CATALOGUE
from scripts.lab.zoo3 import order_block

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    timeframe: str
    asset: str
    ratio: float


CANDIDATES = (
    Candidate("hour21_long", "H1", "fx", 0.5),
    Candidate("level_retest_t0.15_b1.25", "H1", "index", 0.5),
    Candidate("level_retest_t0.15_b1.25", "H1", "metal", 0.5),
    Candidate("retest_big_impulse", "M1", "index", 1.5),
    Candidate("retest_big_impulse", "M5", "index", 2.0),
    Candidate("order_block_live", "M1", "index", 0.5),
    Candidate("order_block_live", "M1", "index", 1.0),
    Candidate("order_block_live", "M5", "index", 0.75),
    Candidate("order_block_live", "M15", "index", 1.0),
    Candidate("order_block_live", "M30", "index", 1.0),
    Candidate("order_block_live", "H1", "index", 1.0),
    Candidate("coin_flip_control", "M1", "index", 0.5),
    Candidate("coin_flip_control", "M5", "index", 0.75),
    Candidate("coin_flip_control", "M1", "index", 1.5),
    Candidate("coin_flip_control", "M5", "index", 2.0),
    Candidate("coin_flip_control", "M15", "index", 1.0),
    Candidate("coin_flip_control", "M30", "index", 1.0),
    Candidate("coin_flip_control", "H1", "index", 1.0),
)


def _hour21(frame: pd.DataFrame, atr: np.ndarray) -> Batch:
    close = frame["close"].to_numpy()
    picked = np.flatnonzero((frame.index.hour == 21) & (frame.index.minute == 0))
    picked = picked[(picked >= 30) & (picked < len(frame) - HORIZON_BARS - 1)]
    picked = picked[np.isfinite(atr[picked]) & (atr[picked] > 0)]
    return Batch(picked, np.ones(len(picked)), close[picked], atr[picked], False)


def _resolve_one(
    frame: pd.DataFrame,
    *,
    index: int,
    direction: int,
    entry: float,
    unit: float,
    ratio: float,
    same_bar: bool,
) -> tuple[int, int] | None:
    stop = entry - direction * unit
    target = entry + direction * unit * ratio
    first = 0 if same_bar else 1
    end = min(index + HORIZON_BARS, len(frame) - 1)
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    for at in range(index + first, end + 1):
        hit_stop = low[at] <= stop if direction > 0 else high[at] >= stop
        hit_target = high[at] >= target if direction > 0 else low[at] <= target
        if hit_stop:
            return 0, at  # same-bar ambiguity is deliberately a loss
        if hit_target:
            return 1, at
    return None


def audit(database: Path, equity: float) -> list[dict[str, object]]:
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    sizer = PositionSizer(settings)
    records: list[dict[str, object]] = []
    data.configure_database(database)
    try:
        for candidate in CANDIDATES:
            for symbol in data.every_symbol():
                if data.asset_class(symbol) != candidate.asset:
                    continue
                frame = data.load(symbol, candidate.timeframe)
                atr = data.atr(frame)
                if candidate.name == "hour21_long":
                    batch = _hour21(frame, atr)
                elif candidate.name == "order_block_live":
                    batch = order_block(frame, atr, tolerance=0.25)
                else:
                    batch = CATALOGUE[candidate.name](frame, atr)
                spec = data.instrument_spec(symbol)
                last_exit = -1
                for lane, at in enumerate(batch.index):
                    at = int(at)
                    if at <= last_exit:
                        continue
                    direction = int(batch.direction[lane])
                    entry = float(batch.entry[lane])
                    unit = float(batch.unit[lane])
                    outcome = _resolve_one(
                        frame,
                        index=at,
                        direction=direction,
                        entry=entry,
                        unit=unit,
                        ratio=candidate.ratio,
                        same_bar=batch.same_bar,
                    )
                    if outcome is None:
                        continue
                    won, exit_at = outcome
                    last_exit = exit_at
                    spread_price = float(frame["spread"].iloc[at]) * spec.point
                    sized = sizer.size(
                        spec=spec,
                        equity=equity,
                        direction=Direction(direction),
                        entry=entry,
                        sl=entry - direction * unit,
                        tp=entry + direction * unit * candidate.ratio,
                        spread_price=spread_price,
                        risk_pct=2.0,
                        enforce_minimum_rr=False,
                    )
                    cost = sizer.cost_share(spec, unit, spread_price)
                    records.append(
                        {
                            "candidate": candidate.name,
                            "timeframe": candidate.timeframe,
                            "asset": candidate.asset,
                            "ratio": candidate.ratio,
                            "symbol": symbol,
                            "stamp": frame.index[at],
                            "won": won,
                            "net_r": (candidate.ratio if won else -1.0) - cost,
                            "cost_r": cost,
                            "approved": sized.approved,
                            "reason": str(sized.reason),
                            "risk_pct": sized.actual_risk_pct,
                            "volume": sized.volume,
                        }
                    )
    finally:
        data.close_database()
    return records


def report(records: list[dict[str, object]], database: Path) -> None:
    with ResearchDataset(database, read_only=True) as stored:
        days, end_text = stored.window() or (180, "")
    end = pd.Timestamp(end_text)
    start = end - pd.Timedelta(days=days)
    split = start + (end - start) * 0.6

    frame = pd.DataFrame(records)
    for keys, group in frame.groupby(["candidate", "timeframe", "asset", "ratio"]):
        name, timeframe, asset, ratio = keys
        print(f"\n{name} {timeframe} {asset} target={ratio:.2f}R")
        reasons = Counter(group.loc[~group["approved"], "reason"])
        print(
            f"  resolved non-overlapping: {len(group)}; "
            f"approved at EUR 203: {group['approved'].sum()}"
        )
        if reasons:
            summary = ", ".join(f"{reason}={count}" for reason, count in reasons.items())
            print("  blocked: " + summary)
        approved = group[group["approved"]]
        for label, part in (
            ("train", approved[approved["stamp"] < split]),
            ("holdout", approved[approved["stamp"] >= split]),
        ):
            if part.empty:
                print(f"  {label:<7} n=0")
                continue
            print(
                f"  {label:<7} n={len(part):4d}  win={part['won'].mean():6.1%}  "
                f"net={part['net_r'].mean():+.3f}R/trade  "
                f"cost={part['cost_r'].median():.3f}R median"
            )
        if not approved.empty:
            symbols = approved.groupby("symbol").agg(
                n=("net_r", "count"), win=("won", "mean"), net=("net_r", "mean")
            )
            print(
                "  symbols: "
                + ", ".join(
                    f"{symbol} n{int(row['n'])} {row['win']:.0%} {row['net']:+.2f}R"
                    for symbol, row in symbols.iterrows()
                )
            )
            by_month = approved.assign(month=approved["stamp"].dt.strftime("%Y-%m")).groupby(
                "month"
            )["net_r"].agg(["count", "mean"])
            months = ", ".join(
                f"{month} {row['mean']:+.2f}R" for month, row in by_month.iterrows()
            )
            print("  months: " + months)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--equity", type=float, default=203.0)
    args = parser.parse_args()
    report(audit(args.database, args.equity), args.database)


if __name__ == "__main__":
    main()
