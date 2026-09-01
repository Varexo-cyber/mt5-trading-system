"""Chronological audit of session-range breaks and failed breaks on broker data."""

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
from scripts.search_walkforward_section import _phase, _portfolio, _summary

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class SessionChoice:
    range_start: int
    range_end: int
    trade_hours: int
    polarity: int
    stop_share: float
    target_r: float


# Fixed, recognisable market sessions.  These are hypotheses, not hours mined
# one by one from the holdout.
WINDOWS = ((0, 6, 4), (0, 7, 3), (6, 8, 3), (7, 9, 3), (12, 14, 3), (13, 15, 3))


def _signals(frame: pd.DataFrame, choice: SessionChoice) -> list[tuple[int, int, float]]:
    dates = pd.Series(frame.index.normalize(), index=frame.index)
    close = frame["close"].to_numpy(float)
    out: list[tuple[int, int, float]] = []
    for _day, daily in frame.groupby(dates):
        window = daily[
            (daily.index.hour >= choice.range_start) & (daily.index.hour < choice.range_end)
        ]
        after = daily[
            (daily.index.hour >= choice.range_end)
            & (daily.index.hour < choice.range_end + choice.trade_hours)
        ]
        if len(window) < 12 or after.empty:
            continue
        top, bottom = float(window["high"].max()), float(window["low"].min())
        width = top - bottom
        if width <= 0.0:
            continue
        for stamp in after.index:
            at = int(frame.index.get_loc(stamp))
            crossed_up = close[at] > top and close[at - 1] <= top
            crossed_down = close[at] < bottom and close[at - 1] >= bottom
            raw = 1 if crossed_up else -1 if crossed_down else 0
            if raw:
                out.append((at, raw * choice.polarity, width * choice.stop_share))
                break
    return out


def _trades(
    frame: pd.DataFrame,
    signals: list[tuple[int, int, float]],
    mask: np.ndarray,
    *,
    choice: SessionChoice,
    symbol: str,
    sizer: PositionSizer,
) -> list[dict[str, object]]:
    spec = data.instrument_spec(symbol)
    close = frame["close"].to_numpy(float)
    out: list[dict[str, object]] = []
    last_exit = -1
    for at, direction, unit in signals:
        if at <= last_exit or not mask[at] or at >= len(frame) - 49:
            continue
        entry = float(close[at])
        result = _resolve_one(
            frame,
            index=at,
            direction=direction,
            entry=entry,
            unit=unit,
            ratio=choice.target_r,
            same_bar=False,
        )
        if result is None:
            continue
        won, exit_at = result
        last_exit = exit_at
        spread_price = float(frame["spread"].iloc[at]) * spec.point
        sized = sizer.size(
            spec=spec,
            equity=203.0,
            direction=Direction(direction),
            entry=entry,
            sl=entry - direction * unit,
            tp=entry + direction * unit * choice.target_r,
            spread_price=spread_price,
            risk_pct=2.0,
            enforce_minimum_rr=False,
        )
        if not sized.approved:
            continue
        cost = sizer.cost_share(spec, unit, spread_price)
        net = (choice.target_r if won else -1.0) - cost
        out.append(
            {
                "stamp": frame.index[at],
                "exit_stamp": frame.index[exit_at],
                "symbol": symbol,
                "won": won,
                "net_r": net,
                "money": net * sized.actual_risk_money,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--symbols", nargs="*", default=[])
    args = parser.parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    sizer = PositionSizer(settings)
    data.configure_database(args.database)
    try:
        days, end_text = data.database_window() or (180, "")
        end = pd.Timestamp(end_text)
        start = end - pd.Timedelta(days=days)
        train_end = start + (end - start) * 0.50
        validation_end = start + (end - start) * 0.75
        requested = set(args.symbols)
        for symbol in data.every_symbol():
            if requested and symbol not in requested:
                continue
            if data.asset_class(symbol) not in {
                "fx",
                "index",
                "metal",
            } or "M5" not in data.available(symbol):
                continue
            frame = data.load(symbol, "M5")
            candidates = []
            for start_hour, end_hour, trade_hours in WINDOWS:
                for polarity in (1, -1):
                    for stop_share in (0.5, 1.0, 1.5):
                        for target_r in (0.5, 1.0, 1.5):
                            choice = SessionChoice(
                                start_hour, end_hour, trade_hours, polarity, stop_share, target_r
                            )
                            rows = _trades(
                                frame,
                                _signals(frame, choice),
                                _phase(frame.index, train_end, validation_end),
                                choice=choice,
                                symbol=symbol,
                                sizer=sizer,
                            )
                            stats = _summary(rows)
                            if stats[0] >= 20:
                                candidates.append((stats[2], stats[0], choice))
            if not candidates:
                print(f"{symbol}: no executable validation candidate with n>=20")
                continue
            validation_net, validation_n, choice = max(candidates)
            signals = _signals(frame, choice)
            holdout_rows = _trades(
                frame,
                signals,
                _phase(frame.index, validation_end, end + pd.Timedelta(seconds=1)),
                choice=choice,
                symbol=symbol,
                sizer=sizer,
            )
            holdout = _summary(holdout_rows)
            print(
                f"{symbol}: val n={validation_n} net={validation_net:+.3f}R; {choice}; "
                f"HOLDOUT n={holdout[0]} win={holdout[1]:.1%} net={holdout[2]:+.3f}R "
                f"money={holdout[3]:+.2f}"
            )
            if holdout_rows:
                curve = _portfolio(holdout_rows).sort_values("stamp")["net_r"].cumsum()
                print(f"  max drawdown {(curve - curve.cummax()).min():+.2f}R")
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
