"""Test fixed session drifts as a simple, falsifiable section-six family."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.search_section_six_rules import _atr, _stability
from scripts.search_walkforward_section import _phase, _portfolio, _summary, _trades

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Choice:
    timeframe: str
    hour: int
    minute: int
    direction: int
    stop_atr: float
    target_r: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    args = parser.parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    sizer = PositionSizer(settings)
    data.configure_database(args.database)
    try:
        days, end_text = data.database_window() or (180, "")
        end = pd.Timestamp(end_text)
        start = end - pd.Timedelta(days=days)
        selection_end = start + (end - start) * 0.75
        for symbol in args.symbols:
            candidates = []
            frames = {tf: data.load(symbol, tf) for tf in ("M5", "M15")}
            for timeframe, frame in frames.items():
                atr = _atr(frame).to_numpy(float)
                minutes = (0, 30)
                for hour in range(24):
                    for minute in minutes:
                        for direction in (1, -1):
                            prediction = np.full(len(frame), np.nan)
                            trigger = (frame.index.hour == hour) & (frame.index.minute == minute)
                            prediction[trigger] = float(direction)
                            for stop_atr in (1.0, 1.5, 2.0, 2.5):
                                for target_r in (0.75, 1.0, 1.5):
                                    rows = _trades(
                                        frame,
                                        prediction,
                                        atr,
                                        _phase(frame.index, start, selection_end),
                                        threshold=0.5,
                                        ratio=target_r,
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        sizer=sizer,
                                        stop_atr=stop_atr,
                                    )
                                    stats = _summary(rows)
                                    if stats[0] < 55 or stats[2] <= 0.02:
                                        continue
                                    conservative, months = _stability(rows)
                                    if months >= 3:
                                        candidates.append(
                                            (
                                                conservative,
                                                stats[2],
                                                stats[0],
                                                Choice(
                                                    timeframe,
                                                    hour,
                                                    minute,
                                                    direction,
                                                    stop_atr,
                                                    target_r,
                                                ),
                                            )
                                        )
            if not candidates:
                print(f"{symbol}: NO stable clock candidate")
                continue
            conservative, selected_net, selected_n, choice = max(candidates)
            frame = frames[choice.timeframe]
            atr = _atr(frame).to_numpy(float)
            prediction = np.full(len(frame), np.nan)
            trigger = (frame.index.hour == choice.hour) & (frame.index.minute == choice.minute)
            prediction[trigger] = float(choice.direction)
            rows = _trades(
                frame,
                prediction,
                atr,
                _phase(frame.index, selection_end, end + pd.Timedelta(seconds=1)),
                threshold=0.5,
                ratio=choice.target_r,
                symbol=symbol,
                timeframe=choice.timeframe,
                sizer=sizer,
                stop_atr=choice.stop_atr,
            )
            stats = _summary(rows)
            curve = _portfolio(rows).sort_values("stamp")["net_r"].cumsum()
            drawdown = float((curve - curve.cummax()).min()) if not curve.empty else 0.0
            print(
                f"{symbol}: SELECT n={selected_n} net={selected_net:+.3f}R "
                f"lower={conservative:+.3f}; {choice}; HOLDOUT n={stats[0]} "
                f"win={stats[1]:.1%} net={stats[2]:+.3f}R money={stats[3]:+.2f} "
                f"dd={drawdown:+.2f}R"
            )
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
