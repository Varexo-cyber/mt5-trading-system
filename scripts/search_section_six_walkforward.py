"""Adaptive nonlinear walk-forward search for section six, per market."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.search_multimarket_section import _fit, _predict
from scripts.search_walkforward_section import _features, _phase, _portfolio, _summary, _trades

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Choice:
    timeframe: str
    polarity: int
    threshold: float
    stop_atr: float
    target_r: float


def _walkforward_prediction(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    for test_start in pd.date_range(start=start, end=end, freq="30D", inclusive="left"):
        test_end = min(test_start + pd.Timedelta(days=30), end)
        train_start = test_start - pd.Timedelta(days=60)
        train = _phase(frame.index, train_start, test_start) & np.isfinite(y)
        good = train & np.isfinite(x).all(axis=1)
        locations = np.flatnonzero(good)
        if len(locations) < 500:
            continue
        if len(locations) > 12_000:
            locations = locations[np.linspace(0, len(locations) - 1, 12_000).astype(int)]
        model = _fit([(x[locations], y[locations])])
        test = _phase(frame.index, test_start, test_end)
        prediction[test] = _predict(x[test], model)
    return prediction


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
        validation_start = start + pd.Timedelta(days=90)
        holdout_start = end - pd.Timedelta(days=30)
        for symbol in args.symbols:
            candidates = []
            stored = {}
            for timeframe, forecast in (("M5", 6), ("M15", 6), ("M30", 6), ("H1", 6)):
                frame = data.load(symbol, timeframe)
                x, y, atr = _features(frame, forecast)
                prediction = _walkforward_prediction(
                    frame, x, y, start=validation_start, end=end + pd.Timedelta(seconds=1)
                )
                stored[timeframe] = (frame, prediction, atr)
                for polarity in (1, -1):
                    for threshold in (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2):
                        for stop_atr in (1.0, 1.5, 2.0, 2.5):
                            for target_r in (0.75, 1.0, 1.5):
                                rows = _trades(
                                    frame,
                                    prediction * polarity,
                                    atr,
                                    _phase(frame.index, validation_start, holdout_start),
                                    threshold=threshold,
                                    ratio=target_r,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    sizer=sizer,
                                    stop_atr=stop_atr,
                                )
                                stats = _summary(rows)
                                portfolio = _portfolio(rows)
                                if stats[0] < 35 or stats[2] <= 0.03 or portfolio.empty:
                                    continue
                                monthly = (
                                    portfolio.assign(month=portfolio["stamp"].dt.strftime("%Y-%m"))
                                    .groupby("month")["net_r"]
                                    .sum()
                                )
                                if int((monthly > 0.0).sum()) < 2:
                                    continue
                                candidates.append(
                                    (
                                        stats[2],
                                        stats[0],
                                        Choice(
                                            timeframe,
                                            polarity,
                                            threshold,
                                            stop_atr,
                                            target_r,
                                        ),
                                    )
                                )
            if not candidates:
                print(f"{symbol}: NO adaptive validation candidate")
                continue
            selected_net, selected_n, choice = max(candidates, key=lambda item: (item[0], item[1]))
            frame, prediction, atr = stored[choice.timeframe]
            rows = _trades(
                frame,
                prediction * choice.polarity,
                atr,
                _phase(frame.index, holdout_start, end + pd.Timedelta(seconds=1)),
                threshold=choice.threshold,
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
                f"{symbol}: VAL n={selected_n} net={selected_net:+.3f}R; {choice}; "
                f"HOLDOUT n={stats[0]} win={stats[1]:.1%} net={stats[2]:+.3f}R "
                f"money={stats[3]:+.2f} dd={drawdown:+.2f}R"
            )
            if stats[0] >= 20 and stats[2] > 0.03:
                x, y, _fresh_atr = _features(frame, 6)
                # Dump the exact model that faced the holdout.  A model retrained
                # at `end` is intended for the next month and cannot reproduce
                # this report because its test period has not happened yet.
                train = _phase(
                    frame.index,
                    holdout_start - pd.Timedelta(days=60),
                    holdout_start,
                ) & np.isfinite(y)
                good = train & np.isfinite(x).all(axis=1)
                locations = np.flatnonzero(good)
                if len(locations) > 12_000:
                    locations = locations[np.linspace(0, len(locations) - 1, 12_000).astype(int)]
                current = _fit([(x[locations], y[locations])])
                print("HOLDOUT_BETA", current.beta.tolist())
                print("HOLDOUT_CENTRE", current.centre.tolist())
                print("HOLDOUT_SCALE", current.scale.tolist())
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
