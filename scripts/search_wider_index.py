"""Test whether wider structural stops make the other indices executable."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.search_walkforward_section import (
    _features,
    _fit_ridge,
    _phase,
    _portfolio,
    _predict,
    _summary,
    _trades,
)

ROOT = Path(__file__).resolve().parent.parent


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
        train_end = start + (end - start) * 0.50
        validation_end = start + (end - start) * 0.75
        symbols = [s for s in data.every_symbol() if data.asset_class(s) == "index"]
        stored = {}
        train_x, train_y = [], []
        for symbol in symbols:
            frame = data.load(symbol, "H1")
            x, y, atr = _features(frame, 12)
            stored[symbol] = (frame, x, atr)
            mask = _phase(frame.index, start, train_end) & np.isfinite(y)
            good = mask & np.isfinite(x).all(axis=1)
            train_x.append(x[good])
            train_y.append(y[good])
        model = _fit_ridge(np.vstack(train_x), np.concatenate(train_y))
        for symbol in symbols:
            if symbol == "SPX500":
                continue
            frame, x, atr = stored[symbol]
            prediction = _predict(x, model) * -1
            candidates = []
            for threshold in (0.03, 0.05, 0.075, 0.10, 0.15):
                for stop_atr in (1.5, 2.0, 2.5, 3.0):
                    for ratio in (0.5, 0.75, 1.0, 1.5):
                        rows = _trades(
                            frame,
                            prediction,
                            atr,
                            _phase(frame.index, train_end, validation_end),
                            threshold=threshold,
                            ratio=ratio,
                            symbol=symbol,
                            timeframe="H1",
                            sizer=sizer,
                            stop_atr=stop_atr,
                        )
                        stats = _summary(rows)
                        if stats[0] >= 30:
                            candidates.append((stats[2], stats[0], threshold, stop_atr, ratio))
            if not candidates:
                print(f"{symbol}: no validation cell with 30 executable trades")
                continue
            net, count, threshold, stop_atr, ratio = max(candidates)
            holdout_rows = _trades(
                frame,
                prediction,
                atr,
                _phase(frame.index, validation_end, end + pd.Timedelta(seconds=1)),
                threshold=threshold,
                ratio=ratio,
                symbol=symbol,
                timeframe="H1",
                sizer=sizer,
                stop_atr=stop_atr,
            )
            holdout = _summary(holdout_rows)
            print(
                f"{symbol}: validation n={count} net={net:+.3f}R; "
                f"choice threshold={threshold} stop={stop_atr}ATR target={ratio}R; "
                f"HOLDOUT n={holdout[0]} win={holdout[1]:.1%} "
                f"net={holdout[2]:+.3f}R money={holdout[3]:+.2f}"
            )
            if holdout_rows:
                curve = _portfolio(holdout_rows).sort_values("stamp")["net_r"].cumsum()
                print(f"  max drawdown {(curve - curve.cummax()).min():+.2f}R")
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
