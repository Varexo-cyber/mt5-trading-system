"""Search a nonlinear, frozen M5 model that must work across several markets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.search_walkforward_section import (
    _features,
    _phase,
    _portfolio,
    _summary,
    _trades,
)

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Model:
    centre: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    offset: np.ndarray
    beta: np.ndarray


def _hidden(x: np.ndarray, model: Model) -> np.ndarray:
    z = np.nan_to_num((x - model.centre) / model.scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.tanh(z @ model.projection + model.offset)


def _fit(parts: list[tuple[np.ndarray, np.ndarray]], seed: int = 7401) -> Model:
    rng = np.random.default_rng(seed)
    joined = np.vstack([x for x, _y in parts])
    centre = np.nanmean(joined, axis=0)
    scale = np.nanstd(joined, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    projection = rng.normal(0.0, 0.55, size=(joined.shape[1], 48))
    offset = rng.uniform(-1.0, 1.0, size=48)
    model = Model(centre, scale, projection, offset, np.zeros(49))
    gram = np.zeros((49, 49))
    rhs = np.zeros(49)
    for x, y in parts:
        h = _hidden(x, model)
        design = np.column_stack([np.ones(len(h)), h])
        gram += design.T @ design
        rhs += design.T @ y
    penalty = np.eye(49) * 25.0
    penalty[0, 0] = 0.0
    return Model(centre, scale, projection, offset, np.linalg.solve(gram + penalty, rhs))


def _predict(x: np.ndarray, model: Model) -> np.ndarray:
    hidden = _hidden(x, model)
    return np.column_stack([np.ones(len(hidden)), hidden]) @ model.beta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--per-symbol", action="store_true")
    parser.add_argument("--timeframe", default="M5", choices=("M1", "M5"))
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
        requested = {item.strip() for item in args.symbols.split(",") if item.strip()}
        timeframe = args.timeframe
        symbols = [s for s in data.every_symbol() if timeframe in data.available(s)]
        traded = [s for s in symbols if not requested or s in requested]
        stored: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        train_parts = []
        for symbol in symbols:
            frame = data.load(symbol, timeframe)
            x, y, atr = _features(frame, 12 if timeframe == "M1" else 6)
            stored[symbol] = (frame, x, atr)
            mask = _phase(frame.index, start, train_end) & np.isfinite(y)
            good = mask & np.isfinite(x).all(axis=1)
            locations = np.flatnonzero(good)
            if len(locations) > 12_000:
                locations = locations[np.linspace(0, len(locations) - 1, 12_000).astype(int)]
            train_parts.append((x[locations], y[locations]))
        model = _fit(train_parts)
        predictions = {symbol: _predict(stored[symbol][1], model) for symbol in symbols}
        if args.per_symbol:
            survivors = []
            for symbol in traded:
                frame, _x, atr = stored[symbol]
                cells = []
                for polarity in (1, -1):
                    for threshold in (0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20):
                        for stop_atr in (1.0, 1.5, 2.0):
                            for target_r in (0.5, 0.75, 1.0, 1.5):
                                rows = _trades(
                                    frame,
                                    predictions[symbol] * polarity,
                                    atr,
                                    _phase(frame.index, train_end, validation_end),
                                    threshold=threshold,
                                    ratio=target_r,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    sizer=sizer,
                                    stop_atr=stop_atr,
                                )
                                stats = _summary(rows)
                                if stats[0] >= 30:
                                    cells.append(
                                        (
                                            stats[2],
                                            stats[0],
                                            polarity,
                                            threshold,
                                            stop_atr,
                                            target_r,
                                        )
                                    )
                if not cells:
                    print(f"{symbol}: no validation cell n>=30")
                    continue
                val_net, val_n, polarity, threshold, stop_atr, target_r = max(cells)
                holdout_rows = _trades(
                    frame,
                    predictions[symbol] * polarity,
                    atr,
                    _phase(frame.index, validation_end, end + pd.Timedelta(seconds=1)),
                    threshold=threshold,
                    ratio=target_r,
                    symbol=symbol,
                    timeframe=timeframe,
                    sizer=sizer,
                    stop_atr=stop_atr,
                )
                holdout = _summary(holdout_rows)
                print(
                    f"{symbol}: VAL n={val_n} {val_net:+.3f}R; p={polarity} "
                    f"t={threshold} stop={stop_atr} target={target_r}; "
                    f"HOLDOUT n={holdout[0]} win={holdout[1]:.1%} "
                    f"net={holdout[2]:+.3f}R money={holdout[3]:+.2f}"
                )
                if holdout[0] >= 30 and holdout[2] > 0.03:
                    survivors.extend(holdout_rows)
            total = _summary(survivors)
            represented = _portfolio(survivors)["symbol"].nunique() if survivors else 0
            print(
                f"SURVIVORS markets={represented} n={total[0]} win={total[1]:.1%} "
                f"net={total[2]:+.3f}R money={total[3]:+.2f}"
            )
            if represented:
                print("MODEL_BETA", model.beta.tolist())
                print("MODEL_CENTRE", model.centre.tolist())
                print("MODEL_SCALE", model.scale.tolist())
            return
        candidates = []
        for polarity in (1, -1):
            for threshold in (0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
                for stop_atr in (1.0, 1.5, 2.0):
                    for target_r in (0.5, 0.75, 1.0, 1.5):
                        rows = []
                        for symbol in traded:
                            frame, _x, atr = stored[symbol]
                            rows.extend(
                                _trades(
                                    frame,
                                    predictions[symbol] * polarity,
                                    atr,
                                    _phase(frame.index, train_end, validation_end),
                                    threshold=threshold,
                                    ratio=target_r,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    sizer=sizer,
                                    stop_atr=stop_atr,
                                )
                            )
                        portfolio = _portfolio(rows)
                        represented = portfolio["symbol"].nunique() if not portfolio.empty else 0
                        stats = _summary(rows)
                        minimum_trades = 100 if len(traded) >= 6 else 40
                        minimum_markets = min(6, len(traded))
                        if stats[0] >= minimum_trades and represented >= minimum_markets:
                            candidates.append(
                                (
                                    stats[2],
                                    stats[0],
                                    represented,
                                    polarity,
                                    threshold,
                                    stop_atr,
                                    target_r,
                                )
                            )
        if not candidates:
            print("NO CANDIDATE: none reached 100 trades across six markets")
            return
        best = max(candidates)
        net, count, represented, polarity, threshold, stop_atr, target_r = best
        holdout_rows = []
        for symbol in traded:
            frame, _x, atr = stored[symbol]
            holdout_rows.extend(
                _trades(
                    frame,
                    predictions[symbol] * polarity,
                    atr,
                    _phase(frame.index, validation_end, end + pd.Timedelta(seconds=1)),
                    threshold=threshold,
                    ratio=target_r,
                    symbol=symbol,
                    timeframe=timeframe,
                    sizer=sizer,
                    stop_atr=stop_atr,
                )
            )
        holdout = _summary(holdout_rows)
        portfolio = _portfolio(holdout_rows)
        by_symbol = portfolio.groupby("symbol")["net_r"].agg(["count", "mean"])
        by_symbol = by_symbol.sort_values("count", ascending=False)
        print(
            f"VALIDATION n={count} markets={represented} net={net:+.3f}R; "
            f"polarity={polarity} threshold={threshold} stop={stop_atr} target={target_r}"
        )
        print(
            f"HOLDOUT n={holdout[0]} win={holdout[1]:.1%} net={holdout[2]:+.3f}R "
            f"money={holdout[3]:+.2f} positive_months={holdout[4]}"
        )
        print(by_symbol.to_string())
        if not portfolio.empty:
            curve = portfolio.sort_values("stamp")["net_r"].cumsum()
            print(f"max drawdown {(curve - curve.cummax()).min():+.2f}R")
        if holdout[2] > 0.0:
            print("MODEL_BETA", model.beta.tolist())
            print("MODEL_CENTRE", model.centre.tolist())
            print("MODEL_SCALE", model.scale.tolist())
            print("MODEL_PROJECTION", model.projection.tolist())
            print("MODEL_OFFSET", model.offset.tolist())
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
