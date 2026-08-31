"""Search for a genuinely new, cost-aware walk-forward section.

This is deliberately not another candle-pattern zoo.  A small ridge model
learns a direction from continuous, ATR-normalised market state on the oldest
50% of the broker archive.  Signal threshold and target are selected on the
next 25%.  The newest 25% is printed once, after the choice is frozen.

Trades are first-touch resolved, same-bar ambiguity is a loss, historical
spread and configured execution costs are charged, the EUR account must be
able to express the position, positions cannot overlap on one symbol, and the
portfolio has four slots.  A high win rate with negative expectancy therefore
cannot pass by construction.
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

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Choice:
    timeframe: str
    asset: str
    forecast: int
    polarity: int
    threshold: float
    target_r: float


def _features(frame: pd.DataFrame, forecast: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    unit = pd.Series(data.atr(frame), index=frame.index).replace(0.0, np.nan)
    span = (high - low).replace(0.0, np.nan)
    fast = close.ewm(span=8, adjust=False).mean()
    slow = close.ewm(span=32, adjust=False).mean()
    atr_slow = unit.rolling(48).mean()
    volume = frame.get("volume", pd.Series(0.0, index=frame.index)).astype(float)
    volume_base = volume.rolling(48).median().replace(0.0, np.nan)
    hour = frame.index.hour + frame.index.minute / 60.0
    columns = [
        close.diff(1) / unit,
        close.diff(3) / unit,
        close.diff(6) / unit,
        close.diff(12) / unit,
        (fast - slow) / unit,
        (close - fast) / unit,
        (close - open_) / unit,
        span / unit,
        (close - low) / span - 0.5,
        unit / atr_slow - 1.0,
        volume / volume_base - 1.0,
        pd.Series(np.sin(2.0 * np.pi * hour / 24.0), index=frame.index),
        pd.Series(np.cos(2.0 * np.pi * hour / 24.0), index=frame.index),
    ]
    x = np.column_stack([column.to_numpy() for column in columns])
    y = ((close.shift(-forecast) - close) / unit).clip(-3.0, 3.0).to_numpy()
    atr = unit.to_numpy()
    return x, y, atr


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    z = np.nan_to_num((x - centre) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta, centre, scale


def _predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    beta, centre, scale = model
    z = np.nan_to_num((x - centre) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([np.ones(len(z)), z]) @ beta


def _phase(stamps: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return (stamps >= start) & (stamps < end)


def _trades(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    atr: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float,
    ratio: float,
    symbol: str,
    timeframe: str,
    sizer: PositionSizer,
) -> list[dict[str, object]]:
    spec = data.instrument_spec(symbol)
    close = frame["close"].to_numpy(dtype=float)
    picked = np.flatnonzero(
        mask
        & np.isfinite(prediction)
        & np.isfinite(atr)
        & (atr > 0.0)
        & (np.abs(prediction) >= threshold)
    )
    out: list[dict[str, object]] = []
    last_exit = -1
    horizon = {"M1": 48, "M5": 36, "M15": 24, "M30": 18, "H1": 12}[timeframe]
    for at in picked:
        if at <= last_exit or at >= len(frame) - horizon - 1:
            continue
        direction = 1 if prediction[at] > 0.0 else -1
        entry = float(close[at])
        unit = float(atr[at])
        resolved = _resolve_one(
            frame,
            index=int(at),
            direction=direction,
            entry=entry,
            unit=unit,
            ratio=ratio,
            same_bar=False,
        )
        if resolved is None:
            continue
        won, exit_at = resolved
        last_exit = exit_at
        spread_price = float(frame["spread"].iloc[at]) * spec.point
        sized = sizer.size(
            spec=spec,
            equity=203.0,
            direction=Direction(direction),
            entry=entry,
            sl=entry - direction * unit,
            tp=entry + direction * unit * ratio,
            spread_price=spread_price,
            risk_pct=2.0,
            enforce_minimum_rr=False,
        )
        if not sized.approved:
            continue
        cost = sizer.cost_share(spec, unit, spread_price)
        out.append(
            {
                "stamp": frame.index[at],
                "exit_stamp": frame.index[exit_at],
                "symbol": symbol,
                "won": won,
                "net_r": (ratio if won else -1.0) - cost,
                "money": ((ratio if won else -1.0) - cost) * sized.actual_risk_money,
            }
        )
    return out


def _portfolio(rows: list[dict[str, object]], slots: int = 4) -> pd.DataFrame:
    open_until: dict[str, pd.Timestamp] = {}
    accepted: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["stamp"]):
        stamp = pd.Timestamp(row["stamp"])
        open_until = {s: until for s, until in open_until.items() if until > stamp}
        symbol = str(row["symbol"])
        if symbol in open_until or len(open_until) >= slots:
            continue
        accepted.append(row)
        open_until[symbol] = pd.Timestamp(row["exit_stamp"])
    return pd.DataFrame(accepted)


def _summary(rows: list[dict[str, object]]) -> tuple[int, float, float, float, int]:
    frame = _portfolio(rows)
    if frame.empty:
        return 0, 0.0, 0.0, 0.0, 0
    monthly = (
        frame.assign(month=frame["stamp"].dt.strftime("%Y-%m")).groupby("month")["net_r"].sum()
    )
    return (
        len(frame),
        float(frame["won"].mean()),
        float(frame["net_r"].mean()),
        float(frame["money"].sum()),
        int((monthly > 0.0).sum()),
    )


def search(database: Path) -> int:
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    sizer = PositionSizer(settings)
    data.configure_database(database)
    try:
        window = data.database_window()
        if window is None:
            raise SystemExit("database has no coverage window")
        days, end_text = window
        end = pd.Timestamp(end_text)
        start = end - pd.Timedelta(days=days)
        train_end = start + (end - start) * 0.50
        validation_end = start + (end - start) * 0.75
        symbols = data.every_symbol()
        candidates: list[tuple[Choice, tuple[int, float, float, float, int], object]] = []
        cached: dict[tuple[str, str, int], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        for timeframe in ("M5", "M15", "M30", "H1"):
            for asset in ("fx", "index", "metal"):
                members = [
                    symbol
                    for symbol in symbols
                    if data.asset_class(symbol) == asset and timeframe in data.available(symbol)
                ]
                if not members:
                    continue
                for forecast in (3, 6, 12):
                    train_x, train_y = [], []
                    for symbol in members:
                        frame = data.load(symbol, timeframe)
                        x, y, atr = _features(frame, forecast)
                        cached[(symbol, timeframe, forecast)] = (frame, x, atr)
                        mask = _phase(frame.index, start, train_end) & np.isfinite(y)
                        good = mask & np.isfinite(x).all(axis=1)
                        train_x.append(x[good])
                        train_y.append(y[good])
                    if not train_x or not sum(len(part) for part in train_x):
                        continue
                    model = _fit_ridge(np.vstack(train_x), np.concatenate(train_y))
                    predictions = {
                        symbol: _predict(cached[(symbol, timeframe, forecast)][1], model)
                        for symbol in members
                    }
                    for polarity in (1, -1):
                        for threshold in (0.03, 0.05, 0.075, 0.10, 0.15, 0.25):
                            for ratio in (0.5, 0.75, 1.0, 1.5):
                                rows: list[dict[str, object]] = []
                                for symbol in members:
                                    frame, _x, atr = cached[(symbol, timeframe, forecast)]
                                    rows.extend(
                                        _trades(
                                            frame,
                                            predictions[symbol] * polarity,
                                            atr,
                                            _phase(frame.index, train_end, validation_end),
                                            threshold=threshold,
                                            ratio=ratio,
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            sizer=sizer,
                                        )
                                    )
                                stats = _summary(rows)
                                choice = Choice(
                                    timeframe,
                                    asset,
                                    forecast,
                                    polarity,
                                    threshold,
                                    ratio,
                                )
                                candidates.append((choice, stats, (members, predictions, model)))
        eligible = [item for item in candidates if item[1][0] >= 75 and item[1][2] > 0.03]
        eligible.sort(key=lambda item: (item[1][2], item[1][0]), reverse=True)
        print(
            f"searched {len(candidates)} validation cells; {len(eligible)} cleared n>=75 and +0.03R"
        )
        if not eligible:
            ranked = sorted(
                (item for item in candidates if item[1][0] >= 75),
                key=lambda item: (item[1][2], item[1][0]),
                reverse=True,
            )
            for choice, stats, _payload in ranked[:10]:
                print(
                    f"  near {choice}: n={stats[0]} win={stats[1]:.1%} "
                    f"net={stats[2]:+.3f}R money={stats[3]:+.2f}"
                )
            return 1
        choice, validation, payload = eligible[0]
        members, predictions, model = payload
        holdout_rows: list[dict[str, object]] = []
        for symbol in members:
            frame, _x, atr = cached[(symbol, choice.timeframe, choice.forecast)]
            holdout_rows.extend(
                _trades(
                    frame,
                    predictions[symbol] * choice.polarity,
                    atr,
                    _phase(frame.index, validation_end, end + pd.Timedelta(seconds=1)),
                    threshold=choice.threshold,
                    ratio=choice.target_r,
                    symbol=symbol,
                    timeframe=choice.timeframe,
                    sizer=sizer,
                )
            )
        holdout = _summary(holdout_rows)
        print(f"chosen on validation: {choice}")
        print(
            f"validation n={validation[0]} win={validation[1]:.1%} "
            f"net={validation[2]:+.3f}R/trade money={validation[3]:+.2f}"
        )
        print(
            f"HOLDOUT    n={holdout[0]} win={holdout[1]:.1%} "
            f"net={holdout[2]:+.3f}R/trade money={holdout[3]:+.2f}"
        )
        holdout_frame = _portfolio(holdout_rows)
        if not holdout_frame.empty:
            by_symbol = holdout_frame.groupby("symbol").agg(
                trades=("net_r", "count"),
                win=("won", "mean"),
                net=("net_r", "sum"),
                money=("money", "sum"),
            )
            print("by symbol")
            for symbol, row in by_symbol.iterrows():
                print(
                    f"  {symbol:<8} n={int(row['trades']):3d} win={row['win']:.1%} "
                    f"net={row['net']:+.2f}R money={row['money']:+.2f}"
                )
            by_month = (
                holdout_frame.assign(month=holdout_frame["stamp"].dt.strftime("%Y-%m"))
                .groupby("month")["net_r"]
                .agg(["count", "sum"])
            )
            print("by month")
            for month, row in by_month.iterrows():
                print(f"  {month} n={int(row['count']):3d} net={row['sum']:+.2f}R")
            curve = holdout_frame.sort_values("stamp")["net_r"].cumsum()
            drawdown = curve - curve.cummax()
            print(f"max drawdown {drawdown.min():+.2f}R")
        beta, centre, scale = model
        print("model beta=" + repr(beta.tolist()))
        print("model centre=" + repr(centre.tolist()))
        print("model scale=" + repr(scale.tolist()))
        return 0 if holdout[0] >= 75 and holdout[2] > 0.03 else 2
    finally:
        data.close_database()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(search(args.database))


if __name__ == "__main__":
    main()
