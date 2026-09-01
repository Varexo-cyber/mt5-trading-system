"""Chronological per-market search for a replacement section six.

The search deliberately uses recognisable rule families rather than fitting a
large model to the holdout.  Parameters are selected on the oldest 75% and are
printed once on the untouched newest 25%, with the real small-account sizer and
recorded spread charged on every trade.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.search_walkforward_section import _phase, _portfolio, _summary, _trades

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Choice:
    timeframe: str
    family: str
    lookback: int
    session: str
    polarity: int
    threshold: float
    stop_atr: float
    target_r: float


SESSIONS = {
    "all": tuple(range(24)),
    "london": tuple(range(6, 11)),
    "overlap": tuple(range(12, 17)),
}


def _atr(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return spans.rolling(14).mean()


def _reading(frame: pd.DataFrame, family: str, lookback: int) -> np.ndarray:
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    unit = _atr(frame).replace(0.0, np.nan)
    returns = close.diff()
    if family == "momentum":
        score = (close - close.shift(lookback)) / unit
    elif family == "ema":
        fast = close.ewm(span=lookback, adjust=False).mean()
        slow = close.ewm(span=lookback * 4, adjust=False).mean()
        score = (fast - slow) / unit
    elif family == "mean_reversion":
        mean = close.rolling(lookback).mean()
        score = (close - mean) / unit
    elif family == "range":
        top = high.shift(1).rolling(lookback).max()
        bottom = low.shift(1).rolling(lookback).min()
        score = (close - (top + bottom) / 2.0) / unit
    elif family == "semivariance":
        positive = returns.clip(lower=0.0).rolling(lookback).sum()
        negative = (-returns.clip(upper=0.0)).rolling(lookback).sum()
        score = (positive - negative) / unit
    elif family == "impulse":
        body = close - open_
        average = body.abs().rolling(lookback).mean().replace(0.0, np.nan)
        score = body / average
    else:
        raise ValueError(f"unknown family {family}")
    return score.to_numpy(float)


def _stability(rows: list[dict[str, object]]) -> tuple[float, int]:
    result = _portfolio(rows)
    if result.empty:
        return -999.0, 0
    values = result["net_r"].to_numpy(float)
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 9.0
    )
    months = (
        result.assign(month=result["stamp"].dt.strftime("%Y-%m")).groupby("month")["net_r"].sum()
    )
    return float(np.mean(values) - 0.5 * standard_error), int((months > 0.0).sum())


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
            candidates: list[tuple[float, float, int, Choice, list[dict[str, object]]]] = []
            cache: dict[tuple[str, str, int], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
            for timeframe in ("M5", "M15"):
                if timeframe not in data.available(symbol):
                    continue
                frame = data.load(symbol, timeframe)
                atr = _atr(frame).to_numpy(float)
                for family in (
                    "momentum",
                    "ema",
                    "mean_reversion",
                    "range",
                    "semivariance",
                    "impulse",
                ):
                    for lookback in (6, 12):
                        prediction = _reading(frame, family, lookback)
                        cache[(timeframe, family, lookback)] = (frame, prediction, atr)
                        for session, hours in SESSIONS.items():
                            session_prediction = prediction.copy()
                            session_prediction[~frame.index.hour.isin(hours)] = np.nan
                            for polarity in (1, -1):
                                for threshold in (0.5, 1.0, 1.5):
                                    for stop_atr in (1.5, 2.5):
                                        for target_r in (1.0, 1.5):
                                            rows = _trades(
                                                frame,
                                                session_prediction * polarity,
                                                atr,
                                                _phase(frame.index, start, selection_end),
                                                threshold=threshold,
                                                ratio=target_r,
                                                symbol=symbol,
                                                timeframe=timeframe,
                                                sizer=sizer,
                                                stop_atr=stop_atr,
                                            )
                                            stats = _summary(rows)
                                            if stats[0] < 40 or stats[2] <= 0.02:
                                                continue
                                            conservative, positive_months = _stability(rows)
                                            if positive_months < 3:
                                                continue
                                            candidates.append(
                                                (
                                                    conservative,
                                                    stats[2],
                                                    stats[0],
                                                    Choice(
                                                        timeframe,
                                                        family,
                                                        lookback,
                                                        session,
                                                        polarity,
                                                        threshold,
                                                        stop_atr,
                                                        target_r,
                                                    ),
                                                    rows,
                                                )
                                            )
            if not candidates:
                print(f"{symbol}: NO stable selection candidate")
                continue
            conservative, selected_net, selected_n, choice, _rows = max(candidates)
            frame, prediction, atr = cache[(choice.timeframe, choice.family, choice.lookback)]
            session_prediction = prediction.copy()
            session_prediction[~frame.index.hour.isin(SESSIONS[choice.session])] = np.nan
            holdout_rows = _trades(
                frame,
                session_prediction * choice.polarity,
                atr,
                _phase(frame.index, selection_end, end + pd.Timedelta(seconds=1)),
                threshold=choice.threshold,
                ratio=choice.target_r,
                symbol=symbol,
                timeframe=choice.timeframe,
                sizer=sizer,
                stop_atr=choice.stop_atr,
            )
            holdout = _summary(holdout_rows)
            curve = _portfolio(holdout_rows).sort_values("stamp")["net_r"].cumsum()
            drawdown = float((curve - curve.cummax()).min()) if not curve.empty else 0.0
            print(
                f"{symbol}: SELECT n={selected_n} net={selected_net:+.3f}R "
                f"lower={conservative:+.3f}; {choice}; HOLDOUT n={holdout[0]} "
                f"win={holdout[1]:.1%} net={holdout[2]:+.3f}R "
                f"money={holdout[3]:+.2f} dd={drawdown:+.2f}R"
            )
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
