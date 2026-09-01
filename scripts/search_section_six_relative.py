"""Chronological relative-index search for the missing section-six markets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_settings
from risk.position_sizer import PositionSizer
from scripts.lab import data
from scripts.search_section_six_rules import SESSIONS, _atr, _stability
from scripts.search_walkforward_section import _phase, _portfolio, _summary, _trades

ROOT = Path(__file__).resolve().parent.parent
INDEX_SYMBOLS = ("GER40", "US30", "NDX100", "SPX500")


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


def _readings(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    lookback: int,
    family: str,
) -> np.ndarray:
    own = frames[symbol]
    index = own.index
    own_close = own["close"].astype(float)
    own_bp = (own_close / own_close.shift(lookback) - 1.0) * 10_000.0
    peers = []
    for peer, frame in frames.items():
        if peer == symbol:
            continue
        close = frame["close"].astype(float).reindex(index).ffill(limit=2)
        peers.append((close / close.shift(lookback) - 1.0) * 10_000.0)
    basket = pd.concat(peers, axis=1).median(axis=1)
    unit_bp = _atr(own) / own_close * 10_000.0
    if family == "gap_close":
        score = (basket - own_bp) / unit_bp
    elif family == "relative_continue":
        score = (own_bp - basket) / unit_bp
    elif family == "basket_momentum":
        score = basket / unit_bp
    elif family == "lag_catchup":
        gap = basket - own_bp
        score = gap.where((gap > 0.0) == (basket > 0.0)) / unit_bp
    else:
        raise ValueError(family)
    return score.to_numpy(float)


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
        holdout_start = start + (end - start) * 0.75
        for symbol in args.symbols:
            candidates = []
            stored = {}
            for timeframe in ("M5", "M15"):
                frames = {member: data.load(member, timeframe) for member in INDEX_SYMBOLS}
                frame = frames[symbol]
                atr = _atr(frame).to_numpy(float)
                for family in (
                    "gap_close",
                    "relative_continue",
                    "basket_momentum",
                    "lag_catchup",
                ):
                    for lookback in (1, 3, 6, 12):
                        base = _readings(symbol, frames, lookback, family)
                        stored[(timeframe, family, lookback)] = (frame, base, atr)
                        for session, hours in SESSIONS.items():
                            prediction = base.copy()
                            prediction[~frame.index.hour.isin(hours)] = np.nan
                            for polarity in (1, -1):
                                for threshold in (0.25, 0.5, 0.75, 1.0, 1.5):
                                    for stop_atr in (1.0, 1.5, 2.0, 2.5):
                                        for target_r in (0.75, 1.0, 1.5):
                                            rows = _trades(
                                                frame,
                                                prediction * polarity,
                                                atr,
                                                _phase(frame.index, start, holdout_start),
                                                threshold=threshold,
                                                ratio=target_r,
                                                symbol=symbol,
                                                timeframe=timeframe,
                                                sizer=sizer,
                                                stop_atr=stop_atr,
                                            )
                                            stats = _summary(rows)
                                            if stats[0] < 45 or stats[2] <= 0.03:
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
                                                            family,
                                                            lookback,
                                                            session,
                                                            polarity,
                                                            threshold,
                                                            stop_atr,
                                                            target_r,
                                                        ),
                                                    )
                                                )
            if not candidates:
                print(f"{symbol}: NO stable relative candidate")
                continue
            conservative, selected_net, selected_n, choice = max(
                candidates, key=lambda item: (item[0], item[1])
            )
            frame, base, atr = stored[(choice.timeframe, choice.family, choice.lookback)]
            prediction = base.copy()
            prediction[~frame.index.hour.isin(SESSIONS[choice.session])] = np.nan
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
                f"{symbol}: SELECT n={selected_n} net={selected_net:+.3f}R "
                f"lower={conservative:+.3f}; {choice}; HOLDOUT n={stats[0]} "
                f"win={stats[1]:.1%} net={stats[2]:+.3f}R money={stats[3]:+.2f} "
                f"dd={drawdown:+.2f}R"
            )
    finally:
        data.close_database()


if __name__ == "__main__":
    main()
