"""Load every market and every timeframe into one shape.

THE BARS ARE NOT IN THE REPOSITORY -- about 750 MB. Fetch them once into
`data/research/`:

    mkdir -p data/research/mkt data/research/idx
    B=https://raw.githubusercontent.com/ejtraderLabs/historical-data/main
    for s in AUDJPY AUDUSD EURCHF EURGBP EURJPY EURUSD GBPJPY GBPUSD \
             USDCAD USDCHF USDJPY XAUUSD; do
      for tf in m15 m30 h1 h4; do
        curl -sSL -o "data/research/mkt/$s.$tf.csv" "$B/$s/$s$tf.csv"
      done
    done
    I=https://raw.githubusercontent.com/FutureSharks/financial-data/master/pyfinancialdata/data/stocks/histdata
    for s in SPXUSD JPXJPY GRXEUR ETXEUR; do for y in $(seq 2010 2018); do
      curl -sSL -o "data/research/idx/${s}_$y.csv" "$I/$s/DAT_ASCII_${s}_M1_$y.csv"
    done; done

Then:  python -m scripts.lab.sweep --out sweep.json
       python -m scripts.lab.report --file sweep.json


TWO FEEDS, because no single free source has both.

    indices   SPXUSD JPXJPY GRXEUR ETXEUR   native M1, 2010-2018 (HistData)
    fx/metal  12 pairs incl. XAUUSD          M15..H4, 2012-2022 (HistData)

The index side is native one-minute, so M5 through H4 are resampled from it
and every timeframe on those four instruments is internally consistent. The FX
side ships as separate files per timeframe and is used as delivered.

WHAT IS NOT HERE, and it matters because it is what the owner asked for by
name: US2000 and NAS100. Neither is published as free intraday history
anywhere reachable. SPX500, DAX30, Nikkei225 and EuroStoxx50 stand in for the
equity-index asset class, which is a real substitution and not the same thing
-- a Russell edge is not established by an S&P measurement.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "research"
IDX = ROOT / "idx"
MKT = ROOT / "mkt"
CACHE = ROOT / "cache"

#: HistData index CFDs, native M1.
INDICES = {
    "SPX500": "SPXUSD",
    "NIKKEI": "JPXJPY",
    "DAX40": "GRXEUR",
    "STOXX50": "ETXEUR",
}
FX = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "EURGBP",
    "EURCHF",
)
METAL = ("XAUUSD",)

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4")
_RULE = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h"}
_NATIVE_FX = {"M15": "m15", "M30": "m30", "H1": "h1", "H4": "h4"}

_DATABASE = None


def configure_database(path: str | Path) -> None:
    """Use the broker SQLite archive instead of the legacy public feeds."""

    global _DATABASE
    from backtesting.research_dataset import ResearchDataset

    if _DATABASE is not None:
        _DATABASE.close()
    _DATABASE = ResearchDataset(path, read_only=True)
    load.cache_clear()


def close_database() -> None:
    global _DATABASE
    if _DATABASE is not None:
        _DATABASE.close()
        _DATABASE = None
    load.cache_clear()


def database_window() -> tuple[int, str] | None:
    return _DATABASE.window() if _DATABASE is not None else None


def asset_class(symbol: str) -> str:
    if _DATABASE is not None:
        value = _DATABASE.spec(symbol).asset_class.value
        return "fx" if value == "forex" else value
    if symbol in INDICES:
        return "index"
    if symbol in METAL:
        return "metal"
    return "fx"


def instrument_spec(symbol: str):  # type: ignore[no-untyped-def]
    """Return the captured broker contract when the SQLite source is active."""

    if _DATABASE is None:
        raise RuntimeError("instrument specifications require --database")
    return _DATABASE.spec(symbol)


def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = frame.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def _load_index_m1(code: str) -> pd.DataFrame:
    parts = []
    for path in sorted(IDX.glob(f"{code}_*.csv")):
        chunk = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["stamp", "open", "high", "low", "close", "volume"],
        )
        chunk["stamp"] = pd.to_datetime(chunk["stamp"], format="%Y%m%d %H%M%S", utc=True)
        parts.append(chunk.set_index("stamp"))
    if not parts:
        raise FileNotFoundError(code)
    frame = pd.concat(parts).sort_index()
    return frame[~frame.index.duplicated(keep="first")]


def _load_fx(symbol: str, timeframe: str) -> pd.DataFrame:
    suffix = _NATIVE_FX[timeframe]
    frame = pd.read_csv(MKT / f"{symbol}.{suffix}.csv", parse_dates=["Date"])
    frame = frame.rename(columns=str.lower).rename(columns={"tick_volume": "volume"})
    frame = frame.set_index("date").sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    # The feed stores prices scaled by 100. Scale cannot matter to an
    # ATR-normalised measurement, but it makes every printout readable.
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] / 100.0
    return frame[~frame.index.duplicated(keep="first")]


def available(symbol: str) -> tuple[str, ...]:
    """Which timeframes this instrument actually has."""
    if _DATABASE is not None:
        return tuple(
            row["timeframe"]
            for row in _DATABASE.coverage()
            if row["broker_symbol"] == _DATABASE.spec(symbol).symbol
        )
    if symbol in INDICES:
        return TIMEFRAMES
    return tuple(_NATIVE_FX)


@lru_cache(maxsize=96)
def load(symbol: str, timeframe: str) -> pd.DataFrame:
    """One instrument at one timeframe, cached as a pickle after the first read.

    Parsing 3.3 million lines of semicolon CSV takes about a minute per
    index-year and the sweep reads each frame dozens of times.
    """
    if _DATABASE is not None:
        frame = _DATABASE.load_frame(symbol, timeframe)
        if "volume" not in frame and "tick_volume" in frame:
            frame = frame.assign(volume=frame["tick_volume"])
        return frame
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{symbol}.{timeframe}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    if symbol in INDICES:
        frame = _load_index_m1(INDICES[symbol])
        if timeframe != "M1":
            frame = _resample(frame, _RULE[timeframe])
    else:
        frame = _load_fx(symbol, timeframe)
    frame = frame.astype(dict.fromkeys(("open", "high", "low", "close"), "float64"))
    frame.to_pickle(cached)
    return frame


def atr(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return spans.rolling(period).mean().to_numpy()


def every_symbol() -> tuple[str, ...]:
    if _DATABASE is not None:
        return tuple(_DATABASE.symbols())
    return tuple(INDICES) + FX + METAL
