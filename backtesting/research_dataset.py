"""One durable, queryable source of broker bars for repeated research.

CSV-per-clock made every new hypothesis depend on a fresh MT5 fetch and made
it easy to accidentally combine different capture windows.  This store keeps
bars, contract specifications and the configuration snapshot in one SQLite
file.  SQLite is deliberately used instead of an optional parquet dependency:
the live Windows environment already has it, the file is portable, and a
partly completed capture remains readable and resumable after interruption.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.instrument import InstrumentSpec
from core.types import Timeframe

SCHEMA_VERSION = 1
BAR_COLUMNS = ("open", "high", "low", "close", "tick_volume", "spread", "real_volume")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS instruments (
    canonical_symbol TEXT PRIMARY KEY,
    broker_symbol TEXT NOT NULL UNIQUE,
    asset_class TEXT NOT NULL,
    spec_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bars (
    broker_symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    time_utc INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    tick_volume INTEGER NOT NULL,
    spread INTEGER NOT NULL,
    real_volume INTEGER NOT NULL,
    PRIMARY KEY (broker_symbol, timeframe, time_utc)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS coverage (
    broker_symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    requested_from_utc INTEGER NOT NULL,
    requested_to_utc INTEGER NOT NULL,
    first_bar_utc INTEGER NOT NULL,
    last_bar_utc INTEGER NOT NULL,
    bars INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    PRIMARY KEY (broker_symbol, timeframe)
) WITHOUT ROWID;
"""


class ResearchDataset:
    """Read and extend a single broker-history database."""

    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        self.path = Path(path)
        if read_only:
            self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.executescript(_SCHEMA)
            self.set_metadata("schema_version", SCHEMA_VERSION)
        self.connection.row_factory = sqlite3.Row

    def __enter__(self) -> ResearchDataset:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def set_metadata(self, key: str, value: object) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        with self.connection:
            self.connection.execute(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                (key, encoded),
            )

    def metadata(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value_json FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else json.loads(str(row["value_json"]))

    def put_instrument(self, canonical_symbol: str, spec: InstrumentSpec) -> None:
        payload = {
            "symbol": spec.symbol,
            "digits": spec.digits,
            "point": spec.point,
            "tick_size": spec.tick_size,
            "tick_value": spec.tick_value,
            "contract_size": spec.contract_size,
            "volume_min": spec.volume_min,
            "volume_max": spec.volume_max,
            "volume_step": spec.volume_step,
            "stops_level": spec.stops_level,
            "freeze_level": spec.freeze_level,
            "currency_base": spec.currency_base,
            "currency_profit": spec.currency_profit,
            "currency_margin": spec.currency_margin,
            "filling_mode_mask": spec.filling_mode_mask,
            "trade_mode": spec.trade_mode,
            "is_forex": spec.is_forex,
            "path": spec.path,
            "description": spec.description,
            "asset_class": spec.asset_class.value,
        }
        with self.connection:
            self.connection.execute(
                "INSERT INTO instruments(canonical_symbol, broker_symbol, asset_class, spec_json) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(canonical_symbol) DO UPDATE SET "
                "broker_symbol=excluded.broker_symbol, asset_class=excluded.asset_class, "
                "spec_json=excluded.spec_json",
                (
                    canonical_symbol,
                    spec.symbol,
                    spec.asset_class.value,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )

    def put_frame(
        self,
        symbol: str,
        timeframe: Timeframe,
        frame: pd.DataFrame,
        *,
        requested_from: pd.Timestamp,
        requested_to: pd.Timestamp,
        captured_at: str,
    ) -> None:
        """Atomically replace overlapping bars and record honest coverage."""

        clean = _validated(frame, symbol, timeframe)
        first = int(clean.index[0].timestamp())
        last = int(clean.index[-1].timestamp())
        start = int(requested_from.timestamp())
        end = int(requested_to.timestamp())

        def records():  # type: ignore[no-untyped-def]
            for stamp, row in clean.iterrows():
                yield (
                    symbol,
                    timeframe.value,
                    int(stamp.timestamp()),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["tick_volume"]),
                    int(row["spread"]),
                    int(row["real_volume"]),
                )

        with self.connection:
            self.connection.execute(
                "DELETE FROM bars WHERE broker_symbol=? AND timeframe=? "
                "AND time_utc BETWEEN ? AND ?",
                (symbol, timeframe.value, first, last),
            )
            self.connection.executemany(
                "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records()
            )
            total = self.connection.execute(
                "SELECT COUNT(*) FROM bars WHERE broker_symbol=? AND timeframe=?",
                (symbol, timeframe.value),
            ).fetchone()[0]
            bounds = self.connection.execute(
                "SELECT MIN(time_utc), MAX(time_utc) FROM bars "
                "WHERE broker_symbol=? AND timeframe=?",
                (symbol, timeframe.value),
            ).fetchone()
            self.connection.execute(
                "INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(broker_symbol, timeframe) DO UPDATE SET "
                "requested_from_utc=excluded.requested_from_utc, "
                "requested_to_utc=excluded.requested_to_utc, "
                "first_bar_utc=excluded.first_bar_utc, last_bar_utc=excluded.last_bar_utc, "
                "bars=excluded.bars, captured_at=excluded.captured_at",
                (
                    symbol,
                    timeframe.value,
                    start,
                    end,
                    int(bounds[0]),
                    int(bounds[1]),
                    int(total),
                    captured_at,
                ),
            )

    def load_frame(self, symbol: str, timeframe: Timeframe | str) -> pd.DataFrame:
        """Load a research-ready UTC frame by canonical or broker symbol."""

        tf = Timeframe.parse(timeframe)
        row = self.connection.execute(
            "SELECT broker_symbol FROM instruments WHERE canonical_symbol=? "
            "OR broker_symbol=? LIMIT 1",
            (symbol, symbol),
        ).fetchone()
        broker_symbol = symbol if row is None else str(row["broker_symbol"])
        rows = self.connection.execute(
            "SELECT time_utc, open, high, low, close, tick_volume, spread, real_volume "
            "FROM bars WHERE broker_symbol=? AND timeframe=? ORDER BY time_utc",
            (broker_symbol, tf.value),
        ).fetchall()
        if not rows:
            raise KeyError(f"no {broker_symbol} {tf.value} bars in {self.path}")
        frame = pd.DataFrame.from_records(
            [tuple(item) for item in rows], columns=("time", *BAR_COLUMNS)
        )
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        return frame.set_index("time")

    def coverage(self) -> list[Mapping[str, object]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM coverage ORDER BY broker_symbol, timeframe"
            ).fetchall()
        ]


def _validated(frame: pd.DataFrame, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
    missing = set(BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} {timeframe.value}: missing columns {sorted(missing)}")
    clean = frame.loc[:, BAR_COLUMNS].copy().sort_index()
    if clean.empty:
        raise ValueError(f"{symbol} {timeframe.value}: no bars")
    if clean.index.tz is None:
        raise ValueError(f"{symbol} {timeframe.value}: timestamps are not timezone-aware")
    if clean.index.has_duplicates:
        raise ValueError(f"{symbol} {timeframe.value}: duplicate timestamps")
    prices = clean[["open", "high", "low", "close"]]
    if prices.isna().to_numpy().any() or (prices <= 0).to_numpy().any():
        raise ValueError(f"{symbol} {timeframe.value}: invalid OHLC values")
    if bool((clean["high"] < prices[["open", "close"]].max(axis=1)).any()):
        raise ValueError(f"{symbol} {timeframe.value}: high below open/close")
    if bool((clean["low"] > prices[["open", "close"]].min(axis=1)).any()):
        raise ValueError(f"{symbol} {timeframe.value}: low above open/close")
    return clean
