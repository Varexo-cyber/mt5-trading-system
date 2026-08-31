"""Bars on disk, so a measurement never pays for the same fetch twice.

    from backtesting.history_store import HistoryStore

    store = HistoryStore(Path("data/history"))
    frame = store.frame("EURUSD.i", Timeframe.M15)

WHY. Every run of `dry_run_sections` pulls its own history out of MT5. A
180-day sweep over sixteen markets and six clocks is roughly four million bars,
and the terminal hands them over one bounded chunk at a time -- the run that
measured itself reported the fetch at 0.2 minutes against 8.3 of compute, but
that was a fourteen-day window with M1 on five markets. At 180 days it is the
larger half, it is paid again on every re-run, and it requires MT5 to be open
and logged in on the machine doing the measuring.

Bars do not change. A closed M15 candle from June is the same object in
September. So they are fetched once and kept.

FORMAT: numpy `.npz`, one file per (symbol, timeframe), compressed.

Not Parquet, which would be the obvious choice and is not installed -- adding
pyarrow to a Python 3.14 VPS for this is a bigger risk than the problem. Not
pickle either: a pandas upgrade would silently invalidate the whole store, and
a cache that breaks on upgrade is worse than no cache. `.npz` holds plain
arrays with their names beside them and will still read in five years.

The index is stored as int64 nanoseconds UTC and every column is stored under
its own name, so the frame that comes back is the frame that went in --
including `spread`, which the cost model reads and which a lossy round trip
would quietly zero.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.instrument import AssetClass, InstrumentSpec
from core.types import Timeframe

#: Bumped when the on-disk layout changes in a way that makes old files
#: unreadable. A store written by an older version is reported and ignored
#: rather than half-read.
LAYOUT_VERSION = 1


def _safe(symbol: str) -> str:
    """A symbol as a directory name. `EURUSD.i` and `US30.cash` both contain
    characters that are legal on one platform and not another."""
    return "".join(ch if ch.isalnum() else "_" for ch in symbol)


class HistoryStore:
    """Read and write cached bars under one directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- layout ------------------------------------------------------------

    def _dir(self, symbol: str) -> Path:
        return self.root / _safe(symbol)

    def path(self, symbol: str, timeframe: Timeframe) -> Path:
        return self._dir(symbol) / f"{timeframe.value}.npz"

    def has(self, symbol: str, timeframe: Timeframe) -> bool:
        return self.path(symbol, timeframe).exists()

    def symbols(self) -> list[str]:
        """Every symbol with a spec on disk, in the broker's own spelling."""
        return sorted(self._manifest().get("symbols", {}))

    def timeframes(self, symbol: str) -> list[str]:
        directory = self._dir(symbol)
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.npz"))

    # -- bars --------------------------------------------------------------

    def write(self, symbol: str, timeframe: Timeframe, frame: pd.DataFrame) -> int:
        """Store one frame. Returns the number of bars written."""
        if frame.empty:
            raise ValueError(f"{symbol} {timeframe.value}: refusing to store an empty frame")
        path = self.path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        index = pd.DatetimeIndex(frame.index)
        if index.tz is None:
            index = index.tz_localize(UTC)
        # NANOSECONDS, EXPLICITLY. `DatetimeIndex.asi8` returns the index's
        # own unit, and pandas 2 hands out microsecond-unit indexes from
        # `date_range` -- so writing asi8 and reading it back as
        # `datetime64[ns]` moved every bar to 1970. Pinning the unit on the way
        # out is the only version-proof form; the unit is stored beside it so a
        # future reader never has to assume.
        stamps = index.tz_convert(UTC)
        stamps = stamps.as_unit("ns") if hasattr(stamps, "as_unit") else stamps
        arrays: dict[str, np.ndarray] = {
            "__time__": np.asarray(stamps.asi8, dtype="int64"),
            "__unit__": np.array(["ns"], dtype=object),
            "__columns__": np.array(list(frame.columns), dtype=object),
        }
        for column in frame.columns:
            arrays[f"col_{column}"] = frame[column].to_numpy()
        np.savez_compressed(path, **arrays)
        return len(frame)

    def frame(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """The stored frame, optionally sliced to a window.

        Raises rather than returning empty when nothing is stored: a caller
        that gets a silent empty frame reports "no setups", and this project
        has already shipped that confusion under six other names.
        """
        path = self.path(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"{symbol} {timeframe.value} is not in the store at {self.root}. "
                f"Run fetch_history.py to fill it."
            )
        with np.load(path, allow_pickle=True) as data:
            unit = str(data["__unit__"][0]) if "__unit__" in data else "ns"
            index = pd.DatetimeIndex(pd.to_datetime(data["__time__"], unit=unit, utc=True))
            columns = [str(name) for name in data["__columns__"]]
            frame = pd.DataFrame({name: data[f"col_{name}"] for name in columns}, index=index)
        frame.index.name = "time"
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        return frame

    # -- specs -------------------------------------------------------------

    def write_spec(self, spec: InstrumentSpec) -> None:
        """Keep the contract specification beside the bars.

        WITHOUT THIS THE STORE IS USELESS ON ITS OWN. Sizing needs
        `volume_min`, `tick_value`, `point` and the rest, and those come from
        `mt5.symbol_info()`. Storing only bars would leave every offline run
        still requiring a live terminal, which is the thing being removed.
        """
        manifest = self._manifest()
        manifest.setdefault("symbols", {})[spec.symbol] = asdict(spec)
        self._write_manifest(manifest)

    def spec(self, symbol: str) -> InstrumentSpec:
        stored = self._manifest().get("symbols", {}).get(symbol)
        if stored is None:
            raise KeyError(f"no stored spec for {symbol} in {self.root}")
        known = {field.name for field in fields(InstrumentSpec)}
        values = {name: value for name, value in stored.items() if name in known}
        values["asset_class"] = AssetClass(values["asset_class"])
        return InstrumentSpec(**values)

    # -- manifest ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"layout": LAYOUT_VERSION, "symbols": {}}
        try:
            loaded = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"layout": LAYOUT_VERSION, "symbols": {}}
        if loaded.get("layout") != LAYOUT_VERSION:
            return {"layout": LAYOUT_VERSION, "symbols": {}}
        return loaded

    def _write_manifest(self, manifest: dict) -> None:
        manifest["layout"] = LAYOUT_VERSION
        manifest["written_at"] = datetime.now(UTC).isoformat()
        self.root.mkdir(parents=True, exist_ok=True)
        # Written whole and moved into place, so an interrupted fetch cannot
        # leave a half-written manifest that makes every stored spec vanish.
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        temporary.replace(self.manifest_path)

    def note_window(self, days: int, end: datetime) -> None:
        manifest = self._manifest()
        manifest["window_days"] = days
        manifest["window_end"] = end.isoformat()
        self._write_manifest(manifest)

    def window(self) -> tuple[int, str] | None:
        manifest = self._manifest()
        if "window_days" not in manifest:
            return None
        return int(manifest["window_days"]), str(manifest.get("window_end", ""))

    # -- reporting ---------------------------------------------------------

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*.npz"))

    def summary(self) -> list[tuple[str, str, int, int]]:
        """(symbol, timeframe, bars, bytes) for everything stored."""
        rows: list[tuple[str, str, int, int]] = []
        for symbol in self.symbols():
            for name in self.timeframes(symbol):
                path = self._dir(symbol) / f"{name}.npz"
                with np.load(path, allow_pickle=True) as data:
                    bars = len(data["__time__"])
                rows.append((symbol, name, bars, path.stat().st_size))
        return rows
