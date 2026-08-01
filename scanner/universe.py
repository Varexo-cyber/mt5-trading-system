"""Staged whole-catalogue scanning without pretending every symbol is liquid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from config.schema import Settings
from core.broker import Broker
from core.instrument import AssetClass
from core.types import SymbolDescriptor, Timeframe


@dataclass(frozen=True, slots=True)
class ScanCandidate:
    symbol: str
    asset_class: AssetClass
    rank: float
    spread_bps: float
    quote_age_seconds: float
    trend_strength_atr: float
    latest_bar: datetime


@dataclass(frozen=True, slots=True)
class ScanBatch:
    candidates: tuple[ScanCandidate, ...]
    inspected: int
    rejected: int
    next_cursor: int
    universe_size: int


class UniverseScanner:
    """Cheaply rank one rotating batch; deep analysis runs only on its winners."""

    def __init__(self, broker: Broker, settings: Settings) -> None:
        self.broker = broker
        self.settings = settings

    def catalogue(self) -> list[SymbolDescriptor]:
        supported = {asset.value for asset in AssetClass if asset is not AssetClass.UNKNOWN}
        return [
            item for item in self.broker.symbols() if self._path_class(item.path).value in supported
        ]

    def scan(self, *, cursor: int = 0, batch_size: int = 25, keep: int = 5) -> ScanBatch:
        universe = self.catalogue()
        if not universe:
            return ScanBatch((), 0, 0, 0, 0)
        cursor %= len(universe)
        indices = [
            (cursor + offset) % len(universe) for offset in range(min(batch_size, len(universe)))
        ]
        candidates: list[ScanCandidate] = []
        rejected = 0
        for index in indices:
            item = universe[index]
            candidate = self._inspect(item)
            if candidate is None:
                rejected += 1
            else:
                candidates.append(candidate)
        candidates.sort(key=lambda item: item.rank, reverse=True)
        next_cursor = (cursor + len(indices)) % len(universe)
        return ScanBatch(
            tuple(candidates[:keep]), len(indices), rejected, next_cursor, len(universe)
        )

    def _inspect(self, descriptor: SymbolDescriptor) -> ScanCandidate | None:
        try:
            spec = self.broker.spec(descriptor.name)
            if not spec.is_tradable or spec.asset_class is AssetClass.UNKNOWN:
                return None
            tick = self.broker.tick(descriptor.name)
            now = datetime.now(UTC)
            age = max(0.0, (now - tick.time).total_seconds())
            max_age = self.settings.filters.spread.max_tick_age_seconds.get(spec.asset_class.value)
            if max_age is None or age > max_age or tick.mid <= 0:
                return None
            spread_bps = tick.spread / tick.mid * 10_000
            cap = self.settings.filters.spread.max_spread_bps.get(spec.asset_class.value)
            if cap is None or spread_bps > cap:
                return None
            raw = self.broker.copy_rates(descriptor.name, Timeframe.H1.mt5_value, 90)
            frame = pd.DataFrame(raw)
            if len(frame) < 55:
                return None
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            previous = close.shift(1)
            atr = (
                pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1)
                .max(axis=1)
                .rolling(14)
                .mean()
                .iloc[-1]
            )
            if not atr or pd.isna(atr):
                return None
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            strength = abs(float(ema20.iloc[-1] - ema50.iloc[-1])) / float(atr)
            activity = min(
                2.0,
                float(frame["tick_volume"].iloc[-1])
                / max(1.0, float(frame["tick_volume"].tail(50).median())),
            )
            spread_quality = max(0.0, 1.0 - spread_bps / cap)
            rank = strength * 2.0 + activity + spread_quality
            last = datetime.fromtimestamp(int(frame["time"].iloc[-1]), tz=UTC)
            return ScanCandidate(
                descriptor.name,
                spec.asset_class,
                rank,
                spread_bps,
                age,
                strength,
                last,
            )
        except Exception:  # noqa: BLE001 - catalogue contains many closed/unavailable CFDs
            return None

    @staticmethod
    def _path_class(path: str) -> AssetClass:
        root = path.split("\\", 1)[0].strip().lower()
        if root == "raw" or root == "forex":
            return AssetClass.FOREX
        if root in {"crypto", "cryptos"}:
            return AssetClass.CRYPTO
        if root in {"stock", "stocks", "shares"}:
            return AssetClass.STOCK
        if root in {"index", "indices"}:
            return AssetClass.INDEX
        if root in {"commodity", "commodities"}:
            return AssetClass.METAL if "metal" in path.lower() else AssetClass.COMMODITY
        return AssetClass.UNKNOWN
