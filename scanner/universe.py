"""Staged whole-catalogue scanning without pretending every symbol is liquid."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pandas as pd

from config.schema import Settings
from core.broker import Broker
from core.clock import Clock, LiveClock
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
    priority_tier: int
    spread_quality: float


@dataclass(frozen=True, slots=True)
class ScanInspection:
    inspected_at: datetime
    symbol: str
    path: str
    asset_class: AssetClass
    status: str
    stage: str
    reason: str
    rank: float | None = None
    spread_bps: float | None = None
    quote_age_seconds: float | None = None
    trend_strength_atr: float | None = None
    latest_bar: datetime | None = None

    def safe_dict(self) -> dict[str, object]:
        return {
            "inspected_at": self.inspected_at.isoformat(),
            "symbol": self.symbol,
            "path": self.path,
            "asset_class": self.asset_class.value,
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "rank": self.rank,
            "spread_bps": self.spread_bps,
            "quote_age_seconds": self.quote_age_seconds,
            "trend_strength_atr": self.trend_strength_atr,
            "latest_bar": self.latest_bar.isoformat() if self.latest_bar else None,
        }


@dataclass(frozen=True, slots=True)
class ScanBatch:
    candidates: tuple[ScanCandidate, ...]
    inspections: tuple[ScanInspection, ...]
    inspected: int
    rejected: int
    next_cursor: int
    universe_size: int


class UniverseScanner:
    """Cheaply rank one rotating batch; deep analysis runs only on its winners."""

    def __init__(self, broker: Broker, settings: Settings, clock: Clock | None = None) -> None:
        self.broker = broker
        self.settings = settings
        # The scanner used `datetime.now(UTC)` directly, against the project rule
        # that nothing outside `core.clock` reads the wall clock. It is not
        # cosmetic: quote age is measured against this, so under a simulated
        # clock every symbol was "stale, market may be closed" and the scanner
        # could not be tested at all. That is how the deep-analysis stage went
        # unexamined for so long.
        self.clock: Clock = clock or LiveClock()

    def catalogue(self) -> list[SymbolDescriptor]:
        """Every broker symbol the operator has asked to look at.

        Two optional narrowings, both off by default. `asset_classes` chooses
        which kinds of market are scanned at all; `symbols_only` reduces it to a
        named handful. Neither changes how anything is judged — they decide what
        gets looked at, and everything they exclude simply never appears.
        """
        wanted = set(self.settings.instruments.asset_classes)
        chosen = self.settings.instruments.symbols_only
        names = (
            {self.settings.instruments.broker_symbol(name) for name in chosen} if chosen else set()
        )

        return [
            item
            for item in self.broker.symbols()
            # Empty means literally every row returned by MT5's symbols_get(),
            # including a future Eightcap folder this version does not yet know.
            # Unknown families are still visible in telemetry and are rejected
            # fail-closed by _inspect if their contract cannot classify them.
            if (not wanted or self._path_class(item.path).value in wanted)
            and (not names or item.name in names)
        ]

    def scan(
        self,
        *,
        cursor: int = 0,
        batch_size: int | None = None,
        keep: int = 5,
        pulse: Callable[[], object] | None = None,
    ) -> ScanBatch:
        universe = self.catalogue()
        if not universe:
            return ScanBatch((), (), 0, 0, 0, 0)
        cursor %= len(universe)
        inspection_count = len(universe) if batch_size is None else min(batch_size, len(universe))
        indices = [(cursor + offset) % len(universe) for offset in range(inspection_count)]
        candidates: list[ScanCandidate] = []
        inspections: list[ScanInspection] = []
        rejected = 0
        for index in indices:
            item = universe[index]
            candidate, inspection = self._inspect(item)
            inspections.append(inspection)
            if candidate is None:
                rejected += 1
            else:
                candidates.append(candidate)
            # MT5 owns one global, non-thread-safe session. Interleave position
            # protection between catalogue reads instead of racing a second
            # thread against the connector while money is open.
            if pulse is not None:
                pulse()
        candidates.sort(key=lambda item: (item.priority_tier, item.rank), reverse=True)
        shortlisted = {item.symbol for item in candidates[:keep]}
        inspections = [
            (
                replace(
                    row,
                    status="SHORTLISTED",
                    stage="deep_analysis_queued",
                    reason="Top-ranked symbol in this rotating batch; queued for full analysis",
                )
                if row.symbol in shortlisted
                else row
            )
            for row in inspections
        ]
        next_cursor = (cursor + len(indices)) % len(universe)
        return ScanBatch(
            tuple(candidates[:keep]),
            tuple(inspections),
            len(indices),
            rejected,
            next_cursor,
            len(universe),
        )

    def _inspect(self, descriptor: SymbolDescriptor) -> tuple[ScanCandidate | None, ScanInspection]:
        inspected_at = self.clock.now()
        path_class = self._path_class(descriptor.path)

        def reject(
            stage: str,
            reason: str,
            *,
            asset_class: AssetClass = path_class,
            spread_bps: float | None = None,
            quote_age_seconds: float | None = None,
        ) -> tuple[None, ScanInspection]:
            return None, ScanInspection(
                inspected_at,
                descriptor.name,
                descriptor.path,
                asset_class,
                "REJECTED",
                stage,
                reason,
                spread_bps=spread_bps,
                quote_age_seconds=quote_age_seconds,
            )

        try:
            spec = self.broker.spec(descriptor.name)
            if not spec.is_tradable:
                return reject(
                    "broker_contract",
                    f"Broker reports trade_mode={spec.trade_mode}; instrument is not tradable",
                    asset_class=spec.asset_class,
                )
            if spec.asset_class is AssetClass.UNKNOWN:
                return reject(
                    "classification",
                    "Instrument asset class is unsupported",
                    asset_class=spec.asset_class,
                )
            tick = self.broker.tick(descriptor.name)
            age = max(0.0, (inspected_at - tick.time).total_seconds())
            max_age = self.settings.filters.spread.max_tick_age_seconds.get(spec.asset_class.value)
            if tick.mid <= 0:
                return reject(
                    "quote",
                    "Invalid non-positive quote",
                    asset_class=spec.asset_class,
                    quote_age_seconds=age,
                )
            spread_bps = tick.spread / tick.mid * 10_000
            if max_age is None or age > max_age:
                return reject(
                    "quote",
                    "Quote is stale; market may be closed",
                    asset_class=spec.asset_class,
                    spread_bps=spread_bps,
                    quote_age_seconds=age,
                )
            cap = self.settings.filters.spread.max_spread_bps.get(spec.asset_class.value)
            if cap is None or spread_bps > cap:
                return reject(
                    "spread",
                    (
                        f"Spread {spread_bps:.3f} bps exceeds {cap:.3f} bps limit"
                        if cap is not None
                        else "No spread limit configured for this asset class"
                    ),
                    asset_class=spec.asset_class,
                    spread_bps=spread_bps,
                    quote_age_seconds=age,
                )
            raw = self.broker.copy_rates(descriptor.name, Timeframe.H1.mt5_value, 90)
            frame = pd.DataFrame(raw)
            if len(frame) < 55:
                return reject(
                    "history",
                    f"Only {len(frame)} H1 bars available; at least 55 required",
                    asset_class=spec.asset_class,
                    spread_bps=spread_bps,
                    quote_age_seconds=age,
                )
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
                return reject(
                    "ranking",
                    "ATR could not be calculated from H1 history",
                    asset_class=spec.asset_class,
                    spread_bps=spread_bps,
                    quote_age_seconds=age,
                )
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            strength = abs(float(ema20.iloc[-1] - ema50.iloc[-1])) / float(atr)
            activity = min(
                2.0,
                float(frame["tick_volume"].iloc[-1])
                / max(1.0, float(frame["tick_volume"].tail(50).median())),
            )
            spread_quality = max(0.0, 1.0 - spread_bps / cap)
            priority_tier = self._priority_tier(descriptor.name, spec.asset_class)
            rank = (
                strength * 2.0
                + activity
                + spread_quality
                + spread_quality * self.settings.scanner.priority_spread_weight
            )
            last = datetime.fromtimestamp(int(frame["time"].iloc[-1]), tz=UTC)
            candidate = ScanCandidate(
                descriptor.name,
                spec.asset_class,
                rank,
                spread_bps,
                age,
                strength,
                last,
                priority_tier,
                spread_quality,
            )
            return candidate, ScanInspection(
                inspected_at,
                descriptor.name,
                descriptor.path,
                spec.asset_class,
                "ELIGIBLE",
                "cheap_scan_passed",
                "Fresh tradable quote, acceptable spread and sufficient H1 history",
                rank=rank,
                spread_bps=spread_bps,
                quote_age_seconds=age,
                trend_strength_atr=strength,
                latest_bar=last,
            )
        except Exception as exc:  # noqa: BLE001 - catalogue has closed/unavailable CFDs
            return reject(
                "broker_data",
                f"Broker data unavailable ({type(exc).__name__})",
            )

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

    def _priority_tier(self, symbol: str, asset_class: AssetClass) -> int:
        """Two preferred lanes above the complete-catalogue fallback."""
        base_symbol = symbol.split(".", 1)[0].upper()
        if base_symbol in self.settings.scanner.priority_symbols:
            return 2
        if asset_class.value in self.settings.scanner.priority_asset_classes:
            return 1
        return 0
