"""Staged whole-catalogue scanning without pretending every symbol is liquid."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pandas as pd

from config.schema import Settings
from core.broker import Broker
from core.clock import Clock, LiveClock
from core.instrument import AssetClass
from core.types import Direction, SymbolDescriptor, Timeframe

#: How long a contract's minimum-lot margin stays usable. It moves with price
#: and price moves slowly relative to a whole catalogue sweep.
_MARGIN_CACHE_SECONDS = 900.0
#: Equity, read once per scan rather than once per symbol.
_EQUITY_CACHE_SECONDS = 30.0


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
class ParkedSpread:
    """A spread measurement kept instead of taken again.

    Only ever written for a market already refused at the spread stage, and
    only when it was far enough past its cap that a re-measurement inside the
    hold cannot plausibly change the answer.
    """

    until: datetime
    asset_class: AssetClass
    spread_bps: float
    cap_bps: float
    measured_at: datetime


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
        # Symbol -> the spread refusal being remembered. Bounded by the
        # catalogue, released by its own deadline, never persisted: a restart
        # measures everything again.
        self._parked: dict[str, ParkedSpread] = {}
        # Symbol -> (expiry, margin for one minimum lot). A property of the
        # contract, not of the account, so it is stable enough to cache and
        # compared against fresh equity every time.
        self._minimum_margin: dict[str, tuple[float, float]] = {}
        self._equity_seen: tuple[float, float] | None = None

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
            and not self.settings.instruments.is_ignored(item.name)
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
        rotating = [(cursor + offset) % len(universe) for offset in range(inspection_count)]
        recurring = (
            [
                index
                for index, item in enumerate(universe)
                if self._priority_tier(item.name, self._path_class(item.path)) > 0
            ]
            if batch_size is not None and self.settings.scanner.priority_every_cycle
            else []
        )
        # dict preserves order: liquid markets are inspected first, then the
        # rotating catalogue slice. A symbol in both is still inspected once.
        indices = list(dict.fromkeys([*recurring, *rotating]))
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
        # Recurring priority rows are additions to the rotating budget. They
        # must not advance the catalogue cursor or fallback symbols get skipped.
        next_cursor = (cursor + inspection_count) % len(universe)
        return ScanBatch(
            tuple(candidates[:keep]),
            tuple(inspections),
            len(indices),
            rejected,
            next_cursor,
            len(universe),
        )

    def _unaffordable(self, symbol: str, spec, tick) -> str | None:  # type: ignore[no-untyped-def]
        """Whether the SMALLEST position this contract allows is out of reach.

        Deliberately the minimum lot and not the sized one. A sized position
        being too large is a fact about today's stop and belongs to the sizer,
        which already refuses it per trade. This asks a different and permanent
        question -- can this account hold ANY position in this instrument --
        and only removes markets where the answer is no on every possible
        setup.

        Unknown is never a refusal. A broker that cannot price the margin, or
        an account whose equity does not read, leaves the symbol in the
        catalogue and lets the per-trade margin check have the last word.
        """
        equity = self._equity()
        if equity <= 0 or spec.volume_min <= 0:
            return None
        margin = self._margin_for_minimum(symbol, spec, tick)
        if margin is None or margin <= 0:
            return None
        factor = self.settings.risk.margin_safety_factor
        needed = margin * factor
        if needed <= equity:
            return None
        return (
            f"Minimum lot {spec.volume_min:g} needs {margin:.2f} margin; with the "
            f"{factor:g}x buffer that is {needed:.2f} against {equity:.2f} equity. "
            f"No setup on this instrument can be sized small enough."
        )

    def _margin_for_minimum(self, symbol: str, spec, tick) -> float | None:  # type: ignore[no-untyped-def]
        cached = self._minimum_margin.get(symbol)
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1]
        estimate = getattr(self.broker, "estimate_margin", None)
        if not callable(estimate):
            return None
        try:
            margin = float(estimate(symbol, Direction.LONG, spec.volume_min, tick.ask))
        except Exception:  # noqa: BLE001 - an unpriceable contract is not a refusal
            return None
        self._minimum_margin[symbol] = (now + _MARGIN_CACHE_SECONDS, margin)
        return margin

    def _equity(self) -> float:
        """Account equity, once per scan rather than once per symbol."""
        cached = self._equity_seen
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1]
        try:
            equity = float(self.broker.account().equity)
        except Exception:  # noqa: BLE001 - unknown equity refuses nothing
            return 0.0
        self._equity_seen = (now + _EQUITY_CACHE_SECONDS, equity)
        return equity

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

        held = self._parked.get(descriptor.name)
        if held is not None:
            if inspected_at < held.until:
                # Same stage as a fresh refusal on purpose: this symbol is
                # blocked on spread, and every report that counts spread
                # blocks must keep counting it.
                return reject(
                    "spread",
                    (
                        f"Spread {held.spread_bps:.3f} bps was "
                        f"{held.spread_bps / held.cap_bps:.0f}x the {held.cap_bps:.3f} bps "
                        f"limit at {held.measured_at:%H:%M} UTC; not re-measured "
                        f"until {held.until:%H:%M} UTC"
                    ),
                    asset_class=held.asset_class,
                    spread_bps=held.spread_bps,
                )
            del self._parked[descriptor.name]

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
                scanner = self.settings.scanner
                hours = scanner.wide_spread_park_hours
                hopeless = cap is not None and spread_bps > cap * scanner.wide_spread_park_multiple
                if hopeless and hours > 0.0:
                    assert cap is not None  # implied by `hopeless`
                    self._parked[descriptor.name] = ParkedSpread(
                        inspected_at + timedelta(hours=hours),
                        spec.asset_class,
                        spread_bps,
                        cap,
                        inspected_at,
                    )
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
            # BEFORE THE EXPENSIVE PART, because this refusal is permanent for
            # the account rather than momentary for the market.
            #
            # The deck was full of INSUFFICIENT_MARGIN on single-name stocks:
            # 0.23 lots of ADS wanting 718 EUR of margin, 0.2 of NXTL wanting
            # 728, on an account holding 176. Those are not near misses. The
            # smallest position the contract allows is already out of reach, so
            # no price, no signal and no adviser opinion could ever turn one
            # into a trade -- and each was costing a full multi-timeframe
            # analysis every cycle on a one-vCPU box, which is time the guard
            # and the two sections that can actually trade were not getting.
            unaffordable = self._unaffordable(descriptor.name, spec, tick)
            if unaffordable is not None:
                return reject(
                    "affordability",
                    unaffordable,
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
