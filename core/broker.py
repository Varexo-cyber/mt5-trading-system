"""Broker-neutral contracts for data and execution.

The strategy and risk layers must not care whether execution happens through
MT5, a REST broker, or a test double. These protocols capture only the domain
operations they need. Venue-specific return codes and payloads stay behind an
adapter, while stop-loss enforcement remains in the immutable OrderRequest.

TradingView is deliberately absent: it is a charting and alert surface, not an
execution venue. A future TradingView companion may display decisions or send
authenticated operator alerts, but it cannot bypass this broker contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from core.instrument import InstrumentSpec
from core.types import AccountSnapshot, OrderRequest, OrderResult, Position, SymbolDescriptor, Tick


@runtime_checkable
class MarketDataProvider(Protocol):
    """Closed-bar and quote access required by analysis and filters."""

    @property
    def server_offset(self) -> timedelta: ...

    def spec(self, symbol: str, *, refresh: bool = False) -> InstrumentSpec: ...

    def tick(self, symbol: str) -> Tick: ...

    def copy_rates(
        self,
        symbol: str,
        timeframe: int,
        count: int,
        start_pos: int = 0,
    ) -> Any: ...

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int,
        start: datetime,
        end: datetime,
    ) -> Any: ...


@runtime_checkable
class Broker(MarketDataProvider, Protocol):
    """Account and order operations every live execution adapter must provide."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> AccountSnapshot: ...

    def ensure_connected(self) -> None: ...

    def shutdown(self) -> None: ...

    def account(self) -> AccountSnapshot: ...

    def positions(
        self,
        symbol: str | None = None,
        magic: int | None = None,
    ) -> list[Position]: ...

    def symbols(self) -> list[SymbolDescriptor]: ...

    def select(self, symbol: str) -> None: ...

    def order_send(self, request: OrderRequest, spec: InstrumentSpec) -> OrderResult: ...

    def modify_stops(self, position: Position, *, sl: float, tp: float) -> OrderResult: ...

    def close_position(self, position: Position, volume: float | None = None) -> OrderResult: ...
