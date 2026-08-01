"""Read-only broker data service used by the local dashboard.

No order method is exposed here. The dashboard may inspect every instrument,
but it cannot bypass the execution engine's risk and validation layers.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from config.schema import Settings
from core.errors import TradingSystemError
from core.instrument import AssetClass, InstrumentSpec
from core.mt5_connector import MT5Connector
from core.types import (
    AccountSnapshot,
    Position,
    SymbolDescriptor,
    Tick,
    Timeframe,
)

PROFILE_TIMEFRAMES: dict[AssetClass, tuple[Timeframe, ...]] = {
    AssetClass.FOREX: (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5),
    AssetClass.CRYPTO: (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5),
    AssetClass.STOCK: (Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15),
    AssetClass.INDEX: (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5),
    AssetClass.METAL: (Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15),
    AssetClass.COMMODITY: (Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1),
    AssetClass.UNKNOWN: (Timeframe.D1, Timeframe.H4, Timeframe.H1),
}


class DashboardService:
    """Short-lived read-only session against the configured terminal."""

    def __init__(self, connector: MT5Connector, settings: Settings) -> None:
        self.connector = connector
        self.settings = settings
        self.account_snapshot: AccountSnapshot | None = None

    def connect(self) -> AccountSnapshot:
        self.account_snapshot = self.connector.connect()
        return self.account_snapshot

    def close(self) -> None:
        self.connector.shutdown()

    def account(self) -> AccountSnapshot:
        return self.account_snapshot or self.connect()

    def positions(self) -> list[Position]:
        return self.connector.positions()

    def symbols(self) -> list[SymbolDescriptor]:
        return self.connector.symbols()

    def spec(self, symbol: str) -> InstrumentSpec:
        return self.connector.spec(symbol)

    def tick(self, symbol: str) -> Tick | None:
        try:
            tick = self.connector.tick(symbol)
        except TradingSystemError:
            return None
        if tick.bid <= 0 or tick.ask <= tick.bid:
            return None
        return tick

    def bars(self, symbol: str, timeframe: Timeframe, count: int = 300) -> pd.DataFrame:
        """Closed OHLCV bars for visual inspection, including stale history.

        The trading data manager is deliberately stricter and rejects stale
        series. A dashboard should still render Friday's chart on Saturday, so
        this method returns history and lets the UI label its age.
        """
        raw = self.connector.copy_rates(symbol, timeframe.mt5_value, count + 1)
        frame = pd.DataFrame(raw)
        if frame.empty:
            return frame
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame["time"] = frame["time"] - pd.to_timedelta(self.connector.server_offset)
        frame = frame.set_index("time").sort_index()
        now = datetime.now(UTC)
        if frame.index[-1].to_pydatetime() + timeframe.duration > now:
            frame = frame.iloc[:-1]
        return frame.tail(count)


def catalogue_asset_class(path: str) -> AssetClass:
    """Classify a catalogue row without selecting it or querying tick values."""
    root = path.split("\\", 1)[0].strip().lower()
    mapping = {
        "raw": AssetClass.FOREX,
        "forex": AssetClass.FOREX,
        "crypto": AssetClass.CRYPTO,
        "cryptos": AssetClass.CRYPTO,
        "stock": AssetClass.STOCK,
        "stocks": AssetClass.STOCK,
        "shares": AssetClass.STOCK,
        "index": AssetClass.INDEX,
        "indices": AssetClass.INDEX,
        "commodity": AssetClass.COMMODITY,
        "commodities": AssetClass.COMMODITY,
    }
    asset = mapping.get(root, AssetClass.UNKNOWN)
    if asset is AssetClass.COMMODITY and "metal" in path.lower():
        return AssetClass.METAL
    return asset
