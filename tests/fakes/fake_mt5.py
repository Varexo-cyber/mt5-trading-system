"""In-memory stand-in for the MetaTrader5 package.

Enough of the real API surface to exercise connection handling, retries, symbol
specs, rate fetching and order execution on any platform. The point is not to
simulate a market — it is to let the *error paths* be tested, since those are
the ones that never get exercised by hand and the ones that lose money.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.mt5_codes import TIMEFRAME_VALUES, OrderType, Retcode, TradeAction

# Mirrors the real package's module-level constants that MT5Connector verifies.
TIMEFRAME_M1 = TIMEFRAME_VALUES["M1"]
TIMEFRAME_M5 = TIMEFRAME_VALUES["M5"]
TIMEFRAME_M15 = TIMEFRAME_VALUES["M15"]
TIMEFRAME_M30 = TIMEFRAME_VALUES["M30"]
TIMEFRAME_H1 = TIMEFRAME_VALUES["H1"]
TIMEFRAME_H4 = TIMEFRAME_VALUES["H4"]
TIMEFRAME_D1 = TIMEFRAME_VALUES["D1"]
TIMEFRAME_W1 = TIMEFRAME_VALUES["W1"]
TIMEFRAME_MN1 = TIMEFRAME_VALUES["MN1"]

ORDER_TYPE_BUY = int(OrderType.BUY)
ORDER_TYPE_SELL = int(OrderType.SELL)
TRADE_ACTION_DEAL = int(TradeAction.DEAL)
TRADE_ACTION_SLTP = int(TradeAction.SLTP)


def eurusd_spec(**overrides: Any) -> SimpleNamespace:
    """A realistic 5-digit EURUSD contract spec on a USD-denominated account."""
    base = {
        "name": "EURUSD",
        "digits": 5,
        "point": 0.00001,
        "trade_tick_size": 0.00001,
        "trade_tick_value": 1.0,  # USD per 0.00001 on 1.00 lot
        "trade_contract_size": 100_000.0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "trade_stops_level": 0,
        "trade_freeze_level": 0,
        "currency_base": "EUR",
        "currency_profit": "USD",
        "currency_margin": "EUR",
        "filling_mode": 2,  # IOC
        "trade_mode": 4,  # full
        "path": "Forex\\Majors\\EURUSD",
        "description": "Euro vs US Dollar",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def usdjpy_spec(**overrides: Any) -> SimpleNamespace:
    """3-digit JPY pair: pip = 0.01, which is where naive pip maths breaks."""
    base = {
        "name": "USDJPY",
        "digits": 3,
        "point": 0.001,
        "trade_tick_size": 0.001,
        "trade_tick_value": 0.0067,  # ~USD per 0.001 on 1.00 lot at 150.00
        "trade_contract_size": 100_000.0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "trade_stops_level": 20,
        "trade_freeze_level": 0,
        "currency_base": "USD",
        "currency_profit": "JPY",
        "currency_margin": "USD",
        "filling_mode": 1,  # FOK only
        "trade_mode": 4,
        "path": "Forex\\Majors\\USDJPY",
        "description": "US Dollar vs Yen",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def xauusd_spec(**overrides: Any) -> SimpleNamespace:
    """Gold: point-quoted, 100 oz contract. Not FX, and the maths must know it."""
    base = {
        "name": "XAUUSD",
        "digits": 2,
        "point": 0.01,
        "trade_tick_size": 0.01,
        "trade_tick_value": 1.0,  # USD per 0.01 on 1.00 lot (100 oz)
        "trade_contract_size": 100.0,
        "volume_min": 0.01,
        "volume_max": 50.0,
        "volume_step": 0.01,
        "trade_stops_level": 50,
        "trade_freeze_level": 20,
        "currency_base": "XAU",
        "currency_profit": "USD",
        "currency_margin": "USD",
        "filling_mode": 2,
        "trade_mode": 4,
        "path": "Commodities\\Metals\\XAUUSD",
        "description": "Gold vs US Dollar",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@dataclass
class FakeMT5:
    """Scriptable fake terminal.

    Attach it to a connector with `MT5Connector(..., mt5_module=fake)`.
    """

    equity: float = 1000.0
    balance: float = 1000.0
    currency: str = "USD"
    login_id: int = 1234567
    server: str = "FakeBroker-Demo"
    is_demo: bool = True
    connected: bool = True

    #: Symbol name -> spec namespace.
    specs: dict[str, SimpleNamespace] = field(default_factory=dict)
    #: Symbol name -> (bid, ask).
    quotes: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Return codes served to consecutive order_send calls, then DONE forever.
    order_retcodes: list[int] = field(default_factory=list)
    #: Fill price offset, in price units, applied to every fill (slippage).
    fill_offset: float = 0.0
    #: Simulate brokers that confirm a market deal but omit the fill price from
    #: MqlTradeResult. The authoritative price remains available in history.
    zero_result_price: bool = False

    initialize_failures: int = 0
    error: tuple[int, str] = (0, "Success")
    #: Broker "current time". Set it to match a SimulatedClock so the newest
    #: synthetic bar is the one that clock would consider still forming.
    now: datetime | None = None

    #: Symbol -> (open hour, close hour) in UTC. Absent means 24-hour trading.
    #: A broker catalogue is mostly *not* spot FX: a share trades eight hours a
    #: day, a grain future halts for several. Bars are generated accordingly so
    #: the gap and staleness checks meet the shape of data they really see.
    session_hours: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: Symbol -> how far behind `now` its last tick is. A closed exchange keeps
    #: serving its final quote, which is what dragged the server-time offset
    #: around in production.
    tick_age: dict[str, timedelta] = field(default_factory=dict)

    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    orders_sent: list[dict[str, Any]] = field(default_factory=list)
    positions: list[SimpleNamespace] = field(default_factory=list)
    deals: list[SimpleNamespace] = field(default_factory=list)
    margin_required: float = 1.0
    _initialized: bool = False
    _retcode_iter: Iterator[int] | None = None

    def __post_init__(self) -> None:
        if not self.specs:
            self.specs = {"EURUSD": eurusd_spec(), "USDJPY": usdjpy_spec()}
        if not self.quotes:
            self.quotes = {"EURUSD": (1.08500, 1.08512), "USDJPY": (150.100, 150.118)}

    # -- session ----------------------------------------------------------

    def initialize(self, path: str | None = None, **kwargs: Any) -> bool:
        self.calls.append(("initialize", (path,)))
        if self.initialize_failures > 0:
            self.initialize_failures -= 1
            self.error = (-10005, "IPC timeout")
            return False
        self._initialized = True
        self.error = (0, "Success")
        return True

    def login(self, login: int, **kwargs: Any) -> bool:
        self.calls.append(("login", (login,)))
        self.login_id = login
        return True

    def shutdown(self) -> None:
        self.calls.append(("shutdown", ()))
        self._initialized = False

    def last_error(self) -> tuple[int, str]:
        return self.error

    def terminal_info(self) -> SimpleNamespace | None:
        if not self._initialized:
            return None
        return SimpleNamespace(connected=self.connected, build=4260)

    def account_info(self) -> SimpleNamespace | None:
        if not self._initialized:
            self.error = (-10004, "No IPC connection")
            return None
        return SimpleNamespace(
            login=self.login_id,
            server=self.server,
            currency=self.currency,
            balance=self.balance,
            equity=self.equity,
            margin=0.0,
            margin_free=self.equity,
            margin_level=0.0,
            leverage=500,
            trade_mode=0 if self.is_demo else 2,
        )

    # -- symbols ----------------------------------------------------------

    def symbols_get(self) -> tuple[SimpleNamespace, ...]:
        self.calls.append(("symbols_get", ()))
        return tuple(
            SimpleNamespace(
                **vars(spec),
                visible=name in self.quotes,
                select=name in self.quotes,
            )
            for name, spec in self.specs.items()
        )

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        self.calls.append(("symbol_select", (symbol,)))
        if symbol not in self.specs:
            self.error = (-10002, "Unknown symbol")
            return False
        return True

    def symbol_info(self, symbol: str) -> SimpleNamespace | None:
        self.calls.append(("symbol_info", (symbol,)))
        return self.specs.get(symbol)

    def _now(self) -> datetime:
        return self.now or datetime.now(UTC)

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace | None:
        self.calls.append(("symbol_info_tick", (symbol,)))
        quote = self.quotes.get(symbol)
        if quote is None:
            self.error = (-10002, "Unknown symbol")
            return None
        bid, ask = quote
        moment = self._now() - self.tick_age.get(symbol, timedelta(0))
        return SimpleNamespace(time=int(moment.timestamp()), bid=bid, ask=ask, last=0.0, volume=0)

    # -- rates ------------------------------------------------------------

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray | None:
        self.calls.append(("copy_rates_from_pos", (symbol, timeframe, start_pos, count)))
        if symbol not in self.specs:
            self.error = (-10002, "Unknown symbol")
            return None
        return synthetic_rates(
            count,
            timeframe,
            base_price=self.quotes[symbol][0],
            end=self._now(),
            session=self.session_hours.get(symbol),
        )

    def copy_rates_range(
        self, symbol: str, timeframe: int, start: datetime, end: datetime
    ) -> np.ndarray | None:
        self.calls.append(("copy_rates_range", (symbol, timeframe, start, end)))
        minutes = _minutes_for(timeframe)
        count = max(int((end - start).total_seconds() / 60 / minutes), 1)
        return synthetic_rates(count, timeframe, base_price=self.quotes[symbol][0], end=end)

    # -- trading ----------------------------------------------------------

    def positions_get(self, symbol: str | None = None) -> tuple[SimpleNamespace, ...]:
        self.calls.append(("positions_get", (symbol,)))
        if symbol is None:
            return tuple(self.positions)
        return tuple(p for p in self.positions if p.symbol == symbol)

    def history_deals_get(
        self, *, position: int | None = None, ticket: int | None = None
    ) -> tuple[SimpleNamespace, ...]:
        self.calls.append(("history_deals_get", (position, ticket)))
        if ticket is not None:
            return tuple(deal for deal in self.deals if int(deal.ticket) == ticket)
        return tuple(deal for deal in self.deals if int(deal.position_id) == int(position or 0))

    def order_calc_margin(
        self, action: int, symbol: str, volume: float, price: float
    ) -> float | None:
        self.calls.append(("order_calc_margin", (action, symbol, volume, price)))
        if symbol not in self.specs:
            self.error = (-10002, "Unknown symbol")
            return None
        return self.margin_required * volume

    def order_send(self, request: dict[str, Any]) -> SimpleNamespace:
        self.calls.append(("order_send", (request.get("symbol"),)))
        self.orders_sent.append(dict(request))

        if self._retcode_iter is None:
            self._retcode_iter = iter(self.order_retcodes)
        retcode = next(self._retcode_iter, int(Retcode.DONE))

        price = float(request.get("price", 0.0))
        filled = price + self.fill_offset if retcode == int(Retcode.DONE) else 0.0
        if retcode == int(Retcode.DONE) and self.zero_result_price:
            self.deals.append(
                SimpleNamespace(
                    ticket=555_001,
                    position_id=555_002,
                    price=filled,
                    volume=float(request.get("volume", 0.0)),
                )
            )
        return SimpleNamespace(
            retcode=retcode,
            deal=555_001 if retcode == int(Retcode.DONE) else 0,
            order=555_002 if retcode == int(Retcode.DONE) else 0,
            volume=float(request.get("volume", 0.0)) if retcode == int(Retcode.DONE) else 0.0,
            price=0.0 if self.zero_result_price else filled,
            bid=price,
            ask=price,
            comment="Request executed" if retcode == int(Retcode.DONE) else "Rejected",
            request_id=1,
            retcode_external=0,
        )


def _minutes_for(timeframe: int) -> int:
    for name, value in TIMEFRAME_VALUES.items():
        if value == timeframe:
            return {
                "M1": 1,
                "M5": 5,
                "M15": 15,
                "M30": 30,
                "H1": 60,
                "H4": 240,
                "D1": 1440,
                "W1": 10080,
                "MN1": 43200,
            }.get(name, 60)
    return 60


def _bar_times(
    last_open: datetime,
    step: timedelta,
    count: int,
    session: tuple[int, int] | None,
    *,
    intraday: bool,
) -> list[datetime]:
    """Walk back from `last_open`, skipping closures, until `count` bars exist."""
    times: list[datetime] = []
    cursor = last_open
    # Bounded so a bad session definition fails the test rather than hanging.
    for _ in range(count * 24):
        if _bar_is_in_session(cursor, session, intraday=intraday):
            times.append(cursor)
            if len(times) == count:
                break
        cursor -= step
    times.reverse()
    return times


def _bar_is_in_session(
    moment: datetime, session: tuple[int, int] | None, *, intraday: bool
) -> bool:
    if moment.weekday() >= 5:  # Saturday, Sunday
        return False
    if session is None or not intraday:
        return True
    opens, closes = session
    return opens <= moment.hour < closes


def synthetic_rates(
    count: int,
    timeframe: int,
    base_price: float = 1.085,
    end: datetime | None = None,
    session: tuple[int, int] | None = None,
) -> np.ndarray:
    """Deterministic, well-formed OHLCV bars ending at the current forming bar.

    Prices follow a fixed pseudo-random walk (seeded) so tests are reproducible
    and never flake on a lucky or unlucky sequence.

    `session` is `(open_hour, close_hour)` in UTC for instruments that do not
    trade around the clock. Weekends are always skipped. Without it the series
    is continuous, which describes spot FX and almost nothing else in a broker
    catalogue.
    """
    minutes = _minutes_for(timeframe)
    step = timedelta(minutes=minutes)
    last_open = (end or datetime.now(UTC)).replace(second=0, microsecond=0)
    # Align to the timeframe grid *measured from midnight*, which is what a
    # broker actually serves. Aligning to the current minute instead produced a
    # grid that drifted with the call time, so H4 bars landed at 03:00/07:00 on
    # one run and 04:00/08:00 on the next and day boundaries fell inside a bar.
    since_midnight = last_open.hour * 60 + last_open.minute
    aligned = last_open - timedelta(minutes=since_midnight % minutes)

    times = _bar_times(aligned, step, count, session, intraday=minutes < 1440)

    rng = np.random.default_rng(seed=42)
    steps = rng.normal(0.0, base_price * 0.0005, size=count)
    closes = base_price + np.cumsum(steps)
    opens = np.concatenate([[base_price], closes[:-1]])
    wick = np.abs(rng.normal(0.0, base_price * 0.0003, size=count))
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick

    dtype = [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "u8"),
        ("spread", "i4"),
        ("real_volume", "u8"),
    ]
    rows = []
    for i in range(count):
        bar_open = times[i]
        rows.append(
            (
                int(bar_open.timestamp()),
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
                float(closes[i]),
                int(500 + i % 100),
                12,
                0,
            )
        )
    return np.array(rows, dtype=dtype)
