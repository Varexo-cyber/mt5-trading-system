"""Domain types shared by every layer.

Everything here is immutable (`frozen=True`) on purpose: a `Signal` or a
`MarketContext` that a module can mutate in place makes the journal unreliable,
because what we logged is no longer what was decided on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum, StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from core.mt5_codes import TIMEFRAME_VALUES

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class Timeframe(Enum):
    """Timeframes we support, carrying their MT5 constant and bar duration."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    @property
    def mt5_value(self) -> int:
        return TIMEFRAME_VALUES[self.value]

    @property
    def duration(self) -> timedelta:
        """Nominal bar duration. MN1 is approximated at 30 days.

        Used for staleness budgets and gap detection only, never for
        calendar arithmetic.
        """
        minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN1": 43200,
        }[self.value]
        return timedelta(minutes=minutes)

    @classmethod
    def parse(cls, value: str | Timeframe) -> Timeframe:
        if isinstance(value, Timeframe):
            return value
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported timeframe {value!r}; expected one of "
                f"{', '.join(tf.value for tf in cls)}"
            ) from exc

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Direction(IntEnum):
    """Trade direction. Values chosen so `score * direction` stays meaningful."""

    LONG = 1
    SHORT = -1

    @property
    def opposite(self) -> Direction:
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class TradingMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    MICRO_LIVE = "micro_live"
    SCALING = "scaling"

    @property
    def is_live(self) -> bool:
        return self in (TradingMode.MICRO_LIVE, TradingMode.SCALING)


@dataclass(frozen=True, slots=True)
class Tick:
    """A single quote. `time` is broker server time, tz-aware UTC."""

    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: int = 0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2.0


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. `time` is the bar's OPEN time, tz-aware UTC."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int = 0
    real_volume: int = 0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass(frozen=True, slots=True)
class Series:
    """A validated, gap-checked frame of CLOSED bars for one symbol/timeframe.

    `df` is indexed by bar open time (tz-aware UTC, strictly increasing) with
    columns open/high/low/close/tick_volume/spread/real_volume.

    Invariant enforced by `DataManager`: the currently forming bar is never
    included. Analysis modules may therefore use the last row freely without
    introducing look-ahead bias.
    """

    symbol: str
    timeframe: Timeframe
    df: pd.DataFrame
    fetched_at: datetime

    def __len__(self) -> int:
        return len(self.df)

    @property
    def last_bar_time(self) -> datetime:
        return self.df.index[-1].to_pydatetime()

    @property
    def last_close(self) -> float:
        return float(self.df["close"].iloc[-1])


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Everything an analysis module is allowed to look at.

    Modules receive this and nothing else — no connector, no clock, no
    filesystem. That constraint is what makes them replayable in a backtest and
    unit-testable without a terminal.
    """

    symbol: str
    now: datetime
    series: dict[Timeframe, Series]
    tick: Tick | None = None
    #: Populated from Phase 3 onward (session, ATR regime, time-to-news, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def bars(self, timeframe: Timeframe) -> Series:
        try:
            return self.series[timeframe]
        except KeyError as exc:
            raise KeyError(
                f"{timeframe} not loaded for {self.symbol}; "
                f"available: {sorted(tf.value for tf in self.series)}"
            ) from exc


@dataclass(frozen=True, slots=True)
class Signal:
    """Output of one analysis module.

    score       -100 (strongly bearish) .. +100 (strongly bullish); 0 = neutral
    confidence  0..1 — how much the module trusts its own read right now
    reasoning   human-readable, goes verbatim into the journal
    """

    module: str
    score: float
    confidence: float
    reasoning: str = ""
    key_levels: tuple[float, ...] = ()
    invalidation_price: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -100.0 <= self.score <= 100.0:
            raise ValueError(f"{self.module}: score {self.score} outside [-100, 100]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.module}: confidence {self.confidence} outside [0, 1]")

    @classmethod
    def neutral(cls, module: str, reasoning: str = "no read") -> Signal:
        """The default a module returns when it has nothing to say.

        Modules must return this rather than a weak directional score when the
        setup they look for is absent. "No opinion" and "slightly bullish" are
        different statements and the confluence engine treats them differently.
        """
        return cls(module=module, score=0.0, confidence=0.0, reasoning=reasoning)


class AnalysisModule(Protocol):
    """Interface every analysis module implements (Phase 4+)."""

    name: str

    def analyze(self, ctx: MarketContext) -> Signal: ...


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Account state at one instant, as reported by the terminal."""

    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    leverage: int
    is_demo: bool
    taken_at: datetime

    @property
    def margin_used_pct(self) -> float:
        return 0.0 if self.equity <= 0 else 100.0 * self.margin / self.equity


@dataclass(frozen=True, slots=True)
class Position:
    """An open position as MT5 reports it (source of truth for reconciliation)."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float
    opened_at: datetime
    magic: int = 0
    comment: str = ""

    @property
    def has_stop(self) -> bool:
        """A position without a stop loss is a hard rule violation."""
        return self.sl != 0.0


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """A market order we intend to send. Built by the risk layer, not strategy."""

    symbol: str
    direction: Direction
    volume: float
    sl: float
    tp: float
    #: Price the decision was based on; used to measure slippage after the fill.
    reference_price: float
    deviation_points: int = 10
    magic: int = 0
    comment: str = ""

    def __post_init__(self) -> None:
        # Hard rule: no order without a stop loss ever leaves this system.
        if self.sl <= 0.0:
            raise ValueError(f"{self.symbol}: order without stop loss is forbidden")
        if self.volume <= 0.0:
            raise ValueError(f"{self.symbol}: volume must be positive, got {self.volume}")
        if self.direction is Direction.LONG and self.sl >= self.reference_price:
            raise ValueError(
                f"{self.symbol}: long stop {self.sl} is at or above entry "
                f"{self.reference_price}"
            )
        if self.direction is Direction.SHORT and self.sl <= self.reference_price:
            raise ValueError(
                f"{self.symbol}: short stop {self.sl} is at or below entry "
                f"{self.reference_price}"
            )


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Outcome of an order attempt. Every field here feeds EXECUTION_REPORT.md."""

    ok: bool
    retcode: int | None
    retcode_name: str
    comment: str
    order_ticket: int | None
    deal_ticket: int | None
    position_ticket: int | None
    requested_volume: float
    filled_volume: float
    requested_price: float
    filled_price: float
    #: Signed, in the instrument's pips. Positive = worse than requested.
    slippage_pips: float
    #: Wall-clock time of the `order_send` call itself.
    latency_ms: float
    spread_at_send: float
    attempts: int
    sent_at: datetime
