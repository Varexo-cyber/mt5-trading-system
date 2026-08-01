"""MetaTrader 5 terminal connection and order execution.

Everything that talks to the broker goes through here, and every call obeys
three rules:

1. **Check the return value.** `mt5` functions return `None` on failure and set
   an error code retrievable with `mt5.last_error()`. A `None` that is not
   checked becomes an `AttributeError` three frames later, or worse, a silently
   skipped order. Every call site here checks.
2. **Retry only what is safe to retry.** Requote, timeout and price-changed are
   transient. `NO_MONEY` and `INVALID_STOPS` are not — retrying those is at
   best noise and at worst a way to eventually squeeze in a trade the account
   cannot support.
3. **Record everything.** Requested vs filled price, slippage in pips,
   call latency, spread at send, attempt count, raw return code. Phase 8 exists
   to compare these numbers to the backtest's assumptions, and that comparison
   is only possible if they were captured at the time.

The `MetaTrader5` package is Windows-only. It is imported lazily so that
analysis, backtesting and the whole test suite run on Linux and macOS.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import Any

from config.schema import MT5Config, MT5Credentials
from core.errors import (
    MT5ConnectionError,
    MT5NotAvailableError,
    OrderFailedError,
    SymbolNotAvailableError,
)
from core.instrument import InstrumentSpec
from core.mt5_codes import (
    RETRYABLE_RETCODES,
    SUCCESS_RETCODES,
    TIMEFRAME_VALUES,
    OrderTime,
    OrderType,
    PositionType,
    TradeAction,
    describe_retcode,
)
from core.types import (
    AccountSnapshot,
    Direction,
    OrderRequest,
    OrderResult,
    Position,
    Tick,
)
from infra.logging import get_logger

log = get_logger(__name__)


def import_mt5() -> ModuleType:
    """Import the MetaTrader5 package, with a message that explains the failure.

    Raised as `MT5NotAvailableError` rather than letting `ImportError` escape,
    because "run this on Windows" is actionable and "No module named
    MetaTrader5" prompts people to pip-install something that does not exist for
    their platform.
    """
    try:
        import MetaTrader5 as mt5  # noqa: N813  (upstream package name)
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise MT5NotAvailableError(
            "the MetaTrader5 package is not importable. It ships Windows-only "
            "binaries; live and paper modes require a Windows host (or a Windows "
            "VPS) with the terminal installed. Backtests and tests run without it."
        ) from exc
    return mt5


class MT5Connector:
    """Owns the terminal session: connect, query, execute, reconcile.

    Not thread-safe. The MT5 API is a single global session per process, so a
    second connector in the same process would fight the first one; construct
    exactly one.
    """

    def __init__(
        self,
        config: MT5Config,
        credentials: MT5Credentials | None = None,
        *,
        mt5_module: ModuleType | None = None,
        terminal_path: str = "",
        pre_send_guard: Callable[[], None] | None = None,
    ) -> None:
        """
        Args:
            mt5_module: injected for tests; the real package is imported on
                first use otherwise.
            pre_send_guard: called immediately before every `order_send`. Wired
                to the kill switch so flipping it mid-cycle still stops the
                order. Should raise to abort.
        """
        self.config = config
        self.credentials = credentials
        self.terminal_path = terminal_path or config.terminal_path
        self._mt5 = mt5_module
        self._pre_send_guard = pre_send_guard

        self._connected = False
        self._spec_cache: dict[str, InstrumentSpec] = {}
        self._selected: set[str] = set()
        self._server_offset = timedelta(0)
        self._last_error: tuple[int, str] | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def mt5(self) -> ModuleType:
        if self._mt5 is None:
            self._mt5 = import_mt5()
        return self._mt5

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def server_offset(self) -> timedelta:
        """Measured (broker server time - UTC). Zero until a tick is seen."""
        return self._server_offset

    def connect(self) -> AccountSnapshot:
        """Initialise the terminal and log in, retrying with backoff.

        Returns the account snapshot so the caller can run its startup guard
        against real equity rather than a configured guess.
        """
        backoff = self.config.reconnect_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_connect_attempts + 1):
            try:
                self._initialise()
                account = self.account()
                self._connected = True
                log.info(
                    "connected to MT5",
                    extra={
                        "event": "mt5_connected",
                        "attempt": attempt,
                        "login": account.login,
                        "server": account.server,
                        "currency": account.currency,
                        "equity": account.equity,
                        "is_demo": account.is_demo,
                        "leverage": account.leverage,
                    },
                )
                self._verify_constants()
                return account
            except (MT5ConnectionError, MT5NotAvailableError) as exc:
                last_error = exc
                if isinstance(exc, MT5NotAvailableError):
                    raise  # retrying an import failure is pointless
                delay = backoff[min(attempt - 1, len(backoff) - 1)]
                log.warning(
                    "MT5 connect failed, backing off",
                    extra={
                        "event": "mt5_connect_retry",
                        "attempt": attempt,
                        "max_attempts": self.config.max_connect_attempts,
                        "delay_s": delay,
                        "reason": str(exc),
                    },
                )
                self._safe_shutdown()
                if attempt < self.config.max_connect_attempts:
                    time.sleep(delay)

        raise MT5ConnectionError(
            f"could not connect after {self.config.max_connect_attempts} attempts: {last_error}"
        )

    def _initialise(self) -> None:
        kwargs: dict[str, Any] = {"timeout": self.config.connect_timeout_ms}
        if self.config.portable:
            kwargs["portable"] = True
        if self.credentials is not None:
            kwargs.update(
                login=self.credentials.login,
                password=self.credentials.password,
                server=self.credentials.server,
            )

        ok = (
            self.mt5.initialize(self.terminal_path, **kwargs)
            if self.terminal_path
            else self.mt5.initialize(**kwargs)
        )
        if not ok:
            raise MT5ConnectionError(f"initialize() failed: {self._error_text()}")

        # `initialize` with credentials already logs in; an explicit login is
        # only needed when the terminal was already running under another
        # account. Doing it unconditionally would fail on some brokers.
        info = self.mt5.account_info()
        if info is None:
            raise MT5ConnectionError(f"account_info() returned None: {self._error_text()}")
        wrong_account = self.credentials is not None and int(info.login) != self.credentials.login
        if wrong_account and not self.mt5.login(
            self.credentials.login,  # type: ignore[union-attr]
            password=self.credentials.password,  # type: ignore[union-attr]
            server=self.credentials.server,  # type: ignore[union-attr]
            timeout=self.config.connect_timeout_ms,
        ):
            raise MT5ConnectionError(f"login() failed: {self._error_text()}")

    def _verify_constants(self) -> None:
        """Assert our mirrored constants still match the installed package.

        Cheap insurance: if a terminal update ever renumbered a timeframe or an
        order type, we would otherwise place an H1 analysis's trade off M1 data
        with no error anywhere.
        """
        mismatches: list[str] = []
        for name, value in TIMEFRAME_VALUES.items():
            actual = getattr(self.mt5, f"TIMEFRAME_{name}", None)
            if actual is not None and int(actual) != value:
                mismatches.append(f"TIMEFRAME_{name}: ours={value} package={actual}")
        for enum_name, member in (
            ("ORDER_TYPE_BUY", OrderType.BUY),
            ("ORDER_TYPE_SELL", OrderType.SELL),
            ("TRADE_ACTION_DEAL", TradeAction.DEAL),
            ("TRADE_ACTION_SLTP", TradeAction.SLTP),
        ):
            actual = getattr(self.mt5, enum_name, None)
            if actual is not None and int(actual) != int(member):
                mismatches.append(f"{enum_name}: ours={int(member)} package={actual}")
        if mismatches:
            raise MT5ConnectionError(
                "MT5 constants differ from the mirrored values in core/mt5_codes.py; "
                "refusing to trade against unknown semantics: " + "; ".join(mismatches)
            )

    def ensure_connected(self) -> None:
        """Verify the link is alive, reconnecting once if it is not.

        Called at the top of every cycle. A terminal that is running but has
        lost its server connection still answers `account_info()` with stale
        data on some builds, so `terminal_info().connected` is checked too.
        """
        healthy = False
        if self._connected:
            try:
                info = self.mt5.terminal_info()
                healthy = info is not None and bool(getattr(info, "connected", True))
            except Exception:  # noqa: BLE001 - the package raises assorted types
                healthy = False
        if healthy:
            return

        log.warning("MT5 link unhealthy, reconnecting", extra={"event": "mt5_reconnect"})
        self._connected = False
        self._safe_shutdown()
        self._spec_cache.clear()
        self._selected.clear()
        self.connect()

    def shutdown(self) -> None:
        """Close the terminal session. Safe to call when never connected."""
        self._safe_shutdown()
        self._connected = False
        log.info("MT5 session closed", extra={"event": "mt5_shutdown"})

    def _safe_shutdown(self) -> None:
        try:
            if self._mt5 is not None:
                self._mt5.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must never raise
            log.debug("shutdown() raised, ignoring", extra={"reason": str(exc)})

    def __enter__(self) -> MT5Connector:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    # -- account and positions ---------------------------------------------

    def account(self) -> AccountSnapshot:
        info = self._call("account_info")
        if info is None:
            raise MT5ConnectionError(f"account_info() returned None: {self._error_text()}")
        # trade_mode: 0=demo, 1=contest, 2=real
        return AccountSnapshot(
            login=int(info.login),
            server=str(info.server),
            currency=str(info.currency),
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            margin_free=float(info.margin_free),
            margin_level=float(info.margin_level),
            leverage=int(info.leverage),
            is_demo=int(getattr(info, "trade_mode", 0)) != 2,
            taken_at=datetime.now(UTC),
        )

    def positions(self, symbol: str | None = None, magic: int | None = None) -> list[Position]:
        """Open positions as the terminal reports them.

        This is the source of truth for reconciliation: what MT5 says beats
        what the system believes, every time.
        """
        raw = self._call("positions_get", symbol=symbol) if symbol else self._call("positions_get")
        if raw is None:
            # `positions_get` returns None both for "error" and, on some builds,
            # for "no positions". Distinguish via last_error.
            code, _ = self._last_error_tuple()
            if code == 0:
                return []
            raise MT5ConnectionError(f"positions_get() failed: {self._error_text()}")

        out: list[Position] = []
        for p in raw:
            if magic is not None and int(p.magic) != magic:
                continue
            out.append(
                Position(
                    ticket=int(p.ticket),
                    symbol=str(p.symbol),
                    direction=(
                        Direction.LONG if int(p.type) == PositionType.BUY else Direction.SHORT
                    ),
                    volume=float(p.volume),
                    price_open=float(p.price_open),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    profit=float(p.profit),
                    swap=float(p.swap),
                    opened_at=datetime.fromtimestamp(int(p.time), tz=UTC),
                    magic=int(p.magic),
                    comment=str(p.comment),
                )
            )
        return out

    # -- symbols -----------------------------------------------------------

    def select(self, symbol: str) -> None:
        """Add a symbol to Market Watch. Required before any data call.

        A symbol that is not selected returns empty rates and a zero tick value,
        which would otherwise surface as "no data" or a division by zero deep
        inside the sizer.
        """
        if symbol in self._selected:
            return
        if not self._call("symbol_select", symbol, True):
            raise SymbolNotAvailableError(
                f"{symbol}: symbol_select failed ({self._error_text()}). Check the "
                f"broker's exact symbol name — many use a suffix such as "
                f"'{symbol}.pro' or '{symbol}m' (see instruments.symbol_suffix)."
            )
        self._selected.add(symbol)

    def spec(self, symbol: str, *, refresh: bool = False) -> InstrumentSpec:
        """Contract specification, cached for the session.

        Cached deliberately: a broker widening the lot step between the sizing
        calculation and the order would otherwise change the meaning of a
        volume we already validated. `refresh=True` re-reads it explicitly.
        """
        if not refresh and symbol in self._spec_cache:
            return self._spec_cache[symbol]

        self.select(symbol)
        info = self._call("symbol_info", symbol)
        if info is None:
            raise SymbolNotAvailableError(f"{symbol}: symbol_info returned None")
        spec = InstrumentSpec.from_mt5(info)
        self._spec_cache[symbol] = spec
        log.debug("symbol spec loaded", extra={"event": "spec", "spec": spec.describe()})
        return spec

    def tick(self, symbol: str) -> Tick:
        self.select(symbol)
        raw = self._call("symbol_info_tick", symbol)
        if raw is None or float(raw.ask) <= 0:
            raise SymbolNotAvailableError(f"{symbol}: no tick available ({self._error_text()})")
        moment = datetime.fromtimestamp(int(raw.time), tz=UTC)
        self._update_server_offset(moment)
        return Tick(
            symbol=symbol,
            time=moment,
            bid=float(raw.bid),
            ask=float(raw.ask),
            last=float(getattr(raw, "last", 0.0)),
            volume=int(getattr(raw, "volume", 0)),
        )

    def _update_server_offset(self, server_time: datetime) -> None:
        """Estimate broker-server-to-UTC offset from a fresh tick timestamp.

        Rounded to whole hours: the real offset is always a whole number of
        hours, and rounding removes the tick's own latency from the estimate.
        """
        delta = server_time - datetime.now(UTC)
        hours = round(delta.total_seconds() / 3600.0)
        offset = timedelta(hours=hours)
        if offset != self._server_offset:
            log.info(
                "broker server time offset updated",
                extra={"event": "server_offset", "offset_hours": hours},
            )
            self._server_offset = offset

    # -- market data --------------------------------------------------------

    def copy_rates(self, symbol: str, timeframe: int, count: int, start_pos: int = 0) -> Any:
        """`count` bars ending at `start_pos` (0 = the forming bar).

        Returns the raw numpy structured array; `DataManager` owns conversion,
        validation and dropping the unfinished bar.
        """
        self.select(symbol)
        rates = self._call("copy_rates_from_pos", symbol, timeframe, start_pos, count)
        if rates is None or len(rates) == 0:
            raise SymbolNotAvailableError(
                f"{symbol}: no rates for timeframe {timeframe} ({self._error_text()})"
            )
        return rates

    def copy_rates_range(self, symbol: str, timeframe: int, start: datetime, end: datetime) -> Any:
        """Bars in [start, end]. Used by the backtester and the data warm-up."""
        self.select(symbol)
        rates = self._call("copy_rates_range", symbol, timeframe, start, end)
        if rates is None:
            raise SymbolNotAvailableError(
                f"{symbol}: copy_rates_range failed ({self._error_text()})"
            )
        return rates

    # -- execution ----------------------------------------------------------

    def order_send(self, request: OrderRequest, spec: InstrumentSpec) -> OrderResult:
        """Send a market order, retrying only transient rejections.

        The returned `OrderResult` is complete whether the order succeeded or
        not — a rejection is data, and Phase 8's execution report is built from
        exactly these records.
        """
        if self._pre_send_guard is not None:
            self._pre_send_guard()

        attempts = 0
        last: OrderResult | None = None
        sent_at = datetime.now(UTC)

        while attempts < self.config.order_max_attempts:
            attempts += 1
            tick = self.tick(request.symbol)
            price = tick.ask if request.direction is Direction.LONG else tick.bid
            payload = self._build_deal_payload(request, spec, price)

            started = time.perf_counter()
            raw = self._call("order_send", payload)
            latency_ms = (time.perf_counter() - started) * 1000.0

            last = self._to_order_result(
                raw=raw,
                request=request,
                spec=spec,
                requested_price=price,
                spread=tick.spread,
                latency_ms=latency_ms,
                attempts=attempts,
                sent_at=sent_at,
            )

            log.info(
                "order attempt",
                extra={
                    "event": "order_attempt",
                    "symbol": request.symbol,
                    "direction": request.direction.name,
                    "attempt": attempts,
                    "requested_volume": request.volume,
                    "filled_volume": last.filled_volume,
                    "requested_price": price,
                    "filled_price": last.filled_price,
                    "slippage_pips": round(last.slippage_pips, 3),
                    "sl": payload["sl"],
                    "tp": payload["tp"],
                    "spread": round(tick.spread, spec.digits),
                    "latency_ms": round(latency_ms, 1),
                    "retcode": last.retcode,
                    "retcode_name": last.retcode_name,
                    "broker_comment": last.comment,
                },
            )

            if last.ok:
                return last
            if last.retcode not in RETRYABLE_RETCODES:
                break
            if attempts < self.config.order_max_attempts:
                time.sleep(self.config.order_retry_delay_ms / 1000.0)

        assert last is not None  # loop runs at least once
        log.error(
            "order rejected",
            extra={
                "event": "order_rejected",
                "symbol": request.symbol,
                "retcode": last.retcode,
                "retcode_name": last.retcode_name,
                "broker_comment": last.comment,
                "attempts": attempts,
            },
        )
        return last

    def _build_deal_payload(
        self, request: OrderRequest, spec: InstrumentSpec, price: float
    ) -> dict[str, Any]:
        sl = spec.normalize_price(request.sl)
        tp = spec.normalize_price(request.tp) if request.tp else 0.0

        # The broker rejects stops closer than `stops_level`. Failing loudly
        # here is right: silently widening the stop would change the risk the
        # sizer computed, and silently narrowing the target changes the R:R the
        # setup was accepted on.
        if spec.violates_stop_level(price, sl):
            raise OrderFailedError(
                f"{request.symbol}: stop {sl} is {abs(price - sl) / spec.point:.0f} points from "
                f"price {price}, inside the broker's {spec.stops_level}-point stop level"
            )

        return {
            "action": int(TradeAction.DEAL),
            "symbol": request.symbol,
            "volume": float(request.volume),
            "type": int(OrderType.BUY if request.direction is Direction.LONG else OrderType.SELL),
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(request.deviation_points or self.config.deviation_points),
            "magic": int(request.magic),
            "comment": request.comment[:31],  # MT5 truncates silently past 31 chars
            "type_time": int(OrderTime.GTC),
            "type_filling": int(spec.preferred_filling()),
        }

    def _to_order_result(
        self,
        *,
        raw: Any,
        request: OrderRequest,
        spec: InstrumentSpec,
        requested_price: float,
        spread: float,
        latency_ms: float,
        attempts: int,
        sent_at: datetime,
    ) -> OrderResult:
        if raw is None:
            return OrderResult(
                ok=False,
                retcode=None,
                retcode_name="NO_RESULT",
                comment=self._error_text(),
                order_ticket=None,
                deal_ticket=None,
                position_ticket=None,
                requested_volume=request.volume,
                filled_volume=0.0,
                requested_price=requested_price,
                filled_price=0.0,
                slippage_pips=0.0,
                latency_ms=latency_ms,
                spread_at_send=spread,
                attempts=attempts,
                sent_at=sent_at,
            )

        retcode = int(raw.retcode)
        filled_price = float(getattr(raw, "price", 0.0) or 0.0)
        ok = retcode in SUCCESS_RETCODES

        # Signed so that positive always means "worse than we asked for",
        # whichever way the trade points. Averaging raw price deltas across
        # longs and shorts would cancel real slippage out to nearly zero.
        slippage = 0.0
        if ok and filled_price > 0:
            raw_delta = filled_price - requested_price
            slippage = spec.price_to_pips(raw_delta * int(request.direction))

        return OrderResult(
            ok=ok,
            retcode=retcode,
            retcode_name=describe_retcode(retcode),
            comment=str(getattr(raw, "comment", "")),
            order_ticket=int(getattr(raw, "order", 0)) or None,
            deal_ticket=int(getattr(raw, "deal", 0)) or None,
            position_ticket=int(getattr(raw, "order", 0)) or None,
            requested_volume=request.volume,
            filled_volume=float(getattr(raw, "volume", 0.0) or 0.0),
            requested_price=requested_price,
            filled_price=filled_price,
            slippage_pips=slippage,
            latency_ms=latency_ms,
            spread_at_send=spread,
            attempts=attempts,
            sent_at=sent_at,
        )

    def modify_stops(self, position: Position, *, sl: float, tp: float) -> OrderResult:
        """Move SL/TP on an existing position (break-even, trailing, partials)."""
        if self._pre_send_guard is not None:
            self._pre_send_guard()

        spec = self.spec(position.symbol)
        payload = {
            "action": int(TradeAction.SLTP),
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": float(spec.normalize_price(sl)),
            "tp": float(spec.normalize_price(tp)) if tp else 0.0,
            "magic": position.magic,
        }
        started = time.perf_counter()
        raw = self._call("order_send", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        retcode = int(raw.retcode) if raw is not None else None
        ok = retcode in SUCCESS_RETCODES if retcode is not None else False
        log.info(
            "stops modified" if ok else "stop modification rejected",
            extra={
                "event": "modify_stops",
                "ticket": position.ticket,
                "symbol": position.symbol,
                "sl": payload["sl"],
                "tp": payload["tp"],
                "retcode": retcode,
                "retcode_name": describe_retcode(retcode),
                "latency_ms": round(latency_ms, 1),
            },
        )
        return OrderResult(
            ok=ok,
            retcode=retcode,
            retcode_name=describe_retcode(retcode),
            comment=str(getattr(raw, "comment", "")) if raw is not None else self._error_text(),
            order_ticket=position.ticket,
            deal_ticket=None,
            position_ticket=position.ticket,
            requested_volume=position.volume,
            filled_volume=position.volume,
            requested_price=position.price_open,
            filled_price=position.price_open,
            slippage_pips=0.0,
            latency_ms=latency_ms,
            spread_at_send=0.0,
            attempts=1,
            sent_at=datetime.now(UTC),
        )

    def close_position(self, position: Position, volume: float | None = None) -> OrderResult:
        """Close all or part of a position at market."""
        if self._pre_send_guard is not None:
            self._pre_send_guard()

        spec = self.spec(position.symbol)
        tick = self.tick(position.symbol)
        closing_long = position.direction is Direction.LONG
        price = tick.bid if closing_long else tick.ask
        close_volume = spec.round_volume_down(volume) if volume else position.volume

        payload = {
            "action": int(TradeAction.DEAL),
            "symbol": position.symbol,
            "position": position.ticket,
            "volume": float(close_volume),
            "type": int(OrderType.SELL if closing_long else OrderType.BUY),
            "price": float(price),
            "deviation": self.config.deviation_points,
            "magic": position.magic,
            "comment": "close",
            "type_time": int(OrderTime.GTC),
            "type_filling": int(spec.preferred_filling()),
        }

        started = time.perf_counter()
        raw = self._call("order_send", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        retcode = int(raw.retcode) if raw is not None else None
        ok = retcode in SUCCESS_RETCODES if retcode is not None else False
        filled_price = float(getattr(raw, "price", 0.0) or 0.0) if raw is not None else 0.0

        # Exit slippage is signed the same way as entry: positive is worse. On a
        # close, "worse" is the mirror of the position's direction.
        slippage = 0.0
        if ok and filled_price > 0:
            slippage = spec.price_to_pips((price - filled_price) * int(position.direction))

        log.info(
            "position closed" if ok else "close rejected",
            extra={
                "event": "close_position",
                "ticket": position.ticket,
                "symbol": position.symbol,
                "volume": close_volume,
                "requested_price": price,
                "filled_price": filled_price,
                "slippage_pips": round(slippage, 3),
                "retcode": retcode,
                "retcode_name": describe_retcode(retcode),
                "latency_ms": round(latency_ms, 1),
            },
        )
        return OrderResult(
            ok=ok,
            retcode=retcode,
            retcode_name=describe_retcode(retcode),
            comment=str(getattr(raw, "comment", "")) if raw is not None else self._error_text(),
            order_ticket=int(getattr(raw, "order", 0)) or None if raw is not None else None,
            deal_ticket=int(getattr(raw, "deal", 0)) or None if raw is not None else None,
            position_ticket=position.ticket,
            requested_volume=close_volume,
            filled_volume=float(getattr(raw, "volume", 0.0) or 0.0) if raw is not None else 0.0,
            requested_price=price,
            filled_price=filled_price,
            slippage_pips=slippage,
            latency_ms=latency_ms,
            spread_at_send=tick.spread,
            attempts=1,
            sent_at=datetime.now(UTC),
        )

    # -- plumbing -----------------------------------------------------------

    def _call(self, function: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke an mt5 function, timing it and clearing stale error state."""
        func = getattr(self.mt5, function)
        started = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > self.config.slow_call_warn_ms:
            log.warning(
                "slow MT5 call",
                extra={"event": "mt5_slow_call", "function": function, "ms": round(elapsed_ms, 1)},
            )
        return result

    def _last_error_tuple(self) -> tuple[int, str]:
        try:
            code, description = self.mt5.last_error()
            return int(code), str(description)
        except Exception:  # noqa: BLE001 - never let error reporting raise
            return -1, "last_error() unavailable"

    def _error_text(self) -> str:
        code, description = self._last_error_tuple()
        return f"[{code}] {description}"
