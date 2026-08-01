"""Connector behaviour, especially the failure paths.

These run against `FakeMT5`, not a terminal. That is the point: requotes,
rejections and dropped connections are exactly what you cannot reproduce on
demand against a real broker, and exactly what must be handled correctly.
"""

from __future__ import annotations

import pytest

from config.schema import MT5Config
from core.errors import MT5ConnectionError, OrderFailedError, SymbolNotAvailableError
from core.mt5_codes import Retcode
from core.mt5_connector import MT5Connector
from core.types import Direction, OrderRequest
from tests.fakes.fake_mt5 import FakeMT5, usdjpy_spec


@pytest.fixture
def fast_config() -> MT5Config:
    """Same logic, no real sleeping."""
    return MT5Config(
        max_connect_attempts=3,
        reconnect_backoff_seconds=(0.001, 0.001, 0.001),
        order_max_attempts=3,
        order_retry_delay_ms=0,
    )


@pytest.fixture
def fake() -> FakeMT5:
    return FakeMT5()


@pytest.fixture
def connector(fast_config: MT5Config, fake: FakeMT5) -> MT5Connector:
    return MT5Connector(fast_config, mt5_module=fake)


class TestConnection:
    def test_connect_returns_account_snapshot(self, connector: MT5Connector, fake: FakeMT5) -> None:
        account = connector.connect()
        assert account.login == fake.login_id
        assert account.equity == fake.equity
        assert account.is_demo is True
        assert connector.is_connected

    def test_connect_retries_transient_failures(
        self, fast_config: MT5Config, fake: FakeMT5
    ) -> None:
        fake.initialize_failures = 2
        connector = MT5Connector(fast_config, mt5_module=fake)
        account = connector.connect()
        assert account.login == fake.login_id
        assert sum(1 for name, _ in fake.calls if name == "initialize") == 3

    def test_connect_gives_up_after_max_attempts(
        self, fast_config: MT5Config, fake: FakeMT5
    ) -> None:
        fake.initialize_failures = 99
        connector = MT5Connector(fast_config, mt5_module=fake)
        with pytest.raises(MT5ConnectionError, match="could not connect after 3"):
            connector.connect()

    def test_constant_mismatch_blocks_startup(self, fast_config: MT5Config, fake: FakeMT5) -> None:
        # Simulate a package whose H1 constant no longer matches our mirror.
        fake.TIMEFRAME_H1 = 999  # type: ignore[attr-defined]
        connector = MT5Connector(fast_config, mt5_module=fake)
        with pytest.raises(MT5ConnectionError, match="constants differ"):
            connector.connect()

    def test_ensure_connected_reconnects_after_a_drop(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        connector.connect()
        before = sum(1 for name, _ in fake.calls if name == "initialize")

        fake.connected = False  # broker link dropped, terminal still running
        fake.initialize_failures = 0
        connector.ensure_connected()

        after = sum(1 for name, _ in fake.calls if name == "initialize")
        assert after > before
        assert connector.is_connected

    def test_reconnect_clears_the_spec_cache(self, connector: MT5Connector, fake: FakeMT5) -> None:
        connector.connect()
        connector.spec("EURUSD")
        fake.connected = False
        connector.ensure_connected()
        # A fresh symbol_info call proves the cache was dropped, so a spec that
        # changed while we were disconnected cannot be used stale.
        calls_before = sum(1 for name, _ in fake.calls if name == "symbol_info")
        connector.spec("EURUSD")
        assert sum(1 for name, _ in fake.calls if name == "symbol_info") == calls_before + 1

    def test_context_manager_shuts_down(self, connector: MT5Connector, fake: FakeMT5) -> None:
        with connector:
            assert connector.is_connected
        assert any(name == "shutdown" for name, _ in fake.calls)


class TestSymbols:
    def test_unknown_symbol_names_the_suffix_problem(self, connector: MT5Connector) -> None:
        connector.connect()
        with pytest.raises(SymbolNotAvailableError, match="suffix"):
            connector.spec("NOPEUSD")

    def test_spec_is_cached(self, connector: MT5Connector, fake: FakeMT5) -> None:
        connector.connect()
        connector.spec("EURUSD")
        count = sum(1 for name, _ in fake.calls if name == "symbol_info")
        connector.spec("EURUSD")
        assert sum(1 for name, _ in fake.calls if name == "symbol_info") == count

    def test_refresh_bypasses_the_cache(self, connector: MT5Connector, fake: FakeMT5) -> None:
        connector.connect()
        connector.spec("EURUSD")
        count = sum(1 for name, _ in fake.calls if name == "symbol_info")
        connector.spec("EURUSD", refresh=True)
        assert sum(1 for name, _ in fake.calls if name == "symbol_info") == count + 1

    def test_tick_exposes_the_spread(self, connector: MT5Connector) -> None:
        connector.connect()
        tick = connector.tick("EURUSD")
        assert tick.ask > tick.bid
        assert tick.spread == pytest.approx(0.00012)


class TestOrderExecution:
    def _request(self, **overrides: object) -> OrderRequest:
        payload = {
            "symbol": "EURUSD",
            "direction": Direction.LONG,
            "volume": 0.01,
            "sl": 1.08300,
            "tp": 1.08900,
            "reference_price": 1.08512,
            "magic": 770101,
        }
        payload.update(overrides)
        return OrderRequest(**payload)  # type: ignore[arg-type]

    def test_successful_order_records_execution_detail(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        connector.connect()
        fake.fill_offset = 0.00003  # filled 0.3 pips above the ask
        spec = connector.spec("EURUSD")

        result = connector.order_send(self._request(), spec)

        assert result.ok
        assert result.retcode_name == "DONE"
        assert result.filled_volume == 0.01
        assert result.slippage_pips == pytest.approx(0.3, abs=1e-6)
        assert result.latency_ms >= 0.0
        assert result.spread_at_send == pytest.approx(0.00012)
        assert result.attempts == 1

    def test_slippage_sign_is_direction_aware(self, connector: MT5Connector, fake: FakeMT5) -> None:
        """Positive always means 'worse than requested', whichever way we trade.

        Averaging raw price deltas across longs and shorts would cancel real
        slippage out to roughly zero and make the execution report useless.
        """
        connector.connect()
        spec = connector.spec("EURUSD")

        # Selling 0.3 pips BELOW the requested bid is worse for a short.
        fake.fill_offset = -0.00003
        worse = connector.order_send(
            self._request(direction=Direction.SHORT, sl=1.08700, tp=1.08100), spec
        )
        assert worse.slippage_pips == pytest.approx(0.3, abs=1e-6)

        # Selling ABOVE the requested bid is favourable.
        fake.fill_offset = 0.00003
        better = connector.order_send(
            self._request(direction=Direction.SHORT, sl=1.08700, tp=1.08100), spec
        )
        assert better.slippage_pips == pytest.approx(-0.3, abs=1e-6)

        # And for a long, filling above the ask is the worse case.
        fake.fill_offset = 0.00003
        long_worse = connector.order_send(self._request(), spec)
        assert long_worse.slippage_pips == pytest.approx(0.3, abs=1e-6)

    def test_requote_is_retried(self, connector: MT5Connector, fake: FakeMT5) -> None:
        connector.connect()
        fake.order_retcodes = [int(Retcode.REQUOTE), int(Retcode.PRICE_CHANGED)]
        spec = connector.spec("EURUSD")

        result = connector.order_send(self._request(), spec)

        assert result.ok
        assert result.attempts == 3

    def test_no_money_is_never_retried(self, connector: MT5Connector, fake: FakeMT5) -> None:
        """Retrying an under-margined order is how an account gets ground down."""
        connector.connect()
        fake.order_retcodes = [int(Retcode.NO_MONEY)]
        spec = connector.spec("EURUSD")

        result = connector.order_send(self._request(), spec)

        assert not result.ok
        assert result.retcode_name == "NO_MONEY"
        assert result.attempts == 1
        assert len(fake.orders_sent) == 1

    def test_invalid_stops_is_never_retried(self, connector: MT5Connector, fake: FakeMT5) -> None:
        connector.connect()
        fake.order_retcodes = [int(Retcode.INVALID_STOPS)]
        spec = connector.spec("EURUSD")
        result = connector.order_send(self._request(), spec)
        assert not result.ok
        assert result.attempts == 1

    def test_persistent_requote_gives_up_and_reports(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        connector.connect()
        fake.order_retcodes = [int(Retcode.REQUOTE)] * 10
        spec = connector.spec("EURUSD")

        result = connector.order_send(self._request(), spec)

        assert not result.ok
        assert result.attempts == 3
        assert result.retcode_name == "REQUOTE"

    def test_stop_inside_broker_stop_level_is_refused_before_sending(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        """Better a loud failure than a broker silently moving our stop."""
        connector.connect()
        fake.specs["USDJPY"] = usdjpy_spec(trade_stops_level=200)  # 0.200 = 20 pips
        spec = connector.spec("USDJPY", refresh=True)

        request = self._request(
            symbol="USDJPY",
            direction=Direction.LONG,
            sl=150.090,
            tp=150.400,
            reference_price=150.118,
        )
        with pytest.raises(OrderFailedError, match="stop level"):
            connector.order_send(request, spec)
        assert not fake.orders_sent

    def test_payload_uses_a_supported_filling_mode(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        connector.connect()
        spec = connector.spec("EURUSD")
        connector.order_send(self._request(), spec)
        assert fake.orders_sent[0]["type_filling"] == 1  # IOC

    def test_pre_send_guard_can_abort(self, fast_config: MT5Config, fake: FakeMT5) -> None:
        """The kill switch must stop an order even mid-cycle."""

        class Halt(Exception):
            pass

        def guard() -> None:
            raise Halt("STOP file present")

        connector = MT5Connector(fast_config, mt5_module=fake, pre_send_guard=guard)
        connector.connect()
        spec = connector.spec("EURUSD")

        with pytest.raises(Halt):
            connector.order_send(self._request(), spec)
        assert not fake.orders_sent


class TestOrderRequestValidation:
    """The last line of defence before an order is even constructed."""

    def test_order_without_a_stop_is_impossible(self) -> None:
        with pytest.raises(ValueError, match="without stop loss"):
            OrderRequest(
                symbol="EURUSD",
                direction=Direction.LONG,
                volume=0.01,
                sl=0.0,
                tp=1.09,
                reference_price=1.085,
            )

    def test_long_stop_above_entry_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at or above entry"):
            OrderRequest(
                symbol="EURUSD",
                direction=Direction.LONG,
                volume=0.01,
                sl=1.09000,
                tp=1.09500,
                reference_price=1.08500,
            )

    def test_short_stop_below_entry_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at or below entry"):
            OrderRequest(
                symbol="EURUSD",
                direction=Direction.SHORT,
                volume=0.01,
                sl=1.08000,
                tp=1.07500,
                reference_price=1.08500,
            )

    def test_zero_volume_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="volume must be positive"):
            OrderRequest(
                symbol="EURUSD",
                direction=Direction.LONG,
                volume=0.0,
                sl=1.08300,
                tp=1.08900,
                reference_price=1.08500,
            )
