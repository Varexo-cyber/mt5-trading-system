"""Connector behaviour, especially the failure paths.

These run against `FakeMT5`, not a terminal. That is the point: requotes,
rejections and dropped connections are exactly what you cannot reproduce on
demand against a real broker, and exactly what must be handled correctly.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

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

    def test_transient_read_ipc_failure_is_retried(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        connector.connect()
        original = fake.copy_rates_from_pos
        calls = 0

        def flaky(*args):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if calls == 1:
                fake.error = (-10001, "IPC send failed")
                return None
            return original(*args)

        fake.copy_rates_from_pos = flaky  # type: ignore[method-assign]

        assert connector.copy_rates("EURUSD", 60, 10) is not None
        assert calls == 2

    def test_non_transient_read_failure_is_not_retried(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        connector.connect()
        calls = 0

        def broken(*_args):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            fake.error = (-10002, "unknown symbol")

        fake.copy_rates_from_pos = broken  # type: ignore[method-assign]

        with pytest.raises(Exception, match="no rates"):
            connector.copy_rates("EURUSD", 60, 10)
        assert calls == 1

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
    def test_catalogue_exposes_path_and_description(self, connector: MT5Connector) -> None:
        connector.connect()
        symbols = connector.symbols()

        eurusd = next(item for item in symbols if item.name == "EURUSD")
        assert eurusd.path.startswith("Forex")
        assert "Euro" in eurusd.description

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

    def test_tick_normalises_a_broker_server_timezone(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        fake.now = datetime.now(UTC) + timedelta(hours=3)
        connector.connect()

        tick = connector.tick("EURUSD")

        assert connector.server_offset == timedelta(hours=3)
        assert abs((tick.time - datetime.now(UTC)).total_seconds()) < 2

    def test_stale_weekend_tick_does_not_corrupt_server_offset(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        fake.now = datetime.now(UTC) - timedelta(hours=16, minutes=37)
        connector.connect()

        connector.tick("EURUSD")

        assert connector.server_offset == timedelta(0)

    def test_a_closed_exchange_does_not_drag_the_offset_around(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        """Regression: the offset flapped many times a second during a full scan.

        Scanning the whole catalogue reads instruments whose exchange is shut.
        A quote exactly a few whole hours stale passes the residual filter and
        is indistinguishable from a timezone, so the offset oscillated between
        the real value and whatever the last closed market implied — silently
        moving every session boundary that depends on it.
        """
        fake.now = datetime.now(UTC) + timedelta(hours=3)  # broker runs at UTC+3
        connector.connect()
        connector.tick("EURUSD")
        assert connector.server_offset == timedelta(hours=3)

        # A stock whose exchange closed two hours ago, then a fresh FX quote,
        # alternating the way a scan interleaves them.
        for _ in range(5):
            fake.now = datetime.now(UTC) + timedelta(hours=1)
            connector.tick("EURUSD")
            assert connector.server_offset == timedelta(hours=3)

            fake.now = datetime.now(UTC) + timedelta(hours=3)
            connector.tick("EURUSD")
            assert connector.server_offset == timedelta(hours=3)

    def test_repetition_alone_never_drags_the_offset_down(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        """Volume of stale quotes must not be mistaken for evidence.

        This is what a full scan looks like: hundreds of closed markets in a
        row. Any rule that counts consecutive lower readings is satisfied by
        stale data alone, which is why the offset is held by time instead.
        """
        fake.now = datetime.now(UTC) + timedelta(hours=3)
        connector.connect()
        connector.tick("EURUSD")

        fake.now = datetime.now(UTC) + timedelta(hours=2)
        for _ in range(200):
            connector.tick("EURUSD")

        assert connector.server_offset == timedelta(hours=3)

    def test_an_unconfirmed_offset_expires_so_dst_can_take_effect(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        """A DST change really does move the broker back an hour, permanently.

        What distinguishes it from a stale quote is not how often it is seen but
        that nothing agrees with the old value any more.
        """
        fake.now = datetime.now(UTC) + timedelta(hours=3)
        connector.connect()
        connector.tick("EURUSD")
        assert connector.server_offset == timedelta(hours=3)

        # Nothing has reconfirmed +3 for longer than the hold window.
        connector._offset_confirmed_at = datetime.now(UTC) - timedelta(minutes=20)
        fake.now = datetime.now(UTC) + timedelta(hours=2)
        connector.tick("EURUSD")

        assert connector.server_offset == timedelta(hours=2)


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

    def test_missing_result_price_is_recovered_from_deal_history(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        """A broker-side zero must not become a fake zero-slippage fill."""
        connector.connect()
        fake.fill_offset = 0.00003
        fake.zero_result_price = True
        spec = connector.spec("EURUSD")

        result = connector.order_send(self._request(), spec)

        assert result.ok
        assert result.filled_price == pytest.approx(1.08515)
        assert result.filled_volume == pytest.approx(0.01)
        assert result.position_ticket == 555_002
        assert result.slippage_pips == pytest.approx(0.3, abs=1e-6)
        assert any(name == "history_deals_get" for name, _ in fake.calls)

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


class TestBrokerAccounting:
    def test_closed_position_recovers_exact_net_deal_result(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        moment = datetime.now(UTC)
        fake.deals = [
            SimpleNamespace(
                ticket=100,
                position_id=42,
                entry=0,
                reason=3,
                time=int((moment - timedelta(hours=2)).timestamp()),
                time_msc=int((moment - timedelta(hours=2)).timestamp() * 1000),
                volume=0.01,
                price=1.085,
                profit=0.0,
                commission=-0.04,
                swap=0.0,
                fee=0.0,
                symbol="EURUSD",
            ),
            SimpleNamespace(
                ticket=101,
                position_id=42,
                entry=1,
                reason=4,
                time=int(moment.timestamp()),
                time_msc=int(moment.timestamp() * 1000),
                volume=0.01,
                price=1.083,
                profit=-2.0,
                commission=-0.04,
                swap=-0.1,
                fee=-0.02,
                symbol="EURUSD",
            ),
        ]
        connector.connect()

        closed = connector.closed_position(42)

        assert closed is not None
        assert closed.exit_price == pytest.approx(1.083)
        assert closed.pnl_money == pytest.approx(-2.20)
        assert closed.reason == "SL"
        assert closed.deal_tickets == (101,)

    def test_margin_uses_broker_calculation(self, connector: MT5Connector, fake: FakeMT5) -> None:
        fake.margin_required = 120.0
        connector.connect()

        required = connector.estimate_margin("EURUSD", Direction.LONG, 0.05, 1.085)

        assert required == pytest.approx(6.0)
        assert any(name == "order_calc_margin" for name, _ in fake.calls)


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


def test_a_call_without_keywords_passes_no_keyword_mapping() -> None:
    """Regression: sixty orders died on an empty `**kwargs`.

    `func(*args, **kwargs)` hands the C extension a keyword mapping even when
    kwargs is empty, and MetaTrader5's `order_send` rejects the presence of one
    outright — "[-2] Unnamed arguments not allowed". The request was valid the
    whole time: order_check returned retcode 0 on the identical payload, and
    calling `mt5.order_send(payload)` directly placed a real trade.
    """
    seen: dict[str, object] = {}

    class Strict:
        def order_send(self, *args: object, **kwargs: object) -> str:
            if kwargs:
                raise TypeError("Unnamed arguments not allowed")
            seen["args"] = args
            return "sent"

        def history_deals_get(self, *args: object, **kwargs: object) -> str:
            seen["kwargs"] = kwargs
            return "history"

    connector = MT5Connector(MT5Config(), mt5_module=Strict())  # type: ignore[arg-type]

    assert connector._call("order_send", {"symbol": "EURUSD"}) == "sent"
    assert seen["args"] == ({"symbol": "EURUSD"},)
    # Calls that genuinely need keywords must still get them.
    assert connector._call("history_deals_get", position=7) == "history"
    assert seen["kwargs"] == {"position": 7}


class TestTheOffsetIsKnownBeforeAnyPositionIsRead:
    """Positions are read first on a fresh process, and the offset starts at zero.

    `_update_server_offset` runs only from `tick()`. The runner reads open
    positions before ticking anything, so a broker clock three hours ahead
    wrote an `opened_at` three hours in the future -- into the journal, and
    afterwards into a price request the terminal answered with a bare "Call
    failed" every fifteen minutes for the life of the row:

        copy_rates_range(start=...T14:40:08+00:00, end=...T12:21:23+00:00)

    It cannot be repaired downstream. The gap between such a timestamp and now
    is the offset minus the position's age, and that age is the very thing
    being read, so no arithmetic recovers the offset from one value. It has to
    be known first.
    """

    def test_connecting_establishes_the_offset(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        fake.now = datetime.now(UTC) + timedelta(hours=3)

        connector.connect()

        assert connector.server_offset == timedelta(hours=3)

    def test_a_position_read_straight_after_connecting_is_in_utc(
        self, connector: MT5Connector, fake: FakeMT5
    ) -> None:
        """The whole point, expressed as the thing that was broken."""
        fake.now = datetime.now(UTC) + timedelta(hours=3)
        connector.connect()

        opened_forty_minutes_ago = int(
            (datetime.now(UTC) + timedelta(hours=3) - timedelta(minutes=40)).timestamp()
        )
        moment = connector._normalise_mt5_timestamp(opened_forty_minutes_ago)

        assert moment < datetime.now(UTC), "a trade cannot have opened in the future"
        assert abs((datetime.now(UTC) - moment).total_seconds() - 2400) < 120

    def test_a_future_timestamp_is_reported_rather_than_quietly_passed_on(
        self, connector: MT5Connector, fake: FakeMT5, caplog
    ) -> None:
        """If one still arrives, something bypassed the priming and the journal
        is about to be handed a trade that opens after now."""
        connector.connect()
        connector._server_offset = timedelta(0)

        with caplog.at_level(logging.ERROR):
            connector._normalise_mt5_timestamp(
                int((datetime.now(UTC) + timedelta(hours=3)).timestamp())
            )

        assert any("lands in the future" in record.message for record in caplog.records)

    def test_an_ordinary_past_timestamp_says_nothing(
        self, connector: MT5Connector, fake: FakeMT5, caplog
    ) -> None:
        connector.connect()

        with caplog.at_level(logging.ERROR):
            moment = connector._normalise_mt5_timestamp(
                int((datetime.now(UTC) - timedelta(hours=2)).timestamp())
            )

        assert abs((datetime.now(UTC) - moment).total_seconds() - 7200) < 5
        assert not [r for r in caplog.records if "lands in the future" in r.message]
