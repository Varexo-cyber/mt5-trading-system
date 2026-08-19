from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.loader import load_settings
from config.schema import MT5Config
from core.clock import SimulatedClock
from core.mt5_connector import MT5Connector
from core.types import Direction, OrderRequest
from execution.manager import PositionManager
from execution.paper_broker import PaperBroker
from runner.service import JarvisRunner, OperationMode
from scanner.universe import UniverseScanner
from tests.fakes.fake_mt5 import FakeMT5, eurusd_spec


def connector(fake: FakeMT5) -> MT5Connector:
    return MT5Connector(MT5Config(), mt5_module=fake)


def test_runner_promotes_filter_context_into_queryable_journal_columns() -> None:
    context = JarvisRunner._journal_cycle_context(
        "EURUSD",
        100.0,
        {
            "session": "london",
            "spread_pips": 0.8,
            "minutes_to_news": 75.0,
            "market_intelligence": {"regime": "trend_up"},
        },
    )

    assert context.session == "london"
    assert context.spread_pips == pytest.approx(0.8)
    assert context.minutes_to_news == pytest.approx(75.0)
    assert context.volatility_regime == "trend_up"


def test_paper_position_survives_restart(tmp_path: Path) -> None:
    fake = FakeMT5()
    market = connector(fake)
    state = tmp_path / "paper.json"
    paper = PaperBroker(market, state)
    paper.connect()
    spec = paper.spec("EURUSD")
    result = paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512), spec
    )
    paper.shutdown()

    restored = PaperBroker(connector(FakeMT5()), state)
    restored.connect()
    positions = restored.positions()

    assert result.ok
    assert len(positions) == 1
    assert positions[0].ticket == result.position_ticket
    restored.shutdown()


def test_paper_broker_uses_the_injected_clock_for_auditable_events(tmp_path: Path) -> None:
    moment = datetime(2026, 8, 9, 12, 34, tzinfo=UTC)
    paper = PaperBroker(
        connector(FakeMT5()),
        tmp_path / "paper.json",
        clock=SimulatedClock(moment),
    )
    paper.connect()
    result = paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512),
        paper.spec("EURUSD"),
    )

    assert result.sent_at == moment
    assert paper.positions()[0].opened_at == moment
    assert paper.account().taken_at == moment
    paper.shutdown()


def test_paper_stop_closes_and_changes_balance(tmp_path: Path) -> None:
    fake = FakeMT5()
    paper = PaperBroker(connector(fake), tmp_path / "paper.json")
    paper.connect()
    spec = paper.spec("EURUSD")
    paper.order_send(OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512), spec)
    fake.quotes["EURUSD"] = (1.079, 1.07912)

    events = paper.mark_to_market()

    assert events[0][1] == "SL"
    assert not paper.positions()
    assert paper.account().balance < 100
    paper.shutdown()


def test_paper_closed_position_survives_crash_window(tmp_path: Path) -> None:
    fake = FakeMT5()
    state = tmp_path / "paper.json"
    paper = PaperBroker(connector(fake), state)
    paper.connect()
    spec = paper.spec("EURUSD")
    result = paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.01, 1.08, 1.10, 1.08512), spec
    )
    fake.quotes["EURUSD"] = (1.079, 1.07912)
    paper.mark_to_market()
    paper.shutdown()

    restored = PaperBroker(connector(FakeMT5()), state)
    restored.connect()
    closed = restored.closed_position(int(result.position_ticket or 0))

    assert closed is not None
    assert closed.reason == "SL"
    assert closed.exit_price == 1.079
    assert closed.pnl_money < 0
    restored.shutdown()


def test_partial_close_is_persistent_and_recoverable(tmp_path: Path) -> None:
    class JournalStub:
        """Enough journal for the manager, including the excursion ratchet.

        `mfe_r` is real state here rather than a constant: the give-back rule
        reads it back on the next pass, and a stub that always returned zero
        would quietly disable it in every test that uses this double.
        """

        def __init__(self) -> None:
            self.peak_r = 0.0

        def open_trade_by_ticket(self, ticket: int):  # type: ignore[no-untyped-def]
            return {"id": 1, "ticket": ticket, "sl": 1.083, "volume": 0.10, "mfe_r": self.peak_r}

        def open_trades(self):  # type: ignore[no-untyped-def]
            return []

        def management_action_exists(self, _ticket, _actions):  # type: ignore[no-untyped-def]
            return False

        def update_excursions(self, _trade_id, *, mae_r, mfe_r):  # type: ignore[no-untyped-def]
            self.peak_r = max(self.peak_r, mfe_r)

    # Pinned to mid-morning, and it has to be. This test used the wall clock,
    # so what it measured depended on the hour the suite happened to run: the
    # closing-hour rule reads the minutes left before the wind-down, and at
    # 14:24 UTC the target 100 pips away needed 363 minutes against 351
    # remaining, so the position was banked as SESSION_DECAY and the partial
    # close this test is about never happened. Correct behaviour, wrong test.
    moment = datetime(2026, 8, 4, 9, 0, tzinfo=UTC)
    fake = FakeMT5(now=moment)
    paper = PaperBroker(connector(fake), tmp_path / "paper.json")
    paper.connect()
    spec = paper.spec("EURUSD")
    paper.order_send(OrderRequest("EURUSD", Direction.LONG, 0.10, 1.083, 1.10, 1.08512), spec)
    fake.quotes["EURUSD"] = (1.09, 1.09012)
    settings = load_settings(env_overrides=False)
    # This test measures partial-close persistence, not whether the synthetic
    # FakeMT5 bars look healthy. Keep earlier exit layers from pre-empting the
    # rule under test.
    management = settings.trade_management.model_copy(
        update={"health_enabled": False, "giveback_arm_r": 0.0}
    )
    settings = settings.model_copy(update={"trade_management": management})
    manager = PositionManager(paper, JournalStub(), settings)  # type: ignore[arg-type]

    manager.manage(paper.positions(), moment)  # break-even first
    events = manager.manage(paper.positions(), moment)

    assert events[0].action == "PARTIAL_CLOSE"
    assert paper.positions()[0].volume == pytest.approx(0.05)
    recovered = manager.reconcile(paper.positions())
    assert any(event.action == "PARTIAL_CLOSE_RECOVERED" for event in recovered)
    remaining = paper.positions()[0]
    paper.close_position(remaining)
    closed = paper.closed_position(remaining.ticket)
    assert closed is not None
    assert closed.volume == pytest.approx(0.10)
    assert len(closed.deal_tickets) == 2
    paper.shutdown()


def test_news_break_even_does_not_close_an_already_protected_position(tmp_path: Path) -> None:
    class NewsStub:
        @staticmethod
        def position_action(*_args):  # type: ignore[no-untyped-def]
            return "break_even"

    fake = FakeMT5(now=datetime.now(UTC))
    paper = PaperBroker(connector(fake), tmp_path / "paper.json")
    paper.connect()
    paper.order_send(
        OrderRequest("EURUSD", Direction.LONG, 0.10, 1.083, 1.10, 1.08512),
        paper.spec("EURUSD"),
    )
    fake.quotes["EURUSD"] = (1.09, 1.09012)
    manager = PositionManager(
        paper,
        object(),
        load_settings(env_overrides=False),  # type: ignore[arg-type]
    )

    first = manager.manage_news(paper.positions(), NewsStub())  # type: ignore[arg-type]
    second = manager.manage_news(paper.positions(), NewsStub())  # type: ignore[arg-type]

    assert first[0].action == "NEWS_BREAK_EVEN"
    assert second == []
    assert len(paper.positions()) == 1
    paper.shutdown()


def test_scanner_scans_and_ranks_full_available_catalogue_by_default() -> None:
    fake = FakeMT5()
    market = connector(fake)
    market.connect()
    settings = load_settings()
    scanner = UniverseScanner(market, settings)

    batch = scanner.scan(cursor=0, keep=2)

    assert batch.inspected == 2
    assert batch.universe_size == 2
    assert batch.next_cursor == 0
    assert {item.symbol for item in batch.candidates} == {"EURUSD", "USDJPY"}
    assert len(batch.inspections) == 2
    assert {item.status for item in batch.inspections} == {"SHORTLISTED"}
    assert all(item.reason for item in batch.inspections)
    market.shutdown()


def test_scanner_yields_to_position_protection_between_symbols() -> None:
    fake = FakeMT5()
    market = connector(fake)
    market.connect()
    scanner = UniverseScanner(market, load_settings())
    pulses: list[int] = []

    batch = scanner.scan(cursor=0, keep=2, pulse=lambda: pulses.append(1))

    assert batch.inspected == 2
    assert len(pulses) == batch.inspected
    market.shutdown()


def test_scanner_queues_core_then_preferred_then_catalogue_fallback() -> None:
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    fake = FakeMT5(
        now=moment,
        specs={
            "EURUSD.i": eurusd_spec(
                name="EURUSD.i", path="RAW\\Raw Majors\\EURUSD.i", description="EURUSD"
            ),
            "BTCUSD": eurusd_spec(
                name="BTCUSD",
                path="Cryptos\\BTCUSD",
                description="Bitcoin",
                trade_contract_size=1.0,
            ),
            "DB1": eurusd_spec(
                name="DB1",
                path="Shares\\Germany\\DB1",
                description="Deutsche Boerse",
                trade_contract_size=1.0,
            ),
        },
        quotes={
            "EURUSD.i": (1.08500, 1.08512),
            "BTCUSD": (1.08500, 1.08512),
            "DB1": (1.08500, 1.08512),
        },
    )
    market = connector(fake)
    market.connect()
    settings = load_settings(env_overrides=False)
    scanner_config = settings.scanner.model_copy(
        update={
            "priority_asset_classes": ("forex", "crypto"),
            "priority_symbols": ("EURUSD",),
            "priority_spread_weight": 10.0,
        }
    )
    settings = settings.model_copy(update={"scanner": scanner_config})
    scanner = UniverseScanner(market, settings, SimulatedClock(moment))

    batch = scanner.scan(keep=3)

    assert [item.symbol for item in batch.candidates] == ["EURUSD.i", "BTCUSD", "DB1"]
    assert [item.priority_tier for item in batch.candidates] == [2, 1, 0]
    market.shutdown()


def test_bounded_scan_revisits_liquid_lanes_without_skipping_the_rotation() -> None:
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    names = ["DB1", "DTE", "BMW", "EURUSD.i", "BTCUSD"]
    paths = {
        "DB1": "Shares\\Germany\\DB1",
        "DTE": "Shares\\Germany\\DTE",
        "BMW": "Shares\\Germany\\BMW",
        "EURUSD.i": "RAW\\Raw Majors\\EURUSD.i",
        "BTCUSD": "Cryptos\\BTCUSD",
    }
    fake = FakeMT5(
        now=moment,
        specs={name: eurusd_spec(name=name, path=paths[name]) for name in names},
        quotes=dict.fromkeys(names, (1.08500, 1.08512)),
    )
    market = connector(fake)
    market.connect()
    settings = load_settings(env_overrides=False)
    scanner_config = settings.scanner.model_copy(
        update={
            "priority_asset_classes": ("forex", "crypto"),
            "priority_symbols": ("EURUSD",),
            "priority_every_cycle": True,
        }
    )
    settings = settings.model_copy(update={"scanner": scanner_config})
    scanner = UniverseScanner(market, settings, SimulatedClock(moment))

    first = scanner.scan(cursor=0, batch_size=1, keep=10)
    second = scanner.scan(cursor=first.next_cursor, batch_size=1, keep=10)

    assert {row.symbol for row in first.inspections} >= {"EURUSD.i", "BTCUSD"}
    assert {row.symbol for row in second.inspections} >= {"EURUSD.i", "BTCUSD"}
    assert first.next_cursor == 1
    assert second.next_cursor == 2
    assert "DB1" in {row.symbol for row in first.inspections}
    assert "DTE" in {row.symbol for row in second.inspections}
    market.shutdown()


def test_empty_asset_filter_keeps_future_broker_catalogue_folders_visible() -> None:
    """A new Eightcap product folder must be inspected, not silently disappear.

    Its contract may still be rejected fail-closed later. The catalogue count
    and scanner telemetry nevertheless need to account for every symbols_get()
    row when the operator selected the complete catalogue.
    """
    fake = FakeMT5(
        specs={
            "FUTURE_PRODUCT": eurusd_spec(
                name="FUTURE_PRODUCT", path="Future Eightcap Products\\FUTURE_PRODUCT"
            )
        },
        quotes={"FUTURE_PRODUCT": (1.08500, 1.08512)},
    )
    market = connector(fake)
    market.connect()
    scanner = UniverseScanner(market, load_settings(env_overrides=False))

    assert [item.name for item in scanner.catalogue()] == ["FUTURE_PRODUCT"]
    market.shutdown()


def test_ignored_symbol_never_enters_the_scan_catalogue() -> None:
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    fake = FakeMT5(
        specs={
            "XAUUSD": eurusd_spec(name="XAUUSD", path="Commodities\\Metals\\XAUUSD"),
            "EURUSD.i": eurusd_spec(name="EURUSD.i", path="RAW\\Raw Majors\\EURUSD.i"),
        },
        quotes={"XAUUSD": (2400.0, 2400.2), "EURUSD.i": (1.08500, 1.08512)},
    )
    market = connector(fake)
    market.connect()
    base = load_settings(env_overrides=False)
    instruments = base.instruments.model_copy(update={"ignored_symbols": ("XAUUSD",)})
    scanner = UniverseScanner(
        market, base.model_copy(update={"instruments": instruments}), SimulatedClock(moment)
    )

    assert [item.name for item in scanner.catalogue()] == ["EURUSD.i"]
    batch = scanner.scan(keep=10)
    assert {row.symbol for row in batch.inspections} == {"EURUSD.i"}
    market.shutdown()


def test_default_scanner_does_not_stop_after_25_markets() -> None:
    symbols = [f"EURUSD{number:02d}" for number in range(30)]
    fake = FakeMT5(
        specs={
            symbol: eurusd_spec(name=symbol, path=f"Forex\\Test\\{symbol}") for symbol in symbols
        },
        quotes=dict.fromkeys(symbols, (1.08500, 1.08512)),
    )
    market = connector(fake)
    market.connect()
    scanner = UniverseScanner(market, load_settings(env_overrides=False))

    batch = scanner.scan(keep=5)

    assert batch.universe_size == 30
    assert batch.inspected == 30
    assert len(batch.inspections) == 30
    assert len(batch.candidates) == 5
    market.shutdown()


def test_scanner_explains_a_closed_or_stale_market() -> None:
    # Older than any plausible broker-server timezone offset, so the connector
    # correctly treats this as a closed-market quote rather than UTC normalization.
    fake = FakeMT5(now=datetime.now(UTC) - timedelta(days=2))
    market = connector(fake)
    market.connect()
    scanner = UniverseScanner(market, load_settings(env_overrides=False))

    batch = scanner.scan(cursor=0, batch_size=1, keep=1)

    assert batch.candidates == ()
    assert batch.rejected == 1
    assert batch.inspections[0].status == "REJECTED"
    assert batch.inspections[0].stage == "quote"
    assert "stale" in batch.inspections[0].reason.casefold()
    market.shutdown()


def test_demo_mode_hard_refuses_a_live_account(tmp_path: Path) -> None:
    fake = FakeMT5(is_demo=False, server="FakeBroker-Live")
    runner = JarvisRunner(
        connector(fake),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.DEMO,
    )

    with pytest.raises(RuntimeError, match="DEMO_ACCOUNT_REQUIRED"):
        runner.connect()

    assert not fake.orders_sent
    assert not runner.broker.is_connected


def _wide_spread_market(moment: datetime) -> FakeMT5:
    """Two forex pairs: one hopeless on spread, one a whisker over the cap.

    Against a 2.0 bps cap, EURUSD quotes 9.2 bps — over four times the limit —
    and GBPUSD quotes 3.0 bps. Both are refused. Only the first is refused in a
    way that six hours cannot plausibly change.
    """
    names = {
        "EURUSD": "Forex\\Majors\\EURUSD",
        "GBPUSD": "Forex\\Majors\\GBPUSD",
    }
    return FakeMT5(
        now=moment,
        specs={
            name: eurusd_spec(name=name, path=path, currency_base=name[:3])
            for name, path in names.items()
        },
        quotes={
            "EURUSD": (1.08500, 1.08600),  # 9.21 bps
            "GBPUSD": (1.08500, 1.085326),  # 3.00 bps
        },
    )


def _scanner_with_forex_cap(
    fake: FakeMT5, moment: datetime, *, cap_bps: float = 2.0, park_hours: float = 6.0
) -> tuple[MT5Connector, UniverseScanner]:
    market = connector(fake)
    market.connect()
    settings = load_settings(env_overrides=False)
    spread = settings.filters.spread.model_copy(update={"max_spread_bps": {"forex": cap_bps}})
    settings = settings.model_copy(
        update={
            "filters": settings.filters.model_copy(update={"spread": spread}),
            "scanner": settings.scanner.model_copy(
                update={"wide_spread_park_multiple": 2.0, "wide_spread_park_hours": park_hours}
            ),
        }
    )
    return market, UniverseScanner(market, settings, SimulatedClock(moment))


def test_hopeless_spread_is_remembered_and_a_borderline_one_is_measured_again() -> None:
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    fake = _wide_spread_market(moment)
    market, scanner = _scanner_with_forex_cap(fake, moment)

    first = scanner.scan(keep=5)
    fake.calls.clear()
    second = scanner.scan(keep=5)

    assert not first.candidates and not second.candidates
    measured_again = {args[0] for name, args in fake.calls if name == "symbol_info_tick"}
    # The nine-bps market is not asked about; the three-bps one still is.
    assert measured_again == {"GBPUSD"}
    market.shutdown()


def test_a_remembered_spread_refusal_still_reports_as_a_spread_block() -> None:
    """`scan_report.py` counts blocked markets by stage. A held one must count.

    If the hold invented its own stage the crypto column would quietly report
    124 markets as passing, which is the opposite of what happened to them.
    """
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    fake = _wide_spread_market(moment)
    market, scanner = _scanner_with_forex_cap(fake, moment)

    scanner.scan(keep=5)
    held = next(row for row in scanner.scan(keep=5).inspections if row.symbol == "EURUSD")

    assert held.status == "REJECTED"
    assert held.stage == "spread"
    assert held.spread_bps == pytest.approx(9.21, abs=0.05)
    assert held.asset_class.value == "forex"
    assert "not re-measured until 18:00 UTC" in held.reason
    market.shutdown()


def test_the_hold_expires_and_the_market_gets_its_next_chance() -> None:
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    fake = _wide_spread_market(moment)
    market, scanner = _scanner_with_forex_cap(fake, moment)

    scanner.scan(keep=5)
    later = moment + timedelta(hours=6, minutes=1)
    scanner.clock = SimulatedClock(later)
    fake.now = later
    fake.quotes["EURUSD"] = (1.08500, 1.085163)  # tightened to 1.50 bps
    fake.calls.clear()

    batch = scanner.scan(keep=5)

    assert "EURUSD" in {row.symbol for row in batch.candidates}
    assert ("symbol_info_tick", ("EURUSD",)) in fake.calls
    market.shutdown()


def test_zero_hours_switches_the_hold_off_entirely() -> None:
    moment = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    fake = _wide_spread_market(moment)
    market, scanner = _scanner_with_forex_cap(fake, moment, park_hours=0.0)

    scanner.scan(keep=5)
    fake.calls.clear()
    scanner.scan(keep=5)

    assert not scanner._parked
    measured_again = {args[0] for name, args in fake.calls if name == "symbol_info_tick"}
    assert measured_again == {"EURUSD", "GBPUSD"}
    market.shutdown()
