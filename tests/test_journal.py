"""Journal: schema, non-trade recording, R arithmetic, execution telemetry."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.loader import load_settings
from config.schema import Settings
from core.clock import SimulatedClock
from core.instrument import InstrumentSpec
from core.types import Direction, OrderResult, Signal
from journal.database import SCHEMA_VERSION, Journal, iso, parse_iso
from journal.recorder import CycleContext, Recorder
from risk.position_sizer import PositionSizer, SizingResult
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec

NOW = datetime(2026, 3, 11, 14, 30, tzinfo=UTC)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_overrides=False)


@pytest.fixture
def journal(tmp_path: Path, clock: SimulatedClock) -> Journal:
    with Journal(tmp_path / "j.db", clock) as j:
        yield j


@pytest.fixture
def recorder(journal: Journal, clock: SimulatedClock, settings: Settings) -> Recorder:
    return Recorder(journal, clock, settings)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


def make_sizing(settings: Settings, spec: InstrumentSpec, **kwargs: object) -> SizingResult:
    return PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=1.08500,
        sl=1.08300,
        tp=1.09100,
        **kwargs,  # type: ignore[arg-type]
    )


class TestSchema:
    def test_opening_creates_the_schema(self, journal: Journal) -> None:
        tables = {
            row["name"]
            for row in journal.query("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "analysis_cycles",
            "module_scores",
            "trades",
            "order_attempts",
            "management_actions",
            "shadow_trades",
            "bar_snapshots",
            "equity_marks",
            "config_snapshots",
            "schema_version",
        } <= tables

    def test_migration_is_idempotent(self, tmp_path: Path, clock: SimulatedClock) -> None:
        path = tmp_path / "j.db"
        with Journal(path, clock) as first:
            applied = first.scalar("SELECT COUNT(*) FROM schema_version")
        with Journal(path, clock) as second:
            assert second.scalar("SELECT COUNT(*) FROM schema_version") == applied
            assert second.scalar("SELECT MAX(version) FROM schema_version") == SCHEMA_VERSION

    def test_future_schema_is_refused(self, tmp_path: Path, clock: SimulatedClock) -> None:
        """Never write into a journal a newer version wrote."""
        path = tmp_path / "j.db"
        with Journal(path, clock) as j:
            j.conn.execute("INSERT INTO schema_version (version) VALUES (99)")
        with pytest.raises(RuntimeError, match="newer journal"):
            Journal(path, clock).open()

    def test_naive_datetimes_are_refused(self) -> None:
        """A naive timestamp in a trading database corrupts it silently."""
        with pytest.raises(ValueError, match="naive datetime"):
            iso(datetime(2026, 3, 11, 14, 30))

    def test_timestamps_round_trip(self) -> None:
        assert parse_iso(iso(NOW)) == NOW


class TestNonTradeRecording:
    """The rows that make filter-effectiveness analysis possible."""

    def test_a_skip_is_recorded_in_full(self, recorder: Recorder, journal: Journal) -> None:
        recorder.record_cycle(
            cycle_id="abc123",
            context=CycleContext(
                symbol="EURUSD",
                equity=1_000.0,
                atr=0.00082,
                spread_pips=0.9,
                session="london",
                volatility_regime="normal",
                minutes_to_news=42.0,
            ),
            reason=Reason.UNDERCAPITALIZED,
            detail="1% of 1000 buys 0.004 lots",
            total_score=71.5,
            score_threshold=65.0,
        )
        row = journal.query("SELECT * FROM analysis_cycles")[0]
        assert row["decision"] == "SKIP"
        assert row["reason"] == "TRADE_SKIPPED_UNDERCAPITALIZED"
        assert row["session"] == "london"
        assert row["minutes_to_news"] == pytest.approx(42.0)
        assert row["total_score"] == pytest.approx(71.5)

    def test_module_scores_are_stored_per_cycle(self, recorder: Recorder, journal: Journal) -> None:
        recorder.record_cycle(
            cycle_id="abc123",
            context=CycleContext(symbol="EURUSD", equity=1_000.0),
            reason=Reason.OK,
            traded=True,
            direction=Direction.LONG,
            signals=[
                Signal(module="market_structure", score=80.0, confidence=0.9, reasoning="BOS up"),
                Signal(module="levels", score=40.0, confidence=0.5, reasoning="at daily open"),
                Signal.neutral("elliott"),
            ],
            weights={"market_structure": 0.5, "levels": 0.3, "elliott": 0.0},
        )
        rows = journal.query("SELECT * FROM module_scores ORDER BY module")
        assert [r["module"] for r in rows] == ["elliott", "levels", "market_structure"]
        assert rows[0]["weight"] == 0.0  # unproven module carries no weight
        assert rows[2]["score"] == pytest.approx(80.0)

    def test_sizing_is_attached_to_the_cycle(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        cycle_pk = recorder.record_cycle(
            cycle_id="x",
            context=CycleContext(symbol="EURUSD", equity=10_000.0),
            reason=Reason.OK,
            traded=True,
        )
        recorder.record_sizing(cycle_pk, make_sizing(settings, spec))

        context = json.loads(journal.query("SELECT context_json FROM analysis_cycles")[0][0])
        assert context["sizing"]["volume"] == pytest.approx(0.50)
        assert context["sizing"]["reason"] == "OK"

    def test_bar_snapshot_round_trips(self, recorder: Recorder, journal: Journal) -> None:
        cycle_pk = recorder.record_cycle(
            cycle_id="x", context=CycleContext(symbol="EURUSD", equity=100.0), reason=Reason.OK
        )
        bars = [{"t": "2026-03-11T13:00:00Z", "o": 1.085, "h": 1.086, "l": 1.084, "c": 1.0855}]
        recorder.record_bar_snapshot(cycle_pk, "EURUSD", "H1", bars)

        stored = json.loads(journal.query("SELECT bars_json FROM bar_snapshots")[0][0])
        assert stored == bars


class TestTradeLifecycle:
    def test_open_and_close_computes_r(
        self,
        recorder: Recorder,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        sizing = make_sizing(settings, spec)  # risks 100.00
        trade_id = recorder.record_trade_open(
            cycle_pk=None,
            sizing=sizing,
            ticket=99,
            entry_price=1.08503,
            equity_before=10_000.0,
        )
        clock.advance(timedelta(hours=2))
        recorder.record_trade_close(
            trade_id,
            exit_price=1.09100,
            pnl_money=300.0,
            exit_reason="TP",
            equity_after=10_300.0,
            mae_r=-0.3,
            mfe_r=3.1,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["pnl_r"] == pytest.approx(3.0)  # 300 on 100 risked
        assert row["duration_seconds"] == 7200
        assert row["mfe_r"] == pytest.approx(3.1)

    def test_r_uses_the_risk_actually_taken(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        """Rounding down means real risk < nominal; R must reflect that."""
        sizing = PositionSizer(settings).size(
            spec=spec,
            equity=1_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08270,
            tp=1.09190,  # 23-pip stop -> 0.04 lots
        )
        assert sizing.actual_risk_money == pytest.approx(9.20)

        trade_id = recorder.record_trade_open(
            cycle_pk=None, sizing=sizing, ticket=1, entry_price=1.085, equity_before=1_000.0
        )
        recorder.record_trade_close(
            trade_id, exit_price=1.0827, pnl_money=-9.20, exit_reason="SL", equity_after=990.80
        )
        row = journal.query("SELECT pnl_r FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["pnl_r"] == pytest.approx(-1.0)

    def test_entry_price_is_the_fill_not_the_request(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        sizing = make_sizing(settings, spec)  # requested 1.08500
        recorder.record_trade_open(
            cycle_pk=None,
            sizing=sizing,
            ticket=1,
            entry_price=1.08507,
            equity_before=10_000.0,
        )
        assert journal.query("SELECT entry_price FROM trades")[0][0] == pytest.approx(1.08507)

    def test_excursions_only_ratchet_outward(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        """A late retrace must not erase that the trade was once 2R up."""
        trade_id = recorder.record_trade_open(
            cycle_pk=None,
            sizing=make_sizing(settings, spec),
            ticket=1,
            entry_price=1.085,
            equity_before=10_000.0,
        )
        recorder.update_excursions(trade_id, mae_r=-0.5, mfe_r=2.0)
        recorder.update_excursions(trade_id, mae_r=-0.2, mfe_r=0.4)

        row = journal.query("SELECT mae_r, mfe_r FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["mae_r"] == pytest.approx(-0.5)
        assert row["mfe_r"] == pytest.approx(2.0)

    def test_open_trades_are_queryable(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        recorder.record_trade_open(
            cycle_pk=None,
            sizing=make_sizing(settings, spec),
            ticket=7,
            entry_price=1.085,
            equity_before=10_000.0,
        )
        assert len(journal.open_trades()) == 1
        assert journal.has_open_position_in("EURUSD")
        assert not journal.has_open_position_in("GBPUSD")

    def test_duplicate_tickets_are_rejected(
        self, recorder: Recorder, settings: Settings, spec: InstrumentSpec
    ) -> None:
        """One MT5 ticket, one journal row — otherwise reconciliation lies."""
        import sqlite3

        sizing = make_sizing(settings, spec)
        recorder.record_trade_open(
            cycle_pk=None,
            sizing=sizing,
            ticket=42,
            entry_price=1.085,
            equity_before=10_000.0,
        )
        with pytest.raises(sqlite3.IntegrityError):
            recorder.record_trade_open(
                cycle_pk=None,
                sizing=sizing,
                ticket=42,
                entry_price=1.085,
                equity_before=10_000.0,
            )


class TestExecutionTelemetry:
    def _result(self, *, ok: bool = True, retcode: int = 10009) -> OrderResult:
        return OrderResult(
            ok=ok,
            retcode=retcode,
            retcode_name="DONE" if ok else "NO_MONEY",
            comment="Request executed",
            order_ticket=1,
            deal_ticket=2,
            position_ticket=1,
            requested_volume=0.5,
            filled_volume=0.5 if ok else 0.0,
            requested_price=1.08500,
            filled_price=1.08503 if ok else 0.0,
            slippage_pips=0.3 if ok else 0.0,
            latency_ms=42.0,
            spread_at_send=0.00012,
            attempts=1,
            sent_at=NOW,
        )

    def test_successful_attempt_is_stored(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        trade_id = recorder.record_trade_open(
            cycle_pk=None,
            sizing=make_sizing(settings, spec),
            ticket=1,
            entry_price=1.08503,
            equity_before=10_000.0,
        )
        recorder.record_order_attempt(
            trade_id=trade_id, kind="ENTRY", symbol="EURUSD", result=self._result()
        )
        row = journal.query("SELECT * FROM order_attempts")[0]
        assert row["slippage_pips"] == pytest.approx(0.3)
        assert row["latency_ms"] == pytest.approx(42.0)
        assert row["ok"] == 1

    def test_rejections_are_stored_too(self, recorder: Recorder, journal: Journal) -> None:
        """A broker rejecting 8% of orders is a finding, not a non-event."""
        recorder.record_order_attempt(
            trade_id=None,
            kind="ENTRY",
            symbol="EURUSD",
            result=self._result(ok=False, retcode=10019),
        )
        row = journal.query("SELECT * FROM order_attempts")[0]
        assert row["ok"] == 0
        assert row["retcode"] == 10019
        assert row["trade_id"] is None

    def test_management_actions_are_stored(
        self, recorder: Recorder, journal: Journal, settings: Settings, spec: InstrumentSpec
    ) -> None:
        trade_id = recorder.record_trade_open(
            cycle_pk=None,
            sizing=make_sizing(settings, spec),
            ticket=1,
            entry_price=1.085,
            equity_before=10_000.0,
        )
        recorder.record_management_action(
            trade_id,
            action="BREAK_EVEN",
            old_sl=1.08300,
            new_sl=1.08512,
            r_at_action=1.0,
            note="+1R reached",
        )
        row = journal.query("SELECT * FROM management_actions")[0]
        assert row["action"] == "BREAK_EVEN"
        assert row["new_sl"] == pytest.approx(1.08512)


class TestShadowTrades:
    def test_blocked_setups_are_tracked_and_resolved(
        self, recorder: Recorder, journal: Journal
    ) -> None:
        """How "is the news filter too strict" becomes measurable."""
        cycle_pk = recorder.record_cycle(
            cycle_id="x",
            context=CycleContext(symbol="EURUSD", equity=1_000.0),
            reason=Reason.UNDERCAPITALIZED,
        )
        shadow_id = recorder.record_shadow_trade(
            cycle_pk=cycle_pk,
            symbol="EURUSD",
            direction=Direction.LONG,
            blocked_by=Reason.UNDERCAPITALIZED,
            entry_price=1.085,
            sl=1.083,
            tp=1.091,
        )
        assert journal.query("SELECT outcome FROM shadow_trades")[0][0] is None

        recorder.resolve_shadow_trade(shadow_id, outcome="TP", pnl_r=3.0)
        row = journal.query("SELECT * FROM shadow_trades")[0]
        assert row["outcome"] == "TP"
        assert row["pnl_r"] == pytest.approx(3.0)


class TestConfigSnapshot:
    def test_snapshot_is_stored_once_per_config(self, recorder: Recorder, journal: Journal) -> None:
        first = recorder.record_config_snapshot()
        second = recorder.record_config_snapshot()
        assert first == second
        assert journal.scalar("SELECT COUNT(*) FROM config_snapshots") == 1

    def test_snapshot_contains_no_credentials(self, recorder: Recorder, journal: Journal) -> None:
        recorder.record_config_snapshot()
        payload = journal.query("SELECT config_json FROM config_snapshots")[0][0].lower()
        assert "password" not in payload
        assert "mt5_login" not in payload
