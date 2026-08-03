"""Risk limits, circuit breaker, and the forbidden-practice assertions."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import Settings
from core.clock import SimulatedClock
from core.errors import ForbiddenStrategyError
from core.instrument import InstrumentSpec
from core.types import AccountSnapshot, Direction, Position
from infra.killswitch import KillSwitch
from journal.database import Journal
from journal.recorder import Recorder
from risk.position_sizer import PositionSizer
from risk.reasons import Reason
from risk.risk_manager import RiskManager
from tests.fakes.fake_mt5 import eurusd_spec

# Wednesday 14:30 UTC — inside the trading day that began Tuesday 21:00.
NOW = datetime(2026, 3, 11, 14, 30, tzinfo=UTC)


@pytest.fixture
def raw() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def settings_for(tmp_path: Path, raw: dict[str, Any], **overrides: Any) -> Settings:
    data = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node: Any = data
        *path, leaf = dotted.split(".")
        for part in path:
            node = node[part]
        node[leaf] = value
    path_ = tmp_path / "config.yaml"
    path_.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_settings(path_, env_overrides=False)


@pytest.fixture
def settings(tmp_path: Path, raw: dict[str, Any]) -> Settings:
    return settings_for(tmp_path, raw, **{"system.mode": "scaling"})


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def journal(tmp_path: Path, clock: SimulatedClock) -> Journal:
    with Journal(tmp_path / "journal.db", clock) as j:
        yield j


@pytest.fixture
def manager(settings: Settings, journal: Journal, clock: SimulatedClock) -> RiskManager:
    return RiskManager(settings=settings, journal=journal, clock=clock)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


def account(equity: float, *, balance: float | None = None, free: float = 1e9) -> AccountSnapshot:
    return AccountSnapshot(
        login=1,
        server="Fake",
        currency="EUR",
        balance=balance if balance is not None else equity,
        equity=equity,
        margin=0.0,
        margin_free=free,
        margin_level=0.0,
        leverage=500,
        is_demo=True,
        taken_at=NOW,
    )


def position(symbol: str = "EURUSD", direction: Direction = Direction.LONG) -> Position:
    return Position(
        ticket=1,
        symbol=symbol,
        direction=direction,
        volume=0.10,
        price_open=1.085,
        sl=1.083,
        tp=1.091,
        profit=0.0,
        swap=0.0,
        opened_at=NOW,
    )


class TestPeriodBoundaries:
    def test_trading_day_starts_at_the_fx_rollover(self, journal: Journal) -> None:
        # 14:30 Wednesday belongs to the day that opened Tuesday 21:00.
        assert journal.day_start(NOW) == datetime(2026, 3, 10, 21, 0, tzinfo=UTC)

    def test_just_before_rollover_is_still_the_old_day(self, journal: Journal) -> None:
        moment = datetime(2026, 3, 11, 20, 59, tzinfo=UTC)
        assert journal.day_start(moment) == datetime(2026, 3, 10, 21, 0, tzinfo=UTC)

    def test_just_after_rollover_is_a_new_day(self, journal: Journal) -> None:
        moment = datetime(2026, 3, 11, 21, 1, tzinfo=UTC)
        assert journal.day_start(moment) == datetime(2026, 3, 11, 21, 0, tzinfo=UTC)

    def test_week_starts_at_the_sunday_boundary(self, journal: Journal) -> None:
        assert journal.week_start(NOW) == datetime(2026, 3, 8, 21, 0, tzinfo=UTC)


class TestEquityAnchors:
    def test_day_anchor_is_written_once(
        self, manager: RiskManager, journal: Journal, clock: SimulatedClock
    ) -> None:
        """A restart mid-day must not hand back a fresh loss budget."""
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(hours=2))
        state = manager.build_state(account(950.0))

        assert state.day_start_equity == pytest.approx(1_000.0)
        assert state.day_pnl_pct == pytest.approx(-5.0)

    def test_new_day_re_anchors(self, manager: RiskManager, clock: SimulatedClock) -> None:
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(days=1))
        state = manager.build_state(account(950.0))

        assert state.day_start_equity == pytest.approx(950.0)
        assert state.day_pnl_pct == pytest.approx(0.0)
        # The week anchor is unchanged, so the weekly loss still shows.
        assert state.week_start_equity == pytest.approx(1_000.0)
        assert state.week_pnl_pct == pytest.approx(-5.0)

    def test_equity_peak_ratchets_and_never_falls(
        self, manager: RiskManager, clock: SimulatedClock
    ) -> None:
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(minutes=5))
        manager.build_state(account(1_200.0))
        clock.advance(timedelta(minutes=5))
        state = manager.build_state(account(900.0))

        assert state.equity_peak == pytest.approx(1_200.0)
        assert state.drawdown_pct == pytest.approx(25.0)


class TestSystemGates:
    def test_clear_state_allows_trading(self, manager: RiskManager) -> None:
        state = manager.build_state(account(1_000.0))
        assert manager.check_can_trade(state).approved

    def test_daily_loss_limit_blocks(self, manager: RiskManager, clock: SimulatedClock) -> None:
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(hours=1))
        state = manager.build_state(account(965.0))  # -3.5%, limit 3%

        decision = manager.check_can_trade(state)
        assert not decision.approved
        assert decision.reason is Reason.DAILY_LOSS_LIMIT
        assert decision.reason.is_halt

    def test_unrealised_loss_counts_towards_the_limit(
        self, manager: RiskManager, clock: SimulatedClock
    ) -> None:
        """Equity, not realised P/L — an open loser still consumes the budget."""
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(hours=1))
        # Balance untouched (nothing closed), equity down on an open position.
        state = manager.build_state(account(960.0, balance=1_000.0))
        assert manager.check_can_trade(state).reason is Reason.DAILY_LOSS_LIMIT

    def test_weekly_limit_outranks_the_daily_one(
        self, manager: RiskManager, clock: SimulatedClock
    ) -> None:
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(days=1))
        manager.build_state(account(1_000.0))  # re-anchor the day at 1000
        clock.advance(timedelta(hours=1))
        state = manager.build_state(account(930.0))  # -7% on the week, -7% today

        decision = manager.check_can_trade(state)
        assert decision.reason is Reason.WEEKLY_LOSS_LIMIT

    def test_circuit_breaker_outranks_everything(
        self, manager: RiskManager, clock: SimulatedClock
    ) -> None:
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(hours=1))
        state = manager.build_state(account(840.0))  # -16% from the peak

        decision = manager.check_can_trade(state)
        assert decision.reason is Reason.CIRCUIT_BREAKER
        assert manager.circuit_breaker_tripped(state)

    def test_kill_switch_outranks_the_breaker(
        self, settings: Settings, journal: Journal, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        switch = KillSwitch.in_dir(tmp_path)
        switch.engage("manual halt")
        manager = RiskManager(settings=settings, journal=journal, clock=clock, kill_switch=switch)
        state = manager.build_state(account(1_000.0))

        decision = manager.check_can_trade(state)
        assert decision.reason is Reason.KILL_SWITCH
        assert "manual halt" in decision.detail

    def test_max_positions_blocks(self, manager: RiskManager) -> None:
        state = manager.build_state(
            account(1_000.0),
            [position("EURUSD"), position("GBPUSD")],  # limit is 2
        )
        assert manager.check_can_trade(state).reason is Reason.MAX_POSITIONS_REACHED

    def test_daily_trade_count_blocks(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        recorder = Recorder(journal, clock, settings)
        sizer = PositionSizer(settings)
        sizing = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
        )
        for ticket in range(6):  # scaling allows 6/day
            recorder.record_trade_open(
                cycle_pk=None,
                sizing=sizing,
                ticket=1000 + ticket,
                entry_price=1.085,
                equity_before=10_000.0,
            )

        state = manager.build_state(account(10_000.0))
        assert state.trades_today == 6
        assert manager.check_can_trade(state).reason is Reason.MAX_TRADES_PER_DAY

    def test_halt_persists_across_cycles(self, manager: RiskManager) -> None:
        manager.halt("operator intervention")
        state = manager.build_state(account(1_000.0))
        decision = manager.check_can_trade(state)
        assert decision.reason is Reason.SYSTEM_HALTED
        assert "operator intervention" in decision.detail


class TestSymbolGates:
    def test_whitelisted_symbol_passes(self, manager: RiskManager, spec: InstrumentSpec) -> None:
        state = manager.build_state(account(1_000.0))
        assert manager.check_symbol("EURUSD", state, spec).approved

    def test_non_whitelisted_symbol_blocks(
        self, manager: RiskManager, spec: InstrumentSpec
    ) -> None:
        state = manager.build_state(account(1_000.0))
        decision = manager.check_symbol("EURTRY", state, spec)
        assert decision.reason is Reason.SYMBOL_NOT_WHITELISTED

    def test_gold_blocks_below_its_equity_floor(
        self, manager: RiskManager, spec: InstrumentSpec
    ) -> None:
        state = manager.build_state(account(300.0))
        decision = manager.check_symbol("XAUUSD", state, spec)
        assert decision.reason is Reason.SYMBOL_BLOCKED_BY_EQUITY

    def test_existing_position_blocks_a_second(
        self, manager: RiskManager, spec: InstrumentSpec
    ) -> None:
        state = manager.build_state(account(1_000.0), [position("EURUSD")])
        decision = manager.check_symbol("EURUSD", state, spec)
        assert decision.reason is Reason.POSITION_ALREADY_OPEN


class TestMargin:
    def test_skipped_without_an_estimator(self, manager: RiskManager) -> None:
        state = manager.build_state(account(1_000.0))
        assert manager.check_margin(state, "EURUSD", Direction.LONG, 1.0, 1.085).approved

    def test_insufficient_free_margin_blocks(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = RiskManager(
            settings=settings,
            journal=journal,
            clock=clock,
            margin_estimator=lambda *_: 300.0,
        )
        state = manager.build_state(account(1_000.0, free=500.0))
        # 300 required x 2.0 safety = 600 > 500 free
        decision = manager.check_margin(state, "EURUSD", Direction.LONG, 1.0, 1.085)
        assert decision.reason is Reason.INSUFFICIENT_MARGIN

    def test_enough_margin_passes(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = RiskManager(
            settings=settings,
            journal=journal,
            clock=clock,
            margin_estimator=lambda *_: 100.0,
        )
        state = manager.build_state(account(1_000.0, free=500.0))
        assert manager.check_margin(state, "EURUSD", Direction.LONG, 1.0, 1.085).approved


class TestAntiMartingale:
    def test_full_risk_below_the_streak_threshold(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        _record_losses(journal, clock, settings, spec, count=2)
        state = manager.build_state(account(1_000.0))
        assert state.consecutive_losses == 2
        assert manager.risk_multiplier(state) == 1.0

    def test_risk_halves_after_three_losses(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        _record_losses(journal, clock, settings, spec, count=3)
        state = manager.build_state(account(1_000.0))
        assert state.consecutive_losses == 3
        assert manager.risk_multiplier(state) == 0.5

    def test_a_winner_resets_the_streak(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        _record_losses(journal, clock, settings, spec, count=3)
        _record_trade(journal, clock, settings, spec, pnl=50.0)
        state = manager.build_state(account(1_000.0))
        assert state.consecutive_losses == 0
        assert manager.risk_multiplier(state) == 1.0

    def test_break_even_neither_extends_nor_resets(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        """A flat trade must not restore full risk after two real losses."""
        _record_losses(journal, clock, settings, spec, count=2)
        _record_trade(journal, clock, settings, spec, pnl=0.0)
        _record_losses(journal, clock, settings, spec, count=1)
        state = manager.build_state(account(1_000.0))
        assert state.consecutive_losses == 3
        assert manager.risk_multiplier(state) == 0.5


class TestForbiddenPractices:
    """These raise. A gate saying "no" is normal; reaching these is not."""

    def _sizing(self, settings: Settings, spec: InstrumentSpec, **kwargs: Any) -> Any:
        return PositionSizer(settings).size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.08500,
            sl=1.08300,
            tp=1.09100,
            **kwargs,
        )

    def test_adding_to_a_winner_is_forbidden(
        self, manager: RiskManager, settings: Settings, spec: InstrumentSpec
    ) -> None:
        sizing = self._sizing(settings, spec)
        state = manager.build_state(account(10_000.0), [position("EURUSD", Direction.LONG)])
        with pytest.raises(ForbiddenStrategyError, match="averaging down / gridding"):
            manager.assert_not_forbidden(sizing, state)

    def test_hedging_the_same_symbol_is_forbidden(
        self, manager: RiskManager, settings: Settings, spec: InstrumentSpec
    ) -> None:
        sizing = self._sizing(settings, spec)
        state = manager.build_state(account(10_000.0), [position("EURUSD", Direction.SHORT)])
        with pytest.raises(ForbiddenStrategyError, match="hedge"):
            manager.assert_not_forbidden(sizing, state)

    def test_sizing_above_what_the_loss_streak_allows_is_forbidden(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        """The decision is what may not rise, and this is a raised decision."""
        _record_trade(journal, clock, settings, spec, pnl=-100.0, risk_pct=0.5)
        state = manager.build_state(account(10_000.0))
        sanctioned = settings.effective_risk_pct() * manager.risk_multiplier(state)
        inflated = replace(self._sizing(settings, spec), intended_risk_pct=sanctioned * 1.5)

        with pytest.raises(ForbiddenStrategyError, match="martingale / recovery"):
            manager.assert_not_forbidden(inflated, state)

    def test_lot_rounding_on_the_previous_trade_is_not_martingale(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        """Regression: every candidate after a loss was refused as martingale.

        The check compared the new trade's *intended* risk against the previous
        trade's *actual* risk, which are different quantities. Volume rounds
        down to the broker's step, so on a small account the realised risk lands
        wherever 0.01 lots happens to put it — 1.44% here, something else on the
        next symbol. Production read that as "risk would rise from 1.440% to
        2.000%" and blocked everything, while the configured risk had not moved
        at all.
        """
        _record_trade(journal, clock, settings, spec, pnl=-100.0, risk_pct=0.72)
        state = manager.build_state(account(10_000.0))
        sizing = self._sizing(settings, spec)

        assert sizing.intended_risk_pct > 0.72, "fixture must reproduce the reported shape"
        manager.assert_not_forbidden(sizing, state)  # must not raise

    def test_same_risk_after_a_loss_is_allowed(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        _record_trade(journal, clock, settings, spec, pnl=-100.0, risk_pct=1.0)
        state = manager.build_state(account(10_000.0))
        manager.assert_not_forbidden(self._sizing(settings, spec), state)

    def test_reduced_risk_after_a_loss_is_allowed(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        _record_trade(journal, clock, settings, spec, pnl=-100.0, risk_pct=1.0)
        state = manager.build_state(account(10_000.0))
        manager.assert_not_forbidden(self._sizing(settings, spec, risk_multiplier=0.5), state)

    def test_increasing_risk_after_a_win_is_allowed(
        self,
        manager: RiskManager,
        journal: Journal,
        clock: SimulatedClock,
        settings: Settings,
        spec: InstrumentSpec,
    ) -> None:
        """Only the post-loss increase is martingale. After a win it is not."""
        _record_trade(journal, clock, settings, spec, pnl=200.0, risk_pct=0.5)
        state = manager.build_state(account(10_000.0))
        manager.assert_not_forbidden(self._sizing(settings, spec), state)


class TestCircuitBreaker:
    def test_tripping_halts_and_engages_the_kill_switch(
        self, settings: Settings, journal: Journal, clock: SimulatedClock, tmp_path: Path
    ) -> None:
        switch = KillSwitch.in_dir(tmp_path)
        manager = RiskManager(settings=settings, journal=journal, clock=clock, kill_switch=switch)
        manager.build_state(account(1_000.0))
        clock.advance(timedelta(hours=1))
        state = manager.build_state(account(800.0))

        message = manager.trip_circuit_breaker(state)

        assert "CIRCUIT BREAKER" in message
        # The STOP file is what makes the halt survive a supervisor restart.
        assert switch.is_engaged()
        assert "CIRCUIT BREAKER" in switch.reason()

        fresh = manager.build_state(account(800.0))
        assert manager.check_can_trade(fresh).reason is Reason.KILL_SWITCH


# ----------------------------------------------------------------- helpers ---


def _record_trade(
    journal: Journal,
    clock: SimulatedClock,
    settings: Settings,
    spec: InstrumentSpec,
    *,
    pnl: float,
    risk_pct: float = 1.0,
) -> None:
    """Open and immediately close one trade with a given outcome."""
    recorder = Recorder(journal, clock, settings)
    sizing = PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=1.08500,
        sl=1.08300,
        tp=1.09100,
        risk_multiplier=risk_pct / settings.effective_risk_pct(),
    )
    trade_id = recorder.record_trade_open(
        cycle_pk=None,
        sizing=sizing,
        ticket=None,
        entry_price=1.085,
        equity_before=10_000.0,
    )
    clock.advance(timedelta(minutes=1))
    recorder.record_trade_close(
        trade_id,
        exit_price=1.083,
        pnl_money=pnl,
        exit_reason="SL" if pnl < 0 else "TP",
        equity_after=10_000.0 + pnl,
    )
    clock.advance(timedelta(minutes=1))


def _record_losses(
    journal: Journal,
    clock: SimulatedClock,
    settings: Settings,
    spec: InstrumentSpec,
    *,
    count: int,
) -> None:
    for _ in range(count):
        _record_trade(journal, clock, settings, spec, pnl=-100.0)
