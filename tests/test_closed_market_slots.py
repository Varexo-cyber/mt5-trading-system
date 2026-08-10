"""A position whose market is shut must not hold a slot hostage.

On 10 August the live account carried three single-name share CFDs (ba, con,
dbk). Their exchange closes at 17:00 Amsterdam. After that nothing about them
is possible: they cannot be closed, their stops cannot be moved, they cannot be
secured. They nevertheless held three of four slots, so the scanner had exactly
one usable slot for the rest of the evening.

The rule these tests pin down: a slot bounds how many trades are being *run*,
and a frozen ticket is not being run. It keeps its risk; it gives up its slot.

The fail-safe direction is the other half, and it is the half worth protecting.
Releasing a slot is the loosening, so it must happen on evidence — an explicit
"this venue is not quoting" — and never on the absence of an answer. Every
unclear case below must leave the position counted.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import Settings
from core.clock import SimulatedClock
from core.types import AccountSnapshot, Direction, Position, Tick
from journal.database import Journal
from risk.reasons import Reason
from risk.risk_manager import RiskManager
from runner.service import JarvisRunner

NOW = datetime(2026, 8, 10, 18, 30, tzinfo=UTC)
SHUT = ("BA", "CON", "DBK")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    raw: dict[str, Any] = copy.deepcopy(
        yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    )
    raw["system"]["mode"] = "scaling"
    raw["risk"]["release_slots_when_unmanageable"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(path, env_overrides=False)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def journal(tmp_path: Path, clock: SimulatedClock) -> Journal:
    with Journal(tmp_path / "journal.db", clock) as j:
        yield j


def account(equity: float = 148.92) -> AccountSnapshot:
    return AccountSnapshot(
        login=5049535,
        server="Fake",
        currency="EUR",
        balance=equity,
        equity=equity,
        margin=0.0,
        margin_free=1e9,
        margin_level=0.0,
        leverage=500,
        is_demo=True,
        taken_at=NOW,
    )


def position(symbol: str, ticket: int) -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        direction=Direction.LONG,
        volume=0.05,
        price_open=33.18,
        sl=32.69,
        tp=34.05,
        profit=0.0,
        swap=0.0,
        opened_at=NOW - timedelta(hours=2),
    )


def manager_with(
    settings: Settings,
    journal: Journal,
    clock: SimulatedClock,
    probe: Any = None,
) -> RiskManager:
    return RiskManager(settings=settings, journal=journal, clock=clock, manageability_probe=probe)


def state_of(manager: RiskManager, symbols: tuple[str, ...]):  # type: ignore[no-untyped-def]
    positions = [position(symbol, 100 + i) for i, symbol in enumerate(symbols)]
    return manager.build_state(account(), positions)


def shut_markets(*symbols: str):  # type: ignore[no-untyped-def]
    closed = set(symbols)
    return lambda symbol: symbol not in closed


class TestSlotAccounting:
    def test_a_shut_market_does_not_consume_a_slot(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = manager_with(settings, journal, clock, shut_markets(*SHUT))
        state = state_of(manager, SHUT)

        assert len(state.open_positions) == 3, "the risk is still on the books"
        assert manager.positions_counted_toward_limit(state) == ()
        assert {p.symbol for p in manager.unmanageable_positions(state)} == set(SHUT)

    def test_the_scanner_may_open_again_with_every_slot_frozen(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        """The whole point. Three shut shares must not end the trading evening."""
        limit = settings.effective_max_positions(148.92)
        held = tuple(f"SHUT{i}" for i in range(limit))
        manager = manager_with(settings, journal, clock, shut_markets(*held))
        state = state_of(manager, held)

        decision = manager.check_can_trade(state)
        assert decision.approved, decision.detail
        assert "market is shut" in decision.detail

    def test_an_open_market_still_consumes_its_slot(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = manager_with(settings, journal, clock, shut_markets())
        limit = settings.effective_max_positions(148.92)
        state = state_of(manager, tuple(f"LIVE{i}" for i in range(limit)))

        decision = manager.check_can_trade(state)
        assert not decision.approved
        assert decision.reason is Reason.MAX_POSITIONS_REACHED

    def test_a_mixed_book_counts_only_the_live_side(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = manager_with(settings, journal, clock, shut_markets("BA", "DBK"))
        state = state_of(manager, ("BA", "EURUSD", "DBK"))

        assert [p.symbol for p in manager.positions_counted_toward_limit(state)] == ["EURUSD"]

    def test_the_refusal_names_the_frozen_tickets(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        """An operator comparing this to the terminal must not have to guess."""
        manager = manager_with(settings, journal, clock, shut_markets("BA"))
        limit = settings.effective_max_positions(148.92)
        held = ("BA", *(f"LIVE{i}" for i in range(limit)))
        state = state_of(manager, held)

        decision = manager.check_can_trade(state)
        assert not decision.approved
        assert "market is shut" in decision.detail
        assert "BA" in decision.detail


class TestFailSafeDirection:
    """Every unclear answer must leave the position counted."""

    def test_off_by_default_so_no_account_changes_silently(
        self, tmp_path: Path, journal: Journal, clock: SimulatedClock
    ) -> None:
        stock = load_settings(DEFAULT_CONFIG_PATH, env_overrides=False)
        assert stock.risk.release_slots_when_unmanageable is False

        manager = manager_with(stock, journal, clock, shut_markets(*SHUT))
        state = state_of(manager, SHUT)
        assert len(manager.positions_counted_toward_limit(state)) == 3

    def test_no_probe_means_every_position_counts(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = manager_with(settings, journal, clock, None)
        state = state_of(manager, SHUT)
        assert len(manager.positions_counted_toward_limit(state)) == 3

    def test_a_raising_probe_keeps_the_slot(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        def broken(symbol: str) -> bool:
            raise RuntimeError("terminal not responding")

        manager = manager_with(settings, journal, clock, broken)
        state = state_of(manager, SHUT)
        assert len(manager.positions_counted_toward_limit(state)) == 3

    def test_a_probe_that_does_not_answer_yes_or_no_keeps_the_slot(
        self, settings: Settings, journal: Journal, clock: SimulatedClock
    ) -> None:
        manager = manager_with(settings, journal, clock, lambda symbol: None)
        state = state_of(manager, SHUT)
        assert len(manager.positions_counted_toward_limit(state)) == 3


def probe_for(
    settings: Settings,
    *,
    tick_age: float | None = 1.0,
    tradable: bool = True,
    raises: type[Exception] | None = None,
    tick_is_none: bool = False,
) -> bool:
    """Run `JarvisRunner._market_is_manageable` against a stubbed broker."""

    def spec(symbol: str, *, refresh: bool = False):  # type: ignore[no-untyped-def]
        if raises is not None:
            raise raises("no spec")
        return SimpleNamespace(is_tradable=tradable)

    def tick(symbol: str):  # type: ignore[no-untyped-def]
        if tick_is_none:
            return None
        assert tick_age is not None
        return Tick(symbol=symbol, time=NOW - timedelta(seconds=tick_age), bid=1.0, ask=1.1)

    runner = SimpleNamespace(
        broker=SimpleNamespace(spec=spec, tick=tick),
        clock=SimulatedClock(NOW),
        settings=settings,
    )
    return JarvisRunner._market_is_manageable(runner, "DBK")  # type: ignore[arg-type]


class TestManageabilityProbe:
    def test_a_fresh_quote_means_the_venue_is_open(self, settings: Settings) -> None:
        assert probe_for(settings, tick_age=5.0) is True

    def test_a_quote_older_than_the_limit_means_the_venue_is_shut(self, settings: Settings) -> None:
        stale = settings.risk.unmanageable_quote_age_seconds + 1
        assert probe_for(settings, tick_age=stale) is False

    def test_the_boundary_itself_is_still_open(self, settings: Settings) -> None:
        exact = settings.risk.unmanageable_quote_age_seconds
        assert probe_for(settings, tick_age=exact) is True

    def test_a_quiet_share_mid_session_keeps_its_slot(self, settings: Settings) -> None:
        """Well past the spread filter's 120s entry gate, well inside this one.

        Reusing that gate here would hand a slot back every time a thin share
        went quiet for two minutes, which is a different question entirely.
        """
        assert probe_for(settings, tick_age=180.0) is True

    def test_a_broker_downgrade_out_of_session_is_decisive(self, settings: Settings) -> None:
        assert probe_for(settings, tradable=False) is False

    def test_an_unreadable_symbol_keeps_its_slot(self, settings: Settings) -> None:
        assert probe_for(settings, raises=RuntimeError) is True

    def test_a_missing_tick_keeps_its_slot(self, settings: Settings) -> None:
        assert probe_for(settings, tick_is_none=True) is True

    def test_a_broker_clock_ahead_of_ours_is_not_a_shut_venue(self, settings: Settings) -> None:
        """A negative age is a clock problem and must never release a slot."""
        assert probe_for(settings, tick_age=-3600.0) is True
