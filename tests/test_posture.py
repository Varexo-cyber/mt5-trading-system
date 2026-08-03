"""Posture may only ever tighten. This is the anti-chasing guarantee.

The temptation after a run of losses is to size up and win the damage back.
That is the single most reliable way to turn a bad week into a dead account,
and it is the thing the project spec says to hardcode a refusal against. These
tests are that refusal, expressed as properties rather than as a comment.
"""

from __future__ import annotations

import pytest

from risk.posture import Posture, assess


def at(losses: int = 0, equity: float = 100.0, peak: float = 100.0):  # type: ignore[no-untyped-def]
    return assess(consecutive_losses=losses, equity=equity, equity_peak=peak)


# --------------------------------------------------------------- the ladder ---


def test_a_clean_account_is_steady() -> None:
    assert at().posture is Posture.STEADY
    assert not at().is_stressed


def test_two_losses_turns_cautious() -> None:
    assert at(losses=2).posture is Posture.CAUTIOUS


def test_four_losses_turns_defensive() -> None:
    assert at(losses=4).posture is Posture.DEFENSIVE


def test_drawdown_alone_can_trigger_it() -> None:
    """The slow bleed a streak counter misses entirely.

    A grind of small losses with the occasional win keeps the streak at zero
    while the balance drains. Only the drawdown reading catches that.
    """
    assert at(losses=0, equity=91.0, peak=100.0).posture is Posture.DEFENSIVE
    assert at(losses=0, equity=95.0, peak=100.0).posture is Posture.CAUTIOUS


def test_a_fresh_account_with_no_peak_is_steady() -> None:
    assert at(equity=100.0, peak=0.0).posture is Posture.STEADY


def test_equity_above_peak_is_not_a_drawdown() -> None:
    assessment = at(equity=120.0, peak=100.0)
    assert assessment.drawdown_pct == 0.0
    assert assessment.posture is Posture.STEADY


# ------------------------------------------------- it only ever moves one way ---


@pytest.mark.parametrize("losses", range(0, 12))
def test_patience_never_exceeds_normal(losses: int) -> None:
    """A stalled trade can be cut sooner. It can never be given more rope."""
    assert at(losses=losses).patience_multiplier <= 1.0


@pytest.mark.parametrize("losses", range(0, 12))
def test_the_entry_bar_never_drops(losses: int) -> None:
    """A drawdown demands more from a setup, never less."""
    assert at(losses=losses).entry_bar_bonus >= 0.0


def test_worse_results_mean_less_patience_not_more() -> None:
    steady = at(losses=0).patience_multiplier
    cautious = at(losses=2).patience_multiplier
    defensive = at(losses=5).patience_multiplier
    assert defensive < cautious < steady


def test_worse_results_mean_a_higher_bar() -> None:
    assert (
        at(losses=5).entry_bar_bonus > at(losses=2).entry_bar_bonus > at(losses=0).entry_bar_bonus
    )


def test_posture_carries_no_sizing_dial() -> None:
    """Sizing is owned by RiskManager and is not reachable from here.

    If a future edit ever adds a multiplier to this object, this test fails and
    the reviewer has to justify it — which is the point.
    """
    fields = set(at(losses=5).__slots__)
    assert not {name for name in fields if "size" in name or "volume" in name or "lot" in name}


# ------------------------------------------------------------------- guidance ---


def test_the_defensive_guidance_forbids_chasing_explicitly() -> None:
    text = at(losses=5).brief()["guidance"].lower()
    assert "do not attempt to win the drawdown back" in text


def test_the_brief_reports_the_numbers_behind_the_call() -> None:
    brief = at(losses=3, equity=94.0, peak=100.0).brief()
    assert brief["consecutive_losses"] == 3
    assert brief["drawdown_from_peak_pct"] == pytest.approx(6.0)
    assert brief["posture"] in {p.value for p in Posture}


# ------------------------------------------- the mechanical half, end to end ---


def test_a_stalled_trade_is_cut_sooner_in_a_drawdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`patience` shortens the time-exit and nothing else.

    The trade below is 15 hours old and going nowhere. At full patience the
    24-hour timeout has not expired and it stays open; in a defensive posture
    the deadline is 9.6 hours and it is closed. That is the concrete meaning of
    "cut dead trades sooner", and it is the only mechanical change posture makes.
    """
    from datetime import UTC, datetime, timedelta

    from config.loader import load_settings
    from config.schema import MT5Config
    from core.clock import SimulatedClock
    from core.mt5_connector import MT5Connector
    from core.types import Direction
    from execution.manager import PositionManager
    from journal.database import Journal
    from journal.recorder import Recorder
    from risk.position_sizer import PositionSizer
    from tests.fakes.fake_mt5 import FakeMT5

    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    settings = load_settings(env_overrides=False)
    broker = MT5Connector(MT5Config(), mt5_module=FakeMT5())
    broker.connect()
    journal = Journal(tmp_path / "j.db", SimulatedClock(now)).open()

    spec = broker.spec("EURUSD")
    tick = broker.tick("EURUSD")
    sizing = PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=tick.ask,
        sl=spec.normalize_price(tick.ask - 0.0020),
        tp=spec.normalize_price(tick.ask + 0.0040),
    )
    recorder = Recorder(journal, SimulatedClock(now), settings)
    opened = now - timedelta(hours=15)
    recorder.record_trade_open(
        cycle_pk=None,
        sizing=sizing,
        ticket=5150,
        entry_price=tick.ask,
        equity_before=10_000.0,
        opened_at=opened,
    )
    journal.conn.commit()

    from core.types import Position

    position = Position(
        ticket=5150,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=sizing.volume,
        price_open=tick.ask,
        sl=sizing.sl,
        tp=sizing.tp,
        profit=0.0,
        swap=0.0,
        opened_at=opened,
    )
    manager = PositionManager(broker, journal, settings)

    steady = manager.manage([position], now, patience=1.0)
    assert "TIME_EXIT" not in [event.action for event in steady]

    defensive = manager.manage([position], now, at(losses=5).patience_multiplier)
    assert [event.action for event in defensive] == ["TIME_EXIT"]
    assert "drawdown posture" in defensive[0].detail
    journal.close()


def test_patience_above_one_cannot_grant_extra_rope(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Clamped at the manager too, not only at the source.

    A future edit that passes 2.0 must not silently double the timeout on a
    dead trade. The 30-hour-old position below is past the 24-hour limit and
    has to be closed regardless of what patience claims.
    """
    from datetime import UTC, datetime, timedelta

    from config.loader import load_settings
    from config.schema import MT5Config
    from core.clock import SimulatedClock
    from core.mt5_connector import MT5Connector
    from core.types import Direction, Position
    from execution.manager import PositionManager
    from journal.database import Journal
    from journal.recorder import Recorder
    from risk.position_sizer import PositionSizer
    from tests.fakes.fake_mt5 import FakeMT5

    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    settings = load_settings(env_overrides=False)
    broker = MT5Connector(MT5Config(), mt5_module=FakeMT5())
    broker.connect()
    journal = Journal(tmp_path / "j.db", SimulatedClock(now)).open()

    spec = broker.spec("EURUSD")
    tick = broker.tick("EURUSD")
    sizing = PositionSizer(settings).size(
        spec=spec,
        equity=10_000.0,
        direction=Direction.LONG,
        entry=tick.ask,
        sl=spec.normalize_price(tick.ask - 0.0020),
        tp=spec.normalize_price(tick.ask + 0.0040),
    )
    opened = now - timedelta(hours=30)
    Recorder(journal, SimulatedClock(now), settings).record_trade_open(
        cycle_pk=None,
        sizing=sizing,
        ticket=5151,
        entry_price=tick.ask,
        equity_before=10_000.0,
        opened_at=opened,
    )
    journal.conn.commit()
    position = Position(
        ticket=5151,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=sizing.volume,
        price_open=tick.ask,
        sl=sizing.sl,
        tp=sizing.tp,
        profit=0.0,
        swap=0.0,
        opened_at=opened,
    )

    events = PositionManager(broker, journal, settings).manage([position], now, patience=99.0)

    assert [event.action for event in events] == ["TIME_EXIT"]
    journal.close()
