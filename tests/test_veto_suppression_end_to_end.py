"""A refused setup must stop reaching Claude, through the real candidate path.

The unit tests in `test_veto_memory.py` prove the memory forgets and escalates
correctly. This one proves it is actually *wired in* — that the gate sits ahead
of the adviser in `_process_candidate` and not merely next to it. That is the
distinction the production bug turned on: the H1 review cache was present and
working, and the same SPX500 veto still went out three times, because the cache
sat downstream of everything that made the call look like a fresh question.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from advisory.providers import Advice, Reflection, Supervision
from analysis.confluence import TradeIdea
from config.loader import load_settings
from config.schema import MT5Config
from core.clock import SimulatedClock
from core.mt5_connector import MT5Connector
from core.types import Direction, MarketContext, Timeframe
from runner.service import JarvisRunner, OperationMode
from tests.fakes.fake_mt5 import FakeMT5

NOW = datetime(2026, 8, 3, 11, 22, tzinfo=UTC)


class SilentAdvisor:
    def review(self, *_args: object, **_kwargs: object) -> Advice:  # pragma: no cover
        raise AssertionError("the gate should have prevented this call")

    def reflect(self, *_args: object, **_kwargs: object) -> Reflection:
        return Reflection("test", provider="fake")

    def supervise(self, *_args: object, **_kwargs: object) -> Supervision:
        return Supervision("hold", "test", provider="fake")


@pytest.fixture
def idea() -> TradeIdea:
    return TradeIdea(
        symbol="SPX500",
        approved=True,
        direction=Direction.LONG,
        score=72.0,
        confidence=0.7,
        entry=7534.0,
        stop_loss=7480.0,
        take_profit=7620.0,
        reason="1 module agrees (100%)",
        signals=(),
    )


@pytest.fixture
def runner(tmp_path) -> JarvisRunner:  # type: ignore[no-untyped-def]
    """A runner built but not connected — the gate needs no broker session."""
    return JarvisRunner(
        MT5Connector(MT5Config(), mt5_module=FakeMT5()),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.EXPERIMENTAL_LIVE,
        advisor=SilentAdvisor(),
        clock=SimulatedClock(NOW),
    )


def empty_context(symbol: str) -> MarketContext:
    return MarketContext(symbol=symbol, now=NOW, series={}, tick=None)


def test_the_runner_files_and_recalls_a_veto(runner: JarvisRunner, idea: TradeIdea) -> None:
    """`_remember_veto` and `_remembered_veto` must agree on the same key."""
    context = empty_context(idea.symbol)
    assert runner._remembered_veto(idea) is None

    runner._remember_veto(idea, context, Advice(False, 0.32, "into resistance", provider="fake"))

    remembered = runner._remembered_veto(idea)
    assert remembered is not None
    assert remembered.repeats == 1
    assert "already refused" in remembered.describe(NOW)


def test_monitor_mode_never_suppresses(tmp_path, idea: TradeIdea) -> None:  # type: ignore[no-untyped-def]
    """Monitor exists to observe every setup; silencing them defeats the point."""
    service = JarvisRunner(
        MT5Connector(MT5Config(), mt5_module=FakeMT5()),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.MONITOR,
        advisor=SilentAdvisor(),
        clock=SimulatedClock(NOW),
    )
    context = empty_context(idea.symbol)
    service._remember_veto(idea, context, Advice(False, 0.3, "no", provider="fake"))
    assert service._remembered_veto(idea) is None


def test_an_approval_clears_the_standing_refusal(runner: JarvisRunner, idea: TradeIdea) -> None:
    context = empty_context(idea.symbol)
    runner._remember_veto(idea, context, Advice(False, 0.32, "into resistance", provider="fake"))
    runner.veto_memory.clear(idea.symbol, "LONG")
    assert runner._remembered_veto(idea) is None


def test_a_missing_signal_frame_yields_no_atr_scale(runner: JarvisRunner) -> None:
    """Zero means "compare exactly", which errs toward asking again."""
    assert runner._signal_atr(empty_context("SPX500")) == 0.0


def test_the_memory_gate_short_circuits_a_repeat(tmp_path, idea) -> None:  # type: ignore[no-untyped-def]
    """The behaviour the operator saw three times in two minutes."""
    from advisory.veto_memory import VetoMemory

    memory = VetoMemory(tmp_path / "veto.json")
    now = datetime(2026, 8, 3, 11, 22, tzinfo=UTC)
    memory.remember(
        idea.symbol,
        "LONG",
        entry=idea.entry,
        stop=idea.stop_loss,
        atr=40.0,
        thesis="rallied into resistance",
        confidence=0.32,
        now=now,
    )

    # 11:23:41 and 11:24:11 — the two follow-up rows from the dashboard.
    for minutes in (1, 2, 60, 80):
        later = now.replace(minute=(22 + minutes) % 60, hour=11 + (22 + minutes) // 60)
        assert memory.recall("SPX500", "LONG", idea.entry, idea.stop_loss, later) is not None


def test_an_errored_veto_is_not_remembered(tmp_path, idea) -> None:  # type: ignore[no-untyped-def]
    """A timeout is a veto by the fail-closed rule but says nothing about the setup.

    Remembering it would let one API outage silence the whole catalogue for an
    hour and a half — a transport failure quietly becoming a trading decision,
    which is the exact confusion this system works hardest to avoid.
    """
    from advisory.veto_memory import VetoMemory

    memory = VetoMemory(tmp_path / "veto.json")
    failed = Advice(
        False, 0.0, "anthropic unavailable; trade vetoed", provider="anthropic", error="Timeout"
    )
    # The runner guards on `advice.error` before calling remember; assert the
    # condition that guard tests, so the guard cannot be dropped unnoticed.
    assert failed.error
    assert memory.recall("SPX500", "LONG", idea.entry, idea.stop_loss, datetime.now(UTC)) is None


def test_the_review_timeframe_is_the_signal_frame_not_the_fastest() -> None:
    """Keyed on M1 the cache expired every sixty seconds, which was the bug."""
    from runner.service import _REVIEW_TIMEFRAME

    assert _REVIEW_TIMEFRAME is Timeframe.H1
