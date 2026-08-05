"""Switching the drawdown breaker and the posture throttle off.

Both were turned off deliberately, on an 88 EUR account 8.2% below its peak.
The breaker had 6.54 EUR of room — three losing trades — while the posture
throttle was refusing fifteen of every sixteen setups over the same drawdown.
The account was too far down to trade and not far enough down to stop.

The dangerous failure here is not that "off" fails to work. It is that a zero
threshold reads as "trip at nought" and halts every account on its first cycle,
which looks exactly like the protection working.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from config.loader import load_settings
from config.schema import NO_DRAWDOWN_BREAKER
from core.clock import SimulatedClock
from journal.database import Journal
from risk.posture import Posture, assess
from risk.reasons import Reason
from risk.risk_manager import RiskManager, RiskState

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def state(equity: float, peak: float) -> RiskState:
    return RiskState(
        now=NOW,
        equity=equity,
        balance=equity,
        margin_free=equity,
        currency="EUR",
        equity_peak=peak,
        day_start_equity=equity,
        week_start_equity=equity,
        day_start=NOW,
        week_start=NOW,
        trades_today=0,
        trades_this_week=0,
        consecutive_losses=0,
        last_trade_risk_pct=None,
    )


def manager(tmp_path, breaker: float) -> RiskManager:  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    risk = settings.risk.model_copy(
        update={
            "max_drawdown_circuit_breaker_pct": breaker,
            # The validator refuses a breaker at or below the weekly limit, and
            # with the breaker off there is nothing to order it against.
            "weekly_loss_limit_pct": 0.0,
            "daily_loss_limit_pct": 0.0,
        }
    )
    clock = SimulatedClock(NOW)
    journal = Journal(tmp_path / "trading.db", clock).open()
    return RiskManager(settings.model_copy(update={"risk": risk}), journal, clock)


# ------------------------------------------------------- circuit breaker ---


def test_the_breaker_still_trips_when_it_is_set(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert manager(tmp_path, 15.0).circuit_breaker_tripped(state(85.0, 100.0))


def test_a_drawdown_below_the_limit_does_not_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert not manager(tmp_path, 15.0).circuit_breaker_tripped(state(90.0, 100.0))


def test_zero_means_off_not_trip_at_nought(tmp_path) -> None:
    """The failure that would look like the protection working.

    Any drawdown clears a bar of zero, including exactly zero, so without an
    explicit check a disabled breaker halts every account on its first cycle —
    and reports a circuit-breaker trip while doing it.
    """
    risk = manager(tmp_path, NO_DRAWDOWN_BREAKER)
    assert not risk.circuit_breaker_tripped(state(100.0, 100.0)), "flat account"
    assert not risk.circuit_breaker_tripped(state(88.0, 96.0)), "8% down"
    assert not risk.circuit_breaker_tripped(state(50.0, 100.0)), "50% down"
    assert not risk.circuit_breaker_tripped(state(1.0, 100.0)), "99% down"


def test_the_drawdown_is_still_measured_with_the_breaker_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Off means "do not act on it", not "stop looking". The number still
    reaches the log, the deck and the adviser."""
    del tmp_path
    assert state(88.28, 96.17).drawdown_pct == pytest.approx(8.2, abs=0.05)


# --------------------------------------------------------------- posture ---


def test_the_throttle_still_bites_when_it_is_on() -> None:
    verdict = assess(consecutive_losses=0, equity=88.28, equity_peak=96.17)
    assert verdict.posture is Posture.DEFENSIVE
    assert verdict.max_candidates == 1


def test_off_leaves_every_candidate_reachable() -> None:
    verdict = assess(consecutive_losses=0, equity=88.28, equity_peak=96.17, enabled=False)
    assert verdict.posture is Posture.STEADY
    assert verdict.max_candidates is None


def test_off_also_restores_full_patience() -> None:
    """The throttle has two dials. Neutralising only the candidate cap would
    leave stalled trades still being cut at 40% of the usual time."""
    verdict = assess(consecutive_losses=9, equity=50.0, equity_peak=100.0, enabled=False)
    assert verdict.patience_multiplier == 1.0


def test_off_still_reports_the_real_numbers() -> None:
    """The operator has to be able to see what the throttle would have done,
    or turning it off means losing the diagnosis along with the brake."""
    verdict = assess(consecutive_losses=3, equity=88.28, equity_peak=96.17, enabled=False)
    assert verdict.consecutive_losses == 3
    assert verdict.drawdown_pct == pytest.approx(8.2, abs=0.05)


def test_off_is_never_stressed() -> None:
    """`is_stressed` drives a warning banner and the adviser's briefing. A
    disabled throttle must not keep announcing a posture it is not applying."""
    assert not assess(
        consecutive_losses=9, equity=10.0, equity_peak=100.0, enabled=False
    ).is_stressed


def test_the_live_overlay_has_both_switched_off() -> None:
    """What the account is actually running, asserted rather than assumed."""
    from config.loader import PACKAGE_ROOT
    from promotion.experimental import apply_experimental_live_limits

    settings = load_settings(overlay=PACKAGE_ROOT / "config" / "eightcap.yaml", env_overrides=False)
    live = apply_experimental_live_limits(settings)
    assert live.risk.posture_throttle is False
    assert live.risk.max_drawdown_circuit_breaker_pct == NO_DRAWDOWN_BREAKER


def test_new_risk_is_not_blocked_when_the_breaker_is_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The bug the first attempt shipped.

    There were two drawdown comparisons. Switching the breaker off fixed
    `circuit_breaker_tripped` and left this one — the gate the runner actually
    calls each cycle — with its bare `>=`. The result on a live account was
    `NEW RISK HALTED: MAX_DRAWDOWN_CIRCUIT_BREAKER - drawdown 8.22% has reached
    the 0.0% circuit breaker; manual restart required`, and every cycle
    analysed nothing.

    Testing the predicate alone could not catch it, which is the lesson: the
    assertion has to sit at the level the caller uses.
    """
    decision = manager(tmp_path, NO_DRAWDOWN_BREAKER).check_can_trade(state(88.28, 96.19))
    assert decision.approved, decision.detail


def test_new_risk_is_still_blocked_when_the_breaker_is_set(tmp_path) -> None:  # type: ignore[no-untyped-def]
    decision = manager(tmp_path, 15.0).check_can_trade(state(80.0, 100.0))
    assert not decision.approved
    assert decision.reason is Reason.CIRCUIT_BREAKER


def test_both_drawdown_checks_agree(tmp_path) -> None:
    """They are one rule and must not be able to disagree — the two copies
    drifting apart is exactly what broke it."""
    risk = manager(tmp_path, 15.0)
    for equity in (100.0, 90.0, 85.0, 84.9, 50.0):
        current = state(equity, 100.0)
        blocked = risk.check_can_trade(current).reason is Reason.CIRCUIT_BREAKER
        assert blocked == risk.circuit_breaker_tripped(current), equity
