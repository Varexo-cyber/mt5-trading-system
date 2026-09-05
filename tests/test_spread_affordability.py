"""Can a trade afford its own spread?

A different question from the one the spread filter asks. That filter compares
the spread against what this instrument normally does at this hour — and in the
evening the answer is that the spread is entirely normal, because the baseline
it learned is itself an evening baseline. The stop, meanwhile, did not widen.

A 2-pip spread against a 6-pip stop means the trade opens a third of the way to
being wrong and has to clear the spread twice before it earns anything. No edge
in the setup survives that. The playbooks already refused on this; the
confluence path did not, which is where the evening stop-outs came from.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.types import MarketContext, Tick
from runner.service import JarvisRunner

NOW = datetime(2026, 8, 4, 21, 30, tzinfo=UTC)


def gate(limit: float = 0.20, by_family=None):  # type: ignore[no-untyped-def]
    """The method under test, on a runner with only the setting it reads."""
    from types import SimpleNamespace

    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = SimpleNamespace(  # type: ignore[assignment]
        analysis=SimpleNamespace(
            confluence=SimpleNamespace(
                max_spread_share_of_stop=limit,
                max_spread_share_of_stop_by_family=by_family or {},
            )
        )
    )
    return instance._spread_is_affordable


def context(spread: float, mid: float = 1.0800) -> MarketContext:
    half = spread / 2
    return MarketContext(
        symbol="EURUSD",
        now=NOW,
        series={},
        tick=Tick(symbol="EURUSD", time=NOW, bid=mid - half, ask=mid + half),
    )


def test_a_tight_spread_against_a_normal_stop_passes() -> None:
    # 0.4 pips against a 20-pip stop.
    ok, share = gate()(context(0.00004), 1.0800, 1.0780)
    assert ok
    assert share < 0.05


def test_the_evening_case_is_refused() -> None:
    """The spread the hour-of-day baseline waves through, against a stop that
    did not widen with it. 2.4 pips of spread on a 6-pip stop is 40%."""
    ok, share = gate()(context(0.00024), 1.0800, 1.0794)
    assert not ok
    assert share > 0.35


def test_the_same_spread_passes_on_a_wider_stop() -> None:
    """The gate is about the ratio, not the spread. A swing trade can carry a
    cost a scalp cannot, and refusing both would throw away good trades in
    exactly the sessions that suit them."""
    ok, _ = gate()(context(0.00024), 1.0800, 1.0770)
    assert ok


def test_the_boundary_is_inclusive() -> None:
    """Exactly at the limit is allowed; the config value reads as a ceiling."""
    ok, share = gate(0.20)(context(0.00020), 1.0800, 1.0790)
    assert ok
    assert share == pytest.approx(0.2)


def test_a_missing_tick_fails_closed() -> None:
    """An unanswerable cost question is not a reason to pay it."""
    blind = MarketContext(symbol="EURUSD", now=NOW, series={}, tick=None)
    ok, share = gate()(blind, 1.0800, 1.0780)
    assert not ok
    assert share == 1.0


def test_a_zero_width_stop_fails_closed() -> None:
    """Any spread is infinitely large against a stop at entry, and dividing by
    it would raise rather than refuse."""
    ok, _ = gate()(context(0.00004), 1.0800, 1.0800)
    assert not ok


def test_the_short_side_is_measured_the_same_way() -> None:
    """Distance, not direction — a short's stop sits above entry."""
    ok, share = gate()(context(0.00004), 1.0800, 1.0820)
    assert ok
    assert share < 0.05


def test_a_looser_limit_admits_more() -> None:
    tight = gate(0.10)(context(0.00024), 1.0800, 1.0784)
    loose = gate(0.30)(context(0.00024), 1.0800, 1.0784)
    assert not tight[0]
    assert loose[0]


def test_a_measured_family_limit_does_not_loosen_every_strategy() -> None:
    measured = gate(0.08, {"section_sixteen_btc_m5": 0.12})
    spread = context(0.00010)
    assert measured(spread, 1.0800, 1.0790, "section_sixteen_btc_m5")[0]
    assert not measured(spread, 1.0800, 1.0790, "some_other_strategy")[0]
