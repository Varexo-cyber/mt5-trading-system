"""Two of the five modules trading real money had no automatic shut-off.

`risk.section_breakers` is what stops a detector that has started losing
without anyone watching. It listed four entries and one of those
(`drift_burst`) is not even on the live allowlist, so the real coverage was
three of five: `market_structure` and `m1_micro_breakout` could trade this
account indefinitely with nothing able to switch them off but a person.

It is the same shape as nearly every defect found on this account: a
protection that exists, is correct, is configured and is tested, sitting
beside the path the code actually takes rather than on it. Nothing was broken
about the breaker mechanism. It just did not cover two of the things it was
there for.

This asserts the RELATION, not the list. A list would have to be edited every
time a module goes live — which is exactly the edit that was missed — while
the relation fails on its own the next time someone adds a module to
`live_enabled_modules` and forgets the rest.
"""

from __future__ import annotations

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings


@pytest.fixture(scope="module")
def settings():  # type: ignore[no-untyped-def]
    return load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False)


def test_nothing_trades_real_money_without_a_way_to_switch_itself_off(settings) -> None:  # type: ignore[no-untyped-def]
    live = set(settings.analysis.confluence.live_enabled_modules)
    protected = set(settings.risk.section_breakers)

    assert live <= protected, (
        f"live with no section breaker: {sorted(live - protected)}. "
        "A module that can open positions must be able to stop itself."
    )


def test_every_breaker_is_actually_armed(settings) -> None:  # type: ignore[no-untyped-def]
    """A breaker entry that exists with `enabled: false` is worse than no
    entry, because the coverage test above would pass on it."""
    live = set(settings.analysis.confluence.live_enabled_modules)

    for module in sorted(live):
        breaker = settings.risk.section_breakers[module]
        assert breaker.enabled, f"{module} has a breaker that is switched off"
        assert breaker.losing_streak > 0, f"{module} has no losing-streak limit"
        assert 0.0 < breaker.maximum_loss_share < 1.0, f"{module} cannot ever trip on loss share"
        assert breaker.minimum_trades > 0, f"{module} would judge itself on an empty sample"


def test_a_breaker_needs_enough_trades_to_mean_something(settings) -> None:  # type: ignore[no-untyped-def]
    """The other failure direction, and the one that costs money quietly: a
    breaker that fires on noise switches off a working detector after a bad
    afternoon and nobody notices it is gone.

    `minimum_trades` may never exceed the window it is judged over, or the
    share can never be computed and the breaker is decorative.
    """
    for module, breaker in settings.risk.section_breakers.items():
        assert breaker.minimum_trades <= breaker.window, (
            f"{module} needs {breaker.minimum_trades} trades inside a window of "
            f"{breaker.window}, which cannot happen"
        )


def test_the_two_swing_modules_are_held_to_the_same_standard(settings) -> None:  # type: ignore[no-untyped-def]
    """`market_structure` and `trend_momentum` plan on the same clock and turn
    their slots over at the same rate. A difference in their emergency stops
    would be a difference resting on nothing, and this is where such a
    difference would be introduced by accident."""
    breakers = settings.risk.section_breakers

    assert breakers["market_structure"].window == breakers["trend_momentum"].window
    assert breakers["market_structure"].losing_streak == breakers["trend_momentum"].losing_streak
    assert (
        breakers["market_structure"].maximum_loss_share
        == breakers["trend_momentum"].maximum_loss_share
    )
