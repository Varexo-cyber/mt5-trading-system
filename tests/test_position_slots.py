"""How many positions the account may carry, given what it holds.

A fixed number of slots is the wrong shape for an account meant to grow. Two
is right at EUR 100 — three simultaneous trades there means three the account
cannot express at the minimum lot anyway — and plainly wrong at EUR 1,000,
where the constraint is no longer size but the fact that somebody once typed
a 2.

The mode's ceiling stays absolute. Growth widens the book toward a limit a
human set; it never walks through it.
"""

from __future__ import annotations

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from promotion.experimental import apply_experimental_live_limits


@pytest.fixture(scope="module")
def settings():  # type: ignore[no-untyped-def]
    return load_settings(env_overrides=False)


def scaled(settings, equity: float, *, step: float, ceiling: int, floor: int = 2) -> int:
    tuned = settings.model_copy(
        update={
            "risk": settings.risk.model_copy(
                update={"equity_per_position": step, "min_concurrent_positions": floor}
            )
        },
        deep=True,
    )
    tuned.active_limits.max_concurrent_positions  # noqa: B018 - fail loudly if absent
    limits = tuned.active_limits.model_copy(update={"max_concurrent_positions": ceiling})
    tuned = tuned.model_copy(update={"modes": {**tuned.modes, tuned.mode.value: limits}}, deep=True)
    return tuned.effective_max_positions(equity)


class TestScaling:
    def test_no_equity_given_returns_the_ceiling(self, settings) -> None:  # type: ignore[no-untyped-def]
        """The startup banner wants the shape of the mode, not today's balance."""
        assert settings.effective_max_positions() == settings.active_limits.max_concurrent_positions

    def test_off_by_default_so_nothing_changes_silently(self, settings) -> None:  # type: ignore[no-untyped-def]
        assert settings.risk.equity_per_position == 0.0
        assert settings.effective_max_positions(10_000.0) == (
            settings.active_limits.max_concurrent_positions
        )

    def test_a_small_account_gets_the_floor_not_less(self, settings) -> None:  # type: ignore[no-untyped-def]
        """EUR 87 earns one slot on a EUR 50 step. One is not enough.

        With a single slot, one open trade blocks every other opportunity for
        as long as it runs.
        """
        assert scaled(settings, 87.0, step=50.0, ceiling=6) == 2

    def test_growth_earns_slots_in_whole_steps(self, settings) -> None:  # type: ignore[no-untyped-def]
        assert scaled(settings, 149.0, step=50.0, ceiling=6) == 2
        assert scaled(settings, 150.0, step=50.0, ceiling=6) == 3
        assert scaled(settings, 300.0, step=50.0, ceiling=6) == 6

    def test_the_ceiling_is_absolute(self, settings) -> None:  # type: ignore[no-untyped-def]
        """A rich account does not walk through the limit a human set."""
        assert scaled(settings, 100_000.0, step=50.0, ceiling=4) == 4

    def test_the_ceiling_also_bounds_the_floor(self, settings) -> None:  # type: ignore[no-untyped-def]
        """A mode allowing one position keeps allowing exactly one.

        The floor is a preference; the mode ceiling is a rule, and a rule that
        a preference can override is not a rule.
        """
        assert scaled(settings, 87.0, step=50.0, ceiling=1, floor=3) == 1

    def test_a_drawdown_narrows_the_book_again(self, settings) -> None:  # type: ignore[no-untyped-def]
        """Symmetric on purpose — the same arithmetic, run downhill."""
        assert scaled(settings, 300.0, step=50.0, ceiling=6) == 6
        assert scaled(settings, 120.0, step=50.0, ceiling=6) == 2

    def test_a_wiped_account_still_reports_at_least_one(self, settings) -> None:  # type: ignore[no-untyped-def]
        """Zero slots would be a division-by-zero waiting to happen elsewhere.

        The equity floor and the circuit breaker stop trading long before this
        matters; the number just has to stay sane.
        """
        assert scaled(settings, 0.0, step=50.0, ceiling=6, floor=1) == 1

    def test_the_slot_count_matches_the_stake_and_the_book_cap(self) -> None:
        """Four slots were calibrated for a 5% stake: 4 x 5 = 20% against a 24%
        book cap. The stake went to 2% on 27 August and the slots did not move,
        so half the sanctioned risk sat unused -- and the SLOT count is what
        decides how many trades a day there are, not any gate.

        A swing trade holds its slot for 24 hours (`horizon_profiles.swing` is
        H1 x 24 bars, and `time_exit_hours` is 24), so four slots is four
        trades a day however many setups appear. That bit harder the same
        afternoon: of the six modules switched off, five were intraday modules
        and two of the three quick ones, so almost everything left classifies
        as swing.

        Eight slots at the ordinary 2% is exactly the 16% book cap, so this
        raises throughput without raising sanctioned risk by a basis point.
        `max_total_open_risk_pct` remains the binding rule and tightens on its
        own as conviction sizes trades up: at the 8% ceiling only two fit.
        """
        overlay = DEFAULT_CONFIG_PATH.with_name("eightcap.yaml")
        settings = apply_experimental_live_limits(
            load_settings(DEFAULT_CONFIG_PATH, overlay=overlay, env_overrides=False)
        )

        assert settings.risk.equity_per_position == 0.0
        assert settings.effective_max_positions(153.03) == 8
        # The property, not just the number: a full book of ordinary trades
        # must land ON the cap, never past it.
        ordinary = settings.risk.conviction_risk.floor_pct
        assert settings.effective_max_positions(153.03) * ordinary == (
            settings.risk.max_total_open_risk_pct
        )
        assert settings.trade_management.pyramiding.enabled
        assert settings.trade_management.pyramiding.max_legs_per_symbol == 3
        assert settings.trade_management.pyramiding.max_active_symbols == 1
        assert not settings.trade_management.pyramiding.counts_toward_position_limit
        assert settings.trade_management.pyramiding.risk_multiplier == 0.25
        assert settings.trade_management.pyramiding.minimum_conviction == 85
        assert settings.trade_management.pyramiding.minimum_ai_confidence == 0.80
