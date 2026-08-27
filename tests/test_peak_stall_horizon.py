"""The stall clock was built for an M5 bar and applied to a 24-hour plan.

`peak_stall_minutes` closes a profitable position whose peak has not advanced
for the configured wait. The rule's own docstring states what that wait was
sized against, in its own words: "roughly a M5 bar plus confirmation". M5 is
the QUICK profile's planning timeframe. The rule was built for a thirty-minute
trade.

It was then applied, as a wall-clock constant, to a trade planned on H1 with a
target twenty-four hours out — where one bar is sixty minutes and four minutes
of no new high is not a move that has finished, it is a trade breathing.

THE ACCOUNT ALREADY MEASURED THE DAMAGE and the number is written twice in
`execution/manager.py`: PEAK_STALL banked +0.54R where leaving the position
alone returned +1.17R, a lift of -0.64R. The recorded response was to make the
rule wait longer, and the file's own comment on that response says it "changes
when it fires and not what it pays". That is exactly right, and it is why
lengthening the constant could never have worked: the wait was still being
counted on a clock that had nothing to do with the trade.

THE DIRECTION IS THE SAFETY ARGUMENT. The configured minutes remain the floor,
so the rule can only ever fire LATER than it does today and never sooner, and a
position with no recorded plan is managed exactly as it is now. Waiting longer
before closing a position in profit risks money left on the table; the account
has a measured -0.64R saying that is the cheaper of the two mistakes here.
"""

from __future__ import annotations

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from execution.manager import PositionManager

# planning_timeframe x target_horizon_bars, which is how the engine builds the
# number it writes on the trade.
SWING, INTRADAY, QUICK = 24 * 60, 12 * 15, 6 * 5


def _config(**changes):  # type: ignore[no-untyped-def]
    management = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    ).trade_management
    return management.model_copy(update=changes) if changes else management


def _wait(planned, **changes) -> float:  # type: ignore[no-untyped-def]
    return PositionManager._peak_stall_wait(_config(**changes), planned)


class TestTheWaitFollowsThePlan:
    def test_a_swing_trade_is_no_longer_closed_after_four_minutes(self) -> None:
        """The defect as a number. A 24-hour plan got the same four minutes as
        a 30-minute one."""
        assert _wait(SWING) == pytest.approx(SWING * 0.15)
        assert _wait(SWING) > _config().peak_stall_minutes

    def test_an_intraday_trade_gets_a_share_of_its_three_hours(self) -> None:
        assert _wait(INTRADAY) == pytest.approx(27.0)

    def test_a_quick_trade_barely_moves_because_the_rule_was_built_for_it(self) -> None:
        """4.5 minutes against the 4.0 it had. The constant was already close
        to right for the one profile it was designed against — which is exactly
        why applying it everywhere looked reasonable."""
        assert _wait(QUICK) == pytest.approx(4.5)


class TestItCanOnlyEverWaitLonger:
    """The whole safety argument, asserted rather than described."""

    @pytest.mark.parametrize("planned", [None, 1.0, QUICK, INTRADAY, SWING, 10_000.0])
    def test_never_shorter_than_the_configured_floor(self, planned: float | None) -> None:
        assert _wait(planned) >= _config().peak_stall_minutes

    def test_a_position_with_no_recorded_plan_is_managed_exactly_as_before(self) -> None:
        """An adopted position, or one opened before the column existed. The
        old constant is the honest answer for it."""
        assert _wait(None) == pytest.approx(_config().peak_stall_minutes)

    def test_a_zero_share_restores_the_flat_constant(self) -> None:
        assert _wait(SWING, peak_stall_share_of_horizon=0.0) == pytest.approx(
            _config().peak_stall_minutes
        )


class TestTheSameDefectInBothPlaces:
    """The time exit and the stall clock had the same fault, in the same file,
    one screen apart: a constant that was right for one horizon applied to all
    three. They now read the plan through one method, so a row that cannot say
    lands on the old behaviour in both."""

    def test_both_rules_read_the_plan_through_the_same_accessor(self) -> None:
        import inspect

        source = inspect.getsource(PositionManager)

        assert source.count("def _planned_minutes") == 1
        assert "self._planned_minutes(row)" in source
        assert "PositionManager._planned_minutes(row)" in source
