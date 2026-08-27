"""The plan knew how long the trade should take. The exit never asked.

`ConfluenceEngine` has decided a horizon per proposal for as long as horizons
have existed — thirty minutes for `quick`, three hours for `intraday`,
twenty-four for `swing` — and writes it as `expected_horizon_minutes` under a
comment that reads, in the engine's own words, "everything downstream derives
its window from this number".

The reach gate derives its window from it. The survival gate does. The runway
check does. The reviewer briefing carries it, the brain stores it, the
scorecard reports it. `execution/manager.py` did not contain the word horizon.
Its stalled-trade deadline was `config.time_exit_hours` — one constant, 24.0,
for every trade this account has ever taken.

TWO THINGS FOLLOW FROM THAT, and they are the two complaints.

THE SLOT. A position slot is released when the position closes, and the number
of slots is the hard ceiling on trades per day: eight slots holding trades for
twenty-four hours each is eight trades a day, whatever the scanner finds. The
same eight slots holding three-hour plans for three hours is not eight.

THE TRADE. A detector that reads the next thirty minutes, held for
twenty-four hours, is judged almost entirely on moves it never claimed to see.
That is a mechanism for the 90-day finding that no detector beat a coin flip
which does not require a single detector to be worthless — only to have been
measured over a window twenty times longer than the one it speaks about.

WHAT DOES NOT CHANGE IS SWING, and that is arithmetic rather than caution:
H1 x 24 bars is 1,440 minutes is 24.0 hours is exactly the constant that was
already there. The two were the same number the whole time; only one of them
was ever applied.
"""

from __future__ import annotations

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from execution.manager import PositionManager


def _config(**changes):  # type: ignore[no-untyped-def]
    management = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    ).trade_management
    return management.model_copy(update=changes) if changes else management


def _deadline(minutes, *, patience: float = 1.0, **changes) -> float | None:  # type: ignore[no-untyped-def]
    """A dict stands in for the journal row on purpose.

    The production row is a `sqlite3.Row`, which raises `IndexError` for an
    unknown column while a dict raises `KeyError`. Both are caught, and a dict
    is the harsher of the two to get right because a missing key looks exactly
    like a present one until it is read.
    """
    row = {} if minutes is _ABSENT else {"expected_horizon_minutes": minutes}
    return PositionManager._time_exit_deadline(_config(**changes), row, patience)


_ABSENT = object()

# What the three profiles plan, in minutes. Not restated from config: these are
# planning_timeframe x target_horizon_bars, which is how the engine builds the
# number it writes on the trade.
SWING, INTRADAY, QUICK = 24 * 60, 12 * 15, 6 * 5


class TestTheDeadlineIsThePlan:
    def test_a_swing_trade_keeps_exactly_the_deadline_it_always_had(self) -> None:
        """The regression that matters most, because it is the case where a
        change would be silent: swing plans 1,440 minutes, the old constant was
        24.0 hours, and 1,440 minutes is 24.0 hours.

        At a multiple of 1.5 the derived figure is 36 hours, and the ceiling
        holds it at 24 — the plan may shorten the leash, never lengthen it.
        """
        assert _deadline(SWING) == pytest.approx(24.0)

    def test_an_intraday_trade_is_not_held_for_a_day(self) -> None:
        """Three hours of plan, four and a half hours of rope. The old
        behaviour gave it twenty-four."""
        assert _deadline(INTRADAY) == pytest.approx(4.5)

    def test_a_quick_trade_is_not_held_for_a_day_either(self) -> None:
        """Thirty minutes of plan, forty-five of rope, against twenty-four
        hours before."""
        assert _deadline(QUICK) == pytest.approx(0.75)

    def test_the_floor_stops_a_tiny_plan_becoming_a_tripwire(self) -> None:
        """A plan of two minutes is not a deadline, it is the spread. The floor
        is the quick profile's whole horizon."""
        assert _deadline(2.0) == pytest.approx(_config().time_exit_minimum_hours)


class TestWhatHasNoPlanKeepsTheOldRule:
    """Three ways a row can fail to say, and all three must land on the
    constant rather than on a guess. This is where a change like this usually
    breaks something: an adopted position has no plan and never will."""

    def test_a_row_from_before_the_column_existed(self) -> None:
        assert _deadline(_ABSENT) == pytest.approx(24.0)

    def test_a_position_adopted_rather_than_planned(self) -> None:
        assert _deadline(None) == pytest.approx(24.0)

    def test_a_zero_is_read_as_absent_and_not_as_an_instant_deadline(self) -> None:
        """The dangerous one. Zero minutes read literally is a deadline that
        has already passed, which would close every trade on its first pass."""
        assert _deadline(0) == pytest.approx(24.0)

    def test_the_whole_thing_can_be_switched_off(self) -> None:
        assert _deadline(QUICK, time_exit_uses_plan_horizon=False) == pytest.approx(24.0)

    def test_no_deadline_at_all_stays_no_deadline(self) -> None:
        assert _deadline(QUICK, time_exit_hours=None) is None


class TestPatience:
    def test_the_drawdown_posture_still_shortens_the_derived_deadline(self) -> None:
        """`patience` scales the timeout downward in a drawdown, and it has to
        keep working on the number the plan produced rather than only on the
        constant it replaced."""
        assert _deadline(INTRADAY, patience=0.5) == pytest.approx(2.25)

    def test_it_is_applied_once_and_not_twice(self) -> None:
        """Stated as its own test because the first version of this method
        applied it to the ceiling before comparing, and then again to the
        result — so a swing trade in a drawdown would have been cut to a
        quarter of its deadline instead of a half."""
        assert _deadline(SWING, patience=0.5) == pytest.approx(12.0)


class TestTheDeadlineIsNotItselfAnExit:
    """Reaching the deadline does not close a working trade, and that is what
    makes shortening it safe. `_time_exit_verdict` closes on two findings only:
    the trade went nowhere, or it is green but never got going."""

    def test_a_trade_that_is_working_survives_its_own_deadline(self) -> None:
        config = _config()

        verdict = PositionManager._time_exit_verdict(
            config, age_hours=10.0, deadline=4.5, r_now=1.8, peak_r=2.4
        )

        assert verdict is None

    def test_a_trade_that_went_nowhere_is_released_at_the_deadline(self) -> None:
        """This is the throughput. Dead capital in a slot is what caps the
        number of trades a day, and the plan is what says when it went dead."""
        config = _config()

        verdict = PositionManager._time_exit_verdict(
            config, age_hours=5.0, deadline=4.5, r_now=0.05, peak_r=0.2
        )

        assert verdict == "went nowhere"

    def test_before_the_deadline_nothing_happens_at_all(self) -> None:
        config = _config()

        verdict = PositionManager._time_exit_verdict(
            config, age_hours=1.0, deadline=4.5, r_now=0.0, peak_r=0.0
        )

        assert verdict is None
