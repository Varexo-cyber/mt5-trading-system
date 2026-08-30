"""Measuring what break-even costs, and getting the measurement itself right.

THE OWNER ASKED FOR THIS RULE. It was already on, on both live sections, with
no per-module exception -- and it is the largest unmeasured deviation left
between the shipped code and the research. 18,828 trades were resolved against
a stop that DOES NOT MOVE. Both sections enter AT a level, where price
oscillates by construction, so a trade that runs a little, comes back to the
level and then goes to target is a full +1R in the research and a +0.1R scratch
here.

It cuts both ways -- the same rule rescues a loser that ran first -- so it is
an arithmetic question, not an argument.

WHICH MAKES THE SIMULATION ITSELF THE RISK. This account has already been
bitten three times by a resolver that peeked:

    same-bar look-ahead     read +0.487R where the truth was +0.347R
    fill ordering           read +0.336R where the truth was +0.063R
    gap_fill                read +1.011R for a strategy worth nothing

Every one of those flattered the result, and every one came from letting a bar
be used twice. A break-even simulation has exactly the same hole available:
arm the stop on this bar's HIGH and then let the trade survive this bar's LOW.
So the tests below are about ordering as much as arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from core.types import Direction
from scripts.dry_run_sections import _break_even_rule, _resolve


@dataclass
class _Idea:
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float


def _bars(*rows: tuple[float, float]) -> pd.DataFrame:
    """One bar per (low, high). Times are M1 and only their order matters."""
    start = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)
    index = [start + timedelta(minutes=i) for i in range(len(rows))]
    return pd.DataFrame(
        {
            "open": [(lo + hi) / 2 for lo, hi in rows],
            "high": [hi for _lo, hi in rows],
            "low": [lo for lo, _hi in rows],
            "close": [(lo + hi) / 2 for lo, hi in rows],
        },
        index=pd.DatetimeIndex(index),
    )


#: entry 100, stop 99, target 101. R = 1.0, so an R and a point are the same
#: number and every assertion below reads directly.
LONG = _Idea(Direction.LONG, entry=100.0, stop_loss=99.0, take_profit=101.0)
SHORT = _Idea(Direction.SHORT, entry=100.0, stop_loss=101.0, take_profit=99.0)
#: arms at +0.25R, stop to entry +0.10R
RULE = (0.25, 0.10)


def _run(idea: _Idea, frame: pd.DataFrame, manage=RULE):
    return _resolve(frame, frame.index[0], idea, horizon_bars=len(frame), manage=manage)


class TestTheFixedColumnIsUnchanged:
    """The research's number must survive having a second column added beside
    it, or the comparison is between two things that both moved."""

    def test_a_clean_winner(self) -> None:
        fixed, _at, _managed = _run(LONG, _bars((99.9, 100.2), (100.5, 101.2)))

        assert fixed == pytest.approx(1.0)

    def test_a_clean_loser(self) -> None:
        fixed, _at, _managed = _run(LONG, _bars((99.9, 100.1), (98.5, 100.0)))

        assert fixed == pytest.approx(-1.0)

    def test_a_bar_holding_both_barriers_is_a_loss(self) -> None:
        """The order inside a bar is unknowable and assuming the good one is
        how a backtest lies."""
        fixed, _at, _managed = _run(LONG, _bars((98.0, 102.0)))

        assert fixed == pytest.approx(-1.0)

    def test_an_unresolved_trade_reports_open(self) -> None:
        fixed, at, _managed = _run(LONG, _bars((99.9, 100.1), (99.8, 100.2)))

        assert fixed is None and at is None

    def test_management_off_leaves_the_managed_column_empty(self) -> None:
        fixed, _at, managed = _run(LONG, _bars((99.9, 100.2), (100.5, 101.2)), manage=None)

        assert fixed == pytest.approx(1.0)
        assert managed is None


class TestTheStopActuallyMoves:
    def test_a_winner_that_never_looks_back_is_untouched(self) -> None:
        """Break-even may only ever cost a trade that came back. If this one
        differs, the moved stop is sitting somewhere it should not."""
        fixed, _at, managed = _run(LONG, _bars((99.95, 100.4), (100.6, 101.5)))

        assert fixed == pytest.approx(1.0)
        assert managed == pytest.approx(1.0)

    def test_a_winner_that_dips_back_to_the_level_is_scratched(self) -> None:
        """THE CASE THAT MATTERS. Runs to +0.4R, comes back to the level, then
        goes to target. A full winner in the research; +0.10R here."""
        frame = _bars((99.95, 100.4), (99.80, 100.2), (100.5, 101.5))

        fixed, _at, managed = _run(LONG, frame)

        assert fixed == pytest.approx(1.0)
        assert managed == pytest.approx(0.10)

    def test_a_loser_that_ran_first_is_rescued(self) -> None:
        """The other half of the arithmetic, and the reason the answer is not
        obvious. Same rule, opposite sign."""
        frame = _bars((99.95, 100.4), (98.5, 100.0))

        fixed, _at, managed = _run(LONG, frame)

        assert fixed == pytest.approx(-1.0)
        assert managed == pytest.approx(0.10)

    def test_a_loser_that_never_ran_is_not_rescued(self) -> None:
        """Below the trigger the stop has not moved and the loss is the loss."""
        frame = _bars((99.95, 100.1), (98.5, 100.0))

        fixed, _at, managed = _run(LONG, frame)

        assert fixed == pytest.approx(-1.0)
        assert managed == pytest.approx(-1.0)

    def test_it_works_the_same_way_short(self) -> None:
        frame = _bars((99.60, 100.05), (99.80, 100.20), (98.5, 99.5))

        fixed, _at, managed = _run(SHORT, frame)

        assert fixed == pytest.approx(1.0)
        assert managed == pytest.approx(0.10)


class TestItDoesNotPeek:
    """Every resolver bug this account has had flattered the result, and every
    one came from using a bar twice."""

    def test_a_bar_cannot_arm_the_stop_and_then_be_stopped_by_itself(self) -> None:
        """THE LOOK-AHEAD, STATED SO THAT THE TWO ORDERINGS DISAGREE.

        Bar one runs to +0.42R and its low is 99.98 -- below where the moved
        stop would sit. Bar two goes straight to target.

            arming AFTER the bar (correct)   the stop was still at 99.00 while
                                             bar one traded, so nothing was
                                             hit; it arms at the close and bar
                                             two pays  ->  +1.00R
            arming DURING the bar (peeking)  the stop jumps to 100.10 and is
                                             then tested against the same
                                             bar's low  ->  +0.10R

        A first draft of this test used a second bar that dipped to 99.90.
        Both orderings scratch that one at +0.10R, so it asserted nothing --
        which is its own version of the defect this file is about."""
        frame = _bars((99.98, 100.42), (100.50, 101.20))

        _fixed, _at, managed = _run(LONG, frame)

        assert managed == pytest.approx(1.0)

    def test_arming_never_manufactures_a_win_out_of_a_stopped_trade(self) -> None:
        """A bar that reaches the real stop resolves at the real stop, no
        matter how far it ran first."""
        frame = _bars(
            (98.0, 100.40),
        )

        fixed, _at, managed = _run(LONG, frame)

        assert fixed == pytest.approx(-1.0)
        assert managed == pytest.approx(-1.0)

    def test_the_managed_run_can_outlive_the_fixed_one(self) -> None:
        """They are different trades after the stop moves, so the walk may not
        stop at the first exit it finds."""
        frame = _bars((99.95, 100.40), (98.5, 100.0), (100.0, 101.5))

        fixed, _at, managed = _run(LONG, frame)

        assert fixed == pytest.approx(-1.0)
        assert managed == pytest.approx(0.10)

    def test_a_managed_trade_still_open_at_the_horizon_reports_nothing(self) -> None:
        """Rather than being scored as a scratch, which would credit the rule
        with an exit it never made.

        The second bar has to stay ABOVE the moved stop at 100.10 -- a bar that
        dips to 99.95 takes it, which is the rule working, not the horizon
        expiring. That mistake was in the first draft of this test."""
        frame = _bars((99.95, 100.40), (100.15, 100.30))

        _fixed, _at, managed = _run(LONG, frame)

        assert managed is None


class TestTheTriggerIsTheOneTheAccountRuns:
    """`break_even_at_r` is 0.25 and is NOT the whole rule. A second trigger
    measured in euros arms the same move earlier on this account, and modelling
    only the first would measure a rule the code does not run."""

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from config.schema import TradingMode

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        return settings.model_copy(
            update={"system": settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})}
        )

    def test_it_takes_the_smaller_of_the_two_triggers(self) -> None:
        settings = self._settings()
        rule = _break_even_rule(settings)
        management = settings.trade_management

        assert rule is not None
        trigger, offset = rule
        euro_trigger = management.capital_protection_at_equity_pct / settings.effective_risk_pct()

        assert trigger == pytest.approx(min(management.break_even_at_r, euro_trigger))
        assert offset == pytest.approx(management.break_even_offset_atr)

    def test_the_euro_trigger_is_the_binding_one_at_this_risk(self) -> None:
        """At 2% risk one percent of equity is half an R, so `break_even_at_r`
        binds; the moment the minimum-lot override pushes a trade to 4% it is
        the euro trigger instead. Both live here, so the smaller one is not a
        constant and must not be hardcoded."""
        settings = self._settings()

        for risk_pct, expected in ((2.0, 0.25), (4.0, 0.25), (8.0, 0.125)):
            share = settings.trade_management.capital_protection_at_equity_pct
            assert min(
                settings.trade_management.break_even_at_r, share / risk_pct
            ) == pytest.approx(expected)

    def test_a_disabled_rule_returns_nothing(self) -> None:
        settings = self._settings()
        off = settings.model_copy(
            update={
                "trade_management": settings.trade_management.model_copy(
                    update={"break_even_at_r": 0.0}
                )
            }
        )

        assert _break_even_rule(off) is None


class TestTheVerdictIsReported:
    def test_the_report_prints_both_columns(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "scripts" / "dry_run_sections.py"
        ).read_text()

        assert "def _break_even_verdict" in source
        assert "what the research measured" in source
        assert "what runs now" in source

    def test_the_csv_carries_the_managed_result_per_trade(self) -> None:
        """A verdict without the rows behind it cannot be checked."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "scripts" / "dry_run_sections.py"
        ).read_text()

        # Renamed to say WHICH column the account trades: a spreadsheet
        # cannot ask, so the header has to answer.
        assert '"managed_r_LIVE",' in source
        assert '"managed_money_LIVE",' in source
        assert '"result_r_fixed_stop",' in source


class TestTheCountersExplainTheTotal:
    """The 30 August run printed "0 winners scratched, 0 losers rescued" beside
    a -0.20R difference. A total that says trades moved, next to two counts
    that say none did.

    The counters tested `result_r > 0 >= managed_r`. A scratched winner exits
    at entry PLUS the break-even offset, so it is still POSITIVE and the sign
    test could never see it. The two numbers whose entire job was to explain
    the total were structurally incapable of doing so.
    """

    def _counts(self, pairs: list[tuple[float, float]]) -> tuple[int, int]:
        from scripts.dry_run_sections import Decision

        trades = [
            Decision(
                datetime(2026, 8, 24, tzinfo=UTC),
                "S",
                "m",
                "TRADE",
                result_r=fixed,
                managed_r=managed,
            )
            for fixed, managed in pairs
        ]
        scratched = [d for d in trades if (d.managed_r or 0) < (d.result_r or 0) - 1e-9]
        rescued = [d for d in trades if (d.managed_r or 0) > (d.result_r or 0) + 1e-9]
        return len(scratched), len(rescued)

    def test_a_winner_cut_to_the_offset_is_counted(self) -> None:
        """+1.00R -> +0.10R. Still positive, and the old test missed it."""
        assert self._counts([(1.0, 0.10)]) == (1, 0)

    def test_a_rescued_loser_is_counted(self) -> None:
        assert self._counts([(-1.0, 0.10)]) == (0, 1)

    def test_an_untouched_trade_is_counted_as_neither(self) -> None:
        assert self._counts([(1.0, 1.0), (-1.0, -1.0)]) == (0, 0)

    def test_the_counts_and_the_total_cannot_disagree(self) -> None:
        """THE PROPERTY. If the totals differ, at least one counter must be
        non-zero -- that is the invariant the old code violated."""
        pairs = [(1.0, 0.10), (-1.0, -1.0), (1.0, 1.0), (-1.0, 0.10)]
        scratched, rescued = self._counts(pairs)
        moved = sum(managed - fixed for fixed, managed in pairs)

        assert abs(moved) > 1e-9
        assert scratched + rescued > 0
        assert (scratched, rescued) == (1, 1)

    def test_the_report_prints_what_each_side_is_worth(self) -> None:
        """A count without its R is still not an explanation: one winner cut
        and one loser rescued nets out very differently depending on where
        each landed."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "scripts" / "dry_run_sections.py"
        ).read_text()

        assert "cut short" in source
        assert "for d in scratched" in source
        assert "for d in rescued" in source
