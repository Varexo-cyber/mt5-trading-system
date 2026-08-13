"""Refuse a target this market does not go to often enough to break even.

Six consecutive live refusals, every one citing this exact number and nothing
else deciding it:

    UK100  SHORT   30.2%
    CADCHF LONG    30.1% up against 22.6% down
    AUDUSD LONG    37.0% up against 37.5% down — "essentially a coin flip"
    AUDSGD LONG    38.1% up against 46.8% DOWN
    GBPUSD LONG    41.2%

The measurement was already being computed, purely to be printed in the review
payload — where the reviewer read it and refused the trade. Five cents a time
for a number the engine had in hand before it asked.

Two properties matter and they pull in opposite directions. It has to be strict
enough to catch the plan that cannot work arithmetically, and it must never
refuse on ignorance: an instrument with too little history has not failed the
test, it has not taken it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.target_reach import ReachVerdict, break_even_rate, measure


def frame(
    drift: float = 0.0, noise: float = 0.0004, bars: int = 500, seed: int = 7
) -> pd.DataFrame:
    """A random walk with a controllable lean.

    `seed` is a parameter rather than a constant because the up and down reach
    rates of any one walk are not symmetric — a single seed cannot produce
    every shape these tests need, and quietly reusing one that happens to fall
    for six hundred bars would make an assertion about direction pass for the
    wrong reason.
    """
    index = pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.Series(1.10 + np.cumsum(rng.normal(drift, noise, bars)), index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 0.0004, "low": close - 0.0004, "close": close},
        index=index,
    )


class TestBreakEvenIsArithmeticNotOpinion:
    """Reach rate is an upper bound on win rate: a trade cannot win without the
    market travelling to its target. Below break-even the plan is beaten before
    the stop, the spread and the commission are even considered."""

    @pytest.mark.parametrize(
        ("reward_risk", "expected"),
        [(1.0, 50.0), (2.0, 33.3), (3.0, 25.0), (4.0, 20.0)],
    )
    def test_the_required_rate_follows_the_plans_own_geometry(
        self, reward_risk: float, expected: float
    ) -> None:
        assert break_even_rate(reward_risk) == pytest.approx(expected, abs=0.1)

    def test_a_plan_with_no_reward_can_never_clear_it(self) -> None:
        """Guards the division. A degenerate plan must fail, not raise."""
        assert break_even_rate(0.0) == 100.0

    def test_the_uk100_case_is_refused(self) -> None:
        """30.2% against a 2:1 plan needing 33.3%. Beaten before it opened."""
        verdict = measure(frame(), distance=0.0055, bars_ahead=24, long=True, reward_risk=2.0)

        assert verdict.forward_pct < verdict.required_pct
        assert not verdict.clears_break_even


#: The live tolerance, in percentage points. Every assertion below passes it
#: rather than zero, because the bare `forward >= opposite` it replaced refused
#: on gaps smaller than the error bar on its own measurement — see
#: `TestNoiseIsNotADisadvantage`.
TOLERANCE = 5.0


class TestTheDirectionWasNeverComparedToTheOtherSide:
    """AUDSGD was proposed LONG on an instrument that falls that far more often
    than it rises that far. The direction came from an EMA and the target from
    a multiplier, and nothing ever put the two side by side."""

    def test_a_long_into_a_falling_market_loses_the_advantage_test(self) -> None:
        verdict = measure(
            frame(drift=-0.00012), distance=0.0020, bars_ahead=24, long=True, reward_risk=2.0
        )

        assert verdict.opposite_pct > verdict.forward_pct
        assert not verdict.beats_the_other_side(TOLERANCE)

    def test_the_same_market_shorted_keeps_it(self) -> None:
        verdict = measure(
            frame(drift=-0.00012), distance=0.0020, bars_ahead=24, long=False, reward_risk=2.0
        )

        assert verdict.beats_the_other_side(TOLERANCE)

    def test_the_two_tests_are_independent(self) -> None:
        """A target can clear break-even and still be the wrong way round,
        which is exactly the AUDSGD shape: 38.1% clears a 33.3% bar and is
        beaten by 46.8% in the other direction. One test alone would have let
        that trade through."""
        verdict = measure(frame(seed=0), distance=0.0010, bars_ahead=24, long=True, reward_risk=2.0)

        assert verdict.clears_break_even
        assert not verdict.beats_the_other_side(TOLERANCE)


class TestItNeverRefusesOnIgnorance:
    """An instrument with no history has not failed the test, it has not taken
    it. Refusing there would silently ban every thinly-quoted market."""

    def test_too_few_bars_reports_unmeasured(self) -> None:
        verdict = measure(frame(bars=10), distance=0.002, bars_ahead=24, long=True, reward_risk=2.0)

        assert not verdict.measured
        assert verdict.windows == 0

    def test_an_empty_frame_reports_unmeasured(self) -> None:
        empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": []})

        assert not measure(
            empty, distance=0.002, bars_ahead=24, long=True, reward_risk=2.0
        ).measured

    def test_a_zero_distance_reports_unmeasured(self) -> None:
        """Otherwise every window trivially "reaches" a zero-width target and
        the gate would wave through a plan with no target at all."""
        assert not measure(
            frame(), distance=0.0, bars_ahead=24, long=True, reward_risk=2.0
        ).measured


class TestItAgreesWithTheSlowVersionItReplaces:
    def test_the_vectorised_count_matches_a_plain_loop(self) -> None:
        """The payload builder keeps its own loop and the two must not drift;
        a gate that disagrees with the number shown to the reviewer would be
        the worst possible outcome of this change."""
        data = frame()
        distance, bars_ahead = 0.0020, 24
        closes = data["close"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        windows = len(closes) - bars_ahead
        up = down = 0
        for start in range(windows):
            ahead = slice(start + 1, start + 1 + bars_ahead)
            if highs[ahead].max() - closes[start] >= distance:
                up += 1
            if closes[start] - lows[ahead].min() >= distance:
                down += 1

        verdict = measure(
            data, distance=distance, bars_ahead=bars_ahead, long=True, reward_risk=2.0
        )

        assert verdict.windows == windows
        assert verdict.forward_pct == pytest.approx(100.0 * up / windows, abs=0.05)
        assert verdict.opposite_pct == pytest.approx(100.0 * down / windows, abs=0.05)

    def test_it_states_both_sides_and_the_bar_in_the_refusal(self) -> None:
        """The operator reading the journal has to be able to check it."""
        text = measure(
            frame(), distance=0.002, bars_ahead=24, long=True, reward_risk=2.0
        ).describe()

        assert "% of the time" in text
        assert "the other way" in text
        assert "break even" in text


class TestNoiseIsNotADisadvantage:
    """The gate was reading its own error bar and calling the sign evidence.

    Live, one hour, on the real account: ASX200 refused LONG at 47.4% against
    49.0%, EURAUD at 35.3% against 35.8%. Over ~388 windows those gaps are 0.63
    and 0.21 standard errors. TARGET_RARELY_REACHED fired 127 times in that
    hour and a large share of it was this — differences a fifth the size of the
    noise on the number being compared.
    """

    @staticmethod
    def _verdict(forward: float, opposite: float, windows: int = 388) -> ReachVerdict:
        return ReachVerdict(
            windows=windows, forward_pct=forward, opposite_pct=opposite, required_pct=33.3
        )

    def test_the_asx200_refusal_would_no_longer_happen(self) -> None:
        assert self._verdict(47.4, 49.0).beats_the_other_side(TOLERANCE)

    def test_the_euraud_refusal_would_no_longer_happen(self) -> None:
        assert self._verdict(35.3, 35.8).beats_the_other_side(TOLERANCE)

    def test_a_real_disadvantage_is_still_refused(self) -> None:
        """AUDSGD: 38.1% up against 46.8% down. 8.7 points, over three standard
        errors, and exactly what this gate was built for."""
        assert not self._verdict(38.1, 46.8).beats_the_other_side(TOLERANCE)

    def test_the_error_bar_wins_when_it_is_larger(self) -> None:
        """A fixed tolerance is a floor, not a ceiling. On a thin sample the
        measured error is bigger than five points and a gap inside it must not
        slip through on the strength of a hardcoded number."""
        thin = self._verdict(40.0, 47.0, windows=25)

        # Seven points apart, which the fixed tolerance alone would refuse...
        assert thin.opposite_pct - thin.forward_pct > TOLERANCE
        # ...but on 25 windows the error bar is nearly ten, so it is not a gap.
        assert thin.standard_error_pct > 7.0
        assert thin.beats_the_other_side(TOLERANCE)

    def test_a_deep_sample_holds_the_line_at_the_tolerance(self) -> None:
        """And with enough history the error bar shrinks below the tolerance,
        so the tolerance is what binds — a deliberate 5-point dead zone rather
        than an accident of sample size."""
        deep = self._verdict(40.0, 52.0, windows=10_000)

        assert deep.standard_error_pct < 5.0
        assert not deep.beats_the_other_side(TOLERANCE)

    def test_zero_tolerance_restores_the_old_behaviour_but_for_the_error_bar(self) -> None:
        """Switching the tolerance off does not switch the statistics off. A
        gap inside the measurement's own noise is still not a disadvantage,
        whatever the config says."""
        assert self._verdict(47.4, 49.0).beats_the_other_side(0.0)
        assert not self._verdict(38.1, 46.8).beats_the_other_side(0.0)
