"""Does this market ever actually travel the distance the target asks for?

The engine sets the target at a multiple of the stop and never asks whether the
instrument goes that far in the time allowed. The measurement that answers it
already exists — `advisory.providers._reachability` — and is computed only to
be printed in the review payload, where the reviewer reads it and refuses the
trade. Six consecutive live refusals, every one citing this number:

    UK100  SHORT   target reached 30.2% of the time
    CADCHF LONG    30.1% up against 22.6% down
    AUDUSD LONG    37.0% up against 37.5% down — "essentially a coin flip"
    AUDSGD LONG    38.1% up against 46.8% DOWN
    GBPUSD LONG    41.2%
    EURCAD SHORT   43.4% down against 34.6% up, "not a decisive edge"

Paying five cents a time to be told a number the engine could have read itself.

TWO TESTS, and the first is arithmetic rather than opinion. Reach rate is an
upper bound on win rate: a trade cannot win without the market travelling to
its target, so if the base rate is below the break-even hit rate implied by the
plan's own reward-to-risk, the plan cannot work even before the stop, the
spread and the commission are considered. At RR 2 break-even is 33%, and UK100
at 30.2% was already beaten before it opened.

The second is the sharper one. AUDSGD was proposed LONG on an instrument that,
over the same horizon and distance, has historically fallen that far more often
than it has risen that far. Nothing in the engine noticed; the direction came
from an EMA and the target from a multiplier, and the two were never compared.

WHAT THIS IS NOT. Reach counts up-moves and down-moves independently over the
same windows, so both can be high in a volatile market and neither is the
probability of reaching the target before the stop. It cannot say a trade will
win. It can only say when a trade cannot: that is why it is written as a floor
and not as a score, and why the margin above break-even is small and
configurable rather than a confident number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ReachVerdict:
    """How often the market went each way, and whether that clears the bar."""

    windows: int
    forward_pct: float
    opposite_pct: float
    required_pct: float

    @property
    def measured(self) -> bool:
        """False when there was not enough history to say anything at all."""
        return self.windows > 0

    @property
    def clears_break_even(self) -> bool:
        return self.forward_pct >= self.required_pct

    @property
    def standard_error_pct(self) -> float:
        """The noise on `forward_pct`, in percentage points.

        A share measured over `windows` samples is not an exact number, and
        this gate was being asked to resolve differences far smaller than its
        own error bar. At 388 windows around 40% the standard error is about
        2.5 points, so two readings less than that apart are the same reading.
        """
        if self.windows <= 0:
            return 0.0
        share = max(0.0, min(1.0, self.forward_pct / 100.0))
        return 100.0 * float(np.sqrt(share * (1.0 - share) / self.windows))

    def beats_the_other_side(self, tolerance_pct: float = 0.0) -> bool:
        """Is the other direction better by more than measurement noise?

        THE BUG THIS FIXES cost 127 refusals an hour on live data. The test was
        a bare `forward >= opposite`, so it refused ASX200 at 47.4% against
        49.0% and EURAUD at 35.3% against 35.8%. Those gaps are 0.63 and 0.21
        standard errors — a fifth of the noise on the number itself. The gate
        was not measuring a disadvantage, it was reading its own error bar and
        calling the sign of it evidence.

        What it exists for is real and survives: AUDSGD proposed LONG at 38.1%
        up against 46.8% down is 8.7 points, well over three standard errors,
        and is still refused. `tolerance_pct` is the floor under what counts as
        a difference; the measured error bar is used when it is larger, so a
        thin sample cannot sneak past on a fixed number.
        """
        margin = max(tolerance_pct, self.standard_error_pct)
        return self.forward_pct >= self.opposite_pct - margin

    def describe(self) -> str:
        return (
            f"this market travelled the target distance {self.forward_pct:.1f}% of the time "
            f"in {self.windows} comparable windows ({self.opposite_pct:.1f}% the other way); "
            f"the plan's reward-to-risk needs {self.required_pct:.1f}% just to break even "
            f"before costs"
        )


@dataclass(frozen=True, slots=True)
class SurvivalVerdict:
    """How often the target was reached WHILE THE TRADE WAS STILL ALIVE.

    The difference from `ReachVerdict` is the stop. Reach counts a window as a
    success whenever price ever covered the distance inside the horizon, even
    if it first fell a full R and would have been closed long before the rally.
    That is the right statistic for "which way does this instrument travel",
    and the wrong one for "does this plan pay", because the plan owns a stop.

    It matters most on this account because the stop it owns is not the stop
    the analysis chose. `_widen_stop_for_costs` pushes the stop out so the
    commission stops being the trade, and on a €160 account it fires on nearly
    every candidate. Widening does two things at once: it lowers
    reward-to-risk, and it makes the position far harder to stop out. The old
    gate charged the first and never credited the second — it re-tested an
    unconditional reach rate, unchanged by the wider stop, against a break-even
    requirement the wider stop had just raised. Every widened trade was scored
    on the cost of the widening and none of its benefit.
    """

    #: Every window the history could supply.
    windows: int
    #: Of those, the ones that ended at the target or at the stop. The rest
    #: expired with neither touched, and pricing those as stop-outs is the
    #: error this gate was making.
    resolved_windows: int
    #: Share of the RESOLVED windows that reached the target first. This is the
    #: population `required_pct` was derived for.
    forward_pct: float
    #: Share of ALL windows that reached it — the looser reading, kept because
    #: it is what an operator means by "how often does this actually happen".
    reach_pct: float
    required_pct: float
    expected_r: float
    reward_risk: float
    cost_r: float

    @property
    def measured(self) -> bool:
        return self.windows > 0

    @property
    def clears_break_even(self) -> bool:
        """Positive measured expectancy, costs included.

        Written as a comparison of rates rather than `expected_r > 0` so that
        an operator margin added to `required_pct` actually bites. With no
        margin the two forms are the same statement: solving `expected_r > 0`
        for the hit rate is where `required_pct` comes from.
        """
        return self.forward_pct >= self.required_pct

    def describe(self) -> str:
        expired = self.windows - self.resolved_windows
        return (
            f"this market reached the target before the {self.reward_risk:.2f}RR stop in "
            f"{self.forward_pct:.1f}% of the {self.resolved_windows} windows that resolved "
            f"({expired} of {self.windows} expired with neither touched); at a "
            f"{self.cost_r:.0%}-of-risk round trip it needs {self.required_pct:.1f}% to "
            f"break even, so one trade is worth {self.expected_r:+.2f}R"
        )


@dataclass(frozen=True, slots=True)
class TargetOdds:
    """One distance, priced — and whether the target is the exit or decoration.

    `expected_r` prices all three outcomes. `resolved_reach` asks a narrower
    question: OF THE WINDOWS THAT ENDED ONE WAY OR THE OTHER, how often did
    this market go our way? That is the population the classic break-even rate
    `(1 + cost) / (1 + RR)` was derived for, and applying it to the whole
    sample — expired windows included — is the error that made every target
    look unpayable.

    Both tests are needed, and they catch different things. A distance the
    market reaches once a month can still show a positive `expected_r` on
    drift alone: every window expires a little in front, nothing is ever
    stopped, and the target is never touched. That is not a plan, because the
    system exits at the target or the stop and neither arrives.
    `resolved_windows` at zero says exactly that, in a number.
    """

    expected_r: float
    reach: float
    resolved_reach: float
    resolved_windows: int

    def target_is_the_exit(self, *, reward_risk: float, cost_r: float) -> bool:
        """Does the target carry the trade, on the windows that resolved?"""
        if self.resolved_windows <= 0 or reward_risk <= 0:
            return False
        return self.resolved_reach >= (1.0 + cost_r) / (1.0 + reward_risk)


@dataclass(frozen=True, slots=True)
class FirstTouchOutcomes:
    """What became of each window, in three kinds and not two.

    THE BUG THIS EXISTS TO KILL WAS MINE AND IT RAN LIVE. A favourable-run
    array alone cannot tell a window that was stopped out from one where price
    drifted sideways until the horizon expired, so a caller holding only that
    array has to treat both as a full stop-out.

    Measured on a synthetic walk carrying a real edge, at a 1R target 46% of
    windows expire unresolved. Charging each of them -1R subtracts about half
    an R of pure fiction from every evaluation: the same market reads -0.15R,
    and is refused, where the truth is +0.31R. Four of five live setups died at
    that gate in a measured window, and the arithmetic is why.

    So the three outcomes stay apart, and the third is MEASURED rather than
    assumed. `settle_r` is where price actually stood at the end of an
    unresolved window, in units of risk, signed in the trade's favour. Calling
    it flat would be a guess with the bars already in hand.
    """

    #: Best favourable excursion before the stop would have closed it, in price.
    run: np.ndarray
    #: True where the stop would have been taken inside the horizon.
    stopped: np.ndarray
    #: Signed R at the last bar, for windows that resolved neither way. Zero
    #: elsewhere, so it is only meaningful read together with the other two.
    settle_r: np.ndarray

    @property
    def windows(self) -> int:
        return int(self.run.size)

    def expectancy_r(self, *, distance: float, risk: float, cost_r: float) -> TargetOdds:
        """What one trade at this distance is worth, and how it got there.

        Every window is scored as what it was. Reaching the target pays the
        target less the round trip; being stopped costs the stop plus the same
        round trip; expiring costs the round trip plus wherever price actually
        ended — small more often than not, sometimes favourable, and never the
        whole stop the two-outcome form charged it.
        """
        if risk <= 0 or self.run.size == 0 or distance <= 0:
            return TargetOdds(0.0, 0.0, 0.0, 0)
        won = self.run >= distance
        lost = self.stopped & ~won
        payoff = np.where(
            won,
            distance / risk - cost_r,
            np.where(lost, -(1.0 + cost_r), self.settle_r - cost_r),
        )
        resolved = int(won.sum() + lost.sum())
        return TargetOdds(
            expected_r=float(payoff.mean()),
            reach=float(won.mean()),
            resolved_reach=float(won.sum() / resolved) if resolved else 0.0,
            resolved_windows=resolved,
        )


def first_touch_outcomes(
    frame: pd.DataFrame,
    *,
    risk: float,
    bars_ahead: int,
    long: bool,
    closes: np.ndarray | None = None,
) -> FirstTouchOutcomes | None:
    """Walk each window once and record which of the three ways it ended.

    Finds the first bar whose adverse extreme would have taken a stop `risk`
    away from the opening close, measures the favourable extreme only up to
    that bar, and — when the stop was never touched — records where price
    stood at the horizon.

    `closes` may be supplied by a caller that already holds the array; it is
    read from the frame otherwise.

    None when the frame lacks the columns or the history to do it, so a caller
    keeps its previous behaviour rather than inventing a number.
    """
    if frame is None or frame.empty or risk <= 0 or bars_ahead <= 0:
        return None
    if not {"high", "low"}.issubset(frame.columns):
        return None
    if closes is None:
        if "close" not in frame.columns:
            return None
        closes = frame["close"].to_numpy(dtype=float)
    windows = len(closes) - bars_ahead
    if windows <= 0:
        return None
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    sign = 1 if long else -1
    favourable = highs if long else lows
    adverse = lows if long else highs
    reached = np.zeros(windows, dtype=float)
    stopped = np.zeros(windows, dtype=bool)
    settle = np.zeros(windows, dtype=float)
    for start in range(windows):
        begin, end = start + 1, start + 1 + bars_ahead
        opened = closes[start]
        stop_level = opened - risk * sign
        adverse_slice = adverse[begin:end]
        breached = adverse_slice <= stop_level if long else adverse_slice >= stop_level
        was_stopped = bool(breached.any())
        stopped[start] = was_stopped
        hit = int(np.argmax(breached)) if was_stopped else len(adverse_slice)
        alive = favourable[begin : begin + hit]
        if alive.size:
            best = alive.max() if long else alive.min()
            reached[start] = (best - opened) * sign
        if not was_stopped:
            settle[start] = (closes[end - 1] - opened) * sign / risk
    return FirstTouchOutcomes(reached, stopped, settle)


def first_touch_runs(
    frame: pd.DataFrame,
    *,
    risk: float,
    bars_ahead: int,
    long: bool,
    closes: np.ndarray | None = None,
) -> np.ndarray | None:
    """Per window: how far price ran our way BEFORE the stop would have hit.

    The favourable-excursion half of `first_touch_outcomes`, kept because that
    is all several callers need. Anything deciding whether a plan PAYS wants
    the outcomes instead: this array cannot tell a stop-out from a window that
    merely ran out of time, and treating those two alike is what refused a
    market worth +0.31R as though it were worth -0.15R.
    """
    outcomes = first_touch_outcomes(
        frame, risk=risk, bars_ahead=bars_ahead, long=long, closes=closes
    )
    return None if outcomes is None else outcomes.run


def measure_first_touch(
    frame: pd.DataFrame,
    *,
    distance: float,
    risk: float,
    bars_ahead: int,
    long: bool,
    cost_r: float = 0.0,
) -> SurvivalVerdict | None:
    """Expectancy of this exact plan, measured against its own stop.

    `distance` is where the target actually sits and `risk` is the stop the
    order will actually carry — the widened one, when it was widened. The
    reward-to-risk is derived from those two rather than passed in, so the
    number tested is the number the broker will receive.

    None when the history cannot answer, never a refusal on ignorance.
    """
    outcomes = first_touch_outcomes(frame, risk=risk, bars_ahead=bars_ahead, long=long)
    if outcomes is None or outcomes.windows == 0 or distance <= 0:
        return None
    reward_risk = distance / risk
    odds = outcomes.expectancy_r(distance=distance, risk=risk, cost_r=cost_r)
    # THE SAME TWO-OUTCOME ERROR LIVED HERE, and this is the gate that kills
    # setups after the engine has approved them: four of five in a measured
    # live window died on TARGET_RARELY_REACHED.
    #
    # `(1 + cost) / (1 + RR)` is the hit rate that solves `edge > 0` for a
    # trade with exactly two endings. Testing it against a share measured over
    # ALL windows — most of which end with neither the target nor the stop
    # touched — compares a rate to a requirement built for a different
    # denominator. At a 1R target on a market with a real edge that reads 50%
    # against 58% needed, and refuses a plan worth +0.42R a trade.
    #
    # So the requirement is unchanged and the rate is measured on the windows
    # it was derived for: the ones that resolved. `expected_r` prices all
    # three endings, the expired ones at where price actually stood.
    required = 100.0 * (1.0 + cost_r) / (1.0 + reward_risk)
    return SurvivalVerdict(
        windows=int(outcomes.windows),
        resolved_windows=odds.resolved_windows,
        forward_pct=100.0 * odds.resolved_reach,
        reach_pct=100.0 * odds.reach,
        required_pct=min(100.0, required),
        expected_r=odds.expected_r,
        reward_risk=reward_risk,
        cost_r=cost_r,
    )


def break_even_rate(reward_risk: float) -> float:
    """The hit rate a plan needs to return zero, ignoring costs.

    Percent, so it compares directly against the reach measurement. A 2:1 plan
    needs one win in three. Costs make the real figure worse, which is why this
    is a floor and not a target.
    """
    if reward_risk <= 0:
        return 100.0
    return 100.0 / (1.0 + reward_risk)


def measure(
    frame: pd.DataFrame,
    *,
    distance: float,
    bars_ahead: int,
    long: bool,
    reward_risk: float,
) -> ReachVerdict:
    """How often price covered `distance` within `bars_ahead`, both ways.

    Vectorised, unlike the copy in the review payload builder, because this one
    runs on every candidate in the catalogue rather than on the handful that
    reach a paid review. A Python loop over four hundred windows for each of
    two hundred symbols is a third of a second per cycle on one vCPU, spent
    every cycle, to compute something that changes once per bar.
    """
    if frame is None or frame.empty or distance <= 0 or bars_ahead <= 0:
        return ReachVerdict(0, 0.0, 0.0, break_even_rate(reward_risk))

    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    windows = len(closes) - bars_ahead
    if windows <= 0:
        return ReachVerdict(0, 0.0, 0.0, break_even_rate(reward_risk))

    # A rolling max of the highs and min of the lows over the window that
    # STARTS one bar after each close. `sliding_window_view` is a view rather
    # than a copy, so this costs one pass instead of four hundred slices.
    ahead_high = np.lib.stride_tricks.sliding_window_view(highs[1:], bars_ahead)[:windows]
    ahead_low = np.lib.stride_tricks.sliding_window_view(lows[1:], bars_ahead)[:windows]
    origin = closes[:windows]
    up = float(np.count_nonzero(ahead_high.max(axis=1) - origin >= distance))
    down = float(np.count_nonzero(origin - ahead_low.min(axis=1) >= distance))

    up_pct = 100.0 * up / windows
    down_pct = 100.0 * down / windows
    return ReachVerdict(
        windows=windows,
        forward_pct=up_pct if long else down_pct,
        opposite_pct=down_pct if long else up_pct,
        required_pct=break_even_rate(reward_risk),
    )
