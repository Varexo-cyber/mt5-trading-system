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
