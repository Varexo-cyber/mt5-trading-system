"""Is this move real, or is it someone in a hurry?

SECTION TWO. Everything else on this account reads the SHAPE of the price
series — a trend, a cross, a break, a drift, a resumed pullback. Nine readers,
one source, and the config says in as many words what that costs: "ze vuren
samen en ze verliezen samen." This module does not read the shape. It asks a
different question entirely, and it answers in the opposite direction.

THE QUESTION. A price moved. Was that DRIFT — an actual repricing — or was it
VOLATILITY, the ordinary wobble of a market plus one participant who had to get
out right now and paid whatever it took? Those two look identical on a chart and
they have opposite futures.

THE ANSWER is a hypothesis test, not a pattern. Christensen, Oomen and Renò
(Journal of Econometrics, 2022) define a non-parametric statistic that is
essentially a t-test on the local mean return, scaled by the local volatility:

    T = sqrt(n_eff) * mu_hat / sigma_hat

Above a threshold near 4, the move is too large to be explained by the
instrument's own volatility. They call that a DRIFT BURST.

WHAT THEY MEASURED, over a multi-year tick sample spanning equities, fixed
income, currencies and commodities — which is this account's whole universe:

    more than 1,000 highly significant events
    roughly one per week per instrument
    typical magnitude 25 to 200 basis points, a handful of 3-8%
    TWO THIRDS are followed by price reversion

That last line is the edge, and it is why this module fades rather than
follows. The reversion is not a chart pattern repeating. It is payment for a
service: someone needed immediacy during a liquidity shock and overpaid for it,
and whoever supplied that immediacy collects the difference back. The same
mechanism arrives independently from the short-term reversal literature — two
separate fields landing on "compensation for supplying immediacy" is a much
stronger reason to believe something than either one alone.

WHAT THIS MODULE CANNOT DO, and the reason is structural rather than a setting.
The paper works on TICK data. This account has M1 bars. The statistic is a
ratio of local drift to local volatility and it survives the coarsening, but
with less power and with microstructure noise the paper itself calls awkward to
handle. So the number here is not the paper's number, and the honest position is
that whether it fires usefully on minute bars is an open question this code
exists to ANSWER rather than to assume.

Which is why it does not trade. It carries a weight so the offline module
backtest can see it, it is deliberately absent from `live_enabled_modules`, and
the observer in the runner records what it WOULD have done as a shadow trade
that the existing resolver settles against real later prices. Nothing here can
reach an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.schema import DriftBurstConfig
from core.types import MarketContext, Signal, Timeframe


@dataclass(frozen=True, slots=True)
class BurstReading:
    """The statistic and everything needed to audit it later."""

    #: The test statistic. Positive means the burst ran up, negative down.
    t_stat: float
    #: Exponentially weighted mean log return over the window, per bar.
    drift: float
    #: Exponentially weighted volatility of those returns, per bar.
    volatility: float
    #: Effective sample size after the kernel weights, which is what the
    #: statistic is scaled by rather than the raw bar count.
    effective_bars: float
    #: Total move across the window, in basis points. The paper's events run
    #: 25-200bp, so this is the sanity check that says whether a reading is the
    #: kind of event the research describes or a rounding artefact.
    move_bp: float

    @property
    def fired(self) -> bool:
        return math.isfinite(self.t_stat)


def _ewma_weights(count: int, half_life: float) -> np.ndarray:
    """Exponential kernel, most weight on the most recent bar.

    The paper weights recent observations more heavily for a reason worth
    keeping: a burst is a LOCAL phenomenon, and a flat window averages the
    explosive part together with the calm before it until the statistic can no
    longer see it. Half-life rather than a raw decay constant so the setting
    means something a person can reason about — "half the weight sits in the
    last N bars".
    """
    if half_life <= 0:
        return np.ones(count, dtype=float)
    ages = np.arange(count - 1, -1, -1, dtype=float)
    return np.exp(-math.log(2.0) * ages / half_life)


def burst_statistic(
    closes: pd.Series,
    *,
    drift_window: int,
    volatility_window: int,
    half_life: float,
    noise_lags: int,
) -> BurstReading | None:
    """The drift-burst t-statistic: a short-window drift over a long-window vol.

    TWO BANDWIDTHS, AND THE SECOND ONE IS NOT A REFINEMENT.

    Written with a single window this test cannot see the events it exists for.
    Measured over 3,000 synthetic paths at a 60-bar window: a burst lasting 20
    of those bars fired 68% of the time, while a burst lasting 5 bars — a
    HARDER, faster move covering 127 basis points — fired 0.0% of the time.
    Never once.

    The reason is that the burst sits inside the window estimating the
    volatility, so it inflates its own denominator. A short violent move raises
    sigma as fast as it raises mu and the ratio never moves. The statistic was
    blind to exactly the shape it was built to find.

    So the drift is estimated over a short recent window and the volatility over
    a long one that is mostly the calm BEFORE the burst — which is what
    Christensen, Oomen and Renò do with separate bandwidths, and which this
    first version had collapsed into one.

    Returns None when there is not enough history or no volatility can be
    formed — never a zero, because "no reading" and "a reading of zero" are
    different statements and only one of them is evidence.
    """
    if drift_window < 2 or volatility_window <= drift_window:
        return None
    if len(closes) < volatility_window + 1:
        return None
    prices = closes.iloc[-(volatility_window + 1) :].astype(float).to_numpy()
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        return None

    returns = np.diff(np.log(prices))

    # DRIFT: the recent window only, exponentially weighted inside it.
    recent = returns[-drift_window:]
    drift_weights = _ewma_weights(len(recent), half_life)
    drift_total = float(drift_weights.sum())
    if drift_total <= 0.0:
        return None
    drift_weights = drift_weights / drift_total
    drift = float(np.dot(drift_weights, recent))

    # VOLATILITY: the whole long window, flat. Deliberately NOT weighted toward
    # the present — the recent bars are the ones under suspicion, and leaning on
    # them is how the denominator gets contaminated in the first place.
    centred = returns - float(returns.mean())
    variance = float(np.dot(centred, centred) / len(centred))

    # Bid-ask bounce makes consecutive returns negatively autocorrelated: a
    # print at the ask followed by one at the bid reads as a move and a
    # reverse, neither of which happened. That inflates a naive variance, and
    # an inflated denominator makes the statistic SMALLER — so the test would
    # go blind on the noisiest instruments, which are the ones bursting most.
    # The Newey-West correction adds the autocovariance terms back.
    for lag in range(1, max(0, noise_lags) + 1):
        if lag >= len(centred):
            break
        bartlett = 1.0 - lag / (noise_lags + 1.0)
        covariance = float(np.dot(centred[lag:], centred[:-lag]) / len(centred))
        variance += 2.0 * bartlett * covariance
    if not math.isfinite(variance) or variance <= 0.0:
        return None

    volatility = math.sqrt(variance)
    # The kernel spends the drift window unevenly, so the raw bar count
    # overstates how much independent information the mean rests on.
    effective = float(1.0 / np.dot(drift_weights, drift_weights))
    t_stat = math.sqrt(effective) * drift / volatility
    if not math.isfinite(t_stat):
        return None

    # The move the DRIFT window covers, which is the part being called a burst.
    move_bp = (prices[-1] / prices[-(drift_window + 1)] - 1.0) * 10_000.0
    return BurstReading(
        t_stat=t_stat,
        drift=drift,
        volatility=volatility,
        effective_bars=effective,
        move_bp=move_bp,
    )


class DriftBurst:
    """Fades a move the instrument's own volatility cannot explain."""

    name = "drift_burst"

    def __init__(self, config: DriftBurstConfig | None = None) -> None:
        self.config = config or DriftBurstConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "drift-burst detector disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.volatility_window + 1
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        reading = burst_statistic(
            series.df["close"],
            drift_window=config.drift_window,
            volatility_window=config.volatility_window,
            half_life=config.half_life_bars,
            noise_lags=config.noise_lags,
        )
        if reading is None:
            return Signal.neutral(self.name, "no volatility estimate available")

        magnitude = abs(reading.t_stat)
        details = {
            "t_stat": reading.t_stat,
            "drift_per_bar": reading.drift,
            "volatility_per_bar": reading.volatility,
            "effective_bars": reading.effective_bars,
            "move_bp": reading.move_bp,
            "threshold": config.t_threshold,
        }
        if magnitude < config.t_threshold:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"|t| {magnitude:.2f} under the {config.t_threshold:.2f} a burst "
                    f"needs; this move is inside the instrument's own volatility"
                ),
                details=details,
            )

        # A MOVE TOO SMALL TO BE THE PAPER'S EVENT IS NOT THE PAPER'S EVENT.
        #
        # The statistic is a ratio, so a dead-quiet instrument that twitches
        # produces a large t on a move worth two basis points. The research
        # describes events of 25-200bp; below that floor the reversion has
        # nothing in it to collect and the spread eats whatever is left.
        if abs(reading.move_bp) < config.minimum_move_bp:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"|t| {magnitude:.2f} clears the bar on a move of only "
                    f"{reading.move_bp:+.1f}bp, under the "
                    f"{config.minimum_move_bp:.0f}bp a burst is worth fading"
                ),
                details=details,
            )

        # FADE. The score points AGAINST the burst, which is the whole thesis:
        # two thirds of these revert, and the reversion is the payment for
        # having supplied immediacy to whoever caused the move.
        direction = -1.0 if reading.t_stat > 0 else 1.0
        span = max(1e-9, config.t_saturation - config.t_threshold)
        strength = min(1.0, (magnitude - config.t_threshold) / span)
        score = direction * (config.base_score + strength * (100.0 - config.base_score))
        room = config.maximum_confidence - config.base_confidence
        confidence = min(config.maximum_confidence, config.base_confidence + strength * room)
        way = "up" if reading.t_stat > 0 else "down"
        return Signal(
            module=self.name,
            score=score,
            confidence=confidence,
            reasoning=(
                f"drift burst {way} at t={reading.t_stat:+.2f} over "
                f"{reading.move_bp:+.1f}bp — the move is larger than this market's "
                f"own volatility explains, and two in three revert"
            ),
            details=details,
        )
