"""This instrument's own record on this weekday, not folklore.

"Sell in May", "Monday reversal", "turn of the month" are stories, and a system
that acts on stories has no way to find out it was wrong. What IS measurable is
whether THIS symbol, over the daily history the broker actually carries, has a
return on this weekday that is distinguishable from zero given its own noise.

So this is written as a t-test and not as a table of beliefs. Mean daily return
on this weekday, divided by its standard error. Below the threshold — which is
where most instruments on most days will sit — it returns neutral and says how
close it came, so the operator can see the module working rather than only
see it silent.

WHY IT IS THE WEAKEST MODULE HERE, on purpose. `score` is 25 against 50-55 for
the others and the confidence ceiling is 0.55, which is below the 0.65 a lone
module needs to produce a setup at all. That is the design: a weekday lean is
background, never a reason to take a trade by itself, and the lone-module floor
already refuses it in that role. What it can do is tip a setup that two other
readers already like, and that is all it should do.

WHAT IT CANNOT ESCAPE. Testing five weekdays and reporting the significant one
is five chances to find a two-sigma result where none exists; at t=2 per test,
one weekday in four instruments will look significant by luck alone. The
threshold is configurable for exactly that reason, and the module backtest is
what decides whether the whole family survives — not this docstring.

Not enabled live. It goes on the allowlist when the module backtest says it
earns its place, and not before.
"""

from __future__ import annotations

import numpy as np

from config.schema import SeasonalityConfig
from core.types import MarketContext, Signal, Timeframe

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class Seasonality:
    """A weekday bias this instrument's own daily history can support."""

    name = "seasonality"

    def __init__(self, config: SeasonalityConfig | None = None) -> None:
        self.config = config or SeasonalityConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "seasonality disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        if series is None or len(series.df) < config.minimum_samples * 5:
            return Signal.neutral(
                self.name, f"needs {config.minimum_samples * 5} closed {timeframe.value} bars"
            )

        frame = series.df.iloc[-config.lookback_days :]
        # Bar-to-bar returns, in percent, so instruments with different price
        # levels are on the same scale and the standard error means the same
        # thing on gold as on EURUSD.
        returns = frame["close"].pct_change().dropna()
        if returns.empty:
            return Signal.neutral(self.name, "no usable daily returns")

        # The weekday being predicted is the one the NEXT bar will belong to,
        # which is the day the trade would be held through. Using the last
        # closed bar's weekday would measure yesterday and act on it today.
        weekday = ctx.now.weekday()
        sample = returns[returns.index.weekday == weekday]
        if len(sample) < config.minimum_samples:
            return Signal.neutral(
                self.name,
                f"only {len(sample)} {_WEEKDAYS[weekday]}s in this instrument's history, "
                f"under the {config.minimum_samples} a weekday claim needs",
            )

        mean = float(sample.mean())
        sd = float(sample.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return Signal.neutral(self.name, "no dispersion in the weekday sample")
        standard_error = sd / np.sqrt(len(sample))
        t = mean / standard_error
        if abs(t) < config.minimum_t:
            return Signal.neutral(
                self.name,
                f"{_WEEKDAYS[weekday]} averages {mean:+.3%} over {len(sample)} samples, "
                f"t = {t:+.2f} against the {config.minimum_t:.1f} needed — that is this "
                f"instrument's noise, not its calendar",
            )

        direction = 1 if t > 0 else -1
        confidence = min(
            config.maximum_confidence,
            config.base_confidence
            + (abs(t) - config.minimum_t) * config.significance_confidence_scale,
        )
        return Signal(
            module=self.name,
            score=config.score * direction,
            confidence=confidence,
            reasoning=(
                f"{_WEEKDAYS[weekday]} has averaged {mean:+.3%} on this instrument over "
                f"{len(sample)} samples, t = {t:+.2f}"
            ),
            details={
                "timeframe": timeframe.value,
                "weekday": _WEEKDAYS[weekday],
                "samples": len(sample),
                "mean_return_pct": round(mean * 100.0, 4),
                "t_statistic": round(t, 2),
            },
        )
