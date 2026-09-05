"""SECTION ELEVEN: the XAUJPY quote against its own two legs, on M5.

WHAT THIS REPLACED, AND WHY THE NAME MOVED CLOCK. Section eleven has been three
things. First a fitted per-metal model on four gold crosses, whose holdout came
back negative in four markets out of four. Then `streak_reversal` on XAUJPY M1,
which lost -0.18 R a trade over 855 trades once the cost model charged M1 what
M1 actually costs. Both are gone. This is the third, and it is the first one
with a counterparty you can name:

    XAUJPY = XAUUSD x USDJPY

Gold moves, the yen does not, and the cross has to follow. Whoever quotes the
cross does that with a lag. The trade is that the quote catches up, and the
person on the other side of it is the market maker whose cross has not caught
up with its own two books yet. Nobody loses money because four one-minute
candles went up; somebody does lose money quoting a stale cross.

WHAT IT MEASURED, AND IT DID NOT CLEAR ITS BAR. `benen.cmd 720`, 720 days of
Eightcap bars, sixteen cells searched, Bonferroni bar 2.96 sigma:

    M5   gap 0.50 ATR   R:R 1.5   152 trades   +54.80 R   +0.361/trade
                                  +2.93 sigma   holdout +1.47 R   0.3 trades/day

+2.93 against a bar of 2.96. It MISSED, by 0.03, and the bar is not moving --
moving it after seeing the number is the one thing a pre-registered bar exists
to prevent. All twelve surviving cells were positive and this was the best of
them, so what is written here is "the strongest thing the search found", not
"a proven edge". It is built so it can be replayed and watched. That is the
whole claim.

THE HONEST HOLE, AND IT IS IN THE SEARCH RATHER THAN IN THIS FILE. The legs are
read on CLOSED bars of the same clock as the cross. A lag that opens and closes
inside one M5 bar is invisible to the measurement and would be invisible here
too. If the real lag is faster than five minutes, the measured +0.361 is a
fraction of what is there and equally a fraction of what this will capture.
Reading the legs on M1 or on ticks is the next experiment, not a tweak.

WHY THE LEGS ARRIVE THROUGH `ctx.meta`. A module receives one MarketContext and
cannot reach a second instrument -- that is the constraint that makes every
module replayable and unit-testable without a terminal, and it is not being
relaxed here. `basket_divergence` already needed the same thing and the route
already exists: the runner, which does have the connector, puts the other
instruments' bars in `context.meta` before the engine runs. This reads them
from there and refuses when they are not there.

STALE LEGS ARE REFUSED, NOT INTERPOLATED. The runner reads 845 markets one at a
time, so a leg reading is always slightly older than the cross. That is fine
and it is measured: the age travels with the reading and a leg older than
`max_leg_age_seconds` is treated as absent. This matters more here than
anywhere else on the account, because a FROZEN leg is precisely what
manufactures a fake gap -- gold pauses daily while the yen trades on, and
without that guard two thirds of this mechanism's profit came out of the pause.
No data is no trade, in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.legs_gap import (
    DEAD_GAP_ATR,
    NORMAL_BARS,
    gap_reading,
    implied_cross,
    signals_from_gap,
)
from analysis.mechanisms import _atr
from core.types import Direction, MarketContext, Signal, Timeframe

#: Key the runner writes the leg bars under. One name, imported by both sides,
#: because a string typed twice is a string that eventually differs -- the same
#: reason `BASKET_META_KEY` exists.
LEGS_META_KEY = "xaujpy_legs"

#: Bars of each leg needed before the reading means anything. The rolling
#: normal is `NORMAL_BARS` long with `NORMAL_BARS // 2` minimum periods, and
#: the ATR beneath it needs its own warmup on top.
MIN_LEG_BARS = NORMAL_BARS + 32


@dataclass(frozen=True, slots=True)
class LegBars:
    """One leg's closed bars, and how old the newest of them is.

    `age_seconds` is measured against the moment the cross was read, not
    against the wall clock, so a replay and the live runner mean the same thing
    by it.
    """

    symbol: str
    frame: pd.DataFrame
    age_seconds: float


class SectionElevenLegs:
    """XAUJPY against XAUUSD x USDJPY, faded back toward its own legs."""

    def __init__(self, name: str, config, broker_symbol: str | None = None) -> None:
        self.name = name
        self.config = config
        # `ctx.symbol` carries the BROKER's name -- the core universe prints
        # `USDJPY.i`, not `USDJPY`. Comparing it against the plain name in the
        # config is how the previous section would have gone silent forever on
        # a broker that only lists the suffixed symbol, with nothing anywhere
        # saying why. The same suffix killed the legs research three symbols in.
        self.broker_symbol = broker_symbol or config.symbol

    # -- the read -----------------------------------------------------------

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        # NOT `Signal.neutral` WITH A ZERO SCORE BY ACCIDENT: a section that
        # could not read its legs has NOT looked at the market and must not be
        # counted as having looked and found nothing.
        quiet = Signal.neutral(self.name, "no read")
        if not cfg.enabled:
            return quiet
        if ctx.symbol not in (self.broker_symbol, cfg.symbol):
            return quiet

        clock = Timeframe.parse(cfg.timeframe)
        series = ctx.series.get(clock)
        if series is None or len(series.df) < MIN_LEG_BARS:
            return quiet
        cross = series.df
        hour = int(cross.index[-1].hour)
        if cfg.allowed_hours and hour not in cfg.allowed_hours:
            return quiet
        if hour in cfg.blocked_hours:
            return quiet

        legs = self.legs_from(ctx)
        if legs is None:
            return quiet
        base, quote = legs

        reading, unit, live_gap = self.reading(cross, base.frame, quote.frame)
        if reading is None or unit is None:
            return quiet
        if live_gap is not None and live_gap < cfg.minimum_gap_atr:
            # THE CROSS IS BEING COMPUTED FROM ITS LEGS, not quoted against
            # them. There is no lag, no counterparty and nothing to trade, and
            # a search that finds an edge inside a gap that cannot exist has
            # found its own rounding error.
            return quiet

        direction_flag = int(signals_from_gap(np.asarray([reading]), cfg.gap_atr)[0])
        if direction_flag == 0:
            return quiet
        direction = Direction.LONG if direction_flag > 0 else Direction.SHORT

        close = float(cross["close"].iloc[-1])
        risk = cfg.stop_atr * unit
        stop = close - risk if direction is Direction.LONG else close + risk
        score = cfg.score if direction is Direction.LONG else -cfg.score
        return Signal(
            module=self.name,
            score=score,
            confidence=cfg.confidence,
            reasoning=(
                f"{self.name}: {cfg.symbol} sits {reading:+.2f} ATR from its "
                f"{base.symbol} x {quote.symbol} normal at {hour:02d}:00 UTC, "
                f"fade {direction.name} toward the legs, stop {cfg.stop_atr:.2f} "
                f"ATR, target {cfg.target_ratio:.2f} R"
            ),
            invalidation_price=stop,
            details={
                "gap_atr": round(reading, 4),
                "base_leg": base.symbol,
                "quote_leg": quote.symbol,
                "base_age_s": round(base.age_seconds, 1),
                "quote_age_s": round(quote.age_seconds, 1),
            },
        )

    # -- the pieces, each drivable on its own by a test ----------------------

    def legs_from(self, ctx: MarketContext) -> tuple[LegBars, LegBars] | None:
        """Both legs out of `ctx.meta`, or None and therefore no trade.

        Returns None rather than raising, and returns None for a leg that is
        merely OLD as well as for one that is missing. Those are the same fact
        for this mechanism: a leg whose last print is minutes stale is exactly
        the frozen quote that invents a gap.
        """

        raw = (ctx.meta or {}).get(LEGS_META_KEY)
        if not isinstance(raw, dict):
            return None
        cfg = self.config
        found: list[LegBars] = []
        for wanted in (cfg.base_leg, cfg.quote_leg):
            leg = raw.get(wanted)
            if not isinstance(leg, LegBars):
                return None
            if leg.age_seconds > cfg.max_leg_age_seconds:
                return None
            if len(leg.frame) < MIN_LEG_BARS:
                return None
            found.append(leg)
        return found[0], found[1]

    def reading(
        self, cross: pd.DataFrame, base: pd.DataFrame, quote: pd.DataFrame
    ) -> tuple[float | None, float | None, float | None]:
        """`(gap in ATR on the newest shared bar, that bar's ATR, median |gap|)`.

        The arithmetic is `analysis.legs_gap`, which is also what
        `scripts/search_xaujpy_legs.py` measured. One implementation, so the
        candidate that was searched is the candidate that trades.
        """

        implied = implied_cross(base, quote)
        if implied.dropna().empty:
            return None, None, None
        frame, gaps, _raw = gap_reading(cross, implied)
        if len(frame) < MIN_LEG_BARS:
            return None, None, None
        # THE NEWEST SHARED BAR MUST BE THE CROSS'S NEWEST BAR. An inner join
        # against a leg that stopped printing an hour ago silently hands back
        # an hour-old reading that looks perfectly current, and every hour of
        # that lag is a gap the mechanism would have invented.
        if frame.index[-1] != cross.index[-1]:
            return None, None, None
        value = float(gaps[-1])
        unit = float(_atr(frame)[-1])
        if not np.isfinite(value) or not np.isfinite(unit) or unit <= 0.0:
            return None, None, None
        finite = gaps[np.isfinite(gaps)]
        median = float(np.median(np.abs(finite))) if finite.size else None
        return value, unit, median


def leg_bars(symbol: str, frame: pd.DataFrame, now: datetime) -> LegBars:
    """Package one leg's frame with the age of its newest bar.

    A helper rather than a constructor call at each site, because the age is
    the part everything gets wrong: it is measured from the leg's last BAR
    TIMESTAMP to the moment the cross is being read, and every caller that
    computes it separately eventually computes it differently.
    """

    last = frame.index[-1].to_pydatetime() if len(frame) else now
    return LegBars(symbol=symbol, frame=frame, age_seconds=max(0.0, (now - last).total_seconds()))


__all__ = [
    "DEAD_GAP_ATR",
    "LEGS_META_KEY",
    "MIN_LEG_BARS",
    "LegBars",
    "SectionElevenLegs",
    "leg_bars",
]
