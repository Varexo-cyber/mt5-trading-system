"""SECTIONS ELEVEN, TWELVE AND THIRTEEN: one mechanism on XAUJPY, three clocks.

WHAT REPLACED WHAT. The old section eleven was a fitted model per metal on the
four gold crosses. Its holdout came back negative in four markets out of four
(-193 / -190 / -149 / -116 R) and it is gone. This is a different thing on a
different question: one instrument, XAUJPY, and a mechanism CHOSEN BY SEARCH
rather than fitted -- a rule with a name you can say out loud, not a linear
head over forty-eight hidden units.

THE THREE SECTIONS ARE ONE CLASS. Section eleven runs M1, twelve runs M5,
thirteen runs M15. A module is one instance reading one timeframe -- every
registry that matters (`weights`, `live_enabled_modules`, `section_breakers`)
is keyed by module name -- so three clocks means three modules, and a single
class means the three cannot drift apart in anything but their config.

WHAT IS DELIBERATELY MISSING, and it is the same shape as the old section
eleven's absent model files. There is no mechanism named in this file. A
section whose `mechanism` is empty is SILENT, not neutral: it emits no read at
all rather than a zero, so a section nobody has searched for cannot be mistaken
for a section that looked and found nothing. `scripts/search_xaujpy.py` is what
fills the name in, and it is built to come back empty.

THE MECHANISMS COME FROM `analysis/mechanisms.py`, which is also what the
search measures. That is the whole reason that module exists: the candidate
that was searched IS the candidate that trades, rather than a re-implementation
of it that nothing compares against.

HOURS ARE PART OF THE MECHANISM HERE, not a filter bolted on afterwards. Every
measurement this account has produced on gold and its crosses has come back
saying the same thing -- 07:00-13:00 UTC cost section ten -0.101 R per trade on
these crosses against +0.064 everywhere else -- so an empty `allowed_hours`
means every hour and is a decision, not a default nobody looked at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.mechanisms import FAMILIES, WARMUP, _atr
from core.types import Direction, MarketContext, Signal, Timeframe

#: Every mechanism any of the three sections may name. It is `FAMILIES["all"]`
#: rather than a second list, so a candidate the search can try is by
#: construction a candidate a section can run -- and a typo in the config names
#: a mechanism that does not exist and is REFUSED at load rather than silently
#: never firing.
MECHANISMS = FAMILIES["all"]


def mechanism_names() -> tuple[str, ...]:
    """Sorted, for the config validator and for error messages."""

    return tuple(sorted(MECHANISMS))


class SectionXauJpy:
    """One searched mechanism, on one clock, on one instrument.

    Silent unless a mechanism has been named for it. The name arrives from a
    search that had to clear a Bonferroni bar, a rate-matched control and an
    untouched holdout to hand it over.
    """

    def __init__(self, name: str, config, broker_symbol: str | None = None) -> None:
        self.name = name
        self.config = config
        # THE NAME THIS BROKER USES, resolved once by the caller that has the
        # config to resolve it with.
        #
        # `ctx.symbol` carries the BROKER's name -- the core universe prints
        # `USDJPY.i`, not `USDJPY` -- and this compared it against the plain
        # `XAUJPY` in its own config. Eightcap happens to list a plain XAUJPY
        # as well, so it worked; on a broker that only lists `XAUJPY.i` this
        # section would have been silent forever with nothing anywhere saying
        # why. The same suffix killed the legs research three symbols in.
        self.broker_symbol = broker_symbol or config.symbol

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        quiet = Signal.neutral(self.name, "no read")
        if not cfg.enabled or not cfg.mechanism:
            # NOT NEUTRAL BECAUSE IT LOOKED. Nothing has been searched for this
            # clock yet, and an absent mechanism reading as a zero is how a
            # section that was never fitted comes to look like a section that
            # found nothing.
            return quiet
        if ctx.symbol not in (self.broker_symbol, cfg.symbol):
            return quiet

        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        if series is None or len(series.df) < WARMUP + 2:
            return quiet
        frame = series.df
        hour = int(frame.index[-1].hour)
        if cfg.allowed_hours and hour not in cfg.allowed_hours:
            return quiet
        if hour in cfg.blocked_hours:
            return quiet

        signals = self.readings(frame)
        direction_flag = int(signals[-1])
        if direction_flag == 0:
            return quiet

        unit = float(_atr(frame)[-1])
        if not np.isfinite(unit) or unit <= 0.0:
            return quiet

        direction = Direction.LONG if direction_flag > 0 else Direction.SHORT
        if cfg.long_only and direction is Direction.SHORT:
            return quiet
        close = float(frame["close"].iloc[-1])
        risk = cfg.stop_atr * unit
        stop = close - risk if direction is Direction.LONG else close + risk

        # `cfg.score`, not a number written here. See the field's comment: a
        # hardcoded 60 against a 35.0 threshold at 0.55 confidence is 33.0, and
        # 33.0 never clears anything.
        score = cfg.score if direction is Direction.LONG else -cfg.score
        return Signal(
            module=self.name,
            score=score,
            confidence=cfg.confidence,
            reasoning=(
                f"{self.name}: {cfg.mechanism} fired {direction.name} on "
                f"{cfg.symbol} {cfg.timeframe} at {hour:02d}:00 UTC, stop "
                f"{cfg.stop_atr:.2f} ATR, target {cfg.target_ratio:.2f} R"
            ),
            invalidation_price=stop,
        )

    def readings(self, frame: pd.DataFrame) -> np.ndarray:
        """The mechanism's per-bar direction, straight from the shared registry.

        A method rather than an inline call so a test can drive the exact array
        the live path reads, instead of a copy of it built beside it.
        """

        return MECHANISMS[self.config.mechanism](frame)
