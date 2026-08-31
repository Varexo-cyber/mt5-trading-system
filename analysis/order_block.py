"""SECTION THREE. The candle that got run over, revisited.

FOUND BY THE SECOND SEARCH, after the first one produced nothing but variants
of section two. `retest_slow` looked like a second strategy and shared 47.5% of
its trades with `impulse_retest` -- the same section wearing a hat. So round
three was constrained to mechanisms that can fire when the retest cannot:
fractal swings, Fibonacci proportions, week boundaries, session VWAP,
cross-market divergence, and this.

Only this survived.

    FX M30, 31,376 trades excluding every bar section two also takes
    hit 62.1%   gross +0.242R   net of control and spread  +0.164R
    train +0.168  /  holdout +0.161

    every year positive, 2012-2022, from +0.132R to +0.289R
    114 of 114 months positive, worst month +0.9R
    72.1% of days positive, worst drawdown -22.5R

WHAT IT LOOKS FOR:

    1. one bar whose BODY spans at least 1.5 ATR      the impulse
    2. the last opposite-coloured candle before it    the block
    3. a limit resting 0.25 ATR inside that block's edge
    4. stop 1.0 ATR from the fill, target 1R

WHY THIS IS NOT SECTION TWO. Section two needs a twenty-bar channel to break;
this needs no level at all, only one violent candle and the candle before it.
Of its entries, 91.4% fall on bars where section two does not fire, and the
numbers above are measured with the other 8.6% REMOVED -- so nothing here is
borrowed from the strategy it sits next to.

THE ECONOMIC CLAIM, and it is different from section two's. Section two stands
in the queue that survives a break. This stands where someone was working an
order that got run over: the last candle going the other way before a 1.5 ATR
move is, by construction, the last price at which the losing side was still
being filled. If they had size left, they did not get it done, and they are
still there when price returns. It is an AREA, not a line -- which is why the
tolerance is measured into the body rather than around a price.

WHY THE IMPULSE MUST BE 1.5 ATR. Measured; the threshold is most of the edge:

    >= 1.0 ATR   122,927 trades   net +0.083R
    >= 1.5 ATR    37,049 trades   net +0.172R
    >= 2.0 ATR    12,441 trades   net +0.185R

A one-ATR candle runs over nobody. At 2.0 the edge is barely better and the
sample is a third the size, so 1.5 is where it settles.

WHY 1:1 AGAIN, and this was not assumed from section two -- it was measured
here separately: 0.75R nets +0.147, 1.0R +0.172, 1.5R +0.099. Both strategies
land on a near target for the same reason: the thing they lean on is a local
supply of orders, and it is used up quickly.

WHERE IT DOES NOT WORK. Indices are thin (+0.029R at M30, and NEGATIVE at M15),
gold thin (+0.018R). Both are cost, not signal -- the gross edge is there in
all three. FX is where it can be paid for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import OrderBlockConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(spans.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class OrderBlock:
    """The last candle the other way before an impulse, offered back."""

    name = "order_block"

    def __init__(self, config: OrderBlockConfig | None = None, name: str | None = None) -> None:
        """`name` gives a second instance on a second clock its own identity.

        Every registry that decides what a module may do is keyed by its name:
        `weights`, `live_enabled_modules`, `section_breakers`, and the
        `module_scores` rows the breaker and the scorecard read back. Two
        instances sharing one name would share a weight, share a breaker, and
        write into each other's history -- so the M1 copy could not be
        switched off without switching off M30, and a losing streak on one
        clock would trip the other.

        Defaults to the class attribute, so single-clock construction is
        unchanged.
        """
        self.config = config or OrderBlockConfig()
        self.name = name or type(self).name

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "order block disabled")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.atr_period + config.lookback_bars + config.block_search_bars + 2
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"{config.timeframe} needs {needed} closed bars")
        frame = series.df
        unit = _atr(frame, config.atr_period)
        if unit <= 0.0:
            return Signal.neutral(self.name, "ATR unavailable")

        stop_atr = config.stop_atr_by_symbol.get(ctx.symbol, config.stop_atr)
        price = ctx.tick.mid if ctx.tick is not None else float(frame["close"].iloc[-1])
        found = self._live_zone(frame, unit, price, stop_atr)
        if found is None:
            return Signal.neutral(
                self.name,
                f"no unspent block behind a {config.minimum_impulse_atr:.1f} ATR impulse "
                f"in the last {config.lookback_bars} {config.timeframe} bars",
            )
        direction, edge, impulse, age = found

        entry = edge + direction * config.zone_tolerance_atr * unit
        # How far price still is from the tradeable edge of the zone, in ATR.
        # Negative means it is already inside.
        away = direction * (price - entry) / unit
        if away > 0.0:
            return Signal.neutral(
                self.name,
                f"{away:.2f} ATR above the block behind the impulse; waiting for the zone",
            )

        conviction = min(1.0, (impulse - config.minimum_impulse_atr) / config.impulse_span_atr)
        score = config.base_score + conviction * config.quality_score_bonus
        confidence = min(
            config.maximum_confidence,
            config.base_confidence + conviction * config.quality_confidence_bonus,
        )
        return Signal(
            module=self.name,
            score=min(score, 100.0) * direction,
            confidence=confidence,
            reasoning=(
                f"price returned to the candle absorbed by a {impulse:.2f} ATR impulse "
                f"{age} bars ago ({config.timeframe}); stop {stop_atr:.2f} ATR from entry"
            ),
            key_levels=(edge,),
            invalidation_price=entry - direction * stop_atr * unit,
            details={
                "timeframe": config.timeframe,
                "setup": self.name,
                "block_edge": round(edge, 6),
                "impulse_atr": round(impulse, 3),
                "impulse_age_bars": age,
            },
        )

    def _live_zone(
        self, frame: pd.DataFrame, unit: float, price: float, stop_atr: float
    ) -> tuple[int, float, float, int] | None:
        """The unspent block nearest to price, or None.

        Nearest rather than newest, for the same reason section two takes the
        nearest level: a run of impulses leaves a ladder of zones, and the one
        being revisited is the one price is at, not the last one printed.

        A zone is dropped once something after it CLOSED through where the stop
        would sit. Past that it is not a zone anyone is defending, and offering
        it again is how a level trade becomes a knife catch.
        """
        config = self.config
        o = frame["open"].to_numpy()
        c = frame["close"].to_numpy()
        last = len(c) - 1
        oldest = max(config.block_search_bars + 1, last - config.lookback_bars)
        best: tuple[float, int, float, float, int] | None = None
        for i in range(last, oldest - 1, -1):
            if not np.isfinite(o[i]) or not np.isfinite(c[i]):
                continue
            impulse = (c[i] - o[i]) / unit
            if abs(impulse) < config.minimum_impulse_atr:
                continue
            direction = 1 if impulse > 0 else -1
            # The last candle going the other way, immediately before it.
            edge = None
            for k in range(i - 1, max(i - config.block_search_bars - 1, 0), -1):
                if (c[k] < o[k]) == (direction > 0):
                    body = (min(o[k], c[k]), max(o[k], c[k]))
                    edge = body[1] if direction > 0 else body[0]
                    break
            if edge is None:
                continue
            entry = edge + direction * config.zone_tolerance_atr * unit
            beyond = entry - direction * stop_atr * unit
            after = c[i + 1 :]
            if len(after) and (
                (direction > 0 and float(after.min()) < beyond)
                or (direction < 0 and float(after.max()) > beyond)
            ):
                continue
            away = direction * (price - entry) / unit
            if away < -stop_atr:
                continue  # price is already through the zone and past the stop
            if best is None or away < best[0]:
                best = (away, direction, edge, abs(impulse), last - i)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]
