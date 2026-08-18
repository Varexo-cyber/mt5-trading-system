"""The quiet session's range, broken when the busy one opens.

Sessions exist in this system only as a way to REFUSE trades: a news blackout,
a runway wind-down, an evening flat, a `MARKET_TOO_QUIET`. The fact that a
market builds a range overnight and resolves it when London arrives has never
been available as a reason to TAKE one — the entire time dimension is a
subtractive filter and never an observation.

WHY IT IS DIFFERENT FROM THE BREAK READERS ALREADY HERE. `impulse_break` and
`m1_micro_breakout` break whatever range happens to be in front of them,
whenever it happens. This one only cares about a specific range built at a
specific time, which means it fires a handful of times a day rather than
continuously, and its population barely overlaps theirs. Two readers that agree
all day are one reader.

THE CLOCK IS THE BROKER'S, and configured rather than inferred. Bars are
stamped in server time, and a server three hours off UTC would silently move
every session by three hours with nothing looking wrong — the ranges would
still be ranges, the breaks would still be breaks, and the module would be
measuring the wrong hours of the day for the life of the account.

Both a floor and a ceiling on the range width. A five-pip overnight range is a
dead market and breaking it means nothing; a range wider than the day's usual
travel has already had its move, and breaking that late is chasing.

Not enabled live. It goes on the allowlist when the module backtest says it
earns its place, and not before.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from config.schema import SessionBreakoutConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> float:
    high, low = frame["high"], frame["low"]
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = float(true_range.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


class SessionBreakout:
    """A break of the range built during the configured quiet window."""

    name = "session_breakout"

    def __init__(self, config: SessionBreakoutConfig | None = None) -> None:
        self.config = config or SessionBreakoutConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "session breakout disabled")
        if config.range_end_hour == config.range_start_hour:
            return Signal.neutral(self.name, "range window has no width")
        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        needed = config.atr_period + 10
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"needs {needed} closed {timeframe.value} bars")

        frame = series.df
        last_time = frame.index[-1]
        # The range belongs to the session that most recently CLOSED. Around
        # midnight that is yesterday's, and using today's date unconditionally
        # would silently compare a break against a window that has not happened.
        session_end = last_time.normalize() + timedelta(hours=config.range_end_hour)
        if last_time < session_end:
            session_end -= timedelta(days=1)
        session_start = session_end.normalize() + timedelta(hours=config.range_start_hour)
        if session_start >= session_end:
            # A window that wraps midnight, e.g. 22:00 to 07:00.
            session_start -= timedelta(days=1)

        if last_time > session_end + timedelta(hours=config.breakout_window_hours):
            return Signal.neutral(
                self.name,
                f"the {config.range_start_hour:02d}:00-{config.range_end_hour:02d}:00 range "
                f"closed more than {config.breakout_window_hours:.0f}h ago; a break now is an "
                f"afternoon move, not that session resolving",
            )

        window = frame[(frame.index >= session_start) & (frame.index < session_end)]
        if len(window) < 3:
            return Signal.neutral(
                self.name,
                f"only {len(window)} {timeframe.value} bars inside the "
                f"{config.range_start_hour:02d}:00-{config.range_end_hour:02d}:00 window",
            )
        top = float(window["high"].max())
        bottom = float(window["low"].min())
        width = top - bottom
        if width <= 0:
            return Signal.neutral(self.name, "the session range has no width")

        atr = _atr(frame, config.atr_period)
        if atr <= 0:
            return Signal.neutral(self.name, "no usable ATR")
        width_atr = width / atr
        if width_atr < config.minimum_range_atr:
            return Signal.neutral(
                self.name,
                f"the session range is {width_atr:.2f} ATR wide, under the "
                f"{config.minimum_range_atr:.2f} floor — breaking a dead range says nothing",
            )
        if width_atr > config.maximum_range_atr:
            return Signal.neutral(
                self.name,
                f"the session range is {width_atr:.2f} ATR wide, over the "
                f"{config.maximum_range_atr:.2f} ceiling — that market has already moved",
            )

        close = float(frame["close"].iloc[-1])
        margin = width * config.breakout_close_share
        if close > top + margin:
            direction, edge, opposite = 1, top, bottom
        elif close < bottom - margin:
            direction, edge, opposite = -1, bottom, top
        else:
            return Signal.neutral(
                self.name,
                f"price is still inside the {width_atr:.2f} ATR session range",
            )

        # Where the break failed: back through the far side of the range it
        # just left. The near edge is inside the range's own noise.
        return Signal(
            module=self.name,
            score=config.score * direction,
            confidence=min(config.maximum_confidence, config.base_confidence),
            reasoning=(
                f"the {config.range_start_hour:02d}:00-{config.range_end_hour:02d}:00 range "
                f"({width_atr:.2f} ATR wide) broke "
                f"{'up' if direction > 0 else 'down'} through {edge:.5g} within "
                f"{config.breakout_window_hours:.0f}h of its close"
            ),
            key_levels=(top, bottom),
            invalidation_price=opposite,
            details={
                "timeframe": timeframe.value,
                "range_top": top,
                "range_bottom": bottom,
                "range_atr": round(width_atr, 2),
                "session_start": str(session_start),
                "session_end": str(session_end),
                "atr": atr,
            },
        )
