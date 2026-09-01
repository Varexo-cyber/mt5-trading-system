"""Fade the first failed SPX500 break of the measured European range."""

from __future__ import annotations

from datetime import timedelta

from config.schema import FailedSessionBreakoutConfig
from core.types import MarketContext, Signal, Timeframe


class FailedSessionBreakout:
    """One M5 trigger per session, frozen from chronological broker replay."""

    name = "failed_session_breakout"

    def __init__(self, config: FailedSessionBreakoutConfig | None = None) -> None:
        self.config = config or FailedSessionBreakoutConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "failed-session section disabled for this symbol")
        timeframe = Timeframe.parse(cfg.timeframe)
        series = ctx.series.get(timeframe)
        if series is None or len(series.df) < 36:
            return Signal.neutral(self.name, "needs 36 closed M5 bars")
        frame = series.df
        now = frame.index[-1]
        day = now.normalize()
        range_start = day + timedelta(hours=cfg.range_start_hour)
        range_end = day + timedelta(hours=cfg.range_end_hour)
        trade_end = range_end + timedelta(hours=cfg.trade_window_hours)
        if now < range_end or now >= trade_end:
            return Signal.neutral(self.name, "outside the measured session-break window")
        window = frame[(frame.index >= range_start) & (frame.index < range_end)]
        if len(window) < 12:
            return Signal.neutral(self.name, "incomplete measured session range")
        top, bottom = float(window["high"].max()), float(window["low"].min())
        width = top - bottom
        if width <= 0.0:
            return Signal.neutral(self.name, "session range has no width")
        close = float(frame["close"].iloc[-1])
        previous = float(frame["close"].iloc[-2])
        crossed_up = close > top and previous <= top
        crossed_down = close < bottom and previous >= bottom
        if not crossed_up and not crossed_down:
            return Signal.neutral(self.name, "no fresh close crossing the session edge")
        earlier = frame[(frame.index >= range_end) & (frame.index < now)]
        if ((earlier["close"] > top) | (earlier["close"] < bottom)).any():
            return Signal.neutral(self.name, "the first session break already occurred")
        # The measured edge is the FADE: a break up is sold and a break down is bought.
        direction = -1 if crossed_up else 1
        stop_distance = width * cfg.stop_range_share
        price = ctx.tick.mid if ctx.tick is not None else close
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning=(
                f"fresh M5 close {'above' if crossed_up else 'below'} the "
                f"{cfg.range_start_hour:02d}:00-{cfg.range_end_hour:02d}:00 range; "
                "fading the failed session break"
            ),
            invalidation_price=price - direction * stop_distance,
            key_levels=(top, bottom),
            details={
                "timeframe": cfg.timeframe,
                "range_top": top,
                "range_bottom": bottom,
                "range_width": width,
            },
        )
