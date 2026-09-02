"""Frozen, market-specific sections eight and nine.

Both routes enter only after a closed bar and can therefore be reproduced by
the live market-order path.  Their market, clock and thresholds are frozen in
the account overlay; neither module silently generalises to another symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import SectionEightTrendDayConfig, SectionNineSessionVwapConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return spans.rolling(period).mean()


class SectionEightTrendDayH1:
    """Follow an SPX500 prior day that closed in its outer decile."""

    name = "section_eight_trend_day_h1"

    def __init__(self, config: SectionEightTrendDayConfig | None = None) -> None:
        self.config = config or SectionEightTrendDayConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "section eight disabled for this market")
        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        if series is None or len(series.df) < 60:
            return Signal.neutral(self.name, "section eight needs 60 closed H1 bars")
        frame = series.df
        now = frame.index[-1]
        if now.hour >= cfg.entry_end_hour_utc:
            return Signal.neutral(self.name, "outside the measured 00:00-02:00 UTC entry window")
        today = now.normalize()
        history = frame[frame.index < today]
        if history.empty:
            return Signal.neutral(self.name, "previous UTC session is unavailable")
        previous_day = history.index[-1].normalize()
        previous = history[history.index.normalize() == previous_day]
        high = float(previous["high"].max())
        low = float(previous["low"].min())
        close = float(previous["close"].iloc[-1])
        if not np.isfinite([high, low, close]).all() or high <= low:
            return Signal.neutral(self.name, "previous UTC session has no usable range")
        location = (close - low) / (high - low)
        if location > cfg.upper_close_location:
            direction = 1
        elif location < cfg.lower_close_location:
            direction = -1
        else:
            return Signal.neutral(self.name, f"prior close location {location:.3f} is not extreme")
        unit = float(_atr(frame).iloc[-1])
        if not np.isfinite(unit) or unit <= 0.0:
            return Signal.neutral(self.name, "H1 ATR unavailable")
        entry = ctx.tick.mid if ctx.tick is not None else float(frame["close"].iloc[-1])
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning=f"prior SPX500 UTC day closed at {location:.1%} of its range",
            invalidation_price=entry - direction * cfg.stop_atr * unit,
            details={"timeframe": cfg.timeframe, "prior_close_location": round(location, 6)},
        )


class SectionNineSessionVwapM30:
    """Fade a two-ATR USDJPY displacement from the current UTC-day VWAP."""

    name = "section_nine_vwap_m30"

    def __init__(self, config: SectionNineSessionVwapConfig | None = None) -> None:
        self.config = config or SectionNineSessionVwapConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "section nine disabled for this market")
        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        if series is None or len(series.df) < 60:
            return Signal.neutral(self.name, "section nine needs 60 closed M30 bars")
        frame = series.df
        unit = float(_atr(frame).iloc[-1])
        if not np.isfinite(unit) or unit <= 0.0:
            return Signal.neutral(self.name, "M30 ATR unavailable")
        day = frame[frame.index.normalize() == frame.index[-1].normalize()]
        volume = day.get("volume", day.get("tick_volume"))
        if volume is None:
            return Signal.neutral(self.name, "session VWAP needs broker volume")
        volume = volume.astype(float)
        total = float(volume.sum())
        if not np.isfinite(total) or total <= 0.0:
            return Signal.neutral(self.name, "session VWAP has no broker volume")
        typical = (
            day["high"].astype(float)
            + day["low"].astype(float)
            + day["close"].astype(float)
        ) / 3.0
        vwap = float((typical * volume).sum() / total)
        close = float(frame["close"].iloc[-1])
        displacement = (close - vwap) / unit
        if abs(displacement) < cfg.minimum_displacement_atr:
            return Signal.neutral(
                self.name,
                f"session VWAP displacement {abs(displacement):.3f} ATR below threshold",
            )
        direction = -1 if displacement > 0.0 else 1
        entry = ctx.tick.mid if ctx.tick is not None else close
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning=f"USDJPY is {displacement:+.2f} ATR from its UTC-session VWAP",
            invalidation_price=entry - direction * cfg.stop_atr * unit,
            key_levels=(vwap,),
            details={
                "timeframe": cfg.timeframe,
                "session_vwap": vwap,
                "vwap_displacement_atr": round(displacement, 6),
            },
        )
