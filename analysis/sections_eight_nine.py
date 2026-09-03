"""Frozen, market-specific sections eight and nine.

Both routes enter only after a closed bar and can therefore be reproduced by
the live market-order path.  Their market, clock and thresholds are frozen in
the account overlay; neither module silently generalises to another symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import (
    SectionEightTrendDayConfig,
    SectionNineSessionVwapConfig,
    SectionTenGoldM1Config,
)
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
            day["high"].astype(float) + day["low"].astype(float) + day["close"].astype(float)
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


class SectionTenGoldM1:
    """Enter an M1 first retest only when the closed M5 trend agrees."""

    name = "section_ten_gold_m1"

    def __init__(self, config: SectionTenGoldM1Config | None = None) -> None:
        self.config = config or SectionTenGoldM1Config()
        self._states: dict[str, dict[str, object]] = {}

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "section ten disabled for this market")
        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        confirmation = ctx.series.get(Timeframe.parse(cfg.confirmation_timeframe))
        needed = max(240, cfg.channel_period + cfg.maximum_wait_bars + 20)
        if series is None or len(series.df) < needed:
            return Signal.neutral(self.name, f"section ten needs {needed} closed M1 bars")
        confirmation_needed = cfg.confirmation_ema_period + cfg.confirmation_slope_bars + 2
        if confirmation is None or len(confirmation.df) < confirmation_needed:
            return Signal.neutral(
                self.name,
                f"section ten needs {confirmation_needed} closed M5 bars for confirmation",
            )

        m5_close = confirmation.df["close"].astype(float)
        m5_ema = m5_close.ewm(span=cfg.confirmation_ema_period, adjust=False).mean()
        m5_slope = float(m5_ema.iloc[-1] - m5_ema.iloc[-1 - cfg.confirmation_slope_bars])

        frame = series.df
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        atr = _atr(frame)
        upper = high.shift(1).rolling(cfg.channel_period).max()
        lower = low.shift(1).rolling(cfg.channel_period).min()
        current_stamp = frame.index[-1]
        state = self._states.get(ctx.symbol)
        if state is not None and state["last_seen"] == current_stamp:
            return state["last_signal"]  # type: ignore[return-value]
        if state is None or state["last_seen"] not in frame.index:
            state = {
                "last_seen": None,
                "candidate": None,
                "last_signal": Signal.neutral(self.name, "section ten warming its lifecycle"),
            }
            start = max(cfg.channel_period + 14, len(frame) - cfg.maximum_wait_bars * 3)
        else:
            start = int(frame.index.searchsorted(state["last_seen"], side="right"))

        signal = Signal.neutral(
            self.name,
            f"no first retest after a {cfg.minimum_break_atr:.2f}-ATR M1 channel break",
        )
        candidate = state["candidate"]
        for at in range(start, len(frame)):
            stamp = frame.index[at]
            consumed = False
            if candidate is not None:
                direction = int(candidate["direction"])
                level = float(candidate["level"])
                unit = float(candidate["unit"])
                age = int(candidate["age"]) + 1
                check_close = float(close.iloc[at])
                failed = (
                    check_close < level - cfg.stop_beyond_atr * unit
                    if direction > 0
                    else check_close > level + cfg.stop_beyond_atr * unit
                )
                touched = (
                    float(low.iloc[at]) <= level + cfg.retest_tolerance_atr * unit
                    if direction > 0
                    else float(high.iloc[at]) >= level - cfg.retest_tolerance_atr * unit
                )
                if failed or age >= cfg.maximum_wait_bars:
                    candidate = None
                elif touched:
                    break_atr = float(candidate["break_atr"])
                    candidate = None
                    consumed = True
                    in_session = (
                        cfg.entry_start_hour_utc <= int(stamp.hour) < cfg.entry_end_hour_utc
                    )
                    in_dead_zone = (
                        cfg.blocked_start_hour_utc <= int(stamp.hour) < cfg.blocked_end_hour_utc
                    )
                    if stamp == current_stamp and in_session and not in_dead_zone:
                        m5_direction = 1 if m5_slope > 0.0 else -1 if m5_slope < 0.0 else 0
                        if m5_direction == direction:
                            entry = ctx.tick.mid if ctx.tick is not None else check_close
                            stop = level - direction * cfg.stop_beyond_atr * unit
                            if direction * (entry - stop) > 0.0:
                                signal = Signal(
                                    module=self.name,
                                    score=cfg.score * direction,
                                    confidence=cfg.confidence,
                                    reasoning=(
                                        "XAUUSD M1 first retest agrees with closed M5 trend"
                                    ),
                                    invalidation_price=stop,
                                    key_levels=(level,),
                                    details={
                                        "timeframe": cfg.timeframe,
                                        "confirmation_timeframe": cfg.confirmation_timeframe,
                                        "break_level": level,
                                        "break_atr": round(break_atr, 6),
                                        "m5_ema_slope": round(m5_slope, 6),
                                        "wait_bars": age,
                                    },
                                )
                        else:
                            signal = Signal.neutral(
                                self.name,
                                "M1 first retest rejected because closed M5 EMA slope disagrees",
                            )
                else:
                    candidate = {**candidate, "age": age}

            # A touch consumes the active setup. A new breakout may only begin
            # on a later bar, never inside the same already-resolved candle.
            if candidate is not None or consumed:
                continue
            unit = float(atr.iloc[at])
            if not np.isfinite(unit) or unit <= 0.0:
                continue
            broken_close = float(close.iloc[at])
            if broken_close > float(upper.iloc[at]):
                direction, level = 1, float(upper.iloc[at])
            elif broken_close < float(lower.iloc[at]):
                direction, level = -1, float(lower.iloc[at])
            else:
                continue
            break_atr = direction * (broken_close - level) / unit
            if break_atr < cfg.minimum_break_atr:
                continue
            candidate = {
                "direction": direction,
                "level": level,
                "unit": unit,
                "age": 0,
                "break_atr": break_atr,
            }

        state["last_seen"] = current_stamp
        state["candidate"] = candidate
        state["last_signal"] = signal
        self._states[ctx.symbol] = state
        return signal
