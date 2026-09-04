"""Frozen own-chart gold-cross candidates awaiting actual-broker validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    return (
        pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(period)
        .mean()
    )


class GoldCrossDiscovery:
    """One market, clock and mechanism so evidence cannot leak across sections."""

    def __init__(self, config, *, name: str) -> None:
        self.config = config
        self.name = name

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        quiet = Signal.neutral(self.name, "no read")
        if not cfg.enabled or ctx.symbol != cfg.allowed_symbols[0]:
            return quiet
        timeframe = Timeframe.parse(cfg.timeframe)
        series = ctx.series.get(timeframe)
        if series is None:
            return quiet
        frame = series.df
        required = max(60, cfg.channel_period + 2, cfg.momentum_bars + 15)
        if len(frame) < required:
            return quiet

        # The replay fills at the next bar open. Test the hour of that fill,
        # not the start time of the signal bar at a session boundary.
        entry_hour = int((frame.index[-1] + timeframe.duration).hour)
        if not cfg.hour_is_open(entry_hour):
            return quiet

        close = frame["close"].astype(float)
        open_ = frame["open"].astype(float)
        fast = close.ewm(span=20, adjust=False).mean()
        slow = close.ewm(span=50, adjust=False).mean()
        natural = 0
        if cfg.mechanism == "channel_breakout":
            ceiling = float(frame["high"].shift(1).rolling(cfg.channel_period).max().iloc[-1])
            floor = float(frame["low"].shift(1).rolling(cfg.channel_period).min().iloc[-1])
            if fast.iloc[-1] > slow.iloc[-1] and close.iloc[-1] > ceiling:
                natural = 1
            elif fast.iloc[-1] < slow.iloc[-1] and close.iloc[-1] < floor:
                natural = -1
        elif cfg.mechanism == "aligned_momentum":
            unit = float(_atr(frame).iloc[-1])
            if not np.isfinite(unit) or unit <= 0:
                return quiet
            impulse = float((close.iloc[-1] - close.iloc[-1 - cfg.momentum_bars]) / unit)
            if (
                fast.iloc[-1] > slow.iloc[-1]
                and impulse > cfg.minimum_momentum_atr
                and close.iloc[-1] > open_.iloc[-1]
            ):
                natural = 1
            elif (
                fast.iloc[-1] < slow.iloc[-1]
                and impulse < -cfg.minimum_momentum_atr
                and close.iloc[-1] < open_.iloc[-1]
            ):
                natural = -1
        if natural == 0:
            return quiet

        signed = natural * cfg.polarity
        unit = float(_atr(frame).iloc[-1])
        if not np.isfinite(unit) or unit <= 0:
            return quiet
        price = float(close.iloc[-1])
        invalidation = price - signed * cfg.stop_atr * unit
        return Signal(
            module=self.name,
            score=cfg.score if signed > 0 else -cfg.score,
            confidence=cfg.confidence,
            reasoning=(
                f"{self.name}: {cfg.mechanism}, {cfg.timeframe}, "
                f"entry hour {entry_hour:02d} UTC"
            ),
            invalidation_price=invalidation,
        )
