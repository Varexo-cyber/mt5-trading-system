"""Frozen H1 SPX500 reversal selected by chronological walk-forward."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.schema import WalkforwardIndexConfig
from core.types import MarketContext, Signal, Timeframe

_BETA = np.asarray(
    [
        0.20329642815181037,
        -0.24011504166516967,
        -0.06442462473482552,
        0.012159435280432061,
        -0.07615356731052322,
        -0.00874235146116553,
        0.18496380853349834,
        0.2419826862890538,
        -0.06748711246986092,
        -0.052981599298536276,
        0.2625544995355195,
        0.020655083745714346,
        0.14066012869827355,
        -0.13029556579160262,
    ]
)
_CENTRE = np.asarray(
    [
        0.02520810438282174,
        0.07648187854860523,
        0.15880260092213971,
        0.35514653940761315,
        0.4568385617673823,
        0.0979130005149801,
        0.029945048995096684,
        1.0265611801885803,
        0.01684560995355213,
        -0.0015447941351359597,
        0.02790813080513689,
        0.0522263641936689,
        -0.0548924897628453,
    ]
)
_SCALE = np.asarray(
    [
        0.7379771146784881,
        1.2347415788716471,
        1.7058062907698497,
        2.3922044562776756,
        1.4012665555097619,
        0.8812342092783573,
        0.7186839213679546,
        0.6589129585190695,
        0.29614871141978644,
        0.276023666900219,
        0.4441695314589642,
        0.70764414470046,
        0.7024948298186425,
    ]
)


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


def model_reading(frame: pd.DataFrame) -> tuple[float, float] | None:
    if len(frame) < 80:
        return None
    close, open_ = frame["close"].astype(float), frame["open"].astype(float)
    high, low = frame["high"].astype(float), frame["low"].astype(float)
    unit = _atr(frame).replace(0.0, np.nan)
    span = (high - low).replace(0.0, np.nan)
    fast, slow = close.ewm(span=8, adjust=False).mean(), close.ewm(span=32, adjust=False).mean()
    volume = frame.get(
        "volume",
        frame.get("tick_volume", pd.Series(0.0, index=frame.index)),
    ).astype(float)
    hour = frame.index[-1].hour + frame.index[-1].minute / 60.0
    values = np.asarray(
        [
            close.diff(1).iloc[-1] / unit.iloc[-1],
            close.diff(3).iloc[-1] / unit.iloc[-1],
            close.diff(6).iloc[-1] / unit.iloc[-1],
            close.diff(12).iloc[-1] / unit.iloc[-1],
            (fast.iloc[-1] - slow.iloc[-1]) / unit.iloc[-1],
            (close.iloc[-1] - fast.iloc[-1]) / unit.iloc[-1],
            (close.iloc[-1] - open_.iloc[-1]) / unit.iloc[-1],
            span.iloc[-1] / unit.iloc[-1],
            (close.iloc[-1] - low.iloc[-1]) / span.iloc[-1] - 0.5,
            unit.iloc[-1] / unit.rolling(48).mean().iloc[-1] - 1.0,
            volume.iloc[-1] / volume.rolling(48).median().replace(0.0, np.nan).iloc[-1] - 1.0,
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
        ]
    )
    if not np.isfinite(values).all():
        return None
    return float(_BETA[0] + np.dot(_BETA[1:], (values - _CENTRE) / _SCALE)), float(unit.iloc[-1])


class WalkforwardIndex:
    name = "walkforward_index"

    def __init__(self, config: WalkforwardIndexConfig | None = None) -> None:
        self.config = config or WalkforwardIndexConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "walk-forward section disabled for this symbol")
        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        found = model_reading(series.df) if series is not None else None
        if found is None:
            return Signal.neutral(self.name, "walk-forward model needs 80 closed H1 bars")
        reading, unit = found
        if abs(reading) < cfg.threshold:
            return Signal.neutral(
                self.name, f"model magnitude {abs(reading):.3f} below {cfg.threshold:.3f}"
            )
        direction = -1 if reading > 0.0 else 1
        price = ctx.tick.mid if ctx.tick is not None else float(series.df["close"].iloc[-1])
        return Signal(
            module=self.name,
            score=cfg.base_score * direction,
            confidence=cfg.confidence,
            reasoning=f"frozen H1 index reversal reading {reading:+.3f}",
            invalidation_price=price - direction * cfg.stop_atr * unit,
            details={"timeframe": cfg.timeframe, "model_reading": round(reading, 6)},
        )
