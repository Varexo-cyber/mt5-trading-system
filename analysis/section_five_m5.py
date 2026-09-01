"""Frozen nonlinear M5 NDX100 model selected by chronological replay."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.schema import SectionFiveM5Config
from core.types import MarketContext, Signal, Timeframe

_BETA = np.asarray(
    [
        0.15322826256792685,
        0.008411209029653819,
        0.0017934194730331579,
        0.039113203513064125,
        -0.04488127104303717,
        0.0321507376770254,
        0.027557682810592725,
        0.042781607016816305,
        0.001313617599492947,
        -0.007627659618344806,
        0.005327586844612847,
        0.009950405466067127,
        0.04187438877576331,
        0.012292826264450945,
        0.004919296434644822,
        -0.015870337104483254,
        -0.07846987033164197,
        0.018776840960658535,
        0.002878571742849899,
        -0.07373977175016896,
        -0.029503236465402132,
        -0.05275238054705102,
        -0.02457232621985641,
        0.02717358477098125,
        -0.037018665231411375,
        0.04272134174984162,
        -0.06122989349660795,
        0.00914581230839449,
        0.027318205875194172,
        0.017821050984937904,
        -0.04820070458704244,
        -0.009589725573855979,
        -0.03573247563026806,
        0.022106419522080187,
        0.0896606993081217,
        -0.008639437993585807,
        0.006974146459447315,
        -0.03561959233230495,
        0.022255484859064614,
        0.028979675431129944,
        0.015658821174253996,
        -0.0005069794164994461,
        -0.01747747334098895,
        0.02955742627297673,
        -0.04182507048945397,
        0.05068648473466053,
        0.046742402690605596,
        -0.018116579376226658,
        0.008409567316445718,
    ]
)
_CENTRE = np.asarray(
    [
        0.0035717234206874763,
        0.009531364646750168,
        0.02387134524240079,
        0.06452566091660898,
        0.08632594618166367,
        0.01553481750307419,
        0.006833312892491378,
        1.004149313812427,
        0.005988366707722066,
        0.025188558887334532,
        0.12281809531260084,
        0.015027473764373562,
        -0.020549238673339947,
    ]
)
_SCALE = np.asarray(
    [
        0.7037256872154103,
        1.1700340724681497,
        1.6048286193610954,
        2.2108404832902115,
        1.2271058615035864,
        0.8236455354325305,
        0.687953072302456,
        0.5087041400735837,
        0.31646318127954426,
        0.37347450208566685,
        0.669081473757011,
        0.7096740188774436,
        0.7040700893749703,
    ]
)
_RNG = np.random.default_rng(7401)
_PROJECTION = _RNG.normal(0.0, 0.55, size=(13, 48))
_OFFSET = _RNG.uniform(-1.0, 1.0, size=48)


def _atr(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return spans.rolling(14).mean()


def model_reading(frame: pd.DataFrame) -> tuple[float, float] | None:
    if len(frame) < 80:
        return None
    close, open_ = frame["close"].astype(float), frame["open"].astype(float)
    high, low = frame["high"].astype(float), frame["low"].astype(float)
    unit = _atr(frame).replace(0.0, np.nan)
    span = (high - low).replace(0.0, np.nan)
    fast, slow = close.ewm(span=8, adjust=False).mean(), close.ewm(span=32, adjust=False).mean()
    volume = frame.get("volume", frame.get("tick_volume", pd.Series(0.0, index=frame.index)))
    volume = volume.astype(float)
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
    hidden = np.tanh(((values - _CENTRE) / _SCALE) @ _PROJECTION + _OFFSET)
    return float(_BETA[0] + hidden @ _BETA[1:]), float(unit.iloc[-1])


class SectionFiveM5:
    name = "section_five_m5"

    def __init__(self, config: SectionFiveM5Config | None = None) -> None:
        self.config = config or SectionFiveM5Config()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "section five disabled for this symbol")
        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        found = model_reading(series.df) if series is not None else None
        if found is None:
            return Signal.neutral(self.name, "section five needs 80 closed M5 bars")
        reading, unit = found
        if abs(reading) < cfg.threshold:
            return Signal.neutral(self.name, f"model magnitude {abs(reading):.3f} below threshold")
        direction = 1 if reading > 0.0 else -1
        price = ctx.tick.mid if ctx.tick is not None else float(series.df["close"].iloc[-1])
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning=f"frozen nonlinear M5 reading {reading:+.3f}",
            invalidation_price=price - direction * cfg.stop_atr * unit,
            details={"timeframe": cfg.timeframe, "model_reading": round(reading, 6)},
        )
