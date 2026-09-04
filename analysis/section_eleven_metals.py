"""SECTION ELEVEN: one trained model per metal, and none shipped until it earns it.

WHAT THIS IS. Section six is a frozen random-feature model on gold: thirteen
scale-free readings of the last bar, standardised, pushed through a fixed
random projection into forty-eight hidden units, then a linear head. It works
on XAUUSD and only on XAUUSD -- `SectionSixGoldM5.symbol` is a hardcoded
string and its coefficients were fitted on gold. Running them on XAUEUR
produces numbers about nothing.

This is the same mechanism with a model of its own per market. The trainer
that produces those models is `scripts/train_section_eleven.py`, and it is
built to come back empty.

WHY THE FEATURES LIVE HERE AND ARE COMPUTED ONE WAY ONLY.

A model trained on one definition of a feature and run on another is not a
weak model, it is a different model, and nothing in its measured numbers says
so. Section six computes its features for the LAST bar only; a trainer needs
them for every bar, and the obvious move is to write a fast vectorised copy.
Two implementations of one definition is exactly the defect this project keeps
producing -- a value that is correct, is tested, and is not the one the other
path uses.

So there is one function, `feature_frame`, and the live path takes its last
row. That is O(n) per call where a hand-written last-bar version would be
O(1), which on a few hundred M5 bars is nothing, and it cannot drift.

WHAT IS DELIBERATELY MISSING. There are no coefficients in this file. A model
arrives as a file under `models/section_eleven/`, written by the trainer,
carrying the window it was fitted on and the out-of-sample result that
justified it. A symbol with no model file does not trade -- it is absent
rather than neutral, and the difference is the whole point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.types import Direction, MarketContext, Signal, Timeframe

#: Bump when `feature_frame` changes what it computes. A model file records the
#: version it was fitted under and is REFUSED against a different one, because
#: a silently redefined feature is a silently different model.
FEATURE_VERSION = 1

#: Bars needed before the first usable feature row: the 48-bar rolling mean on
#: top of the 14-bar ATR, with room to spare.
WARMUP = 80

#: FIXED and shared by every market, exactly as section six has it. It is not
#: fitted, so it costs no degrees of freedom; what is fitted per market is the
#: standardisation and the linear head. The seed lives here rather than in the
#: trainer so a model file cannot be read back against a different random
#: matrix from the one it was fitted on.
_RNG = np.random.default_rng(7401)
PROJECTION = _RNG.normal(0.0, 0.55, size=(13, 48))
OFFSET = _RNG.uniform(-1.0, 1.0, size=48)

#: Where trained models live, relative to the repository root.
MODEL_DIR = "models/section_eleven"


@dataclass(frozen=True, slots=True)
class MetalModel:
    """One market's fitted model, plus what it was fitted on and how it did."""

    symbol: str
    centre: tuple[float, ...]
    scale: tuple[float, ...]
    beta: tuple[float, ...]
    feature_version: int = FEATURE_VERSION
    trained_from: str = ""
    trained_through: str = ""
    #: The out-of-fold result -- data no fit ever saw -- written by the trainer
    #: so the number that justified the model travels with the model instead of
    #: living in somebody's memory of a terminal window.
    holdout_trades: int = 0
    holdout_r: float = 0.0
    holdout_sigma: float = 0.0
    threshold: float = 0.15

    def reading(self, features: np.ndarray) -> float:
        centre = np.asarray(self.centre)
        scale = np.asarray(self.scale)
        hidden = np.tanh(((features - centre) / scale) @ PROJECTION + OFFSET)
        beta = np.asarray(self.beta)
        return float(beta[0] + hidden @ beta[1:])


def _atr(frame: pd.DataFrame) -> pd.Series:
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
        .rolling(14)
        .mean()
    )


def feature_frame(frame: pd.DataFrame) -> np.ndarray:
    """One row of thirteen features per bar, `nan` where a lookback is short.

    THE SAME THIRTEEN SECTION SIX READS, in the same order and with the same
    definitions. Every one is an ATR ratio, a fraction of the bar's own range,
    or a trigonometric hour, so none carries the instrument's price level --
    which is what makes the SHAPE transferable to another metal at all. What is
    not transferable is the fitted part, and that is exactly what gets refitted
    per market.
    """
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    unit = _atr(frame).replace(0.0, np.nan)
    span = (high - low).replace(0.0, np.nan)
    fast = close.ewm(span=8, adjust=False).mean()
    slow = close.ewm(span=32, adjust=False).mean()
    volume = frame.get("volume", frame.get("tick_volume", pd.Series(0.0, index=frame.index)))
    volume = volume.astype(float)
    hours = frame.index.hour.to_numpy() + frame.index.minute.to_numpy() / 60.0

    columns = [
        close.diff(1) / unit,
        close.diff(3) / unit,
        close.diff(6) / unit,
        close.diff(12) / unit,
        (fast - slow) / unit,
        (close - fast) / unit,
        (close - open_) / unit,
        span / unit,
        (close - low) / span - 0.5,
        unit / unit.rolling(48).mean() - 1.0,
        volume / volume.rolling(48).median().replace(0.0, np.nan) - 1.0,
        pd.Series(np.sin(2.0 * np.pi * hours / 24.0), index=frame.index),
        pd.Series(np.cos(2.0 * np.pi * hours / 24.0), index=frame.index),
    ]
    return np.column_stack([column.to_numpy(dtype=float) for column in columns])


def feature_row(frame: pd.DataFrame) -> np.ndarray | None:
    """The live path: the last bar's features, or None when any is not finite.

    Deliberately the last row of `feature_frame` and not a second
    implementation of it.
    """
    if len(frame) < WARMUP:
        return None
    row = feature_frame(frame)[-1]
    return row if np.isfinite(row).all() else None


def hidden_layer(features: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The 48 hidden units for one or many feature rows.

    Shared by the module and the trainer so the thing being fitted is the thing
    being run. Fitting on one transform and predicting through another is the
    same class of mistake as two feature implementations, one layer down.
    """
    standardised = (np.atleast_2d(features) - centre) / scale
    return np.tanh(standardised @ PROJECTION + OFFSET)


def load_models(directory: Path | str) -> dict[str, MetalModel]:
    """Every model file in a directory, keyed by symbol.

    A file fitted under a different `FEATURE_VERSION` is REFUSED rather than
    loaded, because the alternative is running a model against inputs that are
    no longer the inputs it learned. That failure is silent by nature: the
    numbers keep coming and they are simply wrong.
    """
    root = Path(directory)
    if not root.exists():
        return {}
    found: dict[str, MetalModel] = {}
    for path in sorted(root.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = int(raw.get("feature_version", -1))
        if version != FEATURE_VERSION:
            raise ValueError(
                f"{path.name} was fitted under feature version {version} and this build "
                f"computes version {FEATURE_VERSION}. Retrain or delete it -- running it "
                f"would feed the model inputs it never learned."
            )
        model = MetalModel(
            symbol=str(raw["symbol"]),
            centre=tuple(float(v) for v in raw["centre"]),
            scale=tuple(float(v) for v in raw["scale"]),
            beta=tuple(float(v) for v in raw["beta"]),
            feature_version=version,
            trained_from=str(raw.get("trained_from", "")),
            trained_through=str(raw.get("trained_through", "")),
            holdout_trades=int(raw.get("holdout_trades", 0)),
            holdout_r=float(raw.get("holdout_r", 0.0)),
            holdout_sigma=float(raw.get("holdout_sigma", 0.0)),
            threshold=float(raw.get("threshold", 0.15)),
        )
        if len(model.centre) != 13 or len(model.scale) != 13:
            raise ValueError(f"{path.name}: expected 13 feature statistics")
        if len(model.beta) != PROJECTION.shape[1] + 1:
            raise ValueError(
                f"{path.name}: expected {PROJECTION.shape[1] + 1} coefficients, "
                f"got {len(model.beta)}"
            )
        found[model.symbol] = model
    return found


def write_model(model: MetalModel, directory: Path | str) -> Path:
    """Write one fitted model where `load_models` will find it."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{model.symbol}.json"
    path.write_text(
        json.dumps(
            {
                "symbol": model.symbol,
                "feature_version": model.feature_version,
                "trained_from": model.trained_from,
                "trained_through": model.trained_through,
                "threshold": model.threshold,
                "holdout_trades": model.holdout_trades,
                "holdout_r": round(model.holdout_r, 4),
                "holdout_sigma": round(model.holdout_sigma, 4),
                "centre": [round(v, 10) for v in model.centre],
                "scale": [round(v, 10) for v in model.scale],
                "beta": [round(v, 10) for v in model.beta],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class SectionElevenMetals:
    """The live detector. Silent on any market it has no fitted model for."""

    name = "section_eleven_metals"

    def __init__(self, config, models: dict[str, MetalModel] | None = None) -> None:
        self.config = config
        self._models = models if models is not None else load_models(config.model_dir)

    @property
    def models(self) -> dict[str, MetalModel]:
        return self._models

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        quiet = Signal.neutral(self.name, "no read")
        if not cfg.enabled:
            return quiet
        if cfg.allowed_symbols and ctx.symbol not in cfg.allowed_symbols:
            return quiet
        model = self._models.get(ctx.symbol)
        if model is None:
            # NOT NEUTRAL BECAUSE IT SAW NOTHING -- absent because nothing was
            # ever fitted here. A section that silently trades an unfitted
            # market is section six pointed at XAUEUR, which is the whole
            # mistake this module exists to avoid.
            return quiet

        series = ctx.series.get(Timeframe.parse(cfg.timeframe))
        if series is None or len(series.df) < WARMUP:
            return quiet
        hour = int(series.df.index[-1].hour)
        if cfg.hour_is_blocked(ctx.symbol, hour):
            return quiet

        features = feature_row(series.df)
        if features is None:
            return quiet
        reading = model.reading(features) * cfg.polarity
        if abs(reading) < model.threshold:
            return quiet
        if cfg.long_only and reading < 0:
            return quiet

        unit = float(_atr(series.df).iloc[-1])
        if not np.isfinite(unit) or unit <= 0.0:
            return quiet
        direction = Direction.LONG if reading > 0 else Direction.SHORT
        close = float(series.df["close"].iloc[-1])
        stop = close - (
            cfg.stop_atr * unit if direction is Direction.LONG else -cfg.stop_atr * unit
        )
        score = float(np.clip(abs(reading) / max(model.threshold, 1e-9), 0.0, 2.0)) * 50.0
        return Signal(
            module=self.name,
            score=score if direction is Direction.LONG else -score,
            confidence=cfg.confidence,
            reasoning=(
                f"section eleven {ctx.symbol}: reading {reading:+.3f} against "
                f"threshold {model.threshold:.2f}, model fitted through "
                f"{model.trained_through or 'unknown'}"
            ),
            invalidation_price=stop,
        )
