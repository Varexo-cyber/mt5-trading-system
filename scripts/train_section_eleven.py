"""Fit one section-eleven model per metal, and refuse to ship the ones that fail.

    python scripts/train_section_eleven.py --days 720

WHAT MAKES THIS DIFFERENT FROM HOW SECTION SIX WAS BUILT, which is the only
reason to build it at all.

Section six is a frozen model on gold. It went live, came off at -71.65 R over
180 days, went back on at the owner's instruction, and its own config records
the shape of the failure: a strong recent stretch that was regime
concentration rather than an edge. Fitting five more models the same way
produces five more of that.

So every number this script uses to DECIDE comes from data the fit never saw:

  * WALK-FORWARD. The period is cut into folds by date. The model that
    predicts fold i is fitted only on bars before fold i begins, and is thrown
    away afterwards. Every prediction collected is out-of-fold by
    construction, so there is no version of "it fit the sample" available.
  * A HOLDOUT THAT IS NEVER SEARCHED. The newest 20% takes no part in any
    choice -- not the threshold, not the ridge penalty, not which markets get
    shipped. It is read once, at the end, and it can only disqualify.
  * DAY-CLUSTERED SIGMA. Five metals that break on one morning are one
    observation. Counting each trade independently overstates significance by
    roughly the square root of however many moved together, and this project
    has made that mistake before.
  * A RANDOM CONTROL on the same bars at the same stop and target. A bar
    registers a barrier when its EXTREME crosses it, and the overshoot is
    larger on the nearer barrier, so a coin flip does not read zero. Whatever
    it reads is subtracted.
  * BONFERRONI. Five markets times the thresholds tried is not one hypothesis.
    The bar rises with the count and is printed with it.
  * THE REAL COST, from this broker's own schedule through the sizer, charged
    on every trade taken.

A model is written to disk only when its market clears every bar. Nothing here
promotes anything to live: that is a line in `config/eightcap.yaml`, and it
should not be crossed until a replay through `dry_run_sections` agrees.

WHAT IS BEING PREDICTED. The forward move over the trade horizon, in units of
the ATR at entry -- the same quantity the stop and target are expressed in, so
a reading is directly the thing being traded rather than a proxy for it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import erfc, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.section_eleven_metals import (
    FEATURE_VERSION,
    MODEL_DIR,
    WARMUP,
    MetalModel,
    feature_frame,
    hidden_layer,
    write_model,
)
from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from risk.position_sizer import PositionSizer

#: Bars a trade is given to reach a barrier. Beyond it the trade is discarded
#: rather than marked to market: one that reached neither barrier has not
#: answered the question, and calling it a scratch is an answer it did not give.
HORIZON = 24


@dataclass
class Trades:
    r: list[float]
    day: list[object]

    def __len__(self) -> int:
        return len(self.r)


def _atr(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
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
        .to_numpy()
    )


def forward_target(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """The move over the next `horizon` bars, in ATR at the decision bar.

    Deliberately the same unit the stop is in. A model that predicts a number
    in one unit while the risk is expressed in another can be perfectly
    calibrated and still size everything wrong.
    """
    close = frame["close"].to_numpy(dtype=float)
    unit = _atr(frame)
    ahead = np.full(len(close), np.nan)
    ahead[: len(close) - horizon] = close[horizon:] - close[: len(close) - horizon]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ahead / np.where(unit > 0, unit, np.nan)


def fit(features: np.ndarray, target: np.ndarray, penalty: float) -> tuple[np.ndarray, ...]:
    """Standardise, project, and ridge-fit the linear head.

    Returns `(centre, scale, beta)`. The centre and scale come from the
    TRAINING rows only -- computing them over the whole period would leak the
    holdout's distribution into every fold, which is the quiet kind of
    look-ahead that leaves no trace in the result.
    """
    centre = np.nanmean(features, axis=0)
    scale = np.nanstd(features, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    hidden = hidden_layer(features, centre, scale)
    design = np.column_stack([np.ones(len(hidden)), hidden])
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += penalty
    beta = np.linalg.solve(gram, design.T @ target)
    return centre, scale, beta


def resolve(
    frame: pd.DataFrame,
    signals: np.ndarray,
    *,
    stop_atr: float,
    ratio: float,
    cost_r: float,
    horizon: int,
) -> Trades:
    """First touch of stop or target, one position at a time.

    Entry at the signal bar's close, resolution from the NEXT bar, and a bar
    that spans both barriers is a loss because the order inside it is
    unknowable. One position at a time because the account holds one, and
    because an always-on signal otherwise counts one event many times over --
    the correction that reordered the whole gold search.
    """
    out = Trades([], [])
    unit = _atr(frame)
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    index = frame.index
    free_at = 0
    for i in range(len(frame) - 1):
        if i < free_at:
            continue
        direction = int(signals[i])
        if direction == 0 or not np.isfinite(unit[i]) or unit[i] <= 0:
            continue
        entry = close[i]
        risk = stop_atr * unit[i]
        stop = entry - direction * risk
        target = entry + direction * ratio * risk
        last = min(i + 1 + horizon, len(frame))
        result: float | None = None
        exit_at = last
        for j in range(i + 1, last):
            if direction > 0:
                hit_stop, hit_target = low[j] <= stop, high[j] >= target
            else:
                hit_stop, hit_target = high[j] >= stop, low[j] <= target
            if hit_stop:
                result, exit_at = -1.0 - cost_r, j
                break
            if hit_target:
                result, exit_at = ratio - cost_r, j
                break
        free_at = exit_at
        if result is None:
            continue
        out.r.append(result)
        out.day.append(index[i].date())
    return out


def stats(trades: Trades) -> tuple[float, float, float, int]:
    """`(total R, per trade, day-clustered sigma, n)`."""
    n = len(trades)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    by_day: dict[object, float] = {}
    for day, r in zip(trades.day, trades.r, strict=True):
        by_day[day] = by_day.get(day, 0.0) + r
    daily = np.array(list(by_day.values()), dtype=float)
    total = float(daily.sum())
    if len(daily) < 2:
        return total, total / n, 0.0, n
    spread = float(daily.std(ddof=1)) * float(np.sqrt(len(daily)))
    return total, total / n, (total / spread if spread > 0 else 0.0), n


def bonferroni_sigma(cells: int, target_p: float = 0.05) -> float:
    """The sigma one cell must reach when `cells` of them were tried."""
    if cells <= 0:
        return 2.0
    per_cell = target_p / cells
    low, high = 0.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if erfc(mid / sqrt(2.0)) > per_cell:
            low = mid
        else:
            high = mid
    return round(high, 2)


def walk_forward(
    frame: pd.DataFrame,
    *,
    folds: int,
    penalty: float,
    threshold: float,
    horizon: int,
    stop_atr: float,
    ratio: float,
    cost_r: float,
) -> tuple[np.ndarray, Trades]:
    """Out-of-fold readings for every bar, and the trades they produce.

    The model predicting fold i is fitted only on bars BEFORE fold i and
    discarded after. No prediction here was ever in its own training set.
    """
    features = feature_frame(frame)
    target = forward_target(frame, horizon)
    usable = np.isfinite(features).all(axis=1) & np.isfinite(target)
    readings = np.full(len(frame), np.nan)

    first = int(np.argmax(usable)) if usable.any() else len(frame)
    start = max(first, WARMUP)
    bounds = np.linspace(start, len(frame), folds + 1, dtype=int)
    for fold in range(1, folds + 1):
        train_end = bounds[fold - 1]
        test_end = bounds[fold]
        train = usable.copy()
        train[train_end:] = False
        # A fold with too little history behind it teaches nothing; leave its
        # readings nan and it simply takes no trades.
        if int(train.sum()) < 500:
            continue
        centre, scale, beta = fit(features[train], target[train], penalty)
        block = np.arange(train_end, test_end)
        block = block[usable[block]]
        if not len(block):
            continue
        hidden = hidden_layer(features[block], centre, scale)
        readings[block] = beta[0] + hidden @ beta[1:]

    signals = np.zeros(len(frame), dtype=int)
    live = np.isfinite(readings)
    signals[live & (readings >= threshold)] = 1
    signals[live & (readings <= -threshold)] = -1
    return readings, resolve(
        frame,
        signals,
        stop_atr=stop_atr,
        ratio=ratio,
        cost_r=cost_r,
        horizon=horizon,
    )


def random_control(frame: pd.DataFrame, seed: int, rate: float) -> np.ndarray:
    """A coin flip at the same firing rate on the same bars.

    It does NOT read zero: a bar registers a barrier when its extreme crosses
    it and the overshoot is proportionally larger on the nearer one, so the
    resolver manufactures a small edge out of nothing. Matching the rate
    matters because the bias depends on how often you are in.
    """
    rng = np.random.default_rng(seed)
    share = max(min(rate, 0.9), 0.0) / 2.0
    return rng.choice([-1, 0, 1], size=len(frame), p=[share, 1.0 - 2 * share, share]).astype(int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--clock", default="M5")
    parser.add_argument(
        "--symbols",
        default="XAUEUR,XAUGBP,XAUAUD,XAUJPY",
        help="comma list; the four gold crosses by default. XAUUSD is section six's.",
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--stop-atr", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=1.5)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.30",
        help="readings below this take no trade. Every one tried is counted "
        "against the Bonferroni bar.",
    )
    parser.add_argument("--penalty", type=float, default=25.0)
    parser.add_argument(
        "--cells-already-tried",
        type=int,
        default=0,
        help=(
            "cells fitted in EARLIER runs, added to this run's count before the "
            "Bonferroni bar. Sixteen cells were spent on 4 September at "
            "thresholds 0.10 to 0.30; a follow-up that ignores them is paying "
            "for the grid once and searching it twice."
        ),
    )
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--out", default=MODEL_DIR)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write a model file for every market that clears every bar. "
        "Without it nothing is written and the run only reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    clock = Timeframe.parse(args.clock)

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=True),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    sizer = PositionSizer(settings)

    earlier = max(args.cells_already_tried, 0)
    cells = len(symbols) * len(thresholds) + earlier
    bar = bonferroni_sigma(cells)
    print(f"\nSECTION ELEVEN — fitting {len(symbols)} markets x {len(thresholds)} thresholds")
    print(f"  {args.days} days of {clock.value}, {args.folds} walk-forward folds")
    if earlier:
        print(f"  plus {earlier} cells declared from earlier runs")
    print(f"  Bonferroni bar at {cells} cells: {bar:.2f} sigma on the searched part")
    print(f"  the newest {args.holdout:.0%} is never searched and must clear 2.0 on its own\n")

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    results: list[tuple[str, float, dict]] = []

    try:
        for symbol in symbols:
            try:
                spec = connector.spec(symbol)
                frame = fetch_mt5_history(
                    connector, symbol, clock, start - (WARMUP + 40) * clock.duration, end
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                print(f"  {symbol}: no history ({exc})")
                continue
            if len(frame) < 5_000:
                print(f"  {symbol}: only {len(frame)} bars, not enough to fit anything")
                continue

            stop_price = args.stop_atr * float(np.nanmedian(_atr(frame)))
            commission = settings.risk.commission_per_lot(spec.asset_class.value)
            cost_r = sizer._cost_share(spec, stop_price, commission)

            split = int(len(frame) * (1.0 - args.holdout))
            searched, held_out = frame.iloc[:split], frame.iloc[split:]
            print(
                f"  {symbol}: {len(frame):,} bars, cost {cost_r:.1%} of the stop, "
                f"holdout from {held_out.index[0]:%Y-%m-%d}"
            )

            for threshold in thresholds:
                _readings, trades = walk_forward(
                    searched,
                    folds=args.folds,
                    penalty=args.penalty,
                    threshold=threshold,
                    horizon=args.horizon,
                    stop_atr=args.stop_atr,
                    ratio=args.ratio,
                    cost_r=cost_r,
                )
                total, each, sigma, n = stats(trades)
                rate = n / max(len(searched), 1)
                control = resolve(
                    searched,
                    random_control(searched, seed=abs(hash(symbol)) % 9973, rate=rate),
                    stop_atr=args.stop_atr,
                    ratio=args.ratio,
                    cost_r=cost_r,
                    horizon=args.horizon,
                )
                _ct, control_each, _cs, control_n = stats(control)
                print(
                    f"      threshold {threshold:.2f}: {n:>5} trades  {total:>+8.2f} R  "
                    f"{each:>+7.3f} each  {sigma:>+5.2f} sigma  "
                    f"(coin {control_each:+.3f} over {control_n})"
                )
                results.append(
                    (
                        symbol,
                        threshold,
                        {
                            "n": n,
                            "total": total,
                            "each": each,
                            "sigma": sigma,
                            "net": each - control_each,
                            "cost_r": cost_r,
                            "frame": frame,
                            "searched": searched,
                            "held_out": held_out,
                        },
                    )
                )

        _report(results, args, bar)
    finally:
        connector.shutdown()


def _report(results, args, bar: float) -> None:
    print("\n" + "=" * 74)
    print("WHICH MARKETS EARNED A MODEL")
    print("=" * 74)
    if not results:
        print("  Nothing was fitted. No history, or every market was too short.")
        return

    best: dict[str, tuple[float, dict]] = {}
    for symbol, threshold, row in results:
        if symbol not in best or row["sigma"] > best[symbol][1]["sigma"]:
            best[symbol] = (threshold, row)

    winners = []
    for symbol, (threshold, row) in sorted(best.items()):
        why = _why_not(row, bar)
        if why:
            print(f"\n  {symbol}: NO — {why}")
            continue

        held = _on_the_holdout(row, args, threshold)
        holdout_total, _each, holdout_sigma, holdout_n = held
        if holdout_n < 100:
            print(f"\n  {symbol}: NO — only {holdout_n} trades on the untouched holdout")
            continue
        if holdout_total <= 0 or holdout_sigma < 2.0:
            print(
                f"\n  {symbol}: NO — searched part clears the bar, holdout does not "
                f"({holdout_total:+.2f} R at {holdout_sigma:+.2f} sigma over {holdout_n})"
            )
            continue

        print(
            f"\n  {symbol}: YES at threshold {threshold:.2f}\n"
            f"      searched {row['total']:+.2f} R over {row['n']} at {row['sigma']:+.2f} sigma\n"
            f"      holdout  {holdout_total:+.2f} R over {holdout_n} at {holdout_sigma:+.2f} sigma"
        )
        winners.append((symbol, threshold, row, held))

    if not winners:
        print(
            "\n  NOTHING SURVIVED. That is the expected outcome of an honest fit\n"
            "  and it is not a failure of the run. Section six went live on a\n"
            "  number that later read -71.65 R; this is what refusing that looks\n"
            "  like from the inside."
        )
        return

    if not args.write:
        print("\n  Nothing written. Re-run with --write to save these models.")
        return

    for symbol, threshold, row, held in winners:
        frame = row["frame"]
        centre, scale, beta = fit(
            *_all_usable(frame, args.horizon),
            args.penalty,
        )
        model = MetalModel(
            symbol=symbol,
            centre=tuple(float(v) for v in centre),
            scale=tuple(float(v) for v in scale),
            beta=tuple(float(v) for v in beta),
            feature_version=FEATURE_VERSION,
            trained_from=f"{frame.index[0]:%Y-%m-%d}",
            trained_through=f"{frame.index[-1]:%Y-%m-%d}",
            holdout_trades=held[3],
            holdout_r=held[0],
            holdout_sigma=held[2],
            threshold=threshold,
        )
        path = write_model(model, args.out)
        print(f"  wrote {path}")
    print(
        "\n  A model on disk is not a section on the live list. Replay it through\n"
        "  dry_run_sections before anything touches money."
    )


def _all_usable(frame: pd.DataFrame, horizon: int):
    features = feature_frame(frame)
    target = forward_target(frame, horizon)
    usable = np.isfinite(features).all(axis=1) & np.isfinite(target)
    return features[usable], target[usable]


def _why_not(row: dict, bar: float) -> str:
    if row["n"] < 200:
        return f"{row['n']} trades, under the 200 needed to judge it"
    if row["net"] <= 0:
        return f"{row['net']:+.3f} R against a coin flip on the same bars"
    if row["sigma"] < bar:
        return f"{row['sigma']:+.2f} sigma against a bar of {bar:.2f}"
    return ""


def _on_the_holdout(row: dict, args, threshold: float) -> tuple[float, float, float, int]:
    """The newest slice, read once, with a model fitted only on what precedes it."""
    frame = row["frame"]
    split = len(row["searched"])
    features, target = _all_usable(row["searched"], args.horizon)
    if len(features) < 500:
        return 0.0, 0.0, 0.0, 0
    centre, scale, beta = fit(features, target, args.penalty)

    tail = frame.iloc[max(split - WARMUP - args.horizon, 0) :]
    tail_features = feature_frame(tail)
    usable = np.isfinite(tail_features).all(axis=1)
    readings = np.full(len(tail), np.nan)
    if usable.any():
        hidden = hidden_layer(tail_features[usable], centre, scale)
        readings[usable] = beta[0] + hidden @ beta[1:]
    signals = np.zeros(len(tail), dtype=int)
    live = np.isfinite(readings)
    signals[live & (readings >= threshold)] = 1
    signals[live & (readings <= -threshold)] = -1
    # Only the holdout itself trades; the warm-up rows exist to make the
    # features computable and must not contribute trades of their own.
    signals[: max(split - max(split - WARMUP - args.horizon, 0), 0)] = 0
    return stats(
        resolve(
            tail,
            signals,
            stop_atr=args.stop_atr,
            ratio=args.ratio,
            cost_r=row["cost_r"],
            horizon=args.horizon,
        )
    )


if __name__ == "__main__":
    main()
