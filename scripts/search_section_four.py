"""Search this broker's own bars for an edge that survives its own costs.

    python scripts/search_section_four.py --days 365

WHY THIS RUNS ON EIGHTCAP DATA AND NOT ON HISTDATA.

The original research measured 94 detectors over ten years of HistData and
shipped the two that survived a holdout. On this broker, over 1,610 live-shaped
trades and 180 days, those two came back at 49.9% -- a coin flip. The entry
carried no information at all.

So the failure was not "not enough research". It was research on the wrong
data. HistData FX bid bars cannot price this account's spread, cannot see its
commission schedule, and contain none of the session structure that index CFDs
have. Doing more of it would be repeating the mistake with more decimals.

This searches the bars the account actually trades.

WHAT IT GUARDS AGAINST, because a search is a machine for finding noise:

  * BONFERRONI. Testing 40 cells and keeping the best one finds a 2-sigma
    result about 63% of the time on pure noise. The bar rises with the size of
    the grid and is printed with the grid.
  * A HOLDOUT SPLIT BY DATE. Train on the older 60%, and a candidate must
    reach its bar on the newer 40% ON ITS OWN, in the same direction.
  * DAY-CLUSTERED SIGMA. Sixteen markets breaking on one morning are one
    observation. This was the largest single correction in the original
    research and it is the easiest one to lose.
  * A RANDOM CONTROL, measured in the same harness, on the same bars, at the
    same ratio. A bar registers a barrier when its extreme crosses it, and the
    overshoot is proportionally larger on the nearer one, so a coin flip does
    NOT read zero. Whatever it reads is subtracted from every candidate.
  * THE REAL COST. Charged from this broker's own commission schedule and
    slippage assumption, per asset class, exactly as the sizer charges it.

Nothing here is a strategy yet. It is the machine that decides whether one
exists, and it is built to come back empty.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe

#: Bars of history a candidate may look back over before its first signal.
WARMUP = 120
#: Bars of future a trade is given to resolve. Beyond this it is unresolved and
#: excluded rather than marked to market -- a trade that never reached either
#: barrier has not answered the question.
HORIZON = 48


# --------------------------------------------------------------------------
# candidates
#
# Each returns an array of direction per bar: +1 long, -1 short, 0 nothing.
# Deliberately simple and deliberately DIFFERENT from each other -- a grid of
# forty variations on one idea tests one idea forty times and pays the
# Bonferroni price for the privilege.
# --------------------------------------------------------------------------


def _atr(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean().to_numpy()


def gap_continuation(frame: pd.DataFrame) -> np.ndarray:
    """An index CFD opens away from its last close; trade the direction of it.

    UNTESTABLE ON HISTDATA FX, which is why it is here. Spot FX runs
    continuously and gaps only over a weekend; index CFDs gap at every session
    boundary because the underlying was closed. The original 94 detectors could
    not see this mechanism at all.
    """
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    open_ = frame["open"].to_numpy()
    gap = np.zeros(len(frame))
    gap[1:] = (open_[1:] - close[:-1]) / np.where(unit[1:] > 0, unit[1:], np.nan)
    out = np.zeros(len(frame), dtype=int)
    out[gap > 0.5] = 1
    out[gap < -0.5] = -1
    return out


def gap_fade(frame: pd.DataFrame) -> np.ndarray:
    """The same event, traded the other way. Both directions of one mechanism
    must be measured or the one that happens to pay looks like a discovery."""
    return -gap_continuation(frame)


def overnight_drift(frame: pd.DataFrame) -> np.ndarray:
    """Hold from one session's close into the next, direction of the last day.

    Index CFDs have carried a documented close-to-open drift for decades. If it
    survives this broker's costs it is the cheapest edge available; if it does
    not, that is worth knowing before anything cleverer is tried.
    """
    close = frame["close"].to_numpy()
    unit = _atr(frame)
    out = np.zeros(len(frame), dtype=int)
    if len(close) < 3:
        return out
    move = np.zeros(len(close))
    move[1:] = (close[1:] - close[:-1]) / np.where(unit[1:] > 0, unit[1:], np.nan)
    out[move > 0.3] = 1
    out[move < -0.3] = -1
    return out


def streak_reversal(frame: pd.DataFrame, length: int = 4) -> np.ndarray:
    """Four closes the same way, then trade against it."""
    close = frame["close"].to_numpy()
    up = np.zeros(len(close), dtype=int)
    up[1:] = np.sign(np.diff(close)).astype(int)
    out = np.zeros(len(close), dtype=int)
    for i in range(length, len(close)):
        window = up[i - length + 1 : i + 1]
        if np.all(window > 0):
            out[i] = -1
        elif np.all(window < 0):
            out[i] = 1
    return out


def streak_continuation(frame: pd.DataFrame, length: int = 4) -> np.ndarray:
    return -streak_reversal(frame, length)


def range_expansion(frame: pd.DataFrame) -> np.ndarray:
    """A bar twice the recent range, in its own direction.

    Distinct from `impulse_retest`: no channel, no level, no retest. It buys
    the expansion itself, which the research measured at NOTHING on FX. On a
    zero-commission index with a wider stop the cost arithmetic is different,
    and that difference is the whole question.
    """
    unit = _atr(frame)
    span = (frame["high"] - frame["low"]).to_numpy()
    body = (frame["close"] - frame["open"]).to_numpy()
    out = np.zeros(len(frame), dtype=int)
    big = span > 2.0 * unit
    out[big & (body > 0)] = 1
    out[big & (body < 0)] = -1
    return out


def inside_bar_break(frame: pd.DataFrame) -> np.ndarray:
    """A bar wholly inside its predecessor, then the next close decides."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    out = np.zeros(len(frame), dtype=int)
    for i in range(2, len(frame)):
        if high[i - 1] < high[i - 2] and low[i - 1] > low[i - 2]:
            if close[i] > high[i - 1]:
                out[i] = 1
            elif close[i] < low[i - 1]:
                out[i] = -1
    return out


def volatility_contraction(frame: pd.DataFrame) -> np.ndarray:
    """Trade the break out of an unusually quiet stretch, either way it goes."""
    unit = _atr(frame)
    slow = pd.Series(unit).rolling(50).mean().to_numpy()
    close = frame["close"].to_numpy()
    high = pd.Series(frame["high"]).shift(1).rolling(10).max().to_numpy()
    low = pd.Series(frame["low"]).shift(1).rolling(10).min().to_numpy()
    quiet = unit < 0.7 * slow
    out = np.zeros(len(frame), dtype=int)
    out[quiet & (close > high)] = 1
    out[quiet & (close < low)] = -1
    return out


def close_position_in_range(frame: pd.DataFrame) -> np.ndarray:
    """Where the close sits inside the bar. A close on the high after a down
    bar is a rejection; the reverse is exhaustion. One bar, no memory."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    span = np.where(high - low > 0, high - low, np.nan)
    position = (close - low) / span
    out = np.zeros(len(frame), dtype=int)
    out[position > 0.9] = 1
    out[position < 0.1] = -1
    return out


def close_position_fade(frame: pd.DataFrame) -> np.ndarray:
    return -close_position_in_range(frame)


def prior_day_break(frame: pd.DataFrame) -> np.ndarray:
    """The first close beyond yesterday's high or low.

    A level everyone can see, on instruments that have a real yesterday --
    which spot FX, trading through midnight, only half does.
    """
    days = frame.index.normalize()
    high = frame["high"].groupby(days).transform("max").shift(1).to_numpy()
    low = frame["low"].groupby(days).transform("min").shift(1).to_numpy()
    close = frame["close"].to_numpy()
    out = np.zeros(len(frame), dtype=int)
    fresh = np.zeros(len(frame), dtype=bool)
    fresh[1:] = days.to_numpy()[1:] != days.to_numpy()[:-1]
    seen_up = seen_down = False
    for i in range(len(frame)):
        if fresh[i]:
            seen_up = seen_down = False
        if not np.isfinite(high[i]) or not np.isfinite(low[i]):
            continue
        if close[i] > high[i] and not seen_up:
            out[i], seen_up = 1, True
        elif close[i] < low[i] and not seen_down:
            out[i], seen_down = -1, True
    return out


def prior_day_fade(frame: pd.DataFrame) -> np.ndarray:
    return -prior_day_break(frame)


#: Name -> signal function. Every mechanism appears in BOTH directions where
#: that makes sense, so a family cannot be credited for the half that happened
#: to work.
CANDIDATES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "gap_continuation": gap_continuation,
    "gap_fade": gap_fade,
    "overnight_drift": overnight_drift,
    "streak_reversal": streak_reversal,
    "streak_continuation": streak_continuation,
    "range_expansion": range_expansion,
    "inside_bar_break": inside_bar_break,
    "volatility_contraction": volatility_contraction,
    "close_position_in_range": close_position_in_range,
    "close_position_fade": close_position_fade,
    "prior_day_break": prior_day_break,
    "prior_day_fade": prior_day_fade,
}


@dataclass
class Trades:
    """Resolved outcomes, kept as parallel arrays so the statistics are cheap."""

    r: list[float] = field(default_factory=list)
    day: list[object] = field(default_factory=list)
    when: list[datetime] = field(default_factory=list)

    def extend(self, other: Trades) -> None:
        self.r.extend(other.r)
        self.day.extend(other.day)
        self.when.extend(other.when)

    def __len__(self) -> int:
        return len(self.r)


def resolve(
    frame: pd.DataFrame,
    signals: np.ndarray,
    *,
    stop_atr: float,
    ratio: float,
    cost_r: float,
) -> Trades:
    """First touch of stop or target, entry at the signal bar's close.

    THE RULES ARE THE RESEARCH'S, and the two that matter both cost the
    candidate rather than flatter it: resolution starts on the bar AFTER the
    entry bar, and a bar that spans both barriers is a LOSS because the order
    inside it is unknowable.
    """
    out = Trades()
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    index = frame.index

    for i in range(WARMUP, len(frame) - 1):
        direction = int(signals[i])
        if direction == 0 or not np.isfinite(unit[i]) or unit[i] <= 0:
            continue
        entry = close[i]
        risk = stop_atr * unit[i]
        stop = entry - direction * risk
        target = entry + direction * ratio * risk
        for j in range(i + 1, min(i + 1 + HORIZON, len(frame))):
            if direction > 0:
                hit_stop, hit_target = low[j] <= stop, high[j] >= target
            else:
                hit_stop, hit_target = high[j] >= stop, low[j] <= target
            if hit_stop:
                out.r.append(-1.0 - cost_r)
                break
            if hit_target:
                out.r.append(ratio - cost_r)
                break
        else:
            continue
        out.day.append(index[i].date())
        out.when.append(index[i])
    return out


def stats(trades: Trades) -> tuple[float, float, float, int]:
    """`(total R, per trade, day-clustered sigma, n)`.

    SIGMA COMES FROM THE SPREAD OF DAILY TOTALS. Counting each trade as an
    independent observation treats sixteen markets breaking on one morning as
    sixteen, and overstates significance by roughly the square root of however
    many moved together.
    """
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
    se = float(daily.std(ddof=1)) * float(np.sqrt(len(daily)))
    return total, total / n, (total / se if se > 0 else 0.0), n


def _cost_share(settings, asset_class: str, spec, stop_price: float) -> float:
    """What a round trip costs as a fraction of the stop, from the real
    schedule. Only forex pays commission on this account, which is why forex
    is where the cost wall sits."""
    per_side = settings.risk.commission_by_asset_class.get(
        asset_class, settings.risk.commission_per_lot_per_side
    )
    slippage_pips = settings.risk.stop_slippage_pips.get(asset_class, 1.0)
    pip = spec.point * 10.0
    commission_price = (per_side * 2.0) * pip / 10.0
    return (commission_price + slippage_pips * pip) / stop_price if stop_price > 0 else 1.0


def random_control(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """A coin flip on the same bars, so the harness measures its own bias.

    IT DOES NOT READ ZERO, and the original research learned that the
    expensive way: random entries came back at +0.073R and +13.8 sigma at a
    3:1 target. A bar registers a barrier when its EXTREME crosses it, and the
    overshoot is proportionally larger on the nearer barrier, so the harness
    manufactures a small edge out of nothing. Whatever it reads here is
    subtracted from every candidate in the same cell.
    """
    rng = np.random.default_rng(seed)
    out = rng.choice([-1, 0, 1], size=len(frame), p=[0.15, 0.70, 0.15])
    return out.astype(int)


@dataclass
class Cell:
    """One candidate on one clock over one asset class."""

    candidate: str
    clock: str
    asset_class: str
    train: Trades = field(default_factory=Trades)
    test: Trades = field(default_factory=Trades)
    control: Trades = field(default_factory=Trades)


def bonferroni_sigma(cells: int, target_p: float = 0.05) -> float:
    """The sigma a single cell must reach when `cells` of them were tried.

    Testing forty cells and keeping the best finds a 2-sigma result about 63%
    of the time on pure noise. This is not a formality -- it is the difference
    between a search and a story.
    """
    from math import erfc, sqrt

    if cells <= 0:
        return 2.0
    per_cell = target_p / cells
    low, high = 0.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2.0
        # two-sided tail
        if erfc(mid / sqrt(2.0)) > per_cell:
            low = mid
        else:
            high = mid
    return round(high, 2)


def verdict(cell: Cell, bar: float) -> tuple[bool, str]:
    """Does this cell clear every bar, and if not, which one stopped it."""
    _train_total, train_each, train_sigma, train_n = stats(cell.train)
    _test_total, test_each, test_sigma, test_n = stats(cell.test)
    _c_total, control_each, _c_sigma, control_n = stats(cell.control)

    if train_n < 150 or test_n < 100:
        return False, f"too few trades ({train_n} train / {test_n} holdout)"
    net_train = train_each - control_each
    net_test = test_each - control_each
    if net_train <= 0:
        return False, f"train {net_train:+.3f} R net of control"
    if train_sigma < bar:
        return False, f"train {train_sigma:+.2f} sigma, bar is {bar:.2f}"
    if net_test <= 0:
        return False, f"holdout {net_test:+.3f} R net of control — train only"
    if test_sigma < 2.0:
        return False, f"holdout {test_sigma:+.2f} sigma on its own"
    return True, (
        f"train {net_train:+.3f} R at {train_sigma:+.2f} sigma over {train_n}, "
        f"holdout {net_test:+.3f} R at {test_sigma:+.2f} over {test_n} "
        f"(control {control_each:+.3f} R over {control_n})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument(
        "--clocks",
        nargs="*",
        default=["M15", "M30", "H1"],
        help="timeframes to try each candidate on, space or comma separated",
    )
    parser.add_argument("--symbols", default="", help="comma list; default = the core universe")
    parser.add_argument("--stop-atr", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--csv", default="", help="write every cell's numbers here")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clocks = [
        piece.strip().upper()
        for chunk in args.clocks
        for piece in str(chunk).split(",")
        if piece.strip()
    ]

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=True),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        from scanner.universe import UniverseScanner
        from scripts.dry_run_sections import _core_universe

        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = _core_universe(connector, settings)

        end = datetime.now(UTC)
        start = end - timedelta(days=args.days)
        split = start + (end - start) * 0.6

        cells: dict[tuple[str, str, str], Cell] = {}
        print(
            f"\nSEARCHING {len(symbols)} markets x {len(clocks)} clocks "
            f"x {len(CANDIDATES)} candidates, {args.days} days"
        )
        print(f"train up to {split:%Y-%m-%d}, holdout after it\n")

        for position, symbol in enumerate(symbols, 1):
            try:
                spec = connector.spec(symbol)
                asset_class = UniverseScanner._path_class(spec.path).value
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                print(f"  [{position}/{len(symbols)}] {symbol}: no spec ({exc})")
                continue
            for clock_name in clocks:
                clock = Timeframe.parse(clock_name)
                try:
                    frame = fetch_mt5_history(
                        connector, symbol, clock, start - (WARMUP + 40) * clock.duration, end
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{position}/{len(symbols)}] {symbol} {clock_name}: {exc}")
                    continue
                if len(frame) < WARMUP + HORIZON + 200:
                    continue
                stop_price = args.stop_atr * float(np.nanmedian(_atr(frame)))
                cost_r = _cost_share(settings, asset_class, spec, stop_price)

                for name, detector in CANDIDATES.items():
                    key = (name, clock_name, asset_class)
                    cell = cells.setdefault(key, Cell(name, clock_name, asset_class))
                    signals = detector(frame)
                    found = resolve(
                        frame, signals, stop_atr=args.stop_atr, ratio=args.ratio, cost_r=cost_r
                    )
                    _split_into(found, split, cell)
                # ONE control per (clock, asset class), not per candidate: it
                # measures the HARNESS, and running twelve of them would only
                # measure the same thing twelve times with more noise.
                control_key = ("__control__", clock_name, asset_class)
                control = cells.setdefault(
                    control_key, Cell("__control__", clock_name, asset_class)
                )
                found = resolve(
                    frame,
                    random_control(frame, seed=hash((symbol, clock_name)) & 0xFFFF),
                    stop_atr=args.stop_atr,
                    ratio=args.ratio,
                    cost_r=cost_r,
                )
                _split_into(found, split, control)
            print(f"  [{position}/{len(symbols)}] {symbol} done")
    finally:
        connector.shutdown()

    _report(cells, args)


def _split_into(found: Trades, split: datetime, cell: Cell) -> None:
    for r, day, when in zip(found.r, found.day, found.when, strict=True):
        target = cell.train if when < split else cell.test
        target.r.append(r)
        target.day.append(day)
        target.when.append(when)


def _report(cells: dict, args) -> None:
    """What survived, and if nothing did, what stopped each one.

    A SEARCH THAT ONLY PRINTS ITS WINNERS IS A STORY. The near misses are the
    part that says whether the grid was worth running: forty cells all stopped
    at "too few trades" means the window was short, and forty stopped at
    "holdout, train only" means the search is fitting noise and no amount of
    further tuning will help.
    """
    controls = {
        (cell.clock, cell.asset_class): cell
        for cell in cells.values()
        if cell.candidate == "__control__"
    }
    real = [cell for cell in cells.values() if cell.candidate != "__control__"]
    tested = [c for c in real if len(c.train) >= 150 and len(c.test) >= 100]
    bar = bonferroni_sigma(max(len(tested), 1))

    print("\n" + "=" * 78)
    print("SEARCH RESULT")
    print("=" * 78)
    print(f"  {len(real)} cells built, {len(tested)} had enough trades to judge")
    print(f"  Bonferroni bar at {len(tested)} live cells: {bar:.2f} sigma on train")
    print("     (2.0 would be the bar for ONE hypothesis. Keeping the best of")
    print("      many finds a 2-sigma result on pure noise most of the time.)")

    print("\n  THE HARNESS'S OWN BIAS — random entries on the same bars")
    for (clock, asset_class), cell in sorted(controls.items()):
        _t, each, sigma, n = stats(cell.train)
        if n:
            print(
                f"    {asset_class:<10} {clock:<4} {each:+.4f} R over {n:>6} random trades"
                f"   ({sigma:+.2f} sigma)"
            )
    print("    Subtracted from every candidate in its own cell.")

    for cell in real:
        control = controls.get((cell.clock, cell.asset_class))
        if control is not None:
            cell.control = control.train

    passed = [(c, verdict(c, bar)) for c in real]
    winners = [(c, why) for c, (ok, why) in passed if ok]

    if winners:
        print(f"\n  SURVIVED EVERY BAR — {len(winners)} of {len(tested)}")
        for cell, why in winners:
            print(f"\n    {cell.candidate}  {cell.clock}  {cell.asset_class}")
            print(f"      {why}")
        print("\n  These are candidates for section four. Nothing is live until")
        print("  it is built into a module and measured again by history.cmd.")
    else:
        print(f"\n  NOTHING SURVIVED. {len(tested)} cells were judged and none cleared")
        print("  the bar. That is the expected outcome of an honest search and it")
        print("  is not a failure of the run.")

    print("\n  CLOSEST MISSES — what stopped each of the ten best")
    ranked = sorted(
        (c for c in real if len(c.train) >= 150),
        key=lambda c: -stats(c.train)[2],
    )[:10]
    for cell in ranked:
        _t, each, sigma, n = stats(cell.train)
        _ct, control_each, _cs, _cn = stats(cell.control)
        _ok, why = verdict(cell, bar)
        print(
            f"    {cell.candidate:<24} {cell.clock:<4} {cell.asset_class:<10} "
            f"{each - control_each:+.3f} R  {sigma:+5.2f} sigma  n={n:<6} {why}"
        )

    if args.csv:
        import csv as csv_module

        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(
                [
                    "candidate",
                    "clock",
                    "asset_class",
                    "train_n",
                    "train_r_per_trade",
                    "train_sigma",
                    "holdout_n",
                    "holdout_r_per_trade",
                    "holdout_sigma",
                    "control_r_per_trade",
                    "verdict",
                ]
            )
            for cell in real:
                _t1, train_each, train_sigma, train_n = stats(cell.train)
                _t2, test_each, test_sigma, test_n = stats(cell.test)
                _t3, control_each, _s3, _n3 = stats(cell.control)
                ok, why = verdict(cell, bar)
                writer.writerow(
                    [
                        cell.candidate,
                        cell.clock,
                        cell.asset_class,
                        train_n,
                        round(train_each, 4),
                        round(train_sigma, 2),
                        test_n,
                        round(test_each, 4),
                        round(test_sigma, 2),
                        round(control_each, 4),
                        "PASS" if ok else why,
                    ]
                )
        print(f"\n  every cell written to {path}")


if __name__ == "__main__":
    main()
