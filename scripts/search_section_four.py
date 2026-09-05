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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# THE CANDIDATES NOW LIVE IN `analysis/mechanisms.py`, and that move is the
# point rather than tidying. A mechanism this script measures and a live
# section re-implements is two implementations of one definition, and the
# measured numbers cannot tell you which one the account runs. Section eleven
# is built on top of this registry, so the candidate that was searched IS the
# candidate that trades.
# --------------------------------------------------------------------------
from analysis.mechanisms import (
    CANDIDATES,
    FAMILIES,
    GOLD_CANDIDATES,
    HORIZON,
    INDEX_CANDIDATES,
    WARMUP,
    _atr,
)
from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from risk.position_sizer import PositionSizer

__all_mechanisms__ = (CANDIDATES, FAMILIES, GOLD_CANDIDATES, INDEX_CANDIDATES, WARMUP, HORIZON)


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
    horizon: int = HORIZON,
) -> Trades:
    """First touch of stop or target, entry at the signal bar's close.

    THE RULES ARE THE RESEARCH'S, and the two that matter both cost the
    candidate rather than flatter it: resolution starts on the bar AFTER the
    entry bar, and a bar that spans both barriers is a LOSS because the order
    inside it is unknowable.

    ONE POSITION AT A TIME, WHICH THIS DID NOT DO, and it is not a detail --
    it biased the search against exactly the mechanisms it was extended to
    find.

    A signal was taken on EVERY bar that carried one. A breakout candidate
    fires once and does not care. A mean-reverter stays on for as long as
    price is stretched, so one event was entered on ten consecutive bars and
    counted as ten trades: the first at the edge of the move and the other
    nine progressively deeper into it, each worse than the last. The first
    gold run showed the shape plainly -- `stretch_fade` on M5 reported
    150,055 trades where `opening_range_break`, which fires once per
    direction per day by construction, reported 1,796 and carried the largest
    per-trade edge in the grid.

    So the fade candidates were not measured on their mechanism. They were
    measured on their mechanism diluted nine parts to one with late entries
    the account would never take, because the account holds one position per
    symbol and this is what that costs.

    The account's constraint is now the harness's: while a trade is open the
    symbol is busy, and a trade that reaches neither barrier keeps it busy for
    the whole horizon rather than quietly freeing it.
    """
    out = Trades()
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    index = frame.index

    free_at = WARMUP
    for i in range(WARMUP, len(frame) - 1):
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
        # UNRESOLVED IS STILL OCCUPIED. A trade that reached neither barrier
        # is excluded from the statistics -- it has not answered the question
        # -- but the account was in it the whole time and could not take the
        # next signal. Freeing the symbol here would let a candidate skip its
        # own dead trades and pick up the following setup for free.
        free_at = exit_at
        if result is None:
            continue
        out.r.append(result)
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


def _cost_share(sizer, spec, stop_price: float) -> float:
    """What a round trip costs as a fraction of the stop.

    DELEGATED TO THE SIZER, and the first version of this function is why.

    I reimplemented it here as

        pip = spec.point * 10.0
        commission_price = (per_side * 2.0) * pip / 10.0

    which is dimensionally nonsense: commission is account currency per lot,
    and multiplying it by a tenth of a pip does not convert it to price. It
    needs the instrument's pip VALUE, which depends on contract size and
    quote currency -- exactly what `spec.money_per_lot` and
    `spec.pips_to_price` already know.

    The result showed up as gold reading a 62% cost share on H1 against 0.2%
    on M30. Same instrument, same formula, and cost must FALL on a slower
    clock because the stop is wider. A number that moves 300x the wrong way
    is not a property of gold.

    `PositionSizer._cost_share` is the definition the account charges, its own
    docstring warns that "two definitions of the same cost would eventually
    disagree", and I wrote the second one anyway.
    """
    commission = sizer.settings.risk.commission_per_lot(spec.asset_class.value)
    return sizer._cost_share(spec, stop_price, commission)


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
    parser.add_argument(
        "--asset-class",
        default="",
        help="every market the scanner puts in this class, e.g. metal. Ignored "
        "when --symbols is given.",
    )
    parser.add_argument("--stop-atr", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument(
        "--family",
        choices=sorted(FAMILIES),
        default="index",
        help="which candidate grid to run: index (the original twelve), "
        "gold (the sixteen intraday metal mechanisms), or all",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help="bars a trade is given to reach a barrier before it is discarded",
    )
    parser.add_argument(
        "--cells-already-tried",
        type=int,
        default=0,
        help=(
            "cells searched in EARLIER runs of this script, added to this run's "
            "count before the Bonferroni bar is computed. Two searches of forty "
            "cells are eighty hypotheses, and paying for forty twice is how a "
            "search launders itself into a discovery."
        ),
    )
    parser.add_argument("--csv", default="", help="write every cell's numbers here")
    parser.add_argument(
        "--database",
        default="",
        help="read the one-file SQLite research archive instead of contacting MT5",
    )
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
    dataset = None
    connector = None
    if args.database:
        from backtesting.research_dataset import ResearchDataset

        dataset = ResearchDataset(Path(args.database), read_only=True)
    else:
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
        elif args.asset_class:
            # THE SCANNER'S OWN CLASSIFIER, not a substring match on the name.
            # Eightcap's metals are not all called XAU-something and the
            # spelling carries suffixes; asking the classifier means the
            # search finds whatever this broker actually lists rather than
            # whatever I guessed it was called. An empty result is reported
            # rather than run, because zero symbols and zero setups print the
            # same way and this repo has confused the two before.
            assert connector is not None
            wanted = args.asset_class.strip().lower()
            symbols = []
            for item in connector.symbols():
                if settings.instruments.is_ignored(item.name):
                    continue
                try:
                    found = UniverseScanner._path_class(connector.spec(item.name).path).value
                except Exception:  # noqa: BLE001 - a bad symbol is not a reason to stop
                    continue
                if found.lower() == wanted:
                    symbols.append(item.name)
            if not symbols:
                classes = sorted(
                    {
                        UniverseScanner._path_class(connector.spec(i.name).path).value
                        for i in connector.symbols()[:400]
                    }
                )
                raise SystemExit(
                    f"no symbols in asset class {wanted!r}. "
                    f"This broker's catalogue has: {', '.join(classes)}"
                )
        elif dataset is not None:
            symbols = dataset.symbols()
        else:
            assert connector is not None
            symbols = _core_universe(connector, settings)

        stored_window = dataset.window() if dataset is not None else None
        end = (
            datetime.fromisoformat(stored_window[1])
            if stored_window is not None and stored_window[1]
            else datetime.now(UTC)
        )
        start = end - timedelta(days=args.days)
        split = start + (end - start) * 0.6

        sizer = PositionSizer(settings)
        cells: dict[tuple[str, str, str], Cell] = {}
        #: (clock, asset class) -> cost share, printed with the result. It is
        #: the number the whole search turns on and it was invisible.
        costs: dict[tuple[str, str], float] = {}
        grid = FAMILIES[args.family]
        print(
            f"\nSEARCHING {len(symbols)} markets x {len(clocks)} clocks "
            f"x {len(grid)} candidates ({args.family} family), {args.days} days"
        )
        print(f"train up to {split:%Y-%m-%d}, holdout after it")
        print(f"horizon {args.horizon} bars, stop {args.stop_atr} ATR, target {args.ratio}:1\n")

        for position, symbol in enumerate(symbols, 1):
            try:
                spec = dataset.spec(symbol) if dataset is not None else connector.spec(symbol)
                asset_class = (
                    spec.asset_class.value
                    if dataset is not None
                    else UniverseScanner._path_class(spec.path).value
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                print(f"  [{position}/{len(symbols)}] {symbol}: no spec ({exc})")
                continue
            for clock_name in clocks:
                clock = Timeframe.parse(clock_name)
                try:
                    requested_start = start - (WARMUP + 40) * clock.duration
                    frame = (
                        dataset.frame(symbol, clock, requested_start, end)
                        if dataset is not None
                        else fetch_mt5_history(connector, symbol, clock, requested_start, end)
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{position}/{len(symbols)}] {symbol} {clock_name}: {exc}")
                    continue
                if len(frame) < WARMUP + HORIZON + 200:
                    continue
                stop_price = args.stop_atr * float(np.nanmedian(_atr(frame)))
                cost_r = _cost_share(sizer, spec, stop_price)
                costs[(clock_name, asset_class)] = cost_r

                for name, detector in grid.items():
                    key = (name, clock_name, asset_class)
                    cell = cells.setdefault(key, Cell(name, clock_name, asset_class))
                    signals = detector(frame)
                    found = resolve(
                        frame,
                        signals,
                        stop_atr=args.stop_atr,
                        ratio=args.ratio,
                        cost_r=cost_r,
                        horizon=args.horizon,
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
                    horizon=args.horizon,
                )
                _split_into(found, split, control)
            print(f"  [{position}/{len(symbols)}] {symbol} done")
    finally:
        if dataset is not None:
            dataset.close()
        elif connector is not None:
            connector.shutdown()

    _report(cells, args, costs)


def _split_into(found: Trades, split: datetime, cell: Cell) -> None:
    for r, day, when in zip(found.r, found.day, found.when, strict=True):
        target = cell.train if when < split else cell.test
        target.r.append(r)
        target.day.append(day)
        target.when.append(when)


#: Mechanism halves that are exact negations of each other. Folded together
#: before the cross-clock check, or one mechanism reads as two discoveries.
_OPPOSITES: tuple[tuple[str, str], ...] = (
    ("stretch_continuation", "stretch_fade"),
    ("quiet_stretch_continuation", "quiet_stretch_fade"),
    ("london_drive", "london_fade"),
    ("comex_drive", "comex_fade"),
    ("pm_fix_drive", "pm_fix_fade"),
    ("round_number_break", "round_number_fade"),
    ("opening_range_break", "opening_range_fade"),
    ("day_range_exhaustion_break", "day_range_exhaustion_fade"),
    ("gap_continuation", "gap_fade"),
    ("streak_continuation", "streak_reversal"),
    ("close_position_in_range", "close_position_fade"),
    ("prior_day_break", "prior_day_fade"),
)


def _disagreements(real: list[Cell], controls: dict) -> None:
    """One mechanism, two clocks, opposite answers — the signature of noise.

    WHY THIS IS WORTH PRINTING RATHER THAN LEAVING TO WHOEVER READS THE
    TABLE. The gold run's best cell was `london_fade` on M15 at +0.060 R
    against the coin flip, which reads as "the London open reverses on
    metals". Four rows below it sat `london_drive` on M5 at +0.029, which
    says the same open CONTINUES. The same event, the same 360 days, the
    same thirteen markets, and the only difference is how the bars were
    sliced -- so at most one of them can be describing gold, and the other
    is describing the sample.

    Both were among the ten best, neither cleared the bar, and nothing in
    the output connected them. A reader who wanted `london_fade` to be real
    would have had no reason to notice.

    A mechanism whose sign flips between neighbouring clocks is not a
    near miss to tune. It is the clearest evidence the grid produces that
    there is nothing underneath.
    """
    edge: dict[tuple[str, str], float] = {}
    for cell in real:
        if len(cell.train) < 150:
            continue
        control = controls.get((cell.clock, cell.asset_class))
        control_each = stats(control.train)[1] if control is not None else 0.0
        edge[(cell.candidate, cell.clock)] = stats(cell.train)[1] - control_each

    clocks_for: dict[str, list[tuple[str, float]]] = {}
    for (name, clock), value in edge.items():
        clocks_for.setdefault(name, []).append((clock, value))

    # A MECHANISM AND ITS OPPOSITE ARE ONE MECHANISM. The first version of
    # this compared each name against itself across clocks and missed the
    # clearest case in the M30/H1 run: `opening_range_break` was the best cell
    # on M15 at +0.053 against the coin flip, and `opening_range_fade` -- the
    # exact negation of it -- was the best cell on M30 at +0.105. Break wins
    # on one clock, fade wins on another, and both looked like discoveries
    # because they were filed under different names.
    #
    # So the pairs are folded together and the fade half is read with its
    # sign flipped, which puts both halves of one mechanism on one line.
    for one, other in _OPPOSITES:
        if one not in clocks_for and other not in clocks_for:
            continue
        merged = clocks_for.pop(one, []) + [(c, -v) for c, v in clocks_for.pop(other, [])]
        if not merged:
            continue
        # BOTH HALVES LAND ON THE SAME CLOCK and, being negations, say the
        # same thing twice. Averaging them per clock keeps one number per
        # clock -- printing both would show a mechanism agreeing with itself
        # and read as corroboration.
        per_clock: dict[str, list[float]] = {}
        for clock, value in merged:
            per_clock.setdefault(clock, []).append(value)
        clocks_for[f"{one} (fade telt negatief)"] = [
            (clock, sum(values) / len(values)) for clock, values in per_clock.items()
        ]

    split = [
        (name, sorted(rows))
        for name, rows in sorted(clocks_for.items())
        if len(rows) > 1 and min(v for _c, v in rows) < 0 < max(v for _c, v in rows)
    ]
    if not split:
        return
    print(f"\n  DISAGREES WITH ITSELF ACROSS CLOCKS — {len(split)} mechanisms")
    print("    Beating the coin flip on one clock and losing to it on another,")
    print("    on the same event over the same days. At most one can be gold.")
    for name, rows in split:
        detail = ",  ".join(f"{clock} {value:+.3f}" for clock, value in rows)
        print(f"    {name:<34}{detail}")


def _report(cells: dict, args, costs: dict | None = None) -> None:
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
    earlier = max(int(getattr(args, "cells_already_tried", 0) or 0), 0)
    counted = max(len(tested), 1) + earlier
    bar = bonferroni_sigma(counted)

    print("\n" + "=" * 78)
    print("SEARCH RESULT")
    print("=" * 78)
    print(f"  {len(real)} cells built, {len(tested)} had enough trades to judge")
    if earlier:
        print(f"  plus {earlier} cells declared from earlier searches of this project")
    print(f"  Bonferroni bar at {counted} hypotheses: {bar:.2f} sigma on train")
    print("     (2.0 would be the bar for ONE hypothesis. Keeping the best of")
    print("      many finds a 2-sigma result on pure noise most of the time.)")
    print("     The holdout bar stays 2.0, and it has to be cleared on the")
    print("     holdout's own trades -- that half was never searched over.")

    if costs:
        print("\n  WHAT A ROUND TRIP COSTS, as a share of the stop")
        for (clock, asset_class), share in sorted(costs.items(), key=lambda kv: -kv[1]):
            flag = "  <- SUSPECT" if share > 0.5 else ""
            print(f"    {asset_class:<10} {clock:<4} {share:>7.1%}{flag}")
        print("    Above ~25% nothing pays, whatever the entry does. A share")
        print("    that RISES on a slower clock is a bug, not a market.")

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

    # A CANDIDATE THAT NEVER FIRED IS NOT A CANDIDATE THAT FAILED, and the
    # two must not look the same. Silence means the threshold is wrong for
    # this feed, which is a fixable mistake of mine; failure means the
    # mechanism does not pay, which is an answer. A first version of this
    # report showed neither, and six of twelve candidates producing zero
    # trades would have been invisible.
    silent = sorted({c.candidate for c in real if len(c.train) + len(c.test) == 0})
    thin = sorted({c.candidate for c in real if 0 < len(c.train) + len(c.test) < 250} - set(silent))
    if silent:
        print(f"\n  NEVER FIRED — {len(silent)} candidates produced no trades at all")
        print(f"    {', '.join(silent)}")
        print("    That is a threshold that does not match this feed, not a result.")
    if thin:
        print(f"\n  TOO THIN TO JUDGE — fired, but under 250 trades: {', '.join(thin)}")

    _disagreements(real, controls)

    # TWO DIFFERENT QUANTITIES SAT IN ONE ROW WITH NOTHING SAYING SO. The R
    # column was NET of the random control and the sigma column was on the
    # RAW daily totals, so the first gold run printed rows like "+0.049 R,
    # +0.81 sigma" and they read as one measurement disagreeing with itself.
    # They are two questions -- "does it beat a coin flip" and "is it
    # distinguishable from zero" -- and both have to be answered, so both are
    # now named and the raw per-trade number is printed beside them.
    print("\n  CLOSEST MISSES — what stopped each of the ten best")
    print(
        f"    {'candidate':<26}{'clock':<6}{'class':<9}"
        f"{'raw R':>8}{'vs coin':>9}{'sigma':>8}  n"
    )
    print(
        "      raw R    = per trade, cost charged, exactly as measured\n"
        "      vs coin  = the same minus the random control on those bars\n"
        "      sigma    = raw R over the spread of DAILY totals, not per trade"
    )
    ranked = sorted(
        (c for c in real if len(c.train) >= 150),
        key=lambda c: -stats(c.train)[2],
    )[:10]
    for cell in ranked:
        _t, each, sigma, n = stats(cell.train)
        _ct, control_each, _cs, _cn = stats(cell.control)
        print(
            f"    {cell.candidate:<26}{cell.clock:<6}{cell.asset_class:<9}"
            f"{each:>+8.3f}{each - control_each:>+9.3f}{sigma:>+8.2f}  {n}"
        )
        _ok, why = verdict(cell, bar)
        print(f"      {why}")

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
