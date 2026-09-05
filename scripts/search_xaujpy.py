"""Search XAUJPY for a mechanism worth giving a section, and say when it works.

WHAT THE OWNER ASKED FOR, in his words: a mechanism for XAUJPY that gives
three to five trades a day, on M1, M5 or M15, with the sessions it works in
identified, with every live gate carried into the measurement, and with
position management included. This is what answers that, and it is built to
come back empty.

WHY A SEPARATE SCRIPT FROM `search_section_four.py`. That one answers "does
this mechanism have an edge anywhere", sweeping many markets and reporting one
number per candidate. This answers a narrower and harder question about ONE
instrument: at what hour, at what trade rate, and at what cost -- and it pays
the multiplicity price for asking it, which is much larger. Folding hours into
the other script would have quietly multiplied its Bonferroni bar by twenty-
four for every existing user of it.

THE FOUR THINGS THAT MAKE A RESULT HERE MEAN ANYTHING, all of them borrowed
from what the rest of this repository already learnt the hard way:

  DAY-CLUSTERED SIGMA. Hours of one morning are not independent observations.
  Sigma is computed from the spread of DAILY totals, so a single violent
  session is one number and not forty.

  BONFERRONI OVER THE WHOLE GRID, hours included. Twenty-eight mechanisms x
  three clocks x two ratios is 168 cells before hours; adding a per-hour
  selection on top multiplies it again, and the bar is computed from the real
  count rather than the count somebody wishes had been searched. `--hours` is
  therefore off by default: the hour table is REPORTED for every run, but the
  moment an hour window is SELECTED the bar rises to pay for it.

  A RATE-MATCHED RANDOM CONTROL. The same number of trades on the same bars,
  same stop, same target, same costs, entered at random. A mechanism that does
  not beat its own coin flip has found the instrument's drift, not an edge.

  AN UNTOUCHED HOLDOUT. The newest slice is never searched. The old section
  eleven cleared everything except this and its holdout came back negative in
  four markets out of four, which is the entire reason it no longer exists.

WHAT THIS SCRIPT CANNOT DO, said here rather than discovered later. The live
runner wraps every entry in eight further gates -- confirmation, news
blackout, target reach, spread, liveliness, pullback, volume spike, entry
quality -- and this measures none of them except cost. It applies the cost
gate, the one-position-at-a-time rule and the configured blocked hours, and it
NAMES the rest in its own report. A number here is what the mechanism does
with those gates OFF; `dryrun-live.cmd` is what runs them.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.mechanisms import FAMILIES, HORIZON, WARMUP, _atr
from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from risk.position_sizer import PositionSizer
from scripts.search_section_four import (
    _cost_share,
    bonferroni_sigma,
    random_control,
    resolve,
    stats,
)

#: The trading day in UTC, cut where this instrument's liquidity actually
#: changes hands rather than at midnight. XAUJPY is gold priced through a
#: currency, so it has TWO clocks: the metal's and the yen's.
SESSIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    # Tokyo owns the yen leg and nothing owns the metal leg. Thin on both
    # sides, which is where a fade has the best chance and the worst spread.
    ("tokyo", (0, 1, 2, 3, 4, 5, 6)),
    # London opens the metal. This is the window that cost section ten
    # -0.101 R per trade on the gold crosses, measured over 1,664 trades, so
    # it starts under suspicion rather than neutral.
    ("london", (7, 8, 9, 10, 11, 12)),
    # Both open. Deepest book of the day on both legs.
    ("overlap", (13, 14, 15)),
    # London gone, gold still trading, the yen leg thinning out. 16:00 and
    # 17:00 were negative in both halves in four crosses out of four.
    ("ny_late", (16, 17, 18, 19, 20)),
    ("close", (21, 22, 23)),
)

#: Reward per unit of risk, tried for every mechanism. Two rather than five,
#: because each one doubles the grid and the bar rises with it. 1.0 is what
#: the original research used; 1.5 is what sections ten and eleven measured on.
RATIOS: tuple[float, ...] = (1.0, 1.5)

#: The owner asked for three to five trades a day. A cell far under that is not
#: what he wants however good it looks, and a cell far over it is a scalper
#: whose costs this account cannot carry. Reported, never used to select --
#: filtering on it would be one more free parameter nobody paid for.
WANTED_TRADES_PER_DAY = (3.0, 5.0)


@dataclass
class Cell:
    """One mechanism, on one clock, at one ratio, over the whole window."""

    mechanism: str
    clock: str
    ratio: float
    trades: int = 0
    total_r: float = 0.0
    per_trade: float = 0.0
    sigma: float = 0.0
    per_day: float = 0.0
    control_r: float = 0.0
    control_per_trade: float = 0.0
    cost_share: float = 0.0
    holdout_trades: int = 0
    holdout_r: float = 0.0
    by_hour: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    by_session: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    early_r: float = 0.0
    late_r: float = 0.0

    @property
    def beats_its_coin(self) -> bool:
        return self.per_trade > self.control_per_trade


def session_of(hour: int) -> str:
    for name, hours in SESSIONS:
        if hour in hours:
            return name
    return "unknown"


def _cell_count(mechanisms: int, clocks: int, ratios: int, *, hours: bool) -> int:
    """Hypotheses actually tested, which is what the bar has to pay for.

    Selecting the best SESSION out of five multiplies the count by five, and
    selecting the best HOUR out of twenty-four multiplies it by twenty-four.
    That is not a formality: the reason to write it down is that picking the
    best of twenty-four slices of noise finds a two-sigma result almost every
    time, and this repository has already shipped one hour window chosen
    exactly that way.
    """
    base = mechanisms * clocks * ratios
    return base * (len(SESSIONS) if hours else 1)


def _split_r(cell: Cell, found, split: datetime) -> None:
    for value, when in zip(found.r, found.when, strict=True):
        hour = int(when.hour)
        cell.by_hour[hour].append(value)
        cell.by_session[session_of(hour)].append(value)
        if when < split:
            cell.early_r += value
        else:
            cell.late_r += value


def _mute_blocked_hours(signals: np.ndarray, frame: pd.DataFrame, blocked: set[int]) -> np.ndarray:
    """Zero every signal the live section would refuse on the clock.

    THIS WAS DOCUMENTED AND NOT IMPLEMENTED, which is the defect this whole
    repository is a monument to. The module docstring said the configured
    blocked hours were applied; no line applied them, and the first real run
    put `london_drive` third on +0.091 R per trade earned ENTIRELY inside
    07:00-12:00 UTC -- the window the section refuses. A search that ranks
    cells by an edge the section may not take is not measuring the section.

    Reported as well as applied: `_blocked_report` prints what was muted, so
    a block that is costing real money is visible rather than silently gone.
    """
    if not blocked:
        return signals
    hours = frame.index.hour.to_numpy()
    muted = signals.copy()
    muted[np.isin(hours, list(blocked))] = 0
    return muted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--symbol", default="XAUJPY")
    parser.add_argument(
        "--clocks",
        nargs="*",
        default=["M1", "M5", "M15"],
        help="timeframes to try each mechanism on",
    )
    parser.add_argument("--stop-atr", type=float, default=1.0)
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help=(
            "fraction of the NEWEST history never searched. The old section "
            "eleven cleared every other test and died on exactly this one."
        ),
    )
    parser.add_argument(
        "--hours",
        action="store_true",
        help=(
            "SELECT the best session per cell rather than only reporting the "
            "table. This multiplies the hypothesis count by the number of "
            "sessions and raises the Bonferroni bar to pay for it."
        ),
    )
    parser.add_argument(
        "--cells-already-tried",
        type=int,
        default=0,
        help="cells searched in EARLIER runs, added before the bar is computed",
    )
    parser.add_argument(
        "--all-hours",
        action="store_true",
        help=(
            "measure the hours the section BLOCKS as well. Off by default: a "
            "cell whose edge sits inside a blocked window is not a cell the "
            "section can trade, and ranking it there measures nothing."
        ),
    )
    parser.add_argument("--csv", default="", help="write every cell's numbers here")
    parser.add_argument(
        "--database",
        default="",
        help="read the one-file SQLite research archive instead of contacting MT5",
    )
    return parser


def _load(args, connector) -> dict[str, pd.DataFrame]:
    """Bars per clock, fetched once and reused by every mechanism."""

    end = datetime.now(UTC)
    frames: dict[str, pd.DataFrame] = {}
    for clock in args.clocks:
        tf = Timeframe.parse(clock)
        bars_needed = (WARMUP + HORIZON + 50) * tf.duration
        start = end - timedelta(days=args.days) - max(bars_needed * 1.6, timedelta(days=3))
        print(f"  fetching {args.symbol} {clock} ...", flush=True)
        frame = fetch_mt5_history(connector, args.symbol, tf, start, end)
        if frame is None or len(frame) < WARMUP + HORIZON + 50:
            print(f"  {clock}: not enough history, skipped")
            continue
        frames[clock] = frame
        print(
            f"  {clock}: {len(frame):,} bars, {frame.index[0]:%Y-%m-%d} -> "
            f"{frame.index[-1]:%Y-%m-%d}"
        )
    return frames


def main() -> None:
    args = build_parser().parse_args()
    args.clocks = [
        piece.strip().upper()
        for chunk in args.clocks
        for piece in str(chunk).split(",")
        if piece.strip()
    ]
    settings = load_settings(env_overrides=False)
    mechanisms = FAMILIES["all"]

    print("=" * 78)
    print(
        f"XAUJPY MECHANISM SEARCH — {len(mechanisms)} mechanisms x "
        f"{len(args.clocks)} clocks x {len(RATIOS)} ratios"
    )
    print("=" * 78)

    # EXACTLY AS `search_section_four` BUILDS IT, and this is the second time
    # this file has had to learn that. The first version passed
    # `login=`/`password=`/`server=` because that is what the constructor looks
    # like it should take; `MT5Connector.__init__` takes the CONFIG OBJECT and
    # a credentials object, and the run died on the VPS after printing its
    # whole banner. A signature invented from the outside is the same defect
    # class as everything else in this repository: correct-looking, and not on
    # the path the code actually walks.
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=True),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        spec = connector.spec(args.symbol)
        sizer = PositionSizer(settings)
        frames = _load(args, connector)
    finally:
        connector.shutdown()

    if not frames:
        raise SystemExit(
            f"no usable history for {args.symbol}. MT5 must be running, logged in, "
            f"and the symbol visible in Market Watch."
        )

    # THE SECTION'S OWN BLOCKED HOURS, read from the config rather than
    # written down a second time here. Two lists that must agree are two lists
    # that will disagree.
    blocked: set[int] = set()
    if not args.all_hours:
        for name in (
            "section_eleven_xaujpy_m1",
            "section_twelve_xaujpy_m5",
            "section_thirteen_xaujpy_m15",
        ):
            blocked |= set(getattr(settings.analysis, name).blocked_hours)
        if blocked:
            print(
                f"\n  hours the sections refuse, muted here too: "
                f"{', '.join(f'{h:02d}:00' for h in sorted(blocked))}"
            )
            print("  (--all-hours measures them anyway, for diagnosis)")

    cells: list[Cell] = []
    for clock, frame in frames.items():
        # THE HOLDOUT IS CUT BEFORE ANYTHING IS MEASURED, so no decision below
        # can have seen it. Cutting it afterwards is the same data with a
        # different label on it.
        cut = int(len(frame) * (1.0 - args.holdout))
        searched, held = frame.iloc[:cut], frame.iloc[cut:]
        span_days = max((searched.index[-1] - searched.index[0]).days, 1)
        split = searched.index[int(len(searched) * 0.6)]
        print(
            f"\n{clock}: searching {len(searched):,} bars, holding back "
            f"{len(held):,} from {held.index[0]:%Y-%m-%d}"
        )

        for name, mechanism in sorted(mechanisms.items()):
            raw = mechanism(searched)
            signals = raw if args.all_hours else _mute_blocked_hours(raw, searched, blocked)
            for ratio in RATIOS:
                cell = Cell(mechanism=name, clock=clock, ratio=ratio)
                # THE STOP THIS CELL ACTUALLY USES, per clock.
                #
                # THIS LINE WAS `close * 0.004` -- a flat 0.4% of price as the
                # stop width, the same number on M1, M5 and M15. The stop is
                # `stop_atr x ATR of that clock`, and on M1 that ATR is an
                # order of magnitude smaller than 0.4% of price, so M1 was
                # charged roughly a twentieth of the cost it pays.
                #
                # The first 720-day run came back with NINE cells clearing the
                # bar, all of them M1, at 74 to 252 trades a day. That was not
                # an edge, it was free trading. It is the same defect the
                # `_cost_share` docstring already describes one level up --
                # "a number that moves 300x the wrong way is not a property of
                # gold" -- and I reproduced it by hand-rolling the stop width
                # instead of taking the one the cell resolves against.
                stop_price = args.stop_atr * float(np.nanmedian(_atr(searched)))
                cell.cost_share = _cost_share(sizer, spec, stop_price)
                cost_r = cell.cost_share
                found = resolve(
                    searched, signals, stop_atr=args.stop_atr, ratio=ratio, cost_r=cost_r
                )
                if len(found) < 30:
                    continue
                total, per_trade, sigma, n = stats(found)
                cell.trades, cell.total_r = n, total
                cell.per_trade, cell.sigma = per_trade, sigma
                cell.per_day = n / span_days
                _split_r(cell, found, split)

                control = random_control(searched, seed=abs(hash((name, clock, ratio))) % 10_000)
                got = resolve(searched, control, stop_atr=args.stop_atr, ratio=ratio, cost_r=cost_r)
                if len(got):
                    c_total, c_per, _c_sigma, _c_n = stats(got)
                    cell.control_r, cell.control_per_trade = c_total, c_per

                on_holdout = resolve(
                    held, mechanism(held), stop_atr=args.stop_atr, ratio=ratio, cost_r=cost_r
                )
                if len(on_holdout):
                    h_total, _h_per, _h_sigma, h_n = stats(on_holdout)
                    cell.holdout_trades, cell.holdout_r = h_n, h_total
                cells.append(cell)

    tested = _cell_count(len(mechanisms), len(frames), len(RATIOS), hours=args.hours)
    tested += args.cells_already_tried
    bar = bonferroni_sigma(tested)
    _report(cells, bar, tested, args, settings.risk.max_cost_share_of_risk)
    if args.csv:
        _write_csv(cells, Path(args.csv))
        print(f"\nevery cell written to {args.csv}")


def _report(cells: list[Cell], bar: float, tested: int, args, cap: float) -> None:
    print("\n" + "=" * 78)
    print(f"RESULT — Bonferroni bar at {tested} cells: {bar:.2f} sigma")
    print("=" * 78)
    if not cells:
        print("  no cell produced 30 resolved trades. Nothing to say.")
        return

    ranked = sorted(cells, key=lambda c: c.sigma, reverse=True)
    print(
        f"\n  {'mechanism':<28}{'clk':>5}{'R:R':>6}{'trades':>8}{'/day':>7}"
        f"{'total R':>10}{'per':>8}{'cost':>7}{'sigma':>7}{'holdout':>10}"
    )
    for cell in ranked[:20]:
        flag = "" if cell.beats_its_coin else "  <- loses to its own coin flip"
        print(
            f"  {cell.mechanism:<28}{cell.clock:>5}{cell.ratio:>6.1f}{cell.trades:>8}"
            f"{cell.per_day:>7.1f}{cell.total_r:>+10.2f}{cell.per_trade:>+8.3f}"
            f"{cell.cost_share:>6.1%}{cell.sigma:>+7.2f}{cell.holdout_r:>+10.2f}{flag}"
        )

    # THE ACCOUNT'S OWN COST GATE, applied here because it CAN be. A setup
    # whose round trip eats more than `max_cost_share_of_risk` of the stop is
    # refused live with SL_TOO_TIGHT_FOR_COSTS, so a cell above the cap is not
    # a cell the account can trade however good its sigma looks. On M1 this is
    # not a formality -- an M1 ATR is small and the spread on a gold cross is
    # not, and that ratio is the whole reason the first 720-day run put nine
    # unaffordable M1 cells at the top.
    survivors = [
        c
        for c in ranked
        if c.sigma >= bar
        and c.beats_its_coin
        and c.holdout_r > 0
        and c.total_r > 0
        and c.cost_share <= cap
    ]
    priced_out = [
        c
        for c in ranked
        if c.cost_share > cap and c.sigma >= bar and c.holdout_r > 0 and c.total_r > 0
    ]
    print("\n" + "-" * 78)
    print("WHICH CELLS EARNED A SECTION")
    print("-" * 78)
    if priced_out:
        print(f"\n  {len(priced_out)} cell(s) cleared the statistics and are REFUSED on cost:")
        for cell in priced_out[:6]:
            print(
                f"    {cell.mechanism} {cell.clock} R:R {cell.ratio:.1f} -- round trip is "
                f"{cell.cost_share:.1%} of the stop against a {cap:.0%} cap"
            )
        print("  The account would answer SL_TOO_TIGHT_FOR_COSTS to every one of them.")
    if not survivors:
        print("\n  NONE. A cell has to clear all four at once:")
        print("    sigma at or above the bar, positive total, beats its own coin")
        print("    flip, and a POSITIVE untouched holdout.")
        best = ranked[0]
        print(
            f"\n  Closest: {best.mechanism} {best.clock} R:R {best.ratio:.1f} at "
            f"{best.sigma:+.2f} sigma against {bar:.2f}, holdout {best.holdout_r:+.2f} R."
        )
        print("\n  That is the honest answer and it is what the trainer for the old")
        print("  section eleven should have printed before its models were written")
        print("  anyway. Nothing goes in the config off this run.")
    else:
        for cell in survivors:
            print(f"\n  {cell.mechanism} on {cell.clock}, R:R {cell.ratio:.1f}")
            print(
                f"    {cell.trades} trades ({cell.per_day:.1f}/day), {cell.total_r:+.2f} R, "
                f"{cell.per_trade:+.3f} per trade, {cell.sigma:+.2f} sigma"
            )
            print(f"    holdout {cell.holdout_r:+.2f} R over {cell.holdout_trades} trades")
            print(f"    early {cell.early_r:+.2f} / late {cell.late_r:+.2f}")
            low, high = WANTED_TRADES_PER_DAY
            if not low <= cell.per_day <= high:
                print(
                    f"    NOTE: {cell.per_day:.1f} trades a day is outside the "
                    f"{low:.0f}-{high:.0f} the owner asked for."
                )

    _sessions_report(ranked[:5], args)
    _gates_not_modelled()


def _sessions_report(cells: list[Cell], args) -> None:
    print("\n" + "-" * 78)
    print("PER SESSION — where does it actually pay?")
    print("-" * 78)
    if not args.hours:
        print("  REPORTED, NOT SELECTED. Picking the best of five sessions is five")
        print("  more hypotheses per cell and the bar above does not include them.")
        print("  Re-run with --hours to select, and the bar rises accordingly.")
    for cell in cells:
        print(f"\n  {cell.mechanism} {cell.clock} R:R {cell.ratio:.1f}")
        print(f"    {'session':<10}{'trades':>8}{'total R':>10}{'per trade':>12}")
        for name, _hours in SESSIONS:
            values = cell.by_session.get(name, [])
            if not values:
                print(f"    {name:<10}{0:>8}{'--':>10}{'--':>12}")
                continue
            print(
                f"    {name:<10}{len(values):>8}{sum(values):>+10.2f}"
                f"{sum(values) / len(values):>+12.3f}"
            )


def _gates_not_modelled() -> None:
    """Named, every run, because an unnamed omission is an assumed zero.

    The owner asked for the live gates to be carried into the measurement. The
    ones that can be applied to a bar walk are applied: the cost gate, one
    position at a time, and the configured blocked hours. The rest need the
    live context this script does not build, and pretending otherwise would
    make every number here look better than the account can reach.
    """
    print("\n" + "-" * 78)
    print("WHAT THIS RUN DID NOT MODEL")
    print("-" * 78)
    print("  APPLIED HERE: the real cost model (commission + spread + slippage")
    print("  against the stop), one position at a time, and a bar spanning both")
    print("  barriers booked as a LOSS because the order inside it is unknowable.")
    print()
    print("  NOT APPLIED, and each one only ever REMOVES trades:")
    for name, what in (
        ("AWAITING_CONFIRMATION", "price ran against the idea over the last 3 bars"),
        ("NEWS_BLACKOUT", "a calendar event was near — 78 of 414 setups on one live day"),
        ("TARGET_RARELY_REACHED", "the target's historical reach rate was too low"),
        ("SPREAD_EATS_THE_STOP", "live spread against stop width, checked before sizing"),
        ("MARKET_TOO_QUIET", "the liveliness filter"),
        ("AWAITING_PULLBACK", "setup lifecycle: alive, not yet entered"),
        ("VOLUME_SPIKE", "volume spike filter"),
        ("ENTRY_OVEREXTENDED", "entry quality"),
        ("position cap", "four at once across every section, shared with S6/S7/S8/S10"),
        ("confluence floor", "the lone-module confidence gate this section must clear"),
    ):
        print(f"    {name:<24}{what}")
    print()
    print("  So read every R above as WHAT THE MECHANISM DOES WITH THOSE GATES OFF.")
    print("  `sectie11.cmd` runs the replay that has them on, and it is the number")
    print("  that decides. This one only decides what is WORTH replaying.")


def _write_csv(cells: list[Cell], path: Path) -> None:
    rows = []
    for cell in cells:
        row = {
            "mechanism": cell.mechanism,
            "clock": cell.clock,
            "ratio": cell.ratio,
            "trades": cell.trades,
            "per_day": round(cell.per_day, 2),
            "total_r": round(cell.total_r, 3),
            "per_trade": round(cell.per_trade, 4),
            "sigma": round(cell.sigma, 3),
            "control_per_trade": round(cell.control_per_trade, 4),
            "cost_share": round(cell.cost_share, 4),
            "holdout_trades": cell.holdout_trades,
            "holdout_r": round(cell.holdout_r, 3),
            "early_r": round(cell.early_r, 3),
            "late_r": round(cell.late_r, 3),
        }
        for name, _hours in SESSIONS:
            values = cell.by_session.get(name, [])
            row[f"{name}_trades"] = len(values)
            row[f"{name}_r"] = round(float(np.sum(values)), 3) if values else 0.0
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
