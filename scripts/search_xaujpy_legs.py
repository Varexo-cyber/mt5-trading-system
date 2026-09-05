"""XAUJPY against its own two legs, and the first honest question about it.

WHY THIS EXISTS. The twenty-eight mechanisms in `analysis/mechanisms.py` were
written for section four: index CFDs on M15/M30/H1. They are generic one-to-four
bar patterns -- a streak, a gap, where the close sits in the bar -- and not one
of them has a counterparty on a gold cross. Pointed at XAUJPY on M1 the best of
them was `streak_reversal`, "four closes the same way, then trade against it",
which fires on 10.3% of all bars (149 signals a day) and lost 0.18 R a trade in
the replay. Nobody loses money because four one-minute candles went up. There
was no mechanism there to find.

THE MECHANISM THIS MEASURES, and it exists only on a cross.

    XAUJPY = XAUUSD x USDJPY

Two legs, quoted in two different books. When gold moves and the yen does not,
the cross has to follow -- and whoever is quoting the cross does that with a
lag. The counterparty is the market maker whose cross quote has not caught up
with its own legs yet, and the trade pays when it does. That is a sentence you
can say out loud, which is more than any of the twenty-eight could manage here.

THE QUESTION THIS ANSWERS FIRST, BEFORE ANY BACKTEST. Many brokers compute a
cross synthetically from the legs. If Eightcap does, the gap is zero by
construction and there is NOTHING HERE -- no lag, no counterparty, no trade.
This script measures the gap distribution and says so before it resolves a
single trade, because a search that finds an edge in a gap that cannot exist is
a search that has found its own rounding error.

WHAT IS DELIBERATELY NOT DONE. The relationship carries a constant offset --
contract size, a broker markup, a financing component -- and none of that is
tradeable. So the reading is the gap's DEVIATION FROM ITS OWN RECENT NORMAL,
not the raw difference. A structural offset cancels; only the lag survives.

Everything else -- the cost model, the day-clustered sigma, the rate-matched
control, the untouched holdout, the Bonferroni bar, the firing-rate bound -- is
the same machinery `search_xaujpy.py` uses, and for the same reasons.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.mechanisms import HORIZON, WARMUP, _atr
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
from scripts.search_xaujpy import MAX_TRADES_PER_DAY, SESSIONS, session_of

#: Bars the gap is compared against to remove the structural offset. Long
#: enough that a real lag is an outlier against it, short enough that a slow
#: drift in the offset does not become a permanent signal.
NORMAL_BARS = 96

#: How far the gap must sit from its own normal, in ATR of the cross, before
#: it counts as a lag worth trading. Swept rather than chosen.
THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

#: Reward per unit of risk, swept alongside.
RATIOS: tuple[float, ...] = (1.0, 1.5)

#: Below this the cross is being computed from its legs rather than quoted
#: independently, and there is no lag to trade. Expressed in ATR of the cross,
#: so it is scale-free.
DEAD_GAP_ATR = 0.02


@dataclass
class Cell:
    clock: str
    threshold: float
    ratio: float
    trades: int = 0
    total_r: float = 0.0
    per_trade: float = 0.0
    sigma: float = 0.0
    per_day: float = 0.0
    fire_rate: float = 0.0
    control_per_trade: float = 0.0
    cost_share: float = 0.0
    holdout_trades: int = 0
    holdout_r: float = 0.0
    early_r: float = 0.0
    late_r: float = 0.0
    by_session: dict = field(default_factory=dict)

    @property
    def beats_its_coin(self) -> bool:
        return self.per_trade > self.control_per_trade


def implied_cross(gold: pd.DataFrame, yen: pd.DataFrame) -> pd.Series:
    """What XAUJPY has to be, from the two books that actually price it.

    Closes only, and on the shared timestamps only. An inner join is the whole
    alignment: MT5 puts every symbol on the same bar grid, so a timestamp
    present in one and missing in the other is a bar one of them did not trade,
    and inventing it would invent the gap this script is looking for.
    """
    joined = gold[["close"]].join(yen[["close"]], how="inner", lsuffix="_au", rsuffix="_jp")
    return joined["close_au"] * joined["close_jp"]


def gap_reading(
    cross: pd.DataFrame, implied: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """`(the shared bars, gap in ATR, raw gap in price)`.

    RETURNS THE FRAME IT ALIGNED, and that is not tidiness. The first version
    handed back only the readings and left the caller to rebuild the shared
    index itself with a second intersection. Two alignments of one thing is how
    a reading ends up one bar out of step with the bar it labels -- silently,
    and in the direction that flatters, because a gap read against the NEXT
    bar's price is a look-ahead. One alignment, returned.

    THE DEVIATION FROM ITS OWN NORMAL, not the raw difference. Contract size, a
    broker markup and the financing leg all put a constant between the cross and
    the product of its legs, and none of it is tradeable. Subtracting the
    rolling normal removes exactly that and leaves the lag.
    """
    aligned = cross.join(implied.rename("implied"), how="inner")
    frame = aligned[["open", "high", "low", "close"]]
    raw = (aligned["close"] - aligned["implied"]).to_numpy()
    normal = pd.Series(raw).rolling(NORMAL_BARS, min_periods=NORMAL_BARS // 2).mean().to_numpy()
    unit = _atr(frame)
    with np.errstate(invalid="ignore", divide="ignore"):
        reading = (raw - normal) / np.where(unit > 0, unit, np.nan)
    return frame, reading, raw


def signals_from_gap(reading: np.ndarray, threshold: float) -> np.ndarray:
    """Rich cross is sold, cheap cross is bought.

    A cross above its legs is a quote that has not come down yet; the trade is
    that it does. Direction comes from the gap and not from the market, which
    is why this can be right about the trade while being wrong about where gold
    goes -- the same property that makes `basket_divergence` worth having.
    """
    out = np.zeros(len(reading), dtype=int)
    with np.errstate(invalid="ignore"):
        out[reading >= threshold] = -1
        out[reading <= -threshold] = 1
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--cross", default="XAUJPY")
    parser.add_argument("--gold", default="XAUUSD")
    parser.add_argument("--yen", default="USDJPY")
    parser.add_argument("--clocks", nargs="*", default=["M5", "M15"])
    parser.add_argument("--stop-atr", type=float, default=1.0)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--cells-already-tried", type=int, default=0)
    parser.add_argument("--csv", default="")
    return parser


def _fetch(connector, symbol: str, clock: str, days: int) -> pd.DataFrame | None:
    tf = Timeframe.parse(clock)
    end = datetime.now(UTC)
    pad = max((WARMUP + HORIZON + NORMAL_BARS + 50) * tf.duration * 1.6, timedelta(days=3))
    print(f"  fetching {symbol} {clock} ...", flush=True)
    frame = fetch_mt5_history(connector, symbol, tf, end - timedelta(days=days) - pad, end)
    if frame is None or len(frame) < WARMUP + HORIZON + NORMAL_BARS:
        print(f"  {symbol} {clock}: not enough history")
        return None
    print(f"  {symbol} {clock}: {len(frame):,} bars")
    return frame


def main() -> None:
    args = build_parser().parse_args()
    args.clocks = [
        piece.strip().upper()
        for chunk in args.clocks
        for piece in str(chunk).split(",")
        if piece.strip()
    ]
    settings = load_settings(env_overrides=False)

    print("=" * 78)
    print(f"{args.cross} AGAINST ITS LEGS — {args.gold} x {args.yen}")
    print("=" * 78)

    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=True),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        spec = connector.spec(args.cross)
        sizer = PositionSizer(settings)
        frames: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        for clock in args.clocks:
            cross = _fetch(connector, args.cross, clock, args.days)
            gold = _fetch(connector, args.gold, clock, args.days)
            yen = _fetch(connector, args.yen, clock, args.days)
            if cross is None or gold is None or yen is None:
                continue
            frames[clock] = gap_reading(cross, implied_cross(gold, yen))
    finally:
        connector.shutdown()

    if not frames:
        raise SystemExit(
            f"no usable history. MT5 must be running and logged in with "
            f"{args.cross}, {args.gold} and {args.yen} all visible in Market Watch."
        )

    if not _the_gap_is_real(frames):
        return

    cells: list[Cell] = []
    for clock, (frame, reading, _raw) in frames.items():
        cut = int(len(frame) * (1.0 - args.holdout))
        searched, held = frame.iloc[:cut], frame.iloc[cut:]
        read_s, read_h = reading[:cut], reading[cut:]
        span = max((searched.index[-1] - searched.index[0]).days, 1)
        split = searched.index[int(len(searched) * 0.6)]
        stop_price = args.stop_atr * float(np.nanmedian(_atr(searched)))
        cost = _cost_share(sizer, spec, stop_price)
        print(
            f"\n{clock}: searching {len(searched):,} bars, holding back {len(held):,} "
            f"from {held.index[0]:%Y-%m-%d}; round trip {cost:.1%} of the stop"
        )

        for threshold in THRESHOLDS:
            marks = signals_from_gap(read_s, threshold)
            for ratio in RATIOS:
                cell = Cell(clock=clock, threshold=threshold, ratio=ratio)
                cell.cost_share = cost
                cell.fire_rate = float(np.mean(marks != 0))
                found = resolve(searched, marks, stop_atr=args.stop_atr, ratio=ratio, cost_r=cost)
                if len(found) < 30:
                    continue
                total, per_trade, sigma, n = stats(found)
                cell.trades, cell.total_r = n, total
                cell.per_trade, cell.sigma, cell.per_day = per_trade, sigma, n / span
                for value, when in zip(found.r, found.when, strict=True):
                    cell.by_session.setdefault(session_of(int(when.hour)), []).append(value)
                    if when < split:
                        cell.early_r += value
                    else:
                        cell.late_r += value

                control = random_control(searched, seed=abs(hash((clock, threshold, ratio))) % 9973)
                got = resolve(searched, control, stop_atr=args.stop_atr, ratio=ratio, cost_r=cost)
                if len(got):
                    cell.control_per_trade = stats(got)[1]

                out = resolve(
                    held,
                    signals_from_gap(read_h, threshold),
                    stop_atr=args.stop_atr,
                    ratio=ratio,
                    cost_r=cost,
                )
                if len(out):
                    h_total, _p, _s, h_n = stats(out)
                    cell.holdout_trades, cell.holdout_r = h_n, h_total
                cells.append(cell)

    tested = len(THRESHOLDS) * len(RATIOS) * len(frames) + args.cells_already_tried
    _report(cells, bonferroni_sigma(tested), tested, settings)
    if args.csv:
        _write_csv(cells, Path(args.csv))
        print(f"\nevery cell written to {args.csv}")


def _the_gap_is_real(frames: dict) -> bool:
    """The question that has to be answered before any backtest is worth running.

    A broker that computes the cross from its legs leaves no gap, and a search
    over a gap that cannot exist finds its own rounding error and calls it an
    edge. So this prints the distribution and stops if there is nothing there.
    """
    print("\n" + "-" * 78)
    print("IS THERE A GAP AT ALL?")
    print("-" * 78)
    print(f"    {'clock':<7}{'bars':>10}{'median |gap|':>15}{'90th pct':>12}{'max':>12}   (in ATR)")
    alive = False
    for clock, (_frame, reading, raw) in frames.items():
        finite = reading[np.isfinite(reading)]
        if not len(finite):
            print(f"    {clock:<7}{0:>10}{'--':>15}{'--':>12}{'--':>12}")
            continue
        median = float(np.median(np.abs(finite)))
        p90 = float(np.percentile(np.abs(finite), 90))
        biggest = float(np.max(np.abs(finite)))
        print(f"    {clock:<7}{len(finite):>10}{median:>15.3f}{p90:>12.3f}{biggest:>12.3f}")
        raw_median = float(np.nanmedian(np.abs(raw)))
        print(f"    {'':<7}{'':<10}raw median difference {raw_median:,.1f} price units")
        if p90 >= DEAD_GAP_ATR:
            alive = True
    if not alive:
        print(f"\n  NOTHING TO TRADE. The cross never sits more than {DEAD_GAP_ATR} ATR from")
        print("  its own legs, which means this broker computes it FROM those legs")
        print("  rather than quoting it. There is no lag, so there is no counterparty")
        print("  and no mechanism. That is the answer, and it is a useful one: it")
        print("  rules the idea out in a minute instead of after a 90-day replay.")
        return False
    print("\n  There is a gap. Whether it is TRADEABLE after costs is the rest of")
    print("  this report -- a gap smaller than the round trip is not an edge.")
    return True


def _report(cells: list[Cell], bar: float, tested: int, settings) -> None:
    cap = settings.risk.max_cost_share_of_risk
    print("\n" + "=" * 78)
    print(f"RESULT — Bonferroni bar at {tested} cells: {bar:.2f} sigma")
    print("=" * 78)
    if not cells:
        print("  no cell produced 30 resolved trades. Nothing to say.")
        return

    ranked = sorted(cells, key=lambda c: c.sigma, reverse=True)
    print(
        f"\n  {'clk':>5}{'gap':>7}{'R:R':>6}{'trades':>8}{'/day':>7}{'total R':>10}"
        f"{'per':>8}{'cost':>7}{'fires':>7}{'sigma':>7}{'holdout':>10}"
    )
    for cell in ranked[:20]:
        flag = "" if cell.beats_its_coin else "  <- loses to its own coin flip"
        print(
            f"  {cell.clock:>5}{cell.threshold:>7.2f}{cell.ratio:>6.1f}{cell.trades:>8}"
            f"{cell.per_day:>7.1f}{cell.total_r:>+10.2f}{cell.per_trade:>+8.3f}"
            f"{cell.cost_share:>6.1%}{cell.fire_rate:>6.1%}{cell.sigma:>+7.2f}"
            f"{cell.holdout_r:>+10.2f}{flag}"
        )

    survivors = [
        c
        for c in ranked
        if c.sigma >= bar
        and c.beats_its_coin
        and c.holdout_r > 0
        and c.total_r > 0
        and c.cost_share <= cap
        and c.per_day <= MAX_TRADES_PER_DAY
    ]
    print("\n" + "-" * 78)
    print("DID THE LAG EARN A SECTION")
    print("-" * 78)
    if not survivors:
        best = ranked[0]
        print("\n  NO. All five at once: the bar, a positive total, beating its own")
        print("  coin flip, a POSITIVE untouched holdout, and inside the cost and")
        print("  firing bounds.")
        print(
            f"\n  Closest: {best.clock} gap {best.threshold:.2f} R:R {best.ratio:.1f} at "
            f"{best.sigma:+.2f} against {bar:.2f}, holdout {best.holdout_r:+.2f} R, "
            f"{best.per_day:.1f} trades a day."
        )
        print("\n  Nothing goes in the config off this run.")
    else:
        for cell in survivors:
            print(f"\n  {cell.clock}, gap {cell.threshold:.2f} ATR, R:R {cell.ratio:.1f}")
            print(
                f"    {cell.trades} trades ({cell.per_day:.1f}/day), {cell.total_r:+.2f} R, "
                f"{cell.per_trade:+.3f} per trade, {cell.sigma:+.2f} sigma"
            )
            print(f"    holdout {cell.holdout_r:+.2f} R over {cell.holdout_trades} trades")
            print(f"    early {cell.early_r:+.2f} / late {cell.late_r:+.2f}")
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

    print("\n" + "-" * 78)
    print("WHAT THIS RUN DID NOT MODEL")
    print("-" * 78)
    print("  Applied: the account's own cost model, one position at a time, and a")
    print("  bar spanning both barriers booked as a LOSS.")
    print()
    print("  NOT applied, and every one of them only REMOVES trades:")
    print("    the eight live gates (confirmation, news, target reach, spread,")
    print("    liveliness, pullback, volume spike, entry quality), the four-position")
    print("    cap shared with sections 6/7/8/10, and the lone-module floor.")
    print()
    print("  AND ONE THAT IS SPECIFIC TO THIS IDEA. The legs are read on CLOSED bars")
    print("  of the same clock. Live, a lag that closes inside one bar is gone before")
    print("  the next close, so a real implementation would need tick or M1 legs even")
    print("  for an M15 cross. If this pays here, that is the next thing to measure,")
    print("  not the thing to skip.")


def _write_csv(cells: list[Cell], path: Path) -> None:
    rows = []
    for cell in cells:
        row = {
            "clock": cell.clock,
            "gap_atr": cell.threshold,
            "ratio": cell.ratio,
            "trades": cell.trades,
            "per_day": round(cell.per_day, 2),
            "total_r": round(cell.total_r, 3),
            "per_trade": round(cell.per_trade, 4),
            "sigma": round(cell.sigma, 3),
            "fire_rate": round(cell.fire_rate, 4),
            "cost_share": round(cell.cost_share, 4),
            "control_per_trade": round(cell.control_per_trade, 4),
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
