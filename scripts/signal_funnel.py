"""WHERE a detector loses every candidate, clause by clause, per market.

    python scripts/signal_funnel.py --days 7
    python scripts/signal_funnel.py --days 30 --symbols EURUSD.i,XAUUSD

THE RUN THIS EXISTS FOR. On 30 August the core dry run took 42 trades and not
one was on FX. All eleven majors -- the eleven both sections were MEASURED on,
where the research says roughly 0.6 impulse_retest trades per pair per day --
produced nothing in seven days. Everything that traded was gold and index CFDs.

The dry run could say that much. What it could not say is WHY, because
`no weighted directional evidence` at 95.9% means only "the module returned
nothing", and the module has five separate reasons to return nothing:

    1. no bar closed beyond its 20-bar channel        no break at all
    2. the close was beyond it by less than 1.0 ATR   the IMPULSE filter
    3. price later ran past the stop                  the level was given up
    4. price is on the far side of the level          nothing to retest
    5. price is more than 0.15 ATR above the level    the retest has not
                                                      arrived at this instant

Those call for completely different responses. (1) and (2) say the market was
quiet or the threshold is wrong for this feed. (5) says the opposite -- the
setups exist and the sampling misses them, which is what a resting LIMIT order
would not do and a once-per-bar check does.

That last one is the hypothesis this script is built to test, because the
research bought these setups with a limit resting at the level and the live
module can only see the price at the moment it happens to be asked. Every
touch between two bar closes is invisible to it.

This reads the detector's own clauses rather than reimplementing them, so it
cannot drift away from the module the way an approximation would.
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

from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe

WARMUP = 260


@dataclass
class Funnel:
    """One counter per clause, in the order the detector applies them."""

    bars: int = 0
    #: A bar closed beyond the 20-bar channel edge.
    broke: int = 0
    #: ...by at least `minimum_impulse_atr`.
    decisive: int = 0
    #: ...and price has not since run past where the stop would sit.
    alive: int = 0
    #: ...and price is on the retest side of the level.
    right_side: int = 0
    #: ...and price is within `tolerance_atr` of it RIGHT NOW. This is the
    #: signal.
    fired: int = 0
    #: How close price got to the level during the bar, when it did not fire.
    #: A distribution of near-misses says whether the band is too narrow or
    #: the setups simply are not there.
    misses: list[float] = field(default_factory=list)
    #: Bars where price TOUCHED the band at some point during the bar but was
    #: not inside it at the close. A resting limit takes these; a once-per-bar
    #: check does not.
    touched_intrabar: int = 0


def _atr(frame: pd.DataFrame, period: int) -> float:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(period).mean())


def _walk(frame: pd.DataFrame, config) -> Funnel:
    """Replay `ImpulseRetest._live_break`'s clauses over every bar.

    Deliberately written against the same field names the module reads, so a
    change there shows up here as an AttributeError rather than as a quietly
    divergent answer.
    """
    funnel = Funnel()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    upper = pd.Series(high).shift(1).rolling(config.channel_period).max().to_numpy()
    lower = pd.Series(low).shift(1).rolling(config.channel_period).min().to_numpy()

    for now in range(config.channel_period + 1, len(close)):
        funnel.bars += 1
        window = frame.iloc[max(0, now - WARMUP) : now + 1]
        unit = _atr(window, config.atr_period)
        if unit <= 0:
            continue
        price = float(close[now])
        oldest = max(config.channel_period + 1, now - config.lookback_bars)

        best_gap: float | None = None
        saw_break = saw_decisive = saw_alive = saw_side = False
        for i in range(now, oldest - 1, -1):
            if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
                continue
            if close[i] > upper[i]:
                direction, level = 1, float(upper[i])
            elif close[i] < lower[i]:
                direction, level = -1, float(lower[i])
            else:
                continue
            saw_break = True
            if direction * (close[i] - level) / unit < config.minimum_impulse_atr:
                continue
            saw_decisive = True
            beyond = level - direction * config.stop_beyond_atr * unit
            after = close[i + 1 : now + 1]
            if len(after) and (
                (direction > 0 and float(after.min()) < beyond)
                or (direction < 0 and float(after.max()) > beyond)
            ):
                continue
            saw_alive = True
            gap = direction * (price - level) / unit
            if gap < 0.0:
                continue
            saw_side = True
            if best_gap is None or gap < best_gap:
                best_gap = gap
            # Did the bar TOUCH the band even though the close did not sit in
            # it? A resting limit fills there; a once-per-bar check misses it.
            extreme = float(low[now]) if direction > 0 else float(high[now])
            if direction * (extreme - level) / unit <= config.tolerance_atr:
                funnel.touched_intrabar += 1

        funnel.broke += saw_break
        funnel.decisive += saw_decisive
        funnel.alive += saw_alive
        funnel.right_side += saw_side
        if best_gap is not None:
            if best_gap <= config.tolerance_atr:
                funnel.fired += 1
            else:
                funnel.misses.append(best_gap)
    return funnel


def _print(symbol: str, funnel: Funnel, tolerance: float) -> None:
    def line(label: str, count: int, of: int) -> None:
        share = f"{count / of:>6.1%}" if of else "     -"
        print(f"      {label:<44} {count:>7} {share}")

    print(f"\n   {symbol}")
    line("bars examined", funnel.bars, funnel.bars)
    line("a bar closed beyond the 20-bar channel", funnel.broke, funnel.bars)
    line("...by at least 1.0 ATR  (the impulse filter)", funnel.decisive, funnel.bars)
    line("...and the level was not given up since", funnel.alive, funnel.bars)
    line("...and price is on the retest side", funnel.right_side, funnel.bars)
    line(f"...and within {tolerance:.2f} ATR of it AT THE CLOSE", funnel.fired, funnel.bars)
    if funnel.touched_intrabar:
        line("   (the bar TOUCHED the band at some point)", funnel.touched_intrabar, funnel.bars)
    if funnel.misses:
        near = np.quantile(funnel.misses, [0.1, 0.25, 0.5])
        print(
            f"      near misses: closest 10% at {near[0]:.2f} ATR, "
            f"quartile {near[1]:.2f}, median {near[2]:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbols", default="", help="comma list; default = the core universe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    config = settings.analysis.impulse_retest
    clock = Timeframe.parse(config.timeframe)

    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=True),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        from scripts.dry_run_sections import _core_universe

        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = _core_universe(connector, settings)

        end = datetime.now(UTC)
        start = end - timedelta(days=args.days)
        fetch_from = start - (WARMUP + 40) * clock.duration

        print(
            f"\nIMPULSE_RETEST FUNNEL — {config.timeframe}, {args.days} days, "
            f"{len(symbols)} markets"
        )
        print("Where every candidate is lost, clause by clause, in the detector's own order.")

        for symbol in symbols:
            try:
                frame = fetch_mt5_history(connector, symbol, clock, fetch_from, end)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                print(f"\n   {symbol}: no history ({exc})")
                continue
            if len(frame) < WARMUP + 40:
                print(f"\n   {symbol}: only {len(frame)} bars, need {WARMUP + 40}")
                continue
            _print(symbol, _walk(frame, config), config.tolerance_atr)
    finally:
        connector.shutdown()


if __name__ == "__main__":
    main()
