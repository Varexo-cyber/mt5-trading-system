"""Does section six actually work? Nothing has ever been able to answer that.

`modules.cmd` grades the confluence detectors and CANNOT grade this one, for a
reason that is silent rather than obvious. `backtesting/replay.py` fetches D1,
H4, H1, M15 and M5 -- no M1 -- and `candle_momentum` triggers on M1:

    fast = ctx.series.get(Timeframe.parse(config.trigger_timeframe))   # M1
    if fast is None or middle is None or slow is None:
        return Signal.neutral(self.name, "needs M1, M5 and M15 history")

So in every backtest this project has ever run, section six returned a neutral
signal, proposed nothing, and appeared in no table. Not "graded and found
wanting" -- never measured at all, while trading real money since 24 August.

There is a second reason the general replay could not have done it even with
M1 loaded: it decides once per H1 bar. A rule that reads the last closed MINUTE
sampled hourly is a different rule. This walks every closed M1 bar instead.

WHAT IT REPLAYS. The detector at the settings the account runs, and then the
lane's own geometry from `_scalp_plan` rather than the confluence engine's:

    entry  = ask for a long, bid for a short   (the side actually paid)
    stop   = entry -/+ span x stop_candle_spans
    target = entry +/- span x target_candle_spans

with `span` the high-low of the triggering candle. That is the whole plan. The
confluence vote is not involved -- section six has its own lane precisely
because its ceiling of 45 x 0.75 could never clear a bar of 45.

WHAT IT DOES NOT MODEL, and the list is the honest part: the per-second claim
and cut in `_scalp_verdict`, the profit lock, the news blackout, the
concurrency cap. This measures the ENTRY and the plan. If that is negative, no
exit rule saves it; if it is positive, the exit work has something to improve.

    python scripts/backtest_section_six.py --days 30
    python scripts/backtest_section_six.py --days 30 --symbols XAUUSD US30
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.candle_momentum import CandleMomentum
from backtesting.engine import BacktestOrder, PessimisticBacktester
from config.loader import (
    DEFAULT_CONFIG_PATH,
    load_credentials,
    load_settings,
    terminal_path_from_env,
)
from core.data_manager import DataManager
from core.mt5_connector import MT5Connector
from core.types import Direction, MarketContext, Series, Tick, Timeframe

#: Only where commission is zero, because that is the only place the lane may
#: trade. Mirrors `_scalp_plan`'s asset-class refusal rather than restating it
#: as a symbol list that would drift.
DEFAULT_SYMBOLS = ("XAUUSD", "XAUEUR", "US30", "NAS100", "GER40")

#: What the detector needs behind the trigger bar. `candle_lookback` is 30 and
#: the module rejects under `lookback + 2`; 120 leaves room for the M5/M15
#: slope reads without carrying the whole frame into every context.
CONTEXT_BARS = 120


def history(connector: MT5Connector, symbol: str, start: datetime, end: datetime):  # type: ignore[no-untyped-def]
    frames: dict[Timeframe, pd.DataFrame] = {}
    for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15):
        warmup = start - timeframe.duration * (CONTEXT_BARS + 40)
        raw = connector.copy_rates_range(symbol, timeframe.mt5_value, warmup, end)
        frames[timeframe] = DataManager._to_frame(raw)
    return frames


def proposals(
    symbol: str,
    frames: dict[Timeframe, pd.DataFrame],
    settings,  # type: ignore[no-untyped-def]
    *,
    point: float,
    start: datetime,
    end: datetime,
    stride: int = 1,
) -> list[BacktestOrder]:
    """Walk every closed M1 bar and record what the lane would have sent.

    The fill side is the one `_scalp_plan` uses -- a long enters at the ask and
    a short at the bid -- and the spread comes from the broker's own recorded
    value on the trigger bar, so the round trip is charged rather than assumed
    away at the mid.
    """
    config = settings.analysis.candle_momentum
    detector = CandleMomentum(config)
    minute = frames[Timeframe.M1]
    closes = {
        tf: frames[tf].index + tf.duration for tf in (Timeframe.M1, Timeframe.M5, Timeframe.M15)
    }
    orders: list[BacktestOrder] = []

    decided = minute.index + Timeframe.M1.duration
    eligible = minute[(decided >= start) & (decided < end)]
    for sequence, opened_at in enumerate(eligible.index):
        if sequence % stride:
            continue
        decided_at = (opened_at + Timeframe.M1.duration).to_pydatetime()
        moment = pd.Timestamp(decided_at)
        series: dict[Timeframe, Series] = {}
        for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15):
            cut = int(closes[timeframe].searchsorted(moment, side="right"))
            if cut < CONTEXT_BARS:
                break
            window = frames[timeframe].iloc[max(0, cut - CONTEXT_BARS) : cut]
            series[timeframe] = Series(symbol, timeframe, window, decided_at)
        if len(series) < 3:
            continue

        bar = series[Timeframe.M1].df.iloc[-1]
        mid = float(bar["close"])
        spread = max(float(bar.get("spread", 0.0)), 0.0) * point
        tick = Tick(symbol, decided_at, bid=mid - spread / 2, ask=mid + spread / 2)
        signal = detector.analyze(MarketContext(symbol, decided_at, series, tick))
        if not signal.score:
            continue

        direction = Direction.LONG if signal.score > 0 else Direction.SHORT
        entry = tick.ask if direction is Direction.LONG else tick.bid
        span = float(bar["high"]) - float(bar["low"])
        if span <= 0 or entry <= 0:
            continue
        sign = 1.0 if direction is Direction.LONG else -1.0
        stop = entry - sign * span * config.stop_candle_spans
        target = entry + sign * span * config.target_candle_spans
        if min(entry, stop, target) <= 0:
            continue
        orders.append(
            BacktestOrder(
                symbol=symbol,
                decided_at=decided_at,
                direction=direction,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                score=abs(signal.score),
                confidence=signal.confidence,
                modules=("candle_momentum",),
                spread=spread,
            )
        )
    return orders


def render(rows: list[tuple], window: str) -> str:  # type: ignore[no-untyped-def]
    lines = [
        "",
        "=" * 78,
        f"  SECTION SIX, MEASURED FOR THE FIRST TIME  {window}",
        "=" * 78,
        "",
        "  Every closed M1 bar, the detector at live settings, the lane's own",
        "  stop and target. Costs charged: the broker's recorded spread on the",
        "  trigger bar, plus commission and slippage.",
        "",
        "  NOT MODELLED: the per-second claim and cut, the profit lock, the news",
        "  blackout, the two-position cap. This is the entry and the plan. If",
        "  that loses, no exit rule rescues it; if it wins, the exit work has",
        "  something real to improve.",
        "",
        f"  {'symbol':<12}{'setups':>8}{'trades':>8}{'win':>7}{'per trade':>11}"
        f"{'avg win':>10}{'avg loss':>10}{'TP':>6}{'SL':>6}{'total':>10}",
        "  " + "-" * 76,
    ]
    for row in rows:
        symbol, setups, trades, win, per, avg_w, avg_l, tp, sl, total = row
        lines.append(
            f"  {symbol:<12}{setups:>8}{trades:>8}{win:>6.0%}{per:>+10.3f}R"
            f"{avg_w:>+9.2f}R{avg_l:>+9.2f}R{tp:>6.0%}{sl:>6.0%}{total:>+9.2f}R"
        )
    if not rows:
        lines.append("  (no setup reached a closed trade)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="decide every Nth M1 bar. Raise it to trade accuracy for speed",
    )
    args = parser.parse_args(argv)

    settings = load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml")
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    backtester = PessimisticBacktester()

    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    rows: list[tuple] = []
    pooled: list[float] = []
    try:
        connector.connect()
        for symbol in args.symbols:
            print(f"  replaying {symbol} …", flush=True)
            try:
                spec = connector.spec(symbol)
                frames = history(connector, symbol, start, end)
            except Exception as exc:  # noqa: BLE001 - one bad symbol is not the run
                print(f"    skipped: {type(exc).__name__}: {exc}")
                continue
            if any(frame.empty for frame in frames.values()):
                print("    skipped: no history")
                continue
            orders = proposals(
                symbol,
                frames,
                settings,
                point=spec.point,
                start=start,
                end=end,
                stride=args.stride,
            )
            print(f"    {len(orders)} setups")
            if not orders:
                continue
            result = backtester.run_non_overlapping(frames[Timeframe.M1], orders)
            if not result.sample_size:
                continue
            returns = [trade.net_r for trade in result.trades]
            wins = [value for value in returns if value > 0]
            losses = [value for value in returns if value <= 0]
            outcomes: dict[str, int] = defaultdict(int)
            for trade in result.trades:
                outcomes["SL" if trade.outcome.startswith("SL") else trade.outcome] += 1
            pooled.extend(returns)
            rows.append(
                (
                    symbol,
                    len(orders),
                    result.sample_size,
                    result.win_rate,
                    result.expectancy_r,
                    (sum(wins) / len(wins)) if wins else 0.0,
                    (sum(losses) / len(losses)) if losses else 0.0,
                    outcomes["TP"] / result.sample_size,
                    outcomes["SL"] / result.sample_size,
                    result.total_r,
                )
            )
    finally:
        with contextlib.suppress(Exception):
            connector.shutdown()

    print(render(rows, f"{args.days:.0f} days"))
    if pooled:
        values = np.asarray(pooled, dtype=float)
        per = float(values.mean())
        print(f"  Pooled: {len(values)} trades at {per:+.3f}R a trade, {values.sum():+.2f}R total.")
        print(
            "  Positive is the first evidence this lane has ever had.\n"
            if per > 0
            else "  Negative on the entry alone. No exit rule recovers that.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
