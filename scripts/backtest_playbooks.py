"""Ask whether the five theories have ever worked, using free data.

They have never been tested. `momentum_scalp`, `range_fade`, `range_break`,
`failed_break` and `trend_pullback` were each argued into existence, reviewed,
unit-tested against synthetic bars — and never once run against a market. The
live account bets on all five.

This pulls months of real bars from MT5, replays every theory over them one
closed M5 bar at a time, and reports what each would have returned on its own.
No orders, no API calls, nothing written. The only thing it spends is time.

    python scripts/backtest_playbooks.py                     # 90 days, four majors
    python scripts/backtest_playbooks.py --days 180
    python scripts/backtest_playbooks.py --symbols EURUSD.i XAUUSD
    python scripts/backtest_playbooks.py --min-conviction 75  # the live floor

Read the per-theory rows, not the total. A theory that is negative over a real
sample should be switched off, and doing that is worth more — and is far more
certain — than any rule that could be added in its place.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis.playbooks import (
    BreakConfig,
    FadeConfig,
    FailedBreak,
    FailedBreakConfig,
    MomentumScalp,
    Playbook,
    PlaybookEngine,
    PullbackConfig,
    RangeBreak,
    RangeFade,
    ScalpConfig,
    TrendPullback,
)
from backtesting.playbook_replay import (
    REPLAY_TIMEFRAMES,
    PlaybookReplay,
    compare_to_chance,
    evidence_by_playbook,
    render,
    render_comparison,
    render_targets,
    sweep_targets,
)
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.data_manager import DataManager
from core.mt5_connector import MT5Connector
from core.types import Timeframe

DEFAULT_SYMBOLS = ("EURUSD.i", "GBPUSD.i", "USDJPY.i", "AUDUSD.i")


def build_engine(settings, share: float) -> PlaybookEngine:  # type: ignore[no-untyped-def]
    """Every theory, at the settings the account is actually running."""
    chosen: list[Playbook] = [
        MomentumScalp(ScalpConfig(max_spread_share_of_stop=share)),
        RangeFade(FadeConfig(max_spread_share_of_stop=share)),
        RangeBreak(BreakConfig(max_spread_share_of_stop=share)),
        FailedBreak(FailedBreakConfig(max_spread_share_of_stop=share)),
        TrendPullback(PullbackConfig(max_spread_share_of_stop=share)),
    ]
    return PlaybookEngine(chosen, settings.analysis.confluence)


def history(
    connector: MT5Connector, symbol: str, start: datetime, end: datetime
) -> dict[Timeframe, pd.DataFrame]:
    frames: dict[Timeframe, pd.DataFrame] = {}
    for timeframe in REPLAY_TIMEFRAMES:
        # Reach back past `start` so the first decision already has its 300
        # bars of context instead of being skipped for want of history.
        warmup = start - timeframe.duration * 400
        raw = connector.copy_rates_range(symbol, timeframe.mt5_value, warmup, end)
        frames[timeframe] = DataManager._to_frame(raw)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=90.0, help="how far back to replay")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--min-conviction",
        type=float,
        default=0.0,
        help="only proposals above this. 0 measures the theories, the live floor measures "
        "the configuration",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the coin-flip control. It roughly doubles the run and it is the "
        "only part that says whether the analysis is worth anything",
    )
    parser.add_argument(
        "--targets",
        action="store_true",
        help="also sweep the target distance, each with its own coin. Answers "
        "whether reaching for less turns any of this positive",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="decide every Nth M5 bar. Raise it to trade accuracy for speed",
    )
    args = parser.parse_args(argv)

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    engine = build_engine(settings, settings.analysis.playbooks.max_spread_share_of_stop)
    replay = PlaybookReplay(engine, decision_stride_bars=args.stride)

    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        connector.connect()
    except Exception as exc:  # noqa: BLE001 - the caller only needs the reason
        print(f"Could not connect to MT5: {type(exc).__name__}: {exc}")
        print("This needs the terminal running — it reads bar history, nothing else.")
        return 1

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    everything = []
    against_chance = []
    target_rows = []
    try:
        for symbol in args.symbols:
            print(f"  replaying {symbol} …", flush=True)
            try:
                spec = connector.spec(symbol)
                frames = history(connector, symbol, start, end)
                orders = replay.orders(
                    symbol,
                    frames,
                    point=spec.point,
                    start=start,
                    end=end,
                    min_conviction=args.min_conviction,
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not end the run
                print(f"    skipped: {type(exc).__name__}: {exc}")
                continue
            print(f"    {len(orders)} proposals")
            if not orders:
                continue
            evidence = evidence_by_playbook(orders, frames[Timeframe.M5])
            everything.extend(evidence)
            if not args.no_baseline:
                print("    shuffling the directions …", flush=True)
                against_chance.extend(compare_to_chance(orders, frames[Timeframe.M5], evidence))
            if args.targets:
                print("    sweeping the target distance …", flush=True)
                target_rows.extend(sweep_targets(orders, frames[Timeframe.M5]))
    finally:
        with contextlib.suppress(Exception):
            connector.shutdown()

    # One row per theory across every symbol, rather than per symbol per
    # theory: the question is whether the theory works, and splitting a thin
    # sample four ways is how a real result gets buried under noise.
    merged: dict[str, list] = {}
    for item in everything:
        merged.setdefault(item.playbook, []).append(item)
    combined = []
    for name, items in merged.items():
        trades = sum(i.trades for i in items)
        total = sum(i.total_r for i in items)
        wins = sum(i.win_rate * i.trades for i in items)
        combined.append(
            type(items[0])(
                playbook=name,
                proposals=sum(i.proposals for i in items),
                trades=trades,
                total_r=total,
                win_rate=(wins / trades) if trades else 0.0,
                expectancy_r=(total / trades) if trades else 0.0,
                max_drawdown_r=max(i.max_drawdown_r for i in items),
            )
        )

    window = f"{args.days:g} days, {len(args.symbols)} symbols"
    print(render(combined, window=window))

    # Pooled the same way and for the same reason: the question is whether the
    # theory knows something, and four thin per-symbol comparisons answer it
    # four times badly instead of once well.
    if against_chance:
        pooled: dict[str, list] = {}
        for item in against_chance:
            pooled.setdefault(item.real.playbook, []).append(item)
        merged_comparisons = []
        for name, items in pooled.items():
            real = next(row for row in combined if row.playbook == name)
            weights = [item.real.trades for item in items]
            total = sum(weights) or 1
            merged_comparisons.append(
                type(items[0])(
                    real=real,
                    flip_win_rate=sum(
                        i.flip_win_rate * w for i, w in zip(items, weights, strict=True)
                    )
                    / total,
                    flip_expectancy_r=sum(
                        i.flip_expectancy_r * w for i, w in zip(items, weights, strict=True)
                    )
                    / total,
                    flip_best_r=max(i.flip_best_r for i in items),
                    flip_worst_r=min(i.flip_worst_r for i in items),
                )
            )
        print(render_comparison(merged_comparisons, window=window))

    if target_rows:
        # Pooled across symbols per (theory, target), same reasoning again.
        by_key: dict[tuple[str, float], list] = {}
        for row in target_rows:
            by_key.setdefault((row.playbook, row.r_multiple), []).append(row)
        pooled_targets = []
        for (name, multiple), items in by_key.items():
            trades = sum(i.trades for i in items) or 1
            pooled_targets.append(
                type(items[0])(
                    playbook=name,
                    r_multiple=multiple,
                    trades=sum(i.trades for i in items),
                    win_rate=sum(i.win_rate * i.trades for i in items) / trades,
                    expectancy_r=sum(i.expectancy_r * i.trades for i in items) / trades,
                    coin_expectancy_r=sum(i.coin_expectancy_r * i.trades for i in items) / trades,
                )
            )
        print(render_targets(pooled_targets, window=window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
