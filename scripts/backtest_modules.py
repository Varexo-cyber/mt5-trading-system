"""Which detector actually makes money, and which one has been paying for it.

The playbooks got this treatment and it killed three of them: `momentum_scalp`
at -0.561R a trade over 307 trades, `range_fade` at -0.174R over 898,
`range_break` at -76.28R. Not one of the five beat a coin flip taking the same
moments with the same stops and targets.

The eight CONFLUENCE modules have never had it. `trend_momentum`,
`drift_continuation`, `fast_ema_cross`, `impulse_break`, `liquidity_sweep`,
`ema_pullback_resume`, `m1_micro_breakout` and `market_structure` are what the
live account actually trades on, and the only grading they have is
`MIN_TRADES_TO_GRADE_A_MODULE`, which needs twenty closed trades per module.
`verify_brain.py` reports no detector block at all: not one module has reached
twenty. So every argument about which detector to trust, which to weight up and
which to switch off has been made on two or three trades at a time.

This closes that. It replays the live engine over months of real bars, records
which modules were behind every proposal, and reports what each one returned.
Nothing is written, no order is sent, no API is called; the bars are already on
the broker's server.

    python scripts/backtest_modules.py                      # 120 days, majors
    python scripts/backtest_modules.py --days 240
    python scripts/backtest_modules.py --symbols EURUSD.i XAUUSD US30
    python scripts/backtest_modules.py --stride 2           # faster, coarser

THREE TABLES, and the second is the one to read first.

    WHEN PRESENT   every trade a module was part of. Generous, because a good
                   module rides along with bad ones and vice versa.
    ALONE          trades where this module was the ONLY one pointing that way.
                   This is the module's own opinion with nothing to hide behind,
                   and it is exactly the population the 0.65 lone-module floor
                   is currently refusing ~2,400 times an hour.
    AGAINST CHANCE a coin flip taking the same moments, the same stops and the
                   same targets. A module that cannot beat it is not analysis,
                   it is a way of choosing when to pay the spread.

WHAT THIS IS FOR. Killing a module that loses over thousands of samples is
worth more, and is far more certain, than any rule that could be added in its
place. It is not for tuning: nothing here searches a parameter space, and the
configuration it runs is the configuration the account is running.

WHAT IT CANNOT TELL YOU. Attribution is not causation — a module that appears
in profitable trades may be a passenger. The ALONE table is the closest thing
to a clean read and it is also the smallest sample, so check the trade count
before believing any row of it.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis import (
    ConfluenceEngine,
    DriftContinuation,
    EmaPullbackResume,
    FastEmaCross,
    ImpulseBreak,
    LevelReaction,
    LiquiditySweep,
    M1MicroBreakout,
    MarketRegime,
    MarketStructure,
    MeanReversion,
    Seasonality,
    SessionBreakout,
    TrendMomentum,
    VolatilityRegime,
    VolatilitySqueeze,
)
from backtesting.engine import BacktestOrder, PessimisticBacktester
from backtesting.playbook_replay import (
    compare_to_chance,
    evidence_by_playbook,
    render_comparison,
)
from backtesting.replay import REPLAY_TIMEFRAMES, HistoricalContextReplay
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.data_manager import DataManager
from core.mt5_connector import MT5Connector
from core.types import Timeframe

DEFAULT_SYMBOLS = ("EURUSD.i", "GBPUSD.i", "USDJPY.i", "AUDUSD.i", "XAUUSD")


def build_engine(settings):  # type: ignore[no-untyped-def]
    """Every detector, at the settings the account is actually running."""
    return ConfluenceEngine(
        [
            MarketStructure(settings.analysis.market_structure),
            TrendMomentum(settings.analysis.trend_momentum),
            LiquiditySweep(settings.analysis.liquidity_sweep),
            LevelReaction(settings.analysis.level_reaction),
            VolatilityRegime(settings.analysis.volatility_regime),
            MarketRegime(settings.analysis.market_regime),
            DriftContinuation(settings.analysis.drift_continuation),
            FastEmaCross(settings.analysis.fast_ema_cross),
            ImpulseBreak(settings.analysis.impulse_break),
            EmaPullbackResume(settings.analysis.ema_pullback_resume),
            M1MicroBreakout(settings.analysis.m1_micro_breakout),
            VolatilitySqueeze(settings.analysis.volatility_squeeze),
            MeanReversion(settings.analysis.mean_reversion),
            SessionBreakout(settings.analysis.session_breakout),
            Seasonality(settings.analysis.seasonality),
        ],
        settings.analysis.confluence,
    )


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


@dataclass(frozen=True, slots=True)
class ModuleEvidence:
    """One detector's record over the replayed window."""

    module: str
    proposals: int
    trades: int
    total_r: float
    win_rate: float
    expectancy_r: float
    max_drawdown_r: float

    def row(self) -> str:
        return (
            f"  {self.module:<22}{self.proposals:>9}{self.trades:>8}"
            f"{self.win_rate:>8.0%}{self.total_r:>+10.2f}R{self.expectancy_r:>+9.3f}R"
            f"{self.max_drawdown_r:>10.2f}R"
        )


HEADER = (
    f"  {'module':<22}{'proposals':>9}{'trades':>8}{'win':>8}"
    f"{'total':>11}{'per trade':>10}{'max dd':>10}"
)


def evidence_for(
    groups: dict[str, list[BacktestOrder]],
    frames: dict[str, pd.DataFrame],
    backtester: PessimisticBacktester,
) -> list[ModuleEvidence]:
    """Replay each detector's own orders, per symbol, and pool the results.

    Per symbol because a non-overlapping replay is about one instrument's slot
    being occupied; pooling the ORDERS across symbols first would let EURUSD
    suppress a XAUUSD entry that the live account would happily have taken
    alongside it.
    """
    evidence: list[ModuleEvidence] = []
    for module, orders in sorted(groups.items()):
        by_symbol: dict[str, list[BacktestOrder]] = defaultdict(list)
        for order in orders:
            by_symbol[order.symbol].append(order)
        trades = 0
        total_r = 0.0
        wins = 0
        worst = 0.0
        for symbol, group in by_symbol.items():
            frame = frames.get(symbol)
            if frame is None:
                continue
            result = backtester.run_non_overlapping(frame, group)
            trades += result.sample_size
            total_r += result.total_r
            wins += round(result.win_rate * result.sample_size)
            worst = max(worst, result.max_drawdown_r)
        if trades == 0:
            continue
        evidence.append(
            ModuleEvidence(
                module=module,
                proposals=len(orders),
                trades=trades,
                total_r=total_r,
                win_rate=wins / trades,
                expectancy_r=total_r / trades,
                max_drawdown_r=worst,
            )
        )
    return sorted(evidence, key=lambda item: item.expectancy_r)


def render(title: str, evidence: list[ModuleEvidence], note: str) -> str:
    lines = [f"\n{title}", f"  {note}", "", HEADER]
    lines.extend(item.row() for item in evidence)
    if not evidence:
        lines.append("  (nothing reached a closed trade)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=120.0, help="how far back to replay")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="decide every Nth H1 bar. Raise it to trade accuracy for speed",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the coin-flip control. It is the only part that says whether "
        "any of this is analysis rather than a way of paying the spread",
    )
    args = parser.parse_args(argv)

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    replay = HistoricalContextReplay(build_engine(settings), decision_stride_bars=args.stride)

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
    everything: list[BacktestOrder] = []
    execution_frames: dict[str, pd.DataFrame] = {}
    try:
        for symbol in args.symbols:
            print(f"  replaying {symbol} …", flush=True)
            try:
                spec = connector.spec(symbol)
                frames = history(connector, symbol, start, end)
                orders = replay.orders(symbol, frames, point=spec.point, start=start, end=end)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not end the run
                print(f"    skipped: {type(exc).__name__}: {exc}")
                continue
            print(f"    {len(orders)} proposals")
            if not orders:
                continue
            execution_frames[symbol] = frames[Timeframe.M5]
            everything.extend(orders)
    finally:
        with contextlib.suppress(Exception):
            connector.disconnect()

    if not everything:
        print("\nNo proposals in the window. Nothing to measure.")
        return 1

    backtester = PessimisticBacktester()

    present: dict[str, list[BacktestOrder]] = defaultdict(list)
    alone: dict[str, list[BacktestOrder]] = defaultdict(list)
    for order in everything:
        for module in order.modules:
            present[module].append(order)
        if len(order.modules) == 1:
            alone[order.modules[0]].append(order)

    print(
        render(
            "WHEN PRESENT — every trade this detector was part of",
            evidence_for(present, execution_frames, backtester),
            "Generous: a good detector rides along with bad ones and vice versa.",
        )
    )
    print(
        render(
            "ALONE — this detector was the only one pointing that way",
            evidence_for(alone, execution_frames, backtester),
            "The module's own opinion with nothing to hide behind. This is the "
            "population the 0.65 lone-module floor refuses. Check the trade count "
            "before believing a row.",
        )
    )

    if not args.no_baseline:
        # Run on the ALONE population, where `modules[0]` is the module and the
        # grouping the comparison already does is the grouping wanted. A trade
        # with four detectors behind it cannot be attributed to any one of them
        # against a coin, so pitting the blended population against chance
        # would answer a question nobody asked.
        by_symbol: dict[str, list[BacktestOrder]] = defaultdict(list)
        for orders in alone.values():
            for order in orders:
                by_symbol[order.symbol].append(order)
        comparisons = []
        for symbol, group in sorted(by_symbol.items()):
            frame = execution_frames.get(symbol)
            if frame is None:
                continue
            evidence = evidence_by_playbook(group, frame, backtester)
            worth_reading = [item for item in evidence if item.trades >= 20]
            if not worth_reading:
                continue
            comparisons.extend(
                compare_to_chance(group, frame, worth_reading, backtester=backtester)
            )
        if comparisons:
            print(render_comparison(comparisons, window=f"{args.days:.0f} days, lone detectors"))
        else:
            print(
                "\n  No detector reached 20 lone trades on any one symbol. Widen "
                "--days or --symbols before reading anything into the tables above."
            )

    print(
        "\nA detector that is negative over a real sample should be switched off, "
        "and doing that is worth more than any rule that could be added in its place."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
