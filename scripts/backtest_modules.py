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
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis import ConfluenceEngine
from backtesting.engine import BacktestOrder, PessimisticBacktester
from backtesting.exit_study import (
    give_back_curve,
    render_give_back,
    render_policies,
    study,
    sweep_policies,
)
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

#: The same sentence for the detectors, in their own terms. The shared renderer
#: defaults to the playbook wording, which names five pattern rules none of
#: these modules contain.
_MODULE_CONCLUSION = (
    "  These are the detectors the live account trades on, and on this sample\n"
    "  their direction calls are worth about what a coin is worth. Read the\n"
    "  avg win and avg loss columns above before deciding what to do: a\n"
    "  detector whose winners are smaller than its losers is not failing to\n"
    "  read the market, it is failing to get paid for reading it, and those\n"
    "  are opposite repairs."
)


def build_engine(settings):  # type: ignore[no-untyped-def]
    """Every detector, at the settings the account is actually running.

    THROUGH THE RUNNER'S OWN LIST, not a copy. This function used to name its
    fifteen modules by hand, and the account builds eighteen -- so the three it
    had drifted out of sync with were graded by nothing: `drift_burst`,
    `basket_divergence`, and `candle_momentum`, which is section six. A report
    headed "which detector actually makes money" simply did not mention them.

    `build_analysis_modules` says in its own docstring that it exists so this
    script stops keeping a second copy. It was written and this file never
    adopted it. Two lists of the same thing disagree eventually; the only
    question is which one you find out about first, and here it was the one
    that decides what to believe about the other.
    """
    from runner.service import build_analysis_modules

    return ConfluenceEngine(build_analysis_modules(settings), settings.analysis.confluence)


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
    """One detector's record over the replayed window.

    HOW THE TRADES ENDED IS PART OF THE RECORD, and it used to be dropped.
    A table that says only "negative" cannot be acted on: a detector losing
    because its stops are inside the noise and a detector losing because its
    targets are never reached are the same row and opposite repairs. The
    backtester already labels every exit SL, TP or TIME; nothing read it.
    """

    module: str
    proposals: int
    trades: int
    total_r: float
    win_rate: float
    expectancy_r: float
    max_drawdown_r: float
    #: Share of closed trades ending each way. TIME is the time limit, which is
    #: neither the plan working nor the plan failing -- it is the plan not
    #: resolving, and a population dominated by it is being graded on drift.
    tp_share: float = 0.0
    sl_share: float = 0.0
    time_share: float = 0.0
    #: What a winner and a loser are actually worth. The pair says whether a
    #: win rate above 50% can pay for itself at all.
    average_win_r: float = 0.0
    average_loss_r: float = 0.0

    def row(self) -> str:
        return (
            f"  {self.module:<22}{self.trades:>7}"
            f"{self.win_rate:>7.0%}{self.expectancy_r:>+9.3f}R"
            f"{self.average_win_r:>+9.2f}R{self.average_loss_r:>+9.2f}R"
            f"{self.tp_share:>7.0%}{self.sl_share:>6.0%}{self.time_share:>7.0%}"
            f"{self.total_r:>+9.2f}R"
        )


HEADER = (
    f"  {'module':<22}{'trades':>7}{'win':>7}{'per trade':>10}"
    f"{'avg win':>10}{'avg loss':>10}{'TP':>7}{'SL':>6}{'TIME':>7}{'total':>10}"
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
        outcomes: dict[str, int] = defaultdict(int)
        won_r: list[float] = []
        lost_r: list[float] = []
        for symbol, group in by_symbol.items():
            frame = frames.get(symbol)
            if frame is None:
                continue
            result = backtester.run_non_overlapping(frame, group)
            trades += result.sample_size
            total_r += result.total_r
            wins += round(result.win_rate * result.sample_size)
            worst = max(worst, result.max_drawdown_r)
            for trade in result.trades:
                # SL_FIRST_AMBIGUOUS is a bar that touched both levels and is
                # resolved as the stop, so it belongs with the stops.
                outcomes["SL" if trade.outcome.startswith("SL") else trade.outcome] += 1
                (won_r if trade.net_r > 0 else lost_r).append(trade.net_r)
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
                tp_share=outcomes["TP"] / trades,
                sl_share=outcomes["SL"] / trades,
                time_share=outcomes["TIME"] / trades,
                average_win_r=(sum(won_r) / len(won_r)) if won_r else 0.0,
                average_loss_r=(sum(lost_r) / len(lost_r)) if lost_r else 0.0,
            )
        )
    return sorted(evidence, key=lambda item: item.expectancy_r)


def render(title: str, evidence: list[ModuleEvidence], note: str) -> str:
    lines = [f"\n{title}", f"  {note}", "", HEADER]
    lines.extend(item.row() for item in evidence)
    if not evidence:
        lines.append("  (nothing reached a closed trade)")
    return "\n".join(lines)


#: Where the round trip stops being affordable. The gate that admits these
#: trades live is `max_spread_share_of_stop`, which ships at 0.15 and 0.20 --
#: a spread allowed to be a fifth of the stop is a fifth of R paid before the
#: market does anything. The measured gap to break even on this account is
#: 0.094R per trade, which is SMALLER than that, so the cost is not a rounding
#: error on the finding: it may be most of it.
#:
#: Edges chosen around the shipped gate so the table brackets it rather than
#: agreeing with it.
_COST_BANDS = ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 1.00))


def render_cost_bands(orders: list[BacktestOrder], frames, backtester) -> str:  # type: ignore[no-untyped-def]
    """What every trade returned, bucketed by what its round trip cost.

    THE QUESTION NO TABLE HERE ASKED. Every detector loses between 0.063R and
    0.312R a trade, and the spread on the stop widths this account trades is
    itself worth around a tenth of R. If the cheap end of this table is
    positive and the expensive end is not, then the detectors are not the
    thing that needs replacing -- the gate that lets an expensive trade
    through is, and that is one number in the overlay rather than a rewrite.

    If instead every band loses, the cost is not the story and the selection
    really is worthless. Either answer is worth more than another opinion.
    """
    priced = [order for order in orders if order.spread > 0]
    skipped = len(orders) - len(priced)
    rows = []
    for low, high in _COST_BANDS:
        band = [
            order
            for order in priced
            if low <= order.spread / max(1e-12, abs(order.entry - order.stop_loss)) < high
        ]
        if not band:
            continue
        by_symbol: dict[str, list[BacktestOrder]] = defaultdict(list)
        for order in band:
            by_symbol[order.symbol].append(order)
        trades = total = wins = 0.0, 0.0, 0.0
        trades, total, wins = 0, 0.0, 0
        for symbol, group in by_symbol.items():
            frame = frames.get(symbol)
            if frame is None:
                continue
            result = backtester.run_non_overlapping(frame, group)
            trades += result.sample_size
            total += result.total_r
            wins += round(result.win_rate * result.sample_size)
        if trades:
            rows.append((low, high, trades, wins / trades, total / trades, total))

    lines = [
        "",
        "=" * 78,
        "  WHAT THE ROUND TRIP COSTS, AND WHETHER IT IS THE WHOLE STORY",
        "=" * 78,
        "",
        "  Every trade, bucketed by the spread as a share of its own stop. The live",
        "  gate `max_spread_share_of_stop` ships at 0.15-0.20, so the bottom band is",
        "  what the account is allowed to take today.",
        "",
        f"  {'spread / stop':<18}{'trades':>8}{'win':>7}{'per trade':>12}{'total':>11}",
        "  " + "-" * 56,
    ]
    for low, high, trades, win, per, total in rows:
        label = f"{low:.0%} - {high:.0%}" if high < 1 else f"over {low:.0%}"
        lines.append(f"  {label:<18}{trades:>8}{win:>7.0%}{per:>+11.3f}R{total:>+10.2f}R")
    if not rows:
        lines.append("  (no proposal carried a recorded spread)")
    if skipped:
        lines.append("")
        lines.append(f"  {skipped} proposals carried no recorded spread and are left out rather")
        lines.append("  than counted as free.")
    lines.append("")
    return "\n".join(lines)


def by_module_symbol(orders: list[BacktestOrder]) -> dict[str, list[BacktestOrder]]:
    """Every order the detectors proposed, grouped by symbol.

    Pooled across detectors on purpose: the exit question is about what a
    position does after it is open, and a stop that should move at 0.3R does
    not care which reader argued for the entry.
    """
    grouped: dict[str, list[BacktestOrder]] = defaultdict(list)
    for order in orders:
        grouped[order.symbol].append(order)
    return grouped


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
        "--by-regime",
        action="store_true",
        help="also split every detector by what `market_regime` read at the "
        "deciding bar. Answers whether a detector loses everywhere or only "
        "outside the conditions it was written for",
    )
    parser.add_argument(
        "--exits",
        action="store_true",
        help="walk every position bar by bar and sweep exit rules over them. "
        "Answers the question the tables above raise but cannot settle: these "
        "trades win 45-57%% of the time and still lose, so would banking "
        "earlier have paid?",
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

    # SAY WHAT WAS CHARGED. A table of R-multiples means nothing without it, and
    # this run has already been read twice without anyone being able to tell:
    # the spread was silently missing until it was noticed that both replays
    # computed it to build a tick and then dropped it, so two tables produced
    # hours apart were not comparable and nothing on either said so.
    assumptions = backtester.assumptions
    charged = [
        f"commission {assumptions.round_trip_commission_bps:.2f} bps",
        f"exit slippage {assumptions.exit_slippage_bps:.2f} bps",
        f"entry slippage {assumptions.entry_slippage_bps:.2f} bps",
    ]
    with_spread = sum(1 for order in everything if order.spread > 0)
    print(
        f"\n  Costs charged: {', '.join(charged)}, and the broker's own recorded "
        f"spread on {with_spread:,} of {len(everything):,} proposals."
    )
    if with_spread < len(everything):
        print(
            "  Proposals without a recorded spread are filled at the mid on both "
            "sides,\n  so their rows are a LOWER BOUND on what the round trip cost."
        )

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

    if args.by_regime:
        # ONE AVERAGE HIDES TWO ANSWERS.
        #
        # "This detector loses money" and "this detector loses money
        # EVERYWHERE" are different findings and they have different remedies:
        # the first says switch it off, the second says only let it fire where
        # it works. `trend_momentum` was switched off on -0.251R over 148
        # trades without anyone asking which of the two it was — and a trend
        # follower being measured across ranges and transitions is exactly the
        # case where the average is the wrong statistic.
        #
        # The regime was computed on every one of those decisions and thrown
        # away at the order boundary. It is carried now, so the question is
        # answerable instead of arguable.
        by_regime: dict[str, dict[str, list[BacktestOrder]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for order in everything:
            regime = order.regime or "unrecorded"
            for module in order.modules:
                by_regime[regime][module].append(order)
        for regime, groups in sorted(by_regime.items()):
            print(
                render(
                    f"WHEN PRESENT, IN {regime.upper()}",
                    evidence_for(groups, execution_frames, backtester),
                    "A row here is only worth acting on with enough trades behind "
                    "it. Splitting five ways divides the sample five ways.",
                )
            )

    print(render_cost_bands(everything, execution_frames, backtester))

    if args.exits:
        # THE QUESTION THE TABLES ABOVE RAISE AND CANNOT SETTLE.
        #
        # Every detector wins 45-57% of its trades and still loses money, so
        # the winners are smaller than the losers, and the backtester models
        # NO management at all -- fixed stop, fixed target, time limit. The
        # obvious objection is the right one: that is what a profit lock and a
        # ratcheting stop are for. This measures whether that is true on these
        # trades instead of assuming it either way.
        #
        # `HOLD_EVERYTHING` is in the sweep as the control, so every row is
        # read against doing nothing rather than against a hope.
        #
        # This has existed since the playbook study and was only ever pointed
        # at the five dead playbooks. It had never been run on the detectors
        # the account actually trades.
        walked = []
        for symbol, group in sorted(by_module_symbol(everything).items()):
            frame = execution_frames.get(symbol)
            if frame is not None:
                walked.extend(study(group, frame))
        if walked:
            print(render_give_back(give_back_curve(walked)))
            print(render_policies(sweep_policies(walked)))
        else:
            print("\n  Nothing reached a closed position to walk.\n")

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
            # NAME THE SYMBOL. The comparison runs per symbol, so a detector
            # with enough lone trades on four of them produced four rows all
            # labelled `drift_continuation` and nothing else -- which reads as
            # a duplicated row rather than as four separate readings.
            for item in compare_to_chance(group, frame, worth_reading, backtester=backtester):
                named = f"{item.real.playbook[:11]} {symbol[:6]}"
                comparisons.append(replace(item, real=replace(item.real, playbook=named)))
        if comparisons:
            print(
                render_comparison(
                    comparisons,
                    window=f"{args.days:.0f} days, lone detectors",
                    conclusion=_MODULE_CONCLUSION,
                )
            )
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
