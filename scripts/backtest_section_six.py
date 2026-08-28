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
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

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


def sweep(
    orders: list[BacktestOrder], minute: pd.DataFrame, gates: tuple[float, ...]
) -> list[tuple]:
    """The same setups, re-judged at every cost gate, in one pass.

    WHY A SWEEP AND NOT A RUN PER GATE. The expensive half of this script is
    the detector walking every closed M1 bar; the gate is a filter applied to
    what comes out of it. A stricter gate is a strict SUBSET of a looser one,
    so building the setups once with the gate switched off and filtering
    afterwards gives the same answer as re-running, for one detector pass
    instead of eight.

    The non-overlap selection IS redone per gate, because it must be: removing
    setups frees the minutes they occupied, and a later setup that was skipped
    for overlapping a rejected one is genuinely available at the stricter gate.
    Filtering the TRADES instead of the ORDERS would answer a different
    question and would answer it optimistically.

    WHAT THE CURVE IS FOR. The first run of this script returned -0.304R a
    trade over 1,681 trades, and the decomposition said the entry itself is
    worth about +0.03R on gold while the round trip costs 0.235R. Cost and
    gate are the same number seen from two sides -- target is 1.4 spans and
    stop is 1.0 span, so one spread is 1.4/gate of R -- and the only question
    left is whether any gate exists where what survives still has an edge.
    """
    backtester = PessimisticBacktester()
    rows: list[tuple] = []
    for gate in gates:
        kept = [
            order
            for order in orders
            if order.spread > 0 and abs(order.take_profit - order.entry) / order.spread >= gate
        ]
        if not kept:
            rows.append((gate, 0, 0, 0.0, 0.0, 0.0))
            continue
        result = backtester.run_non_overlapping(minute, kept)
        if not result.sample_size:
            rows.append((gate, len(kept), 0, 0.0, 0.0, 0.0))
            continue
        rows.append(
            (
                gate,
                len(kept),
                result.sample_size,
                result.win_rate,
                result.expectancy_r,
                result.total_r,
            )
        )
    return rows


def payoff_sweep(
    orders: list[BacktestOrder], minute: pd.DataFrame, ratios: tuple[float, ...]
) -> list[tuple]:
    """The same entries, re-priced at every payoff ratio, against chance.

    THE COST SWEEP CANNOT ANSWER WHETHER THE ENTRY READS ANYTHING. It varies
    the gate and the gate only moves the cost: across gates 5 to 40 the win
    rate sat at 42% and never budged, while the cost band fell from 28% of R
    to 3.5%. So the whole difference between -0.332R and -0.064R was the
    spread, and the detector's opinion contributed nothing measurable either
    way.

    AND 42% IS EXACTLY WHAT CHANCE PAYS HERE. The lane targets 1.4 spans
    against a 1.0 span stop, so a coin flip resolves in its favour
    1/(1+1.4) = 41.7% of the time. The entry beat that by three tenths of a
    percentage point. The banner already carried the same conclusion in R --
    "it is worth +0.03R" -- without anything anywhere putting it beside the
    number it has to beat.

    So this sweep varies the RATIO instead of the gate, and prints the
    break-even rate next to the achieved one. If the entry sees anything at
    all, there is a ratio where the gap is positive and larger than noise. If
    the gap is flat around zero at every ratio, the detector is a coin and no
    exit rule, no cost gate and no lot size repairs that -- which is a finding
    worth having outright rather than after another month of live trades.

    Every other term is held: same moments, same stops, same non-overlap
    selection. Only the distance to the target moves.
    """
    backtester = PessimisticBacktester()
    rows: list[tuple] = []
    for ratio in ratios:
        repriced = [
            replace(
                order,
                take_profit=order.entry + int(order.direction) * ratio * order.risk,
            )
            for order in orders
            if order.risk > 0
        ]
        if not repriced:
            rows.append((ratio, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        result = backtester.run_non_overlapping(minute, repriced)
        if not result.sample_size:
            rows.append((ratio, 0, 0.0, 0.0, 0.0, 0.0))
            continue

        # ONLY THE TRADES THAT ACTUALLY REACHED A BARRIER, and this is the
        # whole correctness of the table.
        #
        # `1/(1+ratio)` is the first-touch probability for a driftless walk
        # between a stop at -1 and a target at +ratio. It says nothing about a
        # trade that reached neither and was closed by the clock. `win_rate`
        # counts any positive net R as a win, TIME exits included -- so
        # comparing the two mixes a first-touch model with a population that
        # did not touch anything.
        #
        # A CONTROL RUN CAUGHT IT. Fed a pure random walk, the first version of
        # this reported edges of +10% to +17% at every ratio and printed
        # "Positive edge +16.7%". A tool built to say whether a detector beats
        # chance manufactured an edge out of noise, which is worse than the
        # question going unanswered.
        resolved = [
            trade
            for trade in result.trades
            if trade.outcome.startswith("TP") or trade.outcome.startswith("SL")
        ]
        if not resolved:
            rows.append((ratio, 0, 0.0, 1.0 / (1.0 + ratio), 0.0, 0.0))
            continue
        hits = sum(1 for trade in resolved if trade.outcome.startswith("TP"))
        won = hits / len(resolved)
        chance = 1.0 / (1.0 + ratio)
        rows.append(
            (
                ratio,
                len(resolved),
                won,
                chance,
                won - chance,
                float(np.mean([trade.net_r for trade in resolved])),
            )
        )
    return rows


def render_payoff(rows: list[tuple], window: str) -> str:
    lines = [
        "",
        "=" * 78,
        f"  DOES THE ENTRY READ ANYTHING AT ALL?  {window}",
        "=" * 78,
        "",
        "  The same entries and the same stops, with only the target moved. CHANCE is",
        "  what a coin flip resolves at for that ratio -- 1/(1+ratio) -- and EDGE is",
        "  how far the detector beat it. That column is the whole question: a cost",
        "  gate, an exit rule and a lot size all divide into an edge and none of them",
        "  creates one.",
        "",
        "  Only trades that REACHED a barrier are counted. One closed by the clock",
        "  touched neither and says nothing about a first-touch probability.",
        "",
        f"  {'target':>8}{'trades':>8}{'won':>7}{'chance':>9}{'edge':>9}"
        f"{'sigma':>8}{'per trade':>12}",
        "  " + "-" * 63,
    ]
    verdicts: list[tuple[float, float, int, float]] = []
    for ratio, trades, won, chance, edge, expectancy in rows:
        if not trades:
            lines.append(f"  {ratio:>7.1f}R{trades:>8}       (nothing reached a barrier)")
            continue
        sigma = _sigmas(won, chance, trades)
        verdicts.append((sigma, edge, trades, ratio))
        lines.append(
            f"  {ratio:>7.1f}R{trades:>8}{won:>6.0%}{chance:>9.1%}"
            f"{edge:>+8.1%}{sigma:>+8.1f}{expectancy:>+11.3f}R"
        )

    bar = _significance_bar(len(verdicts))
    lines.append("")
    lines.append("  SIGMA is the gap divided by its own standard error, and the bar is NOT two.")
    lines.append(f"  This table reports the best of {len(verdicts)} ratios, so a plain two-sigma")
    lines.append("  test fires on chance about once every ten runs -- which a control on a")
    lines.append(f"  random walk duly did, at +2.1. The bar for {len(verdicts)} comparisons is")
    lines.append(f"  {bar:.2f} sigma, and the widest ratios carry the fewest trades: exactly")
    lines.append("  where a reader most wants to believe a number.")
    lines.append("")
    if not verdicts:
        lines.append("  Nothing reached a barrier at any ratio.")
    else:
        sigma, edge, trades, ratio = max(verdicts)
        if sigma < bar:
            lines.append(
                f"  Best is {edge:+.1%} at {ratio:.1f}R over {trades} trades, {sigma:+.1f} sigma"
                f" against a bar of {bar:.2f}."
            )
            lines.append("  That is chance. No target, no stop and no cost gate turns a coin into")
            lines.append("  a strategy -- the ENTRY has to change, or the lane stays off.")
        else:
            lines.append(
                f"  {edge:+.1%} at {ratio:.1f}R over {trades} trades, {sigma:+.1f} sigma"
                f" against a bar of {bar:.2f}."
            )
            lines.append("  That is outside chance. Now read `per trade`: an edge can still lose")
            lines.append("  money once the round trip is paid, and that is a cost question.")
    lines.append("")
    return "\n".join(lines)


def _significance_bar(comparisons: int, alpha: float = 0.05) -> float:
    """How many sigma the BEST of `comparisons` ratios has to clear.

    NOT TWO, AND THE CONTROL RUN IS WHY. With a plain two-sigma test on four
    ratios, one random-walk seed out of four came back at +2.1 sigma and the
    table would have said "outside chance" about a coin. Four independent looks
    at a 5% test give roughly a one-in-ten chance that at least one fires, and
    this report exists specifically to be scanned for its best row.

    So the bar is corrected for how many times the question was asked --
    Bonferroni, the blunt version, deliberately: the ratios are correlated
    (they share entries and stops) so the true bar is somewhat lower than this,
    and erring toward "not proven" is the direction that costs money the
    account still has.

    At the six shipped ratios this is 2.64 sigma rather than 2.00. That is the
    difference between switching a lane back on because of a coin and leaving
    it off until something real turns up.
    """
    if comparisons <= 0:
        return float("inf")
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * comparisons))


def _sigmas(won: float, chance: float, trades: int) -> float:
    """How many standard errors the gap is, which is the only honest reading.

    THE CONTROL RUN CAUGHT THIS TWICE. Fed a pure random walk, the first
    version of this table reported edges of +10% to +17% and printed "Positive
    edge +16.7%": it compared a first-touch probability against a win rate that
    counted clock exits. Restricting to barrier-resolved trades brought that to
    +4% to +8% -- still positive, still on noise, because the standard error of
    a 33% rate over 34 samples is 8.1%.

    A percentage-point column with no error bar makes small samples look like
    discoveries, and the smallest samples sit at the widest ratios, which is
    exactly where a reader wants to believe one. A tool built to say whether a
    detector beats chance must not manufacture an edge out of noise; that is
    worse than leaving the question unanswered.
    """
    if trades <= 0 or chance <= 0.0 or chance >= 1.0:
        return 0.0
    error = (chance * (1.0 - chance) / trades) ** 0.5
    return (won - chance) / error if error > 0 else 0.0


def render_sweep(rows: list[tuple], window: str) -> str:
    lines = [
        "",
        "=" * 78,
        f"  IS THERE A GATE WHERE THIS PAYS?  {window}",
        "=" * 78,
        "",
        "  `minimum_target_spreads` raised step by step, everything else at live",
        "  settings. The cost column is not a separate measurement -- it IS the",
        "  gate: target is 1.4 spans and stop 1.0, so one spread is 1.4/gate of R.",
        "",
        f"  {'gate':>6}{'cost of R':>12}{'setups':>9}{'trades':>9}{'win':>7}"
        f"{'per trade':>12}{'total':>10}",
        "  " + "-" * 66,
    ]
    for gate, setups, trades, win, per, total in rows:
        cost = f"{1.4 / gate:.1%}" if gate > 0 else "-"
        if not trades:
            lines.append(f"  {gate:>6.0f}{cost:>12}{setups:>9}{0:>9}{'-':>7}{'-':>12}{'-':>10}")
            continue
        lines.append(
            f"  {gate:>6.0f}{cost:>12}{setups:>9}{trades:>9}{win:>6.0%}"
            f"{per:>+11.3f}R{total:>+9.1f}R"
        )
    lines.append("")
    best = [row for row in rows if row[2] and row[4] > 0]
    if best:
        gate, _, trades, _, per, _ = max(best, key=lambda row: row[4])
        lines.append(f"  Positive at gate {gate:.0f}: {per:+.3f}R over {trades} trades. That is")
        lines.append(f"  `minimum_target_spreads: {gate:.1f}` and `own_lane_enabled: true` again.")
        lines.append("  Read the trade count first: a gate leaving twenty trades a month has")
        lines.append("  measured almost nothing, whatever sign it carries.")
    else:
        lines.append("  No gate turns it positive. Cost is not the part that can be fixed --")
        lines.append("  an entry worth hundredths of R cannot carry a scalp's round trip.")
    lines.append("")
    return "\n".join(lines)


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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "also raise `minimum_target_spreads` step by step and report what "
            "survives at each. This is the run that says whether the lane can "
            "be switched back on."
        ),
    )
    parser.add_argument(
        "--payoff",
        action="store_true",
        help=(
            "instead of the cost gate, move the TARGET and print the achieved "
            "win rate beside the rate a coin flip pays at that ratio. The cost "
            "sweep cannot answer whether the entry reads anything; this can."
        ),
    )
    parser.add_argument(
        "--ratios",
        nargs="*",
        type=float,
        default=[0.5, 0.8, 1.0, 1.4, 2.0, 3.0],
        help="the target distances --payoff walks, in multiples of the stop.",
    )
    parser.add_argument(
        "--gates",
        nargs="*",
        type=float,
        default=[5, 7, 10, 14, 20, 28, 40, 56],
        help="the gates --sweep walks. Doubling roughly halves the cost each step.",
    )
    args = parser.parse_args(argv)

    settings = load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml")
    # Both sweeps need the full superset of setups, so the gate comes off
    # for either of them.
    if args.sweep or args.payoff:
        # THE GATE IS SWITCHED OFF FOR THE PASS, NOT LOWERED.
        #
        # The sweep needs the full superset of setups so every gate above can
        # be taken as a subset of it. Leaving the live gate in place would cap
        # the sweep at 7.0 and quietly report the same row eight times, which
        # looks exactly like a flat curve.
        settings = settings.model_copy(
            update={
                "analysis": settings.analysis.model_copy(
                    update={
                        "candle_momentum": settings.analysis.candle_momentum.model_copy(
                            update={"minimum_target_spreads": 0.0}
                        )
                    }
                )
            }
        )
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
    swept: list[tuple] = []
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
            if args.sweep or args.payoff:
                swept.append((symbol, orders, frames[Timeframe.M1]))
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

    if args.sweep:
        # POOLED ACROSS MARKETS, because the question is about the RULE and a
        # per-market table at eight gates is sixty-four rows nobody reads. The
        # per-market split is what a plain run already gives.
        gates = tuple(sorted(float(gate) for gate in args.gates if gate > 0))
        totals: list[tuple] = []
        for gate in gates:
            setups = trades = 0
            weighted = total = 0.0
            wins = 0.0
            for _symbol, orders, minute in swept:
                row = sweep(orders, minute, (gate,))[0]
                setups += row[1]
                trades += row[2]
                wins += row[3] * row[2]
                weighted += row[4] * row[2]
                total += row[5]
            per = weighted / trades if trades else 0.0
            totals.append((gate, setups, trades, wins / trades if trades else 0.0, per, total))
        print(render_sweep(totals, f"{args.days:.0f} days"))
        return 0

    if args.payoff:
        ratios = tuple(sorted(float(r) for r in args.ratios if r > 0))
        totals = []
        for ratio in ratios:
            trades = 0
            wins = weighted = 0.0
            for _symbol, orders, minute in swept:
                row = payoff_sweep(orders, minute, (ratio,))[0]
                trades += row[1]
                wins += row[2] * row[1]
                weighted += row[5] * row[1]
            won = wins / trades if trades else 0.0
            chance = 1.0 / (1.0 + ratio)
            totals.append(
                (ratio, trades, won, chance, won - chance, weighted / trades if trades else 0.0)
            )
        print(render_payoff(totals, f"{args.days:.0f} days"))
        return 0

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
