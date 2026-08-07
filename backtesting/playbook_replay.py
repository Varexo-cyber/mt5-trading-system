"""Run the five theories over history, so somebody knows whether they work.

`HistoricalContextReplay` next door drives the confluence engine — the swing
strategy — and has done since before the playbooks existed. The playbooks were
never wired into it, so `momentum_scalp`, `range_fade`, `range_break`,
`failed_break` and `trend_pullback` have never been tested against a single
day of price history. The live account is betting on all five.

Nobody knows whether `range_fade` is positive over three thousand samples or a
coin flip that pays commission. Not the operator, not the system, not me. That
is the gap this closes, and it closes it for nothing: the bars are already on
the broker's server and no API call is involved.

WHAT THIS IS FOR, AND WHAT IT IS NOT. The question is "is this positive at
all", not "what parameters make this look best". The second question is how
backtests lie: enough knobs turned against enough history will fit any noise.
Nothing here searches a parameter space, and the configuration it runs is the
configuration the account is actually running.

Its most valuable outcome is a theory it can kill. Deleting a playbook that
loses over five thousand samples is worth more, and is far more certain, than
any rule anybody could add.

    from backtesting.playbook_replay import PlaybookReplay
    replay = PlaybookReplay(engine)
    orders = replay.orders("EURUSD.i", frames, point=0.00001, start=..., end=...)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.playbooks import PlaybookEngine
from backtesting.engine import BacktestOrder, PessimisticBacktester
from core.types import Direction, MarketContext, Series, Tick, Timeframe, TradingMode

#: What a playbook may read. M1 is absent deliberately — none of the five uses
#: it, and carrying it would double the slicing cost of every decision for a
#: series nothing looks at.
REPLAY_TIMEFRAMES = (Timeframe.H1, Timeframe.M15, Timeframe.M5)

#: The timeframe decisions are taken on. M5, because that is the fastest thing
#: any of the theories reads and stepping slower would skip the bar a scalp
#: fires on.
DECISION_TIMEFRAME = Timeframe.M5


@dataclass(frozen=True, slots=True)
class PlaybookEvidence:
    """One theory's record over the replayed window."""

    playbook: str
    proposals: int
    trades: int
    total_r: float
    win_rate: float
    expectancy_r: float
    max_drawdown_r: float

    def row(self) -> str:
        return (
            f"  {self.playbook:<18}{self.proposals:>10}{self.trades:>8}"
            f"{self.win_rate:>8.0%}{self.total_r:>+10.2f}R{self.expectancy_r:>+9.3f}R"
            f"{self.max_drawdown_r:>10.2f}R"
        )


class PlaybookReplay:
    """Recreate what each theory could know at each closed M5 bar.

    Look-ahead safety is structural rather than checked: a timeframe's bar is
    only visible once its own close time has passed, so a decision at 10:05
    sees the M15 bar that closed at 10:00 and not the one still forming. Get
    that wrong and every result is a fiction that cannot be detected by
    looking at it.
    """

    def __init__(
        self,
        engine: PlaybookEngine,
        *,
        history_bars: int = 300,
        decision_stride_bars: int = 1,
    ) -> None:
        if history_bars < 120:
            raise ValueError("history_bars must be at least 120")
        if decision_stride_bars < 1:
            raise ValueError("decision_stride_bars must be positive")
        self.engine = engine
        self.history_bars = history_bars
        self.decision_stride_bars = decision_stride_bars

    def orders(
        self,
        symbol: str,
        frames: dict[Timeframe, pd.DataFrame],
        *,
        point: float,
        start: datetime,
        end: datetime,
        min_conviction: float = 0.0,
    ) -> list[BacktestOrder]:
        """Every proposal the theories would have made, as replayable orders."""
        missing = set(REPLAY_TIMEFRAMES) - set(frames)
        if missing:
            raise ValueError(f"replay missing timeframes: {sorted(tf.value for tf in missing)}")

        decisions = frames[DECISION_TIMEFRAME]
        closed_at = decisions.index + DECISION_TIMEFRAME.duration
        eligible = decisions[(closed_at >= start) & (closed_at < end)]

        orders: list[BacktestOrder] = []
        for sequence, opened_at in enumerate(eligible.index):
            if sequence % self.decision_stride_bars:
                continue
            decided_at = (opened_at + DECISION_TIMEFRAME.duration).to_pydatetime()
            context = self._context(symbol, frames, decided_at, point)
            if context is None:
                continue
            verdict = self.engine.evaluate(context, TradingMode.BACKTEST)
            # The live engine stands every theory down when two disagree on
            # direction, and a replay that skipped that would be measuring a
            # system nobody runs.
            if verdict.conflict or verdict.best is None:
                continue
            play = verdict.best
            if play.conviction < min_conviction:
                continue
            orders.append(
                BacktestOrder(
                    symbol=symbol,
                    decided_at=decided_at,
                    direction=play.direction,
                    entry=play.entry,
                    stop_loss=play.stop_loss,
                    take_profit=play.take_profit,
                    score=play.conviction,
                    confidence=play.conviction / 100.0,
                    # The engine's `modules` field carries whatever produced the
                    # order, so the per-theory split below is free.
                    modules=(play.playbook,),
                )
            )
        return orders

    def _context(
        self,
        symbol: str,
        frames: dict[Timeframe, pd.DataFrame],
        decided_at: datetime,
        point: float,
    ) -> MarketContext | None:
        series: dict[Timeframe, Series] = {}
        for timeframe in REPLAY_TIMEFRAMES:
            frame = frames[timeframe]
            # "Every bar whose close time has passed" — the same rule as a
            # boolean mask over `index + duration <= decided_at`, found by
            # binary search instead of by comparing every row. Identical
            # output, and the mask was a fifth of the total runtime because it
            # walked twenty six thousand rows on every one of eighteen
            # thousand decisions.
            end = frame.index.searchsorted(pd.Timestamp(decided_at) - timeframe.duration, "right")
            if end < 120:
                return None
            available = frame.iloc[max(0, end - self.history_bars) : end]
            series[timeframe] = Series(symbol, timeframe, available, decided_at)

        executable = series[DECISION_TIMEFRAME].df.iloc[-1]
        mid = float(executable["close"])
        # The broker's own recorded spread for that bar, not an assumption. On
        # a ten-pip stop the spread decides whether the trade was ever worth
        # taking, and inventing one would quietly answer that question.
        spread = max(float(executable.get("spread", 0.0)), 0.0) * point
        tick = Tick(symbol, decided_at, bid=mid - spread / 2, ask=mid + spread / 2)
        return MarketContext(symbol, decided_at, series, tick)


def evidence_by_playbook(
    orders: list[BacktestOrder],
    frame: pd.DataFrame,
    backtester: PessimisticBacktester | None = None,
) -> list[PlaybookEvidence]:
    """Replay each theory's orders on its own, and report them separately.

    Separately, because a blended number hides the finding. Four theories that
    break even and one that bleeds average out to "slightly negative, needs
    tuning" when the honest answer is "delete the fifth".

    Each theory is run non-overlapping, matching the live account's one-slot-
    at-a-time reality rather than assuming unlimited capital.
    """
    engine = backtester or PessimisticBacktester()
    grouped: dict[str, list[BacktestOrder]] = {}
    for order in orders:
        name = order.modules[0] if order.modules else "unknown"
        grouped.setdefault(name, []).append(order)

    evidence: list[PlaybookEvidence] = []
    for name, group in sorted(grouped.items()):
        result = engine.run_non_overlapping(frame, group)
        evidence.append(
            PlaybookEvidence(
                playbook=name,
                proposals=len(group),
                trades=result.sample_size,
                total_r=result.total_r,
                win_rate=result.win_rate,
                expectancy_r=result.expectancy_r,
                max_drawdown_r=result.max_drawdown_r,
            )
        )
    return evidence


@dataclass(frozen=True, slots=True)
class Comparison:
    """One theory against a coin flip that took the same trades."""

    real: PlaybookEvidence
    flip_win_rate: float
    flip_expectancy_r: float
    #: Best and worst expectancy across the seeds, so a single lucky shuffle
    #: cannot be mistaken for the baseline.
    flip_best_r: float
    flip_worst_r: float

    @property
    def edge_r(self) -> float:
        """What the pattern logic is worth, per trade, over guessing."""
        return self.real.expectancy_r - self.flip_expectancy_r

    @property
    def beats_the_coin(self) -> bool:
        """Better than every shuffle, not merely better than their average.

        A theory that lands inside the range chance produces has not been shown
        to know anything. This is a low bar deliberately — clearing it is
        necessary, nowhere near sufficient.
        """
        return self.real.expectancy_r > self.flip_best_r


def coin_flip(orders: list[BacktestOrder], seed: int) -> list[BacktestOrder]:
    """The same trades with the direction guessed.

    Identical times, identical symbols, identical stop and target distances —
    only the one thing the theory claims to know is replaced by a coin. The
    geometry is mirrored rather than kept, so a flipped long is a real short
    and not a long with its stop above the entry.

    This is the control the whole exercise needs. Five theories all landing
    between 24% and 33% wins, when a 1R stop against a 2R target wins 33% by
    chance alone, is either a coincidence or the answer.
    """
    rng = np.random.default_rng(seed)
    flipped: list[BacktestOrder] = []
    for order in orders:
        risk = abs(order.entry - order.stop_loss)
        reward = abs(order.take_profit - order.entry)
        direction = Direction.LONG if rng.random() < 0.5 else Direction.SHORT
        sign = int(direction)
        flipped.append(
            replace(
                order,
                direction=direction,
                stop_loss=order.entry - risk * sign,
                take_profit=order.entry + reward * sign,
            )
        )
    return flipped


def compare_to_chance(
    orders: list[BacktestOrder],
    frame: pd.DataFrame,
    evidence: list[PlaybookEvidence],
    *,
    seeds: int = 5,
    backtester: PessimisticBacktester | None = None,
) -> list[Comparison]:
    """Each theory beside the coin flip that took its trades.

    Several seeds, because one shuffle is a sample of one and the question is
    whether the theory sits outside what chance produces, not whether it beats
    one particular coin.
    """
    engine = backtester or PessimisticBacktester()
    grouped: dict[str, list[BacktestOrder]] = {}
    for order in orders:
        grouped.setdefault(order.modules[0] if order.modules else "unknown", []).append(order)

    comparisons: list[Comparison] = []
    for item in evidence:
        group = grouped.get(item.playbook, [])
        if not group:
            continue
        runs = [engine.run_non_overlapping(frame, coin_flip(group, seed)) for seed in range(seeds)]
        expectancies = [run.expectancy_r for run in runs]
        comparisons.append(
            Comparison(
                real=item,
                flip_win_rate=float(np.mean([run.win_rate for run in runs])),
                flip_expectancy_r=float(np.mean(expectancies)),
                flip_best_r=max(expectancies),
                flip_worst_r=min(expectancies),
            )
        )
    return comparisons


@dataclass(frozen=True, slots=True)
class TargetRow:
    """One theory at one target distance, beside the coin at the same distance."""

    playbook: str
    r_multiple: float
    trades: int
    win_rate: float
    expectancy_r: float
    coin_expectancy_r: float

    @property
    def edge_r(self) -> float:
        return self.expectancy_r - self.coin_expectancy_r


def retarget(orders: list[BacktestOrder], r_multiple: float) -> list[BacktestOrder]:
    """The same trades with the target moved to a fixed multiple of the stop.

    The stop is untouched. Only the question "how far are we reaching" changes,
    which is the operator's complaint stated as arithmetic: a USDCHF long
    entered at 0.81009 with its target 14.5 pips away peaked 7.1 pips up, never
    reached it, and closed at +2.7. It kept 38% of its best moment because the
    target was somewhere the market was not going.
    """
    moved: list[BacktestOrder] = []
    for order in orders:
        risk = abs(order.entry - order.stop_loss)
        sign = int(order.direction)
        moved.append(replace(order, take_profit=order.entry + risk * r_multiple * sign))
    return moved


def sweep_targets(
    orders: list[BacktestOrder],
    frame: pd.DataFrame,
    *,
    multiples: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    seeds: int = 3,
    backtester: PessimisticBacktester | None = None,
) -> list[TargetRow]:
    """Every theory at every target distance, each beside its own coin.

    The coin is swept too, and that is the entire point. A closer target raises
    any win rate — a coin reaching for 0.5R wins far more often than one
    reaching for 3R — so a theory that improves at a closer target has shown
    nothing unless it improves *more than the coin does*. Without that column
    this table would be a machine for talking yourself into a shorter target.

    This is a parameter search and parameter searches fit noise. Read the
    shape, not the maximum: a broad stretch of target distances where the edge
    is positive is worth something, and a single spike surrounded by negatives
    is the curve fitting itself to ninety days.
    """
    engine = backtester or PessimisticBacktester()
    grouped: dict[str, list[BacktestOrder]] = {}
    for order in orders:
        grouped.setdefault(order.modules[0] if order.modules else "unknown", []).append(order)

    rows: list[TargetRow] = []
    for name, group in sorted(grouped.items()):
        for multiple in multiples:
            moved = retarget(group, multiple)
            real = engine.run_non_overlapping(frame, moved)
            coins = [
                engine.run_non_overlapping(frame, coin_flip(moved, seed)).expectancy_r
                for seed in range(seeds)
            ]
            rows.append(
                TargetRow(
                    playbook=name,
                    r_multiple=multiple,
                    trades=real.sample_size,
                    win_rate=real.win_rate,
                    expectancy_r=real.expectancy_r,
                    coin_expectancy_r=float(np.mean(coins)),
                )
            )
    return rows


def render_targets(rows: list[TargetRow], *, window: str = "") -> str:
    """Does reaching for less turn any of this positive?"""
    lines = [
        "",
        "=" * 78,
        f"  THE SAME TRADES, REACHING FOR LESS{('  ' + window) if window else ''}",
        "=" * 78,
        "",
        "  Stop unchanged, target moved. The coin is moved with it, because a",
        "  closer target raises anybody's win rate — only beating the coin by more",
        "  than before means the analysis got better rather than the arithmetic.",
        "",
    ]
    if not rows:
        lines.extend(["  Nothing to sweep.", ""])
        return "\n".join(lines)

    for name in sorted({row.playbook for row in rows}):
        mine = [row for row in rows if row.playbook == name]
        lines.append(f"  {name}")
        lines.append(
            f"  {'target':>10}{'trades':>9}{'won':>7}{'per trade':>12}{'coin':>11}{'edge':>10}"
        )
        lines.append("  " + "-" * 60)
        for row in sorted(mine, key=lambda r: r.r_multiple):
            lines.append(
                f"  {row.r_multiple:>9.2f}R{row.trades:>9}{row.win_rate:>7.0%}"
                f"{row.expectancy_r:>+11.3f}R{row.coin_expectancy_r:>+10.3f}R"
                f"{row.edge_r:>+9.3f}R"
            )
        lines.append("")

    positive = [row for row in rows if row.expectancy_r > 0 and row.edge_r > 0]
    lines.append("-" * 78)
    if not positive:
        lines.append("  No target distance makes any theory both profitable and better than")
        lines.append("  a coin. Taking profit sooner is the right instinct and it is not")
        lines.append("  enough on its own: it raises the win rate and shrinks the wins by")
        lines.append("  the same arithmetic that raises the coin's.")
    else:
        best = sorted(positive, key=lambda r: -r.edge_r)
        lines.append("  Positive and beating the coin:")
        for row in best[:6]:
            lines.append(
                f"    {row.playbook} at {row.r_multiple:.2f}R — "
                f"{row.expectancy_r:+.3f}R a trade over {row.trades} trades, "
                f"edge {row.edge_r:+.3f}R"
            )
        lines.append("")
        lines.append("  Read the shape and not the maximum. A run of neighbouring targets")
        lines.append("  that are all positive is worth something; one spike between")
        lines.append("  negatives is this curve fitting itself to ninety days.")
    lines.append("")
    return "\n".join(lines)


def render_comparison(comparisons: list[Comparison], *, window: str = "") -> str:
    """The only table that answers whether the analysis is worth anything."""
    lines = [
        "",
        "=" * 78,
        f"  EACH THEORY AGAINST A COIN FLIP THAT TOOK THE SAME TRADES"
        f"{('  ' + window) if window else ''}",
        "=" * 78,
        "",
        "  Same moments, same symbols, same stops and targets. Only the direction",
        "  is guessed. Whatever a theory beats the coin by is what its analysis is",
        "  actually worth.",
        "",
        f"  {'':18}{'won':>7}{'per trade':>12}{'coin won':>10}{'coin/trade':>12}{'edge':>10}",
        "  " + "-" * 74,
    ]
    if not comparisons:
        lines.extend(["  Nothing to compare.", ""])
        return "\n".join(lines)

    for item in sorted(comparisons, key=lambda c: c.edge_r):
        mark = "" if item.beats_the_coin else "   <- inside chance"
        lines.append(
            f"  {item.real.playbook:<18}{item.real.win_rate:>7.0%}"
            f"{item.real.expectancy_r:>+11.3f}R{item.flip_win_rate:>10.0%}"
            f"{item.flip_expectancy_r:>+11.3f}R{item.edge_r:>+9.3f}R{mark}"
        )

    beaten = [c.real.playbook for c in comparisons if c.beats_the_coin]
    lines.append("")
    if not beaten:
        lines.append("  Not one theory beat guessing.")
        lines.append("")
        lines.append("  That is the finding, and it is not about these five theories. Every")
        lines.append("  pattern rule in them — the compression, the shallow pullback, the")
        lines.append("  three-times-tested edge, the rejection wick — adds nothing a coin")
        lines.append("  does not already give you. Writing a sixth would be writing a sixth")
        lines.append("  coin. The approach itself has to change, not the rules inside it.")
    else:
        lines.append(f"  Outside what chance produced: {', '.join(beaten)}.")
        lines.append("  Necessary, nowhere near sufficient — a theory can beat a coin and")
        lines.append("  still lose money once costs are paid. Check its `per trade` column.")
    lines.append("")
    return "\n".join(lines)


def render(evidence: list[PlaybookEvidence], *, window: str = "") -> str:
    """The table, with the honest reading of it printed underneath."""
    lines = [
        "",
        "=" * 78,
        f"  WHAT EACH THEORY WOULD HAVE DONE{('  ' + window) if window else ''}",
        "=" * 78,
        "",
        f"  {'':18}{'proposals':>10}{'trades':>8}{'won':>8}{'net':>11}{'per trade':>10}"
        f"{'worst dd':>11}",
        "  " + "-" * 74,
    ]
    if not evidence:
        lines.append("  No theory proposed anything over this window.")
        lines.append("")
        return "\n".join(lines)

    for item in sorted(evidence, key=lambda e: e.total_r):
        lines.append(item.row())

    thin = [item.playbook for item in evidence if item.trades < 100]
    losing = [item.playbook for item in evidence if item.trades >= 100 and item.total_r < 0]
    lines.append("")
    if losing:
        lines.append(f"  Negative over a real sample: {', '.join(losing)}.")
        lines.append("  Switching one of those off is worth more, and is far more certain,")
        lines.append("  than any rule that could be added in its place.")
    if thin:
        lines.append(f"  Too few trades to read: {', '.join(thin)}. Widen the window.")
    lines.append("")
    lines.append("  Costs: the backtester applies spread and slippage; commission is not")
    lines.append("  in these numbers, so subtract roughly 0.03R to 0.11R per trade at this")
    lines.append("  account's stop widths before believing a thin positive.")
    lines.append("")
    return "\n".join(lines)
