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

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from analysis.playbooks import PlaybookEngine
from backtesting.engine import BacktestOrder, PessimisticBacktester
from core.types import MarketContext, Series, Tick, Timeframe, TradingMode

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
