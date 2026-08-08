"""Measure what actually happens after a position goes into profit.

Every exit threshold in this system is a number somebody chose. `bank_at_r` is
0.3. The give-back rule arms at 0.5R. The health reader looks twelve bars back.
They are applied to every instrument at every hour of the day, and not one of
them was derived from a market — they were argued for, reviewed, and shipped.

The operator's complaint is the direct consequence: a trade goes fifty cents up
and comes back out at a full loss, and `postmortem.py --list 20` says ten of the
last twenty kept under half of their best moment. No amount of reasoning about
that produces the right threshold. Counting does.

So this walks every order the theories would have placed, bar by bar, and at
each bar records two numbers that can be compared honestly:

    take now   what you get for closing at this bar's close
    hold out   what the trade actually went on to return

Both carry the same round-trip cost, so the difference between them is pure
price and nothing else. Positive means holding paid. Negative means the money
was on the table and the system left it there.

That difference, bucketed by the things the guard already knows every second —
how far in profit you are, and whether the move is still running — is the exit
policy. Not a model anybody trained. A table of what happened, over ninety days
of real bars, which the guard can read in a microsecond and which costs nothing
to consult on every tick.

WHAT THIS CANNOT DO. It measures management, not selection. The entries it
walks are the same entries `backtest.cmd` found to be indistinguishable from a
coin flip, and a better exit on a coin flip is a cheaper coin flip. The gap to
break even on this account is 0.25-0.45R per trade; nothing here is going to
find that on its own. It closes a real and separate defect, which is that the
account hands back what it has already earned.

    from backtesting.exit_study import study, hold_table, render_hold_table
    positions = study(orders, frames[Timeframe.M5])
    print(render_hold_table(hold_table(positions)))
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from backtesting.engine import BacktestAssumptions, BacktestOrder

#: Where the drift reading is cut into "running our way", "going nowhere" and
#: "turning against us". Not a new invention: `bank_while_retracing_drift` is
#: already 0.5 in the live configuration and `momentum_turned` already treats
#: 0.25 as a quarter of an ordinary excursion. Using the live number keeps the
#: table's rows meaning the same thing as the rule that will read it.
RUNNING_DRIFT = 0.5

#: Buckets for "how far in profit are you right now". Deliberately coarse.
#: Finer buckets look more precise and are mostly noise: at ninety days and
#: four symbols the thin ones already carry a few hundred samples, and cutting
#: them again buys decimal places nobody can act on.
R_EDGES = (0.0, 0.15, 0.30, 0.50, 0.75, 1.00, 1.50)

#: Buckets for the give-back curve — how much of its best moment a trade kept.
PEAK_EDGES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.50)


@dataclass(frozen=True, slots=True)
class ExitSample:
    """One moment inside one replayed position, and how that position ended.

    Every field except `hold_out_r` is something the live guard can compute
    from what it already has in hand, which is the point: a policy built on
    anything else could be measured here and never run.
    """

    bars_held: int
    #: Net R for closing at this bar's close, exit slippage and commission paid.
    take_now_r: float
    #: Net R the position actually went on to return, same costs.
    hold_out_r: float
    #: Best gross R reached at any point up to and including this bar. The live
    #: journal's `mfe_r` ratchets the same way, off the high rather than the
    #: close, because the guard reads ticks.
    peak_r: float
    bars_since_peak: int
    #: `analysis.position_health.drift_score`, signed for this position.
    #: None when the window has not filled yet.
    drift: float | None

    @property
    def edge_of_holding(self) -> float:
        """What waiting was worth. Both sides paid the identical round trip,
        so this is price and nothing else."""
        return self.hold_out_r - self.take_now_r


@dataclass(frozen=True, slots=True)
class ReplayedPosition:
    """One order walked from fill to exit."""

    symbol: str
    playbook: str
    outcome: str
    final_r: float
    peak_r: float
    bars_held: int
    samples: tuple[ExitSample, ...]

    @property
    def kept(self) -> float | None:
        """Share of its best moment the position took home. None when it never
        went into profit at all — there was nothing to keep."""
        if self.peak_r <= 0:
            return None
        return self.final_r / self.peak_r


def rolling_drift(frame: pd.DataFrame, *, bars: int = 12, atr_period: int = 14) -> np.ndarray:
    """`drift_score` at every bar of a frame, for a long, in one pass.

    Same definition as the live reader — a least-squares slope over the last
    `bars` closes, divided by `sqrt(bars) * ATR` — and `test_exit_study` pins
    it against `drift_score` itself rather than trusting that claim.

    Vectorised because the naive version is a `np.polyfit` per bar per order.
    An OLS slope over a fixed window is a fixed set of weights, so the whole
    series is one dot product against a sliding view; the ATR is a rolling
    mean of true range. Twenty-six thousand bars go from about a second and a
    half to under a millisecond, and the study calls it once per symbol
    instead of once per sample.

    A short's drift is the long's negated, so this is computed once per frame
    and the sign applied at the point of use.
    """
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    count = len(close)
    out = np.full(count, np.nan)
    if count < max(bars, atr_period + 1):
        return out

    # True range, matching the pandas version bar for bar: the first bar has no
    # previous close and falls back to high-low, which is what `.max(axis=1)`
    # over a row containing NaNs produces.
    previous = np.empty(count)
    previous[0] = np.nan
    previous[1:] = close[:-1]
    with np.errstate(invalid="ignore"):
        true_range = np.nanmax(
            np.vstack([high - low, np.abs(high - previous), np.abs(low - previous)]), axis=0
        )

    # Rolling mean of the last `atr_period` true ranges, by prefix sum.
    cumulative = np.concatenate([[0.0], np.cumsum(true_range)])
    atr = np.full(count, np.nan)
    atr[atr_period - 1 :] = (
        cumulative[atr_period:] - cumulative[: count - atr_period + 1]
    ) / atr_period

    # OLS slope over a fixed window is a fixed weight vector: the deviations of
    # the x values divided by their sum of squares.
    x = np.arange(bars, dtype=float)
    deviation = x - x.mean()
    weights = deviation / float(np.dot(deviation, deviation))
    windows = np.lib.stride_tricks.sliding_window_view(close, bars)
    slope = np.full(count, np.nan)
    slope[bars - 1 :] = windows @ weights

    valid = (atr > 0) & np.isfinite(slope)
    out[valid] = slope[valid] * np.sqrt(bars) / atr[valid]
    return out


def study(
    orders: list[BacktestOrder],
    frame: pd.DataFrame,
    *,
    assumptions: BacktestAssumptions | None = None,
    drift_bars: int = 12,
) -> list[ReplayedPosition]:
    """Walk every order to its exit, recording the state at each bar on the way.

    The fill rules are `PessimisticBacktester._replay`'s, deliberately and to
    the letter: entry no earlier than the next bar, the stop wins a bar that
    touches both, gaps fill at the open against us and never for us. A study
    that was kinder than the backtester would produce a policy tuned to a
    market that does not exist.
    """
    rules = assumptions or BacktestAssumptions()
    drift_up = rolling_drift(frame, bars=drift_bars)
    index = frame.index
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)

    positions: list[ReplayedPosition] = []
    for order in orders:
        if order.risk <= 0:
            continue
        first = int(index.searchsorted(pd.Timestamp(order.decided_at), "right"))
        if first >= len(index):
            continue
        last = min(first + rules.max_holding_bars, len(index))
        walked = _walk(order, rules, drift_up, opens, highs, lows, closes, first, last)
        if walked is not None:
            positions.append(walked)
    return positions


def _walk(
    order: BacktestOrder,
    rules: BacktestAssumptions,
    drift_up: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    first: int,
    last: int,
) -> ReplayedPosition | None:
    sign = int(order.direction)
    entry = order.entry * (1 + (rules.entry_slippage_bps / 10_000.0) * sign)
    risk = order.risk
    # Charged identically on every exit this function can produce, so it drops
    # out of `edge_of_holding` and cannot tilt the policy either way.
    costs_r = order.entry * (rules.round_trip_commission_bps / 10_000.0) / risk
    exit_fraction = rules.exit_slippage_bps / 10_000.0

    def net_at(price: float) -> float:
        return (price * (1 - exit_fraction * sign) - entry) * sign / risk - costs_r

    peak_r = 0.0
    peak_bar = 0
    pending: list[tuple[int, float, float, int, float | None]] = []
    exit_price = float(closes[last - 1])
    outcome = "TIME"
    held = last - first

    for step, position in enumerate(range(first, last)):
        favourable = (highs[position] if sign > 0 else lows[position]) - entry
        reached = favourable * sign / risk
        if reached > peak_r:
            peak_r, peak_bar = reached, step

        if sign > 0:
            stop_hit = lows[position] <= order.stop_loss
            target_hit = highs[position] >= order.take_profit
            hit_price = min(opens[position], order.stop_loss) if stop_hit else order.take_profit
        else:
            stop_hit = highs[position] >= order.stop_loss
            target_hit = lows[position] <= order.take_profit
            hit_price = max(opens[position], order.stop_loss) if stop_hit else order.take_profit

        if stop_hit or target_hit:
            exit_price = hit_price
            outcome = ("SL" if not target_hit else "SL_FIRST_AMBIGUOUS") if stop_hit else "TP"
            held = step + 1
            break

        # Only bars the position survived become decisions. The bar that closes
        # the trade is not a moment anybody could have chosen to act on: by the
        # time its close is known, the stop or the target has already filled.
        drift = drift_up[position]
        pending.append(
            (
                step + 1,
                net_at(float(closes[position])),
                peak_r,
                step - peak_bar,
                None if not np.isfinite(drift) else float(drift) * sign,
            )
        )

    final_r = net_at(exit_price)
    samples = tuple(
        ExitSample(
            bars_held=bars_held,
            take_now_r=take_now,
            hold_out_r=final_r,
            peak_r=peak,
            bars_since_peak=since,
            drift=drift,
        )
        for bars_held, take_now, peak, since, drift in pending
    )
    return ReplayedPosition(
        symbol=order.symbol,
        playbook=order.modules[0] if order.modules else "unknown",
        outcome=outcome,
        final_r=final_r,
        peak_r=peak_r,
        bars_held=held,
        samples=samples,
    )


@dataclass(frozen=True, slots=True)
class HoldVerdict:
    """What holding was worth, at one state the guard can recognise."""

    r_floor: float
    r_ceiling: float
    pace: str
    samples: int
    take_now_r: float
    hold_out_r: float

    @property
    def edge(self) -> float:
        return self.hold_out_r - self.take_now_r

    @property
    def verdict(self) -> str:
        return "hold" if self.edge > 0 else "TAKE IT"


def _pace(drift: float | None) -> str:
    if drift is None:
        return "unknown"
    if drift >= RUNNING_DRIFT:
        return "running"
    if drift <= -RUNNING_DRIFT:
        return "against"
    return "stalled"


def _bucket(value: float, edges: tuple[float, ...]) -> tuple[float, float] | None:
    if value < edges[0]:
        return None
    for low, high in pairwise(edges):
        if low <= value < high:
            return low, high
    return edges[-1], float("inf")


def hold_table(positions: list[ReplayedPosition], *, min_samples: int = 30) -> list[HoldVerdict]:
    """Bucket every in-profit moment by where it was and what the move was doing.

    Only moments in profit. "Should I take this" is not a question you can ask
    about a losing position — the answer there is the stop, and the stop is
    already sitting at the broker.

    Buckets under `min_samples` are dropped rather than reported thin. A row
    built on eleven observations is an invitation to act on noise, and acting
    on noise is what this whole exercise exists to stop.
    """
    buckets: dict[tuple[float, float, str], list[ExitSample]] = {}
    for position in positions:
        for sample in position.samples:
            if sample.take_now_r <= 0:
                continue
            span = _bucket(sample.take_now_r, R_EDGES)
            if span is None:
                continue
            buckets.setdefault((span[0], span[1], _pace(sample.drift)), []).append(sample)

    rows = [
        HoldVerdict(
            r_floor=floor,
            r_ceiling=ceiling,
            pace=pace,
            samples=len(group),
            take_now_r=float(np.mean([s.take_now_r for s in group])),
            hold_out_r=float(np.mean([s.hold_out_r for s in group])),
        )
        for (floor, ceiling, pace), group in buckets.items()
        if len(group) >= min_samples
    ]
    order = {"running": 0, "stalled": 1, "against": 2, "unknown": 3}
    return sorted(rows, key=lambda row: (row.r_floor, order.get(row.pace, 9)))


def render_hold_table(rows: list[HoldVerdict]) -> str:
    """The policy as a table, in the units the operator thinks in."""
    if not rows:
        return "\n  Not enough in-profit moments to say anything. Widen the window.\n"

    out = [
        "",
        "  HOLD OR TAKE — measured, not chosen",
        "  " + "-" * 74,
        f"  {'in profit':<14}{'pace':<10}{'moments':>9}{'take now':>11}"
        f"{'hold out':>11}{'waiting is':>13}   what to do",
        "  " + "-" * 74,
    ]
    for row in rows:
        ceiling = "+" if row.r_ceiling == float("inf") else f"{row.r_ceiling:.2f}R"
        window = f"{row.r_floor:.2f}-{ceiling}"
        out.append(
            f"  {window:<14}{row.pace:<10}{row.samples:>9}{row.take_now_r:>+10.3f}R"
            f"{row.hold_out_r:>+10.3f}R{row.edge:>+12.3f}R   {row.verdict}"
        )
    out.append("  " + "-" * 74)
    out.append(
        "  'waiting is' is what an extra minute of patience was worth on average.\n"
        "  Both columns already paid the identical round trip, so it is pure price."
    )
    return "\n".join(out) + "\n"


@dataclass(frozen=True, slots=True)
class GiveBack:
    """How much of its best moment a group of positions took home."""

    peak_floor: float
    peak_ceiling: float
    positions: int
    mean_peak_r: float
    mean_final_r: float

    @property
    def kept(self) -> float:
        return self.mean_final_r / self.mean_peak_r if self.mean_peak_r else 0.0


def give_back_curve(
    positions: list[ReplayedPosition], *, min_positions: int = 10
) -> list[GiveBack]:
    """The operator's actual complaint, counted over ninety days.

    `postmortem.py --list 20` reported ten of twenty trades keeping under half
    of their best moment. Twenty trades cannot tell you whether that is the
    system or the market. This can.
    """
    buckets: dict[tuple[float, float], list[ReplayedPosition]] = {}
    for position in positions:
        if position.peak_r <= 0:
            continue
        span = _bucket(position.peak_r, PEAK_EDGES)
        if span is not None:
            buckets.setdefault(span, []).append(position)

    return sorted(
        (
            GiveBack(
                peak_floor=floor,
                peak_ceiling=ceiling,
                positions=len(group),
                mean_peak_r=float(np.mean([p.peak_r for p in group])),
                mean_final_r=float(np.mean([p.final_r for p in group])),
            )
            for (floor, ceiling), group in buckets.items()
            if len(group) >= min_positions
        ),
        key=lambda row: row.peak_floor,
    )


def render_give_back(rows: list[GiveBack]) -> str:
    if not rows:
        return "\n  No position ever went into profit. That is its own finding.\n"

    out = [
        "",
        "  WHAT EACH TRADE KEPT OF ITS BEST MOMENT",
        "  " + "-" * 62,
        f"  {'peaked at':<16}{'trades':>9}{'best':>10}{'ended':>10}{'kept':>10}",
        "  " + "-" * 62,
    ]
    for row in rows:
        ceiling = "+" if row.peak_ceiling == float("inf") else f"{row.peak_ceiling:.2f}R"
        out.append(
            f"  {f'{row.peak_floor:.2f}-{ceiling}':<16}{row.positions:>9}"
            f"{row.mean_peak_r:>+9.2f}R{row.mean_final_r:>+9.2f}R{row.kept:>9.0%}"
        )
    out.append("  " + "-" * 62)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------- policies ---
#
# The hold-or-take table above answers "given the position is here, is waiting
# worth it". It cannot answer "where should the threshold be", and the two get
# confused constantly. Every row in it is conditional on having REACHED that
# level, so a rule that banks at 0.15R does not collect the 1.50R rows — it
# deletes them. Averaging per moment and reading a threshold off the result is
# how a backtest talks itself into taking profit far too early.
#
# What follows answers the threshold question properly: it replays a whole
# rule over each position's own path, in order, takes the first moment the rule
# fires, and reports what the account would have made per trade. Nothing is
# averaged that did not happen.


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """One complete rule for when to take a profit.

    Shaped to match what `PositionManager._bank_worthwhile_profit` can actually
    do at 4am with a tick and a drift reading, because a policy this measures
    and the guard cannot run is a policy that was never evaluated.
    """

    take_at_r: float
    #: Which pace readings the rule is willing to act on. The 90-day
    #: measurement said running is the worst state to hold in and a retrace is
    #: the only one that pays, which is why the defaults are what they are.
    act_when_running: bool = True
    act_when_stalled: bool = True
    act_when_against: bool = False
    #: Before twelve bars have passed there is no drift reading at all. Acting
    #: is the safer default: it is early in the trade, the profit is small, and
    #: "no information" is not a reason to hold money on the table.
    act_when_unknown: bool = True

    @property
    def name(self) -> str:
        states = "".join(
            letter
            for letter, on in (
                ("R", self.act_when_running),
                ("S", self.act_when_stalled),
                ("A", self.act_when_against),
            )
            if on
        )
        return f"{self.take_at_r:.2f}R/{states or '-'}"

    def fires(self, sample: ExitSample) -> bool:
        if sample.take_now_r < self.take_at_r:
            return False
        return {
            "running": self.act_when_running,
            "stalled": self.act_when_stalled,
            "against": self.act_when_against,
            "unknown": self.act_when_unknown,
        }[_pace(sample.drift)]


#: Doing nothing, so every policy has something honest to be measured against.
HOLD_EVERYTHING = ExitPolicy(
    take_at_r=float("inf"),
    act_when_running=False,
    act_when_stalled=False,
    act_when_against=False,
    act_when_unknown=False,
)


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """What one rule would have returned across every replayed position."""

    policy: ExitPolicy
    trades: int
    banked: int
    total_r: float
    win_rate: float
    worst_r: float

    @property
    def per_trade(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0

    @property
    def banked_share(self) -> float:
        return self.banked / self.trades if self.trades else 0.0


def apply_policy(positions: list[ReplayedPosition], policy: ExitPolicy) -> PolicyOutcome:
    """Replay one rule over every position and report what it returned.

    The first moment the rule fires ends that trade at that bar's close.
    Positions the rule never fires on run to their natural stop or target,
    which is what makes the comparison against `HOLD_EVERYTHING` meaningful:
    the same trades, the same bars, one rule applied or not applied.
    """
    returns: list[float] = []
    banked = 0
    for position in positions:
        hit = next((sample for sample in position.samples if policy.fires(sample)), None)
        if hit is None:
            returns.append(position.final_r)
        else:
            returns.append(hit.take_now_r)
            banked += 1
    if not returns:
        return PolicyOutcome(policy, 0, 0, 0.0, 0.0, 0.0)
    values = np.asarray(returns, dtype=float)
    return PolicyOutcome(
        policy=policy,
        trades=len(values),
        banked=banked,
        total_r=float(values.sum()),
        win_rate=float((values > 0).mean()),
        worst_r=float(values.min()),
    )


#: Thresholds swept. Fine at the bottom because that is where the operator's
#: instinct points and where the answer actually changes; coarse above 1R,
#: where these trades rarely get to.
POLICY_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50)


def sweep_policies(positions: list[ReplayedPosition]) -> list[PolicyOutcome]:
    """Every threshold, against both pace rules, plus doing nothing.

    Two pace variants only. More would be a parameter search, and a parameter
    search over one 90-day window finds the noise — the whole discipline of
    this project is measuring a small number of pre-committed questions rather
    than picking the winner out of a hundred.
    """
    outcomes = [apply_policy(positions, HOLD_EVERYTHING)]
    for threshold in POLICY_THRESHOLDS:
        outcomes.append(apply_policy(positions, ExitPolicy(take_at_r=threshold)))
        outcomes.append(
            apply_policy(
                positions,
                ExitPolicy(take_at_r=threshold, act_when_against=True),
            )
        )
    return outcomes


def render_policies(outcomes: list[PolicyOutcome]) -> str:
    """The threshold question, answered end to end rather than per moment."""
    if not outcomes:
        return "\n  Nothing to replay.\n"

    baseline = outcomes[0]
    out = [
        "",
        "  WHAT A WHOLE EXIT RULE WOULD HAVE RETURNED",
        "  " + "-" * 76,
        f"  {'rule':<14}{'trades':>8}{'banked':>9}{'won':>7}{'per trade':>12}"
        f"{'vs holding':>13}{'worst':>9}",
        "  " + "-" * 76,
        f"  {'hold to stop':<14}{baseline.trades:>8}{'-':>9}{baseline.win_rate:>6.0%}"
        f"{baseline.per_trade:>+11.3f}R{'-':>13}{baseline.worst_r:>+8.2f}R",
    ]
    best = baseline
    for outcome in outcomes[1:]:
        if outcome.per_trade > best.per_trade:
            best = outcome
        out.append(
            f"  {outcome.policy.name:<14}{outcome.trades:>8}{outcome.banked_share:>8.0%}"
            f"{outcome.win_rate:>7.0%}{outcome.per_trade:>+11.3f}R"
            f"{outcome.per_trade - baseline.per_trade:>+12.3f}R{outcome.worst_r:>+8.2f}R"
        )
    out.append("  " + "-" * 76)
    out.append(
        "  R = act while running, S = while stalled, A = while retracing.\n"
        "  'vs holding' is the whole point: what the rule adds over letting every\n"
        "  trade run to its stop or target. Positive means taking profit earlier\n"
        "  was worth doing on these entries."
    )
    if best is baseline:
        out.append("\n  No threshold beat holding. Taking profit earlier is not the answer here.")
    else:
        out.append(
            f"\n  Best on this window: {best.policy.name} at {best.per_trade:+.3f}R per trade,\n"
            f"  {best.per_trade - baseline.per_trade:+.3f}R better than holding, banking "
            f"{best.banked_share:.0%} of trades early."
        )
    return "\n".join(out) + "\n"
