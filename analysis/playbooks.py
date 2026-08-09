"""Several independent theories about the same chart, each with its own plan.

The engine had five modules and one idea. Every module read H1 or H4, their
scores were blended into a single weighted verdict, and that verdict got one
stop shape — a structural level on H1 with an ATR buffer — and one target
shape. Which is a perfectly good swing strategy, and it is the *only* thing the
system could ever see. A clean five-minute impulse with an eight-pip stop was
invisible, because nothing looked at M5 for an entry and because a scalp's plan
does not survive being averaged into an H1 one.

A playbook is a complete, self-contained answer: direction, entry, stop, target,
and how long it expects to take. Not a vote. `MomentumScalp` proposing a long
with a 10-pip stop and `SwingConfluence` proposing a long with a 90-pip stop are
not 'agreeing more strongly' — they are two different trades, and blending them
produces a third that neither theory would take.

So they compete rather than combine, and all of them are shown to the reviewer
including the ones that lost. That disagreement is evidence in itself: two
theories pointing opposite ways on the same chart is a market with no edge in
either direction, which is a reason to stand aside rather than to pick the
higher score.

**Why tight stops matter on this account specifically.** At EUR 100 and 2% risk
the budget is EUR 2. A 90-pip swing stop buys 0.0077 lots, below the 0.01
minimum, and the trade is skipped as undercapitalized — which is most of why
this account has struggled to place anything at all. A 12-pip scalp stop buys
0.058 lots, comfortably above the minimum. The short-horizon playbooks are not
a riskier addition; on an account this size they are frequently the only ones
whose arithmetic works.

**What that costs, and the gate it demands.** Spread is charged per trade
regardless of how long it is held, so it eats a far larger share of a small
stop. Twelve pips with a 1.5-pip spread is 12.5% of the risk gone at entry, and
a scalp playbook without a hard spread-to-stop gate is a machine for paying
spread. Every short-horizon playbook here refuses a setup where the spread is
more than `max_spread_share_of_stop` of the risk, and that gate rejects far more
scalps than the pattern logic does. That is intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from config.schema import ConfluenceConfig
from core.types import Direction, MarketContext, Timeframe, TradingMode


@dataclass(frozen=True, slots=True)
class Play:
    """One theory's complete proposal. Self-contained by design.

    Carries its own stop and target because that is the part a shared engine
    cannot get right for every theory at once: the correct invalidation for a
    five-minute momentum entry is the low of the pullback, and for an H1 swing
    it is the last structural swing point, and there is no single rule that
    produces both.
    """

    playbook: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    conviction: float
    horizon_minutes: int
    thesis: str
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_risk(self) -> float:
        return abs(self.take_profit - self.entry) / self.risk if self.risk > 0 else 0.0

    def summary(self) -> dict[str, object]:
        """Prompt-ready view for the reviewer."""
        return {
            "playbook": self.playbook,
            "direction": self.direction.name,
            "entry": round(self.entry, 6),
            "stop_loss": round(self.stop_loss, 6),
            "take_profit": round(self.take_profit, 6),
            "reward_risk": round(self.reward_risk, 2),
            "conviction": round(self.conviction, 1),
            "expected_minutes_to_target": self.horizon_minutes,
            "thesis": self.thesis,
            "evidence": self.evidence,
        }


class Playbook(Protocol):
    """One theory. Returns a complete plan, or None when it sees nothing."""

    name: str
    horizon_minutes: int
    #: The highest conviction this theory can ever return.
    #:
    #: Declared rather than inferred so a conviction floor set above it fails
    #: loudly at startup instead of silently switching the theory off. "Only
    #: take nine-out-of-ten setups" sounds like strictness and is an off switch
    #: when nothing in the file can score nine.
    max_conviction: float

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None: ...


# --------------------------------------------------------------- shared bits ---


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    """Mean true range over the last `period` bars, in price units.

    In numpy rather than pandas, and only over the tail it needs. The pandas
    version built three aligned Series, concatenated them and rolled a mean
    across the whole frame to read one number off the end — 1.25 ms a call,
    against 0.13 ms here for a bit-identical result.

    That matters because of how often this runs. Five theories evaluate every
    symbol every cycle and most of them ask for an ATR, so it was about
    seven milliseconds of the thirteen each decision costs — half the price of
    scanning the catalogue, and half the runtime of a backtest.
    """
    if len(frame) < period + 1:
        return 0.0
    tail = frame.iloc[-(period + 1) :]
    high = tail["high"].to_numpy(dtype=float)
    low = tail["low"].to_numpy(dtype=float)
    previous = tail["close"].to_numpy(dtype=float)[:-1]
    ranges = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - previous), np.abs(low[1:] - previous)),
    )
    value = float(ranges.mean())
    return value if np.isfinite(value) else 0.0


def _spread_is_affordable(ctx: MarketContext, risk: float, limit: float) -> tuple[bool, float]:
    """Whether entry cost is a tolerable share of what is being risked.

    The gate that decides whether short-horizon trading is viable at all on a
    given instrument. A swing stop of 90 pips barely notices a 1.5-pip spread;
    a 10-pip scalp stop hands over 15% of its risk before the trade has done
    anything. Checked here rather than left to the generic spread filter,
    because that one asks "is this spread normal for this market" and this one
    asks "is this spread survivable for this stop", which are different
    questions with different answers.
    """
    if ctx.tick is None or risk <= 0:
        return False, 1.0
    share = ctx.tick.spread / risk
    return share <= limit, share


def _reachable(
    frame: pd.DataFrame, direction: Direction, bars_ahead: int, quantile: float
) -> float:
    """Typical favourable excursion over `bars_ahead`, in price units.

    The same idea the swing target already uses, applied per playbook on its own
    timeframe: a five-minute theory should size its target against what price
    does in the next twenty five-minute bars, not against what it does in a day.
    A percentile rather than a maximum, so one violent session does not set the
    expectation for every trade after it.
    """
    closes = frame["close"].to_numpy()
    extremes = (frame["high"] if direction is Direction.LONG else frame["low"]).to_numpy()
    windows = len(closes) - bars_ahead
    if windows <= 0:
        return 0.0
    runs = [
        (
            extremes[start + 1 : start + 1 + bars_ahead].max() - closes[start]
            if direction is Direction.LONG
            else closes[start] - extremes[start + 1 : start + 1 + bars_ahead].min()
        )
        for start in range(windows)
    ]
    return max(0.0, float(np.quantile(runs, quantile)))


# ------------------------------------------------------------------ playbooks ---


class MomentumScalp:
    """A clean impulse on M5 that M1 has not yet given back.

    The theory: when price makes a decisive move out of a quiet stretch and the
    pullback is shallow, the move usually has a little further to go. Short
    horizon, and the invalidation is the low of the *pullback* — if price trades
    back through that, the continuation has failed and the reason to be in the
    trade has gone.

    Deliberately demanding, and each requirement rejects a specific way this
    pattern fails:

    * The impulse must be large against ATR, or it is ordinary noise.
    * It must come *out of* compression, or it is the middle of an existing move
      and the easy part has already happened.
    * The pullback must be shallow, or momentum has already failed.
    * M1 must not be actively selling into the entry, which is the difference
      between a pullback and a reversal.
    * Spread must be a small share of the stop, which on a tight stop is the
      constraint that actually binds.
    """

    name = "momentum_scalp"
    max_conviction = 95.0
    horizon_minutes = 60

    def __init__(self, config: ScalpConfig) -> None:
        self.config = config

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:  # noqa: ARG002
        series = ctx.series.get(Timeframe.M5)
        if series is None or len(series.df) < 60 or ctx.tick is None:
            return None
        frame = series.df
        atr = _atr(frame)
        if atr <= 0:
            return None

        legs = self.config.impulse_bars
        window = frame.iloc[-legs:]
        move = float(window["close"].iloc[-1]) - float(window["open"].iloc[0])
        if abs(move) < atr * self.config.min_impulse_atr:
            return None
        direction = Direction.LONG if move > 0 else Direction.SHORT

        # Out of compression, not mid-move: the range before the impulse has to
        # have been quiet, or the easy part of this move is already gone.
        before = frame.iloc[-(legs + self.config.quiet_bars) : -legs]
        if len(before) < self.config.quiet_bars:
            return None
        quiet_range = float(before["high"].max() - before["low"].min())
        if quiet_range > atr * self.config.max_quiet_range_atr:
            return None

        # Shallow pullback. Measured from the impulse extreme, so a deep
        # retrace disqualifies even when the close still looks strong.
        extreme = float(
            window["high"].max() if direction is Direction.LONG else window["low"].min()
        )
        last = float(frame["close"].iloc[-1])
        pullback = abs(extreme - last) / atr
        if pullback > self.config.max_pullback_atr:
            return None

        # M1 must not be running against the entry right now.
        minute = ctx.series.get(Timeframe.M1)
        if minute is not None and len(minute.df) > 10:
            minute_atr = max(_atr(minute.df), 1e-12)
            drift = float(minute.df["close"].iloc[-1]) - float(minute.df["close"].iloc[-6])
            if -(drift * int(direction)) / minute_atr > self.config.max_adverse_m1_atr:
                return None

        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid

        # Under the *pullback* low, not the base of the whole impulse leg.
        #
        # Anchoring to the leg base looks safer and is the wrong trade: on a
        # sharp impulse the leg spans several ATR, so the "scalp" ends up with a
        # stop as wide as a swing entry — which on this account means the sizer
        # refuses it as undercapitalized, the exact problem short-horizon plays
        # exist to solve. It is also the wrong invalidation. The claim is that
        # the shallow pullback holds; the level that disproves it is the low of
        # that pullback, not a price from before the move began.
        # The pullback is what happened *after* the impulse peaked, so the
        # stop is anchored from that bar onward. A fixed lookback would reach
        # back down into the impulse itself and quietly widen the stop toward
        # the leg base again — the thing this rule exists to avoid.
        extremes = (window["high"] if direction is Direction.LONG else window["low"]).to_numpy()
        peak_position = int(extremes.argmax() if direction is Direction.LONG else extremes.argmin())
        since_peak = window.iloc[peak_position:]
        pivot = float(
            since_peak["low"].min() if direction is Direction.LONG else since_peak["high"].max()
        )
        stop = pivot - atr * self.config.stop_buffer_atr * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        # Clear one ordinary bar of this timeframe before anything else. A stop
        # inside that band is not testing the plan, it is testing the noise.
        floor = atr * self.config.min_stop_atr
        if risk < floor:
            stop = entry - floor * int(direction)
            risk = floor
        # And it has to stay a scalp. A stop several ATR wide is a different
        # trade wearing this playbook's name, and it belongs to the swing
        # engine, which sizes and targets it properly.
        if risk > atr * self.config.max_stop_atr:
            return None

        affordable, share = _spread_is_affordable(ctx, risk, self.config.max_spread_share_of_stop)
        if not affordable:
            return None

        typical = _reachable(frame.tail(400), direction, self.config.target_bars, 0.65)
        target_distance = min(risk * self.config.target_r, typical) if typical > 0 else 0.0
        if target_distance <= 0 or target_distance / risk < self.config.min_target_r:
            return None

        conviction = min(
            95.0,
            50.0
            + min(abs(move) / atr, 3.0) * 10.0
            + max(0.0, self.config.max_pullback_atr - pullback) * 15.0,
        )
        return Play(
            playbook=self.name,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=entry + target_distance * int(direction),
            conviction=conviction,
            horizon_minutes=self.horizon_minutes,
            thesis=(
                f"M5 impulse of {abs(move) / atr:.1f} ATR out of a quiet "
                f"{quiet_range / atr:.1f} ATR range, pullback only {pullback:.2f} ATR; "
                f"stop under the pullback low"
            ),
            evidence={
                "timeframe": "M5",
                "impulse_atr": round(abs(move) / atr, 2),
                "prior_range_atr": round(quiet_range / atr, 2),
                "pullback_atr": round(pullback, 2),
                "spread_share_of_stop": round(share, 3),
                "stop_distance": round(risk, 6),
                "typical_run_in_horizon": round(typical, 6),
            },
        )


class RangeFade:
    """A tested range extreme on M15 that price refuses to leave.

    The complement to the momentum theory, and the reason both exist: momentum
    continuation and range reversion are mutually exclusive readings of the same
    chart. A system carrying only one of them is not neutral about which regime
    it is in — it simply assumes one, and takes the wrong trade whenever the
    market is in the other.

    Requires the range to have been respected repeatedly, the current bar to
    have rejected the edge with a wick, and the middle of the range to be far
    enough away to be worth the trip after spread.
    """

    name = "range_fade"
    max_conviction = 90.0
    horizon_minutes = 180

    def __init__(self, config: FadeConfig) -> None:
        self.config = config

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:  # noqa: ARG002
        series = ctx.series.get(Timeframe.M15)
        if series is None or len(series.df) < 80 or ctx.tick is None:
            return None
        frame = series.df
        atr = _atr(frame)
        if atr <= 0:
            return None

        history = frame.iloc[-(self.config.range_bars + 1) : -1]
        top = float(history["high"].max())
        bottom = float(history["low"].min())
        height = top - bottom
        if height <= atr * self.config.min_range_atr:
            return None

        # A range worth fading is one price keeps returning from. A single touch
        # of each side is not a range, it is a leg that has not finished.
        near = atr * self.config.touch_tolerance_atr
        touches_top = int((history["high"] >= top - near).sum())
        touches_bottom = int((history["low"] <= bottom + near).sum())

        candle = frame.iloc[-1]
        body_high = max(float(candle["open"]), float(candle["close"]))
        body_low = min(float(candle["open"]), float(candle["close"]))
        upper_wick = float(candle["high"]) - body_high
        lower_wick = body_low - float(candle["low"])

        direction: Direction | None = None
        if (
            float(candle["high"]) >= top - near
            and float(candle["close"]) < top
            and upper_wick > lower_wick * 1.5
            and touches_top >= self.config.min_touches
        ):
            direction, edge, touches = Direction.SHORT, top, touches_top
        elif (
            float(candle["low"]) <= bottom + near
            and float(candle["close"]) > bottom
            and lower_wick > upper_wick * 1.5
            and touches_bottom >= self.config.min_touches
        ):
            direction, edge, touches = Direction.LONG, bottom, touches_bottom
        if direction is None:
            return None

        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid
        stop = edge - atr * self.config.stop_buffer_atr * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        floor = atr * self.config.min_stop_atr
        if risk < floor:
            stop = entry - floor * int(direction)
            risk = floor

        affordable, share = _spread_is_affordable(ctx, risk, self.config.max_spread_share_of_stop)
        if not affordable:
            return None

        # Target the middle of the range, not the far side: price reaches the
        # midpoint far more often than it crosses the whole range, and a target
        # that is usually reached beats a bigger one that usually is not.
        middle = bottom + height * 0.5
        target_distance = abs(middle - entry)
        if target_distance / risk < self.config.min_target_r:
            return None

        return Play(
            playbook=self.name,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=middle,
            conviction=min(90.0, 45.0 + touches * 6.0 + min(height / atr, 6.0) * 3.0),
            horizon_minutes=self.horizon_minutes,
            thesis=(
                f"M15 range of {height / atr:.1f} ATR held {touches} times; "
                f"current bar rejected the {'top' if direction is Direction.SHORT else 'bottom'} "
                f"with a wick. Target is the range midpoint, not the far side."
            ),
            evidence={
                "timeframe": "M15",
                "range_height_atr": round(height / atr, 2),
                "edge_touches": touches,
                "spread_share_of_stop": round(share, 3),
                "stop_distance": round(risk, 6),
                "range_top": round(top, 6),
                "range_bottom": round(bottom, 6),
            },
        )


class RangeBreak:
    """The range gave way. Trade with it, not against it.

    This is the theory `RangeFade` is missing, and until it existed the system
    was not neutral about which of the two was happening — it assumed the fade
    every time. A range that has held three times and then breaks is the single
    most common way a fade loses, because the same evidence that makes the fade
    attractive (a level respected again and again) is what makes the break
    significant when it finally goes.

    The operator watched it happen and said so: price printed a new high, the
    obvious read was continuation, and the system was short the top.

    **The most valuable thing this does is disagree.** `PlaybookEngine` stands
    every theory down when two of them point opposite ways on one chart, so on
    a genuine breakout this fires long, the fade fires short, and the engine
    takes neither. A losing trade avoided counts the same as a winning one
    taken, and it is available far more often.

    Three requirements, each closing a way breakouts fail:

    * The range must have been real — held repeatedly, the same test the fade
      applies, so the two are reading one definition of a range and not two.
    * The bar must *close* through the edge with its body, not poke through
      with a wick. A wick through the top is the fade's signal, and the two
      readings must never both be available on one bar.
    * The break must come with expansion. A drift through a level on an
      ordinary-sized bar is the level eroding, not breaking, and it is the
      shape that most often reverses straight back.
    """

    name = "range_break"
    max_conviction = 88.0
    horizon_minutes = 120

    def __init__(self, config: BreakConfig) -> None:
        self.config = config

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:  # noqa: ARG002
        series = ctx.series.get(Timeframe.M15)
        if series is None or len(series.df) < 80 or ctx.tick is None:
            return None
        frame = series.df
        atr = _atr(frame)
        if atr <= 0:
            return None

        history = frame.iloc[-(self.config.range_bars + 1) : -1]
        top = float(history["high"].max())
        bottom = float(history["low"].min())
        height = top - bottom
        if height <= atr * self.config.min_range_atr:
            return None

        near = atr * self.config.touch_tolerance_atr
        touches_top = int((history["high"] >= top - near).sum())
        touches_bottom = int((history["low"] <= bottom + near).sum())

        candle = frame.iloc[-1]
        close = float(candle["close"])
        body = abs(close - float(candle["open"]))
        # Body, not range: a bar with a huge wick and a small body travelled and
        # came back, which is rejection rather than a break.
        if body < atr * self.config.min_body_atr:
            return None

        direction: Direction | None = None
        if close > top + near and touches_top >= self.config.min_touches:
            direction, edge, touches = Direction.LONG, top, touches_top
        elif close < bottom - near and touches_bottom >= self.config.min_touches:
            direction, edge, touches = Direction.SHORT, bottom, touches_bottom
        if direction is None:
            return None

        # Already extended. A break found four bars late is a chase: the stop
        # has to sit back at the level, so the risk grows exactly as the
        # remaining move shrinks.
        travelled = abs(close - edge)
        if travelled > atr * self.config.max_extension_atr:
            return None

        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid
        # Behind the broken edge, because that is where the theory fails: a
        # break that gets taken back inside the range was not a break.
        stop = edge - atr * self.config.stop_buffer_atr * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        floor = atr * self.config.min_stop_atr
        if risk < floor:
            stop = entry - floor * int(direction)
            risk = floor

        affordable, share = _spread_is_affordable(ctx, risk, self.config.max_spread_share_of_stop)
        if not affordable:
            return None

        # A measured move: ranges tend to travel about their own height once
        # they let go. Capped in R so a tall range cannot manufacture a target
        # the market has no reason to reach.
        projected = height * self.config.projection
        target_distance = min(projected, risk * self.config.max_target_r)
        if target_distance / risk < self.config.min_target_r:
            return None
        target = entry + target_distance * int(direction)

        return Play(
            playbook=self.name,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            conviction=min(88.0, 42.0 + touches * 6.0 + min(body / atr, 3.0) * 6.0),
            horizon_minutes=self.horizon_minutes,
            thesis=(
                f"M15 range of {height / atr:.1f} ATR held {touches} times and then closed "
                f"through the {'top' if direction is Direction.LONG else 'bottom'} with a "
                f"{body / atr:.1f} ATR body. A level respected that often is worth something "
                f"when it finally gives way."
            ),
            evidence={
                "timeframe": "M15",
                "range_height_atr": round(height / atr, 2),
                "edge_touches": touches,
                "break_body_atr": round(body / atr, 2),
                "extension_atr": round(travelled / atr, 2),
                "spread_share_of_stop": round(share, 3),
                "stop_distance": round(risk, 6),
                "range_top": round(top, 6),
                "range_bottom": round(bottom, 6),
            },
        )


def _drift(frame: pd.DataFrame, bars: int) -> float:
    """How far the market has travelled over `bars`, against how far it wanders.

    A least-squares slope totalled over the window and divided by
    `sqrt(bars) * ATR` — roughly the displacement of a random walk over that
    many bars. The same normalisation the health reader uses, deliberately: a
    trend measured one way for entry and another way for management is two
    definitions of the word, and they would disagree at the worst moment.

    Dimensionless, so one threshold means the same thing on gold and on EURUSD.
    """
    atr = _atr(frame)
    if atr <= 0 or len(frame) < bars:
        return 0.0
    closes = frame["close"].tail(bars).to_numpy(dtype=float)
    positions = np.arange(len(closes), dtype=float)
    slope = float(np.polyfit(positions, closes, 1)[0])
    return slope * bars / (np.sqrt(bars) * atr)


class TrendPullback:
    """An established H1 trend that has paused, and is starting again.

    Neither existing theory covers this. `MomentumScalp` wants compression
    immediately before the move, which a trend that has been running for hours
    does not have; `RangeFade` wants a range, and a trend is the absence of one.
    So the most ordinary thing a market does — go one way, rest, go on — was
    invisible to a system with two theories in it.

    It is also the regime where a small account gets the best of both: the
    stop is a pullback, so it is short, but the context is an H1 trend, so the
    target is not the next few pips.

    The pullback has to be a real one. Too shallow and there is nothing to
    stand behind, so the stop lands inside the noise; too deep and it is not a
    pause any more, it is the trend ending, which is a different trade nobody
    here is proposing. And the current bar must have *turned back* — a pullback
    still falling is a pullback that has not finished.
    """

    name = "trend_pullback"
    max_conviction = 86.0
    horizon_minutes = 240

    def __init__(self, config: PullbackConfig) -> None:
        self.config = config

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:  # noqa: ARG002
        higher = ctx.series.get(Timeframe.H1)
        series = ctx.series.get(Timeframe.M15)
        if higher is None or series is None or ctx.tick is None:
            return None
        if len(higher.df) < self.config.trend_bars + 2 or len(series.df) < 60:
            return None

        drift = _drift(higher.df, self.config.trend_bars)
        if abs(drift) < self.config.min_trend_drift:
            return None
        direction = Direction.LONG if drift > 0 else Direction.SHORT

        frame = series.df
        atr = _atr(frame)
        if atr <= 0:
            return None

        # The extreme the trend reached before it paused, and where the leg
        # that reached it began.
        window = frame.iloc[-self.config.pullback_bars :]
        extreme = float(
            window["high"].max() if direction is Direction.LONG else window["low"].min()
        )
        leg = frame.iloc[
            -(self.config.pullback_bars + self.config.leg_bars) : -self.config.pullback_bars
        ]
        if leg.empty:
            return None
        origin = float(leg["low"].min() if direction is Direction.LONG else leg["high"].max())
        impulse = abs(extreme - origin)
        if impulse <= atr:
            return None

        # Depth as a share of the leg it is retracing, not in ATR.
        #
        # ATR was the obvious unit and it is the wrong one here: it is computed
        # from the recent bars, which during a pullback *are* the pullback. So
        # depth-in-ATR came out at roughly "how many bars has this been going
        # on", not "how far back has it come", and any pullback lasting more
        # than two or three bars was refused however shallow it was. A share of
        # the impulse has no such circularity.
        close = float(frame["close"].iloc[-1])
        retrace = abs(extreme - close) / impulse
        if not (self.config.min_retrace <= retrace <= self.config.max_retrace):
            return None

        # Turned back. A pullback still going is one that has not finished, and
        # entering into it is catching it rather than joining the trend.
        previous = float(frame["close"].iloc[-2])
        if (close - previous) * int(direction) <= 0:
            return None

        # Behind the pullback's own extreme: the level that says the pause was
        # actually a reversal.
        pivot = float(window["low"].min() if direction is Direction.LONG else window["high"].max())
        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid
        stop = pivot - atr * self.config.stop_buffer_atr * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        floor = atr * self.config.min_stop_atr
        if risk < floor:
            stop = entry - floor * int(direction)
            risk = floor
        if risk > atr * self.config.max_stop_atr:
            return None

        affordable, share = _spread_is_affordable(ctx, risk, self.config.max_spread_share_of_stop)
        if not affordable:
            return None

        # The extreme the trend already reached. A level the market has proved
        # it can trade at beats a projection it has never been near.
        target_distance = abs(extreme - entry)
        if target_distance / risk < self.config.min_target_r:
            return None
        capped = min(target_distance, risk * self.config.max_target_r)
        target = entry + capped * int(direction)

        return Play(
            playbook=self.name,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            conviction=min(
                86.0, 40.0 + min(abs(drift), 3.0) * 10.0 + (1.0 - abs(retrace - 0.45)) * 12.0
            ),
            horizon_minutes=self.horizon_minutes,
            thesis=(
                f"H1 drift of {drift:+.1f} over {self.config.trend_bars} bars; M15 gave back "
                f"{retrace:.0%} of its last leg and the final bar turned back with the trend. "
                f"Target is the extreme it already reached."
            ),
            evidence={
                "timeframe": "M15 in H1 context",
                "h1_drift": round(drift, 2),
                "retrace_of_leg": round(retrace, 2),
                "impulse": round(impulse, 6),
                "spread_share_of_stop": round(share, 3),
                "stop_distance": round(risk, 6),
                "trend_extreme": round(extreme, 6),
            },
        )


class FailedBreak:
    """Price left the range, could not stay out, and came back in.

    The third member of the range family and the one a person reacts to
    fastest: everyone who bought the break is now wrong and their stops sit
    just back inside. That is what makes the move away from a failed break
    quick — it is fuelled by the positions the break itself created.

    It cannot fire on the same bar as `RangeBreak`, by construction: that one
    needs a close outside the edge and this one needs a close back inside. They
    read the same event and disagree only about how it ended, which is the
    right way for two theories to be mutually exclusive.
    """

    name = "failed_break"
    max_conviction = 84.0
    horizon_minutes = 150

    def __init__(self, config: FailedBreakConfig) -> None:
        self.config = config

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:  # noqa: ARG002
        series = ctx.series.get(Timeframe.M15)
        if series is None or len(series.df) < 80 or ctx.tick is None:
            return None
        frame = series.df
        atr = _atr(frame)
        if atr <= 0:
            return None

        recent = self.config.break_lookback
        history = frame.iloc[-(self.config.range_bars + recent) : -recent]
        top = float(history["high"].max())
        bottom = float(history["low"].min())
        if top - bottom <= atr * self.config.min_range_atr:
            return None

        near = atr * self.config.touch_tolerance_atr
        if int((history["high"] >= top - near).sum()) < self.config.min_touches:
            top_respected = False
        else:
            top_respected = True
        bottom_respected = int((history["low"] <= bottom + near).sum()) >= self.config.min_touches

        attempt = frame.iloc[-recent:]
        close = float(frame["close"].iloc[-1])
        reclaim = atr * self.config.min_reclaim_atr

        direction: Direction | None = None
        if top_respected and float(attempt["high"].max()) > top + near and close < top - reclaim:
            direction, extreme, edge = Direction.SHORT, float(attempt["high"].max()), top
        elif (
            bottom_respected
            and float(attempt["low"].min()) < bottom - near
            and close > bottom + reclaim
        ):
            direction, extreme, edge = Direction.LONG, float(attempt["low"].min()), bottom
        if direction is None:
            return None

        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid
        # Beyond the high water mark of the failed attempt. If price goes back
        # through that, the break was real after all and simply took its time.
        stop = extreme + atr * self.config.stop_buffer_atr * int(direction) * -1
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        floor = atr * self.config.min_stop_atr
        if risk < floor:
            stop = entry - floor * int(direction)
            risk = floor
        if risk > atr * self.config.max_stop_atr:
            return None

        affordable, share = _spread_is_affordable(ctx, risk, self.config.max_spread_share_of_stop)
        if not affordable:
            return None

        # The far side, where `RangeFade` takes the midpoint. The two are
        # reading different fuel: a fade is betting on the absence of buyers
        # above, while this is betting on the presence of trapped ones, whose
        # stops sit further away and keep the move going once it starts.
        # Capped in R so a very tall range cannot invent a target.
        far = bottom if direction is Direction.SHORT else top
        target_distance = min(abs(far - entry), risk * self.config.max_target_r)
        if target_distance / risk < self.config.min_target_r:
            return None
        target = entry + target_distance * int(direction)

        return Play(
            playbook=self.name,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            conviction=min(
                84.0,
                44.0 + min(abs(extreme - edge) / atr, 2.0) * 8.0 + (abs(close - edge) / atr) * 8.0,
            ),
            horizon_minutes=self.horizon_minutes,
            thesis=(
                f"Price traded {abs(extreme - edge) / atr:.1f} ATR beyond the M15 range "
                f"{'top' if direction is Direction.SHORT else 'bottom'} and closed back inside. "
                f"Everyone who took that break is now wrong, and their stops are the fuel."
            ),
            evidence={
                "timeframe": "M15",
                "broken_edge": round(edge, 6),
                "attempt_extreme": round(extreme, 6),
                "overshoot_atr": round(abs(extreme - edge) / atr, 2),
                "reclaim_atr": round(abs(close - edge) / atr, 2),
                "spread_share_of_stop": round(share, 3),
                "stop_distance": round(risk, 6),
            },
        )


# --------------------------------------------------------------------- config ---


@dataclass(frozen=True, slots=True)
class ScalpConfig:
    impulse_bars: int = 4
    quiet_bars: int = 12
    min_impulse_atr: float = 1.2
    max_quiet_range_atr: float = 2.0
    max_pullback_atr: float = 0.8
    max_adverse_m1_atr: float = 0.4
    stop_buffer_atr: float = 0.25
    #: Hard ceiling on the stop, in ATR. Past this it is not a scalp any more
    #: and the swing engine is the right owner of the trade.
    max_stop_atr: float = 2.5
    #: Floor on the stop, in ATR of this playbook's own timeframe.
    #:
    #: There was a ceiling and no floor, and the floor is the one that was
    #: needed. A shallow pullback puts the pivot within a pip or two of entry,
    #: so the stop came out at a fraction of one M5 bar's range — the adviser
    #: rejected these all day in the same words: "the stop (1.7 pips) is far
    #: smaller than even the M5 ATR (3.8 pips)", "2.4-pip stop, 0.24 ATR H1".
    #:
    #: One ATR of the timeframe the plan lives on. Below that the stop is not
    #: measuring whether the pullback held, it is measuring whether the next
    #: ordinary bar happens to tick through it. The stop is widened to the floor
    #: rather than the play being dropped; if the target can no longer justify
    #: the honest stop, `min_target_r` refuses it a few lines later, which is
    #: the correct place for that answer.
    min_stop_atr: float = 1.0
    target_bars: int = 24
    target_r: float = 2.0
    min_target_r: float = 1.2
    #: Spread as a fraction of the stop. The binding constraint on any short
    #: stop: at 0.15 a 10-pip stop tolerates 1.5 pips of spread and no more.
    max_spread_share_of_stop: float = 0.15


@dataclass(frozen=True, slots=True)
class FadeConfig:
    range_bars: int = 48
    min_range_atr: float = 3.0
    touch_tolerance_atr: float = 0.3
    min_touches: int = 3
    stop_buffer_atr: float = 0.35
    #: Same floor as the scalp, in M15 ATR. A rejection wick at a range edge
    #: can leave the stop a hair beyond the edge itself, which is inside the
    #: noise that produced the wick.
    min_stop_atr: float = 1.0
    min_target_r: float = 1.2
    max_spread_share_of_stop: float = 0.12


@dataclass(frozen=True, slots=True)
class BreakConfig:
    """Deliberately shares `range_bars`, `min_range_atr`, `touch_tolerance_atr`
    and `min_touches` with `FadeConfig`. The two theories must agree on what a
    range *is*, or they would be arguing about different objects and the
    conflict rule that stands them both down would never fire."""

    range_bars: int = 48
    min_range_atr: float = 3.0
    touch_tolerance_atr: float = 0.3
    min_touches: int = 3
    #: Body of the breaking bar, in ATR. A drift through a level on an
    #: ordinary-sized bar is erosion, not a break, and it is the shape that
    #: most often comes straight back.
    min_body_atr: float = 0.8
    #: How far past the edge price may already be. Beyond this the stop has to
    #: sit back at the level while the remaining move has shrunk, which is the
    #: definition of a chase.
    max_extension_atr: float = 1.5
    stop_buffer_atr: float = 0.35
    min_stop_atr: float = 1.0
    #: Share of the range height projected past the break. Ranges tend to
    #: travel roughly their own height; asking for all of it every time is
    #: optimism, so this asks for most of it.
    projection: float = 0.8
    min_target_r: float = 1.3
    #: Ceiling in R, so a tall range cannot manufacture a target the market has
    #: no particular reason to reach.
    max_target_r: float = 3.0
    max_spread_share_of_stop: float = 0.12


@dataclass(frozen=True, slots=True)
class PullbackConfig:
    #: H1 bars the trend is measured over. Twenty-four is a day.
    trend_bars: int = 24
    #: Dimensionless drift — see `_drift`. One is a whole random-walk
    #: excursion's worth of one-way travel, which is a trend rather than a
    #: market that happened to finish higher than it started.
    min_trend_drift: float = 1.0
    pullback_bars: int = 12
    #: Bars before the pullback window in which the impulse leg is looked for.
    leg_bars: int = 24
    #: Depth as a share of the leg being retraced.
    #:
    #: Below the floor nothing has actually pulled back and there is no pivot
    #: to stand a stop behind. Above the ceiling the leg has been mostly given
    #: back, which is the trend ending rather than pausing — a different trade,
    #: and not one anything here is proposing.
    min_retrace: float = 0.2
    max_retrace: float = 0.7
    stop_buffer_atr: float = 0.3
    min_stop_atr: float = 1.0
    max_stop_atr: float = 3.0
    min_target_r: float = 1.5
    max_target_r: float = 4.0
    max_spread_share_of_stop: float = 0.12


@dataclass(frozen=True, slots=True)
class FailedBreakConfig:
    #: Same range definition as the fade and the break, again. Three theories,
    #: one object.
    range_bars: int = 48
    min_range_atr: float = 3.0
    touch_tolerance_atr: float = 0.3
    min_touches: int = 3
    #: Bars in which the break may have been attempted and undone.
    break_lookback: int = 3
    #: How far back inside the range price must have closed. Without a floor,
    #: a bar closing a hair under the edge would count, and that is still a
    #: break in progress.
    min_reclaim_atr: float = 0.3
    stop_buffer_atr: float = 0.3
    min_stop_atr: float = 1.0
    max_stop_atr: float = 3.0
    min_target_r: float = 1.2
    max_target_r: float = 3.0
    max_spread_share_of_stop: float = 0.12


@dataclass(frozen=True, slots=True)
class PlaybookVerdict:
    """Every theory's answer, and what their agreement says."""

    plays: tuple[Play, ...]
    conflict: bool
    note: str

    @property
    def best(self) -> Play | None:
        return self.plays[0] if self.plays else None

    def summary(self) -> dict[str, object]:
        return {
            "proposals": [play.summary() for play in self.plays],
            "playbooks_that_fired": len(self.plays),
            "directional_conflict": self.conflict,
            "note": self.note,
        }


class PlaybookEngine:
    """Runs every theory and reports what each of them saw."""

    def __init__(self, playbooks: list[Playbook], config: ConfluenceConfig) -> None:
        self.playbooks = playbooks
        self.config = config

    def evaluate(self, ctx: MarketContext, mode: TradingMode) -> PlaybookVerdict:
        plays: list[Play] = []
        for playbook in self.playbooks:
            try:
                play = playbook.propose(ctx, mode)
            except Exception:  # noqa: BLE001 - one broken theory must not silence the rest
                continue
            if play is not None:
                plays.append(play)
        plays.sort(key=lambda play: play.conviction, reverse=True)

        directions = {play.direction for play in plays}
        conflict = len(directions) > 1
        if not plays:
            note = "no theory found a setup on this market"
        elif conflict:
            # Not a tie to be broken by score. Two theories reading the same
            # bars in opposite directions is the definition of no edge, and
            # picking the higher number would be inventing one.
            note = (
                "two theories disagree on direction from the same chart. That is a market "
                "with no edge either way, not a close call to be settled by score."
            )
        else:
            note = f"{len(plays)} theor{'y' if len(plays) == 1 else 'ies'} agree on direction"
        return PlaybookVerdict(tuple(plays), conflict, note)
