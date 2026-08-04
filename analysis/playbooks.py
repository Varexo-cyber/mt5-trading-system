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

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None: ...


# --------------------------------------------------------------- shared bits ---


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = ranges.rolling(period).mean().iloc[-1]
    return 0.0 if pd.isna(value) else float(value)


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
    min_target_r: float = 1.2
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
