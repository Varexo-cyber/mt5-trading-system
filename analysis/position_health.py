"""Reading an open position the way a person reads it, once a second.

The mechanical rules — break-even, the trail, the give-back exit — all look at
one number: how many R the trade is up. That number says nothing about *why*.
A trade sitting at +0.4R with the market rolling over underneath it and a trade
sitting at +0.4R grinding steadily toward target are the same number and
opposite situations, and until now the fast layer could not tell them apart. It
had to wait for the supervisor's fifteen-minute visit to find out.

This module is the missing read. It is deliberately not an oracle: it looks at
four things a person would look at — has the structure that justified the trade
broken, has momentum turned, is price running against us bar after bar, has the
cost of getting out blown out — and says how bad it is.

Two design rules do most of the work here:

*One signal is noise, two agreeing is evidence.* Any single reader will fire on
a normal pullback. Exiting on one would bleed the account by a thousand cuts,
which is a worse failure than holding a loser slightly too long. Nothing short
of two independent readers agreeing can produce an exit.

*It may only ever reduce risk.* The verdicts are hold, tighten, secure and
exit. There is no verdict that adds size, widens a stop or reverses, so no
sequence of readings can turn this into a trade the risk layer never approved.
That is a property of the type, not of the caller being careful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from analysis.market_structure import find_swings

Action = Literal["hold", "tighten", "secure", "exit"]
#: `unmanaged` is not a reading, it is the absence of one, and it is a state
#: distinct from every other value here. The four real verdicts all mean "the
#: fast layer looked at this trade and concluded something". `unmanaged` means
#: the loop skipped the position before any reader ran — so no give-back, no
#: profit lock, no peak stall, no time exit. The trade is held by its broker
#: stop alone, and nothing else in this system is watching it.
#:
#: It exists because that state used to be invisible. A skipped position simply
#: never appeared in the health map, and the deck rendered the gap as "geen live
#: oordeel (draait Jarvis?)" — pointing at the one explanation the operator can
#: see with their own eyes is false, while the real one went unsaid.
Verdict = Literal["healthy", "watch", "deteriorating", "broken", "unmanaged"]

#: Below this the DRIFT readers are not consulted. Entry noise — the first tick
#: against you, the spread crossing — would otherwise read as a momentum turn
#: and close good trades before they can breathe.
#:
#: It used to return "too young to judge" for the whole function, and that was
#: the wrong scope. The noise it protects against is specifically a slope or a
#: run computed across the entry tick. Nothing else here has that problem: a
#: broken structure is broken whenever it breaks, a blown-out spread is blown
#: out, and the position's own excursion arms at 0.35R, which is far beyond any
#: spread crossing. Silencing those for two minutes meant a trade that went
#: violently wrong immediately after entry had nothing looking at it at all —
#: and "wrong within two minutes" is a stronger signal than "wrong within two
#: hours", not a weaker one.
MIN_AGE_MINUTES = 2.0

#: Bars of the fast timeframe each drift reader needs to have formed *since the
#: entry* before it is allowed to speak.
#:
#: Stated per reader because one number is wrong for both: the slope reads
#: twelve bars, the run reads six. And stated at all because `MIN_AGE_MINUTES`
#: alone does not do this job — at two minutes old, ten of the slope's twelve
#: bars are from before the trade existed, which is the same stretch of chart
#: the playbook read when it decided to enter. Entry and health then draw
#: opposite conclusions from one set of bars, and health wins because it is
#: asked again every second.
#:
#: Live evidence, 6 August: three trades were cut at 2:14, 2:19 and 2:38
#: against a two-minute floor, for -0.16R, -0.71R and -0.48R. Not one of them
#: reached its stop; the reader closed all three within seconds of becoming
#: eligible to.
_MOMENTUM_BARS = 12
_RUN_BARS = 6

#: Severity thresholds for the combined read. Chosen so that "broken" is out of
#: reach for any single reader at full strength, which is what makes the
#: two-signal rule structural rather than a comment.
BROKEN_AT = 0.75
DETERIORATING_AT = 0.45
WATCH_AT = 0.20


#: Which independent thing a reader is evidence *of*.
#:
#: This is not cosmetic grouping. `momentum_turned` and `adverse_run` are two
#: views of one fact — price is moving against us — and counting them as two
#: signals would satisfy the two-signal rule with a single observation seen
#: twice, which is exactly the failure that rule exists to prevent. Corroboration
#: is required across families, and severity within a family takes the strongest
#: reader rather than adding them up.
#: `trajectory` is the position's OWN path, and it is the only family here that
#: does not read the chart. That is why it exists: the three chart families each
#: need bars to have formed since the entry — twelve for the slope, six for the
#: run — and `corroborated` needs two families to agree, so before twelve
#: minutes agreement is impossible by construction. A trade that goes wrong
#: inside its first quarter of an hour had nothing watching it at all.
Family = Literal["structure", "drift", "liquidity", "trajectory"]


@dataclass(frozen=True, slots=True)
class HealthSignal:
    """One thing that looks wrong, and how wrong."""

    name: str
    severity: float
    detail: str
    family: Family = "drift"


@dataclass(frozen=True, slots=True)
class PositionHealth:
    """What the fast layer thinks of a position right now."""

    verdict: Verdict
    severity: float
    action: Action
    signals: tuple[HealthSignal, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.action != "hold"

    def summary(self) -> dict[str, object]:
        """Compact enough to sit in a prompt or a log line."""
        return {
            "verdict": self.verdict,
            "severity": round(self.severity, 2),
            "action": self.action,
            # Already computed, and the one field this used to drop. On an
            # ordinary reading it condenses the signals into a line; on an
            # `unmanaged` verdict it is the entire answer, because there are no
            # signals and the verdict alone only says a reading was not taken.
            "reason": self.reason,
            "signals": [
                {"name": s.name, "severity": round(s.severity, 2), "detail": s.detail}
                for s in self.signals
            ],
        }


@dataclass(frozen=True, slots=True)
class HealthWeights:
    """How much each family contributes at full strength.

    Chosen so no family reaches `BROKEN_AT` alone and any two of them do. A
    broken structure outweighs the rest because it is the only one that says
    the *reason for the trade* is gone rather than that the market is noisy.
    """

    structure: float = 0.45
    drift: float = 0.35
    liquidity: float = 0.30
    #: Deliberately the smallest. On its own it must not be able to do anything
    #: — "this trade is down" is not a reason to close a trade, it is the
    #: ordinary condition of a trade that has not finished yet. Its whole job is
    #: to be the SECOND family, so a chart reader that is already saying
    #: something is wrong can be acted on at minute six instead of minute
    #: twelve.
    #:
    #: 0.30 and not 0.28, so that `structure` + `trajectory` lands exactly on
    #: BROKEN_AT. That pair is the one combination that should be able to close
    #: a trade at any age, including inside the first two minutes: the swing
    #: that justified the position has broken AND the market has already taken
    #: real money. Neither is a clock reading and neither needs a bar to form,
    #: so waiting adds nothing except the loss.
    #:
    #: Every other pairing still falls short — drift 0.65, liquidity 0.60 — and
    #: alone it is 0.30 against a 0.45 `deteriorating` floor, so "this trade is
    #: down" on its own remains a hold.
    trajectory: float = 0.30

    def of(self, family: Family) -> float:
        return {
            "structure": self.structure,
            "drift": self.drift,
            "liquidity": self.liquidity,
            "trajectory": self.trajectory,
        }[family]


# ------------------------------------------------------------------ readers ---


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    if len(frame) < 2:
        return 0.0
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(period).mean())


def structure_broken(
    frame: pd.DataFrame, sign: int, *, lookback: int = 2, buffer_atr: float = 0.1
) -> HealthSignal | None:
    """Has price closed through the swing that was holding the trade up?

    For a long, that is the most recent confirmed swing low. Losing it is the
    single most informative thing that can happen to an open trade: it does not
    mean the market is against us for a moment, it means the shape that made
    the trade worth taking is no longer there.

    Confirmed swings only. An unconfirmed pivot is a low that has not finished
    being a low, and treating it as broken would fire on every wick.
    """
    atr = _atr(frame)
    if atr <= 0 or len(frame) < lookback * 2 + 2:
        return None
    swings = find_swings(frame, lookback, "internal")
    kind = "low" if sign > 0 else "high"
    pivot = next((swing for swing in reversed(swings) if swing.kind == kind), None)
    if pivot is None:
        return None
    close = float(frame["close"].iloc[-1])
    buffer_ = atr * buffer_atr
    through = (close < pivot.price - buffer_) if sign > 0 else (close > pivot.price + buffer_)
    if not through:
        return None
    distance_atr = abs(close - pivot.price) / atr
    return HealthSignal(
        "structure_broken",
        min(1.0, distance_atr / 0.75),
        f"closed {distance_atr:.2f} ATR through the last swing {kind} at {pivot.price:.5f}",
        "structure",
    )


def drift_score(frame: pd.DataFrame, sign: int, *, bars: int = 12) -> float | None:
    """How hard the market is running in `sign`, in random-walk units.

    A least-squares slope over the last `bars` closes, totalled and divided by
    `sqrt(bars) * ATR` — roughly how far a market wanders over that many bars
    for no reason at all. Positive means running our way, negative against.

    One definition, used twice. `momentum_turned` asks whether this is
    strongly negative; the banking rule asks whether it is strongly positive.
    Two separately derived versions of "is this still moving" would eventually
    disagree, and they would disagree while a position was open.
    """
    atr = _atr(frame)
    if atr <= 0 or len(frame) < bars:
        return None
    closes = frame["close"].tail(bars).to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(closes), dtype=float), closes, 1)[0])
    return slope * sign * bars / atr / np.sqrt(bars)


def momentum_turned(
    frame: pd.DataFrame,
    sign: int,
    *,
    bars: int = 12,
    threshold: float = 0.25,
    saturate: float = 1.5,
) -> HealthSignal | None:
    """Is the recent drift running against the position?

    A least-squares slope over the last `bars` closes, totalled across the
    window. A slope rather than "last close versus first" because one spike at
    either end of the window would otherwise decide the answer.

    The total is then divided by `sqrt(bars) * ATR`, which is roughly how far a
    market wanders over that many bars for no reason at all. That makes the
    reading dimensionless — the same number means the same thing on gold, on
    EURUSD, and on any window length — and it puts the thresholds on a scale
    where they can be reasoned about: 0.25 is a quarter of an ordinary
    excursion, 1.5 is well past anything a quiet market produces.

    Getting this scale wrong is not cosmetic. Normalising by ATR alone put full
    severity at 1.5 ATR across the window, which is *less* than a random walk
    covers, so the reader sat pinned at 1.0 on every trending market and had
    stopped distinguishing a drift from a collapse. A reader that always says
    the same thing is not evidence.
    """
    against = drift_score(frame, -sign, bars=bars)
    if against is None:
        return None
    drift = against
    total_atr = drift * np.sqrt(bars)
    if drift < threshold:
        return None
    fraction = min(1.0, (drift - threshold) / max(saturate - threshold, 1e-9))
    return HealthSignal(
        "momentum_turned",
        0.4 + 0.6 * fraction,
        f"last {bars} bars drifting against the position by {total_atr:.2f} ATR",
        "drift",
    )


def adverse_run(
    frame: pd.DataFrame, sign: int, *, window: int = 5, minimum: int = 4
) -> HealthSignal | None:
    """Bar after bar closing against us — the market is not hesitating.

    Same family as the slope, deliberately. A run of small adverse bars can sit
    inside the slope threshold while being exactly the persistent one-way
    pressure a person reacts to, so it is worth reading separately — but it is
    the same underlying fact, and treating the two as independent corroboration
    would let one drift satisfy the two-signal rule on its own.
    """
    if len(frame) < window + 1:
        return None
    closes = frame["close"].tail(window + 1).to_numpy(dtype=float)
    steps = np.diff(closes)
    against = int(sum(1 for step in steps if step * sign < 0))
    if against < minimum:
        return None
    return HealthSignal(
        "adverse_run",
        min(1.0, 0.5 + 0.5 * (against - minimum) / max(1, window - minimum)),
        f"{against} of the last {window} bars closed against the position",
        "drift",
    )


def _weighted(
    signal: HealthSignal | None, bars_since_entry: float, bars_needed: int
) -> HealthSignal | None:
    """Scale a reading by how much of its window is actually about this trade.

    The alternative was silence below the bar count, and silence says "no
    evidence" when the truth is "partial evidence". On an M1 fast frame that
    silenced the whole drift family for the first twelve minutes of every trade.

    A slope reading two bars into a twelve-bar window is a sixth about this
    position and five sixths about the chart the entry already judged, so it
    contributes a sixth. At the full count it contributes everything, which is
    the behaviour the gate used to produce the moment it opened.

    Dropped entirely below a tenth: a reading that thin is arithmetic on noise,
    and carrying it costs a signal slot and a line in every log.
    """
    if signal is None or bars_needed <= 0:
        return signal
    share = min(1.0, max(0.0, bars_since_entry / bars_needed))
    if share < 0.1:
        return None
    if share >= 1.0:
        return signal
    return HealthSignal(
        signal.name,
        signal.severity * share,
        f"{signal.detail} (weighted to {share:.0%}: {bars_since_entry:.0f} of "
        f"{bars_needed} bars are since the entry)",
        signal.family,
    )


def adverse_excursion(r_now: float, *, arm_r: float = 0.35) -> HealthSignal | None:
    """How far this trade has gone wrong — the one reader that skips the chart.

    Every other reader here needs bars to have formed SINCE the entry: twelve
    for the momentum slope, six for the adverse run, and both belong to the same
    family so they count once. `corroborated` requires two families. On an index
    or a metal the fast frame is M1, so before minute twelve two-family
    agreement is arithmetically impossible and the strongest verdict available
    is a demoted one.

    A live UK100 long shows the shape exactly. Held fifteen minutes, and the
    first reading that could act arrived at fourteen — by then it was -0.83R of
    a -0.99R worst case. The layer was not wrong, it was not allowed to speak.

    The position's own excursion needs no bars. It is available from
    `MIN_AGE_MINUTES` and it is genuinely independent of the chart readers: they
    describe the market, this describes what the market has done to us.

    Deliberately weak on its own — see `HealthWeights.trajectory`. Being down is
    the ordinary condition of an unfinished trade, not a reason to close it. The
    job is to be the SECOND family, so a chart reader already saying something
    is wrong can be acted on at minute six instead of minute twelve.

    `arm_r` is where "down" becomes "materially down". Below it there is nothing
    to corroborate; a trade a fifth of an R offside is inside its own noise.
    """
    if r_now >= -arm_r or arm_r <= 0:
        return None
    beyond = (abs(r_now) - arm_r) / arm_r
    return HealthSignal(
        "adverse_excursion",
        min(1.0, 0.4 + beyond),
        f"trade is {r_now:.2f}R offside, past the {arm_r:.2f}R the plan calls noise",
        "trajectory",
    )


def spread_blowout(spread: float, risk: float, *, limit: float = 0.25) -> HealthSignal | None:
    """Has getting out become expensive?

    A spread that widens to a large share of the trade's own risk is both a
    direct cost and a symptom — it is what liquidity withdrawal and unscheduled
    news look like from inside the terminal. Measured against the position's
    risk rather than in pips, so it means the same on a scalp and a swing.
    """
    if risk <= 0 or spread <= 0:
        return None
    share = spread / risk
    if share <= limit:
        return None
    return HealthSignal(
        "spread_blowout",
        min(1.0, 0.4 + (share - limit) / limit),
        f"spread is {share:.0%} of the trade's risk",
        "liquidity",
    )


# ------------------------------------------------------------------ verdict ---


def assess_position(
    *,
    sign: int,
    r_now: float,
    age_minutes: float,
    fast: pd.DataFrame | None,
    structure: pd.DataFrame | None,
    spread: float = 0.0,
    risk: float = 0.0,
    weights: HealthWeights | None = None,
    secure_at_r: float = 0.5,
    tighten_at_r: float = 0.2,
    fast_bar_minutes: float = 1.0,
) -> PositionHealth:
    """Combine the readers into one verdict and one permitted action.

    `fast` is the short timeframe the momentum and run readers work on; the
    structure reader deliberately uses `structure`, a slower one, because a
    swing on M1 is noise wearing the word "structure".

    The drift readers are additionally held back until their own window has
    filled with bars from after the entry — see `_MOMENTUM_BARS`. The structure
    reader is not: the swing holding a trade up legitimately formed before it,
    and closing through that swing is a real break whenever it happens.
    """
    weights = weights or HealthWeights()

    signals: list[HealthSignal] = []
    if structure is not None and not structure.empty:
        found = structure_broken(structure, sign)
        if found is not None:
            signals.append(found)
    since_entry = age_minutes / fast_bar_minutes if fast_bar_minutes > 0 else 0.0
    if fast is not None and not fast.empty and age_minutes >= MIN_AGE_MINUTES:
        # A READER ALWAYS SPEAKS. WHAT IT SAYS IS WORTH WHAT IT HAS SEEN.
        #
        # These used to be hard gates: below twelve bars the slope said nothing
        # at all, below six the run said nothing at all. The reason was sound —
        # at two minutes old, ten of the slope's twelve bars are from before the
        # trade existed, which is the same stretch of chart the entry read, and
        # letting that count in full means entry and health draw opposite
        # conclusions from one set of bars while health gets asked again every
        # second. Three trades on 6 August were cut at 2:14, 2:19 and 2:38 for
        # exactly that.
        #
        # But silence is the wrong way to express "I have only seen part of
        # this". It made the whole layer blind for the first twelve minutes of
        # every trade on an M1 frame, which is most of the life of a fast one —
        # the live UK100 long was over in fifteen and got its first actionable
        # reading at fourteen.
        #
        # Partial evidence, weighted as partial. A slope with two of its twelve
        # bars since the entry carries a sixth of its strength; at twelve bars
        # it carries all of it. Nothing is silenced and nothing pre-entry is
        # counted at face value, and the 6 August failure stays out of reach by
        # arithmetic: a two-minute-old reading is far too weak to corroborate
        # into an exit no matter what it sees.
        drift = [
            _weighted(momentum_turned(fast, sign), since_entry, _MOMENTUM_BARS),
            _weighted(adverse_run(fast, sign), since_entry, _RUN_BARS),
        ]
        signals.extend(found for found in drift if found)
    blown = spread_blowout(spread, risk)
    if blown is not None:
        signals.append(blown)
    # The position's own path. No bars required, so this is the only reader that
    # can corroborate anything before the drift family becomes eligible — which
    # is the whole reason a trade that went wrong in fifteen minutes had nothing
    # watching it.
    offside = adverse_excursion(r_now)
    if offside is not None:
        signals.append(offside)

    # Strongest reader per family, then sum across families. Adding two readers
    # of the same family would double-count one observation and inflate the
    # severity that the thresholds below are read against.
    strongest: dict[Family, float] = {}
    for signal in signals:
        strongest[signal.family] = max(strongest.get(signal.family, 0.0), signal.severity)
    severity = min(1.0, sum(value * weights.of(family) for family, value in strongest.items()))
    verdict: Verdict = "healthy"
    if severity >= BROKEN_AT:
        verdict = "broken"
    elif severity >= DETERIORATING_AT:
        verdict = "deteriorating"
    elif severity >= WATCH_AT:
        verdict = "watch"

    action: Action = "hold"
    if verdict == "broken":
        action = "secure" if r_now >= secure_at_r else "exit"
    elif verdict == "deteriorating":
        if r_now >= secure_at_r:
            action = "secure"
        elif abs(r_now) >= tighten_at_r:
            # Was `elif r_now >= tighten_at_r`, and that floor is why a losing
            # trade had nothing between its entry and its stop.
            #
            # A deteriorating read on a position already under water fell
            # through to `hold`. Not "hold because the thesis survives" — hold
            # because the only rung below `secure` demanded 0.2R of profit
            # first. So the reading that says "this is going the wrong way" was
            # produced, logged, and could act on every trade EXCEPT the ones it
            # was describing.
            #
            # Three of six trades on 15 August ran from entry to the full stop
            # with no rule able to touch them: CHFJPY peaked at 0.12R, UK100 at
            # 0.00R, ENR at 0.47R. Every protective rule on the account arms at
            # 0.5R or higher, and this one, the only one that could have spoken
            # earlier, required profit it never had.
            #
            # Tightening is the correct response and it always was: it costs
            # nothing when the read is wrong — the trade continues, with less
            # at stake — and it takes a smaller loss when the read is right.
            # The level itself belongs to the caller, which knows the original
            # stop and the live price; what is decided here is only that a
            # deteriorating trade has earned a tighter stop whether it is up or
            # down.
            #
            # `abs` rather than a dropped floor, so the knob keeps meaning
            # something: a trade sitting within 0.2R of its entry has not moved
            # enough for a stop move to be anything but noise, in either
            # direction.
            action = "tighten"

    # One thing wrong is noise; two independent things agreeing is evidence.
    # Counted in families, not signals: a steady drift trips both the slope and
    # the run reader, and letting that pass as corroboration would satisfy this
    # rule with a single observation seen twice — which is the exact failure the
    # rule exists to prevent. Enforced here rather than left to the weights, so
    # retuning one cannot quietly remove the safeguard. A lone family may still
    # tighten the stop: that costs nothing if it is wrong.
    corroborated = len(strongest) >= 2
    if not corroborated:
        if verdict == "broken":
            verdict = "deteriorating"
        if action in ("secure", "exit"):
            action = "tighten"

    names = ", ".join(signal.name for signal in signals)
    return PositionHealth(
        verdict,
        severity,
        action,
        tuple(signals),
        f"{verdict} ({severity:.2f}): {names}" if signals else "nothing wrong",
    )
