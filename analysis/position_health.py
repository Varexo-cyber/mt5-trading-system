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

#: Below this the trade is too young to judge. Entry noise — the first tick
#: against you, the spread crossing — would otherwise read as a momentum turn
#: and close good trades before they can breathe.
MIN_AGE_MINUTES = 2.0

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
Family = Literal["structure", "drift", "liquidity"]


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

    def of(self, family: Family) -> float:
        return {"structure": self.structure, "drift": self.drift, "liquidity": self.liquidity}[
            family
        ]


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
    atr = _atr(frame)
    if atr <= 0 or len(frame) < bars:
        return None
    closes = frame["close"].tail(bars).to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(closes), dtype=float), closes, 1)[0])
    total_atr = -slope * sign * bars / atr  # ATR moved against us across the window
    drift = total_atr / np.sqrt(bars)
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
) -> PositionHealth:
    """Combine the readers into one verdict and one permitted action.

    `fast` is the short timeframe the momentum and run readers work on; the
    structure reader deliberately uses `structure`, a slower one, because a
    swing on M1 is noise wearing the word "structure".
    """
    weights = weights or HealthWeights()
    if age_minutes < MIN_AGE_MINUTES:
        return PositionHealth("healthy", 0.0, "hold", (), "too young to judge")

    signals: list[HealthSignal] = []
    if structure is not None and not structure.empty:
        found = structure_broken(structure, sign)
        if found is not None:
            signals.append(found)
    if fast is not None and not fast.empty:
        signals.extend(
            found for found in (momentum_turned(fast, sign), adverse_run(fast, sign)) if found
        )
    blown = spread_blowout(spread, risk)
    if blown is not None:
        signals.append(blown)

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
        elif r_now >= tighten_at_r:
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
