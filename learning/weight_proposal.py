"""Read what each detector actually earned, and propose a weight for it.

THE GAP THIS FILLS. `ConfigControl` can already version a config, run a
candidate alongside the live one, measure the paired lift with a confidence
bound, and promote it only when the bound clears zero. It refuses anything
that is not a weight change, refuses adding or removing a module, refuses
switching on a module that currently weighs nothing, and caps any single move
at 15%. That is a careful, complete promotion pipeline.

Nothing ever wrote a candidate for it. The whole apparatus has sat there
waiting for a proposal that only ever arrived as a human editing YAML by
judgement — which on this account has meant reading an 8-trade bucket after a
bad afternoon and deciding something. This is that missing half, and its only
job is to be harder to fool than that.

WHAT IT REFUSES TO CONCLUDE, which matters more than what it proposes:

* Fewer than `minimum_trades` on a module: nothing. A module with nine trades
  has no measured record, it has an anecdote.
* A mean that a confidence interval cannot separate from zero: nothing. "Lost
  money" and "lost money detectably" are different findings and only the
  second is worth acting on. This is the exact discipline that was missing
  when a 44-trade bucket got a regime switched off and the next day that
  regime was the best on the card.
* A module already at zero: nothing. Turning one back on is a decision with
  no evidence behind it by definition, and `ConfigControl` refuses it anyway.

WHAT IT PROPOSES is a nudge, never a verdict. The size of the step scales with
how far the evidence sits from zero and is capped well inside the promotion
pipeline's own 15% limit, so a module that is genuinely bad walks down over
several honest measurements rather than being sentenced on one.

Nothing here changes live behaviour. It writes a candidate. The shadow test
decides.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

#: Never step a weight further than this in one proposal, as a share of its
#: current value. `ConfigControl` caps at 15% and refuses beyond it; staying
#: under that means a proposal is never rejected for being too eager, and
#: leaves room for a human to make a larger call deliberately.
MAX_STEP = 0.10

#: A weight this small is off in all but name. Below it, propose zero and let
#: a human decide whether the module comes back — `ConfigControl` will not
#: reinstate it automatically.
FLOOR = 0.05

#: Below this, a proposal is not worth a shadow test. A decided but tiny edge
#: produces a change like 0.6 -> 0.6025, and running a week of paired
#: measurement to settle a quarter of a percent spends the one scarce thing
#: this account has — time with real trades in it — on a difference nothing
#: could detect afterwards.
MINIMUM_STEP = 0.02

#: The per-trade edge at which a step reaches its full size.
#:
#: Detectors on this account live between about -0.3R and +0.5R a trade, so
#: this is the top of the range rather than a round number: a module losing
#: 0.30R decisively has earned the whole step, one losing 0.05R decisively
#: has earned a sixth of it. Set too low, everything that clears the
#: significance bar gets the maximum move and the scaling does nothing, which
#: is what a first pass at 0.10 actually did.
SATURATION_R = 0.30


@dataclass(frozen=True, slots=True)
class ModuleEvidence:
    """What one detector's trades actually did, and how sure we are."""

    module: str
    trades: int
    mean_r: float
    lower_r: float
    upper_r: float

    @property
    def decided(self) -> bool:
        """Does the interval sit entirely on one side of zero?"""
        return self.lower_r > 0.0 or self.upper_r < 0.0

    def describe(self) -> str:
        if not self.trades:
            return f"{self.module}: no attributed trades"
        band = f"[{self.lower_r:+.3f}, {self.upper_r:+.3f}]"
        verdict = "decided" if self.decided else "inside the noise"
        return (
            f"{self.module}: {self.trades} trades, {self.mean_r:+.3f}R each, "
            f"95% {band} — {verdict}"
        )


@dataclass(frozen=True, slots=True)
class WeightProposal:
    module: str
    current: float
    proposed: float
    evidence: ModuleEvidence
    why: str

    @property
    def changed(self) -> bool:
        return abs(self.proposed - self.current) > 1e-9


def measure(
    db: sqlite3.Connection, module: str, *, window: int = 400, confidence: float = 0.95
) -> ModuleEvidence:
    """Every closed trade this module was behind, and the mean's interval.

    Attribution matches `risk/section_breaker.py` and the scorecard's detector
    table exactly — weighted, scored, and pointing the way the trade actually
    went. A detector cannot earn its keep in one report and lose it in
    another.
    """
    try:
        rows = db.execute(
            """
            SELECT t.pnl_r
            FROM trades t
            JOIN module_scores m ON m.cycle_pk = t.cycle_pk
            WHERE m.module = ?
              AND m.weight > 0
              AND m.score != 0
              AND (m.score > 0) = (t.direction = 'LONG')
              AND t.closed_at IS NOT NULL
              AND t.pnl_r IS NOT NULL
            ORDER BY t.closed_at DESC
            LIMIT ?
            """,
            (module, window),
        ).fetchall()
    except sqlite3.OperationalError:
        return ModuleEvidence(module, 0, 0.0, 0.0, 0.0)

    values = [float(row[0]) for row in rows]
    count = len(values)
    if count < 2:
        return ModuleEvidence(module, count, values[0] if values else 0.0, 0.0, 0.0)

    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    # The standard error of the MEAN, not the spread of the trades. The
    # question is "is this detector's average different from zero", and a
    # strategy with wide outcomes can still have a well-determined average
    # once there are enough of them.
    half = NormalDist().inv_cdf(0.5 + confidence / 2.0) * sqrt(variance / count)
    return ModuleEvidence(module, count, mean, mean - half, mean + half)


def propose(
    evidence: ModuleEvidence, current: float, *, minimum_trades: int = 30
) -> WeightProposal:
    """One module's next weight, given what it earned."""
    if current <= 0.0:
        return WeightProposal(
            evidence.module,
            current,
            current,
            evidence,
            "carries no weight; switching one back on needs a decision, not a measurement",
        )
    if evidence.trades < minimum_trades:
        return WeightProposal(
            evidence.module,
            current,
            current,
            evidence,
            f"{evidence.trades} of the {minimum_trades} trades needed to judge it",
        )
    if not evidence.decided:
        return WeightProposal(
            evidence.module,
            current,
            current,
            evidence,
            f"{evidence.mean_r:+.3f}R each, but the interval crosses zero — "
            f"this is not yet distinguishable from luck",
        )

    # THE STEP SCALES WITH THE PART OF THE EVIDENCE THAT IS BEYOND DOUBT, not
    # with the headline mean. `lower_r` on a losing module is the least bad
    # the record can plausibly be; using it means a wide, noisy result moves
    # the weight less than a tight one saying the same thing.
    edge = evidence.lower_r if evidence.mean_r > 0 else evidence.upper_r
    strength = min(1.0, abs(edge) / SATURATION_R)
    step = MAX_STEP * strength * (1.0 if edge > 0 else -1.0)
    if abs(step) < MINIMUM_STEP:
        return WeightProposal(
            evidence.module,
            current,
            current,
            evidence,
            f"{evidence.trades} trades at {evidence.mean_r:+.3f}R each — decided, but the "
            f"edge is too small to be worth a week of shadow testing",
        )
    proposed = round(min(1.0, current * (1.0 + step)), 4)
    if proposed < FLOOR:
        proposed = 0.0
    direction = "earned more" if step > 0 else "cost money"
    return WeightProposal(
        evidence.module,
        current,
        proposed,
        evidence,
        f"{evidence.trades} trades at {evidence.mean_r:+.3f}R each, 95% interval "
        f"{'above' if edge > 0 else 'below'} zero — {direction}, so "
        f"{current:g} -> {proposed:g}",
    )


def proposals(
    db: sqlite3.Connection,
    weights: dict[str, float],
    *,
    minimum_trades: int = 30,
    window: int = 400,
) -> list[WeightProposal]:
    """One entry per weighted module, changed or not.

    Unchanged modules are returned too, deliberately. A report that lists only
    the movers reads as though everything else was examined and found fine,
    when most of them were simply too thin to judge — and that difference is
    the whole point of this file.
    """
    return [
        propose(measure(db, module, window=window), weight, minimum_trades=minimum_trades)
        for module, weight in weights.items()
    ]
