"""How the account should be *carrying itself* right now, given recent results.

Separate from the risk limits, and deliberately so. The limits answer "is this
allowed" — a closed set of hard gates that either pass or block. This answers a
softer question a good trader asks constantly and this system never did: *given
how the last few trades went, how should I be behaving?*

The honest answer, and the one that most people get backwards, is **narrower and
faster, never bigger**. After a run of losses the temptation is to size up and
chase the damage back. That is the single most reliable way to turn a bad week
into a dead account, because it raises exposure exactly when the evidence that
the edge is working is weakest. The arithmetic is brutal and worth stating: down
20%, you need +25% to get level; down 50%, you need +100%.

So posture only ever moves in the protective direction:

* **Losers get cut sooner.** Not winners — losers. A position that is not
  working gets less patience when the account is already down, because the
  thing that actually recovers a drawdown is having capital and slots free for
  the next good setup, not sitting in a dead trade hoping.
* **Fewer entries, and only the best ones.** Not smaller ones, and not a
  higher score bar — the candidates of a cycle are already ranked by
  conviction, so a drawdown simply cuts how far down that list the system is
  willing to go. In a drawdown it takes *better* trades, not *more* trades.
* **Size never moves here.** `RiskManager.risk_multiplier` owns sizing and only
  ever reduces it. Nothing in this module can raise a lot.

Posture is advisory context and a small number of mechanical tightenings. It
cannot loosen anything: every value it produces is clamped so that the cautious
state is always at least as strict as the normal one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Posture(StrEnum):
    """How the account is carrying itself, worst last."""

    #: Normal operation. Full patience, standard entry bar.
    STEADY = "steady"
    #: Recent losses. Cut dead trades sooner, demand more from a new entry.
    CAUTIOUS = "cautious"
    #: A real drawdown. Only clearly exceptional setups are worth a slot.
    DEFENSIVE = "defensive"


@dataclass(frozen=True, slots=True)
class PostureAssessment:
    """The posture, the numbers behind it, and what it changes."""

    posture: Posture
    consecutive_losses: int
    drawdown_pct: float
    #: Multiplier on `time_exit_hours`. Below 1.0 means a stalled trade is
    #: closed sooner. Never above 1.0.
    patience_multiplier: float
    #: How many of the cycle's ranked candidates may be acted on. None means
    #: all of them; 1 means only the single best setup found.
    #:
    #: Rank, not score, and the difference is not cosmetic. The first version
    #: added a fixed bonus to the confluence threshold, which assumes the
    #: threshold sits somewhere with headroom above it. It does not: the engine
    #: produces a narrow band — 2806 live decisions ranged 33 to 42 — so a +7
    #: bonus on a threshold of 40 moved the bar to 47 and shut the system
    #: completely, which is not "more selective", it is off. Worse, how much a
    #: fixed bonus bites depends entirely on where the threshold happens to sit
    #: in the distribution, so the same number means different things after any
    #: recalibration.
    #:
    #: A rank cutoff is scale-free. "Only the best setup of the cycle" means
    #: the same thing whatever the scores look like, always leaves exactly one
    #: candidate reachable, and is a truer statement of the intent: in a
    #: drawdown, take the best thing available rather than everything
    #: acceptable.
    max_candidates: int | None

    @property
    def is_stressed(self) -> bool:
        return self.posture is not Posture.STEADY

    def brief(self) -> dict[str, object]:
        """Prompt-ready summary for the reviewer and the supervisor."""
        return {
            "posture": self.posture.value,
            "consecutive_losses": self.consecutive_losses,
            "drawdown_from_peak_pct": round(self.drawdown_pct, 2),
            "candidates_allowed": self.max_candidates or "all",
            "guidance": _GUIDANCE[self.posture],
        }


_GUIDANCE = {
    Posture.STEADY: (
        "Normal conditions. Judge this on its own merits; recent results are not a "
        "reason to be either bolder or more timid."
    ),
    Posture.CAUTIOUS: (
        "The last few trades lost. That is not a reason to size up or to chase — it is a "
        "reason to be quicker to admit a trade is not working, and to want a bit more "
        "from a new one. Position size is fixed and is not yours to change."
    ),
    Posture.DEFENSIVE: (
        "The account is in a real drawdown. Protecting what is left is now worth more than "
        "any single opportunity: cut dead trades promptly, and only approve a setup you "
        "would call clearly exceptional. Do NOT attempt to win the drawdown back — the "
        "sizing is fixed, no trade here can recover it in one go, and trying is how a bad "
        "week becomes a dead account."
    ),
}


def assess(
    *,
    consecutive_losses: int,
    equity: float,
    equity_peak: float,
    cautious_after_losses: int = 2,
    defensive_after_losses: int = 4,
    defensive_drawdown_pct: float = 8.0,
) -> PostureAssessment:
    """Read the account's recent record and return how it should behave.

    Two independent triggers, because they catch different failures. A losing
    streak is the fast signal — four in a row means something has changed even
    if each loss was small. Drawdown from peak is the slow one, and it catches
    the case a streak counter misses entirely: a long grind of small losses
    with the occasional win keeping the streak at zero while the balance bleeds.
    """
    drawdown = 0.0
    if equity_peak > 0:
        drawdown = max(0.0, (equity_peak - equity) / equity_peak * 100.0)

    if consecutive_losses >= defensive_after_losses or drawdown >= defensive_drawdown_pct:
        posture = Posture.DEFENSIVE
    elif consecutive_losses >= cautious_after_losses or drawdown >= defensive_drawdown_pct / 2:
        posture = Posture.CAUTIOUS
    else:
        posture = Posture.STEADY

    # Both dials move only one way. Clamping here rather than trusting the
    # table means a future edit that fat-fingers a number above 1.0 tightens
    # nothing instead of silently granting extra patience in a drawdown.
    patience, allowed = _DIALS[posture]
    return PostureAssessment(
        posture=posture,
        consecutive_losses=consecutive_losses,
        drawdown_pct=drawdown,
        patience_multiplier=min(1.0, max(0.1, patience)),
        max_candidates=None if allowed is None else max(1, allowed),
    )


#: posture -> (patience on stalled trades, candidates of the cycle allowed).
#:
#: The patience numbers are the "cut losers sooner" half: at DEFENSIVE a trade
#: that has gone nowhere is closed in 40% of the usual time. The candidate
#: counts are the "demand more from a new entry" half — never fewer than one,
#: so the system tightens rather than switching itself off, and there is
#: deliberately no dial anywhere in this module that loosens either.
_DIALS: dict[Posture, tuple[float, int | None]] = {
    Posture.STEADY: (1.0, None),
    Posture.CAUTIOUS: (0.7, 3),
    Posture.DEFENSIVE: (0.4, 1),
}
