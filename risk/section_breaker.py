"""When a new section is going badly, stop it. Automatically, and by itself.

Sections two, five and six went live on the owner's authorisation with ZERO
measured trades between them — every number behind them is reasoning, and
reasoning is what this exists to survive being wrong about.

WHY IT IS COMPUTED AND NOT STORED. The obvious build is a flag written when a
rule trips and read afterwards. That flag then has to survive a restart, stay
in step with trades resolving out of order, and be reset by hand without anyone
forgetting. Every one of those is a way to be wrong about whether real money is
switched on.

So there is no flag. The verdict is derived from the journal on every cycle:
the last N resolved trades this section was behind, and the rule applied to
them. A restart changes nothing, a late resolution corrects itself, and the
state cannot disagree with the evidence because it IS the evidence.

WHAT "THIS SECTION WAS BEHIND IT" MEANS. A trade is attributed to a module when
that module carried weight and scored in the direction actually taken — the
same reading `scripts/scorecard.py` uses for its detector table, deliberately,
so a section cannot look healthy in one place and tripped in another. A trade
several modules agreed on counts for each of them, which is the honest reading:
none of them can be credited alone for a decision the others also made.

RE-ARMING IS MANUAL, and that is the point. A breaker that switches itself back
on is a breaker that trips repeatedly on the same fault while the account pays
for each cycle of it. Turning a section back on means editing
`live_enabled_modules`, which shows up in a diff.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from config.schema import SectionBreakerConfig


@dataclass(frozen=True, slots=True)
class BreakerVerdict:
    """Whether a section may still trade, and the numbers behind the answer."""

    module: str
    tripped: bool
    reason: str
    trades: int
    losses: int
    streak: int

    @property
    def summary(self) -> str:
        return f"{self.module}: {self.reason}"


def _recent_outcomes(db: sqlite3.Connection, module: str, window: int) -> list[float] | None:
    """P&L in R of the last `window` closed trades this module was behind.

    Returns None when the journal cannot answer — an older database without
    `module_scores`, or a fresh one. A breaker that cannot see the evidence
    must not conclude anything from its absence, in either direction: silence
    is not a clean record and it is not a bad one.
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
        return None
    return [float(row[0]) for row in rows]


def assess(db: sqlite3.Connection, module: str, config: SectionBreakerConfig) -> BreakerVerdict:
    """Should this section still be allowed to open a position?"""
    if not config.enabled:
        return BreakerVerdict(module, False, "breaker disabled", 0, 0, 0)

    outcomes = _recent_outcomes(db, module, config.window)
    if outcomes is None:
        return BreakerVerdict(module, False, "no journal evidence available", 0, 0, 0)

    trades = len(outcomes)
    losses = sum(1 for value in outcomes if value <= 0)
    # `outcomes` is newest-first, so the streak reads from the front.
    streak = 0
    for value in outcomes:
        if value > 0:
            break
        streak += 1

    if trades < config.minimum_trades:
        return BreakerVerdict(
            module,
            False,
            f"{trades} of the {config.minimum_trades} trades needed to judge it",
            trades,
            losses,
            streak,
        )

    # TWO RULES, AND THE STREAK IS THE URGENT ONE.
    #
    # A share of losses over a window catches a section that is quietly wrong.
    # A run of consecutive losses catches one that is wrong RIGHT NOW, before
    # the window has filled enough for the share to move — which on a section
    # taking a handful of trades a day is the difference between stopping today
    # and stopping next week.
    if streak >= config.losing_streak:
        return BreakerVerdict(
            module,
            True,
            f"{streak} losses in a row",
            trades,
            losses,
            streak,
        )
    share = losses / trades
    if share > config.maximum_loss_share:
        return BreakerVerdict(
            module,
            True,
            f"{losses} losses in the last {trades} trades ({share:.0%})",
            trades,
            losses,
            streak,
        )
    average_r = sum(outcomes) / trades
    if config.minimum_average_r is not None and average_r < config.minimum_average_r:
        return BreakerVerdict(
            module,
            True,
            f"last {trades} trades average {average_r:+.3f}R",
            trades,
            losses,
            streak,
        )
    return BreakerVerdict(
        module,
        False,
        f"{trades - losses} of {trades} won, average {average_r:+.3f}R",
        trades,
        losses,
        streak,
    )


def tripped_modules(
    db: sqlite3.Connection, breakers: dict[str, SectionBreakerConfig]
) -> dict[str, BreakerVerdict]:
    """Every watched section that has stopped itself."""
    verdicts = {module: assess(db, module, config) for module, config in breakers.items()}
    return {module: verdict for module, verdict in verdicts.items() if verdict.tripped}
