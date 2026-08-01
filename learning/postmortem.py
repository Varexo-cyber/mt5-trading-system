"""Conservative learning proposals; this module never edits live configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from journal.database import Journal, iso


@dataclass(frozen=True, slots=True)
class ModuleEvidence:
    module: str
    trades: int
    wins: int
    posterior_mean: float
    interval_90: tuple[float, float]
    expectancy_r: float
    current_weight: float
    proposed_weight: float | None
    proposal_reason: str


@dataclass(frozen=True, slots=True)
class Postmortem:
    trades: int
    more_trades_needed: int
    score_buckets: dict[str, tuple[int, float]]
    average_mae_r: float | None
    average_mfe_r: float | None
    modules: tuple[ModuleEvidence, ...]


class PostmortemAnalyzer:
    """Bayesian summaries and bounded proposals from at least 100 old trades."""

    def __init__(self, journal: Journal, weights: dict[str, float]) -> None:
        self.journal = journal
        self.weights = weights

    def analyze(self, start: datetime, end: datetime) -> Postmortem:
        trades = self.journal.query(
            "SELECT t.id, t.pnl_r, t.mae_r, t.mfe_r, t.opened_at, c.total_score "
            "FROM trades t LEFT JOIN analysis_cycles c ON c.id = t.cycle_pk "
            "WHERE t.closed_at IS NOT NULL AND t.opened_at >= ? AND t.opened_at < ?",
            (iso(start), iso(end)),
        )
        score_buckets: dict[str, list[float]] = {"<60": [], "60-70": [], "70-80": [], "80+": []}
        for row in trades:
            score = float(row["total_score"] or 0.0)
            bucket = (
                "<60" if score < 60 else "60-70" if score < 70 else "70-80" if score < 80 else "80+"
            )
            score_buckets[bucket].append(float(row["pnl_r"] or 0.0))
        module_rows = self.journal.query(
            "SELECT m.module, t.pnl_r, t.opened_at FROM module_scores m "
            "JOIN trades t ON t.cycle_pk = m.cycle_pk "
            "WHERE t.closed_at IS NOT NULL AND t.opened_at >= ? AND t.opened_at < ? "
            "AND m.weight > 0 AND m.score != 0",
            (iso(start), iso(end)),
        )
        grouped: dict[str, list[tuple[float, datetime]]] = {}
        for row in module_rows:
            grouped.setdefault(str(row["module"]), []).append(
                (float(row["pnl_r"] or 0.0), datetime.fromisoformat(row["opened_at"]))
            )
        evidence = tuple(self._module(name, values) for name, values in sorted(grouped.items()))
        maes = [float(row["mae_r"]) for row in trades if row["mae_r"] is not None]
        mfes = [float(row["mfe_r"]) for row in trades if row["mfe_r"] is not None]
        return Postmortem(
            trades=len(trades),
            more_trades_needed=max(0, 100 - len(trades)),
            score_buckets={
                name: (
                    len(values),
                    sum(value > 0 for value in values) / len(values) if values else 0.0,
                )
                for name, values in score_buckets.items()
            },
            average_mae_r=float(np.mean(maes)) if maes else None,
            average_mfe_r=float(np.mean(mfes)) if mfes else None,
            modules=evidence,
        )

    def _module(self, module: str, values: list[tuple[float, datetime]]) -> ModuleEvidence:
        returns = [value for value, _ in values]
        wins = sum(value > 0 for value in returns)
        alpha, beta = 1 + wins, 1 + len(returns) - wins
        rng = np.random.default_rng(770101 + len(returns))
        samples = rng.beta(alpha, beta, size=50_000)
        interval = (float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95)))
        current = float(self.weights.get(module, 0.0))
        span = max(when for _, when in values) - min(when for _, when in values)
        old_enough = bool(values) and span.days >= 30
        if len(returns) < 100 or not old_enough:
            proposed = None
            reason = f"anti-recency gate: {len(returns)}/100 trades and 30 calendar days required"
        else:
            expectancy = float(np.mean(returns))
            shift = 0.15 if expectancy > 0 else -0.15
            proposed = max(0.05, current * (1.0 + shift))
            reason = f"quarterly proposal capped at {shift:+.0%}; owner approval required"
        return ModuleEvidence(
            module,
            len(returns),
            wins,
            alpha / (alpha + beta),
            interval,
            float(np.mean(returns)) if returns else 0.0,
            current,
            proposed,
            reason,
        )
