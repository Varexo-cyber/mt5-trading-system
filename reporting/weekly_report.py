"""Weekly performance and learning report with anti-recency guardrails."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from config.schema import Settings
from core.types import AccountSnapshot
from journal.database import Journal
from learning.postmortem import PostmortemAnalyzer


class WeeklyReportGenerator:
    def __init__(self, journal: Journal, directory: Path, settings: Settings) -> None:
        self.journal = journal
        self.directory = directory
        self.settings = settings
        self._last_generated: datetime | None = None

    def maybe_generate(self, account: AccountSnapshot, now: datetime) -> tuple[Path, Path] | None:
        if self._last_generated is not None and now - self._last_generated < timedelta(days=1):
            return None
        self._last_generated = now
        return self.generate(account, now)

    def generate(self, account: AccountSnapshot, now: datetime) -> tuple[Path, Path]:
        start = now - timedelta(days=7)
        postmortem = PostmortemAnalyzer(
            self.journal, self.settings.analysis.confluence.weights
        ).analyze(start, now)
        lines = [
            f"# Jarvis weekly report — through {now.date().isoformat()}",
            "",
            f"Equity: {account.equity:.2f} {account.currency}",
            f"Closed sample: {postmortem.trades}/100 minimum",
            f"More trades needed before inference: {postmortem.more_trades_needed}",
            f"Average MAE: {postmortem.average_mae_r}",
            f"Average MFE: {postmortem.average_mfe_r}",
            "",
            "## Score buckets",
            "",
        ]
        lines.extend(
            f"- {bucket}: n={count}, win rate={rate:.1%}"
            for bucket, (count, rate) in postmortem.score_buckets.items()
        )
        lines.extend(["", "## Module proposals (never auto-applied)", ""])
        for item in postmortem.modules:
            proposal = "none" if item.proposed_weight is None else f"{item.proposed_weight:.3f}"
            lines.append(
                f"- {item.module}: n={item.trades}, posterior win={item.posterior_mean:.1%} "
                f"(90% {item.interval_90[0]:.1%}-{item.interval_90[1]:.1%}), "
                f"expectancy={item.expectancy_r:+.3f}R, proposed={proposal}; "
                f"{item.proposal_reason}"
            )
        if not postmortem.modules:
            lines.append("- No closed module-attributed trades.")
        lines.extend(["", "## Rejected-plan counterfactuals", ""])
        lines.append(
            "These are passive original-SL/TP paths, not executed trades. Under 100 "
            "observations per reason is descriptive only."
        )
        for item in postmortem.counterfactuals:
            status = "minimum sample reached" if item.observations >= 100 else "inconclusive"
            lines.append(
                f"- {item.blocked_by}: n={item.observations}, win rate={item.win_rate:.1%}, "
                f"expectancy={item.expectancy_r:+.3f}R; {status}"
            )
        if not postmortem.counterfactuals:
            lines.append("- No resolved rejected-plan observations in this window.")
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.directory / f"weekly-{now.date().isoformat()}"
        markdown, pdf = stem.with_suffix(".md"), stem.with_suffix(".pdf")
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with PdfPages(pdf) as pages:
            for offset in range(0, len(lines), 42):
                figure = Figure(figsize=(8.27, 11.69))
                figure.text(
                    0.06,
                    0.96,
                    "\n".join(
                        line.replace("# ", "").replace("## ", "")
                        for line in lines[offset : offset + 42]
                    ),
                    va="top",
                    family="monospace",
                    fontsize=9,
                )
                pages.savefig(figure, bbox_inches="tight")
        return markdown, pdf
