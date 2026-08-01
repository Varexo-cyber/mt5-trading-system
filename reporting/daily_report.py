"""Automatic audit report generated from journal facts, not model prose."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from core.types import AccountSnapshot
from journal.database import Journal, iso


class DailyReportGenerator:
    def __init__(self, journal: Journal, directory: Path | str, interval_minutes: int = 15) -> None:
        self.journal = journal
        self.directory = Path(directory)
        self.interval = timedelta(minutes=interval_minutes)
        self._last_generated: datetime | None = None

    def maybe_generate(self, account: AccountSnapshot, now: datetime) -> tuple[Path, Path] | None:
        if self._last_generated is not None and now - self._last_generated < self.interval:
            return None
        self._last_generated = now
        return self.generate(account, now)

    def generate(self, account: AccountSnapshot, now: datetime) -> tuple[Path, Path]:
        start = self.journal.day_start(now)
        end = start + timedelta(days=1)
        cycles = self.journal.query(
            "SELECT decision, reason, COUNT(*) AS n FROM analysis_cycles "
            "WHERE ts >= ? AND ts < ? GROUP BY decision, reason ORDER BY n DESC",
            (iso(start), iso(end)),
        )
        trades = self.journal.query(
            "SELECT symbol, direction, volume, entry_price, exit_price, pnl_money, pnl_r, "
            "exit_reason, opened_at, closed_at FROM trades WHERE opened_at >= ? AND opened_at < ? "
            "ORDER BY opened_at",
            (iso(start), iso(end)),
        )
        realised = sum(float(row["pnl_money"] or 0.0) for row in trades if row["closed_at"])
        execution = self.journal.query(
            "SELECT ok, retcode_name, COUNT(*) AS n FROM order_attempts "
            "WHERE ts >= ? AND ts < ? GROUP BY ok, retcode_name ORDER BY n DESC",
            (iso(start), iso(end)),
        )
        reconciliation = self.journal.query(
            "SELECT action, COUNT(*) AS n FROM management_actions "
            "WHERE ts >= ? AND ts < ? AND action LIKE 'BROKER_%' GROUP BY action",
            (iso(start), iso(end)),
        )
        unresolved = sum(
            int(row["n"])
            for row in reconciliation
            if row["action"] == "BROKER_CLOSED_PENDING_HISTORY"
        )
        lines = [
            f"# Jarvis daily report — {start.date().isoformat()}",
            "",
            f"Generated: {now.isoformat()}",
            f"Balance: {account.balance:.2f} {account.currency}",
            f"Equity: {account.equity:.2f} {account.currency}",
            f"Realised today: {realised:+.2f} {account.currency}",
            "",
            "## Decisions",
            "",
        ]
        lines.extend(f"- {row['decision']} / {row['reason']}: {row['n']}" for row in cycles)
        if not cycles:
            lines.append("- No completed analysis cycles.")
        lines.extend(["", "## Trades", ""])
        for row in trades:
            state = (
                f"closed {float(row['pnl_money'] or 0):+.2f}, "
                f"{float(row['pnl_r'] or 0):+.2f}R ({row['exit_reason']})"
                if row["closed_at"]
                else "open"
            )
            lines.append(
                f"- {row['opened_at']} {row['symbol']} {row['direction']} "
                f"{row['volume']:g} lots @ {row['entry_price']:g}: {state}"
            )
        if not trades:
            lines.append("- No trades opened.")
        lines.extend(["", "## Execution integrity", ""])
        lines.append(f"- Unexplained broker/journal differences: {unresolved}")
        lines.extend(
            f"- Orders {'accepted' if row['ok'] else 'rejected'} / "
            f"{row['retcode_name']}: {row['n']}"
            for row in execution
        )
        lines.extend(f"- Reconciliation {row['action']}: {row['n']}" for row in reconciliation)
        if not execution:
            lines.append("- No order attempts.")

        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.directory / f"jarvis-{start.date().isoformat()}"
        markdown = stem.with_suffix(".md")
        pdf = stem.with_suffix(".pdf")
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with PdfPages(pdf) as pages:
            page_lines = lines
            for offset in range(0, len(page_lines), 42):
                figure = Figure(figsize=(8.27, 11.69))
                content = "\n".join(
                    line.replace("# ", "").replace("## ", "")
                    for line in page_lines[offset : offset + 42]
                )
                figure.text(
                    0.06,
                    0.96,
                    content,
                    va="top",
                    family="monospace",
                    fontsize=9,
                )
                pages.savefig(figure, bbox_inches="tight")
        return markdown, pdf
