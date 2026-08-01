"""Generate the micro-live execution acceptance report from journal facts."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

from journal.database import Journal, iso


class ExecutionReportGenerator:
    def __init__(self, journal: Journal, path: Path, interval_minutes: int = 60) -> None:
        self.journal = journal
        self.path = path
        self.interval = timedelta(minutes=interval_minutes)
        self._last_generated: datetime | None = None

    def maybe_generate(self, now: datetime) -> Path | None:
        if self._last_generated is not None and now - self._last_generated < self.interval:
            return None
        self._last_generated = now
        return self.generate(now)

    def generate(self, now: datetime) -> Path:
        rows = self.journal.query("SELECT * FROM order_attempts ORDER BY ts")
        accepted = [row for row in rows if row["ok"]]
        rejected = [row for row in rows if not row["ok"]]
        slippage = [float(row["slippage_pips"]) for row in accepted]
        mismatches = [
            row
            for row in accepted
            if abs(float(row["requested_volume"]) - float(row["filled_volume"])) > 1e-9
        ]
        reconciliation = self.journal.query(
            "SELECT action, COUNT(*) AS n FROM management_actions "
            "WHERE action LIKE 'BROKER_%' GROUP BY action"
        )
        lines = [
            "# Execution report",
            "",
            f"Generated: {iso(now)}",
            f"Attempts / accepted / rejected: {len(rows)} / {len(accepted)} / {len(rejected)}",
            f"Volume mismatches: {len(mismatches)}",
        ]
        if slippage:
            lines.extend(
                [
                    f"Slippage mean: {sum(slippage) / len(slippage):.3f} pips",
                    f"Slippage median: {median(slippage):.3f} pips",
                    f"Slippage worst: {max(slippage):.3f} pips",
                ]
            )
        else:
            lines.append("Slippage: no accepted executions yet.")
        lines.extend(["", "## Return codes", ""])
        codes = self.journal.query(
            "SELECT retcode_name, ok, COUNT(*) AS n FROM order_attempts "
            "GROUP BY retcode_name, ok ORDER BY n DESC"
        )
        lines.extend(
            f"- {row['retcode_name']} ({'accepted' if row['ok'] else 'rejected'}): {row['n']}"
            for row in codes
        )
        if not codes:
            lines.append("- No order attempts.")
        lines.extend(["", "## Reconciliation", ""])
        lines.extend(f"- {row['action']}: {row['n']}" for row in reconciliation)
        if not reconciliation:
            lines.append("- No broker/journal differences recorded.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.path
