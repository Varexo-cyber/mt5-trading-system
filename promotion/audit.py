"""Evidence-based gate between research/demo operation and live money."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backtesting.engine import deflated_sharpe_probability
from config.schema import Settings
from learning.config_control import file_hash


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    name: str
    passed: bool
    detail: str


class PromotionAudit:
    """Derive live eligibility from stored evidence; no manual checkboxes."""

    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = root
        self.settings = settings

    def run(self) -> tuple[PromotionCheck, ...]:
        enabled = tuple(self.settings.analysis.confluence.live_enabled_modules)
        reports = self._reports()
        sessions = self._sessions()
        checks = [
            PromotionCheck(
                "live_module_allowlist",
                bool(enabled),
                ", ".join(enabled) if enabled else "no independently validated modules enabled",
            )
        ]
        for module in enabled:
            returns: list[float] = []
            instruments: set[str] = set()
            configurations = 1
            stability = False
            holdback = False
            for report in reports:
                metadata = report.get("metadata", {})
                validated = set(metadata.get("validated_modules", []))
                if module not in validated:
                    continue
                configurations = max(configurations, int(metadata.get("configurations_tested", 1)))
                stability = stability or bool(metadata.get("parameter_stability_passed", False))
                holdback = holdback or bool(metadata.get("independent_holdback_passed", False))
                for segment in report.get("segments", []):
                    if not str(segment.get("name", "")).endswith("/validation"):
                        continue
                    values = segment.get("returns_by_module", {}).get(module, [])
                    if values:
                        instruments.add(str(segment["name"]).split("/", 1)[0])
                        returns.extend(float(value) for value in values)
            checks.extend(
                [
                    PromotionCheck(
                        f"{module}:oos_sample",
                        len(returns) >= 100,
                        f"{len(returns)}/100 validation trades",
                    ),
                    PromotionCheck(
                        f"{module}:positive_expectancy",
                        bool(returns) and sum(returns) / len(returns) > 0,
                        (
                            f"{sum(returns) / len(returns):.3f}R"
                            if returns
                            else "no validation returns"
                        ),
                    ),
                    PromotionCheck(
                        f"{module}:deflated_sharpe",
                        deflated_sharpe_probability(returns, configurations) >= 0.95,
                        f"probability={deflated_sharpe_probability(returns, configurations):.1%}, "
                        f"configurations={configurations}",
                    ),
                    PromotionCheck(
                        f"{module}:parameter_stability",
                        stability,
                        "broad registered plateau" if stability else "no passing plateau evidence",
                    ),
                    PromotionCheck(
                        f"{module}:cross_instrument",
                        holdback and len(instruments) >= 2,
                        f"validation instruments={sorted(instruments)}; holdback={holdback}",
                    ),
                ]
            )

        paper_seconds = self._duration(sessions, "paper")
        paper_failures = self._sum(sessions, "paper", "unexplained_reconciliations")
        demo_trades = self._sum(sessions, "demo", "trades_opened")
        demo_failures = self._sum(sessions, "demo", "unexplained_reconciliations")
        checks.extend(
            [
                PromotionCheck(
                    "paper_30_days",
                    paper_seconds >= 30 * 86_400,
                    f"{paper_seconds / 86_400:.2f}/30 days",
                ),
                PromotionCheck(
                    "paper_reconciliation",
                    paper_failures == 0,
                    f"{paper_failures} unexplained differences",
                ),
                PromotionCheck(
                    "demo_execution",
                    demo_trades >= 1,
                    f"{demo_trades} demo trades",
                ),
                PromotionCheck(
                    "demo_reconciliation",
                    demo_failures == 0,
                    f"{demo_failures} unexplained differences",
                ),
                self._config_drift_check(),
            ]
        )
        return tuple(checks)

    def assert_passed(self) -> None:
        failures = [check for check in self.run() if not check.passed]
        if failures:
            detail = "; ".join(f"{item.name}: {item.detail}" for item in failures)
            raise RuntimeError(f"LIVE_PROMOTION_BLOCKED: {detail}")

    def _reports(self) -> list[dict]:  # type: ignore[type-arg]
        directory = self.root / "runtime" / "validation"
        reports = []
        for path in directory.glob("strategy-*.json") if directory.exists() else ():
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return reports

    def _sessions(self) -> list[dict]:  # type: ignore[type-arg]
        path = self.root / "runtime" / "operation_history.json"
        if not path.exists():
            return []
        try:
            return list(json.loads(path.read_text(encoding="utf-8")).get("sessions", []))
        except (OSError, json.JSONDecodeError):
            return []

    def _config_drift_check(self) -> PromotionCheck:
        config_path = self.root / "config" / "config.yaml"
        baseline_path = self.root / "config" / "baseline.sha256"
        if not config_path.exists() or not baseline_path.exists():
            return PromotionCheck("config_drift", False, "baseline hash missing")
        current = file_hash(config_path)
        baseline = baseline_path.read_text(encoding="utf-8").strip()
        if current == baseline:
            return PromotionCheck("config_drift", True, "original frozen configuration")
        state_path = self.root / "runtime" / "shadow" / "state.json"
        if not state_path.exists():
            return PromotionCheck("config_drift", False, "changed config has no shadow evidence")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(state["started_at"])
            last_seen = datetime.fromisoformat(state["last_seen_at"])
            passed = (
                state.get("candidate_hash") == current
                and (last_seen - started).days >= 30
                and int(state.get("decisions", 0)) >= 30
            )
            return PromotionCheck(
                "config_drift",
                passed,
                f"shadow days={(last_seen - started).days}, decisions={state.get('decisions', 0)}",
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return PromotionCheck("config_drift", False, "invalid shadow evidence")

    @staticmethod
    def _duration(sessions: list[dict], operation: str) -> float:  # type: ignore[type-arg]
        seconds = 0.0
        for row in sessions:
            if row.get("operation") != operation:
                continue
            try:
                start = datetime.fromisoformat(row["started_at"])
                end = datetime.fromisoformat(row.get("ended_at") or row["last_seen_at"])
                seconds += max(0.0, (end - start).total_seconds())
            except (KeyError, TypeError, ValueError):
                continue
        return seconds

    @staticmethod
    def _sum(sessions: list[dict], operation: str, field: str) -> int:  # type: ignore[type-arg]
        return sum(int(row.get(field, 0)) for row in sessions if row.get("operation") == operation)
