"""Evidence-based gate between research/demo operation and live money."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backtesting.engine import deflated_sharpe_probability
from backtesting.replay import evidence_digest, implementation_digest
from config.schema import Settings
from learning.config_control import _paired_lift, file_hash


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
            overlaps: list[float] = []
            seen_segments: set[tuple[str, str, str, str]] = set()
            for report in reports:
                metadata = report.get("metadata", {})
                validated = set(metadata.get("validated_modules", []))
                if module not in validated:
                    continue
                configurations = max(configurations, int(metadata.get("configurations_tested", 1)))
                stability = stability or bool(metadata.get("parameter_stability_passed", False))
                holdback = holdback or bool(metadata.get("independent_holdback_passed", False))
                if module == "liquidity_sweep":
                    raw_overlap = metadata.get("market_structure_signal_overlap_ratio")
                    if raw_overlap is not None:
                        overlaps.append(float(raw_overlap))
                for segment in report.get("segments", []):
                    if not str(segment.get("name", "")).endswith("/validation"):
                        continue
                    identity = (
                        str(metadata.get("hypothesis", "")),
                        str(segment.get("name", "")),
                        str(segment.get("start", "")),
                        str(segment.get("end", "")),
                    )
                    if not all(identity) or identity in seen_segments:
                        continue
                    seen_segments.add(identity)
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
            if module == "liquidity_sweep":
                overlap = max(overlaps) if overlaps else None
                checks.append(
                    PromotionCheck(
                        f"{module}:independent_evidence",
                        bool(overlaps) and overlap is not None and overlap <= 0.70,
                        (
                            f"same-time same-direction overlap={overlap:.1%}"
                            if overlap is not None
                            else "no market-structure overlap measurement"
                        ),
                    )
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
        current_config = hashlib.sha256(self.settings.model_dump_json().encode()).hexdigest()
        current_implementation = implementation_digest(self.root)
        for path in directory.glob("strategy-*.json") if directory.exists() else ():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                metadata = report.get("metadata", {})
                hypothesis_ref = str(metadata.get("hypothesis", ""))
                hypothesis_path = (self.root / hypothesis_ref).resolve()
                root = self.root.resolve()
                hypothesis_valid = (
                    bool(hypothesis_ref)
                    and hypothesis_path.is_relative_to(root)
                    and hypothesis_path.is_file()
                    and metadata.get("hypothesis_digest") == file_hash(hypothesis_path)
                )
                provenance = bool(
                    hypothesis_valid
                    and metadata.get("effective_config_digest") == current_config
                    and metadata.get("implementation_digest") == current_implementation
                    and metadata.get("historical_data_digests")
                )
                if provenance and report.get("evidence_digest") == evidence_digest(report):
                    reports.append(report)
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
            learning = self.settings.learning
            mean, lower = _paired_lift(state, learning.shadow_confidence_level)
            passed = (
                state.get("candidate_hash") == current
                and (last_seen - started).days >= learning.shadow_min_days
                and int(state.get("resolved_differences", 0)) >= learning.shadow_min_paired_outcomes
                and len(set(state.get("resolved_days", []))) >= learning.shadow_min_unique_days
                and len(set(state.get("resolved_symbols", []))) >= learning.shadow_min_symbols
                and mean >= learning.shadow_min_expectancy_lift_r
                and lower > 0.0
            )
            return PromotionCheck(
                "config_drift",
                passed,
                f"shadow days={(last_seen - started).days}, paired outcomes="
                f"{state.get('resolved_differences', 0)}, mean lift={mean:.3f}R, "
                f"lower bound={lower:.3f}R, unique days="
                f"{len(set(state.get('resolved_days', [])))}, symbols="
                f"{len(set(state.get('resolved_symbols', [])))}",
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
