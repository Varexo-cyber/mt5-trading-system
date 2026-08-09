from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backtesting.replay import evidence_digest, implementation_digest
from config.loader import DEFAULT_CONFIG_PATH, load_settings
from learning.config_control import file_hash
from promotion.audit import PromotionAudit


def enabled_settings():  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    confluence = settings.analysis.confluence.model_copy(
        update={"live_enabled_modules": ("market_structure",)}
    )
    analysis = settings.analysis.model_copy(update={"confluence": confluence})
    return settings.model_copy(update={"analysis": analysis})


def write_hypothesis(root: Path, module: str = "market_structure") -> str:
    target = root / "docs" / "hypotheses" / f"{module}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# Pre-registered {module}\n", encoding="utf-8")
    return file_hash(target)


def evidence_provenance(root: Path) -> dict[str, str]:
    settings = enabled_settings()
    return {
        "effective_config_digest": hashlib.sha256(settings.model_dump_json().encode()).hexdigest(),
        "implementation_digest": implementation_digest(root),
    }


def test_promotion_fails_closed_without_evidence(tmp_path: Path) -> None:
    checks = PromotionAudit(tmp_path, enabled_settings()).run()

    assert checks
    assert not all(check.passed for check in checks)


def test_liquidity_sweep_requires_measured_independence(tmp_path: Path) -> None:
    settings = load_settings(env_overrides=False)
    confluence = settings.analysis.confluence.model_copy(
        update={"live_enabled_modules": ("liquidity_sweep",)}
    )
    configured = settings.model_copy(
        update={"analysis": settings.analysis.model_copy(update={"confluence": confluence})}
    )

    checks = PromotionAudit(tmp_path, configured).run()
    independence = next(
        check for check in checks if check.name == "liquidity_sweep:independent_evidence"
    )

    assert not independence.passed
    assert "no market-structure overlap" in independence.detail


def test_promotion_passes_only_with_research_paper_and_demo_evidence(tmp_path: Path) -> None:
    returns = ([1.0, 0.9, 1.1, 0.8, 1.2] * 12)[:60]
    hypothesis_digest = write_hypothesis(tmp_path)
    report = {
        "metadata": {
            "hypothesis": "docs/hypotheses/market_structure.md",
            "hypothesis_digest": hypothesis_digest,
            "validated_modules": ["market_structure"],
            "configurations_tested": 9,
            "parameter_stability_passed": True,
            "independent_holdback_passed": True,
            **evidence_provenance(tmp_path),
            "historical_data_digests": {"EURUSD.i/H1": "bars-1"},
        },
        "segments": [
            {
                "name": "EURUSD.i/validation",
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2025-01-01T00:00:00+00:00",
                "returns_by_module": {"market_structure": returns},
            },
            {
                "name": "AUDUSD.i/validation",
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2025-01-01T00:00:00+00:00",
                "returns_by_module": {"market_structure": returns},
            },
        ],
    }
    report["evidence_digest"] = evidence_digest(report)
    validation = tmp_path / "runtime" / "validation"
    validation.mkdir(parents=True)
    (validation / "strategy-test.json").write_text(json.dumps(report), encoding="utf-8")
    now = datetime.now(UTC)
    sessions = {
        "sessions": [
            {
                "operation": "paper",
                "started_at": (now - timedelta(days=31)).isoformat(),
                "last_seen_at": now.isoformat(),
                "ended_at": now.isoformat(),
                "unexplained_reconciliations": 0,
                "trades_opened": 10,
            },
            {
                "operation": "demo",
                "started_at": (now - timedelta(hours=2)).isoformat(),
                "last_seen_at": now.isoformat(),
                "ended_at": now.isoformat(),
                "unexplained_reconciliations": 0,
                "trades_opened": 1,
            },
        ]
    }
    history = tmp_path / "runtime" / "operation_history.json"
    history.write_text(json.dumps(sessions), encoding="utf-8")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    (config_dir / "baseline.sha256").write_text(file_hash(config_path), encoding="utf-8")

    checks = PromotionAudit(tmp_path, enabled_settings()).run()

    assert all(check.passed for check in checks), checks


def test_duplicate_validation_reports_do_not_inflate_the_oos_sample(tmp_path: Path) -> None:
    validation = tmp_path / "runtime" / "validation"
    validation.mkdir(parents=True)
    hypothesis_digest = write_hypothesis(tmp_path)
    report = {
        "metadata": {
            "hypothesis": "docs/hypotheses/market_structure.md",
            "hypothesis_digest": hypothesis_digest,
            "validated_modules": ["market_structure"],
            "configurations_tested": 9,
            "parameter_stability_passed": True,
            "independent_holdback_passed": True,
            **evidence_provenance(tmp_path),
            "historical_data_digests": {"EURUSD.i/H1": "bars-1"},
        },
        "segments": [
            {
                "name": "EURUSD.i/validation",
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2025-01-01T00:00:00+00:00",
                "returns_by_module": {"market_structure": [1.0] * 60},
            }
        ],
    }
    report["evidence_digest"] = evidence_digest(report)
    encoded = json.dumps(report)
    (validation / "strategy-a.json").write_text(encoded, encoding="utf-8")
    (validation / "strategy-b.json").write_text(encoded, encoding="utf-8")

    checks = PromotionAudit(tmp_path, enabled_settings()).run()
    sample = next(check for check in checks if check.name == "market_structure:oos_sample")

    assert not sample.passed
    assert sample.detail == "60/100 validation trades"

    hypothesis = tmp_path / "docs" / "hypotheses" / "market_structure.md"
    hypothesis.write_text("# Edited after the result\n", encoding="utf-8")
    changed_checks = PromotionAudit(tmp_path, enabled_settings()).run()
    changed_sample = next(
        check for check in changed_checks if check.name == "market_structure:oos_sample"
    )
    assert changed_sample.detail == "0/100 validation trades"
