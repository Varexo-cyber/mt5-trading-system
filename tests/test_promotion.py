from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


def test_promotion_fails_closed_without_evidence(tmp_path: Path) -> None:
    checks = PromotionAudit(tmp_path, enabled_settings()).run()

    assert checks
    assert not all(check.passed for check in checks)


def test_promotion_passes_only_with_research_paper_and_demo_evidence(tmp_path: Path) -> None:
    returns = ([1.0, 0.9, 1.1, 0.8, 1.2] * 12)[:60]
    report = {
        "metadata": {
            "validated_modules": ["market_structure"],
            "configurations_tested": 9,
            "parameter_stability_passed": True,
            "independent_holdback_passed": True,
        },
        "segments": [
            {
                "name": "EURUSD.i/validation",
                "returns_by_module": {"market_structure": returns},
            },
            {
                "name": "AUDUSD.i/validation",
                "returns_by_module": {"market_structure": returns},
            },
        ],
    }
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
