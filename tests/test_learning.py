from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from analysis.confluence import TradeIdea
from config.loader import DEFAULT_CONFIG_PATH
from core.types import Direction
from learning.config_control import ConfigControl, ShadowRecorder, file_hash


def candidate(root: Path, weight: float, name: str) -> Path:
    payload = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["analysis"]["confluence"]["weights"]["market_structure"] = weight
    path = root / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_shadow_config_enforces_weight_only_and_quarterly_cap(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    control = ConfigControl(tmp_path)

    target = control.start_shadow(candidate(tmp_path, 1.10, "accepted.yaml"))

    assert target.exists()
    with pytest.raises(RuntimeError, match="15% cap"):
        control.start_shadow(candidate(tmp_path, 1.20, "rejected.yaml"))


def _mature_state(root: Path, *, mean_lift: float) -> None:
    state_path = root / "runtime" / "shadow" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    state.update(
        {
            "started_at": (now - timedelta(days=31)).isoformat(),
            "last_seen_at": now.isoformat(),
            "resolved_differences": 100,
            "lift_sum_r": mean_lift * 100,
            "lift_sum_squares_r": mean_lift * mean_lift * 100,
            "resolved_days": [
                (now - timedelta(days=offset)).date().isoformat() for offset in range(20)
            ],
            "resolved_symbols": ["EURUSD", "GBPUSD", "USDJPY"],
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")


def test_shadow_promotion_requires_measured_positive_lift(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    control = ConfigControl(tmp_path)
    control.start_shadow(candidate(tmp_path, 1.10, "candidate.yaml"))
    _mature_state(tmp_path, mean_lift=-0.1)

    with pytest.raises(RuntimeError, match="not proven positive lift"):
        control.promote_shadow()


def test_a_statistically_clear_shadow_can_be_promoted(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    control = ConfigControl(tmp_path)
    target = control.start_shadow(candidate(tmp_path, 1.10, "candidate.yaml"))
    _mature_state(tmp_path, mean_lift=0.2)

    control.promote_shadow()

    assert file_hash(config_dir / "config.yaml") == file_hash(target)


def test_shadow_decisions_are_resolved_as_paired_counterfactuals(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    ConfigControl(tmp_path).start_shadow(candidate(tmp_path, 1.10, "candidate.yaml"))
    recorder = ShadowRecorder(tmp_path)
    opened = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    no_trade = TradeIdea("TEST", False, None, 0, 0, 0, 0, 0, "none", ())
    challenger = TradeIdea(
        "TEST",
        True,
        Direction.LONG,
        70,
        0.8,
        100.0,
        99.0,
        102.0,
        "test thesis",
        (),
    )
    recorder.record("TEST", no_trade, challenger, opened)

    class Broker:
        def copy_rates_range(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            after = int((opened + timedelta(minutes=15)).timestamp())
            return [{"time": after, "open": 100.0, "high": 102.1, "low": 99.5, "close": 102.0}]

    assert recorder.resolve(Broker(), opened + timedelta(hours=1)) == 1  # type: ignore[arg-type]
    state = json.loads((tmp_path / "runtime" / "shadow" / "state.json").read_text())
    assert state["resolved_differences"] == 1
    assert state["challenger_total_r"] == pytest.approx(2.0)
    assert state["lift_sum_r"] == pytest.approx(2.0)
