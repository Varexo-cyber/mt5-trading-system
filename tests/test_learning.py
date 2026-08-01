from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH
from learning.config_control import ConfigControl


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
