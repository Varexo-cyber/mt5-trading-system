"""Loading and validating configuration.

Order of precedence, lowest to highest:

1. defaults declared in `config/schema.py`
2. `config/config.yaml`
3. an optional overlay file (`--config-overlay`, used by walk-forward runs so
   an optimisation never edits the file a live session reads)
4. `TS_` environment variables, for CI and one-off runs

Credentials are loaded separately from `config/.env` and are never merged into
the `Settings` tree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from config.schema import MT5Credentials, Settings
from core.errors import ConfigError

#: Repository-relative root of the trading system package.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "config.yaml"
DEFAULT_ENV_PATH = PACKAGE_ROOT / "config" / ".env"

ENV_PREFIX = "TS_"


def load_settings(
    path: Path | str | None = None,
    *,
    overlay: Path | str | None = None,
    env_overrides: bool = True,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Read, merge and validate the configuration tree.

    Raises `ConfigError` with a readable message on anything wrong — a missing
    file, malformed YAML, an unknown key, or a hard-rule violation.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = _read_yaml(config_path)

    if overlay is not None:
        data = _deep_merge(data, _read_yaml(Path(overlay)))
    if env_overrides:
        data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)

    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {config_path}:\n{_format(exc)}") from exc


def load_credentials(
    env_path: Path | str | None = None, *, required: bool = True
) -> MT5Credentials | None:
    """Read MT5 credentials from `config/.env` and the process environment.

    Returns None when `required=False` and nothing is configured, which is the
    normal case for backtests and unit tests.
    """
    dotenv_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not (login and password and server):
        if not required:
            return None
        missing = [
            name
            for name, value in (
                ("MT5_LOGIN", login),
                ("MT5_PASSWORD", password),
                ("MT5_SERVER", server),
            )
            if not value
        ]
        raise ConfigError(
            f"missing MT5 credential(s): {', '.join(missing)}. "
            f"Copy {dotenv_path.parent / '.env.example'} to {dotenv_path} and fill it in."
        )

    try:
        return MT5Credentials(login=int(login), password=password, server=server)
    except ValueError as exc:
        raise ConfigError(f"MT5_LOGIN must be numeric, got {login!r}") from exc


def terminal_path_from_env() -> str:
    return os.getenv("MT5_TERMINAL_PATH", "")


# ---------------------------------------------------------------- helpers ---


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; scalar and list values in `overlay` replace wholesale.

    Lists are replaced rather than concatenated on purpose: a whitelist you
    meant to shrink must not silently grow.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _env_overrides() -> dict[str, Any]:
    """Translate `TS_SECTION__KEY=value` into a nested dict.

    `TS_SYSTEM__MODE=paper` becomes `{"system": {"mode": "paper"}}`.
    Values are parsed as YAML scalars so numbers and booleans keep their type.
    """
    result: dict[str, Any] = {}
    for name, raw in os.environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name[len(ENV_PREFIX) :].lower().split("__")
        if not path or not path[0]:
            continue
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # pragma: no cover - malformed override
                raise ConfigError(f"env override {name} conflicts with an earlier one")
        node[path[-1]] = value
    return result


def _format(exc: ValidationError) -> str:
    """Render pydantic errors as `section.key: message` lines."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
