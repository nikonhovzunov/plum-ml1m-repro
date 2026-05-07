from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for invalid experiment configuration files."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return data


def require_keys(config: dict[str, Any], keys: list[str], prefix: str = "") -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        label = f"{prefix}: " if prefix else ""
        raise ConfigError(f"{label}missing required keys: {missing}")


def validate_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    require_keys(config, ["name", "stage", "paths"])
    if not isinstance(config["paths"], dict):
        raise ConfigError("paths must be a mapping")
    return config


def validate_config_dir(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    config_dir = Path(config_dir)
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        results[path.name] = validate_config(path)
    if not results:
        raise ConfigError(f"No YAML configs found in {config_dir}")
    return results
