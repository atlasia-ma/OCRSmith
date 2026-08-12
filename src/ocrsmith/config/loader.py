"""Loading and overriding generation configuration."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .schema import GenerationConfig

__all__ = ["DEFAULT_CONFIG_PATH", "apply_overrides", "load_config"]

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> GenerationConfig:
    """Load a config file (or the bundled default) and apply dotted-key overrides."""
    source = Path(path) if path else DEFAULT_CONFIG_PATH
    if not source.exists():
        raise FileNotFoundError(f"Config file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {source} must contain a mapping at the top level")

    data = dict(data)
    if overrides:
        data = apply_overrides(data, overrides)
    return GenerationConfig.model_validate(data)


def apply_overrides(data: dict, overrides: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` strings to a nested dict.

    Values are parsed as JSON when possible, so `run.workers=8`, `page.columns={"1":1}`
    and `fonts.size_range=[20,24]` all do what they look like they do; anything that is
    not valid JSON is kept as a string.
    """
    result = json.loads(json.dumps(data))  # deep copy through plain types
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} is not of the form key.path=value")
        key, raw = override.split("=", 1)
        _set_path(result, key.strip().split("."), _parse(raw.strip()))
    return result


def _parse(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_path(target: dict, path: list[str], value: Any) -> None:
    cursor: Any = target
    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[path[-1]] = value
