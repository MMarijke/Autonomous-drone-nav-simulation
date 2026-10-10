"""YAML-backed configuration with attribute access and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict whose nested dicts are reachable as attributes (``cfg.world.size``)."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        out: dict = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else copy.deepcopy(value)
        return out

    def clone(self) -> "Config":
        return Config(self.to_dict())


def deep_update(base: dict, update: Mapping[str, Any]) -> dict:
    """Recursively merge ``update`` into ``base`` (in place) and return it."""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def apply_overrides(cfg: Config, overrides: Iterable[str] | None) -> Config:
    """Apply ``a.b.c=value`` strings (values parsed as YAML) to ``cfg``."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], Mapping):
                raise KeyError(f"unknown config section {'.'.join(keys[:-1])!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"unknown config key {dotted!r}")
        value = yaml.safe_load(raw)
        if isinstance(value, str):                 # YAML 1.1 reads "1e-3" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        node[keys[-1]] = value
    return cfg


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Config:
    """Load the default config, optionally merge a user file, then CLI overrides."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "r", encoding="utf-8") as fh:
            deep_update(data, yaml.safe_load(fh) or {})
    cfg = Config(data)
    return apply_overrides(cfg, overrides)
