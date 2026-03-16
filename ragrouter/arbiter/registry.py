"""Engine capability registry: loads engine configs from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ragrouter.schemas import EngineConfig, EngineType


_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


class EngineRegistry:
    """Manages the set of available engines and their capabilities."""

    def __init__(self) -> None:
        self._engines: dict[str, EngineConfig] = {}

    @classmethod
    def from_yaml(cls, path: Path | str = _DEFAULT_CONFIG) -> EngineRegistry:
        path = Path(path)
        with path.open() as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        registry = cls()
        for key, cfg in raw.get("engines", {}).items():
            cfg.setdefault("name", key)
            cfg["type"] = EngineType(cfg["type"])
            registry._engines[key] = EngineConfig(**cfg)
        return registry

    @classmethod
    def from_dict(cls, engines: dict[str, dict[str, Any]]) -> EngineRegistry:
        registry = cls()
        for key, cfg in engines.items():
            cfg.setdefault("name", key)
            cfg["type"] = EngineType(cfg["type"])
            registry._engines[key] = EngineConfig(**cfg)
        return registry

    def get(self, name: str) -> EngineConfig | None:
        return self._engines.get(name)

    def enabled_engines(self) -> list[EngineConfig]:
        return [e for e in self._engines.values() if e.enabled]

    def engines_with_strength(self, strength: str) -> list[EngineConfig]:
        return [e for e in self.enabled_engines() if strength in e.strengths]

    def engines_by_cost(self) -> list[EngineConfig]:
        return sorted(self.enabled_engines(), key=lambda e: e.cost_rank)

    def all_engines(self) -> list[EngineConfig]:
        return list(self._engines.values())
