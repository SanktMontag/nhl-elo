"""Rating methods. Each is selected by name and configured by config/methods/<name>.toml."""

from __future__ import annotations

from ..config import deep_merge, load_method_config
from .base import Game, GameUpdate, Method
from .v1 import V1
from .v2_dev import V2Dev

REGISTRY: dict[str, type[Method]] = {
    "v1": V1,
    "v2-dev": V2Dev,
}


def get_method(name: str, overrides: dict | None = None) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"Unknown method {name!r}; choose from {sorted(REGISTRY)}")
    cfg = load_method_config(name)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return REGISTRY[name](cfg)


__all__ = ["Game", "GameUpdate", "Method", "REGISTRY", "get_method"]
