"""Project paths and TOML config loading."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("NHL_ELO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "output"

GAMES_CSV = DATA_DIR / "games.csv"
TEAMS_CSV = DATA_DIR / "teams.csv"
DIVISIONS_CSV = DATA_DIR / "divisions.csv"
SEASONS_CSV = DATA_DIR / "seasons.csv"


@dataclass(frozen=True)
class Settings:
    first_season: int
    score_from: int
    game_types: tuple[int, ...]
    default_method: str
    user_agent: str
    delay: float
    current_ttl_hours: float
    continuations: tuple[dict, ...]
    neutral: tuple[dict, ...]
    simulation: dict = field(default_factory=dict)
    playoffs: dict = field(default_factory=dict)
    site: dict = field(default_factory=dict)
    trade_deadlines: dict = field(default_factory=dict)   # season -> "YYYY-MM-DD"

    def continuation_map(self) -> dict[int, int]:
        """franchise id -> franchise id its rating lives under (followed transitively)."""
        step = {int(c["from"]): int(c["to"]) for c in self.continuations}
        out = {}
        for f in step:
            seen, cur = {f}, step[f]
            while cur in step and cur not in seen:
                seen.add(cur)
                cur = step[cur]
            out[f] = cur
        return out

    def is_neutral(self, season: int, game_type: int) -> bool:
        for rule in self.neutral:
            if int(rule["season"]) == season and int(rule.get("game_type", game_type)) == game_type:
                return True
        return False


def load_settings() -> Settings:
    with open(CONFIG_DIR / "settings.toml", "rb") as fh:
        raw = tomllib.load(fh)
    api = raw.get("api", {})
    return Settings(
        first_season=int(raw["first_season"]),
        score_from=int(raw.get("score_from", raw["first_season"])),
        game_types=tuple(int(x) for x in raw.get("game_types", [2, 3])),
        default_method=raw.get("default_method", "v1"),
        user_agent=api.get("user_agent", "nhl-elo-tracker"),
        delay=float(api.get("delay", 0.5)),
        current_ttl_hours=float(api.get("current_ttl_hours", 1)),
        continuations=tuple(raw.get("continuation", [])),
        neutral=tuple(raw.get("neutral", [])),
        simulation={"sims": 10000, "seed": 2026, "hot": True, "rates_from": 20152016, "backtest_sims": 2000,
                    "checkpoint_sims": 5000, **raw.get("simulation", {})},
        playoffs={"format": "wildcard", "division_spots": 3, "wildcards": 2, **raw.get("playoffs", {})},
        site={"favorite_team": "", "favorite_division": "", **raw.get("site", {})},
        trade_deadlines={int(k): str(v) for k, v in raw.get("trade_deadline", {}).items()},
    )


def load_method_config(name: str) -> dict:
    path = CONFIG_DIR / "methods" / f"{name}.toml"
    if not path.exists():
        raise KeyError(f"No method config {path}")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Copy of `base` with `overrides` applied. Keys may be dotted ("gd.cap") or nested dicts."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        parts = key.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
            node[parts[-1]] = deep_merge(node[parts[-1]], value)
        else:
            node[parts[-1]] = value
    return out


def season_label(season: int) -> str:
    """20242025 -> '2024-25'."""
    s = str(season)
    return f"{s[:4]}-{s[6:]}"
