"""Replay the whole chain with candidate parameters and score the pre-game predictions.

Every prediction is made before the game with ratings built only from earlier games, so the whole
chain is an honest out-of-sample test. Seasons before settings.score_from are burn-in and not scored.

The target is "did the home team win" (OT and shootout wins count as wins). This stays the same
whatever ot_win / so_win are set to, so all settings are compared on the same footing.

Metrics: logloss and brier (lower is better), pick_pct (higher is better), and
home_win_pred vs home_win_actual (should match).
"""

from __future__ import annotations

import itertools

import pandas as pd

from .config import deep_merge, load_method_config, load_settings
from .engine import calibration, replay, score
from .fetch import load_games
from .methods import REGISTRY


def _games() -> pd.DataFrame:
    s = load_settings()
    g = load_games()
    return g.loc[g["season"] >= s.first_season]


def evaluate(method_name: str, overrides: dict | None = None, games: pd.DataFrame | None = None,
             seasons: tuple[int, int] | None = None) -> tuple[dict, pd.DataFrame]:
    """Score one parameter set. seasons=(from, to) limits which seasons are scored (default: score_from on)."""
    settings = load_settings()
    games = _games() if games is None else games
    cfg = deep_merge(load_method_config(method_name), overrides or {})
    rr = replay(games, REGISTRY[method_name](cfg), light=True)
    lo, hi = seasons or (settings.score_from, 99999999)
    sc = rr.scored.loc[rr.scored["season"].between(lo, hi)]
    return score(sc), sc


def parse_grid(specs: list[str]) -> dict[str, list[float]]:
    """['k=6,8,10', 'result.ot_win=0.6,1'] -> {'k': [6, 8, 10], 'result.ot_win': [0.6, 1.0]}"""
    grid = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"grid spec {spec!r} must look like name=v1,v2,...")
        key, vals = spec.split("=", 1)
        grid[key.strip()] = [float(v) for v in vals.split(",") if v.strip()]
    return grid


def grid_search(method_name: str, grid: dict[str, list[float]], seasons: tuple[int, int] | None = None,
                progress=None) -> pd.DataFrame:
    games = _games()
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    rows = []
    for i, combo in enumerate(combos, 1):
        ov = dict(zip(keys, combo))
        res, _ = evaluate(method_name, ov, games, seasons)
        rows.append({**ov, **res})
        if progress:
            progress(i, len(combos))
    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def per_season(scored: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"season": s, **score(g)} for s, g in scored.groupby("season")])


__all__ = ["evaluate", "grid_search", "parse_grid", "per_season", "calibration"]
