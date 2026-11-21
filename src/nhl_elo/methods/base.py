"""Shared method interface and the standard ELO building blocks.

Unlike the KHL model, everything is from the HOME team's point of view, because home ice matters:

    E_home = 1 / (1 + 10^((R_away − (R_home + home_ice)) / scale))     (no home_ice at neutral sites)

E_home doubles as the home team's win probability. NHL games can't end in a tie (2005-06 on), so
there's no Win/Tie/Loss split as in the KHL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Game:
    away_score: int
    home_score: int
    outcome: str = "REG"     # REG / OT / SO / T
    playoff: bool = False
    neutral: bool = False


@dataclass
class GameUpdate:
    home_expected: float     # E for the home team (0..1), incl. home ice
    home_score: float        # S for the home team (0..1)
    home_delta: float
    away_delta: float


class Method:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scale = float(cfg.get("scale", 400))
        self.k = float(cfg.get("k", 8))
        self.home_ice = float(cfg.get("home_ice", 0))
        self.playoff_k = float(cfg.get("playoff_k", 1.0))
        res = cfg.get("result", {})
        self.ot_win = float(res.get("ot_win", 1.0))
        self.so_win = float(res.get("so_win", 1.0))
        season = cfg.get("season", {})
        self.mean = float(season.get("mean", 1500))
        self.offseason_regression = float(season.get("offseason_regression", 0.0))
        self.new_team_offset = float(season.get("new_team_offset", 0))

    @property
    def description(self) -> str:
        return self.cfg.get("description", self.name)

    def params_hash(self) -> str:
        blob = json.dumps(self.cfg, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # --- standard pieces ---
    def expected(self, ra: float, rb: float) -> float:
        """Expected score for A vs B on the logistic curve (base 10, `scale`)."""
        return 1.0 / (1.0 + 10 ** ((rb - ra) / self.scale))

    def home_expected(self, r_away: float, r_home: float, neutral: bool = False) -> float:
        return self.expected(r_home + (0.0 if neutral else self.home_ice), r_away)

    def home_result(self, g: Game) -> float:
        """S for the home team: 1 / ot_win / so_win for a win, mirrored for a loss, ½ for a tie."""
        if g.home_score == g.away_score:
            return 0.5
        s = {"OT": self.ot_win, "SO": self.so_win}.get(g.outcome, 1.0)
        return s if g.home_score > g.away_score else 1.0 - s

    @staticmethod
    def goal_diff(g: Game) -> int:
        """Absolute margin. A shootout adds one goal for the winner; it's still a one-goal game."""
        return 1 if g.outcome == "SO" else abs(g.home_score - g.away_score)

    def gd_multiplier(self, g: Game, home_edge: float) -> float:
        """home_edge = R_home + home_ice − R_away (pre-game)."""
        return 1.0

    def gd_multiplier_vec(self, margin: np.ndarray, winner_edge: np.ndarray) -> np.ndarray:
        """Vectorized G for simulations: margin = goals (≥ 1), winner_edge = winner's pre-game edge incl.
        home ice. Default: call the scalar version per element (subclasses override for speed)."""
        out = np.empty(len(margin))
        for i, (m, e) in enumerate(zip(margin, winner_edge)):
            out[i] = self.gd_multiplier(Game(0, int(m)), float(e))
        return out

    def k_factor(self, g: Game) -> float:
        return self.k * (self.playoff_k if g.playoff else 1.0)

    def update(self, r_away: float, r_home: float, g: Game, ctx: dict | None = None) -> GameUpdate:
        e = self.home_expected(r_away, r_home, g.neutral)
        s = self.home_result(g)
        edge = r_home + (0.0 if g.neutral else self.home_ice) - r_away
        delta = self.k_factor(g) * self.gd_multiplier(g, edge) * (s - e)
        return GameUpdate(e, s, delta, -delta)

    def season_start(self, rating: float) -> float:
        return rating + self.offseason_regression * (self.mean - rating)

    def new_team_rating(self) -> float:
        return self.mean + self.new_team_offset
