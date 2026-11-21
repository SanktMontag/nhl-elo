"""v2-dev: sandbox for method changes. Writes to output/v2-dev/, never touches v1.

Currently: v1 with a log goal-difference multiplier and an autocorrelation correction,

    G = ln(gd + 1) × ac / (0.001 × winner_edge + ac)

where winner_edge is the winner's pre-game rating advantage including home ice. Favourites win
more blowouts, so an uncorrected margin multiplier slowly inflates strong teams' ratings.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Game, Method


class V2Dev(Method):
    name = "v2-dev"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.ac = float(cfg.get("gd", {}).get("autocorrelation", 2.2))

    def gd_multiplier(self, g: Game, home_edge: float) -> float:
        if g.home_score == g.away_score:
            return 1.0
        winner_edge = home_edge if g.home_score > g.away_score else -home_edge
        return math.log(self.goal_diff(g) + 1) * self.ac / (0.001 * winner_edge + self.ac)

    def gd_multiplier_vec(self, margin, winner_edge):
        return np.log(margin + 1) * self.ac / (0.001 * winner_edge + self.ac)
