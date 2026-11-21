"""v1: the KHL v1 model adapted to the NHL.

    E_home = 1 / (1 + 10^((R_away − R_home − home_ice) / scale))
    G      = goal-difference multiplier (1, 1.5, 1.75, then +step per extra goal, capped)
    ΔHome  = K × playoff_k? × G × (S − E_home);  ΔAway = −ΔHome      (zero-sum)

S is 1 for a regulation win, `ot_win` / `so_win` for an OT / shootout win (loser gets 1 − S).
"""

from __future__ import annotations

import numpy as np

from .base import Game, Method


class V1(Method):
    name = "v1"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        gd = cfg.get("gd", {})
        self.gd_one = float(gd.get("one_goal", 1.0))
        self.gd_two = float(gd.get("two_goals", 1.5))
        self.gd_three = float(gd.get("three_goals", 1.75))
        self.gd_step = float(gd.get("step", 0.125))
        self.gd_cap = float(gd.get("cap", 3.25))

    def gd_steps(self, gd: int) -> float:
        if gd <= 1:
            return self.gd_one
        if gd == 2:
            return self.gd_two
        return min(self.gd_cap, self.gd_three + self.gd_step * (gd - 3))

    def gd_multiplier(self, g: Game, home_edge: float) -> float:
        return self.gd_steps(self.goal_diff(g))

    def gd_multiplier_vec(self, margin, winner_edge):
        table = np.array([self.gd_steps(m) for m in range(31)])
        return table[np.minimum(margin, 30)]
