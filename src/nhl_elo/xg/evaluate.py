"""Does xG tell us more about a team than goals, shots or shot attempts do?

For every team-season (regular season), split the games in two and ask how well each team's share of
a metric in one half predicts its goal share in the other half:

    split="halves"   first half of the schedule -> second half (prediction: does it hold up?)
    split="oddeven"  odd games -> even games (repeatability, no time trend)

Metrics: GF% (goals), SF% (shots on goal), CF% (all attempts), FF% (unblocked attempts),
xGF% (raw xG), fxGF% (flurry-adjusted xG), bxGF% (logistic baseline xG). Each for all situations
and 5v5. The score is the correlation with the other half's GF%, pooled across seasons, plus the
season-by-season mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fetch import load_games

METRICS = {"GF%": ("gf", "ga"), "SF%": ("sf", "sa"), "CF%": ("cf", "ca"), "FF%": ("ff", "fa"),
           "xGF%": ("xgf", "xga"), "fxGF%": ("fxgf", "fxga"), "bxGF%": ("xgf_base", "xga_base")}


def _share(df: pd.DataFrame, f: str, a: str) -> pd.Series:
    return df[f] / (df[f] + df[a])


def split_halves(totals: pd.DataFrame, split: str = "halves", min_games: int = 40) -> pd.DataFrame:
    games = load_games()[["game_id", "date"]]
    t = totals.loc[totals["game_type"] == 2].merge(games, on="game_id")
    t = t.sort_values(["season", "team", "date", "game_id"])
    t["n"] = t.groupby(["season", "team"]).cumcount()
    t["gp"] = t.groupby(["season", "team"])["n"].transform("size")
    t = t.loc[t["gp"] >= min_games]
    t["half"] = np.where(t["n"] < t["gp"] / 2, 0, 1) if split == "halves" else t["n"] % 2
    num = [c for c in t.columns if c not in ("game_id", "season", "game_type", "team", "opp", "is_home",
                                              "date", "n", "gp", "half")]
    return t.groupby(["season", "team", "half"])[num].sum().reset_index()


def predictive_table(totals: pd.DataFrame, split: str = "halves", target: str = "GF%") -> pd.DataFrame:
    h = split_halves(totals, split)
    rows = []
    for sfx, label in (("", "all"), ("_5v5", "5v5")):
        tf, ta = METRICS[target]
        y = h.loc[h["half"] == 1].set_index(["season", "team"])
        y = _share(y, tf + sfx, ta + sfx)
        x0 = h.loc[h["half"] == 0].set_index(["season", "team"])
        for name, (f, a) in METRICS.items():
            x = _share(x0, f + sfx, a + sfx)
            d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
            per = d.groupby(level="season").apply(lambda g: g["x"].corr(g["y"]))
            rows.append({"situation": label, "metric": name, "r_pooled": d["x"].corr(d["y"]),
                         "r_season_mean": per.mean(), "seasons": len(per), "team_seasons": len(d)})
    return pd.DataFrame(rows).round(3)
