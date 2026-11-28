"""Arena scorekeeper bias: adjust recorded shot distance rink by rink (Schuckers & Curro, 2013).

Each home rink's scorekeeper records shot locations for both teams, and some record shots closer to or
farther from the net than others. The road teams that visit a rink are a broad sample of the league,
so the distance distribution of road-team shots at that rink should match road-team shots everywhere
else. Where it doesn't, the difference is mostly the scorekeeper.

For each season and home rink:
    F_rink  CDF of road-team unblocked even-strength shot distance at the rink
    F_ref   the same for road-team shots at every other rink that season
    adjusted distance = F_ref⁻¹(F_rink(d))       (quantile mapping, applied to both teams' shots)

The adjustment is blended toward no change by n / (n + SHRINK_N) shots, and rescales the shot's
offset from the net so the angle is unchanged. Only shot locations are used, never outcomes, so it
can see the whole season without leaking results into the xG model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .shots import NET_X

QUANTILES = np.linspace(0, 1, 51)
SHRINK_N = 400
MAX_DIST = 100.0


def rink_tables(shots: pd.DataFrame) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]]:
    """(season, rink team) -> (rink quantiles, reference quantiles, weight)."""
    base = shots.loc[shots["unblocked"] & (shots["state"] == "EV") & ~shots["is_home"] & (shots["dist"] <= MAX_DIST)]
    base = base.assign(rink=base["opp"])
    out = {}
    for season, sg in base.groupby("season"):
        d_all = sg["dist"].to_numpy()
        rinks = sg["rink"].to_numpy()
        for rink in np.unique(rinks):
            mine = d_all[rinks == rink]
            ref = d_all[rinks != rink]
            if len(mine) < 50:
                continue
            w = len(mine) / (len(mine) + SHRINK_N)
            out[(int(season), int(rink))] = (np.quantile(mine, QUANTILES), np.quantile(ref, QUANTILES), w)
    return out


def adjust(shots: pd.DataFrame, tables=None) -> pd.DataFrame:
    """Add dist_adj, x_adj, y_adj and rink_shift (mean adjusted − recorded distance at the rink)."""
    tables = rink_tables(shots) if tables is None else tables
    out = shots.copy()
    rink = np.where(out["is_home"], out["team"], out["opp"])
    dist = out["dist"].to_numpy(dtype=float)
    adj = dist.copy()
    keys = list(zip(out["season"].astype(int), rink.astype(int)))
    idx = pd.Series(range(len(out))).groupby(pd.Series(keys)).indices if len(out) else {}
    for key, rows in idx.items():
        if key not in tables:
            continue
        q_rink, q_ref, w = tables[key]
        d = dist[rows]
        inside = d <= q_rink[-1]
        mapped = np.where(inside, np.interp(d, q_rink, q_ref), d + (q_ref[-1] - q_rink[-1]))
        adj[rows] = np.maximum(0.5, w * mapped + (1 - w) * d)
    scale = np.where(dist > 0, adj / np.where(dist > 0, dist, 1), 1.0)
    out["dist_adj"] = adj
    out["x_adj"] = NET_X - (NET_X - out["x"]) * scale
    out["y_adj"] = out["y"] * scale
    out["rink"] = rink
    return out


def rink_report(shots_adj: pd.DataFrame) -> pd.DataFrame:
    """Per season and rink: mean recorded vs adjusted road-team EV distance."""
    b = shots_adj.loc[shots_adj["unblocked"] & (shots_adj["state"] == "EV") & ~shots_adj["is_home"]]
    return (b.groupby(["season", "rink"]).agg(n=("dist", "size"), recorded=("dist", "mean"), adjusted=("dist_adj", "mean"))
             .assign(shift=lambda x: x["adjusted"] - x["recorded"]).round(2).reset_index())
