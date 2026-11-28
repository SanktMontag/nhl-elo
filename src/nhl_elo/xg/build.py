"""Build the xG tables end to end.

    1. parse every cached play-by-play into shots           (data/cache/xg/shots_raw.parquet)
    2. fetch handedness for new shooters                    (data/players.csv)
    3. rink adjustment, rolling xG models, flurry adjustment (data/cache/xg/shots_xg.parquet)
    4. per-game team totals                                 (output/xg/game_xg.csv)
       and model scores                                     (output/xg/shot_scores.csv, rink_bias.csv)

The shot tables are large (≈2M rows) and rebuildable, so they live in the cache and aren't committed.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, OUTPUT_DIR
from . import model as M
from .pbp import final_games, load_pbp, pbp_path
from .players import add_handedness, fetch_players
from .rink import adjust, rink_report
from .shots import parse_game, shots_frame

XG_CACHE = CACHE_DIR / "xg"
XG_OUT = OUTPUT_DIR / "xg"


def _parse(args):
    gid, season, gt = args
    try:
        return parse_game(load_pbp(gid, season), gt)
    except Exception as e:  # a malformed feed shouldn't stop the build
        return [{"_error": f"{gid}: {e}"}]


def parse_all(games: pd.DataFrame | None = None, workers: int = 4) -> pd.DataFrame:
    games = final_games() if games is None else games
    todo = [(int(g.game_id), int(g.season), int(g.game_type)) for g in games.itertuples()
            if pbp_path(g.game_id, g.season).exists()]
    rows, errors = [], []
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_parse, todo, chunksize=64):
            for r in res:
                (errors.append(r["_error"]) if "_error" in r else rows.append(r))
    if errors:
        print(f"{len(errors)} games failed to parse, e.g. {errors[:3]}")
    return shots_frame(rows)


def game_totals(u: pd.DataFrame, all_shots: pd.DataFrame) -> pd.DataFrame:
    """One row per game and team: goals, shots, attempts, xG (raw and flurry-adjusted), all and 5v5."""
    a = all_shots.assign(ev5=(all_shots["state"] == "EV") & (all_shots["own_sk"] == 5) & (all_shots["opp_sk"] == 5))
    uu = u.assign(ev5=(u["state"] == "EV") & (u["own_sk"] == 5) & (u["opp_sk"] == 5))
    keys = ["game_id", "season", "game_type", "team", "opp", "is_home"]

    tot = []
    for sfx, mask_a, mask_u in (("", np.ones(len(a), bool), np.ones(len(uu), bool)),
                                ("_5v5", a["ev5"].to_numpy(), uu["ev5"].to_numpy())):
        ca = a[mask_a].groupby(keys, observed=True).size().rename(f"cf{sfx}")
        cu = uu[mask_u].groupby(keys, observed=True).agg(**{
            f"ff{sfx}": ("goal", "size"), f"gf{sfx}": ("goal", "sum"),
            f"sf{sfx}": ("event", lambda e: int(((e == "shot-on-goal") | (e == "goal")).sum())),
            f"xgf{sfx}": ("xg", "sum"), f"fxgf{sfx}": ("xg_flurry", "sum"), f"xgf_base{sfx}": ("xg_base", "sum")})
        tot += [ca, cu]
    t = pd.concat(tot, axis=1).fillna(0).reset_index()
    # attach the opponent's numbers as *_against
    opp = t.drop(columns=["opp", "is_home"]).rename(columns={"team": "opp"})
    metric = [c for c in t.columns if c not in keys]
    opp = opp.rename(columns={c: _against(c) for c in metric})
    t = t.merge(opp, on=["game_id", "season", "game_type", "opp"], how="left")
    return t


def _against(col: str) -> str:
    """xgf_5v5 -> xga_5v5, cf -> ca, xgf_base_5v5 -> xga_base_5v5."""
    stem, _, rest = col.partition("_")
    return stem[:-1] + "a" + ("_" + rest if rest else "")


def build(fetch_hands: bool = True, workers: int = 4, log=print) -> dict:
    t0 = time.time()
    XG_CACHE.mkdir(parents=True, exist_ok=True)
    XG_OUT.mkdir(parents=True, exist_ok=True)
    games = final_games()
    shots = parse_all(games, workers)
    shots.to_parquet(XG_CACHE / "shots_raw.parquet")
    log(f"parsed {shots['game_id'].nunique()} games, {len(shots):,} attempts  ({time.time() - t0:.0f}s)")

    if fetch_hands:
        fetch_players(shots["shooter"].dropna().unique())
    shots = add_handedness(shots)
    shots = adjust(shots)
    rink_report(shots).to_csv(XG_OUT / "rink_bias.csv", index=False)
    log(f"rink + handedness done ({time.time() - t0:.0f}s)")

    u = M.rolling_xg(shots, progress=lambda s: log(f"  season {s} ({time.time() - t0:.0f}s)"))
    u["xg_flurry"] = M.flurry_adjust(u, "xg")
    u.to_parquet(XG_CACHE / "shots_xg.parquet")

    scores = []
    for (season, state), g in u.groupby(["season", "state"], observed=True):
        for col in ("xg", "xg_base"):
            scores.append({"season": season, "state": state, "model": col, **M.shot_scores(g, col)})
    for season, g in u.groupby("season"):
        for col in ("xg", "xg_base"):
            scores.append({"season": season, "state": "ALL", "model": col, **M.shot_scores(g, col)})
    pd.DataFrame(scores).round(5).to_csv(XG_OUT / "shot_scores.csv", index=False)

    totals = game_totals(u, shots)
    totals.round(4).to_csv(XG_OUT / "game_xg.csv", index=False)
    log(f"done: {len(u):,} unblocked shots, {len(totals):,} team-games ({time.time() - t0:.0f}s)")
    return {"shots": shots, "u": u, "totals": totals}


if __name__ == "__main__":
    build()
