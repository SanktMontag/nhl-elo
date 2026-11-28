"""Download and cache gamecenter play-by-play for every final game since XG_FIRST_SEASON.

    api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Shot coordinates start in 2009-10 (earlier games have shot events but no x/y). Each response is
~120 KB of JSON, ~11 KB gzipped; the cache is data/cache/pbp/<season>/<game_id>.json.gz and finished
games never expire. Downloads run on a few workers, each keeping the client's per-request delay.
"""

from __future__ import annotations

import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..api import WEB, ApiError, Client
from ..config import CACHE_DIR
from ..fetch import load_games

XG_FIRST_SEASON = 20092010
PBP_DIR = CACHE_DIR / "pbp"


def pbp_path(game_id: int, season: int) -> Path:
    return PBP_DIR / str(season) / f"{game_id}.json.gz"


def load_pbp(game_id: int, season: int) -> dict:
    with gzip.open(pbp_path(game_id, season), "rt", encoding="utf-8") as f:
        return json.load(f)


def final_games(first_season: int = XG_FIRST_SEASON) -> pd.DataFrame:
    g = load_games()
    return g.loc[(g["season"] >= first_season) & (g["state"] == "final")].sort_values("game_id")


def fetch_pbp(games: pd.DataFrame | None = None, workers: int = 3, progress=None) -> dict:
    """Download play-by-play for final games not yet cached. Returns counts."""
    games = final_games() if games is None else games
    todo = [(int(r.game_id), int(r.season)) for r in games.itertuples() if not pbp_path(r.game_id, r.season).exists()]
    local = threading.local()
    done, failed = 0, []

    def one(gid: int, season: int):
        if not hasattr(local, "client"):
            local.client = Client()
        data = local.client._download(WEB + f"gamecenter/{gid}/play-by-play")
        p = pbp_path(gid, season)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        tmp.replace(p)

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, gid, s): gid for gid, s in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except ApiError as e:
                failed.append((futs[fut], str(e)))
            if progress and (done + len(failed)) % 250 == 0:
                progress(done + len(failed), len(todo))
    return {"todo": len(todo), "downloaded": done, "failed": failed}


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    res = fetch_pbp(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 3,
                    progress=lambda i, n: print(f"{i}/{n}  {time.time() - t0:.0f}s", flush=True))
    print({k: (v if k != "failed" else len(v)) for k, v in res.items()}, flush=True)
    for gid, err in res["failed"][:20]:
        print("failed", gid, err)
