"""Player handedness and position, from api-web.nhle.com/v1/player/{id}/landing.

data/players.csv: player_id, name, position, shoots (L/R), birth_date. Only players not already in the
file are fetched, so after the first run it's a handful of requests per new player.

Off-wing: with shots rotated to attack +x, the left side of the ice is +y. A left-handed shooter on
the right side (y < 0) or a right-handed shooter on the left (y > 0) is on his off wing, which opens
up the angle for one-timers and shots across the body.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..api import WEB, ApiError, Client
from ..config import DATA_DIR

PLAYERS_CSV = DATA_DIR / "players.csv"
COLS = ["player_id", "name", "position", "shoots", "birth_date"]


def load_players() -> pd.DataFrame:
    if PLAYERS_CSV.exists():
        return pd.read_csv(PLAYERS_CSV, dtype={"shoots": "string", "position": "string"})
    return pd.DataFrame(columns=COLS)


def fetch_players(ids, workers: int = 3) -> pd.DataFrame:
    have = load_players()
    todo = sorted(set(int(i) for i in ids if pd.notna(i)) - set(have["player_id"].astype(int)))
    local = threading.local()

    def one(pid: int):
        if not hasattr(local, "client"):
            local.client = Client()
        try:
            d = local.client._download(WEB + f"player/{pid}/landing")
        except ApiError:
            return {"player_id": pid}
        name = f"{d.get('firstName', {}).get('default', '')} {d.get('lastName', {}).get('default', '')}".strip()
        return {"player_id": pid, "name": name, "position": d.get("position"),
                "shoots": d.get("shootsCatches"), "birth_date": d.get("birthDate")}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, todo))
    out = pd.concat([have, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    out = out.drop_duplicates("player_id", keep="last").sort_values("player_id")
    out.to_csv(PLAYERS_CSV, index=False)
    return out


def add_handedness(shots: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    players = load_players() if players is None else players
    hand = players.set_index("player_id")["shoots"]
    s = shots["shooter"].map(hand)
    out = shots.copy()
    out["shoots"] = s
    y = out["y"].to_numpy(dtype=float)
    out["off_wing"] = np.select([s.eq("L").fillna(False).to_numpy(), s.eq("R").fillna(False).to_numpy()],
                                [(y < 0).astype(float), (y > 0).astype(float)], default=np.nan)
    return out
