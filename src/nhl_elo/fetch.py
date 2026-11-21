"""Download seasons from the NHL API and write the normalized tables in data/.

data/games.csv      one row per game (played and upcoming), every season since first_season
data/teams.csv      NHL team ids -> franchise, plus the franchise the rating is kept under
data/divisions.csv  conference and division per franchise per season (from end-of-season standings)
data/seasons.csv    season ids, dates and rule flags
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .api import Client
from .config import DIVISIONS_CSV, GAMES_CSV, SEASONS_CSV, TEAMS_CSV, Settings, load_settings

GAME_COLS = ["game_id", "season", "game_type", "date", "start_et", "away_team_id", "home_team_id",
             "away_franchise", "home_franchise", "away", "home", "away_score", "home_score",
             "period", "state", "outcome", "neutral"]

# stats/rest gameStateId: 1 scheduled; 6 and 7 final (7 = official). 2-5 are pre-game/live.
FINAL_STATES = {6, 7}
SCHEDULED_STATES = {1, 2}


@dataclass
class FetchReport:
    seasons: list[int] = field(default_factory=list)
    games_played: int = 0
    games_upcoming: int = 0
    requests: int = 0
    warnings: list[str] = field(default_factory=list)


def outcome(game_type: int, period: int, away_score, home_score) -> str:
    """REG / OT / SO for a finished game, or T for a tie."""
    if away_score == home_score:
        return "T"
    if period <= 3:
        return "REG"
    if game_type == 2 and period >= 5:
        return "SO"  # regular-season OT is a single period; period 5 is the shootout
    return "OT"      # playoff OT periods continue until a goal


def teams_table(client: Client, settings: Settings) -> pd.DataFrame:
    franchises = {f["id"]: f["fullName"] for f in client.franchises()}
    cont = settings.continuation_map()
    rows = []
    for t in client.teams():
        fid = t.get("franchiseId")
        if fid is None:
            continue
        rid = cont.get(fid, fid)
        rows.append({"team_id": t["id"], "tricode": t["triCode"], "name": t["fullName"],
                     "franchise": fid, "rating_franchise": rid, "franchise_name": franchises.get(rid, t["fullName"])})
    return pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)


def normalize_games(raw: list[dict], teams: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, list[str]]:
    """stats/rest game rows -> GAME_COLS. Drops game types not in settings and unrecognised teams."""
    by_id = teams.set_index("team_id")
    rows, warnings = [], []
    for g in raw:
        if g["gameType"] not in settings.game_types:
            continue
        a, h = g["visitingTeamId"], g["homeTeamId"]
        if a not in by_id.index or h not in by_id.index:
            warnings.append(f"game {g['id']}: unknown team id {a if a not in by_id.index else h}")
            continue
        sid = g.get("gameStateId")
        if sid in FINAL_STATES:
            state = "final"
        elif g.get("gameScheduleStateId", 1) != 1:
            continue  # postponed / cancelled; a rescheduled game comes back as scheduled
        elif sid in SCHEDULED_STATES:
            state = "scheduled"
        else:
            state = "live"
        final = state == "final"
        rows.append({
            "game_id": g["id"], "season": g["season"], "game_type": g["gameType"],
            "date": g["gameDate"], "start_et": g.get("easternStartTime", ""),
            "away_team_id": a, "home_team_id": h,
            "away_franchise": int(by_id.at[a, "rating_franchise"]), "home_franchise": int(by_id.at[h, "rating_franchise"]),
            "away": by_id.at[a, "tricode"], "home": by_id.at[h, "tricode"],
            "away_score": g.get("visitingScore") if final else None,
            "home_score": g.get("homeScore") if final else None,
            "period": g.get("period") if final else None,
            "state": state,
            "outcome": outcome(g["gameType"], g["period"], g["visitingScore"], g["homeScore"]) if final else "",
            "neutral": settings.is_neutral(g["season"], g["gameType"]),
        })
    df = pd.DataFrame(rows, columns=GAME_COLS)
    for c in ("away_score", "home_score", "period"):
        df[c] = df[c].astype("Int64")
    return df, warnings


def divisions_table(standings: list[dict], season: int, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Map standings rows (keyed by tricode) to the franchises that played this season."""
    sg = games.loc[games["season"] == season]
    ids = set(sg["away_team_id"]) | set(sg["home_team_id"])
    tri = teams.loc[teams["team_id"].isin(ids)].set_index("tricode")["rating_franchise"].to_dict()
    rows = []
    for s in standings:
        code = s["teamAbbrev"]["default"]
        if code in tri:
            rows.append({"season": season, "franchise": int(tri[code]), "tricode": code,
                         "conference": s.get("conferenceName", ""), "division": s.get("divisionName", "")})
    return pd.DataFrame(rows, columns=["season", "franchise", "tricode", "conference", "division"])


def fetch(refresh_all: bool = False, offline: bool = False, client: Client | None = None) -> FetchReport:
    settings = load_settings()
    client = client or Client(settings, offline=offline)
    rep = FetchReport()

    teams = teams_table(client, settings)
    seasons = [s for s in client.standings_seasons() if s["id"] >= settings.first_season]
    today = date.today().isoformat()
    # The current season is the latest one that has started (or the first upcoming one in the off-season).
    started = [s for s in seasons if s["standingsStart"] <= today]
    current_id = (started or seasons)[-1]["id"]
    seasons = [s for s in seasons if s["id"] <= current_id]

    frames, divs, season_rows = [], [], []
    for s in seasons:
        sid, is_current = s["id"], s["id"] == current_id
        raw = client.season_games(sid, current=is_current, refresh=refresh_all)
        g, warn = normalize_games(raw, teams, settings)
        rep.warnings += warn
        frames.append(g)
        st = client.standings(s["standingsEnd"], sid, current=is_current, refresh=refresh_all)
        d = divisions_table(st, sid, g, teams)
        played_fr = set(g["away_franchise"]) | set(g["home_franchise"])
        if missing := played_fr - set(d["franchise"]):
            rep.warnings.append(f"{sid}: no division for franchise(s) {sorted(missing)}")
        divs.append(d)
        season_rows.append({"season": sid, "start": s["standingsStart"], "end": s["standingsEnd"],
                            "current": is_current, "ties": s.get("tiesInUse", False),
                            "regular_games": int((g["game_type"] == 2).sum()),
                            "playoff_games": int((g["game_type"] == 3).sum())})
        rep.seasons.append(sid)

    games = pd.concat(frames, ignore_index=True).sort_values(["date", "start_et", "game_id"]).reset_index(drop=True)
    dup = games["game_id"].duplicated()
    if dup.any():
        rep.warnings.append(f"{int(dup.sum())} duplicate game ids dropped")
        games = games.loc[~dup]

    TEAMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(GAMES_CSV, index=False)
    teams.to_csv(TEAMS_CSV, index=False)
    pd.concat(divs, ignore_index=True).to_csv(DIVISIONS_CSV, index=False)
    pd.DataFrame(season_rows).to_csv(SEASONS_CSV, index=False)

    rep.games_played = int((games["state"] == "final").sum())
    rep.games_upcoming = int(((games["state"] != "final") & (games["season"] == current_id)).sum())
    rep.requests = client.requests
    return rep


def load_games() -> pd.DataFrame:
    if not GAMES_CSV.exists():
        raise FileNotFoundError(f"{GAMES_CSV} not found. Run `nhl fetch` first.")
    df = pd.read_csv(GAMES_CSV, parse_dates=["date"],
                     dtype={"away_score": "Int64", "home_score": "Int64", "period": "Int64", "outcome": str})
    df["outcome"] = df["outcome"].fillna("")
    return df


def load_teams() -> pd.DataFrame:
    return pd.read_csv(TEAMS_CSV)


def load_divisions() -> pd.DataFrame:
    return pd.read_csv(DIVISIONS_CSV) if DIVISIONS_CSV.exists() else pd.DataFrame(
        columns=["season", "franchise", "tricode", "conference", "division"])


def load_seasons() -> pd.DataFrame:
    return pd.read_csv(SEASONS_CSV)
