"""Replay every season since first_season as one rating chain and write the output tables.

Ratings are keyed by franchise (after continuations), so relocations and renames keep their rating.
Between seasons every rating is pulled toward the mean (method `season.offseason_regression`);
expansion franchises start at mean + new_team_offset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GAMES_CSV, OUTPUT_DIR, load_settings, season_label
from .fetch import load_divisions, load_games, load_teams
from .methods import Game, Method, get_method

RESULT_CODE = {("REG", True): "W", ("REG", False): "L", ("OT", True): "OTW", ("OT", False): "OTL",
               ("SO", True): "SOW", ("SO", False): "SOL"}
GAME_TYPE = {2: "REG", 3: "PO"}


@dataclass
class ChainResult:
    team_games: pd.DataFrame       # empty in light mode
    games: pd.DataFrame            # empty in light mode
    season_start: dict[int, dict[int, float]]
    season_end: dict[int, dict[int, float]]
    final: dict[int, float]
    scored: pd.DataFrame           # one row per played game: season, game_type, e, s, home_win


def replay(games: pd.DataFrame, method: Method, light: bool = False,
           names: dict[int, str] | None = None) -> ChainResult:
    """Apply `method` to every final game in date order, season by season.

    games: GAME_COLS rows (see fetch.py), any order; non-final rows only register teams for their season.
    names: team_id -> display name (light=False only).
    """
    names = names or {}
    games = games.sort_values(["date", "start_et", "game_id"], kind="stable")
    ratings: dict[int, float] = {}
    season_start: dict[int, dict[int, float]] = {}
    season_end: dict[int, dict[int, float]] = {}
    tg_rows, game_rows = [], []
    sc_season, sc_type, sc_e, sc_s, sc_win = [], [], [], [], []

    for season, sg in games.groupby("season", sort=True):
        for f in ratings:
            ratings[f] = method.season_start(ratings[f])
        in_season = set(sg["away_franchise"]) | set(sg["home_franchise"])
        for f in sorted(in_season - set(ratings)):
            ratings[f] = method.new_team_rating()
        season_start[season] = {f: ratings[f] for f in in_season}

        played = sg.loc[sg["state"] == "final"]
        cols = zip(played["away_franchise"], played["home_franchise"], played["away_score"],
                   played["home_score"], played["outcome"], played["game_type"], played["neutral"],
                   played["game_id"], played["date"], played["away_team_id"], played["home_team_id"])
        for af, hf, asc, hsc, oc, gt, neu, gid, d, at, ht in cols:
            g = Game(int(asc), int(hsc), oc, gt == 3, bool(neu))
            ra, rh = ratings[af], ratings[hf]
            u = method.update(ra, rh, g)
            ratings[af] = ra + u.away_delta
            ratings[hf] = rh + u.home_delta
            sc_season.append(season)
            sc_type.append(gt)
            sc_e.append(u.home_expected)
            sc_s.append(u.home_score)
            sc_win.append(1.0 if hsc > asc else 0.0 if hsc < asc else 0.5)
            if light:
                continue
            ds = d.strftime("%Y-%m-%d")
            game_rows.append({
                "game_id": gid, "Date": ds, "Game Type": GAME_TYPE.get(gt, gt),
                "Away": names.get(at, at), "Home": names.get(ht, ht),
                "Away Score": asc, "Home Score": hsc, "Outcome": oc, "Neutral": bool(neu),
                "Away ELO Prior": round(ra), "Home ELO Prior": round(rh),
                "Home Win %": round(u.home_expected * 100, 1),
                "Home Delta": round(u.home_delta, 1),
                "Away ELO Post": round(ratings[af]), "Home ELO Post": round(ratings[hf]),
                "away_franchise": af, "home_franchise": hf,
            })
            for side, fr, of, tid, oid, r, ro, gf, ga, e, delta in (
                ("Home", hf, af, ht, at, rh, ra, hsc, asc, u.home_expected, u.home_delta),
                ("Away", af, hf, at, ht, ra, rh, asc, hsc, 1 - u.home_expected, u.away_delta),
            ):
                tg_rows.append({
                    "Date": ds, "Game Type": GAME_TYPE.get(gt, gt), "Team": names.get(tid, tid),
                    "ELO": round(r), "Opponent": names.get(oid, oid), "Opp ELO": round(ro),
                    "Side": side, "Win %": round(e * 100, 1), "GF": gf, "GA": ga,
                    "Result": "T" if gf == ga else RESULT_CODE[(oc, gf > ga)],
                    "+/- ELO": round(delta, 1), "ELO Post Game": round(r + delta),
                    "game_id": gid, "franchise": fr, "opp_franchise": of,
                })
        season_end[season] = {f: ratings[f] for f in in_season}

    scored = pd.DataFrame({"season": sc_season, "game_type": sc_type, "e": sc_e, "s": sc_s, "home_win": sc_win})
    return ChainResult(pd.DataFrame(tg_rows), pd.DataFrame(game_rows), season_start, season_end,
                       dict(ratings), scored)


# ---------------------------------------------------------------- scoring

def score(scored: pd.DataFrame) -> dict:
    """Prediction quality of E_home as a home-win probability (lower is better except pick_pct)."""
    d = scored.loc[scored["home_win"] != 0.5]
    e, w = d["e"].to_numpy(), d["home_win"].to_numpy()
    if not len(d):
        return {"games": 0}
    p = np.clip(np.where(w == 1, e, 1 - e), 1e-9, 1)
    return {
        "games": int(len(d)),
        "logloss": float(-np.mean(np.log(p))),
        "brier": float(np.mean((w - e) ** 2)),
        "pick_pct": float(np.mean((e > 0.5) == (w == 1))),
        "home_win_pred": float(np.mean(e)),
        "home_win_actual": float(np.mean(w)),
    }


def calibration(scored: pd.DataFrame) -> pd.DataFrame:
    """Favourite's predicted win probability vs how often favourites actually won."""
    d = scored.loc[scored["home_win"] != 0.5]
    e, w = d["e"].to_numpy(), d["home_win"].to_numpy()
    flip = e < 0.5
    df = pd.DataFrame({"e": np.where(flip, 1 - e, e), "w": np.where(flip, 1 - w, w)})
    df["bin"] = pd.cut(df["e"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0], include_lowest=True)
    return df.groupby("bin", observed=True).agg(n=("w", "size"), predicted=("e", "mean"), actual=("w", "mean")).round(3)


# ---------------------------------------------------------------- full run

@dataclass
class RunResult:
    method: Method
    out_dir: Path
    current_season: int
    ratings: pd.DataFrame
    accuracy: pd.DataFrame
    meta: dict


def run(method_name: str = "v1", write: bool = True) -> RunResult:
    settings = load_settings()
    method = get_method(method_name)
    games = load_games()
    games = games.loc[games["season"] >= settings.first_season]
    teams = load_teams()
    divisions = load_divisions()
    names = dict(zip(teams["team_id"], teams["name"]))
    abbrev = dict(zip(teams["team_id"], teams["tricode"]))
    fr_name = dict(zip(teams["rating_franchise"], teams["franchise_name"]))

    rr = replay(games, method, names=names)
    seasons = sorted(games["season"].unique())
    current = int(seasons[-1])

    out = OUTPUT_DIR / method_name
    tables: dict[int, dict[str, pd.DataFrame]] = {}
    history, accuracy = [], []
    for season in seasons:
        season = int(season)
        sg = games.loc[games["season"] == season]
        div = divisions.loc[divisions["season"] == season].set_index("franchise")
        # latest team id per franchise this season (for its name that season)
        side = pd.concat([sg[["date", "away_franchise", "away_team_id"]].set_axis(["date", "f", "t"], axis=1),
                          sg[["date", "home_franchise", "home_team_id"]].set_axis(["date", "f", "t"], axis=1)])
        team_of = side.sort_values("date").drop_duplicates("f", keep="last").set_index("f")["t"].to_dict()
        tg = rr.team_games.loc[rr.team_games["game_id"].isin(set(sg["game_id"]))] if len(rr.team_games) \
            else pd.DataFrame(columns=["franchise", "Game Type", "Result", "GF", "GA", "ELO Post Game"])

        rows = []
        for f, end in rr.season_end[season].items():
            mine = tg.loc[tg["franchise"] == f]
            reg, po = mine.loc[mine["Game Type"] == "REG"], mine.loc[mine["Game Type"] == "PO"]
            res = reg["Result"]
            start = rr.season_start[season][f]
            peak = mine["ELO Post Game"].max() if len(mine) else round(start)
            rows.append({
                "franchise": f, "Team": names.get(team_of[f], fr_name.get(f)), "Abbrev": abbrev.get(team_of[f], ""),
                "Conference": div["conference"].get(f, "") if len(div) else "",
                "Division": div["division"].get(f, "") if len(div) else "",
                "ELO": round(end), "GP": len(reg),
                "W": int(res.isin(["W", "OTW", "SOW"]).sum()), "L": int((res == "L").sum()),
                "OTL": int(res.isin(["OTL", "SOL"]).sum()),
                "PTS": int(2 * res.isin(["W", "OTW", "SOW"]).sum() + res.isin(["OTL", "SOL"]).sum()),
                "GF": int(reg["GF"].sum()), "GA": int(reg["GA"].sum()),
                "PO W": int((po["GF"] > po["GA"]).sum()), "PO L": int((po["GF"] < po["GA"]).sum()),
                "Start ELO": round(start), "Peak ELO": int(peak), "Season +/-": round(end - start),
            })
        ratings = pd.DataFrame(rows).sort_values(["ELO", "Team"], ascending=[False, True], kind="stable").reset_index(drop=True)
        ratings.insert(0, "Rank", ratings.index + 1)
        for r in ratings.to_dict("records"):
            history.append({"season": season, "Season": season_label(season), "franchise": r["franchise"],
                            "Franchise": fr_name.get(r["franchise"], r["Team"]), "Team": r["Team"],
                            "Rank": r["Rank"], "Start ELO": r["Start ELO"], "End ELO": r["ELO"],
                            "Peak ELO": r["Peak ELO"], "GP": r["GP"], "W": r["W"], "L": r["L"],
                            "OTL": r["OTL"], "PTS": r["PTS"], "PO W": r["PO W"], "PO L": r["PO L"]})

        sc = score(rr.scored.loc[rr.scored["season"] == season])
        if sc["games"]:
            accuracy.append({"season": season, "Season": season_label(season), **sc})

        preds = pd.DataFrame()
        if season == current:
            up = sg.loc[sg["state"] != "final"]
            prow = []
            for g in up.itertuples(index=False):
                ra, rh = rr.final[g.away_franchise], rr.final[g.home_franchise]
                e = method.home_expected(ra, rh, bool(g.neutral))
                prow.append({"Date": g.date.strftime("%Y-%m-%d"), "Game Type": GAME_TYPE.get(g.game_type),
                             "Away": names[g.away_team_id], "Home": names[g.home_team_id],
                             "Away ELO": round(ra), "Home ELO": round(rh),
                             "Away Win %": round((1 - e) * 100, 1), "Home Win %": round(e * 100, 1),
                             "game_id": g.game_id})
            preds = pd.DataFrame(prow, columns=["Date", "Game Type", "Away", "Home", "Away ELO", "Home ELO",
                                                "Away Win %", "Home Win %", "game_id"])
        tables[season] = {"ratings": ratings, "team_games": tg.reset_index(drop=True),
                          "games": rr.games.loc[rr.games["game_id"].isin(set(sg["game_id"]))],
                          "predictions": preds}

    accuracy_df = pd.DataFrame(accuracy)
    scored_from = rr.scored.loc[rr.scored["season"] >= settings.score_from]
    played = games.loc[games["state"] == "final"]
    cur_played = played.loc[played["season"] == current]
    meta = {
        "method": method.name, "method_description": method.description, "params": method.cfg,
        "params_hash": method.params_hash(), "first_season": settings.first_season,
        "current_season": current, "current_season_label": season_label(current),
        "seasons": [int(s) for s in seasons],
        "games_played": int(len(played)), "games_played_current": int(len(cur_played)),
        "games_upcoming": int(((games["state"] != "final") & (games["season"] == current)).sum()),
        "last_game": cur_played["date"].max().strftime("%Y-%m-%d") if len(cur_played) else None,
        "score": score(scored_from), "score_from": settings.score_from,
        "games_sha256": _sha(GAMES_CSV), "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if write:
        # Overwrite in place rather than clearing the folder: the simulation files in each season folder
        # (simulation*.csv, written by `nhl simulate`) keep a history that can't be rebuilt from here.
        for season, t in tables.items():
            d = out / str(season)
            d.mkdir(parents=True, exist_ok=True)
            for name, df in t.items():
                if name == "predictions" and df.empty:
                    (d / "predictions.csv").unlink(missing_ok=True)
                    continue
                df.to_csv(d / f"{name}.csv", index=False)
        for d in out.iterdir():
            if d.is_dir() and d.name.isdigit() and int(d.name) not in tables:
                shutil.rmtree(d)  # a season no longer in the data (e.g. first_season moved)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        accuracy_df.round(4).to_csv(out / "accuracy.csv", index=False)
        calibration(scored_from).reset_index().assign(bin=lambda x: x["bin"].astype(str)) \
            .to_csv(out / "calibration.csv", index=False)
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return RunResult(method, out, current, tables[current]["ratings"], accuracy_df, meta)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None
