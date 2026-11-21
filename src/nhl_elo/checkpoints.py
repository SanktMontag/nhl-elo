"""How right were the season simulations? Past seasons re-simulated from fixed points in the season.

At each checkpoint (opening night, US Thanksgiving, Christmas, the All-Star break and the trade deadline)
the season is simulated using only the games played before that morning, exactly as `nhl simulate`
would have done on the day. The predictions are stored next to what actually happened, and next to the
simplest alternative: "the teams in a playoff spot that day make it".

Output: output/<method>/checkpoints.csv (one row per team per checkpoint per season) and
checkpoints_meta.json. Finished seasons are computed once; the current season gains a checkpoint as each
date passes, and gets its results filled in once the Cup is awarded.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, season_label
from .simulate import Simulator, SeasonState, _actual_outcomes, _seed_wildcard, simulate

CHECKPOINTS = ["Opening night", "Thanksgiving", "Christmas", "All-Star break", "Trade deadline"]
WILDCARD_FROM = 20132014                  # first season of the current playoff format
IRREGULAR = (20192020, 20202021)          # 24-team 2020 playoffs; 2021 realigned divisions
ODDS_KEPT = ["Playoffs", "Div Title", "Round 2", "Conf Final", "Cup Final", "Stanley Cup", "Presidents"]
TRUTH = ["made_playoffs", "won_division", "won_cup", "final_pts"]


def eligible_seasons(sim: Simulator) -> list[int]:
    seasons = sorted(int(s) for s in sim.games["season"].unique())
    return [s for s in seasons if s >= WILDCARD_FROM and s not in IRREGULAR]


def checkpoint_dates(sim: Simulator, season: int) -> dict[str, str]:
    """Checkpoint name -> 'as of' date (predictions use games before that day)."""
    reg = sim.games.loc[(sim.games["season"] == season) & (sim.games["game_type"] == 2)]
    dates = pd.DatetimeIndex(sorted(reg["date"].unique()))
    if dates.empty:
        return {}
    y0 = int(str(season)[:4])
    first, last = dates[0].date(), dates[-1].date()
    nov1 = date(y0, 11, 1)
    out = {
        "Opening night": first,
        "Thanksgiving": nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21),   # 4th Thursday of November
        "Christmas": date(y0, 12, 25),
    }
    # The longest league-wide break between Jan 10 and Mar 10: the All-Star break, or in some seasons
    # the Olympic or 4 Nations break. Predictions are made the morning after the last game before it.
    mid = dates[(dates >= pd.Timestamp(y0 + 1, 1, 10)) & (dates <= pd.Timestamp(y0 + 1, 3, 10))]
    if len(mid) > 1:
        gaps = np.diff(mid.values).astype("timedelta64[D]").astype(int)
        i = int(gaps.argmax())
        if gaps[i] >= 3:
            out["All-Star break"] = mid[i].date() + timedelta(days=1)
    deadline = sim.settings.trade_deadlines.get(season)
    if deadline:
        out["Trade deadline"] = date.fromisoformat(deadline)
    return {k: v.isoformat() for k, v in out.items() if first <= v <= last}


def in_playoff_spot(st: SeasonState, playoffs: dict) -> np.ndarray:
    """1/0 per team: in a playoff position on the standings so far, ranked by points percentage (teams
    have played different numbers of games), then regulation wins, ROW, wins and goal differential."""
    if st.gp.sum() == 0:
        return np.full(st.n, np.nan)
    pct = st.pts / (2 * np.maximum(st.gp, 1))
    order = np.lexsort((st.gd, st.w, st.row, st.rw, pct))          # ascending, last key first
    key = np.empty(st.n)
    key[order] = np.arange(st.n)
    *_, seeding = _seed_wildcard(st, key[None, :], int(playoffs.get("division_spots", 3)),
                                 int(playoffs.get("wildcards", 2)))
    return (seeding[0] > 0).astype(float)


def _cup_awarded(st: SeasonState) -> bool:
    final = st.playoffs.loc[(st.playoffs["rnd"] == 4) & st.playoffs["final"]]
    if final.empty:
        return False
    winners = np.where(final["home_won"], final["h"], final["a"])
    return bool(pd.Series(winners).value_counts().max() >= 4)


def _picked(df: pd.DataFrame) -> pd.Series:
    """The 8 teams per conference with the best playoff odds (projected points break ties)."""
    rank = df.sort_values(["Playoffs", "Proj PTS"], ascending=False).groupby("Conference").cumcount()
    return (rank < 8).astype(int).reindex(df.index)


def run_checkpoints(sim: Simulator, sims: int | None = None, force: bool = False,
                    progress=None) -> pd.DataFrame:
    """Compute missing checkpoints (all of them with force) and write checkpoints.csv."""
    cfg = sim.settings.simulation
    sims = int(sims or cfg["checkpoint_sims"])
    out_dir = OUTPUT_DIR / sim.method.name
    path, meta_path = out_dir / "checkpoints.csv", out_dir / "checkpoints_meta.json"
    meta = {"sims": sims, "seed": int(cfg["seed"]), "hot": bool(cfg["hot"]), "method": sim.method.name,
            "params_hash": sim.method.params_hash()}
    old = pd.DataFrame()
    if path.exists() and meta_path.exists() and not force:
        if json.loads(meta_path.read_text()) == meta:      # same model and settings: reuse earlier work
            old = pd.read_csv(path)

    frames, todo = [], []
    for s in eligible_seasons(sim):
        full = sim.state(s)
        done = _cup_awarded(full)
        dates = checkpoint_dates(sim, s)
        if not done and full.through:     # current season: only checkpoints that have passed
            last = date.fromisoformat(full.through) + timedelta(days=1)
            dates = {k: v for k, v in dates.items() if date.fromisoformat(v) <= last}
        if not dates:
            continue
        have = old.loc[old["season"] == s] if len(old) else old
        same = len(have) and set(zip(have["checkpoint"], have["as_of"])) == set(dates.items()) \
            and (not done or have["made_playoffs"].notna().all())
        if same:
            frames.append(have)
        else:
            todo.append((s, dates, full, done))

    for i, (s, dates, full, done) in enumerate(todo, 1):
        truth = _actual_outcomes(sim, s, full) if done else None
        for name, as_of in dates.items():
            st = sim.state(s, as_of)
            res = simulate(st, sim.method, sim.rates, sims=sims, seed=meta["seed"], hot=meta["hot"],
                           playoffs=sim.settings.playoffs)
            spot = pd.DataFrame({"franchise": st.franchises, "in_spot": in_playoff_spot(st, sim.settings.playoffs)})
            t = res.table[["franchise", "Team", "Abbrev", "Conference", "Division", "ELO", "GP", "PTS", "Proj PTS"]
                          + ODDS_KEPT].merge(spot, on="franchise")
            t["picked"] = _picked(t)
            t = t.merge(truth, on="franchise", how="left") if truth is not None \
                else t.assign(**{c: np.nan for c in TRUTH})
            frames.append(t.assign(season=s, checkpoint=name, as_of=as_of, through=res.meta["through"]))
        if progress:
            progress(i, len(todo), s)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df["order"] = df["checkpoint"].map({c: i for i, c in enumerate(CHECKPOINTS)})
        df = df.sort_values(["season", "order", "Playoffs", "Proj PTS"], ascending=[True, True, False, False])
        cols = ["season", "checkpoint", "as_of", "through", "franchise", "Team", "Abbrev", "Conference",
                "Division", "ELO", "GP", "PTS", "in_spot", "Proj PTS", *ODDS_KEPT, "picked", *TRUTH]
        df = df[cols].reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2))
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint, over finished seasons: playoff teams called right (of 16) by the model and by the
    standings that day, Brier scores, and the eventual champion's Cup odds."""
    f = df.loc[df["made_playoffs"].notna()]
    rows = []
    for (s, cp), g in f.groupby(["season", "checkpoint"], sort=False):
        made = g["made_playoffs"] == 1
        champ = g.loc[g["won_cup"] == 1]
        rows.append({
            "season": s, "checkpoint": cp, "as_of": g["as_of"].iat[0],
            "model_right": int((made & (g["picked"] == 1)).sum()),
            "standings_right": int((made & (g["in_spot"] == 1)).sum()) if g["in_spot"].notna().all() else np.nan,
            "model_brier": float(((g["Playoffs"] - made) ** 2).mean()),
            "standings_brier": float(((g["in_spot"] - made) ** 2).mean()) if g["in_spot"].notna().all() else np.nan,
            "champion": champ["Team"].iat[0] if len(champ) else "",
            "champion_cup_odds": float(champ["Stanley Cup"].iat[0]) if len(champ) else np.nan,
            "champion_cup_rank": int((g["Stanley Cup"] > champ["Stanley Cup"].iat[0]).sum() + 1) if len(champ) else np.nan,
        })
    return pd.DataFrame(rows)


__all__ = ["CHECKPOINTS", "checkpoint_dates", "in_playoff_spot", "run_checkpoints", "summary", "season_label"]
