"""Monte Carlo simulation of the rest of a season: standings, playoff bracket and Stanley Cup.

Each run ("sim") plays every remaining game with the method's win probability, builds the standings
and playoff bracket, and plays the playoffs as best-of-7 series. Counting outcomes across many runs
gives the probabilities. All runs are simulated together: every game is one numpy step over arrays
shaped (sims × teams), so 10,000 runs of a full season take seconds.

"Hot" ratings (default) update after every simulated game, the same way real games update them, so a
team's strength can drift during a run as it does in reality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, Settings, load_settings, season_label
from .engine import replay
from .fetch import load_divisions, load_games, load_teams
from .methods import Method, get_method

HOME_GAMES = (0, 1, 4, 6)  # 2-2-1-1-1: games 1, 2, 5 and 7 at the higher seed
ODDS = ["Playoffs", "Div Title", "Presidents", "Round 2", "Conf Final", "Cup Final", "Stanley Cup", "Bottom 3"]
# How a team gets in: its playoff seed. 1st in division is "Div Title"; together with these they add up to
# "Playoffs". Seed codes used internally: 1-3 division place, 4 first wild card, 5 second wild card.
SEEDS = {2: "Div 2nd", 3: "Div 3rd", 4: "WC1", 5: "WC2"}


# ------------------------------------------------------------------ inputs

@dataclass
class Rates:
    p_extra: float          # regular-season games that go past regulation
    p_so: float             # of those, the share decided by a shootout
    p_extra_po: float       # playoff games that go to overtime
    margins: np.ndarray     # regulation winning margins 1..n
    margin_p: np.ndarray

    def as_dict(self) -> dict:
        return {"ot_or_so": round(self.p_extra, 4), "so_share": round(self.p_so, 4),
                "playoff_ot": round(self.p_extra_po, 4),
                "margins": {int(m): round(float(p), 4) for m, p in zip(self.margins, self.margin_p)}}


def measure_rates(games: pd.DataFrame, from_season: int) -> Rates:
    f = games.loc[(games["state"] == "final") & (games["season"] >= from_season)]
    reg, po = f.loc[f["game_type"] == 2], f.loc[f["game_type"] == 3]
    extra = reg["outcome"].isin(["OT", "SO"])
    m = (reg.loc[reg["outcome"] == "REG", "home_score"] - reg.loc[reg["outcome"] == "REG", "away_score"]).abs()
    counts = m.value_counts().sort_index()
    counts = counts.loc[counts.index >= 1]
    return Rates(p_extra=float(extra.mean()), p_so=float((reg["outcome"] == "SO").sum() / max(extra.sum(), 1)),
                 p_extra_po=float((po["outcome"] != "REG").mean()) if len(po) else 0.2,
                 margins=counts.index.to_numpy(dtype=int), margin_p=(counts / counts.sum()).to_numpy())


@dataclass
class SeasonState:
    season: int
    franchises: np.ndarray          # index -> franchise id
    names: list[str]
    abbrevs: list[str]
    conference: np.ndarray
    division: np.ndarray
    ratings: np.ndarray
    # actual regular-season record so far
    gp: np.ndarray
    pts: np.ndarray
    w: np.ndarray
    otl: np.ndarray
    rw: np.ndarray
    row: np.ndarray
    gd: np.ndarray
    remaining: pd.DataFrame         # unplayed regular-season games: a, h (indexes), neutral
    playoffs: pd.DataFrame          # playoff games (any state): rnd, ser, a, h, final, home_won
    through: str | None             # date of the last played game
    neutral_playoffs: bool = False

    @property
    def n(self) -> int:
        return len(self.franchises)


def season_state(games: pd.DataFrame, season: int, ratings: dict[int, float], divisions: pd.DataFrame,
                 teams: pd.DataFrame, settings: Settings | None = None) -> SeasonState:
    settings = settings or load_settings()
    sg = games.loc[games["season"] == season].sort_values(["date", "start_et", "game_id"], kind="stable")
    side = pd.concat([sg[["date", "away_franchise", "away_team_id"]].set_axis(["d", "f", "t"], axis=1),
                      sg[["date", "home_franchise", "home_team_id"]].set_axis(["d", "f", "t"], axis=1)])
    latest = side.sort_values("d").drop_duplicates("f", keep="last").set_index("f")["t"]
    fr = np.array(sorted(latest.index))
    idx = {f: i for i, f in enumerate(fr)}
    tn = teams.set_index("team_id")
    div = divisions.loc[divisions["season"] == season].set_index("franchise")
    missing = [f for f in fr if f not in div.index]
    if missing:
        raise ValueError(f"{season}: no conference/division for franchise(s) {missing}; run `nhl fetch`")

    n = len(fr)
    gp, pts, w, otl, rw, row, gd = (np.zeros(n, dtype=np.int64) for _ in range(7))
    reg = sg.loc[sg["game_type"] == 2]
    played = reg.loc[reg["state"] == "final"]
    for a_f, h_f, asc, hsc, oc in zip(played["away_franchise"], played["home_franchise"], played["away_score"],
                                      played["home_score"], played["outcome"]):
        a, h = idx[a_f], idx[h_f]
        win, lose = (h, a) if hsc > asc else (a, h)
        gp[a] += 1
        gp[h] += 1
        pts[win] += 2
        w[win] += 1
        if oc in ("OT", "SO"):
            pts[lose] += 1
            otl[lose] += 1
        if oc == "REG":
            rw[win] += 1
        if oc != "SO":
            row[win] += 1
        gd[h] += int(hsc) - int(asc)
        gd[a] += int(asc) - int(hsc)

    rem = reg.loc[reg["state"] != "final"]
    remaining = pd.DataFrame({"a": rem["away_franchise"].map(idx).to_numpy(dtype=int),
                              "h": rem["home_franchise"].map(idx).to_numpy(dtype=int),
                              "neutral": rem["neutral"].to_numpy(dtype=bool)})

    po = sg.loc[sg["game_type"] == 3]
    gid = po["game_id"].astype(str)
    playoffs = pd.DataFrame({
        "rnd": gid.str[6:8].astype(int).to_numpy(), "ser": gid.str[8].astype(int).to_numpy(),
        "a": po["away_franchise"].map(idx).to_numpy(dtype=int), "h": po["home_franchise"].map(idx).to_numpy(dtype=int),
        "final": (po["state"] == "final").to_numpy(),
        "home_won": (po["home_score"].fillna(0) > po["away_score"].fillna(0)).to_numpy(),
    })
    last = sg.loc[sg["state"] == "final", "date"]
    return SeasonState(
        season=season, franchises=fr,
        names=[tn.at[latest[f], "name"] for f in fr], abbrevs=[tn.at[latest[f], "tricode"] for f in fr],
        conference=div.loc[fr, "conference"].to_numpy(), division=div.loc[fr, "division"].to_numpy(),
        ratings=np.array([ratings[f] for f in fr], dtype=float),
        gp=gp, pts=pts, w=w, otl=otl, rw=rw, row=row, gd=gd, remaining=remaining, playoffs=playoffs,
        through=last.max().strftime("%Y-%m-%d") if len(last) else None,
        neutral_playoffs=settings.is_neutral(season, 3),
    )


# ------------------------------------------------------------------ simulation

@dataclass
class SimResult:
    table: pd.DataFrame             # one row per team: projections and odds
    points: pd.DataFrame            # franchise, PTS, share of sims
    cut: dict[str, float]           # conference -> median points of the last playoff team
    meta: dict = field(default_factory=dict)


def _play(R, method: Method, rows, a, h, neutral: bool, playoff: bool, rng, rates: Rates, hot: bool,
          margin=None):
    """One game in every sim listed in `rows` (a, h: team index per row, or scalars). Returns
    (home_won, extra, shootout, margin) arrays and updates R in place when hot."""
    n = len(rows)
    ra, rh = R[rows, a], R[rows, h]
    e = method.home_expected(ra, rh, neutral)
    hw = rng.random(n) < e
    extra = rng.random(n) < (rates.p_extra_po if playoff else rates.p_extra)
    so = extra & (rng.random(n) < rates.p_so) if not playoff else np.zeros(n, dtype=bool)
    if margin is None:
        margin = rng.choice(rates.margins, p=rates.margin_p, size=n)
    margin = np.where(extra, 1, margin)
    if hot:
        sw = np.where(so, method.so_win, np.where(extra, method.ot_win, 1.0))
        s = np.where(hw, sw, 1.0 - sw)
        edge = rh + (0.0 if neutral else method.home_ice) - ra
        k = method.k * (method.playoff_k if playoff else 1.0)
        delta = k * method.gd_multiplier_vec(margin, np.where(hw, edge, -edge)) * (s - e)
        R[rows, h] = rh + delta
        R[rows, a] = ra - delta
    return hw, extra, so, margin


def _series(R, method, hi, lo, whi, wlo, neutral, rng, rates, hot):
    """Best-of-7 between hi (home ice) and lo, per sim, from the given series score. Returns the winner."""
    S = len(hi)
    rows_all = np.arange(S)
    whi, wlo = whi.copy(), wlo.copy()
    for _ in range(7):
        active = (whi < 4) & (wlo < 4)
        if not active.any():
            break
        rows = rows_all[active]
        game_no = (whi + wlo)[active]
        hi_home = np.isin(game_no, HOME_GAMES)
        h = np.where(hi_home, hi[active], lo[active])
        a = np.where(hi_home, lo[active], hi[active])
        hw, *_ = _play(R, method, rows, a, h, neutral, True, rng, rates, hot)
        hi_won = hw == hi_home
        whi[active] += hi_won
        wlo[active] += ~hi_won
    return np.where(whi >= 4, hi, lo)


def _sort_key(pts, rw, row, w, gd, rng):
    """NHL standings order (points, regulation wins, regulation+OT wins, wins, goal differential), then
    random. Head-to-head is left out. One float per team per sim; higher ranks first."""
    key = (((pts * 100.0 + rw) * 100.0 + row) * 100.0 + w) * 1000.0 + (gd + 500)
    return key + rng.random(key.shape)


def _seed_wildcard(st: SeasonState, key, spots: int, wildcards: int):
    """Round-1 slots (S × 8 × 2: [higher-ranked side first]) in NHL series order, division winners, the
    last wild card per conference, and each team's seed (S × T: 0 out, 1-3 division place, 4+ wild card)."""
    S = key.shape[0]
    rows = np.arange(S)[:, None]
    seed = np.zeros(key.shape, dtype=np.int8)
    confs = sorted(set(st.conference))
    if len(confs) != 2:
        raise ValueError(f"wildcard format needs 2 conferences, found {confs}")
    slots, div_winners, cut_teams = [], [], {}
    for conf in confs:
        cm = np.where(st.conference == conf)[0]
        divs = sorted(set(st.division[cm]))
        if len(divs) != 2:
            raise ValueError(f"wildcard format needs 2 divisions in {conf}, found {divs}")
        q = np.zeros(key.shape, dtype=bool)
        tops = []
        for d in divs:
            m = np.where(st.division == d)[0]
            top = m[np.argsort(-key[:, m], axis=1)[:, :spots]]          # S × spots
            q[rows, top] = True
            seed[rows, top] = np.arange(1, spots + 1)
            tops.append(top)
        k2 = np.where(q[:, cm], -np.inf, key[:, cm])
        wc = cm[np.argsort(-k2, axis=1)[:, :wildcards]]                 # S × 2
        seed[rows, wc] = np.arange(spots + 1, spots + 1 + wildcards)
        a_better = key[rows[:, 0], tops[0][:, 0]] > key[rows[:, 0], tops[1][:, 0]]
        opp = [np.where(a_better, wc[:, 1], wc[:, 0]), np.where(a_better, wc[:, 0], wc[:, 1])]
        for top, o in zip(tops, opp):
            slots.append(np.stack([top[:, 0], o], axis=1))
            slots.append(np.stack([top[:, 1], top[:, 2]], axis=1))
            div_winners.append(top[:, 0])
        cut_teams[conf] = wc[:, -1]
    return np.stack(slots, axis=1), np.stack(div_winners, axis=1), cut_teams, seed


def simulate(st: SeasonState, method: Method, rates: Rates, sims: int = 10000, seed: int = 0,
             hot: bool = True, playoffs: dict | None = None) -> SimResult:
    playoffs = playoffs or load_settings().playoffs
    if playoffs.get("format", "wildcard") != "wildcard":
        raise ValueError(f"playoff format {playoffs.get('format')!r} is not supported")
    rng = np.random.default_rng(seed)
    S, T = sims, st.n
    rows = np.arange(S)
    R = np.tile(st.ratings, (S, 1))
    pts, w, otl, rw, row, gd = (np.tile(x, (S, 1)) for x in (st.pts, st.w, st.otl, st.rw, st.row, st.gd))
    gp = st.gp + np.bincount(np.r_[st.remaining["a"], st.remaining["h"]].astype(int), minlength=T)

    # ---- regular season
    rem = st.remaining
    if len(rem):
        margins = rng.choice(rates.margins, p=rates.margin_p, size=(len(rem), S))
        for i, (a, h, neu) in enumerate(zip(rem["a"], rem["h"], rem["neutral"])):
            hw, extra, so, margin = _play(R, method, rows, a, h, bool(neu), False, rng, rates, hot, margins[i])
            aw = ~hw
            pts[:, h] += np.where(hw, 2, extra)
            pts[:, a] += np.where(aw, 2, extra)
            w[:, h] += hw
            w[:, a] += aw
            otl[:, h] += aw & extra
            otl[:, a] += hw & extra
            rw[:, h] += hw & ~extra
            rw[:, a] += aw & ~extra
            row[:, h] += hw & ~so
            row[:, a] += aw & ~so
            signed = np.where(hw, margin, -margin)
            gd[:, h] += signed
            gd[:, a] -= signed
    key = _sort_key(pts, rw, row, w, gd, rng)

    # ---- bracket: from the real round-1 series once they're set, otherwise seeded from the standings
    slots, div_winners, cut_teams, seeding = _seed_wildcard(st, key, int(playoffs.get("division_spots", 3)),
                                                         int(playoffs.get("wildcards", 2)))
    r1 = st.playoffs.loc[st.playoffs["rnd"] == 1].drop_duplicates("ser").set_index("ser")
    if len(r1) == 8:
        real = np.array([[r1.at[s, "h"], r1.at[s, "a"]] for s in range(1, 9)])
        slots = np.broadcast_to(real, (S, 8, 2)).copy()

    # actual series scores: (round, team pair) -> wins per team
    actual = {}
    for (rnd, ser), g in st.playoffs.loc[st.playoffs["final"]].groupby(["rnd", "ser"]):
        winners = np.where(g["home_won"], g["h"], g["a"])
        pair = frozenset([int(g["a"].iat[0]), int(g["h"].iat[0])])
        actual[(int(rnd), pair)] = {t: int((winners == t).sum()) for t in pair}

    reached = []                                     # teams alive at the start of each round
    current = [slots[:, k, :] for k in range(8)]     # list of S × 2
    for rnd in range(1, 5):
        reached.append(np.stack([c for pair in current for c in pair.T], axis=1))
        winners = []
        for pair in current:
            t1, t2 = pair[:, 0], pair[:, 1]
            t1_hi = key[rows, t1] >= key[rows, t2]
            hi, lo = np.where(t1_hi, t1, t2), np.where(t1_hi, t2, t1)
            whi, wlo = np.zeros(S, dtype=int), np.zeros(S, dtype=int)
            for (r, pr), wins in actual.items():
                if r != rnd:
                    continue
                x, y = tuple(pr)
                m = ((hi == x) & (lo == y)) | ((hi == y) & (lo == x))
                if m.any():
                    whi[m] = np.where(hi[m] == x, wins[x], wins[y])
                    wlo[m] = np.where(hi[m] == x, wins[y], wins[x])
            winners.append(_series(R, method, hi, lo, whi, wlo, st.neutral_playoffs, rng, rates, hot))
        current = [np.stack([winners[i], winners[i + 1]], axis=1) for i in range(0, len(winners), 2)] \
            if len(winners) > 1 else []
        if len(winners) == 1:
            champion = winners[0]

    # ---- tally
    def share(team_idx: np.ndarray) -> np.ndarray:
        return np.bincount(team_idx.ravel(), minlength=T) / S

    order = np.argsort(-key, axis=1)
    odds = {
        "Playoffs": share(reached[0]), "Div Title": share(div_winners), "Presidents": share(order[:, 0]),
        "Round 2": share(reached[1]), "Conf Final": share(reached[2]), "Cup Final": share(reached[3]),
        "Stanley Cup": share(champion), "Bottom 3": share(order[:, -3:]),
    }
    seeds = {name: (seeding == code).mean(axis=0) for code, name in SEEDS.items()}

    # most likely first-round opponent, as a share of the team's playoff appearances
    opp = np.full((S, T), -1)
    for k in range(slots.shape[1]):
        opp[rows, slots[:, k, 0]] = slots[:, k, 1]
        opp[rows, slots[:, k, 1]] = slots[:, k, 0]
    r1_opp, r1_share = [], []
    for i in range(T):
        o = opp[:, i][opp[:, i] >= 0]
        if len(o):
            c = np.bincount(o, minlength=T)
            r1_opp.append(st.abbrevs[int(c.argmax())])
            r1_share.append(round(c.max() / len(o), 4))
        else:
            r1_opp.append("")
            r1_share.append(0.0)

    l_ = gp[None, :] - w - otl
    table = pd.DataFrame({
        "franchise": st.franchises, "Team": st.names, "Abbrev": st.abbrevs,
        "Conference": st.conference, "Division": st.division, "ELO": np.round(st.ratings).astype(int),
        "GP": st.gp, "PTS": st.pts,
        "Proj PTS": pts.mean(axis=0).round(1),
        "PTS p10": np.percentile(pts, 10, axis=0).round(0).astype(int),
        "PTS p90": np.percentile(pts, 90, axis=0).round(0).astype(int),
        "Proj W": w.mean(axis=0).round(1), "Proj L": l_.mean(axis=0).round(1), "Proj OTL": otl.mean(axis=0).round(1),
        **{k: v.round(4) for k, v in odds.items()},
        **{k: v.round(4) for k, v in seeds.items()},
        "R1 Opp": r1_opp, "R1 Opp %": r1_share,
    }).sort_values(["Stanley Cup", "Playoffs", "Proj PTS"], ascending=False, kind="stable").reset_index(drop=True)

    pts_rows = []
    for i in range(T):
        vals, counts = np.unique(pts[:, i], return_counts=True)
        pts_rows += [{"franchise": st.franchises[i], "PTS": int(v), "share": round(c / S, 5)}
                     for v, c in zip(vals, counts)]
    cut = {c: float(np.median(pts[rows, t])) for c, t in cut_teams.items()}
    meta = {"season": st.season, "through": st.through, "sims": S, "seed": seed, "hot": hot,
            "method": method.name, "params_hash": method.params_hash(), "rates": rates.as_dict(),
            "remaining_games": int(len(rem)), "bracket_from_real_series": bool(len(r1) == 8),
            "playoff_cut": cut}
    return SimResult(table, pd.DataFrame(pts_rows), cut, meta)


# ------------------------------------------------------------------ running it

def truncate(games: pd.DataFrame, season: int, as_of: str) -> pd.DataFrame:
    """Games as they looked on the morning of `as_of` (YYYY-MM-DD): later seasons dropped, later games of
    `season` unplayed, its playoff games removed."""
    g = games.loc[games["season"] <= season].copy()
    later = (g["season"] == season) & (g["date"] >= pd.Timestamp(as_of))
    g = g.loc[~(later & (g["game_type"] == 3))]
    later = (g["season"] == season) & (g["date"] >= pd.Timestamp(as_of))
    g.loc[later, ["away_score", "home_score", "period"]] = pd.NA
    g.loc[later, "state"] = "scheduled"
    g.loc[later, "outcome"] = ""
    return g


class Simulator:
    """Holds the loaded data so many simulations (backfill, backtest) don't reload it."""

    def __init__(self, method_name: str | None = None):
        self.settings = load_settings()
        self.method = get_method(method_name or self.settings.default_method)
        g = load_games()
        self.games = g.loc[g["season"] >= self.settings.first_season]
        self.teams = load_teams()
        self.divisions = load_divisions()
        self.rates = measure_rates(self.games, int(self.settings.simulation["rates_from"]))
        self.current = int(self.games["season"].max())

    def state(self, season: int | None = None, as_of: str | None = None) -> SeasonState:
        season = season or self.current
        g = truncate(self.games, season, as_of) if as_of else self.games.loc[self.games["season"] <= season]
        rr = replay(g, self.method, light=True)
        return season_state(g, season, rr.final, self.divisions, self.teams, self.settings)

    def run(self, season: int | None = None, as_of: str | None = None, sims: int | None = None,
            seed: int | None = None, hot: bool | None = None) -> SimResult:
        cfg = self.settings.simulation
        st = self.state(season, as_of)
        return simulate(st, self.method, self.rates, sims=int(sims or cfg["sims"]),
                        seed=int(cfg["seed"] if seed is None else seed),
                        hot=bool(cfg["hot"] if hot is None else hot), playoffs=self.settings.playoffs)


def write(res: SimResult, method_name: str) -> Path:
    d = OUTPUT_DIR / method_name / str(res.meta["season"])
    d.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(d / "simulation.csv", index=False)
    res.points.to_csv(d / "simulation_points.csv", index=False)
    (d / "simulation_meta.json").write_text(json.dumps(res.meta, indent=2, default=str))
    append_history(res, d / "simulation_history.csv")
    return d


def append_history(res: SimResult, path: Path) -> None:
    """One row per team per 'through' date; a rerun for the same date replaces that date's rows."""
    through = res.meta["through"] or "preseason"
    rows = res.table[["franchise", "Team", "Abbrev", "Conference", "Division", "ELO", "Proj PTS"] + ODDS] \
        .assign(Date=through)
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old.loc[old["Date"] != through], rows], ignore_index=True)
    rows.sort_values(["Date", "Stanley Cup"], ascending=[True, False]).to_csv(path, index=False)


def backfill_dates(sim: Simulator, every_days: int = 7) -> list[str]:
    """Weekly 'as of' dates through the current season so far (each simulated from earlier games only)."""
    sg = sim.games.loc[(sim.games["season"] == sim.current) & (sim.games["state"] == "final")]
    if sg.empty:
        return []
    first, last = sg["date"].min().date(), sg["date"].max().date()
    out, d = [], first + timedelta(days=every_days)
    while d <= last:
        out.append(d.isoformat())
        d += timedelta(days=every_days)
    return out


# ------------------------------------------------------------------ backtest

def backtest_seasons(sim: Simulator) -> list[int]:
    seasons = sorted(int(s) for s in sim.games["season"].unique())
    return [s for s in seasons if 20132014 <= s < sim.current and s not in (20192020, 20202021)]


def backtest(sim: Simulator, mmdd: str = "01-01", sims: int | None = None,
             seasons: list[int] | None = None, progress=None, hot: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """Simulate each past season from `mmdd` using only earlier games; compare with what happened."""
    sims = int(sims or sim.settings.simulation["backtest_sims"])
    seasons = seasons or backtest_seasons(sim)
    month = int(mmdd.split("-")[0])
    rows = []
    for i, s in enumerate(seasons, 1):
        year = int(str(s)[:4]) if month >= 9 else int(str(s)[4:])
        res = sim.run(s, as_of=f"{year}-{mmdd}", sims=sims, hot=hot)
        truth = _actual_outcomes(sim, s)
        t = res.table.merge(truth, on="franchise", how="left")
        rows.append(t.assign(season=s))
        if progress:
            progress(i, len(seasons), s)
    df = pd.concat(rows, ignore_index=True)

    def brier(p, y):
        return float(np.mean((df[p] - df[y]) ** 2))

    base_po = df.groupby("season")["made_playoffs"].transform("mean")
    summary = {
        "seasons": len(seasons), "teams": len(df), "as_of": mmdd, "sims": sims,
        "playoffs_brier": brier("Playoffs", "made_playoffs"),
        "playoffs_brier_no_model": float(np.mean((base_po - df["made_playoffs"]) ** 2)),
        "division_brier": brier("Div Title", "won_division"),
        "cup_logloss": float(-np.mean(np.log(np.clip(df.loc[df["won_cup"] == 1, "Stanley Cup"], 1e-4, 1)))),
        "cup_winner_avg_prob": float(df.loc[df["won_cup"] == 1, "Stanley Cup"].mean()),
    }
    return df, summary


def calibration_table(df: pd.DataFrame, col: str = "Playoffs", truth: str = "made_playoffs") -> pd.DataFrame:
    bins = pd.cut(df[col], [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0], include_lowest=True)
    return df.groupby(bins, observed=True).agg(teams=(truth, "size"), predicted=(col, "mean"),
                                               actual=(truth, "mean")).round(3)


def _actual_outcomes(sim: Simulator, season: int, st: SeasonState | None = None) -> pd.DataFrame:
    st = st or sim.state(season)
    key = (((st.pts * 100.0 + st.rw) * 100.0 + st.row) * 100.0 + st.w) * 1000.0 + (st.gd + 500)
    made = set(st.playoffs["a"]) | set(st.playoffs["h"])
    div_win = {int(np.where(st.division == d)[0][np.argmax(key[st.division == d])]) for d in set(st.division)}
    final = st.playoffs.loc[(st.playoffs["rnd"] == 4) & st.playoffs["final"]]
    winners = np.where(final["home_won"], final["h"], final["a"])
    champ = pd.Series(winners).value_counts().idxmax() if len(winners) else -1
    return pd.DataFrame({"franchise": st.franchises,
                         "made_playoffs": [int(i in made) for i in range(st.n)],
                         "won_division": [int(i in div_win) for i in range(st.n)],
                         "won_cup": [int(i == champ) for i in range(st.n)],
                         "final_pts": st.pts})


__all__ = ["Simulator", "simulate", "season_state", "measure_rates", "truncate", "write", "append_history",
           "backfill_dates", "backtest", "calibration_table", "ODDS", "SEEDS"]
