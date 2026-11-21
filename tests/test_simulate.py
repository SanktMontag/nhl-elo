"""Season simulation tests on a synthetic 32-team league (no data files needed)."""

import numpy as np
import pandas as pd
import pytest

from nhl_elo.methods import get_method
from nhl_elo.simulate import Rates, SeasonState, simulate

PLAYOFFS = {"format": "wildcard", "division_spots": 3, "wildcards": 2}
RATES = Rates(p_extra=0.23, p_so=0.34, p_extra_po=0.23, margins=np.array([1, 2, 3, 4]),
              margin_p=np.array([0.25, 0.26, 0.3, 0.19]))
CONF = np.array(["Eastern"] * 16 + ["Western"] * 16)
DIV = np.array(["Atlantic"] * 8 + ["Metropolitan"] * 8 + ["Central"] * 8 + ["Pacific"] * 8)


def make_state(remaining_rounds=2, pts=None, playoffs=None, ratings=None) -> SeasonState:
    n = 32
    rng = np.random.default_rng(1)
    games = [(a, h) for _ in range(remaining_rounds) for a in range(n) for h in range(n) if a < h and rng.random() < 0.15]
    rem = pd.DataFrame({"a": [g[0] for g in games], "h": [g[1] for g in games], "neutral": False})
    z = np.zeros(n, dtype=np.int64)
    pts = np.array(pts if pts is not None else np.arange(n) % 7 * 3 + 40, dtype=np.int64)
    return SeasonState(
        season=20302031, franchises=np.arange(100, 100 + n), names=[f"T{i}" for i in range(n)],
        abbrevs=[f"T{i}" for i in range(n)], conference=CONF, division=DIV,
        ratings=np.array(ratings if ratings is not None else 1500 + (np.arange(n) % 9) * 10, dtype=float),
        gp=z + 40, pts=pts, w=pts // 2, otl=z, rw=pts // 3, row=pts // 2, gd=z,
        remaining=rem,
        playoffs=playoffs if playoffs is not None else
        pd.DataFrame(columns=["rnd", "ser", "a", "h", "final", "home_won"]),
        through="2031-01-01")


@pytest.fixture(scope="module")
def result():
    return simulate(make_state(), get_method("v1"), RATES, sims=2000, seed=7, playoffs=PLAYOFFS)


def test_totals_per_sim(result):
    t = result.table
    assert t["Playoffs"].sum() == pytest.approx(16)
    assert t["Round 2"].sum() == pytest.approx(8)
    assert t["Conf Final"].sum() == pytest.approx(4)
    assert t["Cup Final"].sum() == pytest.approx(2)
    assert t["Stanley Cup"].sum() == pytest.approx(1)
    assert t["Div Title"].sum() == pytest.approx(4)
    assert t["Presidents"].sum() == pytest.approx(1)
    assert t["Bottom 3"].sum() == pytest.approx(3)
    assert t.groupby("Conference")["Playoffs"].sum().tolist() == pytest.approx([8, 8])


def test_odds_are_monotone(result):
    t = result.table
    for a, b in [("Stanley Cup", "Cup Final"), ("Cup Final", "Conf Final"), ("Conf Final", "Round 2"),
                 ("Round 2", "Playoffs"), ("Div Title", "Playoffs")]:
        assert (t[a] <= t[b] + 1e-12).all()


def test_points_bounds(result):
    st = make_state()
    games_left = np.bincount(np.r_[st.remaining["a"], st.remaining["h"]], minlength=32)
    t = result.table.set_index("franchise").loc[st.franchises]
    # every game is worth at least 0 and at most 2 points to a team
    assert (t["PTS p10"].to_numpy() >= st.pts).all()
    assert (t["PTS p90"].to_numpy() <= st.pts + 2 * games_left).all()
    # points distribution is a full distribution per team
    assert result.points.groupby("franchise")["share"].sum().to_numpy() == pytest.approx(1, abs=1e-3)


def test_points_per_game_total():
    # one game, 1 sim at a time: total points awarded is 2 (regulation) or 3 (OT/SO)
    st = make_state(remaining_rounds=0)
    st.remaining = pd.DataFrame({"a": [0], "h": [1], "neutral": [False]})
    res = simulate(st, get_method("v1"), RATES, sims=4000, seed=3, playoffs=PLAYOFFS)
    t = res.table.set_index("franchise")
    added = (t.loc[[100, 101], "Proj PTS"] - st.pts[[0, 1]]).sum()
    assert 2.0 < added < 3.0
    assert added == pytest.approx(2 + RATES.p_extra, abs=0.03)


def test_same_seed_same_result():
    st = make_state(remaining_rounds=1)
    a = simulate(st, get_method("v1"), RATES, sims=500, seed=11, playoffs=PLAYOFFS).table
    b = simulate(st, get_method("v1"), RATES, sims=500, seed=11, playoffs=PLAYOFFS).table
    pd.testing.assert_frame_equal(a, b)


def test_finished_season_is_certain():
    pts = np.arange(32) * 2 + 50              # no ties in the standings
    st = make_state(remaining_rounds=0, pts=pts)
    res = simulate(st, get_method("v1"), RATES, sims=300, seed=1, playoffs=PLAYOFFS)
    assert set(res.table["Playoffs"].round(6)) <= {0.0, 1.0}
    made = set(res.table.loc[res.table["Playoffs"] == 1, "franchise"] - 100)
    for div in set(DIV):
        top3 = sorted(np.where(DIV == div)[0], key=lambda i: -pts[i])[:3]
        assert set(top3) <= made
    assert len(made) == 16


def _series_games(rnd, ser, home, away, home_wins, away_wins):
    rows = [{"rnd": rnd, "ser": ser, "a": away, "h": home, "final": True, "home_won": True}] * home_wins
    rows += [{"rnd": rnd, "ser": ser, "a": away, "h": home, "final": True, "home_won": False}] * away_wins
    return rows


def _real_bracket(scores):
    """8 real round-1 series between (2k, 2k+1); scores[k] = (higher-index team wins, lower-index wins)."""
    rows = []
    for k in range(8):
        hw, aw = scores.get(k, (0, 0))
        lo, hi = 2 * k + (0 if k < 4 else 8), 2 * k + 1 + (0 if k < 4 else 8)
        rows += _series_games(1, k + 1, hi, lo, hw, aw) or \
            [{"rnd": 1, "ser": k + 1, "a": lo, "h": hi, "final": False, "home_won": False}]
    return pd.DataFrame(rows)


def test_real_series_respected():
    # series 1: team 1 leads 4-0 (done); series 2: team 3 leads 3-0
    po = _real_bracket({0: (4, 0), 1: (3, 0)})
    st = make_state(remaining_rounds=0, playoffs=po)
    res = simulate(st, get_method("v1"), RATES, sims=3000, seed=5, playoffs=PLAYOFFS)
    t = res.table.set_index("franchise")
    assert t.at[101, "Round 2"] == 1.0 and t.at[100, "Round 2"] == 0.0    # finished series never replayed
    assert 0.8 < t.at[103, "Round 2"] < 1.0                                # 3-0 lead: usually, not always
    # the real round-1 teams are exactly the playoff teams
    assert set(t.index[t["Playoffs"] == 1] - 100) == set(po["a"]) | set(po["h"])
