"""xG pipeline: shot parsing, rink adjustment, flurry adjustment."""

import math

import numpy as np
import pandas as pd
import pytest

from nhl_elo.xg import model as M
from nhl_elo.xg.rink import adjust
from nhl_elo.xg.shots import _strength, parse_game

HOME, AWAY = 10, 20


def _play(eid, typ, per, clock, details, code="1551", side=None):
    p = {"eventId": eid, "sortOrder": eid, "typeDescKey": typ, "timeInPeriod": clock, "situationCode": code,
         "periodDescriptor": {"number": per, "periodType": "REG" if per <= 3 else "OT"}, "details": details}
    if side:
        p["homeTeamDefendingSide"] = side
    return p


def _game(plays):
    roster = [{"playerId": 1, "teamId": HOME, "positionCode": "C"}, {"playerId": 2, "teamId": AWAY, "positionCode": "D"},
              {"playerId": 9, "teamId": AWAY, "positionCode": "G"}, {"playerId": 8, "teamId": HOME, "positionCode": "G"}]
    return {"id": 2023020001, "season": 20232024, "gameType": 2, "homeTeam": {"id": HOME}, "awayTeam": {"id": AWAY},
            "rosterSpots": roster, "plays": plays}


def test_orientation_from_defending_side_and_rebound():
    plays = [
        _play(1, "period-start", 1, "00:00", {}, side="left"),
        # home defends left, so home attacks +x: a home shot at x=79 is 10 ft out
        _play(2, "shot-on-goal", 1, "05:00", {"xCoord": 79, "yCoord": 0, "shootingPlayerId": 1, "shotType": "wrist",
                                              "eventOwnerTeamId": HOME}, side="left"),
        _play(3, "goal", 1, "05:02", {"xCoord": 85, "yCoord": 4, "scoringPlayerId": 1, "shotType": "tip-in",
                                      "eventOwnerTeamId": HOME}, side="left"),
        # away attacks -x in period 1: x=-69 is 20 ft out
        _play(4, "missed-shot", 1, "07:00", {"xCoord": -69, "yCoord": 0, "shootingPlayerId": 2, "shotType": "slap",
                                             "eventOwnerTeamId": AWAY}, side="left"),
    ]
    rows = parse_game(_game(plays))
    assert [r["event"] for r in rows] == ["shot-on-goal", "goal", "missed-shot"]
    assert rows[0]["dist"] == pytest.approx(10)
    assert rows[2]["dist"] == pytest.approx(20)
    assert rows[1]["rebound"] and not rows[0]["rebound"]
    assert rows[1]["flurry_id"] == rows[0]["flurry_id"]
    assert rows[1]["score_diff"] == 0 and rows[2]["score_diff"] == -1   # score before the shot


def test_orientation_inferred_without_defending_side():
    plays = [_play(i, "shot-on-goal", 1, f"0{i}:00", {"xCoord": -60 - i, "yCoord": 0, "shootingPlayerId": 1,
                                                      "eventOwnerTeamId": HOME}) for i in range(1, 4)]
    rows = parse_game(_game(plays))
    assert all(r["x"] > 0 for r in rows)


def test_strength_states():
    assert _strength("1551", True, 1, 2)[0] == "EV"
    assert _strength("1451", True, 1, 2)[0] == "PP"     # home 5 skaters vs away 4
    assert _strength("1451", False, 1, 2)[0] == "SH"
    assert _strength("0651", True, 3, 2)[0] == "EN"     # away goalie pulled, home shooting
    assert _strength("0651", False, 3, 2)[0] == "PP"    # away extra attacker vs a goalie
    assert _strength("1331", True, 4, 2)[0] == "PP"     # 3-on-3 regular-season OT
    assert _strength("1010", True, 5, 2) is None        # shootout / penalty shot


def test_flurry_caps_at_one_goal():
    u = pd.DataFrame({"game_id": [1, 1, 1], "flurry_id": [5, 5, 5], "t": [1, 2, 3], "event_id": [1, 2, 3],
                      "xg": [0.5, 0.5, 0.5]})
    f = M.flurry_adjust(u)
    assert list(f.round(3)) == [0.5, 0.25, 0.125]
    assert f.sum() == pytest.approx(1 - 0.5 ** 3)


def test_rink_adjust_removes_a_shift():
    rng = np.random.default_rng(0)
    rows = []
    for rink in range(1, 11):
        bias = 5.0 if rink == 1 else 0.0          # rink 1 records road shots 5 ft too far out
        d = rng.uniform(5, 60, 2000) + bias
        rows.append(pd.DataFrame({"season": 20232024, "team": 99, "opp": rink, "is_home": False, "unblocked": True,
                                  "state": "EV", "dist": d, "x": 89 - d, "y": 0.0}))
    s = adjust(pd.concat(rows, ignore_index=True))
    m = s.groupby("opp")["dist_adj"].mean()
    raw = s.groupby("opp")["dist"].mean()
    assert raw[1] - raw[2:].mean() == pytest.approx(5, abs=0.6)
    assert abs(m[1] - m[2:].mean()) < 1.0
    assert np.allclose(89 - s["x_adj"], s["dist_adj"])   # straight-on shots keep their angle


def test_train_window_is_earlier_seasons():
    seasons = [20092010, 20102011, 20112012, 20122013, 20132014, 20142015, 20152016]
    assert M.train_window(seasons, 20152016) == seasons[1:6]
    assert M.train_window(seasons, 20102011) == [20092010]
    assert M.train_window(seasons, 20092010) == seasons[1:6]   # first season: trained on later ones
