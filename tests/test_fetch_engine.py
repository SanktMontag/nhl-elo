import pandas as pd
import pytest

from nhl_elo.backtest import parse_grid
from nhl_elo.config import Settings
from nhl_elo.engine import replay, score
from nhl_elo.fetch import normalize_games, outcome
from nhl_elo.methods import get_method

SETTINGS = Settings(first_season=20052006, score_from=20062007, game_types=(2, 3), default_method="v1",
                    user_agent="test", delay=0, current_ttl_hours=1,
                    continuations=({"from": 28, "to": 40}, {"from": 40, "to": 41}),
                    neutral=({"season": 20192020, "game_type": 3},))

TEAMS = pd.DataFrame([
    {"team_id": 10, "tricode": "TOR", "name": "Toronto Maple Leafs", "franchise": 5, "rating_franchise": 5},
    {"team_id": 53, "tricode": "ARI", "name": "Arizona Coyotes", "franchise": 28, "rating_franchise": 40},
])


def raw(gid, season=20242025, gtype=2, state=7, period=3, away=10, home=53, vs=2, hs=3, sched=1):
    return {"id": gid, "season": season, "gameType": gtype, "gameStateId": state, "gameScheduleStateId": sched,
            "period": period, "visitingTeamId": away, "homeTeamId": home, "visitingScore": vs, "homeScore": hs,
            "gameDate": "2024-10-10", "easternStartTime": "2024-10-10T19:00:00"}


def test_outcome():
    assert outcome(2, 3, 1, 2) == "REG"
    assert outcome(2, 4, 1, 2) == "OT"
    assert outcome(2, 5, 2, 3) == "SO"
    assert outcome(3, 5, 2, 3) == "OT"   # playoff double overtime, not a shootout
    assert outcome(2, 5, 2, 2) == "T"


def test_continuation_is_transitive():
    assert SETTINGS.continuation_map() == {28: 41, 40: 41}


def test_normalize():
    rows = [raw(1), raw(2, gtype=1), raw(3, state=1), raw(4, period=5), raw(5, state=1, sched=3),
            raw(6, season=20192020, gtype=3, period=5), raw(7, away=999), raw(8, state=4)]
    df, warnings = normalize_games(rows, TEAMS, SETTINGS)
    by = df.set_index("game_id")
    assert 2 not in by.index                    # preseason dropped
    assert 5 not in by.index                    # postponed dropped
    assert 7 not in by.index and len(warnings) == 1
    assert by.at[1, "state"] == "final" and by.at[1, "outcome"] == "REG"
    assert by.at[1, "home_franchise"] == 40     # Arizona rated under its continuation
    assert by.at[3, "state"] == "scheduled" and pd.isna(by.at[3, "home_score"])
    assert by.at[4, "outcome"] == "SO"
    assert by.at[6, "outcome"] == "OT" and bool(by.at[6, "neutral"])
    assert by.at[8, "state"] == "live"


def _games():
    rows = []
    gid = 0
    for season, start in ((20052006, "2005-10-05"), (20062007, "2006-10-05")):
        for i in range(10):
            gid += 1
            rows.append({"game_id": gid, "season": season, "game_type": 2,
                         "date": pd.Timestamp(start) + pd.Timedelta(days=i), "start_et": "", "away_team_id": 1,
                         "home_team_id": 2, "away_franchise": 1, "home_franchise": 2,
                         "away_score": 1, "home_score": 4, "state": "final", "outcome": "REG", "neutral": False})
    rows.append({**rows[-1], "game_id": 99, "away_franchise": 3, "away_team_id": 3, "state": "scheduled",
                 "away_score": pd.NA, "home_score": pd.NA, "outcome": ""})
    df = pd.DataFrame(rows)
    df[["away_score", "home_score"]] = df[["away_score", "home_score"]].astype("Int64")
    return df


def test_chain_regression_and_new_team():
    m = get_method("v1")
    rr = replay(_games(), m)
    end1 = rr.season_end[20052006]
    start2 = rr.season_start[20062007]
    assert end1[2] > 1500 > end1[1]
    assert start2[2] == pytest.approx(m.season_start(end1[2]))
    assert start2[3] == m.new_team_rating()              # appears only in an upcoming game
    assert sum(rr.final.values()) == pytest.approx(1500 * 3)  # zero-sum
    assert len(rr.team_games) == 2 * 20


def test_score_metrics():
    rr = replay(_games(), get_method("v1"), light=True)
    s = score(rr.scored)
    assert s["games"] == 20 and s["home_win_actual"] == 1.0
    assert 0 < s["logloss"] < 0.7


def test_cache_age_uses_stored_fetch_time(tmp_path, monkeypatch):
    import json
    import time

    from nhl_elo.api import Client

    c = Client(SETTINGS, cache_dir=tmp_path)
    calls = []
    monkeypatch.setattr(c, "_download", lambda url: calls.append(url) or {"n": len(calls)})
    assert c.get("u", "x.json", max_age_hours=1) == {"n": 1}
    assert c.get("u", "x.json", max_age_hours=1) == {"n": 1} and len(calls) == 1   # fresh: from cache
    # a restored cache file: new file time, but downloaded 2 hours ago -> must re-download
    (tmp_path / "x.json").write_text(json.dumps({"_fetched_at": time.time() - 7200, "payload": {"n": 1}}))
    assert c.get("u", "x.json", max_age_hours=1) == {"n": 2}
    assert c.get("u", "x.json", max_age_hours=None) == {"n": 2}                   # finished seasons never expire
    # old cache format (bare payload) still reads
    (tmp_path / "old.json").write_text(json.dumps({"data": [1]}))
    assert c.get("u", "old.json", max_age_hours=1) == {"data": [1]}


def test_parse_grid():
    assert parse_grid(["k=4,6", "result.ot_win=1"]) == {"k": [4.0, 6.0], "result.ot_win": [1.0]}
    with pytest.raises(ValueError):
        parse_grid(["k"])
