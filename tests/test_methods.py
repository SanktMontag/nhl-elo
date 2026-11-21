import pytest

from nhl_elo.config import deep_merge, load_method_config
from nhl_elo.methods import Game, get_method
from nhl_elo.methods.v1 import V1


@pytest.fixture
def v1():
    return get_method("v1")


@pytest.fixture(params=["v1", "v2-dev"])
def method(request):
    return get_method(request.param)


def test_zero_sum(method):
    for g in (Game(1, 4), Game(3, 2, "OT"), Game(2, 3, "SO"), Game(5, 0, playoff=True), Game(2, 1, neutral=True)):
        u = method.update(1580, 1420, g)
        assert u.home_delta + u.away_delta == pytest.approx(0)


def test_home_ice(v1):
    assert v1.home_expected(1500, 1500) > 0.5
    assert v1.home_expected(1500, 1500, neutral=True) == pytest.approx(0.5)
    # equal teams at a neutral site: a mirrored result is a mirrored change
    a = v1.update(1500, 1500, Game(1, 3, neutral=True)).home_delta
    b = v1.update(1500, 1500, Game(3, 1, neutral=True)).home_delta
    assert a == pytest.approx(-b)


def test_ot_so_scores():
    m = V1(deep_merge(load_method_config("v1"), {"result.ot_win": 0.6, "result.so_win": 0.55}))
    assert m.home_result(Game(1, 2)) == 1.0
    assert m.home_result(Game(1, 2, "OT")) == pytest.approx(0.6)
    assert m.home_result(Game(2, 1, "OT")) == pytest.approx(0.4)
    assert m.home_result(Game(3, 4, "SO")) == pytest.approx(0.55)
    assert m.home_result(Game(2, 2, "T")) == 0.5


def test_shootout_is_one_goal_game(v1):
    # the NHL adds a goal for the shootout winner; a 5-2 "shootout" score must still count as one goal
    assert v1.goal_diff(Game(2, 5, "SO")) == 1
    assert v1.goal_diff(Game(2, 5, "REG")) == 3


def test_gd_monotone_and_capped(v1):
    vals = [v1.gd_steps(g) for g in range(0, 30)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[1] == v1.gd_one and vals[2] == v1.gd_two
    assert max(vals) == v1.gd_cap


def test_underdog_gains_more(v1):
    fav = v1.update(1450, 1600, Game(2, 3)).home_delta
    dog = v1.update(1600, 1450, Game(2, 3)).home_delta
    assert 0 < fav < dog


def test_ot_win_gains_less_than_regulation(v1):
    assert v1.update(1500, 1500, Game(2, 3, "OT")).home_delta < v1.update(1500, 1500, Game(2, 3)).home_delta


def test_playoff_k():
    m = V1(deep_merge(load_method_config("v1"), {"playoff_k": 2.0}))
    assert m.update(1500, 1500, Game(1, 2, playoff=True)).home_delta == \
        pytest.approx(2 * m.update(1500, 1500, Game(1, 2)).home_delta)


def test_season_regression(v1):
    r = v1.season_start(1700)
    assert r == pytest.approx(1700 - v1.offseason_regression * 200)
    assert v1.season_start(1500) == 1500


def test_deep_merge_dotted_and_nested():
    base = {"k": 1, "gd": {"cap": 3, "step": 0.1}}
    out = deep_merge(base, {"gd.cap": 2, "k": 5})
    assert out == {"k": 5, "gd": {"cap": 2, "step": 0.1}}
    assert base["gd"]["cap"] == 3  # not mutated
