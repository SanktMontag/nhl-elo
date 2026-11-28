"""Shot models: one LightGBM model per strength state, retrained for every season on earlier seasons.

For target season S the training data is the TRAIN_SEASONS seasons before S, so every xG value is
out of sample (the model never saw that season). The first season (2009-10) has no earlier data and
is scored by a model trained on the seasons after it (flagged xg_future_trained).

Talent-free: no shooter, goalie or team identity. The xG of a shot is the chance an average shooter
scores against an average goalie from that situation.

A logistic regression on a handful of features is the baseline the GBM has to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ("EV", "PP", "SH", "EN")
TRAIN_SEASONS = 5

CATEGORICAL = ["shot_type", "prev_type", "prev_zone", "shooter_pos"]
NUMERIC = ["dist_adj", "angle", "x_adj", "y_abs", "off_wing", "prev_seconds", "prev_dist", "prev_same_team",
           "rebound", "angle_change", "rush", "off_turnover", "score_diff", "period", "period_seconds",
           "own_sk", "opp_sk", "own_net_empty", "overtime", "playoff"]
SCORE_FEATURES = ["score_diff"]

# The NHL has relabelled shot types over time (snap vs wrist from 2024-25; bat, poke, between-legs from
# 2022-23; tip-in vs deflected), so labels that have traded places are merged. A missing type is
# set to the most common one: in recent seasons it appears almost only on goals, which would leak.
SHOT_TYPE_MAP = {"wrist": "wrist_snap", "snap": "wrist_snap", "unknown": "wrist_snap",
                 "tip-in": "tip", "deflected": "tip", "bat": "tip",
                 "wrap-around": "wrap", "cradle": "wrap", "poke": "other", "between-legs": "other",
                 "slap": "slap", "backhand": "backhand"}

# Recording keeps changing (rebound-like follow-ups doubled from 2019-20 on, tracking coordinates
# from 2023-24), so recent seasons count more in each training window and the season is a feature:
# trees can't extrapolate, so a new season is treated like the latest season in the window.
RECENCY_DECAY = 0.7
SEASON_FEATURE = True

GBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, seed=7, num_threads=0)
GBM_ROUNDS = 3000
EARLY_STOP = 100


def features(shots: pd.DataFrame) -> pd.DataFrame:
    """Model matrix from the shot table (after rink.adjust and players.add_handedness)."""
    X = pd.DataFrame(index=shots.index)
    for c in CATEGORICAL:
        X[c] = shots[c].astype("string").fillna("unknown").astype("category") if c in shots else "unknown"
    X["shot_type"] = shots["shot_type"].astype("string").map(SHOT_TYPE_MAP).fillna("other").astype("category")
    X["dist_adj"] = shots["dist_adj"]
    X["angle"] = shots["angle"]
    X["x_adj"] = shots["x_adj"]
    X["y_abs"] = shots["y_adj"].abs()
    X["off_wing"] = shots["off_wing"].astype(float) if "off_wing" in shots else np.nan
    X["prev_seconds"] = shots["prev_seconds"].clip(upper=120)
    X["prev_dist"] = shots["prev_dist"]
    for c in ("prev_same_team", "rebound", "rush", "off_turnover", "own_net_empty", "overtime"):
        X[c] = shots[c].astype(float)
    X["angle_change"] = shots["angle_change"]
    X["score_diff"] = shots["score_diff"].clip(-3, 3)
    X["period"] = shots["period"].clip(upper=4)
    X["period_seconds"] = shots["period_seconds"]
    X["own_sk"] = shots["own_sk"]
    X["opp_sk"] = shots["opp_sk"]
    X["playoff"] = (shots["game_type"] == 3).astype(float)
    X["season_year"] = (shots["season"] // 10000).astype(float)
    return X


def _align_categories(X: pd.DataFrame, cats: dict[str, list]) -> pd.DataFrame:
    X = X.copy()
    for c, levels in cats.items():
        X[c] = pd.Categorical(X[c].astype("string").fillna("unknown"), categories=levels)
    return X


def fit_gbm(X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None, y_val=None, rounds: int | None = None,
            weight: np.ndarray | None = None):
    import lightgbm as lgb

    cats = {c: sorted(set(X[c].astype("string").fillna("unknown"))) for c in CATEGORICAL if c in X}
    X = _align_categories(X, cats)
    train = lgb.Dataset(X, y, weight=weight, categorical_feature=list(cats), free_raw_data=False)
    if rounds is None:
        Xv = _align_categories(X_val, cats)
        val = lgb.Dataset(Xv, y_val, reference=train, categorical_feature=list(cats))
        booster = lgb.train(GBM_PARAMS, train, GBM_ROUNDS, valid_sets=[val],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return booster, cats, booster.best_iteration
    booster = lgb.train(GBM_PARAMS, train, rounds)
    return booster, cats, rounds


def predict_gbm(model, X: pd.DataFrame) -> np.ndarray:
    booster, cats, _ = model
    return booster.predict(_align_categories(X, cats))


# ---------------------------------------------------------------- logistic baseline

def baseline_matrix(shots: pd.DataFrame) -> pd.DataFrame:
    d = shots["dist_adj"].clip(upper=100)
    X = pd.DataFrame({"dist": d, "log_dist": np.log1p(d), "angle": shots["angle"].clip(upper=120),
                      "rebound": shots["rebound"].astype(float), "rush": shots["rush"].astype(float)}, index=shots.index)
    X = X.join(pd.get_dummies(shots["shot_type"].astype("string").map(SHOT_TYPE_MAP).fillna("other"), prefix="st", dtype=float))
    X = X.join(pd.get_dummies(shots["state"].astype("string"), prefix="state", dtype=float))
    return X


def fit_baseline(X: pd.DataFrame, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X, y)
    return m, list(X.columns)


def predict_baseline(model, X: pd.DataFrame) -> np.ndarray:
    m, cols = model
    return m.predict_proba(X.reindex(columns=cols, fill_value=0.0))[:, 1]


# ---------------------------------------------------------------- rolling fit

def train_window(seasons: list[int], target: int, n: int = TRAIN_SEASONS) -> list[int]:
    before = [s for s in seasons if s < target][-n:]
    if before:
        return before
    return [s for s in seasons if s > target][:n]


def rolling_xg(shots: pd.DataFrame, drop: list[str] | None = None, states=STATES, with_baseline: bool = True,
               progress=None, decay: float = RECENCY_DECAY, season_feature: bool = SEASON_FEATURE,
               seasons_to_score: list[int] | None = None) -> pd.DataFrame:
    """xG for every unblocked shot, each season predicted by models trained on other (earlier) seasons.

    Returns shots (unblocked only) with xg, xg_base and xg_future_trained columns.
    drop: feature names to leave out (e.g. SCORE_FEATURES for the no-score-state test).
    """
    u = shots.loc[shots["unblocked"]].copy()
    X_all = features(u)
    drop = list(drop or []) + ([] if season_feature else ["season_year"])
    if drop:
        X_all = X_all.drop(columns=drop)
    y_all = u["goal"].to_numpy()
    u["xg"] = np.nan
    u["xg_base"] = np.nan
    seasons = sorted(u["season"].unique())
    first = seasons[0]
    for target in (seasons_to_score or seasons):
        win = train_window(seasons, target)
        newest = max(win)
        age = (newest // 10000 - u["season"] // 10000).clip(lower=0).to_numpy()
        w_all = decay ** age
        for st in states:
            tr = (u["season"].isin(win) & (u["state"] == st)).to_numpy()
            te = ((u["season"] == target) & (u["state"] == st)).to_numpy()
            if not te.any() or tr.sum() < 300:
                continue
            # early stopping on the most recent training season, then refit on the whole window
            last = max(win)
            fit_mask = tr & (u["season"] != last).to_numpy()
            val_mask = tr & (u["season"] == last).to_numpy()
            if fit_mask.sum() >= 500 and val_mask.sum() >= 200:
                _, _, best = fit_gbm(X_all[fit_mask], y_all[fit_mask], X_all[val_mask], y_all[val_mask],
                                     weight=w_all[fit_mask])
                rounds = max(50, int(best * 1.1))
            else:
                rounds = 300
            model = fit_gbm(X_all[tr], y_all[tr], rounds=rounds, weight=w_all[tr])
            u.loc[te, "xg"] = predict_gbm(model, X_all[te])
        if with_baseline:
            trm = u["season"].isin(win).to_numpy()
            tem = (u["season"] == target).to_numpy()
            Xb = baseline_matrix(u)
            bm = fit_baseline(Xb[trm], y_all[trm])
            u.loc[tem, "xg_base"] = predict_baseline(bm, Xb[tem])
        if progress:
            progress(target)
    u["xg_future_trained"] = u["season"] == first
    return u


def flurry_adjust(u: pd.DataFrame, col: str = "xg") -> pd.Series:
    """MoneyPuck-style flurry adjustment: within a flurry, shot n is worth xg_n × Π_{i<n}(1 − xg_i),
    so a flurry is worth at most one goal (its total is the chance that at least one shot scores)."""
    s = u.sort_values(["game_id", "flurry_id", "t", "event_id"])
    keep = 1 - s[col]
    before = keep.groupby([s["game_id"], s["flurry_id"]]).cumprod() / keep
    return (s[col] * before).reindex(u.index)


# ---------------------------------------------------------------- scoring

def shot_scores(u: pd.DataFrame, col: str) -> dict:
    from sklearn.metrics import roc_auc_score

    d = u.dropna(subset=[col])
    p = d[col].clip(1e-6, 1 - 1e-6).to_numpy()
    y = d["goal"].to_numpy()
    return {"shots": int(len(d)), "goals": int(y.sum()), "xg_sum": float(p.sum()),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((y - p) ** 2)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan}

