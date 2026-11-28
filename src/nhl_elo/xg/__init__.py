"""Expected goals (xG): play-by-play download, shot features and the shot models.

    pbp.py       download and cache gamecenter play-by-play (data/cache/pbp/<season>/<game>.json.gz)
    players.py   shooter handedness and position (data/players.csv)
    shots.py     play-by-play -> one row per shot attempt with model features
    rink.py      arena scorekeeper bias adjustment for shot distance
    model.py     rolling, strength-state-specific LightGBM models + logistic baseline
"""
