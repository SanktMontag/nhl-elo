# NHL ELO

ELO ratings for every NHL franchise, one continuous chain from 2005-06 (the first season with
shootouts and no ties) to today. Built on the same design as `../khl-elo`, but the games come straight
from the NHL's public API, so there are no exports to import and no team names to resolve.

**Contents**
1. [One-time setup](#1-one-time-setup)
2. [Updating](#2-updating)
3. [How the ratings work](#3-how-the-ratings-work)
4. [Once a season: check and tune the model](#4-once-a-season-check-and-tune-the-model)
5. [Testing method changes (v2-dev)](#5-testing-method-changes-v2-dev)
6. [The NHL API](#6-the-nhl-api)
7. [Command reference](#7-command-reference)
8. [Project layout](#8-project-layout)
9. [Expected goals (xG)](#9-expected-goals-xg)

---

## 1. One-time setup

```powershell
cd C:\Users\Daniel\Desktop\ELO-main_2\nhl-elo
py -3.14 -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest          # all tests should pass
```

Every session after that: `.venv\Scripts\Activate.ps1`, and `nhl` is available (`nhl -h`).

**Publishing:** as with the KHL site, Streamlit Community Cloud can run this from a GitHub repo, with
`streamlit.py` as the main file. The site only reads `output/`, so changes appear once you commit and push.

## 2. Updating

### Automatic (nightly)
A GitHub Action (`.github/workflows/nightly.yml`) runs `nhl update` every day at 11:23 UTC (3:23am Pacific in
winter, 4:23am in summer), with a backup run at 14:53 UTC (6:53am / 7:53am) because GitHub sometimes skips
scheduled runs. It commits the new results to `main`, and Streamlit Cloud then redeploys the site.
Nights with no new games make no commit. To run it immediately, use **Actions → Nightly update → Run
workflow** on GitHub, or run `gh workflow run nightly.yml`. GitHub emails you if a run fails.

Because the Action pushes to `main`, run **`git pull` before working locally**.

### Manual
```powershell
git pull
nhl update                       # download new results, recompute, simulate, print the top 10
streamlit run app\streamlit_app.py
git add -A; git commit -m "Update ratings through <date>"; git push
```

`nhl update` also re-simulates the rest of the season (see [Season simulation](#season-simulation)) and adds a
row per team to the odds history. It only re-downloads the current season (about four API requests). Finished seasons are cached
in `data/cache/` and never fetched again unless you run `nhl fetch --refresh-all`.

A new season needs no setup: once the NHL schedule for it is out, `nhl update` picks it up, applies the
off-season regression and starts predicting its games.

## 3. How the ratings work

Settings are in `config/methods/v1.toml`, with the evidence for each value.

- **Expected score (home team):** `E = 1 / (1 + 10^((R_away − R_home − 30) / 400))`. The 30-point home-ice
  bonus is about a 54% win chance between equal teams. Neutral-site games (the 2020 bubble playoffs) get
  no bonus.
- **Update:** `ΔHome = K × G × (S − E)` with K = 6. The away team gets −ΔHome, so the average stays 1500.
  - `S` is 1 for a regulation win, 0.75 for an overtime or shootout win, and the loser gets 1 − S.
  - `G` is the goal-difference multiplier: 1× for one goal, 1.25× for two, 1.75× for three, then +0.125×
    per goal, capped at 3.25×. A shootout counts as one goal, because the NHL's final score gives the
    shootout winner an extra goal.
- **Between seasons:** every rating is pulled 33% of the way back to 1500.
- **Franchises:** ratings belong to the NHL franchise id, so Atlanta → Winnipeg carries over on its own.
  The NHL treats Utah as a new franchise. `config/settings.toml` has a `[[continuation]]` that carries
  Arizona's rating into Utah anyway.
- **Expansion teams** (Vegas, Seattle) start at 1500.
- **Win %** on the site is `E`. OT and shootout wins count as wins.

**Accuracy** (26,211 games since 2006-07, each predicted from earlier games only): picks 57.6% of winners,
log-loss 0.6737 vs 0.6898 for always picking the home team. It is well calibrated: when the favourite
is given 62%, favourites win 62%. See the site's **What is ELO?** page or `output/v1/calibration.csv`.

### Season simulation

`nhl simulate` plays out the rest of the season 10,000 times (about 1.5 seconds) and writes playoff,
division, Presidents' Trophy, round-by-round and Stanley Cup odds, plus projected points, to
`output/v1/<season>/simulation*.csv`. The site shows them on the **Season Outlook** page.

- **Games:** each remaining game uses the same home win chance `E` as the Upcoming Games page. The OT/shootout
  rate (22.5%) and regulation goal margins are measured from games since 2015-16 (`[simulation]` in
  `config/settings.toml`).
- **Ratings** update within each run ("hot"). `--cold` keeps them fixed; the backtest found hot slightly better.
- **Standings tiebreakers:** points, regulation wins, regulation + OT wins, wins, goal differential. Head-to-head
  is left out.
- **Playoffs:** top 3 per division + 2 wild cards per conference. The wild cards are the next 2 best teams in
  the conference, from either division; the division winner with more points plays wild card 2, the other
  plays wild card 1. Series are best-of-7 with 2-2-1-1-1 home ice. Once real series exist, the real
  bracket and series scores are used.
- **Path to the playoffs:** the odds of each seed (1st, 2nd or 3rd in the division, wild card 1 or 2; they
  add up to the playoff chance) and the most likely first-round opponent. Shown under **Path to the
  playoffs** on the Season Outlook page.

```powershell
nhl simulate                         # current season (run automatically by nhl update)
nhl simulate --backfill              # also rebuild weekly snapshots so far, for the odds-over-time chart
nhl simulate --backtest --date 01-01 # check the odds on past seasons, using only games before that date
```

### Past predictions (checkpoints)

`nhl simulate --checkpoints` re-simulates every past season (2013-14 on, skipping 2019-20 and 2020-21) at
five points: opening night, US Thanksgiving, Christmas, the All-Star break (the Olympic or 4 Nations break
in seasons that had one) and the trade deadline. Each simulation uses only the games played before that
day. The results go to `output/v1/checkpoints.csv` and the site's **Past Predictions** page, next to what
happened and next to "the teams in a playoff spot that day make it".

`nhl update` keeps it current: finished seasons are reused, the current season gains each checkpoint as it
passes, and its results fill in when the Cup is awarded. Trade deadline dates are in `[trade_deadline]` in
`config/settings.toml`; **add each new season's date** once it's announced. After changing the model,
the file is rebuilt automatically (or force it with `--force`).

Average playoff teams called right, out of 16 (11 seasons):

| | Opening night | Thanksgiving | Christmas | All-Star break | Trade deadline |
|---|---|---|---|---|---|
| Simulation | 10.5 | 12.8 | 13.0 | 13.8 | 14.6 |
| Standings that day | – | 12.4 | 12.5 | 14.0 | 14.4 |

**Accuracy:** from January 1 across 11 past seasons (2013-14 on, skipping 2019-20 and 2020-21), playoff odds
score a Brier score of 0.120, against 0.250 with no model. From March 1 the score is 0.067. The calibration
bins match the actual rates within sampling noise. Simulated from the last day of the 2024-25 regular
season, the 16 qualifiers and all 8 first-round matchups matched the real ones.

## 4. Once a season: check and tune the model

```powershell
nhl backtest --seasons                                       # overall, by season, and calibration
nhl backtest --grid k=4,6,8 home_ice=20,30,40                 # try parameter combinations
nhl backtest --grid result.ot_win=0.5,0.75,1 season.offseason_regression=0.25,0.33,0.4
```

Grid keys are any setting in the method file; use a dot for nested tables (`gd.cap`, `result.so_win`).
Results are sorted by log-loss (lower is better). The target is always "did the home team win", so
settings that change `S` can be compared fairly. `--from 20152016` scores recent seasons only.

To change a setting: edit `config/methods/v1.toml`, note the evidence there, then `nhl run`, commit and push.

## 5. Testing method changes (v2-dev)

`v2-dev` is a sandbox (`config/methods/v2-dev.toml`, `src/nhl_elo/methods/v2_dev.py`). It currently tries
a log goal-difference multiplier with an autocorrelation correction. So far it ties v1 (0.67367 vs 0.67365).

```powershell
nhl run --method v2-dev                 # writes output/v2-dev/, never touches v1
nhl backtest --method v1 v2-dev
```

## 6. The NHL API

The API has no official documentation. The community reference is
<https://github.com/Zmalski/NHL-API-Reference>. This project uses:

| Data | Endpoint |
|---|---|
| Every game of a season, one call | `api.nhle.com/stats/rest/en/game?limit=-1&cayenneExp=season=20242025` |
| Team ids and franchise links | `api.nhle.com/stats/rest/en/team`, `.../franchise?include=teams` |
| Season dates and rule flags | `api-web.nhle.com/v1/standings-season` |
| Divisions (end-of-season standings) | `api-web.nhle.com/v1/standings/{YYYY-MM-DD}` or `/standings/now` |

Things to know:
- **Game types:** 1 preseason, 2 regular season, 3 playoffs, 19/20 4 Nations / All-Star. Only 2 and 3 are used.
- **Finished games** have `gameStateId` 7 (6 for preseason). Unplayed "if necessary" playoff games stay in
  the data as scheduled and are ignored.
- **OT and SO** aren't labelled in the stats API. In the regular season, `period` 4 is OT and 5 is a shootout.
  In the playoffs, period 5 is just double overtime.
- **`/now` endpoints redirect** (HTTP 307) to a dated URL.
- **Other useful endpoints** (`api-web.nhle.com/v1/`): `schedule/{date}` (a week, with
  `gameOutcome.lastPeriodType` and `neutralSite`), `score/{date}`, `club-schedule-season/{TEAM}/{season}`,
  `gamecenter/{id}/boxscore`, `playoff-bracket/{year}`, `season`.
- There is no documented rate limit. The client waits 0.5 s between requests and caches everything.

## 7. Command reference

| Command | Purpose |
|---|---|
| `nhl update [--no-sim]` | Fetch the latest games, recompute and simulate (the weekly command) |
| `nhl simulate [--sims N] [--seed S] [--cold] [--backfill]` | Season odds for the current season |
| `nhl simulate --backtest [--date MM-DD] [--cold]` | Score the simulation on past seasons |
| `nhl simulate --checkpoints [--force]` | Past predictions at Thanksgiving, Christmas, the All-Star break, ... |
| `nhl fetch [--refresh-all] [--offline] [--run]` | Download from the API into `data/` |
| `nhl run [--method v1\|v2-dev] [--top N]` | Recompute all seasons into `output/<method>/` |
| `nhl backtest [--method ...] [--grid NAME=V1,V2 ...] [--from S] [--to S] [--seasons]` | Score and tune the model |

## 8. Project layout

```
config/
  settings.toml           first season, game types, franchise continuations, neutral-site rules
  methods/*.toml          settings for v1 and v2-dev
data/
  games.csv               every game since 2005-06, played and upcoming (built by `nhl fetch`)
  teams.csv               NHL team id -> franchise -> franchise the rating is kept under
  divisions.csv           conference/division per franchise per season
  seasons.csv             season dates and game counts
  cache/                  raw API responses (not committed)
output/<method>/
  <season>/               ratings, team_games, games, predictions (current season only) .csv
                          simulation.csv (odds), simulation_points.csv (points distribution),
                          simulation_history.csv (odds by date), simulation_meta.json
  history.csv             each franchise's start/end/peak ELO and record per season
  accuracy.csv            prediction scores per season
  calibration.csv         predicted vs actual favourite win rate
  checkpoints.csv         past seasons' predictions at each checkpoint, with what happened
  run_meta.json           settings, counts and overall score of the last run
src/nhl_elo/              api.py, fetch.py, engine.py, backtest.py, simulate.py, checkpoints.py,
                          methods/, cli.py
src/nhl_elo/xg/           pbp.py, shots.py, players.py, rink.py, model.py, build.py, evaluate.py
data/players.csv          shooter handedness and position (for the off-wing feature)
output/xg/                game_xg.csv (team xG per game), shot_scores.csv, rink_bias.csv, predictive_*.csv
app/                      Streamlit site (streamlit_app.py + pages/)
tests/                    pytest suite
```

## 9. Expected goals (xG)

A talent-free shot model: the chance that an average shooter scores against an average goalie from a
given situation. Shooter and goalie talent go in a separate layer later (that's the player-rating input).

```powershell
.venv\Scripts\pip install -e .[xg]   # lightgbm, scikit-learn, pyarrow
nhl xg fetch      # play-by-play for every final game since 2009-10 (first run ~1 hour, then only new games)
nhl xg build      # shots -> rink adjustment -> rolling models -> output/xg/ (~10 minutes)
nhl xg evaluate   # how well xG predicts future goals, vs goals, shots and shot attempts
```

Not part of the nightly Action yet: the play-by-play cache (~270 MB) isn't committed.

**Data.** `gamecenter/{id}/play-by-play`, 22,313 games. Shot coordinates start in 2009-10. Unblocked
attempts only (blocked shots are recorded where the block happened). Shootouts and penalty shots are left out.

**Model.**
- One LightGBM model per strength state: EV (5v5, 4v4), PP (incl. extra attacker and 3-on-3 OT), SH, EN.
- Rolling: each season is predicted by a model trained on the 5 seasons before it, so every value is out of
  sample. 2009-10 has no earlier data and uses the 5 seasons after it.
- Features: distance and angle (rink-adjusted), x/|y|, shot type, off-wing, previous event (type, team,
  seconds, distance, zone), rebound + angle change, rush, off a turnover, score state, period, skaters.
- **Rink bias**: each rink's road-team shot distances are quantile-matched to road-team shots everywhere else
  (Schuckers & Curro). MSG records shots ~2.4 ft too close, Minnesota ~2.5 ft too far. The spread has shrunk
  from ~1.9 ft to ~0.7 ft since tracking arrived in 2023-24 (`rink_bias.csv`).
- **Flurry adjustment**: shot n in a quick sequence is worth xG × Π(1 − earlier xG), so a flurry is worth at
  most one goal. Team ratings should use `fxgf`/`fxga`.

**Recording changes the model has to live with.** Rebound-like follow-ups doubled from 2019-20 on and score
about half as often as they used to; "snap" took over from "wrist" in 2024-25; bat/poke/between-legs appear
in 2022-23; a missing shot type appears almost only on goals in recent seasons (a leak if used). Fixes: merged
shot-type groups, recent seasons weighted 0.7^age, and the season as a feature so a new season is treated like
the latest one. That cut the rebound over-prediction from 1.25-1.5× to 1.0-1.3× and season totals are within
±4% except 2019-20 (+10%, the first season after the change). An in-season recalibration was tried and
didn't help (log-loss 0.20872 vs 0.20879), so it isn't used.

**Accuracy** (2010-11 to now, 1.79M shots): log-loss 0.2089, AUC 0.787, total xG 1.01× goals.
Logistic baseline (distance, angle, type, rebound, rush, state): 0.2173, AUC 0.761.

**Does it tell us more than goals?** Each team's share in the first half of a season vs its goal share in the
second half (524 team-seasons):

| | GF% | SF% | CF% | FF% | xGF% | flurry xGF% |
|---|---|---|---|---|---|---|
| All situations | 0.523 | 0.484 | 0.472 | 0.490 | 0.543 | **0.546** |
| 5v5 | 0.414 | 0.416 | 0.412 | 0.417 | 0.450 | **0.454** |

Leaving score state out made no difference per shot but was worse here (0.534), so it stays in.

