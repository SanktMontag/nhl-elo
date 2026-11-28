"""`nhl` command line."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import load_settings, season_label


def _print(df: pd.DataFrame, n: int | None = None) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 500):
        print((df.head(n) if n else df).to_string(index=False))


def cmd_fetch(a) -> int:
    from .fetch import fetch

    rep = fetch(refresh_all=a.refresh_all, offline=a.offline)
    print(f"Seasons {season_label(rep.seasons[0])} to {season_label(rep.seasons[-1])}: "
          f"{rep.games_played} games played, {rep.games_upcoming} upcoming ({rep.requests} API requests).")
    for w in rep.warnings:
        print(f"  warning: {w}")
    if a.run:
        return cmd_run(a)
    return 0


def cmd_run(a) -> int:
    from .engine import run

    method = getattr(a, "method", None) or load_settings().default_method
    res = run(method)
    m = res.meta
    print(f"{method}: {m['games_played']} games through {m['last_game']} "
          f"({m['current_season_label']}: {m['games_played_current']} played, {m['games_upcoming']} upcoming).")
    sc = m["score"]
    print(f"Prediction quality since {season_label(m['score_from'])}: log-loss {sc['logloss']:.4f}, "
          f"brier {sc['brier']:.4f}, picked {sc['pick_pct']:.1%} of winners.")
    print(f"Wrote {res.out_dir}\n")
    _print(res.ratings[["Rank", "Team", "Division", "ELO", "GP", "W", "L", "OTL", "PTS", "Season +/-"]], a.top)
    return 0


def cmd_update(a) -> int:
    a.refresh_all, a.offline, a.run = False, False, True
    rc = cmd_fetch(a)
    if rc == 0 and not a.no_sim:
        print()
        a.sims = a.seed = None
        a.cold = a.backfill = a.backtest = a.checkpoints = a.force = False
        rc = cmd_simulate(a)
    return rc


def _sim_table(t: pd.DataFrame, n: int | None) -> pd.DataFrame:
    view = t[["Team", "Division", "PTS", "Proj PTS", "PTS p10", "PTS p90", "Playoffs", "Div Title",
              "Conf Final", "Stanley Cup"]].copy()
    for c in ("Playoffs", "Div Title", "Conf Final", "Stanley Cup"):
        view[c] = (view[c] * 100).map(lambda x: f"{x:5.1f}%")
    return view.head(n) if n else view


def _checkpoints(sim, sims=None, force=False, quiet=False) -> None:
    from . import checkpoints as C

    df = C.run_checkpoints(sim, sims=sims, force=force,
                           progress=lambda i, t, s: print(f"\r  checkpoints {i}/{t} {season_label(s)}", end="",
                                                          file=sys.stderr))
    if df.empty:
        return
    sm = C.summary(df)
    if quiet or sm.empty:
        print(f"\nCheckpoints: {df.groupby(['season', 'checkpoint']).ngroups} snapshots in "
              f"output/{sim.method.name}/checkpoints.csv", file=sys.stderr)
        return
    print(f"\n\nHow right were the playoff predictions? ({sm['season'].nunique()} finished seasons, "
          "playoff teams called right out of 16)")
    by = sm.groupby("checkpoint", sort=False).agg(
        model=("model_right", "mean"), standings=("standings_right", "mean"), model_brier=("model_brier", "mean"),
        standings_brier=("standings_brier", "mean"), champ_odds=("champion_cup_odds", "mean"),
        champ_rank=("champion_cup_rank", "median"))
    by = by.reindex([c for c in C.CHECKPOINTS if c in by.index])
    print(by.round(3).to_string())


def cmd_simulate(a) -> int:
    from . import simulate as S

    sim = S.Simulator(a.method)
    if a.checkpoints:
        _checkpoints(sim, a.sims, a.force)
        return 0
    if a.backtest:
        mmdd = a.date or "01-01"
        print(f"Simulating past seasons from {mmdd} (only earlier games used)...", file=sys.stderr)
        df, summary = S.backtest(sim, mmdd, a.sims, hot=False if a.cold else None,
                                 progress=lambda i, t, s: print(f"\r  {i}/{t} {season_label(s)}", end="",
                                                                file=sys.stderr))
        print(file=sys.stderr)
        print(f"{summary['seasons']} seasons, {summary['teams']} team-seasons, {summary['sims']} runs each, "
              f"as of {mmdd}:")
        print(f"  Playoffs   Brier {summary['playoffs_brier']:.4f}  "
              f"(no model, each team at its season's playoff rate: {summary['playoffs_brier_no_model']:.4f})")
        print(f"  Division   Brier {summary['division_brier']:.4f}")
        print(f"  Eventual Cup winners were given {summary['cup_winner_avg_prob']:.1%} on average "
              f"(equal odds would be about {1 / (summary['teams'] / summary['seasons']):.1%})")
        print("\nPlayoff calibration: predicted vs actual share that made it")
        print(S.calibration_table(df).to_string())
        return 0

    if a.backfill:
        dates = S.backfill_dates(sim)
        print(f"Backfilling {len(dates)} weekly snapshots of {season_label(sim.current)}...", file=sys.stderr)
        for i, d in enumerate(dates, 1):
            res = sim.run(as_of=d, sims=a.sims, seed=a.seed, hot=False if a.cold else None)
            S.write(res, sim.method.name)
            print(f"\r  {i}/{len(dates)} through {res.meta['through']}", end="", file=sys.stderr)
        print(file=sys.stderr)

    res = sim.run(sims=a.sims, seed=a.seed, hot=False if a.cold else None)
    out = S.write(res, sim.method.name)
    _checkpoints(sim, quiet=True)   # adds checkpoints that have passed; finished seasons are reused
    m = res.meta
    print(f"{m['sims']:,} simulations of {season_label(m['season'])} through {m['through']} "
          f"({m['remaining_games']} regular-season games left, {'hot' if m['hot'] else 'cold'} ratings). "
          f"Median playoff cut: " + ", ".join(f"{c} {v:.0f} pts" for c, v in m["playoff_cut"].items()) + ".")
    print(f"Wrote {out}\n")
    _print(_sim_table(res.table, a.top))
    return 0


def cmd_backtest(a) -> int:
    from .backtest import calibration, evaluate, grid_search, parse_grid, per_season

    seasons = (a.from_season, a.to_season or 99999999) if a.from_season else None
    if a.grid:
        grid = parse_grid(a.grid)
        n = 1
        for v in grid.values():
            n *= len(v)
        print(f"{a.method[0]}: {n} combinations...", file=sys.stderr)
        res = grid_search(a.method[0], grid, seasons,
                          progress=lambda i, t: print(f"\r  {i}/{t}", end="", file=sys.stderr))
        print(file=sys.stderr)
        _print(res.round(5), a.top)
        return 0
    rows, last = [], None
    for m in a.method:
        sc, scored = evaluate(m, seasons=seasons)
        rows.append({"method": m, **sc})
        last = scored
    _print(pd.DataFrame(rows).round(5))
    if a.seasons:
        print("\nBy season (" + a.method[-1] + "):")
        _print(per_season(last).round(4))
    print("\nCalibration (" + a.method[-1] + "): favourite's predicted vs actual win rate")
    print(calibration(last).to_string())
    return 0


def cmd_xg(a) -> int:
    from .xg import build as B
    from .xg.evaluate import predictive_table
    from .xg.pbp import fetch_pbp

    if a.action in ("fetch", "all"):
        res = fetch_pbp(workers=a.workers, progress=lambda i, n: print(f"  {i}/{n}", flush=True))
        print(f"play-by-play: {res['downloaded']} downloaded, {len(res['failed'])} failed of {res['todo']} new games")
    if a.action in ("build", "all"):
        B.build(workers=a.workers)
    if a.action in ("evaluate", "build", "all"):
        totals = pd.read_csv(B.XG_OUT / "game_xg.csv")
        for split in ("halves", "oddeven"):
            t = predictive_table(totals, split)
            t.to_csv(B.XG_OUT / f"predictive_{split}.csv", index=False)
            print(f"\nPredicting goal share in the other half ({split}):")
            _print(t)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nhl", description="NHL ELO ratings")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download games from the NHL API into data/")
    f.add_argument("--refresh-all", action="store_true", help="re-download finished seasons too")
    f.add_argument("--offline", action="store_true", help="use only cached API responses")
    f.add_argument("--run", action="store_true", help="recompute ratings afterwards")
    f.add_argument("--method", default=None)
    f.add_argument("--top", type=int, default=10)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="compute ratings into output/<method>/")
    r.add_argument("--method", default=None, help="v1 (default) or v2-dev")
    r.add_argument("--top", type=int, default=10, help="rows of the ratings table to print")
    r.set_defaults(func=cmd_run)

    u = sub.add_parser("update", help="fetch the latest games, recompute and simulate (the weekly command)")
    u.add_argument("--method", default=None)
    u.add_argument("--top", type=int, default=10)
    u.add_argument("--no-sim", action="store_true", help="skip the season simulation")
    u.set_defaults(func=cmd_update)

    s = sub.add_parser("simulate", help="Monte Carlo the rest of the season: playoff, division and Cup odds")
    s.add_argument("--method", default=None)
    s.add_argument("--sims", type=int, help="number of runs (default from settings.toml)")
    s.add_argument("--seed", type=int)
    s.add_argument("--cold", action="store_true", help="keep ratings fixed within each run")
    s.add_argument("--backfill", action="store_true",
                   help="also rebuild weekly odds for the current season so far (each from earlier games only)")
    s.add_argument("--backtest", action="store_true", help="score the simulation on past seasons")
    s.add_argument("--date", help="with --backtest: simulate each season from this MM-DD (default 01-01)")
    s.add_argument("--checkpoints", action="store_true",
                   help="how right were past predictions at Thanksgiving, Christmas, the All-Star break, ...")
    s.add_argument("--force", action="store_true", help="with --checkpoints: recompute every season")
    s.add_argument("--top", type=int, default=32)
    s.set_defaults(func=cmd_simulate)

    b = sub.add_parser("backtest", help="score predictions; compare methods or grid-search parameters")
    b.add_argument("--method", nargs="+", default=["v1"])
    b.add_argument("--grid", nargs="+", metavar="NAME=V1,V2",
                   help="e.g. k=6,8,10 home_ice=20,30 result.ot_win=0.6,1 season.offseason_regression=0.2,0.3")
    b.add_argument("--from", dest="from_season", type=int, help="first scored season, e.g. 20152016")
    b.add_argument("--to", dest="to_season", type=int, help="last scored season")
    b.add_argument("--seasons", action="store_true", help="also print scores by season")
    b.add_argument("--top", type=int, default=15)
    b.set_defaults(func=cmd_backtest)

    x = sub.add_parser("xg", help="expected goals: download play-by-play, build shot models, evaluate")
    x.add_argument("action", choices=["fetch", "build", "evaluate", "all"])
    x.add_argument("--workers", type=int, default=3)
    x.set_defaults(func=cmd_xg)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # team names like Montréal on Windows consoles
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
