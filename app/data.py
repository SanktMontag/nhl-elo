"""Cached loaders for the Streamlit app. Reads only the generated files in output/<method>/."""

from __future__ import annotations

import json
import math

import altair as alt
import pandas as pd
import streamlit as st

from nhl_elo.config import OUTPUT_DIR, load_settings, season_label

_SETTINGS = load_settings()
METHOD = _SETTINGS.default_method
OUT = OUTPUT_DIR / METHOD
FAV_TEAM = _SETTINGS.site.get("favorite_team", "")          # tricode, e.g. "SEA"
FAV_DIVISION = _SETTINGS.site.get("favorite_division", "")  # e.g. "Pacific"
MAX_SERIES = 8  # more lines than this on one chart can't be told apart
# Every rating since 2005 has been between 1334 and 1650, so this range fits all charts and keeps
# them comparable across seasons and teams. It widens (in steps of 50) if a rating ever goes beyond it.
ELO_RANGE = (1300, 1700)


def _mtime() -> float:
    p = OUT / "run_meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _meta(_key: float) -> dict:
    return json.loads((OUT / "run_meta.json").read_text())


def meta() -> dict:
    if not (OUT / "run_meta.json").exists():
        st.error("No ratings yet. Run `nhl update` first.")
        st.stop()
    return _meta(_mtime())


@st.cache_data(show_spinner=False)
def _season(season: int, _key: float) -> dict:
    d = OUT / str(season)
    tg = pd.read_csv(d / "team_games.csv", parse_dates=["Date"])
    preds = pd.read_csv(d / "predictions.csv", parse_dates=["Date"]) if (d / "predictions.csv").exists() \
        else pd.DataFrame()
    return {"ratings": pd.read_csv(d / "ratings.csv"), "team_games": tg, "predictions": preds,
            "games": pd.read_csv(d / "games.csv", parse_dates=["Date"])}


def season_data(season: int) -> dict:
    return _season(season, _mtime())


@st.cache_data(show_spinner=False)
def _table(name: str, _key: float) -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}.csv")


def history() -> pd.DataFrame:
    return _table("history", _mtime())


def accuracy() -> pd.DataFrame:
    return _table("accuracy", _mtime())


def calibration() -> pd.DataFrame:
    return _table("calibration", _mtime())


# ---- season simulation (written by `nhl simulate`)

def _sim_mtime(season: int) -> float:
    p = OUT / str(season) / "simulation_meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _simulation(season: int, _key: float) -> dict | None:
    d = OUT / str(season)
    if not (d / "simulation.csv").exists():
        return None
    hist = d / "simulation_history.csv"
    return {
        "table": pd.read_csv(d / "simulation.csv"),
        "points": pd.read_csv(d / "simulation_points.csv"),
        "history": pd.read_csv(hist, parse_dates=["Date"]) if hist.exists() else pd.DataFrame(),
        "meta": json.loads((d / "simulation_meta.json").read_text()),
    }


def simulation(season: int) -> dict | None:
    return _simulation(season, _sim_mtime(season))


@st.cache_data(show_spinner=False)
def _checkpoints(_key: float) -> pd.DataFrame:
    return pd.read_csv(OUT / "checkpoints.csv")


def checkpoints() -> pd.DataFrame | None:
    """Past predictions at fixed points of each season (written by `nhl simulate --checkpoints`)."""
    p = OUT / "checkpoints.csv"
    return _checkpoints(p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def _all_games(_key: float) -> pd.DataFrame:
    frames = [pd.read_csv(OUT / str(s) / "team_games.csv", parse_dates=["Date"]).assign(Season=season_label(s))
              for s in _meta(_key)["seasons"] if (OUT / str(s) / "team_games.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def all_team_games() -> pd.DataFrame:
    return _all_games(_mtime())


def season_picker() -> int:
    """Sidebar season selector shared by all pages (newest first)."""
    seasons = sorted(meta()["seasons"], reverse=True)
    idx = st.session_state.get("season_idx", 0)
    choice = st.sidebar.selectbox("Season", range(len(seasons)), index=min(idx, len(seasons) - 1),
                                  format_func=lambda i: season_label(seasons[i]))
    st.session_state["season_idx"] = choice
    return seasons[choice]


def fav_index(options: list, values: list | None = None) -> int:
    """Index of the favourite team in `options` (matched on `values`, e.g. tricodes), else 0."""
    values = values if values is not None else options
    return values.index(FAV_TEAM) if FAV_TEAM in values else 0


def group_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """'All teams / conference / division' picker, starting on the favourite division. Returns the rows."""
    confs = sorted(df["Conference"].dropna().unique())
    divs = sorted(df["Division"].dropna().unique())
    options = ["All teams"] + [f"{c} Conference" for c in confs] + [f"{d} Division" for d in divs]
    start = f"{FAV_DIVISION} Division"
    choice = st.selectbox("Show", options, index=options.index(start) if start in options else 0, key=key,
                          label_visibility="collapsed")
    if choice.endswith(" Conference"):
        return df.loc[df["Conference"] == choice.removesuffix(" Conference")]
    if choice.endswith(" Division"):
        return df.loc[df["Division"] == choice.removesuffix(" Division")]
    return df


def elo_chart(df: pd.DataFrame, color: str | None = None, height: int = 420, key: str = "elo") -> None:
    """Line chart of 'ELO Post Game' over 'Date', with the 1500 average marked.

    The y-range starts at ELO_RANGE and can be narrowed with a slider (exact values), a "Fit to teams"
    toggle, or the mouse (scroll to zoom, drag to pan, double-click to reset to the slider's range).
    """
    y = "ELO Post Game"
    lo = min(ELO_RANGE[0], math.floor(df[y].min() / 50) * 50)
    hi = max(ELO_RANGE[1], math.ceil(df[y].max() / 50) * 50)

    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with right:
        fit = st.toggle("Fit to teams", key=f"{key}_fit", help="Zoom the y-axis to the teams shown")
    default = (max(lo, math.floor(df[y].min() / 10) * 10 - 10), min(hi, math.ceil(df[y].max() / 10) * 10 + 10)) \
        if fit else (lo, hi)
    with left:
        # A new key when "fit" flips, so the slider jumps to the new default.
        rng = st.slider("ELO range", lo, hi, default, step=5, key=f"{key}_range_{fit}",
                        help="Drag either end to zoom the y-axis. You can also scroll on the chart to zoom, "
                             "drag to pan, and double-click to reset.")
    if rng[1] - rng[0] < 10:
        rng = (rng[0], rng[0] + 10)
    tips = [alt.Tooltip("Date:T", format="%b %d, %Y")]
    if color:
        tips.append(alt.Tooltip(f"{color}:N"))
    tips += [alt.Tooltip(f"{y}:Q", title="ELO"), alt.Tooltip("Opponent:N"), alt.Tooltip("Result:N")]
    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])
    line = alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after", clip=True).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y(f"{y}:Q", title="ELO", scale=alt.Scale(domain=list(rng), zero=False, nice=False)),
        tooltip=tips,
        **({"color": alt.Color(f"{color}:N", title=None, legend=alt.Legend(orient="bottom"))} if color else {}),
    ).add_params(zoom)
    avg = alt.Chart(pd.DataFrame({"y": [1500]})).mark_rule(strokeDash=[4, 4], opacity=0.5, clip=True).encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="League average")])
    st.altair_chart((avg + line).properties(height=height), width="stretch")


def odds_chart(df: pd.DataFrame, col: str, height: int = 400) -> None:
    """Probability (0–1 column `col`) over 'Date' per 'Team', on a fixed 0–100% axis. Scroll to zoom,
    drag to pan, double-click to reset."""
    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])
    base = alt.Chart(df).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y(f"{col}:Q", title=col, scale=alt.Scale(domain=[0, 1], nice=False), axis=alt.Axis(format="%")),
        color=alt.Color("Team:N", title=None, legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("Date:T", format="%b %d, %Y"), "Team:N",
                 alt.Tooltip(f"{col}:Q", format=".1%"), alt.Tooltip("Proj PTS:Q", title="Projected points"),
                 alt.Tooltip("ELO:Q")],
    )
    chart = base.mark_line(strokeWidth=2, clip=True) + base.mark_point(size=60, filled=True, clip=True)
    st.altair_chart(chart.add_params(zoom).properties(height=height), width="stretch")


def points_histogram(points: pd.DataFrame, cut: float | None, current: int, height: int = 300) -> None:
    """Share of simulations ending on each point total, with the median playoff cut marked."""
    bars = alt.Chart(points).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, binSpacing=2).encode(
        x=alt.X("PTS:Q", title="Final points", bin=alt.Bin(step=2), scale=alt.Scale(zero=False)),
        y=alt.Y("sum(share):Q", title="Share of simulations", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("PTS:Q", bin=alt.Bin(step=2), title="Points"),
                 alt.Tooltip("sum(share):Q", format=".1%", title="Share")],
    )
    layers = [bars]
    marks = [{"x": current, "label": f"Now: {current}"}]
    if cut is not None:
        marks.append({"x": cut, "label": f"Typical playoff cut: {cut:.0f}"})
    rule = alt.Chart(pd.DataFrame(marks)).mark_rule(strokeDash=[4, 4]).encode(
        x="x:Q", tooltip=[alt.Tooltip("label:N", title="")])
    text = alt.Chart(pd.DataFrame(marks)).mark_text(align="left", dx=4, dy=-4, baseline="bottom").encode(
        x="x:Q", y=alt.value(12), text="label:N")
    layers += [rule, text]
    st.altair_chart(alt.layer(*layers).properties(height=height), width="stretch")


def short_date(d) -> str:
    d = pd.to_datetime(d)
    return f"{d:%b} {d.day}, {d.year}"
