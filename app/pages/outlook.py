import streamlit as st

from data import (FAV_DIVISION, MAX_SERIES, fav_index, group_filter, meta, odds_chart, points_histogram,
                  short_date, simulation)
from nhl_elo.config import season_label

m = meta()
season = m["current_season"]
sim = simulation(season)
st.title(f"Season Outlook — {season_label(season)}")
if sim is None:
    st.write("No simulation yet. Run `nhl simulate`.")
    st.stop()

sm, t = sim["meta"], sim["table"]
st.caption(
    f"{sm['sims']:,} simulations of the rest of the season through {short_date(sm['through'])} "
    f"({sm['remaining_games']} regular-season games left), then the playoffs. "
    "Each team's odds are the share of simulations where it got there."
)

# ---- odds table
view = group_filter(t, key="outlook_group")
view = view.assign(Range=view["PTS p10"].astype(str) + "–" + view["PTS p90"].astype(str))
st.dataframe(
    view[["Team", "Division", "ELO", "PTS", "Proj PTS", "Range", "Playoffs", "Div Title", "Round 2",
          "Conf Final", "Cup Final", "Stanley Cup", "Presidents"]],
    hide_index=True, height=min(36 * (len(view) + 1) + 3, 1200),
    column_config={
        "PTS": st.column_config.NumberColumn("Points now"),
        "Proj PTS": st.column_config.NumberColumn("Projected", format="%.0f"),
        "Range": st.column_config.TextColumn("80% range", help="8 in 10 simulations ended in this range"),
        "Playoffs": st.column_config.ProgressColumn("Playoffs", format="percent", min_value=0, max_value=1),
        "Div Title": st.column_config.NumberColumn("Division", format="percent"),
        "Round 2": st.column_config.NumberColumn("Round 2", format="percent"),
        "Conf Final": st.column_config.NumberColumn("Conf. Final", format="percent"),
        "Cup Final": st.column_config.NumberColumn("Cup Final", format="percent"),
        "Stanley Cup": st.column_config.ProgressColumn("Stanley Cup", format="percent", min_value=0,
                                                       max_value=max(0.25, float(t["Stanley Cup"].max()))),
        "Presidents": st.column_config.NumberColumn("Presidents'", format="percent",
                                                    help="Most points in the league"),
    },
)

# ---- odds over time
st.subheader("How the odds have moved")
hist = sim["history"]
if hist.empty or hist["Date"].nunique() < 2:
    st.write("Odds history builds up as `nhl update` runs each week (or run `nhl simulate --backfill`).")
else:
    c1, c2 = st.columns([1, 3])
    with c1:
        labels = {"Div Title": "Division title", "Conf Final": "Conference final"}
        metric = st.radio("Odds", ["Playoffs", "Div Title", "Conf Final", "Stanley Cup"],
                          format_func=lambda x: labels.get(x, x))
    latest = t.sort_values(metric, ascending=False)
    default = latest.loc[latest["Division"] == FAV_DIVISION, "Team"].tolist()[:MAX_SERIES]
    if not default:
        default = latest["Team"].head(6).tolist()
    with c2:
        pick = st.multiselect("Teams", latest["Team"].tolist(), default=default, max_selections=MAX_SERIES,
                              key=f"odds_teams_{metric}", placeholder="Pick up to 8 teams")
    if pick:
        odds_chart(hist.loc[hist["Team"].isin(pick)], metric)
        st.caption("Each point is a simulation run with the games played up to that date. "
                   "Scroll to zoom, drag to pan, double-click to reset.")

# ---- points distribution
st.subheader("Final points")
by_name = t.sort_values("Team")
team = st.selectbox("Team", by_name["Team"].tolist(), index=fav_index(by_name["Team"].tolist(), by_name["Abbrev"].tolist()),
                    key="points_team")
row = t.loc[t["Team"] == team].iloc[0]
pts = sim["points"].loc[sim["points"]["franchise"] == row["franchise"]]
cut = sm.get("playoff_cut", {}).get(row["Conference"])
a, b = st.columns([3, 1])
with a:
    points_histogram(pts, cut, int(row["PTS"]))
with b:
    st.metric("Projected points", f"{row['Proj PTS']:.0f}", f"80% range {row['PTS p10']}–{row['PTS p90']}",
              delta_color="off")
    st.metric("Playoff chance", f"{row['Playoffs']:.0%}")
    if cut is not None:
        st.caption(f"In a typical simulation the last {row['Conference']} playoff team finishes with about "
                   f"**{cut:.0f} points**.")
