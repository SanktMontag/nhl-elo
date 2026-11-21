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
c1, c2 = st.columns([3, 2], vertical_alignment="center")
with c1:
    view = group_filter(t, key="outlook_group")
with c2:
    mode = st.segmented_control("View", ["Playoff rounds", "Path to the playoffs"], default="Playoff rounds",
                                key="outlook_view", label_visibility="collapsed") or "Playoff rounds"
view = view.assign(Range=view["PTS p10"].astype(str) + "–" + view["PTS p90"].astype(str))
pct = {"format": "percent"}
if mode == "Playoff rounds":
    # Playoff rounds first (each one includes the next), then the two regular-season titles.
    cols = ["Team", "Division", "ELO", "PTS", "Proj PTS", "Range", "Playoffs", "Round 2", "Conf Final",
            "Cup Final", "Stanley Cup", "Div Title", "Presidents"]
    config = {
        "Round 2": st.column_config.NumberColumn("Reach Round 2", help="Make the playoffs and win a "
                                                 "first-round series", **pct),
        "Conf Final": st.column_config.NumberColumn("Reach Conf. Final", help="Win two playoff series", **pct),
        "Cup Final": st.column_config.NumberColumn("Reach Cup Final", help="Win the conference", **pct),
        "Stanley Cup": st.column_config.ProgressColumn("Win Stanley Cup", format="percent", min_value=0,
                                                       max_value=max(0.25, float(t["Stanley Cup"].max()))),
        "Div Title": st.column_config.NumberColumn("Win division", help="Regular season: finish 1st in the "
                                                   "division", **pct),
        "Presidents": st.column_config.NumberColumn("Presidents' Trophy", help="Regular season: most points "
                                                    "in the league", **pct),
    }
else:
    # Older simulation files (before seeds were recorded) don't have these columns.
    if "WC2" not in view:
        st.info("Run `nhl simulate` to add the seeding odds.")
        st.stop()
    view = view.assign(Opp=view["R1 Opp"].fillna("") + view["R1 Opp %"].map(lambda x: f" ({x:.0%})" if x else ""))
    cols = ["Team", "Division", "PTS", "Proj PTS", "Range", "Playoffs", "Div Title", "Div 2nd", "Div 3rd",
            "WC1", "WC2", "Opp"]
    config = {
        "Div Title": st.column_config.NumberColumn("1st in div.", help="Division winner", **pct),
        "Div 2nd": st.column_config.NumberColumn("2nd in div.", **pct),
        "Div 3rd": st.column_config.NumberColumn("3rd in div.", **pct),
        "WC1": st.column_config.NumberColumn("Wild card 1", help="7th seed: plays the division winner with "
                                             "fewer points", **pct),
        "WC2": st.column_config.NumberColumn("Wild card 2", help="8th seed: plays the division winner with "
                                             "more points", **pct),
        "Opp": st.column_config.TextColumn("Likely 1st-round opponent",
                                           help="Most common opponent when the team makes it, and how often"),
    }
st.dataframe(
    view[cols], hide_index=True, height=min(36 * (len(view) + 1) + 3, 1200),
    column_config={
        "PTS": st.column_config.NumberColumn("Points now"),
        "Proj PTS": st.column_config.NumberColumn("Projected", format="%.0f"),
        "Range": st.column_config.TextColumn("80% range", help="8 in 10 simulations ended in this range"),
        "Playoffs": st.column_config.ProgressColumn("Playoffs", format="percent", min_value=0, max_value=1),
        **config,
    },
)
if mode != "Playoff rounds":
    st.caption("How each team gets in. The top 3 in each division qualify; the next 2 best teams in the "
               "conference, from either division, are the wild cards. The five columns add up to the "
               "playoff chance.")

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
