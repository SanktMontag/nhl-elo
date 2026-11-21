import altair as alt
import pandas as pd
import streamlit as st

from data import checkpoints, fav_index, group_filter, short_date
from nhl_elo.config import season_label

CHECKPOINTS = ["Opening night", "Thanksgiving", "Christmas", "All-Star break", "Trade deadline"]

st.title("How right were past predictions?")
df = checkpoints()
if df is None or df.empty:
    st.write("No checkpoints yet. Run `nhl simulate --checkpoints`.")
    st.stop()

st.caption(
    "Each past season was re-simulated at five points (opening night, US Thanksgiving, Christmas, the "
    "All-Star break and the trade deadline), using only the games played before that day, just as the Season "
    "Outlook page would have shown it then. Here are those predictions next to what happened. The "
    "comparison is \"the teams in a playoff spot that day make it\". Seasons start in 2013-14, the first with "
    "the wild card format; 2019-20 and 2020-21 had different playoff formats and are left out."
)

done = df.loc[df["made_playoffs"].notna()]
order = {c: i for i, c in enumerate(CHECKPOINTS)}

# ---- league-wide summary
if not done.empty:
    rows = []
    for (s, cp), g in done.groupby(["season", "checkpoint"]):
        made = g["made_playoffs"] == 1
        rows.append({"season": s, "checkpoint": cp, "Simulation": int((made & (g["picked"] == 1)).sum()),
                     "Standings that day": (made & (g["in_spot"] == 1)).sum() if g["in_spot"].notna().all()
                     else None})
    sm = pd.DataFrame(rows)
    n_seasons = sm["season"].nunique()
    avg = sm.groupby("checkpoint")[["Simulation", "Standings that day"]].mean().reindex(
        [c for c in CHECKPOINTS if c in set(sm["checkpoint"])])
    long = avg.reset_index().melt("checkpoint", var_name="Prediction", value_name="Right").dropna()

    st.subheader(f"Playoff teams called right, out of 16 ({n_seasons} seasons)")
    base = alt.Chart(long).encode(
        x=alt.X("checkpoint:N", sort=CHECKPOINTS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Right:Q", title="Average called right", scale=alt.Scale(domain=[8, 16], nice=False)),
        color=alt.Color("Prediction:N", title=None, legend=alt.Legend(orient="bottom"),
                        sort=["Simulation", "Standings that day"]),
        tooltip=[alt.Tooltip("checkpoint:N", title="When"), alt.Tooltip("Prediction:N"),
                 alt.Tooltip("Right:Q", title="Called right (avg of 16)", format=".1f")],
    )
    st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(size=80, filled=True)).properties(height=300),
                    width="stretch")
    st.caption("The simulation's picks are the 8 teams per conference with the best playoff chances. On "
               "opening night nobody has played, so there are no standings to compare with. An all-or-nothing "
               "pick can't show uncertainty; the simulation's percentages can, and the table further down "
               "shows how close each call was.")

# ---- one season, one checkpoint
st.subheader("A season, as it looked then")
seasons = sorted(df["season"].unique(), reverse=True)
finished = sorted(done["season"].unique(), reverse=True)
c1, c2 = st.columns(2)
with c1:
    season = st.selectbox("Season", seasons, index=seasons.index(finished[0]) if finished else 0,
                          format_func=lambda s: season_label(s) + ("" if s in finished else " (in progress)"))
sd = df.loc[df["season"] == season]
avail = sorted(sd["checkpoint"].unique(), key=order.get)
with c2:
    cp = st.selectbox("When", avail, index=avail.index("Thanksgiving") if "Thanksgiving" in avail else 0,
                      format_func=lambda c: f"{c} ({short_date(sd.loc[sd['checkpoint'] == c, 'as_of'].iat[0])})")
g = sd.loc[sd["checkpoint"] == cp].copy()
complete = g["made_playoffs"].notna().all()

if complete:
    made = g["made_playoffs"] == 1
    right = int((made & (g["picked"] == 1)).sum())
    spot = int((made & (g["in_spot"] == 1)).sum()) if g["in_spot"].notna().all() else None
    champ = g.loc[g["won_cup"] == 1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Playoff teams called right", f"{right} of 16",
              f"standings that day: {spot} of 16" if spot is not None else None, delta_color="off")
    if len(champ):
        c = champ.iloc[0]
        rank = int((g["Stanley Cup"] > c["Stanley Cup"]).sum() + 1)
        m2.metric("Eventual champion", c["Team"], f"{c['Stanley Cup']:.1%} Cup chance, #{rank} in the league",
                  delta_color="off")
    miss = (g["Playoffs"] - g["made_playoffs"]).abs()
    s0 = g.loc[miss.idxmax()]
    m3.metric("Biggest surprise", s0["Team"],
              f"{'made it' if s0['made_playoffs'] == 1 else 'missed'} from a {s0['Playoffs']:.0%} chance",
              delta_color="off")
else:
    st.info("This season isn't finished. The results fill in once the Stanley Cup is awarded.")

g["Through"] = g["GP"].astype(int).astype(str) + " GP, " + g["PTS"].astype(int).astype(str) + " pts"
g["In spot"] = g["in_spot"].map({1.0: "Yes", 0.0: "No"}).fillna("")
g["Made it"] = g["made_playoffs"].map({1.0: "Yes", 0.0: "No"}).fillna("")
g["Call"] = ""
if complete:
    g["Call"] = (g["picked"] == g["made_playoffs"]).map({True: "Right", False: "Wrong"})
view = group_filter(g, key="hindsight_group").sort_values(["Playoffs", "Proj PTS"], ascending=False)
cols = ["Team", "Division", "Through", "In spot", "Proj PTS", "Playoffs", "Made it", "Call", "final_pts",
        "Div Title", "Stanley Cup"]
st.dataframe(
    view[cols], hide_index=True, height=min(36 * (len(view) + 1) + 3, 1200),
    column_config={
        "Through": st.column_config.TextColumn("Record then"),
        "In spot": st.column_config.TextColumn("In a spot then", help="In a playoff position on that day's "
                                                                     "standings, by points percentage"),
        "Proj PTS": st.column_config.NumberColumn("Projected pts", format="%.0f"),
        "Playoffs": st.column_config.ProgressColumn("Playoff chance", format="percent", min_value=0, max_value=1),
        "Made it": st.column_config.TextColumn("Made playoffs"),
        "Call": st.column_config.TextColumn("Simulation's call", help="Right if the team was in its conference's "
                                            "top 8 for playoff chance and made it, or outside it and missed"),
        "final_pts": st.column_config.NumberColumn("Final pts", format="%d"),
        "Div Title": st.column_config.NumberColumn("Division chance", format="percent"),
        "Stanley Cup": st.column_config.NumberColumn("Cup chance", format="percent"),
    },
)

# ---- one team across seasons
st.subheader("One team, every season")
teams = df.sort_values("as_of").drop_duplicates("franchise", keep="last").sort_values("Team")
team = st.selectbox("Team", teams["Team"].tolist(),
                    index=fav_index(teams["Team"].tolist(), teams["Abbrev"].tolist()), key="hindsight_team")
fr = teams.loc[teams["Team"] == team, "franchise"].iat[0]
td = df.loc[df["franchise"] == fr]
grid = td.pivot_table(index="season", columns="checkpoint", values="Playoffs", aggfunc="first")
grid = grid.reindex(columns=[c for c in CHECKPOINTS if c in grid.columns]).sort_index(ascending=False)
result = td.drop_duplicates("season").set_index("season")
grid["Made playoffs"] = result["made_playoffs"].map({1.0: "Yes", 0.0: "No"}).reindex(grid.index).fillna("")
grid["Final pts"] = result["final_pts"].reindex(grid.index)
grid.index = [season_label(s) for s in grid.index]
grid.columns.name = None
st.dataframe(
    grid, height=min(36 * (len(grid) + 1) + 3, 800),
    column_config={"_index": st.column_config.TextColumn("Season"),
                   **{c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
                      for c in CHECKPOINTS},
                   "Final pts": st.column_config.NumberColumn(format="%d")},
)
st.caption(f"{team}'s playoff chance at each point of the season, and how it turned out.")
