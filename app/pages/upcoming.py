import pandas as pd
import streamlit as st

from data import FAV_TEAM, meta, season_data, short_date

m = meta()
st.title(f"Upcoming Games — {m['current_season_label']}")
preds = season_data(m["current_season"])["predictions"]
if preds.empty:
    st.write("No upcoming games in the latest schedule.")
    st.stop()

st.caption(f"Chances use ratings through {short_date(m['last_game'])} and include home ice. "
           "Overtime and shootout wins count as wins. They're estimates, not guarantees.")

c1, c2 = st.columns([2, 3])
with c1:
    first, last = preds["Date"].min().date(), preds["Date"].max().date()
    default_end = min(last, first + pd.Timedelta(days=6))
    rng = st.date_input("Dates", (first, default_end), min_value=first, max_value=last)
with c2:
    teams = sorted(set(preds["Away"]) | set(preds["Home"]))
    r = season_data(m["current_season"])["ratings"]
    fav = r.loc[r["Abbrev"] == FAV_TEAM, "Team"].tolist()
    pick = st.multiselect("Team", teams, default=[x for x in fav if x in teams], placeholder="All teams")

view = preds
if isinstance(rng, tuple) and len(rng) == 2:
    view = view.loc[view["Date"].dt.date.between(*rng)]
if pick:
    view = view.loc[view["Away"].isin(pick) | view["Home"].isin(pick)]

st.dataframe(
    view.drop(columns=["game_id"]),
    hide_index=True, height=min(38 * (len(view) + 1), 800),
    column_config={
        "Date": st.column_config.DateColumn(format="ddd MMM D"),
        "Away Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Home Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    },
)
