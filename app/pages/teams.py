import streamlit as st

from data import FAV_TEAM, all_team_games, elo_chart, history, meta, season_data

m = meta()
st.title("Franchise History")
st.caption(f"Every game since {str(m['first_season'])[:4]}. Ratings follow the franchise through "
           "relocations and renames: Atlanta → Winnipeg, Phoenix/Arizona → Utah.")

h = history()
current = h.loc[h["season"] == m["current_season"]].sort_values("Franchise")
names = dict(zip(current["franchise"], current["Franchise"]))
r = season_data(m["current_season"])["ratings"]
fav = r.loc[r["Abbrev"] == FAV_TEAM, "franchise"].tolist()
ids = list(names)
fid = st.selectbox("Franchise", ids, index=ids.index(fav[0]) if fav and fav[0] in ids else 0, format_func=names.get)

mine = h.loc[h["franchise"] == fid].sort_values("season")
tg = all_team_games()
games = tg.loc[tg["franchise"] == fid].sort_values("Date")

c1, c2, c3, c4 = st.columns(4)
now = mine.iloc[-1]
c1.metric("Current ELO", int(now["End ELO"]), f"{int(now['End ELO'] - now['Start ELO']):+d} this season")
c2.metric("Current rank", f"#{int(now['Rank'])}")
best = mine.loc[mine["Peak ELO"].idxmax()]
c3.metric("All-time peak", int(best["Peak ELO"]), best["Season"], delta_color="off")
c4.metric("Best season finish", f"#{int(mine['Rank'].min())}",
          ", ".join(mine.loc[mine["Rank"] == mine["Rank"].min(), "Season"]), delta_color="off")

if len(games):
    elo_chart(games, height=380, key="franchise_chart")
    st.caption("ELO after each game. The dashed line is the league average (1500). The drop or jump "
               "between seasons is the off-season pull toward 1500.")

st.subheader("By season")
st.dataframe(
    mine.sort_values("season", ascending=False)[
        ["Season", "Team", "Rank", "Start ELO", "End ELO", "Peak ELO", "GP", "W", "L", "OTL", "PTS", "PO W", "PO L"]],
    hide_index=True,
)
with st.expander("All games"):
    st.dataframe(
        games.sort_values("Date", ascending=False)[
            ["Season", "Date", "Game Type", "Team", "Side", "Opponent", "GF", "GA", "Result", "Win %",
             "+/- ELO", "ELO Post Game"]],
        hide_index=True,
        column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                       "Win %": st.column_config.NumberColumn(format="%.0f%%")},
    )
