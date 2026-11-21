import pandas as pd
import streamlit as st

from data import (FAV_DIVISION, FAV_TEAM, MAX_SERIES, elo_chart, fav_index, group_filter, meta, season_data,
                  season_picker, short_date)
from nhl_elo.config import season_label

season = season_picker()
m = meta()
data = season_data(season)
ratings, tg = data["ratings"], data["team_games"]
current = season == m["current_season"]

st.title(f"NHL ELO Rankings — {season_label(season)}")
last = tg["Date"].max() if len(tg) else None
games = tg["game_id"].nunique()
st.caption(f"{'Through' if current else 'Final ·'} {short_date(last) if last is not None else '—'} · "
           f"{games} games · method `{m['method']}`")

divs = (ratings.groupby(["Conference", "Division"])["ELO"].agg(ELO="mean", StDev="std").round(0)
        .reset_index().sort_values("ELO", ascending=False))
leaders = ratings.sort_values("ELO", ascending=False).drop_duplicates("Division")

col1, col2, col3 = st.columns([3, 8, 5], gap="small")
with col1:
    st.subheader("Division ELO")
    st.dataframe(divs[["Division", "ELO", "StDev"]], hide_index=True,
                 column_config={"ELO": st.column_config.NumberColumn(format="%d"),
                                "StDev": st.column_config.NumberColumn(format="%d")})
    st.caption("Leaders: " + ", ".join(f"**{r.Abbrev}** ({r.Division})" for r in leaders.itertuples()))
with col2:
    st.subheader("Rating through the season")
    teams = ratings["Team"].tolist()
    fav_div = ratings.loc[ratings["Division"] == FAV_DIVISION, "Team"].tolist()[:MAX_SERIES]
    pick = st.multiselect("Teams", teams, default=fav_div or leaders["Team"].tolist(), max_selections=MAX_SERIES,
                          label_visibility="collapsed", placeholder="Pick up to 8 teams", key=f"chart_{season}")
    view = tg.loc[tg["Team"].isin(pick)]
    if len(view):
        elo_chart(view, color="Team", height=460, key="season_chart")
    else:
        st.write("No games yet for the selected teams.")
with col3:
    st.subheader("Complete Rankings")
    r = group_filter(ratings, key="rankings_group")
    cols = ["Rank", "Team", "Division", "ELO", "GP", "W", "L", "OTL", "PTS", "Season +/-"]
    if not current and r["PO W"].sum():
        cols += ["PO W", "PO L"]
    st.dataframe(r[cols], hide_index=True, height=520,
                 column_config={"Season +/-": st.column_config.NumberColumn(format="%+d")})

# ---- latest results: every game on the last day with a final score
st.divider()
gm = data["games"]
if len(gm):
    day = gm["Date"].max()
    night = gm.loc[gm["Date"] == day].copy()
    yesterday = (pd.Timestamp.now(tz="America/Los_Angeles") - pd.Timedelta(days=1)).date()
    st.header(f"{'Last night' if day.date() == yesterday else 'Latest results'}: {short_date(day)}")
    home_won = night["Home Score"] > night["Away Score"]
    night["Score"] = night["Away Score"].astype(str) + " – " + night["Home Score"].astype(str)
    night["Final"] = (night["Game Type"].map({"PO": "Playoffs"}).fillna("") + " "
                      + night["Outcome"].map({"OT": "OT", "SO": "SO"}).fillna("")).str.strip()
    night["Winner"] = night["Home"].where(home_won, night["Away"])
    night["Chance"] = (night["Home Win %"].where(home_won, 100 - night["Home Win %"]))
    night["Upset"] = (night["Chance"] < 50).map({True: "Upset", False: ""})
    night["Swing"] = night["Home Delta"].abs()
    w_post = night["Home ELO Post"].where(home_won, night["Away ELO Post"])
    night["Winner ELO"] = w_post.astype(int).astype(str) + " (+" + night["Swing"].map("{:.1f}".format) + ")"
    l_post = night["Away ELO Post"].where(home_won, night["Home ELO Post"])
    night["Loser ELO"] = l_post.astype(int).astype(str) + " (−" + night["Swing"].map("{:.1f}".format) + ")"
    # the favourite team's game first
    fav_name = ratings.loc[ratings["Abbrev"] == FAV_TEAM, "Team"]
    if len(fav_name):
        night = night.assign(_fav=night["Home"].eq(fav_name.iat[0]) | night["Away"].eq(fav_name.iat[0]))             .sort_values(["_fav", "game_id"], ascending=[False, True])
    st.dataframe(
        night[["Away", "Score", "Home", "Final", "Winner", "Chance", "Upset", "Winner ELO", "Loser ELO"]],
        hide_index=True, height=min(36 * (len(night) + 1) + 3, 600),
        column_config={
            "Score": st.column_config.TextColumn("Score", help="Away – Home"),
            "Final": st.column_config.TextColumn("", help="OT = overtime, SO = shootout, Playoffs = playoff game"),
            "Chance": st.column_config.NumberColumn("Winner's chance", format="%.0f%%",
                                                    help="The winner's pre-game win chance"),
            "Upset": st.column_config.TextColumn("", help="The winner was the underdog"),
            "Winner ELO": st.column_config.TextColumn("Winner's ELO", help="After the game (change)"),
            "Loser ELO": st.column_config.TextColumn("Loser's ELO", help="After the game (change)"),
        },
    )
    n_up = int((night["Chance"] < 50).sum())
    st.caption(f"{len(night)} game{'s' if len(night) != 1 else ''}, {n_up} won by the underdog. "
               "Ratings move most when the underdog wins in regulation, or the margin is big; "
               "overtime and shootout wins count for less.")

st.divider()
st.header("Game log")
team = st.selectbox("Team", teams, index=fav_index(teams, ratings["Abbrev"].tolist()), key=f"log_{season}")
f = tg.loc[tg["Team"] == team].sort_values("Date", ascending=False)
st.dataframe(
    f[["Date", "Game Type", "Side", "Opponent", "ELO", "Opp ELO", "Win %", "GF", "GA", "Result",
       "+/- ELO", "ELO Post Game"]],
    hide_index=True, height=420,
    column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                   "Win %": st.column_config.NumberColumn(format="%.0f%%"),
                   "+/- ELO": st.column_config.NumberColumn(format="%+.1f")},
)
