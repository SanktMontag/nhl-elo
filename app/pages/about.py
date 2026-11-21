import streamlit as st

from data import METHOD, accuracy, calibration, meta
from nhl_elo.config import season_label
from nhl_elo.methods import Game, get_method

m = get_method(METHOD)
info = meta()
K, SCALE, HFA = m.k, int(m.scale), int(m.home_ice)

st.title("What is ELO?")
st.markdown(
    """
ELO is a rating system originally developed to rank chess players. It's named for its creator, Arpad Elo,
and has been adapted for many sports to rank teams.

### TL;DR

Every team has a rating, and every game is net zero. When a team wins, it takes some of the loser's
rating. Win as a favourite and you take a little; win as an underdog and you take a lot. Ratings average
1500.

### How it works

Each team enters a game with a rating. The **expected score** for the home team compares the two
ratings on a logistic curve, after giving the home team a bonus for home ice:
"""
)
st.latex(rf"E_{{home}} = \frac{{1}}{{1 + 10^{{(R_{{away}} - R_{{home}} - {HFA})/{SCALE}}}}}")
e_even = m.home_expected(1500, 1500)
st.markdown(
    f"""
* **Home ice is worth {HFA} points.** Two equal teams: the home team wins about **{e_even:.0%}** of the
  time, which is what NHL home teams have done since 2005. Neutral-site games (like the 2020 bubble
  playoffs) get no bonus.
* **Every {SCALE} points** of difference is a 10× difference in odds.

$E$ is also the win chance shown on the Upcoming Games page. After the game:
"""
)
st.latex(r"R' = R + K \cdot G \cdot (S - E)")
st.markdown(
    f"""
* **S** is the result: **1** for a regulation win, **{m.ot_win}** for an overtime win, **{m.so_win}** for a
  shootout win. The loser gets 1 − S (so an overtime loss still earns {1 - m.ot_win:.2f}).
* **K = {K:g}** is how fast ratings move. NHL rosters are stable and 82 games is a lot of evidence, so K is
  much smaller than for rec hockey.
* **G** is the goal-difference multiplier: 1× for one goal, {m.gd_two}× for two, {m.gd_three}× for three,
  then +{m.gd_step}× per extra goal (max {m.gd_cap}×). A shootout counts as a one-goal game, because
  the NHL adds a goal for the shootout winner.

The other team gets exactly the opposite change, so the league average never moves.
"""
)

st.markdown("### Example")
reg = m.update(1450, 1550, Game(2, 4)).home_delta
ot = m.update(1450, 1550, Game(2, 3, "OT")).home_delta
upset = m.update(1450, 1550, Game(4, 1)).away_delta
st.markdown(
    f"""
The home team is rated 1550 and the visitor 1450, so the home team's expected score is
**{m.home_expected(1450, 1550):.2f}**.

* Home wins 4–2 in regulation: home gains **{reg:+.1f}**.
* Home wins 3–2 in overtime: home gains **{ot:+.1f}**.
* Visitor wins 4–1: visitor gains **{upset:+.1f}**.
"""
)

st.markdown(
    f"""
### Between seasons

At the start of each season every rating is pulled **{m.offseason_regression:.0%}** of the way back to
1500, because rosters change over the summer. Expansion teams (Vegas, Seattle) start at
{m.new_team_offset + m.mean:.0f}. Ratings follow the franchise: the Winnipeg Jets carry on from the
Atlanta Thrashers and Utah carries on from Arizona.

The chain starts in {season_label(info['first_season'])} with every team at 1500. That season is a warm-up
and is left out of the accuracy numbers below.
"""
)

st.markdown(
    """
### Season Outlook: how the simulation works

The Season Outlook page plays out the rest of the season 10,000 times:

* Every remaining game is decided using the same win chances as the Upcoming Games page. About 22% of
  games go to overtime or a shootout, the real rate since 2015-16, so the loser picks up a point.
* Ratings keep updating within each simulated season, so a team can get hot or go cold.
* The standings decide the playoff teams: the top three in each division plus two wild cards per
  conference (the next two best teams, from either division). The division winner with more points
  plays the second wild card. Ties are broken by regulation wins, regulation-plus-OT wins, wins and goal
  differential.
* The playoffs are simulated game by game, as best-of-seven series with home ice to the team with
  more points. Once real series start, the real results are used.

A team's playoff chance is the share of those 10,000 seasons in which it made it. Tested on 11 past
seasons from January 1, using only the games played before then, the chances were honest: teams given
about 50% made it about 50% of the time.
"""
)

st.markdown("### How accurate is it?")
sc = info["score"]
c1, c2, c3 = st.columns(3)
c1.metric("Winners picked", f"{sc['pick_pct']:.1%}", help="The team with the higher win chance won")
c2.metric("Games tested", f"{sc['games']:,}", help=f"Every game since {season_label(info['score_from'])}, "
          "each predicted using only earlier games")
c3.metric("Log-loss", f"{sc['logloss']:.4f}", help="Lower is better. Always guessing the home team scores 0.690")
st.markdown(
    """
Hockey is close to a coin flip: even the best prediction models pick only about 60% of NHL winners.
What matters is whether the percentages are honest. When the favourite is given 65%, does it win
about 65% of the time?
"""
)
cal = calibration().rename(columns={"bin": "Favourite's chance", "n": "Games", "predicted": "Predicted",
                                    "actual": "Actually won"})
a, b = st.columns([2, 3])
with a:
    st.dataframe(cal, hide_index=True,
                 column_config={"Predicted": st.column_config.NumberColumn(format="percent"),
                                "Actually won": st.column_config.NumberColumn(format="percent")})
with b:
    acc = accuracy()
    acc = acc.loc[acc["season"] >= info["score_from"]]
    st.dataframe(acc[["Season", "games", "pick_pct", "logloss"]].sort_values("Season", ascending=False)
                 .rename(columns={"games": "Games", "pick_pct": "Winners picked", "logloss": "Log-loss"}),
                 hide_index=True, height=300,
                 column_config={"Winners picked": st.column_config.NumberColumn(format="percent")})

st.caption("This is a side project and is not affiliated with the NHL. Data from the NHL's public web API.")
