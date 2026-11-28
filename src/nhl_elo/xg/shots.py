"""Play-by-play -> one row per shot attempt (goal, shot on goal, missed, blocked) with xG features.

Coordinates are rotated so every shot attacks the net at (+89, 0). The shooting team comes from the
roster (shootingPlayerId / scoringPlayerId), not eventOwnerTeamId, because the owner of a blocked shot
has not always been the shooting team.

Which end a team attacks in each period: `homeTeamDefendingSide` when the feed has it (recent
seasons); otherwise the side where the team's unblocked shots cluster in that period.

situationCode is four digits: away goalie (0/1), away skaters, home skaters, home goalie (0/1).

Strength state, from the shooting team's point of view:
    EN   the opposing net is empty
    EV   equal skaters (5v5, 4v4) in regulation
    PP   more skaters (power play, extra attacker with the other goalie in) or 3-on-3 overtime
    SH   fewer skaters
Shootouts and penalty shots are dropped.

Pre-shot context (public data has no passes, so these are proxies):
    prev_*        the previous event in the period: type, same team, seconds, distance, zone
    rebound       previous event is an unblocked shot by the same team within REBOUND_SECONDS
    angle_change  change in shot angle from the shot being rebounded (degrees, rebounds only)
    rush          previous event in the neutral or defensive zone within RUSH_SECONDS
    off_turnover  a takeaway by the shooter's team or a giveaway by the other team within TURNOVER_SECONDS
    flurry_id     consecutive unblocked shots by one team, each within FLURRY_SECONDS of the last,
                  with no stoppage, faceoff or goal between them
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

NET_X = 89.0
BLUE_LINE = 25.0
SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot", "blocked-shot"}
UNBLOCKED = {"goal", "shot-on-goal", "missed-shot"}
REBOUND_SECONDS = 3
RUSH_SECONDS = 4
TURNOVER_SECONDS = 5
FLURRY_SECONDS = 3
FLURRY_BREAKERS = {"stoppage", "faceoff", "goal", "period-end", "penalty", "period-start", "delayed-penalty"}
CONTEXT_TYPES = {"faceoff", "hit", "giveaway", "takeaway", "shot-on-goal", "missed-shot", "blocked-shot",
                 "goal", "stoppage", "penalty", "delayed-penalty"}


@dataclass
class _Ev:
    type: str
    team: int | None
    t: int            # seconds into the game
    x: float | None   # raw coordinates
    y: float | None


def _seconds(period: int, clock: str) -> int:
    m, s = clock.split(":")
    return (period - 1) * 1200 + int(m) * 60 + int(s)


def _shooter(details: dict) -> int | None:
    return details.get("scoringPlayerId") or details.get("shootingPlayerId")


def _attack_sign(plays: list[dict], roster: dict[int, int], home: int, away: int) -> dict[tuple[int, int], int]:
    """(period, team) -> +1 if the team attacks the +x net in that period, else -1."""
    out: dict[tuple[int, int], int] = {}
    xs: dict[tuple[int, int], list[float]] = {}
    for p in plays:
        per = p["periodDescriptor"]["number"]
        side = p.get("homeTeamDefendingSide")
        if side in ("left", "right") and (per, home) not in out:
            out[(per, home)] = 1 if side == "left" else -1
            out[(per, away)] = -out[(per, home)]
        if p["typeDescKey"] in UNBLOCKED:
            d = p.get("details", {})
            team = roster.get(_shooter(d))
            if team is not None and d.get("xCoord") is not None and abs(d["xCoord"]) > BLUE_LINE:
                xs.setdefault((per, team), []).append(d["xCoord"])
    for (per, team), v in xs.items():
        if (per, team) in out:
            continue
        other = away if team == home else home
        s = np.sign(np.median(v)) if v else 0
        if s == 0 and xs.get((per, other)):
            s = -np.sign(np.median(xs[(per, other)]))
        if s != 0:
            out[(per, team)] = int(s)
            out.setdefault((per, other), -int(s))
    return out


def _strength(code: str | None, shooter_is_home: bool, period: int, game_type: int) -> tuple[str, int, int, bool] | None:
    """-> (state, own skaters, opp skaters, own net empty) or None for shootouts / penalty shots."""
    if not code or len(code) != 4 or not code.isdigit():
        return None
    ag, ask, hsk, hg = (int(c) for c in code)
    own_sk, opp_sk = (hsk, ask) if shooter_is_home else (ask, hsk)
    own_g, opp_g = (hg, ag) if shooter_is_home else (ag, hg)
    if own_sk <= 1 or opp_sk == 0:
        return None  # penalty shot or shootout attempt
    if opp_g == 0:
        state = "EN"
    elif own_sk > opp_sk or (own_sk == opp_sk == 3 and period >= 4 and game_type == 2):
        state = "PP"
    elif own_sk < opp_sk:
        state = "SH"
    else:
        state = "EV"
    return state, own_sk, opp_sk, own_g == 0


def _geom(x: float, y: float) -> tuple[float, float]:
    """Distance (ft) and angle (degrees, 0 = straight on, >90 = from behind the goal line)."""
    dx = NET_X - x
    dist = math.hypot(dx, y)
    angle = math.degrees(math.atan2(abs(y), dx))
    return dist, angle


def parse_game(pbp: dict, game_type: int | None = None) -> list[dict]:
    """All shot attempts in one game with features. Coordinates are rotated to attack +x."""
    home, away = pbp["homeTeam"]["id"], pbp["awayTeam"]["id"]
    gid, season = pbp["id"], pbp["season"]
    gt = game_type if game_type is not None else pbp.get("gameType", 2)
    roster = {r["playerId"]: r["teamId"] for r in pbp.get("rosterSpots", [])}
    pos = {r["playerId"]: r.get("positionCode") for r in pbp.get("rosterSpots", [])}
    plays = sorted(pbp.get("plays", []), key=lambda p: p.get("sortOrder", 0))
    sign = _attack_sign(plays, roster, home, away)

    rows: list[dict] = []
    score = {home: 0, away: 0}
    prev: _Ev | None = None
    prev_shot: dict | None = None   # last unblocked shot row, for rebounds and flurries
    flurry = 0
    last_turnover: dict[int, int] = {}   # team that gained the puck -> time
    for p in plays:
        typ = p["typeDescKey"]
        per_d = p["periodDescriptor"]
        per = per_d["number"]
        if per_d.get("periodType") == "SO":
            continue
        d = p.get("details", {}) or {}
        t = _seconds(per, p.get("timeInPeriod", "00:00"))
        if typ == "period-start":
            prev, prev_shot = None, None
            continue

        if typ in SHOT_TYPES:
            team = roster.get(_shooter(d))
            x, y = d.get("xCoord"), d.get("yCoord")
            s = sign.get((per, team)) if team is not None else None
            st = _strength(p.get("situationCode"), team == home, per, gt) if team is not None else None
            if team is not None and st is not None and x is not None and y is not None and s is not None:
                nx, ny = x * s, y * s
                dist, angle = _geom(nx, ny)
                opp = away if team == home else home
                row = {
                    "game_id": gid, "season": season, "game_type": gt, "event_id": p.get("eventId"),
                    "period": per, "t": t, "team": team, "opp": opp, "is_home": team == home,
                    "shooter": _shooter(d), "shooter_pos": pos.get(_shooter(d)), "goalie": d.get("goalieInNetId"),
                    "event": typ, "goal": int(typ == "goal"), "unblocked": typ in UNBLOCKED,
                    "shot_type": d.get("shotType") or "unknown", "miss_reason": d.get("reason"),
                    "x": nx, "y": ny, "dist": dist, "angle": angle,
                    "state": st[0], "own_sk": st[1], "opp_sk": st[2], "own_net_empty": st[3],
                    "score_diff": score[team] - score[opp], "overtime": per >= 4,
                    "period_seconds": t - (per - 1) * 1200,
                }
                # previous event
                if prev is not None:
                    same = prev.team == team
                    if prev.x is not None and prev.y is not None:
                        # put the previous event in the shooter's frame
                        px, py = prev.x * s, prev.y * s
                        row["prev_dist"] = math.hypot(nx - px, ny - py)
                        row["prev_zone"] = "O" if px > BLUE_LINE else ("D" if px < -BLUE_LINE else "N")
                    else:
                        row["prev_dist"], row["prev_zone"] = np.nan, "unknown"
                    row["prev_type"] = prev.type
                    row["prev_same_team"] = same
                    row["prev_seconds"] = t - prev.t
                else:
                    row.update(prev_type="none", prev_same_team=False, prev_seconds=np.nan,
                               prev_dist=np.nan, prev_zone="unknown")
                dt = row["prev_seconds"]
                row["rebound"] = bool(prev_shot is not None and prev is not None and prev.type in UNBLOCKED
                                      and prev_shot["team"] == team and t - prev_shot["t"] <= REBOUND_SECONDS)
                row["angle_change"] = abs(angle - prev_shot["angle"]) if row["rebound"] else 0.0
                row["rush"] = bool(prev is not None and row["prev_zone"] in ("N", "D") and dt <= RUSH_SECONDS)
                lt = last_turnover.get(team)
                row["off_turnover"] = bool(lt is not None and t - lt <= TURNOVER_SECONDS)
                if row["unblocked"]:
                    cont = (prev_shot is not None and prev_shot["team"] == team and t - prev_shot["t"] <= FLURRY_SECONDS
                            and prev is not None and prev.type in SHOT_TYPES)
                    if not cont:
                        flurry += 1
                    row["flurry_id"] = flurry
                    prev_shot = row
                else:
                    row["flurry_id"] = -1
                rows.append(row)
            if typ == "goal":
                if team is not None:
                    score[team] += 1
                prev_shot = None
            prev = _Ev(typ, team, t, x, y)
            continue

        if typ in ("takeaway", "giveaway"):
            owner = d.get("eventOwnerTeamId")
            gainer = owner if typ == "takeaway" else (away if owner == home else home)
            last_turnover[gainer] = t
        if typ in FLURRY_BREAKERS:
            prev_shot = None
        if typ in CONTEXT_TYPES:
            prev = _Ev(typ, d.get("eventOwnerTeamId"), t, d.get("xCoord"), d.get("yCoord"))
    return rows


def shots_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("prev_type", "prev_zone", "shot_type", "state", "event", "shooter_pos"):
        df[c] = df[c].astype("category")
    return df
