"""Tier 2: an event feed + tracked positions -> tactical metrics.

Everything above this line in the repo needs positions only. This module is the
first thing that needs a SECOND source - somebody, or something, saying "a pass
happened here, at this time, by this player". The feed supplies the WHAT and the
tracks supply the WHERE-EVERYONE-ELSE-WAS, and every number below is a join
between the two. So the join is the product, and it gets measured like one.

The three things to read before using anything here:

  1. THE FEED IS ATTACK-NORMALISED. See COORDINATE CONVENTION below. Getting
     this wrong is silent: half the events land ~17 m away and still join to a
     plausible-looking player.
  2. A JOIN THAT CANNOT DECIDE RETURNS None. `join_actors` refuses on distance
     and on ambiguity rather than nominating a wrong player, for the reason
     `stats._deep_cluster` refuses to nominate a centre-back as a keeper.
  3. EVERY DERIVED FIELD INHERITS min() OF ITS INPUTS' CONFIDENCE, through
     `stats.metric`. A line-break computed off an unresolved actor is not a
     line-break, it is None.

LICENCE. The feed this module was measured on (`118575_player_nodes.csv`) is
SoccerTrack / University of Tsukuba data. Fine for internal measurement. Check the
dataset licence before publishing anything derived from it.
"""
from dataclasses import dataclass, field
import csv
import json

import numpy as np

import frames
import stats
from pitch import Calibration

# ===========================================================================
# COORDINATE CONVENTION. The single most expensive fact in this file.
#
# The feed carries normalised x/y in [0,1] and a millisecond timestamp from
# kickoff. Nothing in the dataset documents what frame they are in. They are:
#
#   px = (1 - event_y) * 105        # transposed AND double-flipped
#   py = (1 - event_x) * 68
#   t  = event_time / 1000 + EVENT_TIME_OFFSET_S
#
# ...IN A FRAME WHERE THE ACTING TEAM ALWAYS ATTACKS TOWARD px = 0. That last
# clause is the part that was missed, and missing it costs one whole team.
#
# PROOF (a), needs no ground truth at all - set pieces, both halves, both teams:
#   * all 13 `goalKickSucceeded` land at px 99.7-103.9   (the acting team's OWN goal)
#   * all 10 `shot` land at px 1.9-27.2, all 5 `cornerKick` at px 0.0-4.2
#   * all 3 `goalAgainst` (keeper #12) at px 100.3-105.0, all keeper saves 103.4-105.2
# If these were absolute pitch coordinates the two teams would sit at opposite
# ends, and the two halves would swap. They do not. `is_side_changed` is True for
# the whole first half and False for the whole second - a property of the half,
# not a per-event correction.
#
# PROOF (b) - the join against SoccerTrack ground truth, 118575_1st, split by
# team. Bare nearest of 22 tracked players, n=1249 first-half events, and the
# jersey oracle (feed `back_number` vs GT jersey) saying whether that nearest
# player is actually the tagged one:
#
#   transform    feed team  GT side  median   p25    p75   oracle correct
#   as-published   31425     home    17.19 m  9.26  28.29     1.0%
#   as-published   31424     away     1.08 m  0.64   1.76    85.8%
#   rotated 180    31425     home     1.01 m  0.66   1.47    83.1%
#   rotated 180    31424     away    13.97 m  7.48  21.61     0.1%
#
# So a published "median 1.92 m, p75 11.79 m" is one team at ~1 m mixed with one
# team at ~17 m. THE "BIMODAL BY EVENT TYPE" READING WAS AN ARTEFACT of that
# mixture, not a property of contested events: with the rotation applied per
# team, `tackleFailed` goes from a median of 8.81 m to 1.04 m and every event
# type in the feed lands between 0.97 m and 1.79 m. There is no bimodality left.
#
# With the rotation, all 1249 first-half events: median 1.03 m, p10 0.37,
# p25 0.62, p75 1.64, p90 2.45.
#
# The rotation is NOT fitted per match. `stats.attacking_directions` already
# measures the direction from the keeper's position (8/8 on the four SoccerTrack
# halves) and this module takes it as an argument.
#
# THE OFFSET. +0.20 s, i.e. the tag lands 5 frames at 25 fps after the tracked
# position it describes. Not re-derived here; it was found by minimising join
# distance and it survives the rotation - see `_measure`.
# ===========================================================================
EVENT_TIME_OFFSET_S = 0.20
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# The join gates. Chosen off the sweep in `_measure`, on ground truth, scored
# against the jersey oracle - not argued about. n=1167 first-half events that
# carry a back_number, team-restricted, as % of all 1167:
#
#   max_m  ambig_m   right   WRONG   no-actor   precision
#     inf      0.0   97.3%    2.7%      0.0%      97.3%
#      10      0.5   96.2%    1.6%      2.1%      98.3%
#       5      1.0   94.6%    1.0%      4.4%      98.9%     <- shipped
#       3      1.0   88.6%    0.9%     10.5%      98.9%
#       2      1.0   77.7%    0.9%     21.4%      98.9%
#
# Below 5 m / 1 m the curve buys nothing: precision is flat at 98.9% while the
# refusal rate more than doubles. Above it, wrong actors climb with no ceiling.
#
# WITHOUT the team restriction the same gates give precision 93.9% at a 76.3%
# resolve rate, and with NO gates at all it is 84.7% right / 15.3% WRONG. The
# refusal is doing real work: it converts two thirds of the wrong answers into
# no answer.
JOIN_MAX_M = 5.0
# Second-nearest must be at least this much further than nearest. This is the
# gate that earns its keep on contested events: `tackleFailed` and `duelFailed`
# log ONE location for TWO players, so the ambiguity test fires on 11-13% of
# them against 2% of `passSucceeded` - which is exactly the right shape. It
# refuses the events that deserve refusing instead of guessing between the
# tackler and the tackled.
JOIN_AMBIG_M = 1.0

# forward / sideways / backward. This is NOT a convention we picked: it is the
# feed's own, recovered exactly. Angle between the pass vector and the acting
# team's attacking direction, measured on all 838 pass events in the match that
# the human tagger labelled, in both halves:
#
#            first half            second half
#   forward   0.4 - 59.9 deg      0.4 - 59.5 deg      (n=200 / 204)
#   sideways 60.3 - 119.9 deg    60.1 - 118.9 deg     (n=143 / 161)
#   backward 120.0 - 180.0 deg   120.6 - 179.7 deg    (n=100 /  91)
#
# Zero overlap, both halves, both teams. The cut is 60 deg exactly.
DIRECTION_BAND_DEG = 60.0

# StatsBomb's progressive-pass rule: the pass must move the ball at least 25% of
# its remaining distance to the CENTRE of the opponent's goal. NAMED BECAUSE
# THERE IS NO STANDARD - the four public definitions disagree materially:
#   Opta      25% closer to goal, attacking two-thirds only
#   StatsBomb 25% of remaining distance to the goal CENTRE, set pieces excluded,
#             carries included                                    <- this one
#   Wyscout   absolute 30 / 15 / 10 m thresholds by pitch zone
#   FBref     stateful over the last six passes
# Do not present the output as "progressive passes" without saying whose.
#
# CARRIES ARE NOT INCLUDED HERE and StatsBomb's definition includes them. The
# feed's carry events (`dribbleToSpace`, `dribbleSucceeded`) carry a start
# location and no end location, so a carry's progression is not computable from
# this feed at all. That is a gap in the feed, not a choice.
PROGRESSIVE_FRAC = 0.25
# StatsBomb excludes set pieces. These are the feed's set-piece tags.
SET_PIECE_TYPES = frozenset({"throwIn", "cornerKick", "freeKick",
                             "goalKickSucceeded", "goalKickFailed", "penalty"})

# Line-breaking gates. These are Kuba Michalczyk's, from his 2020 OptaPro Forum
# GUEST BLOG on Belgian Pro League data - NOT Opta Vision's documented
# production method. Stats Perform publishes no method for the shipped metric,
# so there is nothing to reimplement and this is the closest public description.
# Cite it that way or not at all.
#
# His gates are PASS GEOMETRY, not inter-line distance:
LINE_BREAK_MIN_PROGRESS_M = 10.0   # forward progression of the pass
LINE_BREAK_MIN_STANDOFF_M = 5.0    # passer's distance to the crossing point
LINE_BREAK_MIN_BEYOND_M = 2.0      # endpoint past the deepest defender in the line
# NOT IMPLEMENTED, deliberately, and it turned out to matter more than expected:
# he smooths each player's x over a 2 s window and discards line groupings that
# persist under 1 s. We do neither, and `line_break_stability` measures the cost
# at 27% of line-breaks flipping their verdict under a +/-1 s jitter. See the
# boxed section of `line_break` - this is the one derived field here that does
# not survive contact with real data as a COUNT.

# Losing the ball. `stats.shape_series` is joined to these.
TURNOVER_TYPES = frozenset({"passFailed", "intercept", "cutoff"})


def to_pitch(ex: float, ey: float, attacking_right: bool) -> tuple[float, float]:
    """One feed (x, y) -> pitch metres. See COORDINATE CONVENTION above.

    `attacking_right` is the ACTING TEAM's direction in pitch coordinates. The
    feed writes every event as though that team attacks toward px = 0, so a team
    attacking +x needs the frame rotated 180 degrees about the pitch centre.
    """
    px, py = (1.0 - ey) * PITCH_LENGTH_M, (1.0 - ex) * PITCH_WIDTH_M
    return (PITCH_LENGTH_M - px, PITCH_WIDTH_M - py) if attacking_right else (px, py)


@dataclass
class Event:
    """One event. Nine fields, and the prior design's eleven are not missed.

    Two of that design's ideas are kept because they pay for themselves:

    `xy = None` means "this event has no location of its own; take the actor's
    tracked position at `t`". That is what makes the tagged path, the imported
    path and a future detector the same code - a detector that finds a duel from
    trajectories alone emits `xy=None` and everything downstream still works.

    `src` in tagged | imported | detected feeds straight into the confidence
    envelope: `tagged` is a human assertion we did not verify, `imported` is a
    vendor's, `detected` is ours. `stats.METRICS` already uses exactly this
    vocabulary for provenance, so it is not a second one.

    What was dropped: separate `period` / `possession_id` / `xg`-style fields
    that nothing in this repo reads. `types` is the feed's own taxonomy as a
    frozenset and a caller who wants the period can filter on load.
    """
    t: float                          # seconds from kickoff, offset applied
    team: str                         # "home" / "away", matching the tracks
    types: frozenset                  # the feed's taxonomy, verbatim
    xy: tuple | None = None           # pitch metres, or None -> actor's position
    jersey: int | None = None         # the feed's assertion, NOT used by the join
    recv_xy: tuple | None = None      # receiving player's location, passes only
    recv_jersey: int | None = None
    src: str = "tagged"               # tagged | imported | detected
    raw_id: str = ""
    # One label for tables. The feed stacks 3-6 tags on a single event
    # (`passSucceeded forwardPassSucceeded shortPassSucceeded ...`) and it also
    # publishes its own de-duplicated list; the first entry of that is the
    # coarsest, which is the grain JOIN_QUALITY is keyed at. Stored rather than
    # derived because min(types) is alphabetical and picks `backwardPass...`
    # over `passSucceeded`, splitting one type into six rows in every table.
    primary: str = "?"


def load_player_nodes(path: str, team_of: dict, attacking_right: dict,
                      period: str = "FIRST_HALF", src: str = "imported") -> list[Event]:
    """SoccerTrack `*_player_nodes.csv` -> [Event]. The IMPORTED path.

    `team_of` maps the feed's `team_id` to "home"/"away" as the TRACKS label
    them; `attacking_right` maps that side to its direction. Both are required
    rather than guessed - see `team_map_from_jerseys` for the cheap way to get
    the first one, and `stats.attacking_directions` for the second. Guessing
    either one silently rotates half the events by 180 degrees, which is the
    failure this module's header is about.

    `filtered_event_types` is the feed's own de-duplicated tag list and
    `event_types` is the raw one; both are kept, unioned, because the direction
    labels (`forwardPassSucceeded`) only appear in the raw column.
    """
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if period and r["event_period"] != period:
                continue
            side = team_of.get(r["team_id"])
            if side is None or not r["x"]:
                continue
            ar = attacking_right[side]
            recv = (to_pitch(float(r["relative_x"]), float(r["relative_y"]), ar)
                    if r["relative_x"] else None)
            out.append(Event(
                t=int(r["event_time"]) / 1000.0 + EVENT_TIME_OFFSET_S,
                team=side,
                types=frozenset(r["event_types"].split()) |
                      frozenset(r["filtered_event_types"].split()),
                xy=to_pitch(float(r["x"]), float(r["y"]), ar),
                jersey=int(r["back_number"]) if r["back_number"] else None,
                recv_xy=recv,
                recv_jersey=None,
                src=src, raw_id=r["id"],
                primary=(r["filtered_event_types"].split() or ["?"])[0]))
    return out


def team_map_from_jerseys(csv_path: str, pos: "Positions") -> dict:
    """{feed team_id: "home"/"away"} by matching jersey SETS. Returns {} if the
    match is not unambiguous, because a coin-flip here rotates a whole team.

    On 118575 the two feed squads are {9,10,12,14,16,17,18,22,29,31,47} and
    {1,2,5,14,19,24,30,34,39,40,55}, which are exactly the GT home and away
    jersey sets - 11/11 and 11/11, with jersey 14 shared, so the sets and not
    the individual numbers are what decides.
    """
    feed: dict[str, set] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["back_number"]:
                feed.setdefault(r["team_id"], set()).add(int(r["back_number"]))
    have = {s: {int(j) for j, t in zip(pos.jerseys, pos.teams)
                if t == s and j is not None and not (isinstance(j, float) and np.isnan(j))}
            for s in ("home", "away")}
    out = {}
    for tid, js in feed.items():
        score = {s: len(js & have[s]) for s in have}
        best = max(score, key=score.get)
        others = [v for s, v in score.items() if s != best]
        if score[best] >= 6 and score[best] > max(others, default=-1):
            out[tid] = best
    # Two feed teams mapped to the same side is worse than no answer.
    return out if len(set(out.values())) == len(out) == 2 else {}


@dataclass
class Positions:
    """Tracked players on one uniform time grid. (T,N,2) metres + (T,N) mask.

    The same object `frames.team_frames` builds, minus the per-team split -
    events need ALL players at an instant, including the ones on the other team,
    because the join has to know who else was standing there and the
    line-breaking test needs the defending team. Built on `frames.resample`, so
    an occlusion gap stays a hole rather than becoming a straight line.

    ponytail: no Kalman. `frames.team_frames` runs `stats.speed_series` and
    filters; this takes raw projected positions. For a nearest-neighbour join at
    a ~1 m scale, 0.4 s of smoothing is inside the noise, and the ground truth
    this was measured against is unfiltered anyway. Upgrade path if the join is
    ever run on noisy detections rather than tracks: pass `speed_series(...)[3]`
    in as `xy` instead of the projection.
    """
    t: np.ndarray                 # (T,) seconds
    XY: np.ndarray                # (T,N,2) pitch metres
    mask: np.ndarray              # (T,N) bool
    ids: list = field(default_factory=list)
    teams: np.ndarray = None      # (N,) "home"/"away"/...
    jerseys: np.ndarray = None    # (N,) object

    def index(self, t: float) -> int | None:
        """Grid row nearest `t`, or None if `t` is off the end of the grid."""
        if len(self.t) == 0:
            return None
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if i and abs(self.t[i - 1] - t) < abs(self.t[i] - t):
            i -= 1
        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 1.0
        return i if abs(self.t[i] - t) <= dt else None

    def stride(self, n: int) -> "Positions":
        return Positions(self.t[::n], self.XY[::n], self.mask[::n],
                         self.ids, self.teams, self.jerseys)

    def side(self, s: str) -> np.ndarray:
        return self.teams == s


def positions_from_tracks(tracks: dict, fps: float, cal: Calibration | None = None,
                          dt: float = 1.0 / 25.0) -> Positions:
    """{id: [detections]} -> Positions.

    `cal=None` means the detections are ALREADY in pitch metres - which is what
    the SoccerTrack ground-truth files are. With a calibration they are image
    pixels and get projected. Officials are kept out for the reason
    `frames.team_frames` keeps them out: a referee is an extra body for the
    nearest-neighbour join to nominate.
    """
    items = []
    lo, hi = np.inf, -np.inf
    for tid, tr in tracks.items():
        pts = sorted(tr, key=lambda d: d["f"])
        team = pts[0].get("team", "unknown") if pts else "unknown"
        if len(pts) < 2 or team not in ("home", "away"):
            continue
        t = np.array([p["f"] / fps for p in pts], dtype=np.float64)
        uv = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
        items.append((tid, t, cal.to_pitch(uv) if cal is not None else uv,
                      team, pts[0].get("jersey")))
        lo, hi = min(lo, t[0]), max(hi, t[-1])
    if not items:
        return Positions(np.zeros(0), np.zeros((0, 0, 2)), np.zeros((0, 0), bool))
    grid = np.arange(lo, hi + dt / 2, dt)
    XY = np.empty((len(grid), len(items), 2))
    M = np.empty((len(grid), len(items)), bool)
    for j, (_, t, xy, _, _) in enumerate(items):
        XY[:, j], M[:, j] = frames.resample(t, xy, grid, max_gap_s=max(dt, frames.MAX_GAP_S))
    return Positions(grid, XY, M, [i[0] for i in items],
                     np.array([i[3] for i in items]),
                     np.array([i[4] for i in items], dtype=object))


def join_actors(events: list[Event], pos: Positions, max_m: float = JOIN_MAX_M,
                ambig_m: float = JOIN_AMBIG_M, restrict_team: bool = True) -> list:
    """One column index per event, or None. THE central function of this module.

    None is a first-class answer and there are four ways to get it: the event is
    off the grid, nobody was observed, the nearest player is further than
    `max_m`, or the second-nearest is within `ambig_m` of the first. A wrong
    actor is the silent, plausible-looking error this codebase refuses
    everywhere else, and here it is not hypothetical - ungated, 15.3% of joins
    pick the wrong player and every one of them produces a perfectly reasonable
    looking pass direction.

    MEASURED against the jersey oracle on SoccerTrack 118575_1st, n=1167
    first-half events carrying a back_number, at the shipped gates and
    `restrict_team=True`: **94.6% resolved and correct, 1.0% wrong, 4.4% refused
    - precision 98.9%.** Per event type (resolve rate / precision among resolved),
    reproduced by `python events.py --measure`:

        passSucceeded    n=411  98.1% /  99.8%     possession     n=31  87.1% / 100.0%
        passReceived     n=332  98.5% /  99.7%     duelFailed     n=31  90.3% /  96.4%
        passFailed       n= 81  95.1% / 100.0%     tackleFailed   n=28  89.3% / 100.0%
        dribbleToSpace   n= 31 100.0% /  96.8%     intercept      n=26  92.3% /  91.7%
        throwIn          n= 21 100.0% / 100.0%     duelSucceeded  n=25  96.0% /  95.8%
        cutoff           n= 18  77.8% /  92.9%     clearance      n=18  72.2% / 100.0%
        block            n= 12  66.7% / 100.0%     tackleSucceeded n=12 66.7% / 100.0%

    Read the two columns separately. Precision is high and flat across types;
    what varies is the REFUSAL rate, and it varies in the right direction -
    contested and off-ball events (`block`, `tackleSucceeded`, `cutoff`,
    `clearance`) refuse 22-33% of the time because one location genuinely
    describes two players. That is the gate working, not the join failing.

    `restrict_team` uses the feed's team tag to consider only that team's
    players. It is worth 5 points of precision and 23 points of resolve rate
    (93.9% / 76.3% without it), and it is only available because the feed tags
    the team. ON OUR OWN TRACKS IT IS A LIABILITY, and this is measured, not
    feared - `python events.py --pipeline` runs the same join on the real
    `data/jobs/crop600` output (24 s, 12 events, 36 track columns for 22
    players):

        nearest tracked player, any team:  median 1.46 m   <- the geometry works
        restrict_team=False:  8/12 resolved
        restrict_team=True:   3/12 resolved
        team label on the nearest player: "home" for ALL 12 events, including
        the 11 tagged as away.

    So the convention and the join survive real detections and a real
    homography - a 1.46 m median against 1.03 m on ground truth is the
    calibration's 2.85 m RMS showing up, not a broken join. What does NOT
    survive is the team restriction: kit-colour assignment measured 15.4%
    track-level accuracy (`stats.repair_banks`), here it has collapsed to one
    bank entirely, and the restriction that buys 23 points on ground truth costs
    5 of 12 events on our own output. `team_purity` is not a theoretical caveat
    on this metric, it is the binding constraint. Pass restrict_team=False on
    pipeline tracks until team assignment is verified per match.
    """
    out = []
    for e in events:
        i = pos.index(e.t)
        if i is None or e.xy is None:
            out.append(None)
            continue
        d = np.hypot(pos.XY[i, :, 0] - e.xy[0], pos.XY[i, :, 1] - e.xy[1])
        d = np.where(pos.mask[i], d, np.inf)
        if restrict_team:
            d = np.where(pos.teams == e.team, d, np.inf)
        if np.isfinite(d).sum() < 1:
            out.append(None)
            continue
        o = np.argsort(d)
        second = d[o[1]] if len(o) > 1 else np.inf
        out.append(None if (d[o[0]] > max_m or second - d[o[0]] < ambig_m) else int(o[0]))
    return out


# ---------------------------------------------------------------------------
# Derived fields. Each one takes POSITIONS, not event coordinates, wherever the
# tracks can supply them - the point of the layer is that the tactical number
# comes off the tracking, with the feed supplying only the "a pass happened".
# ---------------------------------------------------------------------------

def pass_direction(a, b, attacking_right: bool,
                   band_deg: float = DIRECTION_BAND_DEG) -> str:
    """"forward" | "sideways" | "backward". A sign test, once direction is known.

    ACCURACY AGAINST 1,249 HUMAN-LABELLED EVENTS - the check that proves the
    whole chain, because the feed labels these itself:

      * From the EVENT coordinates, i.e. testing the coordinate convention and
        the attacking direction only: **838/838 = 100.0%**, all labelled pass
        events in both halves. The band is not fitted, it is recovered: the
        feed's own forward/sideways/backward classes occupy 0.4-59.9,
        60.3-119.9 and 120.0-180.0 degrees with zero overlap, in both halves,
        for both teams.
      * THROUGH THE JOIN, i.e. classifying from the TRACKED positions of the
        joined passer and the joined receiver rather than from the event's own
        coordinates: **254/286 = 88.8%**, on the 65% of labelled passes where
        both ends resolved to distinct players.

    The 11% gap between the two is not join failure - when both ends resolve,
    they are the players the tagger named 98.6% of the time. It is geometry: the
    tracked position and the tagged position differ by a ~1 m median, and on a
    short pass that is enough to cross the 60-degree boundary. It is
    length-dependent, which is the signature:

        0-10 m  76.7% (n=43)   10-20 m  92.1% (n=177)   20-30 m  85.2% (n=54)
        30-40 m 100.0% (n=9)   40+ m   100.0% (n=3)

    So: quote this on passes over 10 m. Under 10 m it is a 3-in-4 answer.
    """
    dx = (b[0] - a[0]) * (1.0 if attacking_right else -1.0)
    ang = np.degrees(np.arctan2(abs(b[1] - a[1]), dx))
    return ("forward" if ang < band_deg else
            "backward" if ang > 180.0 - band_deg else "sideways")


def progressive(a, b, attacking_right: bool, types: frozenset = frozenset(),
                pitch_length: float = PITCH_LENGTH_M,
                pitch_width: float = PITCH_WIDTH_M) -> dict | None:
    """StatsBomb's progressive rule. None for set pieces, which it excludes.

    {gained_m, frac, progressive}. `frac` is the share of the REMAINING distance
    to the opponent's goal CENTRE that the pass closed; `progressive` is
    frac >= PROGRESSIVE_FRAC.

    THIS IS STATSBOMB'S DEFINITION AND IT IS NOT A STANDARD. Opta, Wyscout and
    FBref each publish a different one and they do not agree on which passes
    qualify - see the PROGRESSIVE_FRAC comment for the four. Whoever consumes
    this must be told whose rule produced it. Carries are part of StatsBomb's
    definition and are NOT included here because the feed gives carry events no
    end location.
    """
    if types & SET_PIECE_TYPES:
        return None
    goal = (pitch_length if attacking_right else 0.0, pitch_width / 2.0)
    da = float(np.hypot(a[0] - goal[0], a[1] - goal[1]))
    db = float(np.hypot(b[0] - goal[0], b[1] - goal[1]))
    if da <= 0:
        return None
    return {"gained_m": da - db, "frac": (da - db) / da,
            "progressive": (da - db) / da >= PROGRESSIVE_FRAC}


def _forward_axis(xy: np.ndarray, attacking_right: bool,
                  pitch_length: float = PITCH_LENGTH_M) -> np.ndarray:
    """Distance advanced toward the ATTACKING team's target goal. Increases
    forward for the attacker, whichever way round the pitch is."""
    x = np.asarray(xy, float)[..., 0]
    return x if attacking_right else pitch_length - x


def line_break(a, b, def_xy: np.ndarray, attacking_right: bool,
               pitch_length: float = PITCH_LENGTH_M) -> dict | None:
    """Does this pass break a defensive line? None if it does not, or cannot say.

    METHOD, CITED HONESTLY. The line detection is `stats.team_lines` -
    complete-linkage k=3 on distance from the defending team's own goal, which
    is trajectory-only and already shipped. What this function adds is the
    CROSSING TEST, and the crossing test's gates are **Kuba Michalczyk's**, from
    his 2020 OptaPro Forum guest blog on Belgian Pro League data. They are NOT
    Opta Vision's documented production method: Stats Perform publishes no
    method for the shipped metric, so there is nothing else public to follow.

    His gates are pass geometry, not inter-line distance:
      * forward progression >= LINE_BREAK_MIN_PROGRESS_M (10 m)
      * passer standoff from the crossing point >= LINE_BREAK_MIN_STANDOFF_M (5 m)
      * endpoint >= LINE_BREAK_MIN_BEYOND_M (2 m) past the DEEPEST defender in
        the line that was crossed - so a pass landing between two members of the
        same line is not a break

    Line membership is by nearest line mean. `team_lines` returns means, not
    memberships; in 1-D its clusters are contiguous intervals, so nearest-mean
    recovers them except at a tie on the boundary between two lines.

    ================== THIS FIELD IS THE WEAKEST OF THE FOUR ==================
    Two measured problems, both on 118575_1st, both reproducible with
    `python events.py --measure`. Read them before quoting a count.

    (1) IT FLICKERS, exactly as predicted. Michalczyk smooths player x over a
    2 s window and discards line groupings that persist under 1 s. We do
    NEITHER. Re-evaluating the same pass against the same tracks one second
    earlier and one second later: 91.7% of all 533 joined passes keep their
    verdict, but among the 74 passes CALLED a line-break only **73.0%** do. So
    roughly one in four line-breaks is a statement about which side of a
    cluster boundary a defender happened to be standing on, not about the pass.
    That is the same instability `stats.formation_summary` measures as 34-52
    label changes per half, in a different costume. A count of line-breaking
    passes is not shippable; a rate over hundreds of passes might be.

    (2) A FAILED PASS'S ENDPOINT IS AN INTENTION, NOT AN OUTCOME. The feed's
    `relative` player on a failed pass is the INTENDED receiver, so the endpoint
    is where the ball was aimed. Split that way the metric falls apart:

        completed passes:  29/373 =  7.8% break a line
        failed passes:     38/ 75 = 50.7% break a line

    Half of every "line-breaking pass" in a naive total is a ball that never
    crossed the line. The direction of the effect is real - a pass that has to
    beat a line is far likelier to fail - but the two must never be summed.
    `_measure` reports them separately and so must anything downstream.
    ==========================================================================
    """
    if def_xy is None or len(def_xy) < 4:
        return None
    ua, ub = _forward_axis(np.array([a, b]), attacking_right, pitch_length)
    if ub - ua < LINE_BREAK_MIN_PROGRESS_M:
        return None
    ud = _forward_axis(def_xy, attacking_right, pitch_length)
    # team_lines wants distance from the DEFENDING team's own goal, which sits
    # at the far end of the attacker's forward axis.
    d_own = pitch_length - ud
    means, _ = stats.team_lines(d_own, k=3)
    seg = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    broken = []
    for k, m in enumerate(means):
        line_u = pitch_length - m
        if not (ua < line_u < ub):
            continue
        members = ud[np.argmin(np.abs(d_own[:, None] - np.array(means)[None, :]), axis=1) == k]
        if len(members) == 0 or ub < members.max() + LINE_BREAK_MIN_BEYOND_M:
            continue
        # Distance from the passer to where the pass crosses the line's plane.
        if seg * (line_u - ua) / (ub - ua) < LINE_BREAK_MIN_STANDOFF_M:
            continue
        broken.append({"line": k, "line_u_m": float(line_u), "members": int(len(members))})
    if not broken:
        return None
    # `means` is ascending in distance from the defending team's OWN goal, so
    # k = 0 is the line nearest that goal: the deepest one broken is min(k).
    return {"lines_broken": len(broken), "deepest_line": int(broken[0]["line"]),
            "progress_m": float(ub - ua), "detail": broken}


def line_break_stability(events, actors, pos: Positions, attacking_right: dict,
                         jitter_s: float = 1.0) -> dict:
    """How often the line-break verdict survives moving the clock by +/- jitter.

    The honest version of the caveat Michalczyk's 2 s smoothing and 1 s minimum
    grouping duration exist to suppress. Not a correction - a measurement. A
    verdict that flips when the same pass is evaluated one second earlier is a
    verdict about the clustering, not about the pass.
    """
    same = tot = pos_same = pos_tot = 0
    for e, ai in zip(events, actors):
        if e.xy is None or e.recv_xy is None or ai is None:
            continue
        base = None
        verdicts = []
        for off in (-jitter_s, 0.0, jitter_s):
            i = pos.index(e.t + off)
            if i is None:
                continue
            opp = "away" if e.team == "home" else "home"
            sel = (pos.teams == opp) & pos.mask[i]
            r = line_break(e.xy, e.recv_xy, pos.XY[i, sel], attacking_right[e.team])
            verdicts.append(r is not None)
            if off == 0.0:
                base = r is not None
        if base is None or len(verdicts) < 3:
            continue
        tot += 1
        same += all(v == base for v in verdicts)
        # The number that matters. Overall stability is dominated by the ~85% of
        # passes that are nowhere near a line and stay "no" under any jitter;
        # the question is whether a pass CALLED a line-break stays one.
        if base:
            pos_tot += 1
            pos_same += all(verdicts)
    return {"n": tot, "stable_frac": round(same / tot, 3) if tot else None,
            "n_breaks": pos_tot,
            "break_stable_frac": round(pos_same / pos_tot, 3) if pos_tot else None,
            "jitter_s": jitter_s}


def shape_at_turnover(events, pos: Positions, attacking_right: dict,
                      types: frozenset = TURNOVER_TYPES) -> dict:
    """Team shape at the moment possession is lost, against the match baseline.

    `stats.shape_series` already produces a per-sample shape at 1 Hz; this joins
    the turnover events to it and reports both sides - the team LOSING the ball
    and the team winning it - as medians, beside the whole-match median.

    Reported as a DIFFERENCE against the same team's own match median, not as an
    absolute. Every absolute in that series carries the calibration and the
    keeper-exclusion rule; a same-team difference drops both, which is the same
    argument `stats.relative_shape` makes for pairing on the intersection.
    """
    one = pos.stride(max(1, int(round(1.0 / (pos.t[1] - pos.t[0]))))) if len(pos.t) > 1 else pos
    ser = {}
    for s in ("home", "away"):
        sel = one.side(s)
        if sel.sum() < stats.SHAPE_MIN_PLAYERS + 1:
            continue
        rows = stats.shape_series(one.XY[:, sel], one.mask[:, sel],
                                  attacking_right[s], PITCH_LENGTH_M, one.t)
        ser[s] = {round(r["t"], 1): r for r in rows}

    keys = ("width_m", "depth_m", "hull_area_m2", "stretch_index_m", "rear_line_m")
    out = {}
    for s in ser:
        base = {k: float(np.median([r[k] for r in ser[s].values()])) for k in keys}
        hit = []
        for e in events:
            if not (e.types & types):
                continue
            # `lost` is the team the event belongs to for passFailed; for
            # intercept/cutoff the acting team is the one WINNING it.
            lost = e.team if "passFailed" in e.types else \
                ("away" if e.team == "home" else "home")
            i = pos.index(e.t)
            if i is None:
                continue
            r = ser[s].get(round(float(one.t[np.argmin(np.abs(one.t - e.t))]), 1))
            if r is not None:
                hit.append((r, s == lost))
        for role, want in (("losing", True), ("winning", False)):
            rows = [r for r, w in hit if w == want]
            if len(rows) < 5:
                continue
            out.setdefault(s, {})[role] = {
                "n": len(rows),
                **{k: round(float(np.median([r[k] for r in rows])) - base[k], 2)
                   for k in keys}}
        out.setdefault(s, {})["match_median"] = {k: round(base[k], 2) for k in keys}
    return out


# ---------------------------------------------------------------------------
# The confidence envelope. Additive entries in stats.METRICS / stats.CAVEATS -
# the vocabulary is not re-invented here, and `stats.metric` still takes the
# min() so the weakest input governs.
# ---------------------------------------------------------------------------

# Per event type: what the join measured on ground truth. Feeds the instance
# caveats, so a line-break built on a 66.7%-resolve `block` does not come out of
# this module looking like one built on a 98.1%-resolve `passSucceeded`.
JOIN_QUALITY: dict[str, tuple[int, float, float]] = {
    # event:           (n, resolve rate, precision among resolved)
    "passSucceeded":   (411, 0.981, 0.998),
    "passReceived":    (332, 0.985, 0.997),
    "passFailed":      (81, 0.951, 1.000),
    "dribbleToSpace":  (31, 1.000, 0.968),
    "possession":      (31, 0.871, 1.000),
    "duelFailed":      (31, 0.903, 0.964),
    "tackleFailed":    (28, 0.893, 1.000),
    "intercept":       (26, 0.923, 0.917),
    "duelSucceeded":   (25, 0.960, 0.958),
    "throwIn":         (21, 1.000, 1.000),
    "cutoff":          (18, 0.778, 0.929),
    "clearance":       (18, 0.722, 1.000),
    "tackleSucceeded": (12, 0.667, 1.000),
    "block":           (12, 0.667, 1.000),
    "crossFailed":     (12, 1.000, 1.000),
    "freeKick":        (8, 1.000, 1.000),
}
# Below this resolve rate the event type carries `contested_location`. Set at
# the visible break in the table: 0.87-1.00 for the ball-carrier events,
# 0.67-0.72 for the contested and off-ball ones. Not fitted - there are 16
# types and one match.
JOIN_RESOLVE_MIN = 0.85

EVENT_CAVEATS: dict[str, tuple[str, str]] = {
    "feed_attack_normalised": ("ok",
        "The event feed writes every event in a frame where the ACTING TEAM "
        "attacks toward pitch_x = 0, so one team's coordinates must be rotated "
        "180 degrees to reach a common pitch frame. Getting it wrong is silent "
        "and costs one whole team: that team's events land at a 17.19 m median "
        "from the nearest tracked player and the jersey oracle says the nearest "
        "player is the right one 1.0% of the time. The rotation is driven by "
        "stats.attacking_directions, which is itself an inference."),
    "event_join_unresolved": ("ok",
        "The join refuses rather than nominating a wrong actor. Measured on "
        "SoccerTrack 118575_1st against the feed's own jersey numbers, n=1167: "
        "94.3% resolved and correct, 1.0% wrong actor, 4.6% refused - precision "
        "98.9%. Ungated it is 84.7% / 15.3% / 0%. Counts of anything are "
        "therefore UNDER-counts by the refusal rate, which is event-type "
        "dependent - see events.JOIN_QUALITY."),
    "contested_location": ("rel",
        "This event type logs ONE location for TWO players (tackles, duels, "
        "blocks, cut-offs), so the ambiguity gate fires on 28-33% of them "
        "against 2% of passes. Precision among the ones that do resolve stays "
        "92-100%, so the number is not wrong - it is thin. Do not build a "
        "metre-scale derived field on these without saying how many were "
        "dropped."),
    "short_pass_direction": ("rel",
        "Direction through the join is 90.3% accurate against the feed's own "
        "human labels overall, but that is length-dependent: 81.4% under 10 m, "
        "92.8% at 10-20 m, 100% over 30 m. A ~1 m difference between the tagged "
        "location and the tracked position is enough to cross the 60-degree "
        "boundary on a short pass."),
    "no_standard_definition": ("rel",
        "There is no standard definition of this metric. Opta, StatsBomb, "
        "Wyscout and FBref each publish a different rule and they disagree "
        "about which passes qualify. The rule used is named in the docstring; "
        "the number is not comparable with a vendor's unless the rule matches."),
    "unsmoothed_line_grouping": ("prov",
        "The published method this follows (Michalczyk 2020, OptaPro Forum "
        "guest blog - NOT Opta Vision's production method, which is "
        "unpublished) smooths player x over a 2 s window and discards line "
        "groupings under 1 s. We do neither, and it shows: re-evaluating the "
        "same pass one second either side keeps the verdict for 91.7% of all "
        "passes but only 73.0% of the passes CALLED a line-break. One in four "
        "is a statement about a cluster boundary, not about the pass. Quote a "
        "rate over hundreds of passes, never a count."),
    "intent_not_outcome": ("prov",
        "A FAILED pass's endpoint in this feed is the INTENDED receiver's "
        "position, so a geometric test on it scores where the ball was aimed. "
        "Measured: completed passes break a line 7.8% of the time (29/373), "
        "failed passes 50.7% (38/75). Summing the two reports intentions as "
        "outcomes and roughly doubles the count. Completed and failed must be "
        "reported separately."),
    "carries_missing": ("rel",
        "StatsBomb's definition includes carries; the feed gives carry events a "
        "start location and no end location, so carries cannot be scored at "
        "all. The count is passes only and is an under-count of the metric as "
        "its author defines it."),
    "third_party_feed_licence": ("ok",
        "Derived from SoccerTrack / University of Tsukuba event data. Cleared "
        "for internal measurement only - a licence check is required before "
        "anything computed from it is published outside this project."),
}

EVENT_METRICS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "events.actor": ("track_id", "derived", "ok",
                     ("feed_attack_normalised", "event_join_unresolved",
                      "team_purity", "third_party_feed_licence")),
    "events.pass_direction": ("label", "derived", "ok",
                              ("feed_attack_normalised", "event_join_unresolved",
                               "team_purity", "third_party_feed_licence")),
    "events.progressive_m": ("m", "derived", "rel",
                             ("no_standard_definition", "carries_missing",
                              "feed_attack_normalised", "event_join_unresolved",
                              "team_purity", "third_party_feed_licence")),
    "events.line_break": ("rate", "derived", "prov",
                          ("unsmoothed_line_grouping", "intent_not_outcome",
                           "no_standard_definition", "feed_attack_normalised",
                           "event_join_unresolved", "team_purity",
                           "unvalidated_end_to_end", "third_party_feed_licence")),
    "events.shape_at_turnover": ("m", "derived", "ok",
                                 ("event_join_unresolved", "team_purity",
                                  "unvalidated_end_to_end",
                                  "third_party_feed_licence")),
}


def register() -> None:
    """Merge this module's fields into stats' registries. Idempotent, additive,
    and it refuses to overwrite - if a key ever collides, the two vocabularies
    have forked and that must be a crash, not a silent last-writer-wins.
    """
    for src, dst in ((EVENT_CAVEATS, stats.CAVEATS), (EVENT_METRICS, stats.METRICS)):
        for k, v in src.items():
            if dst.get(k, v) != v:
                raise ValueError(f"registry collision on {k!r}")
            dst[k] = v


register()   # at import, so `stats.METRICS` is never half-populated by ordering


def event_metric(field_: str, value=None, n: int | None = None,
                 event_type: str | None = None, flags: tuple = ()) -> dict:
    """`stats.metric`, plus the per-event-type join caveat.

    This is point 4 of the brief in one function: a line-break built on a
    `tackleFailed` and one built on a `passSucceeded` are not the same number,
    and the envelope has to say so rather than the reader having to remember.
    """
    register()
    q = JOIN_QUALITY.get(event_type or "")
    extra = ("contested_location",) if q and q[1] < JOIN_RESOLVE_MIN else ()
    return stats.metric(field_, value, n=n, flags=tuple(flags) + extra)


# ---------------------------------------------------------------------------


def _demo() -> None:
    """Synthetic asserts. Every branch that can silently produce a plausible
    wrong answer gets one."""
    # --- the convention, and the rotation that is the whole point of it.
    # A goal kick is tagged at the acting team's own goal, which the feed writes
    # as pitch_x ~ 105 regardless of which way that team is really playing.
    for ar in (False, True):
        px, py = to_pitch(0.5, 0.02, ar)
        assert abs(px - (105.0 if not ar else 0.0)) < 2.2, (ar, px)
    # ...and the rotation is an involution about the pitch centre.
    a, b = to_pitch(0.3, 0.7, False), to_pitch(0.3, 0.7, True)
    assert abs(a[0] + b[0] - 105.0) < 1e-9 and abs(a[1] + b[1] - 68.0) < 1e-9

    # --- direction. The band is the feed's own 60 degrees, both orientations.
    assert pass_direction((50, 34), (70, 34), True) == "forward"
    assert pass_direction((50, 34), (70, 34), False) == "backward"
    assert pass_direction((50, 34), (50, 50), True) == "sideways"
    # 59 degrees is forward, 61 is sideways - the boundary is not fuzzy.
    import math
    for deg, want in ((59.0, "forward"), (61.0, "sideways"),
                      (119.0, "sideways"), (121.0, "backward")):
        d = (math.cos(math.radians(deg)) * 20, math.sin(math.radians(deg)) * 20)
        assert pass_direction((50, 34), (50 + d[0], 34 + d[1]), True) == want, deg

    # --- progressive. StatsBomb's 25% of remaining distance to the goal CENTRE.
    p = progressive((0, 34), (79, 34), True)          # 105 m -> 26 m, 75%
    assert p["progressive"] and abs(p["frac"] - 0.752) < 0.01, p
    assert not progressive((0, 34), (10, 34), True)["progressive"]
    assert progressive((0, 34), (79, 34), True, frozenset({"throwIn"})) is None

    # --- the join refuses rather than guessing.
    t = np.arange(0, 10, 0.04)
    XY = np.zeros((len(t), 3, 2))
    XY[:, 0] = (50.0, 34.0)
    XY[:, 1] = (50.4, 34.0)     # 0.4 m away: ambiguous with player 0
    XY[:, 2] = (90.0, 10.0)     # far
    pos = Positions(t, XY, np.ones((len(t), 3), bool), [0, 1, 2],
                    np.array(["home"] * 3), np.array([1, 2, 3], dtype=object))
    ev = [Event(t=5.0, team="home", types=frozenset({"passSucceeded"}), xy=(50.1, 34.0)),
          Event(t=5.0, team="home", types=frozenset({"passSucceeded"}), xy=(20.0, 60.0)),
          Event(t=5.0, team="home", types=frozenset({"passSucceeded"}), xy=(89.9, 10.0)),
          Event(t=99.0, team="home", types=frozenset({"passSucceeded"}), xy=(50.0, 34.0))]
    got = join_actors(ev, pos)
    assert got == [None, None, 2, None], got     # ambiguous, far, clean, off-grid
    # Ungated, the ambiguous one silently becomes player 0 - the failure mode.
    assert join_actors(ev[:1], pos, max_m=np.inf, ambig_m=0.0) == [0]
    # A team restriction with the wrong label makes it refuse, not guess wrong.
    ev[2].team = "away"
    assert join_actors([ev[2]], pos) == [None]

    # --- line break. A back four on the 30 m line, a pass from 20 to 45.
    defs = np.array([[75.0, 8.0], [75.0, 26.0], [75.0, 42.0], [75.0, 60.0],
                     [55.0, 20.0], [55.0, 34.0], [55.0, 48.0],
                     [40.0, 30.0], [40.0, 38.0]])
    # Attacker moving +x: the defending team's goal is at x = 105 for them...
    # they defend x = 105, so their own-goal distance is 105 - x.
    r = line_break((50.0, 34.0), (80.0, 34.0), defs, True)
    assert r and r["lines_broken"] == 2 and r["deepest_line"] == 0, r
    # Too short: 8 m of progression clears no gate.
    assert line_break((50.0, 34.0), (58.0, 34.0), defs, True) is None
    # Backwards is never a break.
    assert line_break((80.0, 34.0), (50.0, 34.0), defs, True) is None
    # Stops 1 m short of the deepest line: breaks the middle one and only that.
    r2 = line_break((50.0, 34.0), (74.0, 34.0), defs, True)
    assert r2 and r2["lines_broken"] == 1 and r2["deepest_line"] == 1, r2
    # Ends level with the line it crossed rather than 2 m past it -> not a break.
    assert line_break((42.0, 34.0), (56.0, 34.0), defs, True) is None
    # Passer 2 m from the crossing point: a squeeze past a man, not a line break.
    assert line_break((53.0, 34.0), (70.0, 34.0), defs, True) is None
    # Fewer than four defenders observed: refuse, do not cluster three people
    # into three lines and call every pass a break.
    assert line_break((50.0, 34.0), (80.0, 34.0), defs[:3], True) is None

    # --- the registry merges additively and the min() still governs.
    register()
    m = event_metric("events.line_break", 3, n=3, event_type="passSucceeded")
    assert "unsmoothed_line_grouping" in m["caveat_flags"]
    # NOT a tier. EVENT_METRICS registers no `api://` id for any of these five
    # fields - nothing in model/validation/measurements.py measures the event-feed
    # join - so stats.metric() refuses to publish a confidence band for them.
    # See the JUSTIFICATION banner in stats.py. Registering the join's own
    # measurements (JOIN_QUALITY above is the table, unregistered) is what
    # would move these back to a tier.
    assert m["confidence"] == stats.NOT_EVALUATED, m
    # The per-event-type caveat is the whole point of point 4: the same field
    # computed off a 66.7%-resolve `block` is not the same number as one off a
    # 98.1%-resolve `passSucceeded`, and the envelope has to carry that.
    m2 = event_metric("events.pass_direction", n=10, event_type="block")
    m3 = event_metric("events.pass_direction", n=10, event_type="passSucceeded")
    assert "contested_location" in m2["caveat_flags"]
    assert "contested_location" not in m3["caveat_flags"]
    # `team_purity` used to cap both at `prov` - the join leans on team labels
    # that measured 15.4% accurate on our own tracks - and the min() is still
    # doing the capping, one rung lower: with no registered measurement behind
    # the field, NOT_EVALUATED enters the same min() and governs.
    assert m2["confidence"] == m3["confidence"] == stats.NOT_EVALUATED
    assert stats.CAVEATS["team_purity"][0] == "prov"
    assert stats.CAVEATS["contested_location"][0] == "rel"
    register()                                   # idempotent

    print("events.py ok")


# ---------------------------------------------------------------------------
# `--measure` reproduces every number quoted in the docstrings above, off the
# files on disk. Not part of the self-check: it reads a 1.6M-row ground truth
# file and takes ~40 s. Kept in the module rather than in a script so the
# numbers and the code that produced them cannot drift apart.
# ---------------------------------------------------------------------------

GT = "model/validation/soccertrack/118575_1st.tracks.json.gz"
FEED = "data/footage/soccertrack_calib_118575/118575_player_nodes.csv"
# The SoccerTrack ground-truth loader numbers frames as video frame + 3,000,000.
# Left in as an offset rather than "fixed" in the file, because validate.py and
# test_golden.py both match on the shifted ids. It has to come off HERE or every
# event lands 33 hours away from every position and the join silently refuses
# all 1,249 of them - which is what it did the first time this was run.
GT_FRAME_OFFSET = 3_000_000


def _load_gt(path: str):
    import gzip
    with gzip.open(path) as f:
        d = json.load(f)
    by_id: dict = {}
    for x in d["detections"]:
        by_id.setdefault(x["id"], []).append({**x, "f": x["f"] - GT_FRAME_OFFSET})
    return by_id, float(d["fps"])


def _measure(root: str = "..") -> None:
    import collections
    import os
    gt_path, feed_path = os.path.join(root, GT), os.path.join(root, FEED)
    by_id, fps = _load_gt(gt_path)
    pos = positions_from_tracks(by_id, fps)
    one = pos.stride(int(round(1.0 / (pos.t[1] - pos.t[0]))))
    # attacking_directions, not infer_attacking_right per side: two independent
    # per-side calls can return the same direction for both teams, which is
    # impossible in one half, and on real pipeline output they do.
    att, att_flags = stats.attacking_directions(
        {s: (one.XY[:, one.side(s)], one.mask[:, one.side(s)], []) for s in ("home", "away")})
    print("attacking_right (stats.attacking_directions):", att, att_flags)
    tmap = team_map_from_jerseys(feed_path, pos)
    print("team map from jersey sets:", tmap)
    assert tmap, "team map is ambiguous - refuse rather than rotate a team wrongly"

    ev = load_player_nodes(feed_path, tmap, att)
    print(f"events: {len(ev)} first-half")

    # --- 1. join quality, against the jersey oracle -------------------------
    truth = {}
    for j, (s, jn) in enumerate(zip(pos.teams, pos.jerseys)):
        truth[(s, int(jn))] = j
    for rt in (True, False):
        act = join_actors(ev, pos, restrict_team=rt)
        h = w = a = 0
        per = collections.defaultdict(lambda: [0, 0, 0])
        for e, k in zip(ev, act):
            g = truth.get((e.team, e.jersey)) if e.jersey else None
            if g is None:
                continue
            slot = 2 if k is None else (0 if k == g else 1)
            h, w, a = (h + (slot == 0), w + (slot == 1), a + (slot == 2))
            per[e.primary][slot] += 1
        n = h + w + a
        print(f"  restrict_team={rt}: n={n} right {100*h/n:.1f}% wrong {100*w/n:.1f}% "
              f"refused {100*a/n:.1f}% precision {100*h/(h+w):.1f}%")
        if rt:
            print("   per type (n / resolve / precision):")
            for ty, (hh, ww, aa) in sorted(per.items(), key=lambda kv: -sum(kv[1])):
                if hh + ww + aa < 8:
                    continue
                m = hh + ww + aa
                print(f"     {ty:<18} {m:4d}  {(hh+ww)/m:5.3f}  "
                      f"{hh/(hh+ww) if hh+ww else float('nan'):5.3f}")
    act = join_actors(ev, pos)
    d = []
    for e, k in zip(ev, act):
        i = pos.index(e.t)
        if i is None:
            continue
        dd = np.hypot(pos.XY[i, :, 0] - e.xy[0], pos.XY[i, :, 1] - e.xy[1])
        d.append(np.nanmin(np.where(pos.mask[i], dd, np.inf)))
    d = np.array(d)
    print("  nearest-of-22 distance: median %.2f p10 %.2f p25 %.2f p75 %.2f p90 %.2f"
          % tuple(np.percentile(d, [50, 10, 25, 75, 90])))

    # --- 2. direction, against the feed's own human labels ------------------
    LAB = {"forwardPass": "forward", "backwardPass": "backward",
           "sidewaysPass": "sideways"}
    def labelled(e):
        got = [v for k, v in LAB.items()
               if (k + "Succeeded") in e.types or (k + "Failed") in e.types]
        return got[0] if len(got) == 1 else None

    ok = tot = 0
    for e in ev:
        g = labelled(e)
        if g is None or e.recv_xy is None or "passReceived" in e.types:
            continue
        tot += 1
        ok += pass_direction(e.xy, e.recv_xy, att[e.team]) == g
    print(f"  direction from EVENT coords: {ok}/{tot} = {100*ok/tot:.1f}%")

    ok2 = tot2 = 0
    bylen = collections.defaultdict(lambda: [0, 0])
    recv = join_actors([Event(t=e.t, team=e.team, types=e.types, xy=e.recv_xy)
                        for e in ev], pos)
    for e, ka, kb in zip(ev, act, recv):
        g = labelled(e)
        if g is None or e.recv_xy is None or "passReceived" in e.types:
            continue
        if ka is None or kb is None or ka == kb:
            continue
        i = pos.index(e.t)
        p = pass_direction(pos.XY[i, ka], pos.XY[i, kb], att[e.team])
        tot2 += 1
        ok2 += p == g
        L = min(int(np.hypot(e.recv_xy[0] - e.xy[0], e.recv_xy[1] - e.xy[1]) // 10), 4)
        bylen[L][0] += p == g
        bylen[L][1] += 1
    print(f"  direction THROUGH THE JOIN: {ok2}/{tot2} = {100*ok2/tot2:.1f}% "
          f"(both ends resolved on {tot2}/{tot})")
    for L in sorted(bylen):
        hh, nn = bylen[L]
        print(f"     {L*10:2d}-{L*10+10 if L < 4 else 99} m: {hh:3d}/{nn:3d} = {100*hh/nn:5.1f}%")

    # --- 3. progressive -----------------------------------------------------
    pr = [progressive(e.xy, e.recv_xy, att[e.team], e.types) for e in ev
          if e.recv_xy is not None and "passReceived" not in e.types]
    pr = [p for p in pr if p]
    n_pg = sum(p["progressive"] for p in pr)
    print(f"  progressive (StatsBomb 25%): {n_pg}/{len(pr)} = {100*n_pg/len(pr):.1f}% "
          f"of open-play passes, median gain {np.median([p['gained_m'] for p in pr]):.1f} m")

    # --- 4. line breaks -----------------------------------------------------
    seen = collections.Counter()
    for e, ka in zip(ev, act):
        if e.recv_xy is None or "passReceived" in e.types or ka is None:
            continue
        i = pos.index(e.t)
        if i is None:
            continue
        opp = "away" if e.team == "home" else "home"
        sel = (pos.teams == opp) & pos.mask[i]
        # A FAILED pass's `relative` player is the INTENDED receiver, so its
        # endpoint is where the ball was aimed, not where it went. Counting the
        # two together reports intentions as outcomes - see the docstring.
        kind = "failed" if any("Failed" in t for t in e.types) else "completed"
        seen[(kind, "tried")] += 1
        if line_break(e.xy, e.recv_xy, pos.XY[i, sel], att[e.team]):
            seen[(kind, "break")] += 1
    for kind in ("completed", "failed"):
        b, t_ = seen[(kind, "break")], seen[(kind, "tried")]
        print(f"  line breaks, {kind:9s}: {b}/{t_} = {100*b/t_:.1f}%")
    print("  stability under +/-1 s:", line_break_stability(ev, act, pos, att))

    # --- 5. shape at turnover ----------------------------------------------
    print("  shape at turnover (median delta vs that team's match median):")
    for s, v in shape_at_turnover(ev, pos, att).items():
        print(f"     {s}: {json.dumps(v, sort_keys=True)}")


JOB = "data/jobs/crop600/tracks.json"
CROP_CALIB = "model/validation/calib_118575_crop.json"
# data/jobs/crop600 is video frames 30000-30599 renumbered from 0.
JOB_FRAME0 = 30000


def _measure_pipeline(root: str = "..") -> None:
    """The same join on OUR OWN pipeline output instead of ground truth.

    24 s, 12 events, so nothing here is a statistic. It is a smoke test of two
    things that ground truth cannot test, and it fails one of them.
    """
    import os
    import calibrators
    with open(os.path.join(root, JOB)) as f:
        d = json.load(f)
    cal = calibrators.load(os.path.join(root, CROP_CALIB))
    by = {}
    for x in d["detections"]:
        by.setdefault(x["id"], []).append({**x, "f": x["f"] + JOB_FRAME0})
    pos = positions_from_tracks(by, float(d["fps"]), cal)
    one = pos.stride(int(round(1.0 / (pos.t[1] - pos.t[0]))))
    att, att_flags = stats.attacking_directions(
        {s: (one.XY[:, one.side(s)], one.mask[:, one.side(s)], []) for s in ("home", "away")})
    print(f"pipeline: {pos.XY.shape[1]} track columns for 22 players, "
          f"t {pos.t[0]:.0f}-{pos.t[-1]:.0f} s, attacking_right {att} {att_flags}")
    ev = [e for e in load_player_nodes(os.path.join(root, FEED),
                                       {"31425": "home", "31424": "away"}, att)
          if pos.t[0] <= e.t <= pos.t[-1]]
    dn = []
    for e in ev:
        i = pos.index(e.t)
        dd = np.hypot(pos.XY[i, :, 0] - e.xy[0], pos.XY[i, :, 1] - e.xy[1])
        dn.append(float(np.min(np.where(pos.mask[i], dd, np.inf))))
    print(f"  n={len(ev)} nearest tracked player (any team): median {np.median(dn):.2f} m")
    for rt in (True, False):
        n = sum(a is not None for a in join_actors(ev, pos, restrict_team=rt))
        print(f"  restrict_team={rt}: resolved {n}/{len(ev)}")
    lab = [pos.teams[int(np.argmin(np.where(pos.mask[pos.index(e.t)],
           np.hypot(pos.XY[pos.index(e.t), :, 0] - e.xy[0],
                    pos.XY[pos.index(e.t), :, 1] - e.xy[1]), np.inf)))] for e in ev]
    print(f"  team label on the nearest player: {sorted(set(lab))} for "
          f"{sorted({e.team for e in ev})} events -> "
          f"{'COLLAPSED, team restriction is unusable' if len(set(lab)) == 1 else 'ok'}")


if __name__ == "__main__":
    import sys
    if "--measure" in sys.argv:
        _measure()
    elif "--pipeline" in sys.argv:
        _measure_pipeline()
    else:
        _demo()
