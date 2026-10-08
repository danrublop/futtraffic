"""TEAM ground truth for the MOT tracks, on ALL TEN STC2025 matches.

WHY THIS EXISTS
    `team_eval.py` scored team assignment on ONE match (118575) against the
    per-detection `team` field in `model/validation/soccertrack/118575_1st.tracks.json.gz`,
    and published 0.6967 per detection / 0.9300 per track. Nobody knew whether
    that was a property of the METHOD or a property of 118575.

    ST2 ships team truth for all ten matches, already on this machine. This file
    is the join that makes it usable: MOT id -> ST2 player -> team + shirt.

WHERE THE TRUTH ACTUALLY LIVES, and it is not where the brief said
    `raw/<m>/<m>_player_nodes.csv` is an EVENT table - one row per touch, so a
    player is present only when they touch the ball. It carries `player_id`,
    `back_number`, `team_id`. It is the ROSTER source here and nothing more.

    `raw/<m>/<m>_tracker_box_data.xml` is the DENSE feed: every frame of the
    whole match, every player on the pitch, `playerId` + normalised pitch `loc`.
    That is what the positional join runs against, and it is why the event
    table's sparsity never becomes a problem: the sparse file supplies the team
    label for a player id, the dense file supplies the geometry that says which
    MOT id that player is. `_tracker_box_metadata.xml` closes the last gap -
    it is the only file that says which team_id is `home` and which is `away`,
    and it covers all 22 on-pitch players even when the event table does not.
    `coverage()` reports, per match, how many of the 22 the event table alone
    would have left unlabelled.

THE JOIN, AND WHY IT IS SELF-FITTED
    Two unknowns: (1) which XML frame is clip frame 0, per match and per PERIOD
    - the frameNumber series restarts each half - and (2) which of MOT's ids
    1..22 is which ST2 playerId. The MOT id order is NOT the identity and NOT
    sorted by team; on 118575 MOT 21 is the home goalkeeper and MOT 1..4 split
    3 away / 1 home.

    The obvious route is the vendor chain (pitch --H--> remapped --mapx-->
    panorama, `st2_overlay.projector`). IT DOES NOT COVER THE SET: 132831 ships
    a rank-2 homography (det 5e-13) and 132877's joins at 215 px against a
    second-best of 294 px with 35% per-frame agreement, i.e. not at all. So the
    map is fitted from the data instead, in two calibration-free steps:

      OFFSET   the mean pitch-x of the 22 players and the mean pixel-u of the 22
               MOT feet are the same signal in two spaces. Correlate the series
               over every candidate offset in both halves and take the peak.
               No calibration, no permutation - a mean is permutation-invariant,
               which is exactly why this step can run before the join.

      PERMUTATION  alternate {fit a pitch->pixel map on the current
               correspondence} and {re-solve the correspondence by Hungarian on
               the aggregated cost}. Initialised by voting the per-frame x-rank
               order. Converges in 2-3 iterations on all ten.

    The offset is then RE-SOLVED geometrically under the fitted map, because a
    scalar correlation is good to tens of frames and tens of frames is a second
    of running. Costs drop 2-10x; 117093 goes from 813 px to 67 px.

    THE MAP IS NOT A HOMOGRAPHY - see `_fit`. These panoramas are stitched and a
    single homography leaves 330-344 px on 117092 and 128057.

    THE VENDOR CHAIN IS STILL USED - as the CHECK, not the method. On 118575 the
    self-fit and the vendor projector return the same permutation for all 22
    ids, from independent geometry. `_selfcheck` pins that.

WHAT A WRONG JOIN LOOKS LIKE, because that is the failure that matters here
    A wrong join does not crash. It produces a plausible accuracy. Four numbers
    are reported for every match and all four are gated in `_ok`, not eyeballed:

      cost_px            mean assigned distance under the fitted map, 24-77 px
      cost2_px           the SECOND-best assignment, 225-290 px, i.e. 3.1-11x
                         worse. A join one swap away from an equally good
                         alternative is not a join
      frame_agreement    solve each frame alone; how often it reproduces the
                         aggregate answer. 0.889-0.966 on the ten
      11/11              the joined ids must split exactly 11 home / 11 away.
                         A join that scrambles identities does not stay balanced

    And `team_eval._selfcheck_clips` runs the other end of it: 118575 re-scored
    with its team labels SHUFFLED must fall from 0.88 to ~0.55.

A JOIN CAN BE RIGHT ABOUT WHO AND WRONG ABOUT WHERE - HANDEDNESS, added 2026-09-14
    The fitted map is free enough to represent a MIRROR: a feed whose y runs the other
    way joins every id correctly, passes all four checks above, and puts every player
    on the wrong side of the pitch. 132877 did exactly that - the review of N26 found
    ST2's own boxes match its feed only once y -> 1 - y (docs/experiments/n26-map-oracle.md §4) -
    and it is why ST2's own chain "does not join" there. Identity and team are fine;
    positions are not. So every join records the sign of its fitted map's Jacobian
    determinant (`feed_handedness`) and of ST2's own chain at the same pitch points
    (`chain_handedness`, None where the chain is unusable), and `feed_mirrored` compares
    them - against ST2_HANDEDNESS, the sign every usable chain shows, where the chain
    is not there to compare with. Two of the ten read mirrored: 132877 against its own
    chain, 132831 against the shared sign. `_ok` is about identity and does not change;
    `dist_eval.stc_gt`, which serves positions, refuses a mirrored feed.

INDEPENDENT CONFIRMATION ON 118575
    The soccertrack gz that `team_eval.py` has always scored against turns out
    to carry `team` as a pure function of the id - gz ids 1-11 are home, 12-22
    are away, with no exception in 1.59 M detections. That is a CONVENTION, not
    an annotation. It is nonetheless correct: routed through the independently
    solved MOT->gz permutation in `model/validation/out/calib_hypothesis_join.json`,
    it agrees with ST2's own team for all 22 of 22 ids - after a global flip,
    because the gz's 'home' is ST2's away side. `_selfcheck` asserts 22/22, and
    that is the guard against a plausible wrong answer.

    THE FIRST VERSION OF THIS JOIN WAS WRONG and this check is what caught it.
    `linear_sum_assignment` returns (row, col); the rows are ST2 players and the
    columns are MOT ids, and reading them the other way round produced a join
    with a 17x margin, 98.8% per-frame agreement, and 11 home / 11 away - every
    quality number a good join has. Only the cross-source check moved: 8 of 22
    instead of 0 or 22. Nothing else would have found it.

WHAT THE EVENT TABLE COST, since the brief was built on it
    Nothing, and it could have cost a day. `player_nodes.csv` has 27,739 rows
    across the ten matches but only 25,818 carry a player_id at all - 7% are
    blank - so "every one with a jersey number and a team" is not true of the
    file. It does not matter: all 220 joined on-pitch players appear in it, its
    team_id and back_number never disagree with the metadata roster (0 conflicts
    in 220), and the metadata roster covers the players it would have missed.
    `--coverage` prints this per match rather than asserting it away.

Usage
    python model/validation/st2_team_gt.py --selfcheck
    python model/validation/st2_team_gt.py --build      # solve all ten, ~3 min
    python model/validation/st2_team_gt.py --coverage   # event-table sparsity
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = _ST2 + "/raw"
MOT = _ST2 + "/mot"
OUT = os.path.join(ROOT, "model/validation/out/st2_team_gt.json")

SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132831", "132877")

#: One line per element in these files, so a line scan beats ElementTree by 40x
#: (1.3 s vs ~60 s on a 220 MB file) and needs no cache.
_FRAME = re.compile(rb'<frame matchTime="[-0-9]+" frameNumber="([0-9]+)" '
                    rb'eventPeriod="([A-Z_]+)"')
_PLAYER = re.compile(rb'<player playerId="([0-9]+)" loc="\[([-0-9.]+), ?([-0-9.]+)\]"')

EVERY = 50    #: clip frames between join samples: 120 frames over a 4 min clip
RANSAC_PX = 20.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- loading

def dense_loc(seq: str) -> dict[tuple[str, int], dict[int, tuple[float, float]]]:
    """{(period, frameNumber): {playerId: (x, y) normalised}} for the WHOLE match.

    Keyed by (period, frameNumber) and not by frameNumber alone: both halves
    restart at 251, so a frameNumber-only key silently overwrites the first half
    with the second and every offset sweep then searches half the match.
    """
    out: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}
    cur: dict[int, tuple[float, float]] = {}
    with open(f"{RAW}/{seq}/{seq}_tracker_box_data.xml", "rb") as fh:
        for line in fh:
            s = line.lstrip()
            if s[:6] == b"<frame":
                m = _FRAME.match(s)
                if m:
                    cur = {}
                    out[(m.group(2).decode(), int(m.group(1)))] = cur
            elif s[:7] == b"<player":
                m = _PLAYER.match(s)
                if m:
                    cur[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def mot_feet(seq: str) -> dict[int, dict[int, tuple[float, float]]]:
    """{clip frame: {mot id: foot (u,v)}}. MOT is top-left x,y,w,h.

    The foot is (x + w/2, y + h). Reading MOT as bottom-centre puts every point
    a full player height high, which is metres of pitch at the far side and
    still produces a join - a worse one, silently.
    """
    out: dict[int, dict[int, tuple[float, float]]] = collections.defaultdict(dict)
    with open(f"{MOT}/{seq}.txt") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 6:
                continue
            x, y, w, h = (float(v) for v in p[2:6])
            out[int(p[0])][int(p[1])] = (x + w / 2.0, y + h)
    return dict(out)


def roster(seq: str) -> dict:
    """{player_id: {team_id, side, back_number, name}} + the team_id -> side map.

    `back_number` and `team_id` come from `player_nodes.csv`, as briefed. `side`
    exists ONLY in the metadata XML - the event table names teams but never says
    which is home - and the metadata is also the fallback for a player who never
    touched the ball, so it is loaded either way and the two are cross-checked
    rather than one being trusted. Disagreements are returned, not swallowed.
    """
    md = ET.parse(f"{RAW}/{seq}/{seq}_tracker_box_metadata.xml").getroot()
    side = {t.get("id"): t.get("side") for t in md.find("teams")}
    # `position` is the team sheet's LISTED position (GK/CB/CM/CF/...), static
    # per player. st2_cache.roles maps it to a line for the G3B role arm.
    meta = {int(p.get("id")): (p.get("teamId"), p.get("shirtNumber"),
                               p.get("nameEn") or p.get("name"), p.get("position") or "")
            for p in md.find("players")}

    events: dict[int, tuple[str, str]] = {}
    n_rows = n_named = 0
    with open(f"{RAW}/{seq}/{seq}_player_nodes.csv") as fh:
        for r in csv.DictReader(fh):
            n_rows += 1
            pid = r.get("player_id") or ""
            if not pid.isdigit():
                continue     # ~7% of rows carry no player_id at all, so "every
                             # row has a jersey number and a team" is not true
            n_named += 1
            events[int(pid)] = (r["team_id"], r["back_number"])

    conflicts = [pid for pid, (tid, bn) in events.items()
                 if pid in meta and (tid, bn) != meta[pid][:2]]
    out = {}
    for pid, (tid, shirt, name, pos) in meta.items():
        etid, ebn = events.get(pid, (None, None))
        out[pid] = {"team_id": etid or tid, "back_number": ebn or shirt,
                    "side": side.get(etid or tid), "name": name, "position": pos,
                    "in_event_table": pid in events}
    return {"players": out, "sides": side, "event_rows": n_rows,
            "event_rows_with_player": n_named,
            "event_players": len(events), "roster_players": len(meta),
            "conflicts": conflicts,
            "event_only": sorted(set(events) - set(meta))}


# ---------------------------------------------------------------- the join

def _offset(seq: str, L: dict, M: dict, n_clip: int) -> tuple[float, str, int]:
    """(correlation, period, offset) so that clip frame f is (period, f+offset).

    Mean pitch-x against mean pixel-u. Both are the same crowd of 22 players
    sloshing up and down the pitch, in two spaces related by a monotone map, and
    a MEAN IS PERMUTATION-INVARIANT - so this runs before the identities are
    known and cannot be corrupted by getting them wrong.
    """
    u = np.array([np.mean([v[0] for v in M[f].values()]) if f in M else np.nan
                  for f in range(n_clip)])
    a = (u - np.nanmean(u)) / (np.nanstd(u) + 1e-9)
    best = None
    for period in ("FIRST_HALF", "SECOND_HALF"):
        fk = sorted(k[1] for k in L if k[0] == period)
        if len(fk) < n_clip:
            continue
        lo, hi = fk[0], fk[-1]
        st = np.full(hi - lo + 1, np.nan)
        for f in range(lo, hi + 1):
            g = L.get((period, f))
            if g:
                st[f - lo] = np.mean([v[0] for v in g.values()])
        for off in range(0, len(st) - n_clip):
            b = st[off:off + n_clip]
            if np.isnan(b).any():
                continue
            b = (b - b.mean()) / (b.std() + 1e-9)
            r = float(np.nanmean(a * b))
            if best is None or r > best[0]:
                best = (r, period, off + lo)
    if best is None:
        raise SystemExit(f"{seq}: no period long enough for a {n_clip} frame clip")
    return best


def _samples(L: dict, M: dict, period: str, off: int, n_clip: int, every: int):
    """[(A normalised loc 22x2, B mot feet 22x2)] on a fixed playerId order.

    Frames whose on-pitch SET differs from the first sampled frame are dropped.
    A substitution changes which 22 ids are present, and silently reusing the
    row order across that boundary shifts every row of the aggregated cost by
    one - which still solves, to the wrong answer.
    """
    ids = None
    out = []
    for cf in range(0, n_clip, every):
        g, mm = L.get((period, cf + off)), M.get(cf)
        if not g or len(g) != 22 or not mm or len(mm) != 22:
            continue
        k = sorted(g)
        if ids is None:
            ids = k
        if k != ids:
            continue
        out.append((np.array([g[i] for i in ids], float),
                    np.array([mm[i] for i in range(1, 23)], float)))
    return ids, out


def _poly(A: np.ndarray) -> np.ndarray:
    """[1, x, y, x^2, xy, y^2] - the design matrix of the pitch -> pixel fit."""
    x, y = A[:, 0], A[:, 1]
    return np.column_stack([np.ones(len(A)), x, y, x * x, x * y, y * y])


def _fit(src: np.ndarray, dst: np.ndarray):
    """Fit pitch (normalised) -> clip pixel, and return it as a callable.

    NOT A HOMOGRAPHY, and that is the whole point. These panoramas are STITCHED:
    `st2_align.py` measures a single homography on the raw panorama at 103 px
    median with 11 of 65 inliers, and the product's own overlay was fixed this
    week for "drawing chords where the pitch curves". A homography-only join
    inherits that error - on 117092 and 128057 it lands at 330-344 px, which is
    a third of the pitch and leaves no margin worth reading.

    A quadratic in (x, y) has six terms per axis, absorbs the stitch curvature,
    and is fitted by least squares on 2,640 correspondences - so it is nowhere
    near flexible enough to fit an arbitrary wrong permutation, which is what
    would make a curved map dangerous here. Costs drop 3-8x on the stitched
    matches and the permutation does not change on any of the ten.
    """
    K, *_ = np.linalg.lstsq(_poly(src), dst, rcond=None)
    return lambda A: _poly(np.asarray(A, float)) @ K


#: The handedness every usable ST2 chain shows, pitch (x, y) -> clip pixel. `_selfcheck`
#: asserts it holds for all of them, so a match without a usable chain is judged by it.
ST2_HANDEDNESS = 1

#: Mirrored feeds kept readable ONLY so measurements registered on them before
#: 2026-09-14 still reproduce. Never add one to a new result.
MIRRORED_FEEDS_REPRODUCED = ("132877",)


def _det_sign(fn, pts: np.ndarray, eps: float) -> int:
    """Majority sign of a pitch -> pixel map's Jacobian determinant over `pts`."""
    dx = (fn(pts + [eps, 0.0]) - fn(pts - [eps, 0.0])) / (2 * eps)
    dy = (fn(pts + [0.0, eps]) - fn(pts - [0.0, eps])) / (2 * eps)
    return int(np.sign(np.sum(np.sign(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]))))


def _handedness(seq: str, fn) -> tuple[int, int | None]:
    """(fitted map's sign, ST2 chain's sign or None) over an interior pitch grid."""
    g = np.stack(np.meshgrid(np.linspace(0.15, 0.85, 8), np.linspace(0.15, 0.85, 6)), -1).reshape(-1, 2)
    fit = _det_sign(fn, g, 1e-3)
    H = np.load(f"{RAW}/{seq}/{seq}_homography.npy")
    if abs(np.linalg.det(H)) < 1e-6:
        return fit, None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import st2_mot as T  # noqa: PLC0415
    from st2_overlay import pitch_of, projector  # noqa: PLC0415
    P, proj = pitch_of(seq), projector(seq, T._wh(seq))
    scale = np.array([P.length, P.width])

    def chain(A):
        px, ok = proj(np.asarray(A, float) * scale)
        return np.where(ok[:, None], px, np.nan)
    # 3 m steps: the chain rounds to whole remap cells, and a finer step reads plateaus.
    eps = 3.0 / P.length
    dx = (chain(g + [eps, 0.0]) - chain(g - [eps, 0.0])) / (2 * eps)
    dy = (chain(g + [0.0, eps]) - chain(g - [0.0, eps])) / (2 * eps)
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    det = det[np.isfinite(det)]
    return fit, (int(np.sign(np.sum(np.sign(det)))) if len(det) else None)


def feed_mirrored(d: dict) -> bool:
    """True if the joined feed's handedness disagrees with ST2's camera - or was never
    checked, which a record written before the check reads as."""
    if d.get("feed_handedness") is None:
        return True
    ref = d.get("chain_handedness") or ST2_HANDEDNESS
    return d["feed_handedness"] != ref


def _permutation(frames, iters: int = 8):
    """{st2 row -> mot column}, by alternating a map fit and Hungarian.

    Initialised by voting the per-frame x-rank order, which is right for most
    ids on a touchline panorama and only has to be close enough for the first
    fit to be sane. RANSAC homography for the first pass, because least squares
    on a partly-wrong correspondence is dragged by its outliers; the quadratic
    takes over once the correspondence has stopped moving.
    """
    import cv2  # noqa: PLC0415
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    V = np.zeros((22, 22))
    for A, B in frames:
        ra = np.argsort(np.argsort(A[:, 0]))
        rb = list(np.argsort(np.argsort(B[:, 0])))
        V[np.arange(22), [rb.index(x) for x in ra]] += 1
    r, c = linear_sum_assignment(-V)
    perm = dict(zip(r.tolist(), c.tolist()))

    fn = None
    for it in range(iters):
        src = np.vstack([A for A, _ in frames])
        dst = np.vstack([B[[perm[i] for i in range(22)]] for _, B in frames])
        if it == 0:
            H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
            if H is None:
                raise SystemExit("homography fit failed - the join has no answer here")

            def fn(A, H=H):
                P = np.c_[np.asarray(A, float), np.ones(len(A))] @ H.T
                return P[:, :2] / P[:, 2:]
        else:
            fn = _fit(src, dst)
        C = np.zeros((22, 22))
        for A, B in frames:
            C += np.linalg.norm(fn(A)[:, None] - B[None], axis=2)
        C /= len(frames)
        r, c = linear_sum_assignment(C)
        new = dict(zip(r.tolist(), c.tolist()))
        if new == perm and it:
            break
        perm = new

    C2 = C.copy()
    C2[r, c] = 1e9
    r2, c2 = linear_sum_assignment(C2)
    agree = tot = 0
    for A, B in frames:
        rr, cc = linear_sum_assignment(np.linalg.norm(fn(A)[:, None] - B[None], axis=2))
        agree += sum(perm[int(x)] == int(y) for x, y in zip(rr, cc))
        tot += 22
    return perm, fn, {"cost_px": round(float(C[r, c].mean()), 2),
                      "cost2_px": round(float(C2[r2, c2].mean()), 2),
                      "frame_agreement": round(agree / max(tot, 1), 4),
                      "frame_n": tot // 22}


def _assigned_cost(frames, perm, fn) -> float:
    """Mean distance from each ST2 player to the MOT id the permutation gives it."""
    tot, n = 0.0, 0
    order = [perm[i] for i in range(22)]
    for A, B in frames:
        tot += float(np.linalg.norm(fn(A) - B[order], axis=1).mean())
        n += 1
    return tot / max(n, 1)


def join(seq: str, every: int = EVERY, verbose: bool = True) -> dict:
    """Solve one match: MOT id -> ST2 player -> side + shirt number."""
    L, M = dense_loc(seq), mot_feet(seq)
    n_clip = max(M) + 1
    corr, period, off0 = _offset(seq, L, M, n_clip)
    # The correlation offset is good to tens of frames, not to the frame: it is
    # a 1 d.o.f. summary of a 44 d.o.f. scene. Tens of frames is a second of
    # running, which is metres, which inflates every cost below and hides
    # whether the permutation is right. So refine it GEOMETRICALLY, under the
    # map the permutation just produced, and re-solve. Costs on the ten matches
    # drop 2-20x doing this; 117093 goes from 813 px to a real join.
    off = off0
    for _ in range(3):
        ids, frames = _samples(L, M, period, off, n_clip, every)
        if len(frames) < 20:
            raise SystemExit(f"{seq}: only {len(frames)} usable frames - not a join")
        perm, fn, q = _permutation(frames)
        cand = []
        for o in range(off - 60, off + 61):
            _i, fr = _samples(L, M, period, o, n_clip, every)
            if _i != ids or len(fr) < 20:
                continue
            cand.append((_assigned_cost(fr, perm, fn), o))
        if not cand:
            break
        newoff = min(cand)[1]
        if newoff == off:
            break
        off = newoff
    ids, frames = _samples(L, M, period, off, n_clip, every)
    perm, fn, q = _permutation(frames)
    q["offset_shift"] = off - off0
    q["feed_handedness"], q["chain_handedness"] = _handedness(seq, fn)

    R = roster(seq)["players"]
    mot2pid = {perm[i] + 1: ids[i] for i in range(22)}
    team = {m: R[p]["side"] for m, p in mot2pid.items() if p in R}
    jersey = {m: R[p]["back_number"] for m, p in mot2pid.items() if p in R}
    counts = collections.Counter(team.values())
    d = {"seq": seq, "period": period, "offset": off, "offset_corr": round(corr, 4),
         "clip_frames": n_clip, "join_frames": len(frames),
         "mot_to_player": {str(k): v for k, v in sorted(mot2pid.items())},
         "team": {str(k): v for k, v in sorted(team.items())},
         "jersey": {str(k): v for k, v in sorted(jersey.items())},
         "home_n": counts.get("home", 0), "away_n": counts.get("away", 0),
         "identity_perm": all(perm[i] + 1 == i + 1 for i in range(22)),
         **q}
    if verbose:
        log(f"{seq}: {period} clip0 -> frame {off} (r={corr:.3f})  "
            f"cost {q['cost_px']:.1f} px vs 2nd {q['cost2_px']:.0f} px  "
            f"agree {q['frame_agreement']:.3f}  "
            f"{counts.get('home', 0)}H/{counts.get('away', 0)}A")
    return d


def build(out: str = OUT, every: int = EVERY) -> dict:
    got = {s: join(s, every) for s in SEQS}
    bad = [s for s, d in got.items() if not _ok(d)]
    mirrored = [s for s, d in got.items() if feed_mirrored(d)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(got, open(out, "w"), indent=1)
    log(f"-> {out}" + (f"  QUESTIONABLE: {bad}" if bad else "  all ten clean")
        + (f"  MIRRORED FEED (positions unusable): {mirrored}" if mirrored else ""))
    return got


def _ok(d: dict) -> bool:
    """The four conditions a join must meet before any accuracy built on it counts."""
    return (d["home_n"] == 11 and d["away_n"] == 11
            and d["frame_agreement"] >= 0.80
            and d["cost2_px"] > 3.0 * d["cost_px"])


def load(path: str = OUT) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing - run: python model/validation/st2_team_gt.py --build")
    return json.load(open(path))


# ---------------------------------------------------------------- sparsity

def coverage() -> dict:
    """Would the EVENT table alone have been enough? Per match, with denominators.

    The answer is no for every match and it does not matter, because the roster
    that fills the gap sits in the same directory. Reported rather than hidden:
    a brief that assumed `player_nodes.csv` was the whole story would have
    dropped these tracks, or worse, guessed them.
    """
    gt = load()
    out = {}
    for s in SEQS:
        R = roster(s)
        on_pitch = [int(p) for p in gt[s]["mot_to_player"].values()]
        seen = sum(R["players"][p]["in_event_table"] for p in on_pitch)
        out[s] = {"event_rows": R["event_rows"],
                  "event_rows_with_player": R["event_rows_with_player"],
                  "roster_players": R["roster_players"],
                  "players_with_any_event": R["event_players"],
                  "joined_on_pitch": len(on_pitch),
                  "of_those_in_event_table": seen,
                  "unlabelled_by_events_alone": len(on_pitch) - seen,
                  "roster_team_conflicts": len(R["conflicts"]),
                  "event_ids_not_in_roster": len(R["event_only"])}
    tot = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))}
    return {"per_match": out, "total": tot}


# ---------------------------------------------------------------- checks

def _vendor_perm(seq: str, every: int = EVERY):
    """The same permutation via ST2's SHIPPED pitch->pixel chain. Check only.

    Independent of `_permutation` in every part that could be wrong: it uses the
    vendor homography and remap tables rather than a fit, so agreement between
    the two is not two runs of one mistake. Returns None where the vendor chain
    is unusable (132831 ships a rank-2 homography; 132877's does not join).
    """
    import cv2  # noqa: PLC0415
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from st2_overlay import pitch_of, projector  # noqa: PLC0415

    H = np.load(f"{RAW}/{seq}/{seq}_homography.npy")
    if abs(np.linalg.det(H)) < 1e-6:
        return None
    L, M = dense_loc(seq), mot_feet(seq)
    n_clip = max(M) + 1
    _corr, period, off = _offset(seq, L, M, n_clip)
    ids, frames = _samples(L, M, period, off, n_clip, every)
    P = pitch_of(seq)
    cap = cv2.VideoCapture(f"{MOT}/clips/{seq}.mp4")
    wh = (int(cap.get(3)), int(cap.get(4)))
    cap.release()
    proj = projector(seq, wh)
    C, n = np.zeros((22, 22)), 0
    for A, B in frames:
        px, ok = proj(np.c_[A[:, 0] * P.length, A[:, 1] * P.width])
        if not ok.all():
            continue
        C += np.linalg.norm(px[:, None] - B[None], axis=2)
        n += 1
    if not n:
        return None
    r, c = linear_sum_assignment(C / n)
    return {int(c[i]) + 1: ids[int(r[i])] for i in range(22)}


def _selfcheck() -> None:
    """Three checks. The first two are cheap; the third is the one that matters.

    1. The permutation solver recovers a KNOWN shuffle from synthetic geometry.
    2. Every solved match is 11 home / 11 away with a 3x margin and >=0.80
       per-frame agreement - `_ok`, applied to the cached solve.
    3. 118575, against two INDEPENDENT sources that never touch this file's
       method: the vendor pitch->pixel chain must return the same 22 ids, and
       the soccertrack gz's team (routed through the separately solved MOT->gz
       permutation in calib_hypothesis_join.json) must agree with ST2's side for
       all 22, up to one global flip. A join that quietly scrambles identities
       still scores a plausible accuracy; only this catches it.
    """
    import gzip  # noqa: PLC0415

    # 1. synthetic: a known homography, a known shuffle, must come back exactly.
    rng = np.random.default_rng(0)
    Htrue = np.array([[30.0, -4.0, 900.0], [1.0, 12.0, 300.0], [0.0, 0.004, 1.0]])
    shuffle = rng.permutation(22)
    frames = []
    for _ in range(40):
        A = rng.uniform([0.02, 0.02], [0.98, 0.98], (22, 2))
        P = np.c_[A, np.ones(22)] @ Htrue.T
        B = (P[:, :2] / P[:, 2:]) + rng.normal(0, 1.5, (22, 2))
        frames.append((A, B[np.argsort(shuffle)]))
    perm, _H, q = _permutation(frames)
    assert all(perm[i] == int(shuffle[i]) for i in range(22)), (perm, shuffle)
    assert q["cost2_px"] > 3 * q["cost_px"], q
    # A mirrored feed joins just as well - and its fitted map has the opposite handedness.
    grid = np.stack(np.meshgrid(np.linspace(0.15, 0.85, 8), np.linspace(0.15, 0.85, 6)), -1).reshape(-1, 2)
    mframes = [(A * [1.0, -1.0] + [0.0, 1.0], B) for A, B in frames]
    mperm, mfn, mq = _permutation(mframes)
    assert all(mperm[i] == int(shuffle[i]) for i in range(22)) and mq["cost2_px"] > 3 * mq["cost_px"], \
        "a mirrored feed passes every identity check"
    assert _det_sign(_H, grid, 1e-3) == -_det_sign(mfn, grid, 1e-3), "and only handedness tells it apart"
    # and it must FAIL loudly on noise, not return a confident permutation
    noise = [(rng.uniform(0, 1, (22, 2)), rng.uniform(0, 3000, (22, 2)))
             for _ in range(40)]
    qn = _permutation(noise)[2]
    assert qn["cost2_px"] < 3 * qn["cost_px"], qn

    gt = load()
    # 2. every match passes its own quality bar
    for s in SEQS:
        d = gt[s]
        assert _ok(d), f"{s} join failed its own gate: {d}"
        assert len(set(d["mot_to_player"].values())) == 22, s
    # ...and handedness: every usable chain shares one sign, and exactly the two feeds
    # found mirrored read mirrored. A change to either is a finding, not noise.
    chains = {d["chain_handedness"] for d in gt.values() if d.get("chain_handedness") is not None}
    assert chains == {ST2_HANDEDNESS}, chains
    assert sorted(s for s in SEQS if feed_mirrored(gt[s])) == ["132831", "132877"], \
        {s: (gt[s].get("feed_handedness"), gt[s].get("chain_handedness")) for s in SEQS}

    # 3. 118575 against two independent sources
    v = _vendor_perm("118575")
    assert v is not None, "vendor chain unusable on 118575 - check lost"
    mine = {int(k): v2 for k, v2 in gt["118575"]["mot_to_player"].items()}
    assert v == mine, f"self-fit and vendor chain disagree: {v} vs {mine}"

    p = os.path.join(ROOT, "model/validation/out/calib_hypothesis_join.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing (model/validation/out/ is gitignored) - "
                         "run: python model/validation/calib_hypothesis.py --join")
    perm_gz = json.load(open(p))["perm"]
    d = json.load(gzip.open(os.path.join(
        ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")))
    gz_team: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in d["detections"]:
        gz_team[str(r["id"])][r["team"]] += 1
    same = sum(gz_team[perm_gz[str(m)]].most_common(1)[0][0] == gt["118575"]["team"][str(m)]
               for m in range(1, 23))
    assert same in (0, 22), (
        f"ST2 side and the soccertrack gz team agree on only {same}/22 ids. "
        "One of the two joins is wrong, and a wrong join still scores.")
    print(f"st2_team_gt.py ok  (gz agreement {same}/22, "
          f"{'flipped naming' if same == 0 else 'same naming'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="solve all ten and cache")
    ap.add_argument("--seq", help="solve one match and print it")
    ap.add_argument("--coverage", action="store_true",
                    help="event-table sparsity, per match")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.build:
        build()
    elif a.seq:
        print(json.dumps(join(a.seq), indent=1))
    elif a.coverage:
        print(json.dumps(coverage(), indent=1))
    else:
        print(json.dumps(load(), indent=1))
