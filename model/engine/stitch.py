"""Offline tracklet stitching: join the fragments that are the same player.

An online tracker cannot look ahead; this runs after it, on the whole file.
  - temporal exclusion: two tracklets sharing a frame are different players
  - physical gate, in METRES through the calibration: MAX_SPEED x gap + SLACK_M,
    for gaps up to MAX_GAP_S
  - cost: pitch distance after coasting each end (COAST); cheapest join first,
    and a component costs its worst junction
  - team gate: two components on opposite team labels never merge
split() severs teleports first. Stitching buys fewer fragments with more
contamination; IDF1 rewards that welding and cannot referee it.

Manual: docs/reference/detection-tracking.md#stitch (evidence: #evidence-stitching)
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

#: Sprint ceiling, m/s. Measured inert on 118575 (cheapest-first ordering rarely
#: reaches an implausible pair); kept as a rail. detection-tracking-evidence (not published)#evidence-stitching
MAX_SPEED = 9.0

#: Position slack, metres, on top of the speed budget: calibration error, not motion.
SLACK_M = 4.0

#: Longest gap to bridge, seconds: the one knob that trades fragments against identity.
MAX_GAP_S = 10.0


class _Tracklet:
    """One tracker id: its frames, its pitch positions in metres, its two ends."""

    __slots__ = ("id", "f", "xy", "start", "end")

    def __init__(self, tid: int, f: np.ndarray, xy: np.ndarray):
        self.id, self.f, self.xy = tid, f, xy
        self.start, self.end = int(f[0]), int(f[-1])


#: Implied speed, m/s, above which one tracker id is severed into two: a teleport
#: threshold, not a motion one. Much lower and the cut measures far-side pixel jitter
#: through the calibration instead of players. detection-tracking-evidence (not published)#evidence-stitching
SPLIT_SPEED = 45.0


def split(rows: list[dict], cal, speed: float = SPLIT_SPEED,
          fps: float = 25.0) -> list[dict]:
    """Sever ids at frame-to-frame teleports, BEFORE stitching.

    Returns track.py-shaped rows with new ids. Splitting the rows rather than
    the tracklets is what keeps this a single function: everything downstream -
    tracklets, the team gate, stitch, apply - runs unchanged on the new ids, and
    the team label each row already carries is inherited by its fragment for
    free.

    Measured on contamination, with and without the team gate:
    detection-tracking-evidence (not published)#evidence-stitching
    """
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[int(r["id"])].append(r)
    out: list[dict] = []
    nxt = 0
    for _, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["f"])
        f = np.array([r["f"] for r in rs], dtype=np.float64)
        xy = np.asarray(cal.to_pitch(np.array([[r["x"], r["y"]] for r in rs], float)),
                        dtype=np.float64)
        cuts = [0]
        if len(rs) > 1:
            # fps, not a literal 25: `f` is in source frames, so a hard-coded
            # 25 made the effective threshold SPLIT_SPEED * fps/25 - inert at
            # 50 fps (90 m/s), twice as aggressive at 12.5. stitch() beside it
            # already takes fps; split did not, and it is the half that is
            # worth 5 points of contamination.
            dt = np.maximum(np.diff(f) / max(fps, 1e-9), 1e-6)
            v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
            cuts += (np.flatnonzero(v > speed) + 1).tolist()
        cuts.append(len(rs))
        for a, b in zip(cuts, cuts[1:]):
            if b <= a:
                continue
            for r in rs[a:b]:
                out.append({**r, "id": nxt})
            nxt += 1
    return out


class HeightScale:
    """A metre scale recovered from PLAYER BOX HEIGHTS. No calibration file.

    WHY THIS EXISTS
        `track.py` stitches only with a calibration, so an uncalibrated run is
        not stitched. This recovers a metre scale from box heights for research
        runs; the job path never uses it. detection-tracking-evidence (not published)#evidence-stitching

    THE GEOMETRY, WHICH IS EXACT FOR THIS CAMERA
        For a level pinhole looking at a ground plane, a standing player of
        height P at depth Z images as `h = f*P/Z`, and their FEET sit at
        `y - y0 = f*C/Z` where C is the camera height and y0 the horizon row.
        Divide: `h / (y - y0) = P / C`, a constant. So box height is LINEAR in
        foot row, and one least-squares fit recovers the horizon y0.

        That gives depth up to a scale: `Z = f*C / (y - y0)`. Two unknowns are
        left, C and f*C, and the pitch supplies exactly two known distances.
        A camera on the touchline at the halfway line sees the pitch LENGTH
        across the image and its WIDTH in depth, so:

            lateral_m = (x - cx) * C * d      spans 105 m   -> gives C
            depth_m   =           (f*C) * d   spans  68 m   -> gives f*C

        where `d = 1/(y - y0)`. Nothing else is needed and nothing is clicked.

    WHAT IT IS NOT
        Not a calibration. Absolute positions are wrong by whatever the width
        assumption is wrong by, and it assumes a level camera. It is only ever
        used to make DISTANCES comparable so the stitcher can rank candidate
        joins, which is exactly what `stitch._link` consumes.

    `ponytail: a real calibration beats this whenever one exists - to_pitch is
    the same interface, so pass the real calibrator instead and this is unused.`
    """

    PITCH_L_M = 105.0     # across the image, for a touchline camera
    PITCH_W_M = 68.0      # into the image

    def __init__(self, y0: float, cx: float, lat_scale: float, depth_scale: float,
                 y_lo: float, y_hi: float):
        self.y0, self.cx = y0, cx
        self.lat_scale, self.depth_scale = lat_scale, depth_scale
        # the box this fit puts the pitch in, in ITS OWN output frame. x is
        # centred on cx by construction; the depth axis is not centred on
        # anything, so it has to be carried rather than inferred.
        self.y_lo, self.y_hi = y_lo, y_hi

    def pitch_box(self):
        """(x0, x1, y0, y1) of the pitch in this calibrator's output frame."""
        return (-self.PITCH_L_M / 2, self.PITCH_L_M / 2, self.y_lo, self.y_hi)

    @classmethod
    def fit(cls, rows: list[dict]) -> "HeightScale | None":
        y = np.array([r["y"] for r in rows], dtype=np.float64)
        h = np.array([r["h"] for r in rows], dtype=np.float64)
        if len(y) < 200 or h.max() <= 0:
            return None
        # h = a*(y - y0)  ->  least squares on (y, h); horizon is the x-intercept
        a, b = np.polyfit(y, h, 1)
        if a <= 1e-9:
            return None                       # boxes do not grow toward the camera
        y0 = -b / a
        if y0 >= y.min():
            return None                       # horizon inside the data: not a ground plane
        d = 1.0 / (y - y0)                    # depth, up to the unknown factor f*C
        x = np.array([r["x"] for r in rows], dtype=np.float64)
        cx = float(np.median(x))
        lat = (x - cx) * d
        # 1st..99th percentile, not min..max: one wild box must not set the scale
        lat_span = float(np.percentile(lat, 99) - np.percentile(lat, 1))
        d_span = float(np.percentile(d, 99) - np.percentile(d, 1))
        if lat_span <= 0 or d_span <= 0:
            return None
        depth_scale = cls.PITCH_W_M / d_span
        d_lo, d_hi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
        return cls(y0, cx, cls.PITCH_L_M / lat_span, depth_scale,
                   d_lo * depth_scale, d_hi * depth_scale)

    def to_pitch(self, pts_px):
        """Pixels -> this fit's OWN frame, not corner-origin pitch coordinates: x is
        lateral metres from the median column `cx` (negative to its left), y is depth
        from the camera in metres (pitch_box() gives the pitch's extent here). Good
        for distances between points, not for absolute positions."""
        p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        d = 1.0 / np.maximum(p[:, 1] - self.y0, 1e-6)
        return np.stack([(p[:, 0] - self.cx) * d * self.lat_scale,
                         d * self.depth_scale], axis=1)


# An off-pitch mask here was WITHDRAWN 2026-09-08: it deleted half the pitch on
# calibrated jobs. detection-tracking-evidence (not published)#evidence-stitching

def tracklets(rows: list[dict], cal) -> dict[int, _Tracklet]:
    """track.py-shaped rows (f, id, x, y in PIXELS) -> tracklets in metres.

    Projected once per row here rather than once per candidate pair: to_pitch on
    a panorama calibrator is a remap plus a homography, and the stitcher asks
    for the same two endpoints thousands of times.
    """
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[int(r["id"])].append(r)
    out = {}
    for tid, rs in by.items():
        rs.sort(key=lambda r: r["f"])
        f = np.array([r["f"] for r in rs], dtype=np.int64)
        px = np.array([[r["x"], r["y"]] for r in rs], dtype=np.float64)
        out[tid] = _Tracklet(tid, f, np.asarray(cal.to_pitch(px), dtype=np.float64))
    return out


COAST = True   # velocity-extrapolated linking; False restores distance-only
# Appearance discount, metres per unit cosine. OFF: measured and refused (same-team
# appearance barely beats box height, and end to end it is noise); the weights are
# research-provenance. detection-tracking-evidence (not published)#evidence-stitching
EMB_W = 0.0


def _link(a: _Tracklet, b: _Tracklet, fps: float, max_speed: float, slack_m: float,
          max_gap_s: float) -> float | None:
    """Cost of joining a's exit to b's entry, or None if the join is impossible.

    `a` must end strictly before `b` starts. That is temporal exclusion, and it
    lives here rather than in the caller because it is the one exact constraint
    in the file and there must be no way around it.

    The cost is a pitch-plane MISS DISTANCE after coasting each end along its own
    velocity (see COAST), falling back to raw distance when that is smaller.
    Deliberately not the implied speed, which separated same-player joins worse
    (detection-tracking-evidence (not published)#evidence-stitching).
    """
    if a.end >= b.start:
        return None
    gap_s = (b.start - a.end) / fps
    if gap_s > max_gap_s:
        return None
    d = float(np.hypot(*(b.xy[0] - a.xy[-1])))
    # `not d <= gate`, not `d > gate`: a NaN end (an unprojectable point) fails every
    # comparison, so it would pass the gate and _feasible's max() would score it 0.
    if not d <= max_speed * gap_s + slack_m:
        return None
    if not COAST:
        return d
    # COASTING. Distance alone charges a sprinter the same as a teleport: a
    # player running 6 m/s across a 2 s occlusion is 12 m away and innocent,
    # while a 12 m jump with both ends standing still is somebody else. So
    # extrapolate each end along its own velocity and score the MISS, not the
    # gap. Symmetric because either end may be the better-observed one, and the
    # min because one clean end is enough to explain the join.
    fwd = float(np.hypot(*(b.xy[0] - (a.xy[-1] + _vel(a, -1, fps) * gap_s))))
    bwd = float(np.hypot(*(a.xy[-1] - (b.xy[0] - _vel(b, 0, fps) * gap_s))))
    c = min(d, fwd, bwd)
    return c if np.isfinite([d, fwd, bwd]).all() else None


def _vel(t: _Tracklet, end: int, fps: float, span: int = 5) -> np.ndarray:
    """Velocity in m/s at one end of a tracklet, over the last `span` samples.

    CLAMPED TO MAX_SPEED, because the unclamped version was worse than useless.
    A two-frame tracklet has a velocity estimated from a single 40 ms
    difference, and `SPLIT_SPEED`'s own note in this file says one pixel of
    far-side wobble reads as 16 m/s. Measured, unclamped: 0.4 m of wobble in one
    frame coasts a **19.6 m** join down to a cost of **0.40 m** - cheaper than a
    genuine 1 m join, so greedy agglomeration does the wild far-side merge
    FIRST. That is the exact failure this file's own commits blame for the
    association ceiling, and coasting was manufacturing part of it.

    Clamped, the same 19.6 m join costs 1.60 m and ranks behind the 1 m join.
    A tracklet too short to have a velocity at all gets zero, which degrades
    `_link` back to plain distance.
    """
    if len(t.f) < 2:
        return np.zeros(2)
    i, j = (max(0, len(t.f) - span), len(t.f) - 1) if end == -1 else (0, min(span, len(t.f)) - 1)
    dt = (t.f[j] - t.f[i]) / fps
    if dt <= 0:
        return np.zeros(2)
    v = (t.xy[j] - t.xy[i]) / dt
    n = float(np.hypot(*v))
    return v if n <= MAX_SPEED else v * (MAX_SPEED / n)


def _feasible(parts: list[_Tracklet], fps: float, max_speed: float, slack_m: float,
              max_gap_s: float) -> float | None:
    """Cost of one player holding exactly `parts`, or None if they cannot be one.

    Every consecutive pair is re-checked, not just the two ends being joined.
    Components interleave: merging A=[0-100]+[300-400] with B=[150-250] creates
    TWO junctions, and validating one of them is how a stitcher quietly bridges
    across a player who was in the way the whole time.

    A component costs what its WORST junction costs. A chain is only as
    believable as its least believable link, and a mean lets one 40 m leap hide
    behind six 2 m ones.
    """
    worst = 0.0
    for u, v in zip(parts, parts[1:]):
        c = _link(u, v, fps, max_speed, slack_m, max_gap_s)
        if c is None:
            return None
        worst = max(worst, c)
    return worst


def stitch(tl: dict[int, _Tracklet], fps: float = 25.0, max_speed: float = MAX_SPEED,
           slack_m: float = SLACK_M, max_gap_s: float = MAX_GAP_S,
           teams: dict[int, str] | None = None,
           emb: dict[int, np.ndarray] | None = None,
           emb_w: float = EMB_W) -> dict[int, int]:
    """Agglomerate tracklets into players. Returns old id -> new id.

    Hierarchical, cheapest junction first, one merge at a time, re-scored after
    each. Greedy over a GLOBAL ranking rather than a left-to-right sweep: the
    closest pair anywhere on the pitch is the one least likely to be wrong, and
    consuming it first removes the ambiguity that would otherwise mislead its
    neighbours. This is GTA-link's clustering with a physical distance where it
    uses an appearance embedding.

    # ponytail: O(n^2) to seed and O(n) per merge, on a dict of pairs. n is the
    # number of tracklets - 189 over 6 minutes, so a few hundred over a half -
    # and it runs in under a second. If a noisier tracker ever pushes n into the
    # thousands, the fix is a heap keyed by cost, not a different algorithm.
    """
    live = {tid: [t] for tid, t in tl.items()}
    teams = teams or {}
    emb = emb or {}

    def feat(k: int) -> np.ndarray | None:
        """One L2-normalised appearance vector per COMPONENT, length-weighted.

        A component's identity is the identity of its longest parts; a 3-frame
        fragment should not drag the mean as far as a 300-frame one.
        """
        v = [(len(p.f), emb[p.id]) for p in live[k] if p.id in emb]
        if not v:
            return None
        m = np.sum([w * e for w, e in v], axis=0) / sum(w for w, _ in v)
        n = float(np.linalg.norm(m))
        return m / n if n > 1e-9 else None

    def sides(k: int) -> set[str]:
        """The known team labels inside one component. Unlabelled parts vote."""
        return {teams[p.id] for p in live[k] if p.id in teams} - {"official", "unknown"}

    def cost(i: int, j: int) -> float | None:
        # Two components that are confidently on OPPOSITE sides are not one
        # player. Unlabelled tracklets are not blocked - absence of a label is
        # not evidence of a different one, and most short tracklets have none.
        if teams:
            a, b = sides(i), sides(j)
            if a and b and not (a & b):
                return None
        c = _feasible(sorted(live[i] + live[j], key=lambda t: t.start),
                      fps, max_speed, slack_m, max_gap_s)
        if c is None or not emb_w:
            return c
        # Appearance DISCOUNTS a geometrically feasible join; it never creates
        # one. Feasibility stays a hard physical gate, so a look-alike on the
        # far side of the pitch is still impossible, not merely expensive.
        fi, fj = feat(i), feat(j)
        return c if fi is None or fj is None else c - emb_w * float(fi @ fj)

    ids = sorted(live)
    pending: dict[tuple[int, int], float] = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            c = cost(ids[a], ids[b])
            if c is not None:
                pending[(ids[a], ids[b])] = c

    while pending:
        i, j = min(pending, key=pending.get)  # type: ignore[arg-type]
        live[i] = sorted(live[i] + live[j], key=lambda t: t.start)
        del live[j]
        # Only the merged component's costs changed; every other pair keeps the
        # cost it already had. Rebuilding the whole matrix per merge is the
        # obvious version and it is O(n^3) - minutes here instead of a second.
        for k in [k for k in pending if i in k or j in k]:
            del pending[k]
        for k in live:
            if k != i:
                c = cost(min(i, k), max(i, k))
                if c is not None:
                    pending[(min(i, k), max(i, k))] = c

    return {p.id: new for new, parts in live.items() for p in parts}


def apply(rows: list[dict], mapping: dict[int, int]) -> list[dict]:
    """Same rows, same boxes, new ids.

    Nothing added, moved or removed - which is why stitching cannot change
    detection precision or recall by a single detection. See the module
    docstring on why that matters for reading the numbers.
    """
    return [{**r, "id": mapping.get(int(r["id"]), int(r["id"]))} for r in rows]


def _measure(frames: int = 9000, tracker: str = "cbiou",
             job: str = "data/jobs/118575_full_1st", **kw) -> None:
    """Reproduce the table in the module docstring against real ground truth.

    Replays the cached detections through `tracker`, scores them with
    model/validation/assoc_eval.score, then scores them again after stitching. Same
    detections, same scorer, one variable.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(root, "model", "validation"))
    sys.path.insert(0, os.path.join(root, "model", "engine"))
    import calibrators  # noqa: PLC0415
    from assoc_eval import load_gt, score  # noqa: PLC0415
    from tracking import get_tracker  # noqa: PLC0415

    cache = json.load(open(os.path.join(root, job, "dets.cache.json")))
    cal = calibrators.load(os.path.join(root, job, "calibration.json"))
    tr = get_tracker(tracker)
    rows = []
    for f in range(frames):
        dets = cache["frames"].get(str(f))
        if dets is None:
            continue
        for d in tr.update(dets, None):
            x1, _, x2, y2 = d["xyxy"]
            # Feet, matching track.py :: track - the homography maps the ground plane.
            rows.append({"f": f, "id": int(d["id"]), "x": (x1 + x2) / 2, "y": y2})

    tl = tracklets(rows, cal)
    gt, _ = load_gt(os.path.join(root, "model/validation/soccertrack/118575_1st.tracks.json.gz"))

    def report(label: str, mapping: dict[int, int]) -> None:
        pred: dict[int, list] = collections.defaultdict(list)
        for tid, t in tl.items():
            for fr, xy in zip(t.f, t.xy):
                pred[int(fr)].append((mapping.get(tid, tid), xy))
        r = score(pred, gt, frames)
        print(f"{label:26s} IDF1 {r['IDF1']:.4f}  frag {r['fragments_per_player']:5.1f}  "
              f"switch {r['id_switches']:5d}  prec {r['precision']:.4f}  "
              f"rec {r['recall']:.4f}  tracks {r['pred_tracks']:4d}  "
              f"life {r['mean_track_life_s']:6.1f}s")

    report(f"{tracker}, no stitching", {})
    report(f"{tracker}, stitched", stitch(tl, **kw))


def _selfcheck() -> None:
    """The invariants, on synthetic tracklets. Two of them are the whole file.

    ONE: a stitched id must never hold two detections in the same frame. That
    is temporal exclusion, and if it ever fails the output is not a player.
    TWO: two players walking side by side must not be welded, however cheap the
    join looks. Welding is the failure no downstream check catches - stats.py's
    teleport clamp never trips between adjacent players, the distance just
    moves from one profile to the other - and it makes track count and track
    life LOOK BETTER while doing it.
    """
    def tk(tid, f0, f1, x0, x1, y=34.0):
        f = np.arange(f0, f1)
        xy = np.column_stack([np.linspace(x0, x1, len(f)), np.full(len(f), y)])
        return _Tracklet(tid, f, xy)

    # split(): a teleport mid-track must sever the id, a normal run must not,
    # and the team label must ride along onto both halves.
    from pitch import Calibration, Pitch  # noqa: PLC0415
    flat = Calibration(pitch=Pitch(), H=np.eye(3))
    walk = [{"f": i, "id": 7, "x": 10.0 + 0.2 * i, "y": 34.0, "team": "home"}
            for i in range(40)]
    assert len({r["id"] for r in split(walk, flat)}) == 1, "a normal run must not split"
    # ...now teleport 60 m between two consecutive frames: 1500 m/s.
    jump = walk[:20] + [{**r, "x": r["x"] + 60.0} for r in walk[20:]]
    got = split(jump, flat)
    assert len({r["id"] for r in got}) == 2, "a teleport must sever the id"
    assert all(r["team"] == "home" for r in got), "the team label must be inherited"
    assert len(got) == len(jump), "split must not add or drop a detection"
    # The threshold is a real threshold: 8 m/s is a fast player, not a teleport.
    fast = [{"f": i, "id": 7, "x": 10.0 + 0.32 * i, "y": 34.0, "team": "home"}
            for i in range(40)]
    assert len({r["id"] for r in split(fast, flat)}) == 1, "8 m/s is a player"

    # One player crossing the pitch, cut in two by a 10-frame occlusion.
    joinable = {1: tk(1, 0, 100, 0.0, 8.0), 2: tk(2, 110, 200, 8.8, 16.0)}
    m = stitch(joinable)
    assert m[1] == m[2], m

    # The team gate, on that same joinable pair. It must block opposite sides,
    # allow the same side, and - the one that matters - stay OUT OF THE WAY when
    # a label is missing. A gate is only as good as its label, and most short
    # tracklets never get one; treating "no label" as "different label" is how
    # the first attempt at this destroyed 23 of 68 correct joins.
    assert stitch(joinable, teams={1: "home", 2: "away"})[1] != \
           stitch(joinable, teams={1: "home", 2: "away"})[2], "opposite sides must not merge"
    assert stitch(joinable, teams={1: "home", 2: "home"})[1] == \
           stitch(joinable, teams={1: "home", 2: "home"})[2], "same side must still merge"
    m = stitch(joinable, teams={1: "home"})
    assert m[1] == m[2], "an unlabelled tracklet must not be blocked"
    m = stitch(joinable, teams={1: "home", 2: "official"})
    assert m[1] == m[2], "'official' is not a side and must not block"

    # Same two halves, but overlapping by one frame: provably two players.
    m = stitch({1: tk(1, 0, 100, 0.0, 8.0), 2: tk(2, 99, 200, 8.8, 16.0)})
    assert m[1] != m[2], m

    # Same two halves, 50 m apart across a 10-frame gap: 125 m/s, refused.
    m = stitch({1: tk(1, 0, 100, 0.0, 8.0), 2: tk(2, 110, 200, 58.0, 66.0)})
    assert m[1] != m[2], m

    # ...and the gap ceiling is enforced independently of the speed budget:
    # 20 s of standing still is inside 9 m/s and outside MAX_GAP_S.
    m = stitch({1: tk(1, 0, 100, 0.0, 8.0), 2: tk(2, 600, 700, 8.0, 12.0)})
    assert m[1] != m[2], m

    # TWO PLAYERS, 2 m apart, walking together; the left one fragments. The
    # right one is continuous, so temporal exclusion alone forbids absorbing
    # it - this is the case that makes stitching safe where an online gate is
    # not, and it must hold even though the wrong join is the nearer one.
    a1, a2 = tk(1, 0, 100, 0.0, 8.0), tk(2, 110, 300, 8.8, 24.0)
    b = tk(3, 0, 300, 0.2, 24.2, y=36.0)
    m = stitch({1: a1, 2: a2, 3: b})
    assert m[1] == m[2] and m[3] != m[1], m

    # The invariant, checked rather than assumed: no merged id may hold two
    # detections in one frame, on a deliberately crowded input.
    tl = {i: tk(i, i * 40, i * 40 + 30, i * 3.0, i * 3.0 + 2.0, y=30.0 + (i % 3))
          for i in range(1, 12)}
    m = stitch(tl)
    seen: dict[int, set] = collections.defaultdict(set)
    for tid, t in tl.items():
        for f in t.f:
            assert int(f) not in seen[m[tid]], (tid, int(f))
            seen[m[tid]].add(int(f))

    # Every input id is accounted for, and apply() relabels without touching
    # anything else - the property the whole precision argument rests on.
    assert set(m) == set(tl)
    rows = [{"f": 0, "id": 1, "x": 5.0, "y": 9.0, "conf": 0.9}]
    out = apply(rows, {1: 7})
    assert out == [{"f": 0, "id": 7, "x": 5.0, "y": 9.0, "conf": 0.9}]
    assert rows[0]["id"] == 1  # not mutated in place

    # tracklets() must project through the calibration, not around it: a stub
    # that halves every coordinate has to show up in the metres.
    class _Cal:
        to_pitch = staticmethod(lambda px: np.asarray(px) * 0.5)

    tls = tracklets([{"f": 3, "id": 9, "x": 10.0, "y": 20.0},
                     {"f": 1, "id": 9, "x": 2.0, "y": 4.0}], _Cal())
    assert tls[9].start == 1 and tls[9].end == 3           # sorted by frame
    assert tls[9].xy[0].tolist() == [1.0, 2.0], tls[9].xy  # and in metres

    # A NaN end (an unprojectable point) must refuse the link. It fails `d > gate`,
    # and _feasible's max(0, nan) is 0 - the cheapest join on the pitch, 70 m apart.
    nan_end = tk(1, 0, 10, 10.0, 11.0)
    nan_end.xy[-1] = np.nan
    m = stitch({1: nan_end, 2: tk(2, 12, 20, 80.0, 81.0)})
    assert m[1] != m[2], m

    print("stitch.py self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true",
                    help="score against ground truth (needs the detection cache)")
    ap.add_argument("--frames", type=int, default=9000)
    ap.add_argument("--tracker", default="cbiou")
    ap.add_argument("--max-gap-s", type=float, default=MAX_GAP_S)
    ap.add_argument("--max-speed", type=float, default=MAX_SPEED)
    ap.add_argument("--slack-m", type=float, default=SLACK_M)
    a = ap.parse_args()
    if a.eval:
        _measure(a.frames, a.tracker, max_gap_s=a.max_gap_s, max_speed=a.max_speed,
                 slack_m=a.slack_m)
    else:
        _selfcheck()
