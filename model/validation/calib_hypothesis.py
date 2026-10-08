"""ROADMAP A: is the product's distance error CALIBRATION, or detection+tracking?

THE HYPOTHESIS (full-pipeline-plan (not published), "First finding")
    recall_px 0.845 / recall_m 0.633 / distance.coverage 0.704. The same boxes
    lose 21 points between image space and metre space, and metre-space recall
    sits seven points from distance coverage. So the coverage hole MAY itself be
    calibration, and "improve the detector" would be the wrong response.

THE TEST
    Substitute ground truth at the CALIBRATION STEP ONLY - swap the pixel ->
    metre map for one fitted on ground truth, hold the detections, the track
    ids and the estimator fixed - and re-run product.distance.*.

WHAT MAKES THIS A TEST AND NOT A NULL READ (postmortem (not published) C2)
    The substitution has to be able to come out BOTH ways, and both ways have to
    be reachable by the machinery, not just by the argument. Two controls run
    every time:

      CEILING  GT positions fed in as if they were our tracker's output, through
               the same attribution, windowing and summation. Must read coverage
               1.00 / ratio 1.00. If it does not, the harness cannot show a
               calibration win even when calibration is perfect, and no verdict
               it produces means anything.

      FLOOR    the oracle map's OWN residual, measured on frames held out of its
               fit. An oracle that is itself 2 m wrong cannot clear a 2 m gate,
               so this number bounds what the experiment could ever have shown.

    And the gate is swept. `contam_eval.GATE_M` is 2.0 m by fiat; C1 in the
    postmortem is a conclusion drawn from a constant nobody checked against the
    distribution it cut. Coverage is reported as a CURVE in the gate, under both
    maps, so the reader sees the shape rather than one point on it.

THE ORACLE, AND WHY IT IS NOT THE ANSWER SUBSTITUTED FOR ITSELF
    It is fitted on ground truth that never touches our pipeline:

        MOT boxes  $ST2_DIR/mot/118575.txt
                   6,000 frames x 22 players, RAW PANORAMA PIXELS, top-left
        metres     model/validation/soccertrack/118575_1st.tracks.json.gz
                   the same 22 players in pitch metres, the harness's own GT

    Those two feeds are the same annotator's, on the same footage, in two
    spaces. Regressing one onto the other gives the best pixel -> metre map this
    footage admits. Our detections are then pushed through it. No detection ever
    gets its position replaced by a GT position - only the MAP changes.

BOX CONVENTIONS, which this repo has got wrong three times
    tracks.json  bottom-centre x,y  (model/engine/track.py :: track writes (fx1+fx2)/2, fy2)
    MOT gt.txt   top-left x,y,w,h -> foot is (x + w/2, y + h)
    Both are asserted in `selfcheck`, against the files, not against a comment.

SPACES, which is the other way to lose a day here
    tracks.split.json is in REMAPPED BAND pixels (11878x870: the vendor canvas
    cropped to rows 390-690 and scaled 2.9x). The MOT GT is in RAW PANORAMA
    pixels (4096x1080). The remap tables answer "which raw pixel feeds this band
    pixel", so band -> raw is a direct table lookup and needs no inversion. That
    is the direction used here; `remap_points` (the other direction, a per-point
    Newton solve) is never called.

    python model/validation/calib_hypothesis.py --selfcheck
    python model/validation/calib_hypothesis.py --join       # offset + id permutation
    python model/validation/calib_hypothesis.py --run        # the experiment
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrators  # noqa: E402
import stats  # noqa: E402
import contam_eval  # noqa: E402
import dist_eval  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT_M = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")
GT_PX = _ST2 + "/mot/118575.txt"
OUT = os.path.join(ROOT, "model/validation/out")

#: Clip frame 0 -> gz 0-based index. st2_metres.OFFSET puts clip frame 0 at
#: FIRST_HALF frameNumber 6246; the metadata puts FIRST_HALF frameStart at 251
#: and the gz file is that series with its base subtracted, so 6246 - 251.
#: NOT taken on trust - `join` sweeps it and asserts the sweep picks this.
OFFSET_GUESS = 5995


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# loading

def load_gt_px(path: str = GT_PX):
    """{clip frame: {mot_id: foot (u,v) in RAW PANORAMA pixels}}.

    MOT is top-left x,y,w,h; the foot the product compares against is
    bottom-centre. Getting this backwards offsets every point by a full player
    height, which on this footage is metres of pitch at the far side.
    """
    out = collections.defaultdict(dict)
    for line in open(path):
        p = line.split(",")
        if len(p) < 6:
            continue
        x, y, w, h = (float(v) for v in p[2:6])
        out[int(p[0])][int(p[1])] = (x + w / 2.0, y + h)
    return dict(out)


def load_gt_m(path: str = GT_M):
    """{gz 0-based frame: {gt_id: (X,Y) metres}}, and the fps."""
    d = json.load(gzip.open(path))
    rows = d["detections"]
    base = min(r["f"] for r in rows)
    per = collections.defaultdict(dict)
    for r in rows:
        per[r["f"] - base][str(r["id"])] = (r["x"], r["y"])
    return float(d["fps"]), dict(per)


def band_to_raw(cal, uv: np.ndarray) -> np.ndarray:
    """Band (remapped) pixels -> raw panorama pixels, by sampling the tables.

    cv2.remap's tables say, for each DESTINATION (band) pixel, which SOURCE
    (raw) pixel feeds it - so they already point the way we want and the
    lookup is one bilinear sample, not an inversion.
    """
    q = np.asarray(uv, np.float64).reshape(-1, 2)
    # cv2.remap asserts both dimensions of the query grid are < SHRT_MAX, so a
    # flat 1 x N query fails above 32,767 points. Pad to a rectangle.
    W = 1024
    n = len(q)
    pad = (-n) % W
    g = np.vstack([q, np.zeros((pad, 2))]).reshape(-1, W, 2).astype(np.float32)
    u = np.ascontiguousarray(g[..., 0])
    v = np.ascontiguousarray(g[..., 1])
    out = np.stack([cv2.remap(m, u, v, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).ravel()[:n]
                    for m in (cal._mapx, cal._mapy)], axis=1)
    return out.astype(np.float64)


def raw_to_band(cal, pts, iters: int = 4):
    """Raw panorama pixels -> band pixels. Coarse grid, then Newton on the map.

    Needed for ONE thing only: putting the GT's own feet into the space the
    shipped calibrator eats, so the shipped map and the oracle can be scored on
    the SAME held-out points. `calibrators.remap_points` does this exactly but
    loops in Python with a full-table argmin per point, which is minutes for
    13,000 points; this is the vectorised version and it is checked against the
    forward table rather than trusted.
    """
    from scipy.spatial import cKDTree
    h, w = cal._mapx.shape
    gy, gx = np.mgrid[0:h:4, 0:w:4]
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    tree = cKDTree(band_to_raw(cal, grid))
    p = np.asarray(pts, float).reshape(-1, 2)
    b = grid[tree.query(p)[1]]
    for _ in range(iters):
        f0 = band_to_raw(cal, b)
        j0 = (band_to_raw(cal, b + [1.0, 0.0]) - f0)
        j1 = (band_to_raw(cal, b + [0.0, 1.0]) - f0)
        J = np.stack([j0, j1], axis=2)          # d(raw)/d(band)
        r = (p - f0)[:, :, None]
        try:
            b = b + np.linalg.solve(J, r)[:, :, 0]
        except np.linalg.LinAlgError:
            break
        b = np.clip(b, [0, 0], [w - 1.001, h - 1.001])
    return b


def coarse_raw_to_m(cal):
    """A rough raw-panorama-pixel -> metre homography, for the JOIN ONLY.

    Built from the job calibration's own 65 landmarks: they are (pitch metres)
    -> (band pixels), and band -> raw is a table lookup, so one `findHomography`
    gives raw -> metres. This is the "single homography on the raw panorama"
    the RemapCalibrator docstring measures at 6.17 m median, and that is fine
    for what it is used for: telling 22 players apart over four minutes of
    aggregated cost, where the wrong answer is tens of metres, not six.
    """
    pts = json.load(open(os.path.join(JOB, "calibration.json")))["points"]
    M = np.array([[float(t) for t in k.strip("()").split(",")] for k in pts], float)
    band = np.array(list(pts.values()), float)
    raw = band_to_raw(cal, band)
    H, _ = cv2.findHomography(raw, M, 0)
    def fn(uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        p = np.c_[uv, np.ones(len(uv))] @ H.T
        return p[:, :2] / p[:, 2:]
    return fn


# --------------------------------------------------------------------------
# the join: which clip frame is which gz frame, and which MOT id is which gz id

def join(cal, every: int = 25, span: int = 40, verbose: bool = True,
         rough=None, guess: int = OFFSET_GUESS) -> dict:
    """Frame offset + MOT id -> gz id, both by search, neither by assumption.

    The offset sweep is the guard the frame-alignment work in this repo has
    needed three times: a wrong offset does not fail, it reports a slightly
    worse everything.

    `rough` is the raw-pixel -> metre map used to compare the two feeds. The
    first pass uses the coarse homography (~7 m); `--join` then refits the
    oracle on that answer and runs a second pass through it, because a 7 m map
    can resolve a 22-way permutation over four minutes but cannot resolve a
    three-frame offset, and the two passes agreeing is the evidence that
    neither is an artefact of the map it was found with.
    """
    px, (_fps, gm) = load_gt_px(), load_gt_m()
    rough = rough or coarse_raw_to_m(cal)
    clip_frames = sorted(px)[::every]

    def cost(off):
        """22x22 aggregated assignment cost at this offset, and its median."""
        C = np.zeros((22, 22))
        n = 0
        for c in clip_frames:
            g = gm.get(c + off)
            if not g or len(g) != 22 or len(px[c]) != 22:
                continue
            A = rough([px[c][i] for i in range(1, 23)])
            B = np.array([g[str(j)] for j in range(1, 23)])
            C += np.linalg.norm(A[:, None] - B[None], axis=2)
            n += 1
        return C / max(n, 1), n

    from scipy.optimize import linear_sum_assignment
    best = None
    for off in range(guess - span, guess + span + 1, 5):
        C, n = cost(off)
        if not n:
            continue
        r, c = linear_sum_assignment(C)
        s = C[r, c].mean()
        if best is None or s < best[0]:
            best = (s, off, C)
    # refine to the frame
    s0, off0, _ = best
    for off in range(off0 - 6, off0 + 7):
        C, n = cost(off)
        if not n:
            continue
        r, c = linear_sum_assignment(C)
        s = C[r, c].mean()
        if s < best[0]:
            best = (s, off, C)
    score, off, C = best
    r, c = linear_sum_assignment(C)
    perm = {int(a) + 1: str(int(b) + 1) for a, b in zip(r, c)}
    # Margin: how much worse the SECOND-best assignment is. A join that is one
    # swap away from an equally good alternative is not a join.
    C2 = C.copy()
    C2[r, c] = 1e6
    r2, c2 = linear_sum_assignment(C2)
    # Per-frame agreement: solve the assignment frame by frame and count how
    # often it reproduces the aggregate answer. An aggregate permutation that
    # only 60% of frames agree with is a coincidence of the averaging.
    agree = tot = 0
    for c_ in clip_frames:
        g = gm.get(c_ + off)
        if not g or len(g) != 22 or len(px[c_]) != 22:
            continue
        A = rough([px[c_][i] for i in range(1, 23)])
        B = np.array([g[str(j)] for j in range(1, 23)])
        rr, cc = linear_sum_assignment(np.linalg.norm(A[:, None] - B[None], axis=2))
        agree += sum(perm[int(x) + 1] == str(int(y) + 1) for x, y in zip(rr, cc))
        tot += 22
    if verbose:
        log(f"  offset sweep -> clip frame 0 = gz frame {off} "
            f"(guess {guess}), mean assigned cost {score:.2f} m")
        log(f"  id permutation: identity? {all(k == int(v) for k, v in perm.items())}; "
            f"2nd-best assignment costs {C2[r2, c2].mean():.2f} m; "
            f"per-frame agreement {agree}/{tot} = {agree / max(tot, 1):.3f}")
    return {"offset": int(off), "perm": perm, "cost_m": float(score),
            "cost2_m": float(C2[r2, c2].mean()),
            "frame_agreement": agree / max(tot, 1), "frame_n": tot,
            "identity_perm": all(k == int(v) for k, v in perm.items())}


# --------------------------------------------------------------------------
# the oracle map

ORACLE_SMOOTHING = 10000.0
"""The only smoothing at which this file reproduces its own published numbers.

`@M` The default used to be 1.0 - in the CLI AND in both function signatures, so
`lint_truth`'s FLAG DRIFT check could not see it: they agreed with each other and
disagreed with reality. `selfcheck()` hardcoded 10000.0, which means **the
self-check was validating a configuration nobody ran.**

At the old default of 1.0 the interpolator adds its own path length and NONE of
the published oracle column reproduces:

    held-out residual   0.549 / 1.186 / 96.2%  ->  0.792 / 1.980 / 90.1%
    recall_m oracle     0.753                  ->  0.726
    ratio_raw oracle    1.046                  ->  1.274
    distance neutrality 0.990                  ->  1.094

`@M` That last row is the tell: 1.094 sits OUTSIDE the 0.95-1.05 band
`selfcheck()` itself asserts, so at the shipped default the "+23% the map
compresses" was mostly map wiggle. 10000.0 is also the value a held-out split
independently selects in `model/validation/st2_metres.py`, by a different route.
"""


class Oracle:
    """raw panorama pixel -> pitch metres, regressed on ground truth.

    Local thin-plate-spline (RBFInterpolator with `neighbors`), because the
    residual of any global fit on this footage is a smooth ramp rather than
    noise (research-farfield-engineering (not published)) and a local fit absorbs it
    without the extrapolation blow-up a global TPS produces off the hull.

    IT REFUSES TO EXTRAPOLATE. Outside the convex hull of the training feet the
    answer is the coarse homography, flagged; a confident number out there would
    be exactly the failure `st2_metres.tps_field` documents at 4.25 m.
    """

    def __init__(self, uv: np.ndarray, xy: np.ndarray, neighbors: int = 64,
                 smoothing: float = ORACLE_SMOOTHING, cell_px: float = 2.0):
        from scipy.interpolate import RBFInterpolator
        from scipy.spatial import Delaunay
        # POOL ONTO A PIXEL GRID FIRST, and this is not an optimisation. A
        # player standing still emits the same box for a hundred frames, so the
        # raw pairs contain thousands of exact duplicates; a 64-neighbour
        # neighbourhood can then be a handful of distinct points sitting on a
        # line, and the local TPS raises "matrix of monomials does not have
        # full column rank". Pooling also stops a stationary player from
        # outvoting a moving one for his own patch of grass.
        key = np.round(uv / cell_px).astype(np.int64)
        _u, inv = np.unique(key, axis=0, return_inverse=True)
        n = np.bincount(inv)
        uv = np.stack([np.bincount(inv, uv[:, k]) / n for k in (0, 1)], 1)
        xy = np.stack([np.bincount(inv, xy[:, k]) / n for k in (0, 1)], 1)
        self.uv, self.xy = uv, xy
        self._rbf = RBFInterpolator(uv, xy, neighbors=min(neighbors, len(uv)),
                                    kernel="thin_plate_spline", smoothing=smoothing)
        self._hull = Delaunay(uv[np.random.default_rng(0).choice(
            len(uv), min(len(uv), 20000), replace=False)])

    def __call__(self, uv, base=None):
        """Correct inside the hull; OUTSIDE IT, hand back `base` untouched.

        `@M` This method used to compute `last_outside` as a FLAG and return the
        RBF result regardless, while the docstring above said it refuses to
        extrapolate. It did not. 8.6% of our detections fall outside the hull -
        99.9% of them on-pitch at the far touchline - and out there the RBF
        displaced them a median 17.1 m, p90 59.4 m, max 81 m.

        Path length is exactly what wild displacements inflate, so the "+23% the
        shipped map compresses" was measured with those displacements in it.
        Honouring the hull moves the oracle's ratio_raw 0.958 -> 0.764 against a
        shipped 0.756: **the gain very largely evaporates.**

        A guard described in a docstring and absent from the code is the failure
        this repo keeps finding in its own instruments. This is that failure
        inside the experiment that produced the week's headline.
        """
        uv = np.asarray(uv, float).reshape(-1, 2)
        out = self._rbf(uv)
        self.last_outside = self._hull.find_simplex(uv) < 0
        if base is not None and self.last_outside.any():
            base = np.asarray(base, float).reshape(-1, 2)
            out[self.last_outside] = base[self.last_outside]
        return out


def build_oracle(cal, jn: dict, fit_every: int = 1, test_every: int = 5,
                 neighbors: int = 64, smoothing: float = ORACLE_SMOOTHING,
                 verbose: bool = True):
    """Fit the oracle on one temporal half of the clip, score it on the other.

    Held out IN TIME, not at random: neighbouring frames of the same player are
    the same patch of grass, so a random split would report the interpolator's
    own smoothness as accuracy.
    """
    px, (_fps, gm) = load_gt_px(), load_gt_m()
    off, perm = jn["offset"], jn["perm"]
    frames = sorted(px)
    cut = frames[len(frames) // 2]

    def pairs(sel, every):
        U, X = [], []
        for c in sel[::every]:
            g = gm.get(c + off)
            if not g:
                continue
            for i, foot in px[c].items():
                j = perm.get(i)
                if j in g:
                    U.append(foot)
                    X.append(g[j])
        return np.array(U, float), np.array(X, float)

    fit_u, fit_x = pairs([f for f in frames if f < cut], fit_every)
    te_u, te_x = pairs([f for f in frames if f >= cut], test_every)
    t0 = time.time()
    orc = Oracle(fit_u, fit_x, neighbors, smoothing)
    pred = orc(te_u)
    d = np.linalg.norm(pred - te_x, axis=1)
    if verbose:
        log(f"  oracle: fit on {len(fit_u)} GT feet (first half of the clip), "
            f"tested on {len(te_u)} (second half), {time.time() - t0:.1f} s")
        log(f"    HELD-OUT residual  median {np.median(d):.3f} m   "
            f"p90 {np.percentile(d, 90):.3f} m   "
            f"within 2 m {100 * (d < 2).mean():.1f}%")
        log(f"    (the reference's own quantisation is 1.05 x 0.68 m, i.e. "
            f"sigma 0.30/0.20 m of pure noise, so this is a ceiling on the "
            f"ceiling)")
    # THE SHIPPED MAP ON EXACTLY THE SAME HELD-OUT POINTS. Without this the
    # oracle's 0.5 m is a number with nothing to be better than, and the
    # registered `calib.metres.118575.overall` cannot stand in for it - that is
    # a different harness, on 20 frames, through the vendor's chain rather than
    # the job's.
    ds = np.linalg.norm(np.asarray(cal.to_pitch(raw_to_band(cal, te_u)), float)
                        - te_x, axis=1)
    if verbose:
        log(f"    SHIPPED map, same points  median {np.median(ds):.3f} m   "
            f"p90 {np.percentile(ds, 90):.3f} m   "
            f"within 2 m {100 * (ds < 2).mean():.1f}%")
    return orc, {"shipped_median_m": float(np.median(ds)),
                 "shipped_p90_m": float(np.percentile(ds, 90)),
                 "shipped_within_2m": float((ds < 2).mean()),
                 "held_out_median_m": float(np.median(d)),
                 "held_out_p90_m": float(np.percentile(d, 90)),
                 "held_out_within_2m": float((d < 2).mean()),
                 "n_fit": int(len(fit_u)), "n_test": int(len(te_u))}


def oracle_distance_control(cal, orc, jn, every: int = 1) -> dict:
    """Is the oracle map DISTANCE-NEUTRAL? The control the ratio depends on.

    A local interpolator fitted on positions quantised to a 1.05 x 0.68 m grid
    can inherit that grid as spatial wiggle: two pixels a step apart map to
    metre values a step apart, and distance is a SUM OF DISPLACEMENTS, so wiggle
    reads as running. `dist_eval`'s whole docstring is about that failure mode,
    and it applies to the MAP as much as to the filter.

    So: push the GT's own pixel feet through the oracle and measure the distance
    that comes out, against the distance of the GT metres those same pixels are
    annotated with, through the identical estimator. A neutral map reads 1.00.
    Anything else is a bandwidth artefact of the map and it lands on `ratio`
    directly, without saying anything about the product.
    """
    px, (_fps, gm) = load_gt_px(), load_gt_m()
    off, perm = jn["offset"], jn["perm"]
    per = collections.defaultdict(lambda: ([], [], []))
    for c in sorted(px)[::every]:
        g = gm.get(c + off)
        if not g:
            continue
        for i, foot in px[c].items():
            j = perm.get(i)
            if j in g:
                f_, u_, x_ = per[j]
                f_.append(c + off)
                u_.append(foot)
                x_.append(g[j])
    a = b = 0.0
    for j, (f_, u_, x_) in per.items():
        m_orc = orc(np.array(u_, float))
        m_gt = np.array(x_, float)
        for target, which in ((m_orc, 0), (m_gt, 1)):
            trk = [{"f": int(f), "id": j, "x": float(u), "y": float(v)}
                   for f, (u, v) in zip(f_, dist_eval.to_px(cal, target))]
            d, _o, _p, _c = dist_eval.track_metrics(trk, cal, 25.0)
            if which == 0:
                a += d
            else:
                b += d
    return {"oracle_km": a / 1000, "gt_km": b / 1000,
            "distance_neutrality": a / b if b else 0.0}


def space_recall(cal, orc, jn, rows, every: int = 25,
                 h_gates=(0.3, 0.6, 1.0), m_gates=(1.0, 2.0, 4.0)) -> dict:
    """The 21-point gap, re-measured on ONE set of rows in THREE spaces.

    `product.detect.recall_px` 0.845 and `recall_m` 0.633 come out of
    `detsweep.score`, which projects GT into pixels with `sceneprior`'s camera
    and gates at 0.6 x expected player height, then projects OUR feet into
    metres and gates at 2.0 m. Two spaces, two gates, and the gates are not
    each other's image: 0.6 of a player height is a VERTICAL extent, while a
    metre of pitch at the far side is a fraction of a pixel row. So part of
    that 21 points can be the gate rather than the geometry, and nothing had
    separated the two.

    Here the image-space comparison uses no calibration at all - our detections
    go band -> raw by table lookup and meet the MOT boxes in their own pixels -
    and the metre comparison is run under both maps. Same rows, same frames,
    exclusive nearest-first matching in every column, both knobs swept.
    """
    px, (_fps, gm) = load_gt_px(), load_gt_m()
    off, perm = jn["offset"], jn["perm"]
    hs = collections.defaultdict(dict)
    for line in open(GT_PX):
        p = line.split(",")
        hs[int(p[0])][int(p[1])] = float(p[5])

    by_f = collections.defaultdict(list)
    for r in rows:
        by_f[int(r["f"])].append(r)

    hit_px = {g: 0 for g in h_gates}
    hit_ms = {g: 0 for g in m_gates}
    hit_mo = {g: 0 for g in m_gates}
    n = 0

    def greedy(D, gate):
        """Exclusive nearest-first, detsweep's rule: one detection satisfies
        one GT player, so a single box in a crowd cannot cover three."""
        tg, td, k = set(), set(), 0
        for i, j in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
            if D[i, j] >= gate:
                break
            if i in tg or j in td:
                continue
            tg.add(int(i))
            td.add(int(j))
            k += 1
        return k

    for c in sorted(px)[::every]:
        ours = by_f.get(c + off)
        g_m = gm.get(c + off)
        if not ours or not g_m:
            continue
        ids = [i for i in px[c] if perm.get(i) in g_m]
        if not ids:
            continue
        G_px = np.array([px[c][i] for i in ids], float)
        G_m = np.array([g_m[perm[i]] for i in ids], float)
        G_h = np.array([hs[c][i] for i in ids], float)
        uv = np.array([[r["x"], r["y"]] for r in ours], float)
        raw = band_to_raw(cal, uv)
        D_px = np.linalg.norm(G_px[:, None] - raw[None], axis=2) / G_h[:, None]
        D_ms = np.linalg.norm(G_m[:, None] - np.asarray(cal.to_pitch(uv),
                                                        float)[None], axis=2)
        D_mo = np.linalg.norm(G_m[:, None] - orc(raw)[None], axis=2)
        for g in h_gates:
            hit_px[g] += greedy(D_px, g)
        for g in m_gates:
            hit_ms[g] += greedy(D_ms, g)
            hit_mo[g] += greedy(D_mo, g)
        n += len(ids)
    return {"n": n,
            "recall_px": {str(g): hit_px[g] / n for g in h_gates},
            "recall_m_shipped": {str(g): hit_ms[g] / n for g in m_gates},
            "recall_m_oracle": {str(g): hit_mo[g] / n for g in m_gates}}


def jitter_control(cal, orc, rows, fps, sigmas=(0.5, 1.0, 2.0, 4.0),
                   seed: int = 0) -> dict:
    """Does the oracle read MORE distance because it is right, or because it
    amplifies the detector's jitter? These are not the same finding.

    Distance is a sum of displacements, so a map with higher metres-per-pixel
    gain turns the same pixel jitter into more running. If the shipped map
    compresses the ground plane where our detections sit, then correcting it
    scales the true motion AND the noise by the same factor - the path length
    genuinely was under-read. If instead the oracle is simply noisier, the extra
    kilometres are an artefact and the ratio means nothing.

    The two are separated by injecting the SAME pixel jitter, at the SAME
    detections, and reading each map's amplification. Equal amplification =>
    the difference between the maps is geometry. Oracle amplifying harder =>
    the difference is noise.

    Jitter goes in BAND pixels, which is where the detector's error actually
    lives: `model/engine/detectors.py` runs on the frame `undistort_frame` produces.
    """
    rng = np.random.default_rng(seed)
    by_id = collections.defaultdict(list)
    for r in rows:
        by_id[int(r["id"])].append(r)
    uv_all = np.array([[r["x"], r["y"]] for r in rows], float)
    raw_all = band_to_raw(cal, uv_all)
    base = {"shipped": np.asarray(cal.to_pitch(uv_all), float),
            "oracle": orc(raw_all)}

    # Local gain: metres of pitch per band pixel, at the places our detections
    # actually are. The mechanism, measured rather than argued.
    gain = {}
    for k, fn in (("shipped", lambda p: np.asarray(cal.to_pitch(p), float)),
                  ("oracle", lambda p: orc(band_to_raw(cal, p)))):
        du = np.linalg.norm(fn(uv_all + [1.0, 0.0]) - base[k], axis=1)
        dv = np.linalg.norm(fn(uv_all + [0.0, 1.0]) - base[k], axis=1)
        gain[k] = {"median_m_per_px_u": float(np.median(du)),
                   "median_m_per_px_v": float(np.median(dv))}

    def total(xy):
        idx, out = 0, 0.0
        for tid, trk in by_id.items():
            n = len(trk)
            if n >= 2:
                trk2 = [{"f": r["f"], "id": tid, "x": float(u), "y": float(v)}
                        for r, (u, v) in zip(trk, dist_eval.to_px(cal, xy[idx:idx + n]))]
                out += dist_eval.track_metrics(trk2, cal, fps)[0]
            idx += n
        return out

    # `total` walks xy in row order, so the rows must be grouped by id and
    # sorted in time first. Getting this wrong does not raise - it silently
    # measures the distance between one player and the next.
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["id"]), rows[i]["f"]))
    rows_g = [rows[i] for i in order]
    by_id = collections.defaultdict(list)
    for r in rows_g:
        by_id[int(r["id"])].append(r)
    uv_g = uv_all[order]
    b = {"shipped": base["shipped"][order], "oracle": base["oracle"][order]}
    out = {"gain": gain, "clean_km": {k: total(v) / 1000 for k, v in b.items()},
           "sigmas": {}}
    for s in sigmas:
        n = rng.normal(0, s, uv_g.shape)
        noisy = {"shipped": np.asarray(cal.to_pitch(uv_g + n), float),
                 "oracle": orc(band_to_raw(cal, uv_g + n))}
        out["sigmas"][str(s)] = {
            k: total(noisy[k]) / 1000 for k in ("shipped", "oracle")}
    return out


def relabel(cal, orc: Oracle, rows: list[dict]) -> tuple[list[dict], dict]:
    """Our detections, re-expressed so the SHIPPED calibrator reads the ORACLE.

    This is how "substitute at the calibration step only" is implemented, and
    the implementation matters. Each detection's band pixel goes band -> raw ->
    oracle metres, and the row is then rewritten as the pseudo-pixel that the
    SHIPPED homography maps to those metres. So:

      * `f` and `id` are untouched - detection and tracking are held fixed, and
        every fragment, gap and swap the tracker made is still there;
      * every downstream consumer keeps using the shipped `cal`, so
        `kalman_ground` differentiates the SAME matrix for its measurement
        noise in both runs. The estimator's bandwidth is not a variable here,
        which is the only way the two ratios are comparable;
      * `dist_eval.to_px`'s round-trip assertion still holds, because the map
        in play is still a homography.

    A pseudo-pixel sits at most a metre or two from the real one, so the
    Jacobian is evaluated at essentially the same patch of grass - and arguably
    the righter one.
    """
    uv = np.array([[r["x"], r["y"]] for r in rows], float)
    t0 = time.time()
    xy = np.empty((len(uv), 2))
    outside = 0
    for i in range(0, len(uv), 200000):
        raw = band_to_raw(cal, uv[i:i + 200000])
        # The shipped map's own answer is the out-of-hull fallback the Oracle
        # docstring promises. Without it the oracle extrapolates onto the far
        # touchline and the path length it invents reads as a calibration win.
        shipped = np.asarray(cal.to_pitch(uv[i:i + 200000]), float)
        xy[i:i + 200000] = orc(raw, base=shipped)
        outside += int(orc.last_outside.sum())
        log(f"    oracle {min(i + 200000, len(uv))}/{len(uv)} "
            f"({time.time() - t0:.0f} s)")
    px = dist_eval.to_px(cal, xy)
    moved = np.linalg.norm(np.asarray(cal.to_pitch(uv), float) - xy, axis=1)
    return ([{**r, "x": float(u), "y": float(v)} for r, (u, v) in zip(rows, px)],
            {"outside_hull": outside, "n": len(uv),
             "shift_median_m": float(np.median(moved)),
             "shift_p90_m": float(np.percentile(moved, 90))})


# --------------------------------------------------------------------------
# scoring: dist_eval's own machinery, with the two calibrators kept apart

def score(rows, fps, cal_ours, cal_ref, gt, gt_frames, stride, gate_m,
          window_s=None):
    """dist_eval.evaluate, with the map under test used ONLY for our tracks.

    The GT reference distance is computed with `cal_ref` in BOTH runs. It is a
    property of the reference alone - GT metres pushed to pixels and back
    through the same matrix - so holding it fixed is what keeps `ratio` a
    statement about our tracks rather than about two different references.
    """
    keep_gate, contam_eval.GATE_M = contam_eval.GATE_M, gate_m
    try:
        owner = contam_eval.owners(rows, cal_ours, gt_frames, stride)
    finally:
        contam_eval.GATE_M = keep_gate

    votes = collections.defaultdict(collections.Counter)
    for i, g in owner.items():
        votes[int(rows[i]["id"])][g] += 1
    dominant = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}

    by_id = collections.defaultdict(list)
    for r in rows:
        by_id[int(r["id"])].append(r)

    ours = collections.defaultdict(float)
    frames = collections.defaultdict(set)
    orphan_m, orphan_ids = 0.0, 0
    for tid, trk in by_id.items():
        if len(trk) < 2:
            continue
        m, _o, _p, _c = dist_eval.track_metrics(trk, cal_ours, fps, window_s=window_s)
        g = dominant.get(tid)
        if g is None:
            orphan_m += m
            orphan_ids += 1
            continue
        ours[g] += m
        frames[g].update(int(r["f"]) for r in trk)

    per, covs = [], []
    for g, t in sorted(gt.items(), key=lambda kv: int(kv[0])):
        gpx = dist_eval.to_px(cal_ref, t["xy"])
        trk = [{"f": int(f), "id": g, "x": x, "y": y}
               for f, (x, y) in zip(t["f"], gpx)]
        full_m, _o, _p, _c = dist_eval.track_metrics(trk, cal_ref, fps, "kalman",
                                                    window_s=window_s)
        seen = frames.get(g, set())
        have = np.fromiter((int(f) in seen for f in t["f"]), bool, len(t["f"]))
        matched_m, _o, _p, _c = dist_eval.track_metrics(
            trk, cal_ref, fps, "kalman", real_override=have[:-1] & have[1:],
            window_s=window_s)
        cov = len(seen & set(t["f"].tolist())) / len(t["f"])
        covs.append(cov)
        per.append({"gt_id": g, "gt_m": full_m, "gt_matched_m": matched_m,
                    "ours_m": ours.get(g, 0.0), "coverage": cov,
                    "ratio_raw": ours.get(g, 0.0) / full_m if full_m else 0.0,
                    "ratio_matched": (ours.get(g, 0.0) / matched_m
                                      if matched_m else 0.0)})
    sq_gt = sum(p["gt_m"] for p in per)
    sq_gtm = sum(p["gt_matched_m"] for p in per)
    sq_ours = sum(p["ours_m"] for p in per)
    return {"coverage": float(np.mean(covs)),
            "ratio_raw": sq_ours / sq_gt if sq_gt else 0.0,
            "ratio_matched": sq_ours / sq_gtm if sq_gtm else 0.0,
            "gt_km": sq_gt / 1000, "gt_matched_km": sq_gtm / 1000,
            "ours_km": sq_ours / 1000, "gate_m": gate_m,
            "attributed_ids": len(dominant), "ids": len(by_id),
            "orphan_ids": orphan_ids, "orphan_km": orphan_m / 1000,
            "players": per}


def load_rows(tracks: str, lo: int | None = None, hi: int | None = None):
    blob = json.load(open(os.path.join(JOB, tracks)))
    rows = blob["detections"]
    if lo is not None:
        rows = [r for r in rows if lo <= r["f"] < hi]
    return rows, float(blob.get("fps", 25.0))


def window_gt(gt, lo, hi):
    """dist_eval's GT dict, clipped to a frame window."""
    out = {}
    for g, t in gt.items():
        m = (t["f"] >= lo) & (t["f"] < hi)
        if m.sum() > 1:
            out[g] = {**t, "f": t["f"][m], "xy": t["xy"][m]}
    return out


# --------------------------------------------------------------------------

def selfcheck():
    """Every convention this file depends on, asserted against the files."""
    cal = calibrators.load(os.path.join(JOB, "calibration.json"))

    # 1. BOX CONVENTIONS. tracks.json is bottom-centre (model/engine/track.py :: track
    #    writes (fx1+fx2)/2, fy2); MOT gt is top-left. Assert the MOT reader
    #    puts the foot BELOW the box top by exactly h, and centred.
    raw = open(GT_PX).readline().split(",")
    x, y, w, h = (float(v) for v in raw[2:6])
    got = load_gt_px()[int(raw[0])][int(raw[1])]
    assert abs(got[0] - (x + w / 2)) < 1e-9 and abs(got[1] - (y + h)) < 1e-9, got
    assert got[1] > y, "MOT foot must be BELOW the box top - top-left origin"
    src = open(os.path.join(ROOT, "model/engine/track.py")).read()
    assert '"x": (fx1 + fx2) / 2, "y": fy2' in src, \
        "model/engine/track.py no longer writes bottom-centre; this harness assumes it"

    # 2. SPACES. tracks.split.json must be in the band canvas, not the raw
    #    panorama - the two differ by 2.9x and a crop, and nothing downstream
    #    would notice.
    assert cal._mapx is not None and cal._mapx.shape == (870, 11878), cal._mapx.shape
    rows, _fps = load_rows("tracks.split.json", 0, 200)
    ux = np.array([r["x"] for r in rows])
    assert ux.max() > 4096, ("tracks look like raw panorama pixels, not band "
                             f"pixels (max x {ux.max():.0f})")

    # 3. band -> raw is the table lookup, and it round-trips through the pitch:
    #    a band point and its raw partner must land on the same patch of grass
    #    under their own maps. This is the check that catches a crop/scale drift.
    probe = np.array([[3000.0, 700.0], [6000.0, 600.0], [9000.0, 750.0]])
    r = band_to_raw(cal, probe)
    assert ((r[:, 0] > 0) & (r[:, 0] < 4096) & (r[:, 1] > 0) & (r[:, 1] < 1080)).all(), r
    d = np.linalg.norm(coarse_raw_to_m(cal)(r) - cal.to_pitch(probe), axis=1)
    assert d.max() < 12.0, ("band->raw->metres disagrees with band->metres by "
                            f"{d.max():.1f} m; the two maps are not the same camera")

    # 4. THE CEILING CONTROL IS REACHABLE. Feed GT through as our own tracks
    #    with a perfect map and the scorer must read 1.00/1.00. Without this the
    #    experiment cannot distinguish "calibration is not the problem" from
    #    "this harness cannot show a calibration win".
    _fps, gtd = dist_eval.load_gt()
    lo, hi = 6000, 6600
    gw = window_gt(gtd, lo, hi)
    ctrl = dist_eval.gt_as_prediction(gw, cal)
    r = score(ctrl, 25.0, cal, cal, gw, contam_eval.load_gt(), 10, 2.0)
    assert r["coverage"] > 0.999, r["coverage"]
    assert abs(r["ratio_matched"] - 1.0) < 0.02, r["ratio_matched"]
    assert abs(r["ratio_raw"] - 1.0) < 0.02, r["ratio_raw"]

    # 5. AND IT IS NOT VACUOUS: the same control through a map that is 3 m
    #    wrong must LOSE coverage. A scorer that reads 1.00 either way is
    #    measuring nothing.
    class Shifted:
        H = np.asarray(cal.H, float)
        pitch = cal.pitch
        @staticmethod
        def to_pitch(px):
            return np.asarray(cal.to_pitch(px), float) + np.array([3.0, 0.0])
    r2 = score(ctrl, 25.0, Shifted, cal, gw, contam_eval.load_gt(), 10, 2.0)
    assert r2["coverage"] < 0.5, ("a 3 m calibration error must cost coverage at "
                                  f"a 2 m gate; got {r2['coverage']:.3f}")

    # 6. raw -> band must invert band -> raw. It is only used to score the
    #    shipped map on the oracle's own held-out points, and a sloppy inverse
    #    there would flatter the oracle by blaming its rival for the inversion.
    back = raw_to_band(cal, band_to_raw(cal, probe))
    assert np.abs(back - probe).max() < 0.5, (back - probe)

    # 7. THE ORACLE HAS TO BE ONE. Fitted and scored on frames held out in
    #    time, it must beat the shipped map on those same points - and it must
    #    not buy that by adding path length of its own, or `ratio` is measuring
    #    the interpolator. Both thresholds are the ones the run is read against.
    jp = os.path.join(OUT, "calib_hypothesis_join.json")
    if os.path.exists(jp):
        jn = json.load(open(jp))
        jn["perm"] = {int(k): v for k, v in jn["perm"].items()}
        _o, st = build_oracle(cal, jn, smoothing=10000.0, verbose=False)
        assert st["held_out_median_m"] < 0.8, st
        assert st["held_out_median_m"] < st["shipped_median_m"], st
        n = oracle_distance_control(cal, _o, jn, every=10)
        assert 0.95 < n["distance_neutrality"] < 1.05, n
        log(f"  oracle held out: {st['held_out_median_m']:.3f} m vs shipped "
            f"{st['shipped_median_m']:.3f} m on the same points; "
            f"distance-neutral at {n['distance_neutrality']:.3f}")

    log("selfcheck ok  (MOT top-left -> foot, tracks in band space, band->raw "
        "agrees with band->metres, GT-as-prediction reads 1.00/1.00 and a 3 m "
        f"shift breaks it to {r2['coverage']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--join", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tracks", default="tracks.split.json")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--gates", type=float, nargs="*",
                    default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--jitter", action="store_true",
                    help="run the geometry-vs-noise control (slow: 5 full passes)")
    ap.add_argument("--clip-only", action="store_true",
                    help="score only the 6,000 frames the oracle was fitted in")
    ap.add_argument("--neighbors", type=int, default=64)
    ap.add_argument("--smoothing", type=float, default=ORACLE_SMOOTHING)
    ap.add_argument("--sweep-smooth", type=float, nargs="*", default=None,
                    help="fit the oracle at each smoothing and report accuracy "
                         "against distance-neutrality, then stop")
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck()

    cal = calibrators.load(os.path.join(JOB, "calibration.json"))
    jpath = os.path.join(OUT, "calib_hypothesis_join.json")
    if a.join or not os.path.exists(jpath):
        log("JOIN pass 1: coarse raw homography (~7 m), frame offset + id map")
        jn = join(cal)
        log("JOIN pass 2: the same search through an oracle fitted on pass 1")
        orc0, _ = build_oracle(cal, jn, fit_every=10, verbose=False)
        jn2 = join(cal, rough=orc0, guess=jn["offset"], span=25)
        if (jn2["offset"], jn2["perm"]) != (jn["offset"], jn["perm"]):
            log(f"  !! pass 2 disagrees with pass 1 (offset {jn['offset']} -> "
                f"{jn2['offset']}); taking pass 2 and re-fitting")
        jn = jn2
        os.makedirs(OUT, exist_ok=True)
        json.dump(jn, open(jpath, "w"), indent=1)
    else:
        jn = json.load(open(jpath))
        # JSON has no integer keys. Left as strings, `perm.get(mot_id)` returns
        # None for every player and the oracle silently fits on zero points.
        jn["perm"] = {int(k): v for k, v in jn["perm"].items()}
        log(f"JOIN (cached): offset {jn['offset']}, identity perm "
            f"{jn['identity_perm']}, cost {jn['cost_m']:.2f} m")
    if not a.run:
        return

    log("\nORACLE: pixel -> metres, regressed on GT, held out in time")
    if a.sweep_smooth:
        # SWEEP THE KNOB. Smoothing trades the map's positional accuracy against
        # its spatial wiggle, and wiggle is distance. Reporting one smoothing
        # value would be C1 again: a conclusion drawn from a constant nobody
        # checked against the distribution it cut.
        log(f"{'smooth':>8} {'held-out med m':>15} {'p90 m':>7} {'<2m':>6} "
            f"{'distance-neutrality':>20}")
        for s in a.sweep_smooth:
            o, st = build_oracle(cal, jn, neighbors=a.neighbors, smoothing=s,
                                 verbose=False)
            n = oracle_distance_control(cal, o, jn, every=2)
            log(f"{s:8.4g} {st['held_out_median_m']:15.3f} "
                f"{st['held_out_p90_m']:7.3f} {st['held_out_within_2m']:6.3f} "
                f"{n['distance_neutrality']:20.3f}")
        return
    orc, ostat = build_oracle(cal, jn, neighbors=a.neighbors,
                              smoothing=a.smoothing)
    neu = oracle_distance_control(cal, orc, jn, every=2)
    ostat.update(neu)
    log(f"    DISTANCE NEUTRALITY: GT's own feet through the oracle read "
        f"{neu['oracle_km']:.2f} km against the GT metres' {neu['gt_km']:.2f} km "
        f"= {neu['distance_neutrality']:.3f}")
    log("    (1.00 = the map adds no path length of its own. Anything above it "
        "is map wiggle and lands on ratio, not on the product.)")

    lo, hi = (jn["offset"], jn["offset"] + 6000) if a.clip_only else (None, None)
    rows, fps = load_rows(a.tracks, lo, hi)
    _gfps, gtd = dist_eval.load_gt()
    gtw = window_gt(gtd, lo, hi) if lo is not None else gtd
    gtf = contam_eval.load_gt()
    scope = (f"frames {lo}-{hi}" if lo is not None else "full 1st half")
    log(f"\nRUN: {a.tracks}, {scope}, {len(rows)} detections, "
        f"{len({r['id'] for r in rows})} ids, {len(gtw)} GT players")

    log("\n  TWO SPACES, ONE SET OF ROWS: where the 21 points actually go")
    sr = space_recall(cal, orc, jn, rows)
    log(f"    n = {sr['n']} GT player-instants (every 25th clip frame)")
    log("    IMAGE, no calibration in it   " + "  ".join(
        f"{g} h: {v:.3f}" for g, v in sr["recall_px"].items()))
    log("    METRES, shipped map           " + "  ".join(
        f"{g} m: {v:.3f}" for g, v in sr["recall_m_shipped"].items()))
    log("    METRES, oracle map            " + "  ".join(
        f"{g} m: {v:.3f}" for g, v in sr["recall_m_oracle"].items()))

    if a.jitter:
        log("\n  GEOMETRY OR NOISE: the same pixel jitter through both maps")
        jt = jitter_control(cal, orc, rows, fps)
        g = jt["gain"]
        log(f"    local gain, metres per band pixel (median over our own "
            f"detections):")
        for k in ("shipped", "oracle"):
            log(f"      {k:8s} u {g[k]['median_m_per_px_u']:.4f}  "
                f"v {g[k]['median_m_per_px_v']:.4f}")
        log(f"    {'sigma px':>9} {'shipped km':>11} {'oracle km':>10} "
            f"{'shipped x':>10} {'oracle x':>9}")
        c = jt["clean_km"]
        log(f"{0.0:>12.1f} {c['shipped']:11.2f} {c['oracle']:10.2f} "
            f"{1.0:10.3f} {1.0:9.3f}")
        for s, v in jt["sigmas"].items():
            log(f"{float(s):>12.1f} {v['shipped']:11.2f} {v['oracle']:10.2f} "
                f"{v['shipped'] / c['shipped']:10.3f} "
                f"{v['oracle'] / c['oracle']:9.3f}")
        out_jt = jt
    else:
        out_jt = None

    log("\n  re-expressing every detection through the oracle map")
    rows_o, rstat = relabel(cal, orc, rows)
    log(f"  oracle moved our detections a median {rstat['shift_median_m']:.2f} m "
        f"(p90 {rstat['shift_p90_m']:.2f} m); "
        f"{rstat['outside_hull']}/{rstat['n']} "
        f"({100 * rstat['outside_hull'] / rstat['n']:.1f}%) fell outside the GT hull")

    # THE CEILING, on the same rows and the same window: GT fed in as our own
    # tracker's output. Whatever this reads is what a PERFECT map plus PERFECT
    # detection and tracking can score here, and it is what says the experiment
    # could have come out the other way.
    ceil = score(dist_eval.gt_as_prediction(gtw, cal), fps, cal, cal, gtw, gtf,
                 a.stride, 2.0)
    log(f"  CEILING control (GT as prediction, 2 m gate): coverage "
        f"{ceil['coverage']:.3f}  ratio_raw {ceil['ratio_raw']:.3f}  "
        f"ratio_matched {ceil['ratio_matched']:.3f}")

    out = {"scope": scope, "tracks": a.tracks, "oracle": ostat, "space_recall": sr, "jitter": out_jt,
           "relabel": rstat, "join": jn,
           "ceiling": {k: v for k, v in ceil.items() if k != "players"},
           "shipped": {}, "oracle_run": {}}
    log(f"\n{'gate m':>7} | {'SHIPPED cov':>11} {'ratio_raw':>10} {'r_match':>8} "
        f"| {'ORACLE cov':>11} {'ratio_raw':>10} {'r_match':>8}")
    for g in a.gates:
        s = score(rows, fps, cal, cal, gtw, gtf, a.stride, g)
        o = score(rows_o, fps, cal, cal, gtw, gtf, a.stride, g)
        out["shipped"][str(g)] = {k: v for k, v in s.items() if k != "players"}
        out["oracle_run"][str(g)] = {k: v for k, v in o.items() if k != "players"}
        if g == 2.0:
            out["players_shipped"], out["players_oracle"] = s["players"], o["players"]
        log(f"{g:7.1f} | {s['coverage']:11.3f} {s['ratio_raw']:10.3f} "
            f"{s['ratio_matched']:8.3f} | {o['coverage']:11.3f} "
            f"{o['ratio_raw']:10.3f} {o['ratio_matched']:8.3f}")
    p = os.path.join(OUT, "calib_hypothesis.json")
    json.dump(out, open(p, "w"), indent=1)
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
