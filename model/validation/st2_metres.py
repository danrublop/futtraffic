"""How good is pixel -> METRES on the STC clips? Scored against ST2's own truth.

WHY THIS EXISTS
    `st2_overlay.check()` scores the FORWARD direction - metres to pixels -
    against the painted lines, and 128057 lands at 3.0 px. That is not the
    direction the product uses. Distance covered, speed, possession zones and
    every game-state channel of the event model need pixel -> metres, and
    nothing had measured that on these clips.

    It is not the same question. The chain compresses 68 m of pitch depth into
    ~213 px on the remapped canvas, so it is well conditioned going out and
    badly conditioned coming back, and a small pixel error at the far side is
    metres.

THE CHAIN IS A LOOKUP, NOT A MATRIX
    pitch --H--> remapped --mapx--> panorama has no closed-form inverse, so the
    forward map is evaluated on a 0.25 m grid and inverted with a KD-tree.

    A DIRECT `cv2.invert(H)` IS THE TRAP HERE. Applying H alone lands on the
    REMAPPED canvas, where the pitch really is a 213 px band - which reads as a
    broken calibration and is not one. Skipping the mapx step is what makes the
    published homography look unusable; with it, the drawn pitch sits 3.0 px
    from the paint.

TRUTH
    `{match}_tracker_box_data.xml` carries a per-frame `loc` in normalised pitch
    coordinates for every player. Clip frame 0 is FIRST_HALF frameNumber 278 on
    128057 (see unused-constraints (not published), established by speed correlation and
    confirmed geometrically at 16.7 px median against 290-400 px at wrong
    offsets). Orientation is checked, not assumed: as-is scores 1.26 m and
    point-reflected 36.97 m, so this feed needs no reflection - unlike the event
    feed, which does.

WHAT IT FOUND (128057, 20 frames, 440 player-instants)

        overall            median 1.26 m   p90 5.26 m
        nearest quartile   median 0.74 m   p90 1.63 m
        farthest quartile  median 3.23 m   p90 9.46 m

    Error scales with distance from the camera, 4.4x across the box-height
    range. Near-side positions are good; far-side positions carry several
    metres. That is the same story as the repo's recorded 1.010 m on 118575 and
    its 0.53 m-per-pixel far band, reached independently on different footage.

    UPPER BOUND, NOT AN ESTIMATE: ST2's own `loc` is written to 1-2 decimals,
    i.e. quantised at ~1.05 m in x and ~0.68 m in y. Roughly 0.35 m of the
    1.26 m median is the truth's own rounding, not our error.

    Hungarian assignment pairs our inverted points to truth points, so a bad
    mapping can mis-pair and UNDERSTATE its own error. The numbers are a floor.
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from st2_overlay import pitch_of, projector  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

GT = _ST2 + "/mot"
RAW = _ST2 + "/raw"
# Clip frame 0 -> FIRST_HALF frameNumber. unused-constraints (not published).
OFFSET = {"118575": 6246, "128057": 278, "132831": 26}
WH = {"118575": (4096, 1080), "128057": (4096, 1080),
      "132831": (3840, 1504)}


CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "footage")


def _knots(xy, P):
    """Pitch metres -> isotropic knot space, so the TPS is not stretched by 105/68."""
    return np.c_[np.asarray(xy, float)[:, 0] / P.length,
                 np.asarray(xy, float)[:, 1] / P.width] * 100.0


TPS_SMOOTHING = 10000.0


def tps_field(seq, fn, P, smoothing=TPS_SMOOTHING):
    """Residual-correction field fitted on the calibration landmarks.

    The shipped chain leaves a residual that is a SMOOTH RAMP, not noise
    (research-farfield-engineering (not published) 1.4), so a thin-plate spline through
    the landmarks removes most of it. Returns a fn(pitch_xy) -> (N,2) pixel
    correction to SUBTRACT from the forward map, or None if the sequence ships
    no landmarks - only 118575 does; they are the vendor's, not ours.

    IT REFUSES TO CORRECT OUTSIDE THE LANDMARK HULL. Extrapolating a TPS is
    4.25 m of confident nonsense (same doc, 3.3), so outside the hull the
    correction is zero and the caller gets the uncorrected chain.

    THE DEFAULT SMOOTHING WAS 100.0 AND THAT WAS A SHIPPED REGRESSION.
    `@M` A claims audit (2026-09-10) found this function hardcoded 100.0 with no
    way to change it, while every result quoted in the docs used 10000.0. At
    100.0 the shipped `--tps` makes the product WORSE - overall median
    1.36 -> 2.11 m, near quartile 0.81 -> 1.53 m - so the flag as shipped did the
    opposite of what the documents claimed for it, and no command line could
    produce the documented numbers.

    10000.0 is the value BOTH halves of the clip select independently when the
    smoothing is chosen on player positions rather than on the landmarks
    (context-pitch-mapping-pivot (not published) 9.10). It is exposed as
    `--tps-smooth` so the sweep behind that choice stays reproducible, and
    `demo()` asserts the default is not the regressing one.
    """
    path = f"{CALIB}/soccertrack_calib_{seq}/{seq}_keypoints.json"
    if not os.path.exists(path):
        return None
    kp = json.load(open(path))
    L = np.array([[float(n) for n in re.findall(r"-?[\d.]+", k)] for k in kp])
    Q = np.array(list(kp.values()), float)
    px, ok = fn(L)
    L, E = L[ok], (px - Q)[ok]
    S = _knots(L, P)
    rbf = RBFInterpolator(S, E, kernel="thin_plate_spline", smoothing=smoothing)
    hull = Delaunay(S)

    def corr(xy):
        s = _knots(xy, P)
        d = rbf(s)
        d[hull.find_simplex(s) < 0] = 0.0
        return d
    return corr


def inverse(seq, step_m=0.25, tps=False, smoothing=TPS_SMOOTHING):
    """pixel -> pitch metres, by KD-tree over the forward map. -> (fn, n_grid)."""
    P = pitch_of(seq)
    fn = projector(seq, WH[seq])
    gx = np.arange(0, P.length + 1e-9, step_m)
    gy = np.arange(0, P.width + 1e-9, step_m)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    px, ok = fn(G)
    if tps:
        corr = tps_field(seq, fn, P, smoothing)
        if corr is None:
            raise SystemExit(f"{seq} ships no keypoints; --tps has nothing to fit")
        px = px - corr(G)
    G, px = G[ok], px[ok]
    if len(G) < 100:
        return None, 0
    tree = cKDTree(px)
    return (lambda uv: G[tree.query(np.asarray(uv, float))[1]]), len(G)


def gt_feet(seq):
    out = defaultdict(list)
    for line in open(f"{GT}/{seq}.txt"):
        p = line.split(",")
        x, y, w, h = (float(v) for v in p[2:6])
        out[int(p[0])].append((x + w / 2, y + h, h))
    return out


def st2_truth(seq, clip_frames):
    """{clip frame: (N,2) normalised loc}. Streams the 200 MB+ XML and stops."""
    off = OFFSET[seq]
    inv = {off + f: f for f in clip_frames}
    hi = max(inv) + 5
    out = {}
    for _, el in ET.iterparse(f"{RAW}/{seq}/{seq}_tracker_box_data.xml",
                              events=("end",)):
        # Only <frame> is cleared. Clearing <player> wipes its attributes before
        # the parent frame is ever read - iterparse ends the child first.
        if el.tag != "frame":
            continue
        if el.get("eventPeriod") == "FIRST_HALF":
            f = int(float(el.get("frameNumber")))
            if f in inv:
                out[inv[f]] = np.array(
                    [[float(t) for t in p.get("loc").strip("[]").split(",")]
                     for p in el.findall("player") if p.get("loc")])
            if f > hi:
                el.clear()
                break
        el.clear()
    return out


def run(seq, every=300, tps=False, smoothing=TPS_SMOOTHING):
    P = pitch_of(seq)
    inv_fn, n = inverse(seq, tps=tps, smoothing=smoothing)
    if inv_fn is None:
        print(f"{seq}: calibration unusable, nothing to measure")
        return float("nan")
    feet = gt_feet(seq)
    frames = list(range(0, max(feet) + 1, every))
    truth = st2_truth(seq, frames)
    rec = []
    for f, T in truth.items():
        if f not in feet or not len(T):
            continue
        T = np.c_[T[:, 0] * P.length, T[:, 1] * P.width]
        uv = np.array([(x, y) for x, y, _ in feet[f]])
        hs = np.array([h for _, _, h in feet[f]])
        M = inv_fn(uv)
        D = np.linalg.norm(M[:, None] - T[None], axis=2)
        r, c = linear_sum_assignment(D)
        rec += [(D[a, b], hs[a]) for a, b in zip(r, c)]
    rec = np.array(rec)
    d = rec[:, 0]
    print(f"\n{seq}{' +TPS' if tps else ''}: pixel -> metres, {len(truth)} frames, "
          f"{len(rec)} instants, "
          f"{n} grid points")
    print(f"  overall            median {np.median(d):5.2f} m   "
          f"p90 {np.percentile(d, 90):5.2f} m")
    # Fraction within a metric threshold, so these numbers are comparable to the
    # fields that actually publish ground-plane error: nuScenes matches by
    # ground-plane centre distance over {0.5, 1, 2, 4} m, and BrnoCompSpeed
    # reports distance error in metres rather than reprojection pixels.
    # research-crossdomain-registration (not published).
    print("  within        " + "  ".join(f"{t:g}m {100 * (d < t).mean():5.1f}%"
                                         for t in (0.5, 1.0, 2.0, 4.0)))
    q = np.percentile(rec[:, 1], [25, 50, 75])
    for lo, hi, lab in ((0, q[0], "farthest quartile"), (q[0], q[1], "25-50%"),
                        (q[1], q[2], "50-75%"), (q[2], 1e9, "nearest quartile")):
        m = (rec[:, 1] >= lo) & (rec[:, 1] < hi)
        if m.sum():
            print(f"  {lab:<18s} median {np.median(d[m]):5.2f} m   "
                  f"p90 {np.percentile(d[m], 90):5.2f} m")
    return float(np.median(d))


def demo():
    P = pitch_of("128057")
    fn = projector("128057", WH["128057"])
    inv_fn, n = inverse("128057", step_m=1.0)
    assert n > 1000, n
    # Round trip: metres -> pixels -> metres must return within the grid step.
    pts = np.array([[52.5, 34.0], [20.0, 10.0], [90.0, 60.0]])
    px, ok = fn(pts)
    assert ok.all()
    back = inv_fn(px)
    assert np.abs(back - pts).max() <= 1.5, back - pts
    # 132831's homography is degenerate; inverse() must say so, not crash.
    f2, n2 = inverse("132831", step_m=2.0)
    assert f2 is None or n2 >= 0

    # The TPS must beat the uncorrected chain ON ITS OWN LANDMARKS, held out one
    # at a time. A bad landmark set otherwise bends the pitch silently.
    P5 = pitch_of("118575")
    f5 = projector("118575", WH["118575"])
    kp = json.load(open(f"{CALIB}/soccertrack_calib_118575/118575_keypoints.json"))
    L = np.array([[float(n) for n in re.findall(r"-?[\d.]+", k)] for k in kp])
    E = f5(L)[0] - np.array(list(kp.values()), float)
    S = _knots(L, P5)
    loo = np.array([RBFInterpolator(np.delete(S, i, 0), np.delete(E, i, 0),
                                    kernel="thin_plate_spline",
                                    smoothing=100.0)(S[i:i + 1])[0]
                    for i in range(len(L))])
    a, b = np.median(np.hypot(*E.T)), np.median(np.hypot(*(E - loo).T))
    assert b < a, (a, b)
    # And it must refuse to extrapolate: no landmarks -> no field, and a point
    # outside the landmark hull gets exactly zero correction.
    assert tps_field("128057", projector("128057", WH["128057"]),
                     pitch_of("128057")) is None
    corr = tps_field("118575", f5, P5)
    assert np.abs(corr(np.array([[-40.0, -40.0], [200.0, 200.0]]))).max() == 0.0
    assert np.abs(corr(np.array([[52.5, 34.0]]))).max() > 0.0
    # The default must not be the value that regressed the product. A flag whose
    # default does the opposite of its documentation is worse than no flag.
    assert TPS_SMOOTHING >= 1000.0, \
        f"default smoothing {TPS_SMOOTHING} is in the regressing range"
    import inspect
    assert inspect.signature(tps_field).parameters["smoothing"].default \
        == TPS_SMOOTHING, "tps_field default must track TPS_SMOOTHING"
    # THE FLAG MUST REACH THE FIT. The first version of this fix rebound a
    # module constant in __main__, which cannot change a default already
    # captured in tps_field's signature - so --tps-smooth was inert and 100 and
    # 10000 produced identical output. Prove the correction differs.
    c_lo = tps_field("118575", f5, P5, smoothing=1.0)
    c_hi = tps_field("118575", f5, P5, smoothing=50000.0)
    probe = np.array([[52.5, 34.0], [20.0, 20.0], [80.0, 50.0]])
    assert np.abs(c_lo(probe) - c_hi(probe)).max() > 1.0, \
        "smoothing does not reach the fit - the flag is inert"
    print(f"ok  (landmark LOO {a:.1f} px -> {b:.1f} px; default smoothing "
          f"{TPS_SMOOTHING:g}; flag reaches the fit)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="128057")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--tps", action="store_true",
                    help="apply the landmark residual-correction field")
    ap.add_argument("--tps-smooth", type=float, default=TPS_SMOOTHING,
                    help="RBF smoothing; 10000 is what a held-out split selects, "
                         "100 (the old hardcoded value) is a regression")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else run(a.seq, a.every, a.tps, a.tps_smooth)
