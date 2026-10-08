"""Solve a real homography for a TeamTrack `soccer_top` clip, from pitch lines.

Why this exists: every validation run to date used H = identity, so the
anisotropic measurement covariance in stats.kalman_ground has never been
exercised. This produces a REAL H so it can be.

Method: the operator (me) reads approximate line positions off a downscaled
frame; this script then refines each pitch line to sub-pixel by fitting a
straight line through the white-pixel centroids inside a narrow band, and takes
landmark pixels as line INTERSECTIONS. Hand-clicking a corner is good to ~5-10
px at 4K; a line fit over 2000+ columns is good to well under one.

Pitch dimensions are NOT assumed. Penalty-area and goal-area sizes are fixed by
the Laws of the Game at every level, so they pin the metric scale; (length,
width) is then searched for as the pair that minimises reprojection RMS.

Run: .venv/bin/python model/validation/teamtrack_calib.py [--clip NAME] [--save]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "engine"))
from paths import FOOTAGE2, VALIDATION      # noqa: E402
from pitch import Calibration, Pitch        # noqa: E402

# Line positions, located by eye on a 1280x720 render of frame 0 of
# soccer_top__test__videos__D_20220220_1_1680_1710 and then snapped to the
# strongest white row/column in a window. Only the SEED matters - each line is
# refined below - but a seed more than ~a band away locks onto the wrong feature.
# name -> (seed pixel, [ranges of the long axis to fit over])
SEED_V = {           # image-vertical lines: x = m*y + c, `a` runs over rows
    "goal_l":  (468, [(140, 1990)]),
    "pen_l":   (917, [(545, 1620)]),
    "gbox_l":  (615, [(850, 1320)]),
    "half":    (1917, [(140, 1990)]),
    "gbox_r":  (3243, [(850, 1320)]),
    "pen_r":   (2930, [(545, 1620)]),
    "goal_r":  (3399, [(140, 1990)]),
}
# The penalty- and goal-area edges parallel to the touchlines are ONE world line
# each, shared by both ends of the pitch, so both ends are fitted together: a
# 115 px lever arm on one goal area gives a slope that is mostly noise, a 2900 px
# baseline across both does not. If the two ends disagreed, the per-line RMS
# printed below would blow up - it is the check on this shortcut.
SEED_H = {           # image-horizontal lines: y = m*x + c, `a` runs over columns
    "touch_t": (112, [(500, 3360)]),
    "pen_t":   (526, [(490, 900), (2950, 3380)]),
    "gbox_t":  (832, [(490, 605), (3255, 3380)]),
    "gbox_b":  (1334, [(490, 605), (3255, 3380)]),
    "pen_b":   (1637, [(490, 900), (2950, 3380)]),
    "touch_b": (2021, [(500, 3360)]),
}
BAND = 22            # +-px around the seed to search for the line
SNAP = 50            # +-px the seed may be re-centred by, to absorb drone pose drift
MIN_WHITE = 3        # pixels of line in a column before it counts


def white_mask(frame: np.ndarray) -> np.ndarray:
    """Pitch markings: bright and unsaturated. Grass is neither."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 70)).astype(np.uint8)


def fit_line(mask: np.ndarray, seed: float, span: tuple, vertical: bool,
             iters: int = 4):
    """Robust straight-line fit to one painted line. Returns (m, c, rms, n).

    `vertical` means the line runs down the image, so we fit x = m*y + c and
    scan one column of the band per row. Fitting the free axis as a function of
    the long axis avoids the infinite-slope case entirely. `span` is a list of
    (lo, hi) ranges so one world line can be fitted from two separated segments.
    """
    axis = [a for lo, hi in span for a in range(int(lo), int(hi))]
    # Snap the seed to the strongest white row/column nearby. The seeds were read
    # off ONE clip and the drone sits at a slightly different pose in each, so
    # without this every other clip silently locks onto grass.
    limit = mask.shape[1] if vertical else mask.shape[0]
    lo, hi = max(0, int(seed) - SNAP), min(limit, int(seed) + SNAP)
    block = mask[np.ix_(axis, range(lo, hi))] if vertical else mask[lo:hi, axis]
    counts = block.sum(axis=0 if vertical else 1)
    seed = lo + int(np.argmax(counts))
    m, c = 0.0, float(seed)
    pts = None
    # Wide band first, then narrow. A line that runs 46 px across the frame is
    # outside a 22 px band at both ends before the slope is known, and the fit
    # then locks onto whichever half it caught. Starting wide costs nothing.
    for band in (70, 40, 25) + (BAND,) * iters:
        a_vals, b_vals = [], []
        for a in axis:
            b0 = m * a + c
            lo, hi = max(0, int(b0 - band)), min(limit, int(b0 + band))
            # vertical line: `a` is a row, we scan across columns.
            strip = mask[a, lo:hi] if vertical else mask[lo:hi, a]
            if strip.sum() < MIN_WHITE:
                continue
            idx = np.flatnonzero(strip)
            # Reject a strip that has caught two things (a player boot and the
            # line): a real line is one contiguous run of a few px.
            if idx[-1] - idx[0] > 14:
                continue
            a_vals.append(a)
            b_vals.append(lo + idx.mean())
        if len(a_vals) < 20:
            raise ValueError(f"line at seed {seed} has only {len(a_vals)} samples")
        A, B = np.array(a_vals, float), np.array(b_vals, float)
        m, c = np.polyfit(A, B, 1)
        resid = B - (m * A + c)
        # MAD, not std: the pitch-name lettering painted beside the top
        # touchline is bright, unsaturated and 1000 px long, so it survives a
        # 3-sigma cut by inflating the sigma it is cut against.
        mad = float(np.median(np.abs(resid - np.median(resid))))
        keep = np.abs(resid) < max(1.5, 4.45 * mad)   # 4.45*MAD ~ 3 sigma
        A, B = A[keep], B[keep]
        m, c = np.polyfit(A, B, 1)
        pts = (A, B)
    resid = pts[1] - (m * pts[0] + c)
    return float(m), float(c), float(np.sqrt(np.mean(resid ** 2))), len(pts[0])


def intersect(v, h) -> tuple[float, float]:
    """(x = mv*y + cv) meets (y = mh*x + ch)."""
    mv, cv = v[0], v[1]
    mh, ch = h[0], h[1]
    y = (mh * cv + ch) / (1.0 - mh * mv)
    return (mv * y + cv, y)


def landmarks_from_frame(frame: np.ndarray) -> tuple[dict, dict]:
    """-> ({landmark name: (x_px, y_px)}, {line name: (m, c, rms_px, n)})."""
    mask = white_mask(frame)
    V = {k: fit_line(mask, s, sp, True) for k, (s, sp) in SEED_V.items()}
    Hl = {k: fit_line(mask, s, sp, False) for k, (s, sp) in SEED_H.items()}

    # Image is a nadir view rotated 90 deg from pitch coordinates:
    #   image left -> pitch x=0, image right -> pitch x=L
    #   image BOTTOM -> pitch y=0, image top -> pitch y=W
    p = {
        "corner_bl": intersect(V["goal_l"], Hl["touch_b"]),
        "corner_tl": intersect(V["goal_l"], Hl["touch_t"]),
        "corner_br": intersect(V["goal_r"], Hl["touch_b"]),
        "corner_tr": intersect(V["goal_r"], Hl["touch_t"]),
        "half_b": intersect(V["half"], Hl["touch_b"]),
        "half_t": intersect(V["half"], Hl["touch_t"]),
        "pen_l_bl": intersect(V["goal_l"], Hl["pen_b"]),
        "pen_l_tl": intersect(V["goal_l"], Hl["pen_t"]),
        "pen_l_br": intersect(V["pen_l"], Hl["pen_b"]),
        "pen_l_tr": intersect(V["pen_l"], Hl["pen_t"]),
        "goal_l_br": intersect(V["gbox_l"], Hl["gbox_b"]),
        "goal_l_tr": intersect(V["gbox_l"], Hl["gbox_t"]),
        "pen_r_bl": intersect(V["pen_r"], Hl["pen_b"]),
        "pen_r_tl": intersect(V["pen_r"], Hl["pen_t"]),
        "pen_r_br": intersect(V["goal_r"], Hl["pen_b"]),
        "pen_r_tr": intersect(V["goal_r"], Hl["pen_t"]),
        "goal_r_bl": intersect(V["gbox_r"], Hl["gbox_b"]),
        "goal_r_tl": intersect(V["gbox_r"], Hl["gbox_t"]),
    }
    return p, {**{f"V:{k}": v for k, v in V.items()},
               **{f"H:{k}": v for k, v in Hl.items()}}


def solve_dims(clicks: dict, grid=(np.arange(95.0, 115.01, 0.1),
                                   np.arange(58.0, 76.01, 0.1))):
    """Search (length, width) for the pair minimising reprojection RMS.

    The penalty and goal areas are law-fixed, so they hold the metric scale
    still while L and W move; the minimum is therefore a MEASUREMENT of the
    pitch, not a fit degree of freedom soaking up error. Never assume 105x68 -
    doing so is a 4-6% systematic on every distance (see pitch.py).
    """
    best = None
    for L in grid[0]:
        for W in grid[1]:
            cal = Calibration.solve(Pitch(length=float(L), width=float(W)), clicks)
            if best is None or cal.rms_error_m < best[0]:
                best = (cal.rms_error_m, float(L), float(W))
    return best[1], best[2], best[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="soccer_top__test__videos__D_20220220_1_1680_1710")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    vid = FOOTAGE2 / (a.clip + ".mp4")
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"cannot read frame {a.frame} of {vid.name}")
        return 1

    clicks, lines = landmarks_from_frame(frame)
    print(f"clip {a.clip}  frame {a.frame}  {frame.shape[1]}x{frame.shape[0]}\n")
    print("line fits (straightness is the perspective-model check):")
    for k, (m, c, rms, n) in sorted(lines.items()):
        print(f"  {k:12s} slope {m:+.5f}  rms {rms:.2f} px  n={n}")

    L, W, rms = solve_dims(clicks)
    pitch = Pitch(length=L, width=W)
    cal = Calibration.solve(pitch, clicks)
    print(f"\nmeasured pitch: {L:.1f} x {W:.1f} m   (105x68 assumed => "
          f"{Calibration.solve(Pitch(105.0, 68.0), clicks).rms_error_m:.3f} m rms)")
    print(f"homography rms {cal.rms_error_m:.3f} m over {len(clicks)} landmarks, "
          f"outliers {cal.outliers or 'none'}")

    lm = pitch.landmarks()
    proj = cal.to_pitch(np.array([clicks[n] for n in sorted(clicks)]))
    print("\nper-landmark residual (m):")
    for n, p in zip(sorted(clicks), proj):
        e = np.linalg.norm(p - np.array(lm[n]))
        print(f"  {n:11s} px({clicks[n][0]:8.1f},{clicks[n][1]:8.1f}) -> "
              f"({p[0]:6.2f},{p[1]:6.2f})  want ({lm[n][0]:6.2f},{lm[n][1]:6.2f})  "
              f"err {e:.3f}")

    import sceneprior
    cam = sceneprior.camera_from_homography(cal.H, (frame.shape[1], frame.shape[0]))
    print(f"\nrecovered camera: f={cam.focal_px:.0f} px, height={cam.height_m:.1f} m, "
          f"centre=({cam.centre_m[0]:.1f}, {cam.centre_m[1]:.1f}) m, "
          f"mount_ok={cam.mount_ok}")

    if a.save:
        out = VALIDATION / f"calib_{a.clip}.json"
        cal.save(str(out))
        print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    """Synthetic lines in, exact intersections out."""
    # A 200x200 canvas with one vertical and one horizontal white line whose
    # crossing we know exactly.
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, 98:101] = 255            # vertical, centre x = 99
    img[48:51, :] = 255             # horizontal, centre y = 49
    mask = white_mask(img)
    v = fit_line(mask, 99, [(80, 190)], True)
    h = fit_line(mask, 49, [(110, 190)], False)
    x, y = intersect(v, h)
    assert abs(x - 99) < 0.2 and abs(y - 49) < 0.2, (x, y)

    # A tilted line must come back with the right slope.
    img2 = np.zeros((400, 400, 3), np.uint8)
    for yy in range(400):
        img2[yy, int(100 + 0.25 * yy):int(100 + 0.25 * yy) + 3] = 255
    v2 = fit_line(white_mask(img2), 100, [(10, 390)], True)
    assert abs(v2[0] - 0.25) < 0.01, v2

    # solve_dims must recover the true pitch from a synthetic camera.
    truth = Pitch(length=103.0, width=65.0)
    Htrue = np.array([[26.0, 0.4, 900.0], [0.3, 27.0, 300.0], [0.0, 0.0, 1.0]])
    lm = truth.landmarks()
    names = [n for n in lm if not n.startswith(("centre", "circle"))]
    world = np.array([lm[n] for n in names], float)
    px = cv2.perspectiveTransform(world.reshape(-1, 1, 2), Htrue).reshape(-1, 2)
    clicks = dict(zip(names, map(tuple, px)))
    L, W, rms = solve_dims(clicks, (np.arange(100.0, 106.01, 0.5),
                                    np.arange(63.0, 68.01, 0.5)))
    assert abs(L - 103.0) < 0.6 and abs(W - 65.0) < 0.6, (L, W, rms)
    print("teamtrack_calib.py ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
