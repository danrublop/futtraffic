"""N34: zero-click calibration. Upload the match; the camera is found from the paint.

WHY
    N31 showed that a spherical camera registered to the painted lines lands held-out interior paint
    within half a metre and gives the same camera to a pixel across seven matches - but it STARTED
    from an operator's 21 dots, and the dots were the largest error in the whole chain (9-19 px off on
    the far side). The dots are only there to say roughly where the camera is. A fixed panoramic rig
    does not need telling: it hangs a few metres up, a few metres behind a touchline, near halfway,
    looking across. That is a small box of possibilities, and the pitch is a known drawing.

WHAT IT DOES, from the video alone - no clicks, no ground truth, nothing from the vendor
    1. 40-frame median (players gone), grass map, paint-ridge map: n31_linereg's.
    2. PAINT ON GRASS -> a truncated distance transform at quarter resolution.
    3. COARSE SEARCH over the mount: horizontal field of view, tilt, height, distance behind the near
       touchline, offset along it, pan. Each candidate is a physical spherical camera with square
       angular pixels; its score is the mean truncated distance from the projected pitch drawing to the
       nearest paint (chamfer). Points that fall outside the frame cost the truncation: a panoramic rig
       sees the whole pitch.
    4. The best few candidates are polished on the same score (Nelder-Mead, all nine physical
       parameters), then handed to n31_linereg.register with NO dots - project-and-search at 25, 12, 8,
       5, 3 px with the square-pixel prior. The one that finds the most paint at the lowest line RMS wins.
    5. ACCEPTANCE, without ground truth: paint hits and line RMS as N31's refusal rule; camera height
       2-20 m and above the pitch; every line family of the drawing represented.

    .venv/bin/python model/validation/n34_autocal.py --fit            # all 8 clips -> model/validation/out/stc2025/n34_autocal.json
    .venv/bin/python model/validation/n34_autocal.py --compare        # against the click-started cameras: px and metres, no ground truth
    .venv/bin/python model/validation/n34_autocal.py --demo           # synthetic panorama, unknown mount
"""
import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

# The zero-click search now lives in the product: model/engine/autocal.py. This file is N34's record and its
# experiment drivers; it holds no second copy of the code.
from autocal import accept, autocal, camera, chamfer, from_install, lineness, paint_dt, paint_mask  # noqa: E402,F401

OUT = os.path.join(ROOT, "model/validation/out/stc2025/n34_autocal.json")
TRUNC = 40.0                                        # px, full resolution
GRID = {"fov_deg": (120, 135, 150, 165, 180), "tilt_deg": (6, 10, 14, 18, 22, 27, 33), "height_m": (4, 6, 8, 11, 15),
        "behind_m": (3, 8, 14, 21, 30), "along_m": (-12, -6, 0, 6, 12), "pan_deg": (-8, -4, 0, 4, 8)}
TOP_K, TOP_REG = 40, 6                                  # candidates polished; polished candidates registered
TRUNC_WIDE = 200.0
TRUNC_M, N_PAINT = 5.0, 1500                            # paint -> drawing: truncation in metres, paint pixels sampled
BOUNDS = {"fov_deg": (100, 200), "tilt_deg": (3, 45), "height_m": (2.5, 20), "behind_m": (1, 40), "along_m": (-25, 25),
          "pan_deg": (-15, 15), "roll_deg": (-4, 4), "cv_off": (-150, 150)}
PAINT_PCT, PAINT_LEN, STRAIGHT = 96.0, 60, 0.15         # lineness percentile on the pitch; stroke length px; minor/major variance
MIN_PITCH_SHARE = 0.15                                  # of the frame: less grass than this is not a pitch we can read
RADII = (25.0, 12.0, 8.0, 5.0, 3.0)


def fit_all():
    import n30_operator_maps as N  # noqa: PLC0415
    import n31_linereg as L  # noqa: PLC0415
    res = {}
    for q in N.SEQS:
        P, _names, _obj, _img = L.clicks_of(q)             # the PITCH SIZE only; no dot is used
        bgr = L._median_cached(q)
        cam, rep = autocal(bgr, P)
        res[q] = {"camera": None if cam is None else cam.tolist(), "width": int(bgr.shape[1]), **rep}
        if cam is not None:
            L.overlay(bgr, cam, P, np.zeros((0, 2)), os.path.join(ROOT, f"model/validation/out/stc2025/n34_autocal_{q}.jpg"))
        print(q, {k: rep.get(k) for k in ("accepted", "hits", "line_rms_px", "f_px_per_rad", "why")}, "centre", np.round(rep.get("camera_centre_m", [0, 0, 0]), 2), flush=True)
    # second pass: a refused match borrows the camera of its install - here, the accepted clips of its frame size
    for q in N.SEQS:
        if res[q]["camera"] is not None or "why" in res[q] and res[q]["why"]:
            continue
        mates = [np.array(v["camera"]) for k, v in res.items() if v["camera"] is not None and v["width"] == res[q]["width"] and v.get("via") != "install"]
        if len(mates) < 2:
            continue
        P, _n, _o, _i = L.clicks_of(q)
        bgr = L._median_cached(q)
        cam, rep = from_install(bgr, P, np.median(np.stack(mates), axis=0))
        res[q] = {"camera": None if cam is None else cam.tolist(), "width": int(bgr.shape[1]), "first_pass": {k: res[q].get(k) for k in ("hits", "line_rms_px")}, **rep}
        print(q, "from its install:", {k: rep[k] for k in ("accepted", "hits", "line_rms_px")}, "centre", np.round(rep["camera_centre_m"], 2), flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("->", OUT)


def compare():
    """Zero-click against click-started (n31b, perimeter rule): the same pixels read through both."""
    import n31_linereg as L  # noqa: PLC0415
    a, b = json.load(open(OUT)), json.load(open(L.OUT_B))
    gx, gy = np.meshgrid(np.linspace(3, 102, 12), np.linspace(3, 65, 8))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    for q in a:
        if a[q]["camera"] is None or b[q]["refused"]:
            print(q, "auto", "REFUSED" if a[q]["camera"] is None else "ok", "| click-started", b[q]["refused"] or "ok")
            continue
        qa, qb = np.array(a[q]["camera"]), np.array(b[q]["camera"])
        w = float(a[q]["width"])
        dpx = np.linalg.norm(L._fwd(qa, grid, w) - L._fwd(qb, grid, w), axis=1)
        dm = np.linalg.norm(L._inv(qa, L._fwd(qb, grid, w), w) - grid, axis=1)
        print(f"{q}  auto vs click-started: median {np.median(dpx):5.2f} px max {dpx.max():5.2f} px | {np.median(dm):4.2f} m median, {dm.max():4.2f} m max"
              f" | hits {a[q]['hits']} vs {b[q]['hits']}, line rms {a[q]['line_rms_px']:.2f} vs {b[q]['line_rms_px']:.2f} px")


def demo():
    """A pitch rendered through a mount the search grid does not contain; the camera must come back."""
    from types import SimpleNamespace  # noqa: PLC0415
    import n31_linereg as L  # noqa: PLC0415
    P = SimpleNamespace(length=105.0, width=68.0, penalty_width=40.32, penalty_depth=16.5,
                        goal_box_width=18.32, goal_box_depth=5.5, centre_circle_r=9.15)
    wh = (4096, 1080)
    true = camera(P, wh, fov_deg=156.0, tilt_deg=19.5, height_m=7.3, behind_m=16.0, along_m=-2.5, pan_deg=1.7, roll_deg=0.4, cv_off=-25.0)
    img = np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8)
    X, _T, _g = L.model_samples(P, 0.05)
    for u, v in L._fwd(true, X, float(wh[0])):
        if np.isfinite(u) and 2 <= u < wh[0] - 2 and 2 <= v < wh[1] - 2:
            img[int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = (235, 235, 235)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    cam, rep = autocal(img, P)
    assert cam is not None and rep["accepted"], rep
    grid = np.stack(np.meshgrid(np.linspace(5, 100, 12), np.linspace(3, 65, 9)), -1).reshape(-1, 2)
    err = np.linalg.norm(L._inv(cam, L._fwd(true, grid, float(wh[0])), float(wh[0])) - grid, axis=1)
    assert np.median(err) < 0.1 and err.max() < 0.5, (np.median(err), err.max(), rep)
    # a frame with no pitch in it is refused, not guessed at
    none, rep2 = autocal(np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8), P)
    assert none is None and not rep2["accepted"], rep2
    night, rep3 = autocal(np.full((wh[1], wh[0], 3), (60, 66, 62), np.uint8), P)          # a grey pitch has no grass to find
    assert night is None and "grass region" in rep3["why"], rep3
    print(f"n34 autocal demo OK: no dots, median map error {np.median(err):.3f} m (max {err.max():.2f}); {rep['hits']} of {rep['samples']} samples on paint")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g_ = ap.add_mutually_exclusive_group(required=True)
    for m_ in ("fit", "compare", "demo"):
        g_.add_argument(f"--{m_}", action="store_true")
    a_ = ap.parse_args()
    (fit_all if a_.fit else compare if a_.compare else demo)()
