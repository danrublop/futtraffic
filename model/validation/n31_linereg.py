"""N31 line registration: the operator's clicks START a spherical camera; the painted lines FINISH it.

WHY
    N30's review left a camera-family map at about 3 m from players' feet and asked what the rest
    was. Two GT-free findings on 118575 answered it (docs/experiments/n31-line-registration.md):
      - read carefully off a player-free median frame, the operator's far-side dots sit a median of
        9 px from the paint, up to 19 - and out there a pixel is about half a metre;
      - on the re-read landmarks an equirectangular camera fits to 3.2 px and puts its own landmarks
        0.30 m out at the median, where a fisheye fits to 7.3 px. The family is spherical
        (research-unstitching-panorama (not published)) and the clicks, not the model, carried the error.
    No operator places a dot to a pixel on a 4096-wide frame shown 1400 wide. The paint does not
    need them to: there are thousands of painted pixels and 21 dots. This is Carr, Sheikh &
    Matthews (WACV 2012) project-and-search, on a spherical camera instead of a pinhole.

WHAT IT DOES, per clip, from the clicks and the video alone - no ground truth anywhere in this file
    1. median of 40 frames across the clip: players gone, paint left.
    2. ridge map: white top-hat of min(B,G,R) - paint is bright in EVERY channel, grass is not.
    3. equirect10 fitted to the clicks, then refitted with a soft-L1 loss so a dot metres off the
       paint cannot drag the camera to it.
    4. sample the pitch's lines and centre circle every 0.5 m, project, and search along each
       sample's image normal for the ridge; keep a hit only if it is a clear local peak.
       On the four PERIMETER lines the strongest ridge is not the paint: a perimeter board's white
       band sits a few pixels beyond a faint far touchline, and N31's first score registered to it
       (docs/experiments/n31-line-registration.md §4). There the search walks from inside the pitch outward and
       takes the FIRST ridge that has grass at its inner foot - paint is where the grass ends.
    5. refit the camera to the hits - normal component only, a line says nothing along itself -
       with the clicks kept as weak robust points. Repeat at search radii 12, 8, 5, 3 px.
    REFUSES (returns the click-only camera, flagged) when fewer than MIN_HITS samples find paint or
    the final line RMS exceeds MAX_LINE_RMS_PX: a registration that did not register is not one.

    .venv/bin/python model/validation/n31_linereg.py --fit      # all 8 clips -> model/validation/out/stc2025/n31b_linereg.json + overlays
    .venv/bin/python model/validation/n31_linereg.py --v1       # without the perimeter rule -> n31_linereg.json, what N31 first scored
    .venv/bin/python model/validation/n31_linereg.py --demo     # synthetic panorama: recovers a camera from displaced clicks
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

# The registration itself now lives in the product: model/engine/autocal.py. This file is the record of how
# it was earned (N31) and still drives those experiments; it holds no second copy of the code.
from autocal import (Camera, MAX_LINE_RMS_PX, MIN_HITS, MIN_PEAK, OUTWARD, WARP_DEG, _fwd, _inv,  # noqa: E402,F401
                     grass_map, median_frame, model_samples, register, ridge_map, search)

OUT = os.path.join(ROOT, "model/validation/out/stc2025/n31_linereg.json")      # N31 as first scored: no perimeter rule (--v1)
OUT_B = os.path.join(ROOT, "model/validation/out/stc2025/n31b_linereg.json")    # with it (--fit)
STEP_M = 0.5
RADII = (12.0, 8.0, 5.0, 3.0)
MIN_PEAK = 6.0            # grey levels the ridge must stand above its search window's median
MIN_HITS = 300
MAX_LINE_RMS_PX = 2.5
CLICK_SCALE_PX = 8.0      # soft-L1 scale for the operator's dots
CLICK_WEIGHT = 0.3


OUTWARD = {"touch_far": (0.0, -1.0), "touch_near": (0.0, 1.0), "goal_l": (-1.0, 0.0), "goal_r": (1.0, 0.0)}


WARP_DEG = 5              # a(u) and b(u) are polynomials of this degree in the normalised column
WARP_RIDGE = 0.05         # px of pseudo-residual per unit coefficient: shrinks the warp toward the rigid camera
WARP_V0, WARP_VS = 540.0, 540.0


def fit_clicks(obj, img, wh, P, robust=True):
    import camera_models as CM  # noqa: PLC0415
    b = CM.fit_form("equirect10", obj, img, wh, P)
    if b is None:
        raise ValueError("no spherical camera fits these clicks")
    if not robust:
        return b[1]
    s = least_squares(lambda v: (_fwd(v, obj) - img).ravel(), b[1], loss="soft_l1", f_scale=6.0, max_nfev=4000)
    return s.x


def clicks_of(seq):
    import calibrators as CAL  # noqa: PLC0415
    import n30_operator_maps as N  # noqa: PLC0415
    from pitch import resolve_landmarks  # noqa: PLC0415
    pay = json.load(open(os.path.join(N.CLICKS, f"{seq}.calibration.json")))
    P = CAL.from_dict(pay).pitch
    names = sorted(pay["points"])
    met = resolve_landmarks(P, {n: pay["points"][n] for n in names})
    return P, names, np.array([met[n] for n in names], float), np.array([pay["points"][n] for n in names], float)


def overlay(bgr, q, P, img, path):
    X, _T, _g = model_samples(P, 0.25)
    out = bgr.copy()
    for u, v in _fwd(q, X, float(bgr.shape[1])):
        if np.isfinite(u) and 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
            out[int(v), int(u)] = (0, 0, 255)
    for u, v in img:
        cv2.circle(out, (int(u), int(v)), 6, (0, 255, 0), 1)
    cv2.imwrite(path, out)


def fit_all(perimeter_rule=True):
    import n30_operator_maps as N  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    res = {}
    for q in N.SEQS:
        P, names, obj, img = clicks_of(q)
        wh = N.CM.T._wh(q)
        bgr = median_frame(f"{ST.CLIPS}/{q}.mp4")
        q0 = fit_clicks(obj, img, wh, P)
        cam, rep = register(ridge_map(bgr), q0, P, obj, img, grass=grass_map(bgr) if perimeter_rule else None)
        res[q] = {"camera": cam.tolist(), "click_only_camera": np.asarray(q0).tolist(), **rep}
        overlay(bgr, cam, P, img, os.path.join(ROOT, f"model/validation/out/stc2025/n31{'b' if perimeter_rule else ''}_linereg_{q}.jpg"))
        print(q, {k: rep[k] for k in ("hits", "line_rms_px", "click_residual_px", "refused")}, flush=True)
    path = OUT_B if perimeter_rule else OUT
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {path}")


FAMILIES = (("circle",), ("halfway",), ("pen_l_front", "pen_r_front"), ("six_l_front", "six_r_front"),
            ("pen_l_far", "pen_r_far", "pen_l_near", "pen_r_near"), ("touch_far",))
OUT_LOGO = os.path.join(ROOT, "model/validation/out/stc2025/n31c_logo.json")
OUT_C = os.path.join(ROOT, "model/validation/out/stc2025/n31c_linereg.json")


def _median_cached(seq):
    import st2_product as ST  # noqa: PLC0415
    path = os.path.join(ROOT, f"model/validation/out/stc2025/{seq}.median40.png")
    if not os.path.exists(path):
        cv2.imwrite(path, median_frame(f"{ST.CLIPS}/{seq}.mp4"))
    return cv2.imread(path)


def logo(seqs=None):
    """Held-out paint, rigid against warped, no ground truth: fit with one family of lines hidden,
    then ask how far the map lands from that family's paint, in metres. GO for scoring the warp at
    feet iff its per-clip mean over families is below the rigid camera's on at least 5 of the 7
    same-install clips - written here before it ran."""
    import n30_operator_maps as N  # noqa: PLC0415
    res = {}
    for q in seqs or [x for x in N.SEQS if x != "117092"]:
        P, _names, obj, img = clicks_of(q)
        bgr = _median_cached(q)
        rid, gr = ridge_map(bgr), grass_map(bgr)
        q0 = fit_clicks(obj, img, N.CM.T._wh(q), P)
        res[q] = {}
        for fam in FAMILIES:
            row = {}
            for name, warp in (("rigid", False), ("warp", True)):
                _cam, rep = register(rid, q0, P, obj, img, grass=gr, warp=warp, leave_out=fam)
                row[name] = rep.get("held_out")
            res[q]["+".join(fam)] = row
        mean = {n: float(np.mean([v[n]["median_m"] for v in res[q].values() if v["rigid"] and v["warp"]])) for n in ("rigid", "warp")}
        res[q]["mean_median_m"] = mean
        print(q, {k: (round(v["rigid"]["median_m"], 2), round(v["warp"]["median_m"], 2)) for k, v in res[q].items()
                  if k != "mean_median_m" and v["rigid"] and v["warp"]}, "| mean", {n: round(x, 2) for n, x in mean.items()}, flush=True)
    wins = sum(v["mean_median_m"]["warp"] < v["mean_median_m"]["rigid"] for v in res.values())
    res["_go"] = {"warp_better_on": f"{wins} of {len(res)}", "go": bool(wins >= 5)}
    with open(OUT_LOGO, "w") as fh:
        json.dump(res, fh, indent=1)
    print(res["_go"], "->", OUT_LOGO)


def fit_warp():
    import n30_operator_maps as N  # noqa: PLC0415
    res = {}
    for q in N.SEQS:
        P, _names, obj, img = clicks_of(q)
        bgr = _median_cached(q)
        q0 = fit_clicks(obj, img, N.CM.T._wh(q), P)
        cam, rep = register(ridge_map(bgr), q0, P, obj, img, grass=grass_map(bgr), warp=True)
        res[q] = {"camera": cam.tolist(), "width": int(bgr.shape[1]), "click_only_camera": np.asarray(q0).tolist(), **rep}
        overlay(bgr, cam, P, img, os.path.join(ROOT, f"model/validation/out/stc2025/n31c_linereg_{q}.jpg"))
        print(q, {k: rep[k] for k in ("hits", "line_rms_px", "refused")}, flush=True)
    with open(OUT_C, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {OUT_C}")


def demo():
    """Render a pitch through a known spherical camera, displace the far-side 'clicks' by 10-18 px
    the way an operator does, and the registration must come back to the true camera where the
    click-only fit does not."""
    from types import SimpleNamespace  # noqa: PLC0415
    import camera_models as CM  # noqa: PLC0415
    P = SimpleNamespace(length=105.0, width=68.0, penalty_width=40.32, penalty_depth=16.5,
                        goal_box_width=18.32, goal_box_depth=5.5, centre_circle_r=9.15)
    Wd, H = 4096, 1080
    # a real install's camera, rounded: equirect10 on 118575's re-read landmarks (camera at 51, 83, 7 m)
    true = np.array([1518.3, -1591.0, 1967.6, 510.9, 4.39943, -0.01316, -0.08535, -49.162, 17.940, 83.007])
    X, _T, _g = model_samples(P, 0.05)
    img = np.full((H, Wd, 3), (40, 110, 40), np.uint8)
    for u, v in _fwd(true, X):
        if np.isfinite(u) and 1 <= u < Wd - 1 and 1 <= v < H - 1:
            img[int(round(v)), int(round(u))] = (235, 235, 235)
    # a perimeter board's white band 6 px beyond the far touchline, brighter than the paint, on a dark surround
    far_line = _fwd(true, np.column_stack([np.linspace(0, 105, 4000), np.zeros(4000)]))
    for u, v in far_line:
        if 1 <= u < Wd - 1 and 12 <= v < H - 1:
            img[int(round(v)) - 11:int(round(v)) - 1, int(round(u))] = (30, 25, 25)
            img[int(round(v)) - 7:int(round(v)) - 5, int(round(u))] = (255, 255, 255)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    lm = {"c0": (0, 0), "c1": (105, 0), "c2": (0, 68), "c3": (105, 68), "h0": (52.5, 0), "h1": (52.5, 68),
          "ce": (52.5, 34), "p0": (16.5, 13.84), "p1": (16.5, 54.16), "p2": (88.5, 13.84), "p3": (88.5, 54.16)}
    obj = np.array(list(lm.values()), float)
    clicks = _fwd(true, obj)
    assert np.isfinite(clicks).all() and (clicks[:, 1] > 0).all() and (clicks[:, 1] < H).all(), clicks
    rng = np.random.default_rng(3)
    far = obj[:, 1] < 20
    clicks = clicks + rng.normal(0, 2.0, clicks.shape)
    clicks[far, 1] -= rng.uniform(10, 18, far.sum())          # too high in the image: beyond the far touchline
    grid = np.stack(np.meshgrid(np.linspace(5, 100, 12), np.linspace(3, 65, 9)), -1).reshape(-1, 2)
    uv = _fwd(true, grid)
    err = lambda q: float(np.median(np.linalg.norm(CM.cyl_inv(q, uv, True) - grid, axis=1)))  # noqa: E731
    q0 = fit_clicks(obj, clicks, (Wd, H), P)
    cam, rep = register(ridge_map(img), q0, P, obj, clicks, grass=grass_map(img))
    assert rep["refused"] is None, rep
    assert err(cam) < 0.15 and err(cam) < 0.25 * err(q0), (err(q0), err(cam), rep)
    # without the perimeter rule the same frame registers its far touchline to the board and lands worse
    blind, _rep = register(ridge_map(img), q0, P, obj, clicks)
    assert err(blind) > 2 * err(cam), (err(blind), err(cam))
    # a frame with no paint registers nothing, and says so
    blank = np.full((H, Wd, 3), (40, 110, 40), np.uint8)
    cam2, rep2 = register(ridge_map(blank), q0, P, obj, clicks, grass=grass_map(blank))
    assert rep2["refused"] and np.allclose(cam2, q0), rep2
    print(f"n31 linereg demo OK: median map error {err(q0):.2f} m from displaced clicks -> {err(cam):.2f} m after the lines "
          f"({err(blind):.2f} m if the board is allowed to be the touchline)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g_ = ap.add_mutually_exclusive_group(required=True)
    g_.add_argument("--fit", action="store_true")
    g_.add_argument("--v1", action="store_true", help="N31 as first scored: no perimeter rule")
    g_.add_argument("--logo", action="store_true", help="held-out paint, rigid against warped; no ground truth")
    g_.add_argument("--warp", action="store_true", help="fit the warped camera on all clips -> n31c_linereg.json")
    g_.add_argument("--demo", action="store_true")
    a_ = ap.parse_args()
    (fit_all(True) if a_.fit else fit_all(False) if a_.v1 else logo() if a_.logo else fit_warp() if a_.warp else demo())
