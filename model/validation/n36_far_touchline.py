"""N36: the far touchline registered to the PAINT, not the boards - the test, written before the fix.

WHY
    The far touchline is where the zero-click map is worst: held-out far-touchline paint 1-3 m out
    (N31 §5), players standing on the line read 0.9-3.5 m INSIDE the pitch (N32). A research probe
    (research-optimal-calibration (not published) Q3) found the cause in the pixels, with no ground truth: on all
    seven same-install clips a thin ridge with GRASS ON BOTH SIDES lies 2.8-5.1 px inside the camera's
    far touchline, and the halfway line ENDS 3.8-4.5 px inside it. The registration's perimeter search
    ("the first ridge at least half the window's tallest") is won by the perimeter boards' white band;
    its simulation says a 3 px lock is -1.1 to -1.7 m at the far touchline and nothing at the centre
    spot. Noise is not the limit there (the Cramer-Rao floor is 0.04-0.09 m); association is.

THE FIX UNDER TEST  model/engine/autocal.search, perimeter samples only: walk the NATIVE min(B,G,R) profile
    from inside the pitch outward and take the first thin local ridge that has grass on BOTH sides - a
    board has no grass beyond it. No top-hat, no extra blur, no "half of tallest". A perimeter sample
    with no such ridge is NOT a hit: it is dropped rather than given to the boards.

THE TEST - none of it reads the position feed
    T  TERMINATIONS, held out by construction: a line sample constrains the camera only ACROSS the
       line, so where the halfway line ENDS is information the registration never uses. Walk the
       halfway line outward on the 40-frame median; its end is where its contrast across the line
       falls to half its median over y = 3..12 m. Offset = that row minus the camera's far-touchline
       row at x = 52.5 (px; + = the line runs on past the camera's touchline, toward the camera side).
    N  N32's five far references (model/validation/n32_valid.json: three throw-in takers and two players
       standing at the far grass edge, reviewed blind before any position was read): the signed depth
       error of the map at the MOT box's foot, as n32_depth_reference scores LINE.
    P  Held-out INTERIOR paint (n31_linereg's families, far touchline excluded), mean over families
       of the median metres, per clip: the fix must not buy the far side with the interior.

RULE, fixed before the fix was written. The new registration (zero-click, per clip) PASSES iff
    T  |offset| <= 1.0 px on at least 6 of the 7 clips          (before: 3.76-4.37 px on all seven);
    N  |median signed error| <= 1.0 m over the five references  (before: +2.15 m);
    P  on at least 6 of 7 clips the interior mean is no more than 0.05 m above the old rule's.
    Anything else FAILS and the old perimeter rule stays. ONE install, development set, n=1 configuration.

EXPLORATORY, added AFTER the first --after run failed the rule (docs/experiments/n36-far-touchline.md): `--width W` runs the
    same three measurements with the pitch W metres wide instead of 68; `--warp` frees N31's column warp (each
    lens's own small vertical correction) once the rigid camera has settled; `--kappa K` sets autocal.VERT_CUBIC,
    one global cubic in the vertical law (the review's finding). Their numbers are not the rule's, and T and N
    stop being independent of the fit once the far touchline's paint is what is fitted (docs/experiments/n36-far-touchline.md).

    .venv/bin/python model/validation/n36_far_touchline.py --before    # the rule's three numbers on the cameras as they are
    .venv/bin/python model/validation/n36_far_touchline.py --after     # re-registers every clip with the fix, then the same
    .venv/bin/python model/validation/n36_far_touchline.py --show      # what those runs wrote, and the rule on each
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
SEQS = ("117093", "118575", "118576", "118577", "118578", "128057", "128058")
STC = os.path.join(ROOT, "model/validation/out/stc2025")
INTERIOR = (("circle",), ("halfway",), ("pen_l_front", "pen_r_front"), ("six_l_front", "six_r_front"),
            ("pen_l_far", "pen_r_far", "pen_l_near", "pen_r_near"))


def termination(bgr, q, length):
    import autocal as A  # noqa: PLC0415
    mn = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), 0.7).min(2)
    y = np.arange(12, -6, -0.05)
    p = A._fwd(q, np.column_stack([np.full_like(y, length / 2.0), y]), float(mn.shape[1]))
    du = np.arange(-3, 3.01, 0.25)
    prof = map_coordinates(mn, [np.repeat(p[:, 1:2], len(du), 1).ravel(), (p[:, 0:1] + du[None]).ravel()], order=1).reshape(len(y), len(du))
    c = prof.max(1) - 0.5 * (prof[:, 0] + prof[:, -1])
    ref = np.median(c[y >= 3])
    k = np.flatnonzero((c < 0.5 * ref) & (y < 3))[0]
    v_end = np.interp(0.5 * ref, [c[k], c[k - 1]], [p[k, 1], p[k - 1, 1]])
    return float(v_end - A._fwd(q, np.array([[length / 2.0, 0.0]]), float(mn.shape[1]))[0, 1])


def far_refs(cams):
    """N32's five far references through `cams` -> signed depth errors in metres (+ = inside the pitch)."""
    import autocal as A  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    import n32_depth_reference as N32  # noqa: PLC0415
    vj = json.load(open(N32.VALID))
    far_keys = set(vj["sides_by_eye"]["far"])
    out = []
    for q in SEQS:
        feet = D.G_mot_feet(q)
        for e in N32.events(q):
            if e["key"] in far_keys and feet.get(e["f"], {}).get(e["mot"]) is not None:
                out.append(float(A._inv(cams[q], np.array([feet[e["f"]][e["mot"]]], float))[0, 1]) - N32.TRUTH["far"])
        for key in vj.get("edge", {}).get("accepted", []):
            if key.startswith(q + ":"):
                mid, f = int(key.split(":")[2]), int(key.split(":")[3])
                uv = feet.get(f, {}).get(mid)
                if uv is not None:
                    out.append(float(A._inv(cams[q], np.array([uv], float))[0, 1]) - 0.0)
    return out


def interior_held_out(bgr, q0, P, native):
    import autocal as A  # noqa: PLC0415
    rid, gr, empty = A.ridge_map(bgr), A.grass_map(bgr), np.zeros((0, 2))
    vals = []
    for fam in INTERIOR:
        _cam, rep = A.register(rid, q0, P, empty, empty, grass=gr, square=1.0, radii=A.AUTO_RADII[1:], leave_out=fam,
                               **({"native": A.native_map(bgr)} if native else {}))
        if rep.get("held_out"):
            vals.append(rep["held_out"]["median_m"])
    return float(np.mean(vals))


def main(after, width=68.0):
    import autocal as A0  # noqa: PLC0415
    if "--kappa" in sys.argv:
        A0.VERT_CUBIC = float(sys.argv[sys.argv.index("--kappa") + 1])
    import dataclasses  # noqa: PLC0415
    import autocal as A  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415
    gold = json.load(open(os.path.join(STC, "n34_autocal.json")))
    cams, T, Pm, res = {}, {}, {}, {}
    for q in SEQS:
        bgr = cv2.imread(os.path.join(STC, f"{q}.median40.png"))
        P = Pitch(105.0, float(width))
        q0 = np.array(gold[q]["camera"], float)
        q0[7:10] -= A._rot(q0[4:7])[:, 1] * (width - 68.0) / 2.0               # the same centre spot in the narrower pitch's frame
        if after:
            cam, rep, Pf = A._settle(A.ridge_map(bgr), A.grass_map(bgr), q0, P, native=A.native_map(bgr))
            if "--warp" in sys.argv:
                e_ = np.zeros((0, 2))
                cam, rep = A.register(A.ridge_map(bgr), cam, Pf, e_, e_, grass=A.grass_map(bgr), warp=True, square=1.0,
                                      radii=A.AUTO_RADII[2:], native=A.native_map(bgr))
            ok = A.accept(cam, rep)[0]
        else:
            cam, rep, Pf = A._settle(A.ridge_map(bgr), A.grass_map(bgr), q0, P)
            ok = A.accept(cam, rep)[0]
        cams[q], T[q] = cam, termination(bgr, cam, Pf.length)
        Pm[q] = interior_held_out(bgr, cam, dataclasses.replace(Pf), after)
        res[q] = {"accepted": ok, "termination_px": round(T[q], 2), "interior_held_out_m": round(Pm[q], 3), "hits": rep["hits"],
                  "touch_far_hits": rep["hits_by_group"]["touch_far"], "length_m": Pf.length, "camera": cam.tolist()}
        print(q, {k: v for k, v in res[q].items() if k != "camera"}, flush=True)
    refs = far_refs(cams)
    res["_far_refs_m"] = [round(x, 2) for x in refs]
    res["_summary"] = {"T_within_1px": int(sum(abs(v) <= 1.0 for v in T.values())), "N_median_m": round(float(np.median(refs)), 2),
                       "P_mean_m": {q: round(v, 3) for q, v in Pm.items()}}
    print(res["_far_refs_m"], res["_summary"])
    json.dump(res, open(os.path.join(STC, f"n36_{'after' if after else 'before'}{'' if width == 68.0 else f'_w{width:g}'}{'_warp' if '--warp' in sys.argv else ''}{('_k' + sys.argv[sys.argv.index('--kappa') + 1]) if '--kappa' in sys.argv else ''}.json"), "w"), indent=1)


def show():
    """The rule on every stored run. P is judged against --before's own interior numbers."""
    base = json.load(open(os.path.join(STC, "n36_before.json")))["_summary"]["P_mean_m"]
    for name in ("before", "after", "after_w67", "after_warp", "after_k-0.2", "after_k-0.4", "after_k-0.6"):
        path = os.path.join(STC, f"n36_{name}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        T = [r[q]["termination_px"] for q in SEQS]
        Pm = r["_summary"]["P_mean_m"]
        far = [r[q]["touch_far_hits"][0] for q in SEQS]
        t_ok, p_ok = sum(abs(v) <= 1.0 for v in T), sum(Pm[q] <= base[q] + 0.05 for q in SEQS)
        n_ok = abs(r["_summary"]["N_median_m"]) <= 1.0
        print(f"{name:11s} T {min(T):+.2f} to {max(T):+.2f} px, {t_ok} of 7 within 1 px | N median {r['_summary']['N_median_m']:+.2f} m of {r['_far_refs_m']} |"
              f" P {min(Pm.values()):.3f} to {max(Pm.values()):.3f} m, {p_ok} of 7 no worse | far-touchline samples on paint {min(far)} to {max(far)} of ~210 |"
              f" {'PASS' if t_ok >= 6 and n_ok and p_ok >= 6 else 'FAIL'}")


if __name__ == "__main__":
    if "--show" in sys.argv:
        raise SystemExit(show())
    main("--after" in sys.argv, float(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 68.0)
