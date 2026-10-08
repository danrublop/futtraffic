"""N38: is kappa a property of the rig? The cubic in the vertical law as a FREE least-squares parameter per clip.

WHY
    N36's review found that one global cubic in the panorama's vertical law, v = cv + fv (e + kappa e^3) with
    kappa = -0.4, passes the letter of N36's rule (T, N, P) where the rigid camera and the column warp fail
    (docs/experiments/n36-far-touchline.md section 2). It was found AFTER the rule, on the same seven clips, by a scan of nine
    values, so it is a candidate and not a result. research-camera-math-limits (not published) section 2 puts kappa's
    Fisher sd at about 0.012 per clip (1 px noise, the 118575 camera) and section 5.2 says what would make it
    credible before new data: if the cubic is a property of the RIG - one panorama law for one stitch - then seven
    clips of the same install must estimate the same kappa to within a few times that sd. If they do not, the
    parameter is absorbing something that varies clip to clip: far-touchline association (0.087 of kappa per px of
    far-touchline misread, section 2), width, or a column-dependent term. This file estimates it.

THE MODEL, and how it is fitted here without touching model/engine/autocal.py
    kappa enters `autocal._fwd` and `autocal._inv` through the module global `autocal.VERT_CUBIC`. The joint fit is
    a copy of `autocal.register`'s inner loop with an 11th parameter: the residual function sets
    `autocal.VERT_CUBIC = v[10]` and then calls `autocal._fwd(v[:10], ...)`. Single-threaded per clip, so the module
    global is the parameter (clips run in separate processes, one global each). Everything else is register's:
    line residual = normal component of (model - paint) for every hit sample; soft-L1 loss at f_scale 2 px;
    max_nfev 2000; the square-pixel prior `square * (|fu| - |fv|)` with square = 1.0 (what `_settle` uses); no
    clicks, no 3-D points, no warp, no length fit. The paint is found by `autocal.search` under the N36 paint rule
    (`native=autocal.native_map(median)`), so the far touchline is the paint, not the boards. Each pass runs the
    radii AUTO_RADII[2:] = (8, 5, 3) px, as N36's `--warp` did; three passes (search -> fit -> search again at the
    new camera), the last camera kept. Held fixed: the pitch, 105 x 68 m with each clip's own length from the start
    file (`length_m`, N36's fitted length, 105.0-105.2); the 40-frame median `model/validation/out/stc2025/<clip>.median40.png`;
    the seven Tsukuba clips (117093 118575 118576 118577 118578 128057 128058).
    kappa's sd is the covariance sd exactly as register computes `length_sd_m`: sqrt(pinv(J'J)[10,10] * sum(r^2)/dof)
    on the last pass's soft-L1 Jacobian. Two starts per clip: A = model/validation/out/stc2025/n36_after.json (paint rule,
    rigid camera, kappa starts at 0) and B = n36_after_k-0.4.json (the same clip settled under kappa -0.4, kappa
    starts at -0.4). The same loop with kappa FIXED (10 parameters) at 0 and at -0.4, from the same starts, gives
    the line RMS and hits the free fit is compared with under an identical protocol (N36's stored files carry no RMS).

WHAT IS MEASURED for the free-kappa camera of each start (functions imported from model/validation/n36_far_touchline.py):
    T   termination: where the halfway line ends minus the camera's far touchline row, px. Under any model that
        reaches the far paint T is the fit's own residual there (N36 CORRECTION), so here it reports convergence,
        not evidence - EXCEPT in the leave-out run below, where the far touchline is not fitted and T is a genuine
        held-out read again.
    N   N32's far references through the camera, signed metres (+ = inside the pitch), per clip; the median over
        all is what N36's rule takes. Truth band 1.1 m wide: it tells 2 m from 0, not 0.02 from 0.5.
    P   held-out interior paint, mean over the five families of the median metres (5 registers per clip, ten camera
        numbers refitted with kappa FIXED at the clip's estimate, one family left out each time), judged as N36
        does: no more than 0.05 m above n36_before's P for that clip.

PRE-REGISTERED, before any run
    RIG   kappa is a property of the rig iff BOTH
            (a) the sample sd of the seven start-A estimates is <= 0.03, and
            (b) |kappa_A - kappa_B| <= 0.02 on every one of the seven clips (the two starts agree).
          Anything else: the per-clip value is absorbing something that varies between clips of one rig - the
          three named above - and the seven cannot be pooled into "the rig's kappa". (Start A is primary; start B's
          sample sd is reported beside it.)
    CANDIDATE  the N36 candidate kappa = -0.4 is consistent with the free estimates iff -0.4 lies within
          [min, max] of the seven start-A estimates. If every estimate is on one side of -0.4, the scan picked
          a value the data does not; the candidate for the next test is then the mean of the seven, or nothing.
    LEAVE-OUT  the whole fit repeated with the far-touchline samples searched but never fitted
          (`leave_out=("touch_far",)` as register spells it), both starts. Three outcomes, decided now:
            REAL     |kappa_LO - kappa_A| <= max(0.05, 2 sd_LO) on at least 6 of 7 clips and sd_LO <= 0.05 on those:
                     the interior and near lines want the same cubic; kappa is a vertical-law term.
            FAR-ROW  |kappa_LO| < 0.5 |kappa_A| on at least 6 of 7 clips (it collapses toward 0 without the far
                     row): kappa is a far-touchline-row parameter - the 0.087-per-px coupling of the research
                     page - and the candidate is a far-touchline association fit wearing a law's name.
            OPEN     neither, or sd_LO > 0.05 on 2 or more clips: the other lines cannot decide it from here.
          Under leave-out T and the far touchline's own held-out normal residual (median signed px, + = the model's
          touchline beyond the paint, toward the boards) are independent of the fit and are reported as such.
    n = 1 configuration of this fit (radii, passes, prior, loss all taken from N36/_settle as they stand; none
    tuned here). Seven clips of ONE install, all seen by every N30-N37 page: whatever this selects, it confirms
    nothing. Development, not a shipped number.

    .venv/bin/python model/validation/n38_kappa_free.py            # ~6 processes, writes model/validation/out/stc2025/n38_kappa_free.json
    .venv/bin/python model/validation/n38_kappa_free.py --show     # the table and the three verdicts from that file
    .venv/bin/python model/validation/n38_kappa_free.py --demo     # a synthetic panorama with a known kappa is recovered
"""
import dataclasses
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
import autocal as A  # noqa: E402
import n36_far_touchline as N36  # noqa: E402

SEQS = N36.SEQS
STC = N36.STC
OUT = os.path.join(STC, "n38_kappa_free.json")
STARTS = {"A": "n36_after.json", "B": "n36_after_k-0.4.json"}
RADII, PASSES, SQUARE = A.AUTO_RADII[2:], 3, 1.0
SD_RIG, AGREE, LO_TOL, LO_SD = 0.03, 0.02, 0.05, 0.05


def fit(ridge, grass, native, q0, P, kappa0, free=True, leave_out=()):
    """register's inner loop with kappa as v[10] (free) or as autocal.VERT_CUBIC held (fixed). -> report."""
    X, T, grp = A.model_samples(P)
    out = np.array([A.OUTWARD.get(g, (0.0, 0.0)) for g in grp])
    use = ~np.isin(grp, list(leave_out))
    width = float(ridge.shape[1])
    v = np.r_[np.asarray(q0, float), kappa0] if free else np.array(q0, float)
    A.VERT_CUBIC = float(kappa0)
    sol = None
    for _ in range(PASSES):
        for radius in RADII:
            hit, d, nrm = A.search(ridge, v[:10], X, T, radius, grass=grass, outward=out, native=native)
            hit &= use
            if hit.sum() < A.MIN_HITS:
                return {"refused": f"{int(hit.sum())} hits at radius {radius}"}
            Xh, dh, nh = X[hit], d[hit], nrm[hit]

            def res(v, Xh=Xh, dh=dh, nh=nh):
                if free:
                    A.VERT_CUBIC = float(v[10])
                c = v[:10]
                line = ((A._fwd(c, Xh, width) - dh) * nh).sum(1)
                return np.r_[line, SQUARE * (abs(c[0]) - abs(c[1]))]
            sol = least_squares(res, v, loss="soft_l1", f_scale=2.0, max_nfev=2000)
            v = sol.x
            A.VERT_CUBIC = float(v[10]) if free else float(kappa0)
    kappa = float(v[10]) if free else float(kappa0)
    sd = None
    if free:
        dof = max(len(sol.fun) - len(sol.x), 1)
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * float((sol.fun ** 2).sum() / dof)
        sd = float(np.sqrt(max(cov[10, 10], 0.0)))
    hit, d, nrm = A.search(ridge, v[:10], X, T, RADII[-1], grass=grass, outward=out, native=native)
    fitted = hit & use
    r = ((A._fwd(v[:10], X[fitted], width) - d[fitted]) * nrm[fitted]).sum(1)
    rep = {"kappa": kappa, "kappa_sd": sd, "hits": int(fitted.sum()), "samples": int(use.sum()),
           "line_rms_px": float(np.sqrt((r ** 2).mean())), "camera": v[:10].tolist(),
           "touch_far_hits": [int((fitted & (grp == "touch_far")).sum()), int((grp == "touch_far").sum())]}
    if leave_out:
        hit8, d8, n8 = A.search(ridge, v[:10], X, T, RADII[0], grass=grass, outward=out, native=native)
        held = hit8 & ~use
        rh = ((A._fwd(v[:10], X[held], width) - d8[held]) * n8[held]).sum(1)
        rep["held_far_px"] = {"hits": int(held.sum()), "median_signed": float(np.median(rh)) if held.any() else None}
    return rep


def far_refs_one(q, cam):
    """N36.far_refs for ONE clip: its loop is over the module's SEQS, so that is narrowed for the call."""
    keep = N36.SEQS
    N36.SEQS = (q,)
    try:
        return N36.far_refs({q: cam})
    finally:
        N36.SEQS = keep


def clip(q):
    from pitch import Pitch  # noqa: PLC0415
    bgr = cv2.imread(os.path.join(STC, f"{q}.median40.png"))
    ridge, grass, native = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr)
    res = {}
    for s, fn in STARTS.items():
        st = json.load(open(os.path.join(STC, fn)))[q]
        q0, P = np.array(st["camera"], float), dataclasses.replace(Pitch(105.0, 68.0), length=float(st["length_m"]))
        k0 = -0.4 if s == "B" else 0.0
        r = {"start_file": fn, "length_m": P.length,
             "free": fit(ridge, grass, native, q0, P, k0),
             "fixed_0": fit(ridge, grass, native, q0, P, 0.0, free=False),
             "fixed_-0.4": fit(ridge, grass, native, q0, P, -0.4, free=False),
             "leave_out_far": fit(ridge, grass, native, q0, P, k0, leave_out=("touch_far",))}
        for key in ("free", "leave_out_far"):
            f = r[key]
            if "refused" in f:
                continue
            A.VERT_CUBIC = f["kappa"]
            cam = np.array(f["camera"])
            f["T_px"] = N36.termination(bgr, cam, P.length)
        f = r["free"]
        if "refused" not in f:
            A.VERT_CUBIC = f["kappa"]
            cam = np.array(f["camera"])
            f["N_refs_m"] = far_refs_one(q, cam)
            f["P_m"] = N36.interior_held_out(bgr, cam, dataclasses.replace(P), True)
        A.VERT_CUBIC = 0.0
        res[s] = r
        print(q, s, {k: (round(v["kappa"], 3), v.get("kappa_sd") and round(v["kappa_sd"], 3), v["hits"], round(v["line_rms_px"], 3))
                     for k, v in r.items() if isinstance(v, dict) and "kappa" in v}, flush=True)
    return q, res


def main():
    with Pool(min(6, len(SEQS))) as pool:
        res = dict(pool.map(clip, SEQS))
    json.dump(res, open(OUT, "w"), indent=1)
    show()


def show():
    r = json.load(open(OUT))
    base = json.load(open(os.path.join(STC, "n36_before.json")))["_summary"]["P_mean_m"]
    print("clip    | start | kappa  +- sd    | rms px hits  (k=0: rms hits) (k=-0.4: rms hits) | T px  | P m   (before+0.05 ok) | N refs m")
    kA, kB, kLO, sdLO, N_all = {}, {}, {}, {}, []
    for q in SEQS:
        for s in ("A", "B"):
            f, z, m = r[q][s]["free"], r[q][s]["fixed_0"], r[q][s]["fixed_-0.4"]
            (kA if s == "A" else kB)[q] = f["kappa"]
            N_all += f["N_refs_m"]
            print(f"{q}  |  {s}    | {f['kappa']:+.3f} +- {f['kappa_sd']:.3f} | {f['line_rms_px']:.3f} {f['hits']:4d}  ({z['line_rms_px']:.3f} {z['hits']:4d})"
                  f"  ({m['line_rms_px']:.3f} {m['hits']:4d}) | {f['T_px']:+.2f} | {f['P_m']:.3f} ({'yes' if f['P_m'] <= base[q] + 0.05 else 'NO'})"
                  f" | {[round(x, 2) for x in f['N_refs_m']]}")
    print("\nleave-out (far touchline searched, never fitted):")
    print("clip    | start | kappa_LO +- sd  | rms px hits | T px (independent) | far paint median signed px (hits)")
    for q in SEQS:
        for s in ("A", "B"):
            f = r[q][s]["leave_out_far"]
            if s == "A":
                kLO[q], sdLO[q] = f["kappa"], f["kappa_sd"]
            print(f"{q}  |  {s}    | {f['kappa']:+.3f} +- {f['kappa_sd']:.3f} | {f['line_rms_px']:.3f} {f['hits']:4d} | {f['T_px']:+.2f}"
                  f" | {f['held_far_px']['median_signed']:+.2f} ({f['held_far_px']['hits']})")
    a, b = np.array([kA[q] for q in SEQS]), np.array([kB[q] for q in SEQS])
    sd_a, sd_b, diff = float(a.std(ddof=1)), float(b.std(ddof=1)), np.abs(a - b)
    print(f"\nstart A: mean {a.mean():+.3f}, sample sd {sd_a:.3f}, range {a.min():+.3f} to {a.max():+.3f}; start B: mean {b.mean():+.3f}, sample sd {sd_b:.3f}")
    print(f"|kappa_A - kappa_B|: max {diff.max():.3f}, {int((diff <= AGREE).sum())} of 7 within {AGREE}")
    rig = sd_a <= SD_RIG and bool((diff <= AGREE).all())
    print(f"RIG: {'PASS' if rig else 'FAIL'} (sd {sd_a:.3f} <= {SD_RIG}: {'yes' if sd_a <= SD_RIG else 'no'}; all 7 agree to {AGREE}: {'yes' if (diff <= AGREE).all() else 'no'})")
    print(f"CANDIDATE -0.4 within [min, max] of start A: {'yes' if a.min() <= -0.4 <= a.max() else 'NO'}")
    lo = np.array([kLO[q] for q in SEQS]); slo = np.array([sdLO[q] for q in SEQS])
    real = ((np.abs(lo - a) <= np.maximum(LO_TOL, 2 * slo)) & (slo <= LO_SD)).sum()
    farrow = (np.abs(lo) < 0.5 * np.abs(a)).sum()
    verdict = "REAL" if real >= 6 else "FAR-ROW" if farrow >= 6 else "OPEN"
    if (slo > LO_SD).sum() >= 2 and verdict != "FAR-ROW":
        verdict = "OPEN"
    print(f"LEAVE-OUT: {verdict} (REAL criterion met on {int(real)} of 7, FAR-ROW on {int(farrow)} of 7, sd_LO > {LO_SD} on {int((slo > LO_SD).sum())};"
          f" kappa_LO mean {lo.mean():+.3f}, sd_LO median {np.median(slo):.3f})")
    print(f"N median over all far refs (both starts' free cameras, {len(N_all)} reads): {np.median(N_all):+.2f} m")
    T_ok = sum(abs(r[q]['A']['free']['T_px']) <= 1.0 for q in SEQS)
    P_ok = sum(r[q]['A']['free']['P_m'] <= base[q] + 0.05 for q in SEQS)
    print(f"N36 rule on start A's free cameras: T within 1 px {T_ok} of 7 (convergence, not evidence), P no worse {P_ok} of 7")


def demo():
    """A synthetic panorama drawn with kappa = -0.25 through the demo camera; the free fit must recover it.
    ponytail: the truth is drawn as white 3 px lines on flat green, no boards, so the perimeter rule has grass on
    both sides and nothing else; this checks the parameter plumbing, not the paint rule."""
    from pitch import Pitch  # noqa: PLC0415
    P, wh = Pitch(105.0, 68.0), (4096, 1080)
    true = A.camera(P, wh, fov_deg=156.0, tilt_deg=19.5, height_m=7.3, behind_m=16.0, along_m=-2.5, pan_deg=1.7, roll_deg=0.4, cv_off=-25.0)
    A.VERT_CUBIC = -0.25
    img = np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8)
    for u, v in A._fwd(true, A.model_samples(P, 0.05)[0], float(wh[0])):
        if np.isfinite(u) and 2 <= u < wh[0] - 2 and 2 <= v < wh[1] - 2:
            img[int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = (235, 235, 235)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    A.VERT_CUBIC = 0.0
    rep = fit(A.ridge_map(img), A.grass_map(img), A.native_map(img), true + np.r_[0, 0, 3.0, -2.0, 0.002, -0.001, 0.001, 0, 0, 0], P, 0.0)
    assert "refused" not in rep and abs(rep["kappa"] + 0.25) < 0.03, rep
    assert rep["kappa_sd"] < 0.03, rep["kappa_sd"]
    fixed = fit(A.ridge_map(img), A.grass_map(img), A.native_map(img), true, P, 0.0, free=False)
    assert fixed["line_rms_px"] > rep["line_rms_px"], (fixed["line_rms_px"], rep["line_rms_px"])
    A.VERT_CUBIC = 0.0
    print(f"n38 ok: kappa -0.25 drawn, {rep['kappa']:+.3f} +- {rep['kappa_sd']:.3f} recovered; rms {rep['line_rms_px']:.3f} px free vs {fixed['line_rms_px']:.3f} rigid")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--show" in sys.argv:
        show()
    else:
        main()
