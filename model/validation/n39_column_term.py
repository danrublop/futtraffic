"""N39: a two-parameter COLUMN term - the shape an imperfect two-lens stitch predicts - against the far-touchline hump.

WHY  (research-camera-math-limits (not published) section 1 and 5.3; docs/experiments/n36-far-touchline.md section 3)
    With the far touchline's paint found (N36), a rigid spherical camera still leaves that paint 2.3-2.8 px
    from where the halfway line ends, and the one candidate that closes it is a global cubic in the vertical
    law, kappa = -0.4 - ten times any lens residual. The derivation says what an imperfect stitch of a lens
    with a residual cubic r = f(theta + k theta^3) does to the panorama: about each LENS AXIS (a_L, e_L)

        e_true = e - k [ (a - a_L)^2 + (e - e_L)^2 ] (e - e_L)

    At the lens-axis column that is the cubic (VERT_CUBIC). Away from it the term is -k (a - a_L)^2 (e - e_L):
    a per-column VERTICAL SCALE with a FIXED shape - N31's warp b(u)(v - v0) with b(u) proportional to
    (u - u_L)^2, zero under each lens axis, largest at the seam between them. The far-touchline paint offset
    from the board-locked n34 camera IS humped: 1.3-2.6 px at the ends of the pitch, 3.3-4.4 px in the middle
    three sevenths on five of seven clips (far_tl_profile.py, re-implemented here as `measured_profile`). A global
    cubic gives a middle/ends ratio of about 1.5 along that line; the two-lens term with axes at +-39 deg about 6.

THE MODEL, written before any fit was run
    On top of the ten-number spherical camera q[0:10] (fu fv cu cv | Rodrigues | translation), the row moves by

        v' = v + beta * s(u) * (v - WARP_V0) / WARP_VS,      s(u) = ((|u - u_c| - d) / 2048)^2

    Two lenses whose axes sit at columns u_c - d and u_c + d, each pixel drawn from the nearer lens, so the
    term is the same parabola about either axis; it is zero under the axes, continuous with a kink at the seam
    u_c (both lenses give beta (d/2048)^2 there), and grows again outside the axes. FREE: beta (px; q[10]) and d
    (q[11] carries d/1000 so that register's WARP_RIDGE pseudo-residual 0.05*q[10:] stays below 0.06 px of pull).
    FIXED: u_c = 2048, the panorama's centre column - the seam of a symmetric two-lens stitch is where the
    stitcher puts it, and the fitted principal column cu is NOT a rig property here (it trades against pan: the
    n36 cameras carry cu from 1976 to 2469 on one rig). v0 = WARP_V0 = 540, vs = WARP_VS = 540 as in autocal;
    the lens-axis row v_L would be the physical choice and cv sits at 510-515 on all seven, so the 25-30 px
    difference is an a(u)-type offset of under half a pixel that the rigid parameters absorb.
    Expected from the derivation: beta = -k (2048/f)^2 WARP_VS = about -990 k px, so a lens residual k of 0.01
    to 0.035 is beta of -10 to -35 px, NEGATIVE (v - v0 < 0 at the far touchline, and the paint is INSIDE, i.e.
    below, the camera's line); d = 1030 px is 39 deg at 1513 px per rad (155 deg / 4). Starts: beta -10, d 1030;
    kappa (arm 4) 0.
    The inverse is closed-form in the column term (s depends on u only) and Newton in kappa, exactly as autocal
    does it; `selfcheck` verifies the round trip to 1e-6 with and without kappa, that the patched functions equal
    autocal's own on a ten-number camera to 1e-9, and that beta and d are recovered from noise-free synthetic
    lines. autocal is monkeypatched (`A._fwd = fwd; A._inv = inv`) so `register`, `search`, `termination`,
    `far_refs` and `interior_held_out` all see the term; model/engine/autocal.py is not edited.

THE ARMS, each from model/validation/out/stc2025/n36_after.json (paint rule, rigid, per-clip length_m), the paint rule
(`native=`), radii AUTO_RADII[2:] = (8, 5, 3), three passes of `register`, square prior 1.0, pitch 105 x 68 with
each clip's n36 length:
    1 rigid            q[0:10]                      VERT_CUBIC 0      (the same-protocol control)
    2 kappa fixed      q[0:10]                      VERT_CUBIC -0.4   (N36's candidate under THIS protocol)
    3 column           q[0:10] + beta, d            VERT_CUBIC 0
    4 column + kappa   q[0:10] + beta, d, kappa     kappa free in q[12]

PRE-REGISTERED JUDGEMENT
    (i) THE HUMP - the weight-bearing test. Measured profile: far_tl_profile.py's method, re-implemented -
        the first thin ridge with grass on both sides walking outward from inside along each column of the n34
        camera's far touchline (blur 0.7, flanks 2 px, contrast >= 6, grass > 5), median per seventh of x in
        [4, 101] m. Predicted profile: the FITTED camera's far-touchline row minus the n34 camera's, at the same
        columns, same bins. Ratio = mean(bins 3,4,5 of 7) / mean(bins 1 and 7). SUCCESS: the predicted ratio
        within 30% of the measured one (|pred/meas - 1| <= 0.30) on at least 5 of 7 clips, for arm 3 or 4.
        Arms 1 and 2 are scored the same way for comparison: the derivation says a rigid camera gives ~1 and
        kappa alone ~1.5.
    (ii) T, N, P by n36_far_touchline's own functions and rule: T within 1 px on 6 of 7; |median N| <= 1 m; P no
        more than 0.05 m worse than n36_before.json on 6 of 7. SAID UP FRONT: T is a fit residual for ANY model
        that reaches the paint (N36 review: T minus the touch_far residual is 0.1-0.25 px), so T passing says the
        optimiser converged; N is three throw-ins from two clips against a 1.1 m truth band. (i) and P carry the
        weight; P is the only held-out number.
    (iii) AGREEMENT ACROSS THE SEVEN CLIPS: one rig, so beta and d must be one number. Tolerance, from the fit
        covariance: the across-clip sd of beta, and of d, each no more than 2.5 x the median per-clip
        linearised sd (J^T J from the final hits' normal residuals at radius 3, sigma^2 = sum r^2 / dof) - the
        same factor the research page asked of kappa (sd 0.012, agree to 0.03). Also: 0 < d < 2048 (the axes
        inside the frame) and beta < 0 (the sign the paint's side dictates). A model whose parameters differ
        by more than that from clip to clip of ONE install is absorbing association or width, not the rig.
    n = 4 arms here, 2 of them new, on top of the >= 30 configurations N36 counts on these same seven clips.
    Seven clips of one install, all previously seen: this SELECTS a candidate for a pre-registered test on data
    it has not seen; it confirms nothing.

    .venv/bin/python model/validation/n39_column_term.py             # self-check (synthetic; no clip is read)
    .venv/bin/python model/validation/n39_column_term.py --run       # the four arms -> model/validation/out/stc2025/n39_column.json
    .venv/bin/python model/validation/n39_column_term.py --show      # every number n39-column-term (not published) quotes
"""
import dataclasses
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
import autocal as A  # noqa: E402
import n36_far_touchline as N36  # noqa: E402

SEQS, STC = N36.SEQS, N36.STC
OUT = os.path.join(STC, "n39_column.json")
U_C, D0, BETA0 = 2048.0, 1030.0, -10.0
ARMS = {"rigid": (False, 0.0, False), "kappa-0.4": (False, -0.4, False), "column": (True, 0.0, False), "column+kappa": (True, 0.0, True)}
BINS = np.linspace(4, 101, 8)
_fwd0, _inv0 = A._fwd, A._inv


def shape(u, d_k):
    return ((np.abs(np.asarray(u, float) - U_C) - 1000.0 * d_k) / 2048.0) ** 2


def fwd(q, xy, width=4096.0):
    q, xy = np.asarray(q, float), np.asarray(xy, float)
    if xy.shape[1] == 2:
        xy = np.column_stack([xy, np.zeros(len(xy))])
    X = xy @ A._rot(q[4:7]).T + q[7:10]
    e = np.arctan2(X[:, 1], np.hypot(X[:, 0], X[:, 2]))
    kappa = q[12] if len(q) > 12 else A.VERT_CUBIC
    u, v = q[2] + q[0] * np.arctan2(X[:, 0], X[:, 2]), q[3] + q[1] * (e + kappa * e ** 3)
    if len(q) > 10:
        v = v + q[10] * shape(u, q[11]) * (v - A.WARP_V0) / A.WARP_VS
    return np.column_stack([u, v])


def inv(q, uv, width=4096.0):
    q, uv = np.asarray(q, float), np.asarray(uv, float).reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]
    if len(q) > 10:
        b = q[10] * shape(u, q[11]) / A.WARP_VS
        v = (v + b * A.WARP_V0) / (1.0 + b)
    kappa = q[12] if len(q) > 12 else A.VERT_CUBIC
    R = A._rot(q[4:7])
    a, e = (u - q[2]) / q[0], (v - q[3]) / q[1]
    if kappa:
        y, e = e, e.copy()
        for _ in range(6):
            e -= (e + kappa * e ** 3 - y) / (1.0 + 3.0 * kappa * e ** 2)
    d = np.column_stack([np.sin(a), np.tan(e), np.cos(a)]) @ R
    C = -R.T @ q[7:10]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -C[2] / d[:, 2]
        P = C[None] + lam[:, None] * d
    P[~(lam > 0)] = np.nan
    return P[:, :2]


A._fwd, A._inv = fwd, inv      # single-threaded; every caller in autocal and n36_far_touchline now sees the term


def start(q10, column, kappa_free):
    q = list(np.asarray(q10, float)[:10])
    if column:
        q += [BETA0, D0 / 1000.0]
    if kappa_free:
        q += [0.0]
    return np.array(q)


def covariance(rid, gr, nat, q, P):
    """Linearised sd of every parameter from the final hits at radius 3 (association bias is NOT in it)."""
    X, T, grp = A.model_samples(P)
    out = np.array([A.OUTWARD.get(g, (0.0, 0.0)) for g in grp])
    hit, d, nrm = A.search(rid, q, X, T, A.AUTO_RADII[-1], grass=gr, outward=out, native=nat)
    w = float(rid.shape[1])

    def r(v):
        return np.r_[((fwd(v, X[hit], w) - d[hit]) * nrm[hit]).sum(1), abs(v[0]) - abs(v[1])]
    r0 = r(q)
    J = np.empty((len(r0), len(q)))
    for i in range(len(q)):
        h = 1e-6 * max(abs(q[i]), 1e-2)
        qp, qm = q.copy(), q.copy()
        qp[i] += h
        qm[i] -= h
        J[:, i] = (r(qp) - r(qm)) / (2 * h)
    cov = np.linalg.pinv(J.T @ J) * float((r0 ** 2).sum() / max(len(r0) - len(q), 1))
    return np.sqrt(np.maximum(np.diag(cov), 0.0)), int(hit.sum())


def measured_profile(bgr, q34):
    """far_tl_profile.py's measurement: paint offset from the n34 camera's far touchline, px, + = inside, by seventh.
    -> (7 medians, share of columns found, columns u, x of each column, offset per column)."""
    img = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), 0.7)
    mn, grass = img.min(2), img[:, :, 1] - np.maximum(img[:, :, 0], img[:, :, 2])
    xs = np.arange(4, 101.01, 0.05)
    k, A.VERT_CUBIC = A.VERT_CUBIC, 0.0        # the n34 REFERENCE is always the equirect law; found on the first run: arm 2's
    p = _fwd0(q34, np.column_stack([xs, np.zeros_like(xs)]), float(img.shape[1]))   # reference had moved with kappa
    A.VERT_CUBIC = k
    _, first = np.unique(np.round(p[:, 0]).astype(int), return_index=True)
    p, xs = p[first], xs[first]
    S = np.arange(9.0, -4.01, -0.25)
    vv, uu = p[:, 1:2] + S[None], np.repeat(p[:, 0:1], len(S), 1)
    Pm = map_coordinates(mn, [vv.ravel(), uu.ravel()], order=1).reshape(vv.shape)
    G = map_coordinates(grass, [vv.ravel(), uu.ravel()], order=1).reshape(vv.shape)
    off = np.full(len(xs), np.nan)
    k2 = 8
    for i in range(len(xs)):
        for j in range(k2, len(S) - k2):
            side = 0.5 * (Pm[i, j - k2] + Pm[i, j + k2])
            if Pm[i, j] >= Pm[i, j - 1] and Pm[i, j] > Pm[i, j + 1] and Pm[i, j] - side >= 6.0 and G[i, j - k2] > 5 and G[i, j + k2] > 5:
                a, b, c = Pm[i, j - 1], Pm[i, j], Pm[i, j + 1]
                off[i] = S[j] - 0.25 * 0.5 * (a - c) / (a - 2 * b + c - 1e-9)
                break
    ok = np.isfinite(off)
    return binned(xs, off), float(ok.mean()), p[:, 0], p[:, 1], xs, off


def binned(xs, val):
    ok = np.isfinite(val)
    return [float(np.median(val[m])) if (m := ok & (xs >= BINS[i]) & (xs < BINS[i + 1])).sum() > 10 else float("nan") for i in range(7)]


def predicted_profile(q, P, u34, v34, xs, width):
    """The fitted camera's far-touchline row minus the n34 camera's, at the n34 columns, same bins."""
    xf = np.arange(-3.0, P.length + 3.0, 0.02)
    pf = fwd(q, np.column_stack([xf, np.zeros_like(xf)]), width)
    o = np.argsort(pf[:, 0])
    return binned(xs, np.interp(u34, pf[o, 0], pf[o, 1]) - v34)


def ratio(prof):
    p = np.asarray(prof, float)
    return float(np.nanmean(p[2:5]) / np.nanmean(p[[0, 6]]))


def run():
    from pitch import Pitch  # noqa: PLC0415
    after = json.load(open(os.path.join(STC, "n36_after.json")))
    n34 = json.load(open(os.path.join(STC, "n34_autocal.json")))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    imgs = {q: cv2.imread(os.path.join(STC, f"{q}.median40.png")) for q in SEQS}
    if "measured" not in res:
        res["measured"] = {}
        for q in SEQS:
            prof, found, *_ = measured_profile(imgs[q], np.array(n34[q]["camera"], float))
            res["measured"][q] = {"bins_px": [round(v, 2) for v in prof], "found": round(found, 3), "ratio": round(ratio(prof), 3)}
            print("measured", q, res["measured"][q], flush=True)
    for arm, (column, kfix, kfree) in ARMS.items():
        if arm in res:
            continue
        A.VERT_CUBIC = kfix
        cams, R = {}, {}
        for q in SEQS:
            bgr = imgs[q]
            P = dataclasses.replace(Pitch(105.0, 68.0), length=float(after[q]["length_m"]))
            rid, gr, nat, empty = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr), np.zeros((0, 2))
            cam = start(after[q]["camera"], column, kfree)
            for _ in range(3):
                cam, rep = A.register(rid, cam, P, empty, empty, grass=gr, square=1.0, radii=A.AUTO_RADII[2:], native=nat)
            sd, nh = covariance(rid, gr, nat, cam, P)
            _m, _f, u34, v34, xs, _o = measured_profile(bgr, np.array(n34[q]["camera"], float))
            pred = predicted_profile(cam, P, u34, v34, xs, float(bgr.shape[1]))
            cams[q] = cam
            R[q] = {"camera": cam.tolist(), "sd": sd.tolist(), "refused": rep["refused"], "hits": rep["hits"], "line_rms_px": round(rep["line_rms_px"], 3),
                    "touch_far_hits": rep["hits_by_group"]["touch_far"], "length_m": P.length,
                    "termination_px": round(N36.termination(bgr, cam, P.length), 2),
                    "interior_held_out_m": round(N36.interior_held_out(bgr, cam, P, True), 3),
                    "pred_bins_px": [round(v, 2) for v in pred], "pred_ratio": round(ratio(pred), 3)}
            if column:
                R[q]["beta_px"], R[q]["beta_sd"], R[q]["d_px"], R[q]["d_sd"] = cam[10], sd[10], 1000 * cam[11], 1000 * sd[11]
            if kfree:
                R[q]["kappa"], R[q]["kappa_sd"] = cam[12], sd[12]
            print(arm, q, {k: v for k, v in R[q].items() if k not in ("camera", "sd")}, flush=True)
        refs = N36.far_refs(cams)
        R["_far_refs_m"] = [round(x, 2) for x in refs]
        R["_N_median_m"] = round(float(np.median(refs)), 2)
        A.VERT_CUBIC = 0.0
        res[arm] = R
        json.dump(res, open(OUT, "w"), indent=1)
        print(arm, "N", R["_N_median_m"], refs, flush=True)


def show():
    r = json.load(open(OUT))
    base = json.load(open(os.path.join(STC, "n36_before.json")))["_summary"]["P_mean_m"]
    print("== (iii) parameters per clip (linearised sd), u_c = 2048 fixed")
    for arm in ("column", "column+kappa"):
        if arm not in r:
            continue
        rows = [r[arm][q] for q in SEQS]
        for q, x in zip(SEQS, rows):
            print(f"{arm:13s} {q} beta {x['beta_px']:+7.2f} +- {x['beta_sd']:5.2f} px  d {x['d_px']:7.1f} +- {x['d_sd']:6.1f} px"
                  + (f"  kappa {x['kappa']:+.3f} +- {x['kappa_sd']:.3f}" if "kappa" in x else "") + f"  rms {x['line_rms_px']:.3f} hits {x['hits']}")
        for name, key, skey in (("beta", "beta_px", "beta_sd"), ("d", "d_px", "d_sd")) + ((("kappa", "kappa", "kappa_sd"),) if "kappa" in rows[0] else ()):
            v, s = np.array([x[key] for x in rows]), np.median([x[skey] for x in rows])
            print(f"{arm:13s} {name}: across-clip sd {v.std(ddof=1):.3f} vs 2.5 x median fit sd {2.5 * s:.3f} -> {'AGREE' if v.std(ddof=1) <= 2.5 * s else 'DISAGREE'};"
                  f" mean {v.mean():+.3f}, range {v.min():+.3f} to {v.max():+.3f}")
        b, d = np.array([x["beta_px"] for x in rows]), np.array([x["d_px"] for x in rows])
        print(f"{arm:13s} sign/frame: beta < 0 on {(b < 0).sum()} of 7, 0 < d < 2048 on {((d > 0) & (d < 2048)).sum()} of 7")
    print("== (i) the hump: far-touchline offset from the n34 camera, px inside, by seventh of x in [4, 101] m; ratio = middle three / two ends")
    for q in SEQS:
        m = r["measured"][q]
        print(f"measured      {q} " + " ".join(f"{v:5.2f}" for v in m["bins_px"]) + f"  ratio {m['ratio']:.2f}  found {m['found']:.2f}")
        for arm in ARMS:
            if arm in r:
                x = r[arm][q]
                print(f"{arm:13s} {q} " + " ".join(f"{v:5.2f}" for v in x["pred_bins_px"]) + f"  ratio {x['pred_ratio']:.2f}  |pred/meas-1| {abs(x['pred_ratio'] / m['ratio'] - 1):.2f}")
    for arm in ARMS:
        if arm in r:
            ok = sum(abs(r[arm][q]["pred_ratio"] / r["measured"][q]["ratio"] - 1) <= 0.30 for q in SEQS)
            print(f"{arm:13s} hump ratio within 30% on {ok} of 7 -> {'REPRODUCES' if ok >= 5 else 'DOES NOT REPRODUCE'}")
    print("== (ii) T, N, P under n36_far_touchline's rule (P against n36_before.json)")
    for arm in ARMS:
        if arm not in r:
            continue
        T = [r[arm][q]["termination_px"] for q in SEQS]
        Pm = {q: r[arm][q]["interior_held_out_m"] for q in SEQS}
        far = [r[arm][q]["touch_far_hits"][0] for q in SEQS]
        t_ok, p_ok = sum(abs(v) <= 1.0 for v in T), sum(Pm[q] <= base[q] + 0.05 for q in SEQS)
        N = r[arm]["_N_median_m"]
        print(f"{arm:13s} T {min(T):+.2f} to {max(T):+.2f} px, {t_ok} of 7 within 1 px | N median {N:+.2f} m of {r[arm]['_far_refs_m']} |"
              f" P {min(Pm.values()):.3f} to {max(Pm.values()):.3f} m, {p_ok} of 7 no worse | far-touchline on paint {min(far)} to {max(far)} |"
              f" refused {sum(r[arm][q]['refused'] is not None for q in SEQS)} | {'PASS' if t_ok >= 6 and abs(N) <= 1.0 and p_ok >= 6 else 'FAIL'}")
    print("== per clip T / P / rms")
    for q in SEQS:
        print(q, " | ".join(f"{arm} T {r[arm][q]['termination_px']:+.2f} P {r[arm][q]['interior_held_out_m']:.3f} rms {r[arm][q]['line_rms_px']:.3f}" for arm in ARMS if arm in r))
    print("== POST HOC (not pre-registered; read after the tables above): shape, and what the term does to the NEAR touchline")
    for arm in ARMS:
        if arm in r:
            rms = [float(np.sqrt(np.nanmean((np.array(r[arm][q]["pred_bins_px"]) - np.array(r["measured"][q]["bins_px"])) ** 2))) for q in SEQS]
            print(f"{arm:13s} RMS of predicted minus measured over the 7 bins, px: " + " ".join(f"{v:.2f}" for v in rms) + f"  mean {np.mean(rms):.2f}")
    for arm in ("column", "column+kappa"):
        if arm not in r:
            continue
        for q in SEQS:
            c = np.array(r[arm][q]["camera"])
            L = r[arm][q]["length_m"]
            far, near = fwd(c, np.array([[L / 2, 0.0]]))[0], fwd(c, np.array([[L / 2, 68.0]]))[0]
            term = lambda p: c[10] * float(shape(p[0], c[11])) * (p[1] - A.WARP_V0) / A.WARP_VS  # noqa: E731
            print(f"{arm:13s} {q} the term's own shift at the halfway column: far touchline {term(far):+.2f} px (row {far[1]:.0f}), near touchline {term(near):+.2f} px (row {near[1]:.0f})")


def selfcheck():
    from pitch import Pitch  # noqa: PLC0415
    P = Pitch(105.0, 68.0)
    true = A.camera(P, (4096, 1080), fov_deg=156.0, tilt_deg=19.5, height_m=7.3, behind_m=16.0, along_m=-2.5, pan_deg=1.7, roll_deg=0.4, cv_off=-25.0)
    grid = np.stack(np.meshgrid(np.linspace(2, 103, 14), np.linspace(1, 67, 9)), -1).reshape(-1, 2)
    for q in (true, np.r_[true, -30.0, 1.03], np.r_[true, -30.0, 1.03, -0.4], np.r_[true, 25.0, 0.7, 0.3]):
        assert np.abs(inv(q, fwd(q, grid), 4096.0) - grid).max() < 1e-6, len(q)
    assert np.abs(fwd(true, grid) - _fwd0(true, grid)).max() < 1e-9 and np.abs(inv(true, fwd(true, grid)) - _inv0(true, fwd(true, grid))).max() < 1e-9
    A.VERT_CUBIC = -0.4
    assert np.abs(fwd(true, grid) - _fwd0(true, grid)).max() < 1e-9 and np.abs(inv(true, fwd(true, grid)) - _inv0(true, fwd(true, grid))).max() < 1e-9
    A.VERT_CUBIC = 0.0
    # the far touchline moves INSIDE (down) most at the seam for beta < 0, and not at all under the axes
    far = fwd(np.r_[true, -30.0, 1.03], np.column_stack([np.linspace(0, 105, 11), np.zeros(11)]))
    off = far[:, 1] - fwd(true, np.column_stack([np.linspace(0, 105, 11), np.zeros(11)]))[:, 1]
    assert off.min() >= 0 and off.argmax() == np.abs(far[:, 0] - U_C).argmin(), off
    # identifiability: beta, d (and kappa) recovered from noise-free line samples, starting where --run starts
    X, T, _g = A.model_samples(P)
    for qt in (np.r_[true, -30.0, 1.03], np.r_[true, -30.0, 1.03, -0.4]):
        d = fwd(qt, X)
        t = fwd(qt, X + 0.25 * T) - fwd(qt, X - 0.25 * T)
        nrm = np.column_stack([-t[:, 1], t[:, 0]]) / np.linalg.norm(t, axis=1, keepdims=True)
        q0 = start(true + np.r_[0, 0, 4.0, -3.0, 0.002, -0.001, 0.001, 0, 0, 0], True, len(qt) > 12)
        sol = least_squares(lambda v: np.r_[((fwd(v, X) - d) * nrm).sum(1), abs(v[0]) - abs(v[1])], q0, max_nfev=2000)
        assert abs(sol.x[10] - qt[10]) < 0.05 and abs(sol.x[11] - qt[11]) < 0.005, (sol.x[10:], qt[10:])
        if len(qt) > 12:
            assert abs(sol.x[12] - qt[12]) < 0.005, sol.x[12]
    print("n39 ok: round trip < 1e-6 with the column term and kappa; identical to autocal on a rigid camera; beta, d, kappa recovered from clean lines")


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    elif "--run" in sys.argv:
        run()
    else:
        selfcheck()
