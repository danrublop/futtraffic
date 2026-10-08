"""Which camera model describes ST2's stitched panorama, and can the operator's clicks fit it.

PROBES, NOT A PROTOCOL. Nothing here was committed before it first ran (the
scratchpad versions ran 2026-09-14 and are ported as-is), nothing ships, and
every number it prints is a lead for a protocol. Findings and their tags:
research-stitched-panorama-calibration (not published).

    .venv/bin/python model/validation/camera_models.py --lut       # 1 s
    .venv/bin/python model/validation/camera_models.py --provenance  # 1 s
    .venv/bin/python model/validation/camera_models.py --perfect   # ~3 min, nine sequences
    .venv/bin/python model/validation/camera_models.py --real      # ~10 s, 118575 only
    .venv/bin/python model/validation/camera_models.py --demo      # self-check

--lut      Is every ST2 mapx/mapy table exactly cv2.fisheye.initUndistortRectifyMap
           of the match's own *_camera_intrinsics.npz? (ST2's recipe:
           src/calibration/generate_calibration_mappings.py:104-132, MIT.)

--perfect  Each form fitted two ways, scored at N12's population (calib_product:
           GT feet every 25th frame, ST2's chain inverted as reference, feet > 8 px
           off its grid dropped):
             capacity  fitted to the chain itself on a 2.5 m pitch grid
             clicks21  fitted to the 21 UI landmarks at the chain's own pixels
           CIRCULAR for the fisheye forms: the reference IS a fisheye + homography,
           and perfect clicks are its pixels. It says whether 21 points identify the
           model that generated them, not whether that model describes the image.

--provenance  Re-runs ST2's own cv2.fisheye.calibrate call on
           data/footage/soccertrack_calib_118575's 65 keypoints and prints it beside the
           shipped npz. They match, so those keypoints ARE the vendor's calibration input.

--real     Real hand-placed pixels instead of the chain's: the 21 UI landmarks taken
           from those same 118575 keypoints, each form fitted to them - plus the UI's own
           solver and N27's perspective_tps on the same clicks - then scored
           (a) at N12's feet against the chain and (b) against ST2's `loc` on
           calib_product.absolute's frozen pairing. One sequence. The clicks are a
           subset of what the reference chain was fitted on (--provenance), so (a) is
           partly circular; (b) is floored by the chain's own ~1.4 m.

Every fit is Levenberg-Marquardt on pixel residuals from a handful of pose
initialisations, best fit RMS kept. Nothing is selected on feet or on `loc`.
`homog8` is a plain least-squares homography, NOT the UI's solver, so it only
approximates the shipped map (N12 carries the shipped one).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(ROOT, "model", "engine")]

import calib_product as CP  # noqa: E402
import st2_mot as T  # noqa: E402
from st2_overlay import RAW, pitch_of, projector  # noqa: E402

S = 2000.0                    # radial normaliser for the homography+radial forms, px
REAL_KP = os.path.join(ROOT, "data/footage/soccertrack_calib_118575/118575_keypoints.json")
REAL_MATCH_M = 1.0            # a hand keypoint within this of a UI landmark stands in for its click


def _rot(rv):
    return cv2.Rodrigues(np.asarray(rv, float))[0]


def _plane_hit(R, t, d):
    """Camera rays d (camera frame) -> pitch-plane xy; NaN behind the camera."""
    Rt = R.T
    dw = d @ Rt.T
    C = -Rt @ t
    lam = -C[2] / dw[:, 2]
    P = C[None] + lam[:, None] * dw
    P[~(lam > 0)] = np.nan
    return P[:, :2]


def _cam(p6, xy):
    return np.column_stack([xy, np.zeros(len(xy))]) @ _rot(p6[:3]).T + p6[3:6]


# ---- forms: forward(full params, pitch xy) -> px ; inverse(full params, px) -> pitch xy ----
def fish_fwd(q, xy):          # q = fx fy cx cy k1 k2 | rvec tvec   (cv2.fisheye, k3=k4=0)
    X = _cam(q[6:12], xy)
    rho = np.hypot(X[:, 0], X[:, 1])
    th = np.arctan2(rho, X[:, 2])
    s = th * (1 + q[4] * th**2 + q[5] * th**4) / np.maximum(rho, 1e-12)
    return np.column_stack([q[2] + q[0] * s * X[:, 0], q[3] + q[1] * s * X[:, 1]])


def fish_inv(q, uv):
    m = np.column_stack([(uv[:, 0] - q[2]) / q[0], (uv[:, 1] - q[3]) / q[1]])
    thd = np.hypot(*m.T)
    th = thd.copy()
    for _ in range(30):       # Newton on theta_d(theta)
        f = th * (1 + q[4] * th**2 + q[5] * th**4) - thd
        th = th - f / (1 + 3 * q[4] * th**2 + 5 * q[5] * th**4)
    dirn = m / np.maximum(thd, 1e-12)[:, None]
    return _plane_hit(_rot(q[6:9]), q[9:12], np.column_stack([np.sin(th)[:, None] * dirn, np.cos(th)]))


def cyl_fwd(q, xy, eq=False):  # q = fu fv cu cv | rvec tvec ; eq: equirectangular (spherical)
    X = _cam(q[4:10], xy)
    rho = np.hypot(X[:, 0], X[:, 2])
    v = np.arctan2(X[:, 1], rho) if eq else X[:, 1] / rho
    return np.column_stack([q[2] + q[0] * np.arctan2(X[:, 0], X[:, 2]), q[3] + q[1] * v])


def cyl_inv(q, uv, eq=False):
    a = (uv[:, 0] - q[2]) / q[0]
    e = (uv[:, 1] - q[3]) / q[1]
    return _plane_hit(_rot(q[4:7]), q[7:10], np.column_stack([np.sin(a), np.tan(e) if eq else e, np.cos(a)]))


def hr_fwd(q, xy):            # q = h11..h32 (h33=1) | cx cy k1 k2
    H = np.append(q[:8], 1.0).reshape(3, 3)
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    d = p[:, :2] / p[:, 2:3] - q[8:10]
    r2 = (d**2).sum(1) / S**2
    return q[8:10] + d * (1 + q[10] * r2 + q[11] * r2**2)[:, None]


def hr_inv(q, uv):
    H = np.append(q[:8], 1.0).reshape(3, 3)
    dd = uv - q[8:10]
    d = dd.copy()
    for _ in range(50):
        r2 = (d**2).sum(1) / S**2
        d = dd / (1 + q[10] * r2 + q[11] * r2**2)[:, None]
    p = np.column_stack([d + q[8:10], np.ones(len(d))]) @ np.linalg.inv(H).T
    return p[:, :2] / p[:, 2:3]


def _eq_fwd(q, x):
    return cyl_fwd(q, x, True)


def _eq_inv(q, u):
    return cyl_inv(q, u, True)


#: name -> (free vector -> full vector, forward, inverse)
FORMS = {
    "fisheye12": (lambda v: v, fish_fwd, fish_inv),
    "fisheye11_fx=fy": (lambda v: np.r_[v[0], v[0], v[1:]], fish_fwd, fish_inv),
    "fisheye10_fx=fy_k2=0": (lambda v: np.r_[v[0], v[0], v[1:4], 0.0, v[4:]], fish_fwd, fish_inv),
    "equidistant9_fx=fy_k=0": (lambda v: np.r_[v[0], v[0], v[1:3], 0.0, 0.0, v[3:]], fish_fwd, fish_inv),
    "cylinder10": (lambda v: v, cyl_fwd, cyl_inv),
    "equirect10": (lambda v: v, _eq_fwd, _eq_inv),
    "homog8": (lambda v: np.r_[v, 0, 0, 0, 0], hr_fwd, hr_inv),
    "homog+radial11": (lambda v: np.r_[v, 0.0], hr_fwd, hr_inv),
    "homog+radial12": (lambda v: v, hr_fwd, hr_inv),
}


def _pose_inits(L, W):
    """Camera beside either touchline, low or high, either roll: 8 starting poses."""
    out = []
    for side in (+1, -1):
        for h in (10.0, 25.0):
            C = np.array([L / 2, W / 2 + side * (W / 2 + 25.0), h])
            z = np.array([L / 2, W / 2, 0.0]) - C
            z /= np.linalg.norm(z)
            for up in (+1, -1):
                x = np.cross(z, [0, 0, up])
                x /= np.linalg.norm(x)
                R = np.stack([x, np.cross(z, x), z])
                out.append(np.r_[cv2.Rodrigues(R)[0].ravel(), -R @ C])
    return out


def fit_form(name, obj, img, wh, P):
    """-> (fit rms px, full params) or None. obj pitch metres, img px."""
    build, fwd, _ = FORMS[name]
    obj, img = np.asarray(obj, float), np.asarray(img, float)
    W, Hh = wh
    f0 = W / np.pi
    if name.startswith(("fisheye", "equidistant")):
        drop = {"fisheye12": [], "fisheye11_fx=fy": [1], "fisheye10_fx=fy_k2=0": [1, 5],
                "equidistant9_fx=fy_k=0": [1, 4, 5]}[name]
        inits = [np.delete(np.r_[f0, f0, W / 2, Hh / 2, 0, 0, p6], drop) for p6 in _pose_inits(P.length, P.width)]
    elif name in ("cylinder10", "equirect10"):
        inits = [np.r_[f0, f0, W / 2, Hh / 2, p6] for p6 in _pose_inits(P.length, P.width)]
    else:
        H, _ = cv2.findHomography(obj, img, 0)
        base = np.r_[(H / H[2, 2]).ravel()[:8], W / 2, Hh / 2, 0.0, 0.0]
        inits = [base[:{"homog8": 8, "homog+radial11": 11}.get(name, 12)]]
    best = None
    for v0 in inits:
        try:
            r = least_squares(lambda v: (fwd(build(v), obj) - img).ravel(), v0,
                              method="lm", max_nfev=4000 * len(v0))
        except (ValueError, np.linalg.LinAlgError):
            continue
        rms = float(np.sqrt(((fwd(build(r.x), obj) - img) ** 2).sum(1).mean()))
        if np.isfinite(rms) and (best is None or rms < best[0]):
            best = (rms, build(r.x))
    return best


class Fitted:
    """A fitted form with the `to_pitch` calib_product.absolute consumes, and the
    `jacobian` stats.kalman_ground reads instead of differentiating an H it lacks."""

    H = None

    def __init__(self, name, q):
        self.name, self.q = name, q

    def to_pitch(self, uv):
        return FORMS[self.name][2](self.q, np.asarray(uv, float).reshape(-1, 2))

    def jacobian(self, uv, e=0.5):
        """d(metres)/d(px), (N,2,2), central differences at +-e px."""
        u = np.asarray(uv, float).reshape(-1, 2)
        dx = (self.to_pitch(u + [e, 0]) - self.to_pitch(u - [e, 0])) / (2 * e)
        dy = (self.to_pitch(u + [0, e]) - self.to_pitch(u - [0, e])) / (2 * e)
        return np.stack([dx, dy], -1)


def _score(ref, pred):
    ok = np.isfinite(pred).all(1)
    if not ok.any():
        return {"med": None, "p90": None, "w1": None, "drop": 1.0}
    d = np.linalg.norm(pred[ok] - ref[ok], axis=1)
    return {"med": round(float(np.median(d)), 3), "p90": round(float(np.percentile(d, 90)), 3),
            "w1": round(float((d < 1).mean()), 3), "drop": round(float(1 - ok.mean()), 4)}


def _eval(obj, img, wh, P, uv, ref):
    out = {}
    for name in FORMS:
        b = fit_form(name, obj, img, wh, P)
        out[name] = ({"fit_rms_px": None, **_score(ref, np.full_like(uv, np.nan))} if b is None else
                     {"fit_rms_px": round(b[0], 2), **_score(ref, Fitted(name, b[1]).to_pitch(uv))})
    return out


# ------------------------------------------------------------------------ modes
def lut(seqs=CP.SEQS + ("132831",)):
    for s in seqs:
        z = np.load(f"{RAW}/{s}/{s}_camera_intrinsics.npz")
        K, D, Kn = z["K"], z["D"], z["Knew"]
        mx, my = np.load(f"{RAW}/{s}/{s}_mapx.npy"), np.load(f"{RAW}/{s}/{s}_mapy.npy")
        h, w = my.shape
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, (w, h), cv2.CV_16SC2)
        fa, fb = cv2.convertMaps(mx, my, cv2.CV_32FC1), cv2.convertMaps(m1, m2, cv2.CV_32FC1)
        print(s, f"{w}x{h}", f"fy/fx={K[1, 1] / K[0, 0]:.1f}", f"rms={float(z['rms']):.2f}",
              f"LUT equal int={float((m1 == mx).all(-1).mean()):.4f} frac={float((m2 == my).mean()):.4f}",
              f"max|d|={float(np.hypot(fa[0] - fb[0], fa[1] - fb[1]).max()):.4f} px")


def provenance(seq="118575"):
    """Were our 118575 keypoints the input of ST2's shipped calibration? Re-run ST2's
    cv2.fisheye.calibrate call (generate_calibration_mappings.py:106-125) on them."""
    z = np.load(f"{RAW}/{seq}/{seq}_camera_intrinsics.npz")
    kp = json.load(open(REAL_KP))
    img = np.array(list(kp.values()), np.float64)
    obj = np.array([[*map(float, k.strip("()").split(",")), 0.0] for k in kp])
    fl = (cv2.CALIB_RECOMPUTE_EXTRINSIC + cv2.CALIB_FIX_SKEW + cv2.CALIB_CHECK_COND
          + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4)
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        [obj.reshape(1, -1, 3)], [img.reshape(1, -1, 2)], T._wh(seq), np.zeros((3, 3)), np.zeros((4, 1)),
        None, None, flags=fl, criteria=(cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6))
    print(f"{seq}: {len(kp)} keypoints, all integer {all(float(x) == int(x) for p in kp.values() for x in p)}")
    print(f"  refit rms {rms:.4f}  shipped {float(z['rms']):.4f}")
    print(f"  fx refit {K[0, 0]:.2f} shipped {z['K'][0, 0]:.2f} | k1 refit {D.ravel()[0]:.5f} shipped {z['D'].ravel()[0]:.5f}"
          f" | k2 refit {D.ravel()[1]:.5f} shipped {z['D'].ravel()[1]:.5f}")


def _table(res, part):
    print(f"== {part}: median m at feet per seq | median of medians, max, max p90")
    for f in FORMS:
        meds = [res[s][part][f]["med"] for s in res]
        ok = [m for m in meds if m is not None]
        p90 = [res[s][part][f]["p90"] for s in res if res[s][part][f]["p90"] is not None]
        print(f"{f:24s}", " ".join(f"{m if m is not None else 'NA':>6}" for m in meds),
              f"| MoM {np.median(ok):.3f} max {max(ok):.3f} p90max {max(p90):.3f}")


def perfect(seqs=CP.SEQS):
    res = {}
    for seq in seqs:
        uv, _, ref, drop, _ = CP._feet_and_ref(seq)
        wh, P = T._wh(seq), pitch_of(seq)
        gx, gy = np.meshgrid(np.arange(0, P.length + 1e-9, 2.5), np.arange(0, P.width + 1e-9, 2.5))
        G = np.column_stack([gx.ravel(), gy.ravel()])
        px, ok = projector(seq, wh)(G)
        P21, clicks = CP.clicks_of(seq)
        lm = P21.landmarks()
        obj, img = np.array([lm[n] for n in clicks]), np.array([clicks[n] for n in clicks])
        res[seq] = {"ref_drop": round(drop, 4), "n_clicks": len(obj),
                    "capacity": _eval(G[ok], px[ok], wh, P, uv, ref),
                    "clicks21": _eval(obj, img, wh, P, uv, ref)}
        print(seq, "done", flush=True)
    _table(res, "capacity")
    _table(res, "clicks21")
    print("seqs", list(res), "ref_drop max", max(r["ref_drop"] for r in res.values()))
    return res


def real_clicks(seq="118575"):
    """The 21 UI landmarks as the hand-annotated keypoints place them."""
    kp = json.load(open(REAL_KP))
    KP = np.array([tuple(map(float, k.strip("()").split(","))) for k in kp])
    PX = np.array(list(kp.values()), float)
    P, perfect_clicks = CP.clicks_of(seq)
    names, obj, img = [], [], []
    for n, xy in P.landmarks().items():
        d = np.hypot(*(KP - xy).T)
        if d.min() <= REAL_MATCH_M and n in perfect_clicks:
            names.append(n)
            obj.append(KP[d.argmin()])
            img.append(PX[d.argmin()])
    return names, np.array(obj), np.array(img)


def real(seq="118575", forms=("homog8", "fisheye12", "fisheye11_fx=fy", "cylinder10", "equirect10")):
    import calibrators  # noqa: PLC0415
    names, obj, img = real_clicks(seq)
    wh, P = T._wh(seq), pitch_of(seq)
    uv, _, ref, drop, _ = CP._feet_and_ref(seq)
    print(f"{seq}: {len(obj)} UI landmarks matched to hand keypoints; feet ref_drop {drop:.4f}")
    cals, res = {}, {}
    # The PRODUCT's maps on the same clicks: the UI's own solver, then N27's spline on it.
    payload = CP.solve(P, {n: [float(u), float(v)] for n, (u, v) in zip(names, img)})
    if payload is None:
        print("UI solver refused the real clicks")
    else:
        for name, d in (("ui_homography", payload), ("ui_homography+tps", {**payload, "projection_type": "perspective_tps"})):
            cals[name] = calibrators.from_dict(d)
            res[name] = {"fit_rms_px": None, "at_feet_vs_chain": _score(ref, np.asarray(cals[name].to_pitch(uv), float))}
        print("tps active", getattr(cals["ui_homography+tps"], "active", None), "| UI solver outliers", payload.get("outliers"))
    for name in forms:
        b = fit_form(name, obj, img, wh, P)
        if b is None:
            print(f"{name:24s} fit FAILED")
            continue
        cals[name] = Fitted(name, b[1])
        res[name] = {"fit_rms_px": round(b[0], 2), "at_feet_vs_chain": _score(ref, cals[name].to_pitch(uv))}
    for name, cal in cals.items():
        a = CP.absolute(seqs=(seq,), cals={seq: cal})["absolute"][seq]
        res[name]["vs_loc"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in a.items()}
    print("\nform                     fit rms px | at feet vs chain med p90 | vs loc med p90 <2m (reference med)")
    for name, r in res.items():
        f, v = r["at_feet_vs_chain"], r["vs_loc"]
        print(f"{name:24s} {str(r['fit_rms_px']):>9} | {f['med']:>6} {f['p90']:>6} | "
              f"{v['clicked_median_m']:>6} {v['clicked_p90_m']:>6} {v['clicked_within_2m']:>5} "
              f"({v['reference_median_m']}, n={v['n']}, drop {v['drop_share']})")
    return res


def demo():
    """Plant a known camera, project 21 points, fit, and invert back exactly."""
    class Pitch:
        length, width = 105.0, 68.0
    wh = (4096, 1080)
    p6 = _pose_inits(Pitch.length, Pitch.width)[0]
    truth = np.r_[1600.0, 1600.0, 2048.0, 540.0, -0.06, 0.02, p6]
    obj = np.array([[x, y] for x in (0, 16.5, 52.5, 88.5, 105) for y in (0, 13.8, 34, 54.2, 68)])[:21]
    img = fish_fwd(truth, obj)
    assert np.isfinite(img).all()
    b = fit_form("fisheye11_fx=fy", obj, img, wh, Pitch)
    assert b is not None and b[0] < 1e-3, b and b[0]
    back = Fitted("fisheye11_fx=fy", b[1]).to_pitch(img)
    assert np.abs(back - obj).max() < 1e-3, np.abs(back - obj).max()
    # the plain homography cannot absorb the lens: its residual must NOT vanish
    h = fit_form("homog8", obj, img, wh, Pitch)
    assert h[0] > 1.0, h[0]
    for fwd, inv, q in ((cyl_fwd, cyl_inv, np.r_[1300.0, 1300.0, 2048.0, 540.0, p6]),
                        (_eq_fwd, _eq_inv, np.r_[1300.0, 1300.0, 2048.0, 540.0, p6])):
        assert np.abs(inv(q, fwd(q, obj)) - obj).max() < 1e-6
    print("camera_models demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("lut", "provenance", "perfect", "real", "demo"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"lut": lut, "provenance": provenance, "perfect": perfect, "real": real, "demo": demo}[next(m for m in ("lut", "provenance", "perfect", "real", "demo") if getattr(a, m))]()
