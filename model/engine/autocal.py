"""Zero-click pitch calibration for a fixed panoramic camera: the camera is found from the paint.

WHAT IT IS FOR
    Replacing the operator's dots on a fixed panoramic rig: the rig's mount is constrained and the
    pitch is a known drawing, so the camera can be found from the paint.

THE MODEL
    One central SPHERICAL camera (equirectangular, square angular pixels) over a flat pitch: the
    column is an azimuth, the row an elevation, about an axis that is tilted. Ten numbers:
    fu fv cu cv | Rodrigues rotation | translation; an optional smooth warp of the row by column
    follows them. Pitch frame: y = 0 is the FAR touchline, the camera sits beyond y = width.

THE PIPELINE  (docs/experiments/n34-zero-click.md)
    median frame -> paint by SHAPE -> coarse search over physical mounts under a two-way score ->
    polish -> project-and-search registration to the paint (Carr, Sheikh & Matthews 2012, on a
    sphere) -> acceptance with no ground truth -> or start from the install's known camera.
    It REFUSES rather than guesses: no readable grass, too little paint found, lines that do not
    close, an implausible mount, or a family of the drawing missing.

EVIDENCE AND LIMITS  one install only. calibration-evidence (not published)#evidence-zero-click and #known-limits.

    .venv/bin/python model/engine/autocal.py            # self-check: a synthetic panorama, an unknown mount
    .venv/bin/python model/engine/autocal.py VIDEO      # calibrate one video, print the report
"""
import itertools
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares, minimize

# ---- registration (model/validation/n31_linereg.py is where each of these was earned) ----
STEP_M = 0.5
RADII = (12.0, 8.0, 5.0, 3.0)
MIN_PEAK = 6.0            # grey levels the ridge must stand above its search window's median
MIN_HITS = 300
MAX_LINE_RMS_PX = 2.5
CLICK_WEIGHT = 0.3
WARP_DEG = 5              # a(u) and b(u) are polynomials of this degree in the normalised column
VERT_CUBIC = 0.0          # kappa in v = cv + fv (e + kappa e^3): 0 is the equirect law; N36's review found kappa -0.4 fits this install
WARP_RIDGE = 0.05         # px of pseudo-residual per unit coefficient: shrinks the warp toward the rigid camera
WARP_V0, WARP_VS = 540.0, 540.0
OUTWARD = {"touch_far": (0.0, -1.0), "touch_near": (0.0, 1.0), "goal_l": (-1.0, 0.0), "goal_r": (1.0, 0.0)}
# ---- the zero-click search (model/validation/n34_autocal.py) ----
TRUNC = 40.0                                            # px, full resolution
TRUNC_WIDE = 200.0
GRID = {"fov_deg": (120, 135, 150, 165, 180), "tilt_deg": (6, 10, 14, 18, 22, 27, 33), "height_m": (4, 6, 8, 11, 15),
        "behind_m": (3, 8, 14, 21, 30), "along_m": (-12, -6, 0, 6, 12), "pan_deg": (-8, -4, 0, 4, 8)}
TOP_K, TOP_REG = 40, 6                                  # candidates polished; polished candidates registered
TRUNC_M, N_PAINT = 5.0, 1500                            # paint -> drawing: truncation in metres, paint pixels sampled
BOUNDS = {"fov_deg": (100, 200), "tilt_deg": (3, 45), "height_m": (2.5, 20), "behind_m": (1, 40), "along_m": (-25, 25),
          "pan_deg": (-15, 15), "roll_deg": (-4, 4), "cv_off": (-150, 150)}
PAINT_PCT, PAINT_LEN, STRAIGHT = 96.0, 60, 0.15         # lineness percentile on the pitch; stroke length px; minor/major variance
MIN_PITCH_SHARE = 0.15                                  # of the frame: less grass than this is not a pitch we can read
AUTO_RADII = (25.0, 12.0, 8.0, 5.0, 3.0)


def _rot(rv):
    return cv2.Rodrigues(np.asarray(rv, float))[0]


def model_samples(P, step=STEP_M, size=None):
    """-> (points (N,2) metres, unit tangents (N,2), group name per point). Lines and the centre
    circle; the penalty arcs and spots are left out. `size` = (length, width) draws the SAME samples -
    the counts stay `P`'s - on a pitch of another size, so a fit can move the pitch's length under them.
    ponytail: no arcs - add them if the box region turns out under-constrained."""
    L, W = (float(P.length), float(P.width)) if size is None else (float(size[0]), float(size[1]))
    L0, W0 = float(P.length), float(P.width)
    pw, pd = float(P.penalty_width) / 2, float(P.penalty_depth)
    gw, gd = float(P.goal_box_width) / 2, float(P.goal_box_depth)
    segs = [("touch_far", (0, 0), (L, 0)), ("touch_near", (0, W), (L, W)), ("goal_l", (0, 0), (0, W)),
            ("goal_r", (L, 0), (L, W)), ("halfway", (L / 2, 0), (L / 2, W))]
    for side, x0, sgn in (("l", 0.0, 1.0), ("r", L, -1.0)):
        for nm, half, depth in (("pen", pw, pd), ("six", gw, gd)):
            xf = x0 + sgn * depth
            segs += [(f"{nm}_{side}_front", (xf, W / 2 - half), (xf, W / 2 + half)),
                     (f"{nm}_{side}_far", (x0, W / 2 - half), (xf, W / 2 - half)),
                     (f"{nm}_{side}_near", (x0, W / 2 + half), (xf, W / 2 + half))]
    counts = {"touch_far": L0, "touch_near": L0, "goal_l": W0, "goal_r": W0, "halfway": W0}   # the only lengths that move with the size
    pts, tan, grp = [], [], []
    for name, a, b in segs:
        a, b = np.array(a, float), np.array(b, float)
        n = max(2, int(counts.get(name, np.linalg.norm(b - a)) / step))
        t = (np.arange(n) + 0.5) / n
        pts.append(a + t[:, None] * (b - a))
        tan.append(np.tile((b - a) / np.linalg.norm(b - a), (n, 1)))
        grp += [name] * n
    r = float(P.centre_circle_r)
    th = np.arange(0, 2 * np.pi, step / r)
    pts.append(np.column_stack([L / 2 + r * np.cos(th), W / 2 + r * np.sin(th)]))
    tan.append(np.column_stack([-np.sin(th), np.cos(th)]))
    grp += ["circle"] * len(th)
    return np.vstack(pts), np.vstack(tan), np.array(grp)


def median_frame(video, n=40):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stack, k = None, 0
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            if stack is None:
                stack = np.empty((n, *f.shape), np.uint8)      # one buffer; a list + np.stack + median was three
            stack[k], k = f, k + 1
    if k < 5:
        raise ValueError(f"{video}: only {k} frames could be read")
    return np.median(stack[:k], axis=0, overwrite_input=True).astype(np.uint8)


def grass_map(bgr):
    """> 0 on grass: green above both other channels. Paint, boards, track and stand are not."""
    b, g, r = (bgr[:, :, i].astype(np.float32) for i in range(3))
    return cv2.GaussianBlur(g - np.maximum(r, b), (0, 0), 1.0)


def ridge_map(bgr):
    w = bgr.min(axis=2).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.GaussianBlur(cv2.morphologyEx(w, cv2.MORPH_TOPHAT, k), (0, 0), 1.0)


def native_map(bgr):
    """min(B,G,R) as the sensor gave it: no top-hat, no blur. A far touchline is ~1 px of paint 3 px from a
    board; ridge_map's 15 px top-hat and sigma-1 blur merge the two (docs/experiments/n36-far-touchline.md)."""
    return bgr.min(axis=2).astype(np.float32)


FLANKS_PX = (2.0, 4.0)     # a far line is ~1 px wide, a near one ~4: the grass either side is looked for at both
PERIM_RADIUS = 8.0        # px: a perimeter window never shrinks below this - the 4 px flanks have to fit inside it, and
                          # a wide window is safe where the rule cannot take a board
GRASS_MIN = 5.0            # grass_map units: green clear of the other channels, not a board's grey


def _basis(u, width):
    return np.polynomial.legendre.legvander(np.clip(2.0 * u / width - 1.0, -1, 1), WARP_DEG)


def _fwd(q, xy, width=4096.0):
    """Pitch metres -> px. `q` is equirect10, optionally followed by a column warp: the row moves by
    a(u) + b(u) (v - v0) / vs. A stitched panorama is one spherical camera only up to each lens's own
    small vertical correction (Gaddam et al. 2014; research-unstitching-panorama (not published) Q1), which
    is nothing on the near side and metres on the far one. The column is untouched, so the inverse
    is closed-form."""
    xy = np.asarray(xy, float)
    if xy.shape[1] == 2:                          # on the pitch plane; three columns is a point OFF it - a crossbar
        xy = np.column_stack([xy, np.zeros(len(xy))])
    X = xy @ _rot(q[4:7]).T + q[7:10]
    e = np.arctan2(X[:, 1], np.hypot(X[:, 0], X[:, 2]))
    p = np.column_stack([q[2] + q[0] * np.arctan2(X[:, 0], X[:, 2]), q[3] + q[1] * (e + VERT_CUBIC * e ** 3)])
    if len(q) > 10:
        B = _basis(p[:, 0], width)
        n = WARP_DEG + 1
        p = np.column_stack([p[:, 0], p[:, 1] + B @ q[10:10 + n] + (B @ q[10 + n:10 + 2 * n]) * (p[:, 1] - WARP_V0) / WARP_VS])
    return p


def _inv(q, uv, width=4096.0):
    uv = np.asarray(uv, float).reshape(-1, 2)
    if len(q) > 10:
        B = _basis(uv[:, 0], width)
        n = WARP_DEG + 1
        a, b = B @ q[10:10 + n], (B @ q[10 + n:10 + 2 * n]) / WARP_VS
        uv = np.column_stack([uv[:, 0], (uv[:, 1] - a + b * WARP_V0) / (1.0 + b)])
    R = _rot(q[4:7])
    a, e = (uv[:, 0] - q[2]) / q[0], (uv[:, 1] - q[3]) / q[1]
    if VERT_CUBIC:
        y, e = e, e.copy()
        for _ in range(4):                                              # Newton on e + k e^3 = y; |k e^2| <= 0.04 in-frame (|e| < 0.26 rad);
            # ponytail: off-frame rows with |3 k e^2| near 1 do not converge in 4 steps - no product path asks for them
            e -= (e + VERT_CUBIC * e ** 3 - y) / (1.0 + 3.0 * VERT_CUBIC * e ** 2)
    d = np.column_stack([np.sin(a), np.tan(e), np.cos(a)]) @ R          # rays in pitch axes
    C = -R.T @ q[7:10]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -C[2] / d[:, 2]
        P = C[None] + lam[:, None] * d
    P[~(lam > 0)] = np.nan                                              # the ray never meets the pitch
    return P[:, :2]


class Camera:
    """What dist_eval and n31_score need of a map: to_pitch, jacobian, H = None."""

    H = None

    def __init__(self, q, width=4096.0):
        self.q, self.width = np.asarray(q, float), float(width)

    def to_pitch(self, uv):
        return _inv(self.q, uv, self.width)

    def jacobian(self, uv, e=0.5):
        u = np.asarray(uv, float).reshape(-1, 2)
        dx = (self.to_pitch(u + [e, 0]) - self.to_pitch(u - [e, 0])) / (2 * e)
        dy = (self.to_pitch(u + [0, e]) - self.to_pitch(u - [0, e])) / (2 * e)
        return np.stack([dx, dy], -1)


def search(ridge, q, X, T, radius, step=0.5, grass=None, outward=None, native=None):
    """-> (hit mask, hit px (N,2), image normals (N,2)). The ridge's centroid along each sample's
    normal, inside `radius` px of where the camera puts it. With `grass` and `outward` (metres,
    zero for interior samples), a perimeter sample takes the first grass-footed ridge from inside.
    With `native` (native_map) a perimeter sample instead takes the first thin ridge with GRASS ON
    BOTH SIDES - paint, which a board is not: the strongest such ridge in the window, or nothing (N36)."""
    if native is not None and outward is not None and grass is not None:
        e = np.abs(outward).sum(1) > 0
        if e.any() and not e.all():                         # interior and perimeter samples, each in its own window
            hit, d, nrm = np.zeros(len(X), bool), np.zeros((len(X), 2)), np.zeros((len(X), 2))
            hit[~e], d[~e], nrm[~e] = search(ridge, q, X[~e], T[~e], radius, step)
            hit[e], d[e], nrm[e] = search(ridge, q, X[e], T[e], max(radius, PERIM_RADIUS), step, grass, outward[e], native)
            return hit, d, nrm
    p = _fwd(q, X)
    t = _fwd(q, X + 0.25 * T) - _fwd(q, X - 0.25 * T)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    edge = np.zeros(len(X), bool)
    if outward is not None and grass is not None:
        edge = np.abs(outward).sum(1) > 0
        o = _fwd(q, X + 0.25 * outward) - p
        nrm[edge & ((o * nrm).sum(1) < 0)] *= -1            # +s now points OUT of the pitch on perimeter samples
    H, Wd = ridge.shape
    s = np.arange(-radius, radius + step / 2, step)
    uu = p[:, 0:1] + s[None] * nrm[:, 0:1]
    vv = p[:, 1:2] + s[None] * nrm[:, 1:2]
    inside = np.isfinite(p).all(1) & (uu.min(1) >= 1) & (uu.max(1) <= Wd - 2) & (vv.min(1) >= 1) & (vv.max(1) <= H - 2)
    prof = np.zeros_like(uu)
    prof[inside] = map_coordinates(ridge, [vv[inside].ravel(), uu[inside].ravel()], order=1).reshape(-1, len(s))
    base = np.median(prof, axis=1, keepdims=True)
    k = prof.argmax(1)
    if edge.any():
        base[edge] = np.percentile(prof[edge], 25, axis=1, keepdims=True)   # half such a window is board or stand
        gr = np.zeros_like(uu)
        rows = inside & edge
        gr[rows] = map_coordinates(grass, [vv[rows].ravel(), uu[rows].ravel()], order=1).reshape(-1, len(s))
        foot = int(round(3.0 / step))                                        # 3 px inside the peak
        if native is not None:
            prof[rows] = map_coordinates(native, [vv[rows].ravel(), uu[rows].ravel()], order=1).reshape(-1, len(s))
        for i in np.flatnonzero(rows):
            k[i] = -1
            if native is not None:
                # the STRONGEST such ridge, not the first: walking from inside, grass texture clears MIN_PEAK on
                # un-smoothed pixels and every perimeter line came back at the window's inner end (N36, first try)
                best = 0.0
                for j in range(1, len(s) - 1):
                    if not (prof[i, j] >= prof[i, j - 1] and prof[i, j] > prof[i, j + 1]):
                        continue
                    for f in (int(round(w / step)) for w in FLANKS_PX):
                        if f <= j < len(s) - f and gr[i, j - f] > GRASS_MIN and gr[i, j + f] > GRASS_MIN:
                            side = 0.5 * (prof[i, j - f] + prof[i, j + f])
                            if prof[i, j] - side >= max(MIN_PEAK, best):
                                best, k[i], base[i, 0] = prof[i, j] - side, j, side
                continue
            # "first" means the first STRONG ridge: at least half the window's tallest. Grass speckle clears
            # MIN_PEAK on its own, and taking it left a bold near touchline with 36 hits of 210.
            need = max(MIN_PEAK, 0.5 * float(prof[i].max() - base[i, 0]))
            for j in range(foot, len(s) - 1):
                if (prof[i, j] - base[i, 0] >= need and prof[i, j] >= prof[i, j - 1] and prof[i, j] > prof[i, j + 1]
                        and gr[i, j - foot] > 0):
                    k[i] = j
                    break
    found = k >= 0
    k = np.where(found, k, 0)
    peak = prof[np.arange(len(prof)), k][:, None]
    # centroid over the peak's own lobe (above half its height), so a wide near-side line gives its middle
    lobe = (prof - base) >= 0.5 * (peak - base)
    idx = np.arange(len(s))[None]
    left, right = k.copy(), k.copy()
    for i in np.flatnonzero(inside):
        while left[i] > 0 and lobe[i, left[i] - 1]:
            left[i] -= 1
        while right[i] < len(s) - 1 and lobe[i, right[i] + 1]:
            right[i] += 1
    run = (idx >= left[:, None]) & (idx <= right[:, None])
    wgt = np.where(run, prof - base, 0.0)
    off = (wgt * s[None]).sum(1) / np.maximum(wgt.sum(1), 1e-9)
    hit = inside & found & ((peak - base)[:, 0] >= MIN_PEAK) & (left > 0) & (right < len(s) - 1)   # the lobe must end inside the window
    return hit, p + off[:, None] * nrm, nrm


def register(ridge, q0, P, obj, img, grass=None, warp=False, leave_out=(), pts3d=None, square=0.0, radii=None,
             fit_length=False, native=None):
    """-> (camera, report). `q0` from the clicks; the lines refine it. `warp` frees the column warp
    for the last two radii, once the rigid camera has put the search on the paint. `leave_out` names
    line groups that are searched and reported but never fitted: the held-out paint a map has to land
    on unaided, which is the only ground-truth-free test of a map BETWEEN its constraints."""
    # `pts3d` = (XYZ metres, uv px): points off the plane. Every line and dot lies ON the pitch, and a
    # plane alone leaves the camera's height and its vertical scale free to trade against each other;
    # a crossbar 2.44 m up does not. `square` weighs a prior that the panorama's angular pixels are
    # square, |fu| = |fv|, which is what a spherical stitch is (research-unstitching-panorama (not published) Q1).
    # `fit_length` frees the pitch's LENGTH for the last radius. The penalty areas, goal areas and centre
    # circle are regulation, so the paint says how long the pitch is; it does not say how WIDE, because
    # width lies along this camera's line of sight (audit-n33 (not published) finding 6) - that has to come from
    # outside. X is exactly linear in the length, so X(L) = X + (L - L0) dX.
    X, T, grp = model_samples(P)
    dXdL = model_samples(P, size=(P.length + 1.0, P.width))[0] - X
    length = float(P.length)
    out = np.array([OUTWARD.get(g, (0.0, 0.0)) for g in grp]) if grass is not None else None
    use = ~np.isin(grp, list(leave_out))
    XYZ, UV = (np.asarray(pts3d[0], float), np.asarray(pts3d[1], float)) if pts3d is not None else (np.zeros((0, 3)), np.zeros((0, 2)))
    radii = tuple(radii or RADII)
    q, rep = np.array(q0, float), {}
    width = float(ridge.shape[1])
    for i, radius in enumerate(radii):
        if warp and i == len(radii) - 2 and len(q) == 10:
            q = np.r_[q, np.zeros(2 * (WARP_DEG + 1))]
        Xc = X + (length - P.length) * dXdL
        hit, d, nrm = search(ridge, q, Xc, T, radius, grass=grass, outward=out, native=native)
        hit &= use
        if hit.sum() < MIN_HITS:
            break
        Xh, dh, nh, Dh = Xc[hit], d[hit], nrm[hit], dXdL[hit]
        free_len = fit_length and i == len(radii) - 1
        nq = len(q)

        def res(v, Xh=Xh, dh=dh, nh=nh, Dh=Dh, nq=nq, base=length):
            c = v[:nq]
            line = ((_fwd(c, Xh + (v[nq] - base) * Dh if len(v) > nq else Xh, width) - dh) * nh).sum(1)
            click = CLICK_WEIGHT * (_fwd(c, obj, width) - img).ravel()
            off = (_fwd(c, XYZ, width) - UV).ravel() if len(XYZ) else np.zeros(0)
            return np.r_[line, click, off, square * (abs(c[0]) - abs(c[1])), WARP_RIDGE * c[10:]]
        if free_len:
            sol = least_squares(res, np.r_[q, length], loss="soft_l1", f_scale=2.0, max_nfev=2000)
            q, length = sol.x[:nq], float(sol.x[nq])
            J = sol.jac
            dof = max(len(sol.fun) - len(sol.x), 1)
            cov = np.linalg.pinv(J.T @ J) * float((sol.fun ** 2).sum() / dof)
            rep["length_m"], rep["length_sd_m"] = length, float(np.sqrt(max(cov[nq, nq], 0.0)))
            continue
        # one loss for both blocks: soft-L1 at 2 px bites the clicks' tens of pixels and leaves sub-2 px line residuals linear
        q = least_squares(res, q, loss="soft_l1", f_scale=2.0, max_nfev=2000).x
    X = X + (length - P.length) * dXdL
    hit_all, d, nrm = search(ridge, q, X, T, radii[-1] if not leave_out else radii[0], grass=grass, outward=out, native=native)
    hit = hit_all & use
    r = ((_fwd(q, X[hit], width) - d[hit]) * nrm[hit]).sum(1) if hit.any() else np.array([np.inf])
    held = hit_all & ~use
    if held.any():
        # held-out paint, in METRES: the normal residual through the map's own local scale
        rp = ((_fwd(q, X[held], width) - d[held]) * nrm[held]).sum(1)
        m = np.linalg.norm(_inv(q, d[held] + nrm[held], width) - _inv(q, d[held], width), axis=1)
        rep["held_out"] = {"groups": list(leave_out), "hits": int(held.sum()), "median_px": float(np.median(np.abs(rp))),
                           "median_m": float(np.median(np.abs(rp) * m)), "p90_m": float(np.percentile(np.abs(rp) * m, 90))}
    rep = {**rep, "hits": int(hit.sum()), "samples": int(len(X)), "line_rms_px": float(np.sqrt((r ** 2).mean())),
           "hits_by_group": {g: [int((hit & (grp == g)).sum()), int((grp == g).sum())] for g in sorted(set(grp))},
           "click_residual_px": ({"median": float(np.median(np.linalg.norm(_fwd(q, obj, width) - img, axis=1))),
                                  "max": float(np.linalg.norm(_fwd(q, obj, width) - img, axis=1).max())} if len(obj) else None)}
    rep["refused"] = (None if rep["hits"] >= MIN_HITS and rep["line_rms_px"] <= MAX_LINE_RMS_PX
                      else f"{rep['hits']} hits, line rms {rep['line_rms_px']:.2f} px")
    return (q if rep["refused"] is None else np.array(q0, float)), rep


LENGTHS_COARSE = np.arange(90.0, 120.01, 2.5)      # IFAB allows 90-120 m
LENGTH_FINE = np.arange(-2.0, 2.01, 0.5)


def _recentred(cam, P, L):
    """The same camera, said in the frame of a pitch of length L that shares `P`'s CENTRE SPOT. The
    pitch frame's origin is a corner, so a longer pitch moves its own origin half the difference."""
    cam = np.array(cam, float)
    cam[7:10] -= _rot(cam[4:7])[:, 0] * (L - P.length) / 2.0
    return cam


def _settle(ridge, grass, start, P, native=None, fix_length=False):
    """Register with no dots, and go round again while it still gains paint: the search radius
    shrinks inside `register`, so a start 20 px out needs more than one pass. -> (camera, report, pitch).

    THE PITCH'S LENGTH IS MEASURED HERE, not assumed: the boxes and circle are regulation, so only
    one length puts them all on paint. A wrong length is outside any search radius and a fit there
    bends the camera instead, so the length is SCANNED (a registration per length, most paint wins),
    then refined jointly with the camera. WIDTH is not measured; it has to come from outside.
    docs/reference/calibration.md#pitch-size (why: #evidence-zero-click)

    `fix_length`: somebody GAVE this pitch's length, and it wins. The camera returned is the one fitted
    at P.length; the scan still runs, and its answer is reported as `length_measured_m` for the record."""
    import dataclasses  # noqa: PLC0415
    empty = np.zeros((0, 2))

    def go(cam, P, fit=False, passes=4):
        rep, radii = None, AUTO_RADII
        for _ in range(passes):
            nxt, nrep = register(ridge, cam, P, empty, empty, grass=grass, square=1.0, radii=radii, fit_length=fit, native=native)
            if rep is not None and nrep["hits"] <= 1.02 * rep["hits"]:
                break
            cam, rep, radii = nxt, nrep, AUTO_RADII[1:]
        return cam, rep
    cam, rep = go(np.asarray(start, float), P)
    if rep["hits"] < MIN_HITS // 2:
        return cam, rep, P                      # no camera to measure a length under
    P0, cam0, rep0 = P, cam, rep
    scan = {}

    def at(L):
        if L not in scan:
            PL = dataclasses.replace(P, length=float(L))
            scan[L] = (*go(_recentred(start, P, L), PL, passes=2), PL)
        return scan[L][1]["hits"]
    scan[float(P.length)] = (cam, rep, P)
    best = max(LENGTHS_COARSE.tolist() + [float(P.length)], key=at)
    best = max([best + d for d in LENGTH_FINE.tolist()], key=at)
    cam, rep, P = scan[best]
    cam, fine = go(cam, P, fit=True)
    if "length_m" in fine and abs(fine["length_m"] - P.length) < 1.0:
        P = dataclasses.replace(P, length=round(float(fine["length_m"]), 2))
        rep = {**fine, "length_m": P.length}
    else:
        rep = {**rep, "length_m": P.length, "length_sd_m": None}
    rep["length_scan"] = {str(k): v[1]["hits"] for k, v in sorted(scan.items())}
    if fix_length:   # the given length stands; the camera is in ITS frame, the paint's answer is kept
        return cam0, {**rep0, "length_m": P0.length, "length_sd_m": None, "length_measured_m": rep["length_m"],
                      "length_measured_sd_m": rep.get("length_sd_m"), "length_scan": rep["length_scan"]}, P0
    return cam, rep, P


def camera(P, wh, fov_deg, tilt_deg, height_m, behind_m, along_m, pan_deg, roll_deg=0.0, cv_off=0.0):
    """A physical mount -> n31_linereg's 10 numbers (equirect, square pixels). The pitch frame has x
    along the touchline to image-right and y TOWARD the camera, which with z up is left-handed; the
    rotation stays proper and the mirror goes in the sign of fu, as every fit in this repo already does."""
    W, H = wh
    f = W / np.radians(fov_deg)
    ps, ti, ro = np.radians(pan_deg), np.radians(tilt_deg), np.radians(roll_deg)
    fwd = np.array([np.sin(ps) * np.cos(ti), -np.cos(ps) * np.cos(ti), -np.sin(ti)])
    right = np.array([-np.cos(ps), -np.sin(ps), 0.0])
    down = np.cross(fwd, right)
    right, down = np.cos(ro) * right + np.sin(ro) * down, np.cos(ro) * down - np.sin(ro) * right
    R = np.stack([right, down, fwd])
    C = np.array([P.length / 2 + along_m, P.width + behind_m, height_m])
    return np.r_[-f, f, W / 2.0, H / 2.0 + cv_off, cv2.Rodrigues(R)[0].ravel(), -R @ C]


def lineness(bgr):
    """Bright-ridge strength from the Hessian of min(B,G,R) at three scales: large where the image
    curves sharply ACROSS a stroke and not along it. A painted line 1-3 px wide stays green in colour
    (it blends with the turf), so colour cannot find the circle or the boxes; its SHAPE can, and grass
    speckle, which curves both ways, scores nothing."""
    w = bgr.min(2).astype(np.float32)
    out = np.zeros_like(w)
    for s in (1.5, 2.5, 4.0):
        g = cv2.GaussianBlur(w, (0, 0), s)
        xx, yy, xy = (cv2.Sobel(g, cv2.CV_32F, a, b, ksize=3) for a, b in ((2, 0), (0, 2), (1, 1)))
        d = np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
        l1, l2 = (xx + yy) / 2 - d, (xx + yy) / 2 + d                 # l1 <= l2; a bright ridge: l1 << 0, l2 ~ 0
        out = np.maximum(out, s * s * np.maximum(np.maximum(-l1, 0) - 1.5 * np.abs(l2), 0))
    return out


def paint_mask(bgr):
    """Paint for the COARSE search: line-shaped, inside the pitch's own grass region, in long thin
    strokes. -> (mask, share of the frame that is pitch). n31_linereg's ridge map is for sub-pixel
    search beside a camera that is already close; as a mask it fires on grass texture (61% of the
    frame) and every camera scores zero.
    ponytail: the pitch REGION is found by greenness, so a night, frosted or desaturated pitch
    (117092) has none and is refused upstream, not guessed at. Upgrade: a learned pitch segmenter."""
    sm = cv2.GaussianBlur(bgr, (0, 0), 1.2).astype(np.float32)
    green = sm[:, :, 1] - np.maximum(sm[:, :, 0], sm[:, :, 2])
    region = cv2.morphologyEx((green > 8).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(region)
    if n < 2:
        return np.zeros(bgr.shape[:2], bool), 0.0
    pitch = cv2.dilate((lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ln = lineness(bgr)
    m = ((ln >= max(float(np.percentile(ln[pitch], PAINT_PCT)), 5.0)) & pitch).astype(np.uint8)   # the floor: a frame with no texture has a zero percentile
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    w, h, a = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    keep = []
    for k in range(1, n):
        if max(w[k], h[k]) < PAINT_LEN or a[k] > 0.35 * w[k] * h[k] + 200:                                 # long, and thin in its box
            continue
        ys, xs = np.nonzero(lab[st[k, 1]:st[k, 1] + h[k], st[k, 0]:st[k, 0] + w[k]] == k)
        ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(float)))
        if ev[0] <= STRAIGHT * ev[1]:                # a line or a shallow arc, not a letter: pitch-side lettering is long strokes too
            keep.append(k)
    mask = np.isin(lab, keep)
    # The near touchline is the longest stroke in a side-on panorama. Whatever is painted BELOW it is
    # off the pitch - a club's name in letters two metres tall - and drags the search toward a camera
    # lying on the grass, for which those letters sit conveniently close to the touchline.
    if keep:
        k = max(keep, key=lambda i: w[i])
        if w[k] >= 0.5 * bgr.shape[1]:
            ys, xs = np.nonzero(lab == k)
            row = np.full(bgr.shape[1], -1)
            np.maximum.at(row, xs, ys)
            cols = np.flatnonzero(row >= 0)
            edge = np.interp(np.arange(bgr.shape[1]), cols, row[cols])
            mask &= np.arange(bgr.shape[0])[:, None] <= edge[None, :] + 4
    return mask, float(pitch.mean())


def paint_dt(bgr, scale=4, trunc=None):
    """Truncated distance (full-res px) to the nearest paint, at 1/scale. -> (dt, scale, pitch share)."""
    paint, share = paint_mask(bgr)
    # the transform is taken at FULL resolution and then thinned: shrinking the mask first averages a
    # 1-2 px stroke away, and the search then scores every camera against an empty pitch
    dt = cv2.distanceTransform((~paint).astype(np.uint8), cv2.DIST_L2, 3)[::scale, ::scale]
    return np.minimum(dt, trunc or TRUNC).astype(np.float32), scale, share


def chamfer(q, X, dt, scale, wh, paint_uv=None, tree=None, trunc=None):
    """Two-way, each half in [0, 1]. DRAWING -> PAINT in pixels: is every line of the pitch on paint?
    PAINT -> DRAWING in METRES, through the camera's inverse: is the paint on the pitch explained by
    a line? One way alone is won by a camera lying on the grass, which folds the whole drawing onto
    one painted stroke; it cannot also explain paint spread over half the frame."""
    p = _fwd(q, X, float(wh[0]))
    ok = np.isfinite(p).all(1) & (p[:, 0] >= 0) & (p[:, 0] < wh[0] - 1) & (p[:, 1] >= 0) & (p[:, 1] < wh[1] - 1)
    trunc = trunc or TRUNC
    cost = np.full(len(X), trunc, np.float32)
    if ok.any():
        cost[ok] = map_coordinates(dt, [p[ok, 1] / scale, p[ok, 0] / scale], order=1, mode="nearest")
    a = float(cost.mean()) / trunc
    if paint_uv is None:
        return a
    g = _inv(q, paint_uv, float(wh[0]))
    d = np.full(len(g), TRUNC_M)
    fin = np.isfinite(g).all(1)
    if fin.any():
        d[fin] = np.minimum(tree.query(g[fin])[0], TRUNC_M)
    return a + float(d.mean()) / TRUNC_M


def autocal(bgr, P, verbose=False, fix_length=False):
    """-> (camera or None, report). No dots."""
    wh = (bgr.shape[1], bgr.shape[0])
    from scipy.spatial import cKDTree  # noqa: PLC0415 - scipy.spatial pulls in a lot; only the search needs it
    dt, scale, share = paint_dt(bgr)
    if share < MIN_PITCH_SHARE:
        return None, {"accepted": False, "why": f"grass region is {share:.2f} of the frame: not a daylight green pitch", "hits": 0, "samples": 0}
    X, _T, grp = model_samples(P, 1.0)
    tree = cKDTree(model_samples(P, 0.25)[0])
    vv, uu = np.nonzero(paint_mask(bgr)[0])
    pick = np.random.default_rng(0).choice(len(uu), min(N_PAINT, len(uu)), replace=False)
    paint_uv = np.column_stack([uu[pick], vv[pick]]).astype(float)
    keys = list(GRID)
    names = keys + ["roll_deg", "cv_off"]
    clip = lambda v: {k: float(np.clip(x, *BOUNDS[k])) for k, x in zip(names, v)}  # noqa: E731
    score = lambda kw: chamfer(camera(P, wh, **kw), X, dt, scale, wh, paint_uv, tree)  # noqa: E731
    # the grid is coarse, so it is scored against a WIDE basin: a drawing 100 px from its paint must still
    # be told from one 300 px away, or the node beside the truth loses to a stranger
    dt_wide = paint_dt(bgr, trunc=TRUNC_WIDE)[0]
    wide = lambda kw: chamfer(camera(P, wh, **kw), X, dt_wide, scale, wh, paint_uv, tree, trunc=TRUNC_WIDE)  # noqa: E731
    scored = sorted(((wide(dict(zip(keys, vals))), dict(zip(keys, vals))) for vals in itertools.product(*GRID.values())),
                    key=lambda s: s[0])
    polished = []
    for c0, kw in scored[:TOP_K]:
        r = minimize(lambda v: wide(clip(v)), np.array([kw[k] for k in keys] + [0.0, 0.0]), method="Nelder-Mead",
                     options={"xatol": 0.1, "fatol": 1e-4, "maxfev": 300})
        r = minimize(lambda v: score(clip(v)), r.x, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-4, "maxfev": 400})
        polished.append((float(r.fun), c0, clip(r.x)))
    polished.sort(key=lambda s: s[0])
    ridge, grass = ridge_map(bgr), grass_map(bgr)
    empty = np.zeros((0, 2))
    best, tried = None, []
    for pol, c0, kw in polished[:TOP_REG]:
        cam, rep, Pf = _settle(ridge, grass, camera(P, wh, **kw), P, fix_length=fix_length)
        ok, C = accept(cam, rep)
        tried.append({"coarse": round(c0, 3), "polished": round(pol, 3), "hits": rep["hits"], "line_rms_px": round(rep["line_rms_px"], 3),
                      "height_m": round(float(abs(C[2])), 2), "accepted": bool(ok)})
        if verbose:
            print("   ", tried[-1], flush=True)
        key = (ok, rep["hits"] // 50, -rep["line_rms_px"])
        if best is None or key > best[0]:
            best = (key, cam, rep, C, Pf)
    _key, cam, rep, C, Pf = best
    report = {"accepted": bool(_key[0]), "hits": rep["hits"], "samples": rep["samples"], "line_rms_px": rep["line_rms_px"],
              "camera_centre_m": [float(v) for v in C], "f_px_per_rad": float(abs(cam[0])), "hits_by_group": rep["hits_by_group"],
              "length_m": Pf.length, "length_sd_m": rep.get("length_sd_m"), "length_measured_m": rep.get("length_measured_m"),
              "length_measured_sd_m": rep.get("length_measured_sd_m"), "candidates": tried, "grid": len(scored),
              "why": why_refused(cam, rep)}
    return (cam if _key[0] else None), report


FAMILIES = {"touch", "goal", "halfway", "circle"}
PEN_FRONTS, PEN_FRONT_SHARE = ("pen_l_front", "pen_r_front"), 0.5


def _missing(rep):
    """The drawing's parts not found on paint. BOTH penalty fronts are required, each at half its samples, not
    "any pen group at 25%": the length scan can land the right mount under a drawing 4 m long, and that keeps one
    front while losing the other (N68, n68-128057-length (not published); docs/reference/calibration.md#known-limits)."""
    g = rep["hits_by_group"]
    fam = {k.split("_")[0] for k, (h, n) in g.items() if h >= 0.25 * n}
    return sorted(FAMILIES - fam) + [k for k in PEN_FRONTS if g[k][0] < PEN_FRONT_SHARE * g[k][1]]


def accept(cam, rep):
    C = -_rot(cam[4:7]).T @ cam[7:10]
    return bool(rep["refused"] is None and 2.0 <= abs(C[2]) <= 20.0 and not _missing(rep)), C


def why_refused(cam, rep):
    """The reason `accept` said no, for the person who has to click instead. None when accepted."""
    ok, C = accept(cam, rep)
    if ok:
        return None
    if rep["refused"]:
        return f"too little of the pitch drawing found on paint: {rep['refused']} (needs {MIN_HITS} hits under {MAX_LINE_RMS_PX} px)"
    if not 2.0 <= abs(C[2]) <= 20.0:
        return f"the camera would be {abs(C[2]):.1f} m above the pitch, which no rig is"
    return f"these parts of the drawing were not found on paint: {', '.join(_missing(rep))}"


def from_install(bgr, P, start, fix_length=False):
    """A match from an install that has been solved before starts from THAT camera: the rig has not
    moved, or has moved by a wobble the 25 px search absorbs. Same registration, same acceptance."""
    cam, rep, Pf = _settle(ridge_map(bgr), grass_map(bgr), np.asarray(start, float), P, fix_length=fix_length)
    ok, C = accept(cam, rep)
    return (cam if ok else None), {"accepted": ok, "via": "install", "hits": rep["hits"], "samples": rep["samples"],
                                   "line_rms_px": rep["line_rms_px"], "camera_centre_m": [float(v) for v in C],
                                   "f_px_per_rad": float(abs(cam[0])), "hits_by_group": rep["hits_by_group"],
                                   "length_m": Pf.length, "length_sd_m": rep.get("length_sd_m"), "why": why_refused(cam, rep),
                                   "length_measured_m": rep.get("length_measured_m"), "length_measured_sd_m": rep.get("length_measured_sd_m")}


def calibrate(video, pitch, install_camera=None, n_frames=40, fix_length=False):
    """The product's one call. -> (camera or None, report). `install_camera` is the camera of an
    earlier match from the same rig, tried first; the search from nothing runs when there is none or
    it no longer fits the paint. `fix_length`: fit at pitch.length, measure the paint's for the record."""
    bgr = median_frame(video, n_frames)
    if install_camera is not None:
        cam, rep = from_install(bgr, pitch, install_camera, fix_length=fix_length)
        if cam is not None:
            return cam, {**rep, "image_wh": [int(bgr.shape[1]), int(bgr.shape[0])]}
    cam, rep = autocal(bgr, pitch, fix_length=fix_length)
    return cam, {**rep, "via": rep.get("via", "search"), "image_wh": [int(bgr.shape[1]), int(bgr.shape[0])]}


def _demo_boards(P, true, wh):
    """A faint far touchline with a bright board 3 px beyond it and no grass past the board: the old perimeter
    rule takes the board, `native` takes the paint (N36)."""
    far = _fwd(true, np.column_stack([np.arange(0, 105, 0.02), np.zeros(5250)]), float(wh[0]))
    img = np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8)
    for u, v in far:
        img[:int(round(v)) - 4, int(round(u))] = (90, 90, 95)             # stand
        img[int(round(v)) - 6:int(round(v)) - 2, int(round(u))] = (250, 250, 250)   # board, centred 4 px beyond the line
        img[int(round(v)), int(round(u))] = (150, 190, 150)                 # 1 px of far paint
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    X, T, grp = model_samples(P)
    m = grp == "touch_far"
    out = np.array([OUTWARD[g] for g in grp[m]])
    off = {}
    for name, nat in (("old", None), ("native", native_map(img))):
        hit, d, nrm = search(ridge_map(img), true, X[m], T[m], 8.0, grass=grass_map(img), outward=out, native=nat)
        off[name] = float(np.median(((d - _fwd(true, X[m])) * nrm).sum(1)[hit])) if hit.any() else None
    assert off["native"] is not None and abs(off["native"]) < 0.5, off
    assert off["old"] is None or off["old"] > 2.0, off                       # outward: on the board, or nothing


def _demo():
    """A pitch rendered through a mount the search grid does not contain; the camera must come back
    with no dots, a frame with no pitch must be refused, and so must a grey one."""
    from pitch import Pitch  # noqa: PLC0415
    P = Pitch(105.0, 68.0)
    wh = (4096, 1080)
    true = camera(P, wh, fov_deg=156.0, tilt_deg=19.5, height_m=7.3, behind_m=16.0, along_m=-2.5, pan_deg=1.7, roll_deg=0.4, cv_off=-25.0)
    img = np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8)
    for u, v in _fwd(true, model_samples(P, 0.05)[0], float(wh[0])):
        if np.isfinite(u) and 2 <= u < wh[0] - 2 and 2 <= v < wh[1] - 2:
            img[int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = (235, 235, 235)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    cam, rep = autocal(img, P)
    assert cam is not None and rep["accepted"], rep
    grid = np.stack(np.meshgrid(np.linspace(5, 100, 12), np.linspace(3, 65, 9)), -1).reshape(-1, 2)
    err = np.linalg.norm(_inv(cam, _fwd(true, grid, float(wh[0])), float(wh[0])) - grid, axis=1)
    assert np.median(err) < 0.1 and err.max() < 0.5, (np.median(err), err.max())
    # a pitch that is not 105 m long says so: drawn at 110.5 x 68, searched as 105 x 68
    import dataclasses  # noqa: PLC0415
    long_p = dataclasses.replace(P, length=110.5)
    img2 = np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8)
    for u, v in _fwd(true, model_samples(long_p, 0.05)[0] - [2.75, 0.0], float(wh[0])):      # same centre spot
        if np.isfinite(u) and 2 <= u < wh[0] - 2 and 2 <= v < wh[1] - 2:
            img2[int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = (235, 235, 235)
    cam2, rep2b = autocal(cv2.GaussianBlur(img2, (0, 0), 0.8), P)
    assert cam2 is not None and abs(rep2b["length_m"] - 110.5) < 0.3, (rep2b.get("length_m"), rep2b.get("accepted"))
    assert abs(rep["length_m"] - 105.0) < 0.3, rep["length_m"]
    # a GIVEN length wins: the 110.5 m paint fitted at a typed 105, the paint's reading kept for the record
    _c, rep2f = from_install(cv2.GaussianBlur(img2, (0, 0), 0.8), P, true, fix_length=True)
    assert rep2f["length_m"] == 105.0 and abs(rep2f["length_measured_m"] - 110.5) < 0.3, rep2f
    # a rig that has been solved before starts from its own camera, nudged
    near, rep1 = from_install(img, P, true + np.r_[0, 0, 6.0, -4.0, 0.004, -0.003, 0.002, 0, 0, 0])
    assert near is not None and rep1["via"] == "install", rep1
    none, rep2 = autocal(np.full((wh[1], wh[0], 3), (40, 110, 40), np.uint8), P)
    assert none is None and not rep2["accepted"], rep2
    night, rep3 = autocal(np.full((wh[1], wh[0], 3), (60, 66, 62), np.uint8), P)
    assert night is None and "grass region" in rep3["why"], rep3
    _demo_boards(P, true, wh)
    # inverse undoes forward, warped or not, and a ray above the horizon meets no pitch
    warped = np.r_[true, 0.5 * np.random.default_rng(1).normal(size=2 * (WARP_DEG + 1))]
    for q in (true, warped):
        assert np.abs(_inv(q, _fwd(q, grid, 4096.0), 4096.0) - grid).max() < 1e-6
    global VERT_CUBIC
    VERT_CUBIC = -0.4
    assert np.abs(_inv(true, _fwd(true, grid, 4096.0), 4096.0) - grid).max() < 1e-6
    VERT_CUBIC = 0.0
    assert np.isnan(_inv(true, np.array([[2048.0, -300.0]]), 4096.0)).all()      # well above this camera's horizon
    print(f"autocal ok: no dots, median map error {np.median(err):.3f} m (max {err.max():.2f}); {rep['hits']} of {rep['samples']} samples on paint; "
          f"a 110.5 m pitch read as {rep2b['length_m']:.2f} m, a 105 m one as {rep['length_m']:.2f} m")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from pitch import Pitch  # noqa: PLC0415
        c, r = calibrate(sys.argv[1], Pitch())
        print({k: r.get(k) for k in ("accepted", "via", "why", "hits", "samples", "line_rms_px", "camera_centre_m", "f_px_per_rad")})
    else:
        _demo()
