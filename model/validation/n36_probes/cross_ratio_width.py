"""EXPLORATORY: the pitch's WIDTH from four points on the halfway line, almost without a camera.

The halfway line's two ends and its two crossings of the centre circle are collinear in the world. Rays from the
camera to collinear points lie in one plane, and position along the line is y = y0 + D tan(theta - theta0) in the
in-plane ray angle theta. The crossings are 9.15 m either side of the centre (regulation), the ends W/2 either
side: four rows measured, four unknowns (y0, D, theta0, W). Only the INTRINSICS of the fitted camera are used
(px -> ray); its position, height and tilt - the things width trades against in a registration - never enter.
"""
import json, sys
import cv2, numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A

cams = json.load(open("model/validation/out/stc2025/n34_autocal.json"))
DU = np.arange(-3, 3.01, 0.25)


def contrast(mn, p):
    prof = map_coordinates(mn, [np.repeat(p[:, 1:2], len(DU), 1).ravel(), (p[:, 0:1] + DU[None]).ravel()], order=1).reshape(len(p), len(DU))
    return prof.max(1) - 0.5 * (prof[:, 0] + prof[:, -1])


def line_end(mn, q, L, ys, ref_mask):
    """Walk the halfway line along `ys` (metres, inside -> outside); -> (u, v) where its contrast halves."""
    p = A._fwd(q, np.column_stack([np.full_like(ys, L / 2), ys]), float(mn.shape[1]))
    c = contrast(mn, p)
    ref = np.median(c[ref_mask])
    k = np.flatnonzero((c < 0.5 * ref) & ~ref_mask)[0]
    t = (0.5 * ref - c[k]) / (c[k - 1] - c[k])
    return p[k] + t * (p[k - 1] - p[k])


def circle_cross(mn, q, L, W, sign):
    """Row where the circle crosses the halfway line: the circle's ridge row at columns either side, quadratic to du = 0."""
    p0 = A._fwd(q, np.array([[L / 2, W / 2 + sign * 9.15]]), float(mn.shape[1]))[0]
    dus, rows = [], []
    dv = np.arange(-6, 6.01, 0.25)
    for du in np.r_[np.arange(-16, -5), np.arange(6, 17)]:
        th = np.arcsin(np.clip(du * 0.0, -1, 1))  # noqa: F841  (kept simple: search each column about the camera's own circle row)
        ang = np.linspace(-0.6, 0.6, 241) + (np.pi / 2 if sign > 0 else -np.pi / 2)
        arc = A._fwd(q, np.column_stack([L / 2 + 9.15 * np.cos(ang), W / 2 + 9.15 * np.sin(ang)]), float(mn.shape[1]))
        v0 = np.interp(p0[0] + du, *((arc[:, 0], arc[:, 1]) if arc[0, 0] < arc[-1, 0] else (arc[::-1, 0], arc[::-1, 1])))
        prof = map_coordinates(mn, [v0 + dv, np.full_like(dv, p0[0] + du)], order=1)
        j = int(prof.argmax())
        if 0 < j < len(dv) - 1 and prof[j] - np.median(prof) >= 6:
            a, b, c = prof[j - 1:j + 2]
            dus.append(du); rows.append(v0 + dv[j] + 0.25 * 0.5 * (a - c) / (a - 2 * b + c - 1e-9))
    co = np.polyfit(dus, rows, 2)
    return np.array([p0[0], co[2]]), len(dus), float(np.std(np.polyval(co, dus) - rows))


def ray(q, uv):
    a, e = (uv[0] - q[2]) / q[0], (uv[1] - q[3]) / q[1]
    return np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])


def width(q, pts, fv_scale=1.0):
    q = np.array(q, float); q[1] *= fv_scale
    r = [ray(q, p) for p in pts]                                   # far end, far crossing, near crossing, near end
    th = np.array([np.arctan2(np.linalg.norm(np.cross(r[0], ri)), r[0] @ ri) for ri in r])
    f = lambda v: v[0] + v[1] * np.tan(th - v[2]) - np.array([-v[3] / 2, -9.15, 9.15, v[3] / 2])  # noqa: E731
    return least_squares(f, [0.0, 60.0, 0.8, 68.0], x_scale=[1, 10, 0.1, 1]).x[3]


for seq in sorted(cams):
    if not cams[seq].get("camera"):
        continue
    q = np.array(cams[seq]["camera"]); L = cams[seq].get("length_m") or 105.0
    mn = cv2.GaussianBlur(cv2.imread(f"model/validation/out/stc2025/{seq}.median40.png").astype(np.float32), (0, 0), 0.7).min(2)
    yf, yn = np.arange(12, -6, -0.05), np.arange(62, 72, 0.02)
    far, near = line_end(mn, q, L, yf, yf >= 3), line_end(mn, q, L, yn, yn <= 66)
    (cf, nf, sf), (cn, nn, sn) = circle_cross(mn, q, L, 68.0, -1), circle_cross(mn, q, L, 68.0, +1)
    pts = [far, cf, cn, near]
    W = width(q, pts)
    sens = {f"{nm}+1px": round(width(q, [p + (np.array([0, 1.0]) if i == k else 0) for i, p in enumerate(pts)]) - W, 2) for k, nm in enumerate(("far_end", "far_x", "near_x", "near_end"))}
    print(f"{seq}  rows far_end {far[1]:.2f} far_x {cf[1]:.2f} ({nf} cols, sd {sf:.2f}) near_x {cn[1]:.2f} ({nn}, sd {sn:.2f}) near_end {near[1]:.2f}  ->  WIDTH {W:.2f} m"
          f"  | fv +-2%: {width(q, pts, 0.98):.2f} / {width(q, pts, 1.02):.2f}  | {sens}")


def touch_cross(img, q, L, y_line, W_model):
    """Row of a TOUCHLINE's centre at the halfway line's column: the strongest grass-flanked thin ridge at columns
    either side (A.search's N36 rule, by hand), quadratic to du = 0. No half-contrast walk, so no inward bias."""
    mn = img.min(2)
    gr = img[:, :, 1] - np.maximum(img[:, :, 0], img[:, :, 2])
    xs = np.linspace(L / 2 - 12, L / 2 + 12, 481)
    ln = A._fwd(q, np.column_stack([xs, np.full_like(xs, y_line)]), float(mn.shape[1]))
    u0 = A._fwd(q, np.array([[L / 2, y_line]]), float(mn.shape[1]))[0, 0]
    dv = np.arange(-9, 9.01, 0.25)
    dus, rows = [], []
    for du in np.r_[np.arange(-40, -5, 2), np.arange(6, 41, 2)]:
        v0 = np.interp(u0 + du, ln[:, 0], ln[:, 1])
        P = map_coordinates(mn, [v0 + dv, np.full_like(dv, u0 + du)], order=1)
        G = map_coordinates(gr, [v0 + dv, np.full_like(dv, u0 + du)], order=1)
        best, row = 6.0, None
        for j in range(1, len(dv) - 1):
            for f in (8, 16):
                if f <= j < len(dv) - f and P[j] >= P[j - 1] and P[j] > P[j + 1] and G[j - f] > 5 and G[j + f] > 5 and P[j] - 0.5 * (P[j - f] + P[j + f]) >= best:
                    best = P[j] - 0.5 * (P[j - f] + P[j + f])
                    a, b, c = P[j - 1:j + 2]
                    row = v0 + dv[j] + 0.25 * 0.5 * (a - c) / (a - 2 * b + c - 1e-9)
        if row is not None:
            dus.append(du); rows.append(row)
    co = np.polyfit(dus, rows, 2)
    res = np.polyval(co, dus) - rows
    keep = np.abs(res - np.median(res)) < 1.0
    co = np.polyfit(np.array(dus)[keep], np.array(rows)[keep], 2)
    return np.array([u0, co[2]]), int(keep.sum()), float(np.std((np.polyval(co, dus) - rows)[keep]))


print("\nthe same, with each END taken as the touchline's own centre at the halfway line's column (no half-contrast walk)")
Ws = []
for seq in sorted(cams):
    if not cams[seq].get("camera"):
        continue
    q = np.array(cams[seq]["camera"]); L = cams[seq].get("length_m") or 105.0
    img = cv2.GaussianBlur(cv2.imread(f"model/validation/out/stc2025/{seq}.median40.png").astype(np.float32), (0, 0), 0.7)
    mn = img.min(2)
    yf = np.arange(12, -6, -0.05)
    walk = line_end(mn, q, L, yf, yf >= 3)
    far, n1, s1 = touch_cross(img, q, L, 0.0, 68.0)
    yn = np.arange(62, 72, 0.02)
    near, n2, s2 = line_end(mn, q, L, yn, yn <= 66), 0, 0.0      # the near line is ~5 px wide and a metre of width is 50 px there: the walk will do
    (cf, _, _), (cn, _, _) = circle_cross(mn, q, L, 68.0, -1), circle_cross(mn, q, L, 68.0, +1)
    W = width(q, [far, cf, cn, near]); Ws.append(W)
    cam_row = A._fwd(q, np.array([[L / 2, 0.0]]), float(mn.shape[1]))[0, 1]
    print(f"{seq}  far touchline centre row {far[1]:.2f} ({n1} cols, sd {s1:.2f}); half-contrast walk ended at {walk[1]:.2f} = {walk[1] - far[1]:+.2f} px inside it;"
          f" n34 camera's far touchline {cam_row:.2f} = {far[1] - cam_row:+.2f} px | near centre {near[1]:.2f} ({n2}, sd {s2:.2f})  ->  WIDTH {W:.2f} m")
print(f"WIDTH median {np.median(Ws):.2f} m, from {min(Ws):.2f} to {max(Ws):.2f}, n=7 clips of ONE pitch and ONE camera (not seven independent measurements)")
