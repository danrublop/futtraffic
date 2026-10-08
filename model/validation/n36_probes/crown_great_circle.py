"""EXPLORATORY: is the pitch FLAT? A straight line in the world images to a great circle (its rays lie in one
plane) whatever the camera's position. A line crossing a CROWNED pitch arches out of that plane. Intrinsics only
for px -> ray; the fitted camera is used for association and to convert the arch from angle to metres."""
import json, sys
import cv2, numpy as np
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A
from pitch import Pitch

cams = json.load(open("model/validation/out/stc2025/n36_after.json"))
LINES = {"goal_l": "x=0", "goal_r": "x=L", "pen_l_front": "x=16.5", "pen_r_front": "x=L-16.5", "halfway": "x=L/2 (control: in the camera's own vertical plane, a crown cannot show)"}


def rays(q, uv):
    a, e = (uv[:, 0] - q[2]) / q[0], (uv[:, 1] - q[3]) / q[1]
    return np.column_stack([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])


for seq in [s for s in sorted(cams) if not s.startswith("_")]:
    q = np.array(cams[seq]["camera"]); P = Pitch(cams[seq]["length_m"], 68.0)
    bgr = cv2.imread(f"model/validation/out/stc2025/{seq}.median40.png")
    X, T, grp = A.model_samples(P, 0.25)
    out = np.array([A.OUTWARD.get(g, (0., 0.)) for g in grp])
    hit, d, _n = A.search(A.ridge_map(bgr), q, X, T, 8.0, grass=A.grass_map(bgr), outward=out, native=A.native_map(bgr))
    row = [seq]
    for g in LINES:
        m = hit & (grp == g) & (np.abs(X[:, 1] - 34.0) > (6.0 if g.startswith("goal") else 0.0))     # not the goal mouth
        r = rays(q, d[m])
        n = np.linalg.svd(r - 0 * r.mean(0), full_matrices=False)[2][-1]                                # plane through the camera
        alpha = np.arcsin(r @ n)                                                                        # out-of-plane angle, rad
        X3 = np.column_stack([X[m], np.zeros(m.sum())])
        up = rays(q, A._fwd(q, X3 + [0, 0, 0.1])) - rays(q, A._fwd(q, X3))
        dadz = (up @ n) / 0.1                                                                           # rad of alpha per metre of height
        y = X[m, 1] - 34.0
        Amat = np.column_stack([np.ones(m.sum()), y, -(y / 34.0) ** 2 * dadz])                          # crown z = -c (y/34)^2: c = drop at the touchline
        co, *_ = np.linalg.lstsq(Amat, alpha, rcond=None)
        res0 = alpha - np.column_stack([np.ones(m.sum()), y]) @ np.linalg.lstsq(Amat[:, :2], alpha, rcond=None)[0]
        res1 = alpha - Amat @ co
        row.append(f"{g}: n={m.sum():3d} crown {co[2]:+.3f} m  arch rms {1513 * res0.std():.2f}->{1513 * res1.std():.2f} px")
    print("  |  ".join(row), flush=True)
