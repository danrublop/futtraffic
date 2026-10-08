"""EXPLORATORY: is the panorama's ROW linear in elevation? v = cv + fv (e + kappa e^3). kappa = 0 is the equirect
the product assumes; 1/3 is (to third order) a CYLINDRICAL panorama, v = fv tan e; negative compresses like a fisheye.
A 67 m pitch made the registration worse and the pitch is flat, so what is left to explain far-touchline paint
2-4 px inside the rigid camera's is the projection's own vertical law."""
import json, sys
import cv2, numpy as np
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A
from pitch import Pitch
from n36_far_touchline import termination

KAPPA = [0.0]


def fwd(q, xy, width=4096.0):
    xy = np.asarray(xy, float)
    if xy.shape[1] == 2:
        xy = np.column_stack([xy, np.zeros(len(xy))])
    X = xy @ A._rot(q[4:7]).T + q[7:10]
    e = np.arctan2(X[:, 1], np.hypot(X[:, 0], X[:, 2]))
    return np.column_stack([q[2] + q[0] * np.arctan2(X[:, 0], X[:, 2]), q[3] + q[1] * (e + KAPPA[0] * e ** 3)])


A._fwd = fwd
gold = json.load(open("model/validation/out/stc2025/n36_after.json"))
emp = np.zeros((0, 2))
for seq in sys.argv[1:]:
    bgr = cv2.imread(f"model/validation/out/stc2025/{seq}.median40.png")
    rid, gr, nat = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr)
    P = Pitch(gold[seq]["length_m"], 68.0)
    X, T, grp = A.model_samples(P)
    out = np.array([A.OUTWARD.get(g, (0., 0.)) for g in grp])
    for kap in (-0.6, -0.4, -0.2, 0.0, 0.2, 1 / 3, 0.5, 0.7, 1.0):
        KAPPA[0] = kap
        c = np.array(gold[seq]["camera"])
        for _ in range(3):
            c, rep = A.register(rid, c, P, emp, emp, grass=gr, square=1.0, radii=A.AUTO_RADII[2:], native=nat, leave_out=())
        hit, d, nrm = A.search(rid, c, X, T, 8.0, grass=gr, outward=out, native=nat)
        r = ((fwd(c, X) - d) * nrm).sum(1)
        per = {g: round(float(np.median(r[hit & (grp == g)])), 2) for g in ("touch_far", "touch_near", "circle", "pen_l_far", "pen_r_far", "pen_l_near")}
        print(f"{seq} kappa {kap:+.2f}  hits {rep['hits']}  rms {rep['line_rms_px']:.3f}  rms@8 {np.sqrt((r[hit] ** 2).mean()):.3f}  T {termination(bgr, c, P.length):+.2f}  fu {c[0]:.0f} fv {c[1]:.0f}  {per}", flush=True)
