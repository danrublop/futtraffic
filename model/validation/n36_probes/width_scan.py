"""EXPLORATORY, after N36's test failed: with the far touchline's paint now found, does the paint say how WIDE the pitch is?"""
import dataclasses, json, sys
import cv2, numpy as np
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A
from pitch import Pitch
from n36_far_touchline import termination
gold = json.load(open("model/validation/out/stc2025/n36_after.json"))
e = np.zeros((0, 2))
for q in sys.argv[1:]:
    bgr = cv2.imread(f"model/validation/out/stc2025/{q}.median40.png")
    rid, gr, nat = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr)
    P0 = Pitch(gold[q]["length_m"], 68.0)
    q0 = np.array(gold[q]["camera"])
    for W in np.arange(64.0, 70.01, 0.5):
        P = dataclasses.replace(P0, width=float(W))
        c = q0.copy(); c[7:10] -= A._rot(c[4:7])[:, 1] * (W - 68.0) / 2.0          # same centre spot
        for _ in range(2):
            c, rep = A.register(rid, c, P, e, e, grass=gr, square=1.0, radii=A.AUTO_RADII[2:], native=nat)
        X, T, grp = A.model_samples(P)
        out = np.array([A.OUTWARD.get(g, (0., 0.)) for g in grp])
        hit, d, nrm = A.search(rid, c, X, T, 8.0, grass=gr, outward=out, native=nat)
        r = ((A._fwd(c, X) - d) * nrm).sum(1)
        per = {g: round(float(np.median(r[hit & (grp == g)])), 2) for g in ("touch_far", "touch_near", "halfway", "circle", "pen_l_far", "pen_l_near")if (hit & (grp == g)).any()}
        print(q, W, "hits", rep["hits"], "rms", round(rep["line_rms_px"], 3), "rms@8px", round(float(np.sqrt((r[hit] ** 2).mean())), 3),
              "T", round(termination(bgr, c, P.length), 2), "h", round(float(A.Camera(c).q[0]), 0), per, flush=True)
