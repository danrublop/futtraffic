"""EXPLORATORY: which held-out family pays for the column warp, per clip, and does a lower-degree warp (3, not 5) stop it?
Starts from n36_after.json (paint rule, rigid). n = 2 warp degrees; both on clips already seen - this chooses a candidate, it confirms nothing."""
import json, sys
import cv2, numpy as np
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A
from pitch import Pitch
import n36_far_touchline as N
base = json.load(open("model/validation/out/stc2025/n36_after.json"))
e = np.zeros((0, 2))
for deg in (5, 3):
    A.WARP_DEG = deg
    for q in N.SEQS:
        bgr = cv2.imread(f"model/validation/out/stc2025/{q}.median40.png")
        rid, gr, nat = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr)
        P = Pitch(base[q]["length_m"], 68.0)
        cam, _rep = A.register(rid, np.array(base[q]["camera"]), P, e, e, grass=gr, warp=True, square=1.0, radii=A.AUTO_RADII[2:], native=nat)
        row = {}
        for fam in N.INTERIOR:
            for name, start in (("rigid", np.array(base[q]["camera"])), ("warp", cam)):
                _c, rep = A.register(rid, start, P, e, e, grass=gr, square=1.0, radii=A.AUTO_RADII[1:], leave_out=fam, native=nat)
                row[f"{fam[0]}:{name}"] = round(rep["held_out"]["median_m"], 2) if rep.get("held_out") else None
        print(f"deg {deg} {q} T {N.termination(bgr, cam, P.length):+.2f}", row, flush=True)
