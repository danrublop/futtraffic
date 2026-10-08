"""EXPLORATORY sweep, n=4 values of WARP_RIDGE: does shrinking the column warp harder stop it wandering into held-out paint
while keeping the far touchline on the paint? Starts from n36_after.json (native rule, rigid), so only the warp differs."""
import json, sys
import cv2, numpy as np
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A
from pitch import Pitch
import n36_far_touchline as N
base = json.load(open("model/validation/out/stc2025/n36_after.json"))
before = json.load(open("model/validation/out/stc2025/n36_before.json"))["_summary"]["P_mean_m"]
e = np.zeros((0, 2))
for ridge in (0.05, 0.2, 0.5, 2.0):
    A.WARP_RIDGE = ridge
    cams, T, Pm = {}, {}, {}
    for q in N.SEQS:
        bgr = cv2.imread(f"model/validation/out/stc2025/{q}.median40.png")
        P = Pitch(base[q]["length_m"], 68.0)
        cam, rep = A.register(A.ridge_map(bgr), np.array(base[q]["camera"]), P, e, e, grass=A.grass_map(bgr), warp=True, square=1.0,
                              radii=A.AUTO_RADII[2:], native=A.native_map(bgr))
        cams[q], T[q], Pm[q] = cam, N.termination(bgr, cam, P.length), N.interior_held_out(bgr, cam, P, True)
    refs = N.far_refs(cams)
    print(f"ridge {ridge}: T {[round(T[q], 2) for q in N.SEQS]} within1px {sum(abs(v) <= 1 for v in T.values())}/7 | N median {np.median(refs):+.2f} {np.round(refs, 2).tolist()} |"
          f" P {[round(Pm[q], 3) for q in N.SEQS]} ok {sum(Pm[q] <= before[q] + 0.05 for q in N.SEQS)}/7", flush=True)
    json.dump({q: cams[q].tolist() for q in N.SEQS}, open(f"model/validation/out/stc2025/n36_cams_ridge{ridge}.json", "w"))
