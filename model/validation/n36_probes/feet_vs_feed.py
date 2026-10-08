"""EXPLORATORY: MOT box feet through three cameras against ST2's feed, split ALONG and ACROSS the camera ray.
Frames f % 20 == 10 (N31's development frames - not held out). The feed itself is suspected of a depth scale
(N32), so ACROSS is the fair column; ALONG is reported as a signed % of range."""
import json, sys
import numpy as np
sys.path[:0] = ["model/engine", "model/validation"]
import autocal as A
from n31_score import feet_pixels

STC = "model/validation/out/stc2025/"
ARMS = {"n34 (ships)": "n34_autocal.json", "native": "n36_after.json", "native+warp": "n36_after_warp.json"}
cams = {a: json.load(open(STC + f)) for a, f in ARMS.items()}
pool = {a: [] for a in ARMS}
for q in ("117093", "118575", "118576", "118577", "118578", "128057", "128058"):
    uv, G = feet_pixels(q)
    for a in ARMS:
        c = np.array(cams[a][q]["camera"])
        est = A._inv(c, uv)
        C = (-A._rot(c[4:7]).T @ c[7:10])[:2]
        ray = G - C; rng = np.linalg.norm(ray, axis=1); ray /= rng[:, None]
        d = est - G
        ok = np.isfinite(d).all(1)
        al, ac = (d * ray).sum(1), d[:, 0] * -ray[:, 1] + d[:, 1] * ray[:, 0]
        pool[a].append(np.column_stack([np.linalg.norm(d, axis=1), al, ac, rng, G[:, 1]])[ok])
print(f"{'arm':14s} {'n':>6s} {'median m':>9s} {'far-half':>9s} {'p90':>6s} | {'|across| med':>12s} {'far':>5s} | {'along % of range (signed median)':>32s} {'far':>6s} {'near':>6s}")
for a, rows in pool.items():
    r = np.vstack(rows); far = r[:, 4] < 34
    band = lambda lo, hi: 100 * np.median(r[(r[:, 3] >= lo) & (r[:, 3] < hi), 1] / r[(r[:, 3] >= lo) & (r[:, 3] < hi), 3])  # noqa: E731
    pos = lambda lo, hi: np.median(r[(r[:, 3] >= lo) & (r[:, 3] < hi), 0])  # noqa: E731
    print(f"   {a}: position median m by range  <25 m {pos(0, 25):.2f}  25-50 {pos(25, 50):.2f}  50-70 {pos(50, 70):.2f}  >70 m {pos(70, 999):.2f}")
    print(f"   {a}: along-ray % by range  <25 m {band(0, 25):+.2f}  25-50 {band(25, 50):+.2f}  50-70 {band(50, 70):+.2f}  >70 m {band(70, 999):+.2f}   (n {len(r)}, non-finite dropped {sum(len(x) for x in rows) - len(r)})")
    print(f"{a:14s} {len(r):6d} {np.median(r[:, 0]):9.2f} {np.median(r[far, 0]):9.2f} {np.percentile(r[:, 0], 90):6.2f} | {np.median(np.abs(r[:, 2])):12.2f} {np.median(np.abs(r[far, 2])):5.2f} |"
          f" {100 * np.median(r[:, 1] / r[:, 3]):32.2f} {100 * np.median(r[far, 1] / r[far, 3]):6.2f} {100 * np.median(r[~far, 1] / r[~far, 3]):6.2f}")
