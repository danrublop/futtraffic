"""N35, raw against the feed - no floor: distance per covered metre and HSR ratio over the seven scored clips."""
import glob, json, numpy as np
r = [json.load(open(p)) for p in sorted(glob.glob("model/validation/out/stc2025/n35_e2e*.json"))]
for a in ("SHIP", "AUTO"):
    for k in ("distance_per_covered_m", "hsr_ratio"):
        v = [x[a][k] for x in r]
        print(f"{a} {k} {min(v):.3f} {max(v):.3f} median {np.median(v):.3f}  (n={len(v)} clips)")
