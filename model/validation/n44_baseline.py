"""The shipped product's seven-clip baseline and the run-to-run noise of its metrics, from stored N41/N42 results.

Prints what benchmark-status (not published) quotes: the C50 (conf 0.5, shipped merge, frozen N34 map) columns of
model/validation/out/stc2025/n41_held.json and n42_M2_dev.json (the product's own `.p35`/`.p35id` tracks on all seven clips),
and the per-clip |delta| of each metric across the two near-identical change sets (N41 dev+held, N42 dev), as the
spread a future pass/fail threshold must clear (2 x the standard error of a four-clip mean).

    .venv/bin/python model/validation/n44_baseline.py
"""
import json
import os

import numpy as np

STC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model/validation/out/stc2025")
KEYS = ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "IDF1", "IDSW", "ids", "dist_raw", "dist_over_floor", "hsr_raw", "hsr_over_floor", "no_owner")


def main():
    held, dev = json.load(open(os.path.join(STC, "n41_held.json"))), json.load(open(os.path.join(STC, "n42_M2_dev.json")))
    n41dev = json.load(open(os.path.join(STC, "n41_dev.json")))
    base = {**held["C50"], **dev["C50"]}
    print("SHIPPED (C50), seven clips:")
    for k in KEYS:
        v = np.array([base[q][k] for q in sorted(base)], float)
        print(f"  {k:16s} mean {v.mean():.4f} median {np.median(v):.4f} min {v.min():.4f} max {v.max():.4f}")
    print("run-to-run spread, per-clip |delta| rms over N41 dev, N41 held and N42 dev (10 clip-pairs), and 2 SE of a 4-clip mean:")
    for k in ("HOTA", "AssA", "IDF1", "dist_raw", "hsr_raw", "no_owner"):
        d = [r["C30"][q][k] - r["C50"][q][k] for r in (n41dev, held, dev) for q in r["C50"]]
        rms = float(np.sqrt(np.mean(np.square(d))))
        print(f"  {k:10s} rms {rms:.4f}  2SE(4-clip mean) {2 * rms / 2:.4f}")


if __name__ == "__main__":
    main()
