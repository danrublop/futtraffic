"""N31 scale validator: WHERE and in WHICH DIRECTION a pitch map reads movement long or short.

WHY
    N30's review (audit-n30 (not published) §3) found its distance rule blind to sign, and every map - ST2's
    own chain included - reading movement several percent short of the feed. One pooled slope cannot
    say whether that is the feed's scale (same everywhere, both axes), a projection-family error
    (varies with pan angle), a crowned pitch (opposite signs near and far) or the box-foot point
    (grows with depth). This splits N27's chord slope by pitch axis and by region. It is a
    DIAGNOSTIC on maps N30 already scored, on the rows N30 already used: no new map, no rule, no
    verdict, nothing selected. n30-operator-maps (not published) §5 item 2.

WHAT IT PRINTS, per arm (N30's five and ST2's chain), pooled over clips and per clip
    all          dist_eval._chord_slope_arrays as N27 and N30 used it (4 s chords, >= 5 m of true movement)
    x, y         the same regression on one pitch axis only: a chord counts when the TRUTH moved >= 5 m
                 along that axis. x runs along the touchline, y across the pitch, y < 34 the far half.
    far, near    chords whose two ends are both in that half (by the truth's y)
    left, mid, right   chords whose two ends are both in that third of the length (truth's x)
    Pooled = the least-squares slope over all clips' chords together, NOT a mean of per-clip slopes.
    `drop` is the share of track rows with no IoU owner, which every one of these excludes.

HELD FIXED  N30's frozen clicks and its five arms (n30_operator_maps.arms_for), `p21idn30` rows
            (N21's product tracks, N6's calibration for the ground truth), stride-2 frames.

    .venv/bin/python model/validation/n31_scale.py --run     # ~15 min, writes model/validation/out/stc2025/n31_scale.json
    .venv/bin/python model/validation/n31_scale.py --demo
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

OUT = os.path.join(ROOT, "model/validation/out/stc2025/n31_scale.json")
REGIONS = {"far": (1, None, 34.0), "near": (1, 34.0, None),
           "left": (0, None, 35.0), "mid": (0, 35.0, 70.0), "right": (0, 70.0, None)}


def chord_inputs(seq, variant):
    """dist_eval._chord_slope's loader, returning the arrays instead of one slope.
    -> (fragment ids, frames, owners, feet px, truth metres, fps, share of rows with no owner)."""
    import contam_eval as C  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    blob = json.load(open(f"{D.STC}/{seq}.{variant}.tracks.json"))
    rows, fps = blob["detections"], float(blob["fps"])
    owner, _counts = C.owners_iou(seq, rows, D.STC_EVERY)
    look = {k: dict(zip(v["f"].tolist(), map(tuple, v["xy"]))) for k, v in D.stc_gt(seq, D.STC_EVERY).items()}
    idx = [i for i in owner if int(rows[i]["f"]) % D.STC_EVERY == 0]
    uv = np.array([[rows[i]["x"], rows[i]["y"]] for i in idx], float)
    G = np.array([look.get(owner[i], {}).get(int(rows[i]["f"]), (np.nan, np.nan)) for i in idx], float)
    return ([int(rows[i]["id"]) for i in idx], [int(rows[i]["f"]) for i in idx], [owner[i] for i in idx],
            uv, G, fps, 1.0 - len(owner) / len(rows))


def _chords(tid, f, g, X, G, fps, secs=4.0):
    """(dX, dG) over N27's chords: runs on one fragment and one owner with no missed frame, ends
    `secs` apart. dist_eval._chord_slope_arrays builds the same set; --demo pins the two together."""
    import dist_eval as D  # noqa: PLC0415
    order = np.lexsort((np.asarray(f), np.asarray(tid)))
    t, fr, gg = np.asarray(tid)[order], np.asarray(f)[order], np.asarray(g)[order]
    X, G = np.asarray(X, float)[order], np.asarray(G, float)[order]
    brk = np.r_[True, (t[1:] != t[:-1]) | (fr[1:] - fr[:-1] != D.STC_EVERY) | (gg[1:] != gg[:-1])]
    st = np.flatnonzero(brk)
    n = int(round(secs * fps / D.STC_EVERY))
    parts = [np.arange(a, b - n) for a, b in zip(st, np.r_[st[1:], len(fr)]) if b - a > n]
    A = np.concatenate(parts) if parts else np.zeros(0, int)
    return X[A + n] - X[A], G[A + n] - G[A], G[A], G[A + n]


def sums(tid, f, g, X, G, fps, min_len=5.0):
    """{split: (sum dX.dG, sum dG.dG, chords)} - the two sums a least-squares slope is made of, so
    clips pool exactly. One axis: that component only, and the truth must move `min_len` along it.
    A region: both ends of the chord inside it, by the truth."""
    dX, dG, Ga, Gb = _chords(tid, f, g, X, G, fps)
    ok = np.isfinite(dX).all(1) & np.isfinite(dG).all(1)

    def acc(mask, axes):
        k = ok & mask & (np.linalg.norm(dG[:, axes], axis=1) >= min_len)
        return (float((dX[k][:, axes] * dG[k][:, axes]).sum()), float((dG[k][:, axes] ** 2).sum()), int(k.sum()))
    every = np.ones(len(dG), bool)
    out = {"all": acc(every, [0, 1]), "x": acc(every, [0]), "y": acc(every, [1])}
    for name, (ax, lo, hi) in REGIONS.items():
        inside = lambda P: (P[:, ax] >= (-np.inf if lo is None else lo)) & (P[:, ax] < (np.inf if hi is None else hi))  # noqa: E731
        out[name] = acc(inside(Ga) & inside(Gb), [0, 1])
    return out


def slopes(acc):
    return {k: (round(v[0] / v[1], 4) if v[1] else None) for k, v in acc.items()}


def run():
    import dist_eval as D  # noqa: PLC0415
    import n30_operator_maps as N  # noqa: PLC0415
    seqs = N._check_freeze()
    arms = (*N.ARMS, "CHAIN")
    per_clip, pooled, drop, chords = {}, {a: {} for a in arms}, {}, {}
    for q in seqs:
        payload = json.load(open(os.path.join(N.CLICKS, f"{q}.calibration.json")))
        cals, why, _rms = N.arms_for(payload, N.CM.T._wh(q))
        cals["CHAIN"], why["CHAIN"] = N._Chain(q), None
        tid, f, g, uv, G, fps, drop[q] = chord_inputs(q, N.VARIANT)
        per_clip[q] = {}
        for a in arms:
            if why[a] or cals[a] is None:
                per_clip[q][a] = None
                continue
            X = np.asarray(cals[a].to_pitch(uv), float).reshape(-1, 2)
            acc = sums(tid, f, g, X, G, fps)
            per_clip[q][a] = slopes(acc)
            chords[q] = {k: v[2] for k, v in acc.items()}
            for k, v in acc.items():
                p = pooled[a].setdefault(k, [0.0, 0.0, 0])
                pooled[a][k] = [p[0] + v[0], p[1] + v[1], p[2] + v[2]]
        print(q, "drop", round(drop[q], 4), {a: (per_clip[q][a] or {}).get("all") for a in arms}, flush=True)
    res = {"seqs": seqs, "variant": N.VARIANT, "drop_share": {q: round(v, 4) for q, v in drop.items()},
           "pooled": {a: slopes(pooled[a]) for a in arms}, "pooled_chords": {a: {k: v[2] for k, v in pooled[a].items()} for a in arms},
           "per_clip": per_clip, "chords": chords,
           "note": "TPS pools 7 clips (refused on 117092); every other arm pools 8"}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    cols = ("all", "x", "y", "far", "near", "left", "mid", "right")
    print("\npooled slope   " + "".join(f"{c:>8}" for c in cols))
    for a in arms:
        print(f"{a:14s} " + "".join(f"{res['pooled'][a].get(c) if res['pooled'][a].get(c) is not None else float('nan'):8.3f}" for c in cols))
    print(f"-> {OUT}")


def demo():
    """A map that reads x 10% short and y 10% long must say so on each axis, and a map that is
    right in the far half and 20% long in the near half must say so by half."""
    import dist_eval as D  # noqa: PLC0415
    n, fps = 4000, 25.0
    tid = np.repeat(np.arange(4), n // 4)
    f = np.tile(np.arange(n // 4) * D.STC_EVERY, 4)
    g = tid + 100
    i = np.arange(n)
    G = np.column_stack([52.5 + 45 * np.sin(i / 60), np.where(tid % 2 == 0, 15.0, 55.0) + 8 * np.sin(i / 40)])
    a = sums(tid, f, g, G * [0.9, 1.1], G, fps)
    s = slopes(a)
    assert abs(s["x"] - 0.9) < 1e-6 and abs(s["y"] - 1.1) < 1e-6, s
    assert 0.9 < s["all"] < 1.1, s
    X = G.copy()
    near = G[:, 1] >= 34
    X[near] = [52.5, 34] + 1.2 * (G[near] - [52.5, 34])
    s = slopes(sums(tid, f, g, X, G, fps))
    assert abs(s["far"] - 1.0) < 1e-6 and abs(s["near"] - 1.2) < 1e-6, s
    # pooled = sums added, not slopes averaged: twice the chords at 0.9 and once at 1.1 is not 1.0
    b = sums(tid, f, g, G * 1.1, G, fps)
    c = sums(tid, f, g, G * 0.9, G, fps)
    pooled = (2 * c["all"][0] + b["all"][0]) / (2 * c["all"][1] + b["all"][1])
    assert abs(pooled - (2 * 0.9 + 1.1) / 3) < 1e-9, pooled
    # and "all" is N27's slope, chord for chord, on a map that is neither exact nor linear
    W = G + 0.02 * (G - [52.5, 34]) ** 2 / 50
    W[::97] = np.nan
    ref = D._chord_slope_arrays(tid, f, g, W, G, fps)
    mine = sums(tid, f, g, W, G, fps)["all"]
    assert abs(mine[0] / mine[1] - ref[0]) < 1e-12 and mine[2] == ref[1], (mine, ref)
    print("n31 scale demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g_ = ap.add_mutually_exclusive_group(required=True)
    g_.add_argument("--run", action="store_true")
    g_.add_argument("--demo", action="store_true")
    a_ = ap.parse_args()
    (run if a_.run else demo)()
