"""A MEASURED taxonomy of association failures on the STC2025 challenge pair.

WHY THIS EXISTS
    `st2_mot.py` says the pipeline is association-bound (AssA 0.263 against
    DetA 0.590) and `st2_team.py` splits that into 1171 track-side and 663
    GT-side switches. Neither says WHAT HAPPENS at the switch, and every fix on
    the table - appearance re-ID, a longer max_lost, a motion gate, a team gate -
    targets a different mechanism. This classifies the switches by the geometry
    at the frame they occur, so the fixes can be ranked instead of argued.

    It reuses `st2_team`'s loaders and Hungarian matcher outright; the counts it
    starts from reproduce that file's docstring exactly (1171 / 663). It adds
    nothing to the matching - only the per-event forensics.

THE TWO AXES ARE NOT ONE COUNT
    contamination  a track id's GT player changes   - identity STOLEN
    fragmentation  a GT player's track id changes   - identity LOST
    One physical swap can raise both. Measured: 40.8% of contamination events
    share a frame+GT-id with a fragmentation event, so the 1834 total is an
    event count on two axes, not 1834 distinct moments in the video.

`ponytail: thresholds are the population's own median box height, not tuned.`
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "model/validation")
import st2_team as T   # noqa: E402

SEQS = ("128057", "132831")


def centre(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def dist(a, b):
    ca, cb = centre(a), centre(b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def max_overlap(gtf, gid):
    """Largest IoU of gid's GT box with any OTHER GT box in that frame."""
    b = gtf[gid]
    return max([T.iou(b, gtf[o]) for o in gtf if o != gid], default=0.0)


def analyse(seq, variant, iou_min):
    gt = T.load_gt(seq)
    tr, _ = T.load_tracks(f"model/validation/out/stc2025/{seq}.{variant}.tracks.json")
    frames = sorted(set(gt) & set(tr))

    by_track, by_gt = defaultdict(list), defaultdict(list)
    for f in frames:
        for tid, gid in T.match(gt[f], tr[f]).items():
            by_track[tid].append((f, gid))
            by_gt[gid].append((f, tid))

    contam, frag = [], []
    for tid, sq in by_track.items():
        for (f0, g0), (f1, g1) in zip(sq, sq[1:]):
            if g0 == g1:
                continue
            gtf = gt[f1]
            contam.append({"seq": seq, "f_prev": f0, "f": f1, "tid": tid,
                           "gid_from": g0, "gid_to": g1, "gap": f1 - f0 - 1,
                           "dist": dist(gtf[g0], gtf[g1]),
                           "pair_iou": T.iou(gtf[g0], gtf[g1]),
                           "h_to": gtf[g1][3], "h_from": gtf[g0][3],
                           "occl_to": max_overlap(gtf, g1)})

    for gid, sq in by_gt.items():
        for (f0, t0), (f1, t1) in zip(sq, sq[1:]):
            if t0 == t1:
                continue
            gap = f1 - f0 - 1
            # During the gap, was ANY predicted box on the player at all?
            # Distinguishes a detector dropout from a handover the matcher lost.
            covered = sum(
                1 for ff in range(f0 + 1, f1)
                if gid in gt.get(ff, {}) and ff in tr
                and any(T.iou(gt[ff][gid], tb) >= iou_min for tb in tr[ff].values()))
            frag.append({"seq": seq, "f_prev": f0, "f": f1, "gid": gid,
                         "tid_from": t0, "tid_to": t1, "gap": gap,
                         "gap_covered": covered, "h": gt[f1][gid][3],
                         "occl": max_overlap(gt[f1], gid)})

    all_h = [b[3] for f in frames for b in gt[f].values()]
    return {"seq": seq, "contam": contam, "frag": frag, "all_h": all_h,
            "n_tracks": len(by_track), "n_gt": len(by_gt), "n_frames": len(frames)}


def classify(o, medh):
    def c(e):
        if e["pair_iou"] > 0:
            return ("A1 duel/crossing swap, no gap" if e["gap"] == 0
                    else "A2 duel/crossing swap, across a gap")
        return ("A3 near-miss swap (<=1 box-height, no overlap)"
                if e["dist"] <= medh else "A4 teleport / id re-use (>1 box-height)")

    def f(e):
        if e["gap"] == 0:
            return "B3 hard id change, no gap"
        return ("B1 dropout gap (no pred box on the player during the gap)"
                if e["gap_covered"] == 0
                else "B2 handover while covered (a pred box was on the player)")
    return c, f


def report(out, iou_min):
    C = [e for s in out for e in s["contam"]]
    F = [e for s in out for e in s["frag"]]
    ah = np.array([h for s in out for h in s["all_h"]])
    medh, p25 = float(np.median(ah)), float(np.percentile(ah, 25))
    cls_c, cls_f = classify(out, medh)
    tot = len(C) + len(F)

    for s in out:
        print(f"{s['seq']}: {s['n_frames']} frames, {s['n_tracks']} track ids for "
              f"{s['n_gt']} GT players | contamination {len(s['contam'])} "
              f"fragmentation {len(s['frag'])}")
        print(f"    {dict(sorted(Counter(cls_c(e)[:2] for e in s['contam']).items()))} "
              f"{dict(sorted(Counter(cls_f(e)[:2] for e in s['frag']).items()))}")
    print(f"\nmedian GT box height over {len(ah)} GT boxes = {medh:.1f} px "
          f"(the 'one box-height' cutoff; not tuned)")

    rows = []
    for evs, fn, kind in ((C, cls_c, "C"), (F, cls_f, "F")):
        b = defaultdict(list)
        for e in evs:
            b[fn(e)].append(e)
        rows += [(k, v, kind) for k, v in b.items()]
    rows.sort(key=lambda r: -len(r[1]))

    print(f"\n{'mechanism':<56s} {'n':>5s} {'share':>7s}  representative example")
    for k, v, kind in rows:
        key = (lambda e: e["pair_iou"]) if kind == "C" else (lambda e: e["gap"])
        ex = sorted(v, key=key)[len(v) // 2]
        s = (f"{ex['seq']} f={ex['f']} track {ex['tid']}: GT {ex['gid_from']}->"
             f"{ex['gid_to']}, pairIoU {ex['pair_iou']:.2f}, {ex['dist']:.1f}px, gap {ex['gap']}"
             if kind == "C" else
             f"{ex['seq']} f={ex['f']} GT {ex['gid']}: track {ex['tid_from']}->"
             f"{ex['tid_to']}, gap {ex['gap']}f, covered {ex['gap_covered']}f")
        print(f"{k:<56s} {len(v):5d} {100 * len(v) / tot:6.1f}%  {s}")
    print(f"{'TOTAL':<56s} {tot:5d} {100.0:6.1f}%")

    pi = np.array([e["pair_iou"] for e in C])
    d = np.array([e["dist"] for e in C])
    print(f"\ncontamination geometry at the switch frame (the two GT players):")
    print(f"  pair IoU >0 {(pi > 0).sum()} ({100 * (pi > 0).mean():.1f}%), "
          f">=0.3 {(pi >= .3).sum()} ({100 * (pi >= .3).mean():.1f}%), "
          f"median {np.median(pi):.3f}")
    print(f"  centre distance px: p10 {np.percentile(d, 10):.1f} p25 {np.percentile(d, 25):.1f} "
          f"median {np.median(d):.1f} p75 {np.percentile(d, 75):.1f} "
          f"p90 {np.percentile(d, 90):.1f} max {d.max():.1f}")

    ht = np.array([e["h_to"] for e in C])
    fh = np.array([e["h"] for e in F])
    print(f"\nbox height at the switch vs the population (median {medh:.1f}, p25 {p25:.1f}):")
    print(f"  contamination new owner: median {np.median(ht):.1f}, "
          f"{100 * (ht < p25).mean():.1f}% below population p25 (25.0% if unrelated)")
    print(f"  fragmentation player   : median {np.median(fh):.1f}, "
          f"{100 * (fh < p25).mean():.1f}% below population p25")

    ot = np.array([e["occl_to"] for e in C])
    fo = np.array([e["occl"] for e in F])
    print(f"\nocclusion (max IoU of that player's GT box with any OTHER GT box):")
    print(f"  contamination new owner: >0 {100 * (ot > 0).mean():.1f}%, "
          f">=0.3 {100 * (ot >= .3).mean():.1f}%, median {np.median(ot):.3f}")
    print(f"  fragmentation player   : >0 {100 * (fo > 0).mean():.1f}%, "
          f">=0.3 {100 * (fo >= .3).mean():.1f}%, median {np.median(fo):.3f}")

    # Are the two axes counting the same moments?
    for s in out:
        fs = {(e["gid"], e["f"]) for e in s["frag"]}
        hit = sum(1 for e in s["contam"]
                  if (e["gid_from"], e["f"]) in fs or (e["gid_to"], e["f"]) in fs)
        print(f"\n{s['seq']}: contamination events sharing frame+GT-id with a "
              f"fragmentation event: {hit}/{len(s['contam'])} "
              f"= {100 * hit / len(s['contam']):.1f}%")


def control(seqs, stride=20):
    """Baseline for the occlusion claim: how crowded is a RANDOM GT box?"""
    for seq in seqs:
        gt = T.load_gt(seq)
        v = [max_overlap(gt[f], i) for f in sorted(gt)[::stride] for i in gt[f]]
        v = np.array(v)
        print(f"{seq}: {len(v)} GT boxes over {len(sorted(gt)[::stride])} frames "
              f"(every {stride}th): overlaps another GT box >0 {100 * (v > 0).mean():.1f}%, "
              f">=0.3 {100 * (v >= .3).mean():.1f}%, median {np.median(v):.3f}")


def demo():
    T.demo()   # pins the bottom-centre -> top-left convention to the harness
    assert dist((0, 0, 10, 10), (0, 30, 10, 10)) == 30.0
    assert max_overlap({1: (0, 0, 10, 10), 2: (0, 0, 10, 10), 3: (99, 99, 1, 1)}, 1) == 1.0
    assert max_overlap({1: (0, 0, 10, 10), 3: (99, 99, 1, 1)}, 1) == 0.0
    c, f = classify(None, 46.9)
    assert c({"pair_iou": 0.4, "gap": 0, "dist": 9})[:2] == "A1"
    assert c({"pair_iou": 0.4, "gap": 3, "dist": 9})[:2] == "A2"
    assert c({"pair_iou": 0.0, "gap": 0, "dist": 40})[:2] == "A3"
    assert c({"pair_iou": 0.0, "gap": 0, "dist": 400})[:2] == "A4"
    assert f({"gap": 0, "gap_covered": 0})[:2] == "B3"
    assert f({"gap": 5, "gap_covered": 0})[:2] == "B1"
    assert f({"gap": 5, "gap_covered": 2})[:2] == "B2"
    print("failtax demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="c03s")
    ap.add_argument("--seqs", nargs="*", default=list(SEQS))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--dump")
    a = ap.parse_args()
    if a.demo:
        demo()
    elif a.control:
        control(a.seqs)
    else:
        T.IOU_MIN = a.iou
        out = [analyse(s, a.variant, a.iou) for s in a.seqs]
        report(out, a.iou)
        if a.dump:
            json.dump(out, open(a.dump, "w"))
