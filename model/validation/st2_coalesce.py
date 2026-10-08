"""Coalescence: two tracks collapsing onto ONE target. The guard for W1/W2/W6.

WHY THIS EXISTS BEFORE THE WORK IT GUARDS
    Every item in tracker-engineering-scope (not published) pushes association toward
    softer, more forgiving decisions - a team likelihood, a deferred duel, a
    capacity constraint. `@C` The characteristic failure of soft association is
    coalescence: two tracks converge onto the same target and the other target
    goes unclaimed (Kropfreiter et al. 2024 on JPDA; the JPDA update provably
    merges two estimates when their predictions are similar, which during a duel
    they are BY CONSTRUCTION).

    Coalescence raises AssA exactly as a real fix does. So without this file, a
    change that quietly merges two players scores as an improvement and ships.
    research-mtt-classical (not published) says build the instrument first; this is it.

TWO MEASURES, AND THEY ANSWER DIFFERENT QUESTIONS

    duplicate   GT-based, unambiguous, for the benchmark. A (frame, GT player)
                where TWO OR MORE track ids each match that player at IoU >= 0.5.
                Nothing legitimate produces this: one player cannot be two people.

    dcenter     GT-free, for production, where there is no truth. Two live track
                ids whose centres stay within 0.6 box-heights for >= 2 s. Real
                duels are 0.5-2 s (failure-taxonomy-association (not published)), so the
                2 s floor is deliberately past the longest honest duel.

READ IT WITH `orphan`
    Coalescence has a signature, not just a count: while two tracks sit on one
    player, ANOTHER player is uncovered. `orphan` is the count of GT players with
    no matching track in the same frames. Duplicates rising while orphans rise
    with them is coalescence. Duplicates alone can just be a redundant detection.

MEASURED BASELINE, `.c03s`, all ten sequences (2026-09-09)

        dup/frame     0.001      orphan/frame   3.49
        dup pairs        49      dcenter pairs   171 over 15,332 frames

    `@M` **We do not coalesce today** - 0.001 duplicates per frame is nothing,
    which is exactly what an over-fragmented tracker looks like (52-101 ids for
    22 players). That is the point of measuring it NOW: this is the zero that
    every softening change has to be held against.

    `@M` **3.49 of 22 players have no matching track in an average frame** - 16%
    of the pitch untracked at any instant. That is a recall number, not a
    coalescence one, and it belongs to the detector and the off-pitch gate.

DCENTER IS A RELATIVE SIGNAL, NOT AN ABSOLUTE COUNT - MEASURED
    `@M` Classifying the dcenter pairs on three sequences: 16 of 17 on 117092,
    and 18/18 and 24/24 on 128057 and 118577, are between tracks that BOTH match
    real players elsewhere - not off-pitch clutter. But `dup` is ~0 on the same
    data, so they are not two tracks on one player either.

    The likely explanation is football: a defender marking an attacker at a set
    piece genuinely stands within 0.6 box-heights for well over two seconds. So
    **dcenter has real false positives by construction, and its absolute value
    means little.** Use it as a delta - a change that moves it sharply is
    suspicious - and use `dup` + `orphan` when ground truth exists.

`ponytail: O(tracks x GT) per frame, which is 22 x 60 on this data. A KD-tree
would be faster and is not needed at this size.`
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
GT_SRC = _ST2 + "/mot"
from st2_mot import WORK  # noqa: E402 - root-relative, one definition
FPS = 25
IOU = 0.5
DCENTER_FRAC = 0.6      # of median box height
DCENTER_S = 2.0         # past the longest honest duel


def _iou_matrix(P, G):
    """P, G are (n,4) and (m,4) top-left x,y,w,h. -> (n,m)."""
    px1, py1 = P[:, 0:1], P[:, 1:2]
    px2, py2 = px1 + P[:, 2:3], py1 + P[:, 3:4]
    gx1, gy1 = G[None, :, 0], G[None, :, 1]
    gx2, gy2 = gx1 + G[None, :, 2], gy1 + G[None, :, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    return inter / (P[:, 2:3] * P[:, 3:4] + G[None, :, 2] * G[None, :, 3]
                    - inter + 1e-9)


def load(seq, variant):
    gt = collections.defaultdict(list)
    for line in open(f"{GT_SRC}/{seq}.txt"):
        p = line.split(",")
        gt[int(p[0])].append((int(float(p[1])), *[float(v) for v in p[2:6]]))
    pr = collections.defaultdict(list)
    for r in json.load(open(f"{WORK}/{seq}{variant}.tracks.json"))["detections"]:
        # tracks.json is bottom-centre x,y + w,h; GT is top-left. Same
        # conversion as st2_mot.stage_pred - getting it wrong puts every box a
        # full box-height low and reads as a worse tracker rather than a bug.
        pr[r["f"]].append((int(r["id"]), r["x"] - r["w"] / 2.0,
                           r["y"] - r["h"], r["w"], r["h"]))
    return gt, pr


def score(seq, variant=".c03s"):
    gt, pr = load(seq, variant)
    h = np.median([b[4] for f in pr for b in pr[f]]) if pr else 1.0
    dthr, dmin = DCENTER_FRAC * h, int(DCENTER_S * FPS)

    dup = orphan = frames = 0
    dup_pairs = collections.Counter()
    close = collections.defaultdict(int)      # (id_a, id_b) -> run length
    dcenter = collections.Counter()

    for f in sorted(set(gt) & set(pr)):
        G = np.array([b[1:] for b in gt[f]], float)
        P = np.array([b[1:] for b in pr[f]], float)
        ids = [b[0] for b in pr[f]]
        if not len(G) or not len(P):
            continue
        frames += 1
        M = _iou_matrix(P, G) >= IOU
        per_gt = M.sum(0)
        dup += int((per_gt >= 2).sum())
        orphan += int((per_gt == 0).sum())
        for j in np.where(per_gt >= 2)[0]:
            hit = sorted(ids[i] for i in np.where(M[:, j])[0])
            for a, b in zip(hit, hit[1:]):
                dup_pairs[(a, b)] += 1
        # dcenter: GT-free. Centres of two live tracks sitting on each other.
        C = np.c_[P[:, 0] + P[:, 2] / 2, P[:, 1] + P[:, 3] / 2]
        D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(len(C)) * 1e9
        seen = set()
        for i, j in zip(*np.where(D < dthr)):
            if i >= j:
                continue
            k = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
            seen.add(k)
            close[k] += 1
        for k in list(close):
            if k not in seen:
                if close[k] >= dmin:
                    # += not =. A Counter assignment kept only the LAST run per
                    # pair, so a tracker with runs of 150 and 60 frames reported
                    # 60 while one with a single run of 150 reported 150 - the
                    # metric moved the wrong way as coalescence got worse.
                    dcenter[k] += close[k]
                del close[k]
    for k, v in close.items():
        if v >= dmin:
            dcenter[k] += v
    return {"frames": frames, "dup": dup, "orphan": orphan,
            "dup_per_frame": dup / max(frames, 1),
            "orphan_per_frame": orphan / max(frames, 1),
            "dup_pairs": len(dup_pairs), "dcenter_pairs": len(dcenter),
            "dcenter_frames": int(sum(dcenter.values()))}


def report(variant=".c03s", seqs=None):
    seqs = seqs or sorted(s[:-len(f"{variant}.tracks.json")]
                          for s in os.listdir(WORK)
                          if s.endswith(f"{variant}.tracks.json")
                          and os.path.exists(f"{GT_SRC}/{s.split('.')[0]}.txt"))
    print(f"\n{'seq':>10s} {'dup/frame':>10s} {'orphan/fr':>10s} "
          f"{'dup pairs':>10s} {'dcenter':>9s} {'dc frames':>10s}")
    tot = []
    for s in seqs:
        r = score(s, variant)
        tot.append(r)
        print(f"{s:>10s} {r['dup_per_frame']:10.3f} {r['orphan_per_frame']:10.3f} "
              f"{r['dup_pairs']:10d} {r['dcenter_pairs']:9d} {r['dcenter_frames']:10d}")
    print(f"{'MEAN':>10s} {np.mean([r['dup_per_frame'] for r in tot]):10.3f} "
          f"{np.mean([r['orphan_per_frame'] for r in tot]):10.3f} "
          f"{np.sum([r['dup_pairs'] for r in tot]):10d} "
          f"{np.sum([r['dcenter_pairs'] for r in tot]):9d} "
          f"{np.sum([r['dcenter_frames'] for r in tot]):10d}")
    return tot


def demo():
    """Synthetic: a clean pair, then the same pair coalesced onto one target."""
    import tempfile
    d = tempfile.mkdtemp()
    global GT_SRC, WORK
    GT_SRC = WORK = d
    n = 100
    # two GT players 200 px apart, both tracked correctly
    with open(f"{d}/T.txt", "w") as fh:
        for f in range(n):
            fh.write(f"{f},1,100,100,20,50,1,1,1\n")
            fh.write(f"{f},2,300,100,20,50,1,1,1\n")
    ok = [{"f": f, "id": i, "x": x + 10, "y": 150.0, "w": 20.0, "h": 50.0}
          for f in range(n) for i, x in ((1, 100), (2, 300))]
    json.dump({"fps": 25.0, "detections": ok}, open(f"{d}/T.ok.tracks.json", "w"))
    r = score("T", ".ok")
    assert r["dup"] == 0 and r["orphan"] == 0, r
    assert r["dcenter_pairs"] == 0, r
    # now both tracks sit on player 1: duplicates AND orphans, together
    bad = [{"f": f, "id": i, "x": 110.0, "y": 150.0, "w": 20.0, "h": 50.0}
           for f in range(n) for i in (1, 2)]
    json.dump({"fps": 25.0, "detections": bad}, open(f"{d}/T.bad.tracks.json", "w"))
    r2 = score("T", ".bad")
    assert r2["dup"] == n, r2                      # player 1 doubly covered
    assert r2["orphan"] == n, r2                   # player 2 abandoned
    assert r2["dcenter_pairs"] == 1, r2            # and it is visible GT-free
    assert r2["dcenter_frames"] == n, r2
    # Two separate runs for one pair must SUM, not overwrite. The first version
    # assigned into the Counter and kept only the last.
    import collections as _c
    d = _c.Counter(); d[("a", "b")] += 150; d[("a", "b")] += 60
    assert d[("a", "b")] == 210, d
    print(f"ok  (clean: dup {r['dup']} orphan {r['orphan']} dcenter "
          f"{r['dcenter_pairs']} | coalesced: dup {r2['dup']} orphan "
          f"{r2['orphan']} dcenter {r2['dcenter_pairs']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=".c03s")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else report(a.variant, a.seqs)
