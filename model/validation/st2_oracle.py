"""Rebuild the association ORACLE: relabel our own tracklets by their true id.

Upper bound on what a perfect merging stitcher could reach over the SAME boxes.
Greedy IoU>=0.5 match per frame, majority GT id per pred track, then resolve
within-frame double-cover by keeping the best-IoU box.
"""
import json, sys, os
from collections import defaultdict, Counter
import numpy as np
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

GT = _ST2 + "/mot"
W = "model/validation/out/stc2025"

def iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    x0 = max(ax0, bx0); y0 = max(ay0, by0)
    x1 = min(ax0+aw, bx0+bw); y1 = min(ay0+ah, by0+bh)
    if x1 <= x0 or y1 <= y0: return 0.0
    i = (x1-x0)*(y1-y0)
    return i / (aw*ah + bw*bh - i)

def build(seq, variant):
    pred = json.load(open(f"{W}/{seq}.{variant}.tracks.json"))["detections"]
    byf = defaultdict(list)
    for r in pred:
        byf[r["f"]].append(r)
    gt = defaultdict(list)
    for line in open(f"{GT}/{seq}.txt"):
        p = line.split(",")
        gt[int(p[0])].append((int(float(p[1])), tuple(float(v) for v in p[2:6])))
    # per-frame greedy match
    match = {}   # (f, idx) -> gid ; and iou
    for f, rows in byf.items():
        g = gt.get(f, [])
        if not g: continue
        pairs = []
        for i, r in enumerate(rows):
            tl = (r["x"] - r["w"]/2.0, r["y"] - r["h"], r["w"], r["h"])
            for j, (gid, gb) in enumerate(g):
                v = iou(tl, gb)
                if v >= 0.5: pairs.append((v, i, j, gid))
        pairs.sort(reverse=True)
        ui, uj = set(), set()
        for v, i, j, gid in pairs:
            if i in ui or j in uj: continue
            ui.add(i); uj.add(j)
            match[(f, i)] = (gid, v)
    # majority GT id per predicted track
    vote = defaultdict(Counter)
    for (f, i), (gid, v) in match.items():
        vote[byf[f][i]["id"]][gid] += 1
    lab = {tid: c.most_common(1)[0][0] for tid, c in vote.items()}
    # relabel + resolve collisions
    out = []
    dropped_nolabel = dropped_collide = 0
    for f, rows in byf.items():
        claim = {}
        for i, r in enumerate(rows):
            tid = r["id"]
            if tid not in lab:
                dropped_nolabel += 1; continue
            gid = lab[tid]
            q = match.get((f, i), (None, 0.0))[1]
            if gid not in claim or q > claim[gid][1]:
                if gid in claim: dropped_collide += 1
                claim[gid] = (r, q)
            else:
                dropped_collide += 1
        for gid, (r, q) in claim.items():
            out.append({**r, "id": int(gid)})
    out.sort(key=lambda r: (r["f"], r["id"]))
    json.dump({"fps": 25.0, "detections": out}, open(f"{W}/{seq}.orc2.tracks.json", "w"))
    print(f"{seq}: {len(pred)} boxes -> {len(out)} ({100*len(out)/len(pred):.1f}%), "
          f"{len({r['id'] for r in pred})} ids -> {len({r['id'] for r in out})}; "
          f"dropped unlabelled {dropped_nolabel} ({100*dropped_nolabel/len(pred):.1f}%), "
          f"double-cover {dropped_collide} ({100*dropped_collide/len(pred):.1f}%)")

def demo():
    """Self-check: the IoU, and the one invariant the whole file exists to keep.

    TrackEval refuses two boxes carrying one identity in one frame. Dropping
    those is the ONLY reason this oracle is not a free relabel, so if the
    resolver silently stopped firing the output would look better and score
    worse. Checked on real output when there is any.
    """
    a = (0.0, 0.0, 10.0, 20.0)
    assert iou(a, a) == 1.0
    assert iou(a, (100.0, 0.0, 10.0, 20.0)) == 0.0
    assert abs(iou(a, (0.0, 10.0, 10.0, 20.0)) - 1 / 3) < 1e-9
    n = 0
    for f in sorted(os.listdir(W)) if os.path.isdir(W) else []:
        if not f.endswith(".orc2.tracks.json"):
            continue
        seen = defaultdict(set)
        for r in json.load(open(f"{W}/{f}"))["detections"]:
            assert r["id"] not in seen[r["f"]], f"double-cover survived in {f}"
            seen[r["f"]].add(r["id"])
        n += 1
    print(f"st2_oracle.py self-check OK ({n} built oracles checked for double-cover)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--demo"]:
        demo(); sys.exit()
    for s in sys.argv[2:]:
        build(s, sys.argv[1])
