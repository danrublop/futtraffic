"""N53 - why the fine-tune lost ONE player, and what it was not.

    .venv/bin/python model/validation/n53_probe.py            # every probe
    .venv/bin/python model/validation/n53_probe.py --demo     # the self-check

On 128057, the only TEST clip whose teams the fine-tune never saw, ft0's
detection recall on GT id 12 - the goalkeeper, all-black kit - falls 0.980 ->
0.352 while the other 21 ids all improve. The clip's POOLED recall rises, so
neither the pooled number nor any of N46's five class gates can see it.

Each probe here kills one explanation. They are kept because a ruled-out
mechanism is a number like any other: without a command it is an opinion.

    conf      not the threshold  - at the raw floor 0.20 the keeper reaches 0.509
                                   while the other ids sit at 0.986
    bands     not the tiling     - in the keeper's own tile band the other ids
                                   score 0.988 against stock's 0.986
    keepers   not keepers        - the extreme-x id pair (the keeper proxy) gains
                                   on the other three TEST clips
    darkness  not the kit        - TRAIN carries darker torsos than this one, and
                                   id 6 on this very clip is as dark and gains

WHAT IS LEFT is this individual's appearance under unseen teams. That is a
statement about coverage, not about a bug, and the fix it points at is training
composition - not a threshold, not a tile size, not a merge rule.

POPULATION for every number: 128057, every GT frame present in the arm's cache
(stride 2, 2988 frames), IoU >= 0.5, conf 0.5 unless the probe sweeps it.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n40_occlusion_recall as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STC = os.path.join(ROOT, "model", "validation", "out", "stc2025")
RAW_ZIP = os.path.join(ROOT, "model", "validation", "out", "n46_stage2", "n46_stage2_results.zip")
Q = "128057"
GK = 12
TEST = ("118577", "118578", "128057", "128058")


def _boxes(q, tag, conf=0.0):
    c = json.load(open(os.path.join(STC, f"{q}.{tag}.dets.cache.json")))
    return {int(k): [d["xyxy"] for d in v if d.get("conf", 1.0) >= conf] for k, v in c["frames"].items()}


def _raw(q=Q, arm="ft0"):
    """The PC's pre-threshold detections, straight out of the stage-2 release."""
    import zipfile  # noqa: PLC0415
    with zipfile.ZipFile(RAW_ZIP) as z:
        name = next(n for n in z.namelist() if q in n and arm in n)
        d = json.loads(z.read(name))
    return {int(k): [(x["xyxy"], x["conf"]) for x in v] for k, v in d["frames"].items()}, d["conf"]


def pooled(gt, boxes):
    """Instance-weighted recall. NOT the mean over ids: in a narrow band the ids
    have wildly different instance counts, and an unweighted mean over them
    answers a different question than "how many of these boxes were found"."""
    n = hit = 0
    for f, rows in gt.items():
        if f not in boxes:
            continue
        for _gi, x, y, w, h in rows:
            n += 1
            hit += any(R._iou((x, y, x + w, y + h), tuple(b)) >= 0.5 for b in boxes[f])
    return hit / max(1, n), n


def recall_by_id(gt, boxes, ids=None):
    cnt = collections.defaultdict(lambda: [0, 0])
    for f, rows in gt.items():
        if f not in boxes:
            continue
        for gi, x, y, w, h in rows:
            if ids is not None and gi not in ids:
                continue
            cnt[gi][0] += 1
            cnt[gi][1] += any(R._iou((x, y, x + w, y + h), tuple(b)) >= 0.5 for b in boxes[f])
    return {k: v[1] / max(1, v[0]) for k, v in cnt.items()}


def conf_sweep():
    """Is it the threshold? Sweep the fine-tune's own raw confidences.

    These read the PC's RAW detections, which are pre-pitch-gate, where every
    other probe here reads the imported caches (conf 0.5, `prior_gates: pitch`).
    So a number from this probe is not interchangeable with one from those: the
    comparison that matters is within this sweep, keeper against the other ids.
    """
    gt = R._gt(Q)
    raw, floor = _raw()
    print(f"[conf] raw floor {floor}, {len(raw)} frames")
    for th in (0.5, 0.4, 0.3, 0.25, 0.2):
        r = recall_by_id(gt, {f: [b for b, c in v if c >= th] for f, v in raw.items()})
        others = float(np.mean([v for k, v in r.items() if k != GK]))
        print(f"[conf] conf>={th:.2f}  id{GK} {r[GK]:.3f}  other 21 ids {others:.3f}")


def bands():
    """Is it the tiling? The keeper's own tile band, with every other id as control."""
    gt, ft, st = R._gt(Q), _boxes(Q, "p46ft0"), _boxes(Q, "p46stock")
    edges = [(0, 3072, "tile 3 only"), (3072, 3328, "tiles 3+4 overlap"), (3328, 4096, "tile 4 only")]
    for lo, hi, name in edges:
        sub = {f: [r for r in rows if lo <= r[1] + r[3] / 2 < hi and r[0] == GK] for f, rows in gt.items()}
        sub = {f: v for f, v in sub.items() if v}
        if not sub:
            continue
        (sv, n), (fv, _) = pooled(sub, st), pooled(sub, ft)
        print(f"[bands] {name:18s} n={n:5d}  stock {sv:.3f}  ft0 {fv:.3f}")
    ctl = {f: [r for r in rows if 3072 <= r[1] + r[3] / 2 < 3328 and r[0] != GK] for f, rows in gt.items()}
    ctl = {f: v for f, v in ctl.items() if v}
    (sv, n), (fv, _) = pooled(ctl, st), pooled(ctl, ft)
    print(f"[bands] {'other ids, same band':18s} n={n:5d}  stock {sv:.3f}  ft0 {fv:.3f}")


def keepers():
    """Is it keepers? The two extreme-mean-x ids per clip stand in for the pair."""
    for q in TEST:
        gt = R._gt(q)
        mx = collections.defaultdict(list)
        for rows in gt.values():
            for gi, x, y, w, h in rows:
                mx[gi].append(x + w / 2)
        order = sorted(mx, key=lambda k: float(np.mean(mx[k])))
        gks = {order[0], order[-1]}
        s, f_ = recall_by_id(gt, _boxes(q, "p46stock")), recall_by_id(gt, _boxes(q, "p46ft0"))
        keep = float(np.mean([f_[g] - s[g] for g in gks]))
        oth = float(np.mean([f_[k] - s[k] for k in s if k not in gks]))
        print(f"[keepers] {q} ids {sorted(gks)}  keeper delta {keep:+.3f}   other 20 ids {oth:+.3f}")


def darkness(frames=(2000, 4000, 6000)):
    """Is it the kit? Median torso V, TRAIN clips against this keeper."""
    import cv2  # noqa: PLC0415
    clips = os.path.expanduser("~/Downloads/st2/mot/clips")
    for q in ("117092", "118575", Q):
        path = os.path.join(clips, f"{q}.mp4")
        if not os.path.exists(path):
            print(f"[darkness] {q}: no clip on disk - skipped")
            continue
        gt, cap, per = R._gt(q), cv2.VideoCapture(path), collections.defaultdict(list)
        for f in frames:
            if f not in gt:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ok, img = cap.read()
            if not ok:
                continue
            v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
            for gi, x, y, w, h in gt[f]:
                t = v[max(0, int(y)):int(y + h * 0.5), max(0, int(x)):int(x + w)]
                if t.size:
                    per[gi].append(float(t.mean()))
        cap.release()
        d = {k: float(np.median(x)) for k, x in per.items() if x}
        if not d:
            continue
        lo = sorted(d, key=lambda k: d[k])[:2]
        extra = f", id{GK} {d[GK]:.0f}" if GK in d else ""
        print(f"[darkness] {q}: darkest id{lo[0]} {d[lo[0]]:.0f}, next id{lo[1]} {d[lo[1]]:.0f}, "
              f"median {np.median(list(d.values())):.0f}{extra}")


def demo():
    """recall_by_id is the one piece of arithmetic here; assert it both ways."""
    gt = {1: [(7, 100.0, 100.0, 20.0, 40.0)], 2: [(7, 100.0, 100.0, 20.0, 40.0)]}
    assert recall_by_id(gt, {1: [[100, 100, 120, 140]], 2: [[100, 100, 120, 140]]})[7] == 1.0
    # a box on the wrong player is not a hit, and a frame absent from the arm's
    # cache is not a miss - scoring it as one would invent a regression
    assert recall_by_id(gt, {1: [[400, 400, 420, 440]], 2: [[100, 100, 120, 140]]})[7] == 0.5
    assert recall_by_id(gt, {1: [[100, 100, 120, 140]]})[7] == 1.0
    # pooled() weights by instance and skips frames the arm never scored, which is
    # what keeps a stride-2 cache from reading as half the players missing
    assert pooled(gt, {1: [[100, 100, 120, 140]], 2: [[400, 400, 420, 440]]}) == (0.5, 2)
    assert pooled(gt, {1: [[100, 100, 120, 140]]}) == (1.0, 1)
    print("n53_probe.py self-check OK")


if __name__ == "__main__":
    if "--demo" in sys.argv[1:]:
        demo()
    else:
        for fn in (conf_sweep, bands, keepers, darkness):
            fn()
