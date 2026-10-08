"""N40b: is the detector the reason tracks end? Recall of the product's detections on OCCLUDED ground-truth boxes.

Protocol first. Population: the seven STC clips, the MOT ground-truth boxes, the product's cached detections
(<seq>.p35.dets.cache.json: coreml, conf 0.5, stride 2 - the frames the cache holds). A GT box is OCCLUDED when
another GT box overlaps it by >= 35% of its own area in that frame (the unverified audit figure's definition,
association-not-detection-clears-the-leaderboard (internal note)); ISOLATED when no GT box overlaps it at all.
Recall = share of GT boxes with a detection at IoU >= 0.5, per class, per depth third (GT foot row: far / mid / near
third of the pitch's image rows per clip). Also: of the frames where a GT player's box is occluded, how often does the
product's RAW track (`.p40raw`, stitching off) change id or have no box at IoU >= 0.5 at all (so lapse >= 1 - recall
by construction; "no box" includes prior-rejected and tracker-dropped ones), versus isolated frames.
No rule: this is a measurement that decides where the next effort goes - if occluded recall is far below isolated,
fragments are a detection problem; if not, they are a tracker problem.

    .venv/bin/python model/validation/n40_occlusion_recall.py        # -> model/validation/out/stc2025/n40_occlusion.json
    .venv/bin/python model/validation/n40_occlusion_recall.py --conf # 118575 only: the same recalls against the confidence
                                                               # floor, from a pass cached down to conf 0.05 (.p40lo), and
                                                               # the non-GT boxes per frame (GT holds 22 players and no
                                                               # officials, so about 2-3 per frame are the referees)
"""
import json
import os
import sys

import numpy as np
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STC = os.path.join(ROOT, "model/validation/out/stc2025")
SEQS = ("117093", "118575", "118576", "118577", "118578", "128057", "128058")
GT_MOT = _ST2 + "/mot"
OCC = 0.35


def _gt(q):
    gt = {}
    for line in open(f"{GT_MOT}/{q}.txt"):
        f, i, x, y, w, h = line.split(",")[:6]
        gt.setdefault(int(f), []).append((int(i), float(x), float(y), float(w), float(h)))
    return gt


def _dets(q):
    c = json.load(open(os.path.join(STC, f"{q}.p35.dets.cache.json")))
    # {frame: [{'xyxy': [x0, y0, x1, y1], 'conf', 'cls'}]}; the product's conf 0.5 is applied at replay, so apply it here
    return {int(k): [d["xyxy"] for d in v if d.get("cls") == "person" and d.get("conf", 1.0) >= 0.5] for k, v in c["frames"].items()}


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - i)


def main():
    res = {}
    for q in SEQS:
        gt, dets = _gt(q), _dets(q)
        if not dets:
            print(q, "no detections cache in a readable layout"); continue
        rows = json.load(open(os.path.join(STC, f"{q}.p40raw.tracks.json")))["detections"]
        # raw track id per (frame, GT id): the detection matched to the GT box at IoU >= 0.5, if any
        by_f = {}
        for r in rows:
            by_f.setdefault(int(r["f"]), []).append(r)
        feet = [g[2] + g[4] for fr in gt.values() for g in fr]
        thirds = np.percentile(feet, [33.3, 66.7])
        cnt = {c: {"n": 0, "hit": 0, "lapse": 0} for c in ("occluded_far", "occluded_mid", "occluded_near", "isolated_far", "isolated_mid", "isolated_near")}
        tid_at = {}
        for f in sorted(dets):
            if f not in gt:
                continue
            boxes = [tuple(b) for b in dets[f]]
            xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
            for (gi, x, y, w, h), g in zip(gt[f], xyxy):
                others = [o for o in xyxy if o is not g]
                cover = max((max(0, min(g[2], o[2]) - max(g[0], o[0])) * max(0, min(g[3], o[3]) - max(g[1], o[1])) / (w * h) for o in others), default=0.0)
                cls = "occluded" if cover >= OCC else "isolated" if cover == 0.0 else None
                if cls is None:
                    continue
                band = "far" if y + h < thirds[0] else "mid" if y + h < thirds[1] else "near"
                key = f"{cls}_{band}"
                cnt[key]["n"] += 1
                hit = any(_iou(g, b) >= 0.5 for b in boxes)
                cnt[key]["hit"] += int(hit)
                m = [r for r in by_f.get(f, []) if _iou(g, (r["x"] - r["w"] / 2, r["y"] - r["h"], r["x"] + r["w"] / 2, r["y"])) >= 0.5]
                tid = int(m[0]["id"]) if m else None
                prev = tid_at.get(gi)
                if prev is not None and (tid is None or tid != prev):
                    cnt[key]["lapse"] += 1
                if tid is not None:
                    tid_at[gi] = tid
        res[q] = {k: {"n": v["n"], "recall": round(v["hit"] / v["n"], 4) if v["n"] else None, "id_lapse_rate": round(v["lapse"] / v["n"], 4) if v["n"] else None} for k, v in cnt.items()}
        print(q, {k: (v["n"], v["recall"], v["id_lapse_rate"]) for k, v in res[q].items()}, flush=True)
    pooled = {}
    for k in next(iter(res.values())):
        n = sum(res[q][k]["n"] for q in res)
        pooled[k] = {"n": n, "recall": round(sum(res[q][k]["recall"] * res[q][k]["n"] for q in res if res[q][k]["n"]) / n, 4),
                     "id_lapse_rate": round(sum(res[q][k]["id_lapse_rate"] * res[q][k]["n"] for q in res if res[q][k]["n"]) / n, 4)}
    res["POOLED"] = pooled
    json.dump(res, open(os.path.join(STC, "n40_occlusion.json"), "w"), indent=1)
    print("POOLED", json.dumps(pooled))
    print("  ".join(f"{k} n={v['n']} recall={v['recall']:.4f} lapse={v['id_lapse_rate']:.4f}" for k, v in pooled.items()))


def conf_sweep(q="118575"):
    c = json.load(open(os.path.join(STC, f"{q}.p40lo.dets.cache.json")))
    frames = {int(k): [(d["xyxy"], d["conf"]) for d in v if d.get("cls") == "person"] for k, v in c["frames"].items()}
    gt = _gt(q)
    feet = [g[2] + g[4] for fr in gt.values() for g in fr]
    th = np.percentile(feet, [33.3, 66.7])
    keys = ("occluded_far", "occluded_mid", "occluded_near", "isolated_far", "isolated_mid", "isolated_near")
    print("conf " + " ".join(f"{k:>13s}" for k in keys) + "  non-GT boxes/frame")
    for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
        cnt = {k: [0, 0] for k in keys}
        fp = nf = 0
        for f, ds in frames.items():
            if f not in gt:
                continue
            nf += 1
            boxes = [tuple(b) for b, cf in ds if cf >= thr]
            xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
            for g, (_gi, x, y, w, h) in zip(xyxy, gt[f]):
                cover = max((max(0, min(g[2], o[2]) - max(g[0], o[0])) * max(0, min(g[3], o[3]) - max(g[1], o[1])) / (w * h) for o in xyxy if o is not g), default=0.0)
                cls = "occluded" if cover >= OCC else "isolated" if cover == 0.0 else None
                if cls is None:
                    continue
                k = f"{cls}_{'far' if y + h < th[0] else 'mid' if y + h < th[1] else 'near'}"
                cnt[k][0] += 1
                cnt[k][1] += any(_iou(g, b) >= 0.5 for b in boxes)
            fp += sum(1 for b in boxes if not any(_iou(g, b) >= 0.5 for g in xyxy))
        print(f"{thr:4.2f} " + " ".join(f"{cnt[k][1] / cnt[k][0]:13.3f}" for k in keys) + f"  {fp / nf:6.2f}")


if __name__ == "__main__":
    conf_sweep() if "--conf" in sys.argv else main()
