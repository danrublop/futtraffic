"""N42: the product's own box merge deletes occluded players - measure it, and its documented upgrade path. Protocol first.

WHY
    RF-DETR is a set-prediction detector with no NMS. The product tiles the panorama and then de-duplicates ALL boxes
    in a frame by intersection-over-smaller >= 0.55 (model/engine/detectors.py `_merge`, `merge_thr`), whose own comment
    names the cost: "a box almost wholly inside another is treated as the same player. Ceiling is a player standing
    directly in front of a much larger near-side player, where the far one is dropped. Upgrade path is IoS only
    between boxes from DIFFERENT tiles and IoU within a tile". N40 found occluded players detected at 0.49-0.68 recall
    (docs/experiments/n40-association.md §2); every occluded pair the detector DID find with > 55% coverage is deleted here by
    design. N41 showed that more boxes at a lower confidence are not better boxes; this is about keeping the good ones.

ARMS - one unmerged detector pass per clip (TiledDetector(raw=True): every tile's boxes with `_tile` and `_clip`
       kept, conf 0.5, tile 1024, overlap as shipped, stride 2, the clip's zero-click calibration for the prior as the
       product runs it), then each merge applied OFFLINE to the same boxes and written as a cache the product's
       tracker replays (same header as the `.p35` cache, so the stamp check passes):
    M0   the shipped merge: `_merge(dets, 0.55)` - the control, and the REPRODUCTION check: its boxes must equal the
         `.p35` cache's on every frame (same detector, same frames) or the harness is wrong.
    M1   the file's upgrade path: IoS >= 0.55 suppresses only between boxes from different tiles; within a tile,
         IoU >= 0.55 suppresses. Clipped boxes lose to whole ones, then confidence, as today.
    M2   M1 with the within-tile IoU at 0.70 (the set-prediction detector rarely duplicates; only near-identical
         boxes are the same player). n = 2 candidates, declared now; nothing else is tried on these clips.

SPLIT (N40/N41's).  DEV = 118575, 118576, 117093.  HELD = 118577, 118578, 128057, 128058, run ONCE.

LEVELS
    L1  detections only, DEV: N40's recall by class (occluded / isolated x far / mid / near, IoU >= 0.5 against the
        MOT boxes) and non-GT boxes per frame, for M0/M1/M2 from the caches. Selection: the candidate with the higher
        far+mid occluded recall goes on, provided its non-GT boxes per frame are within +0.5 of M0 (a merge that stops
        merging duplicates is a false-positive generator, and N41 says what those do downstream).
    L2  the product path from the selected cache: tracker replay -> split + stitch -> identity; scored exactly as N41
        (TrackEval with DetRe and DetPr, distance and HSR raw and over floor, no-owner). DEV first, HELD once.

RULE, fixed now (N41's, so the two candidates are judged alike). The candidate REPLACES the shipped merge iff on HELD:
      HOTA >= M0 + 0.010;  DetPr >= M0 - 0.020 on every clip;  IDSW <= 1.15 x M0;
      distance raw <= M0 + 0.010;  HSR raw <= M0;  no-owner <= M0 + 0.02.
    Anything else: NOT SUPPORTED; the shipped merge stays; the numbers are published.

    .venv/bin/python model/validation/n42_merge.py --detect 118575     # unmerged pass -> <seq>.p42raw.dets.cache.json (~8 min GPU)
    .venv/bin/python model/validation/n42_merge.py --caches            # M0/M1/M2 caches for every clip that has a raw pass
    .venv/bin/python model/validation/n42_merge.py --l1                # recalls on DEV -> n42_l1.json
    .venv/bin/python model/validation/n42_merge.py --l2 M1 dev|held    # product path + N41 scoring
    .venv/bin/python model/validation/n42_merge.py --show
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
DEV = ("118575", "118576", "117093")
HELD = ("118577", "118578", "128057", "128058")
STRIDE, CONF = 2, 0.5
WITHIN = {"M1": 0.55, "M2": 0.70}



def _git_env():
    """verify_claims.git_env - the environment MINUS every GIT_*.

    Without it an exported GIT_DIR (a hook, or any shell started inside one)
    BEATS cwd=, so this guard reads the OTHER tree's HEAD, where this path is
    usually clean - returncode 0, "registered", green. It fails SHORT, in the
    direction that serves an unregistered run as a registered one.
    a-green-gate-is-not-evidence-it-looked (internal note)
    """
    import sys as _sys  # noqa: PLC0415
    _sys.path.insert(0, os.path.join(ROOT, "model", "validation"))
    import verify_claims as _V  # noqa: PLC0415
    return _V.git_env()


def _ios_iou(boxes):
    b = np.asarray(boxes, float)
    x0 = np.maximum(b[:, None, 0], b[None, :, 0]); y0 = np.maximum(b[:, None, 1], b[None, :, 1])
    x1 = np.minimum(b[:, None, 2], b[None, :, 2]); y1 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ios = inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-9)
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
    return ios, iou


def merge(dets, mode):
    """M0 = the product's _merge; M1/M2 = IoS across tiles, IoU within a tile."""
    import detectors as DT  # noqa: PLC0415
    if mode == "M0":
        return DT._merge([dict(d) for d in dets], 0.55)
    if len(dets) < 2:
        return [{k: v for k, v in d.items() if k not in ("_clip", "_tile")} for d in dets]
    order = sorted(range(len(dets)), key=lambda i: (dets[i]["_clip"], -dets[i]["conf"]))
    ios, iou = _ios_iou([d["xyxy"] for d in dets])
    tile = np.array([d["_tile"] for d in dets])
    same = tile[:, None] == tile[None, :]
    sup = np.where(same, iou >= WITHIN[mode], ios >= 0.55)
    keep = []
    for i in order:
        if not keep or not sup[i, keep].any():
            keep.append(i)
    return [{k: v for k, v in dets[i].items() if k not in ("_clip", "_tile")} for i in sorted(keep)]


def detect(seq):
    import detectors as DT  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    out = os.path.join(STC, f"{seq}.p42raw.dets.cache.json")
    if os.path.exists(out) and json.load(open(out)).get("complete"):
        print("have", out); return
    head = json.load(open(os.path.join(STC, f"{seq}.p35.dets.cache.json")))
    head = {k: head[k] for k in ("video", "start", "calib", "calib_sha", "settings")}
    det = DT.TiledDetector(DT.CoreMLDetector(conf=CONF), raw=True)
    cap = cv2.VideoCapture(f"{ST.CLIPS}/{seq}.mp4")
    frames, f = {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % STRIDE == 0:
            frames[str(f)] = [{"xyxy": [float(v) for v in d["xyxy"]], "conf": float(d["conf"]), "cls": d.get("cls"),
                               "_clip": bool(d["_clip"]), "_tile": int(d["_tile"])} for d in det.detect(img)]
            if f % 400 == 0:
                print(f"  frame {f}", flush=True)
        f += 1
    json.dump({**head, "complete": True, "frames": frames}, open(out, "w"))
    print("->", out)


def caches():
    for seq in DEV + HELD:
        src = os.path.join(STC, f"{seq}.p42raw.dets.cache.json")
        if not os.path.exists(src):
            continue
        raw = json.load(open(src))
        # The product writes its cache AFTER the scene prior and a replay does not re-apply it (found on the first
        # run: 2797 of 3000 frames differed before the prior, 0 after; a candidate cache without it replays off-pitch
        # boxes, DetPr 0.762 -> 0.709), so every merge's cache gets the same prior the product used.
        import calibrators as C  # noqa: PLC0415
        poly = np.asarray(C.load(os.path.join(STC, f"{seq}.p35.calibration.json")).pitch_polygon_px(), np.float32)
        inside = lambda b: cv2.pointPolygonTest(poly, (float((b[0] + b[2]) / 2), float(b[3])), False) >= 0  # noqa: E731
        for mode in ("M0", "M1", "M2"):
            out = {**{k: raw[k] for k in ("video", "start", "calib", "calib_sha", "settings", "complete")},
                   "frames": {f: [d for d in merge(ds, mode) if inside(d["xyxy"])] for f, ds in raw["frames"].items()}}
            json.dump(out, open(os.path.join(STC, f"{seq}.p42{mode.lower()}.dets.cache.json"), "w"))
        # reproduction: M0 must equal the product's own cache, box for box
        p35 = json.load(open(os.path.join(STC, f"{seq}.p35.dets.cache.json")))["frames"]
        m0 = json.load(open(os.path.join(STC, f"{seq}.p42m0.dets.cache.json")))["frames"]
        diff = sum(1 for f in p35 if sorted(round(x, 1) for d in p35[f] for x in d["xyxy"])
                   != sorted(round(x, 1) for d in m0.get(f, []) for x in d["xyxy"]))
        print(seq, "M0 (after the prior) vs .p35 cache: frames differing", diff, "of", len(p35), flush=True)
        assert diff == 0, "the harness does not reproduce the product's detections"


def l1():
    import n40_occlusion_recall as R  # noqa: PLC0415
    res = {}
    for mode in ("M0", "M1", "M2"):
        cnt = {k: [0, 0] for k in ("occluded_far", "occluded_mid", "occluded_near", "isolated_far", "isolated_mid", "isolated_near")}
        fp = nf = 0
        for q in DEV:
            gt = R._gt(q)
            frames = {int(k): [d["xyxy"] for d in v if d.get("cls") == "person"] for k, v in json.load(open(os.path.join(STC, f"{q}.p42{mode.lower()}.dets.cache.json")))["frames"].items()}
            feet = [g[2] + g[4] for fr in gt.values() for g in fr]
            th = np.percentile(feet, [33.3, 66.7])
            for f, boxes in frames.items():
                if f not in gt:
                    continue
                nf += 1
                xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
                for g, (_gi, x, y, w, h) in zip(xyxy, gt[f]):
                    cover = max((max(0, min(g[2], o[2]) - max(g[0], o[0])) * max(0, min(g[3], o[3]) - max(g[1], o[1])) / (w * h) for o in xyxy if o is not g), default=0.0)
                    cls = "occluded" if cover >= R.OCC else "isolated" if cover == 0.0 else None
                    if cls is None:
                        continue
                    k = f"{cls}_{'far' if y + h < th[0] else 'mid' if y + h < th[1] else 'near'}"
                    cnt[k][0] += 1
                    cnt[k][1] += any(R._iou(g, tuple(b)) >= 0.5 for b in boxes)
                fp += sum(1 for b in boxes if not any(R._iou(g, tuple(b)) >= 0.5 for g in xyxy))
        res[mode] = {**{k: round(v[1] / v[0], 4) for k, v in cnt.items()}, "non_gt_per_frame": round(fp / nf, 3)}
        print(mode, res[mode], flush=True)
    json.dump(res, open(os.path.join(STC, "n42_l1.json"), "w"), indent=1)


def l2(mode, which):
    import n41_conf as N41  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    seqs = HELD if which == "held" else DEV
    if which == "held" and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n42_merge.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the held-out run must be made by the committed protocol")
    tag = f".p42{mode.lower()}"
    for q in seqs:
        cal = ST._p(q, ".p35.calibration.json")
        ST.track(q, cal, tag=tag, cache_from=tag)
        ST.identify(q, cal, src=tag, dst=tag + "id")
    N41.ARMS = {"C50": ("p35", "p35id"), "C30": (tag[1:], tag[1:] + "id")}   # N41's scorer, its columns and its rule
    N41.score(which, out=os.path.join(STC, f"n42_{mode}_{which}.json"))   # never N41's own file (review: it clobbered n41_dev.json once)


def show():
    p = os.path.join(STC, "n42_l1.json")
    if os.path.exists(p):
        r = json.load(open(p))
        for mode, v in r.items():
            print(mode, " ".join(f"{k} {x}" for k, x in v.items()))
    import glob  # noqa: PLC0415
    for f in sorted(glob.glob(os.path.join(STC, "n42_M*_*.json"))):
        print(os.path.basename(f))
        r = json.load(open(f))
        seqs = list(r["C50"])
        m = lambda a, k: float(np.nanmean([r[a][q][k] for q in seqs]))  # noqa: E731
        keys = ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "IDF1", "IDSW", "dist_raw", "hsr_raw", "no_owner")
        print("  MEAN " + " ".join(f"{k} {m('C50', k):.4f}->{m('C30', k):.4f}" for k in keys))
        if f.endswith("_held.json"):
            ok = [m("C30", "HOTA") >= m("C50", "HOTA") + 0.010, all(r["C30"][q]["DetPr"] >= r["C50"][q]["DetPr"] - 0.020 for q in seqs),
                  m("C30", "IDSW") <= 1.15 * m("C50", "IDSW"), m("C30", "dist_raw") <= m("C50", "dist_raw") + 0.010,
                  m("C30", "hsr_raw") <= m("C50", "hsr_raw"), m("C30", "no_owner") <= m("C50", "no_owner") + 0.02]
            print("  RULE:", "REPLACES the shipped merge" if all(ok) else f"NOT SUPPORTED {ok}")


def demo():
    # two tiles see the same near player (whole in one, clipped in the other): every rule keeps one box;
    # a far player 60%-inside a near player's box in the SAME tile: M0 deletes them, M1/M2 keep both
    near = {"xyxy": (100, 100, 140, 200), "conf": 0.9, "cls": "person", "_clip": False, "_tile": 0}
    near_clip = {"xyxy": (100, 150, 140, 200), "conf": 0.8, "cls": "person", "_clip": True, "_tile": 1}
    far = {"xyxy": (105, 90, 115, 118), "conf": 0.7, "cls": "person", "_clip": False, "_tile": 0}
    assert len(merge([near, near_clip], "M0")) == len(merge([near, near_clip], "M1")) == 1
    assert len(merge([near, far], "M0")) == 1 and len(merge([near, far], "M1")) == 2 and len(merge([near, far], "M2")) == 2
    far_other_tile = {**far, "_tile": 1}
    assert len(merge([near, far_other_tile], "M1")) == 1          # across tiles the IoS rule still applies
    print("n42 demo ok: the shipped merge drops a far player inside a near box; the upgrade path keeps them and still de-duplicates across tiles")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--detect" in a:
        detect(a[a.index("--detect") + 1])
    elif "--caches" in a:
        caches()
    elif "--l1" in a:
        l1()
    elif "--l2" in a:
        l2(a[a.index("--l2") + 1], a[a.index("--l2") + 2])
    elif "--show" in a:
        show()
    else:
        demo()
