"""N43: spend detector pixels where the players are small - a far-band tile pass. Protocol first.

WHY
    Occluded and far-third players are detected weakly (docs/experiments/n40-association.md §2: isolated far recall 0.64-0.88,
    occluded far 0.31-0.49); a lower confidence (N41) and the box merge (N42) did not fix it. The far third of the
    pitch reaches the detector at 20-40 px per player through a 1024 px tile downscaled to the model's 576 input,
    i.e. 0.56x. model/engine/detectors.geometry_tiles already builds strips of small tiles over the far field so a player
    lands ~target_px tall in the model input; it was measured on 40 frames of the remapped 118575 canvas with the
    retired height gate on (far-touchline recall 0.414 -> 0.931 at 2.06x the tiles - its own docstring) and never on
    the raw panorama, never through the product path, never on more than 40 frames. That is what this measures.

ARMS  (same detector, conf 0.5, same merge, same prior, stride 2; the only change is the tile plan)
    T0   the shipped uniform 1024 grid: the `.p35` cache (5 tiles per frame on 4096x1080).
    T1   the uniform grid PLUS far strips from the zero-click camera's own geometry over the FAR THIRD of the pitch
         (ground points with y < 68/3 m from the far touchline), tiles sized so a player there lands ~96 px in the
         576 input, grouped by octave as geometry_tiles does. Expected heights come from the spherical camera (a
         1.8 m post projected at each ground point), because geometry_tiles' pinhole front end does not apply to
         this camera; the strip logic is the same. The tile count is printed: the compute cost is the price.
    n = 1 candidate plan. DISCLOSED: the first draft of this file used the docstring's "under 48 px, target 128";
         on the raw 4096x1080 panorama that covers rows 107-626 with 132 tiles (27x the uniform grid's 5), because
         most of this pitch is under 48 px tall - so the plan was re-cut to the far third and target 96 BEFORE any
         detector pass ran. No result was read in between.

SPLIT (N40-N42's).  DEV = 118575, 118576, 117093.  HELD = 118577, 118578, 128057, 128058, ONCE.

LEVELS AND RULE
    L1  N40's recall by class and non-GT boxes per frame, T0 vs T1, DEV (from the caches, after the product's prior).
        T1 goes on iff far occluded AND far isolated recall both rise by >= 0.05 with non-GT boxes per frame within
        +0.5 of T0. Otherwise it stops here, published.
    L2  the product path (tracker replay -> stitch -> identity), N41's scorer. RULE on HELD, N41's, with ONE change
        learned in N42: the two metre figures (distance raw, HSR raw) are reported but NOT part of the pass/fail,
        because on four-minute clips they swing 0.05-0.1 between near-identical runs (docs/experiments/n42-merge.md); pass/fail is
        HOTA >= T0 + 0.010, DetPr >= T0 - 0.020 on every clip, IDSW <= 1.15 x T0, no-owner <= T0 + 0.02.

    .venv/bin/python model/validation/n43_far_tiles.py --plan 118575      # prints the far strips and the tile count, no GPU
    .venv/bin/python model/validation/n43_far_tiles.py --detect 118575    # T1 pass -> <seq>.p43t1.dets.cache.json (after the prior)
    .venv/bin/python model/validation/n43_far_tiles.py --l1               # DEV recalls -> n43_l1.json
    .venv/bin/python model/validation/n43_far_tiles.py --l2 dev|held
    .venv/bin/python model/validation/n43_far_tiles.py --show
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
STRIDE, CONF, TARGET_PX, FAR_M, MODEL_IN = 2, 0.5, 96, 68.0 / 3, 576



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


def far_strips(cal, wh, target_px=TARGET_PX, far_m=FAR_M, overlap=None, margin_m=3.0):
    """geometry_tiles' strip logic on the spherical camera: (x, y, w, h) tiles over the far third of the pitch only."""
    import autocal as A  # noqa: PLC0415
    import detectors as DT  # noqa: PLC0415
    overlap = DT.OVERLAP if overlap is None else overlap
    w, h = wh
    L, W = cal.pitch.length, cal.pitch.width
    gx, gy = np.meshgrid(np.linspace(-margin_m, L + margin_m, 80), np.linspace(-margin_m, far_m, 30))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    q = np.asarray(cal.camera, float)
    foot = A._fwd(q, xy, float(w))
    head = A._fwd(q, np.column_stack([xy, np.full(len(xy), 1.8)]), float(w))
    ht = foot[:, 1] - head[:, 1]
    ok = np.isfinite(ht) & (ht > 0) & (foot[:, 0] > -w) & (foot[:, 0] < 2 * w) & (foot[:, 1] > -h) & (foot[:, 1] < 2 * h)
    uv, ht = foot[ok], ht[ok]
    tiles = []
    if not len(uv):
        return tiles
    oct_idx = np.floor(np.log2(ht / ht.min())).astype(int)
    for o in np.unique(oct_idx):
        s = oct_idx == o
        side = int(round(MODEL_IN * float(np.median(ht[s])) / target_px))      # a player of this strip's height -> target_px in the input
        x0, x1 = int(np.floor(uv[s, 0].min())), int(np.ceil(uv[s, 0].max()))
        y0, y1 = int(np.floor((uv[s, 1] - 1.5 * ht[s]).min())), int(np.ceil(uv[s, 1].max()))
        tiles += DT._strip_tiles(max(0, x0), min(w, x1), max(0, y0), min(h, y1), side, overlap, w, h)
    return tiles


def plan(seq):
    import calibrators as C  # noqa: PLC0415
    import detectors as DT  # noqa: PLC0415
    cal = C.load(os.path.join(STC, f"{seq}.p35.calibration.json"))
    wh = tuple(cal.image_wh) if getattr(cal, "image_wh", None) else (4096, 1080)
    uni = DT.TiledDetector(DT.StubDetector(), tile=1024).grid(*wh)
    far = far_strips(cal, wh)
    sides = sorted({t[2] for t in far})
    print(f"{seq}: uniform {len(uni)} tiles + far {len(far)} tiles (sides {sides}) = {len(uni) + len(far)}, cost x{(len(uni) + len(far)) / len(uni):.2f}")
    return uni, far, wh


def detect(seq):
    import calibrators as C  # noqa: PLC0415
    import detectors as DT  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    out = os.path.join(STC, f"{seq}.p43t1.dets.cache.json")
    if os.path.exists(out) and json.load(open(out)).get("complete"):
        print("have", out); return
    uni, far, wh = plan(seq)
    cal = C.load(os.path.join(STC, f"{seq}.p35.calibration.json"))
    poly = np.asarray(cal.pitch_polygon_px(), np.float32)
    inside = lambda b: cv2.pointPolygonTest(poly, (float((b[0] + b[2]) / 2), float(b[3])), False) >= 0  # noqa: E731
    head = json.load(open(os.path.join(STC, f"{seq}.p35.dets.cache.json")))
    head = {k: head[k] for k in ("video", "start", "calib", "calib_sha", "settings")}
    det = DT.TiledDetector(DT.CoreMLDetector(conf=CONF), plan=uni + far, batch=16)
    cap = cv2.VideoCapture(f"{ST.CLIPS}/{seq}.mp4")
    frames, f = {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % STRIDE == 0:
            frames[str(f)] = [{"xyxy": [float(v) for v in d["xyxy"]], "conf": float(d["conf"]), "cls": d.get("cls")}
                              for d in det.detect(img) if inside(d["xyxy"])]         # the product's prior, as its cache carries it
            if f % 400 == 0:
                print(f"  frame {f}", flush=True)
        f += 1
    json.dump({**head, "complete": True, "frames": frames}, open(out, "w"))
    print("->", out)


def l1():
    import n40_occlusion_recall as R  # noqa: PLC0415
    res = {}
    for arm, tag in (("T0", "p35"), ("T1", "p43t1")):
        cnt = {k: [0, 0] for k in ("occluded_far", "occluded_mid", "occluded_near", "isolated_far", "isolated_mid", "isolated_near")}
        fp = nf = 0
        for q in DEV:
            gt = R._gt(q)
            frames = {int(k): [d["xyxy"] for d in v if d.get("cls") == "person"] for k, v in json.load(open(os.path.join(STC, f"{q}.{tag}.dets.cache.json")))["frames"].items()}
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
        res[arm] = {**{k: round(v[1] / v[0], 4) for k, v in cnt.items()}, "non_gt_per_frame": round(fp / nf, 3)}
        print(arm, res[arm], flush=True)
    a, b = res["T0"], res["T1"]
    res["goes_on"] = bool(b["occluded_far"] >= a["occluded_far"] + 0.05 and b["isolated_far"] >= a["isolated_far"] + 0.05 and b["non_gt_per_frame"] <= a["non_gt_per_frame"] + 0.5)
    print("L1 selection: T1 goes on to L2" if res["goes_on"] else "L1 selection: T1 stops here")
    json.dump(res, open(os.path.join(STC, "n43_l1.json"), "w"), indent=1)


def l2(which):
    import n41_conf as N41  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    seqs = HELD if which == "held" else DEV
    if which == "held" and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n43_far_tiles.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the held-out run must be made by the committed protocol")
    for q in seqs:
        cal = ST._p(q, ".p35.calibration.json")
        ST.track(q, cal, tag=".p43t1", cache_from=".p43t1")
        ST.identify(q, cal, src=".p43t1", dst=".p43t1id")
    N41.ARMS = {"C50": ("p35", "p35id"), "C30": ("p43t1", "p43t1id")}
    N41.score(which, out=os.path.join(STC, f"n43_{which}.json"))


def why(q="118575"):
    """What the far pass adds and loses, one clip: extra boxes by their best GT IoU, lost boxes by row."""
    import n40_occlusion_recall as R  # noqa: PLC0415
    gt = R._gt(q)
    t0 = {int(k): [tuple(d["xyxy"]) for d in v] for k, v in json.load(open(os.path.join(STC, f"{q}.p35.dets.cache.json")))["frames"].items()}
    t1 = {int(k): [tuple(d["xyxy"]) for d in v] for k, v in json.load(open(os.path.join(STC, f"{q}.p43t1.dets.cache.json")))["frames"].items()}
    lost_rows, extra_iou, extra_rows, nf = [], [], [], 0
    for f in t0:
        if f not in gt:
            continue
        nf += 1
        xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
        lost_rows += [b[3] for b in t0[f] if not any(R._iou(b, c) >= 0.5 for c in t1[f])]
        for b in t1[f]:
            if not any(R._iou(b, c) >= 0.5 for c in t0[f]):
                extra_iou.append(max((R._iou(b, g) for g in xyxy), default=0.0))
                extra_rows.append(b[3])
    e, r = np.array(extra_iou), np.array(extra_rows)
    print(f"{q}: extra boxes per frame {len(e) / nf:.2f}: true new players (GT IoU >= 0.5) {np.mean(e >= 0.5):.2f}, near-miss (0.2-0.5) {np.mean((e >= 0.2) & (e < 0.5)):.2f}, no GT (< 0.2) {np.mean(e < 0.2):.2f};"
          f" no-GT boxes' bottom row median {np.median(r[e < 0.2]):.0f} (the far touchline is near row 150)")
    print(f"{q}: shipped boxes lost per frame {len(lost_rows) / nf:.2f}, bottom rows p10/p50/p90 {np.percentile(lost_rows, [10, 50, 90]).round(0).tolist()} (the far strip ends near row 300)")


def show():
    p = os.path.join(STC, "n43_l1.json")
    if os.path.exists(p):
        r = json.load(open(p))
        for arm in ("T0", "T1"):
            print(arm, " ".join(f"{k} {x}" for k, x in r[arm].items()))
        print("goes_on", r["goes_on"])
    for which in ("dev", "held"):
        f = os.path.join(STC, f"n43_{which}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        seqs = list(r["C50"])
        m = lambda a, k: float(np.nanmean([r[a][q][k] for q in seqs]))  # noqa: E731
        keys = ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "IDF1", "IDSW", "dist_raw", "hsr_raw", "no_owner")
        print(f"{which.upper()} MEAN " + " ".join(f"{k} {m('C50', k):.4f}->{m('C30', k):.4f}" for k in keys))
        if which == "held":
            ok = [m("C30", "HOTA") >= m("C50", "HOTA") + 0.010, all(r["C30"][q]["DetPr"] >= r["C50"][q]["DetPr"] - 0.020 for q in seqs),
                  m("C30", "IDSW") <= 1.15 * m("C50", "IDSW"), m("C30", "no_owner") <= m("C50", "no_owner") + 0.02]
            print("RULE:", "REPLACES the uniform grid" if all(ok) else f"NOT SUPPORTED {ok}")


def demo():
    """A synthetic camera 8 m up and 20 m back: the far strips must lie in the top of the frame, cover the far
    touchline's columns, and use tiles smaller than 1024; the near half gets none."""
    import autocal as A  # noqa: PLC0415
    import calibrators as C  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415
    P, wh = Pitch(105.0, 68.0), (4096, 1080)
    q = A.camera(P, wh, fov_deg=156.0, tilt_deg=19.5, height_m=7.3, behind_m=16.0, along_m=-2.5, pan_deg=1.7)
    cal = C.SphericalCamera(P, q, wh)
    tiles = far_strips(cal, wh)
    assert tiles and all(t[2] < 1024 for t in tiles), tiles[:3]
    far_row = A._fwd(q, np.array([[52.5, 0.0]]), 4096.0)[0, 1]
    near_row = A._fwd(q, np.array([[52.5, 68.0]]), 4096.0)[0, 1]
    assert min(t[1] for t in tiles) <= far_row <= max(t[1] + t[3] for t in tiles)
    third_row = A._fwd(q, np.array([[52.5, 68.0 / 3]]), 4096.0)[0, 1]
    assert max(t[1] + t[3] for t in tiles) < third_row + max(t[3] for t in tiles) < near_row   # the far third, plus one tile of clamp
    print(f"n43 demo ok: {len(tiles)} far tiles, sides {sorted({t[2] for t in tiles})}, rows {min(t[1] for t in tiles)}-{max(t[1] + t[3] for t in tiles)} of 1080")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--plan" in a:
        plan(a[a.index("--plan") + 1])
    elif "--detect" in a:
        detect(a[a.index("--detect") + 1])
    elif "--l1" in a:
        l1()
    elif "--l2" in a:
        l2(a[a.index("--l2") + 1])
    elif "--why" in a:
        why(a[a.index("--why") + 1])
    elif "--show" in a:
        show()
    else:
        demo()
