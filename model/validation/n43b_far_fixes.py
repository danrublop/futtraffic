"""N43b: the two fixes N43 named, applied offline to the far pass's own boxes. Protocol first.

WHY
    N43's far-third tile pass took far occluded recall 0.4285 -> 0.771 and far isolated 0.8146 -> 0.9923 on the dev
    clips and failed its gate on non-GT boxes (2.651 -> 12.793 per frame). `--why` said what they were: 49% of the
    extra boxes match no ground truth and stand at median bottom row 155 - ON the far touchline: officials, subs
    and spectators inside the pitch prior's 3 m margin, detectable once magnified; and 0.65 shipped boxes per frame
    are lost at the strip's bottom edge (rows 300-333), where the merge keeps the far tile's differently cut box
    (docs/experiments/n43-far-tiles.md). Two rules follow, each stated before it is applied:

    F1  FAR MARGIN. A box whose foot maps to y < -FAR_MARGIN_M (more than half a metre BEYOND the far touchline)
        is dropped. The product's 3 m polygon margin stays everywhere else; this only tightens the far edge, where
        the ground truth's 22 players never stand and the crowd does. Through the clip's frozen zero-click camera.
    F2  BOUNDARY. In the band [strip_bottom - 60, strip_bottom + 40] px of foot row, the shipped grid's boxes replace
        the far pass's: the uniform tile sees a whole player there and the far tile a clipped or re-cut one. Emulated
        exactly offline: the far pass's uniform tiles are the shipped tiles, so its boxes in the band are the shipped
        cache's boxes in the band (T0), and every T1 box in the band is discarded.
    F12 both.
    Constants chosen now, from the `--why` rows, not swept: FAR_MARGIN_M = 0.5, BAND = (-60, +40) px about the strip's
    bottom row from `n43_far_tiles.far_strips`. n = 3 variants.

GATE (L1, dev, N40's recall table after the fixes) - a variant goes to the product path iff, against T0:
    far occluded >= +0.05 AND far isolated >= +0.05; mid and near recalls (occluded and isolated) each >= T0 - 0.01;
    non-GT boxes per frame <= T0 + 1.0 (the far pass legitimately finds officials the shipped grid missed; they are
    people, not players, and the tracker sees them either way).
L2, RULE  N43's: HOTA >= T0 + 0.015 (raised from 0.010 after benchmark-status (not published) measured the spread at 0.012),
    DetPr >= T0 - 0.020 on every clip, IDSW <= 1.15 x T0, no-owner <= T0 + 0.02; metre figures reported, not judged.
    DEV first; HELD once, which needs the far pass on the four held-out clips (about 2.5 h each on this machine).

    .venv/bin/python model/validation/n43b_far_fixes.py --caches       # F1/F2/F12 caches from <seq>.p43t1 + <seq>.p35, dev clips
    .venv/bin/python model/validation/n43b_far_fixes.py --l1           # -> n43b_l1.json
    .venv/bin/python model/validation/n43b_far_fixes.py --l2 F12 dev|held
    .venv/bin/python model/validation/n43b_far_fixes.py --show
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
DEV = ("118575", "118576", "117093")
HELD = ("118577", "118578", "128057", "128058")
FAR_MARGIN_M, BAND = 0.5, (-60, 40)
VARIANTS = ("F1", "F2", "F12")



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


def _strip_bottom(seq):
    import calibrators as C  # noqa: PLC0415
    import n43_far_tiles as N  # noqa: PLC0415
    cal = C.load(os.path.join(STC, f"{seq}.p35.calibration.json"))
    tiles = N.far_strips(cal, tuple(cal.image_wh) if getattr(cal, "image_wh", None) else (4096, 1080))
    return max(t[1] + t[3] for t in tiles), cal


def apply(seq, variant, t1_frames, t0_frames):
    bottom, cal = _strip_bottom(seq)
    lo, hi = bottom + BAND[0], bottom + BAND[1]
    out = {}
    for f, dets in t1_frames.items():
        keep = list(dets)
        if "2" in variant:
            keep = [d for d in keep if not (lo <= d["xyxy"][3] < hi)]
            keep += [d for d in t0_frames.get(f, []) if lo <= d["xyxy"][3] < hi]
        if "1" in variant and keep:
            feet = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]] for d in keep], float)
            y = cal.to_pitch(feet)[:, 1]
            keep = [d for d, yy in zip(keep, y) if not (np.isfinite(yy) and yy < -FAR_MARGIN_M)]
        out[f] = keep
    return out


def caches(seqs=DEV):
    for seq in seqs:
        t1 = json.load(open(os.path.join(STC, f"{seq}.p43t1.dets.cache.json")))
        t0 = json.load(open(os.path.join(STC, f"{seq}.p35.dets.cache.json")))["frames"]
        for v in VARIANTS:
            frames = apply(seq, v, t1["frames"], t0)
            json.dump({**{k: t1[k] for k in ("video", "start", "calib", "calib_sha", "settings", "complete")}, "frames": frames},
                      open(os.path.join(STC, f"{seq}.p43{v.lower()}.dets.cache.json"), "w"))
            print(seq, v, "boxes per frame %.2f" % (sum(len(x) for x in frames.values()) / len(frames)), flush=True)


def l1():
    import n40_occlusion_recall as R  # noqa: PLC0415
    res = {}
    for arm, tag in (("T0", "p35"), ("T1", "p43t1"), *[(v, f"p43{v.lower()}") for v in VARIANTS]):
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
    t0 = res["T0"]
    for v in VARIANTS:
        r = res[v]
        ok = (r["occluded_far"] >= t0["occluded_far"] + 0.05 and r["isolated_far"] >= t0["isolated_far"] + 0.05
              and all(r[k] >= t0[k] - 0.01 for k in ("occluded_mid", "occluded_near", "isolated_mid", "isolated_near"))
              and r["non_gt_per_frame"] <= t0["non_gt_per_frame"] + 1.0)
        res[f"{v}_goes_on"] = bool(ok)
        print(v, "goes on to L2" if ok else "stops at L1")
    json.dump(res, open(os.path.join(STC, "n43b_l1.json"), "w"), indent=1)


def l2(variant, which):
    import n41_conf as N41  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    seqs = HELD if which == "held" else DEV
    if which == "held" and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n43b_far_fixes.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the held-out run must be made by the committed protocol")
    tag = f".p43{variant.lower()}"
    if which == "held":
        caches(HELD)
    for q in seqs:
        cal = ST._p(q, ".p35.calibration.json")
        ST.track(q, cal, tag=tag, cache_from=tag)
        ST.identify(q, cal, src=tag, dst=tag + "id")
    N41.ARMS = {"C50": ("p35", "p35id"), "C30": (tag[1:], tag[1:] + "id")}
    N41.score(which, out=os.path.join(STC, f"n43b_{variant}_{which}.json"))


def why(q):
    """The extra non-GT boxes after both fixes: where on the pitch, how persistent (a bench does not move)."""
    import calibrators as C  # noqa: PLC0415
    import n40_occlusion_recall as R  # noqa: PLC0415
    gt, cal = R._gt(q), C.load(os.path.join(STC, f"{q}.p35.calibration.json"))
    f12 = json.load(open(os.path.join(STC, f"{q}.p43f12.dets.cache.json")))["frames"]
    t0 = json.load(open(os.path.join(STC, f"{q}.p35.dets.cache.json")))["frames"]
    rows = []
    for k, v in f12.items():
        f = int(k)
        if f not in gt:
            continue
        xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
        t0b = [tuple(d["xyxy"]) for d in t0.get(k, [])]
        for d in v:
            b = tuple(d["xyxy"])
            if not any(R._iou(g, b) >= 0.5 for g in xyxy) and not any(R._iou(c, b) >= 0.5 for c in t0b):
                rows.append((f, (b[0] + b[2]) / 2, b[3], b[3] - b[1]))
    a = np.array(rows)
    y = cal.to_pitch(a[:, 1:3])[:, 1]
    cells = {}
    for f, x, yy, h in rows:
        cells.setdefault((int(x // 15), int(yy // 8)), set()).add(f)
    busiest = sorted((len(v) for v in cells.values()), reverse=True)[:5]
    print(f"{q}: extra non-GT boxes per frame {len(a) / len(f12):.2f}; pitch y of their feet p10/p50/p90 {np.nanpercentile(y, [10, 50, 90]).round(1).tolist()} m"
          f" (0 = the far touchline; one far pixel is ~0.5 m); box height median {np.median(a[:, 3]):.0f} px; frames occupied by the five busiest 15x8 px spots {busiest} of {len(f12)}")


def show():
    p = os.path.join(STC, "n43b_l1.json")
    if os.path.exists(p):
        r = json.load(open(p))
        for arm in ("T0", "T1", *VARIANTS):
            print(arm, " ".join(f"{k} {x}" for k, x in r[arm].items()))
        print(" ".join(f"{v}:{'on' if r[f'{v}_goes_on'] else 'stop'}" for v in VARIANTS))
    import glob  # noqa: PLC0415
    for f in sorted(glob.glob(os.path.join(STC, "n43b_F*_*.json"))):
        r = json.load(open(f))
        seqs = list(r["C50"])
        m = lambda a, k: float(np.nanmean([r[a][q][k] for q in seqs]))  # noqa: E731
        keys = ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "IDF1", "IDSW", "ids", "dist_raw", "hsr_raw", "no_owner")
        print(os.path.basename(f), "MEAN " + " ".join(f"{k} {m('C50', k):.4f}->{m('C30', k):.4f}" for k in keys))
        for q in seqs:
            print(f"  {q} " + " ".join(f"{k} {r['C50'][q][k]:.3f}->{r['C30'][q][k]:.3f}" for k in keys))
        if f.endswith("_held.json"):
            ok = [m("C30", "HOTA") >= m("C50", "HOTA") + 0.015, all(r["C30"][q]["DetPr"] >= r["C50"][q]["DetPr"] - 0.020 for q in seqs),
                  m("C30", "IDSW") <= 1.15 * m("C50", "IDSW"), m("C30", "no_owner") <= m("C50", "no_owner") + 0.02]
            print("  RULE:", "REPLACES the shipped grid" if all(ok) else f"NOT SUPPORTED {ok}")


def demo():
    import calibrators as C  # noqa: PLC0415
    seq = "118575"
    bottom, cal = _strip_bottom(seq)
    far_row = float(cal.to_image(np.array([[52.5, 0.0]]))[0, 1])
    beyond = {"xyxy": [2000, far_row - 40, 2010, far_row - 6], "conf": 0.6, "cls": "person"}     # 6 px beyond the line: ~3.5 m out
    on = {"xyxy": [2100, far_row - 30, 2110, far_row + 1], "conf": 0.6, "cls": "person"}          # on the line
    band = {"xyxy": [1500, bottom - 60, 1520, bottom - 10], "conf": 0.9, "cls": "person"}         # far-tile box in the band
    t0b = {"xyxy": [1502, bottom - 62, 1522, bottom - 9], "conf": 0.8, "cls": "person"}
    f1 = apply(seq, "F1", {"0": [beyond, on, band]}, {"0": [t0b]})["0"]
    assert on in f1 and beyond not in f1, [d["xyxy"] for d in f1]
    f2 = apply(seq, "F2", {"0": [beyond, on, band]}, {"0": [t0b]})["0"]
    assert band not in f2 and t0b in f2 and on in f2
    print(f"n43b demo ok: a box 3 m beyond the far line is dropped, one on the line kept; in the band ({bottom + BAND[0]}-{bottom + BAND[1]} px) the shipped box replaces the far tile's")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--caches" in a:
        caches()
    elif "--l1" in a:
        l1()
    elif "--l2" in a:
        l2(a[a.index("--l2") + 1], a[a.index("--l2") + 2])
    elif "--why" in a:
        why(a[a.index("--why") + 1])
    elif "--show" in a:
        show()
    else:
        demo()
