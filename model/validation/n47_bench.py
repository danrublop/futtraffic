"""N47: the far-side bench, cut by the paint and by persistence - not by the map. Protocol first.

WHY
    N43's far-third tile pass doubles far recall (occluded 0.43 -> 0.77, isolated 0.81 -> 0.99 on the dev clips) and
    N43b found what stops it shipping: the extra boxes are the BENCH - subs and staff on the steps just behind the far
    touchline, static for thousands of frames. Their feet map 0.3 m INSIDE the pitch because one far pixel is ~0.5 m, so
    no metre margin through the map can separate them from players on the line (docs/experiments/n43b-far-fixes.md). That page
    named the two tests that do not go through the map; this runs them, on N43b's F12 boxes (the far pass with the
    strip-boundary fix and the 0.5 m margin), the dev clips, offline.

    B1  PAINT. The far touchline's paint row per image column, found on the clip's 40-frame median (players gone) by
        model/engine/autocal.search's grass-flanked perimeter rule along the zero-click camera's line (N37's paint_rows). A
        box whose foot row is ABOVE that row by more than TOL px stands beyond the line - off the pitch, whatever the map
        says. Columns without paint found within 20 px on both sides: the box is kept (no evidence is not evidence).
        TOL is set from the PLAYERS, never from the bench: the 99.5th percentile of (paint row - GT foot row) over the
        dev clips' ground-truth boxes whose foot is within 30 px of the far paint, floored at 1.5 px. So at most about
        0.5% of real players standing at the line can be cut, by construction, and the bench's own depth plays no part.
    B2  STATIC. A box whose foot is within FAR_BAND px of the far paint row and whose CELL x CELL px foot cell is occupied
        in >= OCC of the sampled frames within +-WIN video frames is dropped: a spot held for most of a minute next to
        the far line is a bench, not play. Only the far band, so a goalkeeper holding his area is never at risk. Known
        cost, stated before the run: a player crossing a held cell is cut for the frames he is inside it - the F12
        recall line of the gate below is what measures that.
    B12 both.
    Constants fixed now, not swept: FAR_BAND 20 px, CELL 15 x 8 px (N43b's --why cells), WIN 750 frames (30 s either
    side), OCC 0.5, TOL by the rule above. n = 3 arms.

GATE (L1, dev, N40's recall classes, IoU 0.5), an arm goes to the product path iff:
    against T0 (shipped): N43b's gate - far occluded >= +0.05 AND far isolated >= +0.05; mid and near (occluded and
        isolated) each >= T0 - 0.01; non-GT boxes per frame <= T0 + 1.0;
    against F12 (what the filter is applied to): far occluded and far isolated each >= F12 - 0.01 - a filter that
        removes players is deleting the hard inputs, and that is the change this repo distrusts most.
    Also printed per clip: boxes dropped per frame, the share of F12's GT-matched boxes dropped (must read near 0), the
    far paint's coverage of the image width, and TOL.
L2  the product path on the dev clips (EXPLORATORY, as N43b's was): N43's rule read against T0 - HOTA >= +0.015,
    DetPr >= -0.020 on every clip, IDSW <= 1.15x, no-owner <= +0.02. HELD needs the far pass run on the four held-out
    clips (about 10 h of detection on this machine) and is not run from here.

SECOND ROUND, B3 and B4 - written after the first round's result (docs/experiments/n47-bench.md) and a look at the leftover boxes
    (the assistant referee with his flag, substitutes in bibs warming up, staff and spectators on the steps just behind
    the line, many in loose groups that shift across cells), committed before either runs. n = 2 rounds of this rule.
    B3  PAINT FILLED. B12 with the far paint row known at every column the zero-click camera's far touchline covers:
        the camera's line plus the found paint's residual (paint row - camera row at each found sample), a running
        median over RES_MED samples, interpolated linearly across gaps and held flat past the ends. Everything else in
        B1 and B2 unchanged (TOL, FAR_BAND, CELL, WIN, OCC). Gate: the L1 gate above, unchanged.
    B4  LINE-HUGGER, track level, on B3's product-path tracks. A raw track is dropped iff it lasts >= HUG_S seconds and
        the 95th percentile of (foot row - filled paint row) over its rows is <= HUG_PX - it never goes more than about
        three metres into the pitch. The assistant referee runs that line for a half; a player who reaches it leaves it.
        Dropped from the raw tracks and, through raw_id, from the identified ones.
        Gate (product path, dev, EXPLORATORY as above): N43's rule against T0 - HOTA >= +0.015, DetPr >= T0 - 0.020 on
        every clip, IDSW <= 1.15x, no-owner <= +0.02 - AND DetRe of B34 >= DetRe of B3 - 0.005 on every clip: removing
        whole tracks may not remove players.
    Constants fixed now: RES_MED 15 samples, HUG_S 10 s, HUG_PX 6 (one far pixel is about half a metre).

    .venv/bin/python model/validation/n47_bench.py --caches        # TOL, then B1/B2/B12 caches from <seq>.p43f12, dev clips
    .venv/bin/python model/validation/n47_bench.py --b3            # B3 caches + its L1 row (appended to n47_l1.json)
    .venv/bin/python model/validation/n47_bench.py --b4            # B3 and B34 through the product path -> n47_B3_dev.json, n47_B34_dev.json
    .venv/bin/python model/validation/n47_bench.py --l1            # -> model/validation/out/stc2025/n47_l1.json
    .venv/bin/python model/validation/n47_bench.py --l2 B12        # product path, dev, exploratory
    .venv/bin/python model/validation/n47_bench.py --why 118576    # what B12 leaves, against the paint
    .venv/bin/python model/validation/n47_bench.py --show          # the L1 table
    .venv/bin/python model/validation/n47_bench.py --show4         # B3 / B34 product path from the stored runs
    .venv/bin/python model/validation/n47_bench.py                 # self-check
"""
import bisect
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
DEV = ("118575", "118576", "117093")
ARMS = ("B1", "B2", "B12")
FAR_BAND, CELL, WIN, OCC = 20.0, (15, 8), 750, 0.5
TOL_FLOOR, TOL_PCT, TOL_NEAR, GAP = 1.5, 99.5, 30.0, 20.0
TOL_FILE = os.path.join(STC, "n47_tol.json")
RES_MED, HUG_S, HUG_PX = 15, 10.0, 6.0


def far_paint(seq):
    """(u sorted, v) of the far touchline's paint on the clip's median image."""
    import cv2  # noqa: PLC0415
    import n37_feet_on_lines as N37  # noqa: PLC0415
    bgr = cv2.imread(os.path.join(STC, f"{seq}.median40.png"))
    cam = np.array(json.load(open(os.path.join(STC, N37.ARMS["SHIP"])))[seq]["camera"], float)
    return N37.paint_rows(bgr, cam)["touch_far"]


def filled_paint(seq, found=None):
    """B3: the far paint row at every integer column the camera's far touchline covers - camera line + the found
    paint's residual, running-median smoothed and interpolated across the gaps."""
    import n37_feet_on_lines as N37  # noqa: PLC0415
    cam = np.array(json.load(open(os.path.join(STC, N37.ARMS["SHIP"])))[seq]["camera"], float)
    line = N37.line_pixels(cam, 4096)["touch_far"]
    return fill(line, found if found is not None else far_paint(seq))


def fill(line, found):
    """(camera line points (N,2), (pu, pv) found paint) -> (u every column in the line's span, v)."""
    line = line[np.isfinite(line).all(1)]
    line = line[np.argsort(line[:, 0])]
    uc = np.arange(int(np.ceil(line[0, 0])), int(np.floor(line[-1, 0])) + 1, dtype=float)
    vc = np.interp(uc, line[:, 0], line[:, 1])
    pu, pv = found
    r = pv - np.interp(pu, line[:, 0], line[:, 1])
    k = RES_MED // 2
    r = np.array([np.median(r[max(0, i - k):i + k + 1]) for i in range(len(r))])
    return uc, vc + np.interp(uc, pu, r)


def huggers(rows, paint, fps=25.0):
    """B4: raw track ids whose feet stay within HUG_PX of the far paint for >= HUG_S seconds (95th percentile)."""
    by = {}
    for r in rows:
        v = paint_at(r["x"], *paint)
        if v is not None:
            by.setdefault(r["id"], []).append((r["f"], r["y"] - v))
    out = set()
    for tid, fd in by.items():
        f = [a for a, _ in fd]
        if (max(f) - min(f)) / fps >= HUG_S and np.percentile([d for _, d in fd], 95) <= HUG_PX:
            out.add(tid)
    return out


def paint_at(u, pu, pv):
    """The paint row at column u, or None when no paint was found within GAP px on both sides."""
    i = int(np.searchsorted(pu, u))
    if i == 0 or i == len(pu) or u - pu[i - 1] > GAP or pu[i] - u > GAP:
        return None
    return float(np.interp(u, pu, pv))


def tol_from_gt(paints):
    """TOL from the players: the TOL_PCT percentile of (paint row - GT foot row) near the far paint, floored."""
    import n40_occlusion_recall as R  # noqa: PLC0415
    d = []
    for q in DEV:
        pu, pv = paints[q]
        for fr in R._gt(q).values():
            for _i, x, y, w, h in fr:
                v = paint_at(x + w / 2, pu, pv)
                if v is not None and abs(y + h - v) <= TOL_NEAR:
                    d.append(v - (y + h))
    return max(TOL_FLOOR, float(np.percentile(d, TOL_PCT))), len(d)


def static_mask(frames, paint):
    """{frame: [bool per box]} - True where B2 drops the box. `paint` = (pu, pv)."""
    keys = sorted(frames, key=int)
    fr_ids = [int(k) for k in keys]
    occ, where = {}, {}
    for k in keys:
        for j, d in enumerate(frames[k]):
            x1, _y1, x2, y2 = d["xyxy"]
            u = (x1 + x2) / 2
            v = paint_at(u, *paint)
            if v is None or abs(y2 - v) > FAR_BAND:
                continue
            c = (int(u // CELL[0]), int(y2 // CELL[1]))
            occ.setdefault(c, []).append(int(k))
            where[(k, j)] = c
    for c in occ:
        occ[c] = sorted(set(occ[c]))
    out = {}
    for k in keys:
        f = int(k)
        n = bisect.bisect_right(fr_ids, f + WIN) - bisect.bisect_left(fr_ids, f - WIN)
        m = []
        for j in range(len(frames[k])):
            c = where.get((k, j))
            if c is None:
                m.append(False)
                continue
            o = occ[c]
            m.append((bisect.bisect_right(o, f + WIN) - bisect.bisect_left(o, f - WIN)) >= OCC * n)
        out[k] = m
    return out


def apply(arm, frames, paint, tol):
    drop = static_mask(frames, paint) if "2" in arm else None
    out = {}
    for k, dets in frames.items():
        keep = []
        for j, d in enumerate(dets):
            x1, _y1, x2, y2 = d["xyxy"]
            if "1" in arm:
                v = paint_at((x1 + x2) / 2, *paint)
                if v is not None and v - y2 > tol:
                    continue
            if drop is not None and drop[k][j]:
                continue
            keep.append(d)
        out[k] = keep
    return out


def caches():
    paints = {q: far_paint(q) for q in DEV}
    tol, n = tol_from_gt(paints)
    cover = {q: round(float(np.mean([paint_at(u, *paints[q]) is not None for u in range(0, 4096, 8)])), 3) for q in DEV}
    json.dump({"tol_px": round(tol, 3), "gt_boxes_used": n, "far_paint_width_cover": cover}, open(TOL_FILE, "w"), indent=1)
    print(f"TOL {tol:.2f} px from {n} GT boxes near the far paint; far paint covers {cover} of the image width", flush=True)
    for q in DEV:
        f12 = json.load(open(os.path.join(STC, f"{q}.p43f12.dets.cache.json")))
        for arm in ARMS:
            frames = apply(arm, f12["frames"], paints[q], tol)
            json.dump({**{k: f12[k] for k in ("video", "start", "calib", "calib_sha", "settings", "complete")}, "frames": frames},
                      open(os.path.join(STC, f"{q}.p47{arm.lower()}.dets.cache.json"), "w"))
            print(q, arm, "boxes per frame %.2f -> %.2f" % (sum(map(len, f12["frames"].values())) / len(f12["frames"]),
                                                            sum(map(len, frames.values())) / len(frames)), flush=True)


def b3():
    tol = json.load(open(TOL_FILE))["tol_px"]
    for q in DEV:
        f12 = json.load(open(os.path.join(STC, f"{q}.p43f12.dets.cache.json")))
        frames = apply("B12", f12["frames"], filled_paint(q), tol)
        json.dump({**{k: f12[k] for k in ("video", "start", "calib", "calib_sha", "settings", "complete")}, "frames": frames},
                  open(os.path.join(STC, f"{q}.p47b3.dets.cache.json"), "w"))
        print(q, "B3 boxes per frame %.2f" % (sum(map(len, frames.values())) / len(frames)), flush=True)
    res = json.load(open(os.path.join(STC, "n47_l1.json")))
    res["B3"] = recall("p47b3")
    res["B3_goes_on"] = gate(res["B3"], res["T0"], res["F12"])
    print("B3", res["B3"], "goes on" if res["B3_goes_on"] else "stops at L1")
    json.dump(res, open(os.path.join(STC, "n47_l1.json"), "w"), indent=1)


def b4():
    import n41_conf as N41  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    for q in DEV:
        cal = ST._p(q, ".p35.calibration.json")
        ST.track(q, cal, tag=".p47b3", cache_from=".p47b3")
        ST.identify(q, cal, src=".p47b3", dst=".p47b3id")
        raw = json.load(open(os.path.join(STC, f"{q}.p47b3.tracks.json")))
        ided = json.load(open(os.path.join(STC, f"{q}.p47b3id.tracks.json")))
        drop = huggers(raw["detections"], filled_paint(q), raw.get("fps", 25.0))
        print(q, "line-hugger tracks", len(drop), "of", len({r["id"] for r in raw["detections"]}), flush=True)
        for src, dst in ((raw, "p47b34"), (ided, "p47b34id")):
            key = "id" if src is raw else "raw_id"
            json.dump({**src, "detections": [r for r in src["detections"] if r[key] not in drop]},
                      open(os.path.join(STC, f"{q}.{dst}.tracks.json"), "w"))
    for arm in ("B3", "B34"):
        N41.ARMS = {"C50": ("p35", "p35id"), "C30": (f"p47{arm.lower()}", f"p47{arm.lower()}id")}
        N41.score("dev", out=os.path.join(STC, f"n47_{arm}_dev.json"))
    r3 = json.load(open(os.path.join(STC, "n47_B3_dev.json")))
    r4 = json.load(open(os.path.join(STC, "n47_B34_dev.json")))
    m = lambda r, a, k: float(np.mean([r[a][q][k] for q in DEV]))  # noqa: E731
    ok = (m(r4, "C30", "HOTA") >= m(r4, "C50", "HOTA") + 0.015
          and all(r4["C30"][q]["DetPr"] >= r4["C50"][q]["DetPr"] - 0.020 for q in DEV)
          and m(r4, "C30", "IDSW") <= 1.15 * m(r4, "C50", "IDSW")
          and m(r4, "C30", "no_owner") <= m(r4, "C50", "no_owner") + 0.02
          and all(r4["C30"][q]["DetRe"] >= r3["C30"][q]["DetRe"] - 0.005 for q in DEV))
    for q in DEV:
        print(q, " ".join(f"{k} {r4['C50'][q][k]:.3f} / B3 {r3['C30'][q][k]:.3f} / B34 {r4['C30'][q][k]:.3f}"
                          for k in ("HOTA", "DetRe", "DetPr", "AssA", "IDSW", "no_owner")))
    print("B34", "PASSES the product rule (dev, exploratory)" if ok else "does NOT pass the product rule (dev, exploratory)")


def show4():
    """B3 and B34 against T0 from the stored product-path runs, with the B4 gate's parts - no recomputation."""
    r3 = json.load(open(os.path.join(STC, "n47_B3_dev.json")))
    r4 = json.load(open(os.path.join(STC, "n47_B34_dev.json")))
    m = lambda r, a, k: float(np.mean([r[a][q][k] for q in DEV]))  # noqa: E731
    for k in ("HOTA", "DetRe", "DetPr", "AssA", "IDSW", "no_owner"):
        print(f"mean {k}: T0 {m(r4, 'C50', k):.4f}  B3 {m(r3, 'C30', k):.4f}  B34 {m(r4, 'C30', k):.4f}")
    print("gate: HOTA +%.4f (>= 0.015); worst-clip DetPr %.4f (>= -0.020); IDSW x%.3f (<= 1.15); no-owner +%.4f (<= 0.02); "
          "DetRe B34 - B3 worst %.4f (>= -0.005)" % (
              m(r4, "C30", "HOTA") - m(r4, "C50", "HOTA"), min(r4["C30"][q]["DetPr"] - r4["C50"][q]["DetPr"] for q in DEV),
              m(r4, "C30", "IDSW") / m(r4, "C50", "IDSW"), m(r4, "C30", "no_owner") - m(r4, "C50", "no_owner"),
              min(r4["C30"][q]["DetRe"] - r3["C30"][q]["DetRe"] for q in DEV)))


def recall(tag):
    """N40's classes pooled over DEV, plus non-GT per frame - N43b's l1 counting, for one cache tag."""
    import n40_occlusion_recall as R  # noqa: PLC0415
    cnt = {k: [0, 0] for k in ("occluded_far", "occluded_mid", "occluded_near", "isolated_far", "isolated_mid", "isolated_near")}
    fp = nf = 0
    for q in DEV:
        gt = R._gt(q)
        frames = {int(k): [d["xyxy"] for d in v if d.get("cls") == "person"] for k, v in json.load(open(os.path.join(STC, f"{q}.{tag}.dets.cache.json")))["frames"].items()}
        th = np.percentile([g[2] + g[4] for fr in gt.values() for g in fr], [33.3, 66.7])
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
    return {**{k: round(v[1] / v[0], 4) for k, v in cnt.items()}, "non_gt_per_frame": round(fp / nf, 3)}


def gate(r, t0, f12):
    return bool(r["occluded_far"] >= t0["occluded_far"] + 0.05 and r["isolated_far"] >= t0["isolated_far"] + 0.05
                and all(r[k] >= t0[k] - 0.01 for k in ("occluded_mid", "occluded_near", "isolated_mid", "isolated_near"))
                and r["non_gt_per_frame"] <= t0["non_gt_per_frame"] + 1.0
                and r["occluded_far"] >= f12["occluded_far"] - 0.01 and r["isolated_far"] >= f12["isolated_far"] - 0.01)


def dropped(q, arm):
    """Per clip: boxes the arm dropped per frame, and the share of F12's GT-matched boxes among them (must read near 0)."""
    import n40_occlusion_recall as R  # noqa: PLC0415
    gt = R._gt(q)
    f12 = json.load(open(os.path.join(STC, f"{q}.p43f12.dets.cache.json")))["frames"]
    got = json.load(open(os.path.join(STC, f"{q}.p47{arm.lower()}.dets.cache.json")))["frames"]
    n_drop = n_match = n_match_drop = 0
    for k, v in f12.items():
        kept = {tuple(d["xyxy"]) for d in got.get(k, [])}
        xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt.get(int(k), [])]
        for d in v:
            b = tuple(d["xyxy"])
            hit = any(R._iou(g, b) >= 0.5 for g in xyxy)
            n_match += hit
            if b not in kept:
                n_drop += 1
                n_match_drop += hit
    return {"dropped_per_frame": round(n_drop / len(f12), 3), "gt_matched_share_dropped": round(n_match_drop / max(1, n_match), 4)}


def l1():
    res = {arm: recall(tag) for arm, tag in (("T0", "p35"), ("F12", "p43f12"), *[(a, f"p47{a.lower()}") for a in ARMS])}
    for a in ARMS:
        res[f"{a}_per_clip"] = {q: dropped(q, a) for q in DEV}
        print(a, "per clip", res[f"{a}_per_clip"], flush=True)
    for arm, r in res.items():
        print(arm, r, flush=True)
    for a in ARMS:
        res[f"{a}_goes_on"] = gate(res[a], res["T0"], res["F12"])
        print(a, "goes on to L2" if res[f"{a}_goes_on"] else "stops at L1")
    res["tol"] = json.load(open(TOL_FILE))
    json.dump(res, open(os.path.join(STC, "n47_l1.json"), "w"), indent=1)


def l2(arm):
    import n41_conf as N41  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    tag = f".p47{arm.lower()}"
    for q in DEV:
        cal = ST._p(q, ".p35.calibration.json")
        ST.track(q, cal, tag=tag, cache_from=tag)
        ST.identify(q, cal, src=tag, dst=tag + "id")
    N41.ARMS = {"C50": ("p35", "p35id"), "C30": (tag[1:], tag[1:] + "id")}
    N41.score("dev", out=os.path.join(STC, f"n47_{arm}_dev.json"))


def why(q):
    """What B12 leaves: the extra non-GT boxes (in neither the GT nor the shipped cache), where they stand against the paint."""
    import n40_occlusion_recall as R  # noqa: PLC0415
    gt, (pu, pv) = R._gt(q), far_paint(q)
    b12 = json.load(open(os.path.join(STC, f"{q}.p47b12.dets.cache.json")))["frames"]
    t0 = json.load(open(os.path.join(STC, f"{q}.p35.dets.cache.json")))["frames"]
    rows = []
    for k, v in b12.items():
        xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt.get(int(k), [])]
        t0b = [tuple(d["xyxy"]) for d in t0.get(k, [])]
        for d in v:
            b = tuple(d["xyxy"])
            if any(R._iou(g, b) >= 0.5 for g in xyxy) or any(R._iou(c, b) >= 0.5 for c in t0b):
                continue
            p = paint_at((b[0] + b[2]) / 2, pu, pv)
            rows.append((b[3], np.nan if p is None else p - b[3]))
    a = np.array(rows)
    print(f"{q}: extra non-GT boxes after B12 {len(a) / len(b12):.2f} per frame; share with paint found at their column "
          f"{np.isfinite(a[:, 1]).mean():.2f}; paint row minus foot row p10/p50/p90 {np.nanpercentile(a[:, 1], [10, 50, 90]).round(1).tolist()} px; "
          f"foot row p10/p50/p90 {np.percentile(a[:, 0], [10, 50, 90]).round(0).tolist()}")


def show():
    p = os.path.join(STC, "n47_l1.json")
    if not os.path.exists(p):
        raise SystemExit("no n47_l1.json - run --caches then --l1")
    res = json.load(open(p))
    print("TOL", res["tol"])
    for arm in ("T0", "F12", *ARMS):
        print(arm, res[arm], "" if arm in ("T0", "F12") else ("goes on" if res[f"{arm}_goes_on"] else "stops at L1"))
    for arm in ARMS:
        print(arm, "per clip", res[f"{arm}_per_clip"])


def demo():
    pu, pv = np.arange(0.0, 400.0, 10.0), np.full(40, 100.0)          # a flat far touchline at row 100
    assert paint_at(55.0, pu, pv) == 100.0 and paint_at(500.0, pu, pv) is None
    box = lambda x, y2: {"xyxy": (x - 3.0, y2 - 20.0, x + 3.0, y2), "conf": 0.9}  # noqa: E731
    # B1: a foot 5 px above the paint is beyond the line; 1 px above is on it; a column with no paint keeps its box
    fr = {"0": [box(50, 95.0), box(60, 99.0), box(600, 50.0)]}
    assert [d["xyxy"][0] for d in apply("B1", fr, (pu, pv), 2.0)["0"]] == [57.0, 597.0]
    # B2: a box held in one cell for every frame is dropped; one moving along the line (a row below the held cell) is
    # kept; a static one far from the line (a goalkeeper) is kept
    frames = {str(2 * i): [box(52, 102.0), box(20 + 3 * i, 110.0), box(200, 160.0)] for i in range(100)}
    m = static_mask(frames, (pu, pv))
    assert all(v == [True, False, False] for v in m.values()), list(m.values())[:2]
    # ...and a spot held for only a third of the window is not static
    frames = {str(2 * i): ([box(52, 102.0)] if i % 3 == 0 else []) + [box(20 + 3 * i, 110.0)] for i in range(100)}
    assert not any(x for v in static_mask(frames, (pu, pv)).values() for x in v)
    # B3: a flat camera line at 100; paint found at 102 on two stretches -> filled at 102 across the gap too
    cam = np.column_stack([np.arange(0.0, 1001.0), np.full(1001, 100.0)])
    uc, vc = fill(cam, (np.r_[np.arange(0.0, 200.0, 5.0), np.arange(800.0, 1000.0, 5.0)], np.full(80, 102.0)))
    assert uc[0] == 0 and uc[-1] == 1000 and np.allclose(vc, 102.0), (uc[:3], vc[:3])
    # B4: 12 s hugging the line is dropped; 12 s that goes 20 px into the pitch is kept; 5 s on the line is kept
    row = lambda i, f, y: {"id": i, "f": f, "x": 500.0, "y": y}  # noqa: E731
    rows = [row(1, f, 101.0) for f in range(0, 300, 2)] + [row(2, f, 101.0 + (20 if f > 200 else 0)) for f in range(0, 300, 2)] \
        + [row(3, f, 101.0) for f in range(0, 125, 2)]
    assert huggers(rows, (uc, vc)) == {1}, huggers(rows, (uc, vc))
    print("n47 demo ok: the paint test cuts beyond the line only where paint was found; the static test cuts a held spot, not a mover or a goalkeeper")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--caches" in a:
        caches()
    elif "--l1" in a:
        l1()
    elif "--l2" in a:
        l2(a[a.index("--l2") + 1])
    elif "--b3" in a:
        b3()
    elif "--b4" in a:
        b4()
    elif "--show4" in a:
        show4()
    elif "--why" in a:
        why(a[a.index("--why") + 1])
    elif "--show" in a:
        show()
    else:
        demo()
