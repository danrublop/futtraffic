"""N70: does a CLASS-AWARE tile merge improve the shipped ball without touching players? Protocol first.

WHY
    `detectors._merge` de-duplicates every box in a frame against every other by intersection over
    the SMALLER box >= 0.55, whatever its class. A ball box lying inside a player box has IoS 1.0
    with it, and the player (whole, conf >= 0.5) sorts first, so the ball candidate is deleted
    before `ball.BallPicker` ever sees it (docs/reference/detection-tracking.md#ball-merge). Some of
    those deleted candidates are the ball at a player's feet; some are leg decoys the picker is
    better off without. Which dominates is unmeasured. This is the measurement.

    Page: n70-class-aware-merge (not published). Everything below is fixed BEFORE any arm runs.

ARMS - n=2 configurations, declared now; nothing else is tried on these clips
    ONE unmerged detector pass per clip: TiledDetector(CoreMLDetector(conf=0.5, ball=True,
    ball_conf=BALL_CONF), tile=1024, overlap=OVERLAP, raw=True), every frame, read sequentially.
    Both arms are applied OFFLINE to the SAME raw boxes, so the pair differs in the merge and in
    nothing else - no detector run-to-run variance can enter the delta.
    M0  the shipped merge: `detectors._merge(all boxes, 0.55)`.
    M1  `merge_by_class` below: `_merge` run separately on BALL_CLASSES boxes and on the rest,
        same 0.55, same ordering (whole before clipped, then confidence). Nothing else changes.

HELD FIXED (a mismatch in the raw cache's stamped header is VOID, not a result)
    detector   detectors.product_detector() == "coreml": rfdetr-medium, model/models/
               rfdetr-medium-576.mlpackage. INPUT 576x576 - fixed by the export, asserted from
               the worker's reported size (the 384 trap in every-ball-number-was-measured-at-384
               (internal note) is an rfdetr-nano default and does not apply to this path,
               but the size is asserted, not assumed).
    tile       1024, overlap detectors.OVERLAP (0.25), merge_thr 0.55. Tile size is NOT a
               variable (n48-ab-was-confounded-with-tile-size (internal note)).
    thresholds player conf 0.5 (track.run's default, which jobs.py does not override), ball
               BALL_CONF 0.05, applied per class in the worker BEFORE the merge, as the product.
               NOTE: the registered product.ball.* caches were built with conf 0.05 for EVERY
               class (ball_eval.build_cache), so their merge saw players from 0.05 up. M0 here is
               therefore a fresh number and is NOT expected to equal product.ball.correct.
    picker     ball.BallPicker() unmodified (marking_gate off, as jobs.py), each clip's own
               calibration where model/validation/calib_<clip>.json exists (the soccer_side
               clips have none and run without, exactly as the registered metric does).
    stride     detection on every frame, as the registered metric. PRIMARY is scored at stride
               1 (the registered definition); the product's stride 2 is a GUARD, replayed from
               the same caches on even frames only (picker fed even frames only).
    gate       20 px (ball_eval.GATE_PX), truth from ball_eval.load_ball.

POPULATION - the nine clips of product.ball.correct (measurements.BALL_CLIPS), 7650 frames
    HELD   the six ball_cmp.CLIPS, ALL frames (not ball_cmp's first 300): soccer_top test
           1710-1740, 1740-1770, 1770-1800 and soccer_side train 0000-0030, 0030-0060,
           0060-0090. The picker's gates were never fitted on them. The rule reads HELD only.
    TUNED  soccer_top train 0000-0030 and 0030-0060 (ball.py's gates were fitted there,
           ball_eval.py:112). Reported, never ruled on.
    DEAD   soccer_top test 1680-1710 (a dead ball carried forward on every frame). Reported,
           never ruled on.
    Also printed: the nine-clip pooled number in product.ball.correct's own definition.
    Two matches only (D_20220220, F_20200220); every result is same-install.

METRICS (per arm, per clip, and pooled over frames - a SUM, as ball_eval.report_all)
    PRIMARY    `shipped` correct_of_present: share of GT-present frames where the shipped
               picker answers within 20 px (product.ball.correct's definition).
    secondary  `oracle` (gated candidate nearest truth), `wrong_of_present`, `silent`,
               ball candidates per frame after the merge, and THE DROP SHARE:
                 drop = (ball boxes surviving M1 - ball boxes surviving M0) / ball boxes surviving M1
               per clip and pooled, split by what suppressed each lost box in M0 (a player box /
               another ball box). Of the M1-only ball boxes: the share within 20 px of truth
               (the ball, rescued) vs the rest (decoys let back in). Descriptive only.
    PLAYERS    per frame, the multiset of (xyxy, conf, cls) of non-ball boxes under M0 and M1.
               Both arms are subsets of the same raw boxes, so EXACT equality is the test (no
               IoU matching needed). Reported: frames that differ, boxes added, boxes removed.
               Mechanism by which they CAN differ: a whole ball box can suppress a CLIPPED
               player fragment in M0 (whole sorts before clipped), and greedy suppression then
               cascades. So "players unchanged" is a measured outcome, not a given.
    control    `centre` (the constant picker) per arm.

RULE - fixed now. Delta = M1 - M0 on HELD, pooled `shipped`.
    VOID (no verdict, fix and re-run) if ANY of:
      (a) the raw cache header does not record coreml, input 576, tile 1024, overlap 0.25,
          conf 0.5, ball_conf 0.05;
      (b) REPRODUCTION: on the first 10 frames of every clip, M0 applied to the raw boxes is not
          identical to a fresh TiledDetector(raw=False) product pass (same boxes, same conf);
      (c) `centre` scores above 0.01 in either arm.
    SHIP M1 iff ALL of:
      1. Delta >= +0.05 (the bar every ball-picker/detector change here has been held to: N49,
         N50, N58, N59, N64);
      2. the paired cluster-bootstrap 95% CI over the 6 HELD clips (n64_ball_negatives._boot,
         10000 reps, seed 0) has its lower bound > 0 - clips, not frames, because N59 measured
         the clip-to-clip swing on this corpus at ~0.09;
      3. per-clip sign: M1 >= M0 on at least 5 of the 6 HELD clips;
      4. stride-2 guard: pooled HELD delta at stride 2 >= 0;
      5. wrong guard: pooled HELD `wrong_of_present` rises by at most +0.02 (a picker that
         answers more can buy `correct` with `wrong`, and a wrong ball is worse than none);
      6. PLAYERS EXACT: zero player boxes differ between M0 and M1 on every frame of all nine
         clips. Any difference -> PLAYERS CHANGED: M1 does not ship on this experiment whatever
         the ball does; the count, share and frames are published, and a player-side score
         (tracker replay) is a separate registration.
    WORSE iff the CI's upper bound < 0: the class-agnostic merge is acting as a ball gate
         (removing leg decoys), and #ball-merge is rewritten to say so, with the drop share.
    NO EFFECT / NOT SUPPORTED otherwise, naming which condition failed. The shipped merge
         stays; the numbers and the drop share go on the page and into #ball-merge.
    A merge that only DELETES less cannot hide hard inputs, but it can ADD decoys: the drop
    share and the rescued/decoy split are published with the verdict, whatever it is.

ESTABLISHED while writing this protocol (context, not results; the page cites them in order)

    @M 1024,0.25,0.05,0.55,0.5 :: .venv/bin/python -c "import sys,inspect;sys.path.insert(0,'model/engine');import detectors as d,track as t;print(d.tile_of('coreml'),d.OVERLAP,d.BALL_CONF,inspect.signature(d.TiledDetector.__init__).parameters['merge_thr'].default,inspect.signature(t.run).parameters['conf'].default)"
       held fixed: the shipped CoreML path's tile, overlap, ball threshold, merge threshold,
       and track.run's player conf (jobs.py passes none). n=1, no sweep.
    @M 3840x2160 15,6500x1000 9 :: .venv/bin/python -c "import sys;sys.path.insert(0,'model/engine');import detectors as d;f=lambda w,h:len(d._spans(w,1024,d.OVERLAP)[0])*len(d._spans(h,1024,d.OVERLAP)[0]);print('3840x2160',f(3840,2160),'6500x1000',f(6500,1000))"
       held fixed: tile 1024, detectors.OVERLAP; tiles for a 3840x2160 frame, then 6500x1000. n=1.
    @M 0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05 :: .venv/bin/python -c "import gzip,json,sys;sys.path.insert(0,'model/validation');import ball_eval as B;print(','.join(str(json.load(gzip.open(B.cache_path(c)))['conf']) for c in B.ALL_CLIPS))"
       held fixed: the nine candidate caches behind product.ball.*, ball_eval.ALL_CLIPS order;
       the conf stamped in each header, which build_cache applies to EVERY class. n=1.
    @C 0.2364,0.2867 :: model/validation/api-golden.json history 2026-09-29T13:47:39Z, 'replaced' (product.ball.correct, .oracle)
       Re-pointed 2026-09-29: these ids were re-frozen at the shipped merge (4628472), so the command no longer prints them.
       registry ids, nine clips, 7649 GT-present frames, caches at coreml conf 0.05 for all classes.
    @M 900,900,900,900,900,900,750,750,750 :: .venv/bin/python -c "import gzip,json,sys;sys.path.insert(0,'model/validation');import ball_eval as B;print(','.join(str(json.load(gzip.open(B.cache_path(c)))['n_frames']) for c in B.ALL_CLIPS))"
       held fixed: frames per clip in the registered caches, ball_eval.ALL_CLIPS order. n=1.

COST - one CoreML pass per frame is the whole bill; scoring is a CPU replay.
    ~50 min detection (the page derives it), under `caffeinate -i -s`. The CoreML worker holds the
    ANE/GPU: check `ps` for another detector/harness run first and never pkill by name. CPU-only
    replays by other sessions can run alongside.

    .venv/bin/python model/validation/n70_class_merge.py --demo               # merge_by_class self-check
    caffeinate -i -s .venv/bin/python model/validation/n70_class_merge.py --detect   # PHASE 2: raw pass, all 9 clips
    .venv/bin/python model/validation/n70_class_merge.py --score              # PHASE 2: both arms, verdict
    .venv/bin/python model/validation/n70_class_merge.py --export             # M0 -> fixtures/ball/ (committed)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "engine"))

import detectors as D  # noqa: E402

MERGE_THR = 0.55


def merge_by_class(dets: list[dict], thr: float = MERGE_THR) -> list[dict]:
    """Arm M1: the shipped `_merge`, run within the ball class and within everything else,
    never across. Player output is `_merge(non-ball boxes)`, so it differs from M0 only where
    a ball box took part in M0's suppression."""
    ball = [d for d in dets if d["cls"] in D.BALL_CLASSES]
    rest = [d for d in dets if d["cls"] not in D.BALL_CLASSES]
    return D._merge(rest, thr) + D._merge(ball, thr)


def demo() -> None:
    def box(xyxy, conf, cls, clip, tile):
        return {"xyxy": xyxy, "conf": conf, "cls": cls, "_clip": clip, "_tile": tile}

    player = box((100, 100, 140, 200), 0.9, "person", False, 0)
    ball = box((115, 185, 124, 194), 0.2, "sports ball", False, 0)          # at the player's feet
    dup = box((116, 186, 125, 195), 0.1, "sports ball", False, 1)   # same ball, other tile
    m0 = D._merge([player, ball, dup], MERGE_THR)
    m1 = merge_by_class([player, ball, dup])
    assert [d["cls"] for d in m0] == ["person"], m0               # the defect: ball deleted
    assert sorted(d["cls"] for d in m1) == ["person", "sports ball"], m1
    assert [d for d in m1 if d["cls"] == "sports ball"][0]["conf"] == 0.2  # ball dedup kept
    # the cascade the PLAYERS condition exists for: a whole ball suppresses a clipped fragment
    frag = box((110, 150, 130, 200), 0.8, "person", True, 0)
    m0 = D._merge([ball, frag], MERGE_THR)
    assert [d["cls"] for d in m0] == ["sports ball"], m0
    assert sorted(d["cls"] for d in merge_by_class([ball, frag])) == ["person", "sports ball"]
    print("n70_class_merge demo ok")


# ---------------------------------------------------------------- PHASE 2 (written after dd3d219)
sys.path.insert(0, str(ROOT / "model" / "validation"))
HELD = ["soccer_top__test__videos__D_20220220_1_1710_1740",
        "soccer_top__test__videos__D_20220220_1_1740_1770",
        "soccer_top__test__videos__D_20220220_1_1770_1800",
        "soccer_side__train__videos__F_20200220_1_0000_0030",
        "soccer_side__train__videos__F_20200220_1_0030_0060",
        "soccer_side__train__videos__F_20200220_1_0060_0090"]
TUNED = ["soccer_top__train__videos__D_20220220_1_0000_0030",
         "soccer_top__train__videos__D_20220220_1_0030_0060"]
DEAD = ["soccer_top__test__videos__D_20220220_1_1680_1710"]
ALL = TUNED + DEAD + HELD
CONF, REPRO_N, GATE = 0.5, 10, 20.0
KEYS = ("xyxy", "conf", "cls", "_clip", "_tile")


def raw_path(clip: str) -> Path:
    # ball_cands_*.json.gz is gitignored, so the raw pass is too
    return ROOT / "model" / "validation" / f"ball_cands_{clip}.n70raw.json.gz"


def detect(clip: str) -> None:
    """One unmerged pass over every frame; the first REPRO_N frames also go through the
    product's own merged path on the SAME inner detector, for VOID (b)."""
    import gzip, json, time  # noqa: E401, PLC0415
    import cv2  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    out = raw_path(clip)
    if out.exists() and json.load(gzip.open(out)).get("complete"):
        print(f"  {clip}: complete raw cache exists, skipped", flush=True)
        return
    inner = D.CoreMLDetector(conf=CONF, ball=True, ball_conf=D.BALL_CONF)
    raw = D.TiledDetector(inner, tile=1024, overlap=D.OVERLAP, raw=True)
    prod = D.TiledDetector(inner, tile=1024, overlap=D.OVERLAP)
    assert inner.resolution == 576 and raw.size == 576, inner.resolution
    cap = cv2.VideoCapture(str(B.FOOTAGE2 / f"{clip}.mp4"))
    frames, repro, t0, f, h = {}, {}, time.time(), 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h = img.shape[:2]
        frames[f] = [[list(map(float, d["xyxy"])), float(d["conf"]), d["cls"], bool(d["_clip"]), int(d["_tile"])]
                     for d in raw.detect(img)]
        if f < REPRO_N:
            repro[f] = [[list(map(float, d["xyxy"])), float(d["conf"]), d["cls"]] for d in prod.detect(img)]
        if f % 100 == 0:
            print(f"  {clip} frame {f}  {time.time() - t0:.0f}s", flush=True)
        f += 1
    cap.release()
    inner.close()
    hdr = {"clip": clip, "detector": "coreml", "resolution": inner.resolution, "tile": raw.tile,
           "overlap": raw.overlap, "conf": CONF, "ball_conf": D.BALL_CONF, "merge_thr": MERGE_THR,
           "frame_h": h[0], "frame_w": h[1], "n_frames": f, "complete": True,
           "wall_s": round(time.time() - t0, 1)}
    with gzip.open(out, "wt") as fh:
        json.dump({**hdr, "frames": frames, "repro": repro}, fh)
    print(f"  {clip}: {f} frames in {hdr['wall_s']} s -> {out}", flush=True)


def _dicts(row):
    return [dict(zip(KEYS, (tuple(r[0]), r[1], r[2], r[3], r[4]))) for r in row]


def _key(d):
    return (tuple(d["xyxy"]), d["conf"], d["cls"])


def clip_arms(clip: str) -> dict:
    """Both arms for one clip: ball caches in ball_eval's format, player diffs, drop share."""
    import gzip, json  # noqa: E401, PLC0415
    import numpy as np  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    d = json.load(gzip.open(raw_path(clip)))
    void = []
    want = {"detector": "coreml", "resolution": 576, "tile": 1024, "overlap": 0.25,
            "conf": 0.5, "ball_conf": 0.05}
    for k, v in want.items():
        if d.get(k) != v:
            void.append(f"(a) {clip} header {k}={d.get(k)!r}, want {v!r}")
    for f, got in d["repro"].items():
        m0 = sorted(_key(x) for x in D._merge(_dicts(d["frames"][f]), MERGE_THR))
        pr = sorted((tuple(r[0]), r[1], r[2]) for r in got)
        if m0 != pr:
            void.append(f"(b) {clip} frame {f}: M0 {len(m0)} boxes vs product {len(pr)}")
    gt, _wh, _n = B.load_ball(clip)
    arms = {"M0": {}, "M1": {}}
    pl_frames = pl_add = pl_rem = pl_n = 0
    only1 = rescued = by_player = by_ball = only0 = 0
    for fs, row in d["frames"].items():
        f = int(fs)
        dets = _dicts(row)
        m0, m1 = D._merge([dict(x) for x in dets], MERGE_THR), merge_by_class([dict(x) for x in dets])
        b0 = [x for x in m0 if x["cls"] in D.BALL_CLASSES]
        b1 = [x for x in m1 if x["cls"] in D.BALL_CLASSES]
        arms["M0"][f] = [list(x["xyxy"]) + [x["conf"]] for x in b0]
        arms["M1"][f] = [list(x["xyxy"]) + [x["conf"]] for x in b1]
        p0 = sorted(_key(x) for x in m0 if x["cls"] not in D.BALL_CLASSES)
        p1 = sorted(_key(x) for x in m1 if x["cls"] not in D.BALL_CLASSES)
        pl_n += len(p0)
        if p0 != p1:
            pl_frames += 1
            s0, s1 = set(p0), set(p1)
            pl_add += len(s1 - s0)
            pl_rem += len(s0 - s1)
        k0 = {_key(x) for x in b0}
        k1 = {_key(x) for x in b1}
        only0 += len(k0 - k1)
        g = gt.get(f)
        for x in b1:
            if _key(x) in k0:
                continue
            only1 += 1
            c = np.array([(x["xyxy"][0] + x["xyxy"][2]) / 2, (x["xyxy"][1] + x["xyxy"][3]) / 2])
            rescued += g is not None and float(np.hypot(*(c - g))) <= GATE
            box = np.array([x["xyxy"]] + [y["xyxy"] for y in m0], dtype=np.float64)
            ios = D._ios(box)[0, 1:]
            sup = [y for y, v in zip(m0, ios) if v >= MERGE_THR]
            if any(y["cls"] not in D.BALL_CLASSES for y in sup):
                by_player += 1
            else:
                by_ball += 1
    res = {"void": void, "players": {"frames_differ": pl_frames, "added": pl_add, "removed": pl_rem,
                                     "m0_player_boxes": pl_n},
           "ball_boxes": {"M0": sum(map(len, arms["M0"].values())), "M1": sum(map(len, arms["M1"].values())),
                          "m1_only": only1, "m0_only": only0, "rescued": rescued,
                          "sup_player": by_player, "sup_ball": by_ball},
           "n_frames": len(d["frames"])}
    gt_here = {f: gt[f] for f in arms["M0"] if f in gt}
    for arm, fr in arms.items():
        for stride in (1, 2):
            cache = {"clip": clip, "frame_h": d["frame_h"], "frame_w": d["frame_w"],
                     "frames": {f: v for f, v in fr.items() if f % stride == 0}}
            fl = sorted(cache["frames"])
            g = {f: gt_here[f] for f in fl if f in gt_here}
            for m in (("shipped", "oracle", "centre") if stride == 1 else ("shipped",)):
                s = B.score(g, B.predict(cache, m, g), fl, GATE)
                res[f"{arm}/{m}/s{stride}"] = {k: s[k] for k in ("correct", "wrong", "silent", "gt_present")}
    return res


def score() -> int:
    import json  # noqa: PLC0415
    import n64_ball_negatives as N64  # noqa: PLC0415
    R = {c: clip_arms(c) for c in ALL}

    def pooled(clips, key):
        c = sum(R[x][key]["correct"] for x in clips)
        n = sum(R[x][key]["gt_present"] for x in clips)
        return c / n if n else float("nan"), c, n

    def wrong(clips, arm):
        return (sum(R[x][f"{arm}/shipped/s1"]["wrong"] for x in clips)
                / sum(R[x][f"{arm}/shipped/s1"]["gt_present"] for x in clips))

    print(f"{'clip':52s} {'gt':>5s} {'M0':>7s} {'M1':>7s} {'d':>8s} {'or0':>7s} {'or1':>7s} "
          f"{'b/f0':>6s} {'b/f1':>6s} {'drop':>6s} {'resc':>5s} {'plDiff':>6s}")
    for c in ALL:
        r = R[c]
        s0, s1 = r["M0/shipped/s1"], r["M1/shipped/s1"]
        o0, o1 = r["M0/oracle/s1"], r["M1/oracle/s1"]
        bb = r["ball_boxes"]
        n = s0["gt_present"]
        print(f"{c:52s} {n:5d} {s0['correct'] / n:7.4f} {s1['correct'] / n:7.4f} "
              f"{(s1['correct'] - s0['correct']) / n:+8.4f} {o0['correct'] / n:7.4f} {o1['correct'] / n:7.4f} "
              f"{bb['M0'] / r['n_frames']:6.1f} {bb['M1'] / r['n_frames']:6.1f} "
              f"{(bb['M1'] - bb['M0']) / max(bb['M1'], 1):6.3f} {bb['rescued']:5d} "
              f"{r['players']['frames_differ']:6d}")
    out = {}
    for name, cl in (("HELD", HELD), ("TUNED", TUNED), ("DEAD", DEAD), ("ALL9", ALL)):
        a, b = pooled(cl, "M0/shipped/s1"), pooled(cl, "M1/shipped/s1")
        oa, ob = pooled(cl, "M0/oracle/s1"), pooled(cl, "M1/oracle/s1")
        s2a, s2b = pooled(cl, "M0/shipped/s2"), pooled(cl, "M1/shipped/s2")
        ca, cb = pooled(cl, "M0/centre/s1"), pooled(cl, "M1/centre/s1")
        bb = {k: sum(R[x]["ball_boxes"][k] for x in cl) for k in R[cl[0]]["ball_boxes"]}
        nf = sum(R[x]["n_frames"] for x in cl)
        out[name] = {"M0": a[0], "M1": b[0], "delta": b[0] - a[0], "n": a[2],
                     "oracle_M0": oa[0], "oracle_M1": ob[0], "s2_M0": s2a[0], "s2_M1": s2b[0],
                     "wrong_M0": wrong(cl, "M0"), "wrong_M1": wrong(cl, "M1"),
                     "centre_M0": ca[0], "centre_M1": cb[0],
                     "cands_M0": bb["M0"] / nf, "cands_M1": bb["M1"] / nf,
                     "drop": (bb["M1"] - bb["M0"]) / max(bb["M1"], 1), **{f"box_{k}": v for k, v in bb.items()}}
        print(f"POOLED {name}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in out[name].items()}))
    per = {c: {"a": (R[c]["M0/shipped/s1"]["correct"], R[c]["M0/shipped/s1"]["gt_present"]),
               "b": (R[c]["M1/shipped/s1"]["correct"], R[c]["M1/shipped/s1"]["gt_present"])} for c in HELD}
    lo, hi = N64._boot(per, HELD, reps=10000, seed=0)
    wins = sum(per[c]["b"][0] >= per[c]["a"][0] for c in HELD)
    pl = {k: sum(R[c]["players"][k] for c in ALL) for k in R[ALL[0]]["players"]}
    void = [v for c in ALL for v in R[c]["void"]]
    void += [f"(c) centre {k} {out['ALL9'][k]:.4f}" for k in ("centre_M0", "centre_M1") if out["ALL9"][k] > 0.01]
    H = out["HELD"]
    conds = {"1 delta>=+0.05": H["delta"] >= 0.05, "2 CI lo>0": lo > 0, "3 sign>=5/6": wins >= 5,
             "4 stride2>=0": H["s2_M1"] - H["s2_M0"] >= 0,
             "5 wrong<=+0.02": H["wrong_M1"] - H["wrong_M0"] <= 0.02,
             "6 players exact": pl["frames_differ"] == 0}
    print(f"CI95 [{lo:+.4f}, {hi:+.4f}]  sign {wins}/6  players {json.dumps(pl)}")
    print("conditions " + json.dumps(conds))
    if void:
        verdict = "VOID: " + "; ".join(void[:5])
    elif all(conds.values()):
        verdict = "SHIP"
    elif hi < 0:
        verdict = "WORSE"
    else:
        verdict = "NOT SUPPORTED: " + ", ".join(k for k, v in conds.items() if not v)
    print(f"VERDICT {verdict}")
    print(f"TAG {H['M0']:.4f},{H['M1']:.4f},{H['delta']:+.4f},{lo:+.4f},{hi:+.4f},{wins},{pl['frames_differ']}")
    for g in ("HELD", "TUNED", "DEAD", "ALL9"):
        o = out[g]
        print(f"TAG_{g} {o['M0']:.4f},{o['M1']:.4f},{o['delta']:+.4f},{o['oracle_M0']:.4f},{o['oracle_M1']:.4f},"
              f"{o['cands_M0']:.1f},{o['cands_M1']:.1f},{o['drop']:.4f},{o['box_m1_only']},{o['box_rescued']},"
              f"{o['box_sup_player']},{o['box_sup_ball']},{o['box_m0_only']}")
    print(f"TAG_GUARDS {H['s2_M0']:.4f},{H['s2_M1']:.4f},{H['wrong_M0']:.4f},{H['wrong_M1']:.4f},"
          f"{out['ALL9']['centre_M0']:.4f},{out['ALL9']['centre_M1']:.4f},{pl['frames_differ']},{pl['added']},"
          f"{pl['removed']},{pl['m0_player_boxes']}")
    return 0


# ------------------------------------------------ the shipped merge, committed (2026-09-29)
BALL_FIX = ROOT / "model" / "validation" / "fixtures" / "ball"
M0_CANDS = BALL_FIX / "m0_cands.json.gz"
EXPORT_CMD = ".venv/bin/python model/validation/n70_class_merge.py --export"
#: check.py ball --fast's excerpts: one calibrated top clip, one HELD side clip, frames 0..598 at stride 2.
EXCERPTS = {"top": "soccer_top__test__videos__D_20220220_1_1710_1740",
            "side": "soccer_side__train__videos__F_20200220_1_0000_0030"}
EXCERPT_STRIDE, EXCERPT_N = 2, 300


def merge_src_sha() -> str:
    """What made M0: the source of detectors._merge and the IoS it calls. The registry
    producer refuses m0_cands.json.gz when this no longer matches (the merge moved)."""
    import hashlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    return hashlib.sha256((inspect.getsource(D._merge) + inspect.getsource(D._ios)).encode()).hexdigest()


def _gz_json(path: Path, obj) -> None:
    import gzip  # noqa: PLC0415
    import json  # noqa: PLC0415
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:   # byte-reproducible
        fh.write(json.dumps(obj, separators=(",", ":")).encode())


def export() -> int:
    """M0 applied to the N70 raw caches -> the committed inputs of product.ball.* and check.py ball.

    m0_cands.json.gz: per clip, ball_eval's cache format ({clip, frame_h, frame_w, frames: {f:
    [[x1, y1, x2, y2, conf], ...]}}), every frame kept (an empty frame still feeds the scenery memory).
    The excerpts: track()'s cache format ({f: [{xyxy, conf, cls}, ...]}, balls AND players after M0) for
    frames 0, 2, .., 598, with the ball truth on those frames and, for the top clip, its calibration."""
    import hashlib  # noqa: PLC0415
    import gzip  # noqa: PLC0415
    import json  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    clips, src = {}, {}
    for c in ALL:
        p = raw_path(c)
        if not p.exists():
            print(f"missing {p} - build it: caffeinate -i -s .venv/bin/python model/validation/n70_class_merge.py --detect")
            return 2
        src[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = json.load(gzip.open(p))
        m0 = {f: D._merge(_dicts(row), MERGE_THR) for f, row in d["frames"].items()}
        clips[c] = {"clip": c, "frame_h": d["frame_h"], "frame_w": d["frame_w"],
                    "settings": {k: d[k] for k in ("detector", "resolution", "tile", "overlap", "conf", "ball_conf",
                                                   "merge_thr")},
                    "frames": {f: [list(x["xyxy"]) + [x["conf"]] for x in v if x["cls"] in D.BALL_CLASSES]
                               for f, v in m0.items()}}
        if c in EXCERPTS.values():
            keep = [str(EXCERPT_STRIDE * i) for i in range(EXCERPT_N)]
            gt = B.load_ball(c)[0]
            cal = ROOT / "model" / "validation" / f"calib_{c}.json"
            _gz_json(BALL_FIX / f"{c}.m0.s2f0-598.json.gz", {
                "clip": c, "frame_h": d["frame_h"], "frame_w": d["frame_w"], "stride": EXCERPT_STRIDE,
                "made_by": EXPORT_CMD, "source": {p.name: src[p.name]},
                "calibration": json.loads(cal.read_text()) if cal.exists() else None,
                "gt": {f: [float(v) for v in gt[int(f)]] for f in keep if int(f) in gt},
                "frames": {f: [{"xyxy": list(x["xyxy"]), "conf": x["conf"], "cls": x["cls"]} for x in m0[f]]
                           for f in keep}})
    _gz_json(M0_CANDS, {"made_by": EXPORT_CMD, "merge": "detectors._merge(all boxes, 0.55) - M0, the shipped merge",
                        "merge_src_sha256": merge_src_sha(), "source_sha256": src, "clips": clips})
    for p in sorted(BALL_FIX.glob("*.json.gz")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")
    return 0


def main() -> int:
    if "--export" in sys.argv:
        return export()
    if "--demo" in sys.argv:
        demo()
        return 0
    if "--detect" in sys.argv:
        for c in ALL:
            detect(c)
        return 0
    if "--score" in sys.argv:
        return score()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
