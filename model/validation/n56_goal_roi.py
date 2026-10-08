"""N56: a low-threshold detection pass inside the PENALTY AREA only.

    .venv/bin/python model/validation/n56_goal_roi.py --score
    .venv/bin/python model/validation/n56_goal_roi.py --demo

WHY THIS RUNS
    N53: all three fine-tune seeds lose 128057's GT id 12 - the goalkeeper -
    while every other id improves (0.9803 -> 0.3517/0.2112/0.4207). The loss is
    DETECTION: on 64% of his frames no box exists within IoU 0.1 at any
    confidence the detector emitted down to the 0.2 raw floor. No picker, tracker
    or labelling rule can recover a box that was never proposed.

    What geometry CAN do is say where to look harder. Measured on 128057's GT
    through that clip's own calibration: the keeper spends 64.3% of his frames
    inside a penalty area against 19.3% for all 22 ids pooled - a 3.3x
    enrichment, and the smallest fixed region that gives one. That is a POWER
    CHECK, not a tuned threshold: an ROI that never contains him could not help,
    so it was worth knowing before spending the run.

HONEST ABOUT WHAT THIS IS NOT BLIND TO
    The penalty area is fixed by the Laws of the Game and nothing about its
    shape was fitted here. But the DECISION to look there came from already
    knowing which player was lost. So a pass here supports "a low-threshold pass
    inside the penalty area recovers this population", NOT "this will catch
    future losses we have not seen". Saying otherwise would be the landmark
    hold-out mistake in a new costume (landmark-holdout-is-covariate-shift (internal note)).

RULE, FIXED BEFORE THE SCORED RUN (committed with this file, nothing run yet)
    data        the COMMITTED raw detections from releases n46-stage2-1 (ft0) and
                n46-stage2-2 (ft1, ft2), conf floor 0.2, stride 2, tile 1024,
                unmerged. NO detector runs: this is a re-threshold of boxes that
                already exist, so it costs no GPU and cannot be confounded by a
                second detection pass behaving differently.
    arms        BASE  every box with conf >= 0.5, the product's threshold. This
                      reproduces N46's own numbers and is asserted to.
                ROI   the same, PLUS every box with conf >= 0.2 whose FOOT point
                      (box bottom centre) maps inside either penalty area under
                      that clip's committed .p35 calibration.
    population  the four TEST clips, all three seeds. Per-id recall at IoU >= 0.5
                and non-GT boxes per frame, exactly as n46_finetune measures them.
    PASS        BOTH, on the chosen seed ft0:
                  (a) the worst id's recall rises by >= +0.15, and
                  (b) non-GT boxes per frame rise by <= +0.5, which is N46's own
                      precision bar, unchanged.
    REPORT      every id and every seed either way, and the non-GT rise per clip.
                "The ROI only ADDS boxes, so recall cannot fall" was written here
                and is FALSE: `detectors._merge` orders by (_clip, -conf), so a
                WHOLE low-confidence ROI box sorts ahead of a CLIPPED box at conf
                >= 0.5 and can suppress it. Measured on 128057/ft0: 32 BASE boxes
                dropped on 32 frames, and 4.5% of the release carries _clip=True.
                It costs nothing on this run - 0 of 88 ids fall - but the reason
                given for not checking was wrong, so the check is now RUN: no
                per-id recall may fall at all.
    n           n=1 configuration: one region (the penalty area) and one
                threshold (0.2, the raw floor of what the PC saved - not chosen,
                it is the lowest number available). If a second region or
                threshold is ever scored, that n moves and this stops being a
                pre-registration.
    VOID        if BASE does not reproduce n46_finetune's per-id recall for the
                same arm to 1e-9, the re-threshold is not the product's and
                nothing below compares to N46.
"""
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model", "validation", "out", "stc2025")
RELEASES = (os.path.join(ROOT, "model", "validation", "out", "n46_stage2", "n46_stage2_results.zip"),
            os.path.join(ROOT, "model", "validation", "out", "n46_stage2b", "n46_stage2b_results.zip"))
TEST = ("118577", "118578", "128057", "128058")
ARMS = ("ft0", "ft1", "ft2")
CONF, FLOOR = 0.5, 0.2
#: The penalty area, Laws of the Game: 16.5 m deep, 40.32 m wide, centred on the
#: goal. Not fitted - measured against it, never to it.
PEN_DEPTH, PEN_HALFWIDTH, PITCH_L, PITCH_W = 16.5, 20.16, 105.0, 68.0


def in_penalty_area(p) -> bool:
    x, y = float(p[0]), float(p[1])
    return (x < PEN_DEPTH or x > PITCH_L - PEN_DEPTH) and abs(y - PITCH_W / 2) <= PEN_HALFWIDTH


def raw(seq, arm):
    """The PC's pre-threshold detections, straight out of the committed release."""
    for z in RELEASES:
        if not os.path.exists(z):
            continue
        with zipfile.ZipFile(z) as zf:
            hit = [n for n in zf.namelist() if seq in n and f".{arm}." in n]
            if hit:
                return json.loads(zf.read(hit[0]))
    raise SystemExit(f"no raw detections for {seq} {arm} in any committed release")


def boxes(seq, arm, roi):
    """{frame: [xyxy]}, built the way import_raw builds the product's own cache.

    THRESHOLD, THEN MERGE, THEN THE PITCH POLYGON - in that order, and with
    n46_finetune's own constants rather than copies. The first cut of this
    function did none of the last two and BASE came back 0.3531 where N46 has
    0.3517: close enough to read as agreement and not equal, which is what this
    file's VOID condition exists to catch. A re-threshold that skips the
    product's merge is not a re-threshold of the product.
    """
    import calibrators as C  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    import detectors as D  # noqa: PLC0415
    import n46_finetune as N  # noqa: PLC0415

    cal = C.load(os.path.join(STC, f"{seq}.p35.calibration.json"))
    poly = np.asarray(cal.pitch_polygon_px(), np.float32)
    d = raw(seq, arm)
    assert d["conf"] <= FLOOR, f"{seq} {arm}: raw floor is {d['conf']}, not {FLOOR}"
    assert d["merged"] is False, "the release is supposed to be unmerged"
    out = {}
    for f, v in d["frames"].items():
        keep = []
        for b in v:
            if b["conf"] >= CONF:
                keep.append(dict(b))
            elif roi and b["conf"] >= FLOOR:
                x0, _y0, x1, y1 = b["xyxy"]
                p = np.asarray(cal.to_pitch(np.array([[(x0 + x1) / 2, y1]], float))).reshape(-1)[:2]
                if in_penalty_area(p):
                    keep.append(dict(b))
        merged = D._merge(keep, N.MERGE_THR)
        out[int(f)] = [b["xyxy"] for b in merged
                       if cv2.pointPolygonTest(poly, (float((b["xyxy"][0] + b["xyxy"][2]) / 2),
                                                      float(b["xyxy"][3])), False) >= 0]
    return out


def score(arm="ft0"):
    import n40_occlusion_recall as R

    res = {}
    for roi in (False, True):
        per, fp, nf = {}, 0, 0
        for q in TEST:
            gt, bx = R._gt(q), boxes(q, arm, roi)
            cnt = {}
            for f, bs in bx.items():
                if f not in gt:
                    continue
                nf += 1
                g = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
                for gi, x, y, w, h in gt[f]:
                    c = cnt.setdefault((q, gi), [0, 0])
                    c[0] += 1
                    c[1] += any(R._iou((x, y, x + w, y + h), tuple(b)) >= 0.5 for b in bs)
                fp += sum(1 for b in bs if not any(R._iou(tuple(b), tuple(t)) >= 0.5 for t in g))
            per.update({k: v[1] / max(1, v[0]) for k, v in cnt.items()})
        res["ROI" if roi else "BASE"] = {"per_id": per, "non_gt_per_frame": fp / max(1, nf)}
    return res


def show(arm="ft0"):
    import n46_finetune as N  # noqa: PLC0415

    r = score(arm)
    b, o = r["BASE"], r["ROI"]
    # THE VOID CONDITION, RUN. It was written in the docstring and the first
    # scored run violated it silently - a rule that only a reader enforces is
    # the thing docs-guard-must-have-a-runnable-check (internal note) is about.
    ref = N.per_id(f"p46{arm}", TEST)
    # ROUND BEFORE COMPARING: n46_finetune.per_id rounds to 4 dp, so an exact
    # comparison against an unrounded number reports 74 of 88 ids as disagreeing
    # when every one of them agrees. That was this check's first result.
    off = {k: (round(b["per_id"][k], 4), ref[k[0]][str(k[1])]) for k in b["per_id"]
           if abs(round(b["per_id"][k], 4) - ref[k[0]][str(k[1])]) > 1e-9}
    if off:
        raise SystemExit(f"VOID: BASE does not reproduce n46_finetune's per-id recall "
                         f"for {arm} on {len(off)} of {len(b['per_id'])} ids, e.g. "
                         f"{list(off.items())[:3]} - the re-threshold is not the product's.")
    print(f"BASE reproduces n46_finetune on all {len(b['per_id'])} ids (VOID check passed)")
    worst = min(b["per_id"], key=lambda k: b["per_id"][k])
    print(f"arm {arm}, penalty-area pass at conf {FLOOR} (product threshold {CONF})\n")
    print(f"{'':>18} {'BASE':>8} {'ROI':>8} {'delta':>8}")
    print(f"{'worst id ' + str(worst):>18} {b['per_id'][worst]:8.4f} {o['per_id'][worst]:8.4f} "
          f"{o['per_id'][worst] - b['per_id'][worst]:+8.4f}")
    print(f"{'non-GT per frame':>18} {b['non_gt_per_frame']:8.3f} {o['non_gt_per_frame']:8.3f} "
          f"{o['non_gt_per_frame'] - b['non_gt_per_frame']:+8.3f}")
    rises = sorted(((o["per_id"][k] - b["per_id"][k], k) for k in b["per_id"]), reverse=True)[:3]
    print("\nlargest per-id rises: " + ", ".join(f"{q} id{i} {d:+.4f}" for d, (q, i) in rises))
    fell = {k: (b["per_id"][k], o["per_id"][k]) for k in b["per_id"]
            if o["per_id"][k] < b["per_id"][k] - 1e-9}
    if fell:
        print(f"\n  {len(fell)} id(s) LOST recall to the merge: "
              + ", ".join(f"{q} id{i} {bv:.4f}->{ov:.4f}" for (q, i), (bv, ov) in list(fell.items())[:4]))
    a = o["per_id"][worst] - b["per_id"][worst] >= 0.15
    c = o["non_gt_per_frame"] - b["non_gt_per_frame"] <= 0.5
    print(f"\nRULE: worst-id rise >= +0.15 [{a}] and non-GT rise <= +0.5 [{c}] "
          f"and no id falls [{not fell}] -> "
          + ("SUPPORTED" if a and c and not fell else "NOT SUPPORTED")
          + ("" if arm == "ft0" else "   (NOT the ruling seed - ft0 rules; see the docstring)"))
    return r


def demo():
    """The geometry, which is the only arithmetic here that can be silently wrong."""
    assert in_penalty_area((5.0, 34.0)) and in_penalty_area((100.0, 34.0))
    assert not in_penalty_area((52.5, 34.0)), "the centre circle is not a penalty area"
    assert not in_penalty_area((5.0, 5.0)), "the corner is not a penalty area"
    # the boundary, both axes, because off-by-one here silently resizes the ROI
    assert in_penalty_area((16.4, 34.0)) and not in_penalty_area((16.6, 34.0))
    assert in_penalty_area((5.0, 34.0 + 20.1)) and not in_penalty_area((5.0, 34.0 + 20.2))
    print("n56 demo: penalty-area geometry ok (depth 16.5 m, half-width 20.16 m)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--score" in a:
        show(a[a.index("--score") + 1] if len(a) > a.index("--score") + 1 else "ft0")
    else:
        demo()
