"""N62: at the lost goalkeeper, does the detector propose ANYTHING at any score?

    .venv/bin/python model/validation/n62_query_probe.py --probe ft0
    .venv/bin/python model/validation/n62_query_probe.py --demo

WHY THIS RUNS, AND WHY IT IS THE PROBE THAT DECIDES A GPU DAY
    All three fine-tune seeds lose 128057 GT id 12 (N53: 0.9803 -> 0.3517 /
    0.2112 / 0.4207). Two explanations survive and nothing so far separates them:
    the TRAINING COMPOSITION (the only TEST clip whose teams were never seen) or
    the LABEL POLICY (officials, substitutes and ball boys unlabelled, so ~2.65
    people a frame trained as background - N61 puts at most 46.8% of the
    suppressed boxes on the pitch at all).

    N60's reading of the literature says the label-policy failure is a
    CLASSIFICATION failure: the box is still proposed, its class score is pushed
    down. SparseDet measured exactly that shape - proposal recall barely moves
    (0.83 -> 0.79) while AP collapses. So the two explanations predict DIFFERENT
    things about what the detector emits at the keeper's position:

        label policy   a box localises him, carrying a crushed score
        composition    nothing localises him at any score - the representation
                       moved and he is not proposed at all

    N53's `[conf]` probe read the RELEASED detections, whose floor is conf 0.20.
    That cannot see below 0.20, which is exactly where a suppressed-but-proposed
    box would sit. This runs the detector itself.

METHOD - the shipped path, one knob turned
    `model/validation/n46_pc/detect.py`'s exact detector: the product's
    TiledDetector over RFDETRDetector, variant medium, tile 1024, shipped
    overlap, raw (unmerged) - with conf 0.001 instead of 0.20, so essentially
    every query is returned rather than a thresholded subset. Nothing else
    differs, and the weights are the committed `checkpoint_best_total.pth`
    (hash-checked against n46_choice.json before use, or this refuses to run).

READING, FIXED BEFORE THE RUN
    Population: frames where the product's threshold currently LOSES the keeper -
    sampled evenly, N=FRAMES - plus, as the control, the same frames' labelled
    outfield players, who are NOT lost.
        PROPOSED   a returned box with IoU >= 0.5 against the keeper's GT box.
        share      the fraction of sampled frames with such a box.
    Verdict:
        share >= 0.80  CLASSIFIED DOWN. The box is there; the score is crushed.
                       Consistent with the label policy, and the retrain becomes
                       a fix rather than an experiment.
        share <= 0.20  NOT PROPOSED. No query localises him at any score. The
                       label-policy story cannot explain it and the GPU day
                       should go to composition.
        in between     UNRESOLVED, and that is a legitimate answer to report.
    Also reported, because a score has no meaning without one: the same share and
    the score distribution for the labelled outfield players on the same frames.
    A crushed score is only evidence if the controls' scores are not crushed.
    n=1 configuration: one conf floor (0.001), one IoU (0.5), one clip.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
WEIGHTS = os.path.join(ROOT, "model", "validation", "out", "n46_weights")
CLIPS = os.path.expanduser("~/Downloads/st2/mot/clips")
SEQ, KEEPER, FRAMES, FLOOR, TILE = "128057", 12, 60, 0.001, 1024


def _weights(arm):
    """The committed checkpoint, or nothing. A probe on unverified weights is a
    probe on an unknown model - and the file NAME is not the check: the first
    request for these asked for checkpoint_best_ema.pth, which produced none of
    the published detections."""
    import hashlib  # noqa: PLC0415

    p = os.path.join(WEIGHTS, f"{arm}_checkpoint_best_total.pth")
    want = json.load(open(os.path.join(ROOT, "model/validation/n46_choice.json")))["weights_sha256"][arm]
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}: gh release download n46-weights-1")
    got = hashlib.file_digest(open(p, "rb"), "sha256").hexdigest()
    if got != want:
        raise SystemExit(f"VOID: {p} hashes {got[:16]}, n46_choice.json records {want[:16]}")
    return p


def frames_lost(arm):
    """Frames where the product's own threshold returns no keeper box."""
    import n40_occlusion_recall as R  # noqa: PLC0415

    gt = R._gt(SEQ)
    cache = json.load(open(os.path.join(ROOT, "model/validation/out/stc2025",
                                        f"{SEQ}.p46{arm}.dets.cache.json")))["frames"]
    out = []
    for f, rows in sorted(gt.items()):
        k = [r for r in rows if r[0] == KEEPER]
        if not k or str(f) not in cache:
            continue
        _i, x, y, w, h = k[0]
        g = (x, y, x + w, y + h)
        if not any(R._iou(g, tuple(d["xyxy"])) >= 0.5 for d in cache[str(f)]):
            out.append((f, g, [(r[1], r[2], r[1] + r[3], r[2] + r[4]) for r in rows if r[0] != KEEPER]))
    return out


def probe(arm="ft0"):
    import cv2  # noqa: PLC0415
    import detectors as D  # noqa: PLC0415
    import n40_occlusion_recall as R  # noqa: PLC0415

    w = _weights(arm)
    lost = frames_lost(arm)
    if not lost:
        raise SystemExit(f"{arm}: the keeper is never lost at the product threshold")
    pick = lost[:: max(1, len(lost) // FRAMES)][:FRAMES]
    print(f"{arm}: {len(lost)} frames lose the keeper at conf 0.5; probing {len(pick)} of them "
          f"at conf {FLOOR}\n")
    det = D.TiledDetector(D.RFDETRDetector(conf=FLOOR, variant="medium", weights=w),
                          tile=TILE, overlap=D.OVERLAP, batch=8, raw=True)
    cap = cv2.VideoCapture(os.path.join(CLIPS, f"{SEQ}.mp4"))
    hit, ctl_hit, ctl_n, ks, cs = 0, 0, 0, [], []
    for f, g, others in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
        if not ok:
            continue
        boxes = det.detect(img)   # detect.py:62's own call
        # key= on the IoU alone: a tie otherwise compares the dicts and raises
        best = max(((R._iou(g, tuple(b["xyxy"])), b) for b in boxes),
                   key=lambda t: t[0], default=(0.0, None))
        if best[0] >= 0.5:
            hit += 1
            ks.append(float(best[1].get("conf", 0.0)))
        for o in others:                      # the control: labelled outfield players
            ctl_n += 1
            cb = max(((R._iou(o, tuple(b["xyxy"])), b) for b in boxes),
                     key=lambda t: t[0], default=(0.0, None))
            if cb[0] >= 0.5:
                ctl_hit += 1
                cs.append(float(cb[1].get("conf", 0.0)))
    cap.release()
    share = hit / max(1, len(pick))
    verdict = ("CLASSIFIED DOWN" if share >= 0.80 else
               "NOT PROPOSED" if share <= 0.20 else "UNRESOLVED")
    print(f"  keeper PROPOSED on {hit}/{len(pick)} = {share:.3f} of frames it loses")
    if ks:
        print(f"    its score when proposed: p50 {np.median(ks):.4f}  p90 {np.percentile(ks, 90):.4f}")
    print(f"  control, labelled outfield players on the same frames: "
          f"{ctl_hit}/{ctl_n} = {ctl_hit / max(1, ctl_n):.3f}")
    if cs:
        print(f"    their scores: p50 {np.median(cs):.4f}  p10 {np.percentile(cs, 10):.4f}")
    print(f"\n  RULE: >=0.80 CLASSIFIED DOWN, <=0.20 NOT PROPOSED -> {verdict}")
    return {"arm": arm, "share": share, "verdict": verdict, "n": len(pick)}


def demo():
    """The reading rule and the frame selection, which is where this can mislead."""
    assert FLOOR < 0.2, "the released detections already have a 0.20 floor; this must go below it"
    for share, want in ((0.95, "CLASSIFIED DOWN"), (0.05, "NOT PROPOSED"), (0.5, "UNRESOLVED")):
        v = ("CLASSIFIED DOWN" if share >= 0.80 else
             "NOT PROPOSED" if share <= 0.20 else "UNRESOLVED")
        assert v == want, (share, v)
    # the population must be the frames the product LOSES - probing frames it
    # already gets right would answer a question nobody asked
    import n40_occlusion_recall as R  # noqa: PLC0415
    assert callable(R._iou)
    print("n62 demo: reading rule ok (>=0.80 classified down, <=0.20 not proposed)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--probe" in a:
        probe(a[a.index("--probe") + 1] if len(a) > a.index("--probe") + 1 else "ft0")
    else:
        demo()
