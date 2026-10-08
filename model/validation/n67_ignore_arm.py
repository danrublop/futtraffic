"""N67: arm A scored - do ignore regions give the lost goalkeeper back, and at what price?

    .venv/bin/python model/validation/n67_ignore_arm.py            # self-check
    .venv/bin/python model/validation/n67_ignore_arm.py --build    # the ignore-region dataset
    .venv/bin/python model/validation/n67_ignore_arm.py --score    # refuses until IG weights exist

WHY THIS ARM AND NOT THE OTHER
    N46's fine-tune lost 128057 id 12 - a goalkeeper - from recall 0.9803 to 0.3517 on all
    three seeds while the clip mean ROSE. N62: the box is proposed on 0.983-1.000 of the
    lost frames and scored 0.0401 against labelled outfielders' 0.8940, so it is
    classification. N63: keepers were 9.9% of training, above per-capita, so it is not
    rarity. What is left is a conjunction - an appearance TRAIN never carried, inside a
    policy that labels every person who is not one of the 22 as background.

    Arm A treats the policy half. Arm B - label everyone a player and filter downstream -
    was rejected because it stacks on N47's filter, which has not cleared its own gate
    (IDSW 1.261x), and because it changes what the detector is FOR across the whole
    runtime. N66 built the mechanism the library lacks and proved it works.

    ARM A CANNOT TREAT THE OTHER HALF. It cannot make an unseen kit seen, so a partial
    recovery is the expected outcome and the bar below says which part is which.

WHICH BOXES BECOME IGNORE REGIONS, and why not all of them
    N61 bucketed the non-GT boxes on the TRAIN clips. Only ONE bucket is used:
      on-pitch    46.8% - no GT overlap, plausible size, foot inside the GT feet band.
                  N61 calls this the only bucket the officials story is entitled to, and it
                  is the population whose background label plausibly suppresses a
                  novel-kit keeper. USED.
      duplicate   18.2% - a second box on a player the GT HAS. Suppressing a duplicate is
                  desirable, and ignoring it would delete supervision that keeps one box
                  per player. NOT USED.
      wrong-size  15.2% - a merged pair or a fragment. Suppressing those is the point.
                  NOT USED.
      off-pitch   19.8% - bench, staff, spectators. NOT USED, and this is the deliberate
                  one: N43b and N47 spent the far-tile programme REMOVING exactly these
                  boxes, and N47's filter is still blocked at IDSW 1.261x. Ignoring the
                  bench would stop teaching the model to suppress it and could re-introduce
                  what that work exists to delete. It stays available as a SECOND arm if
                  this one works, rather than bundled in where the two could not be told
                  apart.

THE PRICE, which is why the guard below is the load-bearing clause
    The fine-tune's own precision gain is the same mechanism that crushed the keeper: it
    suppressed boxes matching no GT from 2.277 to 0.41-0.53 per frame (N53), and
    n46_dataset records that the STC2025 winner's detector drew 0.2-0.6 non-player boxes a
    frame against our 2.2. So that suppression is what closed the detector gap, and arm A
    gives some of it back BY CONSTRUCTION. One goalkeeper does not buy it back.

RULE, FIXED BEFORE THE DATASET IS BUILT AND BEFORE ANYTHING TRAINS
    population  N46's TEST clips through n46_finetune's own per-id path; 128057 is reported
                alone because it is the only clip whose teams appear nowhere in TRAIN.
    arms        FT  the existing fine-tune, three seeds, id 12 at 0.3517 / 0.21 / 0.42.
                IG  identical in every respect - tiles, epochs, budget, seeds, device,
                    precision - plus the on-pitch ignore regions and n66_ignore.patch().
    PASS, all four
      PRODUCT   id 12's recall on 128057, MEDIAN of three seeds, >= 0.80. The other 21 ids
                sit near 0.98 and stock had him at 0.9803; below 0.80 his distance and speed
                series are still wrong, so a smaller gain is a mechanism result and not a
                product one.
      MECHANISM >= 0.3517 + 0.15 = 0.5017, reported SEPARATELY. If this holds and PRODUCT
                does not, the label policy is implicated and the appearance half is what
                remains - which is a finding, not a ship.
      GUARD-P   non-GT boxes per frame <= 0.60 on the TEST clips: the top of the winner's
                published range and just above the fine-tune's own 0.41-0.53. This is the
                clause the arm is most likely to fail, and it is in the bar rather than
                beside it because E1 cleared an outcome bar while its mechanism column
                moved the wrong way (n64-e1-passed-the-bar-on-a-different-experiment (not published)).
      GUARD-ID  no other TEST id falls by more than PER_ID_FLOOR = 0.10, the floor already
                pre-registered in n46_finetune.py for exactly this arm.
    n           n=3 seeds, median per id, matching n46_finetune's existing practice. One
                draw is not a noise floor and the collapse was consistent across all three,
                so anything less cannot separate the arm from a seed.
    SCREEN, registered 2026-09-28 BEFORE SEED 0 RUNS, because n=3 is 18.4 hours
                FT's own recipe takes 22,029 s per seed (n46_stage1 run records: 17 epochs,
                batch 4, accum 4, lr 1e-4 cosine, fp16, multi_scale on / expanded off), so
                the registered n=3 is 18.4 h of PC GPU.
                RUN SEED 0 ALONE FIRST. Continue to seeds 1 and 2 IFF seed 0's id 12 recall
                on 128057 exceeds 0.42 - the TOP of FT's three observed seeds, which are
                0.3517 / 0.21 / 0.42. The stop rule is deliberately NOT the mechanism bar:
                FT's own spread is 0.21 wide, so a seed-0 value anywhere inside 0.21-0.42
                cannot be distinguished from seed noise and the remaining 12.3 h buys no
                answer. Above 0.42, one seed is outside the range FT ever produced and the
                median becomes worth measuring.
                A stop reads "no evidence that the arm moves him beyond seed noise", NEVER
                "the label policy is not the mechanism" - one draw cannot say that, and the
                bars above still require the median.
                This rule is registered before seed 0 exists precisely so that continuing
                or stopping is not a choice made after seeing it.
    VOID        (a) n66_ignore.patch() did not bind, or its upstream source hash differs;
                (b) ANY ignore region overlaps a GT box at IoU >= 0.5 - that would delete
                    supervision for a labelled player, which is the failure N61 was run to
                    prevent, asserted per region rather than counted;
                (c) the ignore set draws from any bucket but on-pitch;
                (d) the arms differ in anything but the ignore regions - device, precision,
                    seed, epoch count, tile set, or base weights;
                (e) FT does not reproduce id 12 at 0.3517 under this harness, so there is
                    no baseline;
                (f) fewer than 1,000 ignore regions survive the tiling, which would make IG
                    a differently-seeded FT rather than the arm described here.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

#: IMPORTED, never redefined. The sentinel label the dataset writes must be the one the loss
#: strips - two copies of it could drift, and then the ignore regions would train as a
#: phantom class instead of being withheld. n66_ignore owns it.
from n66_ignore import IGNORE_LABEL  # noqa: E402

def _bbox_iou(a, b):
    """IoU of two COCO [x, y, w, h] boxes. Used on the TILED boxes, where cropping can raise
    an overlap the whole-frame check never saw."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    i = (ix1 - ix0) * (iy1 - iy0)
    return i / (aw * ah + bw * bh - i)


#: The detector variant the regions are found with. Named here rather than inline in two
#: places, so the assert in build() has something to compare.
VARIANT = "medium"

#: The only bucket that becomes an ignore region. See the docstring for the other three.
BUCKET = "on-pitch"
#: A region overlapping a labelled player this much deletes supervision instead of
#: withholding it - VOID (b), asserted per region.
GT_IOU_MAX = 0.5
#: VOID (f): below this the arm is not the arm.
MIN_REGIONS = 1000
#: The pre-registered bars, here so they cannot move after the number exists.
PRODUCT_BAR, MECHANISM_BAR, NON_GT_MAX = 0.80, 0.5017, 0.60
#: IMPORTED, not re-declared. It existed here as +0.10 and in n46_finetune as -0.10 - two
#: copies of one number with opposite signs, in a file that imports IGNORE_LABEL precisely to
#: stop a constant being copied.
from n46_finetune import PER_ID_FLOOR  # noqa: E402
#: The screen's continue threshold: the TOP of FT's three observed id-12 recalls, so a
#: seed-0 value inside FT's own spread stops the arm rather than buying 12.3 more hours.
SCREEN_CONTINUE = 0.42
#: FT's three seeds on 128057 id 12, from n46_finetune --per-id. The spread IS the reason
#: the screen is not set at the mechanism bar.
FT_SEEDS = (0.3517, 0.21, 0.42)


def regions(clip, det=None):
    """[(frame, (x0,y0,x1,y1))] - the ON-PITCH non-GT boxes on one TRAIN clip.

    Detected FRESH rather than read from out/stc2025/*.p21.dets.cache.json, because only
    four of N46's five TRAIN clips have such a cache (132831 has none) and a builder with a
    per-clip hole would either drop a fifth of the arm or need a special case. Fresh, with
    the SHIPPED configuration asserted against shipped_config, is uniform across all five.

    `bucket()` is IMPORTED from n61_unlabelled_audit, not reproduced: the four bucket rules
    are that file's and a second copy of them here would be free to drift from the shares
    the arm's design rests on.
    """
    import numpy as np  # noqa: PLC0415

    import detectors as D  # noqa: PLC0415
    import n40_occlusion_recall as R  # noqa: PLC0415
    import n46_dataset as N46  # noqa: PLC0415
    import n61_unlabelled_audit as N61  # noqa: PLC0415

    cache = os.path.join(str(N46.OUT).replace("n46_dataset", "n67_ignore"),
                         f"regions_{clip}.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        print(f"  {clip}: {len(d)} regions from cache", flush=True)
        return [(f, tuple(b)) for f, b in d]

    gt = R._gt(clip)
    feet = [y + h for rows in gt.values() for _i, _x, y, _w, h in rows]
    hs = [h for rows in gt.values() for _i, _x, _y, _w, h in rows]
    lo, hi = float(np.percentile(feet, 1)), float(np.percentile(feet, 99))
    med_h = float(np.median(hs))

    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(f"{N46.CLIPS}/{clip}.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = N46.FRAME_STEP * max(1, round(len(N46.grid(w, h)) / 5))   # the build's own rule
    det = det or D.TiledDetector(D.RFDETRDetector(conf=0.5, variant=VARIANT),
                                 tile=1024, overlap=D.OVERLAP)
    out, f, kept, seen = [], 0, 0, 0
    while True:
        if not cap.grab():
            break
        if f % step == 0 and f in gt:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = [(x, y, x + bw, y + bh) for _i, x, y, bw, bh in gt[f]]
            for d in det.detect(frame):
                if d.get("cls") not in D.PLAYER_CLASSES:
                    continue
                b = tuple(float(v) for v in d["xyxy"])
                seen += 1
                if not _keep(b, g, lo, hi, med_h):
                    continue
                # (VOID (b) is enforced in build(), on the TILED boxes - the only place
                # cropping can change the answer. The assert that used to sit here restated
                # the line above it and could not fail.)
                out.append((f, b))
                kept += 1
            if f % (step * 20) == 0:
                print(f"    {clip} frame {f}, {kept} regions so far", flush=True)
        f += 1
    cap.release()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump([[f, list(b)] for f, b in out], open(cache, "w"))
    print(f"  {clip}: {kept} on-pitch regions from {seen} detected boxes, step {step}",
          flush=True)
    return out


def _void_bf(anns, added):
    """VOID (b) and (f) on the ANNOTATIONS THAT REACH THE LOSS. Factored so demo() can run it.

    (b) was a tautology where it used to live: regions() asserted no GT overlap immediately
    after `continue`-ing on GT overlap, and it ran on whole-frame boxes BEFORE tile_boxes cut
    them - cropping can RAISE an IoU. (f) lived inline in build(), so neither constant could be
    exercised by a self-check; planting GT_IOU_MAX=0.99 and MIN_REGIONS=0 both survived.
    """
    if added < MIN_REGIONS:
        raise SystemExit(f"VOID (f): {added} tiled ignore regions, need {MIN_REGIONS}")
    byimg = {}
    for a in anns:
        byimg.setdefault(a["image_id"], ([], []))[0 if a["category_id"] == 0 else 1].append(
            a["bbox"])
    worst, over = 0.0, 0
    for players, igs in byimg.values():
        for g in igs:
            m = max((_bbox_iou(g, pl) for pl in players), default=0.0)
            worst = max(worst, m)
            over += m >= GT_IOU_MAX
    print(f"  VOID (b) on the TILED boxes: {over} over IoU {GT_IOU_MAX}, worst {worst:.4f}")
    if over:
        raise SystemExit(f"VOID (b): {over} tiled ignore regions cover a labelled player at "
                         f"IoU >= {GT_IOU_MAX}, which would DELETE supervision instead of "
                         f"withholding it")
    return worst


def build(which="ig"):
    """The IG dataset: N46's tiles unchanged, plus the ignore regions as category 1."""
    import detectors as D  # noqa: PLC0415
    import n46_dataset as N46  # noqa: PLC0415
    import shipped_config as SC  # noqa: PLC0415

    # VOID (d) in part: the regions must be found with the configuration the PRODUCT runs,
    # or they are not the population the product's detector would have proposed.
    # VOID (a), made runnable: the forked loss must still match upstream, checked BEFORE a
    # dataset is written. n66_ignore.patch() was called from nowhere in the tree, so the
    # regions could have been built for a mechanism that no longer applies.
    import n66_ignore as N66  # noqa: PLC0415
    if N66.source():
        raise SystemExit("VOID (a): the forked loss_labels no longer matches upstream - "
                         "re-read it and re-pin before building a dataset for it")
    cfg = SC.product_config()
    # VOID (d) in part. `variant` is asserted too now: it was hardcoded in two places while
    # the docstring claimed the shipped configuration was checked, and N65 just showed the
    # model's input geometry is worth 0.11 on the ball.
    assert cfg["conf"] == 0.5 and cfg["tile"] == 1024, cfg
    assert VARIANT == "medium", VARIANT
    src = os.path.join(str(N46.OUT), "train")
    coco = json.load(open(os.path.join(src, "_annotations.coco.json")))
    by_frame = {}
    for im in coco["images"]:
        by_frame.setdefault((im["seq"], im["frame"]), []).append(im)

    det = D.TiledDetector(D.RFDETRDetector(conf=cfg["conf"], variant=VARIANT),
                          tile=cfg["tile"], overlap=D.OVERLAP)
    anns = list(coco["annotations"])
    nid = max((a["id"] for a in anns), default=0) + 1
    added = 0
    for clip in N46.SPLIT["train"]:
        for f, box in regions(clip, det):
            for im in by_frame.get((clip, f), []):
                # the build's OWN cut, so a region lands where a GT box would have
                for b in N46.tile_boxes([box], tuple(im["tile"])):
                    anns.append({"id": nid, "image_id": im["id"],
                                 "category_id": IGNORE_LABEL, "bbox": b,
                                 "area": round(b[2] * b[3], 2), "iscrowd": 0})
                    nid += 1
                    added += 1
    _void_bf(anns, added)
    dest = os.path.join(str(N46.OUT).replace("n46_dataset", "n67_ignore"))
    os.makedirs(os.path.join(dest, "train"), exist_ok=True)
    for im in coco["images"]:
        link = os.path.join(dest, "train", im["file_name"])
        if not os.path.exists(link):
            os.symlink(os.path.join(src, im["file_name"]), link)
    cats = list(coco["categories"]) + [{"id": IGNORE_LABEL, "name": "ignore",
                                        "supercategory": "none"}]
    json.dump({**coco, "annotations": anns, "categories": cats},
              open(os.path.join(dest, "train", "_annotations.coco.json"), "w"))
    os.makedirs(os.path.join(dest, "valid"), exist_ok=True)
    vsrc = os.path.join(str(N46.OUT), "valid")
    for fn in sorted(os.listdir(vsrc)):
        link = os.path.join(dest, "valid", fn)
        if not os.path.exists(link):
            os.symlink(os.path.join(vsrc, fn), link)
    print(f"\n  IG: {len(coco['images'])} tiles, "
          f"{len(coco['annotations'])} player boxes "
          f"+ {added} ignore regions (category {IGNORE_LABEL}) -> "
          f"{os.path.relpath(dest, ROOT)}")
    print(f"  valid linked whole and UNCHANGED - no ignore regions there, so the "
          f"checkpoint selection metric is the same quantity as FT's")
    return 0


def _keep(b, gt, lo, hi, med_h):
    """The predicate build() uses, in ONE place so demo() can exercise the real thing.

    regions() used to inline this, so demo() - which called neither regions() nor build() -
    could not fail on any defect in it. Five planted: BUCKET="off-pitch", GT_IOU_MAX=0.99,
    MIN_REGIONS=0, PER_ID_FLOOR sign, IGNORE_LABEL desynced. All five survived.
    """
    import n40_occlusion_recall as R  # noqa: PLC0415
    import n61_unlabelled_audit as N61  # noqa: PLC0415
    if any(R._iou(b, t) >= GT_IOU_MAX for t in gt):
        return False                       # it IS a labelled player
    return N61.bucket(b, gt, lo, hi, med_h) == BUCKET


def demo():
    """The bars are declared, the predicate is exercised, and the constants are not copies."""
    import n61_unlabelled_audit as N61
    # the bucket this arm uses must be one n61 actually produces, spelled the same way
    src = open(N61.__file__).read()
    assert BUCKET in src, f"{BUCKET} is not a bucket n61_unlabelled_audit produces"
    for other in ("duplicate", "wrong-size", "off-pitch"):
        assert other in src, other
    # the bars must be ordered: a product pass implies a mechanism pass, never the reverse
    assert PRODUCT_BAR > MECHANISM_BAR > 0.3517, (PRODUCT_BAR, MECHANISM_BAR)
    # and the precision guard must sit above the fine-tune's measured rate, not below it,
    # or the arm is refused before it starts
    assert 0.53 < NON_GT_MAX < 2.277, NON_GT_MAX

    # THE PREDICATE, on a synthetic frame where each bucket is represented exactly once.
    # GT: one player 100 px tall with feet at y=500. lo/hi bracket it, med_h = 100.
    gt = [(200.0, 400.0, 240.0, 500.0)]
    lo, hi, med_h = 450.0, 550.0, 100.0
    cases = {
        "on-pitch":   ((600.0, 400.0, 640.0, 500.0), True),   # no overlap, right size, in band
        "duplicate":  ((210.0, 410.0, 250.0, 510.0), False),  # overlaps the GT player
        "wrong-size": ((600.0, 100.0, 640.0, 500.0), False),  # 400 px tall, 4x med_h
        "off-pitch":  ((600.0, 100.0, 640.0, 200.0), False),  # feet at 200, outside the band
    }
    for name, (box, want) in cases.items():
        got = _keep(box, gt, lo, hi, med_h)
        assert got == want, (name, box, got, want)
    # and the bucket rule must actually NAME them the way this file expects, not merely
    # contain the string somewhere in its source
    import n61_unlabelled_audit as _N61  # noqa: PLC0415
    assert _N61.bucket(cases["on-pitch"][0], gt, lo, hi, med_h) == BUCKET
    assert _N61.bucket(cases["off-pitch"][0], gt, lo, hi, med_h) == "off-pitch"
    assert _N61.bucket(cases["wrong-size"][0], gt, lo, hi, med_h) == "wrong-size"
    assert _N61.bucket(cases["duplicate"][0], gt, lo, hi, med_h) == "duplicate"

    # CONSTANTS MUST BE SHARED, NOT COPIED. IGNORE_LABEL is imported from n66 and
    # PER_ID_FLOOR from n46_finetune; assert they still are, since a copy could drift and
    # then the dataset's sentinel would not be the one the loss strips.
    import n46_finetune as _FT  # noqa: PLC0415
    import n66_ignore as _N66  # noqa: PLC0415
    assert IGNORE_LABEL == _N66.IGNORE_LABEL, (IGNORE_LABEL, _N66.IGNORE_LABEL)
    assert PER_ID_FLOOR == _FT.PER_ID_FLOOR, (PER_ID_FLOOR, _FT.PER_ID_FLOOR)
    # the tiled-IoU helper the VOID uses must actually compute IoU
    assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _bbox_iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert abs(_bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) - 1 / 3) < 1e-9

    # VOID (b) AND (f), exercised. A clean set passes; a region ON a player must be refused;
    # and too few regions must be refused. Both constants are now reachable from here.
    clean = [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
             {"image_id": 1, "category_id": IGNORE_LABEL, "bbox": [50, 50, 10, 10]}]
    assert _void_bf(clean, MIN_REGIONS) == 0.0
    # ABSOLUTE values, not values derived from the constants. `MIN_REGIONS - 1` is below the
    # floor whatever the floor is, and an IoU of 1.0 exceeds any threshold - so both of those
    # tests passed with the constants set to 0 and 0.99, which is the mutation surviving.
    # 999 must be refused by a floor of 1000, and IoU 2/3 must be refused by a bar of 0.5.
    onplayer = [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                {"image_id": 1, "category_id": IGNORE_LABEL, "bbox": [2, 0, 10, 10]}]
    assert abs(_bbox_iou([0, 0, 10, 10], [2, 0, 10, 10]) - 2 / 3) < 1e-9
    for bad, added, why in ((onplayer, MIN_REGIONS, "(b)"), (clean, 999, "(f)")):
        try:
            _void_bf(bad, added)
        except SystemExit as e:
            assert f"VOID {why}" in str(e), (why, str(e))
        else:
            raise AssertionError(f"VOID {why} did not refuse: added={added}, "
                                 f"MIN_REGIONS={MIN_REGIONS}, GT_IOU_MAX={GT_IOU_MAX}")
    # THE SCREEN MUST BE ABOVE EVERY SEED FT PRODUCED, or it would continue on noise; and
    # BELOW the mechanism bar, or it would stop on a result the rule calls a finding.
    assert SCREEN_CONTINUE >= max(FT_SEEDS), (SCREEN_CONTINUE, FT_SEEDS)
    assert SCREEN_CONTINUE < MECHANISM_BAR, (SCREEN_CONTINUE, MECHANISM_BAR)
    print("n67_ignore_arm.py ok - bars declared, bucket matches n61's own spelling")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--score" in a:
        raise SystemExit("N67 is pre-registered and NOT scored: no IG weights exist. Build "
                         "the dataset, train it on the PC with n66_ignore.patch() bound, "
                         "then score. The rule is in this file's docstring.")
    if "--build" in a:
        raise SystemExit(build())
    raise SystemExit(demo() or 0)
