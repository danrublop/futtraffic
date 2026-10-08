"""N61 probe A: WHAT are the boxes the fine-tune learned to suppress?

    .venv/bin/python model/validation/n61_unlabelled_audit.py --audit
    .venv/bin/python model/validation/n61_unlabelled_audit.py --demo

WHY THIS RUNS FIRST
    N53 says the three fine-tune seeds suppress "boxes matching no GT box" from
    2.277 to ~0.41-0.53 per frame, and reads that as the unlabelled people -
    officials, substitutes, ball boys - training as background. Review measured
    that 20.3% of those boxes on TEST actually overlap a GT player at
    0 < IoU < 0.5, i.e. they are second or mislocalised boxes on players the GT
    DOES contain. So the population has never been named, and every sentence
    about "officials" rests on it.

    N60: the literature's recipe (pseudo-label, then ignore) needs boxes that are
    really the missing people. Pseudo-labelling duplicates-on-real-players would
    mask out the very instances we want learned.

    This is a DESCRIPTIVE AUDIT. It has no bar, no arm and no verdict; it names a
    population. It runs on the TRAIN clips, because those are the frames that
    trained the model - N53's figure was TEST.

THE BUCKETS, fixed here so the counting cannot be tuned afterwards
    duplicate      overlaps a GT player at 0 < IoU < 0.5. NOT a person the GT
                   lacks; a second box on one it has.
    wrong-size     no GT overlap, and taller than 1.6x or shorter than 0.6x the
                   median GT height at that image row. One person is not that
                   shape; a merged pair or a fragment is.
    off-pitch      no GT overlap, plausible size, and its foot point falls
                   outside the 1st-99th PERCENTILE band of GT feet rows. The
                   first cut used the clip-wide min/max and scored 0.0% on every
                   clip: over 300 frames 22 players cover nearly every row, so
                   nothing could fall outside it. A bucket that cannot be
                   populated is not evidence that it is empty.
    on-pitch       no GT overlap, plausible size, foot point inside the GT band.
                   THIS is the bucket that can be an official, and the only one
                   the officials story is entitled to.
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model", "validation", "out", "stc2025")
TRAIN = ("117092", "117093", "118575", "132877")
TALL, SHORT = 1.6, 0.6


def bucket(b, gt_boxes, rows_lo, rows_hi, med_h):
    """One non-GT box -> one bucket. See the module docstring for the rules."""
    import n40_occlusion_recall as R  # noqa: PLC0415

    best = max((R._iou(tuple(b), tuple(g)) for g in gt_boxes), default=0.0)
    if best > 0:
        return "duplicate"
    h = b[3] - b[1]
    if h > TALL * med_h or h < SHORT * med_h:
        return "wrong-size"
    return "on-pitch" if rows_lo <= b[3] <= rows_hi else "off-pitch"


def audit():
    import n40_occlusion_recall as R  # noqa: PLC0415

    tot = collections.Counter()
    for q in TRAIN:
        path = os.path.join(STC, f"{q}.p21.dets.cache.json")
        if not os.path.exists(path):
            print(f"{q}: no stock cache, skipped")
            continue
        gt = R._gt(q)
        frames = {int(k): [d["xyxy"] for d in v if d.get("cls") in ("person", "player")]
                  for k, v in json.load(open(path))["frames"].items()}
        feet = [y + h for rows in gt.values() for _i, _x, y, _w, h in rows]
        hs = [h for rows in gt.values() for _i, _x, _y, _w, h in rows]
        lo, hi = (float(np.percentile(feet, 1)), float(np.percentile(feet, 99)))
        med_h = float(np.median(hs))
        c = collections.Counter()
        for f, boxes in frames.items():
            if f not in gt:
                continue
            g = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
            for b in boxes:
                if any(R._iou(tuple(b), tuple(t)) >= 0.5 for t in g):
                    continue                      # it IS a labelled player
                c[bucket(b, g, lo, hi, med_h)] += 1
        n = sum(c.values())
        print(f"{q}: {n:6d} non-GT boxes  " + "  ".join(
            f"{k} {100 * c[k] / max(1, n):5.1f}%" for k in
            ("duplicate", "wrong-size", "off-pitch", "on-pitch")))
        tot.update(c)
    n = sum(tot.values())
    print(f"\nPOOLED {n} non-GT boxes over the TRAIN clips:")
    for k in ("duplicate", "wrong-size", "off-pitch", "on-pitch"):
        print(f"   {tot[k]:7d}  {100 * tot[k] / max(1, n):5.1f}%  {k}")
    print(f"\nThe officials story is entitled to the on-pitch bucket only: "
          f"{100 * tot['on-pitch'] / max(1, n):.1f}% of what the fine-tune suppressed.")
    return tot


def demo():
    """The bucket rule, on boxes whose answer is known by construction."""
    gt = [(100.0, 100.0, 140.0, 200.0)]          # one GT player, 100 px tall
    lo, hi, med = 150.0, 900.0, 100.0
    assert bucket((105.0, 105.0, 135.0, 190.0), gt, lo, hi, med) == "duplicate"
    assert bucket((500.0, 300.0, 540.0, 800.0), gt, lo, hi, med) == "wrong-size"   # 500 px
    assert bucket((500.0, 690.0, 540.0, 730.0), gt, lo, hi, med) == "wrong-size"   # 40 px
    # feet ABOVE the topmost row any GT player occupies: the stand, not the pitch
    assert bucket((500.0, 40.0, 540.0, 140.0), gt, lo, hi, med) == "off-pitch"
    # ...and below the bottom-most: the near touchline bench
    assert bucket((500.0, 850.0, 540.0, 950.0), gt, lo, hi, med) == "off-pitch"
    assert bucket((500.0, 400.0, 540.0, 500.0), gt, lo, hi, med) == "on-pitch"
    # a box overlapping the GT AT ALL is a duplicate, however slightly - the point
    # is that it is not evidence of a person the GT lacks
    assert bucket((139.0, 199.0, 179.0, 299.0), gt, lo, hi, med) == "duplicate"
    print("n61 demo: bucket rule ok (duplicate / wrong-size / off-pitch / on-pitch)")


if __name__ == "__main__":
    if "--audit" in sys.argv[1:]:
        audit()
    else:
        demo()
