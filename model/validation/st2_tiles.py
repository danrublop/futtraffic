"""Build an RF-DETR training set of TILES from SoccerTrack v2's MOT ground truth.

WHY TILES AND NOT FRAMES
    RF-DETR has no letterbox. `square_resize_div_64` squashes the input to a
    square, so a 4096x1080 frame at 576 compresses x by 7.1x and y by 1.9x and
    shears every player. Training on whole frames is not a worse option, it is
    not an option, and Roboflow have said rectangular training is not coming.

    A 1024x1024 tile resized to 576 is isotropic. More importantly it is the
    SAME geometry we infer at (`detectors.TiledDetector`, at the tile size and
    overlap the shipped constructor passes it - `TILE` and `OVERLAP` below are
    READ from model/engine/detectors.py, never copied), so the model sees players at
    the same patch count in training and in production. That match is the whole
    point; a model trained at one scale and run at another is a different model.

FOUR DECISIONS, EACH MEASURED OR REASONED, NONE ARBITRARY
    KEEP_FRAC 0.60   A player clipped by a seam is a half-player, and feeding it
                     as a whole box teaches the model to fire on torsos. The
                     detector's tile overlap (`detectors.OVERLAP`) means the
                     same player is whole in the neighbouring tile, so dropping
                     the fragment costs nothing. SAHI uses the same convention.
    EMPTY_KEEP 0.10  Some tiles hold no player. A few are useful negatives
                     against crowd and advertising; all of them waste epochs.
    STRIDE 50        6,000 frames per clip at 25 fps is perhaps 240 independent
                     samples. Stride 50 is the honest count, and it is ~200 GB
                     of PNG the STC recipe re-decoded every epoch for nothing.
    SPLIT BY MATCH   The STC baseline shuffles frames WITHIN a clip, so frame
                     1041 trains and 1042 validates - their val mAP is train
                     accuracy. This repo has paid for that lesson once already
                     (`detector-eval-caches-share-clips`). 128057 and 132831 are
                     what `st2_mot.py` scores; they never appear in training.

`ponytail: writes JPEG, not PNG - a 1024 tile is 100 KB against 1.6 MB and the
detector never sees the difference. Upgrade path is quality=100 if a measurement
ever says otherwise.`
"""
import argparse
import collections
import csv
import inspect
import json
import os
import random
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "engine"))
import detectors  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

GT = _ST2 + "/mot"
CLIPS = f"{GT}/clips"
# Inference geometry, read off the constructor the product runs - the same place
# model/validation/shipped_config.py resolves it. This line was a copy,
# `TILE, OVERLAP = 1024, 0.10`: 8472cef put the shipped overlap back to 0.25 and
# the copy stayed at 0.10, while demo() compared spans() with detectors._spans
# at the copy's own value and so stayed green.
TILE = inspect.signature(detectors._coreml).parameters["tile"].default
OVERLAP = detectors.OVERLAP
KEEP_FRAC, EMPTY_KEEP, STRIDE = 0.60, 0.10, 50
# the official challenge split - test is what model/validation/st2_mot.py scores
SPLIT = {"train": ["117092", "117093", "118575", "118576", "118577", "118578"],
         "valid": ["128058"],
         "test": ["128057", "132831"]}


def spans(length, tile, overlap):
    """(offsets, tile_size) along one axis - the detector's own function.

    Imported rather than reimplemented, and it returns the SIZE as well as the
    offsets. An earlier version copied the body and dropped the size, which was
    exactly the bug: on a 1080-tall frame `_spans` returns a **1080**-tall tile
    (one row covers the axis), so training on a hardcoded 1024 fed the model
    players 5.5% shorter than inference does, and never showed it the bottom 56
    rows. rf-detr square-resizes, so that scale gap is real and invisible.
    """
    import sys
    sys.path.insert(0, "model/engine")
    import detectors
    return detectors._spans(length, tile, overlap)


def build(split, out_root, stride=STRIDE, seed=0):
    rng = random.Random(seed)
    out = os.path.join(out_root, split)
    os.makedirs(out, exist_ok=True)
    images, anns, iid, aid = [], [], 0, 0
    kept_boxes = clipped = empty_kept = 0

    for m in SPLIT[split]:
        clip = f"{CLIPS}/{m}.mp4"
        if not os.path.exists(clip):
            print(f"  {m}: NO CLIP, skipped")
            continue
        gt = collections.defaultdict(list)
        with open(f"{GT}/{m}.txt") as fh:
            for r in csv.reader(fh):
                if len(r) >= 6:
                    gt[int(r[0])].append(tuple(float(v) for v in r[2:6]))
        cap = cv2.VideoCapture(clip)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ys, th_ = spans(H, TILE, OVERLAP)
        xs, tw_ = spans(W, TILE, OVERLAP)
        offs = [(x, y) for y in ys for x in xs]
        n0 = iid
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % stride or f not in gt:
                f += 1
                continue
            for tx, ty in offs:
                th, tw = th_, tw_
                keep = []
                for x, y, w, h in gt[f]:
                    cx1, cy1 = max(x, tx), max(y, ty)
                    cx2, cy2 = min(x + w, tx + tw), min(y + h, ty + th)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    if (cx2 - cx1) * (cy2 - cy1) < KEEP_FRAC * w * h:
                        clipped += 1
                        continue
                    keep.append([cx1 - tx, cy1 - ty, cx2 - cx1, cy2 - cy1])
                if not keep:
                    if rng.random() > EMPTY_KEEP:
                        continue
                    empty_kept += 1
                name = f"{m}_{f:06d}_{tx}_{ty}.jpg"
                cv2.imwrite(os.path.join(out, name), frame[ty:ty + th, tx:tx + tw],
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                images.append({"id": iid, "file_name": name, "width": tw, "height": th})
                for b in keep:
                    anns.append({"id": aid, "image_id": iid, "category_id": 1,
                                 "bbox": [round(v, 2) for v in b],
                                 "area": round(b[2] * b[3], 2), "iscrowd": 0})
                    aid += 1
                kept_boxes += len(keep)
                iid += 1
            f += 1
        cap.release()
        print(f"  {m}: {iid - n0} tiles  ({W}x{H}, {len(offs)} per frame)")

    # category 0 is Roboflow's unannotated parent; rfdetr drops it and resolves
    # num_classes to 1. Keeping it puts us on the code path they test.
    json.dump({"images": images,
               "categories": [{"id": 0, "name": "objects", "supercategory": "none"},
                              {"id": 1, "name": "player", "supercategory": "objects"}],
               "annotations": anns},
              open(os.path.join(out, "_annotations.coco.json"), "w"))
    print(f"{split}: {len(images)} tiles, {kept_boxes} boxes "
          f"({kept_boxes / max(len(images), 1):.1f}/tile), {empty_kept} empty kept, "
          f"{clipped} seam fragments dropped")
    return len(images), kept_boxes


def demo():
    """Self-check: tile offsets must match the detector's, and no annotation may
    fall outside its tile."""
    import sys
    sys.path.insert(0, "model/engine")
    import detectors
    for length in (1080, 1504, 4096, 3840):
        # offsets AND size - dropping the size was the bug this used to hide
        assert spans(length, TILE, OVERLAP) == detectors._spans(length, TILE, OVERLAP), length
    root = "/tmp/st2_tiles_demo"
    n, b = build("valid", root, stride=600)
    d = json.load(open(f"{root}/valid/_annotations.coco.json"))
    wh = {i["id"]: (i["width"], i["height"]) for i in d["images"]}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        W, H = wh[a["image_id"]]
        assert x >= -0.01 and y >= -0.01 and x + w <= W + 0.01 and y + h <= H + 0.01, a
    assert n and b, (n, b)
    # THE CENTRAL CLAIM IS THAT THE TILE ON DISK IS THE TILE WE RECORDED.
    # Writing a half-width JPEG while recording "width": 1024 used to pass -
    # and would have trained the model at a scale the annotations do not
    # describe, which no downstream check could see.
    im = d["images"][0]
    got = cv2.imread(os.path.join(root, "valid", im["file_name"])).shape[:2]
    assert got == (im["height"], im["width"]), (got, im["height"], im["width"])
    print(f"demo ok: offsets match detectors._spans, {b} boxes all inside their tile")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_ST2 + "/ft_tiles")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        for s in ("train", "valid", "test"):
            build(s, a.out, a.stride)
