"""N46 dataset: SoccerTrack v2 boxes cut into the exact tiles the product feeds the detector, in the COCO layout RF-DETR trains on.

WHAT IT REPRODUCES
    The STC2025 winner fine-tuned its detector on this rig's own training clips (research-stc2025-winners (not published)) and
    that detector is 75% of the gap to them (docs/experiments/n45-winner-boxes.md). This builds the same kind of training set for
    OUR detector (RF-DETR Medium, Apache-2.0) from the same data (SoccerTrack v2, CC BY 4.0: credit, link the licence,
    indicate changes), but in the product's own geometry: model/engine/detectors.TiledDetector's grid (tile 1024, overlap
    0.25) and its exact crop + resize to the model's 576 input (INTER_AREA when downscaling), so the model is trained on
    what it is shown at inference - the finding the winner's pixels and SAHI's +7-8 AP both point at.

SPLIT, by match (every clip is a different match on a different date; matchTitle/matchDatetime in raw/<seq>/)
    TRAIN  117092 117093 118575 132831 132877      (three canvases: 4096x1080/1084, 3840x1504, 3840x1906 - the night rig)
    VALID  118576                                   (checkpoint / epoch selection only)
    TEST   118577 118578 128057 128058              NOT WRITTEN HERE. The held-out clips of N40-N45; opened once, by
                                                    model/validation/n46_finetune.py, never by training.
    Leakage stated, not hidden: one club's teams recur - 3 of the 4 TEST clips share a team with TRAIN (B, C1, the soccer
    circle, B2; 118578 is the same pairing, B vs C1, as 117092). 128057 (TEST, the challenge clip) has teams found
    nowhere else: it is the only clean reading and is reported on its own.

DECLARED CONSTANTS (chosen before any training; the reviewer and the smoke test may question them, nothing tunes them)
    FRAME_STEP = 10       every 10th frame (2.5 Hz) on a 5-tile canvas: consecutive frames are near-duplicates. Scaled per
                          clip by tiles-per-frame / 5 so every training MATCH contributes about equally (the night rig's
                          3840x1906 canvas cuts into 15 tiles; at a flat step it was 34% of all training tiles - found on
                          the first build's counts, fixed before any training)
    VISIBLE_MIN = 0.4     a box cut by a tile edge is kept if at least 40% of its area is inside the tile
    MIN_PX = 2            ...and at least 2 px wide and tall after the resize to 576
    EMPTY_KEEP = 4        tiles with no ground-truth box: 1 in 4 kept (they carry the bench, stands and staff)
    CLASS = "person"      one class, named "person" like the COCO model's: model/engine/track.assign_teams forces any track
                          named "player" onto a team, overruling the kit-colour officials gate, so a "player" name would
                          send the fine-tuned arm down a different team/stitch/identity path than STOCK (review,
                          2026-09-18, before any training). The ground truth holds the 22 players and NO officials, so officials, benches and
                          crowds are background - the winner's choice, and the reason their detector drew 0.2-0.6 non-
                          player boxes per frame to our 2.2. Risk stated: a model that learns "this referee kit is
                          background" may also suppress a player in a similar kit on another ground.

    .venv/bin/python model/validation/n46_dataset.py            # self-check (synthetic frame: a box lands where it should)
    .venv/bin/python model/validation/n46_dataset.py --build    # -> model/validation/out/n46_dataset/{train,valid}/ + manifest.json
    .venv/bin/python model/validation/n46_dataset.py --smoke    # -> model/validation/out/n46_smoke/ (about 1% of it, for the PC smoke test)
    .venv/bin/python model/validation/n46_dataset.py --sheet    # -> a contact sheet of random tiles with their boxes, to LOOK at
    .venv/bin/python model/validation/n46_dataset.py --pack     # -> zips of both, plus the PC scripts, for copying to the training PC
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "engine"), os.path.join(ROOT, "model", "validation")]
MOT = _ST2 + "/mot"
CLIPS = f"{MOT}/clips"
OUT = os.path.join(ROOT, "model/validation/out/n46_dataset")
SMOKE = os.path.join(ROOT, "model/validation/out/n46_smoke")
SPLIT = {"train": ("117092", "117093", "118575", "132831", "132877"), "valid": ("118576",)}
TEST = ("118577", "118578", "128057", "128058")
FRAME_STEP, VISIBLE_MIN, MIN_PX, EMPTY_KEEP, TILE, SIZE = 10, 0.4, 2, 4, 1024, 576
CLASS = "person"
SEED = 0


def gt_boxes(seq):
    """{frame (0-based): [(x0, y0, x1, y1), ...]} in native pixels, from the MOT ground truth."""
    out = {}
    for line in open(f"{MOT}/{seq}.txt"):
        f, _i, x, y, w, h = line.split(",")[:6]
        x, y, w, h = float(x), float(y), float(w), float(h)
        out.setdefault(int(f), []).append((x, y, x + w, y + h))
    return out


def grid(w, h):
    import detectors as D  # noqa: PLC0415
    return D.TiledDetector(D.StubDetector(), tile=TILE).grid(w, h)


def crop(frame, tile):
    """The product's exact crop + resize (model/engine/detectors.TiledDetector.detect)."""
    x, y, tw, th = tile
    return cv2.resize(frame[y:y + th, x:x + tw], (SIZE, SIZE), interpolation=cv2.INTER_AREA if tw >= SIZE else cv2.INTER_LINEAR)


def tile_boxes(boxes, tile):
    """Native-pixel boxes -> [x, y, w, h] in the tile's 576 x 576 image, cut to the tile, VISIBLE_MIN / MIN_PX applied."""
    x, y, tw, th = tile
    sx, sy = SIZE / tw, SIZE / th
    out = []
    for x0, y0, x1, y1 in boxes:
        cx0, cy0, cx1, cy1 = max(x0, x), max(y0, y), min(x1, x + tw), min(y1, y + th)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < VISIBLE_MIN * (x1 - x0) * (y1 - y0):
            continue
        bx, by, bw, bh = (cx0 - x) * sx, (cy0 - y) * sy, (cx1 - cx0) * sx, (cy1 - cy0) * sy
        if bw >= MIN_PX and bh >= MIN_PX:
            out.append([round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)])
    return out


def build_split(name, seqs, root, frame_step=FRAME_STEP, limit=None):
    rng = random.Random(f"{SEED}-{name}")
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    images, anns, counts = [], [], {}
    per_seq = -(-limit // len(seqs)) if limit else None                 # a smoke set samples every clip, not the first one
    for seq in seqs:
        start = len(images)
        gt = gt_boxes(seq)
        cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tiles = grid(w, h)
        step = frame_step * max(1, round(len(tiles) / 5))
        f, kept, empty = 0, 0, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if f % step == 0 and f in gt:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                for ti, tile in enumerate(tiles):
                    tb = tile_boxes(gt[f], tile)
                    if not tb:
                        if rng.randrange(EMPTY_KEEP):
                            continue
                        empty += 1
                    fn = f"{seq}_f{f:05d}_t{ti:02d}.jpg"
                    cv2.imwrite(os.path.join(d, fn), crop(frame, tile), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    iid = len(images)
                    images.append({"id": iid, "file_name": fn, "width": SIZE, "height": SIZE, "seq": seq, "frame": f, "tile": list(tile)})
                    for b in tb:
                        anns.append({"id": len(anns), "image_id": iid, "category_id": 0, "bbox": b, "area": round(b[2] * b[3], 2), "iscrowd": 0})
                    kept += 1
                    if per_seq and len(images) - start >= per_seq:
                        break
            if per_seq and len(images) - start >= per_seq:
                break
            f += 1
        counts[seq] = {"tiles": kept, "empty_tiles": empty, "frame_size": [w, h], "tiles_per_frame": len(tiles), "frame_step": step}
        print(f"  {name} {seq}: {kept} tiles ({empty} empty) from a {w}x{h} canvas, {len(tiles)} tiles per frame", flush=True)
    coco = {"info": {"description": "SoccerTrack v2 (CC BY 4.0) boxes in the fut-traffic product's tile geometry"},
            "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
            "categories": [{"id": 0, "name": CLASS, "supercategory": "none"}], "images": images, "annotations": anns}
    json.dump(coco, open(os.path.join(d, "_annotations.coco.json"), "w"))
    return {"images": len(images), "boxes": len(anns), "per_clip": counts}


def _sha(path):
    """sha256 of a file - `hashlib.file_digest` (3.11+) streams it."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def build(root=OUT, frame_step=FRAME_STEP, limits=None):
    for s in TEST:
        assert all(s not in v for v in SPLIT.values()), f"test clip {s} in a training split"
    assert not set(SPLIT["train"]) & set(SPLIT["valid"])
    if os.path.exists(root):
        shutil.rmtree(root)
    res = {name: build_split(name, seqs, root, frame_step, (limits or {}).get(name)) for name, seqs in SPLIT.items()}
    manifest = {"split": {**{k: list(v) for k, v in SPLIT.items()}, "test_not_written": list(TEST)},
                "constants": {"FRAME_STEP": frame_step, "VISIBLE_MIN": VISIBLE_MIN, "MIN_PX": MIN_PX, "EMPTY_KEEP": EMPTY_KEEP,
                              "TILE": TILE, "SIZE": SIZE, "CLASS": CLASS, "SEED": SEED},
                "gt_sha256": {s: _sha(f"{MOT}/{s}.txt") for s in sum(SPLIT.values(), ())},
                "clip_bytes": {s: os.path.getsize(f"{CLIPS}/{s}.mp4") for s in sum(SPLIT.values(), ())},
                "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                "counts": res, "attribution": "SoccerTrack v2, CC BY 4.0, https://github.com/AtomScott/SoccerTrack-v2 - cropped, resized, re-encoded"}
    json.dump(manifest, open(os.path.join(root, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_clip"} for k, v in res.items()}))
    return manifest


def sheet(root=OUT, n=12):
    rng = random.Random(1)
    coco = json.load(open(os.path.join(root, "train", "_annotations.coco.json")))
    by = {}
    for a in coco["annotations"]:
        by.setdefault(a["image_id"], []).append(a["bbox"])
    with_boxes = [im for im in coco["images"] if im["id"] in by]
    picks = rng.sample(with_boxes, min(n, len(with_boxes)))
    tiles = []
    for im in picks:
        img = cv2.imread(os.path.join(root, "train", im["file_name"]))
        for x, y, w, h in by[im["id"]]:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
        cv2.putText(img, im["file_name"], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out = os.path.join(root, "_check_sheet.jpg")
    cv2.imwrite(out, np.vstack(rows))
    print("->", out)


def pack():
    """Zips for a GitHub release (at most 2 GB a file), all rooted at n46_pc/, so extracting every one of them into C:\\
    gives the layout the scripts expect: n46_kit.zip (scripts, the product's detectors.py, the smoke set),
    n46_dataset_<k>.zip (the training set, cut by size; part 1 carries the manifest and annotations) and n46_clips.zip
    (the five held-out clips detect.py reads)."""
    import zipfile  # noqa: PLC0415
    pc = os.path.join(ROOT, "model/validation/n46_pc")
    dst = os.path.join(ROOT, "model/validation/out/n46_pack")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)

    def tree(src):
        return sorted((os.path.join(dp, fn), os.path.join("n46_pc", os.path.relpath(os.path.join(dp, fn), os.path.dirname(src))))
                      for dp, _dn, fns in os.walk(src) for fn in fns)

    with zipfile.ZipFile(os.path.join(dst, "n46_kit.zip"), "w", zipfile.ZIP_STORED) as z:     # JPEGs do not compress further
        for fn in ("README.md", "train.py", "smoke.py", "detect.py"):
            z.write(os.path.join(pc, fn), os.path.join("n46_pc", fn))
        z.write(os.path.join(ROOT, "model", "engine", "detectors.py"), os.path.join("n46_pc", "detectors.py"))   # the product's tiler
        for f, arc in tree(SMOKE):
            z.write(f, arc)
    files = sorted(tree(OUT), key=lambda fa: not fa[0].endswith(".json"))    # manifest + annotations land in part 1
    part, size, z = 0, 0, None
    for f, arc in files:
        if z is None or size + os.path.getsize(f) > 1.9e9:
            z and z.close()
            part, size = part + 1, 0
            z = zipfile.ZipFile(os.path.join(dst, f"n46_dataset_{part}.zip"), "w", zipfile.ZIP_STORED)
        z.write(f, arc)
        size += os.path.getsize(f)
    z.close()
    with zipfile.ZipFile(os.path.join(dst, "n46_clips.zip"), "w", zipfile.ZIP_STORED) as z:
        for seq in ("118576", "118577", "118578", "128057", "128058"):
            z.write(os.path.join(CLIPS, f"{seq}.mp4"), os.path.join("n46_pc", "clips", f"{seq}.mp4"))
    for name in sorted(os.listdir(dst)):
        print("->", os.path.join(dst, name), f"{os.path.getsize(os.path.join(dst, name)) / 1e6:.0f} MB")


def demo():
    """A synthetic 4096x1080 frame with one white box: after the product's crop + resize, the box's label must sit
    on the white pixels in every tile that holds enough of it, and a box mostly outside a tile must be dropped."""
    frame = np.zeros((1080, 4096, 3), np.uint8)
    box = (1000.0, 500.0, 1040.0, 600.0)                                  # straddles the first two tiles' overlap
    frame[500:600, 1000:1040] = 255
    hits = 0
    for tile in grid(4096, 1080):
        tb = tile_boxes([box], tile)
        img = crop(frame, tile)
        for x, y, w, h in tb:
            inner = img[int(y + 1):int(y + h - 1), int(x + 1):int(x + w - 1)]
            assert inner.size and inner.mean() > 200, (tile, x, y, w, h, inner.mean() if inner.size else None)
            hits += 1
    assert hits == 2, hits                                                # tiles at x 0 and 768 both hold it whole
    assert tile_boxes([(1020.0, 500.0, 1100.0, 600.0)], (0, 0, 1024, 1080)) == []      # 5% inside: dropped
    assert len(tile_boxes([(990.0, 500.0, 1030.0, 600.0)], (0, 0, 1024, 1080))) == 1   # 85% inside: kept, cut
    print("n46 dataset demo ok: labels land on their pixels after the product's crop and resize; edge rule holds")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--build" in a:
        build()
    elif "--smoke" in a:
        build(SMOKE, frame_step=FRAME_STEP * 20, limits={"train": 240, "valid": 60})
    elif "--sheet" in a:
        sheet()
    elif "--pack" in a:
        pack()
    else:
        demo()
