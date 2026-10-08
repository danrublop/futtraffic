"""A ball detector of its own: RF-DETR nano at 640, fine-tuned on native-scale ball tiles.

WHY THIS EXISTS
    The only ball the product could ever see is COCO's "sports ball" class on the
    stock player detector. The player fine-tune (plan-detector-finetune.md) is
    single-class `person`, so the day it replaces COCO that source is gone. The
    ball needs its own model whatever the fine-tune does.

    The first one (ball-detector-trained-not-yet-scored (internal note), 2026-08-29)
    lived in a session scratchpad under /tmp and was lost with it: checkpoint,
    tiler and trainer. This file is the tiler and trainer, recovered from that
    session's log and committed; every output goes under model/validation/out/ball/.

WHAT CHANGED FROM THE LOST RUN, AND WHY
    1. The lost run split its four held-out matches two/two too (916/159/162 tiles,
       from that session's log), but which two were valid is not recorded. Here it
       is a rule: sorted, the first two are valid, the last two are test. The
       tile counts (916/162/159) suggest it is the lost run's split reversed, so
       the old 0.913 was scored on what is valid here, and is not comparable to
       this file's test number.
    2. A box cut by the tile edge kept its full width at a clamped origin; it is
       clipped to the tile now.
    Everything else is as the lost run: HF martinjolif/football-ball-detection,
    640 px windows at native scale containing the ball at a random offset (seed 0),
    nano@640, batch 4 x accum 4, lr 1e-4, no multi-scale.

THE DATA IS GONE, AND `fetch` MUST NOT BE RUN AGAIN
    2026-09-19 licence audit: martinjolif/football-ball-detection tags itself CC BY 4.0
    but is a Roboflow re-export of the DFL Bundesliga Data Shootout clips - the same
    chain as `role_clf.pt`, by the same route (filenames are DFL clip ids; six of its
    fourteen clips are train_roles.py's own HOLD_OUT/VAL tuples). DFL's terms are
    competition-use-only with delete-all-copies, and a re-uploader cannot grant rights
    they never held. `out/ball/{src,tiles}` were deleted, 395 MB (dfl-dataset (not published)).
      - `fetch` and `tiles` below would RE-BREACH those terms. Do not run them.
      - the checkpoint is kept and flagged, as role_clf.pt is: research only, NEVER
        shipped, wired into nothing. jobs.py passes ball=True since 66fba82 (2026-09-23),
        but that runs the STOCK detector's ball - no fine-tuned checkpoint is wired.
      - the clean replacement trains this same nano on TeamTrack ball annotations
        (MIT, data/footage/soccertrack/README.md), which is already the population N48
        scores on - so it needs a clip-disjoint split against that.

RULE, FIXED BEFORE THE RUN (committed with this file)
    split      train = every clip not held out; valid = 42ba34, 4b770a; test =
               538438, a9f16c. Whole matches, never frames.
    run        one run, 12 epochs, n=1 configuration. No sweep.
    select     the checkpoint rfdetr keeps as best on VALID (checkpoint_best_ema).
    test       scored ONCE on TEST with that checkpoint, after training ends.
    pass       test mAP50 >= 0.85. Below that, the tile-level model is not good
               enough to be worth scoring on SoccerTrack.
    NOT a product claim. A tile mAP on broadcast footage says nothing about our
    panorama; the claim that matters is the next step - ball_eval.py on the six
    held-out TeamTrack clips (soccer_top test 1710-1800, soccer_side x3; MIT, and this
    repo has long mislabelled them SoccerTrack v1),
    scored against the COCO detector on the same frames, conf 0.05, 20 px gate,
    with the frame subset and its share stated.

    .venv/bin/python model/validation/ball_train.py fetch     # dataset -> out/ball/src
    .venv/bin/python model/validation/ball_train.py tiles     # -> out/ball/tiles/{train,valid,test}
    .venv/bin/python model/validation/ball_train.py train     # -> out/ball/run/ (~9 min/epoch on the Mac)
    .venv/bin/python model/validation/ball_train.py test      # scores TEST once, writes out/ball/test.json
    .venv/bin/python model/validation/ball_train.py --demo    # self-check of the split and the box clip
"""
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "model", "validation", "out", "ball")
SRC = os.path.join(OUT, "src")
TILES = os.path.join(OUT, "tiles")
RUN = os.path.join(OUT, "run")
T = 640
HELD = sorted({"538438", "a9f16c", "42ba34", "4b770a"})
VALID, TEST = set(HELD[:2]), set(HELD[2:])
EPOCHS = 12


def clip_of(path):
    return os.path.basename(path).split("_")[0]


def split_of(path):
    c = clip_of(path)
    return "valid" if c in VALID else "test" if c in TEST else "train"


def clip_box(x, y, w, h):
    """A box in tile coordinates, clipped to the tile; None if nothing is left."""
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, T), min(y + h, T)
    return None if x1 <= x0 or y1 <= y0 else [x0, y0, x1 - x0, y1 - y0]


#: The licence refusal, as a runnable check rather than a paragraph. A guard that
#: lives only in a docstring has cost this repo a headline twice
#: (docs-guard-must-have-a-runnable-check (internal note)).
DFL_REFUSAL = (
    "REFUSED: martinjolif/football-ball-detection is a Roboflow re-export of the DFL\n"
    "Bundesliga Data Shootout clips, whose terms are competition-use-only with\n"
    "delete-all-copies. The copies here were deleted on 2026-09-19 (dfl-dataset (not published));\n"
    "fetching or re-tiling them would re-breach those terms. Train the clean replacement\n"
    "on TeamTrack ball annotations (MIT) instead - see this file's header.")


def fetch():
    raise SystemExit(DFL_REFUSAL)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="martinjolif/football-ball-detection", repo_type="dataset",
                      local_dir=SRC, max_workers=2)


def tiles():
    raise SystemExit(DFL_REFUSAL)
    import cv2
    random.seed(0)
    data = glob.glob(f"{SRC}/**/data.yaml", recursive=True)
    base = os.path.dirname(data[0]) if data else SRC
    srcs = sorted(q for sp in ("train", "valid", "test")
                  for q in glob.glob(f"{base}/{sp}/images/*.jpg"))
    for split in ("train", "valid", "test"):
        d = os.path.join(TILES, split)
        os.makedirs(d, exist_ok=True)
        imgs, anns, iid, aid = [], [], 0, 0
        for ip in (q for q in srcs if split_of(q) == split):
            im = cv2.imread(ip)
            ih, iw = im.shape[:2]
            lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            boxes = []
            for line in (open(lp) if os.path.exists(lp) else []):
                p = line.split()
                if len(p) >= 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:5])
                    boxes.append((cx * iw, cy * ih, bw * iw, bh * ih))
            for (cx, cy, _, _) in boxes:
                # a window CONTAINING the ball at a random offset, so the model does
                # not learn "the ball is in the middle of the tile"
                x0 = int(min(max(cx - random.uniform(0.15, 0.85) * T, 0), iw - T))
                y0 = int(min(max(cy - random.uniform(0.15, 0.85) * T, 0), ih - T))
                tile = im[y0:y0 + T, x0:x0 + T]
                if tile.shape[:2] != (T, T):
                    continue
                iid += 1
                name = f"{iid:06d}.jpg"
                cv2.imwrite(os.path.join(d, name), tile)
                imgs.append({"id": iid, "file_name": name, "width": T, "height": T})
                for (bx, by, bw, bh) in boxes:
                    b = clip_box(bx - bw / 2 - x0, by - bh / 2 - y0, bw, bh)
                    if b:
                        aid += 1
                        anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": b,
                                     "area": b[2] * b[3], "iscrowd": 0})
        json.dump({"images": imgs, "annotations": anns,
                   "categories": [{"id": 1, "name": "ball", "supercategory": "none"}]},
                  open(os.path.join(d, "_annotations.coco.json"), "w"))
        clips = sorted({clip_of(q) for q in srcs if split_of(q) == split})
        print(f"  {split}: {len(imgs)} tiles, {len(anns)} balls, {len(clips)} clips")


#: EVERY hyperparameter, in ONE place, because an arm trained at different settings
#: from its baseline measures the settings. That confound has already cost this repo a
#: headline once - N48's +0.3311 was tiling, not the fine-tune
#: (n48-ab-was-confounded-with-tile-size (internal note)) - and a second copy of these
#: numbers in a training script on another machine is the same failure waiting.
#: `n64_ball_negatives.pack()` writes this dict out so the PC cannot disagree with it.
#: SEED IS IN HERE NOW. rfdetr calls seed_everything only when TrainConfig.seed is not
#: None (rfdetr/training/module_model.py), and run_d1/training_config.json records
#: "seed": null - so D1, E1 and every ball arm before 2026-09-26 shuffled, augmented and
#: EMA'd unseeded. A paired arm at n=1 per side then carries one unmeasured seed draw
#: against a +0.05 bar, and D1' cannot reproduce D1 even on identical hardware.
#: n46_finetune.py already runs three seeds and takes the median for the player
#: fine-tune; the ball arms were the ones that did not.
HP = dict(batch_size=4, grad_accum_steps=4, lr=1e-4, num_workers=4, multi_scale=False,
          expanded_scales=False, tensorboard=False, early_stopping=False,
          checkpoint_interval=4, seed=0)


def train(dataset_dir: str = TILES, output_dir: str = RUN, epochs: int = EPOCHS):
    """The one training call. Parameterised so N52's clean arms reuse it rather
    than re-declaring a second set of hyperparameters that could drift from these."""
    from rfdetr import RFDETRNano
    RFDETRNano(resolution=T).train(
        dataset_dir=dataset_dir, output_dir=output_dir, epochs=epochs, **HP)


def test():
    """TEST, once, with the valid-selected checkpoint; COCO mAP50 via pycocotools."""
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from rfdetr import RFDETRNano
    out = os.path.join(OUT, "test.json")
    assert not os.path.exists(out), "TEST is scored once; test.json already exists"
    ck = os.path.join(RUN, "checkpoint_best_ema.pth")
    m = RFDETRNano(resolution=T, pretrain_weights=ck)
    gt = COCO(os.path.join(TILES, "test", "_annotations.coco.json"))
    dets = []
    for im in gt.dataset["images"]:
        r = m.predict(Image.open(os.path.join(TILES, "test", im["file_name"])), threshold=0.001)
        for (x0, y0, x1, y1), s in zip(r.xyxy, r.confidence):
            dets.append({"image_id": im["id"], "category_id": 1,
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "score": float(s)})
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    res = {"checkpoint": ck, "test_clips": sorted(TEST), "tiles": len(gt.dataset["images"]),
           "mAP50": round(float(ev.stats[1]), 4), "mAP50_95": round(float(ev.stats[0]), 4),
           "pass": bool(ev.stats[1] >= 0.85)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def demo():
    # The licence refusal must REFUSE, not merely be documented.
    for fn in (fetch, tiles):
        try:
            fn()
        except SystemExit as e:
            assert "REFUSED" in str(e), fn.__name__
        else:
            raise AssertionError(f"{fn.__name__} did not refuse: the DFL guard is decoration")
    assert VALID | TEST == set(HELD) and not VALID & TEST, "held-out matches split, disjoint"
    assert split_of("x/538438_12.jpg") == "test" and split_of("x/42ba34_1.jpg") == "valid"
    assert split_of("x/ffffff_1.jpg") == "train"
    assert clip_box(-5, 10, 20, 20) == [0, 10, 15, 20], "left edge clipped, not shifted"
    assert clip_box(630, 630, 20, 20) == [630, 630, 10, 10]
    assert clip_box(-30, 0, 20, 20) is None
    print("ball_train demo: ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--demo"
    {"fetch": fetch, "tiles": tiles, "train": train, "test": test, "--demo": demo}[cmd]()
