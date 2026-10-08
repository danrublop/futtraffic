"""N46 training, run on the training PC (CUDA). Fine-tunes RF-DETR Medium (Apache-2.0) on the N46 tile dataset.

Every run writes run_record.json next to its weights: the resolved settings, the seed, the dataset manifest's hash and
split, the GPU, the library versions, wall-clock, seconds per image, and peak GPU memory - so a number and the conditions
that made it never travel separately (research-training-protocol (not published) §2.2). The TEST clips are not in the dataset
and nothing here can open them; checkpoint selection uses the VALID split only.

    python train.py --epochs E --batch B --seed 0        # n46_dataset/ -> runs/seed0/
    (interrupted? run the same line again: it continues from runs/seed0/last.ckpt)

RECIPE, fixed before any training (plan-detector-finetune (not published) §7):
    lr 1e-4 (rfdetr default), COSINE decay over the run - rfdetr's default step drop is at epoch 100, so in a 30-epoch
    run the rate would never fall; multi-scale kept but NOT expanded (480-704 px, not 416-736: training does not shrink
    the smallest players below 0.83x of what inference sees); a checkpoint every 5 epochs kept, so the epoch can be
    re-chosen on the rule's own recall later without retraining; the model used is checkpoint_best_total (valid mAP50-95).
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LR = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(HERE, "n46_dataset"))
    ap.add_argument("--out", default="", help="default runs/seed<seed>")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=0, help="gradient accumulation; 0 = 16 // batch (effective batch 16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda", help="cuda on the training PC; mps/cpu only to check the script runs")
    a = ap.parse_args()
    a.out = a.out or os.path.join(HERE, "runs", f"seed{a.seed}")
    accum = a.accum or max(1, 16 // a.batch)

    import traceback
    from importlib.metadata import version
    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, "run_record.json")
    previous = json.load(open(rec_path)) if os.path.exists(rec_path) else None
    last = os.path.join(a.out, "last.ckpt")
    resume = last if previous and previous.get("status") != "done" and os.path.exists(last) else None
    if previous and previous.get("status") == "done":
        sys.exit(f"{a.out} already holds a finished run - use another --out or --seed")
    record = {"args": vars(a), "status": "importing", "previous": previous}
    try:
        import numpy as np
        import torch
        import rfdetr.training  # noqa: F401 - the training extra; a missing Windows wheel fails here, in minute one
        from rfdetr import RFDETRMedium
    except Exception:                                        # a report is written even when nothing can start
        record["status"] = "error: " + traceback.format_exc()[-1500:]
        json.dump(record, open(rec_path, "w"), indent=1)
        sys.exit(record["status"])

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    cuda = a.device == "cuda"
    if cuda and not torch.cuda.is_available():
        sys.exit("no CUDA GPU visible to PyTorch - see README.md section 1")
    cap = torch.cuda.get_device_capability(0) if cuda else (0, 0)
    # Turing (7.x, e.g. RTX 2060 SUPER) has no native bfloat16; name fp16 explicitly rather than trust "auto".
    amp_dtype = "fp16" if cuda and cap[0] < 8 else "auto"

    manifest = json.load(open(os.path.join(a.dataset, "manifest.json")))
    seqs = {sp: {im["seq"] for im in json.load(open(os.path.join(a.dataset, sp, "_annotations.coco.json")))["images"]} for sp in ("train", "valid")}
    sp = manifest["split"]                                   # by each image's own match id, not its file name
    assert seqs["train"] <= set(sp["train"]) and seqs["valid"] <= set(sp["valid"]), seqs
    assert not (seqs["train"] & seqs["valid"]) and not ((seqs["train"] | seqs["valid"]) & set(sp["test_not_written"])), seqs
    n_train = len(json.load(open(os.path.join(a.dataset, "train", "_annotations.coco.json")))["images"])
    start_epoch, bad_ckpt = 0, None
    if resume:
        try:                                                 # our own checkpoint; rfdetr's args are plain objects
            start_epoch = torch.load(resume, map_location="cpu", weights_only=False)["epoch"] + 1
        except Exception:                                    # a save cut off by a power loss: set it aside, fall back
            bad_ckpt = resume + ".bad"
            os.replace(resume, bad_ckpt)
            older = sorted(glob.glob(os.path.join(a.out, "checkpoint_*.ckpt")), key=os.path.getmtime)
            resume = older[-1] if older else None
            start_epoch = torch.load(resume, map_location="cpu", weights_only=False)["epoch"] + 1 if resume else 0
        if os.path.exists(os.path.join(a.out, "metrics.csv")):  # Lightning rewrites it on resume; keep the earlier epochs
            shutil.copy(os.path.join(a.out, "metrics.csv"), os.path.join(a.out, f"metrics_before_epoch{start_epoch}.csv"))

    record = {"args": vars(a), "resumed_from_epoch": start_epoch if resume else None, "resumed_from": resume,
              "corrupt_checkpoint_set_aside": bad_ckpt, "previous": previous, "grad_accum_steps": accum, "effective_batch": a.batch * accum, "amp_dtype": amp_dtype,
              "dataset": {"dir": os.path.abspath(a.dataset), "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                          "split": manifest["split"], "constants": manifest["constants"], "train_images": n_train},
              "env": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "rfdetr": version("rfdetr"),
                      "device": a.device, "gpu": torch.cuda.get_device_name(0) if cuda else None, "capability": list(cap),
                      "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None},
              "status": "started", "lr": LR, "lr_scheduler": "cosine to 0 over the run, no warmup",
              "multi_scale": "on, expanded_scales off (480-704 px); the product infers at 576 only"}
    json.dump(record, open(rec_path, "w"), indent=1)

    model = RFDETRMedium()                                   # downloads rf-detr-medium.pth (Apache-2.0) on first use
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model.train(device=a.device, resume=resume, dataset_dir=a.dataset, output_dir=a.out, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=accum,
                    lr=LR, lr_scheduler="cosine", expanded_scales=False, seed=a.seed, num_workers=a.workers, amp_dtype=amp_dtype,
                    tensorboard=False, early_stopping=False, run_test=False, checkpoint_interval=5, progress_bar="tqdm")
        record["status"] = "done"
    except torch.cuda.OutOfMemoryError as e:
        record["status"] = f"OOM: {str(e)[:200]}"
    except RuntimeError as e:                                # some OOMs surface as RuntimeError from cuDNN
        record["status"] = ("OOM: " if "out of memory" in str(e).lower() else "error: ") + str(e)[:300]
    except Exception:                                        # anything else: say what, instead of leaving "started"
        record["status"] = "error: " + traceback.format_exc()[-1500:]
    secs = time.time() - t0
    ran = a.epochs - start_epoch                             # only a finished run says how many epochs these seconds bought
    record.update(seconds=round(secs, 1),
                  seconds_per_image_epoch=round(secs / (n_train * ran), 4) if record["status"] == "done" and ran > 0 else None,
                  peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
                  peak_gpu_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2) if cuda else None,
                  outputs=sorted(f for f in os.listdir(a.out) if f.endswith(".pth")))
    if record["status"] == "done" and "checkpoint_best_total.pth" in record["outputs"]:
        best = os.path.join(a.out, "checkpoint_best_total.pth")
        try:
            record["best_total_source"] = torch.load(best, map_location="cpu", weights_only=False).get("best_total_source")
            record["reload_check"] = reload_check(best, a.dataset, a.device)
        except Exception:
            record["reload_check"] = {"reading": "error: " + traceback.format_exc()[-1500:]}
    json.dump(record, open(rec_path, "w"), indent=1)
    print(json.dumps({k: record.get(k) for k in ("status", "seconds", "seconds_per_image_epoch", "peak_gpu_mem_gb", "outputs", "reload_check")}))
    sys.exit(0 if record["status"] == "done" else 3)


def reload_check(ckpt, dataset, device, n=20):
    """Reload the saved checkpoint with rfdetr AND through the product's own wrapper (model/engine/detectors.RFDETRDetector,
    the packed copy) and run both on validation tiles.
    Found on the Mac dry run: a one-epoch smoke model reloads correctly but its top score is ~0.02, so at the product's
    conf 0.5 it draws nothing - fine for a smoke run, fatal to discover after a night of training. Recorded, not judged:
    a REAL run should show tens of boxes per tile at >= 0.5 against the ground truth's count beside it."""
    import glob
    import numpy as np
    import cv2
    from rfdetr import RFDETRMedium
    kw = {"device": device} if device != "cuda" else {}
    m = RFDETRMedium(pretrain_weights=ckpt, num_classes=1, **kw)
    coco = json.load(open(os.path.join(dataset, "valid", "_annotations.coco.json")))
    gt_per = {}
    for ann in coco["annotations"]:
        gt_per[ann["image_id"]] = gt_per.get(ann["image_id"], 0) + 1
    with_gt = [im for im in coco["images"] if gt_per.get(im["id"])]
    ims = with_gt[:: max(1, len(with_gt) // n)][:n]                 # spread over the whole validation match, not its first frames
    tops, at05, gt = [], [], []
    for im in ims:
        img = np.ascontiguousarray(cv2.imread(os.path.join(dataset, "valid", im["file_name"]))[:, :, ::-1])
        d = m.predict(img, threshold=0.001)
        c = np.asarray(d.confidence) if len(d) else np.zeros(0)
        tops.append(float(c.max()) if len(c) else 0.0)
        at05.append(int((c >= 0.5).sum()))
        gt.append(gt_per[im["id"]])
    # ...and through the PRODUCT's wrapper, which maps class ids to names off the checkpoint (a mapping that once silently
    # dropped every box of a fine-tuned model - model/engine/detectors._checkpoint_classes) and keeps only player classes at 0.5
    sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "engine")]
    import detectors as D
    prod = D.RFDETRDetector(weights=ckpt, conf=0.5, variant="medium", **kw)
    via_product = [len(prod.detect(cv2.imread(os.path.join(dataset, "valid", im["file_name"])))) for im in ims]
    ok = np.mean(at05) >= 0.5 * np.mean(gt) and np.mean(via_product) >= 0.9 * np.mean(at05)
    return {"tiles": len(ims), "top_score_median": round(float(np.median(tops)), 4), "boxes_at_0.5_per_tile": round(float(np.mean(at05)), 2),
            "product_wrapper_boxes_per_tile": round(float(np.mean(via_product)), 2), "gt_boxes_per_tile": round(float(np.mean(gt)), 2),
            "reading": "usable" if ok else ("scores too low for conf 0.5 - undertrained or a load problem" if np.mean(at05) < 0.5 * np.mean(gt)
                                            else "the product's wrapper drops boxes rfdetr keeps - a class-name mapping problem")}


if __name__ == "__main__":                                   # required on Windows: the data loader spawns workers
    main()
