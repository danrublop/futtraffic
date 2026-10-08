"""N46 smoke test, run on the training PC first. About 45 minutes. Answers, before any real training:
    1. does PyTorch see the GPU, with how much memory, and can it do fp16;
    2. which batch size fits in its memory for RF-DETR Medium at 576 px (tries 2, 4, 8; stops at the first failure);
    3. how many seconds a full epoch of the real dataset will take, and so what 30 epochs x 3 seeds costs - timed as
       (3 epochs - 1 epoch) / 2 at the chosen batch, so start-up and checkpoint writing do not inflate it.
Writes smoke_report.json - send that file back; it is what the training plan's numbers will be made from.

    python smoke.py                       # expects n46_smoke/ beside this file; takes no arguments
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    import torch
    rep = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if not rep["cuda_available"]:
        rep["verdict"] = "PyTorch cannot see a CUDA GPU: install the CUDA build of torch (README.md section 1)"
        json.dump(rep, open(os.path.join(HERE, "smoke_report.json"), "w"), indent=1)
        sys.exit(rep["verdict"])
    p = torch.cuda.get_device_properties(0)
    rep.update(gpu=torch.cuda.get_device_name(0), gpu_mem_gb=round(p.total_memory / 2**30, 2),
               capability=list(torch.cuda.get_device_capability(0)), cuda=torch.version.cuda)
    smoke = os.path.join(HERE, "n46_smoke")
    full = os.path.join(HERE, "n46_dataset")
    count = lambda d, s: len(json.load(open(os.path.join(d, s, "_annotations.coco.json")))["images"])  # noqa: E731
    n_smoke, n_smoke_valid = count(smoke, "train"), count(smoke, "valid")
    n_full, n_full_valid = (count(full, "train"), count(full, "valid")) if os.path.exists(full) else (None, None)
    free0, total = torch.cuda.mem_get_info()
    rep["gpu_free_at_start_gb"] = round(free0 / 2**30, 2)

    def run(bs, epochs, name):
        out = os.path.join(HERE, "runs", name)
        shutil.rmtree(out, ignore_errors=True)               # a rerun starts clean; train.py would otherwise resume it
        # --accum 1: one optimizer step per batch, so a one-epoch probe draws 30-120 steps of rfdetr's per-step training
        # scale (480-704 px, expanded_scales off) and meets the largest one; with accumulation it takes 15 steps and may not
        r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), "--dataset", smoke, "--out", out,
                            "--epochs", str(epochs), "--batch", str(bs), "--accum", "1"], cwd=HERE)
        p = os.path.join(out, "run_record.json")
        rec = json.load(open(p)) if os.path.exists(p) else {"status": f"no run record (exit {r.returncode})"}
        return r.returncode, {k: rec.get(k) for k in ("status", "seconds", "peak_gpu_mem_gb", "peak_gpu_reserved_gb", "reload_check")}

    rep["probes"] = {}
    for bs in (2, 4, 8):
        code, rep["probes"][bs] = run(bs, 1, f"smoke_b{bs}")
        print(bs, rep["probes"][bs], flush=True)
        if code != 0:
            break
    # headroom from RESERVED memory against what was free when we started (the display takes some; the driver's sysmem
    # fallback can turn a real OOM into a silent slowdown: 15% headroom keeps the choice clear of it, and the README says
    # to stop a run whose epochs take far longer than estimated)
    ok = [bs for bs, v in rep["probes"].items() if v["status"] == "done" and (v["peak_gpu_reserved_gb"] or 99) <= 0.85 * free0 / 2**30]
    if not ok:
        rep["verdict"] = "no batch size fitted with 15% headroom - send this report back"
    else:
        bs = max(ok)
        _c, three = run(bs, 3, f"smoke_b{bs}_3ep")
        rep["timing_3_epochs"] = three
        per_epoch = (three["seconds"] - rep["probes"][bs]["seconds"]) / 2 if three.get("status") == "done" and three.get("seconds") else None
        rep["recommended_batch"] = bs
        rep["smoke_images"] = {"train": n_smoke, "valid": n_smoke_valid}
        if per_epoch:
            # an epoch = training over the train split + validation (regular and EMA) over the valid split; scaled by
            # images, treating a validation image as costing what a training one does (it costs less: an upper estimate)
            per_image = per_epoch / (n_smoke + n_smoke_valid)
            rep["seconds_per_image_steady"] = round(per_image, 4)
        if per_epoch and n_full:
            rep["full_images"] = {"train": n_full, "valid": n_full_valid}
            rep["est_hours_per_epoch"] = round(per_image * (n_full + n_full_valid) / 3600, 2)
            rep["est_hours_30_epochs_3_seeds"] = round(per_image * (n_full + n_full_valid) * 90 / 3600, 1)
        rep["verdict"] = (f"batch {bs} fits with 15% headroom; est_hours_* are an upper estimate" if per_epoch else
                          f"the 3-epoch run at batch {bs} failed ({three.get('status')}) - send this report back")
    json.dump(rep, open(os.path.join(HERE, "smoke_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
