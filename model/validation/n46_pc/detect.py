"""N46 detection on held-out clips, run on the training PC (CUDA) or the Mac (MPS). The product's own tiled detector
(model/engine/detectors.py, copied beside this file by the pack step) with given weights: tile 1024, overlap as shipped,
576 input, every 2nd frame. It saves every TILE's boxes at conf >= 0.2 BEFORE the merge (TiledDetector raw mode); the Mac
applies the product's conf 0.5, its merge and its pitch prior on import (model/validation/n46_finetune.py --import). Filtering
at 0.5 before the merge is exactly what the product does, and a different threshold can be replayed later without a GPU.

    python detect.py --weights stock --seqs 118576              # clips/118576.mp4 -> detections/118576.stock.raw.json
    python detect.py --weights runs/seed0 --seqs 118576         # its checkpoint_best_total.pth, tagged ft0
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                                     # the packed copy of model/engine/detectors.py
sys.path.append(os.path.join(HERE, "..", "..", "engine"))                    # or the repo's own, when run on the Mac
STRIDE, CONF = 2, 0.2          # every 2nd frame, as the product; saved floor - the Mac applies the product's 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="'stock', a run folder such as runs/seed0, or a .pth checkpoint")
    ap.add_argument("--tag", default="", help="default: stock, or ft<N> for runs/seed<N>")
    ap.add_argument("--clips", default=os.path.join(HERE, "clips"))
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "detections"))
    ap.add_argument("--max-frames", type=int, default=0, help="stop early (a check, not a result)")
    a = ap.parse_args()
    if os.path.isdir(a.weights):
        run = os.path.basename(os.path.normpath(a.weights))
        a.tag = a.tag or (f"ft{run[4:]}" if run.startswith("seed") else run)
        a.weights = os.path.join(a.weights, "checkpoint_best_total.pth")
    a.tag = a.tag or ("stock" if a.weights == "stock" else sys.exit("--tag is needed for a bare .pth"))
    import detectors as D
    kw = {"conf": CONF, "variant": "medium"}
    sha = None
    if a.weights != "stock":
        kw["weights"] = a.weights
        sha = hashlib.sha256(open(a.weights, "rb").read()).hexdigest()
    det = D.TiledDetector(D.RFDETRDetector(**kw), tile=1024, overlap=D.OVERLAP, batch=8, raw=True)   # the product's _rfdetr, unmerged
    os.makedirs(a.out, exist_ok=True)
    for seq in a.seqs:
        path = os.path.join(a.clips, f"{seq}.mp4")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"cannot open {path} (missing, or a folder name OpenCV cannot read - keep the kit at C:\\n46_pc)")
        frames, f, t0 = {}, 0, time.time()
        while True:
            ok = cap.grab()
            if not ok:
                break
            if f % STRIDE == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                frames[str(f)] = [{"xyxy": [round(float(v), 2) for v in d["xyxy"]], "conf": round(float(d["conf"]), 4), "cls": d.get("cls"),
                                   "_clip": bool(d["_clip"]), "_tile": int(d["_tile"])} for d in det.detect(img)]
                if f % 1000 == 0:
                    print(f"  {seq} frame {f}  {f / max(time.time() - t0, 1e-9):.1f} source fps", flush=True)
            f += 1
            if a.max_frames and f >= a.max_frames:
                break
        if not frames:
            sys.exit(f"no frames read from {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = os.path.join(a.out, f"{seq}.{a.tag}.raw.json")
        json.dump({"seq": seq, "tag": a.tag, "weights": a.weights, "weights_sha256": sha, "conf": CONF, "stride": STRIDE,
                   "tile": 1024, "overlap": D.OVERLAP, "merged": False, "prior": "NOT APPLIED", "frames_read": f, "frame_count": total, "max_frames": a.max_frames, "seconds": round(time.time() - t0, 1), "frames": frames},
                  open(out, "w"))
        print("->", out, flush=True)


if __name__ == "__main__":
    main()
