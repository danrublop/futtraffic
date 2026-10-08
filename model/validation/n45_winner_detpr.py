"""The STC2025 winner's raw detector output (model/validation/out/stc2025/winner/<seq>_yolo, from github.com/ron941/GTATrack-STC2025) against ST2 GT - research-stc2025-winners (not published) §0.
Per-frame greedy IoU>=0.5 match of a MOT det file against ST2 GT: recall, precision, boxes/frame.

    .venv/bin/python model/validation/n45_winner_detpr.py
Also splits GT recall by depth third of the foot row (same idea as model/validation/n40_occlusion_recall.py,
occlusion = another GT box covers >=35% of it), so it reads against N40's table."""
import sys, numpy as np
from collections import defaultdict
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
GT = _ST2 + "/mot"
def load(fn, one_based):
    d = defaultdict(list)
    for line in open(fn):
        p = line.replace(",", " ").split()
        if len(p) < 6: continue
        f = int(p[0]) - (1 if one_based else 0)
        d[f].append([float(p[2]), float(p[3]), float(p[2]) + float(p[4]), float(p[3]) + float(p[5])])
    return {k: np.array(v) for k, v in d.items()}
def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9), inter / (aa[:, None] + 1e-9)
import os
WIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "stc2025", "winner")
for seq, det_fn, H in [("128057", os.path.join(WIN, "128057_yolo"), 1080), ("132831", os.path.join(WIN, "132831_yolo"), 1504)]:
    gt = load(f"{GT}/{seq}.txt", False); det = load(det_fn, True)
    tp = fp = fn = 0; cls = defaultdict(lambda: [0, 0])
    for f, g in gt.items():
        d = det.get(f, np.zeros((0, 4)))
        # occlusion + depth class per GT box
        _, cov = iou(g, g); np.fill_diagonal(cov, 0); occ = cov.max(1) >= 0.35
        third = np.minimum((g[:, 3] / H * 3).astype(int), 2)   # 0 far (top) .. 2 near (bottom)
        matched = np.zeros(len(g), bool)
        if len(d):
            m, _ = iou(d, g); m = m.copy()
            while True:
                i, j = np.unravel_index(m.argmax(), m.shape)
                if m[i, j] < 0.5: break
                matched[j] = True; m[i, :] = 0; m[:, j] = 0
        tp += matched.sum(); fn += (~matched).sum(); fp += len(d) - matched.sum()
        for o, t, mm in zip(occ, third, matched):
            k = ("occ" if o else "iso", ["far", "mid", "near"][t]); cls[k][0] += mm; cls[k][1] += 1
    print(f"{seq}: recall {tp/(tp+fn):.4f} precision {tp/(tp+fp):.4f} det/frame {sum(len(v) for v in det.values())/len(gt):.2f} FP/frame {fp/len(gt):.2f}")
    print("   " + "  ".join(f"{a}-{b} {cls[(a,b)][0]/max(cls[(a,b)][1],1):.3f} (n={cls[(a,b)][1]})" for a in ("occ", "iso") for b in ("far", "mid", "near")))
