"""N46: raw detect.py (conf 0.2, unmerged) -> filter 0.5 -> _merge(0.55) must equal the product's get_detector('rfdetr', conf=0.5).

    .venv/bin/python model/validation/n46_equiv.py [stock | path/to/checkpoint.pth]    # 118576, frames 0-99, stride 2
"""
import json, subprocess, sys, cv2, numpy as np
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
sys.path.insert(0, "model/engine"); import detectors as D
S, W = "model/validation/out/n46_proof", (sys.argv[1:] or ["stock"])[0]   # out dir, weights ('stock' or .pth)
N = 100
subprocess.run([".venv/bin/python", "model/validation/n46_pc/detect.py", "--weights", W, "--tag", "eq", "--clips",
                _ST2 + "/mot/clips", "--seqs", "118576", "--out", S, "--max-frames", str(N)], check=True)
raw = json.load(open(f"{S}/118576.eq.raw.json"))["frames"]
kw = {"conf": 0.5, "variant": "medium"} | ({} if W == "stock" else {"weights": W})
prod = D.get_detector("rfdetr", **kw)
cap = cv2.VideoCapture(_ST2 + "/mot/clips/118576.mp4")
worst, nb, mism = 0.0, 0, 0
for f in range(N):
    ok, img = cap.read()
    if f % 2: continue
    a = sorted(prod.detect(img), key=lambda d: d["xyxy"])
    b = sorted(D._merge([d | {"xyxy": tuple(d["xyxy"])} for d in raw[str(f)] if d["conf"] >= 0.5], 0.55), key=lambda d: d["xyxy"])
    if len(a) != len(b): mism += 1; print("frame", f, len(a), len(b)); continue
    nb += len(a)
    if a: worst = max(worst, float(np.abs(np.array([d["xyxy"] for d in a]) - np.array([d["xyxy"] for d in b])).max()))
print(f"frames {N//2}  boxes {nb}  count mismatches {mism}  worst coord diff {worst:.3f} px")
