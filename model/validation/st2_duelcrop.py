"""Duel-aware crops: does removing the other player's pixels rescue the label?

ANSWER: NO, and the reason is arithmetic. `@M` Six variants over 974 cross-team
duels; not one beats the plain baseline (41.6% resolved / 54.5% abstain), and
masking trades abstentions for outright errors roughly one for one.

`@M` Measured on 1,148 duel crops: the median torso crop is **149 px** (~12x12),
the other player's box covers **70.6%** of it, and masking leaves fewer than six
pixels in **38.5%** of crops. There is not enough of the occluded player visible
to classify, so no crop geometry rescues it and a perfect segmentation mask
would not either.

Kept as a runnable negative result: this is the experiment that closed the
single-frame route and sent the work to the temporal one. See
`model/validation/st2_kit_at_duel.py` for the numbers it is measured against.
"""
import sys, random
from collections import Counter, defaultdict
import cv2, numpy as np
sys.path.insert(0, "model/validation")
import st2_failtax as F, st2_team as T
import st2_kit_at_duel as K
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

CLIPS = _ST2 + "/mot/clips"

def torso_box(box):
    x, y, w, h = box
    return (int(x+0.25*w), int(x+0.75*w), int(y+0.15*h), int(y+0.45*h))

def narrow_box(box):
    x, y, w, h = box
    return (int(x+0.35*w), int(x+0.65*w), int(y+0.12*h), int(y+0.42*h))

def crop_px(limg, bx, other=None, mask_other=False):
    """Pixels of the torso region, optionally with the other player's box removed."""
    x0, x1, y0, y1 = bx
    x0, y0 = max(x0, 0), max(y0, 0)
    if x1 <= x0+1 or y1 <= y0+1: return None
    c = limg[y0:y1, x0:x1]
    if c.size < 12: return None
    P = c.reshape(-1, 3).astype(float)
    if mask_other and other is not None:
        ox, oy, ow, oh = other
        ys, xs = np.mgrid[y0:y1, x0:x1]
        inside = (xs >= ox) & (xs < ox+ow) & (ys >= oy) & (ys < oy+oh)
        keep = ~inside.reshape(-1)
        if keep.sum() >= 6:
            P = P[keep]
    return P

def label_from(P, cen, mode):
    if P is None or len(P) == 0: return None
    if mode == "median":
        col = np.median(P, axis=0)
        return min(cen, key=lambda k: np.linalg.norm(col - cen[k]))
    if mode == "vote":                       # per-pixel nearest centre, majority
        d = np.stack([np.linalg.norm(P - cen[k], axis=1) for k in sorted(cen)])
        v = np.bincount(d.argmin(0), minlength=len(cen))
        return int(v.argmax())
    raise ValueError(mode)

VARIANTS = (("baseline      ", torso_box, False, "median"),
            ("mask-other    ", torso_box, True,  "median"),
            ("narrow        ", narrow_box, False, "median"),
            ("narrow+mask   ", narrow_box, True,  "median"),
            ("pixel-vote    ", torso_box, False, "vote"),
            ("mask+vote     ", torso_box, True,  "vote"))

def run(seq, n_per=200):
    gt = T.load_gt(seq)
    lab, cen = K.centres(seq, gt)
    out = F.analyse(seq, "c03s", 0.5)
    ev = [e for e in (out["contam"] if isinstance(out, dict) else out[0])
          if e["gap"] == 0 and e["pair_iou"] > 0
          and lab.get(e["gid_from"]) is not None and lab.get(e["gid_to"]) is not None
          and lab[e["gid_from"]] != lab[e["gid_to"]]
          and e["gid_from"] in gt.get(e["f"], {}) and e["gid_to"] in gt[e["f"]]]
    random.Random(2).shuffle(ev); ev = ev[:n_per]
    by_f = defaultdict(list)
    for e in ev: by_f[e["f"]].append(e)
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    R = {v[0]: Counter() for v in VARIANTS}
    for f in sorted(by_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        got, img = cap.read()
        if not got: continue
        limg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for e in by_f[f]:
            bf, bt = gt[f][e["gid_from"]], gt[f][e["gid_to"]]
            own = lab[e["gid_from"]]
            for name, bxfn, mask, mode in VARIANTS:
                pf = label_from(crop_px(limg, bxfn(bf), bt, mask), cen, mode)
                pt = label_from(crop_px(limg, bxfn(bt), bf, mask), cen, mode)
                if pf is None or pt is None: continue
                R[name]["n"] += 1
                if pf == pt: R[name]["abstain"] += 1
                elif pf == own: R[name]["ok"] += 1
                else: R[name]["wrong"] += 1
    cap.release()
    return R

def demo():
    """The masked reader must drop the other player's pixels and only those."""
    cen = {0: np.zeros(3), 1: np.full(3, 255.0)}
    limg = np.zeros((100, 100, 3), np.uint8)
    limg[20:60, 20:60] = 255                    # a bright block
    box = (20.0, 10.0, 40.0, 80.0)
    assert label_from(crop_px(limg, torso_box(box)), cen, "median") in (0, 1)
    away = (500.0, 500.0, 10.0, 10.0)
    a = crop_px(limg, torso_box(box), away, mask_other=True)
    b = crop_px(limg, torso_box(box), away, mask_other=False)
    assert a is not None and len(a) == len(b), "a distant box must mask nothing"
    on_top = (0.0, 0.0, 100.0, 100.0)
    c = crop_px(limg, torso_box(box), on_top, mask_other=True)
    assert len(c) == len(b), "full cover falls back rather than returning empty"
    print(f"ok  (crop {len(b)} px; distant box masks 0; full cover falls back)")


if "--demo" in sys.argv:
    demo(); sys.exit(0)

tot = {v[0]: Counter() for v in VARIANTS}
for s in ("128057", "132831", "118575", "118576", "128058"):
    for k, v in run(s).items(): tot[k].update(v)
print(f"\n{'variant':>16s} {'n':>6s} {'resolved':>10s} {'wrong':>8s} {'abstain':>9s}")
for name, *_ in VARIANTS:
    r = tot[name]; n = max(r["n"], 1)
    print(f"{name:>16s} {r['n']:6d} {100*r['ok']/n:9.1f}% {100*r['wrong']/n:7.1f}% "
          f"{100*r['abstain']/n:8.1f}%")
