"""Render what the team classifier actually sees at a duel. Eyeball the pixels.

Every number in `st2_kit_at_duel.py` and `st2_duelcrop.py` rests on a ~12x12 px
torso crop of which 70.6% is the other player. That is easy to assert and hard
to believe, so this draws it: the frame in context, both players' boxes, the
torso regions those numbers are computed from, and the same crops with the other
player's box removed - at 12x magnification, next to what the classifier decided.

    .venv/bin/python model/validation/st2_duelview.py --seq 128057 --n 12
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, "model/validation")
import st2_failtax as F  # noqa: E402
import st2_kit_at_duel as K  # noqa: E402
import st2_team as T  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

CLIPS = _ST2 + "/mot/clips"
OUT = "renders"
Z = 12                      # magnification for the torso crops
CTX = 110                   # context box half-width in source pixels
KIT = ((60, 200, 255), (255, 190, 60))     # BGR, kit 0 / kit 1


def torso_rect(box):
    x, y, w, h = box
    return (int(x + 0.25 * w), int(y + 0.15 * h),
            int(x + 0.75 * w), int(y + 0.45 * h))


def tile(limg, bgr, box, other, cen, lab_true):
    """(raw crop, masked crop, predicted kit, n_kept) at magnification Z."""
    x0, y0, x1, y1 = torso_rect(box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, bgr.shape[1]), min(y1, bgr.shape[0])
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    raw = bgr[y0:y1, x0:x1]
    L = limg[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    col = np.median(L, axis=0)
    pred = min(cen, key=lambda k: np.linalg.norm(col - cen[k]))
    ox, oy, ow, oh = other
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs >= ox) & (xs < ox + ow) & (ys >= oy) & (ys < oy + oh)
    masked = raw.copy()
    masked[inside] = (0, 0, 255)            # red = the other player's box
    kept = int((~inside).sum())
    up = lambda a: cv2.resize(a, None, fx=Z, fy=Z,
                              interpolation=cv2.INTER_NEAREST)
    return up(raw), up(masked), pred, kept, inside.size


def render(seq, n=12, variant=".c03s", seed=2):
    gt = T.load_gt(seq)
    lab, cen = K.centres(seq, gt)
    out = F.analyse(seq, variant.lstrip("."), 0.5)
    ev = [e for e in (out["contam"] if isinstance(out, dict) else out[0])
          if e["gap"] == 0 and e["pair_iou"] > 0
          and lab.get(e["gid_from"]) is not None
          and lab.get(e["gid_to"]) is not None
          and lab[e["gid_from"]] != lab[e["gid_to"]]
          and e["gid_from"] in gt.get(e["f"], {}) and e["gid_to"] in gt[e["f"]]]
    random.Random(seed).shuffle(ev)
    ev = ev[:n]
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    rows = []
    for e in ev:
        f = e["f"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        limg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        bf, bt = gt[f][e["gid_from"]], gt[f][e["gid_to"]]
        A = tile(limg, img, bf, bt, cen, lab[e["gid_from"]])
        B = tile(limg, img, bt, bf, cen, lab[e["gid_to"]])
        if A is None or B is None:
            continue
        # context: both boxes, drawn in their TRUE kit colour, torso dashed
        cx = int(bf[0] + bf[2] / 2), int(bf[1] + bf[3] / 2)
        x0, y0 = max(cx[0] - CTX, 0), max(cx[1] - CTX, 0)
        ctx = img[y0:y0 + 2 * CTX, x0:x0 + 2 * CTX].copy()
        for box, gid in ((bf, e["gid_from"]), (bt, e["gid_to"])):
            c = KIT[lab[gid]]
            cv2.rectangle(ctx, (int(box[0]) - x0, int(box[1]) - y0),
                          (int(box[0] + box[2]) - x0, int(box[1] + box[3]) - y0),
                          c, 1)
            tx0, ty0, tx1, ty1 = torso_rect(box)
            cv2.rectangle(ctx, (tx0 - x0, ty0 - y0), (tx1 - x0, ty1 - y0),
                          (255, 255, 255), 1)
        ctx = cv2.resize(ctx, (330, 330), interpolation=cv2.INTER_NEAREST)
        h = max(ctx.shape[0], A[0].shape[0], B[0].shape[0])
        pad = lambda a: cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 8,
                                           cv2.BORDER_CONSTANT, value=(30, 30, 30))
        row = np.hstack([pad(ctx), pad(A[0]), pad(A[1]), pad(B[0]), pad(B[1])])
        row = cv2.copyMakeBorder(row, 34, 6, 6, 6, cv2.BORDER_CONSTANT,
                                 value=(30, 30, 30))
        okA, okB = A[2] == lab[e["gid_from"]], B[2] == lab[e["gid_to"]]
        verdict = ("ABSTAIN (both read same kit)" if A[2] == B[2]
                   else ("RESOLVED" if okA else "WRONG"))
        cv2.putText(row, f"{seq} f={f} IoU={e['pair_iou']:.2f}  "
                         f"kept {A[3]}/{A[4]} & {B[3]}/{B[4]} px  -> {verdict}",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (90, 255, 90) if verdict == "RESOLVED" else
                    (90, 200, 255) if verdict.startswith("ABSTAIN") else
                    (90, 90, 255), 1, cv2.LINE_AA)
        rows.append(row)
    cap.release()
    if not rows:
        return None
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(30, 30, 30))
            for r in rows]
    sheet = np.vstack(rows)
    head = np.full((44, w, 3), 30, np.uint8)
    cv2.putText(head, "context (white = torso region the colour is read from)"
                      " | A raw | A masked | B raw | B masked   "
                      "red = the OTHER player's box",
                (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                cv2.LINE_AA)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/duel_{seq}.png"
    cv2.imwrite(path, np.vstack([head, sheet]))
    print(f"  {path}  ({len(rows)} duels)")
    return path


def demo():
    """Pin the crop geometry. A wrong rect renders convincing, wrong evidence."""
    box = (100.0, 200.0, 40.0, 100.0)          # x, y, w, h
    x0, y0, x1, y1 = torso_rect(box)
    assert (x0, y0, x1, y1) == (110, 215, 130, 245), (x0, y0, x1, y1)
    assert x0 > box[0] and x1 < box[0] + box[2], "torso must sit inside the box"
    assert y0 > box[1] and y1 < box[1] + box[3], "torso must sit inside the box"
    # The mask must mark the OTHER player's pixels and nothing else.
    img = np.zeros((400, 400, 3), np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cen = {0: np.zeros(3), 1: np.full(3, 255.0)}
    far = (300.0, 200.0, 40.0, 100.0)          # nowhere near `box`
    r = tile(lab, img, box, far, cen, 0)
    assert r is not None and r[3] == r[4], "no overlap -> every pixel kept"
    near = (100.0, 200.0, 40.0, 100.0)         # exactly on top of it
    r2 = tile(lab, img, box, near, cen, 0)
    assert r2[3] == 0, f"full overlap -> no pixels kept, got {r2[3]}"
    print(f"ok  (torso rect {x0},{y0}-{x1},{y1} inside the box; "
          f"mask keeps {r[3]}/{r[4]} with no overlap and {r2[3]} with full)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", default=["128057"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        for s in a.seq:
            render(s, a.n)
