"""N46: export a fine-tuned RF-DETR checkpoint to the product's CoreML format, and check it draws what PyTorch draws.

WHY
    The product runs a CoreML .mlpackage on the Neural Engine (model/engine/detectors.CoreMLDetector, a Python 3.12 worker),
    and N46 judges the fine-tune through PyTorch. A fine-tune that passes N46 ships only if its CoreML export reproduces
    its PyTorch boxes - the repo had no export script (the shipped package was made by hand) and the worker read every
    class through COCO's table, where a fine-tuned model's class 0 ("person") is "" - so every box would have been named
    "0" and dropped. The export below writes the checkpoint's class names into the package; the worker reads them and
    refuses a non-COCO head without them.

THE CHECK, fixed before any run
    frames      118576 (VALID - never TEST), every 120th frame from 0: 50 frames across the clip
    arms        PyTorch  get_detector("rfdetr", weights=CKPT, conf=0.5)  - the path N46 judges
                CoreML   get_detector("coreml", weights=PKG, conf=0.5)   - the path that would ship
                both tiled and merged as the product runs them; no scene prior (identical for both, so it cannot differ)
    metric      greedy one-to-one matching at IoU >= 0.5, highest confidence first; agreement recall (PyTorch boxes
                CoreML also drew) and precision (CoreML boxes PyTorch also drew), pooled over frames
    null        the same CoreML boxes against the PyTorch boxes of the NEXT sampled frame (4.8 s later): if that reads
                near the real number, the metric measures box density, not agreement
    rule        PASS iff recall >= 0.98, precision >= 0.98, null recall < 0.3, and every CoreML box is named a player
                class. The reference is the shipped stock export on its own harness: 0.986 / 0.995 (detectors.py
                CoreMLDetector docstring, 118575 band frames) - run here as the CONTROL on the same 50 frames first.

    ~/.venvs/ftcoreml/bin/python model/validation/n46_coreml.py export CKPT OUT.mlpackage     # the 3.12 side (coremltools)
    .venv/bin/python model/validation/n46_coreml.py check stock model/models/rfdetr-medium-576.mlpackage   # the control
    .venv/bin/python model/validation/n46_coreml.py check CKPT OUT.mlpackage
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
CLIP = os.path.expanduser("~/Downloads/st2/mot/clips/118576.mp4")
EVERY, N = 120, 50
REC_MIN, PREC_MIN, NULL_MAX = 0.98, 0.98, 0.3


def export(ckpt, out):
    """-> OUT.mlpackage at 576, batch 1, fp16 (what the shipped package is), with class_names and the weights' sha256
    in its metadata."""
    import torch  # noqa: PLC0415
    import coremltools as ct  # noqa: PLC0415
    from rfdetr import RFDETRMedium  # noqa: PLC0415
    # weights_only=False: our own training output, and rfdetr stores its args as plain objects; read where the product's
    # model/engine/detectors._checkpoint_classes reads them, so the two paths cannot disagree about the names
    names = (torch.load(ckpt, map_location="cpu", weights_only=False).get("args") or {}).get("class_names")
    assert names, f"{ckpt} carries no class_names - not a fine-tuned rfdetr checkpoint"
    tmp = out + ".export"
    shutil.rmtree(tmp, ignore_errors=True)
    path = RFDETRMedium(pretrain_weights=ckpt, num_classes=len(names)).export(
        output_dir=tmp, format="coreml", coreml_precision="float16", shape=(576, 576), batch_size=1)
    m = ct.models.MLModel(str(path))
    m.user_defined_metadata["class_names"] = json.dumps(list(names))
    m.user_defined_metadata["weights_sha256"] = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    shutil.rmtree(out, ignore_errors=True)
    m.save(out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("->", out, "classes", names)


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def matched(ref, got):
    """Greedy one-to-one at IoU >= 0.5, highest confidence first -> number of pairs."""
    used, n = set(), 0
    for g in sorted(got, key=lambda d: -d["conf"]):
        best = max(((i, _iou(g["xyxy"], r["xyxy"])) for i, r in enumerate(ref) if i not in used), key=lambda t: t[1], default=(None, 0.0))
        if best[1] >= 0.5:
            used.add(best[0])
            n += 1
    return n


def agreement(pt, cm):
    """Per-frame box lists -> (recall, precision, null recall). Null = CoreML frame i against PyTorch frame i+1."""
    m = sum(matched(p, c) for p, c in zip(pt, cm))
    null = sum(matched(p, c) for p, c in zip(pt[1:], cm[:-1]))
    return m / max(1, sum(map(len, pt))), m / max(1, sum(map(len, cm))), null / max(1, sum(map(len, pt[1:])))


def check(ckpt, pkg):
    import cv2  # noqa: PLC0415
    import detectors as D  # noqa: PLC0415
    kw = {} if ckpt == "stock" else {"weights": ckpt}
    pt_det = D.get_detector("rfdetr", conf=0.5, **kw)
    cm_det = D.get_detector("coreml", conf=0.5, weights=pkg)
    cap, f, pt, cm = cv2.VideoCapture(CLIP), 0, [], []
    while len(pt) < N and cap.grab():
        if f % EVERY == 0:
            _ok, img = cap.retrieve()
            pt.append(pt_det.detect(img))
            cm.append(cm_det.detect(img))
        f += 1
    rec, prec, null = agreement(pt, cm)
    named = all(d["cls"] in D.PLAYER_CLASSES for fr in cm for d in fr)
    ok = rec >= REC_MIN and prec >= PREC_MIN and null < NULL_MAX and named
    print(f"{ckpt} vs {os.path.basename(pkg)}: frames {len(pt)}, boxes PyTorch {sum(map(len, pt))} CoreML {sum(map(len, cm))}, "
          f"recall {rec:.3f}, precision {prec:.3f}, null {null:.3f}, all named players {named} -> {'PASS' if ok else 'FAIL'}")
    return ok


def demo():
    box = lambda x, c=0.9: {"xyxy": (x, 0.0, x + 10.0, 20.0), "conf": c}  # noqa: E731
    a = [[box(0), box(50)], [box(100)]]
    assert agreement(a, a)[:2] == (1.0, 1.0)
    assert agreement(a, [[box(0)], [box(100)]])[:2] == (2 / 3, 1.0)           # one PyTorch box missed
    assert matched([box(0)], [box(0), box(1)]) == 1                           # one-to-one: a duplicate is not a second match
    assert agreement(a, a)[2] == 0.0                                          # the null reads nothing where frames differ
    print("n46_coreml demo ok: agreement counts one-to-one matches, and the null can read zero")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["export"]:
        export(a[1], a[2])
    elif a[:1] == ["check"]:
        sys.exit(0 if check(a[1], a[2]) else 1)
    else:
        demo()
