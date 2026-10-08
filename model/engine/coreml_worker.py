"""The Python 3.12 half of the CoreML detector. Not imported - it is exec'd.

WHY A SUBPROCESS AT ALL. RF-DETR on the Apple Neural Engine is faster than the
PyTorch MPS path (the measured ratios and how they moved:
detection-tracking-evidence (not published)#evidence-coreml), and the only way to reach the ANE is CoreML.
coremltools has no working native extension for this repo's Python 3.14: it
installs and then dies with "BlobWriter not loaded". It works on 3.12. So the
model runs in a 3.12 interpreter and tiles cross the process boundary.

NUMPY ONLY, NO TORCH, NO RFDETR, and that is what makes this shippable. This
file used to import torch for rfdetr's post_process (a sigmoid, a topk, a
cxcywh->xyxy and a scale - fifteen lines) and rfdetr for an 80-entry name
table. Both are inlined below. The runtime closure went from 1.1 GB to a clean
3.12 venv of 145 MB (coremltools 26 + sympy 72 + numpy 34 + protobuf/attrs/yaml
~13), which is what a desktop build can carry beside its 3.14 interpreter. The
inlining is not free-hand: _selfcheck compares both against rfdetr when rfdetr
is importable, and against frozen goldens when it is not.

THE PROTOCOL, stdin/stdout, binary in and one JSON line out:

    parent -> worker   uint32 n, then n * S * S * 3 bytes, uint8, BGR, C-order
    worker -> parent   {"seq": k, "tiles": [[[x1,y1,x2,y2,conf,cls], ...], ...]}

n == 0 means shut down. argv: model, compute units, conf, and optionally a JSON
{class name: min conf} filter (the parent's detectors._keep rule, applied here so
dropped boxes never cross the pipe). The handshake before any of that is one JSON line on
stdout carrying the model's input square, so a parent that expects 576 and gets
a 640 .mlpackage finds out at startup rather than through wrong boxes.

BGR, not RGB, and the worker flips the channels. The parent has the tiles as
OpenCV gives them; flipping there is a full extra copy of 16 MB per frame, and
here it is free - the normalize below has to walk every pixel anyway.

`seq` exists because the failure this whole exercise is exposed to is a worker
that quietly returns stale or empty boxes while the clock still reads fast. The
parent asserts it.

Run standalone for the self-check:  ~/.venvs/ftcoreml/bin/python coreml_worker.py --selfcheck
"""
import json
import struct
import sys
import time

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def _pre(buf: np.ndarray) -> np.ndarray:
    """(n,S,S,3) uint8 BGR -> (n,3,S,S) float32, exactly infer_transforms().

    infer_transforms is ToImage + ToDtype(scale) + Resize + Normalize. The
    resize is a no-op here (tiles arrive at the model's own square, resized by
    cv2 in the parent where the interpolation choice is already made and
    measured), so what is left is a transpose, a /255 and ImageNet stats.
    _selfcheck asserts that against the real transform rather than by reading.
    """
    x = np.ascontiguousarray(buf[..., ::-1].transpose(0, 3, 1, 2), np.float32)
    x /= 255.0
    x -= MEAN
    x /= STD
    return x


# rfdetr.assets.coco_classes.COCO_CLASSES, by id, "" for the gaps COCO leaves.
# Inlined so the worker does not import rfdetr (and through it torch) for a
# lookup table; _selfcheck diffs it against the real one when rfdetr is around.
NAMES = (
    ";person;bicycle;car;motorcycle;airplane;bus;train;truck;boat;traffic light;"
    "fire hydrant;;stop sign;parking meter;bench;bird;cat;dog;horse;sheep;cow;"
    "elephant;bear;zebra;giraffe;;backpack;umbrella;;;handbag;tie;suitcase;frisbee;"
    "skis;snowboard;sports ball;kite;baseball bat;baseball glove;skateboard;"
    "surfboard;tennis racket;bottle;;wine glass;cup;fork;knife;spoon;bowl;banana;"
    "apple;sandwich;orange;broccoli;carrot;hot dog;pizza;donut;cake;chair;couch;"
    "potted plant;bed;;dining table;;;toilet;;tv;laptop;mouse;remote;keyboard;"
    "cell phone;microwave;oven;toaster;sink;refrigerator;;book;clock;vase;scissors;"
    "teddy bear;hair drier;toothbrush"
).split(";")


def _name(i: int, names: list | None = None) -> str:
    """Class id -> name: the model's own names when it carries them (a fine-tuned export, where id 0 is "person"),
    else COCO's; an unknown id reads as the id so it is visible - and never kept as a player."""
    table = NAMES if names is None else names
    return table[i] if 0 <= i < len(table) and table[i] else str(int(i))


def _post(labels: np.ndarray, dets: np.ndarray, size: int, n: int = 300):
    """rfdetr.export.benchmark.post_process, in numpy. -> (scores, cls, xyxy).

    Sigmoid the (1,300,91) logits, take the global top-300 over queries x
    classes (a query can win twice, which is the point of the flatten), turn
    the matching (1,300,4) cxcywh boxes into xyxy and scale by the model's own
    square - the tiles are already at that square, so target_sizes is (S,S).

    Ties: torch.topk and argsort break exact float ties in different orders, so
    the two disagree on the ORDER of equal-scored entries and never on the set.
    _selfcheck asserts set equality, not row equality, because row equality is
    the assertion that would fail for a reason that does not matter.
    """
    p = np.exp(-np.logaddexp(0.0, -labels[0].astype(np.float32)))   # stable sigmoid
    flat = p.reshape(-1)
    k = min(n, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx], kind="stable")]
    q, cls = idx // labels.shape[2], idx % labels.shape[2]
    cx, cy, w, h = dets[0][q].astype(np.float32).T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * size
    return flat[idx], cls, boxes


class Runner:
    def __init__(self, model: str, units: str = "CPU_AND_NE"):
        import coremltools as ct

        self.m = ct.models.MLModel(model, compute_units=getattr(ct.ComputeUnit, units))
        spec = self.m.get_spec()
        self.inp = spec.description.input[0].name
        shape = list(spec.description.input[0].type.multiArrayType.shape)
        assert len(shape) == 4 and shape[0] == 1 and shape[2] == shape[3], shape
        self.size = int(shape[2])
        # coremltools renames the outputs. Map them by SHAPE: both are
        # (1,300,N) float and getting labels and dets the wrong way round is
        # silent. See model/validation/coreml_check.py, where this was first got right.
        # ...and DO NOT key on the class count. A COCO model emits (1,300,91); a
        # model fine-tuned on one class emits (1,300,2), because the head is
        # Linear(hidden, num_classes + 1). Hard-coding 91 turns "we trained our
        # own detector" into a KeyError at load, which is the least useful place
        # to find out. dets is the only (1,300,4); labels is the other one.
        by_shape = {tuple(o.type.multiArrayType.shape): o.name for o in spec.description.output}
        dets = by_shape[(1, 300, 4)]
        lab_shape = next(sh for sh in by_shape if sh != (1, 300, 4))
        self.out = {"labels": by_shape[lab_shape], "dets": dets}
        self.ncls = int(lab_shape[2])
        # A fine-tuned model's class 0 is "person", not COCO's "" - read through the COCO table every box would be
        # named "0" and the product would drop them all. model/validation/n46_coreml.py writes the names into the package;
        # a non-COCO head without them is refused here, at load, not discovered as an empty track file.
        meta = dict(spec.description.metadata.userDefined)
        self.names = json.loads(meta["class_names"]) if "class_names" in meta else None
        # rfdetr's head is num_classes + 1 wide; a names list that does not cover it would silently drop boxes
        if self.names is not None and not (isinstance(self.names, list) and len(self.names) in (self.ncls - 1, self.ncls)):
            raise SystemExit(f"{model}: class_names {self.names!r} does not fit a {self.ncls}-wide head")
        if self.names is None and self.ncls != 91:
            raise SystemExit(f"{model}: {self.ncls - 1}-class head with no class_names metadata - "
                             "export it with model/validation/n46_coreml.py")

    def run(self, buf: np.ndarray, conf: float, thr: dict | None = None) -> list[list]:
        """`thr` {class name: min conf}, optional: drop every other class here instead of
        piping it (the parent's detectors._keep applies the same rule to the same rounded
        conf, so its output is unchanged)."""
        x = _pre(buf)
        rows = []
        for i in range(len(x)):
            o = self.m.predict({self.inp: x[i:i + 1]})
            scores, cls, boxes = _post(o[self.out["labels"]], o[self.out["dets"]], self.size)
            keep = scores >= conf
            r = [[round(float(v), 2) for v in b] + [round(float(s), 4), _name(c, self.names)]
                 for b, s, c in zip(boxes[keep], scores[keep], cls[keep])]
            rows.append(r if thr is None else [x for x in r if x[4] >= thr.get(x[5], 2.0)])
        return rows


def serve(model: str, units: str, conf: float, thr: dict | None = None) -> None:
    r = Runner(model, units)
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    out.write((json.dumps({"ready": True, "size": r.size, "units": units}) + "\n").encode())
    out.flush()
    n_px = r.size * r.size * 3
    seq = 0
    while True:
        head = inp.read(4)
        if len(head) < 4:
            return                                  # parent closed the pipe
        n = struct.unpack("<I", head)[0]
        if n == 0:
            return
        raw = inp.read(n * n_px)
        if len(raw) < n * n_px:
            raise SystemExit(f"short read: wanted {n * n_px}, got {len(raw)}")
        buf = np.frombuffer(raw, np.uint8).reshape(n, r.size, r.size, 3)
        t0 = time.perf_counter()
        rows = r.run(buf, conf, thr)
        # Reported so the parent can subtract it and see the boundary's real
        # cost. Without it, "IPC overhead" is a number nobody can check.
        out.write((json.dumps({"seq": seq, "tiles": rows,
                               "ms": (time.perf_counter() - t0) * 1000}) + "\n").encode())
        out.flush()
        seq += 1


def _selfcheck(model: str) -> None:
    """The preprocessing and the decode, against what they claim to reproduce.

    Two tiers on purpose. The rfdetr comparisons are the real references, but
    they need torch, and the whole point of this file now is that the shipping
    interpreter has no torch - so a check that only runs where rfdetr is
    installed would never run where it matters. The goldens below are frozen
    FROM those comparisons and run everywhere.
    """
    rng = np.random.default_rng(0)
    S = 64

    # --- _pre, golden. One pixel, hand-checkable: BGR (10,20,30) is RGB
    # (30,20,10), /255, then ImageNet stats. Anchored to infer_transforms by
    # the rfdetr block below.
    px = np.array([[[[10, 20, 30]]]], np.uint8)
    got = _pre(px)[0, :, 0, 0]
    want = np.array([(30 / 255 - 0.485) / 0.229, (20 / 255 - 0.456) / 0.224,
                     (10 / 255 - 0.406) / 0.225], np.float32)
    assert np.abs(got - want).max() < 1e-5, (got, want)
    # ...and it must be able to fail: RGB in where BGR is expected is not the
    # same tensor. (A grey-ish random image would agree; a red one cannot.)
    red = np.zeros((1, S, S, 3), np.uint8)
    red[..., 2] = 255
    assert np.abs(_pre(red)[0] - _pre(red[..., ::-1].copy())[0]).max() > 1.0

    # --- _post, golden. One query screaming class 1 at a box that is trivially
    # readable back out: centre (0.5,0.25), size (0.2,0.5) -> xyxy (0.4,0,0.6,0.5)
    # of the square. Everything else is -20 logits, i.e. sigmoid ~2e-9.
    lab = np.full((1, 300, 91), -20.0, np.float32)
    lab[0, 7, 1] = 20.0
    det = np.tile(np.array([0.5, 0.25, 0.2, 0.5], np.float32), (1, 300, 1))
    sc, cl, bx = _post(lab, det, 576)
    assert len(sc) == len(cl) == len(bx) == 300, (len(sc), len(cl), len(bx))
    assert sc[0] > 0.999 and cl[0] == 1, (sc[0], cl[0])
    assert np.abs(bx[0] - np.array([0.4, 0.0, 0.6, 0.5]) * 576).max() < 1e-3, bx[0]
    assert (np.diff(sc) <= 0).all(), "scores must come back sorted descending"
    assert _name(1) == "person" and _name(37) == "sports ball", (_name(1), _name(37))
    assert _name(12) == "12", _name(12)      # a COCO gap reads as its id, not ""
    assert _name(0, ["person"]) == "person" and _name(1, ["person"]) == "1"     # a fine-tuned head: its own names
    assert _name(0) == "0"                   # ...which COCO's table would have lost
    # ...and it must be able to fail: the winning query must be the one with the
    # logit, not query 0. Move it and the answer has to move with it.
    lab2 = np.full((1, 300, 91), -20.0, np.float32)
    lab2[0, 7, 37] = 20.0
    det2 = det.copy()
    det2[0, 7] = [0.1, 0.1, 0.02, 0.02]
    assert _post(lab2, det2, 576)[1][0] == 37, _post(lab2, det2, 576)[1][0]
    assert abs(_post(lab2, det2, 576)[2][0][0] - 0.09 * 576) < 1e-3

    # --- the real references, where they are installed.
    try:
        import torch                                          # noqa: PLC0415
        from PIL import Image                                 # noqa: PLC0415
        from rfdetr.assets.coco_classes import COCO_CLASSES   # noqa: PLC0415
        from rfdetr.export.benchmark import infer_transforms, post_process  # noqa: PLC0415
    except ImportError as e:
        print(f"coreml_worker: rfdetr references skipped ({e}); goldens only")
    else:
        assert {i: n for i, n in enumerate(NAMES) if n} == COCO_CLASSES, "NAMES drifted"
        bgr = rng.integers(0, 256, (2, S, S, 3), dtype=np.uint8)
        tf = infer_transforms((S, S))
        for i in range(len(bgr)):
            ref, _ = tf(Image.fromarray(bgr[i][..., ::-1]), None)   # PIL wants RGB
            assert np.abs(ref.numpy() - _pre(bgr[i:i + 1])[0]).max() < 1e-5
        for _ in range(5):
            lab = (rng.standard_normal((1, 300, 91)) * 3).astype(np.float32)
            det = rng.random((1, 300, 4)).astype(np.float32)
            r = post_process({"labels": torch.from_numpy(lab), "dets": torch.from_numpy(det)},
                             torch.tensor([[576.0, 576.0]]))[0]
            sc, cl, bx = _post(lab, det, 576)
            # 1e-6, not ==: numpy's stable sigmoid and torch's differ by a
            # float32 ULP (8.9e-08 measured). Anything that is actually wrong
            # here is wrong by O(1), not by 1e-7.
            assert np.abs(np.sort(r["scores"].numpy()) - np.sort(sc)).max() < 1e-6
            # Set, not row: exact float ties (saturated sigmoids) permute inside
            # a tie group and both orders are a correct topk. The SET is the
            # contract, and a wrong decode cannot preserve it.
            mine = sorted(map(tuple, np.column_stack([cl, bx]).round(4)))
            theirs = sorted(map(tuple, np.column_stack(
                [r["labels"].numpy(), r["boxes"].numpy()]).round(4)))
            assert mine == theirs, [x for x in mine if x not in theirs][:3]

    if model:
        r = Runner(model, "CPU_AND_NE")
        rows = r.run(np.zeros((1, r.size, r.size, 3), np.uint8), 0.5)
        assert rows == [[]], rows          # a black tile holds no players
        rows = r.run(np.zeros((1, r.size, r.size, 3), np.uint8), 0.0)
        assert len(rows[0]) == 300, len(rows[0])   # ...but the model did run
        only = r.run(np.zeros((1, r.size, r.size, 3), np.uint8), 0.0, {"person": 0.0})
        assert only == [[x for x in rows[0] if x[5] == "person"]], "the class filter must be exactly _keep's"
        print(f"coreml_worker self-check OK (model {r.size}px, 300 queries decoded)")
    else:
        print("coreml_worker self-check OK (preprocessing + decode, no model given)")


if __name__ == "__main__":
    a = sys.argv[1:]
    # No args self-checks, like every other module here - `python
    # model/engine/coreml_worker.py` used to die on an IndexError reaching for the
    # model path. Bare, it checks the preprocessing only, which is the half that
    # runs on any interpreter; pass a .mlpackage on the 3.12 one to also run the
    # model.
    if not a or a[0] == "--selfcheck":
        _selfcheck(a[1] if len(a) > 1 else "")
    else:
        serve(a[0], a[1] if len(a) > 1 else "CPU_AND_NE", float(a[2]) if len(a) > 2 else 0.5,
              json.loads(a[3]) if len(a) > 3 else None)
