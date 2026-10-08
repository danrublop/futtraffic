"""Does RF-DETR under CoreML on the ANE detect the same players as PyTorch/MPS?

The speed case is already made (detectors.py: 25.6 ms/tile against 68.5, 2.4x).
This file is the QUALITY case, and it exists because the first attempt at one
could not fail: it compared a single tile that held no confident players, so
"14 of 15 boxes match" said the model computes the same thing and nothing at all
about detection.

WHAT IT MEASURES, on real frames spread across a half:

  1. Agreement with the SHIPPING detections. data/jobs/118575_full_1st/
     dets.cache.json holds every PyTorch/MPS box for all 71,850 frames - the
     reference, at the production config (medium, default 576, 16 tiles of 1024
     on the 11878x870 band, conf 0.5, scene prior applied). Greedy IoU matching,
     reported per depth band.
  2. Absolute recall against GROUND TRUTH, both paths, through detsweep.score -
     so a disagreement can be read as "CoreML lost a player" or "CoreML found
     one PyTorch missed" rather than just "they differ".
  3. A NULL: the same CoreML boxes scored against a different frame's PyTorch
     boxes. If that reads anywhere near the real number, the metric is measuring
     box density and not agreement, and nothing else here is worth reading.

The cache is post-prior (track.py filters BEFORE it writes), so the CoreML boxes
get the same make_prior gate before anything is compared. Skipping that reads as
a CoreML precision collapse that is really an unfiltered crowd.

THREE PHASES, because coremltools has no working native extension on this repo's
Python 3.14 ("BlobWriter not loaded") and works on 3.12:

    .venv/bin/python model/validation/coreml_check.py dump [n]     # opencv, video
    ~/.venvs/ftcoreml/bin/python model/validation/coreml_check.py infer [units]
    .venv/bin/python model/validation/coreml_check.py score

Tiles cross the gap as one uint8 .npy (RGB, 576x576, production tiling already
applied), which is why the 3.12 side needs no opencv.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

WORK = os.environ.get("CLAUDE_JOB_DIR", "/tmp") + "/tmp/coremlcheck"
MODEL = os.environ.get("CLAUDE_JOB_DIR", "/tmp") + "/tmp/coreml/output/rfdetr-medium.mlpackage"
JOB = "data/jobs/118575_full_1st"
VIDEO = "data/footage/118575_panorama_1st_half.mp4"
SIZE = 576          # RF-DETR medium's default input square - what ships
TILE = 1024
KEEP_CONF = 0.10    # dumped low so `score` can sweep; the product runs 0.5


def _grid(w: int, h: int):
    from detectors import TiledDetector

    class _S:
        resolution = SIZE
    return TiledDetector(_S(), tile=TILE).grid(w, h)


# --------------------------------------------------------------------- dump
def dump(n_frames: int = 112, every: int = 600, first: int = 3000) -> None:
    """Production tiling for N frames -> one uint8 (N*16, 576, 576, 3) RGB npy."""
    import cv2

    import calibrators
    import detsweep

    os.makedirs(WORK, exist_ok=True)
    cal = calibrators.load(f"{JOB}/calibration.json")
    gt = detsweep.load_gt()
    idx = [f for f in range(first, first + n_frames * every, every) if f in gt]
    h, w = cal._mapx.shape[:2]
    tiles = _grid(w, h)
    print(f"{len(idx)} frames of {w}x{h}, {len(tiles)} tiles each, "
          f"frames {idx[0]}..{idx[-1]}", flush=True)

    arr = np.lib.format.open_memmap(f"{WORK}/tiles.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(idx) * len(tiles), SIZE, SIZE, 3))
    cap = cv2.VideoCapture(VIDEO)
    try:
        for k, f in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, raw = cap.read()
            if not ok:
                raise SystemExit(f"could not read frame {f}")
            band = cal.undistort_frame(raw)
            for t, (x, y, tw, th) in enumerate(tiles):
                # Same interpolation rule as TiledDetector.detect - INTER_AREA
                # is what keeps a 3-px-wide far player from aliasing away, and
                # swapping it would be measuring the resize, not the model.
                crop = cv2.resize(band[y:y + th, x:x + tw], (SIZE, SIZE),
                                  interpolation=cv2.INTER_AREA if tw >= SIZE else cv2.INTER_LINEAR)
                arr[k * len(tiles) + t] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if k % 20 == 0:
                print(f"  frame {f} ({k + 1}/{len(idx)})", flush=True)
    finally:
        cap.release()
    arr.flush()
    json.dump({"frames": idx, "tiles": tiles, "wh": [w, h], "size": SIZE},
              open(f"{WORK}/meta.json", "w"))
    print(f"wrote {WORK}/tiles.npy {arr.shape} and meta.json", flush=True)


# -------------------------------------------------------------------- infer
def infer(units: str = "CPU_AND_NE", model: str = MODEL, out: str | None = None) -> None:
    """Every dumped tile through CoreML. Writes per-tile boxes in 576-space."""
    import time

    import coremltools as ct
    import torch
    from PIL import Image
    from rfdetr.assets.coco_classes import COCO_CLASSES
    from rfdetr.export.benchmark import infer_transforms, post_process

    meta = json.load(open(f"{WORK}/meta.json"))
    arr = np.load(f"{WORK}/tiles.npy", mmap_mode="r")
    person = [k for k, v in COCO_CLASSES.items() if v == "person"]
    assert len(person) == 1, COCO_CLASSES
    person = person[0]

    m = ct.models.MLModel(model, compute_units=getattr(ct.ComputeUnit, units))
    spec = m.get_spec()
    in_name = spec.description.input[0].name
    # coremltools renames the outputs, so map them by SHAPE. Getting labels and
    # dets the wrong way round is silent: both are (1,300,N) float16.
    out_names = {tuple(o.type.multiArrayType.shape): o.name for o in spec.description.output}
    names = {"labels": out_names[(1, 300, 91)], "dets": out_names[(1, 300, 4)]}

    tf = infer_transforms((SIZE, SIZE))
    tgt = torch.tensor([[SIZE, SIZE]], dtype=torch.float32)
    rows, secs = [], []
    for i in range(len(arr)):
        t, _ = tf(Image.fromarray(np.asarray(arr[i])), None)
        t0 = time.perf_counter()
        o = m.predict({in_name: t.unsqueeze(0).numpy().astype(np.float32)})
        secs.append(time.perf_counter() - t0)
        r = post_process({k: torch.from_numpy(o[v].astype(np.float32)) for k, v in names.items()},
                         tgt)[0]
        keep = (r["scores"] >= KEEP_CONF) & (r["labels"] == person)
        rows.append([[round(float(v), 2) for v in b] + [round(float(s), 4)]
                     for b, s in zip(r["boxes"][keep], r["scores"][keep])])
        if i % 200 == 0:
            print(f"  tile {i}/{len(arr)}", flush=True)
    path = out or f"{WORK}/coreml.{units}.json"
    json.dump({"units": units, "model": os.path.basename(model), "tiles": rows,
               "ms_median": float(np.median(secs[1:]) * 1000)}, open(path, "w"))
    print(f"{units}: {np.median(secs[1:]) * 1000:.1f} ms/tile median, "
          f"{sum(len(r) for r in rows)} boxes >= {KEEP_CONF} -> {path}", flush=True)


def infer_torch(out: str | None = None) -> None:
    """The same tiles through PyTorch fp32 and the SAME post_process.

    Only needed if CoreML disagrees with the cache: it separates "fp16 on the
    ANE lost a box" from "benchmark.post_process is not predict()'s postprocess".
    """
    import torch
    from PIL import Image
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES
    from rfdetr.export.benchmark import infer_transforms, post_process

    arr = np.load(f"{WORK}/tiles.npy", mmap_mode="r")
    person = [k for k, v in COCO_CLASSES.items() if v == "person"][0]
    net = RFDETRMedium().model.model.eval()
    tf = infer_transforms((SIZE, SIZE))
    tgt = torch.tensor([[SIZE, SIZE]], dtype=torch.float32)
    rows = []
    with torch.inference_mode():
        for i in range(len(arr)):
            t, _ = tf(Image.fromarray(np.asarray(arr[i])), None)
            o = net(t.unsqueeze(0))
            r = post_process({"labels": o["pred_logits"].float(),
                              "dets": o["pred_boxes"].float()}, tgt)[0]
            keep = (r["scores"] >= KEEP_CONF) & (r["labels"] == person)
            rows.append([[round(float(v), 2) for v in b] + [round(float(s), 4)]
                         for b, s in zip(r["boxes"][keep], r["scores"][keep])])
            if i % 200 == 0:
                print(f"  tile {i}/{len(arr)}", flush=True)
    path = out or f"{WORK}/torch.fp32.json"
    json.dump({"units": "torch-fp32-cpu", "tiles": rows}, open(path, "w"))
    print(f"wrote {path}", flush=True)


# -------------------------------------------------------------------- score
def _replay(per_tile: list, tiles: list, wh: tuple, conf: float) -> list[list[dict]]:
    """Per-tile boxes -> per-frame boxes, through TiledDetector itself.

    Deliberately not re-derived here. The tile->frame mapping and the clipped-box
    merge are the classic tiling bug and they look like a bad model rather than
    arithmetic, so the production code does it: a stub inner detector replays the
    CoreML boxes in grid order and TiledDetector.detect does the rest on a blank
    frame of the right size (the pixels are never used - the boxes are already
    decoded).
    """
    from detectors import TiledDetector

    w, h = wh
    n = len(tiles)

    class _Replay:
        resolution = SIZE

        def __init__(self, rows):
            self.rows, self.i = rows, 0

        def detect_batch(self, frames):
            out = []
            for _ in frames:
                out.append([{"xyxy": tuple(r[:4]), "conf": r[4], "cls": "person"}
                            for r in self.rows[self.i] if r[4] >= conf])
                self.i += 1
            return out

    inner = _Replay(per_tile)
    det = TiledDetector(inner, tile=TILE, plan=tiles)
    blank = np.zeros((h, w, 3), np.uint8)
    return [det.detect(blank) for _ in range(len(per_tile) // n)]


def _match(a: list[dict], b: list[dict], thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy highest-IoU-first pairing, exclusive on both sides."""
    if not a or not b:
        return []
    A = np.array([d["xyxy"] for d in a], float)
    B = np.array([d["xyxy"] for d in b], float)
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda M: (M[:, 2] - M[:, 0]) * (M[:, 3] - M[:, 1])  # noqa: E731
    iou = inter / np.maximum(ar(A)[:, None] + ar(B)[None, :] - inter, 1e-9)
    pairs, ta, tb = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape)):
        if iou[i, j] < thr:
            break
        if i in ta or j in tb:
            continue
        ta.add(int(i)); tb.add(int(j)); pairs.append((int(i), int(j)))
    return pairs


def _band(dets: list[dict], cal) -> np.ndarray:
    """Depth band of each box, 0 = far touchline, same 6.8 m bins as detsweep."""
    if not dets:
        return np.zeros(0, int)
    b = np.array([d["xyxy"] for d in dets], float)
    feet = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
    y = cal.to_pitch(feet)[:, 1]
    return np.clip(np.floor(y / 6.8), 0, 9).astype(int)


def _agree(ref: list, new: list, cal, label: str) -> None:
    """Recall and precision of `new` against `ref`, overall and per band."""
    import collections

    hit = collections.Counter(); tot = collections.Counter()
    fp = collections.Counter(); nn = collections.Counter()
    for r, c in zip(ref, new):
        br, bc = _band(r, cal), _band(c, cal)
        for v in br:
            tot[int(v)] += 1
        for v in bc:
            nn[int(v)] += 1
        pairs = _match(r, c)
        for i, _ in pairs:
            hit[int(br[i])] += 1
        for j in set(range(len(c))) - {j for _, j in pairs}:
            fp[int(bc[j])] += 1
    ks = sorted(tot)
    R = sum(hit.values()) / max(1, sum(tot.values()))
    P = sum(hit.values()) / max(1, sum(nn.values()))
    print(f"\n  {label}: recall {R:.3f}  precision {P:.3f}  "
          f"({sum(tot.values())} ref boxes, {sum(nn.values())} coreml boxes)")
    print("    band    " + " ".join(f"{k:6d}" for k in ks))
    print("    n_ref   " + " ".join(f"{tot[k]:6d}" for k in ks))
    print("    recall  " + " ".join(f"{hit[k] / max(1, tot[k]):6.3f}" for k in ks))
    print("    prec    " + " ".join(f"{hit[k] / max(1, nn[k]):6.3f}" for k in ks))


def score(units: str = "CPU_AND_NE", conf: float = 0.5, extra: str | None = None) -> None:
    import calibrators
    import detsweep
    import sceneprior
    from track import make_prior

    meta = json.load(open(f"{WORK}/meta.json"))
    idx, tiles, wh = meta["frames"], [tuple(t) for t in meta["tiles"]], tuple(meta["wh"])
    cal = calibrators.load(f"{JOB}/calibration.json")
    cam = sceneprior.camera_from_homography(cal.H, wh)
    prior = make_prior(cal, wh)
    assert prior is not None, "no scene prior: the cache is post-prior, so this is not comparable"

    gt = detsweep.load_gt()
    gts = [gt[f] for f in idx]
    cache = {int(k): v for k, v in json.load(open(f"{JOB}/dets.cache.json"))["frames"].items()}
    py = [[{"xyxy": tuple(d["xyxy"]), "conf": d["conf"], "cls": d["cls"]}
           for d in cache[f] if d["conf"] >= conf] for f in idx]

    def load(path):
        raw = json.load(open(path))
        out = _replay(raw["tiles"], tiles, wh, conf)
        # The cache is written AFTER track.py's scene prior; comparing raw boxes
        # to filtered ones reads as a precision collapse that is really a filter.
        return [[d for d, k in zip(f, prior(np.array([x["xyxy"] for x in f], float)))
                 if k] if f else [] for f in out], raw

    runs = [(units, f"{WORK}/coreml.{units}.json")]
    if extra:
        runs.append((os.path.basename(extra), extra))

    print(f"{len(idx)} frames, conf {conf}, {sum(len(g) for g in gts)} GT player-frames")
    print(detsweep.HEAD)
    rp = detsweep.score(py, gts, cal, cam, conf, prior=True)
    print(detsweep._fmt("pytorch cache (shipping)", rp, 0.0))
    print(detsweep.bands(rp))
    print(f"    det/frame {np.mean([len(f) for f in py]):.2f}")

    for name, path in runs:
        cm, raw = load(path)
        rc = detsweep.score(cm, gts, cal, cam, conf, prior=True)
        print(detsweep._fmt(f"coreml {name}", rc, raw.get("ms_median", 0) / 1000 * len(tiles)))
        print(detsweep.bands(rc))
        print(f"    det/frame {np.mean([len(f) for f in cm]):.2f}")
        _agree(py, cm, cal, f"coreml {name} vs pytorch cache")
        # THE NULL. Same boxes, next frame's reference. Frames are 24 s apart, so
        # every player has moved; if this scores near the real number then the
        # metric is counting box density and nothing above it can be believed.
        _agree(py[1:] + py[:1], cm, cal, "NULL (shuffled reference)")


# -------------------------------------------------------------------- bench
def bench(n: int = 20, every: int = 600, first: int = 3000, conf: float = 0.5) -> None:
    """END-TO-END frames/s, PyTorch vs CoreML, through the real tiled path.

    Per-tile latency is the number that looks great while the pipeline gets
    slower: it is measured at batch 1, and the shipping PyTorch path traces at
    batch 8, and it excludes the 16 MB per frame that a subprocess detector has
    to move. So this times TiledDetector.detect() on real band frames - the
    whole thing, resize and merge included - and prints what the subprocess
    boundary cost inside that.

    The two paths run SEQUENTIALLY (both models in memory at once is 16 GB of
    trouble), on the same decoded frames, and both are compared to the shipping
    dets.cache.json with the same scene prior the cache was written behind.
    """
    import time

    import cv2

    import calibrators
    from detectors import CoreMLDetector, RFDETRDetector, TiledDetector
    from track import make_prior

    cal = calibrators.load(f"{JOB}/calibration.json")
    cache = {int(k): v for k, v in json.load(open(f"{JOB}/dets.cache.json"))["frames"].items()}
    idx = [f for f in range(first, first + n * every, every) if f in cache]
    assert len(idx) == n, f"{len(idx)} of {n} frames are in the cache"

    cap = cv2.VideoCapture(VIDEO)
    bands, t0 = [], time.perf_counter()
    try:
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, raw = cap.read()
            assert ok, f"could not read frame {f}"
            bands.append(cal.undistort_frame(raw))
    finally:
        cap.release()
    h, w = bands[0].shape[:2]
    prior = make_prior(cal, (w, h))
    assert prior is not None, "no scene prior: the cache is post-prior, not comparable"
    print(f"{n} frames of {w}x{h} decoded in {time.perf_counter() - t0:.1f}s", flush=True)

    def timed(det, label):
        out, secs = [], []
        for b in bands:
            t = time.perf_counter()
            d = det.detect(b)
            secs.append(time.perf_counter() - t)
            out.append(d)
        # Warm-up is real for both paths (MPS traces, the ANE loads); drop it.
        med = float(np.median(secs[1:]))
        print(f"  {label:34s} {med:6.3f} s/frame  {1 / med:5.2f} frames/s  "
              f"{np.mean([len(d) for d in out]):5.2f} det/frame", flush=True)
        return out, med

    res = {}
    py = RFDETRDetector(conf=conf, variant="medium")
    res["pytorch"] = timed(TiledDetector(py, tile=TILE), "PyTorch MPS fp16 (batch 8)")
    del py
    import gc

    gc.collect()

    cm = CoreMLDetector(conf=conf)
    tiled = TiledDetector(cm, tile=TILE, batch=16)
    res["coreml"] = timed(tiled, "CoreML fp16 CPU_AND_NE (16/write)")
    # THE BOUNDARY, on pre-resized tiles so the cv2 resize the PyTorch path
    # also pays does not get charged to the pipe. The worker reports what it
    # spent between reading the tiles and writing the reply; everything else in
    # the round trip is the boundary.
    tiles = [cv2.resize(bands[0][y:y + th, x:x + tw], (SIZE, SIZE),
                        interpolation=cv2.INTER_AREA)
             for x, y, tw, th in tiled.grid(w, h)]
    nb = len(tiles)

    def boundary(label, chunks):
        cm.worker_s = 0.0
        t = time.perf_counter()
        for _ in range(5):
            for c in chunks:
                cm.detect_batch(c)
        rt = (time.perf_counter() - t) / 5
        print(f"    {label:38s} {rt * 1000:6.1f} ms/frame, worker {cm.worker_s / 5 * 1000:6.1f}"
              f" -> {(rt - cm.worker_s / 5) * 1000:5.1f} ms of boundary")

    boundary(f"one write of {nb} tiles (what ships)", [tiles])
    boundary(f"{nb} single-tile round trips", [[c] for c in tiles])
    import pickle

    t = time.perf_counter()
    for _ in range(10):
        pickle.loads(pickle.dumps(tiles, protocol=5))
    print(f"    {'pickle of the same tiles (encode only)':38s} "
          f"{(time.perf_counter() - t) / 10 * 1000:6.1f} ms/frame")
    t = time.perf_counter()
    for _ in range(5):
        [cv2.resize(bands[0][y:y + th, x:x + tw], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
         for x, y, tw, th in tiled.grid(w, h)]
    print(f"    {'cv2 resize of the tiles (both paths pay)':38s} "
          f"{(time.perf_counter() - t) / 5 * 1000:6.1f} ms/frame")
    cm.close()

    # ------------------------------------------------------------- agreement
    ref = [[{"xyxy": tuple(d["xyxy"]), "conf": d["conf"], "cls": d["cls"]}
            for d in cache[f] if d["conf"] >= conf] for f in idx]

    def gate(runs):
        return [[d for d, k in zip(f, prior(np.array([x["xyxy"] for x in f], float))) if k]
                if f else [] for f in runs]

    for name in ("pytorch", "coreml"):
        _agree(ref, gate(res[name][0]), cal, f"{name} (live) vs dets.cache.json")
    _agree(gate(res["pytorch"][0]), gate(res["coreml"][0]), cal, "coreml vs pytorch, both live")
    # THE NULL: the same CoreML boxes against the previous frame's cache. Frames
    # are 24 s apart, so if this reads anywhere near the numbers above then the
    # metric counts box density and none of it means anything.
    _agree(ref[1:] + ref[:1], gate(res["coreml"][0]), cal, "NULL (shifted reference)")

    sp, sc = res["pytorch"][1], res["coreml"][1]
    print(f"\n  END TO END: {sp / sc:.2f}x  ({1 / sp:.2f} -> {1 / sc:.2f} frames/s); "
          f"a 71850-frame half at stride 2: {sp * 35925 / 3600:.1f} h -> "
          f"{sc * 35925 / 3600:.1f} h")


def _selfcheck() -> None:
    """The matcher and the replay, before any number off them is believed."""
    a = [{"xyxy": (0.0, 0.0, 10.0, 20.0), "conf": .9, "cls": "person"},
         {"xyxy": (50.0, 0.0, 60.0, 20.0), "conf": .9, "cls": "person"}]
    assert _match(a, a) == [(0, 0), (1, 1)] or sorted(_match(a, a)) == [(0, 0), (1, 1)]
    assert _match(a, a[::-1]) and len(_match(a, a[::-1])) == 2
    # One box cannot satisfy two references, and a miss stays a miss.
    assert len(_match(a, a[:1])) == 1
    assert _match(a, [{"xyxy": (200.0, 0.0, 210.0, 20.0), "conf": .9, "cls": "person"}]) == []
    # Shifted by half a box: IoU 0.33, below the 0.5 gate.
    assert _match(a, [{"xyxy": (0.0, 10.0, 10.0, 30.0), "conf": .9, "cls": "person"}]) == []

    # Replay puts a tile-space box at the right FULL-FRAME coordinates. Tile
    # (100, 50, 1024, 1024) scaled by 1024/576; a box at the tile origin must
    # land at the tile origin.
    tiles = [(100, 50, TILE, TILE), (1000, 50, TILE, TILE)]
    per_tile = [[[0.0, 0.0, 57.6, 115.2, 0.9]], [[0.0, 0.0, 57.6, 115.2, 0.3]]]
    got = _replay(per_tile, tiles, (4096, 1080), 0.5)
    assert len(got) == 1 and len(got[0]) == 1, got          # conf 0.3 filtered out
    x1, y1, x2, y2 = got[0][0]["xyxy"]
    s = TILE / SIZE
    assert abs(x1 - 100) < 1e-6 and abs(y1 - 50) < 1e-6, got
    assert abs(x2 - (100 + 57.6 * s)) < 1e-3 and abs(y2 - (50 + 115.2 * s)) < 1e-3, got
    print("coreml_check self-check OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    args = sys.argv[2:]
    if cmd == "dump":
        dump(*(int(a) for a in args))
    elif cmd == "infer":
        infer(*args)
    elif cmd == "infer-torch":
        infer_torch(*args)
    elif cmd == "bench":
        bench(*(int(a) for a in args[:3]), *(float(a) for a in args[3:4]))
    elif cmd == "score":
        score(args[0] if args else "CPU_AND_NE",
              float(args[1]) if len(args) > 1 else 0.5,
              args[2] if len(args) > 2 else None)
    else:
        _selfcheck()
