"""Detection stage: one frame -> boxes in original-frame pixels.

Backends through get_detector; "rfdetr" and "coreml" are tiled by default
(TiledDetector), because the model squeezes its input to a square and a wide
frame loses its small players. Player and ball classes are thresholded
separately (BALL_CONF, _keep).

Licensing note, load-bearing: RF-DETR is Apache-2.0 for both the code and the
COCO checkpoints Roboflow publish, for variants nano/small/medium/large ONLY -
XL and 2XL ship under PML 1.0 and are refused below. Ultralytics YOLO (AGPL-3.0),
YOLOv9 (GPL), YOLO-NAS weights (no managed-service use) and BoxMOT (AGPL) must
not appear here, even transitively. AGPL triggers on network use and we ship
this as a SaaS. See README.

Also load-bearing: nothing here reads faces or gait. A detector returns boxes.
See README on BIPA.

Manual: docs/reference/detection-tracking.md#detection
(scale and tiling: #evidence-tiling; the far touchline: #evidence-far-touchline;
the confidence decision: #conf)
"""
import atexit
import json
import os
import struct
import sys
from typing import Protocol

import cv2
import numpy as np

# Classes we keep. "person" is what a COCO-pretrained detector actually emits;
# the soccer-specific names are here for a future finetune.
PLAYER_CLASSES = frozenset({"person", "player", "goalkeeper", "referee"})

# The ball has its own threshold; players stay at `conf`. When a caller asks for
# the ball the detector runs at min(conf, BALL_CONF) and _keep splits the two
# classes afterwards, so player output is unchanged.
# Evidence: detection-tracking-evidence (not published)#evidence-ball
BALL_CONF = 0.05


def _keep(name: str, conf: float, player_conf: float, ball_conf: float,
          ball: bool) -> bool:
    """One box, two thresholds. See BALL_CONF for why they cannot be one."""
    if name in PLAYER_CLASSES:
        return conf >= player_conf
    if ball and name in BALL_CLASSES:
        return conf >= ball_conf
    return False

# The ball, kept SEPARATE from players (it must never reach the player tracker;
# see ball.py) and off by default: on some footage COCO's "sports ball" fires on
# legs and line markings, and ball.py's shape gate is what tells them apart.
# Evidence: detection-tracking-evidence (not published)#evidence-ball
BALL_CLASSES = frozenset({"sports ball", "ball"})

# Apache-2.0 checkpoints. XL/2XL are PML 1.0 - not licence-compatible with a
# hosted service, so they are not reachable from here by any argument.
RFDETR_VARIANTS = ("nano", "small", "medium", "large")


class Detector(Protocol):
    """frame (BGR, as OpenCV hands it over) -> list of boxes.

    Each box: {"xyxy": (x1, y1, x2, y2), "conf": float, "cls": str}.
    Coordinates are pixels in the ORIGINAL frame, whatever the model ran at.
    """

    def detect(self, frame: np.ndarray) -> list[dict]: ...


class StubDetector:
    """Deterministic moving boxes. No weights, no GPU, no network.

    Exists so the whole pipeline - tracking, kit-colour teams, JSON contract -
    is runnable and testable on any machine. Boxes drift at a constant velocity
    so association is genuinely exercised rather than trivially satisfied.
    """

    def __init__(self, n: int = 4, size: tuple[int, int] = (24, 60),
                 step: tuple[float, float] = (6.0, 1.0), origin: tuple[float, float] = (60.0, 80.0),
                 spacing: float = 110.0, conf: float = 0.9):
        self.n, self.size, self.step = n, size, step
        self.origin, self.spacing, self.conf = origin, spacing, conf
        self._f = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        w, h = self.size
        out = []
        for i in range(self.n):
            x = self.origin[0] + i * self.spacing + self.step[0] * self._f
            y = self.origin[1] + self.step[1] * self._f
            out.append({"xyxy": (x, y, x + w, y + h), "conf": self.conf, "cls": "person"})
        self._f += 1
        return out


def _round_to(value: int, block: int) -> int:
    """Nearest positive multiple of `block`."""
    return max(block, int(round(value / block)) * block)


def _checkpoint_classes(path: str) -> dict[int, str] | None:
    """Class id -> name off a finetuned checkpoint, or None for a COCO one.

    rfdetr writes the training set's category names into the checkpoint as
    `args.class_names`, in the order the COCO json listed them, and a FINETUNED
    model emits them 0-BASED: a single-class ball model returns class_id 0 for
    the ball. This started at 1, on the reasoning that id 0 is a reserved
    background slot - which is true of the COCO table this falls back to and
    NOT of a finetuned head. The cost was total and silent: every detection
    mapped to a name nothing recognised, `_boxes` dropped all of them, and a
    model that finds the ball on 18 of 20 held-out tiles returned nothing at
    all through this path.

    Never fatal: a checkpoint without names is a COCO one, and the caller
    carries on with the COCO table.
    """
    try:
        import torch  # noqa: PLC0415

        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                       # noqa: BLE001
        print(f"  could not read class names from {path}: {e}", flush=True)
        return None
    args = ck.get("args") if isinstance(ck, dict) else None
    names = getattr(args, "class_names", None) or (
        args.get("class_names") if isinstance(args, dict) else None)
    if not names:
        return None
    return {i: str(n) for i, n in enumerate(names)}


class RFDETRDetector:
    """RF-DETR (Apache-2.0). COCO-pretrained by default: class "person".

    Wrap this in TiledDetector for wide shots - see the module docstring. On its
    own it sees one square 576 (medium) view of the whole frame.

    `resolution` is the model's input square. It MUST be a multiple of
    patch_size * num_windows (32 for every variant here) or RF-DETR raises at
    predict time, i.e. after the weights have loaded, minutes into a run. It is
    rounded here rather than validated: the caller asked for "bigger", and
    refusing over 16 px is not a service. The old code additionally passed the
    same number as `shape=` to predict(), which duplicated the resize and made
    `-r 1008` a guaranteed ValueError; predict() now gets no shape at all and
    uses the resolution the model was built with.
    """

    def __init__(self, weights: str | None = None, conf: float = 0.5,
                 variant: str = "medium", resolution: int | None = None,
                 device: str | None = None, half: bool = True,
                 num_queries: int | None = None, ball: bool = False,
                 ball_conf: float = BALL_CONF):
        # num_queries: the decoder's object slots (None = the checkpoint's). Left at
        # the default on purpose: fewer queries lost recall on this footage.
        # Evidence: detection-tracking-evidence (not published)#evidence-queries
        if variant not in RFDETR_VARIANTS:
            raise SystemExit(f"variant {variant!r} not allowed; have {list(RFDETR_VARIANTS)} "
                             "(XL/2XL are PML 1.0, see module docstring)")
        import rfdetr  # noqa: PLC0415 - heavy optional import, torch pulls ~2GB
        from rfdetr import config as rfconfig  # noqa: PLC0415

        name = variant.capitalize()
        cfg = getattr(rfconfig, f"RFDETR{name}Config")()  # cheap: pydantic, no weights
        block = cfg.patch_size * cfg.num_windows

        kw: dict = {}
        if weights:
            kw["pretrain_weights"] = weights
        if resolution:
            kw["resolution"] = _round_to(resolution, block)
            if kw["resolution"] != resolution:
                print(f"  resolution {resolution} -> {kw['resolution']} (must be a multiple of {block})")
        if device:
            kw["device"] = device
        if num_queries:
            if num_queries > cfg.num_queries:
                raise SystemExit(
                    f"num_queries {num_queries} > the checkpoint's {cfg.num_queries}; "
                    "raising it adds slots the pretrained weights do not have")
            kw["num_queries"] = num_queries
        self.model = getattr(rfdetr, f"RFDETR{name}")(**kw)
        self.resolution = kw.get("resolution", cfg.resolution)
        self.conf = conf
        # See BALL_CONF: detect at the lower of the two, split by class after.
        self.ball_conf = ball_conf
        self._det_conf = min(conf, ball_conf) if ball else conf
        # fp16 + torch.jit.trace: 1.33x throughput on MPS at no measurable cost
        # in recall (40 real frames: 24.3 vs 24.4 det/frame, 27 ids either way).
        # Off on CPU, where fp16 is emulated and slower. Any failure to trace
        # falls back to fp32 rather than taking the run down - see detect_batch.
        self.half = half and "cpu" not in str(getattr(self.model.model, "device", "cpu")).lower()
        self._bs = 0

        # CLASS NAMES. A COCO checkpoint calls everyone "person" - including the
        # referee, which is why kit colour has to guess at officials and cannot
        # tell a referee from a goalkeeper (track._officials_by_contrast). A
        # checkpoint FINETUNED on football names them, and those names are worth
        # more than any colour test, so they are read off the checkpoint itself
        # rather than assumed.
        from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: PLC0415

        self.names = COCO_CLASSES
        if weights:
            named = _checkpoint_classes(weights)
            if named:
                self.names = named
                print(f"  {os.path.basename(weights)} names its classes: "
                      f"{', '.join(sorted(set(named.values())))}", flush=True)
        # See BALL_CLASSES: off unless a caller has decided this footage can
        # carry a ball, because on the wrong footage it emits confident legs.
        self.ball = ball

    def _boxes(self, det) -> list[dict]:
        out = []
        for box, c, cid in zip(det.xyxy, det.confidence, det.class_id):
            name = self.names.get(int(cid), str(cid))
            if not _keep(name, float(c), self.conf, self.ball_conf, self.ball):
                continue
            out.append({"xyxy": tuple(float(v) for v in box), "conf": float(c), "cls": name})
        return out

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """One forward pass for N frames/tiles. ~1.7x cheaper per image than N
        separate calls (measured, MPS): the per-call overhead is real."""
        # rfdetr wants RGB; OpenCV gives BGR. Getting this backwards costs
        # roughly half the recall and looks like a bad model rather than a bug.
        # cvtColor, not a [::-1] view: torch refuses negative-stride arrays.
        imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        n = len(imgs)
        if self.half and not self._bs:
            self.half, self._bs = False, 0
            try:
                import torch  # noqa: PLC0415

                self.model.inference(compile=True, batch_size=n, dtype=torch.float16)
                self._bs = n
            except Exception as e:  # noqa: BLE001 - any failure means "run fp32"
                print(f"  fp16 unavailable ({type(e).__name__}: {e}); running fp32", flush=True)
        if self._bs:
            # A traced model accepts exactly the batch size it was traced at.
            # The first call is always the largest chunk (TiledDetector emits
            # full chunks first), so later batches are only ever short: pad them
            # and drop the padding's results.
            imgs = imgs + [imgs[-1]] * (self._bs - n)
        det = self.model.predict(imgs, threshold=self._det_conf, include_source_image=False)
        return [self._boxes(d) for d in det[:n]]

    def detect(self, frame: np.ndarray) -> list[dict]:
        return self.detect_batch([frame])[0]


def _bundled(name: str) -> str:
    """<binary's own directory>/coreml/<name> in a PyInstaller build, else "".

    THE TWO LAYOUTS. In a dev checkout the CoreML half is a 3.12 venv the
    developer made and a .mlpackage under model/models/, and nothing is bundled. In
    the desktop app there is no checkout: PyInstaller writes one directory,
    electron-builder copies it whole to Contents/Resources/backend, and
    sys.executable is the frozen binary sitting at the top of it. So
    dirname(sys.executable) is the anchor - NOT sys._MEIPASS, which points at
    _internal and would put the worker inside PyInstaller's own tree.

    app/desktop/scripts/build-backend.sh fills coreml/ with a second, frozen
    PyInstaller build of coreml_worker.py against a Python 3.12, plus the
    .mlpackage. Both are optional: if the build could not make them, this
    returns a path that does not exist, the caller's os.path.exists says no,
    and jobs.py records "rfdetr". A path is not a file, and only the second one
    is worth anything - see _selfcheck.
    """
    if not getattr(sys, "frozen", False):
        return ""
    return os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "coreml", name)


# The frozen worker is a whole interpreter, so it stands in for one; it takes
# the same argv (model, units, conf, thresholds) because coreml_worker.py reads sys.argv[1:]
# either way. CoreMLDetector spots it by name and drops the script argument.
COREML_WORKER_EXE = "coreml-worker"
COREML_PYTHON = (os.environ.get("FT_COREML_PYTHON") or _bundled(COREML_WORKER_EXE)
                 or os.path.expanduser("~/.venvs/ftcoreml/bin/python"))
#: The FINE-TUNED BALL detector (N52 arm D1: RF-DETR Nano on TeamTrack tiles, MIT),
#: distributed like COREML_MODEL below: gitignored, bundled at build time or pointed at
#: by an env var, never in git. Run at BALL_RESOLUTION and BALL_TILE, the geometry it
#: was trained in. OFF unless a caller asks for it: nothing reads this constant on its
#: own. Evidence: detection-tracking-evidence (not published)#evidence-ball
BALL_MODEL = (os.environ.get("FT_BALL_MODEL") or _bundled("ball-nano-640.pth")
              or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "model", "models", "ball-nano-640.pth"))
BALL_VARIANT, BALL_TILE, BALL_RESOLUTION = "nano", 640, 640
#: THE LICENCE GATE. Two fine-tuned nano ball checkpoints exist and they are not
#: interchangeable: N52 arm D1 was trained on TeamTrack (MIT, commercial use permitted) and
#: arm B on DFL Bundesliga data, which is COMPETITION USE ONLY and must never ship
#: (internal note `ball-detector-trained-not-yet-scored`). They are both a 117 MB
#: `checkpoint_best_ema.pth`, so the file NAME cannot tell them apart - dropping arm B at
#: BALL_MODEL would ship licence-blocked weights through the product with nothing noticing.
#: track.run asserts this hash before the ball detector loads.
BALL_MODEL_SHA256 = "f1cef6d7cb16c10e2ca4991502ec0c2c471ed910415daa101ace48974d5d0999"

COREML_MODEL = (os.environ.get("FT_COREML_MODEL") or _bundled("rfdetr-medium-576.mlpackage")
                or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "model", "models", "rfdetr-medium-576.mlpackage"))


class CoreMLDetector:
    """RF-DETR on the Apple Neural Engine, through a Python 3.12 subprocess.

    Same contract as RFDETRDetector - detect/detect_batch, boxes in the
    coordinates of whatever image was handed over - so TiledDetector wraps it
    unchanged. A job runs it when both halves exist (product_detector); "rfdetr"
    stays PyTorch.

    WHY A SUBPROCESS: the ANE is reachable only through CoreML, and coremltools
    has no working native extension for this repo's Python 3.14 - it installs
    and dies at runtime with "BlobWriter not loaded". A 3.12 interpreter runs
    the .mlpackage; model/engine/coreml_worker.py is that half and documents the
    wire format.

    NO SILENT FALLBACK: a missing 3.12 interpreter or .mlpackage raises here, and
    every reply carries a sequence number the parent asserts, so a worker that
    returns stale or empty boxes cannot pass for a fast one.

    Throughput against PyTorch, agreement with the PyTorch boxes and long-run
    behaviour are measured in detection-tracking-evidence (not published)#evidence-coreml.

    WHAT REACHES IT. --conf and -w both do (track.py builds {"conf", "weights"}
    for this detector); --variant and --resolution do NOT, because a .mlpackage
    is exported at one variant and one input square and would raise a TypeError
    rather than no-op. That fixes the shape of the whole knob set: everything
    about this model except the threshold is baked at export time, so retuning
    resolution means exporting again, not passing a flag. $FT_COREML_MODEL and
    $FT_COREML_PYTHON pick the bundle and the interpreter.
    """

    def __init__(self, weights: str | None = None, conf: float = 0.5,
                 units: str = "CPU_AND_NE", python: str | None = None,
                 ball: bool = False, ball_conf: float = BALL_CONF):
        import subprocess  # noqa: PLC0415

        model = weights or COREML_MODEL
        exe = python or COREML_PYTHON
        if not os.path.exists(model):
            raise SystemExit(
                f"CoreML model not found: {model}\n"
                "  export one with, on the 3.12 interpreter:\n"
                "    RFDETRMedium().export(format='coreml', coreml_precision='float16', "
                "shape=(576,576), batch_size=1)\n"
                "  (a fine-tuned checkpoint: model/validation/n46_coreml.py export, which also stores its class names)\n"
                "  then point FT_COREML_MODEL at the .mlpackage.")
        if not os.path.exists(exe):
            raise SystemExit(
                f"no CoreML worker at {exe}\n"
                "  coremltools has no working build for this repo's 3.14, so the worker needs "
                "a 3.12.\n"
                "  dev: python3.12 -m venv ~/.venvs/ftcoreml && ~/.venvs/ftcoreml/bin/pip "
                "install coremltools numpy\n"
                "  packaged: app/desktop/scripts/build-backend.sh freezes one into "
                "resources/backend/coreml/;\n"
                "    this build did not get one. $FT_COREML_PYTHON overrides either.")
        self.conf, self.ball, self.units = conf, ball, units
        # See BALL_CONF. The worker is handed the LOWER threshold; the split by
        # class happens here, on the way back.
        self.ball_conf = ball_conf
        self._det_conf = min(conf, ball_conf) if ball else conf
        # A frozen worker IS the script; an interpreter needs to be handed one.
        cmd = [exe] if os.path.basename(exe) == COREML_WORKER_EXE else [
            exe, os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml_worker.py")]
        # Per-class thresholds, so the worker pipes only what _keep below will keep.
        thr = {c: conf for c in PLAYER_CLASSES} | ({c: ball_conf for c in BALL_CLASSES} if ball else {})
        self.proc = subprocess.Popen(cmd + [model, units, str(self._det_conf), json.dumps(thr)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        atexit.register(self.close)
        hello = self.proc.stdout.readline()
        if not hello:
            raise SystemExit(f"CoreML worker died on startup (exit {self.proc.wait()}); "
                             "its traceback is above")
        info = json.loads(hello)
        self.resolution = int(info["size"])
        self._seq = 0
        self.worker_s = 0.0

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        if self.proc.poll() is not None:
            raise SystemExit(f"CoreML worker exited ({self.proc.returncode}) mid-run")
        s = self.resolution
        w = self.proc.stdin
        w.write(struct.pack("<I", len(frames)))
        for f in frames:
            if f.shape != (s, s, 3) or f.dtype != np.uint8:
                raise SystemExit(f"CoreML worker takes {s}x{s}x3 uint8 tiles, got "
                                 f"{f.shape} {f.dtype}")
            # Raw bytes, no copy: cv2.resize already returns C-contiguous.
            w.write(f if f.flags.c_contiguous else np.ascontiguousarray(f))
        w.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"CoreML worker died mid-run (exit {self.proc.wait()}); "
                             "its traceback is above")
        reply = json.loads(line)
        if reply["seq"] != self._seq or len(reply["tiles"]) != len(frames):
            raise SystemExit(f"CoreML worker out of step: asked for {len(frames)} tiles as "
                             f"seq {self._seq}, got {len(reply['tiles'])} as {reply['seq']}")
        self._seq += 1
        self.worker_s += reply.get("ms", 0.0) / 1000  # what the ANE side spent
        return [[{"xyxy": (r[0], r[1], r[2], r[3]), "conf": r[4], "cls": r[5]}
                 for r in tile
                 if _keep(r[5], r[4], self.conf, self.ball_conf, self.ball)]
                for tile in reply["tiles"]]

    def detect(self, frame: np.ndarray) -> list[dict]:
        return self.detect_batch([frame])[0]

    def close(self) -> None:
        """Shut the worker down. Idempotent - atexit and __del__ both call it."""
        p = getattr(self, "proc", None)
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.write(struct.pack("<I", 0))     # the "no more tiles" header
            p.stdin.flush()
            p.stdin.close()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001 - a wedged worker gets killed, not debugged
            p.kill()
            p.wait()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter teardown; nothing to report to
            pass


def _spans(length: int, tile: int, overlap: float) -> tuple[list[int], int]:
    """Tile offsets and tile size along one axis, evenly spread.

    If the axis is barely longer than the tile, one tile covers it: two nearly
    identical rows would double the cost to re-see the same pixels.
    """
    if length <= tile * (1 + overlap):
        return [0], length
    stride = max(1, int(tile * (1.0 - overlap)))
    n = (length - tile + stride - 1) // stride + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)], tile


def _ios(boxes: np.ndarray) -> np.ndarray:
    """(N,4) xyxy -> (N,N) intersection over the SMALLER area.

    Not IoU. The duplicate a tiling scheme actually produces is a box cut in
    half by a tile edge sitting inside the whole box from the next tile over:
    IoU 0.5, which survives any sane NMS threshold, but IoS 1.0.
    """
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-9)


def _merge(dets: list[dict], thr: float) -> list[dict]:
    """Cross-tile de-duplication. Whole boxes beat clipped ones, then confidence.

    Ordering is the point. A box clipped by a tile edge is half a player, and
    track.py takes the ground point from the box BOTTOM - keeping the clipped
    copy of a vertically split player puts their feet at their waist. So a
    detection that touches an inner tile edge can only ever be the survivor if
    nothing whole overlaps it.

    # ponytail: IoS suppression is deliberately aggressive - a box almost
    # wholly inside another is treated as the same player. Ceiling is a player
    # standing directly in front of a much larger near-side player, where the
    # far one is dropped. Upgrade path is IoS only between boxes from DIFFERENT
    # tiles and IoU within a tile; measure a corner kick before paying for it.
    """
    if len(dets) < 2:
        return [{k: v for k, v in d.items() if k not in ("_clip", "_tile")} for d in dets]
    order = sorted(range(len(dets)), key=lambda i: (dets[i]["_clip"], -dets[i]["conf"]))
    ios = _ios(np.array([d["xyxy"] for d in dets], dtype=np.float64))
    keep: list[int] = []
    for i in order:
        if not keep or ios[i, keep].max() < thr:
            keep.append(i)
    return [{k: v for k, v in dets[i].items() if k not in ("_clip", "_tile")} for i in sorted(keep)]


# Throughput history (the CoreML port, fp16, detecting every second frame):
# detection-tracking-evidence (not published)#evidence-stride


#: Tile overlap. More overlap gives a far player a second look from another offset,
#: which matters where one look lands under the threshold. Chosen by a paired,
#: pre-registered rule on the remapped pitch band (detsweep.overlap_decision); it costs
#: extra tiles on the CoreML path. docs/reference/detection-tracking.md#overlap
OVERLAP = 0.25


class TiledDetector:
    """Overlapping-tile inference (the SAHI idea, ~40 lines instead of a dep).

    Crop the frame into a grid of ~`tile`-px tiles with `overlap` between
    neighbours, resize each to the detector's input square, detect, map the
    boxes back to full-frame pixels, merge.

    What this actually fixes, on this footage, is ASPECT more than size. RF-DETR
    stretches its input to a square, so 4096x1080 is squeezed 7.1x across and
    1.9x down: a far-side player 25 px tall and 10 px wide arrives 13x1.4 px, a
    vertical line at 9.5:1 where a person is 2.5:1. At tile=1024 the same player
    arrives 13x5.6 px at 2.4:1 - barely taller, but shaped like a human. Smaller
    tiles buy real size on top of that (tile 768 -> 19x7.5) at a steeper price.

    Cost is tiles-per-frame, near enough: 5 tiles is 5 forward passes. There is
    no free lunch here, only a dial. tile=None/0 turns tiling off.

    Boundaries: with 25% overlap any player smaller than ~0.25*tile is wholly
    inside at least one tile, so nothing has to survive as a clipped fragment,
    and _merge prefers the whole copy. Fragments are still emitted and merged
    rather than dropped, so a player larger than the overlap degrades to a
    clipped box instead of vanishing.

    # ponytail: fixed grid, every tile every frame. Ceiling is throughput - the
    # far touchline is re-scanned at full cost even when it is empty. Upgrade
    # path is a foreground/motion gate (MOG2) or the sceneprior per-row scale to
    # skip tiles, worth roughly 2x; measure that the tracker survives the
    # resulting gaps before paying for it.
    """

    def __init__(self, inner: Detector, tile: int = 1024, overlap: float = OVERLAP,
                 batch: int = 8, merge_thr: float = 0.55,
                 plan: list[tuple[int, int, int, int]] | None = None, raw: bool = False):
        self.inner, self.tile, self.overlap = inner, tile, overlap
        self.batch, self.merge_thr = batch, merge_thr
        # `raw`: skip _merge and keep every tile's boxes with `_tile` (index) and `_clip`, so a merge rule can be
        # applied and measured offline (model/validation/n42_merge.py). Never the product's path.
        self.raw = raw
        self.size = int(getattr(inner, "resolution", 576))
        # An explicit plan overrides the uniform grid: tiles of different sizes
        # in different places, which is the only way to spend more pixels on the
        # far touchline than on the near one. See geometry_tiles().
        self.plan = None if plan is None else [tuple(int(v) for v in t) for t in plan]
        self._grid: tuple | None = None

    def grid(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """Cached (x, y, tw, th) tiles for a frame size."""
        if self.plan is not None:
            return self.plan
        if self._grid is None or self._grid[0] != (w, h):
            xs, tw = _spans(w, self.tile, self.overlap)
            ys, th = _spans(h, self.tile, self.overlap)
            self._grid = ((w, h), [(x, y, tw, th) for y in ys for x in xs])
        return self._grid[1]

    def detect(self, frame: np.ndarray) -> list[dict]:
        """One frame -> merged boxes in FRAME pixels. Tiles go through the inner
        detector in chunks of `batch`. raw=True returns every tile's boxes unmerged,
        with `_tile` (tile index) and `_clip` (touches an inner tile edge)."""
        h, w = frame.shape[:2]
        tiles = self.grid(w, h)
        # INTER_AREA when the tile is bigger than the model's input, because
        # that is a downscale and area-averaging is what keeps a 3-px-wide far
        # player from aliasing away. A geometry plan inverts the case: its far
        # tiles are SMALLER than the input and get upscaled, and cv2's
        # INTER_AREA degenerates to nearest-neighbour when it upsamples - which
        # would hand the model a blocky 3.5x zoom and then blame the model.
        crops = [cv2.resize(frame[y:y + th, x:x + tw], (self.size, self.size),
                            interpolation=cv2.INTER_AREA if tw >= self.size else cv2.INTER_LINEAR)
                 for x, y, tw, th in tiles]
        m = 2.0  # px of slack when asking "did this box hit the tile edge?"
        dets: list[dict] = []
        for i in range(0, len(crops), self.batch):
            chunk, meta = crops[i:i + self.batch], tiles[i:i + self.batch]
            for j, ((x, y, tw, th), found) in enumerate(zip(meta, _run(self.inner, chunk))):
                sx, sy = tw / self.size, th / self.size
                for d in found:
                    x1, y1, x2, y2 = d["xyxy"]
                    clip = ((x1 < m and x > 0) or (y1 < m and y > 0)
                            or (x2 > self.size - m and x + tw < w)
                            or (y2 > self.size - m and y + th < h))
                    dets.append({**d, "_clip": clip, "_tile": i + j,
                                 "xyxy": (x + x1 * sx, y + y1 * sy, x + x2 * sx, y + y2 * sy)})
        return dets if self.raw else _merge(dets, self.merge_thr)


def _strip_tiles(x0: int, x1: int, y0: int, y1: int, side: int, overlap: float,
                 w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Square `side` tiles covering the box, clamped to the frame.

    SQUARE, deliberately. A wide-short strip fed to the model as one crop is
    free vertical magnification, and it is worthless: the model stretches its
    input to a square, so a 3050x150 strip turns a 10x33 px player into a 5x170
    smear at 34:1 where a person is 3.3:1. Isotropy is the whole point of tiling
    (see the class docstring), so a strip shorter than the tile is CENTRED in a
    square crop rather than stretched into one.
    """
    side = max(32, min(side, min(w, h)))

    def axis(a0: int, a1: int, limit: int) -> tuple[list[int], int]:
        if a1 - a0 <= side:                 # centre the span in a square crop
            return [max(0, min(limit - side, (a0 + a1) // 2 - side // 2))], side
        offs, sz = _spans(a1 - a0, side, overlap)
        return [max(0, min(limit - sz, a0 + o)) for o in offs], sz

    xs, tw = axis(x0, x1, w)
    ys, th = axis(y0, y1, h)
    return [(x, y, tw, th) for y in ys for x in xs]


def geometry_tiles(cam, cal, image_wh: tuple[int, int], res: int = 768,
                   target_px: int = 64, overlap: float = OVERLAP,
                   margin_m: float = 3.0, max_h_px: float | None = None,
                   ) -> list[tuple[int, int, int, int]]:
    """A tiling built from the scene prior instead of from the frame's shape.

    A uniform grid spends the same pixels on every part of the image, and the
    image is a perspective view: on 118575's 11878x870 band a player is 33 px
    tall at the far touchline and 165 px at the near one, so tile=1024 hands the
    far player 0.75x magnification he cannot use and the near player 0.75x he
    does not need. sceneprior.expected_height_px says exactly how tall a player
    is at every point, so tile size can follow it.

    The other half of the geometry, and the reason this is affordable: the far
    field is NOT wide. The pitch is a trapezoid, and on 118575 the far touchline
    occupies columns 4751..6705 of 11878 - 1954 px - while the near touchline
    spans 274..11095. Magnifying the far field 3x therefore costs ~13 extra
    tiles, not the ~100 a full-width far strip would.

    Tiles are grouped by OCTAVE of expected height (each strip covers a 2x
    range, which one tile size serves), and each strip's tile side is chosen so
    a player there lands ~`target_px` tall in the model's square input.

    `max_h_px` restricts the plan to the far field: pass e.g. 48 to get only the
    strips where players are under 48 px, to add to a uniform grid rather than
    replace it. Returns (x, y, w, h) tiles for TiledDetector(plan=...).

    Measured on 118575 against the uniform grid, and why the far pass is not
    wired: detection-tracking-evidence (not published)#evidence-tiling.

    # ponytail: strips are octaves of expected height, one tile size each. The
    # ceiling is that a 2x height range inside one strip gets one compromise
    # scale. Upgrade path is per-row tile sizing, which costs tiles for a
    # magnification the detector's own scale tolerance already covers - measure
    # a strip's inner spread before paying for it.
    """
    import sceneprior  # noqa: PLC0415 - only geometry tiling needs the prior

    w, h = image_wh
    L, W = cal.pitch.length, cal.pitch.width
    gx, gy = np.meshgrid(np.linspace(-margin_m, L + margin_m, 80),
                         np.linspace(-margin_m, W + margin_m, 60))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    ph = np.column_stack([xy, np.ones(len(xy))])
    q = (cam.G @ ph.T).T
    uv = q[:, :2] / q[:, 2:3]
    ht = sceneprior.expected_height_px(cam, xy)
    # Only points that can actually hold a visible player. The head is above the
    # ground point, so a tile must reach uv_y - height to contain the whole box.
    ok = (ht > 0) & (uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)
    if max_h_px is not None:
        ok &= ht < max_h_px
    uv, ht = uv[ok], ht[ok]
    if not len(uv):
        return []

    tiles: list[tuple[int, int, int, int]] = []
    oct_idx = np.floor(np.log2(ht / ht.min())).astype(int)
    for o in np.unique(oct_idx):
        s = oct_idx == o
        x0, x1 = int(np.floor(uv[s, 0].min())), int(np.ceil(uv[s, 0].max()))
        y0 = int(np.floor((uv[s, 1] - ht[s]).min()))
        y1 = int(np.ceil(uv[s, 1].max()))
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        # tile side such that res/side * expected_height = target_px
        side = int(round(res * float(np.median(ht[s])) / target_px))
        tiles += _strip_tiles(x0, x1, y0, y1, side, overlap, w, h)
    return sorted(set(tiles))


def _run(det: Detector, frames: list[np.ndarray]) -> list[list[dict]]:
    """Batch if the detector can, loop if it cannot (StubDetector cannot)."""
    fn = getattr(det, "detect_batch", None)
    return fn(frames) if fn else [det.detect(f) for f in frames]




def _rfdetr(tile: int | None = 1024, overlap: float = OVERLAP, batch: int = 8, **kw) -> Detector:
    """RF-DETR, tiled by default: whole-frame inference loses the small players on
    a wide frame (detection-tracking-evidence (not published)#evidence-tiling). `tile=0` selects the untiled path."""
    inner = RFDETRDetector(**kw)
    return TiledDetector(inner, tile=tile, overlap=overlap, batch=batch) if tile else inner


def _coreml(tile: int | None = 1024, overlap: float = OVERLAP, batch: int = 16, **kw) -> Detector:
    """Same tiling as _rfdetr, on the ANE. batch is the tiles per pipe write:
    the .mlpackage is batch 1 so the worker loops either way, and 16 sends a
    whole frame in one crossing instead of two."""
    inner = CoreMLDetector(**kw)
    return TiledDetector(inner, tile=tile, overlap=overlap, batch=batch) if tile else inner


def _nvidia(which: str):
    """CUDA / TensorRT, imported ONLY when asked for.

    nvidia.py imports torch expecting a CUDA build and raises on this machine by
    design, so it must not be imported at module scope - `import detectors` has
    to keep working on a Mac. Lazy is the whole point here, not an optimisation.
    """
    def make(**kw):
        import nvidia  # noqa: PLC0415
        return getattr(nvidia, which)(**kw)
    return make


_DETECTORS = {"rfdetr": _rfdetr, "rfdetr-whole": RFDETRDetector, "coreml": _coreml,
              "cuda": _nvidia("_cuda"), "tensorrt": _nvidia("_tensorrt"),
              "stub": StubDetector}


def get_detector(name: str, **kw) -> Detector:
    """Build the detector registered as `name` (_DETECTORS), passing `kw` to its
    constructor; unknown names exit. What a job runs is product_detector()."""
    if name not in _DETECTORS:
        raise SystemExit(f"unknown detector {name!r}; have {sorted(_DETECTORS)}")
    return _DETECTORS[name](**kw)


def product_detector() -> str:
    """What a job runs: coreml when BOTH halves (bundle + 3.12 worker) exist, else rfdetr.
    The one rule, used by app/server/jobs.py and model/validation/check.py."""
    return "coreml" if os.path.exists(COREML_MODEL) and os.path.exists(COREML_PYTHON) else "rfdetr"


def tile_of(name: str) -> int | None:
    """The tile the named detector's constructor runs at (None: untiled, or not readable -
    the nvidia makers take **kw and cannot be imported on a Mac)."""
    import inspect  # noqa: PLC0415
    p = inspect.signature(_DETECTORS[name]).parameters.get("tile") if name in _DETECTORS else None
    return p.default if p is not None and p.default is not p.empty else None


def model_identity(name: str, weights: str | None = None, variant: str = "medium") -> dict | None:
    """{path, sha256} of the weights that run, or None for a detector with none of ours
    (stub, cuda, tensorrt). CoreML: the .mlpackage's Data/ tree (spec + weights; the
    Manifest only maps item UUIDs to those paths). RF-DETR: `weights`, else the variant's
    pretrain .pth rfdetr resolves. sha256 None when the file is absent."""
    import hashlib  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    h = hashlib.sha256()
    if name == "coreml":
        root = Path(weights or COREML_MODEL)
        if not root.exists():
            return {"path": str(root), "sha256": None}
        for f in sorted(x for x in (root / "Data").rglob("*") if x.is_file()):
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
        return {"path": str(root), "sha256": h.hexdigest()}
    if name not in ("rfdetr", "rfdetr-whole"):
        return None
    if weights:
        path = Path(weights)
    else:
        from rfdetr import config as rfconfig  # noqa: PLC0415
        path = (Path(os.environ.get("RF_HOME", "~/.roboflow/models")).expanduser()
                / getattr(rfconfig, f"RFDETR{variant.capitalize()}Config")().pretrain_weights)
    if not path.exists():
        return {"path": str(path), "sha256": None}
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return {"path": str(path), "sha256": h.hexdigest()}


if __name__ == "__main__":
    # Grid geometry: 4096x1080 at tile 1024 is 5 columns and ONE row - 1080 is
    # only 5% over the tile, and a second row would double the bill to re-see
    # the same pixels.
    xs, tw = _spans(4096, 1024, 0.25)
    ys, th = _spans(1080, 1024, 0.25)
    assert (len(xs), tw) == (5, 1024), (xs, tw)
    assert (ys, th) == ([0], 1080), (ys, th)
    assert xs[0] == 0 and xs[-1] == 4096 - 1024, xs
    assert min(b - a for a, b in zip(xs, xs[1:])) <= 1024 * (1 - 0.25) + 1, xs  # overlap held
    assert _spans(4096, 576, 0.25)[0][0] == 0
    assert _spans(500, 1024, 0.25) == ([0], 500)  # frame smaller than a tile

    # Full frame coverage: no gaps between tiles, whatever the size.
    for W, H, t in ((4096, 1080, 1024), (4096, 1080, 576), (1920, 1080, 768), (640, 480, 1024)):
        sx, w_ = _spans(W, t, 0.25)
        assert sx[0] == 0 and sx[-1] + w_ == W, (W, t, sx, w_)
        assert all(b <= a + w_ for a, b in zip(sx, sx[1:])), (W, t, sx)

    # Resolution rounding - the bug that made `-r 1008` a guaranteed ValueError
    # after the weights had loaded.
    assert _round_to(1008, 32) == 1024 and _round_to(576, 32) == 576
    assert _round_to(1000, 32) == 992 and _round_to(1, 32) == 32

    # IoS, not IoU: a box cut in half by a tile edge has IoU 0.5 with the whole
    # box (survives NMS) and IoS 1.0 (does not).
    whole = (0.0, 0.0, 10.0, 100.0)
    half = (0.0, 0.0, 10.0, 50.0)
    s = _ios(np.array([whole, half]))
    assert s[0, 1] == 1.0 and s[1, 0] == 1.0, s
    assert _ios(np.array([whole, (50.0, 0.0, 60.0, 100.0)]))[0, 1] == 0.0

    # Merge keeps the whole box and drops the clipped duplicate, whichever is
    # more confident - feet come from the box bottom, so half a player is worse
    # than a less confident whole one.
    merged = _merge([{"xyxy": half, "conf": 0.9, "cls": "person", "_clip": True},
                     {"xyxy": whole, "conf": 0.6, "cls": "person", "_clip": False}], 0.55)
    assert len(merged) == 1 and merged[0]["xyxy"] == whole, merged
    assert "_clip" not in merged[0]
    # ...but two genuinely separate players are both kept.
    assert len(_merge([{"xyxy": whole, "conf": 0.9, "cls": "person", "_clip": False},
                       {"xyxy": (50.0, 0.0, 60.0, 100.0), "conf": 0.8, "cls": "person",
                        "_clip": False}], 0.55)) == 2

    # End to end without weights: a box detected in tile coordinates comes back
    # at the right FULL-FRAME coordinates, scaled by tile/input and offset by
    # the tile origin. Getting this wrong is the classic tiling bug and it looks
    # like a bad model, not like arithmetic.
    class _OneBox:
        """Reports a fixed box in tile coordinates, i.e. once per tile."""

        resolution = 64

        def detect(self, frame):
            return [{"xyxy": (20.0, 20.0, 30.0, 44.0), "conf": 0.9, "cls": "person"}]

    t = TiledDetector(_OneBox(), tile=128, overlap=OVERLAP)
    frame = np.zeros((256, 512, 3), np.uint8)
    assert len(t.grid(512, 256)) == 5 * 3, t.grid(512, 256)
    got = t.detect(frame)
    assert len(got) == 15, len(got)  # 15 distinct tiles, 15 distinct boxes
    x1, y1, x2, y2 = got[0]["xyxy"]
    assert (x1, y1, x2, y2) == (40.0, 40.0, 60.0, 88.0), got[0]  # tile 128 / input 64 = 2x

    # Geometry-driven tiling. Synthetic rig with a known focal and height, same
    # construction as sceneprior._demo, so the tiles can be checked against
    # arithmetic rather than against themselves.
    from pitch import Calibration, Pitch  # noqa: PLC0415

    W_, H_ = 1920, 1080
    C = np.array([52.5, -25.0, 12.0])
    fwd = np.array([0.0, 25.0, -12.0]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    R = np.column_stack([right, np.cross(fwd, right), fwd]).T
    K = np.array([[1400.0, 0, W_ / 2], [0, 1400.0, H_ / 2], [0, 0, 1.0]])
    P = K @ np.column_stack([R, -R @ C])
    G_true = P[:, [0, 1, 3]]
    cal_ = Calibration(pitch=Pitch(105.0, 68.0), H=np.linalg.inv(G_true / G_true[2, 2]))
    import sceneprior  # noqa: PLC0415

    cam_ = sceneprior.camera_from_homography(cal_.H, (W_, H_))
    plan = geometry_tiles(cam_, cal_, (W_, H_), res=768, target_px=64)
    assert plan, plan
    assert all(0 <= x and 0 <= y and x + tw <= W_ and y + th <= H_ for x, y, tw, th in plan), plan
    # Square crops: the model stretches its input to a square, so a non-square
    # crop is anisotropic magnification and turns a player into a smear.
    assert all(tw == th for _, _, tw, th in plan), plan
    # The point of the whole function: far tiles are SMALLER than near ones, so
    # the far player gets more magnification than the near one.
    far = min(t[3] for t in plan)
    near = max(t[3] for t in plan)
    assert far < near, plan
    # ...and restricting to the far field is both a subset in size and cheaper.
    # On this rig y=68 is the FAR touchline and players there are 29 px; 60 px
    # cuts the plan at the halfway line.
    far_only = geometry_tiles(cam_, cal_, (W_, H_), res=768, target_px=64, max_h_px=60)
    assert far_only and len(far_only) < len(plan), (len(far_only), len(plan))
    assert max(t[3] for t in far_only) <= near, (far_only, near)

    # A plan overrides the uniform grid, tile size and all.
    t2 = TiledDetector(_OneBox(), tile=128, plan=[(0, 0, 64, 64), (100, 50, 32, 32)])
    assert t2.grid(512, 256) == [(0, 0, 64, 64), (100, 50, 32, 32)]
    assert len(t2.detect(np.zeros((256, 512, 3), np.uint8))) == 2

    # Tiling a detector with no detect_batch falls back to a loop, not a crash.
    assert TiledDetector(StubDetector(), tile=256).detect(np.zeros((256, 512, 3), np.uint8))

    assert isinstance(get_detector("stub"), StubDetector)

    # --- where the CoreML halves are looked for, in BOTH layouts.
    # The trap this is written against: os.path.exists() on a path you just
    # built is True in a dev checkout and False in the app, so a check that
    # only asserts "the resolver returned a string" passes forever. Here the
    # unfrozen answer is asserted empty, the frozen answer is asserted to be
    # the real bundle directory, and its CONTENTS are reported, not asserted -
    # a checkout that has never run build-backend.sh has no bundle to check.
    assert not getattr(sys, "frozen", False), "a frozen interpreter cannot self-check"
    assert _bundled("x") == "", _bundled("x")
    _exe, _froz = sys.executable, getattr(sys, "frozen", False)
    try:
        sys.frozen, sys.executable = True, "/tmp/nowhere/backend/futtraffic-api"
        assert _bundled("m.mlpackage") == "/tmp/nowhere/backend/coreml/m.mlpackage", \
            _bundled("m.mlpackage")
        assert not os.path.exists(_bundled("m.mlpackage")), "a resolved path is not a file"
        # The ACTUAL bundle, if this checkout has one. os.path.dirname of the
        # real frozen binary is the anchor the app will use.
        real = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "app", "desktop", "resources", "backend", "futtraffic-api")
        if os.path.exists(real):
            sys.executable = real
            have = {n: os.path.exists(_bundled(n))
                    for n in (COREML_WORKER_EXE, "rfdetr-medium-576.mlpackage")}
            print(f"detectors.py: bundle at {os.path.dirname(real)} -> {have}"
                  + ("" if all(have.values()) else
                     "  <- packaged installs will run rfdetr, not coreml"))
        else:
            print("detectors.py: no app/desktop/resources/backend in this checkout; "
                  "bundle layout unchecked")
    finally:
        sys.executable = _exe
        if not _froz:
            del sys.frozen

    # CoreML: it must REFUSE rather than fall back. A silent fallback turns a
    # "2.4x" run into a secretly 1x one, which is the worst outcome available.
    try:
        CoreMLDetector(weights="/nonexistent/model.mlpackage")
    except SystemExit as e:
        assert "not found" in str(e), e
    else:
        raise AssertionError("a missing .mlpackage must raise, not fall back")

    if not (os.path.exists(COREML_MODEL) and os.path.exists(COREML_PYTHON)):
        print(f"detectors.py: CoreML path skipped (no {COREML_MODEL} or {COREML_PYTHON})")
    else:
        try:
            CoreMLDetector(python="/nonexistent/python3.12")
        except SystemExit as e:
            assert "no CoreML worker" in str(e), e
        else:
            raise AssertionError("a missing 3.12 interpreter must raise, not fall back")

        # conf 0 so every reply carries boxes: the failure this guards against is
        # a worker that returns stale or empty results while the clock still
        # reads 2.4x, and an assert on an empty list cannot catch it.
        cm = get_detector("coreml", conf=0.0)
        assert isinstance(cm, TiledDetector) and cm.size == cm.inner.resolution == 576, cm.size
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (576, 576, 3), dtype=np.uint8)
        black = np.zeros((576, 576, 3), np.uint8)
        a, b = cm.inner.detect_batch([noise, black])
        assert a and b, (len(a), len(b))                    # the model ran
        assert a != b, "noise and black decoded identically - the worker is not reading the tiles"
        assert cm.inner.detect(noise) == a, "the same tile decoded differently twice"
        # Boxes are the model's, not clipped: at conf 0 the junk queries hang off
        # the edges. What must hold is that they are boxes at all.
        assert all(d["xyxy"][0] < d["xyxy"][2] and d["xyxy"][1] < d["xyxy"][3]
                   and -576 < min(d["xyxy"]) and max(d["xyxy"]) < 2 * 576 for d in a), a[:3]
        assert all(0.0 <= d["conf"] <= 1.0 and d["cls"] in PLAYER_CLASSES for d in a), a[:3]

        # ...and through the tiling, in FULL-FRAME coordinates on a band-shaped
        # frame. 3 tiles, so the seq the worker echoes has to advance too.
        band = rng.integers(0, 256, (870, 2500, 3), dtype=np.uint8)
        seq = cm.inner._seq
        got = cm.detect(band)
        assert got and cm.inner._seq > seq, (len(got), seq, cm.inner._seq)
        # The tile offsets were applied: 2500 px is 3 tiles, and boxes land past
        # the first one. (Unclipped model boxes overhang the frame, so a
        # containment assert here would be an assert about clipping.)
        assert max(d["xyxy"][2] for d in got) > 1024, max(d["xyxy"][2] for d in got)
        assert min(d["xyxy"][0] for d in got) < 1024
        pid = cm.inner.proc.pid
        cm.inner.close(); cm.inner.close()                  # idempotent
        assert cm.inner.proc.poll() is not None, pid
        print("detectors.py: CoreML self-check OK (worker started, decoded, and reaped)")

    try:
        __import__("rfdetr")
    except ImportError:
        print("detectors.py self-check OK (rfdetr not installed, model path skipped)")
    else:
        # The bug that started this: `-r 1008` used to load ~200MB of weights
        # and THEN raise ValueError, minutes into a run. Nano and fp32 keep this
        # to ~30s; it is the only part of the self-check that needs a GPU-ish
        # machine, and it is the part worth the wait.
        d = get_detector("rfdetr", variant="nano", resolution=1008, conf=0.5, tile=1024,
                         half=False)  # tracing is a slow one-off; not what this checks
        assert d.inner.resolution == 1024, d.inner.resolution
        assert d.size == 1024
        out = d.detect(np.zeros((1080, 4096, 3), np.uint8))
        assert out == [], out  # a black frame has no players, but it must not raise
        assert len(d.grid(4096, 1080)) == 5
        for variant in ("xl", "2xl"):
            try:
                RFDETRDetector(variant=variant)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"{variant} is PML 1.0 and must be refused")
        print("detectors.py self-check OK (including RF-DETR at -r 1008)")
