"""RF-DETR on NVIDIA: plain CUDA PyTorch, and TensorRT. Opt-in, never default.

NOTHING IN THIS FILE HAS BEEN RUN ON AN NVIDIA GPU. It was written on an M3 Mac
with no CUDA device present. Every number below is either MEASURED ON THIS MAC
(and labelled so) or PUBLISHED BY SOMEBODY ELSE (and cited). There are no
measured NVIDIA timings here and there must not be any until the PC has run
`check` and printed its own.

WHY IT IS WORTH BUILDING, and the honest provenance of each number:

  MEASURED HERE, base M3 (10 GPU cores, 16 GB), RF-DETR medium, 576x576, batch
  1, median of 15 (see detectors.py, which is where these live):
      PyTorch MPS fp32              68.5 ms/tile
      PyTorch MPS fp16              61.0 ms/tile
      CoreML fp16, Neural Engine    25.6-28.1 ms/tile
  and the pipeline as a whole runs at 1.18 frames/s, ~17 h per half, because
  detection is 92-98% of the wall clock.

  PUBLISHED BY ROBOFLOW, not measured by anyone here: RF-DETR medium at 4.52
  ms/image on an NVIDIA T4 with TensorRT 10.4, fp16, batch 1. The T4 is Turing,
  the same generation as the RTX 2060 this is aimed at. A 2060 is not a T4 and
  this figure is not a prediction for it; it is the reason to try.

THE PART THAT IS NOT THE GPU, flagged because it is the likeliest way this
disappoints. Per frame the CPU must decode 4096x1080, remap it to the
11878x870 pitch band (roughly 30 MB of output), cut 16 tiles and resize each to
576. On this Mac that work is invisible because detection costs 1000 ms/frame.
If TensorRT takes detection to ~80 ms/frame the same CPU work is suddenly the
same order of magnitude, and an i7 is not obviously faster at it than an M3.
THIS HAS NOT BEEN MEASURED, here or anywhere. `check --timing` prints the two
halves separately on the PC so it can be measured rather than argued about. If
the CPU side does dominate, the mitigations are NVDEC decode (PyNvVideoCodec or
FFmpeg with -hwaccel cuda) and doing cv2.remap on the GPU (cv2.cuda.remap, or a
torch grid_sample); neither is implemented here.

TWO DETECTORS, both satisfying detectors.Detector - detect(frame) -> [{"xyxy",
"conf", "cls"}] and detect_batch - so TiledDetector wraps either unchanged:

  CUDADetector   RFDETRDetector with device="cuda". No engine to build, no
                 TensorRT install, works the moment torch sees the GPU. This is
                 the fallback and the thing to try first.
  TensorRTDetector  A prebuilt .trt engine, fed from torch CUDA tensors. Faster,
                 and version- and GPU-locked: the engine MUST be built on the
                 machine that runs it.

NO SILENT FALLBACK, and this is the whole reason the file is shaped like this.
A missing CUDA device, a missing tensorrt module or a missing engine raises with
the exact package to install. The failure that must never happen is a quiet
drop to CPU, which produces a "12x" run that is secretly 0.3x and looks like a
success for six hours.

LICENSING. RF-DETR is Apache-2.0 (nano/small/medium/large only - XL/2XL are PML
1.0 and detectors.py refuses them). TensorRT is proprietary NVIDIA, installed by
the user on their own machine; nothing from it is vendored or redistributed
here, and this file only imports it. No AGPL anywhere.

RUNNING IT, on the PC, in order - see SETUP below for the full instructions:

    python model/engine/nvidia.py check --detector=cuda        # no engine needed
    python model/engine/nvidia.py build                        # ~5-20 min, once
    python model/engine/nvidia.py check --detector=tensorrt

and on any machine, including this Mac with no GPU at all:

    python model/engine/nvidia.py                              # self-check
    python model/engine/nvidia.py check --detector=torch       # checks the FIXTURE

THE ONE END-TO-END NUMBER THAT EXISTS, measured on this Mac 2026-08-18 with
`check --detector=torch`: PyTorch MPS **fp16** against the fixture's MPS **fp32**
reference scores recall 1.000, precision 1.000, median IoU 0.990, median score
delta 0.0006, with the rotated null control at 0.018. That is a real fp16-vs-fp32
run of this exact check, on the wrong vendor's silicon. It says the bar below is
reachable and the null discriminates; it says NOTHING about NVIDIA.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:                    # run as a script, import flat siblings
    sys.path.insert(0, _HERE)

SIZE = 576                  # RF-DETR medium's default input square - what ships
TILE = 1024                 # production tiling on the 11878x870 band, 16 tiles
# Imported, not restated. These two factories hardcoded 0.25 while the CoreML and
# rfdetr paths shipped 0.10 - a real behavioural split, under a docstring saying
# "Same tiling as detectors._rfdetr". A constant copied is a constant that drifts.
from detectors import OVERLAP  # noqa: E402
FIXTURE = os.path.join(_ROOT, "validation", "nvidia_fixture.npz")
# Same formula build() renames to, so the two cannot drift apart. The input
# square is IN THE NAME on purpose: engines are silent about what they were
# built for until TensorRTDetector's shape guard fires.
ENGINE = os.environ.get("FT_TRT_ENGINE",
                        os.path.join(_ROOT, "models", f"rfdetr-medium-{SIZE}.trt"))

# ImageNet stats, i.e. rfdetr's infer_transforms: ToImage + ToDtype(scale) +
# Resize (a no-op here, the tiles already arrive at the model's square) +
# Normalize. _selfcheck asserts _pre against the real transform rather than
# against this comment - getting it wrong is silent and costs half the recall.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# THE PASS BAR. fp16 does not reproduce fp32 bit-exactly and demanding that it
# does would be a check nobody can pass. The shape of a sane tolerance is
# MEASURED ON THIS MAC, CoreML fp16 on the ANE against PyTorch MPS over 112 real
# frames: recall 0.990, precision 0.995, median IoU 0.98, median score delta
# 0.0035 (detectors.py). The bars below sit slightly under that because a
# different vendor's fp16 has not been observed here.
PASS_R = 0.97
PASS_P = 0.97
PASS_IOU = 0.90             # median IoU over matched pairs
PASS_NULL = 0.25            # the shuffled control must stay far below PASS_R


# --------------------------------------------------------------------- guards
def _require_cuda():
    """The torch module, with a CUDA device actually present. Raises otherwise."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "torch is not installed.\n"
            "  This path needs a CUDA build of PyTorch, and there is NO torch wheel for\n"
            "  Python 3.14 - use Python 3.12. On the PC:\n"
            "    py -3.12 -m venv .venv-cuda\n"
            "    .venv-cuda\\Scripts\\pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "    .venv-cuda\\Scripts\\pip install rfdetr opencv-python") from e
    if not torch.cuda.is_available():
        raise SystemExit(
            f"torch {torch.__version__} is installed but sees no CUDA device.\n"
            f"  torch.version.cuda = {torch.version.cuda!r} "
            f"(None means this is a CPU-only wheel).\n"
            "  Either the NVIDIA driver is missing/too old, or pip installed the CPU wheel.\n"
            "  Reinstall with:  pip install --force-reinstall torch --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "  Check the driver with:  nvidia-smi\n"
            "  Refusing to run on the CPU: a silent CPU fallback here is ~30x SLOWER than\n"
            "  the Mac this is meant to beat, and it looks like a successful run.")
    return torch


def _require_tensorrt():
    """The tensorrt module. Raises with the install line if it is absent."""
    try:
        import tensorrt as trt  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "tensorrt is not installed.\n"
            "  pip install tensorrt          (NVIDIA's own wheel, proprietary, local only)\n"
            "  and for building engines from rfdetr:  pip install 'rfdetr[tensorrt]'\n"
            "  Python 3.12 - there are no TensorRT wheels for 3.14.\n"
            "  If you only want the GPU without the engine build, use --detector cuda.") from e
    return trt


# ------------------------------------------------------------------ detectors
class CUDADetector:
    """RFDETRDetector pinned to CUDA. The fallback: no engine, no TensorRT.

    Deliberately a thin wrapper rather than a reimplementation - the tracing,
    the fp16 fallback, the BGR->RGB flip and the class filtering are all already
    right in detectors.RFDETRDetector and are not worth a second copy. What this
    adds is the guard: it refuses to exist without a CUDA device, and it asserts
    afterwards that the model really landed on one.

    `half=True` traces at fp16, which is what makes a Turing card worth having
    (its tensor cores are fp16). Pass half=False for a fp32 run, which is the
    tighter comparison against the Mac's fp32 reference.
    """

    def __init__(self, conf: float = 0.5, variant: str = "medium",
                 weights: str | None = None, resolution: int | None = None,
                 half: bool = True, ball: bool = False):
        torch = _require_cuda()
        from detectors import OVERLAP, RFDETRDetector  # noqa: PLC0415

        self.inner = RFDETRDetector(weights=weights, conf=conf, variant=variant,
                                    resolution=resolution, device="cuda", half=half,
                                    ball=ball)
        dev = str(getattr(self.inner.model.model, "device", "?"))
        if "cuda" not in dev.lower():
            raise SystemExit(
                f"rfdetr was asked for device='cuda' and reports device={dev!r}.\n"
                "  Refusing to continue - see the module docstring on silent fallbacks.")
        self.resolution = self.inner.resolution
        self.conf = conf
        self.device_name = torch.cuda.get_device_name(0)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        return self.inner.detect_batch(frames)

    def detect(self, frame: np.ndarray) -> list[dict]:
        return self.inner.detect_batch([frame])[0]


class TensorRTDetector:
    """A prebuilt RF-DETR .trt engine, fed and drained through torch CUDA tensors.

    NO pycuda, NO cuda-python, NO `inference_models`. torch is already a hard
    dependency and a torch CUDA tensor is a device allocation with a .data_ptr(),
    which is exactly what TensorRT's set_tensor_address wants. That removes an
    entire dependency and an entire class of buffer-lifetime bug for about
    fifteen lines.

    THE ENGINE IS NOT PORTABLE. TensorRT serializes for one GPU architecture and
    one TensorRT version; an engine built elsewhere either refuses to
    deserialize or - worse - is silently rebuilt/rejected. So the repo ships no
    engine and `build()` runs on the target machine. See SETUP.

    BATCH. The engine's own input shape decides it. A fixed-batch engine
    (rfdetr's default, batch_size=1) is used at that batch and short chunks are
    padded; a dynamic-batch engine is queried for its optimization profile's max.
    This is read off the engine rather than assumed, because guessing wrong is a
    shape error thirty seconds into a seventeen-hour run.

    Preprocessing runs ON THE GPU: the uint8 BGR tiles are uploaded as-is and
    the channel flip, the /255 and the ImageNet normalize happen in torch on the
    device. That is both the fast way and the one that keeps the i7 out of the
    inner loop (see the module docstring on the CPU-side risk).

    HOW MUCH OF THIS IS ACTUALLY VERIFIED, since nothing here has met a GPU.
    Everything except TensorRT itself has been run on this Mac against the
    fixture: _pre's arithmetic asserted against rfdetr's infer_transforms
    (_selfcheck), the decode asserted against rfdetr's own PostProcess
    (_selfcheck), and the two of them together - _pre, then RF-DETR's forward,
    then post_process + the clamp + the class filter, i.e. this class with the
    engine swapped for PyTorch - MEASURED 2026-08-18 on MPS fp32 over the 14
    fixture tiles at recall 1.000, precision 1.000, median IoU 0.9995 at both
    conf 0.5 and 0.25. So a disagreement on the PC is the ENGINE, the fp16, or
    the four things below, and is not the decode.

    WHAT IS STILL UNVERIFIED, and can only be: deserialize, the I/O-name and
    shape discovery, set_tensor_address against torch's data_ptr, and
    execute_async_v3 with its stream. Those are the four calls `check
    --detector=tensorrt` exists to exercise.
    """

    def __init__(self, engine: str | None = None, conf: float = 0.5,
                 ball: bool = False, size: int = SIZE):
        torch = _require_cuda()
        trt = _require_tensorrt()
        path = engine or ENGINE
        if not os.path.exists(path):
            raise SystemExit(
                f"TensorRT engine not found: {path}\n"
                "  Engines are per-GPU and per-TensorRT-version, so this repo ships none.\n"
                "  Build one ON THIS MACHINE:\n"
                f"    python model/engine/nvidia.py build --out={os.path.dirname(path) or '.'}\n"
                "  or set FT_TRT_ENGINE to an engine you built earlier.")

        from detectors import BALL_CLASSES, PLAYER_CLASSES  # noqa: PLC0415
        from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: PLC0415
        from rfdetr.export.benchmark import post_process  # noqa: PLC0415

        self.torch, self.post, self.names = torch, post_process, COCO_CLASSES
        self.keep = PLAYER_CLASSES | (BALL_CLASSES if ball else frozenset())
        self.conf, self.resolution = conf, size
        self.device_name = torch.cuda.get_device_name(0)

        # BOTH of these are kept on self on purpose. TensorRT's C++ objects are
        # owned in a chain - Logger < Runtime < ICudaEngine < IExecutionContext -
        # and destroying a parent while a child is alive is undefined. As locals
        # they are collected the moment __init__ returns, and the engine then
        # runs on a dangling runtime and logs through a dangling logger: a
        # segfault that typically arrives on the first WARNING the engine emits,
        # i.e. hours into a run rather than in a test. UNVERIFIED without a GPU,
        # but it is what NVIDIA's own samples do and it costs two attributes.
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as fh:
            blob = fh.read()
        self.runtime = trt.Runtime(self.logger)
        eng = self.runtime.deserialize_cuda_engine(blob)
        if eng is None:
            raise SystemExit(
                f"TensorRT {trt.__version__} could not deserialize {path}.\n"
                "  An engine is locked to the GPU architecture and TensorRT version that\n"
                "  built it. Rebuild it here:  python model/engine/nvidia.py build")
        if not hasattr(eng, "num_io_tensors"):
            raise SystemExit(
                f"TensorRT {trt.__version__} is too old: this uses the name-based I/O API\n"
                "  (num_io_tensors / set_tensor_address / execute_async_v3), which needs\n"
                "  TensorRT >= 8.5.  pip install -U tensorrt")
        self.engine, self.ctx = eng, eng.create_execution_context()
        if self.ctx is None:
            raise SystemExit(
                f"TensorRT could not create an execution context for {path}.\n"
                "  Almost always GPU memory: the engine deserialized but its activation\n"
                "  workspace does not fit. Check free VRAM with  nvidia-smi , close other\n"
                "  CUDA processes, or rebuild at a smaller batch:\n"
                "    python model/engine/nvidia.py build --batch=1")

        # I/O tensors by name. rfdetr's ONNX export names them input/dets/labels
        # (rfdetr/detr.py: output_names = ["dets", "labels"]), but the mapping is
        # confirmed by SHAPE too: both outputs are float (N,300,k) and swapping
        # labels for dets is silent - see coreml_worker.py, where that bug was
        # found the first time.
        ins, outs = [], []
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            (ins if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT else outs).append(n)
        if len(ins) != 1 or len(outs) < 2:
            raise SystemExit(f"engine has inputs {ins} and outputs {outs}; expected one input "
                             "and at least dets+labels. Was it exported with backbone_only?")
        self.in_name = ins[0]
        shp = tuple(eng.get_tensor_shape(self.in_name))
        if len(shp) != 4 or shp[1] != 3:
            raise SystemExit(f"engine input {self.in_name} has shape {shp}; expected (N,3,S,S)")
        if shp[2] != size or shp[3] != size:
            raise SystemExit(
                f"engine input is {shp[2]}x{shp[3]} but this pipeline feeds {size}x{size} tiles.\n"
                f"  Rebuild with:  python model/engine/nvidia.py build --size={size}")
        if shp[0] == -1:                       # dynamic batch: take the profile's max
            self.batch = int(eng.get_tensor_profile_shape(self.in_name, 0)[2][0])
            # Only for a dynamic input: TensorRT requires set_input_shape there,
            # and on a fully static engine some versions reject it outright.
            self.ctx.set_input_shape(self.in_name, (self.batch, 3, size, size))
        else:
            self.batch = int(shp[0])

        def _dtype(name):
            return torch.from_numpy(np.zeros(1, trt.nptype(eng.get_tensor_dtype(name)))).dtype

        self.buf = {self.in_name: torch.empty((self.batch, 3, size, size),
                                              dtype=_dtype(self.in_name), device="cuda")}
        self.out = {}
        for n in outs:
            s = tuple(int(v) for v in self.ctx.get_tensor_shape(n))
            if any(v < 0 for v in s):
                # -1 means the context still has an unresolved shape, which can
                # only happen if the engine has a dynamic dimension other than
                # batch. torch.empty would raise something incomprehensible.
                raise SystemExit(
                    f"engine output {n} still has an unresolved shape {s} after the input was "
                    "fixed; this code only handles a dynamic BATCH.\n"
                    f"  Rebuild with a static shape:  python model/engine/nvidia.py build "
                    f"--size={size}")
            self.buf[n] = torch.empty(s, dtype=_dtype(n), device="cuda")
            if len(s) == 3 and s[2] == 4:
                self.out["dets"] = n
            elif len(s) == 3 and s[2] > 4:
                self.out["labels"] = n
        by_name = {"dets", "labels"} <= set(outs)
        if set(self.out) != {"dets", "labels"}:
            raise SystemExit(f"could not identify dets/labels among {outs} with shapes "
                             f"{[tuple(self.buf[n].shape) for n in outs]}")
        if by_name and (self.out["dets"] != "dets" or self.out["labels"] != "labels"):
            raise SystemExit(f"engine output names {outs} disagree with their shapes "
                             f"({self.out}); refusing to guess which is which")
        # self.buf is what keeps these allocations alive for as long as the
        # engine holds their addresses. If it were ever rebuilt per call, torch's
        # caching allocator would hand the memory to something else while
        # TensorRT still pointed at it - a use-after-free that reads as
        # plausible-but-wrong boxes rather than as a crash.
        for n, t in self.buf.items():
            if not self.ctx.set_tensor_address(n, int(t.data_ptr())):
                raise SystemExit(f"TensorRT rejected the device address for tensor {n!r}. "
                                 "The engine and this code disagree about its I/O names.")
        # How many query/class pairs post_process should select. rfdetr's
        # PostProcess uses args.num_select, which equals num_queries for every
        # variant detectors.py allows; the engine's own query dimension is that
        # number, and reading it beats the library default (300) silently being
        # right for medium and wrong for something else.
        self.nq = int(self.buf[self.out["labels"]].shape[1])
        self.tgt = torch.tensor([[size, size]] * self.batch, dtype=torch.float32, device="cuda")
        self._mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        self.gpu_s = 0.0        # inference-only seconds, so the CPU side can be subtracted

    def _pre(self, frames: list[np.ndarray]):
        """(n,S,S,3) uint8 BGR on the host -> (n,3,S,S) normalized on the device.

        Exactly rfdetr's infer_transforms; _selfcheck asserts that against the
        real transform on CPU, which is the only part of this class that can be
        verified without a GPU and is also the part most likely to be wrong.
        """
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to("cuda", non_blocking=True)
        x = x[..., [2, 1, 0]].permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
        return x.sub_(self._mean).div_(self._std)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        torch = self.torch
        s = self.resolution
        for f in frames:
            if f.shape != (s, s, 3) or f.dtype != np.uint8:
                raise SystemExit(f"TensorRT path takes {s}x{s}x3 uint8 tiles, got "
                                 f"{f.shape} {f.dtype}")
        rows: list[list[dict]] = []
        for i in range(0, len(frames), self.batch):
            chunk = frames[i:i + self.batch]
            n = len(chunk)
            # A TensorRT engine accepts exactly the batch it was built for. Pad
            # the short chunk and drop the padding's results - same trick as
            # RFDETRDetector's traced path.
            pad = chunk + [chunk[-1]] * (self.batch - n)
            self.buf[self.in_name].copy_(self._pre(pad))   # copy_ converts dtype
            # torch's CURRENT stream, read HERE and not cached in __init__: the
            # H2D copy above went to whatever stream is current now, and enqueuing
            # the kernel anywhere else is the unsynchronised race that shows up as
            # boxes from the previous tile. Same stream, and the device-wide
            # synchronize below also puts the upload strictly before the kernel.
            stream = torch.cuda.current_stream()
            # Synchronize BEFORE the clock so gpu_s is the engine and not the
            # upload and preprocess queued ahead of it. One extra sync per chunk
            # of 16 tiles, against ~80 ms of work, and it is what makes the
            # "inference vs host" split in check() an honest number.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ok = self.ctx.execute_async_v3(stream.cuda_stream)
            # MUST be before the outputs are read: execute_async_v3 only ENQUEUES.
            # Without this the buffers hold the previous chunk's numbers, which
            # look entirely plausible and are wrong.
            stream.synchronize()
            self.gpu_s += time.perf_counter() - t0
            if not ok:
                raise SystemExit("TensorRT execute_async_v3 returned False; the engine did "
                                 "not run. Rebuild it:  python model/engine/nvidia.py build")
            res = self.post({"labels": self.buf[self.out["labels"]].float(),
                             "dets": self.buf[self.out["dets"]].float()},
                            self.tgt, num_queries=self.nq)
            for r in res[:n]:
                # PARITY. rfdetr's own PostProcess (model/models/postprocess.py)
                # clamps boxes into the image; the benchmark post_process this
                # file imports does NOT, and the fixture's reference came
                # through the clamping one. MEASURED ON THIS MAC over the 14
                # fixture tiles: at conf 0.25 not one box overhangs the square
                # by more than 0.1 px, so the clamp moves median IoU by 0.0001
                # and changes no verdict. It is here because it is free and
                # because TiledDetector multiplies these coordinates by
                # tile/576 and adds the tile origin - an unclamped box lands
                # inside the NEIGHBOURING tile's territory, where _merge and
                # the _clip edge test both see something the reference path
                # would never have produced. UNVERIFIED on real overhang: the
                # 22 boxes that do sit on a boundary here are all under conf
                # 0.25, so how far a 2060's fp16 pushes them is unknown.
                r["boxes"].clamp_(0.0, float(s))
                # One device->host transfer for the whole tile. Iterating CUDA
                # tensors element by element costs a sync per number, which at
                # 16 tiles a frame is the kind of overhead that eats the win.
                keep = r["scores"] >= self.conf
                bx = r["boxes"][keep].cpu().numpy()
                sc = r["scores"][keep].cpu().numpy()
                cl = r["labels"][keep].cpu().numpy()
                names = [self.names.get(int(c), str(int(c))) for c in cl]
                rows.append([{"xyxy": tuple(round(float(v), 2) for v in b),
                              "conf": round(float(q), 4), "cls": nm}
                             for b, q, nm in zip(bx, sc, names) if nm in self.keep])
        return rows

    def detect(self, frame: np.ndarray) -> list[dict]:
        return self.detect_batch([frame])[0]


# --------------------------------------------------------------- tiled wiring
def _cuda(tile: int | None = TILE, overlap: float = OVERLAP, batch: int = 8, **kw):
    """Same tiling as detectors._rfdetr, on CUDA."""
    from detectors import TiledDetector  # noqa: PLC0415

    inner = CUDADetector(**kw)
    return TiledDetector(inner, tile=tile, overlap=overlap, batch=batch) if tile else inner


def _tensorrt(tile: int | None = TILE, overlap: float = OVERLAP, batch: int = 16, **kw):
    """Same tiling, on a .trt engine. batch here is tiles per detect_batch call;
    the engine's own batch decides how they are actually chunked."""
    from detectors import TiledDetector  # noqa: PLC0415

    inner = TensorRTDetector(**kw)
    return TiledDetector(inner, tile=tile, overlap=overlap, batch=batch) if tile else inner


# ------------------------------------------------------------------- scoring
def _match(a: list[dict], b: list[dict], thr: float = 0.5):
    """Greedy highest-IoU-first pairing, exclusive on both sides. -> (pairs, ious)

    # ponytail: this is the same twenty lines as model/validation/coreml_check._match.
    # Duplicated on purpose - model/engine/ importing from model/validation/ inverts the
    # dependency, and a shared home for it is a refactor of two files this change
    # is not allowed to touch.
    """
    if not a or not b:
        return [], []
    A = np.array([d["xyxy"] for d in a], float)
    B = np.array([d["xyxy"] for d in b], float)
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda M: np.clip(M[:, 2] - M[:, 0], 0, None) * np.clip(M[:, 3] - M[:, 1], 0, None)  # noqa: E731
    iou = inter / np.maximum(ar(A)[:, None] + ar(B)[None, :] - inter, 1e-9)
    pairs, ious, ta, tb = [], [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape)):
        if iou[i, j] < thr:
            break
        if i in ta or j in tb:
            continue
        ta.add(int(i)); tb.add(int(j))
        pairs.append((int(i), int(j))); ious.append(float(iou[i, j]))
    return pairs, ious


def _agree(ref: list[list[dict]], got: list[list[dict]], thr: float = 0.5) -> dict:
    """Recall/precision of `got` against `ref`, tile by tile, plus IoU and score drift."""
    hit = nref = ngot = 0
    ious: list[float] = []
    dscore: list[float] = []
    misses: list[dict] = []
    for r, g in zip(ref, got):
        nref += len(r); ngot += len(g)
        pairs, iou = _match(r, g, thr)
        hit += len(pairs)
        ious += iou
        dscore += [abs(r[i]["conf"] - g[j]["conf"]) for i, j in pairs]
        misses += [r[i] for i in set(range(len(r))) - {i for i, _ in pairs}]
    return {"R": hit / max(1, nref), "P": hit / max(1, ngot), "hit": hit,
            "n_ref": nref, "n_got": ngot,
            "med_iou": float(np.median(ious)) if ious else 0.0,
            "med_dscore": float(np.median(dscore)) if dscore else float("nan"),
            "misses": misses}


def _sanity(got: list[list[dict]], ref: list[list[dict]], size: int) -> list[str]:
    """Is this even the right kind of output? -> list of complaints, empty is good.

    These exist because "0.000 agreement" is a useless diagnosis. A box in
    normalized coordinates, a cxcywh box read as xyxy, and a box computed at the
    wrong input square all score zero and all have different fixes.
    """
    bad = []
    flat = [b for t in got for b in t]
    if not flat:
        return ["every tile returned ZERO boxes - the model ran but decoded nothing, or the "
                "engine output was never read back. Check dets/labels are not swapped."]
    xy = np.array([b["xyxy"] for b in flat], float)
    if not np.isfinite(xy).all():
        bad.append("non-finite box coordinates (NaN/Inf) - fp16 overflow, or an uninitialized "
                   "output buffer.")
    if (xy[:, 2] <= xy[:, 0]).mean() > 0.05 or (xy[:, 3] <= xy[:, 1]).mean() > 0.05:
        bad.append("boxes with x2<=x1 or y2<=y1 - these look like cxcywh read as xyxy. "
                   "post_process must run box_cxcywh_to_xyxy before scaling.")
    if np.nanmax(np.abs(xy)) < 8.0:
        bad.append(f"every coordinate is under 8 px on a {size} px tile - the boxes are still "
                   "NORMALIZED 0..1. post_process must multiply by target_sizes.")
    if np.nanmax(xy) > size * 1.5 or np.nanmin(xy) < -size * 0.5:
        bad.append(f"coordinates far outside 0..{size} (min {np.nanmin(xy):.0f}, "
                   f"max {np.nanmax(xy):.0f}) - target_sizes is probably the wrong square.")
    ra = np.array([b["xyxy"] for t in ref for b in t], float)
    if len(ra):
        h_got = float(np.median(xy[:, 3] - xy[:, 1]))
        h_ref = float(np.median(ra[:, 3] - ra[:, 1]))
        if h_ref > 0 and not (0.2 < h_got / h_ref < 5.0):
            bad.append(f"median box height {h_got:.0f} px against the reference's {h_ref:.0f} px "
                       "- a scale factor, not a detection difference. Wrong input square, or "
                       "the tiles were resized before the detector saw them.")
    return bad


# --------------------------------------------------------------------- check
def load_fixture(path: str = FIXTURE):
    """-> (tiles, expected_at_conf_floor, meta). Verifies the tiles decode identically."""
    import hashlib  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    if not os.path.exists(path):
        raise SystemExit(f"fixture not found: {path}\n"
                         "  It is committed to the repo; if it is missing, regenerate it ON A\n"
                         "  MAC WITH THE FOOTAGE:  python model/engine/nvidia.py make-fixture")
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    tiles = [cv2.imdecode(z[f"t{i}"], cv2.IMREAD_COLOR) for i in range(meta["n"])]
    if any(t is None for t in tiles):
        raise SystemExit(f"{path}: a tile failed to decode. cv2 build without WebP support?")
    h = hashlib.sha256()
    for t in tiles:
        h.update(np.ascontiguousarray(t).tobytes())
    if h.hexdigest() != meta["sha256"]:
        print(f"  WARNING: decoded tiles hash {h.hexdigest()[:16]} but the fixture records "
              f"{meta['sha256'][:16]}.\n"
              "    The two machines are NOT seeing identical pixels, so any disagreement "
              "below is partly the image decoder, not the model.", flush=True)
    return tiles, [[{"xyxy": tuple(r[:4]), "conf": r[4], "cls": r[5]} for r in t]
                   for t in meta["expected"]], meta


def check(det, path: str = FIXTURE, confs=(0.5, 0.25), label: str = "detector",
          timing: bool = True, batch: int = 8) -> bool:
    """Run `det` over the fixture tiles and say whether it agrees with the Mac.

    WHAT THIS IS DEFENDED AGAINST, since a check written for hardware the author
    cannot run is the most dangerous kind:

      empty output      -> recall 0, and _sanity names it explicitly.
      stale/shifted     -> tiles come from FOUR different frames, so returning
                           tile k-1's boxes for tile k scores near the null. The
                           ROTATED control below measures exactly that, and is
                           printed every run whether it fails or not.
      wrong coordinates -> _sanity, before agreement is even computed.
      "the fixture's own boxes" -> the detector never reads the fixture; the only
                           way to get the expected boxes is to compute them. The
                           rotated control is also printed so a reader can see
                           the metric discriminating rather than take it on faith.
      silent CPU        -> CUDADetector/TensorRTDetector both refuse to build
                           without a CUDA device, and the timing line below would
                           read in the hundreds of ms.

    Returns True if it passed. The caller exits non-zero if not.
    """
    tiles, ref_all, meta = load_fixture(path)
    n = len(tiles)
    print(f"fixture {os.path.basename(path)}: {n} tiles of {meta['size']}px from "
          f"{len(set(f for f, _ in meta['tile_index']))} frames of {meta['source']['video']}")
    print(f"  reference: {meta['reference']['what']}, conf floor "
          f"{meta['reference']['conf_floor']}, generated {meta['reference']['generated']}")
    dev = getattr(det, "device_name", None)
    print(f"  detector:  {label}" + (f" on {dev}" if dev else ""))
    # A detector thresholded ABOVE the fixture's floor cannot be scored against
    # it: every reference box between the two numbers is a guaranteed miss, so
    # the conf 0.25 row reads as lost recall when it is only arithmetic. Refuse
    # rather than print a number that means nothing.
    floor, dc = meta["reference"]["conf_floor"], getattr(det, "conf", None)
    if dc is not None and dc > floor + 1e-9:
        raise SystemExit(
            f"this detector is thresholded at conf {dc} but the fixture's reference only goes "
            f"down to {floor}.\n  Every reference box below {dc} would count as a miss and the "
            f"recall printed here would be meaningless.\n  Re-run with --conf={floor} (the "
            "default); check() scores at 0.5 and 0.25 by itself.")

    # Warm up with the SAME batch as the timed run, not with one tile. Both
    # RFDETRDetector and a fixed-batch TensorRT engine LOCK to the batch of the
    # first call, so a one-tile warm-up makes the real call a shape error - which
    # is how this was found.
    def run(imgs):
        # Chunked at `batch`, exactly as TiledDetector feeds a real frame, so a
        # detector that only works at one batch size fails here and not in hour
        # three of a run. The first chunk is the largest, which is what the
        # batch-locked paths (traced torch, fixed-batch TensorRT) need.
        out = []
        for i in range(0, len(imgs), batch):
            out += det.detect_batch(imgs[i:i + batch])
        return out

    t0 = time.perf_counter()
    run(tiles)                                      # warm-up: trace/engine/alloc
    warm = time.perf_counter() - t0
    if hasattr(det, "gpu_s"):
        det.gpu_s = 0.0
    t0 = time.perf_counter()
    got_all = run(tiles)
    wall = time.perf_counter() - t0
    if len(got_all) != n:
        raise SystemExit(f"detect_batch returned {len(got_all)} results for {n} tiles - the "
                         "contract is one list per tile, in order.")
    if timing:
        gpu = getattr(det, "gpu_s", None)
        extra = ""
        if gpu:
            extra = (f"   (inference {gpu / n * 1000:6.2f} ms, host side "
                     f"{(wall - gpu) / n * 1000:5.2f} ms)")
        print(f"  timing:    {wall / n * 1000:6.2f} ms/tile over {n} tiles, "
              f"{warm * 1000:.0f} ms warm-up{extra}")
        print("             MEASURED ON THIS MACHINE, just now. For scale, measured on the "
              "Mac: 61.0 ms/tile (PyTorch MPS fp16), 25.6 ms/tile (CoreML/ANE).")
        print("             Roboflow PUBLISH 4.52 ms for medium on a T4 with TensorRT 10.4 "
              "fp16 batch 1.")
        print("             None of that is a pass criterion; the agreement below is.")
        if wall / n > 0.100:
            print("  WARNING: over 100 ms/tile. That is slower than this Mac's Neural Engine. "
                  "Check nvidia-smi shows the GPU busy during the run.")

    ok = True
    bad = _sanity(got_all, ref_all, meta["size"])
    if not bad:
        # LIVENESS. Run the same tiles MIRRORED. Anything that computes from
        # pixels detects the same players in a mirrored pitch (football is not
        # chiral) but puts the boxes somewhere else; anything replaying a stored
        # answer returns the identical boxes. This is the control for the one
        # failure the agreement score cannot see - a path that returns the
        # fixture's own expected boxes, or any other constant, would score
        # 1.000 above and is caught only here.
        got_f = run([np.ascontiguousarray(t[:, ::-1]) for t in tiles])
        n_f = sum(len(t) for t in got_f)
        n_g = sum(len(t) for t in got_all)
        same = _agree(got_all, got_f)["R"]
        print(f"\n  liveness:  mirrored tiles return {n_f} boxes against {n_g} upright, "
              f"{same:.3f} of them in the same place")
        if same > 0.5:
            ok = False
            print("    FAIL: the mirrored image produced the SAME boxes. This detector is not "
                  "looking at its input - it is replaying something.")
        elif not (0.4 < n_f / max(1, n_g) < 2.5):
            ok = False
            print(f"    FAIL: {n_f} boxes on mirrored tiles against {n_g} upright. A mirrored "
                  "pitch holds the same players; this is not a working detector.")
    if bad:
        ok = False
        print("\n  SANITY FAILED - the output is not the right kind of thing:")
        for b in bad:
            print(f"    * {b}")

    for conf in confs:
        ref = [[b for b in t if b["conf"] >= conf] for t in ref_all]
        got = [[b for b in t if b["conf"] >= conf] for t in got_all]
        st = _agree(ref, got)
        # THE NULL. Same boxes, the next tile's reference. The tiles come from
        # four frames minutes apart and different parts of the pitch, so a real
        # agreement and a bookkeeping accident are miles apart here. If this is
        # not far below the real number, nothing above it can be believed.
        null = _agree(ref[1:] + ref[:1], got)
        flips = sum(1 for m in st["misses"] if abs(m["conf"] - conf) < 0.05)
        print(f"\n  conf {conf}:  recall {st['R']:.3f}  precision {st['P']:.3f}   "
              f"({st['n_ref']} reference boxes, {st['n_got']} returned, {st['hit']} matched)")
        print(f"             median IoU {st['med_iou']:.3f}   median |score delta| "
              f"{st['med_dscore']:.4f}")
        print(f"             NULL (rotated reference): recall {null['R']:.3f}  "
              f"precision {null['P']:.3f}")
        if st["misses"]:
            print(f"             {len(st['misses'])} missed reference boxes, of which {flips} "
                  f"scored within 0.05 of the {conf} threshold (i.e. threshold flips, not "
                  "lost players)")
        if null["R"] > PASS_NULL:
            ok = False
            print(f"    FAIL: the null control scores {null['R']:.3f} (> {PASS_NULL}). This "
                  "metric is measuring box density, not agreement - ignore every number above.")
        if st["R"] < PASS_R or st["P"] < PASS_P:
            if flips and flips == len(st["misses"]) and st["R"] >= PASS_R - 0.03:
                print(f"    WARN: under the {PASS_R} bar, but every miss is a threshold flip. "
                      "fp16 moves scores by ~0.004 and conf 0.5 sits on a cliff for this "
                      "footage; see detectors.py. Look at the conf 0.25 row.")
            else:
                ok = False
                print(f"    FAIL: recall/precision under {PASS_R}/{PASS_P}.")
        if st["hit"] and st["med_iou"] < PASS_IOU:
            ok = False
            print(f"    FAIL: median IoU {st['med_iou']:.3f} < {PASS_IOU} - boxes are matching "
                  "but sitting in the wrong place.")

    print(f"\n  {'PASS' if ok else 'FAIL'}: {label} "
          f"{'agrees with' if ok else 'DOES NOT agree with'} the Mac reference.")
    return ok


# ------------------------------------------------------------------- fixture
class _Canned:
    """Replays canned results, but MIRRORS them when handed a mirrored tile.

    Test double for _selfcheck ONLY, never reachable from get_detector(). It
    mirrors because a stand-in that ignored its input would be indistinguishable
    from the cheat check()'s liveness control exists to catch - so the double has
    to behave like something that reads pixels, and _Cheat is the one that does
    not.
    """

    resolution = SIZE

    def __init__(self, rows, tiles=None):
        self.rows, self.tiles, self.i = rows, tiles, 0

    def detect_batch(self, frames):
        out = []
        for f in frames:
            k, self.i = self.i, (self.i + 1) % len(self.rows)   # check() calls in chunks
            r = self.rows[k]
            if self.tiles is not None and not np.array_equal(f, self.tiles[k]):
                S = self.resolution
                r = [{**d, "xyxy": (S - d["xyxy"][2], d["xyxy"][1],
                                    S - d["xyxy"][0], d["xyxy"][3])} for d in r]
            out.append(r)
        return out

    def detect(self, frame):
        return self.detect_batch([frame])[0]


class _Cheat(_Canned):
    """Returns the same answer whatever it is shown. The failure check() must
    catch and agreement alone cannot: it scores a perfect 1.000."""

    def detect_batch(self, frames):
        out = []
        for _ in frames:
            out.append(self.rows[self.i])
            self.i = (self.i + 1) % len(self.rows)
        return out


def make_fixture(out: str = FIXTURE, job: str = "data/jobs/118575_full_1st",
                 video: str = "data/footage/118575_panorama_1st_half.mp4",
                 frames=(3000, 20000, 40000, 60000), per_frame: int = 3,
                 quiet_tiles: int = 2, conf_floor: float = 0.10) -> None:
    """Build the portable fixture. RUN ON THE MAC, once; it needs the footage.

    Real frames, real production tiling, real detector. What is stored is the
    576x576 BGR tile EXACTLY as TiledDetector would hand it to a detector, WebP
    LOSSLESS (bit-exact by spec, unlike JPEG whose SIMD IDCT is only accurate to
    within 1 LSB and would differ between NEON and AVX2), plus every box the
    Mac's PyTorch path found down to `conf_floor` so the PC can score at any
    threshold without regenerating.

    TILE SELECTION is a rule, not a taste: the `per_frame` busiest tiles of each
    frame (busiest = most boxes at conf 0.5 - that is where the signal is), plus
    the `quiet_tiles` tiles that have NO boxes at 0.5 but the most below it -
    i.e. the grass most likely to produce a false positive, so precision is
    measured somewhere it can actually fail. On this footage the players cluster
    into 4-5 of the 16 tiles, so an even sample would be mostly empty grass.
    """
    import cv2  # noqa: PLC0415

    import calibrators  # noqa: PLC0415
    from detectors import RFDETRDetector, TiledDetector  # noqa: PLC0415

    cal = calibrators.load(os.path.join(_ROOT, job, "calibration.json"))
    h, w = cal._mapx.shape[:2]

    class _S:
        resolution = SIZE

    grid = TiledDetector(_S(), tile=TILE).grid(w, h)
    cap = cv2.VideoCapture(os.path.join(_ROOT, video))
    tiles, index = [], []
    try:
        for fi, f in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, raw = cap.read()
            if not ok:
                raise SystemExit(f"could not read frame {f} of {video}")
            band = cal.undistort_frame(raw)
            for ti, (x, y, tw, th) in enumerate(grid):
                tiles.append(cv2.resize(band[y:y + th, x:x + tw], (SIZE, SIZE),
                                        interpolation=cv2.INTER_AREA if tw >= SIZE
                                        else cv2.INTER_LINEAR))
                index.append((fi, ti))
    finally:
        cap.release()
    print(f"{len(frames)} frames of {w}x{h}, {len(grid)} tiles each -> {len(tiles)} candidates",
          flush=True)

    # fp32, not the traced fp16 the pipeline ships. The fixture is the REFERENCE:
    # it should be the least approximate thing this machine can produce, so that
    # a disagreement on the PC is the PC's fp16 and not two fp16 paths drifting
    # apart in different directions.
    det = RFDETRDetector(variant="medium", conf=conf_floor, half=False)
    res = []
    for i in range(0, len(tiles), 8):
        res += det.detect_batch(tiles[i:i + 8])
        print(f"  {min(i + 8, len(tiles))}/{len(tiles)}", flush=True)

    n5 = [sum(1 for b in t if b["conf"] >= 0.5) for t in res]
    pick: list[int] = []
    for fi in range(len(frames)):
        cand = [i for i in range(len(tiles)) if index[i][0] == fi]
        pick += sorted(cand, key=lambda i: -n5[i])[:per_frame]
    quiet = sorted((i for i in range(len(tiles)) if n5[i] == 0 and i not in pick),
                   key=lambda i: -len(res[i]))[:quiet_tiles]
    pick = sorted(pick + quiet)
    print(f"kept {len(pick)} tiles: {sum(n5[i] for i in pick)} boxes at conf 0.5, "
          f"{sum(len(res[i]) for i in pick)} at {conf_floor}", flush=True)

    import hashlib  # noqa: PLC0415
    import platform  # noqa: PLC0415

    blobs, hsh = {}, hashlib.sha256()
    for k, i in enumerate(pick):
        enc = cv2.imencode(".webp", tiles[i], [cv2.IMWRITE_WEBP_QUALITY, 101])[1]
        blobs[f"t{k}"] = enc
        # Hash the DECODED tile, not the source: what the PC must reproduce is
        # what comes back out of cv2.imdecode, and lossless is a claim to check.
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if not np.array_equal(dec, tiles[i]):
            raise SystemExit("WebP round trip is not lossless on this cv2 build; "
                             "the fixture would not be a reference.")
        hsh.update(np.ascontiguousarray(dec).tobytes())
    meta = {
        "n": len(pick), "size": SIZE, "sha256": hsh.hexdigest(),
        "source": {"video": os.path.basename(video), "job": job,
                   "frames": [int(frames[index[i][0]]) for i in pick],
                   "tile_rects": [list(grid[index[i][1]]) for i in pick],
                   "band_wh": [w, h], "tile_px": TILE},
        "tile_index": [list(index[i]) for i in pick],
        "reference": {
            "what": "detectors.RFDETRDetector(variant='medium', half=False) on PyTorch MPS fp32",
            "conf_floor": conf_floor, "resolution": det.resolution,
            "generated": time.strftime("%Y-%m-%d"),
            "machine": f"{platform.machine()} {platform.system()}",
        },
        "expected": [[[round(float(v), 2) for v in b["xyxy"]] + [round(b["conf"], 4), b["cls"]]
                      for b in res[i]] for i in pick],
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, meta=json.dumps(meta), **blobs)
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)


def build(out: str | None = None, variant: str = "medium", size: int = SIZE,
          batch: int = 1) -> None:
    """Export ONNX and build a TensorRT engine, ON THIS MACHINE. NVIDIA only.

    Roughly 5-20 minutes on a 2060 - almost all of it TensorRT's kernel autotune,
    which is why the engine is not portable and cannot be shipped. UNVERIFIED:
    that range is what NVIDIA's own docs describe for a model this size, not
    something measured here.
    """
    _require_cuda()
    _require_tensorrt()
    import rfdetr  # noqa: PLC0415

    outdir = out or os.path.join(_ROOT, "models")
    os.makedirs(outdir, exist_ok=True)
    model = getattr(rfdetr, f"RFDETR{variant.capitalize()}")()
    print(f"exporting {variant} at {size}x{size}, batch {batch} -> {outdir}", flush=True)
    # dynamic_batch is left OFF deliberately: polygraphy builds a default
    # optimization profile from the ONNX shape hints, and whether that profile
    # actually spans a batch range is not something that could be verified from a
    # Mac. A fixed-batch engine always works and TensorRTDetector pads short
    # chunks. If you want batch 4, pass --batch 4 and rebuild.
    path = str(model.export(output_dir=outdir, format="tensorrt", shape=(size, size),
                            batch_size=batch, dynamic_batch=False, fp16=True, verbose=True))
    # rfdetr names the artifact after the VARIANT only - "rfdetr-medium.trt"
    # (export/_onnx/exporter.py: f"{variant_name}.onnx", then _tensorrt swaps the
    # suffix). Two things are wrong with leaving it there: it is not the name
    # ENGINE looks for, so `check --detector=tensorrt` would say "engine not
    # found" straight after a twenty-minute build; and it carries no input
    # square, so building at 640 would silently overwrite a 576 engine and the
    # only symptom would be TensorRTDetector's shape guard firing next run.
    final = os.path.join(outdir, f"rfdetr-{variant}-{size}.trt")
    if os.path.abspath(path) != os.path.abspath(final):
        os.replace(path, final)
    print(f"engine: {final}", flush=True)
    if os.path.abspath(final) == os.path.abspath(ENGINE):
        print("  this is the default path; `check --detector=tensorrt` will find it.", flush=True)
    else:
        print(f"  point the pipeline at it with:  export FT_TRT_ENGINE={final}", flush=True)


# ----------------------------------------------------------------- self-check
def _selfcheck() -> None:
    """Runs on ANY machine, GPU or not. What is skipped without CUDA is named.

    Exercised here, with no GPU: the import guards and their messages, the
    preprocessing against rfdetr's own transform, the fixture loader and its
    hash, the sanity gate on five ways of being wrong, the greedy matcher, and
    check()'s full verdict path - PASS and FAIL both, driven by a canned
    detector. Skipped: constructing a CUDA/TensorRT detector and running an
    engine. That is two constructors; everything downstream of them is checked.
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    have_cuda = False
    with contextlib.suppress(Exception):
        import torch  # noqa: PLC0415

        have_cuda = torch.cuda.is_available()

    # -- the guards fire, and say what to install --------------------------
    if not have_cuda:
        # Both branches of each guard (absent package / present but no device)
        # must name the thing that is missing AND a command that fixes it - an
        # error that only says "no GPU" costs the user an afternoon.
        for fn, must in ((_require_cuda, ("cuda", "pip install")),
                         (_require_tensorrt, ("tensorrt", "pip install"))):
            try:
                fn()
            except SystemExit as e:
                msg = str(e).lower()
                assert all(m in msg for m in must), (fn.__name__, msg)
            else:
                raise AssertionError(f"{fn.__name__} did not raise without CUDA/TensorRT")
        # ...and a detector cannot be built at all.
        for cls in (CUDADetector, TensorRTDetector):
            try:
                cls()
            except SystemExit:
                pass
            else:
                raise AssertionError(f"{cls.__name__} was constructible with no CUDA device")

    # -- the matcher -------------------------------------------------------
    b = lambda x1, y1, x2, y2, c=0.9: {"xyxy": (x1, y1, x2, y2), "conf": c, "cls": "person"}  # noqa: E731
    p, i = _match([b(0, 0, 10, 100)], [b(0, 0, 10, 100)])
    assert p == [(0, 0)] and abs(i[0] - 1.0) < 1e-9, (p, i)
    assert _match([b(0, 0, 10, 100)], [b(0, 0, 10, 40)])[0] == []      # IoU 0.4 < 0.5
    assert _match([b(0, 0, 10, 100)], [])[0] == []
    # exclusive on both sides: two references, one detection -> one pair
    assert len(_match([b(0, 0, 10, 100), b(1, 1, 11, 101)], [b(0, 0, 10, 100)])[0]) == 1

    ref = [[b(10, 10, 30, 70), b(100, 20, 118, 74)], [b(200, 5, 224, 65)], []]
    st = _agree(ref, ref)
    assert st["R"] == 1.0 and st["P"] == 1.0 and st["med_iou"] == 1.0, st
    assert _agree(ref, [[], [], []])["R"] == 0.0
    # a 40 px shift is a different player, not the same box
    shifted = [[{**d, "xyxy": tuple(v + 40 for v in d["xyxy"])} for d in t] for t in ref]
    assert _agree(ref, shifted)["R"] == 0.0, _agree(ref, shifted)
    # a 3 px shift still matches, with a visibly lower IoU - so IoU is doing work
    nudged = [[{**d, "xyxy": tuple(v + 3 for v in d["xyxy"])} for d in t] for t in ref]
    s3 = _agree(ref, nudged)
    assert s3["R"] == 1.0 and 0.6 < s3["med_iou"] < 0.95, s3

    # -- the sanity gate, on every way of being wrong it claims to name -----
    assert _sanity(ref, ref, SIZE) == []
    assert "ZERO boxes" in _sanity([[], [], []], ref, SIZE)[0]
    norm = [[{**d, "xyxy": tuple(v / SIZE for v in d["xyxy"])} for d in t] for t in ref]
    assert any("NORMALIZED" in s for s in _sanity(norm, ref, SIZE)), _sanity(norm, ref, SIZE)
    cxcy = [[{**d, "xyxy": ((d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2,
                            d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1])} for d in t]
            for t in ref]
    assert any("cxcywh" in s for s in _sanity(cxcy, ref, SIZE)), _sanity(cxcy, ref, SIZE)
    huge = [[{**d, "xyxy": tuple(v * 4 for v in d["xyxy"])} for d in t] for t in ref]
    assert _sanity(huge, ref, SIZE), "a 4x scale error must not pass the sanity gate"
    nan = [[{**d, "xyxy": (float("nan"),) + d["xyxy"][1:]} for d in t] for t in ref]
    assert any("finite" in s for s in _sanity(nan, ref, SIZE)), _sanity(nan, ref, SIZE)

    # -- the preprocessing, against rfdetr's own transform ------------------
    # The one piece of the TensorRT path that CAN be verified without a GPU, and
    # the one most likely to be silently wrong: BGR/RGB and the ImageNet stats.
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from rfdetr.export.benchmark import infer_transforms  # noqa: PLC0415
    except ImportError:
        print("  (skipped: preprocessing vs rfdetr's infer_transforms - rfdetr/torch absent)")
    else:
        S = 64
        rng = np.random.default_rng(0)
        bgr = rng.integers(0, 256, (2, S, S, 3), dtype=np.uint8)
        # The same arithmetic TensorRTDetector._pre does, on the CPU. Written out
        # rather than called because _pre hard-codes device="cuda".
        x = torch.from_numpy(bgr)[..., [2, 1, 0]].permute(0, 3, 1, 2).to(torch.float32).div_(255)
        x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
        tf = infer_transforms((S, S))
        for k in range(len(bgr)):
            want, _ = tf(Image.fromarray(bgr[k][..., ::-1]), None)   # PIL wants RGB
            assert float((want - x[k]).abs().max()) < 1e-5, float((want - x[k]).abs().max())
        # ...and it can fail: a red image through the wrong channel order is not
        # the same tensor. (A grey random image would agree by accident.)
        red = np.zeros((1, S, S, 3), np.uint8)
        red[..., 2] = 255
        def norm3(t):
            t = t.permute(0, 3, 1, 2).to(torch.float32) / 255
            return (t - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)

        rx = norm3(torch.from_numpy(red)[..., [2, 1, 0]])       # correct: BGR -> RGB
        wrong = norm3(torch.from_numpy(red))                    # the bug: no flip
        assert float((rx - wrong).abs().max()) > 1.0, "BGR and RGB must not agree on red"

    # -- the postprocess, against rfdetr's OWN PostProcess -----------------
    # The second thing about the TensorRT path that can be checked with no GPU,
    # and the one that WAS wrong. TensorRTDetector decodes with
    # rfdetr.export.benchmark.post_process; the fixture's reference came out of
    # rfdetr.models.postprocess.PostProcess, via predict(). They are NOT the
    # same function: PostProcess clamps boxes into the image and post_process
    # does not. On this fixture the gap is sub-pixel (see detect_batch), so the
    # value of this assert is not today's number - it is that the two functions
    # live in the same library and can drift without anything else noticing.
    # It checks both halves: that the clamp is still needed, and still enough.
    try:
        import torch  # noqa: PLC0415
        from rfdetr.export.benchmark import post_process  # noqa: PLC0415
        from rfdetr.models.postprocess import PostProcess  # noqa: PLC0415
    except ImportError:
        print("  (skipped: postprocess parity vs rfdetr's PostProcess - rfdetr/torch absent)")
    else:
        g = torch.Generator().manual_seed(3)
        B, Q, C = 2, 300, 91
        logits = torch.randn(B, Q, C, generator=g)
        # cxcywh normalized: with random centres and widths plenty of these
        # leave the square, which is the only case where the clamp shows.
        boxes = torch.rand(B, Q, 4, generator=g)
        # Deliberately NOT square, so an h/w transposition cannot hide.
        tgt = torch.tensor([[400.0, 576.0]] * B)
        scale = torch.tensor([576.0, 400.0, 576.0, 400.0])       # [w, h, w, h]
        want = PostProcess(num_select=Q)({"pred_logits": logits, "pred_boxes": boxes}, tgt)
        got = post_process({"labels": logits, "dets": boxes}, tgt, num_queries=Q)
        assert len(want) == len(got) == B
        for w, q in zip(want, got):
            assert torch.equal(w["scores"], q["scores"]), "score selection diverged"
            assert torch.equal(w["labels"], q["labels"]), "class index selection diverged"
            assert not torch.allclose(w["boxes"], q["boxes"]), (
                "post_process now agrees with PostProcess unclamped - upstream added the clamp, "
                "so the clamp_ in TensorRTDetector.detect_batch is dead code; delete it.")
            assert torch.allclose(w["boxes"], torch.minimum(q["boxes"].clamp_min(0.0), scale),
                                  atol=1e-4), "the clamp is no longer enough to reach parity"
        # ...and on the square this pipeline actually feeds, that reduces to the
        # one-liner detect_batch runs.
        sq = torch.tensor([[float(SIZE), float(SIZE)]] * B)
        for w, q in zip(PostProcess(num_select=Q)({"pred_logits": logits, "pred_boxes": boxes}, sq),
                        post_process({"labels": logits, "dets": boxes}, sq, num_queries=Q)):
            assert torch.allclose(w["boxes"], q["boxes"].clamp(0.0, float(SIZE)), atol=1e-4)

    # -- the fixture, and check() end to end -------------------------------
    if not os.path.exists(FIXTURE):
        print(f"  (skipped: {os.path.relpath(FIXTURE, _ROOT)} is absent - "
              "run `make-fixture` on a Mac with the footage)")
        print("nvidia.py self-check OK (guards, matcher, sanity gate, preprocessing)")
        return
    tiles, exp, meta = load_fixture()
    assert len(tiles) == len(exp) == meta["n"] > 0
    assert all(t.shape == (meta["size"], meta["size"], 3) for t in tiles)
    assert sum(len(t) for t in exp) > 50, "a fixture this thin cannot measure recall"
    # At the shipping threshold at least one tile must be empty grass, or a
    # detector that hallucinates over an empty pitch is never charged for it.
    assert sum(1 for t in exp if not [b for b in t if b["conf"] >= 0.5]) >= 1, \
        "no tile that is empty at conf 0.5: precision has nowhere to fail"

    def _run(rows, cls=_Canned, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = check(cls(rows, tiles), timing=False, **kw)
        return got, buf.getvalue()

    # A perfect detector passes...
    ok, txt = _run(exp)
    assert ok, txt
    assert "recall 1.000" in txt and "PASS" in txt, txt
    # ...and the null control it prints is LOW, which is what makes the 1.000
    # mean anything. If this ever reads high, the metric is counting boxes.
    null = [float(line.split("recall")[1].split()[0])
            for line in txt.splitlines() if "NULL" in line]
    assert null and max(null) < PASS_NULL, txt

    # Every failure the docstring claims to catch, actually caught:
    for name, rows, want in (
        ("empty", [[] for _ in exp], "ZERO boxes"),
        ("rotated/stale", exp[1:] + exp[:1], "FAIL"),
        ("normalized", [[{**d, "xyxy": tuple(v / meta["size"] for v in d["xyxy"])} for d in t]
                        for t in exp], "NORMALIZED"),
        ("shifted 40px", [[{**d, "xyxy": tuple(v + 40 for v in d["xyxy"])} for d in t]
                          for t in exp], "FAIL"),
        ("half the boxes", [t[::2] for t in exp], "FAIL"),
    ):
        ok, txt = _run(rows)
        assert not ok, f"{name} must FAIL the check\n{txt}"
        assert want in txt, f"{name}: expected {want!r} in\n{txt}"

    # THE CHEAT. A detector that returns the fixture's own expected boxes scores
    # a perfect 1.000 on agreement and is caught by liveness alone. If this ever
    # passes, every other number in this file is decoration.
    ok, txt = _run(exp, cls=_Cheat)
    assert not ok, f"a detector replaying the expected boxes must FAIL\n{txt}"
    assert "not looking at its input" in txt, txt

    # An fp16-shaped disagreement - every box nudged sub-pixel and rescored -
    # must still PASS, or the bar is set where no real GPU can reach it.
    rng = np.random.default_rng(7)
    fp16ish = [[{**d, "xyxy": tuple(v + rng.normal(0, 0.3) for v in d["xyxy"]),
                 "conf": float(np.clip(d["conf"] + rng.normal(0, 0.004), 0, 1))} for d in t]
               for t in exp]
    ok, txt = _run(fp16ish, label="fp16-like")
    assert ok, f"a sub-pixel fp16-shaped difference must still pass\n{txt}"

    print(f"nvidia.py self-check OK (guards, matcher, sanity gate, preprocessing, "
          f"fixture {meta['n']} tiles / {sum(len(t) for t in exp)} boxes, "
          f"check() passes and fails on cue)")
    if not have_cuda:
        print("  SKIPPED, and only this: building a CUDADetector/TensorRTDetector and running "
              "an engine. No CUDA device on this machine.")


# --------------------------------------------------------------------- SETUP
SETUP = """
SETTING THIS UP ON THE PC (RTX 2060, i7). Nothing below has been run by the
author - it is assembled from the vendors' own instructions, and step 1 exists
so you find out in thirty seconds rather than after the engine build.

0. PYTHON 3.12. Not 3.14. There is no torch wheel and no tensorrt wheel for
   3.14 - the same wall the CoreML work hit. Check with `python --version`.

       py -3.12 -m venv .venv-cuda            (Windows)
       python3.12 -m venv .venv-cuda          (Linux)

1. DRIVER AND TORCH. A 2060 is Turing (compute 7.5), supported by every current
   CUDA build. You need an NVIDIA driver >= 525; `nvidia-smi` prints it.

       .venv-cuda/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
       .venv-cuda/bin/pip install rfdetr opencv-python
       .venv-cuda/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

   That last line must print True. If it prints False you have the CPU wheel.

2. THE CUDA PATH FIRST - it needs no engine and proves the model and the fixture
   before TensorRT is in the picture:

       .venv-cuda/bin/python model/engine/nvidia.py check --detector=cuda

   Expect ~1-2 minutes, most of it downloading the RF-DETR checkpoint the first
   time. If this fails, TensorRT will not help.

   If it fails in a way that looks like the FIXTURE rather than the GPU, run
   `check --detector=torch`, which uses whatever device rfdetr picks on its own.
   On this Mac that command scores 1.000/1.000, so a failure there is the
   fixture or the install, not CUDA.

3. TENSORRT.

       .venv-cuda/bin/pip install "rfdetr[tensorrt]"      (pulls tensorrt + polygraphy)
       .venv-cuda/bin/python model/engine/nvidia.py build

   Roughly 5-20 minutes, nearly all of it TensorRT autotuning kernels for YOUR
   card. UNVERIFIED - that is the range NVIDIA describe for a model this size,
   not a measurement. It writes model/models/rfdetr-medium-576.trt, which is exactly
   what step 4 looks for. (rfdetr's own exporter names it rfdetr-medium.trt,
   with no input square in the name; build() renames it so that a 640 engine
   cannot overwrite a 576 one.) The engine is locked to this GPU and this
   TensorRT version: do not copy it anywhere, and rebuild it after upgrading
   either.

4. THE VERIFICATION, the one command that answers "is this right?":

       .venv-cuda/bin/python model/engine/nvidia.py check --detector=tensorrt

   WHAT GOOD LOOKS LIKE. Two rows, conf 0.5 and conf 0.25:

       recall >= 0.97, precision >= 0.97, median IoU >= 0.90
       median |score delta| around 0.004
       NULL (rotated reference) recall below 0.25 - ideally under 0.05
       liveness: mirrored tiles return a similar NUMBER of boxes, and under 0.05
                 of them in the same place (on the Mac: 321 vs 340, 0.021)

   Add --fp32 to `check --detector=cuda` for the tighter comparison: fp32 on the
   GPU against the fixture's fp32 reference should be closer still. If fp32
   agrees and fp16 does not, the engine is fine and it is a precision question.

   For scale: CoreML fp16 on the Mac's Neural Engine hit R 0.990 / P 0.995 /
   median IoU 0.98 / score delta 0.0035 against PyTorch over 112 real frames,
   and PyTorch MPS fp16 hit 1.000 / 1.000 / 0.990 / 0.0006 against this very
   fixture. Exact equality is NOT expected and NOT required - fp16 is not fp32.
   Anything at or above the CoreML row is a healthy TensorRT engine.

   The command exits 0 on PASS and 1 on FAIL, so it can go in a script.

   If conf 0.5 warns about threshold flips while conf 0.25 is clean, that is the
   known and expected fp16 behaviour on this footage: conf 0.5 sits on a cliff
   and 0.2-0.3 is where this pipeline belongs anyway (detectors.py).

5. WIRING IT INTO THE PIPELINE. detectors.py is ALREADY DONE - checked, not
   assumed: it has a `_nvidia(which)` factory and

       _DETECTORS = {..., "cuda": _nvidia("_cuda"), "tensorrt": _nvidia("_tensorrt")}

   with the `import nvidia` inside the factory so `import detectors` still works
   on a machine with no GPU. Nothing to add there.

   track.py is NOT done, and it is one edit. Its --conf/--variant plumbing only
   fires for "rfdetr", so `--detector cuda` today silently runs at the default
   conf 0.5 instead of the 0.25 this pipeline wants. At the
   `if detector == "rfdetr":` block (around line 621):

       if detector in ("rfdetr", "cuda"):
           dkw = {"conf": conf, "variant": variant}
           dkw.update({k: v for k, v in (("weights", weights),
                                         ("resolution", resolution)) if v})
       elif detector == "tensorrt":
           dkw = {"conf": conf}

   TENSORRT MUST BE ITS OWN BRANCH. Do not write
   `if detector in ("rfdetr", "cuda", "tensorrt")`: TensorRTDetector takes no
   `variant`, `weights` or `resolution` - the engine fixes all three - and that
   one-liner is a TypeError at startup. There is also no --engine flag on
   track.py; select a non-default engine with FT_TRT_ENGINE in the environment.

   After that:  --detector tensorrt --conf 0.25

6. WHAT MIGHT STILL BE SLOW, and it is not the GPU. Per frame the CPU decodes
   4096x1080, remaps it to an 11878x870 band (~30 MB), and cuts and resizes 16
   tiles. At 1 frame/s that is invisible; at 12 frames/s it may be the whole
   bill. `check` prints inference time and host time separately for the
   TensorRT path - if host time is the larger number, the next moves are NVDEC
   decode and doing the remap on the GPU. NOT MEASURED, here or anywhere.
"""


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "selfcheck"
    # Both "--k=v" and "--k v". The first version only took "--k=v", which made
    # `check --detector tensorrt` silently run the CUDA path - a flag that is
    # ignored rather than rejected is exactly the class of bug this file exists
    # to avoid.
    args, flags, rest = {}, [], list(a[1:])
    while rest:
        x = rest.pop(0)
        if not x.startswith("--"):
            raise SystemExit(f"unexpected argument {x!r}; options look like --name=value")
        if "=" in x:
            k, v = x[2:].split("=", 1)
            args[k.replace("-", "_")] = v
        elif rest and not rest[0].startswith("--"):
            args[x[2:].replace("-", "_")] = rest.pop(0)
        else:
            flags.append(x)
    # ...and the same argument applies to a name nobody reads. `--engien=x` or
    # `--fp32` on a path that has no fp32 knob used to be accepted and dropped,
    # which is the same silent-no-op bug in a different costume.
    _OPTS = {"selfcheck": (set(), set()),
             "setup": (set(), set()),
             "make-fixture": ({"out", "job", "video", "per_frame", "quiet_tiles", "conf_floor"},
                              set()),
             "build": ({"out", "variant", "size", "batch"}, set()),
             "check": ({"detector", "conf", "engine"}, {"--fp32"})}
    if cmd in _OPTS:
        stray = sorted((set(args) - _OPTS[cmd][0]) | (set(flags) - _OPTS[cmd][1]))
        if stray:
            raise SystemExit(f"{cmd}: unknown option(s) {stray}; it takes "
                             f"{sorted(_OPTS[cmd][0] | _OPTS[cmd][1]) or 'none'}")
    if cmd == "selfcheck":
        _selfcheck()
    elif cmd == "setup":
        print(SETUP)
    elif cmd == "make-fixture":
        # argv is all strings; make_fixture indexes and compares with these.
        make_fixture(**{k: (int(v) if k in ("per_frame", "quiet_tiles")
                            else float(v) if k == "conf_floor" else v)
                        for k, v in args.items()})
    elif cmd == "build":
        build(out=args.get("out"), variant=args.get("variant", "medium"),
              size=int(args.get("size", SIZE)), batch=int(args.get("batch", 1)))
    elif cmd == "check":
        which = args.get("detector", "cuda")
        conf = float(args.get("conf", 0.10))    # the floor; check() scores at 0.5 and 0.25
        if which == "cuda":
            det = CUDADetector(conf=conf, half="--fp32" not in flags)
        elif which == "tensorrt":
            if "--fp32" in flags:
                raise SystemExit("--fp32 means nothing to an engine: precision is baked in at "
                                 "build time. Rebuild with fp16 off, or drop the flag.")
            det = TensorRTDetector(engine=args.get("engine"), conf=conf)
        elif which == "torch":
            # Whatever device rfdetr picks by itself - cuda on the PC, MPS here.
            # Not an NVIDIA path: it is how the FIXTURE is checked on a machine
            # with no GPU, so "the PC disagrees" can be told apart from "the
            # fixture is broken". MEASURED HERE on MPS fp16 against the fp32
            # reference: R 1.000, P 1.000, median IoU 0.990, NULL 0.018.
            from detectors import RFDETRDetector  # noqa: PLC0415

            det = RFDETRDetector(variant="medium", conf=conf, half="--fp32" not in flags)
            # This is the ONE path in the file with no CUDA guard, so it is the
            # one that could quietly be running on a CPU. Print the device it
            # actually landed on rather than let a 900 ms/tile run look like bad
            # agreement.
            det.device_name = str(getattr(det.model.model, "device", "?"))
        else:
            raise SystemExit(f"unknown --detector {which!r}; have cuda, tensorrt, torch")
        sys.exit(0 if check(det, label=which) else 1)
    else:
        raise SystemExit(f"unknown command {cmd!r}\n"
                         "  selfcheck | setup | check --detector=cuda|tensorrt | build | "
                         "make-fixture")
