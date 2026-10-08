"""Association: per-frame boxes -> stable track ids.

Renamed from trackers.py, which shadowed the pip package `trackers`
(roboflow/trackers, Apache-2.0); this name does not. The shipped tracker is
C-BIoU from that package (_RoboflowTracker, via get_tracker("best")). ByteTrack
(MIT) and a plain IoU tracker are vendored below as the zero-dependency fallback.

Motion and IoU only. No appearance embeddings, no faces, no gait - see README
on BIPA.

Manual: docs/reference/detection-tracking.md#association
"""
from typing import Protocol

import numpy as np


class Tracker(Protocol):
    """Detections for one frame -> the same detections with an "id" added.

    `frame` is accepted and ignored by everything here; it is in the signature
    so an appearance-based tracker could be dropped in without changing callers.
    Unassociated tracks emit nothing rather than a predicted box: stats.py
    already treats gaps as unobserved, and inventing positions would turn an
    occlusion into fake distance covered.
    """

    def update(self, detections: list[dict], frame: np.ndarray | None = None) -> list[dict]: ...


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    wh = lambda z: np.clip(z[:, 2] - z[:, 0], 0, None) * np.clip(z[:, 3] - z[:, 1], 0, None)  # noqa: E731
    union = wh(a)[:, None] + wh(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _expand(b: np.ndarray, k: float) -> np.ndarray:
    """Grow every box by k x its own width/height on each side.

    Expansion-IoU (Deep-EIoU's idea, reimplemented - that repo ships no LICENCE).
    Plain IoU is degenerate at this box size: the median detected player on the
    remapped 118575 panorama is 24 px tall and 10 px wide, moves 1.7 px between
    frames, and the MEASURED median IoU between two consecutive detections of
    the same ground-truth player is 0.36 - with 30% of pairs under the 0.2 gate
    and 21% at exactly zero. That is the whole fragmentation story: the match
    fails, the track dies, a new id is born, median track span 0.4 s.

    Proportional, not a fixed pixel margin. Both were measured; a fixed 10 px
    lifts the median to 0.72 but it is 0.4x a near player's height and 0.5x a
    far player's, so it over-gates exactly where players are densest and lifts
    the best WRONG-player score from 0.02 to 0.13. k x box keeps the margin in
    units of the player, which is what the ~4.8x near-to-far vertical scale
    gradient demands.
    """
    w = (b[:, 2] - b[:, 0]) * k
    h = (b[:, 3] - b[:, 1]) * k
    return b + np.column_stack([-w, -h, w, h])


def _height_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,M) overlap of the VERTICAL extent only.

    The measured pathology on this footage is box-extent jitter, not motion: a
    median player is 10 px wide and 24 px tall, so a +/-1 px wobble per edge is a
    20% width change but only an 8% height change. Height is the more stable
    measurement by ~2.4x, and multiplying it into the expanded IoU says so.

    MeMoSORT's Mo-IoU makes the same argument for occlusion ("a target's height
    can remain a highly distinguishable feature"); here the motivation is
    smallness rather than occlusion, and both point the same way.
    """
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, 1], b[None, :, 1])
    hi = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(hi - lo, 0, None)
    ha = np.clip(a[:, 3] - a[:, 1], 0, None)[:, None]
    hb = np.clip(b[:, 3] - b[:, 1], 0, None)[None, :]
    union = ha + hb - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _greedy(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """Highest-IoU-first matching.

    # ponytail: greedy, not Hungarian. With 22 players it is within a hair of
    # optimal and needs no scipy. If ID switches in crowds turn out to be
    # dominated by association order, swap in
    # scipy.optimize.linear_sum_assignment on (1 - iou).
    """
    m, pairs = iou.copy(), []
    while m.size:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        if m[i, j] < thr:
            break
        pairs.append((int(i), int(j)))
        m[i, :] = -1.0
        m[:, j] = -1.0
    return pairs


def _hungarian(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """Globally optimal matching on (1 - iou), gated at `thr`.

    Greedy takes the best pair first and lets it block a pair it crosses; with
    22 players in a crowd that ordering IS the id switch. scipy is already a
    dependency (it arrives with the detector stack), so the only cost is the
    solve itself, which is microseconds at this size.
    """
    if not iou.size:
        return []
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    ri, ci = linear_sum_assignment(1.0 - iou)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if iou[i, j] >= thr]


class _Track:
    __slots__ = ("id", "box", "vel", "lost", "hits")

    def __init__(self, tid: int, box: np.ndarray):
        self.id, self.box = tid, box.astype(np.float64)
        self.vel = np.zeros(4)
        self.lost, self.hits = 0, 1

    def predict(self) -> np.ndarray:
        # ponytail: EMA constant-velocity extrapolation, not a Kalman filter.
        # Ceiling is long occlusions - a Kalman's growing covariance widens the
        # gate as confidence drops, while a fixed extrapolation just drifts off
        # the player. Upgrade path: a 7-state constant-velocity Kalman (~40 lines).
        return self.box + self.vel * (1 + self.lost)

    def absorb(self, box: np.ndarray) -> None:
        self.vel = 0.5 * self.vel + 0.5 * (box - self.box)
        self.box = box.astype(np.float64)
        self.lost, self.hits = 0, self.hits + 1


class _Base:
    """Shared association core. ByteTrack is this plus one extra pass over the
    low-confidence detections that everything else throws away."""

    second_pass = False

    # Defaults of the FALLBACK trackers, left as measured: no knob swept on 118575
    # moved IDF1 by more than a few thousandths. Not evidence about C-BIoU.
    # Evidence: detection-tracking-evidence (not published)#evidence-association
    def __init__(self, iou_thr: float = 0.2, second_iou_thr: float = 0.5, max_lost: int = 30,
                 new_track_conf: float = 0.5, low_conf: float = 0.1, expand: float = 0.5,
                 hungarian: bool = False, isu: tuple = (), lost_gain: float = 0.0,
                 height_w: float = 0.0):
        self.iou_thr, self.second_iou_thr = iou_thr, second_iou_thr
        self.max_lost, self.new_track_conf, self.low_conf = max_lost, new_track_conf, low_conf
        # expand=0 is plain IoU, i.e. the old behaviour, for A/B measurement.
        self.expand = expand
        # Every knob below defaults OFF so the shipped behaviour is unchanged
        # until a measurement says otherwise - model/validation/assoc_eval.py scores
        # each one against real ground truth in about seven seconds.
        self.hungarian = hungarian
        # Iterative scale-up: extra expansion rounds over what the first pass
        # left unmatched. Deep-EIoU measured +1.94 HOTA for 0.7/0.8/0.9, the
        # point being that no single expansion suits both a 10 px far player
        # and a 115 px near one.
        self.isu = tuple(isu)
        # Widen the gate as a track goes unseen. predict() already scales the
        # DISPLACEMENT by (1 + lost); this scales the UNCERTAINTY, which is the
        # growing-covariance behaviour a Kalman would give, in one term.
        self.lost_gain = lost_gain
        # Blend in vertical-extent overlap; see _height_iou.
        self.height_w = height_w
        self.tracks: list[_Track] = []
        self._next_id = 1

    def _cost(self, tracks: list[_Track], dets: list[dict], k: float) -> np.ndarray:
        pred = np.array([t.predict() for t in tracks])
        boxes = np.array([d["xyxy"] for d in dets], dtype=np.float64)
        if self.lost_gain:
            # Per-track expansion: a track lost for 20 frames gets a wider gate
            # than one seen last frame, which is the whole point.
            grow = np.array([k * (1.0 + self.lost_gain * t.lost) for t in tracks])
            w = (pred[:, 2] - pred[:, 0]) * grow
            h = (pred[:, 3] - pred[:, 1]) * grow
            pred = pred + np.column_stack([-w, -h, w, h])
            boxes = _expand(boxes, k) if k else boxes
        elif k:
            pred, boxes = _expand(pred, k), _expand(boxes, k)
        c = _iou(pred, boxes)
        if self.height_w:
            # MULTIPLIED into the expanded IoU, never blended: a weighted sum lets a
            # far box on the same image row (height-IoU 1.0) weld two players.
            # Multiplying lets height refine an overlap, never create one.
            # detection-tracking-evidence (not published)#evidence-association
            c = c * np.power(_height_iou(pred, boxes), self.height_w)
        return c

    def _match(self, tracks: list[_Track], dets: list[dict], thr: float) -> list[tuple[int, int]]:
        if not tracks or not dets:
            return []
        solve = _hungarian if self.hungarian else _greedy
        pairs = solve(self._cost(tracks, dets, self.expand), thr)
        if not self.isu:
            return pairs
        # Retry what is still free at successively wider expansions. Indices
        # stay in the caller's space throughout, so the caller needs no changes.
        used_t = {i for i, _ in pairs}
        used_d = {j for _, j in pairs}
        for k in self.isu:
            ft = [i for i in range(len(tracks)) if i not in used_t]
            fd = [j for j in range(len(dets)) if j not in used_d]
            if not ft or not fd:
                break
            for a, b in solve(self._cost([tracks[i] for i in ft], [dets[j] for j in fd], k), thr):
                pairs.append((ft[a], fd[b]))
                used_t.add(ft[a])
                used_d.add(fd[b])
        return pairs

    def update(self, detections: list[dict], frame: np.ndarray | None = None) -> list[dict]:
        hi = [d for d in detections if d["conf"] >= self.new_track_conf]
        lo = [d for d in detections if self.low_conf <= d["conf"] < self.new_track_conf]

        live = list(self.tracks)
        free_t, free_d = set(range(len(live))), set(range(len(hi)))
        out: list[dict] = []

        def take(t: _Track, d: dict) -> None:
            t.absorb(np.asarray(d["xyxy"], dtype=np.float64))
            out.append({**d, "id": t.id})

        for ti, di in self._match(live, hi, self.iou_thr):
            take(live[ti], hi[di])
            free_t.discard(ti)
            free_d.discard(di)

        if self.second_pass and lo and free_t:
            # ByteTrack's actual idea: a low-confidence box sitting on an
            # already-tracked player is usually that player occluded, not noise.
            # Recovering it here is what keeps the id alive through a crowd.
            rest = sorted(free_t)
            for ri, di in self._match([live[i] for i in rest], lo, self.second_iou_thr):
                take(live[rest[ri]], lo[di])
                free_t.discard(rest[ri])

        for di in sorted(free_d):
            t = _Track(self._next_id, np.asarray(hi[di]["xyxy"], dtype=np.float64))
            self._next_id += 1
            self.tracks.append(t)
            out.append({**hi[di], "id": t.id})

        for ti in free_t:
            live[ti].lost += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return out


class IoUTracker(_Base):
    """Single-pass greedy IoU. Zero dependencies beyond numpy, so the pipeline
    always runs even when nothing else is installed."""

    second_pass = False


class ByteTrackTracker(_Base):
    """ByteTrack (MIT), vendored: high-confidence pass, then a second pass that
    re-associates low-confidence detections to still-unmatched tracks."""

    second_pass = True


class _RoboflowTracker:
    """Adapter onto roboflow/trackers (Apache-2.0), measured better than ours.

    Kept behind an adapter rather than replacing the hand-rolled trackers,
    because those are the zero-dependency floor this module promises: the
    pipeline has to run when nothing optional is installed. This import is
    therefore lazy and its absence is a clear error, not a crash at frame 40000.

    Measured better than the vendored trackers on 118575 against ground truth
    (detection-tracking-evidence (not published)#evidence-association). get_tracker builds it with no
    kwargs, so every constant is the library default (docs/reference/detection-tracking.md#tracker).
    """

    def __init__(self, kind: str = "cbiou", iou: str | None = None, **kw):
        try:
            import trackers as _rf  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - environment, not logic
            raise SystemExit(
                f"tracker {kind!r} needs the 'trackers' package (Apache-2.0): "
                "pip install trackers") from e
        import supervision as _sv  # noqa: PLC0415

        self._sv = _sv
        cls = {"cbiou": _rf.CBIoUTracker, "rf-bytetrack": _rf.ByteTrackTracker,
               "ocsort": _rf.OCSORTTracker, "botsort": _rf.BoTSORTTracker}[kind]
        if iou:
            kw["iou"] = {"diou": _rf.DIoU, "ciou": _rf.CIoU,
                         "giou": _rf.GIoU, "biou": _rf.BIoU}[iou]()
        self._t = cls(**kw)

    def update(self, detections: list[dict], frame: np.ndarray | None = None) -> list[dict]:
        # An empty frame still goes in: it is what ages the Kalman state and the
        # lost-track counters (trackers 2.6.0 accepts an empty Detections).
        # Class names are carried across the tracker by INDEX into this frame's
        # own list, not through supervision's class_id, and never dropped: a
        # detector that names "referee" or "goalkeeper" is the only thing that
        # can tell those two apart from a player (kit colour provably cannot -
        # see track._officials_by_contrast), so losing the name here would make
        # a class-aware checkpoint pointless on this tracker while working fine
        # on IoUTracker. class_id stays all-zero deliberately: it is the inner
        # tracker's association gate, and gating on class would refuse to link a
        # box the detector relabelled mid-track.
        names = [d.get("cls", "person") for d in detections]
        det = self._sv.Detections(
            xyxy=np.array([d["xyxy"] for d in detections], np.float32).reshape(-1, 4),
            confidence=np.array([d["conf"] for d in detections], np.float32),
            class_id=np.zeros(len(detections), int),
            data={"name": np.array(names, dtype=object)})
        r = self._t.update(det)
        ids = getattr(r, "tracker_id", None)
        if ids is None:
            return []
        out = []
        for i, tid in enumerate(ids):
            if tid is None or tid < 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in r.xyxy[i])
            out.append({"xyxy": (x1, y1, x2, y2),
                        "conf": float(r.confidence[i]) if r.confidence is not None else 1.0,
                        "cls": str(r.data["name"][i]) if "name" in getattr(r, "data", {})
                        else "person",
                        "id": int(tid)})
        return out


# One metre becomes this many tracker units. The inner trackers were tuned on
# pixel-scale boxes, and handing them coordinates in the range 0-105 would put
# every player inside everyone else's buffer; 100 puts a pitch at 10500x6800,
# which is the magnitude a 4K frame has.
PITCH_SCALE = 100.0

# The ground footprint we give a player, in metres. NOT their height: on the
# pitch plane a body is a footprint, and 1 m is roughly shoulder width plus the
# stance. It sets what "overlapping" means - two feet inside 1 m of each other -
# and it is the same number at both touchlines, which is the entire point.
PLAYER_FOOTPRINT_M = 1.0


class PitchTracker:
    """Associate on the PITCH PLANE, in metres, instead of in image pixels.

    WHY. A pixel-space IoU gate means different things at the near and far
    touchlines of a wide panorama; in metres it means one thing.
    detection-tracking-evidence (not published)#evidence-association

    HOW. The inner tracker is untouched - this is a change of coordinate system,
    not of algorithm, so C-BIoU keeps its cascaded buffers and its Kalman. Each
    detection's FOOT (box bottom-centre, the same point track.py stores and the
    only part of a body the ground-plane homography actually maps) is projected
    to pitch metres and re-expressed as a fixed PLAYER_FOOTPRINT_M box. The
    tracker's constant-velocity model then runs in metres, where it is
    physically meaningful, rather than in a perspective image where a player
    running toward the camera accelerates.

    A box is NOT projected as a box, deliberately: a bounding box is not planar,
    so its corners do not lie on the ground and projecting them produces a
    trapezoid whose area encodes depth rather than size. Only the foot projects
    honestly.

    Ids come back attached to the ORIGINAL pixel boxes, so nothing downstream
    can tell the difference.
    """

    def __init__(self, cal, inner: str | None = None,
                 footprint_m: float = PLAYER_FOOTPRINT_M, **kw):
        # Resolved at CALL time, not as a default argument: BEST is defined
        # below this class, and a def-time default would freeze whatever it was
        # at import rather than following it.
        inner = inner or BEST
        if cal is None:
            raise SystemExit(
                "PitchTracker needs a calibration - association in metres is "
                "not possible without one. Use --tracker cbiou for the "
                "pixel-space path.")
        self.cal = cal
        self.footprint_m = float(footprint_m)
        self.inner = get_tracker(inner, **kw)

    def _boxes(self, dets: list[dict]) -> np.ndarray:
        feet = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2.0, d["xyxy"][3]]
                         for d in dets], float)
        m = np.asarray(self.cal.to_pitch(feet), float) * PITCH_SCALE
        h = self.footprint_m * PITCH_SCALE / 2.0
        return np.column_stack([m[:, 0] - h, m[:, 1] - h, m[:, 0] + h, m[:, 1] + h])

    def update(self, detections: list[dict], frame: np.ndarray | None = None) -> list[dict]:
        if not detections:
            return []
        pitch = self._boxes(detections)
        # A non-finite projection is a detection the homography cannot place -
        # behind the camera, or outside the remap. Dropping it here is right:
        # the alternative is feeding NaN into the tracker's Kalman, which
        # poisons the track rather than losing one box.
        ok = np.isfinite(pitch).all(axis=1)
        proxies, keep = [], []
        for i, good in enumerate(ok):
            if not good:
                continue
            proxies.append({**detections[i], "xyxy": tuple(float(v) for v in pitch[i])})
            keep.append(i)
        if not proxies:
            return []
        got = self.inner.update(proxies, frame)
        if not got:
            return []
        # Map each tracked proxy back to its pixel box by NEAREST CENTRE, not by
        # equality: supervision returns float32, so exact lookups silently miss.
        # detection-tracking-evidence (not published)#evidence-association
        src_c = np.array([[(p["xyxy"][0] + p["xyxy"][2]) / 2.0,
                           (p["xyxy"][1] + p["xyxy"][3]) / 2.0] for p in proxies])
        out_c = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2.0,
                           (d["xyxy"][1] + d["xyxy"][3]) / 2.0] for d in got])
        dist = np.linalg.norm(out_c[:, None, :] - src_c[None, :, :], axis=2)
        j = dist.argmin(1)
        out = []
        for i, d in enumerate(got):
            # 1 unit is 1 cm at PITCH_SCALE. float32 on a 10,000-unit coordinate
            # is good to ~0.001, so anything past this is a different box, not
            # rounding - and returning the wrong player's pixels would be worse
            # than dropping one detection.
            if dist[i, j[i]] > 1.0:
                continue
            out.append({**detections[keep[j[i]]], "id": int(d["id"])})
        return out


_TRACKERS = {"bytetrack": ByteTrackTracker, "iou": IoUTracker}
#: The measured-better options, each a thin adapter onto roboflow/trackers.
_RF_KINDS = {"cbiou": {}, "rf-bytetrack": {}, "ocsort": {}, "botsort": {},
             "rf-bytetrack-diou": {"kind": "rf-bytetrack", "iou": "diou"},
             "cbiou-diou": {"kind": "cbiou", "iou": "diou"}}


#: What "best" resolves to. A name, not a class, so the fallback below can be
#: reported honestly rather than silently substituted.
BEST = "cbiou"


def best_available(quiet: bool = False) -> str:
    """`BEST` if its dependency is installed, else the zero-dependency tracker.

    The pipeline must run on a machine with nothing optional installed - that is
    what IoUTracker and ByteTrackTracker are for - but when the better one IS
    available it is used (detection-tracking-evidence (not published)#evidence-association).

    So: pick the best, and SAY which one was picked. A silent substitution is
    how a run gets attributed to the wrong tracker three weeks later.
    """
    try:
        import trackers  # noqa: F401, PLC0415
    except ImportError:
        if not quiet:
            print("  tracker: 'trackers' not installed, falling back to the built-in "
                  "bytetrack (measured IDF1 0.211 vs 0.263); pip install trackers",
                  flush=True)
        return "bytetrack"
    return BEST


def get_tracker(name: str, **kw) -> Tracker:
    """Tracker by name: "best" -> best_available(); "<inner>-pitch" -> PitchTracker
    (needs kw["cal"]); a roboflow kind (_RF_KINDS) -> _RoboflowTracker; else a
    vendored one (_TRACKERS). Unknown names exit; `kw` goes to the constructor."""
    if name == "best":
        name = best_available()
    # "<inner>-pitch" associates on the pitch plane in metres. It needs a
    # calibration, which is why it is a name rather than a default: an
    # uncalibrated run must still track, in pixels, rather than fail.
    if name.endswith("-pitch"):
        return PitchTracker(kw.pop("cal", None), inner=name[:-len("-pitch")], **kw)
    if name in _RF_KINDS:
        cfg = dict(_RF_KINDS[name])
        cfg.setdefault("kind", name)
        return _RoboflowTracker(**{**cfg, **kw})
    if name not in _TRACKERS:
        raise SystemExit(
            f"unknown tracker {name!r}; have {sorted(_TRACKERS) + sorted(_RF_KINDS)}")
    return _TRACKERS[name](**kw)


if __name__ == "__main__":
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert _iou(a, a)[0, 0] == 1.0
    assert _iou(a, np.array([[20.0, 20.0, 30.0, 30.0]]))[0, 0] == 0.0
    assert abs(_iou(a, np.array([[0.0, 0.0, 10.0, 20.0]]))[0, 0] - 0.5) < 1e-9
    assert _iou(np.zeros((0, 4)), a).shape == (0, 1)

    # Expansion-IoU. 10x24 px is the real median box on the 118575 panorama;
    # at that size a 3.6 px step - the measured p90 displacement - already
    # halves plain IoU, which is why tracks lasted 0.4 s.
    sm = np.array([[100.0, 100.0, 110.0, 124.0]])
    step = sm + np.array([2.0, 3.0, 2.0, 3.0])
    near = sm + np.array([12.0, 0.0, 12.0, 0.0])     # a neighbour one width over
    assert _expand(sm, 0.0).tolist() == sm.tolist()  # k=0 must be the old behaviour
    assert _iou(sm, step)[0, 0] < 0.55 and _iou(sm, near)[0, 0] == 0.0
    e = lambda z: _expand(z, 0.5)  # noqa: E731
    assert _iou(e(sm), e(step))[0, 0] > 0.7
    # ...and the neighbour must stay well below the right box. Expansion that
    # buys track length by welding two players together is worse than
    # fragmenting: stats.py's teleport clamp never trips between adjacent
    # players, so the distance just moves from one profile to the other.
    assert _iou(e(sm), e(near))[0, 0] < 0.5 * _iou(e(sm), e(step))[0, 0]

    # Greedy takes the best pair first, and nothing is matched twice.
    m = np.array([[0.9, 0.4], [0.5, 0.3]])
    assert _greedy(m, 0.2) == [(0, 0), (1, 1)]
    assert _greedy(m, 0.45) == [(0, 0)]  # (1,1) is below threshold

    # The CLASS NAME survives the tracker. Whatever names the detector emits are
    # the only thing that can separate a referee or a keeper from a player, so a
    # tracker that quietly rewrites them to "person" makes a class-aware
    # checkpoint useless. Every tracker on the registry is checked, not just the
    # hand-rolled ones - the roboflow wrapper is where this was hardcoded.
    for name in sorted(_TRACKERS) + sorted(_RF_KINDS):
        try:
            t = get_tracker(name)
        except Exception:   # optional package absent, or a version drift in it
            continue
        got = None
        for f in range(3):
            out = t.update([{"xyxy": (10.0 + 3 * f, 20.0, 34.0 + 3 * f, 80.0),
                             "conf": 0.9, "cls": "referee"}])
            if out:
                got = out[0].get("cls")
        assert got in (None, "referee"), f"{name} rewrote the class to {got!r}"

    # A drifting box keeps its id; a box that teleports gets a new one.
    for cls in (IoUTracker, ByteTrackTracker):
        t = cls()
        ids = [t.update([{"xyxy": (10.0 + 3 * f, 20.0, 34.0 + 3 * f, 80.0), "conf": 0.9,
                          "cls": "person"}])[0]["id"] for f in range(20)]
        assert len(set(ids)) == 1, (cls, ids)
        assert t.update([{"xyxy": (900.0, 20.0, 924.0, 80.0), "conf": 0.9, "cls": "person"}]
                        )[0]["id"] != ids[0]

    # Two small players side by side, both drifting: expansion must keep TWO
    # ids, not merge them. This is the purity half of the trade and the one
    # that silently corrupts numbers when it fails.
    for cls in (IoUTracker, ByteTrackTracker):
        t, seen = cls(), []
        for f in range(30):
            out = t.update([{"xyxy": (100.0 + 2 * f, 100.0 + f, 110.0 + 2 * f, 124.0 + f),
                             "conf": 0.9, "cls": "person"},
                            {"xyxy": (112.0 + 2 * f, 100.0 + f, 122.0 + 2 * f, 124.0 + f),
                             "conf": 0.9, "cls": "person"}])
            assert len(out) == 2, (cls, f, out)
            seen.append(tuple(d["id"] for d in out))
        assert len(set(seen)) == 1, (cls, set(seen))   # no swap, no new ids

    # ByteTrack's whole point: a low-confidence box on a tracked player keeps
    # the id alive where a plain IoU tracker drops the frame entirely. This is
    # what tiling makes matter - far-side players detect faint.
    faint = [{"xyxy": (13.0, 20.0, 37.0, 80.0), "conf": 0.2, "cls": "person"}]
    bt, io = ByteTrackTracker(), IoUTracker()
    for trk in (bt, io):
        trk.update([{"xyxy": (10.0, 20.0, 34.0, 80.0), "conf": 0.9, "cls": "person"}])
    assert bt.update(faint) and bt.update(faint)[0]["id"] == 1
    assert io.update(faint) == []

    # Unassociated tracks emit nothing rather than a guessed box, then expire.
    t = ByteTrackTracker(max_lost=3)
    t.update([{"xyxy": (10.0, 20.0, 34.0, 80.0), "conf": 0.9, "cls": "person"}])
    for _ in range(4):
        assert t.update([]) == []
    assert t.tracks == []

    assert isinstance(get_tracker("bytetrack"), ByteTrackTracker)

    # An empty frame AGES a C-BIoU track (Kalman predict + lost counter); returning
    # before the inner update froze it, so a gap cost nothing.
    try:
        rf = get_tracker("cbiou")
    except SystemExit:
        rf = None                                   # 'trackers' not installed
    if rf is not None:
        rf.update([{"xyxy": (10.0, 20.0, 34.0, 80.0), "conf": 0.9, "cls": "person"}])
        assert rf.update([]) == [] and rf.update([]) == []
        assert [t.time_since_update for t in rf._t.tracks] == [2], \
            [t.time_since_update for t in rf._t.tracks]
    print("tracking.py self-check OK")
