"""Track identity: anonymous tracklets -> merged groups with optional shirt numbers.

Input is the tracks.json contract from track.py:
    {"fps": float, "detections": [{"f", "id", "team", "x", "y", "w"?, "h"?, "jersey"}]}
x, y are the FEET in image pixels of the detector's canvas (the flattened canvas on remap
footage); w, h are the detector's box in the same pixels, where the tracker recorded one.
Output: {track_id: PlayerId} from a strategy (get_identity_strategy).

Contract and config: docs/reference/identity.md. History and measurements:
identity-evidence (not published). With no arguments this runs the self-check; with a
tracks file (-v video, -c calibration.json) it prints the assisted report.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
import argparse
import json

import cv2
import numpy as np

from roster import Roster
from stats import MAX_SPEED  # one speed convention for the whole pipeline

# --- merging -----------------------------------------------------------------
# A break longer than this is not an occlusion, it's a substitution or a walk
# off-frame, and bridging it invents a path we never observed.
MAX_GAP_S = 5.0
# Cost is a weighted mean of normalised terms, so it lives in [0, 1].
W_SPEED, W_GAP = 0.6, 0.2
MAX_COST = 0.55
# The best candidate must beat the runner-up by this much, both ways: a crossing gives near-equal
# costs and is refused. Raise it to merge less. docs/reference/identity.md#merge
MERGE_MARGIN = 0.12
MAX_PASSES = 8

# --- legibility / OCR --------------------------------------------------------
# Height in px of the TORSO CROP, not of the player. identity-evidence (not published)#evidence-crop
MIN_TORSO_PX = 12
# w/h of the crop. A real shape test when the detection carries the detector's box `w`; with the
# synthesised width (torso_box) it fires only when the crop is clipped at a frame edge.
TORSO_ASPECT = (1.0, 1.7)
MIN_SHARPNESS = 40.0                # variance of Laplacian
MIN_NEIGHBOUR_PX = 18               # a closer teammate means the number is covered
MIN_OCR_VOTES = 3
# Body height in px, linear in image row, for a detection with no box `h`.
# ponytail: linear-in-y height model. Ceiling: wrong at extreme lens distortion
# and for tall/short players. Upgrade path is projecting a 1.8m vertical through
# the calibration homography - do that only if crops measurably miss the torso.
HEIGHT_PX_FAR, HEIGHT_PX_NEAR = 20.0, 50.0
TORSO_TOP, TORSO_BOT = 0.20, 0.50   # same band as track.py's team_by_shirt


@dataclass(frozen=True)
class PlayerId:
    """Who a track is. `number is None` means "a human needs to look at this"."""

    team: str
    number: int | None = None
    name: str | None = None
    player_id: str | None = None
    confidence: float = 0.0
    # manual | roster | ocr | auto | unassigned
    source: str = "unassigned"
    track_group: int | None = None  # merged tracklet this track landed in

    @property
    def assigned(self) -> bool:
        return self.number is not None or self.player_id is not None


def by_track(tracks: dict) -> dict[int, list[dict]]:
    """tracks.json -> {track_id: detections sorted by frame}."""
    out: dict[int, list[dict]] = {}
    for d in tracks["detections"]:
        out.setdefault(int(d["id"]), []).append(d)
    for v in out.values():
        v.sort(key=lambda d: d["f"])
    return out


# ---------------------------------------------------------------------------
# Tracklet merging
# ---------------------------------------------------------------------------

@dataclass
class Tracklet:
    tid: int                # representative id = smallest member id
    members: list[int]
    team: str
    t0: float
    t1: float
    p0: np.ndarray          # first position
    p1: np.ndarray          # last position
    n: int


def _positions(dets: list[dict], cal) -> np.ndarray:
    """Feet of `dets` -> (N, 2) float64 pitch metres via cal.to_pitch; unchanged when cal is None."""
    xy = np.array([[d["x"], d["y"]] for d in dets], dtype=np.float64)
    # no cal: right for pitch-metre inputs only; raw pixels make MAX_SPEED refuse every merge
    return cal.to_pitch(xy) if cal is not None else xy


def _build(members: list[int], by_id: dict[int, list[dict]], fps: float,
           cal) -> Tracklet:
    dets = sorted((d for m in members for d in by_id[m]), key=lambda d: d["f"])
    xy = _positions(dets, cal)
    return Tracklet(
        tid=min(members), members=sorted(members),
        team=dets[0].get("team", "unknown"),
        t0=dets[0]["f"] / fps, t1=dets[-1]["f"] / fps,
        p0=xy[0], p1=xy[-1], n=len(dets),
    )


def _pair_cost(a: Tracklet, b: Tracklet, fps: float, max_gap_s: float
               ) -> float | None:
    """Cost in [0, 1] of claiming b continues a; None = refused.

    Motion and the gap only. Positions in pitch metres, times in seconds. Refused: different
    teams, gap <= 0 or > max_gap_s, implied speed > stats.MAX_SPEED (m/s).
    docs/reference/identity.md#merge
    """
    if a.team != b.team:
        return None                      # kit colour is the one signal we trust
    gap = b.t0 - a.t1
    if gap <= 0 or gap > max_gap_s:
        return None                      # overlapping in time = two different people
    d = float(np.linalg.norm(b.p0 - a.p1))
    v = d / max(gap, 1.0 / fps)
    if v > MAX_SPEED:
        return None                      # players don't teleport (stats.py convention)

    return (v / MAX_SPEED * W_SPEED + gap / max_gap_s * W_GAP) / (W_SPEED + W_GAP)


# Known limit: under dense correlated fragmentation motion cost cannot tell a continuation from a
# different player, and the merger welds. docs/reference/identity.md#welds

def _mutual_best(cost: dict[tuple[int, int], float]) -> set[tuple[int, int]]:
    """Pairs that are each other's best match AND clear the margin both ways."""
    def best_two(items):
        s = sorted(items, key=lambda kv: kv[1])
        return s[0][0], s[0][1], (s[1][1] - s[0][1]) if len(s) > 1 else float("inf")

    succ, pred = defaultdict(list), defaultdict(list)
    for (a, b), c in cost.items():
        succ[a].append((b, c))
        pred[b].append((a, c))

    out = set()
    for a, items in succ.items():
        b, c, margin_f = best_two(items)
        if c > MAX_COST or margin_f < MERGE_MARGIN:
            continue
        a2, _, margin_b = best_two(pred[b])
        # Mutual: b's most plausible predecessor must also be a, and that choice
        # must be unambiguous too. One-sided agreement is how a crossing gets
        # welded together.
        if a2 == a and margin_b >= MERGE_MARGIN:
            out.add((a, b))
    return out


def _accepted_pairs(tls: dict[int, Tracklet], fps: float, max_gap_s: float
                    ) -> list[tuple[int, int]]:
    """The merges motion and the gap support, mutually and unambiguously, as sorted (a, b) pairs.

    History of the removed appearance arm: identity-evidence (not published)#evidence-appearance
    """
    # ponytail: O(n^2) over tracklets. Ceiling: fine to a few thousand tracklets
    # (a match is hundreds). Bucket by end-time if a season ever runs at once.
    cost: dict[tuple[int, int], float] = {}
    for a in tls.values():
        for b in tls.values():
            if a.tid == b.tid:
                continue
            c = _pair_cost(a, b, fps, max_gap_s)
            if c is not None:
                cost[(a.tid, b.tid)] = c
    return sorted(_mutual_best(cost))


def merge_tracklets(by_id: dict[int, list[dict]], fps: float, cal=None,
                    max_gap_s: float = MAX_GAP_S) -> tuple[dict[int, int], dict]:
    """Join tracklets that are the same player. -> ({track_id: group_id}, report).

    by_id: track.py detections, feet in canvas px (pitch metres when cal is None); fps in
    frames/s; max_gap_s in seconds. group_id is the smallest track_id in the group, so it is
    stable across runs. report: {"tracklets_before", "tracklets_after"}.
    docs/reference/identity.md#merge
    """
    groups = {t: [t] for t in by_id}
    for _ in range(MAX_PASSES):
        tls = {t: _build(m, by_id, fps, cal) for t, m in groups.items()}
        pairs = _accepted_pairs(tls, fps, max_gap_s)
        if not pairs:
            break
        # One merge per tracklet per pass: a chain A->B->C is safe, but two
        # tracklets claiming the same successor in one pass is not.
        used: set[int] = set()
        for a, b in sorted(pairs):
            if a in used or b in used or a not in groups or b not in groups:
                continue
            used |= {a, b}
            merged = groups.pop(a) + groups.pop(b)
            groups[min(merged)] = merged

    mapping = {t: g for g, members in groups.items() for t in members}
    report = {"tracklets_before": len(by_id), "tracklets_after": len(groups)}
    return mapping, report


# ---------------------------------------------------------------------------
# Sparse jersey OCR (one video pass)
# ---------------------------------------------------------------------------

def torso_box(x: float, y: float, frame_w: int, frame_h: int,
              box_h: float | None = None,
              box_w: float | None = None) -> tuple[int, int, int, int]:
    """Feet (x, y) in canvas px -> torso crop (x1, y1, x2, y2) in px, clipped to the frame.

    The TORSO_TOP..TORSO_BOT band of the body. HEAD DELIBERATELY EXCLUDED: never cropped,
    embedded or stored (docs/reference/identity.md#ocr). `box_h`/`box_w` are the detector's
    box, preferred; without them the height is the HEIGHT_PX_* row model and the width is
    synthesised from the height. identity-evidence (not published)#evidence-crop
    """
    h = box_h if box_h else (
        HEIGHT_PX_FAR + (HEIGHT_PX_NEAR - HEIGHT_PX_FAR) * (y / max(frame_h, 1)))
    top = y - h
    y1, y2 = int(top + TORSO_TOP * h), int(top + TORSO_BOT * h)
    half_w = (box_w / 2.0) if box_w else 0.21 * h  # a torso is ~0.42 body-heights wide
    x1, x2 = int(x - half_w), int(x + half_w)
    return (max(x1, 0), max(y1, 0), min(x2, frame_w), min(y2, frame_h))


def legible(crop: np.ndarray, box: tuple[int, int, int, int],
            nearest_px: float) -> bool:
    """Is this crop worth spending OCR on? Reject early and reject most.

    False when the crop is empty or shorter than MIN_TORSO_PX, its w/h (from `box`, px) is
    outside TORSO_ASPECT, another track's feet are within MIN_NEIGHBOUR_PX (`nearest_px`), or
    the Laplacian variance is under MIN_SHARPNESS. It does not see orientation (facing_away
    ranks sightings upstream) and only partly sees blur. docs/reference/identity.md#ocr
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if crop.size == 0 or h < MIN_TORSO_PX:
        return False
    if not (TORSO_ASPECT[0] <= w / h <= TORSO_ASPECT[1]):
        return False
    if nearest_px < MIN_NEIGHBOUR_PX:
        return False                      # someone is standing in front of the number
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var()) >= MIN_SHARPNESS


def get_ocr_backend(name: str = "auto"):
    """-> callable(crop BGR uint8) -> (text, conf 0-1), or None if no backend imports.

    "auto" tries paddle, then doctr (both Apache-2.0); whichever imports first runs. Absent is
    a supported case. Precision: docs/reference/identity.md#ocr-precision
    """
    for n in (("paddle", "doctr") if name == "auto" else (name,)):
        try:
            return _OCR_BACKENDS[n]()
        except Exception:  # not installed, or model download failed
            continue
    return None


def _paddle():
    from paddleocr import PaddleOCR  # noqa: PLC0415

    eng = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

    def read(crop):
        res = eng.ocr(crop, cls=False)
        best = ("", 0.0)
        for line in (res[0] or []) if res else []:
            txt, conf = line[1]
            if conf > best[1]:
                best = (txt, float(conf))
        return best
    return read


def _doctr():
    """docTR RECOGNITION ONLY: the crop is already the region, and a text detector finds nothing
    in it. identity-evidence (not published)#evidence-ocr"""
    from doctr.models import recognition_predictor  # noqa: PLC0415

    rec = recognition_predictor(pretrained=True)

    def read(crop):
        out = rec([crop])
        if not out:
            return ("", 0.0)
        txt, conf = out[0]
        return (str(txt), float(conf))
    return read


_OCR_BACKENDS = {"paddle": _paddle, "doctr": _doctr}


#: Vote weight of a CHEST read against a back read (1.0): less than a back, more than nothing,
#: because facing is a velocity proxy. identity-evidence (not published)#evidence-ocr
FRONT_WEIGHT = 0.45


def facing_away(track: list[dict], i: int, cam_xy, cal, span: int = 10) -> int:
    """Is this player showing their BACK to the camera? +1 yes, -1 no, 0 unknown.

    The sign of (displacement . away-from-camera), in pitch metres, between detections
    i - span and i + span of `track` (`span` counts detections, not frames). cam_xy: the
    camera centre in pitch metres, or None. cal: needs to_pitch. 0 when cam_xy is None, the
    window is empty, or the displacement is under the stand-still distance in the body.
    docs/reference/identity.md#ocr; evidence: identity-evidence (not published)#evidence-facing

    ponytail: velocity as a proxy for facing. It is wrong for a backpedalling
    defender or a player shuffling sideways, and it cannot see a player turning
    on the spot. The upgrade path is a back/front classifier over the same crop
    (the role model in rolenet.py is the same shape).
    """
    if not (0 <= i < len(track)) or cam_xy is None:
        return 0
    j0, j1 = max(0, i - span), min(len(track) - 1, i + span)
    if j1 <= j0:
        return 0
    a = cal.to_pitch(np.array([[track[j0]["x"], track[j0]["y"]]], float))[0]
    b = cal.to_pitch(np.array([[track[j1]["x"], track[j1]["y"]]], float))[0]
    v = b - a
    if float(np.hypot(*v)) < 0.5:
        return 0
    return 1 if float(v @ ((a + b) / 2 - np.asarray(cam_xy, float))) > 0 else -1


def scan_video(video_path: str, by_id: dict[int, list[dict]], ocr=None,
               samples_per_track: int = 60, cal=None) -> tuple[dict, dict]:
    """One decode pass -> (votes {track_id: [(number, weighted conf)]}, report).

    by_id: detections in the detector's canvas px. Frames go through cal.undistort_frame
    (when the calibration has one) before cropping, so the crop is taken on the same pixels.
    Per track, the best `samples_per_track` sightings by (facing_away, box h); with no `h`
    recorded, spread evenly over the track. report: crops_attempted, crops_legible,
    legible_frame_rate, ocr_backend, ocr_reads, ocr_reads_facing_away, and `warnings` only
    when degraded. Raises SystemExit when the video cannot be opened.
    docs/reference/identity.md#ocr
    """
    # detections are on the flattened canvas on remap jobs; a no-op elsewhere.
    # identity-evidence (not published)#evidence-crop
    flatten = getattr(cal, "undistort_frame", None) if cal is not None else None

    cam_xy, warnings = None, []
    if cal is not None:
        try:
            import sceneprior  # noqa: PLC0415

            probe = cv2.VideoCapture(video_path)
            ok_p, frame0 = probe.read()
            probe.release()
            if not ok_p:
                raise ValueError("could not read a frame to size the canvas")
            # the canvas size, not the upload's: cal.H is in canvas space
            if flatten is not None:
                frame0 = flatten(frame0)
            h, w = frame0.shape[:2]
            # a calibration that knows where its camera is says so; recovering one from a stand-in H
            # on a panorama would put it somewhere it is not (calibrators.SphericalCamera)
            own = getattr(cal, "camera_centre_m", None)
            cam_xy = (np.array(own[:2], float) if own is not None
                      else np.array(sceneprior.camera_from_homography(cal.H, (w, h)).centre_m[:2]))
        except Exception as e:                  # noqa: BLE001 - never fatal, never silent
            cam_xy = None
            warnings.append(f"facing gate off: camera probe failed ({type(e).__name__}: {e})")

    wanted: dict[int, list[tuple[int, float, float, int,
                                 float | None, float | None]]] = defaultdict(list)
    for tid, dets in by_id.items():
        scored = []
        # `h` is the box height where the tracker recorded one. Older tracks.json
        # files carry only the foot point: then the candidates go in every-stride-th
        # first, so the stable sort below takes ties evenly across the track rather
        # than the first samples_per_track in time.
        stride = max(1, len(dets) // samples_per_track)
        order = (range(len(dets)) if any(d.get("h") for d in dets)
                 else sorted(range(len(dets)), key=lambda i: (i % stride, i)))
        for i in order:
            d = dets[i]
            fa = facing_away(dets, i, cam_xy, cal) if cal else 0
            h = float(d.get("h") or 0.0)
            scored.append((fa, h, i, d, fa))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        for _, _, i, d, fa in scored[:samples_per_track]:
            wanted[d["f"]].append((tid, d["x"], d["y"], fa,
                                   d.get("h"), d.get("w")))

    # Neighbour distances need everyone in the frame, not just the sampled track.
    all_feet: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for tid, dets in by_id.items():
        for d in dets:
            if d["f"] in wanted:
                all_feet[d["f"]].append((tid, d["x"], d["y"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fw = fh = 0   # taken from the first FLATTENED frame, below

    votes: dict[int, list[tuple[int, float]]] = defaultdict(list)
    attempted = legible_n = read_n = facing_n = 0
    f = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        if f not in wanted:
            continue          # flattening a frame nobody sampled is pure cost
        if flatten is not None:
            frame = flatten(frame)
        if not fw:
            fh, fw = frame.shape[:2]
        for tid, x, y, facing, box_h, box_w in wanted.get(f, ()):
            box = torso_box(x, y, fw, fh, box_h, box_w)
            crop = frame[box[1]:box[3], box[0]:box[2]]
            if crop.size == 0:
                continue
            attempted += 1
            # Exclude self by track id, not by position: two players standing on
            # the same pixel is exactly the occlusion this is looking for.
            others = [np.hypot(x - ox, y - oy) for ot, ox, oy in all_feet[f]
                      if ot != tid]
            if not legible(crop, box, min(others) if others else 1e9):
                continue
            legible_n += 1
            if ocr is None:
                continue
            txt, conf = ocr(crop)
            num = _as_jersey(txt)
            if num is not None:
                # a chest read is down-weighted, never dropped (FRONT_WEIGHT)
                votes[tid].append((num, conf * (1.0 if facing >= 0 else FRONT_WEIGHT)))
                read_n += 1
                facing_n += facing == 1
    cap.release()

    report = {
        "crops_attempted": attempted,
        "crops_legible": legible_n,
        # the heuristic's acceptance rate, not the true readable rate
        "legible_frame_rate": round(legible_n / attempted, 4) if attempted else 0.0,
        "ocr_backend": "none" if ocr is None else "present",
        "ocr_reads": read_n,
        "ocr_reads_facing_away": facing_n,
        **({"warnings": warnings} if warnings else {}),   # absent on a clean run
    }
    return dict(votes), report


def _as_jersey(txt: str) -> int | None:
    s = "".join(c for c in (txt or "") if c.isdigit())
    if not s or len(s) > 2:
        return None
    n = int(s)
    return n if 0 < n <= 99 else None


def vote_number(votes: list[tuple[int, float]]) -> tuple[int | None, float]:
    """Vote over a whole MERGED tracklet, never per frame. -> (number | None, confidence 0-1).

    Confidence is the winner's share of the confidence mass, discounted until
    there are MIN_OCR_VOTES reads.
    """
    if not votes:
        return None, 0.0
    weight: dict[int, float] = defaultdict(float)
    count: dict[int, int] = defaultdict(int)
    for n, c in votes:
        weight[n] += max(c, 1e-3)
        count[n] += 1
    num = max(weight, key=weight.get)
    share = weight[num] / sum(weight.values())
    return num, round(share * min(1.0, count[num] / MIN_OCR_VOTES), 3)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class IdentityStrategy(ABC):
    """tracks.json -> {track_id: PlayerId}. Every track gets an entry.

    Unassigned tracks come back with number=None rather than being dropped, so
    the correction UI knows what to ask about instead of guessing.
    """

    name = "base"

    def __init__(self, **_):
        self.report: dict = {}

    @abstractmethod
    def assign(self, tracks: dict, video_path: str | None = None) -> dict[int, PlayerId]:
        ...


class ManualIdentity(IdentityStrategy):
    """A human said so: `mapping` {track_id: number | dict | PlayerId}. Needs no video or OCR."""

    name = "manual"

    def __init__(self, mapping: dict | None = None, roster: Roster | None = None,
                 match_id: str = "match", **kw):
        super().__init__(**kw)
        self.mapping = {int(k): v for k, v in (mapping or {}).items()}
        self.roster = roster
        self.match_id = match_id

    def assign(self, tracks, video_path=None):
        by_id = by_track(tracks)
        teams = {t: d[0].get("team", "unknown") for t, d in by_id.items()}
        out = {}
        for tid, team in teams.items():
            v = self.mapping.get(tid)
            out[tid] = self._resolve(v, team) if v is not None else PlayerId(team=team)
            if self.roster is not None and out[tid].player_id:
                self.roster.bind(self.match_id, tid, out[tid].player_id, 1.0, "manual")
        self.report = {"strategy": self.name, "tracks": len(out),
                       "assigned": sum(p.assigned for p in out.values())}
        return out

    def _resolve(self, v, team: str) -> PlayerId:
        if isinstance(v, PlayerId):
            return v
        if isinstance(v, dict):
            return PlayerId(team=v.get("team", team), number=v.get("number"),
                            name=v.get("name"), player_id=v.get("player_id"),
                            confidence=1.0, source="manual")
        num = v if isinstance(v, int) else _as_jersey(str(v))
        p = self.roster.by_number(num) if (self.roster and num) else None
        return PlayerId(team=team, number=num, name=p.name if p else None,
                        player_id=p.player_id if p else None,
                        confidence=1.0, source="manual")


class AssistedIdentity(IdentityStrategy):
    """What a job runs: merge, vote sparse OCR per merged group, leave the rest to a human.

    Anything under `min_conf`, a one-digit OCR number, or the less confident of two same-team
    groups voting one number comes back unassigned. `cal` is required for a metre-space
    merge. `roster`/`match_id`: None/"match" on the job path, so no roster is read or written.
    docs/reference/identity.md#entry-point
    """

    name = "assisted"

    def __init__(self, roster: Roster | None = None, match_id: str = "match",
                 min_conf: float = 0.6, ocr: str = "auto", cal=None,
                 max_gap_s: float = MAX_GAP_S, **kw):
        super().__init__(**kw)
        self.roster, self.match_id, self.min_conf = roster, match_id, min_conf
        self.ocr_name, self.cal = ocr, cal
        self.max_gap_s = max_gap_s

    def assign(self, tracks, video_path=None):
        by_id = by_track(tracks)
        fps = float(tracks.get("fps", 25.0))

        votes, scan = {}, {}
        if video_path:
            ocr = get_ocr_backend(self.ocr_name)
            # cal lets each read be weighted by facing (facing_away); without it all count the same
            votes, scan = scan_video(video_path, by_id, ocr, cal=self.cal)
            if ocr is None:     # jobs._try_identity surfaces warnings as identity_note
                scan["warnings"] = [f"no OCR backend loaded ({self.ocr_name!r}): no numbers read",
                                    *scan.get("warnings", [])]

        mapping, mrep = merge_tracklets(by_id, fps, self.cal, self.max_gap_s)
        groups: dict[int, list[int]] = defaultdict(list)
        for tid, g in mapping.items():
            groups[g].append(tid)

        teams = {t: d[0].get("team", "unknown") for t, d in by_id.items()}
        confirmed = self.roster.confirmed(self.match_id) if self.roster else {}

        # One decision per merged group, applied to every member track.
        decided: dict[int, PlayerId] = {}
        for g, members in groups.items():
            team = teams[g]
            pid = next((confirmed[m] for m in members if m in confirmed), None)
            if pid and self.roster:
                p = self.roster.players[pid]
                decided[g] = PlayerId(team, p.number, p.name, p.player_id, 1.0,
                                      "manual", g)
                continue
            num, conf = vote_number([v for m in members for v in votes.get(m, [])])
            decided[g] = PlayerId(team, num, None, None, conf, "ocr" if num else
                                  "unassigned", g)
        self._resolve_roster(decided)
        decided = self._drop_duplicates(decided)

        # A one-digit OCR number is never written; its votes stay in the scan.
        # docs/reference/identity.md#ocr-precision
        one_digit = {g for g, p in decided.items() if p.source in ("ocr", "roster")
                     and p.number is not None and p.number < 10}
        out = {t: replace(decided[g], track_group=g) for t, g in mapping.items()}
        out = {t: (p if p.confidence >= self.min_conf and p.assigned and mapping[t] not in one_digit
                   else PlayerId(team=teams[t], track_group=p.track_group))
               for t, p in out.items()}
        self._persist(out)

        self.report = {"strategy": self.name, **mrep, **scan,
                       "groups": len(groups),
                       "one_digit_withheld": sum(decided[g].confidence >= self.min_conf for g in one_digit),
                       "tracks_assigned": sum(p.assigned for p in out.values()),
                       "tracks_needing_human": sum(not p.assigned for p in out.values()),
                       "groups_needing_human": sum(
                           1 for g in groups if not out[g].assigned)}
        return out

    def _resolve_roster(self, decided: dict[int, PlayerId]) -> None:
        if not self.roster:
            return
        for g, p in decided.items():
            if p.source == "manual":
                continue  # a human already named this one; don't relabel it a guess
            r = self.roster.by_number(p.number) if p.number else None
            if r:
                decided[g] = replace(p, name=r.name, player_id=r.player_id,
                                     source="roster")

    def _drop_duplicates(self, decided: dict[int, PlayerId]) -> dict[int, PlayerId]:
        """Two groups on one team voting the same shirt number: keep the more confident, hand the
        other to a human. Keyed on (team, number), never the number alone."""
        best: dict[tuple[str | None, int], int] = {}
        for g, p in decided.items():
            if p.number is None:
                continue
            key = (p.team, p.number)
            cur = best.get(key)
            if cur is None or decided[cur].confidence < p.confidence:
                best[key] = g
        return {g: (p if p.number is None or best[(p.team, p.number)] == g
                    else PlayerId(team=p.team, track_group=p.track_group))
                for g, p in decided.items()}

    def _persist(self, out: dict[int, PlayerId]) -> None:
        """Write machine bindings into the roster so profiles join across matches.

        Human bindings are never overwritten - `confirmed()` already fed them in
        above, and a later re-run must not downgrade them to a guess. The caller
        owns Roster.save(); this only mutates.
        """
        if not self.roster:
            return
        confirmed = self.roster.confirmed(self.match_id)
        for tid, p in out.items():
            if p.player_id and tid not in confirmed:
                self.roster.bind(self.match_id, tid, p.player_id, p.confidence,
                                 p.source)


class AutoIdentity(AssistedIdentity):
    """A baseline to measure, not to ship: every OCR proposal at any confidence, and the
    remaining groups filled with placeholder numbers at confidence 0, source "auto"."""

    name = "auto"

    def __init__(self, **kw):
        kw.setdefault("min_conf", 0.0)
        super().__init__(**kw)

    def assign(self, tracks, video_path=None):
        out = super().assign(tracks, video_path)
        groups: dict[int, list[int]] = defaultdict(list)
        for tid, p in out.items():
            groups[p.track_group].append(tid)

        used = defaultdict(set)
        for g, members in groups.items():
            p = out[members[0]]
            if p.number:
                used[p.team].add(p.number)
        for g in sorted(groups):  # sorted = deterministic, so runs are comparable
            members = groups[g]
            p = out[members[0]]
            if p.assigned:
                continue
            n = next(i for i in range(1, 100) if i not in used[p.team])
            used[p.team].add(n)
            # confidence stays 0: this is a slot, not a read. Anything consuming
            # it must be able to tell a guess from a number we actually saw.
            filled = PlayerId(p.team, n, None, None, 0.0, "auto", g)
            for t in members:
                out[t] = filled
        self.report["strategy"] = self.name
        self.report["tracks_assigned"] = sum(p.assigned for p in out.values())
        self.report["auto_filled_groups"] = sum(
            1 for g, m in groups.items() if out[m[0]].source == "auto")
        return out


_STRATEGIES = {"manual": ManualIdentity, "assisted": AssistedIdentity,
               "auto": AutoIdentity}


def get_identity_strategy(name: str, **kw) -> IdentityStrategy:
    if name not in _STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; have {sorted(_STRATEGIES)}")
    return _STRATEGIES[name](**kw)


# ---------------------------------------------------------------------------

def _selfcheck_no_appearance() -> None:
    """Fail if an appearance or body-descriptor term comes back into the merge.

    Reads the CODE of the merge path only (comments and docstrings tokenised away).
    Rule: internal note `no-biometric-templates-in-the-product`; history:
    identity-evidence (not published)#evidence-appearance
    """
    import dataclasses  # noqa: PLC0415 - self-check only
    import inspect      # noqa: PLC0415
    import io           # noqa: PLC0415
    import textwrap     # noqa: PLC0415
    import tokenize     # noqa: PLC0415

    watched = (Tracklet, _build, _pair_cost, _accepted_pairs, _mutual_best,
               merge_tracklets, scan_video, AssistedIdentity.__init__,
               AssistedIdentity.assign)
    banned = ("torso_hist", "calchist", "embedding", "appearance", "descriptor",
              "reid", "siglip", "osnet", "gait")
    for obj in watched:
        src = textwrap.dedent(inspect.getsource(obj))
        code = " ".join(
            t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type not in (tokenize.COMMENT, tokenize.STRING)).lower()
        hit = [b for b in banned if b in code]
        assert not hit, (
            f"{getattr(obj, '__qualname__', obj)} reintroduces {hit} into the "
            f"identity merge - see appearance-term-decision (not published) before "
            f"adding any appearance term back")

    field_ban = {"emb", "embedding", "hist", "histogram", "descriptor",
                 "feature", "appearance"}
    bad = [f.name for f in dataclasses.fields(Tracklet) if f.name in field_ban]
    assert not bad, f"Tracklet carries a per-person vector again: {bad}"

    # And the merge must offer no channel for one: a caller cannot pass what the
    # signature does not take. Substrings, not exact names - and NOT bare "emb",
    # which is inside "members".
    for fn in (merge_tracklets, _pair_cost, _build, AssistedIdentity.__init__):
        bad = [p for p in inspect.signature(fn).parameters
               if any(b in p.lower() for b in
                      ("embedding", "appearance", "hist", "descriptor", "reid"))]
        assert not bad, f"{fn.__qualname__} takes {bad}"


def _synth(tid, team, x0, y0, vx, vy, f0, f1, step=5, fps=25.0):
    return [{"f": f, "id": tid, "team": team, "jersey": None,
             "x": x0 + vx * (f - f0) / fps, "y": y0 + vy * (f - f0) / fps}
            for f in range(f0, f1 + 1, step)]


def _demo() -> None:
    """Ground truth in, merges out. The second case is the one that matters.

    A wrong merge does not crash anything - it silently doubles one player's
    distance and deletes another's. That is the failure this asserts against.
    """
    fps = 25.0

    # 0. FACING. Running AWAY from the camera shows the back; the SIGN must be right
    # (identity-evidence (not published)#evidence-facing).
    class _C:
        H = np.array([[0.125, 0.0, 0.0], [0.0, 0.125, 0.0], [0.0, 0.0, 1.0]])

        @staticmethod
        def to_pitch(pts):
            return np.asarray(pts, float) / 8.0

    cam = (52.5, 90.0)                      # camera beyond the near touchline
    away = [{"f": k, "x": 400.0, "y": 8.0 * (60.0 - 1.0 * k)} for k in range(21)]
    toward = [{"f": k, "x": 400.0, "y": 8.0 * (20.0 + 1.0 * k)} for k in range(21)]
    still = [{"f": k, "x": 400.0, "y": 8.0 * 34.0} for k in range(21)]
    assert facing_away(away, 10, cam, _C()) == 1, "running to the far side is not 'away'"
    assert facing_away(toward, 10, cam, _C()) == -1, "running at the camera is not 'toward'"
    assert facing_away(still, 10, cam, _C()) == 0, "a standing player was given a facing"
    assert facing_away(away, 10, None, _C()) == 0, "it guessed with no camera"

    # ...and the SELECTION must prefer big, back-turned sightings over even spacing.
    big_back = [{"f": 100 + k, "x": 400.0, "y": 8.0 * (60.0 - k), "h": 90.0}
                for k in range(21)]
    small_front = [{"f": 200 + k, "x": 400.0, "y": 8.0 * (20.0 + k), "h": 20.0}
                   for k in range(21)]
    dets = big_back + small_front
    picked = []
    for i, d in enumerate(dets):
        picked.append((facing_away(dets, i, cam, _C()), float(d.get("h") or 0.0), i))
    picked.sort(key=lambda t: (-t[0], -t[1]))
    top = [dets[i] for _, _, i in picked[:10]]
    assert all(d["h"] == 90.0 for d in top), \
        f"selection did not prefer the big back-turned crops: {[d['h'] for d in top]}"

    # 1. One player, tracklets 1 and 2 split by a 1s occlusion. Must join.
    a1 = _synth(1, "home", 20.0, 30.0, 2.0, 0.0, 0, 250)   # ends (40, 30) at t=10
    a2 = _synth(2, "home", 42.0, 30.0, 2.0, 0.0, 275, 500)  # resumes 2m on, 1s later
    # A player 30m away must never be pulled in - that is 30m in 1s.
    far = _synth(3, "home", 60.0, 60.0, 0.0, 0.0, 275, 500)
    tracks = {"fps": fps, "detections": a1 + a2 + far}
    m, rep = merge_tracklets(by_track(tracks), fps)
    assert m[1] == m[2], f"occlusion split not rejoined: {m}"
    assert m[3] != m[1], f"distant player merged in: {m}"
    assert (rep["tracklets_before"], rep["tracklets_after"]) == (3, 2), rep

    # 2. THE failure mode. Two same-team players cross within 0.2m during a 0.4s
    # occlusion. Both continuations are physically plausible, so merging either
    # is a coin flip - and a wrong flip swaps two players' whole stat lines.
    # Correct behaviour is to refuse and leave it for a human.
    p1a = _synth(11, "home", 10.0, 34.0, 2.0, 0.0, 0, 250)      # -> (30, 34)
    p2a = _synth(12, "home", 50.0, 33.8, -2.0, 0.0, 0, 250)     # -> (30, 33.8)
    p1b = _synth(13, "home", 30.8, 34.0, 2.0, 0.0, 260, 500)
    p2b = _synth(14, "home", 29.2, 33.8, -2.0, 0.0, 260, 500)
    cross = {"fps": fps, "detections": p1a + p2a + p1b + p2b}
    mc, _ = merge_tracklets(by_track(cross), fps)
    assert len(set(mc.values())) == 4, f"crossing players were merged: {mc}"

    # 3. The crossing STAYS refused, and no appearance signal may be offered to
    # rescue it. This used to be the test that appearance breaks the tie; the
    # term was removed on 2026-09-20 and the guard below is what replaced it.
    _selfcheck_no_appearance()

    # 4. Opposing teams never merge, however plausible the motion.
    opp = {"fps": fps, "detections": _synth(21, "home", 10.0, 34.0, 2.0, 0.0, 0, 250)
           + _synth(22, "away", 30.8, 34.0, 2.0, 0.0, 260, 500)}
    mo, _ = merge_tracklets(by_track(opp), fps)
    assert mo[21] != mo[22], mo

    # 5. Manual always works, with no video and no OCR anywhere.
    man = get_identity_strategy("manual", mapping={1: 9, 2: 9, 3: 4})
    ids = man.assign(tracks)
    assert ids[1].number == 9 and ids[1].confidence == 1.0, ids[1]
    assert set(ids) == {1, 2, 3}, ids

    # 6. Assisted with no video: it must merge, assign nothing, and say so.
    ass = get_identity_strategy("assisted")
    ids = ass.assign(tracks)
    assert all(not p.assigned for p in ids.values()), ids
    assert ass.report["tracks_needing_human"] == 3, ass.report
    assert ids[1].track_group == ids[2].track_group, ids

    # 7. Auto fills every group so downstream sees a squad - at zero confidence,
    # because nothing was actually read.
    au = get_identity_strategy("auto")
    ids = au.assign(tracks)
    assert all(p.assigned for p in ids.values()), ids
    assert ids[1].number == ids[2].number != ids[3].number, ids
    assert all(p.confidence == 0.0 for p in ids.values()), ids

    # 8. Voting is over the merged tracklet, and a single lucky read is not proof.
    assert vote_number([(9, 0.9)])[1] < 0.4, vote_number([(9, 0.9)])
    assert vote_number([(9, .8), (9, .8), (9, .8), (6, .9)])[0] == 9
    assert _as_jersey("O9") == 9 and _as_jersey("123") is None

    # 9. Roster: the cross-match join is the whole reason the file exists.
    import tempfile, os  # noqa: PLC0415 - self-check only
    path = os.path.join(tempfile.mkdtemp(), "roster.json")
    r = Roster(team="D3 Home", season="2026")
    p = r.add(number=9, name="A Player")
    r.bind("match-01", 1, p.player_id, 1.0, "manual")
    r.save(path)
    r2 = Roster.load(path)
    r2.add(number=17, name="A Player")          # changed shirt, same human
    r2.bind("match-02", 7, p.player_id, 1.0, "manual")
    assert r2.tracks_for(p.player_id) == [("match-01", 1), ("match-02", 7)], \
        r2.tracks_for(p.player_id)
    assert r2.player_for("match-02", 7).number == 17
    assert r2.confirmed("match-01") == {1: p.player_id}

    # 10. A confirmed binding beats anything the pipeline decides.
    r2.bind("match-03", 1, p.player_id, 1.0, "manual")
    ids = AssistedIdentity(roster=r2, match_id="match-03").assign(tracks)
    assert ids[1].number == 17 and ids[1].source == "manual", ids[1]
    assert ids[2].number == 17, "merged group must inherit the human's answer"

    # 11. A one-digit OCR read is withheld, a two-digit one written; and a missing OCR backend is a
    # warning in the report, not a silent zero. The scan is stubbed: no video here.
    g = globals()
    real = g["scan_video"], g["get_ocr_backend"]
    g["get_ocr_backend"] = lambda name="auto": None
    g["scan_video"] = lambda v, b, o, cal=None: ({1: [(7, 1.0)] * 5, 3: [(23, 1.0)] * 5}, {})
    try:
        s = AssistedIdentity()
        ids = s.assign(tracks, "stub.mp4")
    finally:
        g["scan_video"], g["get_ocr_backend"] = real
    assert ids[1].number is None and ids[2].number is None, ids
    assert ids[3].number == 23 and ids[3].source == "ocr", ids[3]
    assert s.report["one_digit_withheld"] == 1, s.report
    assert s.report["warnings"][0].startswith("no OCR backend"), s.report

    # 12. With no box height, scan_video samples evenly across the track, not the first 60 in time.
    # Stubbed capture/crop: x carries the frame so the sampled frames can be read back.
    class _Cap:
        n = 0

        def isOpened(self):  # noqa: N802
            return True

        def read(self):
            self.n += 1
            return self.n <= 600, np.zeros((8, 8, 3), np.uint8)

        def release(self):
            pass
    seen = []
    real = g["cv2"], g["torso_box"], g["legible"]
    g["cv2"] = type("CV", (), {"VideoCapture": staticmethod(lambda p: _Cap())})
    g["torso_box"] = lambda x, *a, **k: seen.append(x) or (0, 0, 4, 4)
    g["legible"] = lambda *a: False
    try:
        scan_video("stub.mp4", {1: [{"f": f, "x": float(f), "y": 1.0} for f in range(600)]})
    finally:
        g["cv2"], g["torso_box"], g["legible"] = real
    assert len(seen) == 60 and max(seen) >= 500, f"no-h sampling is not spread: {min(seen)}..{max(seen)}"

    print("identity.py ok")


def _measure(tracks_path: str, video_path: str | None, calib: str | None) -> None:
    """Print AssistedIdentity's report for one tracks file (calibration via calibrators.load)."""
    cal = None
    if calib:
        import calibrators  # noqa: PLC0415 - the product's loader: remap/spherical too
        cal = calibrators.load(calib)
    with open(tracks_path) as f:
        tracks = json.load(f)
    s = AssistedIdentity(cal=cal)
    s.assign(tracks, video_path)
    print(json.dumps(s.report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracks", nargs="?", help="tracks.json; omit to run the self-check")
    p.add_argument("-v", "--video", default=None)
    p.add_argument("-c", "--calibration", default=None)
    a = p.parse_args()
    _measure(a.tracks, a.video, a.calibration) if a.tracks else _demo()
