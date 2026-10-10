"""Score TEAM ASSIGNMENT against ground truth, without re-running the detector.

Why this exists
---------------
`model/validation/soccertrack/118575_1st.tracks.json.gz` carries a per-detection
`team` field that this repo had never read. For this half the truth is exactly
11 v 11 and 793,111 home / 793,098 away detections - a ratio of 1.0000. The
shipped pipeline produces 2.50, and the demo report reads home 73.8 km vs away
27.2 km. Two real teams are within ~10% of each other, so that is a systematic
bias in `track.assign_teams`, not noise, and until now nothing measured it.

Same shape as `assoc_eval.py`: replay a stage against cached intermediates and
match predictions to GT in METRES with the same 2.0 m gate. The difference is
that team assignment needs PIXELS - `dets.cache.json` holds boxes only and no
colour was ever cached - so this file samples frames from the video, remaps
them exactly as `track.run` does, and recomputes the kit crops. That sampling
is cached to an .npz so the loop after the first build is seconds, not minutes.

Frames are sampled EVENLY ACROSS THE HALF, never the first N: the teams swap
ends, the sun moves, and the first two minutes are not the half.

BOTH LABEL MAPPINGS ARE SCORED AND THE BETTER ONE IS TAKEN. 'home' out of a
clustering is a cluster id, not an identity - `track.assign_teams` itself says
so - and scoring one arbitrary mapping would report 20% for a labelling that is
80% right. This is the standard convention for unsupervised cluster evaluation
(it is what the Hungarian/best-map accuracy does), but it is stated here
because it IS a free parameter and a reader deserves to know it was spent.

What it found
-------------
300 frames spread evenly across the half, 5,998 tracked detections, 4,101 of
them matched to a GT player inside 2 m. `detection_accuracy` is the share of
those matched detections whose team label is right, under the better of the two
label mappings:

    team assignment                          acc    home/away   track acc
    mean Lab + plain 3-means (shipped)      51.6%     2.83        56.0%
    mean Lab, no bogus official cluster     65.0%     1.24        80.0%
    chroma+L, within-cluster metric (now)   70.7%     1.18        93.0%
    SigLIP shirt crops, teamembed.py        67.8-69.4%  0.76-1.02  89-90%
    ---- best any PER-TRACK label can do    73.1%                100.0%

Repeated on an independent 211-frame sample: 70.8% / 1.25 / 91.8%, so the
result is not a property of which frames were picked.

The 51.6% row needed the OLD `track.py` to reproduce; both halves of that
function have since changed. `--method mean_lab` puts the retired readout
through the current clustering and scores 56.4%, which is the honest
attribution: the readout was worth ~14 points and the clustering metric ~5,
and neither alone gets past 57%.

Read the last row before celebrating any of the others. `stats.py:550` gives a
whole track the team of its FIRST detection, so team assignment is a per-track
decision, and on this job the stitched tracks weld players from opposite sides
together - the largest track in the sample is 149 detections, 57 home and 92
away. Labelling every track by the majority of its own detections, which is the
best any labeller could ever do, still gets only 73.1% of detections right. The
colour work is now within 2.4 points of that. EVERYTHING LEFT IS ASSOCIATION,
and `assoc_eval.py` is the file for it.

What it can NOT flatter
-----------------------
Accuracy here goes DOWN when the clustering degrades. A labelling that collapses
to one bank scores ~50% (GT is balanced 11/11) and shows an infinite ratio; a
labelling that calls everything 'official' scores 0. There is no way to raise
this number by merging things together, which is the failure mode that has
already fooled this analysis twice.

TEN MATCHES: was 69.7% a property of the method, or of 118575?
--------------------------------------------------------------
Everything above was ONE match. `model/validation/st2_team_gt.py` joins ST2's own
team truth onto the MOT ground truth for all ten STC2025 clips, and `--clips`
scores the shipped labeller against it. 217,954 matched detections, IoU 0.5,
1 clip frame in 5:

    match     det acc   ceiling   det/ceil   track acc   matched
    117092     0.4657    0.7444     0.626      0.5000      21,127
    117093     0.8434    0.8961     0.941      0.9423      20,982
    118575     0.8819    0.9296     0.949      0.9388      20,487
    118576     0.7109    0.8652     0.822      0.8727      20,779
    118577     0.8416    0.9213     0.914      0.7931      22,457
    118578     0.8389    0.9109     0.921      0.8367      22,252
    128057     0.7936    0.9003     0.882      0.8824      23,086
    128058     0.8035    0.8871     0.906      0.9444      22,106
    132831     0.8921    0.8936     0.998      0.9556      21,927
    132877     0.8115    0.8604     0.943      0.8372      22,751
    ------    POOLED 0.7894 over 217,954 matched detections; ceiling 0.8813

NO, 69.7% WAS NOT REPRESENTATIVE - and not because the method is better than
that. It is not the same measurement. The registered 0.6967 is the full first
half of 118575 through a product job, matched in metres; 118575's own clip
scores 0.8819. The gap is almost entirely the CEILING: the product job's
stitched tracks weld opposite-side players together and cap any per-track label
at 0.7313, while the clip's tracks cap it at 0.9296. Divide it out and the two
land on top of each other - 0.6967/0.7313 = 0.953 and 0.8819/0.9296 = 0.949.
The labeller reaches ~95% of whatever the association leaves it, in both.

SPREAD: 0.4657 to 0.8921, sd 0.124 over ten matches. One match, 117092, is the
outlier and it fails visibly without any ground truth: its labelling collapses
to a 7.40 home/away ratio and it puts 4,787 of 21,127 matched real players in
the bogus 'official' cluster. It is also the only clip on a different rig
(3840x1906; the other nine are 4096x1080 or 1084). Drop it and the range is
0.711-0.892. The pooled 0.7894 is the honest headline; the registered 0.6967 is
a correct number for a different question.

STABLE, not a matcher artefact: pooled reads 0.7885 / 0.7894 / 0.7939 at IoU
0.3 / 0.5 / 0.7, 0.7895 at every frame instead of every fifth, and 0.7888 on
the UNSTITCHED c03 tracks. The per-match ordering does not move.

N9 (2026-09-12): THE SAME TEN-MATCH PATH, AT THE CONFIG THE PRODUCT RUNS
    Everything above is `.c03s` - conf 0.3, stride 1, `--roles`, no calibration,
    HeightScale stitch. `app/server/jobs.py` runs none of that, so 0.7894 is a team
    number for a tracker nobody ships (model-map (not published) §3). `--variant p05`
    scores the product's own tracks, and adds the two columns 0.7894 hides.

    PROTOCOL, fixed before the run:

    population   the NINE clips with product tracks on disk. 132831 is excluded
                 - the product's clicked calibration cannot be produced for it
                 at all (st2_product.py) - so this pool is nine, not ten, and
                 132831 is one of the two clips `.c03s` was TUNED on, which
                 makes the drop favourable to `.c03s` in the comparison. Stated
                 before the run, not after.
    held fixed   coreml r576 t1024, conf 0.5, stride 2, clicked calibration ->
                 scene prior and metric stitch, no RoleNet. The team labels are
                 read straight out of the tracks file, so this is the shipped
                 `assign_teams` on the shipped association, not a re-clustering.
    frames       `--every 10`. The product tracks EVEN frames only (stride 2), so
                 the default `--every 5` samples 600 frames our tracker never ran
                 on and reads as a 50% recall collapse that is pure cadence.
                 Every 10 is "one in five of the frames the product produced",
                 which is what `--every 5` means for `.c03s`. Both variants are
                 re-run at `--every 10` so config and cadence are not confounded.
    matcher      unchanged: Hungarian on IoU >= 0.5 against ST2's MOT boxes,
                 team truth from `st2_team_gt`, gated by its own `_ok`.
    half-time    ST2's `team` flips at half time in the EVENT table
                 (st2-raw-player-nodes-is-the-real-feed (internal note)). It
                 cannot bite here: the side label comes from
                 `_tracker_box_metadata.xml`, which is a roster property and not
                 a per-period one, and each clip lies inside a single period
                 (`period` in the join). Checked, not assumed.
    metrics      pooled detection accuracy as before, PLUS
                 `phantom_frac` - the share of our sampled boxes that match no
                 GT box AND carry a home/away label. Those never enter the
                 accuracy and they are the ones that invent a player on a side
                 total. `unscored_frac` is the wider share that never enters the
                 accuracy at all (model-map §3's 0.2075); the difference between
                 the two is boxes the labeller REFUSED ('official'/'unknown'),
                 which cost a team total nothing.
                 `detection_accuracy_as_named` - the same accuracy WITHOUT the
                 per-match best-of-two mapping, i.e. with 'home' meaning home.
                 `named_correctly` per clip and `matches_named_correctly` over
                 the pool. Floor: `assign_teams` names the larger cluster 'home',
                 so chance is a coin flip per clip and 4-5 of 9 is chance.
    pass rule    fixed before running, and it is about the FIGURE, not the
                 method: the current 0.7894 is reportable as a product number
                 iff the product config lands within its stated per-match sd
                 (0.124) of it AND naming is better than chance. If naming is at
                 chance, `detection_accuracy` is a CLUSTERING score and must not
                 be printed next to the words home and away anywhere in the
                 product - which is a documentation fix, not a model one.

Usage
-----
    python model/validation/team_eval.py --build              # sample video, ~2 min
    python model/validation/team_eval.py                      # score current method
    python model/validation/team_eval.py --method mean_lab   # the readout it replaced
    python model/validation/team_eval.py --clips              # ALL TEN, ~3 s
    python model/validation/team_eval.py --seq 117092         # one clip
    python model/validation/team_eval.py --clips --variant p05 --every 10 \
        --seqs 117092,117093,118575,118576,118577,118578,128057,128058,132877   # N9
    python model/validation/team_eval.py --selfcheck
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assoc_eval  # noqa: E402 - the gate and the per-frame match, imported, never copied
import calibrators  # noqa: E402
import track as trackmod  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")
VIDEO = os.path.join(ROOT, "data/footage/118575_panorama_1st_half.mp4")
CROPS = os.path.join(JOB, "team_eval.crops.npz")

# The same gate, and now literally the same value: assoc_eval owns it, with the
# reason (the calibration's own residual is around a metre, so anything tighter
# scores the calibration).
GATE_M = assoc_eval.GATE_M
BOX = 32  # cached crop is the whole box at BOXxBOX; the torso is rows 0.2-0.5


def load_gt(path: str = GT) -> dict[int, dict[str, tuple[np.ndarray, str]]]:
    """frame -> {gt id: (xy metres, 'home'|'away')}. Frame numbers rebased to 0."""
    d = json.load(gzip.open(path))
    rows = d["detections"]
    base = min(r["f"] for r in rows)
    per_frame: dict[int, dict[str, tuple[np.ndarray, str]]] = collections.defaultdict(dict)
    for r in rows:
        per_frame[r["f"] - base][str(r["id"])] = (np.array([r["x"], r["y"]]), r["team"])
    return per_frame


def build(n_frames: int = 300, job: str = JOB, video: str = VIDEO,
          out: str = CROPS, tracks: str = "tracks.stitched.json") -> str:
    """Sample frames across the half and cache one crop per tracked detection.

    The tracks file records the FOOT point only (box centre-x, box bottom-y), so
    the full box is recovered by joining back to `dets.cache.json` on that point
    - which is exact to 2e-4 px, the float32 rounding the tracks file went
    through. No re-detection, no re-association: this measures team assignment
    on the association the job actually shipped.
    """
    import cv2  # noqa: PLC0415

    cal = calibrators.load(f"{job}/calibration.json")
    cache = json.load(open(f"{job}/dets.cache.json"))["frames"]
    dets = json.load(open(f"{job}/{tracks}"))["detections"]
    by_frame: dict[int, list] = collections.defaultdict(list)
    for d in dets:
        by_frame[d["f"]].append(d)

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(0, total - 1, n_frames).astype(int)

    fs, tids, feet, crops, feats = [], [], [], [], []
    for k, f in enumerate(picks):
        rows = by_frame.get(int(f))
        boxes = cache.get(str(int(f)))
        if not rows or not boxes:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, raw = cap.read()
        if not ok:
            continue
        frame = cal.undistort_frame(raw)
        B = np.array([[(b["xyxy"][0] + b["xyxy"][2]) / 2, b["xyxy"][3]] for b in boxes])
        for r in rows:
            j = int(np.hypot(*(np.array([r["x"], r["y"]]) - B).T).argmin())
            if np.hypot(*(np.array([r["x"], r["y"]]) - B[j])) > 0.01:
                continue  # no box for this track this frame; drop rather than guess
            x1, y1, x2, y2 = (int(v) for v in boxes[j]["xyxy"])
            c = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if c.size == 0:
                continue
            fs.append(int(f))
            tids.append(int(r["id"]))
            feet.append([r["x"], r["y"]])
            crops.append(cv2.resize(c, (BOX, BOX), interpolation=cv2.INTER_AREA))
            # The PRODUCTION feature, at full resolution, so "current accuracy"
            # is the shipped function's number and not the resized crop's.
            feats.append(trackmod.team_by_shirt(frame, (x1, y1, x2, y2)))
        if k % 50 == 0:
            print(f"  frame {f} ({k}/{len(picks)})  {len(fs)} crops", flush=True)
    cap.release()
    # REFUSE to write an empty cache over a good one. cv2.VideoCapture fails
    # SILENTLY on a missing file, so the frame loop collected nothing and the
    # unconditional savez below then overwrote an irreplaceable 16 MB / 160 MB
    # cache with ~1 KB of nothing, printed "0 crops", and exited 0. That is the
    # worst shape a harness can have: it does not merely report success while
    # testing nothing, it destroys the evidence on the way past.
    if not fs:
        raise SystemExit(
            f"collected 0 crops from {video!r} - refusing to overwrite {out!r}. "
            "The capture opened empty (a missing or unreadable video reports no "
            "error), so there is nothing to cache.")
    np.savez_compressed(out, f=np.array(fs, np.int32), tid=np.array(tids, np.int32),
                        foot=np.array(feet, np.float32), crop=np.array(crops, np.uint8),
                        feat=np.array(feats, np.float32))
    print(f"{len(fs)} crops over {len(set(fs))} frames -> {out}")
    return out


def features(z, method: str) -> np.ndarray:
    """One vector per cached detection, for whichever readout is under test.

    `shipped` is `track.team_by_shirt` as it stood when the cache was built, at
    FULL resolution. The others are recomputed from the 32x32 crop the cache
    stores, so a new idea can be tried without touching the video again; the
    price is the resize, which `--method chroma` exists to bound (it recomputes
    the shipped readout off the crop and must land within a point of it).
    """
    if method == "shipped":
        return z["feat"].astype(np.float32)
    import cv2  # noqa: PLC0415
    torso = z["crop"][:, int(0.2 * BOX):int(0.5 * BOX)]   # rows 0.2-0.5 of the box
    if method == "mean_lab":     # the RETIRED readout, kept so the loss is reproducible
        out = [cv2.cvtColor(t, cv2.COLOR_BGR2Lab).reshape(-1, 3).mean(0) for t in torso]
        return np.array(out, np.float32)
    if method == "chroma":       # the current readout, off the crop
        out = []
        for t in torso:
            px = t.reshape(-1, 3).astype(np.float32)
            ch = (px / (px.sum(1, keepdims=True) + 1e-6)).mean(0)[:2] * 255.0
            out.append([ch[0], ch[1], cv2.cvtColor(t, cv2.COLOR_BGR2Lab)[:, :, 0].mean()])
        return np.array(out, np.float32)
    if method == "chroma_only":  # drop L, to show that L is load-bearing
        return features(z, "chroma")[:, :2]
    raise SystemExit(f"unknown method {method}")


def score(labels: dict[int, str], z, gt: dict, cal, gate: float = GATE_M) -> dict:
    """Match sampled predictions to GT in metres, then compare team labels.

    The match is `assoc_eval.greedy_match` - that file owns the gate and the
    order. Detections the tracker never produced are not this file's problem -
    they are scored by assoc_eval - so only MATCHED detections enter the accuracy.
    """
    xy = cal.to_pitch(z["foot"].astype(np.float64))
    by_frame: dict[int, list] = collections.defaultdict(list)
    for i, f in enumerate(z["f"]):
        by_frame[int(f)].append(i)

    pairs: list[tuple[int, str]] = []   # (pred label index, gt team)
    pred_lab, gt_team = [], []
    for f, idx in by_frame.items():
        g = gt.get(f, {})
        m = assoc_eval.greedy_match({k: v[0] for k, v in g.items()},
                                    {i: xy[i] for i in idx}, gate)
        for gid, (_gxy, team) in g.items():
            best = m.get(gid)
            if best is None:
                continue
            pred_lab.append(labels.get(int(z["tid"][best]), "unknown"))
            gt_team.append(team)
            pairs.append((int(z["tid"][best]), team))

    pred_lab = np.array(pred_lab)
    gt_team = np.array(gt_team)
    n = len(gt_team)

    def acc_for(mapping: dict[str, str]) -> tuple[float, dict[str, float]]:
        ok = np.array([mapping.get(p) == g for p, g in zip(pred_lab, gt_team)])
        per = {t: float(ok[gt_team == t].mean()) for t in ("home", "away")
               if (gt_team == t).any()}
        return float(ok.mean()) if n else 0.0, per

    a, per_a = acc_for({"home": "home", "away": "away"})
    b, per_b = acc_for({"home": "away", "away": "home"})
    acc, per, flip = (a, per_a, False) if a >= b else (b, per_b, True)

    # The ratio the finding is about, over EVERY sampled detection - not only
    # the matched ones - because that is what the shipped report aggregates.
    all_lab = [labels.get(int(t), "unknown") for t in z["tid"]]
    c = collections.Counter(all_lab)
    hi, lo = (c["away"], c["home"]) if flip else (c["home"], c["away"])
    tracks = sorted(set(int(t) for t in z["tid"]))

    # Per-track accuracy: each track's GT team is the majority of its matched
    # detections. Unmatched tracks have no truth and are excluded, not guessed.
    tgt: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for tid, team in pairs:
        tgt[tid][team] += 1
    m = {"home": "away", "away": "home"} if flip else {"home": "home", "away": "away"}
    t_ok = [m.get(labels.get(tid)) == c2.most_common(1)[0][0] for tid, c2 in tgt.items()]

    # THE CEILING, and it is not this file's to raise. ONE team label per track
    # is the contract stats.py reads (stats.py:550 takes the team of a track's
    # first detection), so no clustering can beat the best possible per-track
    # label: give every track the team of the majority of its own detections and
    # count what that gets right. On the shipped 118575 association that is
    # 73.1%, because the stitched tracks weld players from opposite sides
    # together - the biggest track in the sample is 149 detections split 57/92
    # across the two teams. A team-assignment score close to this number means
    # the colour work is finished and the remaining loss is ASSOCIATION.
    ceiling = (sum(c2.most_common(1)[0][1] for c2 in tgt.values()) / n) if n else 0.0

    return {
        "per_track_ceiling": round(ceiling, 4),
        "matched_detections": n,
        "sampled_detections": int(len(z["tid"])),
        "sampled_frames": len(by_frame),
        "detection_accuracy": round(acc, 4),
        "per_team_accuracy": {k: round(v, 4) for k, v in per.items()},
        "track_accuracy": round(float(np.mean(t_ok)), 4) if t_ok else 0.0,
        "tracks_scored": len(t_ok),
        "tracks_labelled": len(tracks),
        "pred_home_away_ratio": round(hi / lo, 4) if lo else float("inf"),
        "true_home_away_ratio": 1.0,
        "label_counts": dict(c),
        "label_mapping_flipped": flip,
    }


def evaluate(method: str = "shipped", crops: str = CROPS, job: str = JOB,
             assigner=None) -> dict:
    if not os.path.exists(crops):
        raise SystemExit(f"{crops} missing - run: python model/validation/team_eval.py --build")
    z = np.load(crops)
    cal = calibrators.load(f"{job}/calibration.json")
    X = features(z, method)
    colours: dict[int, list] = collections.defaultdict(list)
    for tid, v in zip(z["tid"], X):
        colours[int(tid)].append(v)
    labels = (assigner or trackmod.assign_teams)(dict(colours))
    out = score(labels, z, load_gt(), cal)
    out["method"] = method
    return out


# ==========================================================================
# TEN MATCHES. Everything above this line scores 118575's full first half and
# is untouched, so `api://team.accuracy.detection` = 0.6967 still reproduces.
#
# WHY A SECOND PATH RATHER THAN THE SAME ONE TEN TIMES
#     The path above needs a product JOB (a calibration, a dets cache, the
#     panorama mp4). That exists for 118575 and for nothing else, and building
#     nine more is a week. The STC2025 clips need none of it: our tracks for all
#     ten already sit in model/validation/out/stc2025/, they already carry a per-
#     detection `team`, and the GT is boxes in the same pixel space. So the ten-
#     match comparison is done in PIXELS by IoU rather than in metres by a 2 m
#     gate. That is a different matcher, so the two numbers are not the same
#     measurement - which is exactly why 118575 is scored BOTH ways below and
#     the gap between its two numbers is reported rather than averaged away.
#
# THE LABEL MAPPING IS CHOSEN PER MATCH, and it has to be. 'home' out of
# `track.assign_teams` is a cluster id; the clustering is run independently per
# clip, so match A's 'home' and match B's 'home' are unrelated. Picking one
# mapping for the pool would score half the matches inverted. Correct counts
# are summed under each match's own better mapping - stated because it is a
# free parameter spent ten times instead of once.

STC_TRACKS = os.path.join(ROOT, "model/validation/out/stc2025")
STC_GT = _ST2 + "/mot"
IOU_GATE = 0.5      # st2_team.py's gate, so "matched" means the same thing there
CLIP_EVERY = 5      # 1 frame in 5: ~1,200 frames and ~26,000 GT boxes per match


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,M) IoU between two sets of top-left x,y,w,h boxes."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) -
                 np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) -
                 np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.clip(union, 1e-9, None)


def score_clip(seq: str, variant: str = "c03s", every: int = CLIP_EVERY,
               gate: float = IOU_GATE, gt_join: dict | None = None) -> dict:
    """Team accuracy for one STC2025 clip, against ST2's own team truth.

    Our per-detection `team` is read straight out of the tracks file - these
    clips were tracked by `st2_run.py` with the shipped `assign_teams`, so this
    is the shipped labeller and not a re-clustering. GT team comes from
    `st2_team_gt`, which is the file that makes this possible at all.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import st2_team  # noqa: PLC0415  - to_tlwh lives there and must not fork
    import st2_team_gt  # noqa: PLC0415

    join = (gt_join or st2_team_gt.load())[seq]
    if not st2_team_gt._ok(join):
        raise SystemExit(f"{seq}: the join failed its own gate; refusing to score it")
    gt_team = {int(k): v for k, v in join["team"].items()}

    gt: dict[int, list] = collections.defaultdict(list)
    with open(f"{STC_GT}/{seq}.txt") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 6:
                continue
            f = int(p[0])
            if f % every == 0:
                gt[f].append((int(p[1]), *(float(v) for v in p[2:6])))

    ours: dict[int, list] = collections.defaultdict(list)
    labels: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    n_ours = 0
    for d in json.load(open(f"{STC_TRACKS}/{seq}.{variant}.tracks.json"))["detections"]:
        labels[int(d["id"])][d.get("team") or "unknown"] += 1
        if d["f"] % every == 0:
            ours[d["f"]].append(d)
            n_ours += 1

    pred, truth, pairs = [], [], []
    n_gt = 0
    matched_box: set[tuple[int, int]] = set()   # (frame, index into ours[frame])
    for f, g in gt.items():
        n_gt += len(g)
        o = ours.get(f)
        if not o:
            continue
        G = np.array([[x, y, w, h] for _i, x, y, w, h in g], float)
        Ob = np.array([st2_team.to_tlwh(d) for d in o], float)
        M = _iou(Ob, G)
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] < gate:
                continue
            # Matched is matched, even if the GT id carries no team label: this
            # box is a real player, so it is NOT a phantom and must not land in
            # the phantom denominator. It is still excluded from accuracy.
            matched_box.add((f, i))
            t = gt_team.get(g[j][0])
            if t is None:
                continue
            pred.append(o[i].get("team") or "unknown")
            truth.append(t)
            pairs.append((int(o[i]["id"]), t))

    # N9: THE UNMATCHED BOXES, which the accuracy above never sees. Our boxes
    # that match no GT box at all still carry a team label, and a 'home' or
    # 'away' on one of them puts a player on the pitch who was not there - it
    # flows straight into the side totals stats.py sums. The label breakdown is
    # the point: 'official' and 'unknown' are refusals, and a refusal on a false
    # positive costs a team total nothing.
    unmatched = collections.Counter()
    for f, o in ours.items():
        for i, d in enumerate(o):
            if (f, i) not in matched_box:
                unmatched[d.get("team") or "unknown"] += 1

    pred, truth = np.array(pred), np.array(truth)
    n = len(truth)
    if not n:
        raise SystemExit(f"{seq}: nothing matched at IoU {gate} - not a score")

    def acc_for(m):
        ok = np.array([m.get(p) == t for p, t in zip(pred, truth)])
        return int(ok.sum()), {k: round(float(ok[truth == k].mean()), 4)
                               for k in ("home", "away") if (truth == k).any()}

    a, per_a = acc_for({"home": "home", "away": "away"})
    b, per_b = acc_for({"home": "away", "away": "home"})
    hits, per, flip = (a, per_a, False) if a >= b else (b, per_b, True)
    m = {"home": "away", "away": "home"} if flip else {"home": "home", "away": "away"}

    # Per track: our majority label vs the majority GT team of its matched
    # detections. Unmatched tracks have no truth and are excluded, not guessed.
    tgt: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for tid, t in pairs:
        tgt[tid][t] += 1
    t_ok = [m.get(labels[tid].most_common(1)[0][0]) == c.most_common(1)[0][0]
            for tid, c in tgt.items()]
    ceiling = sum(c.most_common(1)[0][1] for c in tgt.values()) / n

    cnt = collections.Counter(pred)
    hi, lo = (cnt["away"], cnt["home"]) if flip else (cnt["home"], cnt["away"])
    return {
        "seq": seq, "variant": variant,
        "per_track_ceiling": round(ceiling, 4),
        # Accuracy divided by the ceiling: the LABELLER's share, with the
        # association it was handed divided out. Comparing raw accuracies across
        # matches compares two things at once, because the ceiling itself moves
        # 0.74-0.93 across these ten - a clip whose tracks weld two players
        # together caps the labeller before it starts.
        "accuracy_over_ceiling": round(hits / n / ceiling, 4) if ceiling else 0.0,
        "matched_detections": n,
        "gt_detections_sampled": n_gt,
        "our_detections_sampled": n_ours,
        "recall_of_gt": round(n / n_gt, 4) if n_gt else 0.0,
        "sampled_frames": len(gt),
        "detection_accuracy": round(hits / n, 4),
        "detection_hits": hits,
        "per_team_accuracy": per,
        "track_accuracy": round(float(np.mean(t_ok)), 4) if t_ok else 0.0,
        "tracks_scored": len(t_ok),
        "tracks_labelled": len(labels),
        "pred_home_away_ratio": round(hi / lo, 4) if lo else float("inf"),
        "true_home_away_ratio": 1.0,
        "matched_label_counts": dict(cnt),
        "label_mapping_flipped": flip,
        # ---- N9, the two columns the figure above hides ---------------------
        # 1. PHANTOMS. `detection_accuracy` is over matched boxes only, so every
        #    box below is invisible to it. `unscored_frac` is the share of our
        #    sampled boxes that never enter the accuracy at all - the 0.2075 in
        #    model-map §3 - and `phantom_frac` is the part of it that is worse
        #    than merely unscored: no GT box AND a home/away label, i.e. a
        #    player invented on one side's total.
        "our_detections_unmatched": int(sum(unmatched.values())),
        "unmatched_label_counts": dict(unmatched),
        "unscored_frac": round(1 - n / n_ours, 4) if n_ours else 0.0,
        "phantom_frac": round((unmatched["home"] + unmatched["away"]) / n_ours, 4)
        if n_ours else 0.0,
        "phantom_refused_frac": round(
            (unmatched["official"] + unmatched["unknown"]) / n_ours, 4) if n_ours else 0.0,
        # 2. NAMING. `detection_accuracy` takes the better of the two label
        #    mappings, so a clip whose sides are named backwards scores exactly
        #    as well as one named right - naming errors are forgiven by
        #    construction. This is the same accuracy WITHOUT that forgiveness:
        #    'home' means home. `track.assign_teams` calls the larger cluster
        #    'home', so chance here is a coin flip per clip and the useful
        #    statistic is how many of the nine come up heads.
        "detection_accuracy_as_named": round(a / n, 4),
        "detection_hits_as_named": a,
        "named_correctly": not flip,
        "join_cost_px": join["cost_px"], "join_cost2_px": join["cost2_px"],
        "join_frame_agreement": join["frame_agreement"],
    }


def evaluate_clips(variant: str = "c03s", every: int = CLIP_EVERY,
                   gate: float = IOU_GATE, verbose: bool = True,
                   seqs: tuple[str, ...] | None = None) -> dict:
    """All ten matches, per match and pooled. The pooled number is the honest one.

    Pooled accuracy is SUM(hits) / SUM(matched), not the mean of ten rates: the
    matches contribute different numbers of matched detections and a mean of
    rates would weight a match with half the detections equally. Both are
    printed, because if they disagree the weighting is doing work.

    `seqs` exists for N9: the product config has no tracks for 132831 (its ST2
    chain cannot be clicked at all - see st2_product.py), so the ten-match pool
    is a NINE-match pool there. Passing the shorter list explicitly, rather than
    skipping missing files, keeps the population in the caller's hand and in the
    output; `matches` in the result is the number that was actually scored.
    """
    import st2_team_gt  # noqa: PLC0415
    join = st2_team_gt.load()
    rows = []
    for seq in (seqs or st2_team_gt.SEQS):
        r = score_clip(seq, variant, every, gate, join)
        rows.append(r)
        if verbose:
            print(f"{seq}  det {r['detection_accuracy']:.4f} "
                f"({r['detection_hits']}/{r['matched_detections']})  "
                f"track {r['track_accuracy']:.4f} ({r['tracks_scored']} tracks)  "
                f"ceiling {r['per_track_ceiling']:.4f}  "
                f"ratio {r['pred_home_away_ratio']:.2f}  "
                f"recall {r['recall_of_gt']:.3f}  "
                f"phantom {r['phantom_frac']:.3f}  "
                f"named {'ok' if r['named_correctly'] else 'BACKWARDS'}",
                flush=True, file=sys.stderr)
    hits = sum(r["detection_hits"] for r in rows)
    n = sum(r["matched_detections"] for r in rows)
    n_ours = sum(r["our_detections_sampled"] for r in rows)
    accs = [r["detection_accuracy"] for r in rows]
    taccs = [r["track_accuracy"] for r in rows]
    return {
        "variant": variant, "iou_gate": gate, "frame_stride": every,
        "matches": len(rows), "sequences": [r["seq"] for r in rows],
        # ---- N9's two columns, pooled ---------------------------------------
        "pooled_our_detections_sampled": n_ours,
        "pooled_unscored_frac": round(1 - n / n_ours, 4) if n_ours else 0.0,
        "pooled_phantom_frac": round(
            sum(r["unmatched_label_counts"].get("home", 0) +
                r["unmatched_label_counts"].get("away", 0) for r in rows) / n_ours, 4)
        if n_ours else 0.0,
        "pooled_detection_accuracy_as_named": round(
            sum(r["detection_hits_as_named"] for r in rows) / n, 4),
        "matches_named_correctly": sum(r["named_correctly"] for r in rows),
        "pooled_detection_accuracy": round(hits / n, 4),
        "pooled_matched_detections": n,
        "pooled_gt_detections_sampled": sum(r["gt_detections_sampled"] for r in rows),
        "mean_of_per_match_accuracy": round(float(np.mean(accs)), 4),
        "per_match_accuracy_min": round(min(accs), 4),
        "per_match_accuracy_max": round(max(accs), 4),
        "per_match_accuracy_sd": round(float(np.std(accs, ddof=1)), 4),
        "pooled_per_track_ceiling": round(
            sum(r["per_track_ceiling"] * r["matched_detections"] for r in rows) / n, 4),
        "pooled_accuracy_over_ceiling": round(
            hits / sum(r["per_track_ceiling"] * r["matched_detections"] for r in rows), 4),
        "per_match_accuracy_over_ceiling_min": round(
            min(r["accuracy_over_ceiling"] for r in rows), 4),
        "per_match_accuracy_over_ceiling_max": round(
            max(r["accuracy_over_ceiling"] for r in rows), 4),
        "mean_of_per_match_track_accuracy": round(float(np.mean(taccs)), 4),
        "per_match_track_accuracy_min": round(min(taccs), 4),
        "per_match_track_accuracy_max": round(max(taccs), 4),
        "per_match": {r["seq"]: r for r in rows},
    }


def cluster_report(method: str = "shipped", crops: str = CROPS) -> dict:
    """Is k-means separating kits at all? Balance and separation, no labels.

    This is the check that tells "the clustering found the kits and the naming
    convention is wrong" apart from "the clustering never found the kits". They
    look identical from an accuracy number alone.
    """
    import cv2  # noqa: PLC0415
    z = np.load(crops)
    X = np.ascontiguousarray(features(z, method), np.float32)
    gt = load_gt()
    out = {}
    for k in (2, 3):
        # track._kmeans, not cv2's, so the distances printed here are in the
        # metric the pipeline actually clusters in. The two disagree by enough
        # to flip the official test - that was the bug.
        lab, cen = trackmod._kmeans(X, k)
        sizes = sorted(int((lab == j).sum()) for j in range(k))
        d = sorted(round(float(np.linalg.norm(cen[i] - cen[j])), 2)
                   for i in range(k) for j in range(i + 1, k))
        out[f"k{k}"] = {"sizes": sizes, "centre_distances": d,
                        "smallest_share": round(sizes[0] / len(X), 3)}
        if k == 3:
            # The two numbers assign_teams actually gates on. Below 25% and
            # above 0.5 and a third cluster is called an officiating crew.
            out["k3"]["separation_ratio"] = round(d[0] / d[-1], 2)
    # The ceiling this feature can reach: cluster the GT-labelled detections and
    # ask how separable the two truths are in THIS space. If a Fisher ratio near
    # 0 comes back, no clustering of this feature can work and the fix is the
    # feature, not the algorithm.
    xy = None
    from collections import defaultdict
    idx_by_frame = defaultdict(list)
    for i, f in enumerate(z["f"]):
        idx_by_frame[int(f)].append(i)
    cal = calibrators.load(f"{JOB}/calibration.json")
    xy = cal.to_pitch(z["foot"].astype(np.float64))
    truth = np.full(len(X), "", dtype=object)
    for f, idx in idx_by_frame.items():
        g = gt.get(f, {})
        m = assoc_eval.greedy_match({k: v[0] for k, v in g.items()},
                                    {i: xy[i] for i in idx})
        for gid, best in m.items():
            truth[best] = g[gid][1]
    m = truth != ""
    A, B = X[m][truth[m] == "home"], X[m][truth[m] == "away"]
    between = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    within = float((np.linalg.norm(A - A.mean(0), axis=1).mean() +
                    np.linalg.norm(B - B.mean(0), axis=1).mean()) / 2)
    out["truth_separation"] = {
        "n_home": int(len(A)), "n_away": int(len(B)),
        "between_class_distance": round(between, 2),
        "mean_within_class_distance": round(within, 2),
        "ratio": round(between / within, 3) if within else 0.0,
        "home_centre": [round(float(v), 1) for v in A.mean(0)],
        "away_centre": [round(float(v), 1) for v in B.mean(0)],
    }
    return out


def _selfcheck() -> None:
    """The scorer, on a case whose answer is known by construction.

    Four GT players, two per team, standing still. The predictions sit on top of
    them with the team names SWAPPED, which must score 1.0 - that is the whole
    point of trying both mappings. Then one label is corrupted, which must drop
    the number; a scorer that cannot go down is not a scorer.
    """
    class _Cal:
        def to_pitch(self, px):
            return np.asarray(px, float)          # pixels are metres here

    pos = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0)]
    teams = ["home", "home", "away", "away"]
    gt = {f: {str(i + 1): (np.array(p), t) for i, (p, t) in enumerate(zip(pos, teams))}
          for f in range(5)}
    z = {"f": [], "tid": [], "foot": []}
    for f in range(5):
        for i, p in enumerate(pos):
            z["f"].append(f)
            z["tid"].append(i + 1)
            z["foot"].append(p)
    z = {k: np.array(v) for k, v in z.items()}

    swapped = {1: "away", 2: "away", 3: "home", 4: "home"}
    r = score(swapped, z, gt, _Cal())
    assert r["matched_detections"] == 20, r
    assert r["detection_accuracy"] == 1.0, r
    assert r["per_track_ceiling"] == 1.0, r   # pure tracks: nothing to lose here
    assert r["label_mapping_flipped"] is True, r
    assert r["pred_home_away_ratio"] == 1.0, r
    assert r["track_accuracy"] == 1.0, r

    # One track on the wrong side: 15/20, and one of the two banks drops to 50%.
    bad = {**swapped, 1: "home"}
    r2 = score(bad, z, gt, _Cal())
    assert r2["detection_accuracy"] == 0.75, r2
    assert r2["per_team_accuracy"]["home"] == 0.5, r2

    # Collapse to one bank - the failure mode that must NOT look like success.
    r3 = score({i: "home" for i in (1, 2, 3, 4)}, z, gt, _Cal())
    assert r3["detection_accuracy"] == 0.5, r3
    assert r3["pred_home_away_ratio"] == float("inf"), r3

    # 'official' and 'unknown' match no GT team under either mapping.
    r4 = score({i: "official" for i in (1, 2, 3, 4)}, z, gt, _Cal())
    assert r4["detection_accuracy"] == 0.0, r4

    # Out of gate is not scored at all, rather than scored as correct.
    far = {k: (v.copy() if k != "foot" else v + 50.0) for k, v in z.items()}
    r5 = score(swapped, far, gt, _Cal())
    assert r5["matched_detections"] == 0, r5

    _selfcheck_clips()
    print("team_eval.py ok")


def _selfcheck_clips() -> None:
    """The ten-match path: the IoU matcher, and a NEGATIVE CONTROL on the join.

    The control is the point. A wrong MOT-id -> team join does not crash and
    does not look wrong; it produces a plausible accuracy. So one real match is
    re-scored with its 11 home / 11 away labels SHUFFLED, and that has to land
    near chance. If a scrambled join still scores in the eighties, the number
    is not coming from the join and nothing built on it means anything.
    """
    import st2_team_gt  # noqa: PLC0415

    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert abs(_iou(a, a)[0, 0] - 1.0) < 1e-9
    assert _iou(a, np.array([[20.0, 20.0, 10.0, 10.0]]))[0, 0] == 0.0
    # half-overlap: 50 intersection, 150 union
    assert abs(_iou(a, np.array([[5.0, 0.0, 10.0, 10.0]]))[0, 0] - 50 / 150) < 1e-9

    seq = "118575"
    join = st2_team_gt.load()
    real = score_clip(seq, every=25, gt_join=join)
    assert real["detection_accuracy"] > 0.70, real

    # N9's two columns, on the same real clip. The accounting identity is the
    # check that matters: every sampled box of ours is either matched or
    # unmatched, and every unmatched one carries exactly one label. If that ever
    # stops holding, `phantom_frac` is being divided by the wrong denominator.
    assert (real["our_detections_unmatched"] + sum(real["matched_label_counts"].values())
            <= real["our_detections_sampled"]), real
    assert sum(real["unmatched_label_counts"].values()) == real["our_detections_unmatched"]
    assert 0.0 <= real["phantom_frac"] <= real["unscored_frac"] <= 1.0, real
    # The naming column must be the UNFORGIVEN accuracy: never better than the
    # best-of-two, and exactly equal to it when no flip was needed.
    assert real["detection_accuracy_as_named"] <= real["detection_accuracy"] + 1e-9, real
    assert (real["named_correctly"] ==
            (real["detection_accuracy_as_named"] == real["detection_accuracy"])), real
    # And a clip whose labels are globally inverted must keep `detection_accuracy`
    # and LOSE `named_correctly`. That is the whole point of the column.
    flipped = {**join, seq: {**join[seq],
                             "team": {k: ("away" if v == "home" else "home")
                                      for k, v in join[seq]["team"].items()}}}
    inv = score_clip(seq, every=25, gt_join=flipped)
    assert abs(inv["detection_accuracy"] - real["detection_accuracy"]) < 1e-9, (
        "best-of-two mapping must be blind to a global side swap")
    assert inv["named_correctly"] is not real["named_correctly"], (
        "named_correctly must NOT be blind to a global side swap")

    # Five shuffles, not one: a single draw of an 11/11 relabelling lands
    # anywhere from 0.47 to 0.67 on this match, so one draw would be a coin
    # toss dressed up as a test.
    ids = list(join[seq]["team"])
    got = []
    for seed in range(5):
        vals = [join[seq]["team"][i] for i in ids]
        np.random.default_rng(seed).shuffle(vals)
        bad = {**join, seq: {**join[seq], "team": dict(zip(ids, vals))}}
        got.append(score_clip(seq, every=25, gt_join=bad)["detection_accuracy"])
    assert max(got) < 0.75 and float(np.mean(got)) < 0.65, (
        f"SHUFFLED team joins score {got} against a real {real['detection_accuracy']:.3f} - "
        "this harness cannot tell a good join from a bad one")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="sample the video and cache crops")
    ap.add_argument("--frames", type=int, default=300, help="frames to sample for --build")
    ap.add_argument("--method", default="shipped",
                    help="shipped | chroma | chroma_only | mean_lab")
    ap.add_argument("--clusters", action="store_true", help="cluster balance/separation only")
    ap.add_argument("--clips", action="store_true",
                    help="score ALL TEN STC2025 clips against ST2's own team truth")
    ap.add_argument("--seq", help="score one STC2025 clip")
    ap.add_argument("--seqs", help="comma-separated clip list for --clips (N9: the nine)")
    ap.add_argument("--variant", default="c03s", help="stc2025 tracks variant")
    ap.add_argument("--every", type=int, default=CLIP_EVERY, help="clip frame stride")
    ap.add_argument("--gate", type=float, default=IOU_GATE, help="IoU match gate")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.clips:
        print(json.dumps(evaluate_clips(a.variant, a.every, a.gate,
                                        seqs=tuple(a.seqs.split(",")) if a.seqs else None),
                         indent=2))
    elif a.seq:
        print(json.dumps(score_clip(a.seq, a.variant, a.every, a.gate), indent=2))
    elif a.build:
        build(a.frames)
    elif a.clusters:
        print(json.dumps(cluster_report(a.method), indent=2))
    else:
        print(json.dumps(evaluate(a.method), indent=2))
