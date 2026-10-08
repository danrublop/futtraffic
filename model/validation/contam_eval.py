"""CONTAMINATION: what share of a player row belongs to somebody else.

WHY THIS EXISTS
    Every identity decision this pipeline makes was judged on IDF1 until it was
    caught rewarding the exact failure it was supposed to detect. IDF1 credits a
    ground-truth player with the frames carrying its single most common
    predicted id, so ONE LARGE IMPURE TRACK SCORES WELL for whoever dominates
    it, and splitting that track into two pure ones scores WORSE. Two real
    improvements - the team gate in `stitch.stitch` and the teleport split in
    `stitch.split` - were measured, rejected on IDF1, and only shipped after
    being re-measured here. A third, `--stride 2`, looked like a quality
    improvement on IDF1 and is really a wash.

    So the numbers this repo quotes for identity - 49.6% with neither, 33.2%
    with both - come from this file, and until it existed they came from scratch
    scripts that no longer exist. CORRECTED 2026-09-28: "neither" still had
    stitching on - 49.6% is the UNGATED stitch; raw tracks before stitching score
    15.1% (ledger-vision (not published), IDENTITY rows), so stitching raises
    contamination. One match, tracks config unrecorded; at what ships see
    n21-product-now (not published) (N8 row). A metric the whole argument rests on has to be
    runnable by someone else, or it is an anecdote.

WHAT IT MEASURES
    For every predicted id, the ground-truth player who owns the most of its
    detections is its dominant player. Contamination is the share of all
    attributed detections that sit under an id whose dominant player is somebody
    else. Perfect tracking is 0%. Welding two players into one id puts the
    smaller of them straight into the numerator, which is precisely what IDF1
    cannot see.

    PER DETECTION, NOT PER TRACK. Attributing one owner to a whole track first
    and then aggregating gives a flattering number - it hides the swap inside
    the track that made the track wrong. The two denominators differ by about
    two points on real output (31.0% per track against 33.2% per detection), so
    the distinction is worth stating whenever a figure is quoted.

WHAT IT IS NOT
    It says nothing about whether a detection is in the right place, only about
    whose id it is under. Detection quality belongs to `detsweep.py` and
    association-over-time to `assoc_eval.py`; all three are needed and none
    substitutes for another.

    It also cannot see a player nobody detected. A pipeline that found one
    player and tracked them perfectly would score 0% here. Read it beside the
    recall in `assoc_eval`, never alone.

N8 (2026-09-12): CONTAMINATION AFTER THE IDENTITY MERGE, AT THE PRODUCT CONFIG
    `api://product.identity.contaminated` = 33.2% runs on `tracks.split.json` in
    one job directory, and that file records no detector, no conf and no stride
    (model-map (not published) §3). So the shipped identity number is measured on a
    configuration nobody can name. `--stc` re-measures it on tracks that were
    produced by `app/server/jobs.py`'s own call.

    PROTOCOL, written here before the first run, with the pass rule:

    population   the nine STC2025 clips that have product tracks on disk.
                 132831 is EXCLUDED because its ST2 chain maps all 16 in-frame
                 landmarks onto one pixel, so the product's clicked calibration
                 cannot be produced for it at all (`st2_product.py`) - a limit
                 of the benchmark data, stated before any score existed.
    split        NOT a dataset split, and it must not be read as one. 117093 and
                 128058 are STC2025 training sequences and 118575 sits in ST2's
                 train split (model-map §4). All nine are held out from what
                 `.c03s` was tuned on. n=1 configuration: nothing is swept here.
    held fixed   coreml r576 tile 1024, conf 0.5, stride 2, scene prior built
                 from the clicked calibration, metric stitch, then
                 `jobs.apply_identity` - i.e. the files `{seq}.p05.tracks.json`
                 (before the merge) and `{seq}.p05id.tracks.json` (after it).
    attribution  PIXELS, not metres, and this is the one deliberate departure
                 from the 118575 path above. The 2.0 m gate needs a calibration
                 whose own residual is well under a metre; the product's clicked
                 fit is 5.8-6.3 m RMS on these panoramas (N6). A metre gate on
                 top of that attributes almost nothing, and what it did
                 attribute would make this a calibration measurement wearing an
                 identity costume. So: Hungarian on IoU >= 0.5 against ST2's own
                 MOT boxes - the same matcher `team_eval.score_clip` uses, so
                 "matched" means the same thing in both files.
    frames       every frame the stride-2 tracker ran on (the even ones).
                 Dropped by the product's own cadence, never by selection.
    metric       contamination PER DETECTION, `contamination()` unchanged, floor
                 0.0% (perfect) and rising when two players share an id. It is
                 NOT comparable with the 33.2% above: different footage,
                 different matcher, a quarter of a match instead of a half.
    pass rule    the claim under test is N6's - the identity merge is redundant
                 after metric stitch. PASSES if contamination after the merge
                 equals contamination before it on every sequence AND every
                 `.p05id` file's `_identity_report` records zero merges. FAILS,
                 and the merge is doing work for better or worse, if either
                 moves. A fail is a result; so is a pass.
    drop share   printed on every row and never optional: the share of our
                 sampled detections that match no GT box (they are excluded from
                 the metric entirely) and the share of GT boxes we never matched.

    python model/validation/contam_eval.py --selfcheck
    python model/validation/contam_eval.py --tracks tracks.split.json
    python model/validation/contam_eval.py --grid        # reproduce the shipped table
    python model/validation/contam_eval.py --stc         # N8: nine clips, both stages
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

import calibrators  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")

#: How close a prediction must be to a ground-truth player to be attributed to
#: them. The same 2.0 m `assoc_eval` uses, for the same reason: the calibration's
#: own error against real players is about 1.0 m, so a tighter gate would score
#: the geometry and a looser one would attribute a detection to a player who was
#: never there.
GATE_M = 2.0

#: Sample every Nth frame when attributing owners. 10 is 7,185 frames of a half,
#: which is ~100k attributed detections - far past the point where the answer
#: moves. Attribution is the whole cost of this file; the metric itself is free.
STRIDE = 10


def load_gt(path: str = GT) -> dict[int, dict[str, np.ndarray]]:
    """Ground-truth positions in metres, per frame, 0-based.

    The file numbers frames from 3,000,001; the offset is removed so both sides
    speak the video's own frame index.
    """
    rows = json.load(gzip.open(path))["detections"]
    base = min(r["f"] for r in rows)
    per: dict[int, dict[str, np.ndarray]] = collections.defaultdict(dict)
    for r in rows:
        per[r["f"] - base][str(r["id"])] = np.array([r["x"], r["y"]])
    return per


def owners(rows: list[dict], cal, gt: dict[int, dict[str, np.ndarray]],
           stride: int = STRIDE) -> dict[int, str]:
    """Row index -> ground-truth player id, for the rows close enough to claim.

    Greedy nearest ground-truth player inside GATE_M. Rows with no player near
    them are left out entirely rather than assigned to the closest one: a false
    positive on an empty patch of grass is a DETECTION problem, and folding it
    in here would let detection noise move an identity number.
    """
    by_frame: dict[int, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_frame[int(r["f"])].append(i)
    if not by_frame:
        return {}

    xy = cal.to_pitch(np.array([[r["x"], r["y"]] for r in rows], float))
    out: dict[int, str] = {}
    for f in range(0, max(by_frame) + 1, stride):
        g, idx = gt.get(f), by_frame.get(f)
        if not g or not idx:
            continue
        gid = list(g)
        G = np.array([g[k] for k in gid])
        d = np.linalg.norm(xy[idx][:, None, :] - G[None, :, :], axis=2)
        j = d.argmin(1)
        for k, i in enumerate(idx):
            if d[k, j[k]] < GATE_M:
                out[i] = gid[j[k]]
    return out


def contamination(rows: list[dict], owner: dict[int, str]) -> dict:
    """The metric. Share of attributed detections under a wrong-player id."""
    per_id: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for i, g in owner.items():
        per_id[int(rows[i]["id"])][g] += 1

    total = wrong = 0
    impure = 0
    for counts in per_id.values():
        n = sum(counts.values())
        total += n
        wrong += n - counts.most_common(1)[0][1]
        impure += len(counts) > 1
    # How many ids a ground-truth player is spread across - the other half of
    # the story, and the one contamination deliberately does not punish.
    spread = collections.defaultdict(set)
    for i, g in owner.items():
        spread[g].add(int(rows[i]["id"]))
    return {
        "contaminated": round(100 * wrong / total, 1) if total else 0.0,
        # The numerator as well as the rate, so several sequences can be POOLED
        # by summing wrong and attributed. A mean of nine percentages weights a
        # clip with half the detections equally, and re-deriving `wrong` from a
        # rounded percentage is a number that can disagree with itself.
        "wrong": wrong,
        "attributed": total,
        "ids": len(per_id),
        "impure_ids": impure,
        "ids_per_gt_player": round(float(np.mean([len(v) for v in spread.values()])), 1)
        if spread else 0.0,
        "gt_players_seen": len(spread),
    }


def score(tracks: str, job: str = JOB, gt_path: str = GT, stride: int = STRIDE) -> dict:
    rows = json.load(open(os.path.join(job, tracks)))["detections"]
    cal = calibrators.load(os.path.join(job, "calibration.json"))
    return contamination(rows, owners(rows, cal, load_gt(gt_path), stride))


def _selfcheck() -> None:
    """Synthetic tracks with a known answer, including the cases that lie.

    A metric that cannot go up when two players are merged is the whole reason
    this file exists, so merging is what gets asserted hardest.
    """
    class Flat:
        """Identity calibration: an image coordinate IS a pitch metre."""

        @staticmethod
        def to_pitch(px):
            return np.asarray(px, float)

    # Two players, 20 m apart, 100 frames each.
    gt = {f: {"A": np.array([10.0, 10.0]), "B": np.array([30.0, 30.0])}
          for f in range(100)}

    def rows_for(assign):
        out = []
        for f in range(0, 100, STRIDE):
            out.append({"f": f, "id": assign(f, "A"), "x": 10.0, "y": 10.0})
            out.append({"f": f, "id": assign(f, "B"), "x": 30.0, "y": 30.0})
        return out

    # Perfect: one id per player, never crossed.
    perfect = rows_for(lambda f, p: 1 if p == "A" else 2)
    r = contamination(perfect, owners(perfect, Flat, gt))
    assert r["contaminated"] == 0.0, r
    assert r["ids"] == 2 and r["impure_ids"] == 0, r
    assert r["ids_per_gt_player"] == 1.0, r

    # WELDED: both players under one id. Half of it is the wrong player, and
    # this is the case IDF1 scores as an improvement.
    welded = rows_for(lambda f, p: 1)
    r = contamination(welded, owners(welded, Flat, gt))
    assert r["contaminated"] == 50.0, r
    assert r["ids"] == 1 and r["impure_ids"] == 1, r

    # FRAGMENTED: each player cut into five ids, but never mixed. Contamination
    # must stay at zero - it measures purity, not fragmentation - while
    # ids_per_gt_player carries the damage. If a "fix" trades one for the other,
    # both numbers have to be read to see it.
    frag = rows_for(lambda f, p: (f // 20) * 2 + (1 if p == "A" else 2))
    r = contamination(frag, owners(frag, Flat, gt))
    assert r["contaminated"] == 0.0, r
    assert r["ids_per_gt_player"] == 5.0, r

    # A detection nowhere near anybody is not attributed, so pure grass cannot
    # move the number in either direction.
    noise = perfect + [{"f": f, "id": 99, "x": 90.0, "y": 60.0} for f in range(0, 100, STRIDE)]
    r = contamination(noise, owners(noise, Flat, gt))
    assert r["contaminated"] == 0.0, r
    assert r["ids"] == 2, f"an unattributed id must not become a player: {r}"

    # THE CONTROL. Shuffling which id each detection carries must WRECK the
    # score. If it does not, the metric is reading the data rather than the
    # labelling, and every number taken off it is worthless.
    rng = np.random.default_rng(0)
    shuffled = [dict(r_, id=int(rng.integers(1, 3))) for r_ in perfect]
    r = contamination(shuffled, owners(shuffled, Flat, gt))
    assert r["contaminated"] > 25.0, f"random ids must score badly, got {r}"

    _selfcheck_stc()
    print("contam_eval.py ok  (perfect 0.0%, welded 50.0%, fragmented-but-pure "
          "0.0%, random ids wrecked)")


def _selfcheck_stc() -> None:
    """The N8 path, with the control that separates the matcher from the metric.

    `owners_iou` is the part that can break silently. Read `to_tlwh` the wrong
    way round, or sample an odd frame our stride-2 tracker never ran on, and it
    attributes almost nothing - which reads as a clean tracker, not as a broken
    harness. So: a real clip must attribute a plausible share, and the SAME
    attribution with our track ids shuffled must wreck contamination. If the
    shuffle does not move it, the number is coming from the boxes rather than
    from the labelling.
    """
    if not os.path.exists(f"{STC}/118575.p05id.tracks.json"):
        return
    rows = json.load(open(f"{STC}/118575.p05id.tracks.json"))["detections"]
    owner, counts = owners_iou("118575", rows, every=50)
    assert counts["our_boxes_sampled"] > 500, counts
    real = contamination(rows, owner)
    assert 0.3 < real["attributed"] / counts["our_boxes_sampled"] < 1.0, (real, counts)
    assert real["ids"] > 10, real

    # Shuffle the id COLUMN, not the id names: contamination is a property of
    # the partition, so renaming ids is invariant by construction and would
    # assert nothing. This keeps every id's size and destroys who is in it.
    rng = np.random.default_rng(0)
    col = rng.permutation([int(r["id"]) for r in rows])
    shuffled = [dict(r, id=int(i)) for r, i in zip(rows, col)]
    bad = contamination(shuffled, owner)
    assert bad["contaminated"] > real["contaminated"] + 10, (
        f"shuffling our track ids scores {bad['contaminated']}% against a real "
        f"{real['contaminated']}% - this harness cannot see the labelling")
    # An odd frame is a frame the product never tracked. Asserting it here is
    # what stops a future `--every 5` reading as a 50% recall collapse.
    assert not any(int(rows[i]["f"]) % 2 for i in owner), "stride-2 tracks on an odd frame"
    assert merges("118575") == 0, "N6 recorded zero merges; that is the pass rule's input"


#: The configurations quoted across this repo, so the table can be regenerated
#: rather than believed. Names match the files `stitch.py` and `track.py` write.
GRID = (("raw association (C-BIoU)", "tracks.cbiou.json"),
        ("+ team-gated stitch", "tracks.final.json"),
        ("+ split at 45 m/s", "tracks.split.json"))


# ==========================================================================
# N8. Everything above scores one job directory in METRES and is untouched, so
# `api://product.identity.contaminated` still reproduces. See the protocol in
# the module docstring; the pass rule is there and was fixed before the run.

STC = os.path.join(ROOT, "model/validation/out/stc2025")
STC_GT = _ST2 + "/mot"
#: The nine with a product calibration. 132831 has none - see the protocol.
STC_SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
            "128057", "128058", "132877")
IOU_GATE = 0.5
#: Our stride-2 tracks only exist on even frames, so an odd sampling frame would
#: score zero of our detections and read as a tracker that misses half the match.
STC_EVERY = 2


def owners_iou(seq: str, rows: list[dict], every: int = STC_EVERY,
               gate: float = IOU_GATE) -> tuple[dict[int, str], dict]:
    """Row index -> MOT ground-truth id, matched by IoU in PIXELS.

    The metre attribution `owners()` does cannot be used on the STC clips: the
    product's clicked homography is ~6 m off there, so a 2 m gate would measure
    the calibration. This is `team_eval.score_clip`'s matcher, imported rather
    than forked so "matched at IoU 0.5" means one thing in this repo.

    Returns the owner map AND the counts the drop share is computed from -
    never one without the other.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import st2_team  # noqa: PLC0415  - to_tlwh, the bottom-centre -> MOT convention
    import team_eval  # noqa: PLC0415  - _iou

    gt: dict[int, list] = collections.defaultdict(list)
    n_gt = 0
    with open(f"{STC_GT}/{seq}.txt") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 6 or int(p[0]) % every:
                continue
            gt[int(p[0])].append((int(p[1]), *(float(v) for v in p[2:6])))
            n_gt += 1

    ours: dict[int, list] = collections.defaultdict(list)
    n_ours = 0
    for i, r in enumerate(rows):
        if int(r["f"]) % every:
            continue
        ours[int(r["f"])].append(i)
        n_ours += 1

    owner: dict[int, str] = {}
    for f, g in gt.items():
        idx = ours.get(f)
        if not idx:
            continue
        G = np.array([[x, y, w, h] for _i, x, y, w, h in g], float)
        O = np.array([st2_team.to_tlwh(rows[i]) for i in idx], float)
        M = team_eval._iou(O, G)
        for i, j in zip(*linear_sum_assignment(-M)):
            if M[i, j] >= gate:
                owner[idx[i]] = str(g[j][0])
    return owner, {"gt_boxes_sampled": n_gt, "our_boxes_sampled": n_ours,
                   "sampled_frames": len(gt)}


def merges(seq: str, variant: str = "p05id") -> int:
    """How many tracklets `jobs.apply_identity` actually merged on this clip.

    N6 says zero on every sequence. Read out of the file the stage wrote rather
    than re-run, because the file IS the record of the run that produced the
    tracks being scored.
    """
    rep = json.load(open(f"{STC}/{seq}.{variant}.tracks.json")).get("_identity_report", {})
    try:
        return int(rep["tracklets_before"]) - int(rep["tracklets_after"])
    except KeyError as e:   # the field moved; a silent 0 would PASS the rule
        raise SystemExit(f"{seq}: _identity_report has no {e}: {sorted(rep)}") from None


def score_clip(seq: str, variant: str = "p05id", every: int = STC_EVERY,
               gate: float = IOU_GATE) -> dict:
    rows = json.load(open(f"{STC}/{seq}.{variant}.tracks.json"))["detections"]
    owner, counts = owners_iou(seq, rows, every, gate)
    r = contamination(rows, owner)
    r.update(seq=seq, variant=variant, **counts)
    # THE DROP SHARE, on the row, every time. Contamination is computed over
    # `attributed` only: a detection that matched no GT box is not scored at
    # all, and a tracker that emitted fewer, cleaner boxes would look better
    # here for exactly the wrong reason.
    r["unattributed_frac"] = round(1 - r["attributed"] / counts["our_boxes_sampled"], 4) \
        if counts["our_boxes_sampled"] else 0.0
    r["gt_recall"] = round(r["attributed"] / counts["gt_boxes_sampled"], 4) \
        if counts["gt_boxes_sampled"] else 0.0
    return r


def evaluate_stc(variants=("p05", "p05id"), seqs=STC_SEQS, every: int = STC_EVERY,
                 gate: float = IOU_GATE, verbose: bool = True) -> dict:
    """N8. Both stages on all nine, per clip and pooled. Pooled = sum/sum."""
    out: dict = {"variants": list(variants), "iou_gate": gate, "frame_stride": every,
                 "sequences": list(seqs), "per_variant": {}}
    for v in variants:
        rows = [score_clip(s, v, every, gate) for s in seqs]
        if verbose:
            for r in rows:
                print(f"{r['seq']} {v:6s} contaminated {r['contaminated']:5.1f}%  "
                      f"ids {r['ids']:4d} impure {r['impure_ids']:3d}  "
                      f"ids/player {r['ids_per_gt_player']:5.1f}  "
                      f"attributed {r['attributed']:6d}  "
                      f"unattributed {r['unattributed_frac']:.3f}  "
                      f"gt_recall {r['gt_recall']:.3f}", flush=True, file=sys.stderr)
        w = sum(r["wrong"] for r in rows)
        n = sum(r["attributed"] for r in rows)
        ours = sum(r["our_boxes_sampled"] for r in rows)
        gtn = sum(r["gt_boxes_sampled"] for r in rows)
        out["per_variant"][v] = {
            "pooled_contaminated": round(100 * w / n, 1) if n else 0.0,
            "pooled_wrong": w, "pooled_attributed": n,
            "pooled_unattributed_frac": round(1 - n / ours, 4) if ours else 0.0,
            "pooled_gt_recall": round(n / gtn, 4) if gtn else 0.0,
            "per_clip": {r["seq"]: r for r in rows},
        }
    # Off the id stage THIS call scored - variants[-1], the one the rule below
    # reads - and not a hard-coded .p05id, which handed every other run N6's merges.
    out["merges"] = {s: merges(s, variants[-1]) for s in seqs}
    a, b = variants[0], variants[-1]
    moved = [s for s in seqs
             if out["per_variant"][a]["per_clip"][s]["contaminated"]
             != out["per_variant"][b]["per_clip"][s]["contaminated"]]
    # The pass rule from the docstring, evaluated in code rather than by eye.
    out["identity_merge_is_redundant_after_stitch"] = (
        not moved and set(out["merges"].values()) == {0})
    out["sequences_whose_contamination_moved"] = moved
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="tracks.split.json")
    ap.add_argument("--job", default=JOB)
    ap.add_argument("--stride", type=int, default=STRIDE)
    # Reachable from the CLI so this can be pointed at a SECOND MATCH. score()
    # always took gt_path; only argparse could not reach it, which is why every
    # row in the ledger still reads n=1.
    ap.add_argument("--gt", default=GT, help="ground-truth .tracks.json.gz")
    ap.add_argument("--grid", action="store_true", help="every shipped configuration")
    ap.add_argument("--stc", action="store_true",
                    help="N8: the nine STC2025 clips at the product config, both stages")
    ap.add_argument("--variant", action="append", help="stc2025 variant (repeatable)")
    ap.add_argument("--seq", action="append", help="one stc2025 sequence (repeatable)")
    ap.add_argument("--every", type=int, default=STC_EVERY, help="clip frame stride")
    ap.add_argument("--gate", type=float, default=IOU_GATE, help="IoU match gate")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.stc:
        print(json.dumps(evaluate_stc(tuple(a.variant or ("p05", "p05id")),
                                      tuple(a.seq or STC_SEQS), a.every, a.gate),
                         indent=2))
    elif a.grid:
        for name, f in GRID:
            if not os.path.exists(os.path.join(a.job, f)):
                print(f"{name:28s} (missing {f})")
                continue
            r = score(f, a.job, a.gt, stride=a.stride)
            print(f"{name:28s} contaminated {r['contaminated']:5.1f}%   "
                  f"ids {r['ids']:5d}   ids/player {r['ids_per_gt_player']:5.1f}   "
                  f"attributed {r['attributed']}", flush=True)
    else:
        print(json.dumps(score(a.tracks, a.job, a.gt, stride=a.stride), indent=2))
