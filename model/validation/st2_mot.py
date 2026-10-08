"""SoccerTrack Challenge 2025 MOT protocol: our tracker, their HOTA, their clips.

WHY THIS EXISTS
    Every association number this repo quotes is bespoke - IDF1 0.155, 47.3
    fragments per player, 33.2% contamination, 796 tracks for 22 players. None has
    a published peer, so none says whether we are bad or merely honest.

    The MMSports 2025 SoccerTrack Challenge scores HOTA on SoccerTrack v2's own
    4-minute clips, same camera, same 22 identities. Published: ByteTrack 0.42,
    Deep-EIoU 0.54, winner GTATrack 0.60 (DetA 0.76 / AssA 0.47 / IDSW 331.5).
    That is the first yardstick this stack has ever had.

TWO CONVENTIONS THAT WILL SILENTLY COST YOU
    1. The GT ships **0-based** (frames 0..5975); MOTChallenge and TrackEval are
       **1-based**. Both GT and predictions are shifted +1 here. Get this wrong and
       every box is one frame stale, which reads as a slightly worse tracker rather
       than as a bug.
       (`118578.txt` arrived 1-based from the organisers and was normalised to
       0-based in our copy - so it is consistent with the other nine here.)
    2. GT columns 7-10 are all `-1`. TrackEval's MOT17 parser reads col 7 as
       confidence and col 9 as visibility, and a -1 confidence is dropped. They are
       rewritten to `1,1,1` on the way in.

NO REMAP, DELIBERATELY
    The GT is in native panorama pixels, so the tracker runs on the raw 4096x1080
    frame with no `--remap` and no `--calib`. That is also the honest reading: it is
    what our shipped stack does to this footage with no dataset-specific help.

`ponytail: sequences are hardcoded to the pair we score. The challenge split is
under query - see docs/experiments/st2-input-ledger.md - so SEQS is one line to change.`
"""
import argparse
import atexit
import configparser
import json
import math
import os
import shutil
import subprocess
import sys
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

GT_SRC = _ST2 + "/mot"
# Root-relative, not cwd-relative: run from model/validation/ the relative form staged
# into model/validation/validation/out while api.py hashed model/validation/out, and
# st2_ab.available() found no sequences at all.
WORK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "model/validation/out/stc2025")
BENCH, SPLIT, TRACKER = "STC", "test", "ours"
VARIANT = os.environ.get("ST2_VARIANT", "")

# The CHALLENGE (final) phase, which is what the published 0.42/0.60 pair scores:
# organisers' "Challenge Videos" folder, and also the ST2 README's own test default.
# 117093/128058 are TRAINING sequences - useful to sanity-check the harness, never
# to quote. Geometry is per-sequence: 132831 is 3840x1504, the rest 4096x1080.
SEQS = ("128057", "132831")
CLIPS = _ST2 + "/mot/clips"
FPS = 25


def _wh(seq):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         f"{CLIPS}/{seq}.mp4"], capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


#: Every staging root this process has handed out. A SET, not a recomputation
#: from VARIANT: every harness here rebinds that global per arm (st2_product
#: scores six variants in one process, n45_winner_boxes three), so a cleanup that
#: re-derives one path at exit deletes the last arm's tree and leaks the rest.
#: It did: review found 31 trees and 1.7 GB left behind, one pid holding 14.
_STAGED: set = set()


def _cleanup_staging():
    """Drop every staging tree THIS process created. Best effort: a stale tree is
    gitignored scratch, not a correctness problem, and a cleanup that raises on
    exit would turn a green run red for nothing."""
    import shutil  # noqa: PLC0415
    for d in list(_STAGED):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


atexit.register(_cleanup_staging)

def _root(sub):
    """Staging root, SCOPED BY VARIANT.

    It was not, and that made the whole harness non-deterministic: every variant
    staged its predictions into one shared directory, so `st2_ab.py` scoring A
    then B could read B against A's staged files. Measured 2026-09-10 - the same
    command on the same inputs, twice in a row:

        run 1:  AssA improved on 6/8 sequences, p = 0.145
        run 2:  AssA improved on 0/0 sequences, p = 1.000

    An A/B harness that returns a different answer to the same question is worse
    than no harness, because it produces verdicts. The variant goes in the path.

    ...AND BY PROCESS, for the same reason one level out. `stage_gt` opens its
    gt.txt with "w", which truncates a 5.3 MB file in place and refills it line by
    line, so a concurrent reader of the SAME variant scores against a partial
    ground truth. On 2026-09-23 that blocked a push: audit-claims (not published):989 came
    back DRIFT in 32.1 s against 29 previous OK rows at 65.1-80.4 s - half the
    time, because half the boxes. The value had never moved, and re-run alone it
    printed exactly what was recorded.

    The gate races ITSELF: `verify_claims.resources()` derives a resource from the
    module named in the command string, so `st2_ab.py`, `status.py` and
    `st2_mot.py` read as three disjoint resources while all three write eval/c03s.
    A cross-process guard could not have helped either - the collision was inside
    one -j4 run. Scoping the path by pid closes both, costs ~100 MB of scratch,
    and is safe because nothing here is a cache: stage_gt and stage_pred run
    unconditionally on every invocation, and every reader stages then scores
    inside one process.
    """
    v = (VARIANT or "bare").lstrip(".").replace("/", "_")
    root = os.path.join(WORK, "eval", f"{v}.{os.getpid()}")
    _STAGED.add(root)
    return os.path.join(root, sub, f"{BENCH}-{SPLIT}")


def stage_gt(seqs, every=1):
    """Their .txt -> TrackEval's expected tree, 1-based, with a usable conf column.

    `every` keeps GT only on every Nth frame (0-based f % N == 0). A stride-2
    tracker - the product's - emits boxes on even frames only, and against all
    GT frames every odd frame reads as a miss: half the recall lost to a
    cadence, not to detection. A PARAMETER, not a module global: as a global it
    was set to 2 by one measurement and silently inherited by the next caller
    (st2_ab.score), which scored three A/B verdicts on half the frames."""
    for s in seqs:
        d = os.path.join(_root("gt"), s, "gt")
        os.makedirs(d, exist_ok=True)
        n = 0
        with open(f"{GT_SRC}/{s}.txt") as fh, open(f"{d}/gt.txt", "w") as out:
            for line in fh:
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                n = max(n, int(p[0]) + 1)      # seqLength stays the full clip
                if int(p[0]) % every:
                    continue
                out.write(f"{int(p[0]) + 1},{int(float(p[1]))},"
                          f"{p[2]},{p[3]},{p[4]},{p[5]},1,1,1\n")
        w, h = _wh(s)
        _seqinfo(s, n, w, h, every)
        print(f"  gt {s}: {n} frames, {w}x{h}, every {every}")
    with open(os.path.join(_root("gt"), f"{BENCH}-{SPLIT}.txt"), "w") as fh:
        fh.write("\n".join(seqs) + "\n")


def _seqinfo(s, n, w, h, every):
    """seqinfo.ini, with the GT stride it was staged at. TrackEval reads only its own
    keys; `gtEvery` is for stage_pred, which must score against the stride that WAS
    staged - a file, not a module global, because the global leaked (stage_gt)."""
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {"name": s, "imDir": "img1", "frameRate": str(FPS),
                       "seqLength": str(n), "imWidth": str(w),
                       "imHeight": str(h), "imExt": ".jpg", "gtEvery": str(every)}
    os.makedirs(os.path.join(_root("gt"), s), exist_ok=True)
    with open(os.path.join(_root("gt"), s, "seqinfo.ini"), "w") as fh:
        cfg.write(fh)


def _check_cadence(s, frames, every, n):
    """Refuse predictions whose frames do not sit on the staged GT's frames.

    Two traps, both silent in TrackEval. A stride-2 tracker scored against GT on
    EVERY frame reads each odd frame as a miss - half the recall lost to a cadence
    (`n40_association` staged `.p35` exactly so). Predictions on frames the GT was
    thinned OFF score as false positives against no boxes at all."""
    if not frames:
        raise ValueError(f"{s}: no predictions to stage - a tracker that emitted nothing "
                         "is a failure, not a HOTA of 0")
    off = [f for f in frames if f % every]
    if off:
        raise ValueError(f"{s}: {len(off)} prediction frames are off the GT stride {every} "
                         f"(first {off[0]}) and would score as false positives")
    step = math.gcd(*(b - a for a, b in zip(frames, frames[1:]))) if len(frames) > 1 else 0
    if step > every:
        raise ValueError(f"{s}: predictions every {step} frames against GT every {every} - "
                         f"the GT frames between them read as misses; stage_gt(every={step})")
    if frames[-1] >= n:
        raise ValueError(f"{s}: prediction frame {frames[-1]} is past the staged GT's {n} frames")


def stage_pred(seqs, every=None):
    """tracks.json (bottom-centre x,y + w,h, 0-based f) -> MOTChallenge txt.

    `every` is the GT stride the caller believes it staged; it is checked against
    the one stage_gt recorded in this process's seqinfo.ini, and the prediction
    frames against that stride (_check_cadence). No staged GT, no staging."""
    d = os.path.join(_root("trackers"), TRACKER, "data")
    os.makedirs(d, exist_ok=True)
    for s in seqs:
        src = f"{WORK}/{s}{VARIANT}.tracks.json"
        if not os.path.exists(src):
            sys.exit(f"missing {src} - run the tracker first")
        cfg = configparser.ConfigParser()
        if not cfg.read(os.path.join(_root("gt"), s, "seqinfo.ini")) or "gtEvery" not in cfg["Sequence"]:
            raise ValueError(f"{s}: no GT staged by this process - stage_gt before stage_pred")
        staged = int(cfg["Sequence"]["gtEvery"])
        if every is not None and every != staged:
            raise ValueError(f"{s}: stage_pred(every={every}) but the GT was staged every {staged}")
        rows = json.load(open(src))["detections"]
        _check_cadence(s, sorted({int(r["f"]) for r in rows}), staged,
                       int(cfg["Sequence"]["seqLength"]))
        with open(f"{d}/{s}.txt", "w") as out:
            for r in rows:
                x = r["x"] - r["w"] / 2.0          # bottom-centre -> top-left
                y = r["y"] - r["h"]
                out.write(f"{r['f'] + 1},{r['id']},{x:.2f},{y:.2f},"
                          f"{r['w']:.2f},{r['h']:.2f},1,-1,-1,-1\n")
        print(f"  pred {s}: {len(rows)} boxes, {len({r['id'] for r in rows})} ids")


def run(seqs):
    import trackeval
    ev = trackeval.Evaluator({**trackeval.Evaluator.get_default_eval_config(),
                              "PRINT_CONFIG": False, "DISPLAY_LESS_PROGRESS": True,
                              "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                              "PLOT_CURVES": False, "USE_PARALLEL": False})
    ds = trackeval.datasets.MotChallenge2DBox({
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": _root("gt"), "TRACKERS_FOLDER": _root("trackers"),
        "BENCHMARK": BENCH, "SPLIT_TO_EVAL": SPLIT, "TRACKERS_TO_EVAL": [TRACKER],
        "SEQ_INFO": {s: None for s in seqs}, "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False, "PRINT_CONFIG": False})
    res, _ = ev.evaluate([ds], [trackeval.metrics.HOTA(), trackeval.metrics.Identity(),
                                trackeval.metrics.CLEAR()])
    key = list(res)[0]
    out = res[key][TRACKER]
    print(f"\n{'seq':>10s} {'HOTA':>7s} {'DetA':>7s} {'AssA':>7s} {'LocA':>7s} "
          f"{'IDF1':>7s} {'IDSW':>6s}")
    hs = []
    for s in list(seqs) + ["COMBINED_SEQ"]:
        m = out[s]["pedestrian"]
        h = m["HOTA"]["HOTA"].mean()
        print(f"{s:>10s} {h:7.3f} {m['HOTA']['DetA'].mean():7.3f} "
              f"{m['HOTA']['AssA'].mean():7.3f} {m['HOTA']['LocA'].mean():7.3f} "
              f"{m['Identity']['IDF1']:7.3f} {m['CLEAR']['IDSW']:6.0f}")
        if s != "COMBINED_SEQ":
            hs.append(h)
    print(f"\n  challenge score (mean over sequences): HOTA {sum(hs)/len(hs):.3f}")
    print("  published: ByteTrack 0.42 | Deep-EIoU 0.54 | GTATrack (winner) 0.60")
    return {s: {"HOTA": float(out[s]["pedestrian"]["HOTA"]["HOTA"].mean()),
                "DetA": float(out[s]["pedestrian"]["HOTA"]["DetA"].mean()),
                "AssA": float(out[s]["pedestrian"]["HOTA"]["AssA"].mean()),
                "LocA": float(out[s]["pedestrian"]["HOTA"]["LocA"].mean()),
                # Detection recall AND precision: a filter that raises one by
                # spending the other must be visible as both (N19).
                "DetRe": float(out[s]["pedestrian"]["HOTA"]["DetRe"].mean()),
                "DetPr": float(out[s]["pedestrian"]["HOTA"]["DetPr"].mean()),
                "IDF1": float(out[s]["pedestrian"]["Identity"]["IDF1"]),
                "IDSW": float(out[s]["pedestrian"]["CLEAR"]["IDSW"])}
            for s in seqs}


def demo():
    """Self-check: staging must preserve the box count and shift frames by exactly 1."""
    s = SEQS[0]
    stage_gt([s])
    raw = [l for l in open(f"{GT_SRC}/{s}.txt") if l.strip()]
    st = [l for l in open(os.path.join(_root("gt"), s, "gt", "gt.txt")) if l.strip()]
    assert len(raw) == len(st), (len(raw), len(st))
    assert int(st[0].split(",")[0]) == int(raw[0].split(",")[0]) + 1
    assert st[0].split(",")[6:9] == ["1", "1", "1\n"], st[0]
    # stage_pred was never exercised, and it carries the more dangerous of the
    # two conventions: tracks.json is bottom-centre + height, MOTChallenge is
    # top-left. Stage one synthetic box and check the corner arithmetic.
    os.makedirs(WORK, exist_ok=True)
    probe = f"{WORK}/_demo.tracks.json"
    json.dump({"fps": 25.0, "detections": [
        {"f": 0, "id": 7, "x": 100.0, "y": 200.0, "w": 20.0, "h": 50.0}]},
        open(probe, "w"))
    global VARIANT
    keep = VARIANT
    VARIANT = "_demo"
    try:
        _seqinfo("", 10, 4096, 1080, 1)
        stage_pred(["", ])          # seq "" -> f"{WORK}/_demo.tracks.json"
        line = open(os.path.join(_root("trackers"), TRACKER, "data", ".txt")).read().split(",")
        # THE CADENCE TRAP, planted: a stride-2 tracker's even-only frames against
        # GT staged on every frame. TrackEval would score it - every odd frame a
        # miss, recall halved by a cadence - so stage_pred must refuse instead.
        json.dump({"fps": 25.0, "detections": [
            {"f": f, "id": 7, "x": 100.0, "y": 200.0, "w": 20.0, "h": 50.0}
            for f in range(0, 10, 2)]}, open(probe, "w"))
        for gt_every, claimed, why in ((1, None, "even-only predictions vs every-frame GT"),
                                       (2, 1, "a caller's stride that is not the staged one")):
            _seqinfo("", 10, 4096, 1080, gt_every)
            try:
                stage_pred([""], claimed)
                raise AssertionError(f"stage_pred scored {why}")
            except ValueError:
                pass
        _seqinfo("", 10, 4096, 1080, 2)
        stage_pred([""], 2)          # the same frames on the stride they were made at
        json.dump({"fps": 25.0, "detections": [
            {"f": f, "id": 7, "x": 100.0, "y": 200.0, "w": 20.0, "h": 50.0}
            for f in range(10)]}, open(probe, "w"))
        try:
            stage_pred([""], 2)
            raise AssertionError("odd-frame predictions against GT staged every 2 were scored")
        except ValueError:
            pass
    finally:
        VARIANT = keep
        os.remove(probe)
    assert int(line[0]) == 1, f"frame not shifted +1: {line[0]}"
    assert abs(float(line[2]) - 90.0) < 1e-6, f"x should be 100-20/2=90, got {line[2]}"
    assert abs(float(line[3]) - 150.0) < 1e-6, f"y should be 200-50=150, got {line[3]}"

    # A CONCURRENT STAGE MUST NOT TOUCH THIS PROCESS'S gt.txt. stage_gt opens
    # with "w", so before the path carried a pid a second run truncated the file
    # this one was scoring against: on 2026-09-23 that turned a stable claim into
    # DRIFT twice, at 32.1 s and 14.0 s against a normal 60-80 s - half the boxes,
    # half the time. Fork a real competitor and watch the size.
    mine = os.path.join(_root("gt"), s, "gt", "gt.txt")
    size = os.path.getsize(mine)
    code = ("import sys;sys.path.insert(0,%r);import st2_mot as M;"
            "M.stage_gt([%r])" % (os.path.dirname(os.path.abspath(__file__)), s))
    proc = subprocess.Popen([sys.executable, "-c", code])
    seen = set()
    while proc.poll() is None:
        try:
            seen.add(os.path.getsize(mine))
        except OSError:
            seen.add(-1)
    proc.wait()
    assert seen, "the probe never sampled - the child exited before the first poll"
    assert seen <= {size}, (
        f"a concurrent stage_gt moved this process's gt.txt: sizes {sorted(seen)} "
        f"against {size}. The staging root is not process-scoped, and every "
        f"TrackEval score racing it is measured against a partial ground truth.")

    # EVERY root this process hands out must be registered for cleanup, not just
    # the one VARIANT happens to hold at exit. st2_product scores six variants in
    # one process; re-deriving a single path at exit leaked the other five, and 68
    # trees / 3.1 GB were on disk when review looked.
    _before = set(_STAGED)
    _keep = VARIANT
    try:
        for _v in ("_demo_a", "_demo_b"):
            VARIANT = _v
            _root("gt")
    finally:
        VARIANT = _keep
    _new = _STAGED - _before
    assert len(_new) == 2, f"a root was handed out without being registered: {_new}"
    assert all(str(os.getpid()) in d for d in _new), _new
    _STAGED.difference_update(_new)   # nothing was created on disk; do not queue them

    print(f"demo ok: {len(st)} gt boxes staged, frames shifted +1, conf column "
          f"usable, bottom-centre -> top-left verified, a stride-2 tracker against "
          f"every-frame GT refused")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seqs", nargs="*", default=list(SEQS))
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        stage_gt(a.seqs)
        stage_pred(a.seqs)
        run(a.seqs)
