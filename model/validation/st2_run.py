"""Produce the shipped `.c03`/`.c03s` tracker output for an ST2 sequence, from a CLI.

WHY THIS EXISTS
    `generalisation-audit (not published)` §6.3: the shipped `.c03s` config is exactly
    reproducible but **no checked-in command produced it** - `model/engine/track.py`
    gates stitching on `cal is not None` and never references `HeightScale`. That
    gap is how `.oracle` became unreproducible: two files with different
    constructions, no producer, and no way to tell what either represented.

    So this is the producer. Two stages, both addressable:

        stage 1  model/engine/track.py at conf 0.3, no --remap, no --calib   -> .c03
        stage 2  HeightScale + split + tracklets + stitch + apply      -> .c03s

    `demo()` reproduces the shipped `128057.c03s` from its `.c03` and asserts
    every box matches, so the stage-2 chain cannot drift from what was measured.

NO --dets-cache, DELIBERATELY
    `detection-tracking-evidence (not published)#evidence-ball` ("ONE TRAP", moved from detectors.py;
    caches now record conf and refuse a mismatch, #cache): the cache keyed on video and start frame, NOT
    on conf. Reusing one across a confidence sweep replays the old boxes and
    silently returns the old answer - which is exactly how
    `128057.cbiou.c0.{10,25,40}` came to hold byte-identical detection sets and
    void the confidence sweep that was read off them. A re-run that shares a
    cache is not a re-run.

THE DETECTOR IS coreml, AND THAT IS NOT A PERFORMANCE CHOICE
    `track.py` defaults to `-d rfdetr` (PyTorch). The shipped `.c03` files were
    NOT made with it. Probed on 128057's first 200 frames against the file on
    disk: **coreml reproduces 4,045 of 4,045 boxes exactly; rfdetr matches 15**
    (0.4%), and returns a different box count and a different track count. The
    two backends are not interchangeable and the difference is silent.

    So a held-out run on the PyTorch path would be scored against a tuned pair
    built by a different detector, and every held-out delta would carry that
    difference as if it were generalisation. `demo()` asserts the identity.

    (coreml is also 2.7x faster here - 3.35 fps against 1.23 - which is how the
    discrepancy was noticed at all. Speed is the symptom, not the reason.)

THE OUTPUT IS PANORAMA PIXELS WITH w/h
    Same convention as `st2_mot.py`: raw 4096x1080 frame, no remap, no calib. So
    unlike `data/jobs/118575_full_1st/tracks.*` (remapped space, no box size),
    these files CAN carry the detector box-bottom bias measurement that
    `context-pitch-mapping-pivot (not published)` §8.2 lists as unmeasured.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.stitch as st  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

CLIPS = _ST2 + "/mot/clips"
OUT = "model/validation/out/stc2025"
FPS = 25
CONF = 0.3
DETECTOR = "coreml"   # see the module docstring - NOT a performance choice
# Never tracked at all, then tracked only at conf 0.5. Together these take the
# held-out split from 3 sequences to 8 (generalisation-audit.md §7.1, §7.2).
MISSING = ("117092", "118575", "118577", "117093", "128058")


def track(seq, force=False):
    """Stage 1 -> {seq}.c03.tracks.json. Returns the path."""
    dst = f"{OUT}/{seq}.c03.tracks.json"
    if os.path.exists(dst) and not force:
        print(f"  {seq}: .c03 exists, skipping detector")
        return dst
    cmd = [sys.executable, "model/engine/track.py", f"{CLIPS}/{seq}.mp4",
           "-c", str(CONF), "-d", DETECTOR, "--roles", "-o", dst]
    print(f"  {seq}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return dst


def stitch_file(src, dst):
    """Stage 2. The chain is generalisation-audit.md §6.3's, verified by demo()."""
    raw = json.load(open(src))["detections"]
    cal = st.HeightScale.fit(raw)
    if cal is None:
        raise SystemExit(f"{src}: HeightScale.fit returned None, cannot stitch")
    rows = st.split(raw, cal, fps=FPS)
    lab = collections.defaultdict(collections.Counter)
    for r in rows:
        lab[int(r["id"])][r.get("team", "unknown")] += 1
    labels = {k: v.most_common(1)[0][0] for k, v in lab.items()}
    out = st.apply(rows, st.stitch(st.tracklets(rows, cal), FPS, teams=labels))
    # Record what produced this. The absence of exactly this is why .oracle is
    # unreproducible; a file that cannot say what it is becomes a mystery.
    json.dump({"fps": float(FPS), "detections": out,
               "_config": {"stage1": f"model/engine/track.py -c {CONF} -d {DETECTOR} --roles",
                           "stage2": "stitch.HeightScale+split+tracklets+stitch+apply",
                           "producer": "model/validation/st2_run.py", "dets_cache": None}},
              open(dst, "w"))
    print(f"  -> {dst}: {len(out)} boxes, {len({r['id'] for r in out})} ids "
          f"(from {len({r['id'] for r in raw})} raw)")


def run(seqs, force=False):
    for seq in seqs:
        if not os.path.exists(f"{CLIPS}/{seq}.mp4"):
            print(f"  {seq}: no clip, skipping")
            continue
        print(f"\n=== {seq} ===", flush=True)
        stitch_file(track(seq, force), f"{OUT}/{seq}.c03s.tracks.json")


def probe_detector(seq="128057", n=200):
    """Stage 1's detector must reproduce the shipped .c03 on the same frames.

    This is the check that matters: the backends disagree silently, and a
    held-out run built by the wrong one is scored against a tuned pair built by
    the other. Costs about a minute.
    """
    ref = f"{OUT}/{seq}.c03.tracks.json"
    if not os.path.exists(ref):
        print(f"skip detector probe: no {ref}")
        return
    tmp = f"{OUT}/_probe.tracks.json"
    subprocess.run([sys.executable, "model/engine/track.py", f"{CLIPS}/{seq}.mp4",
                    "-c", str(CONF), "-d", DETECTOR, "-n", str(n), "-o", tmp],
                   check=True, capture_output=True)
    key = lambda R: sorted((r["f"], round(r["x"], 1), round(r["y"], 1),
                            round(r["w"], 1)) for r in R if r["f"] < n)
    a = key(json.load(open(ref))["detections"])
    b = key(json.load(open(tmp))["detections"])
    os.remove(tmp)
    hit = len(set(a) & set(b))
    assert hit == len(a), (f"detector {DETECTOR} reproduces {hit}/{len(a)} boxes "
                           f"of the shipped {seq}.c03 - wrong backend")
    print(f"ok  (detector {DETECTOR} reproduces {hit}/{len(a)} boxes of {seq}.c03)")


def demo():
    """Stage 2 must reproduce the shipped 128057.c03s box for box."""
    src = f"{OUT}/128057.c03.tracks.json"
    shipped = f"{OUT}/128057.c03s.tracks.json"
    if not (os.path.exists(src) and os.path.exists(shipped)):
        print("skip: 128057 .c03/.c03s not on disk")
        return
    tmp = f"{OUT}/_demo.c03s.tracks.json"
    stitch_file(src, tmp)
    key = lambda R: sorted((r["f"], round(r["x"], 2), round(r["y"], 2), r["id"])
                           for r in R)
    a = key(json.load(open(tmp))["detections"])
    b = key(json.load(open(shipped))["detections"])
    os.remove(tmp)
    assert a == b, f"stage 2 drifted: {len(a)} vs {len(b)} boxes"
    print(f"ok  ({len(a)} boxes reproduce 128057.c03s exactly)")
    probe_detector()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=list(MISSING))
    ap.add_argument("--force", action="store_true", help="re-detect even if .c03 exists")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else run(a.seqs, a.force)
