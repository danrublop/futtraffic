"""A/B the identity stage's torso-histogram term: use_appearance True vs False.

THE TWO-ARM A/B IS RETIRED (2026-09-20). It ran at commit 0a15ea6, before the
removal; its result is `out/appearance-policy/appearance_ab.json` and
appearance-term-decision (not published) §2 - identical mapping on 9 of 9, 0 of 512
tracks regrouped. `identity.use_appearance` and `identity.torso_hist` no longer
exist, so there is no second arm to run and no `torso_hist` left to time; what
proves the removal is `--shipped` / `--compare`, below. The protocol is kept
because it is the record of what the retired arm did.

PROTOCOL (committed before the run). Population: the ST2 sequences N21 scored,
read from their cached `.p21.tracks.json` - the product's own tracker output at
what ships. Held fixed: the same tracks, the same calibration file N21 used
(`{seq}.p05.calibration.json`), the same `scan_video` pass (one decode, the
shipped docTR backend, samples_per_track=60), the same `merge_tracklets`.

ONE DECODE, TWO MERGES, and that is exactly the flag. `AssistedIdentity.assign`
differs between the two arms in one statement - `if not self.use_appearance:
embeddings = {}` - which happens AFTER `scan_video` returns and BEFORE
`merge_tracklets` is called. So scanning once and merging twice is the flag
bit-for-bit, and it halves the video decode.

Success criterion, named before the run: the flag BUYS something iff the two
arms produce a different track_id -> group mapping on at least one sequence.
Counts alone are not the test - two different merges can give the same count.

    .venv/bin/python model/validation/appearance_ab.py --seq 118575

--------------------------------------------------------------------------------
`--shipped`: THE REMOVAL'S REGRESSION PROOF (added 2026-09-20, before the removal)
--------------------------------------------------------------------------------
The A/B above compares two arms inside one process. It does not prove that
DELETING the code reproduces what ships, because the deletion also rewrites
`merge_tracklets`, `_pair_costs` and `scan_video`.

So `--shipped` runs `jobs.apply_identity` - the job's own stage, the function
`model/validation/st2_product.identify` calls - and records the only thing the
claim is about: the `raw_id -> id` map each sequence comes out with, per
sequence, written to `shipped_<label>.json`. Run it BEFORE the removal
(`--label before`) and AFTER (`--label after`), then:

    .venv/bin/python model/validation/appearance_ab.py --compare before after

which diffs the maps key by key and exits non-zero on the first difference. The
claim is identity of the mapping on all nine sequences, against the
appearance-ON arm - not against the OFF arm.

It also records `seconds` per sequence (the whole stage, decode included). The
runtime the removal gave back was measured the same way while `torso_hist` still
existed and is in appearance-term-decision (not published); `--time-merge` still bounds
what the merge costs without the video.
"""
import argparse
import json
import math
import os
import sys
import time
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "engine"),
                os.path.join(ROOT, "app", "server"),
                os.path.join(ROOT, "model", "validation")]

OUT = os.path.join(ROOT, "model", "validation", "out", "appearance-policy")
CLIPS = _ST2 + "/mot/clips"
WORK = os.path.join(ROOT, "model", "validation", "out", "stc2025")
ALL9 = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132877")
#: How many times --time-merge repeats the merge before taking the MINIMUM. Five
#: is enough for the minimum to settle on this machine and keeps the claim under
#: a second; the quantity is a bound on the merge's cost, not a benchmark.
REPEATS = 5


def shipped(seq: str, tag: str = ".p21") -> dict:
    """`jobs.apply_identity` on one sequence -> the raw_id -> id map it produces.

    The stage the job runs, not a reimplementation of it: same entry point as
    `st2_product.identify`.
    """
    import identity
    import jobs

    tracks = json.load(open(os.path.join(WORK, f"{seq}{tag}.tracks.json")))
    cal = jobs.load_calibration(os.path.join(WORK, f"{seq}.p05.calibration.json"))
    t0 = time.perf_counter()
    out, report = jobs.apply_identity(
        identity, tracks, os.path.join(CLIPS, f"{seq}.mp4"), cal)
    seconds = time.perf_counter() - t0
    # raw_id is written by apply_identity for exactly this traceability. A
    # detection the stage did not touch carries no raw_id and keeps its own id.
    mapping = {str(d.get("raw_id", d["id"])): int(d["id"]) for d in out["detections"]}
    return {"seq": seq, "tag": tag, "tracks": len(mapping),
            "groups": len(set(mapping.values())), "mapping": mapping,
            "seconds": round(seconds, 2),
            "crops_attempted": report.get("crops_attempted")}


def time_merge(tag: str = ".p21") -> None:
    """The merge alone, no video: how much of the stage the merger is at all."""
    import identity
    import jobs

    # THE MINIMUM OVER REPEATS, not one sample. A single timing is a measurement
    # of the machine's load as much as of the code, and this claim is re-run by a
    # gate that deliberately runs four things at once: on 2026-09-24 it came back
    # DRIFT at -j4 and reproduced exactly when run alone. Contention can only
    # make a sample SLOWER, so the minimum is the robust estimator - it converges
    # on the uncontended cost from above and cannot be flattered by load.
    n, best = 0, math.inf
    loaded = []
    for s in ALL9:
        d = json.load(open(os.path.join(WORK, f"{s}{tag}.tracks.json")))
        cal = jobs.load_calibration(os.path.join(WORK, f"{s}.p05.calibration.json"))
        loaded.append((identity.by_track(d), float(d["fps"]), cal))
    n = sum(len(by) for by, _f, _c in loaded)
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for by, fps, cal in loaded:
            identity.merge_tracklets(by, fps, cal)
        best = min(best, time.perf_counter() - t0)
    t = best
    # An UPPER BOUND, rounded up to the next 0.1 s, because the raw value
    # jitters run to run and a claim the checker re-runs must not drift across a
    # rounding boundary. A cost claim is honest as a bound anyway.
    print("merge_only_seconds_at_most", math.ceil(t * 10) / 10,
          "tracks", n, "raw", round(t, 3))


def compare(a_label: str, b_label: str) -> int:
    A = json.load(open(os.path.join(OUT, f"shipped_{a_label}.json")))
    B = json.load(open(os.path.join(OUT, f"shipped_{b_label}.json")))
    ra = {r["seq"]: r for r in A["rows"]}
    rb = {r["seq"]: r for r in B["rows"]}
    assert set(ra) == set(rb), f"different sequences: {sorted(ra)} vs {sorted(rb)}"
    bad = []
    for s in sorted(ra):
        if ra[s]["mapping"] != rb[s]["mapping"]:
            d = [k for k in ra[s]["mapping"]
                 if ra[s]["mapping"].get(k) != rb[s]["mapping"].get(k)]
            bad.append((s, len(d), d[:5]))
    tracks = sum(r["tracks"] for r in ra.values())
    groups = sum(r["groups"] for r in ra.values())
    print(f"seqs {len(ra)} tracks {tracks} groups {groups} "
          f"identical {len(ra) - len(bad)} of {len(ra)} differing_tracks "
          f"{sum(b[1] for b in bad)}")
    for s, n, ex in bad:
        print(f"  {s}: {n} tracks regrouped, e.g. {ex}")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", action="append", default=None)
    ap.add_argument("--tag", default=".p21")
    ap.add_argument("--shipped", action="store_true",
                    help="run jobs.apply_identity and record the raw_id -> id map")
    ap.add_argument("--label", default="before", help="suffix for --shipped's output")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--time-merge", action="store_true",
                    help="time merge_tracklets alone on the cached tracks")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.compare:
        raise SystemExit(compare(*a.compare))
    if a.time_merge:
        return time_merge(a.tag)
    if a.shipped:
        rows = [shipped(s, a.tag) for s in (a.seq or list(ALL9))]
        for r in rows:
            print(json.dumps({k: v for k, v in r.items() if k != "mapping"}),
                  flush=True)
        with open(os.path.join(OUT, f"shipped_{a.label}.json"), "w") as fh:
            json.dump({"protocol": "model/validation/appearance_ab.py docstring",
                       "label": a.label, "rows": rows}, fh)
        print("stage seconds", round(sum(r["seconds"] for r in rows), 1))
        return
    ap.error("nothing to do: --shipped, --compare A B or --time-merge. The "
             "two-arm A/B is retired - see this file's header")


if __name__ == "__main__":
    main()
