"""N68: 128057 accepted from nothing at a pitch length it does not have - every run of n68-128057-length (not published).

    .venv/bin/python model/validation/n68_128057_length.py q1               # is 109.14 m wrong? + reference lengths, per-group table
    .venv/bin/python model/validation/n68_128057_length.py q2 [--seq 128058] # which step: candidates, fix_length, from_install
    .venv/bin/python model/validation/n68_128057_length.py q3               # map displacement, accepted vs install camera
    .venv/bin/python model/validation/n68_128057_length.py q4               # autocal of each commit on the same frame

Inputs: the committed medians model/validation/out/stc2025/<clip>.median40.png, Pitch() (105 x 68), CPU. Runs go
to a process pool, one process per run. Promoted from earlier scratch drivers (i128_run.py, i128_q3.py,
i128_old.py); the pre-registration below is the doc's section 0, copied, not rewritten.

PRE-REGISTRATION (n68-128057-length (not published) section 0)

The trigger, reported in review and not yet re-run here: on
`model/validation/out/stc2025/128057.median40.png`, `autocal.autocal(img, Pitch())` accepts with 681 hits,
line RMS 0.862 px and length 109.14 m (sd 0.088). N34 refused this match from nothing (292 hits) and
accepted it only via the install camera; seven same-install matches were said to read 105.0-105.25 m.

**Ground truth.** SoccerTrack v2's own metadata, `raw/128057/128057_tracker_box_metadata.xml`, element
`<pitch width="105" height="68"/>` (ST2 calls the long side `width`). Caveat stated up front: every one of
the ten ST2 matches carries the identical element, including the other venues (132831, 132877), so it may
be a dataset default rather than a survey of this ground. It is therefore paired with a REFERENCE (not
truth): the length the current `autocal` reads on the other same-install matches from their own median
frames. GT_L = 105.0 m.

**Q1 - is 109.14 m wrong?** Wrong iff |length - GT_L| > max(3 x length_sd, 0.5 m). Also answered against
the reference with the same rule.

**Q2 - which step produces it?** Candidates in this order, each with its discriminating run:
- (a) the coarse length grid / `_settle` choosing between local optima - the length scan (hits vs length over
  `LENGTHS_COARSE` + `LENGTH_FINE`) for 128057 vs 118575; is there a 105 m peak that loses?
- (b) the camera search converging to a different mount than the install - camera centre / height / f of the
  accepted 128057 camera vs the other same-install matches; `from_install` from the N34 install camera.
- (c) the paint on this frame pulling the fit - `autocal(..., fix_length=True)` at GT: does `accept()` still
  pass, with what hits and RMS; hits by line group at 109 vs 105.
  Deciding rule: (a) if the scan has a peak within 1 m of GT whose hits are within 10% of the winner's; (b) if
  the accepted camera's centre is > 1 m from the install's median centre; (c) if at GT the fit loses a line
  family or > 10% of hits that it keeps at 109. More than one may hold; each is reported.

**Q3 - what does the wrong camera cost?** Map displacement between the accepted 109 m camera and the best
reference camera (from_install at GT length if accepted, else fix_length at GT) on the 12 x 8 pitch grid
(`check.py`'s `_grid`): the same pixel read through both, median and max metres.

**Q4 - why did N34 refuse and today's code accept?** Run the autocal of successive commits on the same frame
(`git show <commit>:<path>` into a scratch module) and name the first commit whose run accepts.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEDIAN = ROOT / "model" / "validation" / "out" / "stc2025" / "{}.median40.png"
N34 = ROOT / "model" / "validation" / "out" / "stc2025" / "n34_autocal.json"
REFS = ("117093", "118575", "118576", "118577", "118578", "128058")   # same install, daylight, not 128057
GT_L = 105.0
COMMITS = ("887cbee", "9e1bd73", "caca57e", "55506c0", "d6040d9", "b75395c", "HEAD")
PEN = ("pen_l_front", "pen_r_front", "pen_r_far")


def _job(kind, seq, code=None):
    """One run in its own process (max_tasks_per_child=1), so an old commit's modules never meet today's.
    kind: scan = autocal free length, every _settle recorded | fix = autocal(fix_length=True)
          | install = from_install from the N34 install camera, free and fixed | old = autocal of `code`."""
    import cv2  # noqa: PLC0415
    base = code or str(ROOT / "model")
    sys.path[:0] = [os.path.join(base, "validation"), os.path.join(base, "engine")]
    if code:
        os.chdir(base)                                         # N34's module resolves its paths from here
    from pitch import Pitch  # noqa: PLC0415
    if os.path.exists(os.path.join(base, "engine", "autocal.py")):
        import autocal as A  # noqa: PLC0415
    else:
        import n34_autocal as A  # noqa: PLC0415 - 887cbee: autocal lived in validation/
    img, P, calls = cv2.imread(str(MEDIAN).format(seq)), Pitch(), []
    if kind != "old" and hasattr(A, "_missing"):   # F1 shipped (doc section 5): this investigates the rule it replaced
        A._missing = lambda rep: sorted({"touch", "goal", "halfway", "pen", "circle"}
                                        - {k.split("_")[0] for k, (h, n) in rep["hits_by_group"].items() if h >= 0.25 * n})
    if kind in ("scan", "fix"):
        settle = A._settle

        def rec(*a, **k):
            cam, rep, Pf = settle(*a, **k)
            ok, C = A.accept(cam, rep)
            calls.append({"accepted": ok, "hits": rep["hits"], "rms": rep["line_rms_px"], "length_m": Pf.length,
                          "centre": C.tolist(), "f": float(abs(cam[0])), "groups": rep["hits_by_group"],
                          "scan": rep.get("length_scan")})
            return cam, rep, Pf
        A._settle = rec
    if kind == "install":
        n34 = json.load(open(N34))
        start = np.median([v["camera"] for q, v in n34.items()
                           if v["camera"] is not None and v.get("via") != "install" and q != seq], axis=0)
        out = {}
        for fl in (False, True):
            cam, rep = A.from_install(img, P, start, fix_length=fl)
            out[f"fixed={fl}"] = {"camera": None if cam is None else np.asarray(cam).tolist(), **rep}
        return out
    cam, rep = A.autocal(img, P, fix_length=True) if kind == "fix" else A.autocal(img, P)
    return {"camera": None if cam is None else np.asarray(cam).tolist(), "calls": calls,
            **{k: rep.get(k) for k in ("accepted", "hits", "line_rms_px", "length_m", "length_sd_m",
                                       "camera_centre_m", "f_px_per_rad", "hits_by_group", "why")}}


def _run(jobs):
    with ProcessPoolExecutor(max_workers=min(len(jobs), max(1, os.cpu_count() // 2)), max_tasks_per_child=1) as ex:
        futs = {j: ex.submit(_job, *j) for j in jobs}
        return {j: f.result() for j, f in futs.items()}


def _row(name, r):
    c = r.get("camera_centre_m") or [np.nan] * 3
    rms = r.get("line_rms_px")
    print(f"{name:34s} accepted={r['accepted']} hits={r['hits']} rms={rms and round(rms, 3)} length={r['length_m']}"
          f" sd={r.get('length_sd_m') and round(r['length_sd_m'], 3)} centre={np.round(c, 2).tolist()}"
          f" f={r.get('f_px_per_rad') and round(r['f_px_per_rad'])}")


def _share(g, name):
    h, n = g[name]
    return f"{h}/{n}={h / n:.2f}"


def q1():
    res = _run([("scan", "128057")] + [("scan", q) for q in REFS] + [("install", "128057")])
    r = res[("scan", "128057")]
    _row("128057 from nothing", r)
    off, thr = abs(r["length_m"] - GT_L), max(3 * (r["length_sd_m"] or 0), 0.5)
    print(f"|length - GT_L| = {off:.2f} m vs max(3 sd, 0.5) = {thr:.2f} m -> {'WRONG' if off > thr else 'not wrong'}")
    print("reference lengths (from nothing):")
    for q in REFS:
        _row(f"  {q}", res[("scan", q)])
    _row("  128057 from_install free length", res[("install", "128057")]["fixed=False"])
    # the per-group table: every accepted _settle result on the seven daylight medians, plus 128057 from_install
    rows = [(q, c) for (k, q), v in res.items() if k == "scan" for c in v["calls"] if c["accepted"]]
    inst = res[("install", "128057")]["fixed=False"]
    rows.append(("128057/install", {"length_m": inst["length_m"], "hits": inst["hits"], "groups": inst["hits_by_group"]}))
    wrong = [(q, c) for q, c in rows if abs(c["length_m"] - GT_L) > 2.0]
    right = {(q, c["hits"], c["length_m"]): c for q, c in rows if abs(c["length_m"] - GT_L) <= 2.0}
    print(f"per-group share on paint; wrong = accepted with |length - {GT_L}| > 2 m; correct: n={len(right)} distinct (clip, hits, length)")
    for q, c in wrong:
        print(f"  wrong {q} @{c['length_m']}: " + " ".join(f"{g} {_share(c['groups'], g)}" for g in PEN))
    print("  correct min: " + " ".join(f"{g} {round(min(h / n for h, n in [c['groups'][g] for c in right.values()]), 2)}" for g in PEN))


def q2(seq):
    jobs = [("scan", seq)] + ([("fix", seq), ("install", seq)] if seq == "128057" else [])
    res = _run(jobs)
    s = res[("scan", seq)]
    for i, c in enumerate(s["calls"]):
        sc = c["scan"] or {}
        peak = max(sc, key=sc.get) if sc else None
        print(f"  candidate {i}: accepted={c['accepted']} hits={c['hits']} length={c['length_m']} "
              f"height={c['centre'][2]:.2f} hits@105.0={sc.get('105.0')} scan peak {peak}:{sc.get(peak)} "
              f"hits@110.0={sc.get('110.0')}")
    _row(f"{seq} autocal free length", s)
    if seq != "128057":
        return
    f, i = res[("fix", seq)], res[("install", seq)]
    at105 = [c["scan"]["105.0"] for c in s["calls"] if c["scan"]]
    print(f"candidates' hits at 105.0: {min(at105)}-{max(at105)}")
    _row(f"{seq} autocal fix_length 105.0", f)
    _row(f"{seq} from_install fix_length 105.0", i["fixed=True"])
    _row(f"{seq} from_install free length", i["fixed=False"])
    for name, g in (("accepted free", s["hits_by_group"]), ("from_install free", i["fixed=False"]["hits_by_group"])):
        print(f"  {name}: pen_r_front {_share(g, 'pen_r_front')} pen_l_front {_share(g, 'pen_l_front')}")


def q3():
    """The same pixel through the accepted camera and the install-started cameras, on check.py's 12 x 8 grid."""
    sys.path[:0] = [str(ROOT / "model" / "engine"), str(ROOT / "model" / "validation")]
    import autocal as A  # noqa: PLC0415
    from check import _grid  # noqa: PLC0415
    res = _run([("scan", "128057"), ("install", "128057")])
    s = res[("scan", "128057")]
    qa, La = np.array(s["camera"]), s["length_m"]
    for name in ("fixed=True", "fixed=False"):
        ref = res[("install", "128057")][name]
        qb, Lb = np.array(ref["camera"]), ref["length_m"]
        gb = _grid() + [(Lb - 105.0) / 2, 0]                     # the same centre-relative grid in B's frame
        ra = A._inv(qa, A._fwd(qb, gb, 4096.0), 4096.0)
        dc = np.linalg.norm(ra - [La / 2, 0] - (gb - [Lb / 2, 0]), axis=1)   # centre-spot aligned
        dk = np.linalg.norm(ra - gb, axis=1)                                 # corner aligned (what the output says)
        sx = np.polyfit(gb[:, 0] - Lb / 2, ra[:, 0] - La / 2, 1)[0]
        print(f"ref from_install {name} L_ref={Lb} L_acc={La} | centre-aligned median {np.median(dc):.2f} max {dc.max():.2f} m"
              f" | corner-aligned median {np.median(dk):.2f} max {dk.max():.2f} m | x slope {sx:.3f}")


def q4():
    """autocal of each commit (git archive of its engine/ and validation/), then the working copy."""
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for c in COMMITS:
            d = os.path.join(tmp, c)
            os.makedirs(d)
            paths = subprocess.run(["git", "ls-tree", "--name-only", c], cwd=ROOT, capture_output=True, text=True,
                                   check=True).stdout.split()
            pre = "model/" if "model" in paths else ""
            blob = subprocess.run(["git", "archive", c, f"{pre}engine", f"{pre}validation"], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            tf = tempfile.NamedTemporaryFile(suffix=".tar", dir=tmp, delete=False)
            tf.write(blob)
            tf.close()
            with tarfile.open(tf.name) as t:
                t.extractall(d, filter="data")
            dirs[c] = os.path.join(d, pre.rstrip("/"))
        res = _run([("old", "128057", dirs[c]) for c in COMMITS] + [("old", "128057", None)])
    for c in COMMITS + ("working copy",):
        r = res[("old", "128057", dirs.get(c))]
        print(f"{c:12s} accepted={r['accepted']} hits={r['hits']} rms={round(r['line_rms_px'], 3)} length={r['length_m']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("q", choices=("q1", "q2", "q3", "q4"))
    ap.add_argument("--seq", default="128057", help="q2 only: the median to scan (128057, or 128058 for the latent case)")
    a = ap.parse_args()
    {"q1": q1, "q2": lambda: q2(a.seq), "q3": q3, "q4": q4}[a.q]()
