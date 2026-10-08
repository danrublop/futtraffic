"""Engine regression checks, one subcommand per section: calib, track, detect, identity, measure, ball.

    .venv/bin/python model/validation/check.py calib --fast            # self-checks + golden round-trip
    .venv/bin/python model/validation/check.py calib --real            # the shipped search on real frames
    .venv/bin/python model/validation/check.py calib --rebless "why"   # re-freeze the golden, with a reason
    .venv/bin/python model/validation/check.py calib --prove           # each planted defect must FAIL
    .venv/bin/python model/validation/check.py track  --fast|--prove|--rebless "why"
    .venv/bin/python model/validation/check.py detect --fast|--prove|--rebless "why"
    .venv/bin/python model/validation/check.py identity --fast|--real|--prove|--rebless "why"
    .venv/bin/python model/validation/check.py measure --fast|--real|--prove|--rebless "why"|--make-fixture
    .venv/bin/python model/validation/check.py ball --fast|--real|--prove|--rebless "why"
    .venv/bin/python model/validation/check.py status [--section P] [--json]   # READ-ONLY: goldens, cache
                     staleness, runs, processes, app jobs - no producer runs (rules in check_status.py)

    Every section run appends {t, section, mode, exit, git} to model/validation/out/check-runs.jsonl
    (gitignored, local state); `status` shows the latest per section and mode.

    exit 0 pass | 1 regression | 2 could not run (missing fixture, hash mismatch, killed self-check).
    "Could not run" is never a pass.

CALIB - WHY
    The shipped panorama path (jobs.auto_calibrate -> autocal.calibrate -> calibrators.SphericalCamera)
    had no guard on real footage: its self-checks render synthetic pitches or mock the search, so kappa,
    MIN_HITS, the grid or a dropped to_dict key could all move or refuse every real panorama in green.

    --fast   pitch.py (runs calibrators._demo), track.py, sceneprior.py and jobs.py self-checks, in
             parallel subprocesses; plus the committed golden 118575 calibration.json through
             calibrators.from_dict: to_dict() keys must equal the golden's EXACTLY, and to_image/to_pitch
             on a fixed 12x8 pitch grid must equal the frozen values within 1e-6.
    --real   jobs.auto_calibrate for real on committed median frames (fixtures/calib/), no install camera,
             no typed length. Only the video decode (autocal.median_frame) and the frame-shape probe are
             stubbed. 118575 must be accepted and 117092 refused, as frozen; the reason is compared too.
             128057 must be REFUSED, by accept() itself: from nothing it was accepted at 109.14 m (ST2 says
             105) until the N68 fix F1 required both penalty fronts on paint (n68-128057-length (not published)
             section 5). That is the fixture refused BY accept() that 117092 (refused by the grass-share
             gate, before accept() is reached) could not be, so a loosened accept() now fails --real;
             --prove plants the pooled rule back and must see it fail. KNOWN_WRONG is empty; an entry there
             prints a KNOWN-WRONG line every run, meaning "unchanged", never "correct".

TOLERANCES - registered 2026-09-28 from run-to-run variation, BEFORE any comparison to the golden was judged.
    n=2 runs of 118575 and n=1 of 117092, same code, same fixture, same machine: the search is seeded
    (rng 0) and deterministic - run 2 reproduced run 1's camera numbers bit for bit, and its hits; the
    117092 run was refused on grass share, as N34 recorded.
    So the bounds are tight, not the 0.05 m / 0.2 m / 5% a reviewer suggested:
        map displacement vs golden on the 12x8 grid   median <= 0.02 m, max <= 0.1 m
        hits                                          within +-2% of golden
        accepted / refused                            exact; the refusal reason exact
        calibration.json keys                         exact
    A platform or BLAS change that breaks bit-determinism has this much slack before it reads as a
    regression; anything that moves the real camera more than that is a regression or a --rebless.

REBLESS writes fixtures/calib/golden.json in one piece (never partially), stamped with the reason,
    date, git HEAD and the constants in force, and refuses to freeze a golden in which 118575 is not
    accepted or 117092 is not refused: those two are the spec, not the snapshot.

TRACK - the stored 118575 detection cache (fixtures/track/, the 2026-09-17 .p35 run, CoreML, 3000
    detected frames) replayed through track.run at the shipped call shape: stride, conf, prior gates,
    stitch and tracker resolved from app/server/jobs.py's call by shipped_config.product_config(), never
    typed here. The detector is not constructed (a complete cache replays), but the clip's PIXELS are
    read - team_by_shirt crops shirts - so the full clip is REFERENCED in place, at st2_mot.CLIPS, and
    identified by its size and full-file sha256 (0.7 s). Absent or different -> exit 2, never 0.
    PASS is the sha256 of json.dumps(detections, sort_keys=True) equal to the golden, EXACTLY: n=2
    replays were bit-identical to each other and to the stored .p35.tracks.json, so any slack would be
    invented. A mismatch prints rows, ids and per-team rows/ids against the golden so a rebless can
    say what moved (HOTA is not computed: st2_mot needs its own staged run).
    BALL IS NOT COVERED. This fixture's cache holds no ball candidates, and track.run refuses to
    replay such a cache when the ball is requested, so the replay runs with the ball disabled and
    every ball row, picker and ball stamp is outside this check (the shipped job runs the ball).
    Rebuilding the fixture cache with ball candidates costs a full 3000-frame detection pass; the
    ball gets its own section instead.

DETECT - the detector the product would pick (detectors.product_detector(), jobs.py's rule) on a committed 20-frame excerpt of the same clip (fixtures/detect/, an h264
    stream copy - no re-encode, its 20 decoded frames equal the source's bit for bit), through track.run
    at the shipped stride/conf/ball/prior, so 10 frames are detected at the shipped overlap and tile.
    The frames compared are the detection cache's, ball candidates included (since 2026-09-28).
    Goldens are stored per detector; the golden records which one the product picked at bless time, and
    a different pick now is a FAIL (exit 1), not a silent switch of reference.
    TOLERANCES - registered 2026-09-28 BEFORE any comparison against these goldens was judged:
        coreml   exact: every frame's boxes, xyxy, conf and cls identical (prior review: 204/204, 0 px, 0 conf)
        rfdetr   |boxes - golden| <= 2 in total; one-to-one IoU >= 0.5 matching per frame and class, at
                 most 2 boxes unmatched on either side in total; matched |dxy| <= 2 px, |dconf| <= 0.02
    The model identity (path + sha256 of the .mlpackage's Data/ tree, or of the RF-DETR .pth) is in
    the golden; a different model is a FAIL too.

IDENTITY - jobs.apply_identity (FUTTRAFFIC_IDENTITY unset = "assisted", min_conf, ocr "auto" = docTR) on
    EXACTLY what `check track` reproduces: the track fixture's replay (its sha must equal the one this
    golden was blessed on, else exit 2) and its own calibration (TCAL). PASS is the exact sha256 of
    (raw_id -> id, per group the written jersey/jersey_conf and the sorted (number, weighted conf) votes,
    the strategy's report). Exact because docTR recognition was bit-identical within and across
    processes on 40 crops (reviewer's s3r_ocrdet.py) and every other step is deterministic.
    Two inputs, both judged exactly in both modes:
      replay   118575, the track --fast replay: 44 tracks and ~0 merges, so it guards selection, crops,
               legibility, OCR, the vote and min_conf - NOT the merge.
      merge    fixtures/identity/118575.nostitch.f0-2999.tracks.json.gz, the same replay with stitch off,
               first 3000 frames, through apply_identity with no video: merges only, product defaults.
    --real   the clip's pixels (referenced in place, as track does) and docTR for real. Also re-records
             the crop/OCR lookup and fails if it differs from the committed one (--fast would be serving
             stale reads). Prints a NON-GATING diagnostic vs ST2 GT jerseys when the local
             out/st2_team_gt.json join exists - never part of pass/fail.
    --fast   the same replay, with no pixels and no docTR: identity.cv2's VideoCapture serves black frames of
             the recorded shape, and the committed lookup (fixtures/identity/118575.crops.json.gz, keyed
             frame:x1,y1,x2,y2 -> crop sha256 + Laplacian variance; docTR read per crop sha256) answers
             legible()'s sharpness and the OCR. identity.py's own selection, torso_box, legible thresholds,
             facing weight, vote, merge and duplicate rule all run. A requested crop missing from the
             lookup, or a recorded one never requested, FAILS "selection or crop moved".
             What only --real sees: a change in the decode, docTR itself, or the pixels of a crop whose
             box did not move. Plus identity.py / roster.py / jobs.py self-checks, in parallel.
    KNOWN BUGS are carried by jobs.py's SELFCHECK_KNOWN_BUGS and printed on its "jobs.py ok" line.

MEASURE - stats.build_match, the stage that writes match.json, on what the identity stage hands it.
    --fast   fixtures/measure/118575.id.tracks.json.gz (jobs.apply_identity's output on identity --fast's input,
             officials included; regenerate with --make-fixture, which needs the clip and refuses unless identity
             reproduces its golden) + fixtures/measure/corrections.json (one merge, one split) through
             jobs._derive_tracks, then build_match on TCAL. PASS is the exact sha256 of json.dumps(match,
             sort_keys=True). Exact because n=6 processes under PYTHONHASHSEED 0-5 gave one sha, after
             formation_summary's tie-break stopped depending on set order (it flipped 4-3-3 / 3-5-2 here).
             The one live input is pinned: metric() resolves tiers against api-golden.json, so the golden freezes
             the accepted ids METRICS names and --fast patches stats.accepted_measurements to that set; a live
             set that differs prints a NOTE, never a FAIL. A mismatch prints players/officials counts and the
             per-field [tier, flags, n], per-team and per-player columns that moved. Plus stats.py, frames.py,
             roles.py and test_golden.py (which SKIPs with exit 0 when its source is absent - that is exit 2 here).
    --real   dist_eval.evaluate_stc_all(REAL_VARIANT) on the nine STC2025 clips against ST2 GT; the whole result
             dict must equal the golden's (n=5 runs identical on 2026-09-28, plus the reviewer's). Our side's inputs
             (out/stc2025, gitignored) are identified by sha; absent or different, or GT unreadable -> exit 2.

BALL - the shipped ball path (track.track's class split -> ball.BallPicker, marking gate at its default, off)
    on the SHIPPED merge's candidates: fixtures/ball/ is n70_class_merge.py --export's output, M0 of the N70 raw
    caches (coreml 576, tile 1024, overlap 0.25, player conf 0.5, ball conf 0.05; the raw caches are gitignored,
    their sha256s are recorded inside m0_cands.json.gz).
    --fast   ball.py's self-check, then two excerpts (balls + players after M0, track()'s cache format, frames
             0..598 at the shipped stride 2, blank frames of the recorded shape, no detector) through the real
             track.track(): soccer_top test 1710-1740 on its own perspective calibration, and HELD soccer_side
             train 0000-0030 with none (it has none). PASS = exact sha256 of the ball rows AND exact correct/wrong/
             silent/gt_present vs the ball truth stored in the excerpt. Also the detect golden's 10 frames (ball
             candidates included) on TCAL, the spherical camera: the path a spherical calibration takes through
             BallPicker._expected_px (4 rows of 10 frames before the 2026-09-29 size-gate fix, 10 after). The prior is not applied (a cache hit skips it, as in the product).
             Ball boxes under detectors.BALL_CONF are dropped first - a no-op at the shipped value - so --prove
             can plant a raised threshold.
    --real   ball_eval.score_m0 on m0_cands.json.gz: per clip correct/wrong/silent/gt_present, shipped stride 1,
             oracle stride 1, shipped stride 2 - the numbers behind product.ball.* - exact vs golden. Needs the
             SoccerTrack truth CSVs (gitignored); absent -> exit 2.
    --prove  SIZE_RATIO (0.4,2.0), MARGIN_M 30, Scenery min_frames 3, cal=None handed to the picker by track.py,
             BALL_CONF 0.10, and the size gate read back through inv(H) (the pre-2026-09-29 _expected_px) must
             FAIL; MAX_BLOB_EXTENT 3.0 must PASS (inert: the marking gate is off), and unpatched.

PROVE plants, in-process: identity (--fast lookups) - MERGE_MARGIN 0.06, MAX_GAP_S 2.5, FRONT_WEIGHT 1.0,
    facing sort reversed, torso_box ignoring box_h, MIN_SHARPNESS 3000, min_conf 0.5, _drop_duplicates
    keyed on the number alone must FAIL; MIN_SHARPNESS x2 must PASS (inert: no crop here is under 2249). track - tracking.BEST="bytetrack", stitch MAX_SPEED 7, split handed fps/2,
    split SPLIT_SPEED x2; detect - OVERLAP 0.30. Each must FAIL and the unpatched run must PASS.
    OVERLAP 0.20 is run too and must PASS: on a 4096x1080 frame at tile 1024 it tiles identically to 0.25.
    measure (--fast's compare) - DISTANCE_WINDOW_S x2, MAX_SPEED 12, MIN_MINUTES x2, officials kept in players, a
    caveat flag dropped from distance_km, heatmap max_points 60 must FAIL; distance_km's tier ok -> rel must PASS
    (inert: its caveats already cap it at prov), and so must SPRINT_SPEED 6.0 and SPRINT_MIN_S x2 (inert: sprints
    are withheld from match.json; --real's pooled_sprint_ratio still moves with them).
"""
import contextlib
import datetime
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "model" / "engine"), str(ROOT / "app" / "server")]
FIX = ROOT / "model" / "validation" / "fixtures" / "calib"
GOLDEN = FIX / "golden.json"
PY = sys.executable
EXPECT = {"118575": True, "117092": False, "128057": False}   # accepted? - the spec a rebless may not change
# 128057: refused by accept() since N68 F1 (was accepted at 109.14 m, ST2 says 105; n68-128057-length (not published))
KNOWN_WRONG = {}   # {clip: {"gt_length_m", "doc", "source"}} - a fixture frozen at behaviour known to be wrong
MADE_BY = ('.venv/bin/python -c "import sys, cv2; sys.path[:0] = [\'model/engine\']; import autocal; '
           'cv2.imwrite(\'{q}.median40.png\', autocal.median_frame(\'$ST2/mot/clips/{q}.mp4\'))"'
           '   # autocal.median_frame n=40 (n31_linereg._median_cached); ST2 = SoccerTrack v2 root')
MED_M, MAX_M, HITS_REL = 0.02, 0.1, 0.02            # see TOLERANCES
SELFCHECKS = ("model/engine/pitch.py", "model/engine/track.py", "model/engine/sceneprior.py", "app/server/jobs.py")
TFIX = ROOT / "model" / "validation" / "fixtures" / "track"
TGOLDEN, TCACHE, TCAL = TFIX / "golden.json", TFIX / "118575.p35.dets.cache.json.gz", TFIX / "118575.p35.calibration.json"
DFIX = ROOT / "model" / "validation" / "fixtures" / "detect"
DGOLDEN, DCLIP = DFIX / "golden.json", DFIX / "118575.f0-19.mp4"
DMADE_BY = "ffmpeg -i $ST2/mot/clips/118575.mp4 -map 0:v -c copy -an -frames:v 20 118575.f0-19.mp4   # stream copy"
DX_PX, DCONF, DBOXES = 2.0, 0.02, 2                  # rfdetr tolerances, see DETECT


class CouldNotRun(Exception):
    pass


def _grid():
    import numpy as np
    return np.stack(np.meshgrid(np.linspace(3, 102, 12), np.linspace(3, 65, 8)), -1).reshape(-1, 2)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _golden(p=GOLDEN):
    if not p.exists():
        raise CouldNotRun(f"{p.relative_to(ROOT)} missing: run --rebless")
    return json.loads(p.read_text())


def _stamp(reason):
    """The fields every golden carries: why, when, on what code."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "model/engine", "app/server/jobs.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.split("\n")
    return {"reason": reason, "date": datetime.date.today().isoformat(), "git_head": head,
            "uncommitted_engine_files": sorted(x[3:] for x in dirty if x.strip())}


def _write(path, g):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(g, indent=1))
    os.replace(tmp, path)                                # all or nothing
    print(f"re-blessed {path.relative_to(ROOT)}: {g['reason']}")


def _start(paths):
    return {s: subprocess.Popen([PY, s], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for s in paths}


def selfchecks(paths=SELFCHECKS, procs=None):
    """-> [failure lines]. Parallel subprocesses; a killed one is could-not-run, not a failure."""
    procs = procs or _start(paths)
    bad = []
    for s, p in procs.items():
        out = p.communicate()[0]
        if p.returncode < 0:
            raise CouldNotRun(f"{s} self-check killed by signal {-p.returncode}")
        print(f"  {'ok  ' if p.returncode == 0 else 'FAIL'} {s}: {out.strip().splitlines()[-1][:110] if out.strip() else ''}")
        if p.returncode:
            bad.append(f"{s} self-check failed:\n{out[-1500:]}")
    return bad


def roundtrip(g):
    """Golden calibration.json -> from_dict -> to_dict keys and the map on the grid. -> [failure lines]."""
    import calibrators
    import numpy as np
    d = g["fixtures"]["118575"]["calibration"]
    cal = calibrators.from_dict(d)
    bad = []
    if set(cal.to_dict()) != set(d):
        bad.append(f"to_dict keys {sorted(cal.to_dict())} != golden {sorted(d)}")
    px = cal.to_image(_grid())
    e1 = float(np.nanmax(np.abs(px - np.array(g["grid_px"])))) if np.isfinite(px).all() else float("inf")
    e2 = float(np.max(np.abs(cal.to_pitch(np.array(g["grid_px"])) - _grid())))
    if not (e1 <= 1e-6 and e2 <= 1e-6):
        bad.append(f"golden map moved: to_image off by {e1:.3g} px, to_pitch off by {e2:.3g} m (tolerance 1e-6)")
    return bad


def run_fixture(q):
    """jobs.auto_calibrate on one committed median frame, in a throwaway data dir. -> result dict."""
    import cv2
    png = FIX / f"{q}.median40.png"
    if not png.exists():
        raise CouldNotRun(f"fixture {png.relative_to(ROOT)} missing")
    bgr = cv2.imread(str(png))
    if "jobs" not in sys.modules:                  # jobs reads both at import: a throwaway store, autocal on
        os.environ["FUTTRAFFIC_DATA"] = tempfile.mkdtemp(prefix="checkcal-")
        os.environ.pop("FUTTRAFFIC_AUTOCAL", None)
    import autocal
    import jobs
    assert jobs.AUTOCAL and "checkcal-" in str(jobs.DATA_DIR), jobs.DATA_DIR    # never the real store
    st = jobs.create(f"check calib {q}", None, ".mp4")
    jobs.save(st)
    t = time.time()
    with mock.patch.object(jobs, "_video_shape", lambda p: (bgr.shape[1], bgr.shape[0])), \
            mock.patch.object(autocal, "median_frame", lambda *a, **k: bgr):
        ok = jobs.auto_calibrate(jobs.load(st["id"]))
    st = jobs.load(st["id"])
    cal = jobs.job_dir(st["id"]) / "calibration.json"
    auto = st.get("calibration_auto") or {}
    return {"accepted": bool(ok), "why": auto.get("why"), "hits": auto.get("hits"),
            "line_rms_px": auto.get("line_rms_px"), "length_m": auto.get("length_m"), "secs": round(time.time() - t, 1),
            "calibration": json.loads(cal.read_text()) if ok and cal.exists() else None}


def displacement(a, b):
    """Metres the map b moves a grid point that map a puts at pixel p: |b.to_pitch(a.to_image(g)) - g|."""
    import calibrators
    import numpy as np
    A, B = calibrators.from_dict(a), calibrators.from_dict(b)
    d = np.linalg.norm(B.to_pitch(A.to_image(_grid())) - _grid(), axis=1)
    return float(np.median(d)), float(d.max())


def real(g):
    bad = []
    for q, want in g["fixtures"].items():
        png = FIX / f"{q}.median40.png"
        if not png.exists() or _sha(png) != want["sha256"]:
            raise CouldNotRun(f"fixture {q}: missing or sha256 differs from golden - regenerate with: {want['made_by']}")
        r = run_fixture(q)
        line = f"  {q}: accepted={r['accepted']} hits={r['hits']} why={r['why']!r} ({r['secs']} s)"
        if r["accepted"] != want["accepted"] or r["accepted"] != EXPECT[q]:
            bad.append(f"{q}: accepted={r['accepted']}, golden {want['accepted']}, spec {EXPECT[q]} - why: {r['why']!r}")
        elif not r["accepted"]:
            if r["why"] != want["why"]:
                bad.append(f"{q}: refused for {r['why']!r}, golden {want['why']!r}")
        else:
            med, mx = displacement(want["calibration"], r["calibration"])
            line += f" map vs golden median {med:.4f} m max {mx:.4f} m"
            if med > MED_M or mx > MAX_M:
                bad.append(f"{q}: map moved median {med:.4f} m / max {mx:.4f} m (tolerance {MED_M} / {MAX_M})")
            if abs(r["hits"] - want["hits"]) > HITS_REL * want["hits"]:
                bad.append(f"{q}: hits {r['hits']} vs golden {want['hits']} (tolerance +-{HITS_REL:.0%})")
            if set(r["calibration"]) != set(want["calibration"]):
                bad.append(f"{q}: calibration.json keys {sorted(r['calibration'])} != golden {sorted(want['calibration'])}")
        print(line)
        kw = want.get("known_wrong")
        if kw:
            print(f"  KNOWN-WRONG {q}: accepted={r['accepted']} at length {r['length_m']} m, ground truth {kw['gt_length_m']} m"
                  f" - a pass here means UNCHANGED, not correct; see {kw['doc']}")
    return bad


def constants():
    import autocal
    import shipped_config
    c = {k: v for k, v in vars(autocal).items() if k.isupper() and isinstance(v, (int, float, tuple, dict))}
    c = json.loads(json.dumps(c, default=list))
    c["jobs.PANORAMA_ASPECT"] = shipped_config._ast_constant(shipped_config.JOBS, "PANORAMA_ASPECT")[-1][1]
    return c


def rebless(reason):
    if not reason.strip():
        print("refused: --rebless needs a non-empty reason")
        return 2
    fixtures = {}
    for q, want in EXPECT.items():
        r = run_fixture(q)
        if r["accepted"] != want:
            print(f"refused: {q} accepted={r['accepted']} ({r['why']!r}); the spec says {want}. Not freezing a regression.")
            return 1
        fixtures[q] = {"sha256": _sha(FIX / f"{q}.median40.png"), "made_by": MADE_BY.format(q=q),
                       **{k: v for k, v in r.items() if k != "secs"}}
        if q in KNOWN_WRONG:
            fixtures[q]["known_wrong"] = {"length_m": r["length_m"], **KNOWN_WRONG[q]}
        print(f"  {q}: accepted={r['accepted']} hits={r['hits']} why={r['why']!r}")
    import calibrators
    _write(GOLDEN, {**_stamp(reason), "constants": constants(),
                    "tolerances": {"median_m": MED_M, "max_m": MAX_M, "hits_rel": HITS_REL}, "fixtures": fixtures,
                    "grid_px": calibrators.from_dict(fixtures["118575"]["calibration"]).to_image(_grid()).tolist()})
    return 0


def _quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def _pooled_missing(rep):
    """accept()'s rule before N68 F1: "pen" present if ANY penalty group has 25% of its samples on paint."""
    fam = {k.split("_")[0] for k, (h, n) in rep["hits_by_group"].items() if h >= 0.25 * n}
    return sorted({"touch", "goal", "halfway", "pen", "circle"} - fam)


def prove(g):
    """A check is not trusted until it has been seen to fail: each planted defect, in-process."""
    import autocal
    import calibrators
    real_to_dict = calibrators.SphericalCamera.to_dict
    cases = [("unpatched", None, "fast", False), ("unpatched", None, "real", False),
             ("VERT_CUBIC=-0.4", mock.patch.object(autocal, "VERT_CUBIC", -0.4), "fast", True),
             ("MIN_HITS=900", mock.patch.object(autocal, "MIN_HITS", 900), "real", True),
             ("pooled pen family (pre-N68)", mock.patch.object(autocal, "_missing", _pooled_missing), "real", True),
             ("to_dict drops rms_error_m",
              mock.patch.object(calibrators.SphericalCamera, "to_dict",
                                lambda self: {k: v for k, v in real_to_dict(self).items() if k != "rms_error_m"}),
              "fast", True)]
    wrong = 0
    for name, patch, mode, must_fail in cases:
        with patch or contextlib.nullcontext():
            bad = _quiet(roundtrip if mode == "fast" else real, g)
        ok = bool(bad) == must_fail
        wrong += not ok
        print(f"  {'ok  ' if ok else 'BAD '} {name:28s} {mode:4s} -> {'FAILS' if bad else 'passes'}"
              f"{': ' + bad[0][:100] if bad else ''}")
    return 1 if wrong else 0


def _run_section(name, argv, fast, prove_fn, rebless_fn, golden, real_fn=None):
    """The one exit-code contract: 0 pass, 1 regression, 2 could not run."""
    mode = argv[0] if argv else ""
    try:
        if mode == "--rebless":
            return rebless_fn(argv[1] if len(argv) > 1 else "")
        g = _golden(golden)
        if mode == "--fast":
            bad = fast(g)
        elif mode == "--real" and real_fn:
            bad = real_fn(g)
        elif mode == "--prove":
            return prove_fn(g)
        else:
            print(__doc__)
            return 2
    except CouldNotRun as e:
        print(f"COULD NOT RUN: {e}")
        return 2
    for b in bad:
        print(f"REGRESSION  {b}")
    print(f"check {name} {mode}: {'FAIL' if bad else 'ok'}")
    return 1 if bad else 0


def calib(argv):
    try:
        return _run_section("calib", argv, lambda g: selfchecks() + roundtrip(g), prove, rebless, GOLDEN, real)
    finally:                                         # run_fixture's throwaway store, never the real one
        d = os.environ.get("FUTTRAFFIC_DATA", "")
        if Path(d).name.startswith("checkcal-"):
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- track
def _product():
    """What a job runs, from jobs.py's call site outward (shipped_config), incl. what 'best' resolves to."""
    import shipped_config
    return shipped_config.product_config()


def _clip():
    """The full 118575 clip under $ST2_DIR (st2_mot.CLIPS); never committed."""
    return Path(os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))) / "mot" / "clips" / "118575.mp4"


def _fixture_shas(paths, g):
    for p in paths:
        if not p.exists() or (g and _sha(p) != g["fixtures"][p.name]):
            raise CouldNotRun(f"fixture {p.relative_to(ROOT)} missing or sha256 differs from the golden")


def _run_track(video, detector, p, ball, cache_src=None, **kw):
    """track.run at the shipped call shape, in a throwaway dir. -> (tracks.json dict, cache dict)."""
    import track
    tmp = Path(tempfile.mkdtemp(prefix="checktrack-"))
    try:
        cache = tmp / "dets.cache.json"
        if cache_src:
            with gzip.open(cache_src) as a, open(cache, "wb") as b:
                shutil.copyfileobj(a, b)
        with contextlib.redirect_stdout(io.StringIO()):
            track.run(str(video), str(tmp / "tracks.json"), detector=detector, tracker=p["tracker"], conf=p["conf"],
                      calib=str(TCAL), stride=p["stride"], stitch=p["stitch"], prior_gates=tuple(p["prior_gates"]),
                      ball=ball, dets_cache=str(cache), **kw)
        return json.loads((tmp / "tracks.json").read_text()), json.loads(cache.read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _track_tracks(g=None):
    """The track fixture replayed: -> (tracks.json dict, clip identity, config, secs). Checks g first."""
    _fixture_shas((TCACHE, TCAL), g)
    clip = _clip()
    if not clip.exists():
        raise CouldNotRun(f"{clip} is absent - the full clip is referenced in place, not committed")
    ident = {"path": str(clip), "size": clip.stat().st_size, "sha256": _sha(clip)}
    if g and (ident["size"], ident["sha256"]) != (g["clip"]["size"], g["clip"]["sha256"]):
        raise CouldNotRun(f"{clip} is not the clip the golden was blessed on (size/sha256 differ)")
    with gzip.open(TCACHE) as fh:
        det = json.load(fh)["settings"]["detector"]          # replayed, never constructed: the cache is complete
    p = _product()
    t = time.time()
    res, _ = _run_track(clip, det, p, ball=False, cache_src=TCACHE)
    return res, ident, p, round(time.time() - t, 1)


def _rows_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def track_replay(g=None):
    """-> (summary, clip identity, config). Verifies fixtures and clip against g first when given."""
    res, ident, p, secs = _track_tracks(g)
    rows, teams = res["detections"], {}
    for r in rows:
        teams.setdefault(str(r.get("team")), []).append(r["id"])
    return ({"sha256": _rows_sha(rows), "rows": len(rows), "ids": len({r["id"] for r in rows}), "secs": secs,
             "teams": {k: {"rows": len(v), "ids": len(set(v))} for k, v in sorted(teams.items())},
             "stamp": res["_perf"].get("config")}, ident, p)


def track_fast(g):
    r, _, p = track_replay(g)
    w = g["result"]
    print(f"  replay: {r['rows']} rows, {r['ids']} ids, sha {r['sha256'][:16]}, tracker {p['tracker']}->"
          f"{p['tracker_resolved']}, stride {p['stride']}, stitch {p['stitch']} ({r['secs']} s)")
    if r["sha256"] == w["sha256"]:
        return []
    return [f"detections sha {r['sha256'][:16]} != golden {w['sha256'][:16]}: rows {r['rows']} vs {w['rows']}, "
            f"ids {r['ids']} vs {w['ids']}, teams {r['teams']} vs golden {w['teams']}; tracker resolved "
            f"{p['tracker_resolved']} (golden {g['config']['tracker_resolved']})"]


def track_rebless(reason):
    if not reason.strip():
        print("refused: --rebless needs a non-empty reason")
        return 2
    try:
        r, ident, p = track_replay()
    except CouldNotRun as e:
        print(f"COULD NOT RUN: {e}")
        return 2
    print(f"  replay: {r['rows']} rows, {r['ids']} ids, sha {r['sha256'][:16]} ({r['secs']} s)")
    r.pop("secs")
    _write(TGOLDEN, {**_stamp(reason), "config": {**{k: p[k] for k in ("stride", "conf", "prior_gates", "stitch",
                                                                       "tracker", "tracker_resolved")},
                                                  "ball": False, "ball_note": "NOT COVERED: the cache is ball=False"},
                     "clip": {**ident, "identity": "size + full-file sha256; path = st2_mot.CLIPS/118575.mp4"},
                     "fixtures": {TCACHE.name: _sha(TCACHE), TCAL.name: _sha(TCAL)},
                     "made_by": "gzip -9 of model/validation/out/stc2025/118575.p35.dets.cache.json (2026-09-17)",
                     "result": r})
    return 0


def _redefault(fn, name, f):
    """Patch one bound default of fn (the value a module constant was copied into at def time)."""
    import inspect
    names = [q.name for q in inspect.signature(fn).parameters.values()
             if q.default is not q.empty and q.kind != q.KEYWORD_ONLY]
    d = list(fn.__defaults__)
    d[names.index(name)] = f(d[names.index(name)])
    return mock.patch.object(fn, "__defaults__", tuple(d))


def _prove_cases(cases, check, g):
    wrong = 0
    for name, patches, must_fail in cases:
        with contextlib.ExitStack() as st:
            for pt in patches:
                st.enter_context(pt)
            try:
                bad = _quiet(check, g)
            except CouldNotRun as e:
                bad, must_fail = [f"COULD NOT RUN: {e}"], None
        ok = bool(bad) == must_fail
        wrong += not ok
        print(f"  {'ok  ' if ok else 'BAD '} {name:34s} -> {'FAILS' if bad else 'passes'}"
              f"{': ' + bad[0][:130] if bad else ''}", flush=True)
    return 1 if wrong else 0


def track_prove(g):
    import stitch
    import tracking
    split = stitch.split
    return _prove_cases([
        ("unpatched", [], False),
        ('tracking.BEST="bytetrack"', [mock.patch.object(tracking, "BEST", "bytetrack")], True),
        ("stitch MAX_SPEED 9 -> 7", [mock.patch.object(stitch, "MAX_SPEED", 7.0),
                                     _redefault(stitch.stitch, "max_speed", lambda v: 7.0)], True),
        ("split handed fps/2", [mock.patch.object(stitch, "split", lambda rows, cal, speed=stitch.SPLIT_SPEED,
                                                  fps=25.0: split(rows, cal, speed, fps=fps / 2))], True),
        ("split SPLIT_SPEED x2", [mock.patch.object(stitch, "SPLIT_SPEED", 2 * stitch.SPLIT_SPEED),
                                  _redefault(stitch.split, "speed", lambda v: 2 * v)], True),
    ], track_fast, g)


def track_section(argv):
    return _run_section("track", argv, track_fast, track_prove, track_rebless, TGOLDEN)


# ---------------------------------------------------------------- detect
def detect_pick():
    """jobs.py's rule, from the one place it lives (detectors.product_detector)."""
    import detectors
    return detectors.product_detector()


def model_identity(det):
    """path + sha256 of the weights that run (detectors.model_identity, which track.run stamps too)."""
    import detectors
    return detectors.model_identity(det)


def detect_run(det, p=None):
    """-> {frame: boxes} from track.run on the excerpt, at the shipped stride/conf/ball/prior."""
    p = p or _product()
    _, cache = _run_track(DCLIP, det, p, ball=p["ball"])
    return cache["frames"], cache["settings"]


def _iou(a, b):
    import numpy as np
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lo, hi = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(hi - lo, 0, None), -1)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def det_compare(got, want, det):
    """-> [failure lines], by the tolerances registered in the docstring."""
    ng, nw = sum(map(len, got.values())), sum(map(len, want.values()))
    if det == "coreml":
        diff = sorted(set(got) ^ set(want)) + [f for f in want if f in got and got[f] != want[f]]
        return [f"coreml boxes not identical: {len(diff)} frame(s) differ ({diff[:5]}), boxes {ng} vs golden {nw}"] \
            if diff else []
    unmatched, dxy, dconf = 0, 0.0, 0.0
    for f in set(got) | set(want):
        for cls in {b["cls"] for b in got.get(f, []) + want.get(f, [])}:
            a = [b for b in got.get(f, []) if b["cls"] == cls]
            b = [b for b in want.get(f, []) if b["cls"] == cls]
            M = _iou([x["xyxy"] for x in a], [x["xyxy"] for x in b]) if a and b else None
            pairs = []
            while M is not None and M.size and M.max() >= 0.5:      # greedy one-to-one, best IoU first
                i, j = divmod(int(M.argmax()), M.shape[1])
                pairs.append((a[i], b[j]))
                M[i, :], M[:, j] = -1, -1
            unmatched += len(a) + len(b) - 2 * len(pairs)
            for x, y in pairs:
                dxy = max(dxy, max(abs(u - v) for u, v in zip(x["xyxy"], y["xyxy"])))
                dconf = max(dconf, abs(x["conf"] - y["conf"]))
    print(f"  rfdetr vs golden: boxes {ng} vs {nw}, unmatched {unmatched}, max |dxy| {dxy:.3f} px, "
          f"max |dconf| {dconf:.4f}")
    bad = []
    if abs(ng - nw) > DBOXES or unmatched > DBOXES:
        bad.append(f"rfdetr boxes {ng} vs golden {nw}, {unmatched} unmatched at IoU 0.5 (tolerance {DBOXES})")
    if dxy > DX_PX or dconf > DCONF:
        bad.append(f"rfdetr matched boxes moved |dxy| {dxy:.3f} px, |dconf| {dconf:.4f} (tolerance {DX_PX} / {DCONF})")
    return bad


def detect_fast(g):
    _fixture_shas((DCLIP, TCAL), g)
    det = detect_pick()
    if det != g["picked"] or det not in g["detectors"]:
        return [f"the product would pick {det!r} here, but the golden was blessed on {g['picked']!r}"
                f" (goldens held: {sorted(g['detectors'])})"]
    w = g["detectors"][det]
    t = time.time()
    got, _ = detect_run(det)
    ident = model_identity(det)
    print(f"  {det}: {sum(map(len, got.values()))} boxes on {len(got)} frames ({time.time() - t:.1f} s), "
          f"model sha {str(ident['sha256'])[:16]}")
    bad = [] if ident["sha256"] == w["model"]["sha256"] else [f"{det} model {ident} != golden {w['model']}"]   # path may move (worktree)
    return bad + det_compare(got, w["frames"], det)


def detect_rebless(reason):
    if not reason.strip():
        print("refused: --rebless needs a non-empty reason")
        return 2
    if not DCLIP.exists():
        print(f"COULD NOT RUN: {DCLIP.relative_to(ROOT)} missing - regenerate with: {DMADE_BY}")
        return 2
    pick, p, dets = detect_pick(), _product(), {}
    for det in ("coreml", "rfdetr"):
        try:
            t = time.time()
            frames, settings = detect_run(det, p)
        except (SystemExit, ImportError, OSError) as e:          # a detector this machine cannot run
            print(f"  {det}: not available here ({str(e)[:120]})")
            continue
        dets[det] = {"model": model_identity(det), "settings": settings, "boxes": sum(map(len, frames.values())),
                     "frames": frames}
        print(f"  {det}: {dets[det]['boxes']} boxes on {len(frames)} frames ({time.time() - t:.1f} s)")
    if pick not in dets:
        print(f"refused: the product picks {pick!r} here and it did not run")
        return 2
    _write(DGOLDEN, {**_stamp(reason), "picked": pick,
                     "config": {k: p[k] for k in ("stride", "conf", "ball", "prior_gates", "overlap", "tile")},
                     "fixtures": {DCLIP.name: _sha(DCLIP), TCAL.name: _sha(TCAL)}, "made_by": DMADE_BY,
                     "tolerances": {"coreml": "exact", "rfdetr": {"boxes": DBOXES, "iou": 0.5, "dxy_px": DX_PX,
                                                                  "dconf": DCONF}},
                     "detectors": dets})
    return 0


def detect_prove(g):
    import detectors

    def overlap(v):
        return [mock.patch.object(detectors, "OVERLAP", v)] + [_redefault(fn, "overlap", lambda _: v)
                                                               for fn in (detectors._coreml, detectors._rfdetr)]
    # 0.20 is INERT here, and must pass: detectors._spans(4096, 1024, o) is the same five tiles
    # [0, 768, ..., 3072] for every o from 0.10 to 0.25 - the grid is pinned to both frame edges and
    # only the tile COUNT moves. 0.30 is six tiles, so it is the planted defect this check can see.
    return _prove_cases([
        ("unpatched", [], False),
        ("OVERLAP 0.20 (inert: same 5 tiles)", overlap(0.20), False),
        ("OVERLAP 0.25 -> 0.30", overlap(0.30), True),
    ], detect_fast, g)


def detect_section(argv):
    return _run_section("detect", argv, detect_fast, detect_prove, detect_rebless, DGOLDEN)


# ---------------------------------------------------------------- identity
IFIX = ROOT / "model" / "validation" / "fixtures" / "identity"
IGOLDEN, ICROPS = IFIX / "golden.json", IFIX / "118575.crops.json.gz"
IMERGE = IFIX / "118575.nostitch.f0-2999.tracks.json.gz"
IMERGE_MADE_BY = ("check._run_track on the track fixture (cache + TCAL + the 118575 clip) at the shipped call shape "
                  "with stitch=False, rows with f < 3000 reduced to f,id,team,x,y; gzip -9, mtime 0")
ISELFCHECKS = ("model/engine/identity.py", "model/engine/roster.py", "app/server/jobs.py")
_TRACKS = {}


class _Moved(Exception):
    pass


class _Pixels:
    """identity.scan_video's pixels and OCR: recorded from the clip and docTR (--real, table None), or
    served from the committed lookup table (--fast). identity's own code runs unchanged either way: only
    its cv2.VideoCapture, cv2.Laplacian, legible (to learn the crop's key) and OCR backend are wrapped."""

    def __init__(self, table=None):
        import cv2
        import identity
        self.cv2, self.idm, self.table = cv2, identity, table
        self.crops, self.ocr, self.seen, self.votes = {}, {}, set(), {}
        self.f, self.cur, self.shape, self.frames, self.ocr_s = -1, None, None, 0, 0.0
        self.real_ocr = None

    def patches(self):
        idm, px, inner, real_legible = self.idm, self, self.idm.scan_video, self.idm.legible
        if self.table is None:
            self.real_ocr = idm.get_ocr_backend("auto")
            if self.real_ocr is None:
                raise CouldNotRun("no OCR backend (docTR) importable - --real needs the one jobs.py runs")

        class Cv:
            def __getattr__(self, k):
                return getattr(px.cv2, k)

            def VideoCapture(self, path):  # noqa: N802
                return px.capture(path)

            def Laplacian(self, img, depth):  # noqa: N802
                return px.laplacian(img, depth)

        def legible(crop, box, near):
            key = f"{px.f}:{','.join(map(str, box))}"
            px.seen.add(key)
            if px.table is None:
                px.cur = px.crops[key] = {"sha": hashlib.sha256(crop.tobytes()).hexdigest()}
            else:
                px.cur = px.table["crops"].get(key)
                if px.cur is None:
                    raise _Moved(f"selection or crop moved: frame {px.f} torso box {box} is not in the fixture")
            return real_legible(crop, box, near)

        def scan(*a, **k):
            px.votes, rep = inner(*a, **k)
            return px.votes, rep

        return [mock.patch.object(idm, "cv2", Cv()), mock.patch.object(idm, "legible", legible),
                mock.patch.object(idm, "get_ocr_backend", lambda name="auto": px.read),
                mock.patch.object(idm, "scan_video", scan)]

    def capture(self, path):
        import numpy as np
        self.f, px = -1, self
        if self.table is None:
            cap = self.cv2.VideoCapture(path)
        else:
            z = np.zeros(self.table["frame_shape"], np.uint8)

        class Cap:
            def isOpened(self):  # noqa: N802
                return px.table is not None or cap.isOpened()

            def release(self):
                if px.table is None:
                    cap.release()

            def read(self):
                if px.table is not None:
                    px.f += 1
                    return px.f < px.table["frames"], z
                ok, fr = cap.read()
                if ok:
                    px.f += 1
                    px.frames, px.shape = max(px.frames, px.f + 1), list(fr.shape)
                return ok, fr
        return Cap()

    def laplacian(self, img, depth):
        if self.table is None:
            a = self.cv2.Laplacian(img, depth)
            self.cur["lap"] = float(a.var())
            return a
        if "lap" not in self.cur:
            raise _Moved(f"selection or crop moved: frame {self.f} reached the sharpness test, not recorded")
        v = self.cur["lap"]
        return type("Var", (), {"var": lambda _: v})()

    def read(self, crop):
        if self.table is None:
            t = time.time()
            r = self.real_ocr(crop)
            self.ocr_s += time.time() - t
            self.ocr[self.cur["sha"]] = list(r)
            return r
        r = self.table["ocr"].get(self.cur["sha"])
        if r is None:
            raise _Moved(f"selection or crop moved: crop {self.cur['sha'][:16]} (frame {self.f}) has no recorded read")
        return tuple(r)

    def lookup(self):
        return {"frame_shape": self.shape, "frames": self.frames, "crops": self.crops, "ocr": self.ocr}


def _jobs():
    """jobs.apply_identity and load_calibration, imported over a throwaway store (calib's rule)."""
    if "jobs" not in sys.modules:
        os.environ["FUTTRAFFIC_DATA"] = tempfile.mkdtemp(prefix="checkcal-")
    import jobs
    return jobs


def _canon(x):
    x = json.loads(json.dumps(x))
    return x, hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def _grouped(out, rep, votes):
    """-> the frozen result: raw_id -> id, per group (written jersey/conf, sorted votes), the report."""
    of, written = {}, {}
    for x in out["detections"]:
        of[str(x.get("raw_id", x["id"]))] = x["id"]
        written.setdefault(str(x["id"]), set()).add((x.get("jersey"), x.get("jersey_conf")))
    members = {}
    for r, gid in of.items():
        members.setdefault(str(gid), []).append(int(r))
    groups = {gid: {"written": sorted(w, key=str),
                    "votes": sorted(tuple(v) for m in members[gid] for v in votes.get(m, []))}
              for gid, w in written.items()}
    return _canon({"group_of": of, "groups": groups, "report": rep})


def identity_merge():
    """The merge-sensitive input: jobs.apply_identity with no video (merge only, product defaults)."""
    import identity
    with gzip.open(IMERGE) as fh:
        tr = json.load(fh)
    jobs = _jobs()
    os.environ.pop("FUTTRAFFIC_IDENTITY", None)
    out, rep = jobs.apply_identity(identity, tr, None, jobs.load_calibration(TCAL))
    return _grouped(out, rep, {})


def identity_run(g, table):
    """check track's replay (once per process) -> jobs.apply_identity at the shipped default. -> (result, sha, px)."""
    import identity
    if "res" not in _TRACKS:
        _TRACKS["res"] = _track_tracks(_golden(TGOLDEN))
    res, ident, _, _ = _TRACKS["res"]
    res = json.loads(json.dumps(res))
    tsha = _rows_sha(res["detections"])
    if g and tsha != g["tracks_sha256"]:
        raise CouldNotRun(f"the track replay (sha {tsha[:16]}) is not the one this golden was blessed on "
                          f"({g['tracks_sha256'][:16]}): check track first, then identity --rebless")
    jobs = _jobs()
    cal = jobs.load_calibration(TCAL)
    os.environ.pop("FUTTRAFFIC_IDENTITY", None)            # the shipped default, "assisted"
    px = _Pixels(table)
    t = time.time()
    with contextlib.ExitStack() as st:
        for pt in px.patches():
            st.enter_context(pt)
        out, rep = jobs.apply_identity(identity, res, ident["path"], cal)
    px.secs, px.out = round(time.time() - t, 1), out       # out: measure --make-fixture's input
    if table is not None and px.seen != set(table["crops"]):
        raise _Moved(f"selection or crop moved: {len(set(table['crops']) - px.seen)} fixture crops never requested")
    r, sha = _grouped(out, rep, px.votes)
    return r, sha, px, tsha


def _id_compare(r, sha, w, name):
    if sha == w["sha256"]:
        return []
    moved = sorted(k for k in set(r["group_of"]) | set(w["result"]["group_of"])
                   if r["group_of"].get(k) != w["result"]["group_of"].get(k))
    gm = sorted(k for k in set(r["groups"]) | set(w["result"]["groups"])
                if r["groups"].get(k) != w["result"]["groups"].get(k))
    rp = {k: (r["report"].get(k), w["result"]["report"].get(k)) for k in set(r["report"]) | set(w["result"]["report"])
          if r["report"].get(k) != w["result"]["report"].get(k)}
    return [f"{name} sha {sha[:16]} != golden {w['sha256'][:16]}: {len(moved)} track(s) regrouped {moved[:6]}, "
            f"{len(gm)} group(s) with other votes/number {gm[:6]}, report moved {rp}"]


def _gt_diag(r):
    """NON-GATING: written jerseys vs ST2 GT on this clip, when the local join exists. Never pass/fail."""
    try:
        import collections
        import st2_team_gt as T
        gt = T.load()["118575"]
        feet, hit = T.mot_feet("118575"), collections.defaultdict(collections.Counter)
        res = _TRACKS["res"][0]
        of = r["group_of"]
        for x in res["detections"]:
            cand = [(abs(u - x["x"]) + abs(v - x["y"]), m) for m, (u, v) in feet.get(int(x["f"]), {}).items()]
            if cand and min(cand)[0] <= 30:
                hit[str(of[str(x["id"])])][min(cand)[1]] += 1
        written = right = 0
        players = set()
        for gid, grp in r["groups"].items():
            num = grp["written"][0][0]
            if num is None or not hit[gid]:
                continue
            m, n = hit[gid].most_common(1)[0]
            if n < 20 or n < 0.6 * sum(hit[gid].values()):
                continue
            written += 1
            if str(num) == str(gt["jersey"].get(str(m))):
                right += 1
                players.add(m)
        print(f"  diagnostic (NOT gated): {right} of {written} written groups with a GT player carry its number; "
              f"{len(players)} of {len(gt['jersey'])} GT players named right (feet within 30 px L1, modal id, >=20 and 60%)")
    except (Exception, SystemExit) as e:  # noqa: BLE001 - a diagnostic never decides
        print(f"  diagnostic unavailable: {type(e).__name__}: {str(e)[:100]}")


def _crops_table(g):
    _fixture_shas((ICROPS,), g)
    with gzip.open(ICROPS) as fh:
        return json.load(fh)


def identity_check(g, table=None):
    """Merge fixture + the replay; table given = --fast lookups, else the real clip and docTR."""
    _fixture_shas((IMERGE,), g)
    m, msha = identity_merge()
    bad = _id_compare(m, msha, g["merge"], "merge fixture")
    print(f"  merge fixture: {m['report']['tracklets_before']} -> {m['report']['tracklets_after']} tracklets, "
          f"sha {msha[:16]}")
    try:
        r, sha, px, _ = identity_run(g, table)
    except _Moved as e:
        return bad + [str(e)]
    rep = r["report"]
    print(f"  replay: {rep.get('tracklets_before')} tracks -> {rep.get('groups')} groups, {rep.get('crops_attempted')} "
          f"crops, {rep.get('ocr_reads')} reads, {rep.get('tracks_assigned')} tracks named, sha {sha[:16]} "
          f"(identity {px.secs} s{f', OCR {px.ocr_s:.1f} s' if table is None else ''})")
    bad += _id_compare(r, sha, g["identity"], "identity")
    if table is None:
        _gt_diag(r)
        now = px.lookup()
        with gzip.open(ICROPS) as fh:
            was = json.load(fh)
        if now != was:
            bad.append(f"the committed crop/OCR lookup no longer matches the clip + docTR: crops {len(now['crops'])} "
                       f"vs {len(was['crops'])}, reads {len(now['ocr'])} vs {len(was['ocr'])} - --fast is serving "
                       f"stale pixels; --rebless")
    return bad


def identity_fast(g):
    procs = _start(ISELFCHECKS)                    # in parallel with the 19 s track replay
    table = _crops_table(g)
    bad = identity_check(g, table)
    return selfchecks(ISELFCHECKS, procs) + bad


def identity_real(g):
    _crops_table(g)
    return identity_check(g)


def identity_rebless(reason):
    if not reason.strip():
        print("refused: --rebless needs a non-empty reason")
        return 2
    import identity
    try:
        m, msha = identity_merge()
        r, sha, px, tsha = identity_run(None, None)
    except CouldNotRun as e:
        print(f"COULD NOT RUN: {e}")
        return 2
    if r["report"].get("strategy") != "assisted" or r["report"].get("ocr_backend") != "present":
        print(f"refused: strategy {r['report'].get('strategy')!r}, OCR {r['report'].get('ocr_backend')!r} - the shipped "
              "default is 'assisted' with docTR present; not freezing anything else")
        return 1
    tmp = ICROPS.with_suffix(".tmp")
    with gzip.GzipFile(tmp, "wb", 9, mtime=0) as fh:
        fh.write(json.dumps(px.lookup(), sort_keys=True).encode())
    os.replace(tmp, ICROPS)
    print(f"  merge fixture {m['report']}; replay {r['report']} ({px.secs} s, OCR {px.ocr_s:.1f} s)")
    _write(IGOLDEN, {**_stamp(reason),
                     "config": {"strategy": "assisted (FUTTRAFFIC_IDENTITY unset)", "ocr": "auto",
                                "min_conf": identity.AssistedIdentity().min_conf, "max_gap_s": identity.MAX_GAP_S,
                                "merge_margin": identity.MERGE_MARGIN, "front_weight": identity.FRONT_WEIGHT,
                                "min_sharpness": identity.MIN_SHARPNESS, "calibration": TCAL.name},
                     "tracks_sha256": tsha, "clip": _TRACKS["res"][1],
                     "fixtures": {ICROPS.name: _sha(ICROPS), IMERGE.name: _sha(IMERGE)},
                     "made_by": {IMERGE.name: IMERGE_MADE_BY,
                                 ICROPS.name: "identity --rebless: every torso crop scan_video requested, keyed "
                                              "frame:x1,y1,x2,y2 -> crop sha256 + Laplacian variance; docTR read per "
                                              "crop sha256. No pixels."},
                     "merge": {"sha256": msha, "result": m}, "identity": {"sha256": sha, "result": r}})
    return 0


def _mutsrc(owner, name, old, new):
    """Patch owner.name with its own source, `old` replaced by `new` (a mutant a constant cannot express)."""
    import inspect
    import textwrap
    fn = getattr(owner, name)
    src = textwrap.dedent(inspect.getsource(fn))
    assert old in src, f"{name}: {old!r} not in its source - the mutant is stale"
    ns = {}
    exec(compile(src.replace(old, new), f"<mutant {name}>", "exec"), fn.__globals__, ns)  # noqa: S102
    return mock.patch.object(owner, name, ns[name])


def identity_prove(g):
    import identity
    A, real_torso = identity.AssistedIdentity, identity.torso_box
    table = _crops_table(g)
    return _prove_cases([
        ("unpatched", [], False),
        ("MERGE_MARGIN 0.12 -> 0.06", [mock.patch.object(identity, "MERGE_MARGIN", 0.06)], True),
        ("MAX_GAP_S 5 -> 2.5", [mock.patch.object(identity, "MAX_GAP_S", 2.5),
                                _redefault(identity.merge_tracklets, "max_gap_s", lambda v: 2.5),
                                _redefault(A.__init__, "max_gap_s", lambda v: 2.5)], True),
        ("FRONT_WEIGHT -> 1.0", [mock.patch.object(identity, "FRONT_WEIGHT", 1.0)], True),
        ("facing sort reversed", [_mutsrc(identity, "scan_video", "key=lambda t: (-t[0], -t[1])",
                                          "key=lambda t: (t[0], -t[1])")], True),
        ("torso_box ignores box_h", [mock.patch.object(identity, "torso_box", lambda x, y, fw, fh, box_h=None,
                                                       box_w=None: real_torso(x, y, fw, fh, None, box_w))], True),
        # x2 is INERT here, and must pass: every crop that reaches the sharpness test on this clip has a
        # Laplacian variance of 2249 or more (the committed lookup), 56x the threshold of 40, so no factor
        # short of that moves one. 3000 drops crops, so it is the planted defect this check can see.
        ("MIN_SHARPNESS x2 (inert: min var 2249)", [mock.patch.object(identity, "MIN_SHARPNESS",
                                                                      2 * identity.MIN_SHARPNESS)], False),
        ("MIN_SHARPNESS -> 3000", [mock.patch.object(identity, "MIN_SHARPNESS", 3000.0)], True),
        ("AssistedIdentity min_conf 0.6 -> 0.5", [_redefault(A.__init__, "min_conf", lambda v: 0.5)], True),
        ("_drop_duplicates keyed on number", [_mutsrc(A, "_drop_duplicates", "(p.team, p.number)", "p.number")], True),
    ], lambda g: identity_check(g, table), g)


def identity_section(argv):
    try:
        return _run_section("identity", argv, identity_fast, identity_prove, identity_rebless, IGOLDEN, identity_real)
    finally:                                         # _jobs' throwaway store, never the real one
        d = os.environ.get("FUTTRAFFIC_DATA", "")
        if Path(d).name.startswith("checkcal-"):
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- measure
MFIX = ROOT / "model" / "validation" / "fixtures" / "measure"
MGOLDEN, MTRACKS, MCORR = MFIX / "golden.json", MFIX / "118575.id.tracks.json.gz", MFIX / "corrections.json"
MSELFCHECKS = ("model/engine/stats.py", "model/engine/frames.py", "model/engine/roles.py", "model/engine/test_golden.py")
TG_SOURCE = ROOT / "model" / "validation" / "soccertrack" / "118575_1st.tracks.json.gz"   # test_golden SKIPs, exit 0, without it
MMADE_BY = {MTRACKS.name: "check.py measure --make-fixture: jobs.apply_identity's output on identity --fast's input (the track "
                          "replay + committed crop/OCR lookups), all keys but _perf's wall-clock two, gzip -9, mtime 0",
            MCORR.name: "hand-written in apply_corrections' stored form: one merge [keep, drop] of two tracks that never "
                        "overlap, one split of a full-length track at frame 3000"}
REAL_VARIANT = "p21id"   # dist_eval.evaluate_stc_all's tracks variant the golden freezes
REAL_KEYS = ("pooled_ratio_raw", "pooled_coverage", "pooled_ratio_matched", "player_ratio_median", "player_ratio_cv",
             "player_ratio_n", "pooled_hsr_ratio", "pooled_sprint_ratio", "player_peak_median_ms",
             "player_gt_peak_median_ms", "pooled_clamped_frac")


def _live_accepted():
    """The api:// ids METRICS names that api-golden.json accepts today - the live input build_match reads."""
    import stats
    ids = {e[4] for e in stats.METRICS.values() if len(e) > 4 and e[4]}
    return sorted(ids & stats.accepted_measurements())


def measure_run(accepted):
    """fixture -> jobs._derive_tracks (with the corrections fixture) -> stats.build_match on TCAL, tiers resolved
    against `accepted` (frozen), never the live api-golden.json. -> (match, sha, secs)."""
    import frames  # noqa: F401 - before any patch: frames copies stats.MIN_MINUTES at its (lazy) import
    import stats
    jobs = _jobs()
    tmp = Path(tempfile.mkdtemp(prefix="checkmeasure-"))
    try:
        with gzip.open(MTRACKS) as a, open(tmp / "tracks.id.json", "wb") as b:
            shutil.copyfileobj(a, b)
        shutil.copy(MCORR, tmp / "corrections.json")
        t = time.time()
        jobs._derive_tracks(tmp)
        with mock.patch.object(stats, "accepted_measurements", lambda: frozenset(accepted)), \
                contextlib.redirect_stdout(io.StringIO()):
            m = stats.build_match(str(tmp / "tracks.json"), str(TCAL), str(tmp / "match.json"))
        return (*_canon(m), round(time.time() - t, 1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _msummary(m):
    """What a mismatch is explained by: per-player rows, officials, team totals, per-field tier/caveats/n."""
    return {"players": {str(p["id"]): {k: p[k] for k in ("team", "minutes", "distance_km", "top_speed_kmh", "sprints")}
                        for p in m["players"]},
            "officials": m["meta"]["officials_excluded"], "territory": m["territory"],
            "teams": {s: {k: v for k, v in t.items() if k != "shape"} for s, t in m["teams"].items()},
            "fields": {k: [v.get("confidence"), v.get("caveat_flags"), v.get("n"), v.get("value")]
                       for k, v in m["fields"].items()}}


def _mcompare(s, sha, w):
    if sha == w["sha256"]:
        return []
    ws = w["summary"]
    diff = lambda a, b: {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}  # noqa: E731
    pl = {i: diff(a or {}, b or {}) if a and b else (a, b) for i, (a, b) in diff(s["players"], ws["players"]).items()}
    return [f"match sha {sha[:16]} != golden {w['sha256'][:16]}: players {len(s['players'])} vs {len(ws['players'])}, "
            f"officials {s['officials']} vs {ws['officials']}",
            f"fields moved (now, golden) [tier, flags, n, value]: {diff(s['fields'], ws['fields'])}",
            f"teams moved: {diff(s['teams'], ws['teams'])}; territory {s.get('territory')} vs {ws.get('territory')}",
            f"players moved (now, golden) per column, or the whole row when one side lacks it: {pl}"]


def measure_check(g):
    m, sha, secs = measure_run(g["accepted"])
    s = _msummary(m)
    print(f"  build_match: {len(s['players'])} players, {s['officials']} officials, {len(s['fields'])} fields, "
          f"sha {sha[:16]} ({secs} s)")
    return _mcompare(s, sha, g["fast"])


def measure_fast(g):
    if not TG_SOURCE.exists():
        raise CouldNotRun(f"{TG_SOURCE.relative_to(ROOT)} missing: test_golden.py would SKIP and exit 0")
    _fixture_shas((MTRACKS, MCORR, TCAL), g)
    procs = _start(MSELFCHECKS)
    bad = measure_check(g)
    live = _live_accepted()
    if live != g["accepted"]:
        print(f"  NOTE (not gated): api-golden.json now accepts {sorted(set(live) - set(g['accepted']))} and drops "
              f"{sorted(set(g['accepted']) - set(live))} vs the frozen set - a job today resolves tiers against the live one")
    return selfchecks(MSELFCHECKS, procs) + bad


def _real_inputs():
    """Our side of --real: the variant's tracks + calibrations per STC clip, by sha. Absent -> could not run."""
    import contam_eval
    import dist_eval
    ps = [Path(dist_eval.STC) / f"{q}.{v}" for q in contam_eval.STC_SEQS
          for v in (f"{REAL_VARIANT}.tracks.json", "p05.calibration.json")]
    miss = [p.name for p in ps if not p.exists()]
    if miss:
        raise CouldNotRun(f"{dist_eval.STC} lacks {miss[:4]}{'...' if len(miss) > 4 else ''} (local, gitignored)")
    return {p.name: _sha(p) for p in ps}


def measure_real_run():
    import dist_eval
    ins, t = _real_inputs(), time.time()
    try:
        r = dist_eval.evaluate_stc_all(variant=REAL_VARIANT, verbose=False)
    except (OSError, KeyError, SystemExit) as e:                       # ST2 GT / the join absent
        raise CouldNotRun(f"evaluate_stc_all could not read its ground truth: {type(e).__name__}: {e}") from e
    return json.loads(json.dumps(r)), ins, round(time.time() - t, 1)


def measure_real(g):
    r, ins, secs = measure_real_run()
    w = g["real"]
    if ins != w["inputs"]:
        raise CouldNotRun(f"out/stc2025 inputs differ from the golden's: {sorted(k for k in ins if ins[k] != w['inputs'].get(k))}")
    print(f"  evaluate_stc_all({REAL_VARIANT!r}): " + ", ".join(f"{k} {r.get(k)}" for k in REAL_KEYS) + f" ({secs} s)")
    if r == w["result"]:
        return []
    moved = {k: (r.get(k), w["result"].get(k)) for k in sorted(set(r) | set(w["result"])) if r.get(k) != w["result"].get(k)}
    return [f"evaluate_stc_all moved (now, golden): {moved}"]


def measure_rebless(reason):
    if not reason.strip():
        print("refused: --rebless needs a non-empty reason")
        return 2
    try:
        _fixture_shas((MTRACKS, MCORR, TCAL), None)
        acc = _live_accepted()
        m, sha, secs = measure_run(acc)
        r, ins, rsecs = measure_real_run()
    except CouldNotRun as e:
        print(f"COULD NOT RUN: {e} - a measure golden freezes --fast and --real together")
        return 2
    s = _msummary(m)
    print(f"  build_match: {len(s['players'])} players, {s['officials']} officials, sha {sha[:16]} ({secs} s); "
          f"evaluate_stc_all {rsecs} s")
    import stats
    _write(MGOLDEN, {**_stamp(reason),
                     "constants": {k: getattr(stats, k) for k in ("MAX_SPEED", "SPRINT_SPEED", "SPRINT_MIN_S",
                                                                  "MIN_MINUTES", "DISTANCE_WINDOW_S")},
                     "accepted": acc, "accepted_note": "api-golden.json ids (METRICS' only) frozen at bless time; "
                                                       "--fast resolves tiers against this set, not the live file",
                     "fixtures": {p.name: _sha(p) for p in (MTRACKS, MCORR, TCAL)}, "made_by": MMADE_BY,
                     "fast": {"sha256": sha, "summary": s},
                     "real": {"variant": REAL_VARIANT, "inputs": ins, "result": r}})
    return 0


def measure_make_fixture():
    """Regenerates MTRACKS from identity's --fast path (needs the clip, as identity does). Refuses unless identity
    reproduces its own golden: the fixture is that stage's output, not a new one."""
    ig = _golden(IGOLDEN)
    r, sha, px, _ = identity_run(ig, _crops_table(ig))
    if sha != ig["identity"]["sha256"]:
        print(f"refused: identity sha {sha[:16]} != its golden {ig['identity']['sha256'][:16]} - check identity first")
        return 1
    for k in ("seconds", "throughput_fps"):                 # wall clock: the only bytes that differ run to run
        px.out.get("_perf", {}).pop(k, None)
    MFIX.mkdir(parents=True, exist_ok=True)
    tmp = MTRACKS.with_suffix(".tmp")
    with gzip.GzipFile(tmp, "wb", 9, mtime=0) as fh:
        fh.write(json.dumps(px.out).encode())
    os.replace(tmp, MTRACKS)
    print(f"wrote {MTRACKS.relative_to(ROOT)}: {len(px.out['detections'])} rows, sha {_sha(MTRACKS)[:16]}")
    return 0


def measure_prove(g):
    import frames
    import stats
    E = stats.METRICS["players.distance_km"]
    assert E[2] == "ok" and E[3], E              # the two METRICS mutants below assume this shape
    const = lambda mod, k, f: mock.patch.object(mod, k, f(getattr(mod, k)))  # noqa: E731
    return _prove_cases([
        ("unpatched", [], False),
        ("DISTANCE_WINDOW_S x2", [const(stats, "DISTANCE_WINDOW_S", lambda v: 2 * v)], True),
        # INERT here since sprints are withheld (null in match.json); --real still reads them (pooled_sprint_ratio).
        ("SPRINT_SPEED 5.5 -> 6.0 (inert: withheld)", [const(stats, "SPRINT_SPEED", lambda v: 6.0)], False),
        ("SPRINT_MIN_S x2 (inert: withheld)", [const(stats, "SPRINT_MIN_S", lambda v: 2 * v)], False),
        ("MAX_SPEED 9.5 -> 12", [const(stats, "MAX_SPEED", lambda v: 12.0)], True),
        ("MIN_MINUTES x2", [const(stats, "MIN_MINUTES", lambda v: 2 * v), const(frames, "MIN_MINUTES", lambda v: 2 * v)], True),
        ("officials kept in players", [_mutsrc(stats, "build_match", 'players = [p for p in players if p.team != "official"]',
                                               "players = list(players)")], True),
        ("distance_km drops a caveat flag", [mock.patch.dict(stats.METRICS, {"players.distance_km": E[:3] + (E[3][1:],) + E[4:]})], True),
        ("heatmap_points max_points 600 -> 60", [_redefault(stats.heatmap_points, "max_points", lambda v: 60)], True),
        # INERT here, and must pass: distance_km's caveats already cap its tier at prov, so the class ceiling is not
        # the min() that decides it (stats.metric).
        ("distance_km tier ok -> rel (inert)", [mock.patch.dict(stats.METRICS, {"players.distance_km": E[:2] + ("rel",) + E[3:]})], False),
    ], measure_check, g)


def measure_section(argv):
    try:
        if argv[:1] == ["--make-fixture"]:
            try:
                return measure_make_fixture()
            except CouldNotRun as e:
                print(f"COULD NOT RUN: {e}")
                return 2
        return _run_section("measure", argv, measure_fast, measure_prove, measure_rebless, MGOLDEN, measure_real)
    finally:                                         # _jobs' throwaway store, never the real one
        d = os.environ.get("FUTTRAFFIC_DATA", "")
        if Path(d).name.startswith("checkcal-"):
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- ball
BFIX = ROOT / "model" / "validation" / "fixtures" / "ball"
BGOLDEN, BM0 = BFIX / "golden.json", BFIX / "m0_cands.json.gz"
BSELFCHECKS = ("model/engine/ball.py",)
BMADE_BY = ".venv/bin/python model/validation/n70_class_merge.py --export   # M0 of the N70 raw caches (gitignored)"


def _bexcerpts():
    import n70_class_merge as N
    return {k: BFIX / f"{c}.m0.s2f0-598.json.gz" for k, c in N.EXCERPTS.items()}


def _ball_rows(frames, shape, stride, cal):
    """track.track() replaying a complete cache on blank frames: the product's ball path (split, picker,
    marking gate at its default) with detector None. -> ball rows."""
    import numpy as np
    import track
    import tracking
    z = np.zeros(shape, np.uint8)
    res = track.track(iter([z] * len(frames)), 25.0, None, tracking.get_tracker(_product()["tracker"]),
                      cache={int(f): v for f, v in frames.items()}, stride=stride, ball=True, cal=cal)
    return res["ball"]


def ball_fast_run():
    """-> {name: {sha256, rows[, correct, wrong, silent, gt_present]}} for the two M0 excerpts and the
    detect fixture's frames on TCAL."""
    import calibrators
    import cv2
    import detectors
    import ball_eval
    import numpy as np
    stride, out = _product()["stride"], {}
    for name, p in _bexcerpts().items():
        if not p.exists():
            raise CouldNotRun(f"{p.relative_to(ROOT)} missing - regenerate: {BMADE_BY}")
        with gzip.open(p) as fh:
            fx = json.load(fh)
        if fx["stride"] != stride:
            raise CouldNotRun(f"{p.name} was cut at stride {fx['stride']}, the product runs {stride}")
        # the worker's per-class ball threshold, re-applied: a no-op at the shipped BALL_CONF (every
        # fixture ball is >= it), so a raised one is seen here (--prove)
        frames = {f: [d for d in v if d["cls"] not in detectors.BALL_CLASSES or d["conf"] >= detectors.BALL_CONF]
                  for f, v in fx["frames"].items()}
        cal = calibrators.from_dict(fx["calibration"]) if fx["calibration"] else None
        rows = _ball_rows(frames, (fx["frame_h"], fx["frame_w"], 3), stride, cal)
        gt = {int(f): np.array(v) for f, v in fx["gt"].items()}
        sc = ball_eval.score(gt, {r["f"]: np.array([r["x"], r["y"]]) for r in rows}, sorted(map(int, frames)))
        out[name] = {"sha256": _rows_sha(rows), "rows": len(rows),
                     **{k: sc[k] for k in ("correct", "wrong", "silent", "gt_present")}}
    # the detect fixture's 10 frames (ball candidates included) on the track fixture's SphericalCamera: the
    # path a spherical calibration takes through the size gate
    dg = _golden(DGOLDEN)
    cap = cv2.VideoCapture(str(DCLIP))
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
    cap.release()
    fr = dg["detectors"][dg["picked"]]["frames"]
    if sorted(map(int, fr)) != list(range(0, stride * len(fr), stride)):
        raise CouldNotRun(f"detect golden frames {sorted(map(int, fr))} are not stride {stride} from 0")
    rows = _ball_rows(fr, shape, stride, calibrators.load(str(TCAL)))
    out["tcal"] = {"sha256": _rows_sha(rows), "rows": len(rows), "input_sha256": _rows_sha(fr),
                   "candidates": sum(d["cls"] in detectors.BALL_CLASSES for v in fr.values() for d in v)}
    return out


def ball_real_run():
    """The 9-clip metric, per clip, from the committed M0 candidates. -> {clip: {arm: counts}}."""
    import ball_eval
    if not BM0.exists():
        raise CouldNotRun(f"{BM0.relative_to(ROOT)} missing - regenerate: {BMADE_BY}")
    try:
        return _quiet(ball_eval.score_m0, str(BM0))["clips"]
    except (SystemExit, OSError, RuntimeError, KeyError) as e:      # truth CSVs absent, merge moved
        raise CouldNotRun(f"ball_eval.score_m0: {e}") from e


def _bcmp(got, want, what):
    return [f"{what} {k}: {got.get(k)} != golden {want.get(k)}" for k in sorted(set(got) | set(want))
            if got.get(k) != want.get(k)]


def ball_fast(g):
    _fixture_shas(list(_bexcerpts().values()) + [TCAL], g)
    bad = selfchecks(BSELFCHECKS)
    t = time.time()
    r = ball_fast_run()
    for k, v in r.items():
        print(f"  {k}: {v}")
    print(f"  ({time.time() - t:.1f} s)")
    return bad + _bcmp(r, g["fast"], "ball rows")


def ball_real(g):
    t = time.time()
    r = ball_real_run()
    pooled = {k: sum(c[k]["correct"] for c in r.values()) / sum(c[k]["gt_present"] for c in r.values())
              for k in ("shipped/s1", "oracle/s1", "shipped/s2")}
    print("  pooled " + ", ".join(f"{k} {v:.4f}" for k, v in pooled.items()) + f" ({time.time() - t:.1f} s)")
    return _bcmp(r, g["real"], "clip")


def ball_rebless(reason):
    if not reason.strip():
        print("refused: --rebless needs a non-empty reason")
        return 2
    import ball
    import detectors
    try:
        fast, real_ = ball_fast_run(), ball_real_run()
    except CouldNotRun as e:
        print(f"COULD NOT RUN: {e}")
        return 2
    print(f"  fast: {fast}")
    sc = ball.Scenery()
    _write(BGOLDEN, {**_stamp(reason),
                     "config": {"stride": _product()["stride"], "tracker": _product()["tracker"],
                                "ball_markings": False, "BALL_CONF": detectors.BALL_CONF,
                                "SIZE_RATIO": list(ball.BallPicker.SIZE_RATIO), "MARGIN_M": ball.BallPicker.MARGIN_M,
                                "scenery": {k: getattr(sc, k) for k in ("radius", "window", "share", "min_frames")}},
                     "fixtures": {p.name: _sha(p) for p in list(_bexcerpts().values()) + [TCAL]},
                     "real_input": {BM0.name: _sha(BM0)}, "made_by": BMADE_BY,
                     "fast": fast, "real": real_})
    return 0


def ball_prove(g):
    import ball
    import detectors
    import track
    Picker = ball.BallPicker
    return _prove_cases([
        ("unpatched", [], False),
        ("MAX_BLOB_EXTENT 3.0 (inert: gate off)", [mock.patch.object(ball, "MAX_BLOB_EXTENT", 3.0)], False),
        ("SIZE_RATIO (0.4,3.0)->(0.4,2.0)", [mock.patch.object(Picker, "SIZE_RATIO", (0.4, 2.0))], True),
        ("MARGIN_M 3 -> 30", [mock.patch.object(Picker, "MARGIN_M", 30.0)], True),
        ("Scenery min_frames 25 -> 3", [_redefault(ball.Scenery.__init__, "min_frames", lambda v: 3)], True),
        ("track.py hands the picker cal=None", [mock.patch.object(track._ball, "BallPicker",
                                                                  lambda cal=None, **k: Picker(cal=None, **k))], True),
        ("BALL_CONF 0.05 -> 0.10", [mock.patch.object(detectors, "BALL_CONF", 0.10)], True),
        ("size gate back through inv(H) (pre-fix)", [mock.patch.object(Picker, "_expected_px", _expected_px_via_h)],
         True),
    ], ball_fast, g)


def _expected_px_via_h(self, xy):
    """BallPicker._expected_px before 2026-09-29: forward through to_pitch, back through inv(H) - wrong
    wherever H is a stand-in (SphericalCamera). The TCAL replay must see it."""
    import numpy as np
    m = self.cal.to_pitch(xy)
    q = np.c_[m + np.array([0.22, 0.0]), np.ones(len(m))] @ np.linalg.inv(np.asarray(self.cal.H, float)).T
    return np.hypot(*((q[:, :2] / np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])) - xy).T)


def ball_section(argv):
    return _run_section("ball", argv, ball_fast, ball_prove, ball_rebless, BGOLDEN, ball_real)


SECTIONS = {"calib": calib, "track": track_section, "detect": detect_section, "identity": identity_section,
            "measure": measure_section, "ball": ball_section}

def log_run(section, argv, code):
    """One JSON line per section run, append-only. A failed write never changes the exit code."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")}).stdout.strip()
    try:
        log = ROOT / "model" / "validation" / "out" / "check-runs.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "section": section,
                                 "mode": argv[0] if argv else "", "exit": code, "git": head}) + "\n")
    except OSError as e:
        print(f"(run not logged: {e})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "status":
        import check_status
        raise SystemExit(check_status.main(sys.argv[2:], list(SECTIONS), {"calib": GOLDEN, "track": TGOLDEN, "detect": DGOLDEN,
                                                                   "identity": IGOLDEN, "measure": MGOLDEN,
                                                                   "ball": BGOLDEN}))
    if len(sys.argv) < 2 or sys.argv[1] not in SECTIONS:
        print(__doc__)
        raise SystemExit(2)
    code = SECTIONS[sys.argv[1]](sys.argv[2:])
    log_run(sys.argv[1], sys.argv[2:], code)
    raise SystemExit(code)
