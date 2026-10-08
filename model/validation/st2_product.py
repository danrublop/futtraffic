"""N6: the tracker at the PRODUCT config, on the ten ST2 sequences.

WHY THIS EXISTS
    Every registered tracker id scores `.c03s` - conf 0.3, stride 1, --roles,
    no calibration, HeightScale stitch (model/validation/st2_run.py). app/server/jobs.py
    runs none of that. model-map (not published) §3 found the product's tracking had
    never been scored at all, and §5 N6 is the run every product tracking
    claim waits on. This is that run.

PROTOCOL, fixed before the first run (2026-09-11), not after seeing a number:

    population   all ten ST2 sequences; the 8 held out and the 2 `.c03s` was
                 tuned on are reported APART. n=1 configuration - nothing is
                 tuned here, so there is no sweep for an argmax to hide in.
    calibration  SIMULATED CLICKS. The UI's 21 landmarks (Pitch.landmarks)
                 put through ST2's own pitch->pixel chain (st2_overlay.
                 projector); every one that lands in frame is "clicked", and
                 the PRODUCT's solver fits them - app/frontend/src/lib/pitch.ts
                 solveClicks, run headless by app/frontend/scripts/solve-clicks.mts
                 - and emits the payload the UI posts. NOT model/engine/pitch.py's
                 Calibration.solve: on this stitched panorama its LMEDS threw
                 out 16 of 21 perfect clicks on 128057 and found no homography
                 at all on 132831, where pitch.ts stops rejecting once a fifth
                 of the points disagree. A PERFECT operator, so this is a
                 ceiling on what a user's clicks give, not an estimate of them.
    tracker      model/engine/track.run exactly as app/server/jobs.py calls it: coreml,
                 conf at its default (0.5), stride 2, calib -> scene prior and
                 metric stitch.                        -> {seq}.p05.tracks.json
    identity     jobs.apply_identity on that output - the stage a job runs
                 next, the same function.            -> {seq}.p05id.tracks.json
    frames       GT restricted to the frames a stride-2 tracker runs on
                 (st2_mot.stage_gt(every=2)): half of them, dropped by cadence, not
                 by selection. `.c03s` is re-scored on the same frames as
                 `.c03s_e2`, so the comparison is like for like.
    metrics      HOTA, DetA, AssA, IDSW via st2_mot (TrackEval).
    variant B    ".p05m": the same tracker and identity, but calibrated with ST2's
                 OWN chain expressed as the product's `remap` calibration - the
                 vendor homography and tables, cropped to calibrators.pitch_band
                 and upscaled isotropically by the measured squash, exactly as
                 calibrators._measure_real derives them; nothing is fitted. The
                 UI cannot produce this, so it is not the product: it is the
                 tracker with calibration taken out of the picture, and A minus
                 B is what the operator-clickable calibration costs. Its tracks
                 come out in canvas pixels and are mapped back through the same
                 tables to be scored; rows whose feet leave the table are
                 dropped and counted. A rig whose band is the whole canvas
                 (117092: 162 Mpx a frame, ~15x the others) is excluded as a
                 COMPUTE limit, stated here before running; A-B compares the
                 sequences both variants ran.
    N18 replay   ".p05r": B's detections (its cache holds boxes AFTER B's scene
                 prior, built from a good calibration) mapped to clip pixels
                 through B's own tables, then REPLAYED through track.run with
                 A's calibration - a complete cache means no detector and no A
                 prior (track.track skips the prior on a cache hit), so the
                 metric stitch runs through A's ~6 m homography on B's boxes.
                 Fixed readings, stated before running: .p05r vs A is what A's
                 prior-filtered detection costs; .p05r vs B is what the bad
                 homography costs ASSOCIATION - confounded by pixel space, since
                 B's tracker matched boxes on the flattened canvas and this one
                 on the raw frame. Boxes are the extent of the 4 mapped corners;
                 dropped boxes are counted.
    N19 arms     one detection pass per sequence with NO prior (track.run with
                 no calibration, so its cache holds unfiltered boxes), then the
                 scene prior applied OFFLINE to those same boxes through
                 track.make_prior(gates=...) built from A's calibration, and
                 each arm replayed through track.run with A's calibration:
                 ".p19v0" both gates (the shipped prior - must reproduce A),
                 ".p19v1" none, ".p19v2" pitch polygon only, ".p19v3" height
                 gate only. Scored with DetRe/DetPr through the tracker and a
                 box-level precision/recall on the cached boxes (IoU >= 0.5,
                 greedy, even frames), paired deltas vs V0 with a 95% t CI.
                 Pass rule, fixed in model-map (not published) §3 N19 before running.
    N20 arms     N19's four arms again, on the GOOD calibration: B's remap, on
                 B's canvas. N19 could not say whether the height gate EARNS
                 its keep when the camera is recovered from a fit that is
                 right - every calibration there was ~6 m off. Detection runs
                 once per sequence with B's calibration and prior_gates=()
                 (the calibration is needed to flatten the frame at all, the
                 gate is not), then the same offline arms and replays; tracks
                 come back through to_panorama for scoring, as B's do.
                 ".p20v0" both gates (must reproduce B), ".p20v1" none,
                 ".p20v2" polygon only, ".p20v3" height only. On the 7 held-out
                 sequences (COMMON7), which is what the paired rule reads.
                 PASS RULE, fixed before running: the height gate earns its
                 keep on a good calibration iff v0 beats v2 on HOTA on >= 6 of
                 the 7 held-out sequences. If it does, the product change is to
                 GATE the height check on calibration evidence and a threshold
                 has to be found (N12 data); if it does not, on either
                 calibration, the change is to delete the height check.
    excluded     a sequence whose ST2 chain puts the landmarks on fewer than 4
                 distinct pixels cannot be clicked at all: 132831 maps all 16
                 in-frame landmarks to ONE pixel. Reported as excluded with that
                 reason - a limit of the benchmark data, not a product result.

N17: CONF 0.3 AGAINST CONF 0.5 THROUGH THE TRACKER
--------------------------------------------------
PROTOCOL, fixed 2026-09-12 before the first detection ran.

WHY. model/engine/detectors.py has measured twice now that conf 0.5 throws away the
far touchline: band-0 image recall is 0.260 at conf 0.5 against 0.655 at 0.3,
and image precision pays 0.866 -> 0.827. That is THE DETECTOR IN ISOLATION.
app/server/jobs.py:860 keeps conf at 0.5 anyway, and its stated reason is not that
the gain is fake - it is that more marginal boxes into an ONLINE tracker is the
input that welds two players into one id, and "the metric that would show that
- contamination through the shipping path - has no runnable implementation in
this repo". N8 wrote it (model/validation/contam_eval.py, 2026-09-12). So the reason
conf stays at 0.5 has expired and this is the run that replaces it.

POPULATION  the val split, 117093 and 132877 (model/validation/st2_eval.py SPLIT).
            TWO SEQUENCES. That is the whole population and it is small: there
            is no interval to quote off n=2, so the pass rule below is an
            AGREEMENT rule over sequences and the report says so rather than
            dressing two numbers as an estimate.
HELD FIXED  everything `track()` holds fixed for N6: the UI-solved 21-click
            calibration already on disk ({seq}.p05.calibration.json), coreml,
            stride 2, prior_gates ("pitch", "height"), the same clips, and the
            same stride-2 GT frames. The conf-0.5 arm is N6's OWN `.p05` /
            `.p05id` output, not a re-run, so the two arms differ in one thing.
CHANGED     the DETECTOR's conf, 0.5 -> 0.3. n=2 configurations, both reported.
THE TRAP    `--dets-cache` keys on video and start frame and NOT on conf
            (model/engine/detectors.py, "ONE TRAP"), so a conf sweep that reuses a
            cache replays the old boxes and reports that confidence does
            nothing. The 0.3 arm writes its own cache path, and `run_n17`
            ASSERTS that it holds strictly more boxes than the 0.5 arm's before
            it scores anything. If that assert ever passes silently, the run is
            void, which is why it is an assert and not a printed warning.
METRIC      HOTA primary, with DetA / AssA / IDSW / DetRe / DetPr beside it -
            N6's set - because the whole question is whether a detection gain
            survives association, and model-map (not published) §3 records association
            as the weak half. Contamination through the shipping path
            (contam_eval.score_clip on the `id` stage) is the guard.
FLOOR       the conf-0.5 arm itself: this is a paired change to a shipped
            config, so the thing to beat is what ships.
DROP SHARE  every contamination row carries `unattributed_frac` (our boxes that
            matched no GT box and are therefore not scored at all) and
            `gt_recall`; both are printed for both arms, because a tracker that
            emits fewer, cleaner boxes looks better on contamination for
            exactly the wrong reason.
PASS RULE   MOVE the product to conf 0.3 iff HOTA rises on BOTH val sequences
            AND contamination does not rise on either.
            KEEP conf 0.5 if HOTA falls on either sequence, or contamination
            rises on either.
            A SPLIT - HOTA up on one and down on the other - is KEEP, and the
            honest report is "two sequences cannot decide this; here are the
            other eight". `_n17_verdict` applies it in code; `demo()` asserts
            it fires in both directions.
FAILING     anything but the first leaves app/server/jobs.py at conf 0.5. Nothing
            in this file changes the shipped conf; it produces the number that
            lets someone else change it.

N21: THE PRODUCT AS IT SHIPS TODAY, RE-RUN
------------------------------------------
PROTOCOL, fixed 2026-09-13 before the first detection ran.

WHY. N6, N8-N11 and N17 were all measured on tracks detected 2026-09-11 at tile
overlap 0.10 behind both prior gates. The product has shipped overlap 0.25 since
8472cef and the pitch polygon alone since c6b8fb8. Nothing on disk is that
configuration: `.p19v2` has the gate right and the overlap wrong, and a cache
written before 051c7f2 cannot say what it was detected under. This run makes the
tracks the downstream re-measures read.

POPULATION  ALL10 less 132831, whose ST2 chain cannot be clicked (N6's own
            exclusion: 1 of 10 sequences, before running). 128057 is `.c03s`'s
            tuned sequence and is reported apart, as N6 does. n=1
            configuration: nothing is tuned, so there is no argmax.
HELD FIXED  N6's `{seq}.p05.calibration.json`, REUSED and not re-solved, so the
            calibration cannot move between `.p05` and `.p21`; the clips; the
            stride-2 GT frames; `track()` and `identify()` exactly as N6 called
            them, which model/validation/shipped_config.py checks against
            app/server/jobs.py.
CHANGED     the product code, from what ran on 2026-09-11 to what ships: gates
            ("pitch", "height") -> ("pitch",) (c6b8fb8), tile overlap 0.10 ->
            0.25 (8472cef). No commit since touches model/engine/tracking.py,
            stitch.py, identity.py or calibrators.py, and app/server/jobs.py's one
            (202254e) moved the identity stage into the function identify()
            calls. So `.p21` against `.p05` is the product then against the
            product now - TWO changes at once - and is never reported as the
            effect of either.
OUTPUTS     {seq}.p21.tracks.json, a fresh detection pass behind its own cache,
            then {seq}.p21id.tracks.json, the job's identity stage. Scores go to
            n21_results.json and NOT p05_results.json: run_n17 re-scored `.p05`
            on its two sequences into that file and the update replaced N6's
            record of the rest.
VOID IF     any `.p21` tracks file does not diff ON against
            shipped_config.product_config(), a missing stamp included. Asserted
            before anything is scored: a file that is not what ships measures
            something else, whatever it scores.
METRICS     N6's - HOTA, DetA, AssA, IDSW, DetRe, DetPr - on `.p21`/`.p21id`
            beside `.p05`/`.p05id`, same sequences, same frames. Downstream, the
            scorers N8-N11 already use, pointed at variant p21/p21id, each with
            the floor and drop share its own docstring names.
DROP SHARE  132831, above. A sequence that fails to track is recorded with its
            error in `skipped`, never silently absent.
RULE        none: this decides nothing. Its values REPLACE the `.p05` ids, by
            name and with a reason, through the one writer of the golden
            (parallel-work-plan (not published) §3). A VOID run replaces nothing.
NOT HERE    N7, below: it replays `.p21`'s cache.

N7: DOES THE METRIC STITCH EARN ITS PLACE
-----------------------------------------
PROTOCOL, fixed 2026-09-13 before either arm was tracked.

WHY. Every job's tracks go through track.run's metric stitch, and the A/B that
would justify it has never been run on the product: model-map (not published) §5 N7
waited on N6, whose tracks carried a prior the product has since retired. N21
is the product, so N7 runs on it.

ARMS        both REPLAYED from N21's own detection cache, with N6's calibration,
            stride 2 and the shipped gates - nothing is re-detected, so the arms
            differ in one thing:
              .p21s   track.run(stitch=True)   the product. Must reproduce
                      .p21's detections exactly, or the replay is not what
                      N21 measured and N7 is VOID.
              .p21ns  track.run(stitch=False)  the same, unstitched.
GUARD       shipped_config.compare_guard on the two arms' own stamps must say OK
            with varied={"stitch"} before anything is scored. CONFOUNDED,
            NO_DIFFERENCE or UNSTAMPED is VOID.
POPULATION  N21's nine sequences; 128057 is `.c03s`'s tuned sequence and st2_ab
            marks it, counting every sequence as its rule does. n=2
            configurations.
METRIC      AssA primary - association is what a stitch changes - with HOTA and
            DetA beside it, IDSW, and st2_ab's coalescence guard (duplicate ids
            per frame): a stitch that merges two players raises AssA too.
RULE        the stitch SHIPS iff st2_ab.ab(".p21ns", ".p21s", "AssA") passes - a
            one-sided sign test at p < 0.05 over the sequences that move, a
            positive mean, no coalescence - AND mean held-out IDSW does not rise
            (the rule model-map (not published) §5 N7 already declared).
            DOES NOT SHIP otherwise, with st2_ab's own label kept beside it -
            INCONCLUSIVE when too few sequences move to reach p < 0.05 at all.
FAILING     DOES NOT SHIP is a product decision this file does not make: the
            stitch stays in the job until someone removes it, and model-map
            §6's "two mergers run in series: pick one after N7" gets its input.
OUTPUTS     n7_results.json beside the tracks.

    .venv/bin/python model/validation/st2_product.py --n21      # N21, the nine clickable sequences
    .venv/bin/python model/validation/st2_product.py --n7       # N7, stitch on/off off N21's cache
    .venv/bin/python model/validation/st2_product.py --n17      # N17, val split only

    .venv/bin/python model/validation/st2_product.py            # variant A, every stage, resumes
    .venv/bin/python model/validation/st2_product.py --map      # variant B, every stage, resumes
    .venv/bin/python model/validation/st2_product.py --replay   # N18, needs A and B done
    .venv/bin/python model/validation/st2_product.py --n19      # N19, on the sequences B ran
    .venv/bin/python model/validation/st2_product.py --n20      # N20, the same arms on B's map
    .venv/bin/python model/validation/st2_product.py --score    # score what exists
    .venv/bin/python model/validation/st2_product.py --demo     # calibration only, seconds
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import st2_mot as T  # noqa: E402
from st2_overlay import pitch_of, projector  # noqa: E402

ALL10 = ("117092", "117093", "118575", "118576", "118577", "118578",
         "128057", "128058", "132831", "132877")
TUNED = ("128057", "132831")
CLIPS = f"{T.GT_SRC}/clips"
STRIDE = 2          # app/server/jobs.py: track_run(..., stride=2)
VARIANTS = (".p05", ".p05id", ".c03s_e2", ".p05m", ".p05mid", ".p05r")
N19 = {".p19v0": ("pitch", "height"), ".p19v1": (), ".p19v2": ("pitch",),
       ".p19v3": ("height",)}
N20 = {f".p20v{i}": g for i, g in enumerate(N19.values())}
#: Variant B only. The ST2 rigs flatten to 9-11 Mpx; 117092's does not.
MAX_CANVAS_MPX = 20


def _p(seq, suffix):
    return os.path.join(T.WORK, f"{seq}{suffix}")


def calibrate(seq, out_dir=None):
    """Simulated clicks, solved by the UI's own solver ->
    {seq}.p05.calibration.json. -> (path, report). `out_dir` lets --demo
    write elsewhere: the run's calibration files are N6 inputs and its record."""
    pitch = pitch_of(seq)
    lm = pitch.landmarks()
    px, ok = projector(seq, T._wh(seq))(list(lm.values()))
    clicks = {n: [float(u), float(v)] for n, (u, v), k in zip(lm, px, ok) if k}
    if len({(round(u), round(v)) for u, v in clicks.values()}) < 4:
        raise ValueError(f"{seq}: ST2's pitch->pixel chain is degenerate here "
                         f"({len(clicks)} landmarks on fewer than 4 distinct pixels)")
    r = subprocess.run(
        ["npx", "vite-node", "--config", "vitest.config.mts", "scripts/solve-clicks.mts"],
        cwd=os.path.join(ROOT, "app", "frontend"), check=True, capture_output=True, text=True,
        input=json.dumps({"pitch": pitch.__dict__, "clicks": clicks}))
    payload = json.loads(r.stdout)
    if "error" in payload:
        raise RuntimeError(f"{seq}: the UI's solver found no calibration")
    path = (os.path.join(out_dir, f"{seq}.p05.calibration.json") if out_dir
            else _p(seq, ".p05.calibration.json"))
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return path, {"clicked": len(clicks), "of": len(lm),
                  "outliers": payload["outliers"],
                  "fit_rms_m": round(payload["rms_error_m"], 3)}


def track(seq, calib, tag=".p05", gates=None, conf=None, stitch=None, cache_from=None,
          ball=True):
    """The job's tracker call, one process per sequence (MPS does not give
    memory back, model/engine/detsweep.py). Resumes from its detection cache.
    `gates` is None - DEFER to whatever the product ships - unless a
    measurement asks otherwise. It used to default to ("pitch", "height"), a
    copy of the product default that went stale the moment 0a5a7cd moved it,
    and a harness holding its own copy of the shipped config is the thing this
    file exists to catch.

    `ball` DEFAULTS TO TRUE because app/server/jobs.py passes ball=True since
    66fba82 (2026-09-23) and this function's whole job is to be that call. It was
    absent, so it inherited track.run's ball=False and the harness ran a
    configuration the product does not - which shipped_config could not report
    until `ball` was resolved here, one day after being added to its KEYS. The
    artefacts already on disk were detected WITHOUT it and therefore diff
    OFF-CONFIG on this key alone; see score()'s VOID note.

    `conf` is the DETECTOR's confidence and None leaves track.run's own 0.5,
    which is what app/server/jobs.py runs. It is passed only when asked for, so no
    existing variant's command string moves. The cache is per-`tag` and that is
    load-bearing: `--dets-cache` keys on video and start frame, NOT on conf
    (model/engine/detectors.py, "ONE TRAP"), so two confs sharing one cache path is a
    conf sweep that reports the first conf's boxes twice. N17 asserts the box
    counts differ rather than trusting the path to be different.

    `stitch` and `cache_from` exist for N7: None and None leave every existing
    variant's command string exactly as it was. `cache_from` replays ANOTHER
    tag's detections, which is safe only since 051c7f2 - a cache records the
    settings its boxes were detected under and track.run refuses a mismatch.
    """
    out = _p(seq, f"{tag}.tracks.json")
    if os.path.exists(out):
        return out
    code = ("import sys; sys.path.insert(0, 'model/engine'); import track; "
            f"track.run({f'{CLIPS}/{seq}.mp4'!r}, {out!r}, calib={calib!r}, "
            f"detector='coreml', stride={STRIDE}, prior_gates={gates!r}, "
            + (f"conf={conf!r}, " if conf is not None else "")
            + (f"stitch={stitch!r}, " if stitch is not None else "")
            + f"ball={ball!r}, "
            + f"dets_cache={_p(seq, f'{cache_from or tag}.dets.cache.json')!r})")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    return out


def identify(seq, calib, src=".p05", dst=".p05id"):
    """jobs.apply_identity, the job's own identity stage."""
    out = _p(seq, f"{dst}.tracks.json")
    if os.path.exists(out):
        return out
    import identity  # noqa: PLC0415
    import jobs  # noqa: PLC0415
    with open(_p(seq, f"{src}.tracks.json")) as fh:
        tracks = json.load(fh)
    tracks, report = jobs.apply_identity(identity, tracks, f"{CLIPS}/{seq}.mp4",
                                         jobs.load_calibration(calib))
    tracks["_identity_report"] = report
    with open(out, "w") as fh:
        json.dump(tracks, fh)
    return out


def even_frames(seq):
    """`.c03s` on the stride-2 frames only -> {seq}.c03s_e2.tracks.json."""
    out = _p(seq, ".c03s_e2.tracks.json")
    with open(_p(seq, ".c03s.tracks.json")) as fh:
        d = json.load(fh)
    d["detections"] = [r for r in d["detections"] if r["f"] % STRIDE == 0]
    d["_config"] = dict(d.get("_config") or {}, derived=f"f % {STRIDE} == 0")
    with open(out, "w") as fh:
        json.dump(d, fh)


def calibrate_map(seq):
    """Variant B's calibration: ST2's chain as a `remap` calibration. -> (path, report)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import calibrators as C  # noqa: PLC0415
    from pitch import Calibration  # noqa: PLC0415
    raw = os.path.join(os.path.dirname(T.GT_SRC), "raw", seq)
    Hs = np.asarray(np.load(f"{raw}/{seq}_homography.npy"), np.float64)  # pitch -> canvas
    pitch = pitch_of(seq)
    mx, my = C.load_maps(raw)
    inner = lambda H: C.PerspectiveCalibrator(Calibration(pitch=pitch, H=H))  # noqa: E731
    band = C.pitch_band(C.RemapCalibrator(pitch, None, None, inner(np.linalg.inv(Hs))),
                        (mx.shape[1], mx.shape[0]))
    # The squash, measured over the pitch interior as _measure_real does.
    gx, gy = np.meshgrid(np.linspace(2, pitch.length - 2, 60),
                         np.linspace(2, pitch.width - 2, 30))
    q = cv2.perspectiveTransform(np.column_stack([gx.ravel(), gy.ravel()])
                                 .reshape(-1, 1, 2), Hs).reshape(-1, 2).astype(int)
    q = q[(q[:, 0] >= 0) & (q[:, 0] < mx.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < mx.shape[0])]
    scale = round(float(np.median(np.gradient(my, axis=0)[q[:, 1], q[:, 0]])), 1)
    # The crop+scale is an affine window (calibrators._measure_real, incl. the
    # +-0.5 for cv2.resize's pixel centres), so the cropped homography follows
    # from the vendor's exactly - no fit.
    A = np.array([[scale, 0.0, 0.5 * scale - 0.5],
                  [0.0, scale, (0.5 - band[0]) * scale - 0.5],
                  [0.0, 0.0, 1.0]])
    cal = C.RemapCalibrator(pitch, None, None, inner(np.linalg.inv(A @ Hs)),
                            maps=raw, crop=band, scale=scale)
    mpx = mx.shape[1] * scale * (band[1] - band[0]) * scale / 1e6
    if mpx > MAX_CANVAS_MPX:
        raise ValueError(f"{seq}: detection canvas {mpx:.0f} Mpx exceeds "
                         f"{MAX_CANVAS_MPX} (band {band}, x{scale}) - a compute limit")
    path = _p(seq, ".p05m.calibration.json")
    with open(path, "w") as fh:
        json.dump(cal.to_dict(), fh)
    return path, {"band": list(band), "scale": scale, "canvas_mpx": round(mpx, 1)}


def _canvas_to_clip(calib):
    """-> P(u, v) mapping canvas px to clip px through a remap calibration's
    own (cropped, scaled) tables: (x, y, inside) arrays, nearest table cell."""
    import numpy as np  # noqa: PLC0415
    import calibrators as C  # noqa: PLC0415
    with open(calib) as fh:
        c = json.load(fh)
    mx, my = C.load_maps(c["maps"], c.get("maps_yscale", 1.0), c.get("maps_crop"),
                         c.get("maps_scale", 1.0))
    h, w = mx.shape

    def P(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        ui = np.clip(np.round(u).astype(int), 0, w - 1)
        vi = np.clip(np.round(v).astype(int), 0, h - 1)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return mx[vi, ui], my[vi, ui], inside & (mx[vi, ui] >= 0) & (my[vi, ui] >= 0)
    return P


def to_panorama(src, dst, calib):
    """Canvas-space tracks -> clip pixels through the calibration's own tables.
    Feet are mapped as a point; w/h from the mapped box corners. -> share of
    rows dropped because their feet fall outside the table."""
    import numpy as np  # noqa: PLC0415
    P = _canvas_to_clip(calib)
    with open(src) as fh:
        d = json.load(fh)
    R = d["detections"]
    x, y = np.array([r["x"] for r in R]), np.array([r["y"] for r in R])
    bw, bh = np.array([r["w"] for r in R]), np.array([r["h"] for r in R])

    fx, fy, ok = P(x, y)
    cx = np.stack([P(x + sx * bw / 2, y - sy * bh)[0] for sx in (-1, 1) for sy in (0, 1)])
    cy = np.stack([P(x + sx * bw / 2, y - sy * bh)[1] for sx in (-1, 1) for sy in (0, 1)])
    out = []
    for i, r in enumerate(R):
        if ok[i]:
            out.append(dict(r, x=float(fx[i]), y=float(fy[i]),
                            w=float(cx[:, i].max() - cx[:, i].min()),
                            h=float(cy[:, i].max() - cy[:, i].min())))
    d["detections"] = out
    d["_to_panorama"] = {"rows": len(R), "dropped": len(R) - len(out)}
    with open(dst, "w") as fh:
        json.dump(d, fh)
    return (len(R) - len(out)) / max(len(R), 1)


def replay(seq):
    """N18: B's post-prior boxes, in clip pixels, replayed with A's calibration.
    -> share of boxes dropped by the mapping."""
    import numpy as np  # noqa: PLC0415
    out = _p(seq, ".p05r.tracks.json")
    cache = _p(seq, ".p05r.dets.cache.json")
    with open(_p(seq, ".p05m_canvas.dets.cache.json")) as fh:
        b = json.load(fh)
    assert b.get("complete"), f"{seq}: B's detection cache is not complete"
    P = _canvas_to_clip(_p(seq, ".p05m.calibration.json"))
    frames, n, kept = {}, 0, 0
    for f, dets in b["frames"].items():
        if not dets:
            frames[f] = []
            continue
        a = np.array([d["xyxy"] for d in dets], float)
        cx, cy, ok = zip(*(P(a[:, i], a[:, j]) for i, j in ((0, 1), (2, 1), (0, 3), (2, 3))))
        cx, cy, ok = np.stack(cx), np.stack(cy), np.all(np.stack(ok), axis=0)
        n += len(dets)
        frames[f] = [dict(d, xyxy=[float(cx[:, k].min()), float(cy[:, k].min()),
                                   float(cx[:, k].max()), float(cy[:, k].max())])
                     for k, d in enumerate(dets) if ok[k]]
        kept += len(frames[f])
    with open(cache, "w") as fh:
        # calib_sha None on purpose, and said so: these boxes passed B's prior,
        # not this calibration's, which is the experiment.
        json.dump({"video": f"{seq}.mp4", "start": 0, "calib": None,
                   "calib_sha": None, "complete": True, "frames": frames,
                   "note": "N18 replay: B's post-prior canvas boxes mapped to clip "
                           "pixels; deliberately NOT filtered by A's scene prior"}, fh)
    if not os.path.exists(out):
        track(seq, _p(seq, ".p05.calibration.json"), tag=".p05r")
    return (n - kept) / max(n, 1)


def run_replay(seqs):
    """N18 end to end on the sequences B ran. -> merged into p05_results.json."""
    done = [s for s in seqs if os.path.exists(_p(s, ".p05m_canvas.dets.cache.json"))]
    drops = {}
    for s in done:
        drops[s] = round(replay(s), 5)
        print(f"=== {s}: replay dropped {drops[s]}", flush=True)
    res = score(done, (".p05r",))
    out = os.path.join(T.WORK, "p05_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev["replay_dropped"] = drops
    prev.setdefault("scores", {}).update(res)
    with open(out, "w") as fh:
        json.dump(prev, fh, indent=1)
    _table(res)


def raw_detect(seq):
    """The job's detector on the raw frames with NO prior -> a complete cache of
    unfiltered boxes. Same call as track() except calib=None, which is exactly
    what leaves the prior out."""
    cache = _p(seq, ".p19raw.dets.cache.json")
    if os.path.exists(cache) and json.load(open(cache)).get("complete"):
        return cache
    code = ("import sys; sys.path.insert(0, 'model/engine'); import track; "
            f"track.run({f'{CLIPS}/{seq}.mp4'!r}, {_p(seq, '.p19raw.tracks.json')!r}, "
            f"detector='coreml', stride={STRIDE}, dets_cache={cache!r})")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    return cache


def _gt_boxes(seq):
    """{0-based even frame: (N,4) xyxy} of ST2's MOT ground truth."""
    import numpy as np  # noqa: PLC0415
    out = {}
    with open(f"{T.GT_SRC}/{seq}.txt") as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 6 or int(p[0]) % STRIDE:
                continue
            x, y, w, h = (float(v) for v in p[2:6])
            out.setdefault(int(p[0]), []).append((x, y, x + w, y + h))
    return {f: np.array(b) for f, b in out.items()}


def _box_pr(frames, gt):
    """Box-level (recall, precision) over the frames: greedy IoU >= 0.5."""
    import numpy as np  # noqa: PLC0415
    tp = nd = ng = 0
    for f, g in gt.items():
        d = np.array([b["xyxy"] for b in frames.get(str(f), [])], float).reshape(-1, 4)
        ng, nd = ng + len(g), nd + len(d)
        if not len(d):
            continue
        ix = np.maximum(0, np.minimum(d[:, None, 2], g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0]))
        iy = np.maximum(0, np.minimum(d[:, None, 3], g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1]))
        inter = ix * iy
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
        iou = inter / (area(d)[:, None] + area(g)[None, :] - inter + 1e-9)
        used_d, used_g = set(), set()
        for k in np.argsort(-iou, axis=None):
            i, j = divmod(int(k), iou.shape[1])
            if iou[i, j] < 0.5:
                break
            if i not in used_d and j not in used_g:
                used_d.add(i)
                used_g.add(j)
        tp += len(used_d)
    return tp / max(ng, 1), tp / max(nd, 1)


def n19_arms(seq):
    """Each arm: the raw cache through make_prior(gates) from A's calibration,
    written as its own cache and replayed. -> {arm: box-level stats}."""
    import numpy as np  # noqa: PLC0415
    import jobs  # noqa: PLC0415
    from track import make_prior  # noqa: PLC0415
    raw = json.load(open(raw_detect(seq)))
    calib = _p(seq, ".p05.calibration.json")
    cal = jobs.load_calibration(calib)
    wh = T._wh(seq)
    gt = _gt_boxes(seq)
    stats = {"raw": dict(zip(("box_recall", "box_precision"), _box_pr(raw["frames"], gt)))}
    n_raw = sum(len(v) for v in raw["frames"].values())
    for arm, gates in N19.items():
        keep = make_prior(cal, wh, gates) if gates else None
        frames = {}
        for f, dets in raw["frames"].items():
            if keep is None or not dets:
                frames[f] = dets
                continue
            m = keep(np.array([d["xyxy"] for d in dets], float))
            frames[f] = [d for d, k in zip(dets, m) if k]
        kept = sum(len(v) for v in frames.values())
        with open(_p(seq, f"{arm}.dets.cache.json"), "w") as fh:
            json.dump({"video": f"{seq}.mp4", "start": 0, "calib": None,
                       "calib_sha": None, "complete": True, "frames": frames,
                       "note": f"N19 arm {arm}: raw boxes, prior gates {gates} "
                               "from A's calibration, applied offline"}, fh)
        r, p = _box_pr(frames, gt)
        stats[arm] = {"gates": list(gates), "removed": round(1 - kept / max(n_raw, 1), 5),
                      "box_recall": r, "box_precision": p}
        if not os.path.exists(_p(seq, f"{arm}.tracks.json")):
            track(seq, calib, tag=arm)
    return stats


def raw_detect_canvas(seq):
    """The job's detector on B's flattened canvas with NO gate: B's calibration
    (a remap needs it to flatten at all) and prior_gates=()."""
    cache = _p(seq, ".p20raw.dets.cache.json")
    if os.path.exists(cache) and json.load(open(cache)).get("complete"):
        return cache
    code = ("import sys; sys.path.insert(0, 'model/engine'); import track; "
            f"track.run({f'{CLIPS}/{seq}.mp4'!r}, {_p(seq, '.p20raw.tracks.json')!r}, "
            f"calib={_p(seq, '.p05m.calibration.json')!r}, detector='coreml', "
            f"stride={STRIDE}, prior_gates=(), dets_cache={cache!r})")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    return cache


def n20_arms(seq):
    """N19's arms on B's calibration and canvas. -> {arm: removed share}."""
    import numpy as np  # noqa: PLC0415
    import calibrators as C  # noqa: PLC0415
    import jobs  # noqa: PLC0415
    from track import make_prior  # noqa: PLC0415
    raw = json.load(open(raw_detect_canvas(seq)))
    calib = _p(seq, ".p05m.calibration.json")
    cal = jobs.load_calibration(calib)
    with open(calib) as fh:
        c = json.load(fh)
    mx, _ = C.load_maps(c["maps"], c.get("maps_yscale", 1.0), c.get("maps_crop"),
                        c.get("maps_scale", 1.0))
    wh = (mx.shape[1], mx.shape[0])          # the CANVAS, not the clip
    n_raw = sum(len(v) for v in raw["frames"].values())
    out = {}
    for arm, gates in N20.items():
        keep = make_prior(cal, wh, gates) if gates else None
        frames = {}
        for f, dets in raw["frames"].items():
            if keep is None or not dets:
                frames[f] = dets
                continue
            m = keep(np.array([d["xyxy"] for d in dets], float))
            frames[f] = [d for d, k in zip(dets, m) if k]
        out[arm] = round(1 - sum(len(v) for v in frames.values()) / max(n_raw, 1), 5)
        with open(_p(seq, f"{arm}_canvas.dets.cache.json"), "w") as fh:
            json.dump({"video": f"{seq}.mp4", "start": 0, "calib": None,
                       "calib_sha": None, "complete": True, "frames": frames,
                       "note": f"N20 arm {arm}: canvas boxes, prior gates {gates} "
                               "from B's calibration, applied offline"}, fh)
        if not os.path.exists(_p(seq, f"{arm}.tracks.json")):
            track(seq, calib, tag=f"{arm}_canvas")
            to_panorama(_p(seq, f"{arm}_canvas.tracks.json"),
                        _p(seq, f"{arm}.tracks.json"), calib)
    return out


def run_n20(seqs):
    """N20 end to end, applying the rule as written in the docstring."""
    # B's completed detection cache, not its calibration FILE: a calibration
    # written before the canvas cap existed is still on disk for 117092, whose
    # 162 Mpx canvas is exactly what the cap excludes - and picking by the file
    # started N20 on it at 0.3 fps.
    done = [s for s in seqs if os.path.exists(_p(s, ".p05m_canvas.dets.cache.json"))]
    removed = {}
    print(f"N20 on {done}", flush=True)
    for s in done:
        removed[s] = n20_arms(s)
        print(f"=== {s}: removed {json.dumps(removed[s])}", flush=True)
    res = score(done, tuple(N20) + (".p05m",))
    common = [s for s in done if s in COMMON7]
    paired = {f"v0_vs_{a[-2:]}": {k: _paired(res, ".p20v0", a, k, common)
                                  for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr")}
              for a in (".p20v1", ".p20v2", ".p20v3")}
    paired["v0_vs_B"] = {k: _paired(res, ".p20v0", ".p05m", k, common)
                         for k in ("HOTA", "DetRe", "DetPr")}
    earns = paired["v0_vs_v2"]["HOTA"]["wins"] >= 6
    out = os.path.join(T.WORK, "p05_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev["n20"] = {"removed": removed, "paired": paired,
                   "height_gate_earns_its_keep_on_a_good_calibration": earns,
                   "rule": "v0 beats v2 on HOTA on >=6 of 7 (st2_product docstring)"}
    prev.setdefault("scores", {}).update(res)
    with open(out, "w") as fh:
        json.dump(prev, fh, indent=1)
    _table({k: v for k, v in res.items() if k in N20})
    print(json.dumps(prev["n20"] | {"removed": "see file"}, indent=1))


def _paired(res, a, b, key, seqs):
    """mean, 95% t half-width, wins of (a - b) over seqs."""
    import math  # noqa: PLC0415
    d = [res[a][s][key] - res[b][s][key] for s in seqs]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
    t = {6: 2.447, 7: 2.365, 8: 2.306}.get(len(d) - 1, 2.0)
    return {"mean": m, "ci95": t * sd / math.sqrt(len(d)), "wins": sum(x > 0 for x in d),
            "n": len(d)}


def run_n19(seqs):
    """N19 end to end on the sequences B ran; the pass rule is applied as
    written in model-map (not published) §3 N19, not re-chosen after the numbers."""
    done = [s for s in seqs if os.path.exists(_p(s, ".p05m_canvas.dets.cache.json"))]
    box = {}
    for s in done:
        box[s] = n19_arms(s)
        print(f"=== {s}: {json.dumps({k: {q: round(v, 4) if isinstance(v, float) else v for q, v in d.items()} for k, d in box[s].items()})}", flush=True)
    res = score(done, tuple(N19) + (".p05",))
    common = [s for s in done if s in COMMON7]
    paired = {arm: {k: _paired(res, arm, ".p19v0", k, common)
                    for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr")}
              for arm in (".p19v1", ".p19v2", ".p19v3")}
    paired["v0_vs_A"] = {k: _paired(res, ".p19v0", ".p05", k, common)
                         for k in ("HOTA", "DetRe", "DetPr")}
    tol = paired["v0_vs_A"]["DetPr"]["ci95"]
    verdict = {arm: ("BETTER" if paired[arm]["HOTA"]["wins"] >= 6
                     and paired[arm]["DetPr"]["mean"] >= -tol else "NOT BETTER")
               for arm in (".p19v1", ".p19v2", ".p19v3")}
    out = os.path.join(T.WORK, "p05_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev["n19"] = {"box": box, "paired_vs_v0": paired, "verdict": verdict,
                   "rule": "HOTA up on >=6 of 7 AND mean DetPr drop no larger "
                           "than the V0-vs-A DetPr CI (model-map (not published) N19)"}
    prev.setdefault("scores", {}).update(res)
    with open(out, "w") as fh:
        json.dump(prev, fh, indent=1)
    _table({k: v for k, v in res.items() if k in N19})
    print(json.dumps({"paired_vs_v0": paired, "verdict": verdict}, indent=1))


COMMON7 = ("117093", "118575", "118576", "118577", "118578", "128058", "132877")
#: N17's population: the official val split (model/validation/st2_eval.py SPLIT).
VAL = ("117093", "132877")
N17_CONF = 0.3


def _cache_boxes(seq, tag):
    """Boxes in a variant's detection cache. The conf arms MUST differ here."""
    c = json.load(open(_p(seq, f"{tag}.dets.cache.json")))
    return sum(len(v) for v in c["frames"].values())


def _n17_verdict(hota, contam, conf=N17_CONF):
    """The pass rule from the docstring, in code. `hota` and `contam` are
    {seq: (at 0.5, at conf)}. Agreement over sequences, not an interval: n=2.

    Returns (move?, verdict). A split is KEEP and says which way each went.
    """
    up = {s: b > a for s, (a, b) in hota.items()}
    dirty = {s: b > a for s, (a, b) in contam.items()}
    if all(up.values()) and not any(dirty.values()):
        return True, f"MOVE the product to conf {conf}"
    why = []
    if not all(up.values()):
        why.append("HOTA falls on " + ",".join(s for s, v in up.items() if not v)
                   + (" (a SPLIT - two sequences cannot decide this)"
                      if any(up.values()) else ""))
    if any(dirty.values()):
        why.append("contamination rises on "
                   + ",".join(s for s, v in dirty.items() if v))
    return False, "KEEP conf 0.5 - " + "; ".join(why)


def run_n17(seqs=VAL, conf=N17_CONF, tag=".p03"):
    """N17. One fresh detection pass per val sequence at `conf`, then the job's
    identity stage, against N6's own `.p05` / `.p05id` at conf 0.5."""
    import contam_eval  # noqa: PLC0415
    boxes = {}
    for s in seqs:
        calib = _p(s, ".p05.calibration.json")
        track(s, calib, tag=tag, conf=conf)
        identify(s, calib, src=tag, dst=f"{tag}id")
        boxes[s] = {".p05": _cache_boxes(s, ".p05"), tag: _cache_boxes(s, tag)}
        # THE TRAP, asserted rather than hoped for: --dets-cache keys on video
        # and start frame and not on conf, so a replayed cache is a conf sweep
        # that measures nothing. Fewer-or-equal boxes at a LOWER conf is that.
        assert boxes[s][tag] > boxes[s][".p05"], (
            f"{s}: conf {conf} produced {boxes[s][tag]} boxes against conf 0.5's "
            f"{boxes[s]['.p05']} - the detection cache was replayed, not re-run; "
            f"this run measures nothing")
        print(f"=== {s}: boxes {boxes[s]['.p05']} at 0.5 -> {boxes[s][tag]} at {conf}",
              flush=True)
    res = score(seqs, (".p05", tag, ".p05id", f"{tag}id"))
    # score_clip names the variant without the leading dot, and it carries its
    # own drop share (unattributed_frac, gt_recall) on every row.
    cont = {v: {s: contam_eval.score_clip(s, v[1:]) for s in seqs}
            for v in (".p05id", f"{tag}id")}
    hota = {s: (res[".p05"][s]["HOTA"], res[tag][s]["HOTA"]) for s in seqs}
    dirt = {s: (cont[".p05id"][s]["contaminated"], cont[f"{tag}id"][s]["contaminated"])
            for s in seqs}
    move, verdict = _n17_verdict(hota, dirt, conf)
    out = os.path.join(T.WORK, "p05_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev["n17"] = {"conf": conf, "seqs": list(seqs), "boxes": boxes,
                   "hota_0.5_then_conf": hota, "contaminated_0.5_then_conf": dirt,
                   "contamination": cont, "move_the_product": move,
                   "verdict": verdict, "n_configurations": 2,
                   "rule": "HOTA up on BOTH val sequences AND contamination up "
                           "on neither (st2_product docstring N17)"}
    prev.setdefault("scores", {}).update(res)
    with open(out, "w") as fh:
        json.dump(prev, fh, indent=1)
    _table({k: v for k, v in res.items() if k in (".p05", tag, ".p05id", f"{tag}id")})
    for s in seqs:
        a, b = cont[".p05id"][s], cont[f"{tag}id"][s]
        print(f"{s}  HOTA {hota[s][0]:.4f} -> {hota[s][1]:.4f}   "
              f"contaminated {a['contaminated']:.1f}% -> {b['contaminated']:.1f}%   "
              f"unattributed {a['unattributed_frac']:.3f} -> {b['unattributed_frac']:.3f}   "
              f"gt_recall {a['gt_recall']:.3f} -> {b['gt_recall']:.3f}   "
              f"ids {a['ids']} -> {b['ids']}")
    print(f"VERDICT: {verdict}  (n=2 configurations, n=2 sequences - an "
          f"agreement over sequences, not an interval)")


def run_n21(seqs=ALL10, tag=".p21"):
    """N21: the product as it ships today, on every sequence N6 could click.
    Protocol in the module docstring. Resumes: track() skips a finished file
    and its cache resumes a partial one."""
    import shipped_config  # noqa: PLC0415
    done, skipped = [], {}
    for s in seqs:
        calib = _p(s, ".p05.calibration.json")
        if not os.path.exists(calib):
            skipped[s] = "no N6 calibration - its ST2 chain cannot be clicked"
            print(f"=== {s}: EXCLUDED - {skipped[s]}", flush=True)
            continue
        try:
            track(s, calib, tag=tag)
            identify(s, calib, src=tag, dst=f"{tag}id")
        except Exception as e:  # noqa: BLE001 - record the sequence, run the rest
            skipped[s] = f"FAILED {type(e).__name__}: {e}"
            print(f"=== {s}: {skipped[s]}", flush=True)
            continue
        done.append(s)
    # VOID IF, before a single score. UNSTAMPED is not ON, so a file rebuilt from
    # a cache that predates recorded settings voids the run too.
    stamps = {s: shipped_config.config_of(_p(s, f"{tag}.tracks.json")) for s in done}
    for s, stamp in stamps.items():
        state, diff = shipped_config.diff_artefact(stamp)
        assert state == "ON", f"{s}: {tag} is {state} against what ships {diff} - N21 is VOID"
    res = score(done, (".p05", tag, ".p05id", f"{tag}id"))
    perf = {s: {k: json.load(open(_p(s, f"{tag}.tracks.json")))["_perf"].get(k)
                for k in ("seconds", "throughput_fps", "prior_rejected")} for s in done}
    with open(os.path.join(T.WORK, "n21_results.json"), "w") as fh:
        json.dump({"seqs": done, "skipped": skipped, "stamps": stamps, "perf": perf,
                   "scores": res, "protocol": "model/validation/st2_product.py docstring N21"},
                  fh, indent=1)
    _table(res)


def run_n7(seqs=ALL10, base=".p21"):
    """N7: the metric stitch on and off, replayed off N21's own detection cache.
    Protocol in the module docstring."""
    import shipped_config  # noqa: PLC0415
    import st2_ab  # noqa: PLC0415
    done = [s for s in seqs if os.path.exists(_p(s, f"{base}.dets.cache.json"))]
    on, off = f"{base}s", f"{base}ns"
    for s in done:
        calib = _p(s, ".p05.calibration.json")
        track(s, calib, tag=on, stitch=True, cache_from=base)
        track(s, calib, tag=off, stitch=False, cache_from=base)
        # VOID IF, per sequence and before any score.
        same = (json.load(open(_p(s, f"{base}.tracks.json")))["detections"]
                == json.load(open(_p(s, f"{on}.tracks.json")))["detections"])
        assert same, f"{s}: the stitched replay does not reproduce {base} - N7 is VOID"
        verdict, why = shipped_config.compare_guard(
            shipped_config.config_of(_p(s, f"{on}.tracks.json")),
            shipped_config.config_of(_p(s, f"{off}.tracks.json")), {"stitch"})
        assert verdict == "OK", f"{s}: arms are {verdict} {why} - N7 is VOID"
    res = score(done, (off, on))
    held = [s for s in done if s not in TUNED]
    idsw = {v: sum(res[v][s]["IDSW"] for s in held) / len(held) for v in (off, on)}
    ab_ships = bool(st2_ab.ab(off, on, "AssA"))
    ships = ab_ships and idsw[on] <= idsw[off]
    with open(os.path.join(T.WORK, "n7_results.json"), "w") as fh:
        json.dump({"seqs": done, "arms": {"stitch": on, "no_stitch": off}, "scores": res,
                   "held_out_mean_idsw": idsw, "st2_ab_ships": ab_ships,
                   "verdict": "SHIPS" if ships else "DOES NOT SHIP",
                   "rule": "st2_ab AssA sign test SHIPS and held-out IDSW does not rise "
                           "(st2_product docstring N7)"}, fh, indent=1)
    _table(res)
    print(f"held-out mean IDSW: no stitch {idsw[off]:.1f} -> stitch {idsw[on]:.1f}")
    print(f"N7 VERDICT: {'SHIPS' if ships else 'DOES NOT SHIP'} "
          f"(st2_ab {'passes' if ab_ships else 'does not pass'}; n=2 configurations)")

def run_map(seqs):
    """Variant B end to end. -> results merged into p05_results.json."""
    cals, done = {}, []
    for s in seqs:
        try:
            path, rep = calibrate_map(s)
        except Exception as e:  # noqa: BLE001 - report the sequence, run the rest
            cals[s] = {"excluded": f"{type(e).__name__}: {e}"}
            print(f"=== {s}: EXCLUDED (map) - {e}", flush=True)
            continue
        print(f"=== {s}: map calibration {rep}", flush=True)
        track(s, path, tag=".p05m_canvas")
        identify(s, path, src=".p05m_canvas", dst=".p05mid_canvas")
        rep["dropped"] = {v: round(to_panorama(_p(s, f"{v}_canvas.tracks.json"),
                                               _p(s, f"{v}.tracks.json"), path), 5)
                          for v in (".p05m", ".p05mid")}
        cals[s] = rep
        done.append(s)
    res = score(done)
    out = os.path.join(T.WORK, "p05_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev["calibration_map"] = cals
    prev.setdefault("scores", {}).update({v: res[v] for v in (".p05m", ".p05mid") if v in res})
    with open(out, "w") as fh:
        json.dump(prev, fh, indent=1)
    _table({v: res[v] for v in (".p05m", ".p05mid") if v in res})


def score(seqs, variants=VARIANTS):
    res = {}
    for v in variants:
        have = [s for s in seqs if os.path.exists(_p(s, f"{v}.tracks.json"))]
        if not have:
            continue
        T.VARIANT = v
        T.stage_gt(have, STRIDE)
        T.stage_pred(have)
        r = T.run(have)
        keys = ("HOTA", "DetA", "AssA", "IDSW", "DetRe", "DetPr")
        res[v] = {s: {k: r[s][k] for k in keys if k in r[s]} for s in have}
    return res


def _table(res):
    for v, per in res.items():
        for name, grp in (("held out", [s for s in per if s not in TUNED]),
                          ("tuned", [s for s in per if s in TUNED])):
            if grp:
                m = {k: sum(per[s][k] for s in grp) / len(grp)
                     for k in ("HOTA", "DetA", "AssA", "IDSW")}
                print(f"{v:10s} {name:9s} n={len(grp):2d}  HOTA {m['HOTA']:.4f}  "
                      f"DetA {m['DetA']:.4f}  AssA {m['AssA']:.4f}  IDSW {m['IDSW']:.1f}")


def run(seqs):
    cals, done = {}, []
    for s in seqs:
        try:
            path, rep = calibrate(s)
        except (ValueError, RuntimeError) as e:
            cals[s] = {"excluded": str(e)}
            print(f"=== {s}: EXCLUDED - {e}", flush=True)
            continue
        cals[s] = rep
        print(f"=== {s}: calibration {rep}", flush=True)
        track(s, path)
        identify(s, path)
        even_frames(s)
        done.append(s)
    res = score(done)
    with open(os.path.join(T.WORK, "p05_results.json"), "w") as fh:
        json.dump({"calibration": cals, "scores": res,
                   "protocol": "model/validation/st2_product.py docstring"}, fh, indent=1)
    _table(res)


def demo():
    """Calibration only, on every sequence: at least four clicks, a finite fit,
    and a file calibrators accepts. Seconds, no detector."""
    import tempfile  # noqa: PLC0415
    import jobs  # noqa: PLC0415
    tmp = tempfile.mkdtemp(prefix="st2prod-")
    for s in ALL10:
        try:
            path, rep = calibrate(s, tmp)
        except ValueError as e:
            assert s == "132831", e     # the one known degenerate chain
            print(f"  {s}: excluded - {e}")
            continue
        assert rep["clicked"] >= 4, (s, rep)
        jobs.load_calibration(path)
        print(f"  {s}: {rep}")

    # N17's pass rule, both directions, because a rule that can only say one of
    # them is not a rule. HOTA and contamination as (at 0.5, at 0.3) pairs.
    up = {"117093": (0.40, 0.42), "132877": (0.35, 0.37)}
    clean = {"117093": (30.0, 29.0), "132877": (31.0, 31.0)}
    assert _n17_verdict(up, clean)[0], _n17_verdict(up, clean)
    # one sequence going the other way is a SPLIT, and a split is KEEP
    split = {"117093": (0.40, 0.42), "132877": (0.35, 0.34)}
    m, why = _n17_verdict(split, clean)
    assert not m and "SPLIT" in why, why
    # ...and a HOTA win bought with contamination is KEEP too
    m, why = _n17_verdict(up, {"117093": (30.0, 33.0), "132877": (31.0, 31.0)})
    assert not m and "contamination rises on 117093" in why, why
    print("st2_product demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--n21" in sys.argv:
        run_n21()
    elif "--n7" in sys.argv:
        run_n7()
    elif "--n17" in sys.argv:
        # The val split only, which is N17's declared population.
        run_n17()
    elif "--n20" in sys.argv:
        # The 7 held out, not all 10: the paired rule reads those, and the
        # tuned sequence costs another canvas detection pass for nothing.
        run_n20(list(COMMON7))
    elif "--n19" in sys.argv:
        run_n19(list(ALL10))
    elif "--replay" in sys.argv:
        run_replay(list(ALL10))
    elif "--map" in sys.argv:
        run_map(list(ALL10))
    elif "--score" in sys.argv:
        _table(score(list(ALL10)))
    else:
        run(list(ALL10))
