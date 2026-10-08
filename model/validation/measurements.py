"""Every measurement this repo publishes. The REGISTRY, not the mechanism.

`api.py` owns the contract - scope and denominator mandatory, failures that stay
failures, a cache keyed on inputs AND producing code, a golden snapshot that only
moves with a written reason. This file owns the LIST.

They are separate files because they change for different reasons and by
different hands: adding a measurement is routine, changing the contract is not.
Splitting them also means two people can add measurements and audit the
mechanism at the same time without fighting over one file.

TO ADD A MEASUREMENT
    1. Write the producer as a thin call into the scorer that ALREADY owns the
       logic. If there is no scorer, write the scorer first - this file must not
       become a second place where a number is computed.
    2. Declare `inputs` (the data it reads) and `code` (the modules that produce
       it). Both are required. A measurement with empty inputs hashes to sha256
       of nothing, which is a constant, so it can never invalidate - four of the
       first twelve did exactly that.
    3. Give it an `n`. A rate without its denominator is how a subpopulation
       figure gets compared to a full-population one (postmortem (not published) B2).
    4. Run `api.py --verify`. If anything ELSE moved, you changed more than you
       meant to. Then `--snapshot "reason"`.

A COUNTERFACTUAL - an oracle, a ceiling, a control - is a measurement here too,
and the argument for that (and against it) is written out above the
`calib.oracle.*` block, with the fencing it is registered behind.
"""
import functools
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL10 = ("117092", "117093", "118575", "118576", "118577", "118578",
         "128057", "128058", "132831", "132877")
TUNED = ("128057", "132831")
HELDOUT = tuple(s for s in ALL10 if s not in TUNED)


@dataclass
class Spec:
    """A measurement's definition. Nothing here is optional, on purpose.

    `n` lives HERE, not in a parallel dict keyed by id - that was the copy step
    this whole layer exists to remove, one level down, and an audit caught it the
    day it was written.

    `inputs` and `code` are both required and both non-empty. A measurement that
    declares no inputs hashes to sha256 of nothing, which is a CONSTANT, so it
    can never invalidate; four of the first twelve did exactly that, which is
    postmortem (not published) B4 rebuilt inside the fix for B4.
    """
    desc: str
    unit: str
    scope: str
    n: int
    fn: Callable
    inputs: tuple
    code: tuple = ("model/validation/api.py",)


@functools.cache
def _mot(variant, over=ALL10, every=1):
    """TrackEval over `over`, once per (variant, set, GT stride) per process.

    Five ids read `.c03s` and each used to stage and score its own subset, so
    one verify ran TrackEval five times. Per-sequence HOTA does not depend on
    which other sequences were scored beside it, so subsets are read off one run.
    `every` is stage_gt's: 2 scores a stride-2 tracker on its own frames.
    """
    import st2_mot as T
    T.VARIANT = variant
    T.stage_gt(list(over), every)
    T.stage_pred(list(over), every)
    return T.run(list(over))


def _mean(variant, seqs, key, over=ALL10, every=1):
    r = _mot(variant, over, every)
    return sum(r[s][key] for s in seqs) / len(seqs)


def _ab(a, b, metric):
    """(mean delta, k improved, n moved, p, verdict) - the shipping decision."""
    import st2_ab as A
    import st2_coalesce as C
    seqs = [s for s in A.available(a) if s in A.available(b)]
    RA, RB = A.score(a, seqs), A.score(b, seqs)
    d = [RB[s][metric] - RA[s][metric] for s in seqs]
    ties = sum(1 for x in d if abs(x) < A.TIE)
    wins = sum(1 for x in d if x > A.TIE)
    n = len(d) - ties
    p = A.sign_p(wins, n)
    mean = sum(d) / len(d)
    ca = sum(C.score(s, a)["dup_per_frame"] for s in seqs) / len(seqs)
    cb = sum(C.score(s, b)["dup_per_frame"] for s in seqs) / len(seqs)
    coalesced = cb > max(2 * ca, 0.01)
    ships = p < 0.05 and mean > 0 and not coalesced
    return {"mean": mean, "k": wins, "n": n, "p": p,
            "verdict": "SHIPS" if ships else "DOES NOT SHIP"}, len(seqs)


# IN-PROCESS, both of them. They used to shell out to the script, and a
# subprocess's imports are invisible to api._deps - which is how model/engine/pitch.py
# came to sit under team accuracy with no key covering it. Same functions, same
# defaults as the scripts' CLIs (team_eval: method "shipped"; st2_metres: every
# 300, smoothing TPS_SMOOTHING).
@functools.cache
def _team_eval():
    import team_eval
    return team_eval.evaluate()


def _team(key):
    return float(_team_eval()[key])


@functools.cache
def _metres_out(seq, tps=False):
    import contextlib
    import io
    import st2_metres
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st2_metres.run(seq, tps=tps)
    return buf.getvalue()


def _metres(seq, band, tps=False):
    out = _metres_out(seq, tps)
    m = re.search(rf"{band}\s+median\s+([0-9.]+) m", out)
    if not m:
        raise RuntimeError(f"st2_metres printed no {band} for {seq}")
    return float(m.group(1))


# --------------------------------------------------------------------------
# PRODUCT PRODUCERS. Every one is `lru_cache` over a single call into the
# scorer that owns the logic, so the four distance rows below cost ONE run of
# dist_eval rather than four. None of them computes anything: if a number is not
# a key in the scorer's own return value, it does not belong here.
#
# The scorers were changed to RETURN their tables rather than only print them.
# `_metres` above still regex-scrapes st2_metres' printed table (in-process,
# via redirect_stdout) and that is the pattern being retired, not extended - a
# tidied print format silently stops a scraped number resolving, and nothing
# fails until someone reads the doc.
# --------------------------------------------------------------------------
from functools import lru_cache  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

JOB = "data/jobs/118575_full_1st"
SPLIT_TRACKS = f"{JOB}/tracks.split.json"
GT_118575 = "model/validation/soccertrack/118575_1st.tracks.json.gz"
CALIB_118575 = f"{JOB}/calibration.json"


@lru_cache(maxsize=None)
def _dist(tracks="tracks.split.json"):
    import dist_eval
    return dist_eval.evaluate(tracks)


@lru_cache(maxsize=None)
def _dist_stc():
    """N10/N11: the same estimator on the PRODUCT config's own tracks.

    `_dist` above scores one job directory whose tracks file predates the
    current tracker. This scores the nine STC2025 clips `app/server/jobs.py`'s own
    pipeline produced, against ST2's dense positional feed. Same code, same
    estimator, different population - which is the only thing that changed and
    the only reason the values below move.
    """
    import dist_eval
    # variant p21, not the function's p05 default: a caller's explicit argument is
    # the only thing that holds when the callee's default moves. The clips too: STC8,
    # not contam_eval.STC_SEQS, whose nine include 132877's mirrored feed.
    return dist_eval.evaluate_stc_all(STC8, variant="p21id", verbose=False)


@lru_cache(maxsize=None)
def _poss(t0=300.0, t1=1800.0):
    import possession_eval
    return possession_eval.score(t0, t1)


@lru_cache(maxsize=None)
def _floors(split="val"):
    import st2_eval
    return st2_eval.floors(split)


@lru_cache(maxsize=None)
def _det(name="base r768 t1024", conf=0.5, prior=True):
    sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
    import detsweep
    # detsweep keeps its own box cache in /tmp, keyed on config name and frames
    # but not on weights, detector or remap. With it on, a moved key re-ran the
    # producer and it replayed the OLD boxes under the new key. Off: when the
    # key moves, the detector actually runs.
    detsweep.BOX_CACHE = None
    return detsweep.run(name, conf, prior)


@lru_cache(maxsize=None)
def _ball():
    import ball_eval
    return ball_eval.score_m0(os.path.join(ROOT, BALL_M0))


@lru_cache(maxsize=None)
def _touch():
    import touch_eval
    return touch_eval.gt_isolate()


@lru_cache(maxsize=None)
def _contam(tracks="tracks.split.json"):
    import contam_eval
    return contam_eval.score(tracks)


# The two gaps. These are producers like any other - they call the scorer and
# the scorer refuses, so `api.py` records FAILED with the real reason rather
# than a plausible number. A gap that reports nothing is invisible; a gap that
# reports FAILED is a work item, and that is the whole point of registering
# them. Neither is a placeholder: both go green on their own the day the thing
# they are waiting for lands.
def _poss_pipeline():
    import possession_eval
    return possession_eval.score_pipeline()


def _event_model():
    import st2_eval
    return st2_eval.model_map()


TRACKS = tuple(f"model/validation/out/stc2025/{s}.c03s.tracks.json" for s in ALL10)
RAW = tuple(f"model/validation/out/stc2025/{s}.c03.tracks.json" for s in ALL10)
GATED = tuple(f"model/validation/out/stc2025/{s}.c03s_op.tracks.json" for s in ALL10)

# THE DATA IN THE KEY. Every path below is read by a producer and, until
# 2026-09-10, was missing from its key: new ground truth, new weights or an
# upgraded scorer served the old number as fresh. The gate resolves from the
# cache now, so a read that is missing here is a FALSE GREEN, not a slow run -
# add the file here in the same change that makes a producer read it.
from st2_mot import GT_SRC  # noqa: E402 - ST2 MOT ground truth, outside the repo

ST2_GT = tuple(f"{GT_SRC}/{s}.txt" for s in ALL10)
ST2_RAW_118575 = os.path.join(os.path.dirname(GT_SRC), "raw", "118575")


def _pkg(name):
    """An installed package's source directory. A scorer upgraded under us
    (trackeval) or a detector library bumped (rfdetr) moves the number, so it
    is an input. A missing package keys as a constant string and fails loudly
    in the producer, which is where a missing package should fail."""
    spec = importlib.util.find_spec(name)
    return spec.submodule_search_locations[0] if spec else f"MISSING-PACKAGE:{name}"


PKG_TRACKEVAL = (_pkg("trackeval"),)


def _coreml_inputs():
    """What the ANE worker reads, from the paths detectors.py itself resolves -
    so FT_COREML_MODEL / FT_COREML_PYTHON are honoured instead of a hard-coded
    path that watches files nobody used. The worker is a subprocess, so
    api._deps cannot see its imports: its model, script, interpreter and
    coremltools version are declared. A later coremltools lands in a new
    dist-info directory, which the glob finds, so an upgrade moves the key."""
    import glob
    sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
    import detectors
    py = detectors.COREML_PYTHON
    ct = glob.glob(os.path.join(os.path.dirname(os.path.dirname(py)), "lib", "python3*",
                                "site-packages", "coremltools-*.dist-info", "METADATA"))
    return (detectors.COREML_MODEL, "model/engine/coreml_worker.py", py) + tuple(sorted(ct))


IN_MOT = TRACKS + ST2_GT + PKG_TRACKEVAL

# N6 (model/validation/st2_product.py): the PRODUCT config, scored for the first time.
# COMMON7 is the held-out set both of its calibrations could run: 132831's ST2
# chain is degenerate and 117092's remap canvas is a compute limit, both
# excluded with reasons before running. Every row is on the stride-2 frames.
COMMON7 = ("117093", "118575", "118576", "118577", "118578", "128058", "132877")
IN_N6 = (tuple(f"model/validation/out/stc2025/{s}{v}.tracks.json" for s in COMMON7
               for v in (".p05", ".p05m", ".c03s_e2"))
         + tuple(f"model/validation/out/stc2025/{s}{v}.calibration.json" for s in COMMON7
                 for v in (".p05", ".p05m"))
         + tuple(f"{GT_SRC}/{s}.txt" for s in COMMON7) + PKG_TRACKEVAL)
# N10/N11 (2026-09-12): the distance and speed rows on the product config.
# Nine clips, not COMMON7's seven - those two exclusions are properties of the
# TRACKING benchmark (a degenerate ST2 chain, a 162 Mpx canvas) and neither
# stops a distance being scored, so widening here is honest rather than
# convenient. 132831 is still out: its chain cannot produce a clicked
# calibration at all.
# 2026-09-29: 132877 DROPPED (author decision). Its joined position feed is mirrored
# against ST2's own camera (st2-132877-feed-is-mirrored (internal note)), and
# dist_eval.stc_gt refuses mirrored feeds except to reproduce results registered on
# them - which is all these ids were doing. STC8 is N27's population
# (dist_eval.N27_SEQS): 1.2333 over 197 players on nine clips became 1.2278 over 175.
STC8 = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")
IN_STC_DIST = (tuple(f"model/validation/out/stc2025/{s}.p21id.tracks.json" for s in STC8)
               + tuple(f"model/validation/out/stc2025/{s}.p05.calibration.json" for s in STC8)
               + tuple(f"{GT_SRC}/{s}.txt" for s in STC8)
               # dist_eval.stc_gt: the join, its cache, and the vendor XML it scans on a miss
               + ("model/validation/out/st2_team_gt.json", "model/validation/out/n10_stc_gt.json")
               + tuple(f"{os.path.dirname(GT_SRC)}/raw/{s}/{s}_tracker_box_data.xml" for s in STC8))
# Dated rather than described, for N6's reason below: the prior moved under it.
# `_dist_stc` scores N21's .p21id tracks (model/validation/st2_product.py, 2026-09-13):
# the product as it ships since c6b8fb8 and 8472cef, detected fresh on all nine
# clips, stamped, and diffed ON against shipped_config before a score was taken -
# AFTER the identity stage, because that is the file app/server/jobs.py builds
# match.json from. Review found the first re-point read .p21, the tracker's output
# before identity, which differs on 117092 where the stage merged a pair.
# Until then these two ids read the .p05 tracks, written 2026-09-11 behind BOTH
# prior gates; those values stay in the golden's history under f89b9ba's scope.
# The calibration is N6's, reused rather than re-solved, so it did not move.
STC8_SCOPE = ("PRODUCT config per app/server/jobs.py AS IT SHIPPED at c6b8fb8 and "
              "8472cef (coreml r576, conf 0.5, stride 2, pitch polygon only, tile "
              "overlap 0.25, metric stitch, then the identity stage): N21's .p21id "
              "tracks, written 2026-09-13 - STORED, their stamp is whatever check.py "
              "status reads now, not a claim here; N6's calibration reused; "
              "8 STC2025 clips: 132831 excluded by the benchmark and 132877 DROPPED "
              "for its mirrored ground-truth feed, per dist_eval.stc_gt's own rule; "
              "attribution in PIXELS because a 2 m metre "
              "gate under an 8.27 m map measures the map (n12-calibration (not published)); "
              "n=1 configuration")

# The scene prior's gate set moved under this constant twice in two days, so it
# is dated rather than described. 0a5a7cd retired the height gate in make_prior
# but left track.run passing its own copy, so the product kept BOTH gates until
# c6b8fb8 made the seam defer. Every .p05* artefact on disk predates that fix.
N6 = ("PRODUCT config per app/server/jobs.py (coreml, conf 0.5, stride 2, scene "
      "prior, metric stitch) AS IT SHIPPED UNTIL c6b8fb8 - i.e. with BOTH prior "
      "gates. From c6b8fb8 the product applies the pitch polygon ALONE, and the "
      "tracker.*.p19v2.common7 family is that configuration measured; these "
      "rows are the superseded one, kept because N19/N20 are what judged it. "
      "7 held-out ST2 sequences; GT every 2nd frame; n=1 configuration")

# NOT model/validation/api.py. It is the mechanism, not a producer: it calls spec.fn
# and stores what comes back, so no edit to it can move a value - and while it
# sat in every tuple below, a comment changed there re-ran every producer.
#
# The code whose change must invalidate a cached value. Rule 3 CLAIMED the key
# included the producing code and the first implementation did not - an audit
# caught it the same day. Over-invalidating costs a re-run; under-invalidating
# serves a stale number as fresh, which is the whole failure this module exists
# to stop.
CODE_MOT = ("model/validation/st2_mot.py", "model/validation/status.py")
CODE_N6 = CODE_MOT + ("model/validation/st2_product.py",)
CODE_AB = CODE_MOT + ("model/validation/st2_ab.py", "model/validation/st2_coalesce.py")
CODE_TEAM = ("model/validation/team_eval.py", "model/engine/track.py")
CODE_METRES = ("model/validation/st2_metres.py",
               "model/validation/st2_overlay.py")

# The product chain. `model/engine/stats.py` is in CODE_DIST because distance IS
# stats.py - dist_eval imports `speed_series` and `_windowed` rather than
# reimplementing them, so a change to the filter or the window must invalidate
# the number even though dist_eval.py itself did not move.
CODE_DIST = ("model/validation/dist_eval.py",
             "model/validation/contam_eval.py", "model/engine/stats.py",
             "model/engine/calibrators.py")
CODE_POSS = ("model/validation/possession_eval.py",
             "model/validation/touch_eval.py", "model/engine/possession.py",
             "model/engine/events.py")
CODE_FLOOR = ("model/validation/st2_eval.py",)
CODE_DET = ("model/engine/detsweep.py", "model/engine/detectors.py",
            "model/engine/sceneprior.py", "model/engine/calibrators.py")
CODE_BALL = ("model/validation/ball_eval.py", "model/engine/ball.py",
             "model/validation/teamtrack_tracks.py",
             # the merge that made BALL_M0 (score_m0 refuses the file if _merge moved)
             "model/engine/detectors.py", "model/validation/n70_class_merge.py",
             # ball_eval.check_settings compares the fixture to the product's config
             "model/validation/shipped_config.py")
CODE_TOUCH = ("model/validation/touch_eval.py",)
CODE_CONTAM = ("model/validation/contam_eval.py",
               "model/engine/calibrators.py")

# The 118575 chain reads these three files and nothing else. Declared as a
# tuple so every row that depends on the panorama half invalidates together -
# re-running the tracker moves tracks.split.json and every distance, identity
# and possession number with it.
# The remap tables calibration.json POINTS at ("maps": a directory that
# calibrators.load_maps globs). Hashing the pointer is not hashing the tables.
CAL_MAPS = ("data/footage/soccertrack_calib_118575/118575_mapx.npy",
            "data/footage/soccertrack_calib_118575/118575_mapy.npy")
IN_118575 = (SPLIT_TRACKS, GT_118575, CALIB_118575) + CAL_MAPS
IN_BAS = ("data/soccertrack_v2/bas_index.csv",)
# st2_eval.floors reads the CORRECTED labels by default: st2_cache's .npz per val match,
# not bas_index.csv. The split is read off the same constant st2_eval uses.
from st2_index import EVENT_SPLIT  # noqa: E402
IN_EVENTS_NPZ = IN_BAS + tuple(f"model/validation/out/st2_events/{m}.npz"
                               for m, sp in sorted(EVENT_SPLIT.items()) if sp == "val")
BALL_CLIPS = ("soccer_top__train__videos__D_20220220_1_0000_0030",
              "soccer_top__train__videos__D_20220220_1_0030_0060",
              "soccer_top__test__videos__D_20220220_1_1680_1710",
              "soccer_top__test__videos__D_20220220_1_1710_1740",
              "soccer_top__test__videos__D_20220220_1_1740_1770",
              "soccer_top__test__videos__D_20220220_1_1770_1800",
              "soccer_side__train__videos__F_20200220_1_0000_0030",
              "soccer_side__train__videos__F_20200220_1_0030_0060",
              "soccer_side__train__videos__F_20200220_1_0060_0090")
# The SHIPPED merge's candidates, committed: M0 of the N70 raw caches (coreml 576, tile 1024,
# overlap 0.25, player conf 0.5, ball conf 0.05), ball boxes only. Provenance (command, source
# cache sha256s, merge source sha256) is inside the file; n70_class_merge.export writes it.
# The old ball_cands_<clip>.json.gz ran EVERY class at conf 0.05 into the merge (N70).
BALL_M0 = "model/validation/fixtures/ball/m0_cands.json.gz"
# Candidates AND the ball truth they are scored against: the truth is a CSV
# per clip (ball_eval.load_ball), and it was not in the key at all.
IN_BALL_GT = (BALL_M0,) + tuple(
    p for c in BALL_CLIPS for p in (
        f"data/footage/soccertrack/{c.replace('videos', 'annotations')}.csv",
        # ball_eval._clip_cal: the shipped picker projects through it. Declared
        # for every clip - absent ones hash as MISSING and move when created.
        f"model/validation/calib_{c}.json")) + (
    # read as TEXT by shipped_config.product_config, so no import records it
    "app/server/jobs.py",)
IN_TEAM = (f"{JOB}/team_eval.crops.npz", CALIB_118575, GT_118575) + CAL_MAPS
IN_METRES = (f"{GT_SRC}/118575.txt", ST2_RAW_118575,
             "data/footage/soccertrack_calib_118575/118575_keypoints.json")
# PathCRF's weights and the vendored code that runs them (model/engine/possession.py
# adds data/vendor/pathcrf to the path and loads saved/<TRIAL>).
PATHCRF = ("data/vendor/pathcrf/saved/140", "data/vendor/pathcrf/models",
           "data/vendor/pathcrf/set_transformer", "data/vendor/pathcrf/datatools")
IN_TOUCH = (GT_118575, "data/soccertrack_v2/ball__118575_1st_ball.npz",
            "data/soccertrack_v2/bas__118575__118575_12_class_events.json")
IN_EVENTS = ("data/footage/soccertrack_calib_118575/118575_12_class_events.json",)
# possession_eval.score reads the GT tracks, the labels and PathCRF - NOT the job's
# tracks, calibration or maps. Only the pipeline row (score_pipeline) reads those.
IN_POSS = (GT_118575,) + IN_EVENTS + PATHCRF
IN_POSS_PIPELINE = IN_POSS + IN_118575

#: What st2_metres inverts: ST2's published homography + mapx chain. The
#: product's map is the operator's <=21 clicks solved in app/frontend/src/lib/pitch.ts,
#: which nothing scores (model-map (not published) §5 N12).
VENDOR_MAP = "ST2's own homography+mapx, NOT the operator-clicked map the product uses"
#: What every `.c03s` / `.c03` / `.c03s_op` id ran, per model/validation/st2_run.py
#: stages 1-2. It is NOT the product: app/server/jobs.py runs conf 0.5, stride 2,
#: calibrated scene prior, no RoleNet, metric stitch, then identity.merge_tracklets.
#: Said in every scope because the id names alone read as "the tracker". Only
#: the 5 files st2_run wrote record it (`_config`); the other 5, both tuned
#: sequences among them, predate the record and carry only `_perf`.
C03S = ("RESEARCH CONFIG per st2_run.py: coreml, conf 0.3, stride 1, --roles, "
        "no calibration, HeightScale stitch - not the product path; recorded in "
        "5 of the 10 files")


RUN_RECORD = "model/models/role_clf.run.json"
CODE_ROLE = ("model/validation/train_roles.py",)


def _role(path):
    """A metric from the training run record, as its SPREAD over seeds.

    The value is a dict, deliberately. `train_roles.py` was published at
    accuracy 0.950 and referee recall 0.971 from ONE unseeded run; seeded three
    ways, the identical protocol puts both outside the resulting range. A single
    scalar id here would rebuild exactly the mistake that discovery corrected -
    so the measurement IS the spread, and a reader cannot quote the median
    without seeing how far the seeds move.
    """
    import json as _json
    r = _json.load(open(os.path.join(ROOT, RUN_RECORD)))
    node = r["test"]
    for k in path.split("."):
        node = node[k]
    return {k: node[k] for k in ("min", "median", "max", "n_seeds") if k in node}


# --------------------------------------------------------------------------
# ROADMAP A: the oracle-substitution experiment, `model/validation/calib_hypothesis.py`
#
# SHOULD A COUNTERFACTUAL BE IN A REGISTRY OF WHAT THE PRODUCT DOES? Both ways,
# and then the answer.
#
#   AGAINST. The oracle map cannot ship - it is regressed on the annotations it
#   is scored against, on one match, one camera. This registry is read as the
#   product's ledger, so an id that resolves to `ratio_raw 1.046` invites
#   exactly the comparison `postmortem (not published)` B2 is: a number from one
#   population quoted as if it came from another. Nothing here can stop a
#   reader pasting the oracle's number into a sentence about the product.
#
#   FOR, and this is the answer. The counterfactual is what the roadmap is
#   being re-ordered on: it is the evidence that calibration owns the 21-point
#   recall gap and detection owns the 30% hole. (It was also read as owning
#   the 20%-low distance; that reading was hull extrapolation and is retracted,
#   b9a2106 - gated, the map is worth ~2 points of distance.)
#   Its numbers already exist - in `state-and-roadmap` (not published) A, as literals,
#   under an `api:ok` waiver pointing at a json file. That is `+0.024`'s exact
#   shape and the reason it survived seven documents. A counterfactual is also
#   MORE fragile than a product number, not less: this one hangs on a 22-way id
#   permutation, a frame offset, a convex hull and an interpolator's smoothing,
#   and the only way a move in any of them is ever noticed is if the value is
#   keyed on the code that produces it. Leaving it out does not stop it being
#   quoted; it stops it being CHECKED.
#
#   So: registered, and fenced. Every id is `calib.oracle.*` and never
#   `product.*`; every scope says ORACLE MAP - not shippable; every desc names
#   the shipped id it must be read against; and the two controls that decide
#   whether the counterfactual means anything are registered beside it.
#
# THE REFUTATION IS REGISTERED TOO. `calib.oracle.distance.coverage` is the one
# that came out NO - coverage 0.704 -> 0.695, the result that stopped us
# blaming calibration for the 30% hole. It is the row most likely to be quietly
# dropped in a re-run, so it is the row that most needs an id.
#
# WHAT HAD TO CHANGE, AND IT IS NOT IN THIS FILE. `calib_hypothesis.py` exposes
# no function returning the run's numbers - `main()` prints and writes a json -
# so these producers compose its OWN functions (`build_oracle`,
# `oracle_distance_control`, `space_recall`, `relabel`, `score`) exactly as
# `main()` does, which is the same thing `_ab` and `_mot` above do. Nothing is
# recomputed here. Reading `model/validation/out/calib_hypothesis.json` instead was
# the cheaper option and was rejected on evidence: that file is currently a
# `--clip-only` run whose distance rows do NOT match the full-half numbers the
# doc cites it for, and it records neither the scope-independent flags nor the
# smoothing, so nothing reading it can tell which run it is holding.
# --------------------------------------------------------------------------

GT_PX_MOT = _ST2 + "/mot/118575.txt"
JOIN_118575 = "model/validation/out/calib_hypothesis_join.json"

#: The oracle's TPS smoothing, and it is load-bearing rather than taste.
#: `calib_hypothesis.main()` defaulted to 1.0 until 637ff1d (it now defaults
#: to this constant); at 1.0 the published run
#: reproduces NONE of its numbers (held-out residual 0.792 m not 0.549,
#: recall_m 0.726 not 0.753, ratio_raw 1.274 not 1.046) and the neutrality
#: control reads 1.094, outside the band `selfcheck()` itself asserts - i.e. at
#: the shipped default the +23% would be map wiggle. 10000.0 is the value
#: `selfcheck()` hard-codes and the only one that reproduces the run. It lives
#: HERE, in the registry, because api.py hashes this module into every key: move
#: it and every row below re-runs.
ORACLE_SMOOTHING = 10000.0
ORACLE_NEIGHBORS = 64
ORACLE_GATE_M = 2.0
ORACLE_STRIDE = 10

IN_CALIB_HYP = (SPLIT_TRACKS, GT_118575, CALIB_118575, GT_PX_MOT, JOIN_118575) + CAL_MAPS
CODE_CALIB_HYP = ("model/validation/calib_hypothesis.py",
                  "model/validation/dist_eval.py", "model/validation/contam_eval.py",
                  "model/engine/calibrators.py", "model/engine/stats.py")


@lru_cache(maxsize=None)
def _calib():
    """The whole oracle substitution, once, so nine rows cost one run.

    Composed out of `calib_hypothesis`'s own functions in `main()`'s order. The
    shipped-map distance column that `main()` also computes is deliberately NOT
    run: it reproduces `product.distance.ratio_raw` and
    `product.distance.coverage` to the digit, and a second id for the same
    number is the copy step this layer exists to delete.
    """
    import json as _json
    import calib_hypothesis as CH
    import calibrators
    import contam_eval
    import dist_eval
    cal = calibrators.load(os.path.join(ROOT, CALIB_118575))
    jn = _json.load(open(os.path.join(ROOT, JOIN_118575)))
    # JSON has no integer keys. Left as strings, `perm.get(mot_id)` misses every
    # player, the oracle fits on zero points and NOTHING RAISES - it just gets
    # worse. calib_hypothesis.main() carries the same conversion and warning.
    jn["perm"] = {int(k): v for k, v in jn["perm"].items()}
    assert len(jn["perm"]) == 22, jn["perm"]
    orc, ostat = CH.build_oracle(cal, jn, neighbors=ORACLE_NEIGHBORS,
                                 smoothing=ORACLE_SMOOTHING, verbose=False)
    ostat.update(CH.oracle_distance_control(cal, orc, jn, every=2))
    rows, fps = CH.load_rows("tracks.split.json")
    _gfps, gtw = dist_eval.load_gt()
    gtf = contam_eval.load_gt()
    sr = CH.space_recall(cal, orc, jn, rows)
    rows_o, _rstat = CH.relabel(cal, orc, rows)
    args = (fps, cal, cal, gtw, gtf, ORACLE_STRIDE, ORACLE_GATE_M)
    return {"oracle": ostat, "space_recall": sr,
            "ceiling": CH.score(dist_eval.gt_as_prediction(gtw, cal), *args),
            "oracle_run": CH.score(rows_o, *args)}


def _calib_map_error(which):
    """median / p90 / within-2m for one map, as ONE value.

    Three ids would let the median be quoted alone, and the shipped map's
    median is the half of its distribution that looks fine.
    """
    o = _calib()["oracle"]
    return {k: o[f"{which}_{k}"] for k in ("median_m", "p90_m", "within_2m")}


def _calib_distance(key):
    """An oracle distance row, refused unless the oracle is distance-neutral.

    Distance is a sum of displacements, so a map that wiggles reads as running.
    Below 0.95 or above 1.05 the ratio is measuring the interpolator and says
    nothing about the product, and `calib_hypothesis.selfcheck` asserts the same
    band - so this is that assertion, moved to where the number is published.
    """
    c = _calib()
    neu = c["oracle"]["distance_neutrality"]
    if not 0.95 < neu < 1.05:
        raise RuntimeError(
            f"oracle map is not distance-neutral ({neu:.3f}, band 0.95-1.05): "
            f"a ratio through it is map wiggle, not path length")
    return c["oracle_run"][key]


MEASUREMENTS = {
    "tracker.hota.c03s.all10": Spec(
        "mean HOTA at the frozen config over the whole ST2 set", "HOTA",
        f"all 10 sequences; {C03S}", 10, lambda: _mean(".c03s", ALL10, "HOTA"),
        IN_MOT, CODE_MOT),
    "tracker.hota.c03s.tuned": Spec(
        "mean HOTA on the two sequences the config was tuned on", "HOTA",
        f"TUNED 128057+132831; {C03S}", 2, lambda: _mean(".c03s", TUNED, "HOTA"),
        IN_MOT, CODE_MOT),
    "tracker.hota.c03s.heldout": Spec(
        "mean HOTA on the eight sequences never used for tuning", "HOTA",
        f"HELD OUT, 8 sequences; {C03S}", 8,
        lambda: _mean(".c03s", HELDOUT, "HOTA"), IN_MOT, CODE_MOT),
    "tracker.deta.c03s.all10": Spec(
        "mean DetA at the frozen config", "DetA",
        f"all 10 sequences; {C03S}", 10, lambda: _mean(".c03s", ALL10, "DetA"),
        IN_MOT, CODE_MOT),
    "tracker.assa.c03s.all10": Spec(
        "mean AssA at the frozen config - the association headroom", "AssA",
        f"all 10 sequences; {C03S}", 10, lambda: _mean(".c03s", ALL10, "AssA"),
        IN_MOT, CODE_MOT),
    # --- N6: the product's own tracking, two calibrations, one comparator ----
    # A minus B is what the operator-clickable calibration costs the tracker;
    # B against c03s_e2 is the tracker itself. model-map (not published) §5 N6.
    "tracker.hota.p05.common7": Spec(
        "mean HOTA of the product's tracker with the product's calibration: "
        "perfect simulated clicks on the UI's 21 landmarks, solved by the UI's "
        "own solver (app/frontend/src/lib/pitch.ts) - one homography, ~6 m off on "
        "a stitched panorama", "HOTA", f"{N6}; calibration SIMULATED CLICKS",
        7, lambda: _mean(".p05", COMMON7, "HOTA", COMMON7, 2), IN_N6, CODE_N6),
    "tracker.hota.p05m.common7": Spec(
        "mean HOTA of the same tracker with ST2's own chain as a remap "
        "calibration - calibration taken out of the picture. Not something the "
        "UI can produce", "HOTA", f"{N6}; calibration ST2's homography+mapx as remap",
        7, lambda: _mean(".p05m", COMMON7, "HOTA", COMMON7, 2), IN_N6, CODE_N6),
    "tracker.hota.c03s_e2.common7": Spec(
        "mean HOTA of the research config .c03s on the same 7 sequences and the "
        "same frames - the comparator every registered tracker id has used",
        "HOTA", f"{C03S}; the same 7 sequences, GT every 2nd frame", 7,
        lambda: _mean(".c03s_e2", COMMON7, "HOTA", COMMON7, 2), IN_N6, CODE_N6),
    "tracker.deta.p05.common7": Spec(
        "mean DetA, product tracker with the simulated-click calibration - "
        "where the calibration's cost lands (its scene prior rejects players)",
        "DetA", f"{N6}; calibration SIMULATED CLICKS", 7,
        lambda: _mean(".p05", COMMON7, "DetA", COMMON7, 2), IN_N6, CODE_N6),
    # AssA too: the calibration's cost lands on association as well as on
    # detection (the metric stitch runs through the same ~6 m fit), and a
    # sentence saying so is not a measurement.
    "tracker.assa.p05.common7": Spec(
        "mean AssA, product tracker with the simulated-click calibration",
        "AssA", f"{N6}; calibration SIMULATED CLICKS", 7,
        lambda: _mean(".p05", COMMON7, "AssA", COMMON7, 2), IN_N6, CODE_N6),
    "tracker.assa.p05m.common7": Spec(
        "mean AssA, product tracker with ST2's own chain as remap calibration",
        "AssA", f"{N6}; calibration ST2's homography+mapx as remap", 7,
        lambda: _mean(".p05m", COMMON7, "AssA", COMMON7, 2), IN_N6, CODE_N6),
    # --- N18: B's post-prior detections replayed through A's calibration ---
    # No detector and no A prior (a cache hit skips it), so the metric stitch
    # runs through the ~6 m homography on B's boxes. Against A: what A's prior
    # cost detection. Against B: what the bad homography costs association
    # (confounded by pixel space: B matched boxes on the flattened canvas).
    "tracker.hota.p05r.common7": Spec(
        "mean HOTA of B's detections replayed through the product's clicked "
        "calibration with its scene prior out of the way (N18)", "HOTA",
        f"{N6}; B's post-prior boxes in clip px, A's calibration, no A prior",
        7, lambda: _mean(".p05r", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p05r.tracks.json", ".p05r.dets.cache.json")), CODE_N6),
    "tracker.deta.p05r.common7": Spec(
        "mean DetA of B's detections replayed through the product's clicked "
        "calibration with its scene prior out of the way (N18)", "DetA",
        f"{N6}; B's post-prior boxes in clip px, A's calibration, no A prior",
        7, lambda: _mean(".p05r", COMMON7, "DetA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p05r.tracks.json", ".p05r.dets.cache.json")), CODE_N6),
    "tracker.assa.p05r.common7": Spec(
        "mean AssA of B's detections replayed through the product's clicked "
        "calibration with its scene prior out of the way (N18)", "AssA",
        f"{N6}; B's post-prior boxes in clip px, A's calibration, no A prior",
        7, lambda: _mean(".p05r", COMMON7, "AssA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p05r.tracks.json", ".p05r.dets.cache.json")), CODE_N6),
    "tracker.deta.p05m.common7": Spec(
        "mean DetA, product tracker with ST2's own chain as remap calibration",
        "DetA", f"{N6}; calibration ST2's homography+mapx as remap", 7,
        lambda: _mean(".p05m", COMMON7, "DetA", COMMON7, 2), IN_N6, CODE_N6),

    # --- N19: the scene prior's two gates, one detection pass, four arms -----
    # Detected ONCE per sequence with no prior; each arm is that same cache
    # filtered offline by track.make_prior(gates=...) from A's calibration and
    # replayed. n=4 arms, pre-registered in model-map (not published) §3 N19 with the
    # pass rule, before any of them ran. DetPr is registered beside DetRe
    # because a filter that buys recall with precision must show both.
    "tracker.hota.p19v0.common7": Spec(
        "N19 arm v0: both gates - the SHIPPED prior; must reproduce tracker.*.p05.common7 - mean HOTA", "HOTA",
        f"{N6}; N19 arm v0 (both gates - the SHIPPED prior; must reproduce tracker.*.p05.common7); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v0", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v0.tracks.json")), CODE_N6),
    "tracker.detre.p19v0.common7": Spec(
        "N19 arm v0: both gates - the SHIPPED prior; must reproduce tracker.*.p05.common7 - mean DetRe", "DetRe",
        f"{N6}; N19 arm v0 (both gates - the SHIPPED prior; must reproduce tracker.*.p05.common7); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v0", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v0.tracks.json")), CODE_N6),
    "tracker.detpr.p19v0.common7": Spec(
        "N19 arm v0: both gates - the SHIPPED prior; must reproduce tracker.*.p05.common7 - mean DetPr", "DetPr",
        f"{N6}; N19 arm v0 (both gates - the SHIPPED prior; must reproduce tracker.*.p05.common7); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v0", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v0.tracks.json")), CODE_N6),
    "tracker.hota.p19v1.common7": Spec(
        "N19 arm v1: no prior at all - mean HOTA", "HOTA",
        f"{N6}; N19 arm v1 (no prior at all); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v1", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v1.tracks.json")), CODE_N6),
    "tracker.detre.p19v1.common7": Spec(
        "N19 arm v1: no prior at all - mean DetRe", "DetRe",
        f"{N6}; N19 arm v1 (no prior at all); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v1", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v1.tracks.json")), CODE_N6),
    "tracker.detpr.p19v1.common7": Spec(
        "N19 arm v1: no prior at all - mean DetPr", "DetPr",
        f"{N6}; N19 arm v1 (no prior at all); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v1", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v1.tracks.json")), CODE_N6),
    "tracker.hota.p19v2.common7": Spec(
        "N19 arm v2: pitch polygon only, height gate OFF - mean HOTA", "HOTA",
        f"{N6}; N19 arm v2 (pitch polygon only, height gate OFF) - THIS IS THE "
        f"SHIPPED PRIOR from c6b8fb8 on; one no-prior detection pass, gates "
        f"applied offline, which N19's v0 arm licenses by reproducing the "
        f"inline path to 0.000; n=4 arms", 7,
        lambda: _mean(".p19v2", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v2.tracks.json")), CODE_N6),
    "tracker.detre.p19v2.common7": Spec(
        "N19 arm v2: pitch polygon only, height gate OFF - mean DetRe", "DetRe",
        f"{N6}; N19 arm v2 (pitch polygon only, height gate OFF) - THIS IS THE "
        f"SHIPPED PRIOR from c6b8fb8 on; one no-prior detection pass, gates "
        f"applied offline, which N19's v0 arm licenses by reproducing the "
        f"inline path to 0.000; n=4 arms", 7,
        lambda: _mean(".p19v2", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v2.tracks.json")), CODE_N6),
    "tracker.detpr.p19v2.common7": Spec(
        "N19 arm v2: pitch polygon only, height gate OFF - mean DetPr", "DetPr",
        f"{N6}; N19 arm v2 (pitch polygon only, height gate OFF) - THIS IS THE "
        f"SHIPPED PRIOR from c6b8fb8 on; one no-prior detection pass, gates "
        f"applied offline, which N19's v0 arm licenses by reproducing the "
        f"inline path to 0.000; n=4 arms", 7,
        lambda: _mean(".p19v2", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v2.tracks.json")), CODE_N6),
    "tracker.hota.p19v3.common7": Spec(
        "N19 arm v3: height gate only, polygon OFF - mean HOTA", "HOTA",
        f"{N6}; N19 arm v3 (height gate only, polygon OFF); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v3", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v3.tracks.json")), CODE_N6),
    "tracker.detre.p19v3.common7": Spec(
        "N19 arm v3: height gate only, polygon OFF - mean DetRe", "DetRe",
        f"{N6}; N19 arm v3 (height gate only, polygon OFF); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v3", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v3.tracks.json")), CODE_N6),
    "tracker.detpr.p19v3.common7": Spec(
        "N19 arm v3: height gate only, polygon OFF - mean DetPr", "DetPr",
        f"{N6}; N19 arm v3 (height gate only, polygon OFF); one no-prior detection pass, gates "
        f"applied offline, n=4 arms", 7,
        lambda: _mean(".p19v3", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p19raw.dets.cache.json", ".p19v3.tracks.json")), CODE_N6),
    # --- N20: the same four arms on the GOOD calibration (B's remap) ---------
    # N19 could not say whether the height gate EARNS its keep when the camera
    # is recovered from a fit that is right. Same offline-gate method, B's
    # calibration and canvas, tracks mapped back for scoring. Pre-registered
    # rule: the gate earns its keep iff v0 beats v2 on HOTA on >=6 of 7. It won
    # on 1 of 7, so it does not - on either calibration.
    "tracker.hota.p20v0.common7": Spec(
        "N20 arm v0: both gates - the shipped prior on a GOOD calibration; must reproduce tracker.*.p05m.common7 - mean HOTA", "HOTA",
        f"{N6}; N20 arm v0 (both gates - the shipped prior on a GOOD calibration; must reproduce tracker.*.p05m.common7); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v0", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v0.tracks.json")), CODE_N6),
    "tracker.detre.p20v0.common7": Spec(
        "N20 arm v0: both gates - the shipped prior on a GOOD calibration; must reproduce tracker.*.p05m.common7 - mean DetRe", "DetRe",
        f"{N6}; N20 arm v0 (both gates - the shipped prior on a GOOD calibration; must reproduce tracker.*.p05m.common7); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v0", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v0.tracks.json")), CODE_N6),
    "tracker.detpr.p20v0.common7": Spec(
        "N20 arm v0: both gates - the shipped prior on a GOOD calibration; must reproduce tracker.*.p05m.common7 - mean DetPr", "DetPr",
        f"{N6}; N20 arm v0 (both gates - the shipped prior on a GOOD calibration; must reproduce tracker.*.p05m.common7); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v0", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v0.tracks.json")), CODE_N6),
    "tracker.hota.p20v1.common7": Spec(
        "N20 arm v1: no prior at all - mean HOTA", "HOTA",
        f"{N6}; N20 arm v1 (no prior at all); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v1", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v1.tracks.json")), CODE_N6),
    "tracker.detre.p20v1.common7": Spec(
        "N20 arm v1: no prior at all - mean DetRe", "DetRe",
        f"{N6}; N20 arm v1 (no prior at all); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v1", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v1.tracks.json")), CODE_N6),
    "tracker.detpr.p20v1.common7": Spec(
        "N20 arm v1: no prior at all - mean DetPr", "DetPr",
        f"{N6}; N20 arm v1 (no prior at all); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v1", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v1.tracks.json")), CODE_N6),
    "tracker.hota.p20v2.common7": Spec(
        "N20 arm v2: pitch polygon only, height gate OFF - mean HOTA", "HOTA",
        f"{N6}; N20 arm v2 (pitch polygon only, height gate OFF); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v2", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v2.tracks.json")), CODE_N6),
    "tracker.detre.p20v2.common7": Spec(
        "N20 arm v2: pitch polygon only, height gate OFF - mean DetRe", "DetRe",
        f"{N6}; N20 arm v2 (pitch polygon only, height gate OFF); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v2", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v2.tracks.json")), CODE_N6),
    "tracker.detpr.p20v2.common7": Spec(
        "N20 arm v2: pitch polygon only, height gate OFF - mean DetPr", "DetPr",
        f"{N6}; N20 arm v2 (pitch polygon only, height gate OFF); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v2", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v2.tracks.json")), CODE_N6),
    "tracker.hota.p20v3.common7": Spec(
        "N20 arm v3: height gate only, polygon OFF - mean HOTA", "HOTA",
        f"{N6}; N20 arm v3 (height gate only, polygon OFF); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v3", COMMON7, "HOTA", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v3.tracks.json")), CODE_N6),
    "tracker.detre.p20v3.common7": Spec(
        "N20 arm v3: height gate only, polygon OFF - mean DetRe", "DetRe",
        f"{N6}; N20 arm v3 (height gate only, polygon OFF); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v3", COMMON7, "DetRe", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v3.tracks.json")), CODE_N6),
    "tracker.detpr.p20v3.common7": Spec(
        "N20 arm v3: height gate only, polygon OFF - mean DetPr", "DetPr",
        f"{N6}; N20 arm v3 (height gate only, polygon OFF); GOOD calibration (ST2 remap), one "
        f"no-gate detection pass on its canvas, gates applied offline, n=4 arms", 7,
        lambda: _mean(".p20v3", COMMON7, "DetPr", COMMON7, 2),
        IN_N6 + tuple(f"model/validation/out/stc2025/{s}{v}" for s in COMMON7
                      for v in (".p20raw.dets.cache.json", ".p20v3.tracks.json")), CODE_N6),
    # The verdict SAYS "DOES NOT SHIP" and the stitch runs on every product job
    # anyway (track.py). One of the two is wrong; model-map (not published) §5 N7.
    "ab.stitch.assa": Spec(
        "does the stitch beat raw tracker output, on the primary metric",
        "verdict", f"all 10 incl. the 2 tuned, AssA primary; .c03 vs .c03s; "
        f"{C03S}", 10,
        lambda: _ab(".c03", ".c03s", "AssA")[0], IN_MOT + RAW, CODE_AB),
    # The gate this scores is st2_onpitch.py's TRACK-level gate, and nothing in
    # model/engine/ or app/server/ applies it, whatever the verdict says. The product has
    # a different, BOX-level one: the scene prior drops boxes whose feet fall
    # outside the pitch polygon (model/engine/track.py, `keep`).
    "ab.offpitch.hota": Spec(
        "does the off-pitch gate ship, on HOTA", "verdict",
        f"all 10 incl. the 2 tuned, HOTA; {C03S}; this TRACK-level gate is "
        f"not in the product (its scene prior gates boxes)", 10, lambda: _ab(".c03s", ".c03s_op", "HOTA")[0],
        IN_MOT + GATED, CODE_AB),
    "ab.offpitch.assa": Spec(
        "does the off-pitch gate ship on AssA, the DECLARED primary", "verdict",
        f"all 10 incl. the 2 tuned, AssA primary; {C03S}; this TRACK-level "
        f"gate is not in the product (its scene prior gates boxes)", 10, lambda: _ab(".c03s", ".c03s_op", "AssA")[0],
        IN_MOT + GATED, CODE_AB),
    "team.accuracy.detection": Spec(
        "per-detection team accuracy, best of two label mappings", "fraction",
        "118575 1st half, 300 frames", 4101, lambda: _team("detection_accuracy"),
        IN_TEAM, CODE_TEAM),
    "team.accuracy.track": Spec(
        "per-track team accuracy", "fraction",
        "118575 1st half, 300 frames", 30, lambda: _team("track_accuracy"),
        IN_TEAM, CODE_TEAM),
    "calib.metres.118575.overall": Spec(
        "pixel->metres median error, the direction the product uses", "metres",
        f"118575, 20 frames, 440 instants; {VENDOR_MAP}", 440,
        lambda: _metres("118575", "overall"), IN_METRES, CODE_METRES),
    "calib.metres.118575.far": Spec(
        "pixel->metres median error in the farthest quartile", "metres",
        f"118575, farthest quartile; {VENDOR_MAP}", 110,
        lambda: _metres("118575", "farthest quartile"), IN_METRES, CODE_METRES),

    # ---------------------------------------------------------------------
    # PRODUCT. Everything above this line is plumbing: it answers "whose box is
    # this" and stops there. `full-pipeline-plan (not published)` opens on the reason
    # that is not enough - we could say the tracker scores HOTA 0.364 and could
    # not say what that costs a distance figure, and `model/engine/stats.py` has
    # carried `unvalidated_end_to_end` since the day it was written.
    #
    # Rows 6-8 of that plan's table are below. They are worse numbers than the
    # ones above them and that is the point: a bad number that is measured
    # beats a good number that is component-level and unconnected.
    # ---------------------------------------------------------------------

    # --- row 6: DISTANCE. The headline. ----------------------------------
    # Read `ratio_raw` and `coverage` together or not at all. Raw is what a
    # coach would be shown today; it is BELOW 1 because we lose 30% of the
    # frames, not because the metres are short - and reading it as accuracy is
    # postmortem (not published) B2 with new numbers.
    "product.distance.ratio_raw": Spec(
        "squad distance we would ship, over full GT distance - recall included",
        "ratio", "118575 1st half, tracks.split.json, 22 GT players, "
        "0.4 s window", 22,
        lambda: (_dist()["squad"]["ratio_raw"], len(_dist()["players"])),
        IN_118575, CODE_DIST),
    "product.distance.coverage": Spec(
        "mean share of a GT player's frames we hold an id on. Without this the "
        "row above is unreadable", "fraction",
        "118575 1st half, tracks.split.json, 22 GT players", 22,
        lambda: (_dist()["squad"]["coverage"], len(_dist()["players"])), IN_118575, CODE_DIST),
    "product.distance.ratio_matched": Spec(
        "squad distance over GT RE-SCORED WITH OUR OWN COVERAGE as its gap "
        "mask - the same windows dropped both sides, so this contains no "
        "recall. The estimator's accuracy where it can see", "ratio",
        "118575 1st half, tracks.split.json, 22 GT players, coverage-matched",
        22, lambda: (_dist()["squad"]["ratio_matched"], len(_dist()["players"])),
        IN_118575, CODE_DIST),
    # PER-PLAYER, from the run the three rows above already pay for. The squad
    # rows pool 22 players into one ratio; `match.json` ships a row PER PLAYER,
    # so the pooled number is a measurement of a population the product does
    # not emit (api-adequacy (not published) 1.2, 3). dist_eval has returned this block
    # since it was written and the registry was taking the mean and dropping it
    # on the floor - which is why `median 1.139 (IQR 1.082-1.234)` sits in
    # docs/experiments/st2-input-ledger.md:49 as an uncited literal and `cv 0.107` sits in
    # app/frontend/src/lib/report.ts:96 as a paragraph shown to readers.
    # `model/engine/stats.py::METRICS` resolves players.distance_km, derived.share
    # and derived.per90 to this id.
    "product.player.distance.spread": Spec(
        "per-player distance ratio_matched across the squad: median, IQR, min, "
        "max, cv, and the correlation with local metres-per-pixel - detector "
        "noise predicts that correlation, contamination predicts none",
        "ratios", STC8_SCOPE + "; PER PLAYER not pooled; 175 of the 176 GT players "
        "form a ratio - the one with no covered metres drops out, which n says", 175,
        lambda: ({k: _dist_stc()[k] for k in (
            "player_ratio_median", "player_ratio_cv", "player_ratio_min",
            "player_ratio_max", "player_ratio_n", "player_ratio_corr_m_per_px")},
            _dist_stc()["player_ratio_n"]),
        IN_STC_DIST, CODE_DIST + CODE_CONTAM),
    # Both sides of the peak comparison, because neither is quotable alone:
    # ours is a max over fragments that contain id switches and GT's is a max
    # over one real player. `match.json.players[].top_speed_kmh` and
    # `teams[].top_speed_kmh` publish OUR half of this and nothing measured it
    # until now. NOT the `clamped_frac` sitting in the same block - that one is
    # already registered below and a number registered twice is a number that
    # can disagree with itself.
    "product.speed.peak_ms": Spec(
        "GT peak speed against ours, max and median over the squad",
        "m/s", STC8_SCOPE + "; PER PLAYER not pooled; each player's peak over that "
        "player's fragments, so ours is a max across id switches and GT's is not; ours "
        "over the 175 players holding a fragment, GT's over all 176", 175,
        lambda: ({k: _dist_stc()[k] for k in (
            "gt_peak_max_ms", "player_gt_peak_median_ms",
            "our_peak_max_ms", "player_peak_median_ms")}, _dist_stc()["player_peak_n"]),
        IN_STC_DIST, CODE_DIST + CODE_CONTAM),
    # The squad-POOLED raw ratio on the tracks the product ships (stride 2, .p21id),
    # which is what model/engine/stats.py's teams.distance_m is justified by. It was
    # justified by product.distance.ratio_raw: one August job, stride 1, a tracks file
    # older than the current tracker. Same scorer as the two ids above, one run.
    "product.distance.ratio_raw.p21id": Spec(
        "squad distance we would ship, over full GT distance, pooled over every GT "
        "player (sum of metres, then the ratio) - recall included, so read it beside "
        "coverage", "ratio", STC8_SCOPE + "; POOLED over all 176 GT players", 176,
        lambda: (_dist_stc()["pooled_ratio_raw"], _dist_stc()["player_gt_n"]),
        IN_STC_DIST, CODE_DIST + CODE_CONTAM),
    # NAME COLLISION, LIVE: this is a share of FRAGMENTS (0.4661).
    # `match.json.players[].clamped_frac` is a share of STEPS (0.00117),
    # ~400x apart, and reading one as the other opened a roadmap item on a
    # misread (commit c3e0089). The scope string says "per FRAGMENT" and that
    # is what caught it. OWED: rename this id to
    # `product.speed.clamped_frag_frac` (api-adequacy (not published) R2) - the rename
    # moves the golden, so it belongs with the snapshot, not here. Until then
    # stats.py deliberately cites NOTHING for that field.
    "product.speed.clamped_frac": Spec(
        "share of our fragments whose unclamped peak exceeds stats.MAX_SPEED, "
        "i.e. that would show the 34.2 km/h clamp instead of a speed",
        "fraction", "118575 1st half, tracks.split.json, per FRAGMENT", 384,
        lambda: (_dist()["peak"]["clamped_frac"], _dist()["peak"]["our_frags"]),
        IN_118575, CODE_DIST),

    # --- row 7: POSSESSION. PathCRF, no ball anywhere. --------------------
    # GROUND-TRUTH trajectories, so this is the MODEL's ceiling on our footage
    # and says nothing about our tracking. The row that would say that is
    # `product.possession.f1.contact.pipeline` and it FAILS.
    "product.possession.f1.contact": Spec(
        "PathCRF possession changes vs the 118575 CONTACT labels, F1 at 1.0 s",
        "F1", "118575 half 1, 300-1800 s, GROUND-TRUTH tracks, trial 140; GT-fed ceiling, "
        "not the product",
        529, lambda: _poss()["rows"]["CONTACT.1.0"]["f1"],
        IN_POSS, CODE_POSS),
    "product.possession.chance.contact": Spec(
        "the same count of events thrown at random - the floor the row above "
        "has to clear, in the same units on the same labels", "F1",
        "118575 half 1, 300-1800 s, 300 random trials; GT-fed ceiling, not the "
        "product", 529,
        lambda: _poss()["rows"]["CONTACT.1.0"]["chance"],
        IN_POSS, CODE_POSS),
    # THE FLOOR THAT BINDS, and the one that was never run. The same count of
    # events spaced EVENLY: on these labels it scores ABOVE the model on both
    # headline rows, so neither headline is a result (model-map (not published) §0.4).
    "product.possession.metronome.contact": Spec(
        "the same count of events spaced evenly, mean of 20 phases - the floor "
        "product.possession.f1.contact must clear, and on these labels does not",
        "F1", "118575 half 1, 300-1800 s, CONTACT, tol 1.0 s, 20 phases; GT-fed "
        "ceiling, not the product", 529,
        lambda: _poss()["rows"]["CONTACT.1.0"]["metronome_f1"],
        IN_POSS, CODE_POSS),
    "product.possession.metronome.recall.pass": Spec(
        "PASS-family recall of the same count of evenly spaced events at 2.5 s "
        "- the floor product.possession.recall.pass must clear, and does not",
        "recall", "118575 half 1, 300-1800 s, PASS family, tol 2.5 s, 20 phases; "
        "GT-fed ceiling, not the product",
        283, lambda: _poss()["rows"]["PASS.2.5"]["metronome_rec"],
        IN_POSS, CODE_POSS),
    "product.possession.recall.pass": Spec(
        "share of PASS-family labels matched by a possession change at 2.5 s. "
        "Was the ball-free route's headline; it is BELOW "
        "product.possession.metronome.recall.pass, so it is not one", "recall",
        "118575 half 1, 300-1800 s, GROUND-TRUTH tracks, tol 2.5 s; GT-fed ceiling, "
        "not the product", 283,
        lambda: _poss()["rows"]["PASS.2.5"]["rec"], IN_POSS, CODE_POSS),
    "product.possession.f1.contact.pipeline": Spec(
        "the same, on OUR tracks instead of ground truth - what tracking costs "
        "possession. FAILS: PathCRF needs 22 tracks and gets 413, and the "
        "job's calibration rms is above possession.MAX_RMS_M", "F1",
        "118575 half 1, tracks.split.json - sprint P2, not yet runnable", 529,
        _poss_pipeline,
        IN_POSS_PIPELINE, CODE_POSS),

    # --- row 8: EVENTS. The floors, and the fact that nothing clears them. --
    # A model that does not beat a metronome is not a model. `metronome` emits
    # one event every mean-gap second: on this corpus events come every 1.6 s,
    # so the clock alone is worth more than chance and it is the binding floor.
    "events.floor.chance.pass.ap": Spec(
        "AP of uniformly random spots on PASS, tol 1.0 s - the absolute floor",
        "AP", "ST2 val split (117093+132877), 4 periods, mean of 5 seeds",
        2029, lambda: _floors()["AP"]["chance.PASS.1.0"], IN_EVENTS_NPZ, CODE_FLOOR),
    "events.floor.metronome.pass.ap": Spec(
        "AP of one event every mean-gap second on PASS, tol 1.0 s - the floor "
        "that catches a model which learned the clock", "AP",
        "ST2 val split (117093+132877), 4 periods, mean of 5 seeds", 2029,
        lambda: _floors()["AP"]["metronome.PASS.1.0"], IN_EVENTS_NPZ, CODE_FLOOR),
    "events.floor.chance.drive.ap": Spec(
        "AP of uniformly random spots on DRIVE, tol 1.0 s", "AP",
        "ST2 val split (117093+132877), 4 periods, mean of 5 seeds", 1775,
        lambda: _floors()["AP"]["chance.DRIVE.1.0"], IN_EVENTS_NPZ, CODE_FLOOR),
    "events.floor.metronome.drive.ap": Spec(
        "AP of one event every mean-gap second on DRIVE, tol 1.0 s", "AP",
        "ST2 val split (117093+132877), 4 periods, mean of 5 seeds", 1775,
        lambda: _floors()["AP"]["metronome.DRIVE.1.0"], IN_EVENTS_NPZ, CODE_FLOOR),
    "events.ceiling.shiftcopy.drive.ap": Spec(
        "AP of DRIVE emitted at every GT PASS + 1.40 s, tol 1.0 s. GT-FED, so "
        "a CEILING on what a PASS->DRIVE offset buys, not a floor - and DRIVE "
        "is 38% of the corpus", "AP",
        "ST2 val split (117093+132877), GT-fed, mean of 5 seeds", 1775,
        lambda: _floors()["AP"]["shift-copy*.DRIVE.1.0"], IN_EVENTS_NPZ, CODE_FLOOR),
    "events.model.map.val": Spec(
        "what OUR event model scores against those floors. FAILS: sprint P1 is "
        "not trained, so nothing in this repo is compared to them - the floors "
        "are half a gate on their own", "AP",
        "ST2 val split (117093+132877), tol 1.0 s - no model exists", 4532,
        _event_model, IN_EVENTS_NPZ + ("model/models/event_model.pt",),
        CODE_FLOOR),

    # --- row 1: DETECTION, so the chain is complete at the front. ---------
    # The two recalls are the same boxes scored in two spaces and the GAP
    # BETWEEN THEM IS CALIBRATION, not detection: the box is on the player and
    # the metres are wrong. A far-side "detection fix" measured only in metres
    # is measuring calibrators.py.
    "product.detect.recall_m": Spec(
        "share of GT players with a detection whose FOOT projects within 2 m - "
        "the product metric; the tracker and every stat live here", "recall",
        "118575 band canvas (the job's remap calibration), coreml r576 t1024 - "
        "the SHIPPED detector, built by detectors.get_detector('coreml') - "
        "conf 0.5, tile overlap 0.25 (reverted in 8472cef), prior = the PITCH "
        "POLYGON ALONE: detsweep calls make_prior with no gates and so has "
        "measured the polygon since 0a5a7cd, while the product only joined it "
        "at c6b8fb8. 40 frames from 3000 every 1500", 880,
        # The ANE worker runs in its own Python, so its model, script and
        # coremltools are declared: api._deps cannot see a subprocess's imports.
        lambda: _det("coreml r576 t1024")["recall_m"],
        IN_118575 + ("data/footage/118575_panorama_1st_half.mp4",) + _coreml_inputs(), CODE_DET),
    "product.detect.recall_px": Spec(
        "share of GT players with a box within 0.6x expected height IN THE "
        "IMAGE - the detector's own metric. The gap to recall_m is calibration",
        "recall", "118575 band canvas (the job's remap calibration), coreml "
        "r576 t1024 - the SHIPPED detector - conf 0.5, tile overlap 0.25 "
        "(reverted in 8472cef), prior = the PITCH POLYGON ALONE (detsweep has "
        "measured the polygon since 0a5a7cd; the product joined it at c6b8fb8), "
        "40 frames from 3000 every 1500", 880,
        lambda: _det("coreml r576 t1024")["recall_px"],
        IN_118575 + ("data/footage/118575_panorama_1st_half.mp4",) + _coreml_inputs(), CODE_DET),

    # --- the role classifier, reported as a SPREAD over seeds -------------
    "model.role.accuracy": Spec(
        "role classifier overall accuracy, min/median/max over 3 seeds - the "
        "spread IS the measurement, see _role()", "spread",
        "4 held-out clips, 3,081 crops, epoch chosen on val, test opened once "
        "per seed", 3081, lambda: _role("accuracy"), (RUN_RECORD,), CODE_ROLE),
    "model.role.referee_recall": Spec(
        "referee recall, min/median/max over 3 seeds. The 0.971 this repo "
        "published for months was one unseeded run and sits above the max. "
        "TERMINAL: the model is licence blocker B3 and cannot ship, and this "
        "class cannot be re-measured on anything we may legally use - ST2 "
        "annotates zero referees. Kept because it is evidence; see README and "
        "model/validation/train_roles_st2.py",
        "spread", "4 held-out clips, 312 referee crops", 312,
        lambda: _role("recall.referee"), (RUN_RECORD,), CODE_ROLE),
    "model.role.goalkeeper_recall": Spec(
        "goalkeeper recall, min/median/max over 3 seeds. The 0.613 that a "
        "paragraph of rolenet.py was written to explain sits BELOW the min. "
        "The model cannot ship (B3); THIS class is reproducible on ST2 and is "
        "the bar train_roles_st2.py pre-registered against",
        "spread", "4 held-out clips, 119 keeper crops", 119,
        lambda: _role("recall.goalkeeper"), (RUN_RECORD,), CODE_ROLE),

    # --- the rest of the chain that produces a product-facing number ------
    "product.identity.contaminated": Spec(
        "share of attributed detections sitting under an id whose dominant GT "
        "player is somebody else - what fraction of a player's row is another "
        "player", "percent",
        "118575 1st half, tracks.split.json, PER DETECTION (stride 10)",
        100114, lambda: _contam()["contaminated"], IN_118575, CODE_CONTAM),
    "product.ball.correct": Spec(
        "share of ball-truth frames where the shipped picker answers within "
        "20 px, pooled over frames, detection on EVERY frame (the registered "
        "definition; .s2 is what jobs.py runs)", "fraction",
        "9 SoccerTrack clips from 2 matches, stride 1 (every frame; the product "
        "runs stride 2, see .s2); otherwise the SHIPPED settings: coreml "
        "rfdetr-medium 576, tile 1024, overlap 0.25, player conf 0.5, ball conf "
        "0.05 (detectors.BALL_CONF), the shipped class-agnostic merge "
        "detectors._merge 0.55 (N70 M0), BallPicker marking gate off, each "
        "clip's calib_<clip>.json where one exists (6 top clips, perspective; "
        "the 3 side clips have none); STOCK weights, no fine-tuned checkpoint "
        "wired. Candidates: " + BALL_M0, 7649,
        lambda: (_ball()["pooled"]["shipped/s1"]["rate"], _ball()["pooled"]["shipped/s1"]["n"]),
        IN_BALL_GT, CODE_BALL),
    "product.ball.correct.s2": Spec(
        "product.ball.correct at the product's stride - the picker fed even "
        "frames only, as app/server/jobs.py runs it", "fraction",
        "9 SoccerTrack clips from 2 matches at the SHIPPED settings: stride 2 "
        "(even frames), coreml rfdetr-medium 576, tile 1024, overlap 0.25, "
        "player conf 0.5, ball conf 0.05, merge detectors._merge 0.55 (N70 M0), "
        "marking gate off, calibrations as product.ball.correct; the "
        "denominator is the ball-truth frames among the even frames", 3824,
        lambda: (_ball()["pooled"]["shipped/s2"]["rate"], _ball()["pooled"]["shipped/s2"]["n"]),
        IN_BALL_GT, CODE_BALL),
    "product.ball.oracle": Spec(
        "the same, for the gated candidate NEAREST truth - the ceiling over "
        "this detector and merge", "fraction",
        "same 9 clips and candidates as product.ball.correct, stride 1: coreml "
        "576, tile 1024, overlap 0.25, player conf 0.5, ball conf 0.05, merge "
        "detectors._merge 0.55 (N70 M0); the oracle applies the shape gates "
        "only (no calibration)", 7649,
        lambda: (_ball()["pooled"]["oracle/s1"]["rate"], _ball()["pooled"]["oracle/s1"]["n"]),
        IN_BALL_GT, CODE_BALL),
    # --- ROADMAP A: the oracle-substitution experiment --------------------
    # `state-and-roadmap` (not published) A carries these as literals under an
    # `api:ok` waiver pointing at a json. That is `+0.024`'s exact shape: a
    # number with no id, quoted in a table, read onward. They are ids now.
    #
    # Read `_calib()` for the smoothing that had to be pinned to get here.
    "calib.oracle.map_error.shipped": Spec(
        "the SHIPPED pixel->metre map's own error, on GT feet held out in time "
        "from the oracle's fit. A dict, deliberately: the median alone reads "
        "like a working map, and it is the p90 and the within-2m share that "
        "explain why recall in metres collapses - see model.role.* for the "
        "same argument", "m/m/frac",
        "118575, 13,200 GT feet held out in time, raw-panorama MOT feet; 'shipped' "
        "map = the job's calibration.json, a REMAP built from ST2's own "
        "homography+mapx chain - not an operator-clicked client map",
        13200, lambda: _calib_map_error("shipped"), IN_CALIB_HYP,
        CODE_CALIB_HYP),
    "calib.oracle.map_error.oracle": Spec(
        "the same three numbers for the ORACLE map on the same held-out feet. "
        "COUNTERFACTUAL, not the product. The annotation it is fitted and "
        "scored against is quantised at 1.05 x 0.68 m, so this is a ceiling on "
        "the ceiling", "m/m/frac",
        "118575, 13,200 GT feet held out in time, ORACLE MAP - not shippable",
        13200, lambda: _calib_map_error("held_out"), IN_CALIB_HYP,
        CODE_CALIB_HYP),

    # The 21-point gap, one set of rows in three spaces. NOT the same as
    # product.detect.recall_px/recall_m: those are detsweep's, n=880, and their
    # two gates are not each other's image - separating that is what this
    # harness is for. Same rows, same frames, exclusive nearest-first in every
    # column.
    "calib.oracle.recall_m.shipped": Spec(
        "share of GT players with one of OUR detections within 2 m under the "
        "SHIPPED map", "recall",
        "118575, 5,280 GT player-instants, every 25th clip frame, 2 m gate; "
        "'shipped' map = the job's calibration.json, a REMAP built from ST2's own "
        "homography+mapx chain - not an operator-clicked client map",
        5280, lambda: _calib()["space_recall"]["recall_m_shipped"]["2.0"],
        IN_CALIB_HYP, CODE_CALIB_HYP),
    "calib.oracle.recall_m.oracle": Spec(
        "the same detections, the same 2 m gate, through the ORACLE map. The "
        "confirmation: the image-to-metre recall gap IS calibration. Read "
        "against calib.oracle.recall_px.ceiling, which is what was available",
        "recall",
        "118575, 5,280 GT player-instants, ORACLE MAP - not shippable", 5280,
        lambda: _calib()["space_recall"]["recall_m_oracle"]["2.0"],
        IN_CALIB_HYP, CODE_CALIB_HYP),
    "calib.oracle.recall_px.ceiling": Spec(
        "the same rows matched IN THE IMAGE, at the most generous gate swept "
        "(1.0 x player height) and with no calibration anywhere in it - so no "
        "map can beat it. This is the number that shows coverage was already "
        "saturated at the detection limit before any map changed", "recall",
        "118575, 5,280 GT player-instants, IMAGE SPACE, 1.0 h gate", 5280,
        lambda: _calib()["space_recall"]["recall_px"]["1.0"], IN_CALIB_HYP,
        CODE_CALIB_HYP),

    # Distance. The shipped side of this table is NOT re-registered: it comes
    # out of the same scorer and reproduces product.distance.ratio_raw and
    # product.distance.coverage to the digit. Registering it twice would be the
    # copy step this layer exists to delete.
    "calib.oracle.distance.ratio_raw": Spec(
        "squad distance through the ORACLE map over full GT distance, same "
        "rows, same ids, same estimator - only the map changes. HULL-GATED "
        "since b9a2106 (the relabel feeding the distance rows ONLY - "
        "map_error, recall_m and neutrality still call the oracle ungated): "
        "outside the training feet's hull the shipped map "
        "answers. Ungated it read 1.046 and was published as 'the shipped map "
        "compresses path length by 23%' - hull extrapolation, retracted. "
        "Refused unless the neutrality control holds", "ratio",
        "118575 1st half, tracks.split.json, 22 GT players, 2 m gate, "
        "ORACLE MAP - not shippable", 22,
        lambda: _calib_distance("ratio_raw"), IN_CALIB_HYP, CODE_CALIB_HYP),
    "calib.oracle.distance.coverage": Spec(
        "THE REFUTATION. Mean share of a GT player's frames we hold an id on, "
        "with calibration made perfect. It goes DOWN against "
        "product.distance.coverage 0.704, and stays flat across a 1-8 m gate "
        "sweep under both maps: the 30% hole is detection and tracking, not "
        "the map. Registered because a registry that holds only the "
        "confirmations is a registry that lets the refutation rot", "fraction",
        "118575 1st half, tracks.split.json, 22 GT players, 2 m gate, "
        "ORACLE MAP - not shippable", 22,
        lambda: _calib_distance("coverage"), IN_CALIB_HYP, CODE_CALIB_HYP),

    # The two controls. An unregistered control on a registered result is a
    # result nobody can re-check: both of these can invalidate every oracle row
    # above, and neither would be noticed if it moved in a log file.
    "calib.oracle.control.neutrality": Spec(
        "GT's OWN pixel feet through the oracle, summed by the same estimator, "
        "over the distance of the GT metres those pixels are annotated with. "
        "1.00 = the map adds no path length of its own. This bounds "
        "calib.oracle.distance.ratio_raw and gates it: at smoothing 1.0, "
        "main()'s default until 637ff1d, it read 1.094, outside the 0.95-1.05 "
        "band. The oracle is called here WITHOUT the hull gate",
        "ratio", "118575, 22 GT players, GT feet through the oracle", 22,
        lambda: _calib()["oracle"]["distance_neutrality"], IN_CALIB_HYP,
        CODE_CALIB_HYP),
    "calib.oracle.control.ceiling": Spec(
        "GT positions fed in as if they were our tracker's output, through the "
        "same attribution, windowing and summation. Must read 1.00/1.00/1.00 "
        "or the harness cannot show a calibration win even when calibration is "
        "perfect, and no verdict it produces means anything "
        "(postmortem (not published) C2)", "ratios",
        "118575 1st half, GT as prediction, 22 GT players, 2 m gate", 22,
        lambda: {k: _calib()["ceiling"][k]
                 for k in ("coverage", "ratio_raw", "ratio_matched")},
        IN_CALIB_HYP, CODE_CALIB_HYP),

    "product.touch.f1.contact.gtball": Spec(
        "the ball-based owner rule with a PERFECT ball and perfect tracks, F1 "
        "on CONTACT at 1.0 s. The comparator that decides the roadmap against "
        "product.possession.f1.contact, a contact rule needing no ball input", "F1",
        "118575 half 1 entire, GT ball + GT tracks, near_m 1.5 CHOSEN FROM A "
        "5-POINT SWEEP on this same half - tuned, not held out", 1020,
        lambda: _touch()["rows"]["CONTACT.1.5.1.0"]["f1"], IN_TOUCH, CODE_TOUCH),
}



if __name__ == "__main__":
    # The GT stride is a stage_gt PARAMETER. As a st2_mot module global it was
    # set to 2 by the N6 ids and inherited by st2_ab.score, which then scored
    # three A/B verdicts on half the frames with valid cache keys.
    import inspect
    import st2_mot as _T
    assert not hasattr(_T, "GT_EVERY"), "the GT stride is module state again - it leaks"
    assert inspect.signature(_T.stage_gt).parameters["every"].default == 1

    # The one branch in this file that can suppress a published number: the
    # neutrality gate on the oracle distance rows. Planted both ways, because a
    # gate that cannot refuse is `postmortem (not published)` A - a self-check that
    # could not fail. 1.094 is not a made-up number: it is what
    # `calib_hypothesis.main()` produced at smoothing 1.0, its default until 637ff1d.
    _real = _calib
    for _neu, _should_pass in ((0.990, True), (1.094, False)):
        _calib = lambda n=_neu: {"oracle": {"distance_neutrality": n},
                                 "oracle_run": {"ratio_raw": 1.046}}
        try:
            _got, _refused = _calib_distance("ratio_raw"), False
        except RuntimeError:
            _got, _refused = None, True
        assert _refused != _should_pass, (_neu, _got, _refused)
    _calib = _real
    print(f"ok  ({len(MEASUREMENTS)} measurements; the neutrality gate admits "
          f"0.990 and refuses 1.094)")
