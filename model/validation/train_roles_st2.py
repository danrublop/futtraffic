"""The licence-clean replacement for `role_clf.pt`: ST2 crops, and TWO classes.

    python model/validation/train_roles_st2.py --selfcheck   # the protocol assertions
    python model/validation/train_roles_st2.py --probe       # eyeball the GK labels
    python model/validation/train_roles_st2.py --build       # cut crops -> out/roleclf_st2
    python model/validation/train_roles_st2.py --train       # 7.6 min, 3 seeds
    python model/validation/train_roles_st2.py --show        # the result against the bar

WHAT THIS REPLACES AND WHY
    `model/models/role_clf.pt` is licence blocker B3
    (licence-audit (not published) §7.3, row 3 of §8): its training
    chain runs back to the DFL Bundesliga Data Shootout under a competition-use-only
    grant with a delete-all-copies clause (dfl-dataset (not published)).
    `model/validation/train_roles.py` is the script that trained it; its protocol -
    whole-clip split, three-way, three seeds, sealed test set, run record - is good
    and is REUSED HERE UNCHANGED. `--train` rebinds that module's constants and calls
    its `train()`. Only the data and the class list are new.

    Nothing here touches `train_roles.fetch`, the deleted `role_crops.npz`, or
    `role_clf.pt`. The old weights are not a teacher and not an initialisation: the
    ImageNet weights torchvision ships are the only prior, exactly as before.

THE HEADLINE FINDING, AND IT IS BAD NEWS
    **ST2 has no referees.** Not in the MOT ground truth (exactly 22 ids per frame,
    all ten matches), not in the dense tracker XML, not in the roster, and not in the
    GSR `role` field either - cited from licence-audit (not published)
    APPENDIX scanned `gsr/128057/128057_1st.json` whole and got player 1,353,000 /
    goalkeeper 135,300 / **referee 0**. The referee IS on the pitch and our detector
    finds him; he is simply not annotated.

    `role_clf.pt` is a player / goalkeeper / **referee** classifier and the referee
    is the class it exists for: `model/engine/rolenet.py`'s docstring says so ("this
    asks what a person LOOKS like ... and it is the one that answers the referee"),
    and `model/engine/track.py` calls it so a confident referee vote can override kit
    colour. **So this replacement cannot reproduce the surface it replaces.** It is
    two classes. Referee recall is not lowered here, it is UNMEASURABLE, and a
    two-class model shipped in a three-class socket is a product decision, not a
    retrain - see BAR.

    This is research-data-hunt (not published) §5.3 taken
    literally ("Ship player / goalkeeper only ... **Do this first.** It decouples
    'get the DFL-derived model out of the tree' from 'solve refereeing'"). §5.1-5.2
    cost the referee class at 2-3 person-days of *verification* (an on-pitch track
    that matches no ST2 id is by construction a candidate official) and that work is
    NOT done here.

WHERE THE LABELS COME FROM, AND WHAT THEY COST TO BELIEVE
    Not from the GSR files - those are 2.7 GB per half, 20 of them, and their
    annotation canvas is 3840x1504 for every match while the clips we hold are
    4096x1080, 3840x1906 and 3840x1504. A canvas mismatch is how this repo got its
    9.35 m baseline (`detect-on-the-raw-panorama-not-the-vendor-remap` (internal note)).
    Everything below is already on disk and already aligned:

      boxes   `~/Downloads/st2/mot/<match>.txt`, MOT format, 22 ids x ~6,000 frames,
              in the pixel space of `~/Downloads/st2/mot/clips/<match>.mp4`, which is
              frame-for-frame the same length. `st2_tiles.py` already cuts tiles this
              way; `--probe` re-proves it by eye.
      id      `model/validation/out/st2_team_gt.json`'s `mot_to_player`, the solved
              permutation `st2_team_gt.py` writes.
      role    `position="GK"` on `<player>` in
              `~/Downloads/st2/raw/<m>/<m>_tracker_box_metadata.xml`. Exactly two per
              match, every match. Read through `st2_team_gt.roster()`, which owns
              that file's parse and carries @position since 2026-09-20.

    IT IS A PROXY - a squad-registration field, not a per-frame observation - so it
    is measured rather than assumed. Against the dataset's own GSR `role` field,
    which is an INDEPENDENT annotation of the same ten matches:

    @M agree 88,of 88 :: .venv/bin/python model/validation/train_roles_st2.py --gsr-check  #slow
       88 of 88 player ids agree, 0 disagreements, over ~880 annotation records in
       4 matches (117092, 118575, 128057, 132877), first-half GSR, 400 KB window
       per match at byte 100,000,000. Nothing was dropped: every id present in the
       window is scored. Bench goalkeepers (117092 registers 4 GKs, 128057 three)
       never enter the joined 22, so they cannot leak in.

    So the residual risk is NOT the position field, it is the PERMUTATION. If
    `mot_to_player` assigns a goalkeeper id to an outfield track, that track is
    wrong for its whole length, and two tracks per match carry the entire
    goalkeeper class.

    @M 0.8890,0.9663 :: .venv/bin/python model/validation/train_roles_st2.py --selfcheck
       min and max `frame_agreement` over the ten matches in
       model/validation/out/st2_team_gt.json, printed with cost2_px/cost_px
       (second-best permutation against best, 3.4-11.1) beside it. No threshold is
       asserted on either: nobody has calibrated one, and an uncalibrated gate is
       decoration.

    `--probe` is the check that settles it, because it is the artefact and not the
    keys (`reproduced-means-diffed-against-the-golden` (internal note)): one montage,
    every match's two goalkeeper crops beside two outfield crops. Run on
    2026-09-20, all ten rows show both goalkeepers in a kit distinct from BOTH
    outfield kits. That is the label chain verified end to end by eye, ten for ten.

THE SPLIT IS BY MATCH AND THE KITS STILL LEAK - SAY SO
    `train_roles.py`'s rule is whole clips, three ways, nothing shared. Here whole
    MATCHES, three ways. But ST2's ten matches are nine university sides playing each
    other, and a kit classifier's unit of generalisation is the KIT, not the match:

        117092  C1 vs B          118577  Medical vs Football Society
        117093  Tsukuba vs B     118578  B vs C1
        118575  C3 vs Soc.Soc    128057  TSC vs Sakushin
        118576  B vs C3          128058  B1 vs B2
                                 132831  Chuo Gakuin vs C1
                                 132877  B2 vs Matsudo City

    Derived: build the graph whose nodes are teams and whose edges are matches: **nine of
    the ten matches are one connected component** (only 128057 is isolated), so a
    team-disjoint three-way split DOES NOT EXIST on this dataset. Claiming one would
    be `landmark-holdout-is-covariate-shift` (internal note) a second time.

    So the rule is stated as what it is, pre-registered, and re-derived in
    `_selfcheck` so editing the tuples without editing the rule fails:

      RANK each match by how many OTHER matches share one of its two teams
      (ascending, ties by ascending match id). TEST = ranks 1-3, VAL = ranks 4-5,
      TRAIN = the rest. It is a property of the fixture list and of nothing that
      was scored.

        rank  match   shared-with  ->  split
          1   128057       0           TEST
          2   118577       1           TEST
          3   132877       1           TEST
          4   118575       2           VAL
          5   132831       2           VAL
          6   117093       4           TRAIN
          7   118576       5           TRAIN
          8   128058       5           TRAIN
          9   117092       6           TRAIN
         10   118578       6           TRAIN

    One of the six TEST teams (B2, in 128058) appears in TRAIN; the other five do
    not. Report that number with the score or the score is a rate over an unnamed
    population.

    132831 and 132877 have mirrored position feeds (`st2-132877-feed-is-mirrored` (internal note),
    `dist_eval.stc_gt` refuses them). Crops are image space and identity/team are fine
    there, so they are usable here - recorded because the next reader will ask.

PRE-REGISTERED BEFORE ANY TRAINING RUN (commit 1 of this branch, nothing scored yet)
    split     whole matches, the rank rule above. Never frames, never halves.
    metric    per-class recall on the TEST matches, **min / median / max over seeds
              0 1 2**, median is the headline - the same form `role_clf.run.json`
              publishes, so the two are comparable on the one class they share.
    bar       @C 0.8403,0.8811 :: model/models/role_clf.run.json test.recall
              goalkeeper and player recall, medians of 3 seeds, 4 held-out DFL clips,
              3,081 crops. The replacement must MATCH goalkeeper
              recall 0.840. Below it, the licence argument alone does not justify the
              swap, because the thing being swapped in is worse at its remaining job.
    and       referee recall is **WITHDRAWN, not regressed**. No ST2 split can
              produce it. The shipped surface reads 0.8429 (cited on the same run record) and
              it must be struck
              from README, not re-quoted against this model.
    drop      every box that `rolenet.crop` refuses, every undecodable frame, and the
              sampling stride, reported per match in the build manifest. A metric that
              rose because short boxes were filtered out is indistinguishable from a
              real gain without it (docs/README.md#working-rules).
    n         this is ONE configuration, not an argmax over a sweep. If a sweep
              happens the number carries its `n=`.

    THE HONEST RISK, WRITTEN DOWN BEFORE THE RUN: the goalkeeper class here has
    **20 distinct appearances** - two men per match, ten matches - however many crops
    are cut from them. The DFL model had 169 goalkeeper crops from 11 clips and
    scored 0.840. More crops off the same twenty tracks is not more signal, and the
    TEST matches contribute six of those twenty. A miss is a plausible outcome and
    the pre-registered response is to report it, not to re-split.

CROP GEOMETRY IS `rolenet.crop` AND NOTHING ELSE
    Built through `rolenet.crop`, not a copy of it, for the reason `train_roles.build`
    gives: a training crop padded or resized differently from the inference crop is a
    silent accuracy loss that looks like a bad model. One caveat this data adds:

    @M 12,12/ 34,12/34/ 46,12/34/46/ 74,168 :: .venv/bin/python model/validation/train_roles_st2.py --selfcheck
       source box height in px, min/p10/p50/p90/max, pooled over all 26,400 crops of
       the built set (stride 50, all ten matches, nothing dropped). A 46 px box is
       upscaled 2.1x into rolenet's 96 px crop. The model this replaces was trained
       on 1080p broadcast crops, which is one more reason no number carries across.
       Held fixed: stride=50, the ten mot/clips videos, rolenet W,H,PAD = 48,96,0.1.

THE RESULT: THE KEEPER CLASS MISSES THE BAR. THE PLAYER CLASS DOES NOT.
    RUN 2026-09-20, NOT RE-SPLIT.
    The rule above was committed in 1a4e805 before anything was trained. This is
    what it produced, reported against the bar it named, with the population beside
    it. One configuration. Nothing was re-split, re-striden or re-tuned afterwards,
    which is the only reason the miss means anything.

    CORRECTED 2026-09-20 after review: this block said "IT MISSES THE BAR, ON BOTH
    CLASSES". Only one of the two deltas clears the bar's OWN seed noise. The bar's
    player recall runs 0.8257 / 0.8811 / 0.8921 across its three seeds - a spread of
    0.0664, WIDER than the -0.0551 below, and its worst seed sits under this run's
    0.8260. The player row is reported as indistinguishable from seed noise, not as a
    miss. The goalkeeper row stands: the bar's worst seed is 0.8151 and this run is
    0.5889, far outside it. Both spreads are in role_clf.run.json test.recall.

    @M 0.5889,0.8260,0.8044,0.7074 :: .venv/bin/python model/validation/train_roles_st2.py --show
       goalkeeper recall / player recall / accuracy / macro recall, MEDIAN of
       seeds 0 1 2, on the 3 pre-registered TEST matches (118577, 128057, 132877),
       7,920 crops = 720 goalkeeper + 7,200 player, test opened once per seed,
       epoch chosen on the 2 VAL matches. Stride 50, nothing dropped (drop share
       0.0 on every match). n=1 configuration tried.

        metric          bar median (spread)        this run, median   delta   reading
        goalkeeper  0.8403 (0.8151-0.8908)             0.5889        -0.2514  MISSES
        player      0.8811 (0.8257-0.8921)             0.8260        -0.0551  within
                                                                              the bar's
                                                                              own spread
        referee     0.8429 (0.8397-0.8622)          UNMEASURABLE        -     no class

       And the two bar medians are different SEEDS (goalkeeper 0.8403 is seed 2,
       player 0.8811 is seed 0), so no single DFL model ever achieved both.

       Spread, not the median, because the median is one draw of three:
        goalkeeper  0.5306 / 0.5889 / 0.6000     player  0.8243 / 0.8260 / 0.8267
        accuracy    0.7976 / 0.8044 / 0.8061     macro   0.6774 / 0.7074 / 0.7133

    POPULATION, stated with the score and not after it: 1 of the 6 TEST teams
    (34317, University of Tsukuba B2) also appears in TRAIN, in 128058. The other
    five are unseen. A team-disjoint split does not exist on ST2 (see above), so
    this is the best obtainable and the overlap is part of the result.

    WHAT THE NUMBERS SAY, AND WHAT THEY DO NOT. Validation macro recall ran
    0.78-0.84 while test macro recall came out 0.68-0.71 - a ~0.12 drop from two
    held-out matches to three others. That is the pre-registered risk arriving
    exactly as written: the goalkeeper class has twenty distinct appearances in the
    whole dataset, the model is learning *those twenty kits*, and a sixth and
    seventh keeper it has never seen is a different problem. More crops off the
    same twenty tracks cannot fix it; more MATCHES might, and ST2 has no more.

    It does NOT say the DFL model is better at goalkeepers in the product. The two
    numbers are on different footage (1080p broadcast vs a 4K fixed panorama where
    the median box is 46 px), different splits and different class counts, and
    `licence-audit` (not published) §7.3 said in advance not to carry numbers across. The
    honest reading is narrower: **on the only footage we may legally train on, a role
    model of this size reaches GK recall 0.5889 where the shipped one reached 0.84 on
    broadcast crops, there is no referee class at all, and the shipped one cannot
    ship.** The player class is not part of that claim.

    NOT DONE, ON PURPOSE: no epoch sweep, no stride sweep, no rebalanced sampler,
    no longer schedule, no second split. Each of those is a defensible next
    experiment and every one of them, run now, would be selection on the test set
    this number was just taken from (Cawley & Talbot, JMLR 11 (2010)). The
    next configuration needs a fresh pre-registration and its own `n=`.

    @M 454.3 :: .venv/bin/python model/validation/train_roles_st2.py --show
       wall seconds for the whole run: 3 seeds x 8 epochs on 13,200 TRAIN crops,
       MPS, plus val and test inference. The pre-run estimate was ~6 min; it
       took 7.6, with another session's api.py run competing for the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rolenet  # noqa: E402
import st2_team_gt as G  # noqa: E402 - the roster read (ids, sides, @position)
import train_roles  # noqa: E402  - reused for the protocol, never for its data

# ST2 lives outside the repo and is not env-configurable; 31 files under
# model/validation/ hardcode this same path and this one does not invent a 32nd
# convention. ponytail: one constant, parameterise it when a second machine needs it.
ST2 = os.path.expanduser("~/Downloads/st2")
JOIN = os.path.join(ROOT, "model", "validation", "out", "st2_team_gt.json")
OUT = os.path.join(ROOT, "model", "validation", "out", "roleclf_st2")
CROPS = os.path.join(OUT, "role_crops_st2.npz")

#: CC BY 4.0 attribution is attribution to a VERSION (train_roles.py's REVISION
#: note, postmortem B6/C4). This sha is what
#: `curl -sL https://huggingface.co/api/datasets/atomscott/soccertrack-v2` returned
#: on 2026-09-20. No sigil: it is an identifier, not a measurement, and
#: train_roles.py's REVISION carries the same comment in the same form.
DATASET = "atomscott/soccertrack-v2"
REVISION = "eae5179377d3b189d9f15c6b4c3a6a6c41a63eaa"
LICENCE = "CC BY 4.0"
ATTRIBUTION = ("SoccerTrack v2 (atomscott/soccertrack-v2), Scott et al., University "
               "of Tsukuba, CC BY 4.0 - MOT boxes and roster metadata")

#: TWO. There is no third class in this data and inventing one is the failure mode
#: this file exists to avoid.
NAMES = ("goalkeeper", "player")

MATCHES = ("117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877")

#: Pre-registered, by the rank rule in the docstring, re-derived in `_selfcheck`.
TEST = ("118577", "128057", "132877")
VAL = ("118575", "132831")

#: Every 50th frame = one crop per track per 2 s. Not 25: at 22 boxes x 240 frames x
#: 10 matches the float32 tensors `train_roles._prep` builds are ~3 GB and an OOM
#: twenty minutes in is a worse failure than half the crops. Neighbouring seconds of
#: one track are near-duplicates anyway - the class has 20 distinct appearances
#: whatever the stride, which is the ceiling the docstring names.
STRIDE = 50

#: Cited from model/models/role_clf.run.json test.recall, median of seeds 0/1/2 on the
#: 4 held-out DFL clips (3,081 crops). The bar, fixed before the run.
BAR = {"goalkeeper": 0.8403, "player": 0.8811,
       "referee": "WITHDRAWN - 0.8429 shipped, unmeasurable on ST2"}

#: The bar's OWN across-seed spread, same artefact, test.recall[k] min/max. The bar
#: is a median of three draws; comparing a scalar to a scalar hides that the player
#: figure moves 0.0664 between seeds - wider than the delta this run was called a
#: MISS on. Registered rule unchanged (it rules on the median); this is the noise
#: context printed beside it, added 2026-09-20 after review.
BAR_SPREAD = {"goalkeeper": (0.8151, 0.8908), "player": (0.8257, 0.8921),
              "referee": (0.8397, 0.8622)}

PREREG = {
    "written": "2026-09-20",
    "replaces": "model/models/role_clf.pt (licence blocker B3)",
    "classes": list(NAMES),
    "referee": "absent from ST2 at source; not a regression, an unmeasurable metric",
    "split_rule": "whole matches; rank by number of other matches sharing a team "
                  "(asc, ties by match id); TEST = ranks 1-3, VAL = ranks 4-5",
    "test_matches": list(TEST), "val_matches": list(VAL),
    "train_matches": [m for m in MATCHES if m not in TEST and m not in VAL],
    "team_disjoint": False,
    "team_disjoint_why": "9 of 10 matches are one connected component of the "
                         "team-vs-team graph; only 128057 is isolated",
    "metric": "per-class recall on TEST matches, min/median/max over seeds 0 1 2, "
              "median is the headline",
    "bar": BAR,
    "stride": STRIDE,
    "n_configurations": 1,
    "report_drop_share": True,
    "prestated_risk": "the goalkeeper class has 20 distinct appearances (2 per match "
                      "x 10 matches) however many crops are cut; 6 of the 20 are in "
                      "TEST. A miss is plausible and is to be reported, not re-split.",
}


# ---------------------------------------------------------------- labels

def roster_gk(match: str) -> set[str]:
    """ST2 player ids whose squad position is GK, for one match.

    `st2_team_gt.roster` reads this file, and since 2026-09-20 it carries
    @position (st2_cache.roles needs it too). This was a second, regex parse of
    the same three attributes, written when roster() dropped them; one parse of
    one vendor file is the rule. Proven equal to the regex on all ten matches
    before the swap, both here and for `teams`.
    """
    return {str(pid) for pid, v in G.roster(match)["players"].items()
            if v["position"] == "GK"}


def teams(match: str) -> set[str]:
    """The two team ids of one match, from the same roster read."""
    return set(G.roster(match)["sides"])


def gk_mot_ids(match: str, join: dict) -> set[str]:
    """MOT track ids that are a goalkeeper, via the solved id permutation."""
    gk = roster_gk(match)
    return {mid for mid, pid in join[match]["mot_to_player"].items() if str(pid) in gk}


def split_rank() -> list[tuple[str, int]]:
    """The pre-registered rule, computed rather than remembered."""
    t = {m: teams(m) for m in MATCHES}
    return sorted(((m, sum(bool(t[m] & t[o]) for o in MATCHES if o != m))
                   for m in MATCHES), key=lambda r: (r[1], r[0]))


def boxes_by_frame(match: str) -> dict[int, list[tuple[str, float, float, float, float]]]:
    out: dict[int, list] = {}
    with open(f"{ST2}/mot/{match}.txt") as f:
        for line in f:
            p = line.split(",")
            if len(p) < 6:
                continue
            out.setdefault(int(float(p[0])), []).append(
                (p[1], *(float(v) for v in p[2:6])))
    return out


def _sha(path: str) -> str:
    """sha256 of a file - `hashlib.file_digest` (3.11+) streams it."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ---------------------------------------------------------------- build

def build(stride: int = STRIDE, matches: tuple[str, ...] = MATCHES,
          out: str = CROPS, limit_frames: int | None = None) -> dict:
    """Every sampled MOT box -> one crop at rolenet's own geometry, plus a manifest.

    Decoding is sequential with `grab()` on the skipped frames: a `CAP_PROP_POS_FRAMES`
    seek per sample re-decodes from the nearest keyframe and is ~20x slower.
    """
    t0 = time.time()
    join = json.load(open(JOIN))
    X, y, clip, box_h = [], [], [], []
    man: dict = {"matches": {}, "stride": stride, "classes": list(NAMES)}

    for m in matches:
        gk = gk_mot_ids(m, join)
        assert len(gk) == 2, f"{m}: {len(gk)} goalkeeper tracks, expected 2"
        by_frame = boxes_by_frame(m)
        want = sorted(f for f in by_frame if f % stride == 0)
        if limit_frames:
            want = want[:limit_frames]
        cap = cv2.VideoCapture(f"{ST2}/mot/clips/{m}.mp4")
        kept = Counter()
        dropped_nocrop = dropped_nodecode = 0
        heights: list[float] = []
        pos, wanted = 0, set(want)
        last = max(want) if want else -1
        while pos <= last:
            if pos in wanted:
                ok, im = cap.read()
                if not ok:
                    dropped_nodecode += len(by_frame[pos])
                    pos += 1
                    continue
                for mid, x, yy, w, h in by_frame[pos]:
                    c = rolenet.crop(im, (x, yy, x + w, yy + h))
                    if c is None:
                        dropped_nocrop += 1
                        continue
                    name = "goalkeeper" if mid in gk else "player"
                    X.append(c)
                    y.append(NAMES.index(name))
                    clip.append(m)
                    box_h.append(h)
                    kept[name] += 1
                    heights.append(h)
            elif not cap.grab():
                dropped_nodecode += len(by_frame.get(pos, ()))
            pos += 1
        cap.release()
        hs = np.array(heights) if heights else np.zeros(1)
        man["matches"][m] = {
            "frames_sampled": len(want), "gk_mot_ids": sorted(gk),
            "n_per_class": dict(kept),
            "dropped_crop_none": dropped_nocrop,
            "dropped_decode": dropped_nodecode,
            "drop_share": round((dropped_nocrop + dropped_nodecode)
                                / max(1, sum(kept.values()) + dropped_nocrop
                                      + dropped_nodecode), 5),
            "box_h_px": {k: round(float(np.percentile(hs, p)), 1)
                         for k, p in (("min", 0), ("p10", 10), ("p50", 50),
                                      ("p90", 90), ("max", 100))},
            "split": "test" if m in TEST else "val" if m in VAL else "train",
            "sources": {f"mot/{m}.txt": _sha(f"{ST2}/mot/{m}.txt")[:16],
                        f"raw/{m}/{m}_tracker_box_metadata.xml":
                            _sha(f"{ST2}/raw/{m}/{m}_tracker_box_metadata.xml")[:16]},
        }
        print(f"  {m} {man['matches'][m]['split']:5} {sum(kept.values()):6} crops "
              f"(gk {kept['goalkeeper']}) dropped {dropped_nocrop + dropped_nodecode}",
              flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # `h` is the SOURCE box height in panorama pixels, kept because every crop is
    # resized to 96 and the scale it came from is then unrecoverable - and the
    # scale is the caveat the licence audit asked to check before promising numbers.
    np.savez_compressed(out, X=np.array(X, np.uint8), y=np.array(y, np.int64),
                        clip=np.array(clip), revision=np.array(REVISION),
                        names=np.array(NAMES), h=np.array(box_h, np.float32))
    man.update(n=len(X), wall_s=round(time.time() - t0, 1), crops=out,
               crops_sha256=_sha(out), dataset=DATASET, revision=REVISION,
               licence=LICENCE, attribution=ATTRIBUTION, prereg=PREREG,
               crop_hw=[rolenet.H, rolenet.W], crop_pad=rolenet.PAD)
    with open(os.path.join(OUT, "build.manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"{len(X)} crops from {len(set(clip))} matches -> {out}")
    return man


#: Matches the GSR cross-check reads, and the byte window it reads from each. Any
#: 400 KB of the annotation stream holds ~10 consecutive frames = all 22 ids; the
#: offset is past the `images` array and inside `annotations`.
GSR_CHECK = ("117092", "118575", "128057", "132877")
GSR_URL = ("https://huggingface.co/datasets/atomscott/soccertrack-v2/resolve/main/"
           "gsr/{m}/{m}_1st.json")
GSR_RANGE = (100_000_000, 100_400_000)
_GSR_RE = re.compile(r'"role": "(\w+)",\s*\n\s*"jersey": "\w*",\s*\n\s*'
                     r'"team": "\w+",\s*\n\s*"player_id": (\d+)')


def gsr_check() -> tuple[int, int]:
    """Bound the proxy: does `position="GK"` agree with ST2's own `role` field?

    The roster position is a squad registration, not an observation of who kept
    goal. The GSR annotations carry `role` per box and are an INDEPENDENT
    annotation of the same matches, so they price the proxy instead of vouching
    for it. 400 KB per match is range-requested, not the 2.7 GB file, and cached
    under out/ so the tag re-runs offline afterwards.
    """
    import urllib.request  # noqa: PLC0415 - only this path needs it

    tok = os.path.expanduser("~/.cache/huggingface/token")
    cache = os.path.join(OUT, "gsr_cache")
    os.makedirs(cache, exist_ok=True)
    ok = bad = 0
    for m in GSR_CHECK:
        p = os.path.join(cache, f"{m}_1st.chunk")
        if not os.path.exists(p):
            if not os.path.exists(tok):
                raise SystemExit(
                    "no HuggingFace token at ~/.cache/huggingface/token and no cached\n"
                    f"chunk at {p}. The dataset is CC BY 4.0 and gated:auto - accept\n"
                    "the terms once at https://huggingface.co/datasets/"
                    "atomscott/soccertrack-v2")
            req = urllib.request.Request(
                GSR_URL.format(m=m),
                headers={"Authorization": f"Bearer {open(tok).read().strip()}",
                         "Range": f"bytes={GSR_RANGE[0]}-{GSR_RANGE[1]}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                f.write(r.read())
        gk = roster_gk(m)
        seen: dict[str, set[str]] = {}
        with open(p, encoding="utf-8", errors="ignore") as f:
            for role, pid in _GSR_RE.findall(f.read()):
                seen.setdefault(pid, set()).add(role)
        for pid, roles in seen.items():
            want = "goalkeeper" if pid in gk else "player"
            if roles == {want}:
                ok += 1
            else:
                bad += 1
                print(f"  DISAGREE {m} {pid}: GSR {sorted(roles)}, roster says {want}")
        print(f"  {m}: {len(seen)} player ids, {sum(1 for r in seen.values() if r == {'goalkeeper'})} "
              f"goalkeeper by GSR, roster GK in the joined 22 "
              f"{sorted(gk & set(seen))}")
    print(f"player ids agree {ok} of {ok + bad} between roster position and GSR role")
    return ok, ok + bad


def show() -> dict:
    """The run record's headline, against the bar, with the population beside it.

    Exists so the docstring's numbers are a command and not a memory. It reads the
    record; it never re-trains, so quoting it costs nothing and cannot drift from
    what was actually run.
    """
    rec = os.path.join(OUT, "role_clf_st2.run.json")
    if not os.path.exists(rec):
        raise SystemExit(f"nothing trained yet: {rec}")
    r = json.load(open(rec))
    t = r["test"]
    print(f"TEST {r['split']['test_clips']}  n={t['n']} crops  "
          f"{t['n_per_class']}  opened {r['test_opens']}x (once per seed)")
    print(f"TRAIN {r['split']['train_clips']}   VAL {r['split']['val_clips']}")
    print(f"{'metric':18}{'bar':>10}{'barspread':>17}{'min':>9}{'median':>9}"
          f"{'max':>9}{'delta':>9}")
    for k in NAMES:
        s = t["recall"][k]
        lo, hi = BAR_SPREAD[k]
        # The registered rule rules on the median (MISS). The reading beside it
        # asks the question the median cannot: is the gap bigger than the bar's
        # own seed-to-seed movement? If not, the two runs are not separable.
        read = "MISSES the bar's worst seed" if s["median"] < lo else \
               "inside the bar's own seed spread - not separable"
        print(f"{k + ' recall':18}{BAR[k]:10.4f}{f'({lo:.4f}-{hi:.4f})':>17}"
              f"{s['min']:9.4f}{s['median']:9.4f}"
              f"{s['max']:9.4f}{s['median'] - BAR[k]:+9.4f}"
              f"   {'MISS' if s['median'] < BAR[k] else 'meets bar'}: {read}")
    for k in ("accuracy", "macro_recall"):
        s = t[k]
        print(f"{k:18}{'-':>10}{s['min']:9.4f}{s['median']:9.4f}{s['max']:9.4f}")
    print(f"referee recall     {BAR['referee']}")
    # The kit overlap is part of the score, not a footnote to it.
    tr = set().union(*(teams(m) for m in r["split"]["train_clips"]))
    te = set().union(*(teams(m) for m in r["split"]["test_clips"]))
    print(f"population: {len(te & tr)} of {len(te)} TEST teams appear in TRAIN "
          f"{sorted(te & tr)}; a team-disjoint split does not exist on ST2")
    print(f"n=1 configuration tried.  wall_s {r['wall_s']}  device {r['device']}")
    return r


def probe(frame: int = 2500) -> str:
    """One montage of every match's two goalkeeper crops, beside two outfield ones.

    The permutation is the only unmeasured link in the label chain, and a goalkeeper
    id pointing at an outfield track is visible in one look. This is the check
    `reproduced-means-diffed-against-the-golden` (internal note) asks for: look at the
    artefact, do not trust that the keys were valid.
    """
    join = json.load(open(JOIN))
    rows = []
    for m in MATCHES:
        gk = gk_mot_ids(m, join)
        bf = boxes_by_frame(m)
        f = min(bf, key=lambda k: abs(k - frame))
        cap = cv2.VideoCapture(f"{ST2}/mot/clips/{m}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        cap.release()
        assert ok, f"{m}: frame {f} did not decode"
        pick = ([b for b in bf[f] if b[0] in gk] + [b for b in bf[f] if b[0] not in gk])[:4]
        cs = [rolenet.crop(im, (b[1], b[2], b[1] + b[3], b[2] + b[4])) for b in pick]
        cs = [c for c in cs if c is not None]
        rows.append(np.hstack(cs + [np.zeros((rolenet.H, rolenet.W, 3), np.uint8)]
                              * (4 - len(cs))))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"probe_gk_all10_f{frame}.png")
    cv2.imwrite(p, np.vstack(rows))
    print("rows are " + ", ".join(MATCHES))
    print("columns: GK, GK, outfield, outfield  ->", p)
    return p


# ---------------------------------------------------------------- train

def train(**kw):
    """Reuse `train_roles.train` verbatim by rebinding its data constants.

    Explicit and loud rather than clever: that module is the audited protocol
    (seeded, sealed test set, run record with assertions) and forking it would fork
    the assertions too. What changes is the data, the class list and the attribution.
    """
    train_roles.NAMES = list(NAMES)
    train_roles.CROPS = CROPS
    train_roles.HOLD_OUT = TEST
    train_roles.VAL = VAL
    train_roles.DATASET = DATASET
    train_roles.LICENCE = LICENCE
    train_roles.ATTRIBUTION = ATTRIBUTION
    train_roles.OUT = os.path.join(OUT, "role_clf_st2.pt")
    train_roles.RUN = os.path.join(OUT, "role_clf_st2.run.json")
    train_roles.LEAK_AUDIT = {"finding": "not applicable - this model has no leaky "
                                         "predecessor; the DFL audit is in train_roles.py"}
    assert train_roles.NAMES == ["goalkeeper", "player"], "the rebind did not take"
    assert os.path.exists(CROPS), f"build first: {CROPS} does not exist"
    return train_roles.train(argv=sys.argv, **kw)


# ---------------------------------------------------------------- self-check

def _selfcheck() -> None:
    """Fail if the pre-registration drifts from the rule that justified it.

    The GSR cross-check quoted in the docstring, for the next reader:

        curl -sL -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \\
          -r 100000000-100400000 \\
          "https://huggingface.co/datasets/atomscott/soccertrack-v2/resolve/main/\\
gsr/118575/118575_1st.json" | grep -A3 '"role"'

    and compare its `player_id`s against `roster_gk("118575")`.
    """
    # 1. No third class. The whole point of this file is that it does not invent one.
    assert NAMES == ("goalkeeper", "player"), NAMES
    assert "referee" not in NAMES, "ST2 has no referees; a referee class here is a lie"
    assert isinstance(BAR["referee"], str) and "WITHDRAWN" in BAR["referee"]

    # 2. The split is whole matches, three ways, nothing shared, and the hardcoded
    #    tuples still match the rank rule that chose them.
    assert set(TEST).isdisjoint(VAL)
    assert set(TEST) | set(VAL) <= set(MATCHES)
    assert len(TEST) == 3 and len(VAL) == 2
    rank = split_rank()
    assert set(m for m, _ in rank[:3]) == set(TEST), f"TEST drifted from the rule: {rank[:3]}"
    assert set(m for m, _ in rank[3:5]) == set(VAL), f"VAL drifted from the rule: {rank[3:5]}"

    # 3. The kit leak is reported, not hidden. A team-disjoint split does not exist
    #    here and the count of TEST teams seen in TRAIN is part of the result.
    tr = [m for m in MATCHES if m not in TEST and m not in VAL]
    train_t = set().union(*(teams(m) for m in tr))
    test_t = set().union(*(teams(m) for m in TEST))
    leaked = test_t & train_t
    assert PREREG["team_disjoint"] is False
    print(f"  kit leak: {len(leaked)} of {len(test_t)} TEST teams appear in TRAIN "
          f"{sorted(leaked)}")

    # 4. The label chain, end to end, on every match - two goalkeepers, joined,
    #    and no id claimed by both classes.
    join = json.load(open(JOIN))
    for m in MATCHES:
        gk = gk_mot_ids(m, join)
        assert len(gk) == 2, f"{m}: {len(gk)} GK tracks from the join, expected 2"
        assert gk <= set(join[m]["mot_to_player"]), f"{m}: GK id outside the join"
    print("  10/10 matches: exactly 2 goalkeeper tracks from roster position + join")

    # 5. The permutation is the unmeasured link; print its quality rather than
    #    assert a threshold nobody has calibrated.
    print(f"  {'match':8}{'agree':>8}{'cost_px':>9}{'cost2/cost':>12}")
    for m in MATCHES:
        v = join[m]
        print(f"  {m:8}{v['frame_agreement']:8.4f}{v['cost_px']:9.2f}"
              f"{v['cost2_px'] / v['cost_px']:12.1f}")

    # 6. Crop geometry is rolenet's, not a copy, and the build calls it.
    img = np.zeros((200, 200, 3), np.uint8)
    assert rolenet.crop(img, (50, 50, 90, 150)).shape == (rolenet.H, rolenet.W, 3)
    assert "rolenet.crop" in open(os.path.abspath(__file__)).read()

    # 7. The DFL chain stays refused. If this file ever makes train_roles.fetch
    #    reachable, that is the one failure that matters.
    try:
        train_roles.fetch()
    except SystemExit as e:
        assert "REFUSED" in str(e)
    else:
        raise AssertionError("train_roles.fetch() did not refuse")

    # 8. If the crops exist: disjoint matches, right shape, both classes everywhere.
    if os.path.exists(CROPS):
        z = np.load(CROPS)
        te, va = np.isin(z["clip"], TEST), np.isin(z["clip"], VAL)
        tr_m = ~te & ~va
        for a, b in ((te, tr_m), (te, va), (va, tr_m)):
            assert set(z["clip"][a]).isdisjoint(set(z["clip"][b])), "match leaked"
        assert z["X"].shape[1:] == (rolenet.H, rolenet.W, 3), z["X"].shape
        assert len(str(z["revision"])) == 40, "crops built unpinned"
        for nm, msk in (("val", va), ("test", te), ("train", tr_m)):
            for i, n in enumerate(NAMES):
                assert ((z["y"] == i) & msk).sum() > 0, f"no {n} in {nm}"
        print(f"  crops: {len(z['y'])} total, "
              f"{int((z['y'] == 0).sum())} goalkeeper, {int((z['y'] == 1).sum())} player")
        # The scale the crops came from. 96 px of network input over a ~45 px box
        # is a 2.1x upscale, and the model this replaces saw 1080p broadcast crops.
        q = [int(round(np.percentile(z["h"], p))) for p in (0, 10, 50, 90, 100)]
        print(f"  source box height px min/p10/p50/p90/max: {'/'.join(map(str, q))}")
    else:
        print(f"  crops not built yet ({CROPS})")
    print("train_roles_st2.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--show", action="store_true", help="the result against the bar")
    ap.add_argument("--probe", action="store_true", help="GK montage, all 10 matches")
    ap.add_argument("--gsr-check", action="store_true",
                    help="price the roster-position proxy against ST2's own role field")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="frames per match; a tiny sample for proving the path")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.show:
        show()
    elif a.probe:
        probe()
    elif a.gsr_check:
        gsr_check()
    elif a.build:
        # A truncated build is a SAMPLE and must not land on the path the trainer
        # reads, or a proof-of-path run silently becomes the training set.
        build(a.stride, limit_frames=a.limit_frames,
              out=CROPS if not a.limit_frames
              else CROPS.replace(".npz", f".sample{a.limit_frames}.npz"))
    elif a.train:
        train(epochs=a.epochs, seeds=tuple(a.seeds))
    else:
        _selfcheck()
