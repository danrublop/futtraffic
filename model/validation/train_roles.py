"""Train model/engine/rolenet.py's classifier: player / goalkeeper / referee.

    python model/validation/train_roles.py --fetch     # download + build crops (once)
    python model/validation/train_roles.py             # train -> model/models/role_clf.pt
    python model/validation/train_roles.py --selfcheck # the protocol assertions

WHY THE MODEL IS TRAINED HERE RATHER THAN DOWNLOADED
    There is no football detector on the hub that is both class-aware and
    usable. The one 4-class checkpoint carrying an Apache tag
    (`julianzu9612/RFDETR-Soccernet`) is trained on SoccerNet-Tracking, whose
    terms are research-only and which the uploader was not in a position to
    relicense; SoccerNet's own repos are GPL-3.0. Every other football
    checkpoint found is Ultralytics-trained, which is AGPL-3.0 WHATEVER THE
    MODEL CARD SAYS - two of them claim MIT and Apache over YOLO weights. See
    STACK (not published)'s legal traps.

    So the licence-clean path is to train it: torchvision's ResNet-18 (BSD-3,
    ImageNet weights) on `martinjolif/football-player-detection` (CC BY 4.0), a
    re-export of Roboflow's football-players-detection. That dataset is cut from
    the DFL Bundesliga Data Shootout clips, NOT from SoccerNet - which is the
    whole reason it is usable. Attribution is required by CC BY 4.0, and CC BY
    4.0 attribution is attribution to a VERSION: see REVISION below.

THE SPLIT IS BY CLIP, AND THAT IS THE POINT
    The dataset's own train/valid/test split shares every one of its 15 clips
    between all three, so scoring on it measures "another frame of the same
    match, with the same kits, seconds apart". The product question is a match
    the model has never seen, so whole clips are held out instead. The
    difference is not cosmetic: same-clip scoring flatters this task badly,
    because a kit colour memorised from frame 3 answers frame 4.

    That much this file always did, and it is the one piece of hygiene
    `research-training-protocol (not published)` §9 says to keep. What follows is what
    it got WRONG.

THE SPLIT IS THREE-WAY NOW, AND THE OLD NUMBERS ARE WITHDRAWN
    Until 2026-09-10 this script scored the four held-out clips INSIDE the epoch
    loop and reported the last epoch's line. That makes the held-out set a
    VALIDATION set - a set consulted eight times per run, whose numbers were
    then published as a held-out estimate. Whether or not `epochs=8` was chosen
    by reading those lines, the published figure is a selection-adjacent
    statistic on the test set and is optimistically biased.
    `@C` Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent
    Selection Bias in Performance Evaluation*, JMLR 11 (2010) -
    https://jmlr.org/papers/v11/cawley10a.html
    It is `postmortem (not published)` C5 - "the TPS smoothing chosen by looking at the
    scorer, then reported against it" - committed a second time, in the only
    training script in the repo.

    SUPERSEDED, do not quote (last-epoch score on the TEST clips, 8 epochs,
    unseeded torch RNG, unpinned dataset):

        referee recall     0.971      overall accuracy   0.950
        goalkeeper recall  0.613      referee precision  0.799 @0.5 / 0.875 @0.8

    Those numbers are left here rather than deleted because a withdrawn number
    that vanishes cannot be recognised in the seven documents that copied it
    (`postmortem (not published)` B1). They are superseded by the block below.

    The split is now three-way and by clip, 15 clips, 8 / 3 / 4:
      TRAIN  8 clips, 4,220 crops   fits the weights.
      VAL    3 clips, 1,277 crops   picks the epoch, and nothing else.
                       Consulted every epoch, which is what a validation set is
                       FOR. Carved out of the 11 clips that were already
                       training data, so the test set is untouched by the
                       change and the two protocols stay comparable.
      TEST   4 clips, 3,081 crops   the same four as before. Opened ONCE per
                       seed, after the epoch is chosen, and the run record
                       counts the opens.

    The corrected protocol therefore trains on 4,220 crops where the leaky one
    trained on 5,497, so a drop in the headline could be the missing 23% rather
    than the leak. That confound is not argued about below, it is measured: a
    third run applies the OLD protocol to the NEW 8 training clips and separates
    the two. See "WHAT THIS COST" - the answer is that it is almost all the
    missing 23%.

    Why the epoch is picked on macro recall and not accuracy: the loss is class
    weighted (4,777 players against 169 keepers), so selecting on plain accuracy
    would select against the objective the model is trained on. Macro recall is
    the mean of the per-class recalls present in the split.

    Why not leave-one-clip-out over all 15, which the protocol doc recommends
    (§3.4, item 11): LOCO is the right protocol for COMPARING two configurations
    at n=15 paired observations instead of n=4, and this script compares
    nothing - it trains one configuration and reports its number. Adding 15
    folds would multiply an hour of compute to produce the same headline. It is
    the next change, and it is the change to make before any A/B on this model.

THREE SEEDS, SPREAD BEFORE MEAN
    `torch.manual_seed` was never called here. `m.fc = Linear(512, 3)` - the one
    randomly initialised layer - drew from the global unseeded RNG, so this file
    could not reproduce its own docstring, today or in a month. Now one `--seed`
    drives `torch`, `numpy` and `random` together, which is also the right SHAPE
    of variation: `@C` Bouthillier et al., *Accounting for Variance in Machine
    Learning Benchmarks*, MLSys 2021 (https://arxiv.org/abs/2103.03098) find
    that randomising several sources together approaches the ideal estimator
    better than fixing everything and varying one seed. A single-seed number is
    not a result, so the default is three seeds (`--seeds 0 1 2`) and the report
    prints min/median/max BEFORE the mean.

    The seeding is not asserted, it is tested. `python model/validation/train_roles.py
    --seeds 0 1 2` was run three times on 2026-09-10 and produced a
    **byte-identical `model/models/role_clf.pt`** every time, sha256
    `3a497f7d8a776a19...`, with every per-epoch line matching. That is the only real check that a seed reached everything it
    had to, and on this box (MPS, torch 2.13.0) it passes. It is not promised on
    other hardware: the run record carries the device and the library versions
    precisely because a backend swap is a config change (postmortem's
    CoreML-vs-PyTorch catch, 4,045 boxes against 15).

MEASURED, on the four held-out matches (3,081 crops), 3 seeds, epoch chosen on
the validation clips, test opened once per seed. Spread first, then the median.
`model/models/role_clf.run.json` is the authority; this is a copy for readers.

    accuracy            0.828 / 0.875 / 0.889   min / median / max over seeds 0,1,2
    macro recall        0.835 / 0.846 / 0.882
    referee recall      0.840 / 0.843 / 0.862   n = 312
    goalkeeper recall   0.815 / 0.840 / 0.891   n = 119
    player recall       0.826 / 0.881 / 0.892   n = 2,650
    referee precision   0.542 / 0.596 / 0.729   at p>=0.5
                        0.599 / 0.637 / 0.808   at p>=0.8
    goalkeeper called referee: 1 of 119, on every one of the three seeds
                        <- the confusion that would delete a keeper from a squad

    Read the spread, not the median. On accuracy the three seeds cover six
    points and on referee precision they cover nineteen. Any single-seed number
    from this script, old or new, is one draw from that.

WHAT THIS COST, AND WHAT THE LEAK WAS ACTUALLY WORTH
    The headline fell from 0.950 to 0.875 accuracy and 0.971 to 0.843 referee
    recall. Almost none of that is the leak. Three runs, same 3 seeds, same
    3,081 test crops, medians (see LEAK_AUDIT below for the spreads):

      protocol                             train     acc   macro  ref rec  ref prec
      leaky, last epoch  = the published   11 clips  0.940  0.895   0.968    0.917
      leaky, best epoch chosen ON test     11 clips  0.942  0.914   0.958    0.884
      leaky, last epoch                     8 clips  0.875  0.846   0.888    0.574
      CORRECTED, epoch chosen on VAL        8 clips  0.875  0.846   0.843    0.596

    Line 3 against line 4 is the leak with the training data held fixed, and on
    accuracy and macro recall it is **zero to three decimal places** - on
    referee precision the corrected protocol is two points BETTER. The old
    reporting rule was "the last epoch", not "the best epoch", and the last
    epoch turns out not to be systematically the best one. The bias Cawley &
    Talbot describe was available - line 2 shows +0.019 macro recall and +0.084
    goalkeeper recall sitting there for anyone who read the per-epoch lines and
    picked - but the specific rule this script used did not collect much of it.
    Saying so is the point: the honest re-measurement is allowed to exonerate
    the thing it was run to convict. The protocol was still wrong; it was wrong
    in a way that happened not to pay.

    Line 1 against line 3 is where the loss went: **the three clips that had to
    become a validation set.** At 15 clips, buying a validation set costs 23% of
    the training crops and, worse, three of eleven kits - and referee precision,
    which is the metric that decides whether a player gets called an official,
    falls by 34 points on that alone. That is the real price of the fix, and it
    is a price the protocol doc already told us how to avoid: leave-one-clip-out
    (§3.4, item 11) trains on 14 of 15 clips per fold and needs no permanent
    validation split at all. **That is the next change, and roughly an hour of
    laptop compute buys back most of what this one cost.**

    One more thing the seeds settled, and it is the sharpest argument on this
    page for seeding. The superseded docstring reported, from a single unseeded
    run: accuracy 0.950, referee recall 0.971, referee precision 0.799,
    goalkeeper recall 0.613. Re-run the IDENTICAL protocol at three seeds:

        accuracy           published 0.950  vs seeds 0.903-0.948   ABOVE the max
        referee recall     published 0.971  vs seeds 0.917-0.978   near the max
        referee precision  published 0.799  vs seeds 0.859-0.921   BELOW the min
        goalkeeper recall  published 0.613  vs seeds 0.773-0.857   BELOW the min

    One draw, sitting outside the three-seed range on three of four metrics, in
    both directions. Not a measurement - a sample. And a whole paragraph of this
    file's prose ("goalkeeper is weak and is left weak") was written to explain
    0.613, which is noise. `@C` Reimers & Gurevych, EMNLP 2017;
    `@C` Bouthillier et al., MLSys 2021.

    Left standing, and partly overturned. Goalkeeper is still the THINNEST class
    - 169 training crops against 4,777 players - and the argument for tolerating
    it never depended on 0.613: a keeper called a player is harmless, while a
    keeper called a referee would delete him from his squad, and that confusion
    is **1 of 119 on all three seeds**. But "goalkeeper is the weak one" is no
    longer true of recall. At 0.840 median it now sits level with referee
    (0.843) and below player (0.881), because selecting the epoch on macro
    recall selects for exactly the class the weighted loss is fighting for. The
    weakest number in this file is no longer a recall at all - it is **referee
    precision, 0.596 at p>=0.5**, which is `rolenet.vote`'s problem to survive
    and the reason that function requires a majority share over a whole track
    rather than trusting one crop.

WHAT THE RUN RECORD IS FOR
    `research-training-protocol (not published)` §2.2 calls a run record written by the
    training script itself the single highest-value practice available to us: it
    maps to nine of the twenty-two incidents in `postmortem (not published)`. It is one
    JSON file beside the weights, and it carries the resolved config (not argv -
    A4 and A5 were both "the default in the code disagreed with the default in
    the docs", and argv records neither), every seed, the dataset revision, the
    split membership by clip id, n per class per clip, library versions, the
    per-epoch VALIDATION table, wall time, and the hash of the artefact.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))

import rolenet  # noqa: E402

DATASET = "martinjolif/football-player-detection"
# CC BY 4.0 attribution is attribution to a version, and an unpinned
# snapshot_download lets the training data change under us with no signal
# (`research-training-protocol (not published)` §9.3, postmortem B6/C4). This sha is
# what `huggingface.co/api/datasets/<repo>` returned on 2026-09-10.
REVISION = "e8b8cea002692efd74c945fcdad63e729adc5671"
# Model-card fields travel WITH the weights, so a checkpoint that escapes the repo
# still carries its licence chain. Named constants, not an f-string inside `train`,
# because train_roles_st2.py rebinds them: a checkpoint cut from ST2 that claims a
# DFL provenance would be a worse defect than the one this repo is unwinding.
LICENCE = "CC BY 4.0"
ATTRIBUTION = (f"{DATASET} (CC BY 4.0), re-export of Roboflow "
               "football-players-detection, cut from DFL Bundesliga Data Shootout "
               "clips - COMPETITION-USE-ONLY, see dfl-dataset (not published)")
CROPS = os.path.join(ROOT, "model", "validation", "role_crops.npz")
OUT = os.path.join(ROOT, "model", "models", "role_clf.pt")
RUN = os.path.join(ROOT, "model", "models", "role_clf.run.json")

# YOLO class ids in that dataset: 0 ball, 1 goalkeeper, 2 player, 3 referee.
# The ball is dropped - it is not a person and ball.py owns it.
KEEP = {1: "goalkeeper", 2: "player", 3: "referee"}
NAMES = list(KEEP.values())

# TEST. Four whole matches, chosen as the largest clips so the held-out set is
# big enough to mean something (3,081 of 8,578 crops). UNCHANGED from the
# superseded run, deliberately: changing the test set at the same time as the
# protocol would make the old and new numbers incomparable, and the point of
# this change is to say what the leak was worth.
HOLD_OUT = ("538438", "744b27", "42ba34", "a9f16c")

# VALIDATION. The three MEDIAN-sized of the eleven remaining clips - ranks 5, 6
# and 7 of 11 by crop count. The rule is a property of clip SIZE and of nothing
# that was scored, and it is re-derived from the crops in `_selfcheck` so that
# editing this tuple without editing the rule fails.
#   not the three largest  - they carry 2,650 of 5,497 training crops, and
#                            halving the training set to fix the protocol would
#                            confound "what did the leak cost" with "what did
#                            the smaller training set cost".
#   not the three smallest - 554 crops and 22 goalkeepers is too thin a signal
#                            to choose an epoch on macro recall.
# At ranks 5-7: 1,277 crops, 121 referees, 38 goalkeepers, and 4,220 crops left
# to train on.
VAL = ("573e61", "40cd38", "54745b")

# What the leak was worth, measured rather than asserted. Three protocols, the
# same seeds (0,1,2) and the same 3,081 test crops, min/median/max over seeds.
# Reproduce with `python model/validation/train_roles.py --audit-leak`, which is
# `audit_leak()` below - the withdrawn protocol kept RUNNABLE rather than
# described, so this block is a checkable claim and not a memory (postmortem
# §E: a number in prose cannot be re-run). It also rides in the run record.
LEAK_AUDIT = {
    "test_crops": 3081, "test_clips": 4, "seeds": [0, 1, 2],
    "leaky_last_epoch_11clips": {  # what the docstring published, under seeds
        "train_crops": 5497, "accuracy": [0.903, 0.940, 0.948],
        "macro_recall": [0.892, 0.895, 0.914], "referee": [0.917, 0.968, 0.978],
        "goalkeeper": [0.773, 0.815, 0.857],
        "referee_precision_p0.5": [0.859, 0.917, 0.921],
        "referee_precision_p0.8": [0.920, 0.947, 0.959]},
    "leaky_best_epoch_on_test_11clips": {  # the bias that WAS available
        "train_crops": 5497, "accuracy": [0.868, 0.942, 0.948],
        "macro_recall": [0.907, 0.914, 0.932], "referee": [0.955, 0.958, 0.978],
        "goalkeeper": [0.815, 0.899, 0.908],
        "referee_precision_p0.5": [0.872, 0.884, 0.921],
        "referee_precision_p0.8": [0.927, 0.944, 0.959]},
    "leaky_last_epoch_8clips": {  # the leak, with training data held fixed
        "train_crops": 4220, "accuracy": [0.862, 0.875, 0.894],
        "macro_recall": [0.835, 0.846, 0.868], "referee": [0.843, 0.888, 0.897],
        "goalkeeper": [0.714, 0.815, 0.849],
        "referee_precision_p0.5": [0.542, 0.574, 0.578],
        "referee_precision_p0.8": [0.599, 0.621, 0.622]},
    "finding": "leaky vs corrected at the SAME 4,220 training crops is 0.875 vs "
               "0.875 accuracy and 0.846 vs 0.846 macro recall - the leak was "
               "worth ~0 under this script's last-epoch reporting rule. The 6.5 "
               "points lost are the 3 clips that became the validation set. "
               "Leave-one-clip-out (protocol doc item 11) recovers them.",
}

# Every field the run record must carry to be re-runnable a month later
# (`research-training-protocol (not published)` §2.2). _selfcheck asserts on this tuple,
# so the checklist and the check are the same object.
REQUIRED = (
    "git_commit", "git_dirty", "argv", "config", "seeds", "dataset", "revision",
    "crops_sha256", "split", "n_per_class_per_clip", "libraries", "device",
    "epochs_val", "test", "per_seed", "test_opens", "wall_s", "host",
    "artefact_sha256",
)


#: The same refusal ball_train.py carries, for the same chain. Written 2026-09-19,
#: after a review found that `role_data/` (87 MB of raw export) had never been recorded
#: by any DFL sweep and that `role_crops.npz` - deleted 2026-09-09 - was BACK on disk,
#: rebuilt 2026-09-10 by this function. A deletion note is worthless while the code that
#: rebuilds the data has no refusal in it (dfl-dataset (not published), licence-audit (not published) B3).
DFL_REFUSAL = (
    "REFUSED: martinjolif/football-player-detection is a re-export of the DFL Bundesliga\n"
    "Data Shootout clips, whose terms are competition-use-only with delete-all-copies.\n"
    "Every copy on this machine was deleted on 2026-09-19; fetching would re-create them.\n"
    "role_clf.pt is blocker B3 for exactly this reason. The clean replacement trains the\n"
    "same ResNet-18 on ST2 crops (CC BY 4.0) - research-roboflow-and-oss (not published) 4.5.")


def fetch() -> str:
    raise SystemExit(DFL_REFUSAL)
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    d = snapshot_download(repo_id=DATASET, repo_type="dataset", revision=REVISION,
                          local_dir=os.path.join(ROOT, "model", "validation", "role_data"))
    return os.path.join(d, "data")


def build(src: str) -> None:
    """Every labelled person box -> one crop, at rolenet's own crop geometry.

    Built through `rolenet.crop` rather than a copy of it: a training crop that
    is padded or resized differently from the inference crop is a silent
    accuracy loss that looks like a bad model.

    The dataset's own split directories are walked and then IGNORED - the clip
    id in the filename is the only grouping this file trusts, because all 15
    clips appear in all three of the dataset's splits.
    """
    X, y, clip = [], [], []
    for split in ("train", "valid", "test"):
        for ip in sorted(glob.glob(f"{src}/{split}/images/*.jpg")):
            im = cv2.imread(ip)
            ih, iw = im.shape[:2]
            lp = f"{src}/{split}/labels/{os.path.basename(ip).rsplit('.', 1)[0]}.txt"
            if not os.path.exists(lp):
                continue
            for line in open(lp):
                p = line.split()
                if len(p) < 5 or int(p[0]) not in KEEP:
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                c = rolenet.crop(im, ((cx - bw / 2) * iw, (cy - bh / 2) * ih,
                                      (cx + bw / 2) * iw, (cy + bh / 2) * ih))
                if c is None:
                    continue
                X.append(c)
                y.append(NAMES.index(KEEP[int(p[0])]))
                clip.append(os.path.basename(ip).split("_")[0])
    np.savez_compressed(CROPS, X=np.array(X, np.uint8), y=np.array(y, np.int64),
                        clip=np.array(clip), revision=np.array(REVISION))
    print(f"{len(X)} crops from {len(set(clip))} clips @ {REVISION[:8]} -> {CROPS}")


# ---------------------------------------------------------------- run record

def _git() -> tuple[str, bool]:
    """Commit and DIRTY. Dirty matters more: a run from a dirty tree is not
    reproducible and has to say so (postmortem B5). The stamp itself is
    verify_claims.head - full sha here, and GIT_* dropped, which this one did
    not do."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import verify_claims as V  # noqa: PLC0415 - reused, never copied
    return V.head(short=False)


def _sha(path: str) -> str:
    """sha256 of a file, streamed - `hashlib.file_digest` (3.11+) does it."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _seed_all(seed: int) -> None:
    """One seed, three RNGs. Init, shuffling and augmentation vary TOGETHER,
    which is the form Bouthillier et al. measured as the efficient one."""
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # warn_only: several ops used here have no deterministic MPS kernel, and a
    # hard failure would trade a recorded seed for no run at all.
    torch.use_deterministic_algorithms(True, warn_only=True)


class _Sealed:
    """The test split, and a counter of how many times it has been opened.

    This is the mechanical form of "the test set is touched once, at the end".
    It is not a comment asking the next reader to be careful; the count lands in
    the run record and `_selfcheck` asserts it. `research-training-protocol (not published)`
    §3.3 condition 4: touched a bounded number of times, and the count published.
    """

    def __init__(self, *payload):
        self._payload, self.opens = payload, 0

    def open(self):
        self.opens += 1
        return self._payload


# ---------------------------------------------------------------- training

def _prep(a):
    """BGR uint8 crops -> normalised NCHW float. One definition, because the
    leak audit is only a comparison if both arms preprocess identically."""
    import torch  # noqa: PLC0415

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    a = (a[..., ::-1].astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(a.transpose(0, 3, 1, 2).copy())


def _metrics(P, yt, prob=None) -> dict:
    """Every rate with its denominator in the same object (postmortem B2)."""
    import torch  # noqa: PLC0415

    out = {"n": int(len(yt)), "accuracy": round(float((P == yt).float().mean()), 4),
           "recall": {}, "n_per_class": {}}
    for c, name in enumerate(NAMES):
        m = yt == c
        out["n_per_class"][name] = int(m.sum())
        if m.any():
            out["recall"][name] = round(float((P[m] == c).float().mean()), 4)
    out["macro_recall"] = round(float(np.mean(list(out["recall"].values()))), 4)
    # The licence-clean replacement (train_roles_st2.py) rebinds NAMES to two
    # classes, because ST2 annotates no referees at all. Every referee statistic
    # below is then a rate over nothing, and postmortem B2 is that a rate without
    # its denominator must not be printed - so it is ABSENT, not zero.
    if "referee" not in NAMES:
        return out
    ref = NAMES.index("referee")
    if prob is not None:
        for t in (0.5, 0.8):
            call = (prob[:, ref] >= t)
            out[f"referee_precision_p{t}"] = (
                round(float((yt[call] == ref).float().mean()), 4) if call.any() else None)
            out[f"referee_calls_p{t}"] = int(call.sum())
    # The one confusion that costs something: a keeper called a referee is
    # deleted from his squad. A keeper called a player is a player.
    gk, rf = NAMES.index("goalkeeper"), ref
    out["goalkeeper_called_referee"] = int(((yt == gk) & (P == rf)).sum())
    out["goalkeeper_n"] = int((yt == gk).sum())
    return out


def _fit(seed: int, Xtr, ytr, Xva, yva, sealed, cfg, dev) -> tuple[dict, list, dict]:
    """One seed: fit, choose the epoch on VAL, then open the test set once."""
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    _seed_all(seed)
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(512, len(NAMES))
    m = m.to(dev)

    # Class weights, from the TRAIN clips only. 4,777 players against 169
    # keepers: unweighted, the cheapest model that scores 87% is one that
    # answers "player" to everything.
    cnt = np.bincount(ytr.numpy(), minlength=len(NAMES))
    w = torch.tensor(cnt.sum() / (len(NAMES) * np.maximum(cnt, 1)), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w.to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    def predict(Xs):
        m.eval()
        L = []
        with torch.no_grad():
            for i in range(0, len(Xs), 256):
                L.append(m(Xs[i:i + 256].contiguous().to(dev)).softmax(1).cpu())
        p = torch.cat(L)
        return p.argmax(1), p

    rng = np.random.default_rng(seed)
    table, best = [], None
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        m.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        tot = 0.0
        for i in range(0, len(Xtr), 64):
            xb = Xtr[perm[i:i + 64]].contiguous()
            # Horizontal flip is the only augmentation a kit crop plausibly
            # wants: a player is symmetric, and a rotated or colour-jittered
            # kit is a DIFFERENT kit, which is the one thing being learnt.
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3]).contiguous()
            loss = lossf(m(xb.to(dev)), ytr[perm[i:i + 64]].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)

        P, _ = predict(Xva)
        row = _metrics(P, yva)
        # Only VAL keys land in this table. _selfcheck asserts that no key here
        # starts with "test": the absence IS the guarantee that the epoch was
        # not chosen on the test set.
        row = {"epoch": ep + 1, "train_loss": round(tot / len(Xtr), 4),
               "val_accuracy": row["accuracy"], "val_macro_recall": row["macro_recall"],
               "val_recall": row["recall"], "wall_s": round(time.time() - t0, 1)}
        table.append(row)
        print(f"  seed{seed} ep{ep + 1:>2} loss={row['train_loss']:.4f} "
              f"val_macro_recall={row['val_macro_recall']:.3f} "
              f"val_acc={row['val_accuracy']:.3f} {row['wall_s']:.0f}s", flush=True)
        if best is None or row["val_macro_recall"] > best[1]:
            best = (ep + 1, row["val_macro_recall"],
                    {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})

    ep_best, val_best, state = best
    print(f"  seed{seed} epoch chosen on VAL: {ep_best} "
          f"(val macro recall {val_best:.3f}) -> opening TEST once", flush=True)
    m.load_state_dict(state)

    Xte, yte = sealed.open()          # <- the only read of the test split
    P, prob = predict(Xte)
    test = _metrics(P, yte, prob)
    test["selected_epoch"] = ep_best
    test["selected_on"] = "val_macro_recall"
    return test, table, state


def train(epochs: int = 8, lr: float = 3e-4, seeds: tuple[int, ...] = (0, 1, 2),
          argv: list | None = None) -> dict:
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    # Fail here, not after twenty minutes of compute, and not in the run
    # record's assertions at the very end (postmortem D1: notice the wrong
    # SIZE early).
    assert len(seeds) >= 3, f"a single-seed number is not a result: {seeds}"

    t_start = time.time()
    z = np.load(CROPS)  # arrays only; no pickle, so no code path from the file to exec
    X, y, clip = z["X"], z["y"], z["clip"]
    rev = str(z["revision"]) if "revision" in z else "UNPINNED-REBUILD-THE-CROPS"

    te = np.isin(clip, HOLD_OUT)
    va = np.isin(clip, VAL)
    tr = ~te & ~va
    assert not (te & va).any()
    trc, vac, tec = sorted(set(clip[tr])), sorted(set(clip[va])), sorted(set(clip[te]))
    print(f"train {tr.sum()} crops / {len(trc)} clips   "
          f"val {va.sum()} / {len(vac)} clips   "
          f"TEST {te.sum()} / {len(tec)} clips (opened once per seed)", flush=True)

    Xtr, ytr = _prep(X[tr]), torch.from_numpy(y[tr])
    Xva, yva = _prep(X[va]), torch.from_numpy(y[va])
    sealed = _Sealed(_prep(X[te]), torch.from_numpy(y[te]))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    cfg = {"epochs": epochs, "lr": lr, "batch": 64, "weight_decay": 1e-4,
           "arch": "resnet18", "weights": "IMAGENET1K_V1", "crop_hw": [rolenet.H, rolenet.W],
           "select_on": "val_macro_recall", "class_weighted_loss": True,
           "augmentation": "hflip p=0.5 per batch"}

    per_seed, tables, states = [], {}, {}
    for s in seeds:
        test, table, state = _fit(s, Xtr, ytr, Xva, yva, sealed, cfg, dev)
        per_seed.append({"seed": s, **test})
        tables[str(s)] = table
        states[s] = state

    # Spread BEFORE mean. A single-seed number is not a result, and the median
    # seed is the one that ships - not the best, which would be selection on the
    # test set by another route.
    def spread(key, sub=None):
        # .get, not [], for the referee keys _metrics omits when NAMES has no
        # referee in it. An absent metric reports None; it never reports 0.
        v = [(p.get(key, {}).get(sub) if sub else p.get(key)) for p in per_seed]
        v = [x for x in v if x is not None]
        if not v:
            return None
        return {"min": round(min(v), 4), "median": round(float(np.median(v)), 4),
                "max": round(max(v), 4), "mean": round(float(np.mean(v)), 4),
                "n_seeds": len(v)}

    headline = {
        "accuracy": spread("accuracy"),
        "macro_recall": spread("macro_recall"),
        "recall": {n: spread("recall", n) for n in NAMES},
        "referee_precision_p0.5": spread("referee_precision_p0.5"),
        "referee_precision_p0.8": spread("referee_precision_p0.8"),
        "goalkeeper_called_referee": spread("goalkeeper_called_referee"),
        "n": per_seed[0]["n"],
        "n_per_class": per_seed[0]["n_per_class"],
        "scope": f"{len(tec)} held-out clips, opened once per seed after epoch selection",
    }

    # The shipped checkpoint is the MEDIAN seed by test accuracy. Ranking three
    # seeds is not model selection - all three are reported, and the median is
    # named in advance - but it is a test-set consultation and is recorded.
    order = sorted(per_seed, key=lambda p: p["accuracy"])
    ship = order[len(order) // 2]["seed"]

    m = torchvision.models.resnet18()
    m.fc = torch.nn.Linear(512, len(NAMES))
    m.load_state_dict(states[ship])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"state": states[ship], "names": NAMES,
                # Model-card fields travel WITH the weights, so a checkpoint
                # that escapes the repo still carries its licence chain.
                "dataset": DATASET, "revision": rev, "licence": LICENCE,
                "attribution": ATTRIBUTION,
                "train_clips": trc, "val_clips": vac, "held_out": tec,
                "seed": ship, "seeds": list(seeds),
                "intended_use": "role vote over a whole track (rolenet.vote)",
                "out_of_scope": "single-crop decisions; non-broadcast camera geometry",
                "test": headline}, OUT)

    record = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "model/validation/train_roles.py",
        "git_commit": _git()[0], "git_dirty": _git()[1],
        "argv": argv if argv is not None else sys.argv,
        "config": cfg,
        "seeds": list(seeds), "shipped_seed": ship,
        "dataset": DATASET, "revision": rev, "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "crops_sha256": _sha(CROPS),
        "split": {"train_clips": trc, "val_clips": vac, "test_clips": tec,
                  "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                  "rule": "whole clips; test = the 4 largest, val = the 3 largest of "
                          "the remaining 11; the dataset's own splits are ignored "
                          "because all 15 clips appear in all three"},
        "n_per_class_per_clip": {
            c: {n: int(((clip == c) & (y == i)).sum()) for i, n in enumerate(NAMES)}
            for c in sorted(set(clip.tolist()))},
        "libraries": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                      "numpy": np.__version__, "opencv": cv2.__version__,
                      "python": sys.version.split()[0]},
        "device": dev,
        "host": f"{platform.system()} {platform.release()} {platform.machine()}",
        "epochs_val": tables,
        "per_seed": per_seed,
        "test": headline,
        "test_opens": sealed.opens,
        "wall_s": round(time.time() - t_start, 1),
        "artefact_sha256": _sha(OUT),
        "superseded": {
            "accuracy": 0.950, "referee_recall": 0.971, "goalkeeper_recall": 0.613,
            "referee_precision_p0.5": 0.799, "referee_precision_p0.8": 0.875,
            "why": "last epoch of a run that scored the TEST clips every epoch; "
                   "torch RNG unseeded; dataset revision unpinned. "
                   "Cawley & Talbot JMLR 2010; postmortem C5."},
        "leak_audit": LEAK_AUDIT,
    }
    with open(RUN, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\ntest opened {sealed.opens}x (once per seed, after selection)")
    for k in ("accuracy", "macro_recall"):
        s = headline[k]
        print(f"  {k:<14} min {s['min']:.3f}  median {s['median']:.3f}  "
              f"max {s['max']:.3f}   (mean {s['mean']:.3f}, {s['n_seeds']} seeds)")
    for n in NAMES:
        s = headline["recall"][n]
        print(f"  recall {n:<11} min {s['min']:.3f}  median {s['median']:.3f}  "
              f"max {s['max']:.3f}   (mean {s['mean']:.3f}, n={headline['n_per_class'][n]})")
    print(f"-> {OUT}  (seed {ship})\n-> {RUN}")
    _assert_record(record)
    return record


# ---------------------------------------------------------------- leak audit

def audit_leak(epochs: int = 8, lr: float = 3e-4,
               seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Re-run the SUPERSEDED protocol, so LEAK_AUDIT is a claim and not a memory.

        python model/validation/train_roles.py --audit-leak

    Three configurations, same seeds, same 3,081 test crops:
      11 clips, last epoch  - what the old docstring published
      11 clips, best epoch chosen ON test - the bias that was sitting there
       8 clips, last epoch  - the leak with the training data held FIXED, which
                              is the only line that prices the leak alone

    It deliberately writes no run record and no checkpoint. Its whole method is
    to score the test set every epoch, which is exactly what `_assert_record`
    refuses to accept - the audit is allowed to do it precisely because its
    output is a withdrawal, not a result. `postmortem (not published)`'s own
    what-worked list: keep failed experiments RUNNABLE.
    """
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    z = np.load(CROPS)
    X, y, clip = z["X"], z["y"], z["clip"]
    te = np.isin(clip, HOLD_OUT)
    Xte, yte = _prep(X[te]), torch.from_numpy(y[te])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {"test_crops": int(te.sum()), "test_clips": len(HOLD_OUT), "seeds": list(seeds)}

    for tag, tr in (("11clips", ~te), ("8clips", ~te & ~np.isin(clip, VAL))):
        Xtr, ytr = _prep(X[tr]), torch.from_numpy(y[tr])
        cnt = np.bincount(y[tr], minlength=len(NAMES))
        w = torch.tensor(cnt.sum() / (len(NAMES) * np.maximum(cnt, 1)), dtype=torch.float32)
        runs = []
        for seed in seeds:
            _seed_all(seed)
            m = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = torch.nn.Linear(512, len(NAMES))
            m = m.to(dev)
            lossf = torch.nn.CrossEntropyLoss(weight=w.to(dev))
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
            rng = np.random.default_rng(seed)
            rows = []
            for ep in range(epochs):
                m.train()
                perm = torch.from_numpy(rng.permutation(len(Xtr)))
                for i in range(0, len(Xtr), 64):
                    xb = Xtr[perm[i:i + 64]].contiguous()
                    if rng.random() < 0.5:
                        xb = torch.flip(xb, dims=[3]).contiguous()
                    loss = lossf(m(xb.to(dev)), ytr[perm[i:i + 64]].to(dev))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                m.eval()
                L = []
                with torch.no_grad():
                    for i in range(0, len(Xte), 256):
                        L.append(m(Xte[i:i + 256].contiguous().to(dev)).softmax(1).cpu())
                p = torch.cat(L)
                rows.append(_metrics(p.argmax(1), yte, p))
                print(f"  {tag} seed{seed} ep{ep + 1} TEST acc={rows[-1]['accuracy']:.3f} "
                      f"macro={rows[-1]['macro_recall']:.3f}", flush=True)
            runs.append((rows[-1], max(rows, key=lambda r: r["macro_recall"])))

        for which, idx in (("last_epoch", 0), ("best_epoch_on_test", 1)):
            key = f"leaky_{which}_{tag}"
            if which == "best_epoch_on_test" and tag == "8clips":
                continue                      # not one of the three published lines
            out[key] = {"train_crops": int(tr.sum())}
            for metric, get in (("accuracy", lambda r: r["accuracy"]),
                                ("macro_recall", lambda r: r["macro_recall"]),
                                ("referee", lambda r: r["recall"]["referee"]),
                                ("goalkeeper", lambda r: r["recall"]["goalkeeper"]),
                                ("referee_precision_p0.5",
                                 lambda r: r["referee_precision_p0.5"]),
                                ("referee_precision_p0.8",
                                 lambda r: r["referee_precision_p0.8"])):
                out[key][metric] = sorted(round(get(r[idx]), 3) for r in runs)

    # The stored constant is a claim about this machine. Print the diff rather
    # than assert it: MPS on another box may not reproduce to three decimals,
    # and a hard failure there would teach the reader to delete the check.
    for key, got in out.items():
        if not isinstance(got, dict) or key not in LEAK_AUDIT:
            continue
        for metric, v in got.items():
            was = LEAK_AUDIT[key].get(metric)
            flag = "" if was == v else f"   <- LEAK_AUDIT says {was}"
            print(f"{key:36} {metric:24} {v}{flag}")
    return out


# ---------------------------------------------------------------- self-check

def _assert_record(r: dict) -> None:
    """The run record is only worth writing if it cannot be written incomplete."""
    missing = [k for k in REQUIRED if k not in r]
    assert not missing, f"run record missing {missing}"
    assert r["seeds"] and all(isinstance(s, int) for s in r["seeds"]), r["seeds"]
    assert len(r["seeds"]) >= 3, f"a single-seed number is not a result: {r['seeds']}"
    assert len(r["revision"]) == 40, f"dataset revision not pinned: {r['revision']!r}"
    assert r["test_opens"] == len(r["seeds"]), (r["test_opens"], r["seeds"])
    # THE protocol assertion: the epoch table cannot contain a test column.
    for s, table in r["epochs_val"].items():
        for row in table:
            bad = [k for k in row if "test" in k or "held" in k]
            assert not bad, f"seed {s} epoch table scored the test set: {bad}"
    assert set(r["split"]["test_clips"]).isdisjoint(r["split"]["train_clips"])
    assert set(r["split"]["test_clips"]).isdisjoint(r["split"]["val_clips"])
    assert set(r["split"]["val_clips"]).isdisjoint(r["split"]["train_clips"])
    assert r["test"]["n"] and r["test"]["n_per_class"], "a rate without its denominator"


def _selfcheck() -> None:
    """Fail if the protocol regresses.

    Three things this file got wrong once and must not get wrong again: the test
    split used for selection, unseeded RNGs, and a run record with holes.
    """
    # The licence refusal must REFUSE. A deletion note is not a guard
    # (docs-guard-must-have-a-runnable-check (internal note)).
    try:
        fetch()
    except SystemExit as e:
        assert "REFUSED" in str(e), "fetch() must refuse: the DFL data is deleted"
    else:
        raise AssertionError("fetch() did not refuse - the DFL guard is decoration")
    # 1. The split holds out whole MATCHES, three ways, with nothing shared.
    assert set(HOLD_OUT).isdisjoint(VAL), (HOLD_OUT, VAL)
    assert len(set(HOLD_OUT)) == 4 and len(set(VAL)) == 3

    # 2. Seeds. Not "seeded" - the same seed must give the same first weights.
    import torch  # noqa: PLC0415

    def first_fc(seed):
        _seed_all(seed)
        return torch.nn.Linear(512, len(NAMES)).weight[0, :4].tolist()
    assert first_fc(0) == first_fc(0), "seeding does not reproduce the fc init"
    assert first_fc(0) != first_fc(1), "the seed does not reach the fc init"

    # 3. The run record's own assertions must reject the failures they name.
    ok = {k: v for k, v in {
        "git_commit": "x", "git_dirty": False, "argv": [], "config": {},
        "seeds": [0, 1, 2], "dataset": DATASET, "revision": "a" * 40,
        "crops_sha256": "s", "split": {"train_clips": ["a"], "val_clips": ["b"],
                                       "test_clips": ["c"]},
        "n_per_class_per_clip": {}, "libraries": {}, "device": "cpu",
        "epochs_val": {"0": [{"epoch": 1, "val_accuracy": 1.0}]},
        "test": {"n": 1, "n_per_class": {"player": 1}}, "per_seed": [],
        "test_opens": 3, "wall_s": 1.0, "host": "h", "artefact_sha256": "a",
    }.items()}
    _assert_record(ok)
    for mutate, why in (
        (lambda r: r.pop("revision"), "missing field"),
        (lambda r: r.update(revision="main"), "unpinned revision"),
        (lambda r: r.update(seeds=[0]), "single seed"),
        (lambda r: r.update(test_opens=24), "test consulted more than once per seed"),
        (lambda r: r["epochs_val"]["0"][0].update(test_accuracy=0.95),
         "test scored inside the epoch loop"),
        (lambda r: r.update(split={"train_clips": ["a"], "val_clips": ["a"],
                                   "test_clips": ["a"]}), "clip in two splits"),
        (lambda r: r["test"].update(n=0), "a rate without its denominator"),
    ):
        bad = json.loads(json.dumps(ok))
        mutate(bad)
        try:
            _assert_record(bad)
        except (AssertionError, KeyError):
            continue
        raise AssertionError(f"_assert_record accepted: {why}")

    # 4. The crops, if built: group disjointness, geometry, and the revision.
    if os.path.exists(CROPS):
        z = np.load(CROPS)  # arrays only; no pickle, so no code path from the file to exec
        te, va = np.isin(z["clip"], HOLD_OUT), np.isin(z["clip"], VAL)
        tr = ~te & ~va
        for a, b in ((te, tr), (te, va), (va, tr)):
            assert set(z["clip"][a]).isdisjoint(set(z["clip"][b])), "clip leaked"
        assert te.sum() > 500 and tr.sum() > 2000 and va.sum() > 200, \
            (tr.sum(), va.sum(), te.sum())
        assert z["X"].shape[1:] == (rolenet.H, rolenet.W, 3), z["X"].shape
        assert "revision" in z and len(str(z["revision"])) == 40, "crops built unpinned"
        # The VAL tuple is a hardcoded id list, which is what makes a change to
        # it visible in the run record - but a hardcoded list can drift from the
        # rule that justified it. Re-derive the rule and assert they agree.
        rest = sorted((c for c in set(z["clip"].tolist()) if c not in HOLD_OUT),
                      key=lambda c: (-(z["clip"] == c).sum(), c))
        assert set(rest[4:7]) == set(VAL), \
            f"VAL is not the three median-sized non-test clips: {rest[4:7]} vs {VAL}"
        # Every class present in every split, or a recall in the report is a
        # rate over nothing (postmortem B2).
        for name, ids in (("val", VAL), ("test", HOLD_OUT)):
            m = np.isin(z["clip"], ids)
            for i, n in enumerate(NAMES):
                assert ((z["y"] == i) & m).sum() > 0, f"no {n} in the {name} split"

    # 5. If a run exists, it must satisfy the same assertions the run made.
    if os.path.exists(RUN):
        with open(RUN) as f:
            _assert_record(json.load(f))
    print("train_roles.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the dataset and build crops")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one run per seed; the spread is the result, not the mean")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--audit-leak", action="store_true",
                    help="re-run the superseded protocol; writes nothing")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.audit_leak:
        audit_leak(a.epochs, a.lr, tuple(a.seeds))
    elif a.fetch:
        build(fetch())
    else:
        # The RESOLVED config is what goes in the record, not argv alone -
        # postmortem A4/A5 were both a code default disagreeing with a doc.
        train(a.epochs, a.lr, tuple(a.seeds), argv=sys.argv)
