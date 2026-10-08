"""The geometry-first per-player event model - sprint P1, and G2 (the overfit gate).

    .venv/bin/python model/validation/st2_train.py --demo    # self-check, no data
    .venv/bin/python model/validation/st2_train.py --g2      # the gate below

WHAT THE MODEL SEES
    Game-state channels only, from the vendor's GT tracks in st2_cache's .npz:
    per player per frame [x, y, vx, vy, speed, accel, dist to each goal, side,
    on-pitch]. Velocity and acceleration come from `possession._features`, which
    smooths POSITION with a Savitzky-Golay before differencing - that module
    measured F1 0.054 unsmoothed against 0.60 smoothed, so it is not a detail.

    NO BALL CHANNEL. The feed's ball is st2_eval's nearest-player reference point
    and nothing else (sprint §P1): at ship confidence the product's ball has 6.8%
    recall and its ST2 GT is a 1 m grid, so a ball feature would reintroduce the
    exact dependency this route exists to avoid. NO PIXELS either - that is P3.
    No role channel: the cache carries side and jersey, not role.

    STILL TRUE OF WHAT SHIPS AND OF ARM A, 2026-09-20. The G3B addendum at the
    foot of this docstring adds two SWITCHABLE arms - a GT-ball arm and a team
    sheet role arm - behind `--arm`, to attribute G3's failure. Arm B's ball is
    an ORACLE input and its numbers are never model performance.

    Shape is PAVE's: attention ACROSS players within each frame, then a temporal
    read per player. PAVE's own ablation is that game-state channels alone beat
    game-state plus visual features, which is why this is the first thing built.

WHAT IT EMITS
    Per centre frame, per player, a distribution over the 12 BAS classes + null.
    A spot for class c is a local maximum of max-over-players p(c), greedily
    de-duplicated within NMS_S. Scored by `st2_eval.ap_tolerant`, the same
    11-point interpolated, one-to-one scorer the G1 floors were measured with.
    Do not write a second scorer.

PROTOCOL - G2, the deliberate-overfit gate. Committed BEFORE the first training
run, as sprint §11 requires and model/validation/protocol.py checks.

    G2 IS A SANITY CHECK, NOT A RESULT. It trains and scores on the SAME data.
    A number from it says only "the harness can fit what it has seen"; it says
    nothing whatever about whether this model works. No number from G2 may be
    quoted as model performance anywhere, ever.

    HALF      117093 period 1 (the first half) and nothing else. 117093 is a VAL
              match, so no train match is spent on a sanity check and test
              (128057) is never opened. `_selfcheck` refuses any match whose
              EVENT_SPLIT is "test" or "excluded".
    CENTRES   every STRIDE-th frame of that half. The training set and the
              scored set are the SAME centres - that is what "overfit" means
              here, and it is stated in the artefact rather than implied.
    TARGET    centre t, player p is class c iff a LANDED label of class c has
              its corrected row within RADIUS frames of t and p is its actor;
              otherwise null. Off-pitch players are excluded from the loss.
              Landing and the label offset are st2_index's PROTOCOL block; the
              share of the half's labels that did NOT land is reported beside
              every number.
    TIME      an event's time is read from the feed clock at its CORRECTED row,
              never from the raw label clock, so predictions and GT share one
              clock after the LABEL OFFSET shift.
    SCORE     mAP@+-1 s = the unweighted mean over the classes PRESENT in the
              half of ap_tolerant(spots_c, gt_c, 1.0). Per-class AP and n are
              reported with it; a class absent from the half is not scored and
              is listed as absent.
    PASS      mAP@1s > 0.80 on that same half. Below it the harness is broken -
              wrong targets, wrong masking, a feature/label misalignment G0
              missed - and G3 does not start. It is never a modelling result.
    RECORDED  every run prints the git commit, the sha256 of bas_index.csv and
              of the half's cache, the number of configurations tried, and the
              drop share. n= is the count of configurations tried on this half.

ADDENDUM 2026-09-19, committed before any attribution number is produced.

    ATTRIBUTION  a spot carries the slot of the player holding the maximum for
              its class at its peak centre. Spots are matched to GT exactly as
              the AP matches them - `st2_eval._greedy`, score order, same
              tolerance - and a matched spot is ATTRIBUTED CORRECTLY iff that
              slot is the labelled actor's slot. Score = correct / matched,
              per class and pooled, reported beside n_matched / n_gt. It is
              never quoted alone: it rises when recall falls, so the pair is
              one number in two halves.
    BASELINES on the identical matched set, per sprint §3: the nearest-player
              credit for each matched event (the on-pitch player closest to the
              FEED'S ball at the label frame, a k-way tie containing the actor
              scoring 1/k) and chance (1/on-pitch). Events whose frame carries
              no feed ball have no nearest-player credit; they stay in the GT
              and are counted out of the floor, and the count is printed.
    NOT LIKE FOR LIKE, and this is the point. The nearest-player floor is
              allowed to look at the ball. The model has NO ball channel at all
              (sprint §P1). So it is a deliberately harder bar than a matched
              comparison, and a model below it has not necessarily failed - it
              has failed to beat a baseline with a sense it does not possess.
              State that wherever the two appear together.
    G2's attribution is, like everything else here, on a half the model has
              memorised. It is a wiring check on the actor path, not a result,
              and it has no pass rule.

PROTOCOL - G3, the real training run. Committed BEFORE it is run.

    BUDGET IS EQUAL OPTIMISER STEPS, NOT EQUAL EPOCHS. G2 needed 150 epochs on
              one half, and 150 was chosen by looking at a val half - carrying it
              into G3 would import a val-selected hyper-parameter straight into
              the result. G3's train set is 170,294 centres against G2's 13,530,
              so epochs = round(150 * 13530 / 170294) = 12 and the number of
              gradient updates is held fixed instead. `_g3_epochs()` asserts that
              arithmetic against the data at run time, so the number cannot drift
              away from the centres it was derived from.
              n=1 configuration. Seed 0. IF 12 EPOCHS UNDERFITS, THAT IS THE
              RESULT: the epoch count is not tuned upward against val and no
              sweep is run. A budget chosen after seeing val is not a budget.
    TRAIN     the six train matches of EVENT_SPLIT, both periods, twelve halves.
              Test (128057) is not loaded by this file at all.
    SCORE     val = 117093 + 132877, at tolerance 1.0 s. Per class, AP is the
              mean over the periods in the reporting set, which is what
              st2_eval.floors averages, so the two are comparable. Reported for
              117093 alone, 132877 alone, and pooled - never only pooled, because
              132877 is the un-mirrored feed and drops 16.8% of its labels, and
              pooling hides both.
    GATE      sprint §P1: beat EVERY floor st2_eval.py prints, per class, at that
              tolerance, AND beat the nearest-player floor on attribution. A class
              is judged against the largest of its floors. Classes whose floor is
              0.000 are named as such rather than counted as wins.
    DROPS     every 132877 number carries its drop share; every pooled number
              carries it too, because 132877 is half the pooled periods.

ADDENDUM 2026-09-19, the SECOND budget. Committed before it is run, after the
author ruled G3 FAILED under the first one and asked for the underfit removed.

    WHY       the first G3 spent 12 epochs (equal optimiser steps to G2) and its
              training loss was still falling: 0.5909 -> 0.3700, no flattening.
              So "PASS and DRIVE lose to shift-copy" had two possible causes and
              that run could not separate them: the model was UNDERFIT, or the
              geometry channels cannot do PASS and DRIVE. This run removes the
              first so only the second can be left.
    STOP      after each epoch e > 3, with L the mean TRAINING loss per epoch:
              stop iff (L[e-3] - L[e]) / L[e-3] < 0.01. Hard cap 90 minutes of
              training, checked at an epoch boundary. Seed 0, n=1 configuration.
              The artefact records which of the two stopped it. IF THE CAP FIRES
              FIRST, that is reported as the result at the cap and the cap is
              NOT extended.
    VAL IS NEVER CONSULTED FOR THE STOP. That is the whole point of the rule: a
              stop chosen on val is a val-selected hyper-parameter wearing a
              different hat, and this run exists to make a claim about the
              inputs, which only survives if nothing about it was fitted to val.
    ALSO SCORED at fixed epoch 12, so step 8's published numbers come out of the
              same artefact instead of a second half-hour run. Training is
              deterministic, so the weights at epoch 12 of this run ARE the
              weights the 12-epoch run ended on. NEITHER CHECKPOINT IS SELECTED
              OVER THE OTHER: the plateau is the result, epoch 12 is the
              previous result re-emitted, and both are always printed.
    WHAT EACH OUTCOME MEANS, written before the number exists:
              PASS/DRIVE rise above shift-copy -> the first run was underfit and
                  the gate is cleared on the second reading; the author re-rules.
              PASS/DRIVE rise but still lose -> the budget was part of it and the
                  inputs are the rest.
              PASS/DRIVE do not move -> it is NOT the budget. That is a finding
                  about the INPUTS: no ball channel and no role channel, on a
                  task whose two biggest classes are defined by where the ball
                  went. Say so plainly and let it scope P3.
    ARTEFACT  the run writes model/validation/out/st2_events/G3.json and every
              number in st2-p1-step8 (not published) is quoted through `--show`, which
              reads it in under a second. Exactly ONE #slow tag in that document
              names the producer. No G3 number needs a re-train to verify.

PROTOCOL - G3B, the BALL/ROLE DIAGNOSTIC. Committed BEFORE any arm is run, in a
commit that contains no result. st2-p1-step9 (not published) is the page it writes.

    WHY  G3 FAILED and step 8 named the suspect: PASS 0.437 and DRIVE 0.453
              against floors of 0.544 and 0.578, and those two classes are 84%
              of the val corpus and are DEFINED BY WHERE THE BALL WENT. The
              model has no ball channel and no role channel. The floor that
              beats it is handed the GT PASS times; the attribution floor it
              loses to (0.900) names the player nearest the FEED'S BALL. This
              run asks one question: how much of that gap is the missing ball,
              and how much the missing role.

    ARM B IS AN ORACLE INPUT AND ITS NUMBERS ARE NOT PRODUCT PERFORMANCE.
              It is fed the vendor feed's GROUND-TRUTH ball position, per frame.
              The product has no such ball: at ship confidence its ball detector
              has 6.8% recall (sprint §P1) and ST2's own ball GT is a 1 m grid
              (st2-ball-gt-is-a-1m-grid (internal note)). So arm B is an UPPER
              BOUND no shipped pipeline can reach, and it exists to attribute a
              failure, never to report a capability. Every arm-B number is
              written "ORACLE" in the same sentence, in the doc, in a commit
              message and in chat; the artefact carries `"oracle": true` and
              `--show` prints the banner on every arm-B block so the label
              cannot be lost by quoting. Sprint §P1 allows exactly this - "if it
              goes in at all it goes in as a switchable ablation, never a base
              channel" - and this is the switch.
    ARM C IS ORACLE TOO, MORE QUIETLY. The listed position is a team sheet, which
              an operator could type (jersey-numbers-come-from-the-operator (internal note)).
              Attaching it to a TRACK needs per-track identity for
              the whole half, which this repo does not have: fragments are a
              detection problem (N40) and appearance re-ID is dead on this
              footage. So arm C is also a diagnostic, not a product number.

    ARMS      three, each one configuration, all pre-registered here and none
              selected over another after the fact:
                A geo    the 10 geometry channels. Step 8's model, unchanged.
                B ball   geo + 4: (ball_x - x)/10, (ball_y - y)/10, |ball - p|/10
                         in metres, and a ball-present flag. Absent ball -> the
                         first three are 0 AND the flag is 0, so "no ball" is
                         distinguishable from "the ball is on my feet", which a
                         zero-filled displacement alone would not be.
                C role   geo + 4: a one-hot over GK/DEF/MID/FWD from the vendor
                         team sheet (st2_cache.roles). No position listed -> an
                         all-zero one-hot, counted and printed.
              WHY THIS FORM. Relative displacement + distance is exactly the
              statistic the nearest-player attribution floor is computed from,
              so arm B can learn that floor's rule; absolute ball position
              broadcast to every player is not, because within a frame it is
              constant and cannot pick an actor out. Over the +-0.96 s window
              the displacement SEQUENCE already carries ball-leaves-me (PASS)
              against ball-arrives-at-me (DRIVE), so no ball-velocity or
              closing-rate channel is added - which would need a second
              smoothing constant on a 1.05 m x 0.68 m quantised ball, and a
              constant chosen here is a knob nobody registered. What arm B
              still CANNOT learn: ball height (HIGH PASS vs PASS by flight),
              and who the ball is going to before it gets there.
              The arms are NOT parameter-matched: B and C each add 4x64 = 256
              weights in `embed`. If either wins by a hair, that is the first
              thing to suspect, and the margin rule below is set above it.
    BUDGET    IDENTICAL TO G3's SECOND BUDGET, per arm, copied not re-derived:
              after each epoch e > 3 stop iff the mean TRAINING loss's gain over
              G3_PLATEAU[1] epochs is under G3_PLATEAU[0]; hard cap G3_CAP_MIN
              minutes at an epoch boundary; seed 0; n=1 configuration per arm.
              VAL IS NEVER CONSULTED - not for the stop, not for the channel
              form, not for the 10 m scale. Val is also scored at fixed epoch
              G3_CHECKPOINTS, which is the ONLY equal-epoch comparison between
              arms; the plateau epoch may differ between them and that is a
              result, not a confound to correct.
    POPULATION  EVENT_SPLIT, unchanged. Train = the six train matches, twelve
              halves. Val = 117093 + 132877, four periods. 132831 excluded.
              128057 (test) not loaded. Floors are st2_eval's on the CORRECTED
              clock, the matcher is st2_eval._greedy, the scorer is
              ap_tolerant, the attribution rule is the ADDENDUM above - all
              shared with step 8, none re-implemented. Every val number carries
              117093's match-level drop share and 132877's 0.168.
    MEASURED BEFORE THE RUN, because it conditions the reading rather than
              following from it: the feed's ball is present on 0.999 / 1.000 of
              117093's two periods and 0.863 / 0.763 of 132877's, longest gaps
              2 s and 114 s. ARM B IS PARTLY BALL-BLIND ON 132877 BY
              CONSTRUCTION. The gate below is pooled, like step 8's; the
              per-match columns are printed and 117093 is the clean read.
    "THE BALL EXPLAINS IT" IFF, at the plateau checkpoint, pooled over the four
              val periods: arm B's PASS AP@1s > its worst floor AND arm B's
              DRIVE AP@1s > its worst floor - the same per-class gate step 8 was
              judged by, against the same floors. Attribution is reported the
              same way: arm B against the 0.900 nearest-player floor, which is
              now like-for-like for the first time, since arm B can see the ball
              that floor is computed from.
    WHAT EACH OUTCOME MEANS, written before the numbers exist:
              B CLEARS BOTH -> the missing input was the ball. The geometry-only
                  route's ceiling on 84% of the corpus is set by ball
                  observability, and the next question is the ball detector
                  (6.8% recall at ship conf), NOT a bigger event model and NOT
                  pixels. Still not a product result: see ORACLE above.
              B MOVES PASS OR DRIVE BY >= 0.05 AND STILL LOSES -> the ball is
                  part of it and something else is the rest. Scope P3 on the
                  residual, per class, and say which classes moved.
              B MOVES BOTH BY < FLOOR_NOISE (0.012) -> THE BALL IS NOT THE
                  MISSING INPUT. That is the surprising outcome and the valuable
                  one: it would mean +-1 s localisation of release and reception
                  needs something neither positions nor the ball's ground-plane
                  position carries, and step 8's closing paragraph is wrong.
              C, against A, reads the same three ways. B and C are never added
                  together into a claim about a fourth arm that was not run.
    VOID IF   any one of these, and the whole page is withdrawn rather than
              footnoted:
              - arm A does not reproduce step 8's run: stop epoch 18, pooled
                PASS 0.437, DRIVE 0.453, attribution 0.466. Training is
                deterministic at seed 0 and arm A's code path is unchanged, so a
                difference means the harness moved under the comparison and NO
                arm is comparable to step 8.
              - a cached .npz's sha differs from the one arm A recorded, or the
                metadata XML's sha differs between arms.
              - any arm stops at the CAP instead of the plateau. That arm is
                reported at the cap, is NOT compared against a plateau arm as if
                it had converged, and only its fixed-epoch column stands.
              - val is consulted for anything at all, including a re-run "to
                check" after seeing a number. A second run of an arm makes this
                a sweep and it then carries n= configurations tried.
              - the channel form or the 10 m scale is changed after any arm has
                been scored.
    DEFECT 1 OF STEP 8 IS COUNTED HERE, NOT FIXED. `_mark` still lets the later
              label win a collided target cell, exactly as step 8's run did, so
              arm A reproduces bit for bit and the arms stay comparable to it.
              What changes is that the count is no longer silent: collisions,
              cross-class collisions and un-masked cells are printed per half
              and carried in the artefact, which is what this repo's own rule
              about dropped input requires. Fixing it means a multi-label target
              and a different loss; that is its own registered run.

`ponytail: one .npz held in RAM (62 MB) instead of mmap + workers - the half fits
and a DataLoader would be more code than the training loop. Upgrade path for G3
(six matches, ~20 halves) is np.load(mmap_mode='r') per half, as sprint §11 says.`
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "model/engine"))
import possession as P  # noqa: E402
import st2_cache as C  # noqa: E402
import st2_eval as E  # noqa: E402
from st2_index import EVENT_SPLIT, HALF_MS  # noqa: E402

FPS = 1000.0 / C.FRAME_MS          # 25.0, the feed's own rate
STRIDE = 5                          # centres every 0.2 s
RADIUS = 2                          # frames each side of a label that carry its class
OFFSETS = np.arange(-24, 25, 6)     # the window: +-0.96 s, 9 steps
NMS_S = 0.4                         # spots closer than this, same class, are one spot
SPOT_MIN = 0.02                     # below this a peak is not emitted at all
TOL = 1.0                           # the gate's tolerance, seconds
NULL = 0                            # class 0 is "no event"
N_CH = 10                           # geometry channels per player per frame
BALL_SCALE_M = 10.0                 # metres per unit on the ball-relative channels
ROLE_CH = len(C.ROLE_LINES)         # GK / DEF / MID / FWD one-hot
# The G3B arms (PROTOCOL addendum). "geo" IS step 8's model, byte for byte.
ARM_CH = {"geo": N_CH, "ball": N_CH + 4, "role": N_CH + ROLE_CH}
CLASSES = ("BALL PLAYER BLOCK", "CROSS", "DRIVE", "FREE KICK", "GOAL", "HEADER",
           "HIGH PASS", "OUT", "PASS", "PLAYER SUCCESSFUL TACKLE", "SHOT", "THROW IN")
IDX = {c: i + 1 for i, c in enumerate(CLASSES)}
G2_MATCH, G2_PERIOD = "117093", 0
G2_PASS_MAP = 0.80
G2_EPOCHS, G2_CENTRES = 150, 13530      # what G2 spent, and on how many centres


def _g3_epochs(centres: int) -> int:
    """G2's optimiser steps, re-derived from the data actually loaded.

    The budget is equal STEPS, not equal epochs, so it is a function of the train
    set size and must be recomputed rather than pasted. Written 2026-09-19 after a
    review found the docstring promising this check and the check absent.
    """
    n = max(1, round(G2_EPOCHS * G2_CENTRES / max(centres, 1)))
    assert 1 <= n <= G2_EPOCHS, (n, centres)
    return n


# ---------------------------------------------------------------- features
def features(xy, pitch, side, smooth_s=P.SMOOTH_S, ball=None, role=None):
    """(T,N,2) metres -> (T,N,C) channels + (T,N) on-pitch mask.

    `possession._features` owns the smoothing; SMOOTH_S is set around the call
    rather than copied, so there is one Savitzky-Golay in this repo and not two.

    WITH NEITHER `ball` NOR `role` THIS IS STEP 8's FEATURE SET, unchanged and
    in the same order - arm A of G3B has to reproduce step 8 bit for bit, so
    the geometry block is appended to rather than rewritten.

    `ball` (T,2) metres adds the ORACLE ball arm's four channels; `role` (N,)
    line indices from st2_cache.roles adds the role arm's one-hot. Both are the
    G3B PROTOCOL addendum's, and neither is on by default anywhere.
    """
    on = np.isfinite(xy[:, :, 0])
    old, P.SMOOTH_S = P.SMOOTH_S, smooth_s
    try:
        f, _ = P._features(SimpleNamespace(XY=xy, mask=on), FPS)
    finally:
        P.SMOOTH_S = old
    lx, ly = float(pitch[0]), float(pitch[1])
    x, y, vx, vy, sp, ac = (f[..., i] for i in range(6))
    gy = (y - ly / 2) ** 2
    ch = np.stack([x / lx, y / ly, vx / 10, vy / 10, sp / 10, ac / 10,
                   np.sqrt(x ** 2 + gy) / lx, np.sqrt((x - lx) ** 2 + gy) / lx,
                   np.where(side < 0, 0.5, side)[None, :] * np.ones_like(x),
                   on.astype(np.float32)], axis=2)
    extra = []
    if ball is not None:
        bx, by = ball[:, :1], ball[:, 1:]                 # (T,1), broadcast over players
        seen = np.isfinite(bx) & np.isfinite(by)          # the FEED's ball, not the player
        dx, dy = np.where(seen, bx - x, 0.0), np.where(seen, by - y, 0.0)
        # A missing ball is (0, 0, 0, flag 0). Without the flag that reads as
        # "the ball is on my feet", which is the one imputation that inverts the
        # channel's meaning on exactly the frames it is missing.
        extra.append(np.stack([dx / BALL_SCALE_M, dy / BALL_SCALE_M,
                               np.hypot(dx, dy) / BALL_SCALE_M,
                               np.broadcast_to(seen, x.shape).astype(np.float32)], axis=2))
    if role is not None:
        oh = np.asarray(role)[:, None] == np.arange(ROLE_CH)      # -1 -> all zeros
        extra.append(np.broadcast_to(oh, x.shape + (ROLE_CH,)).astype(np.float32))
    if extra:
        ch = np.concatenate([ch, *extra], axis=2)
    return np.nan_to_num(ch).astype(np.float32), on


def _mark(tgt, j, a, cls, n):
    """Write one label into a target cell, counting what it wrote over.

    `n` is [collisions, cross-class collisions, un-masked cells]. STEP 8's
    DEFECT 1 IS COUNTED HERE AND NOT FIXED: the later label still wins, exactly
    as the published run did, so arm A of G3B reproduces it bit for bit and the
    arms stay comparable to step 8. What changes is that the loss is no longer
    silent - 51 of 16,948 labels never reached the loss and appeared in no drop
    share, which this repo's own rule forbids. The fix is a multi-label target
    and a different loss; that is its own registered run.
    """
    prev = tgt[j, a]
    if prev == -100:
        n[2] += 1                       # un-masks a player whose channels are zeros
    elif prev != NULL:
        n[0] += 1
        n[1] += int(prev != IDX[cls])
    tgt[j, a] = IDX[cls]


def half(match, period, arm="geo"):
    """The cache's half -> (channels, on-pitch, centre rows, targets, gt times).

    Targets are (n_centres, N) int64 over 13 classes, -100 where the player is
    off the pitch, which torch's cross-entropy ignores.

    `arm` selects the channel set (G3B PROTOCOL addendum). "geo" is step 8's.
    """
    c = C.load(match)
    assert EVENT_SPLIT[match] in ("train", "val"), f"{match} is {EVENT_SPLIT[match]}"
    assert arm in ARM_CH, arm
    k = period
    xy, t_ms = c[f"p{k}_xy"], c[f"p{k}_t_ms"]
    ball = c[f"p{k}_ball"]
    role = C.roles(match, c[f"p{k}_pid"]) if arm == "role" else None
    ch, on = features(xy, c["pitch"], c[f"p{k}_side"],
                      ball=ball if arm == "ball" else None, role=role)
    assert ch.shape[2] == ARM_CH[arm], (ch.shape, arm)
    centres = np.arange(0, len(t_ms), STRIDE)
    tgt = np.zeros((len(centres), xy.shape[1]), np.int64)
    tgt[~on[centres]] = -100
    sel = (c["ev_period"] == k) & (c["ev_drop"] == "")
    rows, acts, labs = c["ev_row"][sel], c["ev_actor"][sel], c["ev_label"][sel]
    gt, no_ball, coll = {}, 0, [0, 0, 0]
    for r, a, lab in zip(rows, acts, labs):
        # the corrected row IS the clock; ev_t_s is the raw label time (PROTOCOL)
        if np.isnan(ball[r, 0]):
            # no feed ball on this frame, so the nearest-player floor is undefined
            # for it. The event still counts as GT; the floor skips it, exactly as
            # st2_eval.nearest_player does, and the count is reported.
            no_ball += 1
            near, chance = float("nan"), float("nan")
        else:
            near, chance = E._credit(np.linalg.norm(c[f"p{k}_xy"][r] - ball[r], axis=1), a)
        gt.setdefault(str(lab), []).append(
            (float(t_ms[r] - k * HALF_MS) / 1000, int(a), near, chance))
        for j in range(int(np.searchsorted(centres, r - RADIUS, "left")),
                       int(np.searchsorted(centres, r + RADIUS, "right"))):
            _mark(tgt, j, a, str(lab), coll)
    n_lab = int((c["ev_period"] == k).sum())
    drop = 1.0 - len(rows) / n_lab
    # THE PERIOD'S OWN DROP SHARE IS THE SMALL ONE AND IT FLATTERS 132877. Its
    # third period has no footage at all, so those rows never appear in period 0
    # or 1 and `drop` cannot see them. The match-level share is what step 6
    # measured (132877: 0.168) and what every val number has to carry.
    m_drop = float((c["ev_drop"] != "").sum()) / len(c["ev_drop"])
    return SimpleNamespace(ch=ch, on=on, centres=centres, tgt=tgt, gt=gt,
                           lo=np.zeros(len(centres), np.int64),
                           hi=np.full(len(centres), len(ch) - 1, np.int64),
                           t=(t_ms[centres] - k * HALF_MS) / 1000.0, no_ball=no_ball,
                           n_labels=n_lab, n_landed=len(rows), drop=drop,
                           match_drop=m_drop, match=match, arm=arm, coll=coll,
                           n_role_unknown=0 if role is None else int((role < 0).sum()),
                           period=k, sha=_sha(C.path(match)))


def halves(split, arm="geo"):
    return [half(m, k, arm)
            for m in sorted(x for x, s in EVENT_SPLIT.items() if s == split)
            for k in (0, 1)]


def dataset(hs):
    """Several halves as ONE time-concatenated store, windows clipped per half.

    Players are padded to the widest half's slot count; the padding is simply
    off-pitch, which the mask already handles, so a 23-player half and a 27-player
    half batch together. `lo`/`hi` are that centre's OWN half's first and last row,
    so no window can reach across a half boundary into another match's football -
    which is the one silent way a concatenated store goes wrong.

    Each half keeps `sl`, its slice of the centre array, so scoring stays per half:
    val is reported per match and the floors are a mean over periods.
    """
    nmax = max(h.ch.shape[1] for h in hs)
    tot = sum(h.ch.shape[0] for h in hs)
    n_ch = hs[0].ch.shape[2]
    assert all(h.ch.shape[2] == n_ch for h in hs), "halves built on different arms"
    ch = np.zeros((tot, nmax, n_ch), np.float32)
    on = np.zeros((tot, nmax), bool)
    tgt = np.full((sum(len(h.centres) for h in hs), nmax), -100, np.int64)
    centres, lo, hi, off, c0 = [], [], [], 0, 0
    for h in hs:
        t, n = h.ch.shape[:2]
        ch[off:off + t, :n], on[off:off + t, :n] = h.ch, h.on
        tgt[c0:c0 + len(h.centres), :n] = h.tgt
        centres.append(h.centres + off)
        lo.append(np.full(len(h.centres), off, np.int64))
        hi.append(np.full(len(h.centres), off + t - 1, np.int64))
        h.sl = slice(c0, c0 + len(h.centres))
        h.ch = h.on = h.tgt = None                      # the big copy is the store now
        off, c0 = off + t, c0 + len(h.centres)
    return SimpleNamespace(ch=ch, on=on, tgt=tgt, halves=hs,
                           centres=np.concatenate(centres),
                           lo=np.concatenate(lo), hi=np.concatenate(hi))


# ---------------------------------------------------------------- model
class EventNet(nn.Module):
    """Spatial attention across players at each step, then a temporal read."""

    def __init__(self, f_in=10, d=64, heads=4, steps=len(OFFSETS)):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(f_in, d), nn.ReLU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(steps * d, 128), nn.ReLU(),
                                  nn.Linear(128, len(CLASSES) + 1))

    def forward(self, x, pad):
        """x (B,S,N,F), pad (B,S,N) True where the player is off the pitch."""
        b, s, n, _ = x.shape
        h = self.embed(x).reshape(b * s, n, -1)
        m = pad.reshape(b * s, n).clone()
        m[m.all(1)] = False                      # an all-padded frame would give NaN
        h = self.norm(h + self.attn(h, h, h, key_padding_mask=m, need_weights=False)[0])
        h = h.reshape(b, s, n, -1).permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.head(h)                      # (B,N,13)


def _windows(d, idx, device):
    rows = np.clip(d.centres[idx][:, None] + OFFSETS[None, :],
                   d.lo[idx][:, None], d.hi[idx][:, None])
    x = torch.from_numpy(d.ch[rows]).to(device)
    return x, torch.from_numpy(~d.on[rows]).to(device)


def fit(d, epochs=40, batch=64, lr=2e-3, seed=0, device="cpu", quiet=False,
        after=None, plateau=None, cap_min=None):
    """-> (net, per-epoch mean training loss, why it stopped).

    `plateau=(rel, window)` stops when the training loss has improved by less
    than `rel` over `window` consecutive epochs; `cap_min` stops at an epoch
    boundary once that many minutes of training have passed. BOTH READ TRAINING
    LOSS AND THE CLOCK ONLY - val is never consulted, which is the whole point
    of the rule (G3 PROTOCOL, second budget). `after(ep, net, losses)` runs at
    each epoch boundary for fixed-epoch checkpoints.
    """
    torch.manual_seed(seed)
    net = EventNet(f_in=d.ch.shape[2]).to(device)
    n = np.bincount(d.tgt[d.tgt >= 0].ravel(), minlength=len(CLASSES) + 1)
    w = np.where(n > 0, 1 / np.sqrt(np.maximum(n, 1)), 0.0)      # 1/sqrt(N), sprint §P1
    w = w / w[w > 0].mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                  ignore_index=-100)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses, why, t0 = [], "epochs", time.time()
    for ep in range(epochs):
        net.train()
        order, tot = rng.permutation(len(d.centres)), 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x, pad = _windows(d, idx, device)
            y = torch.from_numpy(d.tgt[idx]).to(device)
            loss = loss_fn(net(x, pad).reshape(-1, len(CLASSES) + 1), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        losses.append(tot / len(order))
        mins = (time.time() - t0) / 60
        if not quiet:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {losses[-1]:.4f}  {mins:5.1f} min",
                  flush=True)
        if after:
            after(ep + 1, net, losses)
        if plateau and len(losses) > plateau[1]:
            base = losses[-1 - plateau[1]]
            if (base - losses[-1]) / base < plateau[0]:
                why = "plateau"
                break
        if cap_min is not None and mins >= cap_min:
            why = "cap"
            break
    return net, losses, why


@torch.no_grad()
def predict(net, d, batch=256, device="cpu"):
    """-> (n_centres, 13) max-over-on-pitch-players probability, and (n_centres, 13)
    the SLOT of the player holding that maximum - the actor the spot fires on."""
    net.eval()
    out = np.zeros((len(d.centres), len(CLASSES) + 1), np.float32)
    who = np.zeros((len(d.centres), len(CLASSES) + 1), np.int32)
    for i in range(0, len(d.centres), batch):
        idx = np.arange(i, min(i + batch, len(d.centres)))
        x, pad = _windows(d, idx, device)
        p = net(x, pad).softmax(-1)
        p = p.masked_fill(pad[:, len(OFFSETS) // 2].unsqueeze(-1), 0.0)
        mx = p.max(1)
        out[idx], who[idx] = mx.values.cpu().numpy(), mx.indices.cpu().numpy()
    return out, who


def spots(prob, t, who=None, nms_s=NMS_S, thr=SPOT_MIN):
    """Per class, local maxima of prob, greedily thinned within nms_s seconds.

    -> {class: [(t, score, actor slot)]}. The actor is the player holding the
    maximum at the peak centre; with `who=None` it is -1, which scores as a miss
    against every real actor rather than as a free pass."""
    out = {}
    for ci, cls in enumerate(CLASSES, 1):
        s = prob[:, ci]
        keep = np.flatnonzero((s >= thr) & (s >= np.r_[0, s[:-1]]) & (s >= np.r_[s[1:], 0]))
        taken, res = [], []
        for j in keep[np.argsort(-s[keep])]:
            if all(abs(t[j] - t[q]) > nms_s for q in taken):
                taken.append(j)
                res.append((float(t[j]), float(s[j]), -1 if who is None else int(who[j, ci])))
        out[cls] = res
    return out


def score(pred, gt, tol=TOL):
    """-> {class: (n_gt, AP)} for the classes present, plus ("mAP", value)."""
    per = {c: (len(gt[c]), E.ap_tolerant(pred.get(c, []), [e[0] for e in gt[c]], tol))
           for c in CLASSES if gt.get(c)}
    return per, float(np.mean([a for _, a in per.values()]))


# ---------------------------------------------------------------- G2
def _sha(p):
    """sha256 of a file, first 12 - `hashlib.file_digest` (3.11+) streams it."""
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()[:12]


def _selfcheck(match):
    """The three asserts sprint §11 says cannot be skipped."""
    assert EVENT_SPLIT[match] not in ("test", "excluded"), \
        f"{match} is {EVENT_SPLIT[match]}: st2_train.py never loads it"
    assert len(set(EVENT_SPLIT.values())) == 4, EVENT_SPLIT
    assert not set(m for m, s in EVENT_SPLIT.items() if s == "train") & \
        set(m for m, s in EVENT_SPLIT.items() if s in ("val", "test")), "split overlaps"


def G2_JSON(epochs):
    """One artefact per configuration, mirroring G3.json's one-file-per-run shape."""
    return os.path.join(C.OUT, f"G2-e{epochs}.json")


def show_g2(epochs, path=None):
    """Print G2-e<epochs>.json. THE ONLY PRODUCER OF QUOTABLE NUMBERS FOR STEP 7.

    Same contract as `show()` does for G3, and for the same reason: RE-DONE WORK.
    The training's result is this file. Re-running the training per tag re-derives
    what it already wrote, which is the defect - `g2()` writing nothing at all was
    the older half of it.

    THE CLOCK IS SUPPORTING DETAIL, NOT THE ARGUMENT, and it is worth stating
    accurately because an earlier reading of it was wrong. Two quiet-box
    measurements of the same 150-epoch command on 2026-09-20:

        365.1 / 356.9 / 397.7 s   a second session's quiet run, all three INSIDE
                                  the 600 s cap
        497.9 s OK, 600.0 s TIMEOUT x2   this file's own quiet serial run

    So it is roughly 60-83% of the cap: it PASSES ALONE and fails UNDER
    CONTENTION. It is not structurally uncheckable, and calling it one would be
    the kind of overstated reason this repo keeps having to withdraw.
    """
    path = path or G2_JSON(epochs)
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing - run THE PRODUCER:\n"
            f"  .venv/bin/python model/validation/st2_train.py --g2 --device cpu "
            f"--epochs {epochs} --configs 2")
    r = json.load(open(path))
    lb = r["labels"]
    print(f"G2 {r['commit']} labels {r['labels_sha']} cache {r['cache_sha']} "
          f"seed {r['seed']} {r['device']} n={r['n_configs']} epochs {r['epochs']}")
    print(f"  {r['match']} P{r['period'] + 1}, TRAIN AND SCORE ON THE SAME HALF - "
          f"a harness sanity check, NOT model performance.")
    print(f"  labels {lb['n_labels']}, landed {lb['n_landed']}, drop share "
          f"{lb['drop']:.4f}; centres {lb['centres']} at stride {r['stride']}")
    print(f"  {'class':26s} {'n_gt':>5s} {'n_spot':>7s} {'AP@1s':>7s}")
    for c, v in sorted(r["classes"].items(), key=lambda kv: -kv[1]["n_gt"]):
        print(f"  {c:26s} {v['n_gt']:5d} {v['n_spot']:7d} {v['ap']:7.3f}")
    print(f"  {'-' * 50}\n  mAP@1s over {len(r['classes'])} classes present: "
          f"{r['map']:.3f}")
    if r["absent"]:
        print(f"  absent from this half, not scored: {', '.join(r['absent'])}")
    print(f"  G2 {'PASSES' if r['passes'] else 'FAILS'} "
          f"(rule: mAP@1s > {r['pass_map']} on this same half)")
    print(f"\n  ATTRIBUTION given a correct spot - wiring check only. "
          f"{lb['no_ball']} of {lb['n_landed']} landed events have no feed ball.")
    print(f"  {'class':26s} {'matched':>8s} {'n_gt':>5s} {'model':>7s} "
          f"{'nearest':>8s} {'chance':>7s}")
    for c, v in r["attribution"].items():
        print(f"  {c:26s} {v['matched']:8d} {v['n_gt']:5d} {v['model']:7.3f} "
              f"{v['nearest']:8.3f} {v['chance']:7.3f}")
    return r


def g2(epochs=40, seed=0, device=None, n_configs=1):
    """Train on 117093 P1 and score that same half. A sanity check, not a result."""
    _selfcheck(G2_MATCH)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    h = half(G2_MATCH, G2_PERIOD)
    d = dataset([h])
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"G2 OVERFIT - {G2_MATCH} P{G2_PERIOD + 1}, TRAIN AND SCORE ON THE SAME HALF.\n"
          f"  This is a sanity check on the harness. It is NOT model performance.\n"
          f"  commit {commit}  labels {_sha(os.path.join(ROOT, E.INDEX))}  "
          f"cache {h.sha}  device {device}  n={n_configs} configurations tried\n"
          f"  labels {h.n_labels}, landed {h.n_landed}, drop share {h.drop:.4f} "
          f"(st2_index PROTOCOL); centres {len(d.centres)} at stride {STRIDE}")
    t0 = time.time()
    net, _, _ = fit(d, epochs=epochs, seed=seed, device=device)
    prob, who = predict(net, d, device=device)
    pred = spots(prob, h.t, who)
    per, m = score(pred, h.gt)
    print(f"\n  {'class':26s} {'n_gt':>5s} {'n_spot':>7s} {'AP@1s':>7s}")
    for c, (n, ap) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"  {c:26s} {n:5d} {len(pred[c]):7d} {ap:7.3f}")
    absent = [c for c in CLASSES if not h.gt.get(c)]
    print(f"  {'-' * 50}\n  mAP@1s over {len(per)} classes present: {m:.3f}"
          f"   ({time.time() - t0:.0f} s)")
    if absent:
        print(f"  absent from this half, not scored: {', '.join(absent)}")
    print(f"  G2 {'PASSES' if m > G2_PASS_MAP else 'FAILS'} "
          f"(rule: mAP@1s > {G2_PASS_MAP} on this same half)")

    att = E.attribution(pred, h.gt, TOL)
    # THE ARTEFACT. Written for the same reason G3.json is, and the reason is that
    # this result belongs in a file: without one, every tag on step 7 re-runs the
    # training to re-derive numbers the training already produced. `g2()` writing
    # nothing was the defect. The run writes the file once and every tag in
    # st2-p1-step7 (not published) reads it back through `--show-g2` in under a second.
    # (On the clock, see show_g2's docstring: 365-500 s against a 600 s cap across
    # two quiet runs - it passes alone and is at risk only under contention. That
    # is supporting detail, not the argument.)
    # Keyed by epochs because step 7 publishes two configurations (150 and 40) and
    # one file per run is G3's shape; a single shared file would have the second
    # run silently overwrite the first's numbers.
    art = {"commit": commit, "labels_sha": _sha(os.path.join(ROOT, E.INDEX)),
           "seed": seed, "device": device, "n_configs": n_configs,
           "epochs": epochs, "match": G2_MATCH, "period": G2_PERIOD,
           "cache_sha": h.sha, "stride": STRIDE, "tol_s": TOL,
           "labels": {"n_labels": h.n_labels, "n_landed": h.n_landed,
                      "drop": round(h.drop, 4), "centres": len(d.centres),
                      "no_ball": h.no_ball},
           "map": round(m, 3), "pass_map": G2_PASS_MAP, "passes": m > G2_PASS_MAP,
           "classes": {c: {"n_gt": n, "n_spot": len(pred[c]), "ap": round(ap, 3)}
                       for c, (n, ap) in per.items()},
           "absent": absent,
           "attribution": {c: {"matched": att[c][0], "n_gt": att[c][1],
                               "model": round(att[c][2], 3),
                               "nearest": round(att[c][3], 3),
                               "chance": round(att[c][4], 3)} for c in att},
           "minutes": {"total": round((time.time() - t0) / 60, 1)}}
    os.makedirs(C.OUT, exist_ok=True)
    json.dump(art, open(G2_JSON(epochs), "w"), indent=1)
    print(f"\n  -> {G2_JSON(epochs)}")
    print(f"\n  ATTRIBUTION given a correct spot - no pass rule, wiring check only.\n"
          f"  The nearest-player column may look at the FEED'S BALL; this model has no\n"
          f"  ball channel, so it is a harder bar than a like-for-like comparison.\n"
          f"  {h.no_ball} of {h.n_landed} landed events have no feed ball and are out of "
          f"that column.\n"
          f"  {'class':26s} {'matched':>8s} {'n_gt':>5s} {'model':>7s} {'nearest':>8s} {'chance':>7s}")
    for c in [x for x in CLASSES if x in att] + ["all"]:
        n, ng, mo, ne, ch = att[c]
        print(f"  {c:26s} {n:8d} {ng:5d} {mo:7.3f} {ne:8.3f} {ch:7.3f}")
    return m, per, att, h


G3_JSON = os.path.join(C.OUT, "G3.json")
# G3B writes to its OWN directory. out/st2_events is step 6-8's and a push gate
# reads it; nothing here may write there.
G3B_OUT = os.path.join(ROOT, "model/validation/out/st2_ball_oracle")


def g3b_path(arm):
    return os.path.join(G3B_OUT, f"G3B_{arm}.json")
G3_PLATEAU = (0.01, 3)          # stop when train loss gains < 1% over 3 epochs
G3_CAP_MIN = 90                 # hard ceiling on training minutes
G3_CHECKPOINTS = (12,)          # ALSO score val here: step 8's published budget
# The floors are 5 random trials per cell, so they have their own spread. Measured
# over the 26 cells at +-1 s: re-seeding moves a floor by up to 0.0116
# (st2-p1-step7 (not published)). A model that clears its floor by less than that has not
# cleared anything, so it is reported as a TIE and never counted as a gate win.
FLOOR_NOISE = 0.012


def _score_val(net, va, device):
    """-> {reporting set: {...}} for one set of weights. Pure reporting."""
    prob, who = predict(net, va, device=device)
    by = [spots(prob[h.sl], h.t, who[h.sl]) for h in va.halves]
    out = {}
    for name in ("117093", "132877", "all"):
        hs = [(h, by[i]) for i, h in enumerate(va.halves)
              if name == "all" or h.match == name]
        fl = E.floors("val", 0, True, only=None if name == "all" else name)
        cls_out = {}
        for cls in CLASSES:
            n = sum(len(h.gt.get(cls, [])) for h, _ in hs)
            if not n:
                continue
            ap = float(np.nanmean([E.ap_tolerant(pr.get(cls, []),
                                                 [e[0] for e in h.gt.get(cls, [])], TOL)
                                   for h, pr in hs]))
            f = {b: fl["AP"].get(f"{b}.{cls}.{TOL}") for b, _ in E.BASELINES}
            worst = max(v for v in f.values() if v is not None and not np.isnan(v))
            cls_out[cls] = {"n": n, "ap": round(ap, 3),
                            **{b: (None if f[b] is None or np.isnan(f[b])
                                   else round(f[b], 3)) for b, _ in E.BASELINES},
                            "worst": round(worst, 3), "beats": bool(ap > worst),
                            "tie": bool(ap > worst and ap - worst < FLOOR_NOISE),
                            "zero_floor": bool(worst == 0)}
        n_m, n_g, mo, ne, ch_ = E.attribution_many([(pr, h.gt) for h, pr in hs], TOL)["all"]
        out[name] = {"periods": len(hs), "classes": cls_out,
                     "match_drop": {h.match: round(float(h.match_drop), 3) for h, _ in hs},
                     "period_drop": {f"{h.match} P{h.period + 1}": round(float(h.drop), 3)
                                     for h, _ in hs},
                     "attribution": {"matched": n_m, "n_gt": n_g, "model": round(mo, 3),
                                     "nearest": round(ne, 3), "chance": round(ch_, 3),
                                     "beats": bool(mo > ne)}}
    return out


def g3(seed=0, device=None, n_configs=1, smoke=False, arm="geo", out=None):
    """Train on the six train matches, score val, write the artefact. Then `--show`.

    THE BUDGET IS THE G3 PROTOCOL'S and is read off the TRAINING loss and the
    clock alone. `smoke` forces one epoch to exercise the path; its numbers are
    not a result and the artefact records that it was a smoke run.

    `arm` is the G3B addendum's channel set and `out` its artefact. The defaults
    ARE step 8's run: arm "geo" into out/st2_events/G3.json, unchanged.
    """
    for m in EVENT_SPLIT:
        if EVENT_SPLIT[m] in ("train", "val"):
            _selfcheck(m)
    out = out or G3_JSON
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    tr = dataset(halves("train", arm))
    va = dataset(halves("val", arm))
    n_tr = len(tr.centres)
    coll = [sum(h.coll[i] for h in tr.halves + va.halves) for i in range(3)]
    n_lab = sum(h.n_landed for h in tr.halves + va.halves)
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"G3 - train on {len(tr.halves)} train halves, score val. "
          f"128057 (test) not loaded.\n"
          f"  ARM {arm.upper()} - {tr.ch.shape[2]} channels per player"
          + ("   *** ORACLE INPUT: the GT ball. NOT PRODUCT PERFORMANCE ***"
             if arm == "ball" else "") + "\n"
          f"  commit {commit}  labels {_sha(os.path.join(ROOT, E.INDEX))}  device {device}  "
          f"seed {seed}  n={n_configs} configurations tried\n"
          + ("  *** SMOKE RUN, ONE EPOCH, NOT A RESULT ***\n" if smoke else "")
          + f"  BUDGET = train-loss plateau: stop when the gain over "
          f"{G3_PLATEAU[1]} epochs is under {G3_PLATEAU[0]:.0%}, hard cap "
          f"{G3_CAP_MIN} min. Val is never consulted.\n"
          f"  also scoring val at fixed epoch {G3_CHECKPOINTS} (step 8's budget)\n"
          f"  target collisions {coll[0]} ({coll[1]} across classes), un-masked cells "
          f"{coll[2]}, of {n_lab} landed labels - step 8 defect 1, COUNTED not fixed\n"
          f"  train centres {n_tr}, {tr.ch.shape[1]} player slots, "
          f"{tr.ch.nbytes / 1e9:.1f} GB resident", flush=True)
    cps = {}

    def after(ep, net, losses):
        if ep in G3_CHECKPOINTS:
            cps[str(ep)] = _score_val(net, va, device)
            print(f"    [checkpoint at epoch {ep} scored]", flush=True)

    net, losses, why = fit(tr, epochs=1 if smoke else 400, seed=seed, device=device,
                           after=after, plateau=None if smoke else G3_PLATEAU,
                           cap_min=None if smoke else G3_CAP_MIN)
    train_min = (time.time() - t0) / 60
    cps["stop"] = _score_val(net, va, device)
    art = {"commit": commit, "labels_sha": _sha(os.path.join(ROOT, E.INDEX)),
           "seed": seed, "device": device, "n_configs": n_configs, "smoke": smoke,
           "arm": arm, "channels": int(tr.ch.shape[2]),
           "oracle": arm == "ball",
           "oracle_note": ("the GT ball is an ORACLE input - the product has no ball "
                           "at 6.8% recall. A DIAGNOSTIC, never model performance."
                           if arm == "ball" else
                           "the team sheet needs per-track identity, which this repo "
                           "does not have. A DIAGNOSTIC, not a product number."
                           if arm == "role" else ""),
           "collisions": {"cells": coll[0], "across_classes": coll[1],
                          "unmasked": coll[2], "landed_labels": n_lab},
           "role_unknown": sum(h.n_role_unknown for h in tr.halves + va.halves),
           "cache_sha": {f"{h.match} P{h.period + 1}": h.sha
                         for h in tr.halves + va.halves},
           "budget": {"rule": "train-loss plateau", "rel": G3_PLATEAU[0],
                      "window": G3_PLATEAU[1], "cap_min": G3_CAP_MIN,
                      "stopped_by": why, "epochs": len(losses)},
           "losses": [round(x, 4) for x in losses],
           "train": {"halves": len(tr.halves), "centres": n_tr,
                     "slots": int(tr.ch.shape[1]), "gb": round(tr.ch.nbytes / 1e9, 1)},
           "minutes": {"train": round(train_min, 1),
                       "total": round((time.time() - t0) / 60, 1)},
           "checkpoints": cps}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(art, open(out, "w"), indent=1)
    print(f"\n  -> {out}")
    show(out)
    return art


def show(path=G3_JSON):
    """Print G3.json. THE ONLY PRODUCER OF QUOTABLE NUMBERS FOR THE DOC.

    The training run is tens of minutes, which is past the timeout the push gate
    gives verify_claims. So the run writes the artefact once and every measured
    tag in st2-p1-step8 (not published) reads it through here in a second - the N-series
    pattern (model/validation/n43b_far_fixes.py --show). Exactly one #slow tag in
    that document names the producer.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing - run: python model/validation/st2_train.py --g3")
    r = json.load(open(path))
    b = r["budget"]
    arm = r.get("arm", "geo")
    print(f"G3 {r['commit']} labels {r['labels_sha']} seed {r['seed']} {r['device']} "
          f"n={r['n_configs']}" + ("  *** SMOKE, NOT A RESULT ***" if r["smoke"] else ""))
    print(f"  ARM {arm.upper()}, {r.get('channels', N_CH)} channels"
          + (f"; role unknown on {r['role_unknown']} slots" if arm == "role" else ""))
    if r.get("oracle_note"):
        print(f"  *** {'ORACLE INPUT (GT BALL)' if r.get('oracle') else 'DIAGNOSTIC ARM'}: "
              f"{r['oracle_note']} ***")
    if r.get("collisions"):
        c = r["collisions"]
        print(f"  target collisions {c['cells']} ({c['across_classes']} across classes), "
              f"un-masked cells {c['unmasked']}, of {c['landed_labels']} landed labels "
              f"(step 8 defect 1, counted not fixed)")
    print(f"  budget {b['rule']}: gain under {b['rel']:.0%} over {b['window']} epochs, "
          f"cap {b['cap_min']} min -> STOPPED BY {b['stopped_by'].upper()} "
          f"at epoch {b['epochs']}")
    print(f"  train {r['train']['halves']} halves, {r['train']['centres']} centres, "
          f"{r['train']['slots']} slots, {r['train']['gb']} GB; "
          f"{r['minutes']['train']} min training, {r['minutes']['total']} min total")
    print(f"  loss epoch 1 {r['losses'][0]:.4f} -> epoch {len(r['losses'])} "
          f"{r['losses'][-1]:.4f}"
          + "".join(f"; at checkpoint epoch {cp} {r['losses'][int(cp) - 1]:.4f}"
                    for cp in r["checkpoints"] if cp != "stop"))
    for cp, sets in r["checkpoints"].items():
        for name, v in sets.items():
            print(f"\n=== ARM {arm.upper()}"
                  + (" (ORACLE GT BALL - NOT PRODUCT PERFORMANCE)" if r.get("oracle")
                     else "")
                  + f" / {'STOP' if cp == 'stop' else 'EPOCH ' + cp} / VAL {name} - "
                  f"{v['periods']} periods. MATCH-LEVEL DROP "
                  + ", ".join(f"{k} {x:.3f}" for k, x in v["match_drop"].items())
                  + " (132877's is its no-footage third period; step 6)")
            print(f"  {'class':26s} {'n':>5s} {'AP@1s':>6s} | {'chance':>6s} {'metro':>6s} "
                  f"{'shift':>6s} {'worst':>6s}  verdict")
            for cls, c in v["classes"].items():
                print(f"  {cls:26s} {c['n']:5d} {c['ap']:6.3f} | "
                      + " ".join(f"{c[bn]:6.3f}" if c[bn] is not None else f"{'-':>6s}"
                                 for bn, _ in E.BASELINES)
                      + f" {c['worst']:6.3f}  "
                      + ("TIE  " if c.get("tie") else "BEATS" if c["beats"] else "loses")
                      + (f"  (inside floor noise {FLOOR_NOISE})" if c.get("tie") else "")
                      + ("  (floor 0.000)" if c.get("zero_floor") else ""))
            a = v["attribution"]
            cl = v["classes"].values()
            clean = sum(c["beats"] and not c.get("tie") and not c.get("zero_floor")
                        for c in cl)
            print(f"  {'-' * 70}\n  above their worst floor {sum(c['beats'] for c in cl)}"
                  f"/{len(v['classes'])}, of which CLEAN (not a tie inside floor noise, "
                  f"not over a 0.000 floor): {clean}")
            print(f"  ATTRIBUTION  matched {a['matched']}/{a['n_gt']}  model {a['model']:.3f}"
                  f"  nearest-player {a['nearest']:.3f}  chance {a['chance']:.3f}  -> "
                  + ("BEATS" if a["beats"] else "loses")
                  + " (nearest-player sees the feed ball; this model has none)")


def demo():
    """Targets, spots and the split guard, on data small enough to check by hand."""
    T, N = 400, 3
    xy = np.full((T, N, 2), np.nan, np.float32)
    xy[:, :2, 0] = np.linspace(0, 50, T)[:, None]
    xy[:, :2, 1] = np.array([10.0, 30.0])
    ch, on = features(xy, np.array([105.0, 68.0]), np.array([0, 1, -1], np.int8), smooth_s=0.4)
    assert ch.shape == (T, N, 10) and on[:, 2].sum() == 0
    assert abs(ch[T // 2, 0, 4] * 10 - 50 / (T / FPS)) < 0.1, ch[T // 2, 0, 4]   # speed m/s
    assert ch[0, 0, 9] == 1.0 and ch[0, 2, 9] == 0.0                             # visibility
    assert abs(ch[0, 0, 6]) < abs(ch[0, 0, 7])                                   # goal distances

    # G3B's ARMS. The geometry block must be byte-identical with the extra
    # channels on - arm A of G3B has to reproduce step 8, so an arm that
    # perturbed the first ten channels would void the whole comparison.
    ball = np.full((T, 2), np.nan, np.float32)
    ball[:, 0], ball[:, 1] = 20.0, 10.0
    ball[5] = np.nan                                       # a frame with no feed ball
    role = np.array([0, 3, -1], np.int8)                   # GK, FWD, and not listed
    chb, _ = features(xy, np.array([105.0, 68.0]), np.array([0, 1, -1], np.int8),
                      smooth_s=0.4, ball=ball)
    chr_, _ = features(xy, np.array([105.0, 68.0]), np.array([0, 1, -1], np.int8),
                       smooth_s=0.4, role=role)
    assert chb.shape[2] == ARM_CH["ball"] and chr_.shape[2] == ARM_CH["role"]
    assert (chb[:, :, :N_CH] == ch).all() and (chr_[:, :, :N_CH] == ch).all()
    # player 0 at (x, 10) with the ball at (20, 10): dy is 0, dx and the distance
    # agree, and both are in units of BALL_SCALE_M metres.
    assert abs(chb[0, 0, N_CH] - 20.0 / BALL_SCALE_M) < 1e-5, chb[0, 0, N_CH]
    assert abs(chb[0, 0, N_CH + 1]) < 1e-6
    assert abs(chb[0, 0, N_CH + 2] - 20.0 / BALL_SCALE_M) < 1e-5
    assert chb[0, 0, N_CH + 3] == 1.0 and chb[5, 0, N_CH + 3] == 0.0   # present flag
    # NO BALL IS ZERO AND FLAGGED, never "the ball is on my feet": the flag is the
    # only thing separating those two, and the displacement alone cannot.
    assert (chb[5, :, N_CH:N_CH + 3] == 0).all()
    # role one-hot: GK, FWD, and an unlisted player as all zeros, not a fifth line
    assert list(chr_[0, 0, N_CH:]) == [1, 0, 0, 0] and list(chr_[0, 1, N_CH:]) == [0, 0, 0, 1]
    assert not chr_[0, 2, N_CH:].any()

    # _mark: step 8's defect 1 is COUNTED, and the later label still wins, so
    # arm A stays bit-identical to step 8. A miscount here is the whole point.
    n = [0, 0, 0]
    t9 = np.zeros((3, 2), np.int64)
    t9[2, 1] = -100
    _mark(t9, 0, 0, "PASS", n)
    assert t9[0, 0] == IDX["PASS"] and n == [0, 0, 0]          # an empty cell is not a collision
    _mark(t9, 0, 0, "DRIVE", n)
    assert t9[0, 0] == IDX["DRIVE"] and n == [1, 1, 0]         # the LATER label wins
    _mark(t9, 0, 0, "DRIVE", n)
    assert n == [2, 1, 0]                                      # same class: a collision, not cross
    _mark(t9, 2, 1, "SHOT", n)
    assert t9[2, 1] == IDX["SHOT"] and n == [2, 1, 1]          # un-masked an off-pitch slot

    # TARGETS: one label at row 100 for player 1 marks centres within RADIUS and
    # nothing else, and an off-pitch player is ignored (-100), never null.
    centres = np.arange(0, T, STRIDE)
    tgt = np.zeros((len(centres), N), np.int64)
    tgt[~on[centres]] = -100
    for j in range(int(np.searchsorted(centres, 100 - RADIUS, "left")),
                   int(np.searchsorted(centres, 100 + RADIUS, "right"))):
        tgt[j, 1] = IDX["PASS"]
    assert (tgt == IDX["PASS"]).sum() == 1 and tgt[20, 1] == IDX["PASS"]   # centre 100 only
    assert (tgt[:, 2] == -100).all() and (tgt[:, 0] == NULL).all()

    # SPOTS: two peaks 0.8 s apart are two spots, 0.2 s apart are one, and the
    # emitted time is the peak's own. This is the NMS the gate's AP depends on.
    t = centres / FPS
    prob = np.zeros((len(centres), len(CLASSES) + 1), np.float32)
    prob[[20, 24], IDX["PASS"]] = [0.9, 0.8]          # 0.8 s apart at stride 5
    assert len(spots(prob, t)["PASS"]) == 2
    prob[:] = 0
    prob[[20, 22], IDX["PASS"]] = [0.9, 0.8]          # 0.4 s apart, two local maxima
    s = spots(prob, t, nms_s=0.5)["PASS"]             # inside nms -> the higher wins
    assert len(s) == 1 and abs(s[0][0] - centres[20] / FPS) < 1e-6, s
    assert spots(prob, t)["SHOT"] == []

    # THE ACTOR RIDES WITH THE SPOT. `who` holds the slot per (centre, class), so
    # the spot at centre 20 must carry slot 4 and not slot 9 from another class.
    who = np.zeros((len(centres), len(CLASSES) + 1), np.int32)
    who[20, IDX["PASS"]], who[20, IDX["SHOT"]] = 4, 9
    assert spots(prob, t, who, nms_s=0.5)["PASS"][0][2] == 4
    assert spots(prob, t)["PASS"][0][2] == -1              # no `who` -> never a free pass

    # A perfect spotter must score 1.0 through the shared scorer, and the split
    # guard must refuse test. Both are one line and both have been wrong before.
    gt = {"PASS": [(10.0, 3, 1.0, 0.05), (30.0, 7, 0.0, 0.05)]}
    assert abs(score({"PASS": [(10.0, 1.0, 3), (30.0, 0.9, 7)]}, gt)[1] - 1.0) < 1e-9
    assert score({"PASS": [(60.0, 1.0, 3)]}, gt)[1] == 0.0
    for m in ("128057", "132831"):
        try:
            _selfcheck(m)
        except AssertionError:
            continue
        raise AssertionError(f"_selfcheck let {m} through")
    # END TO END, on synthetic halves. `fit` grew a 3-tuple return, and a review
    # reported that a call site still wrote `net = fit(...)`. IT DID NOT - both
    # sites unpack - but nothing in the repo could have told you either way, and
    # a mistake of that shape only surfaces minutes later inside `predict`. This
    # check is the answer to that, and writing it found a different, real bug:
    # `if cap_min and` meant a zero-minute cap never fired.
    def fake(t, n, sd, n_ch=N_CH):
        r = np.random.default_rng(sd)
        cs = np.arange(0, t, STRIDE)
        tg = np.zeros((len(cs), n), np.int64)
        tg[3, 0] = IDX["PASS"]
        return SimpleNamespace(ch=r.normal(size=(t, n, n_ch)).astype(np.float32),
                               on=np.ones((t, n), bool), centres=cs, tgt=tg,
                               t=cs / FPS,
                               gt={"PASS": [(cs[3] / FPS, 0, 1.0, 0.05)]})
    h1, h2 = fake(200, 4, 0), fake(150, 6, 1)
    ds = dataset([h1, h2])
    assert ds.ch.shape == (350, 6, N_CH) and ds.tgt.shape[1] == 6      # padded to the wider
    assert (ds.tgt[h1.sl, 4:] == -100).all()   # h1 is the narrow half; its pad is ignored
    assert (ds.on[:200, 4:] == False).all()    # and those slots are off-pitch, not zeros
    assert ds.lo[h2.sl.start] == 200 and ds.hi[h2.sl.start] == 349     # no reach across
    # THE MODEL'S INPUT WIDTH COMES FROM THE DATA, not from N_CH: a G3B arm with
    # extra channels has to reach `embed`, and EventNet's default is still 10.
    wide = dataset([fake(120, 4, 2, ARM_CH["ball"]), fake(90, 5, 3, ARM_CH["ball"])])
    assert wide.ch.shape[2] == ARM_CH["ball"]
    assert fit(wide, epochs=1, batch=8, quiet=True)[0].embed[0].in_features == ARM_CH["ball"]
    net, losses, why = fit(ds, epochs=2, batch=8, quiet=True)
    assert len(losses) == 2 and why == "epochs", (losses, why)
    prob, who = predict(net, ds, batch=8)
    assert prob.shape == (len(ds.centres), len(CLASSES) + 1) == who.shape
    assert spots(prob[h2.sl], h2.t, who[h2.sl]) is not None
    # the two stopping rules fire, and say which one did
    assert fit(ds, epochs=9, batch=8, quiet=True, plateau=(1.0, 1))[2] == "plateau"
    assert fit(ds, epochs=9, batch=8, quiet=True, cap_min=0.0)[2] == "cap"

    print("st2_train demo ok: channels, targets, NMS, scorer, split guard, "
          "multi-half store, fit/predict/spots end to end, both stop rules, "
          "G3B ball/role arms and the collision counter")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", action="store_true", help="the deliberate-overfit gate")
    ap.add_argument("--g3", action="store_true", help="train on train, score val")
    ap.add_argument("--g3b", action="store_true",
                    help="the G3B diagnostic: --arm geo|ball|role into out/st2_ball_oracle")
    ap.add_argument("--arm", default="geo", choices=sorted(ARM_CH),
                    help="G3B channel set; ball is an ORACLE input")
    ap.add_argument("--smoke", action="store_true", help="g3 with one epoch, not a result")
    ap.add_argument("--show", action="store_true", help="print G3.json - the doc's tags")
    ap.add_argument("--show-g2", action="store_true",
                    help="print G2-e<epochs>.json - step 7's tags, without re-training")
    # G2_EPOCHS, not 40: every published step-7 tag runs --epochs 150, so a
    # different default is a reproduction hazard on the file's own headline.
    ap.add_argument("--epochs", type=int, default=G2_EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--configs", type=int, default=1, help="configurations tried, for n=")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.show:
        show(g3b_path(a.arm) if a.g3b else G3_JSON)
    elif a.show_g2:
        show_g2(a.epochs)
    elif a.g3b:
        g3(seed=a.seed, device=a.device, n_configs=a.configs, smoke=a.smoke,
           arm=a.arm, out=g3b_path(a.arm))
    elif a.g3:
        g3(seed=a.seed, device=a.device, n_configs=a.configs, smoke=a.smoke)
    elif a.g2:
        g2(epochs=a.epochs, seed=a.seed, device=a.device, n_configs=a.configs)
    else:
        demo()
