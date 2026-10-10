# N52: the clean ball training set — 101 MIT clips fetched, tiles built, split registered, and STILL only two matches

2026-09-20, branch `ball-clean`; **§7b added 2026-09-23, when D1 was trained and
scored.** Everything before §7b was written while nothing had run and is left in the
tense it was written in - that is the record, not an oversight. D1 now exists and is
SUPPORTED; **D2 still does not**, so §7a's four-outcome table is unresolved.
The rule is the docstring of
[`model/validation/ball_clean.py`](../../model/validation/ball_clean.py), committed with
this page and before any run it could judge.

Follows `n48-ball-detector-ab.md` (branch `picker-path`,
267f934), whose detector works and cannot ship, and
licence-audit (not published) blocker **B4**.

## The result in one line

TeamTrack (MIT) is the right source, and at scoping time **zero of the nine clips on
disk were eligible to train on** — @C 9,7649,0 :: this repo at commit d0856ec, `ball_clean.py census`
before the fetch (clips present, ball positions, clips whose `role()` is train or valid).
Six of the nine *are* the population N48–N51 score on, two are where `ball.BallPicker`'s
gates were fitted, one is the dead ball `ball_cmp` excludes by name. So the clean rebuild
was never blocked on a cleverer split; it was blocked on **fetching clips that exist, are
MIT, and are free**.

**They have now been fetched**, and the state today is:

@M 110,101 :: .venv/bin/python model/validation/ball_clean.py fetch
   clips in `data/footage/soccertrack/` and clips eligible to train or validate on.
   `fetch` is resumable and re-running it is a no-op that re-prints, which is why this
   tag still runs. Licence line printed by the same command.

@M 110,89015,101 :: .venv/bin/python model/validation/ball_clean.py census
   clips on disk, ball positions across all of their annotation CSVs, and clips whose
   `role()` is train or valid. n=1, no configuration.

## 1. What TeamTrack actually gives — and the two-match limit, up front

Two sports splits, and every clip of each is **one match**: `soccer_top` is
`D_20220220_1`, `soccer_side` is `F_*_20220220_1`. **Two matches, two venues, two
cameras — read this before any number below.** A clip-disjoint split inside one match
holds the camera, the pitch, the ball, the kits and the light fixed, so a held-out clip
here is a held-out **minute**, not a held-out match. Every score this dataset can ever
produce, including every score in N48–N51, is a **same-match** score.

**What that costs, as a number rather than a warning:**

@C 0.00987,0.48761,0.0198,0.18,1.0 :: `ball_clean.py variance` in the working repository (the `.cmpB` input is not distributed)
   between-match and within-match sums of squares, the between share, then the lowest
   and highest of N48's six per-clip arm-B `oracle` rates. One-way decomposition by
   match over the `.cmpB` caches; a replay, nothing detects. n=1. Those caches are output of the
   DFL-trained detector, whose data may not be redistributed, so they are not in this repository.

Only @D 1.98% :: 0.00987 / (0.00987 + 0.48761) [the two sums of squares above] of the
per-clip spread is between the two matches;
@D 98.02% :: 1 - 0.0198 is within one. **That is not permission to score same-match.**
Two groups give one degree of freedom, and the within-match spread runs from a clip the
detector gets on every frame to one it gets on under a fifth — that swamps any venue
term, so this decomposition cannot resolve a match effect, it can only show that this
data has no room to look for one. The consequence is concrete: a clean model that beats
stock here has been shown to beat it **on this camera, on this pitch, on this ball**, and
nothing more. Generalisation needs a third commercially-licensed source
(`data/footage/soccertrack/README.md` §5), which remains the open item.

@M 126,101,matches 2 :: .venv/bin/python model/validation/ball_clean.py plan
   soccer clips in `teamtrack_zip_index.json`, clips this page's rule makes eligible to
   train or validate on, and distinct matches. Counting dates would say three — `soccer_side`'s train filenames carry an upstream
   typo (`F_20200220` to 870 s, `F_20220220` after) for one match.

The nine clips that were on disk before the fetch, and what each is already spent on.
`census` now prints all 110, these nine among them:

| clip | res | ball frames | GT box p5/p50/p95 px | median step px | frozen | role |
|---|---|---|---|---|---|---|
| side train 0000-0030 | 6500x1000 | 750/750 | 7 / 9 / 10 | 3.00 | 26.3% | **scored** |
| side train 0030-0060 | 6500x1000 | 750/750 | 6 / 8 / 10 | 2.24 | 10.4% | **scored** |
| side train 0060-0090 | 6500x1000 | 750/750 | 8 / 21 / 21 | 5.00 | 0.0% | **scored** |
| top test 1680-1710 | 3840x2160 | 900/900 | 20 / 20 / 20 | 0.03 | 65.6% | excluded, degenerate |
| top test 1710-1740 | 3840x2160 | 899/900 | 11 / 11 / 14 | 0.10 | 50.3% | **scored** |
| top test 1740-1770 | 3840x2160 | 900/900 | 12 / 13 / 15 | 4.18 | 30.9% | **scored** |
| top test 1770-1800 | 3840x2160 | 900/900 | 15 / 15 / 15 | 4.69 | 1.6% | **scored** |
| top train 0000-0030 | 3840x2160 | 900/900 | 10 / 12 / 14 | 5.00 | 7.1% | excluded, gates fitted |
| top train 0030-0060 | 3840x2160 | 900/900 | 11 / 12 / 13 | 4.12 | 1.2% | excluded, gates fitted |

@M 899,0.03,0.10,4.18,4.69,5.00,4.12,65.6,50.3,30.9,1.6,7.1,1.2 :: .venv/bin/python model/validation/ball_clean.py census
   the one clip with 899 of 900 ball frames, then the median
   frame-to-frame GT ball step for the four `soccer_top` clips in printed order, then
   the share of steps under 0.5 px for those clips plus the two train clips. **Whole
   clips**, not N48's first-300-frame window, so these do not restate N49's numbers —
   `top test 1710-1740` reads 50.3% frozen over 899 steps where N49 read 299 of 299 over
   the first 300. Boxes are TeamTrack's annotation boxes, not measurements
   (`ball_eval.py` header). n=1, no configuration.

**The ball is annotated on essentially every frame** — @D 0.9999 :: 7649/7650 — which is
why this is the only ball ground truth the repo has.

## 2. How many genuinely independent clips survive: 101 clips, and still 0 independent matches

The three exclusions stack, and before the fetch they took everything:

* **scored** — `ball_cmp.CLIPS`, six clips. Training on them then scoring on them is the
  overtuning this repo exists to catch.
* **gate-fitted** — `soccer_top` train 0000-0060. `ball.BallPicker`'s `ASPECT` and
  `SIZE_RATIO` were fitted there (`ball_eval.py:112`). A detector trained on the same
  pixels those constants were shaped on is fitted to them a second time, indirectly.
* **degenerate** — `top test 1680-1710`: the ball is carried forward motionless, which
  the census reads as @M 0.03 :: .venv/bin/python model/validation/ball_clean.py census
  px per frame with a constant 20 px box. `ball_cmp` already refuses it.

@M train 40,valid 9,train 42,valid 10,excluded 8,excluded 11 :: .venv/bin/python model/validation/ball_clean.py plan
   `soccer_top` train / valid / `soccer_side` train / valid clips under this page's rule,
   then the excluded count for each sport. All 101 are now on disk.

**And the limit no split fixes**, restated because it is the part a reader skips. The
101 eligible clips are 30 s windows of the *same two matches* as the scored ones — same
camera, pitch, ball, kits and light, §1's 1.98%. So:

> Any score this data produces is a **same-match** score. It can answer "does the clean
> model beat stock at the same tile size on this footage". It cannot answer "does it
> generalise", and no rearrangement of TeamTrack will. That needs a third
> commercially-licensed source, which is `README.md` §5's open item.

## 3. The split, registered here, before any training run

`ball_clean.role()` is the rule; `--demo` asserts it rather than describing it.

| role | what | why |
|---|---|---|
| **scored** | the six `ball_cmp.CLIPS` | never trained, never validated, never looked at while choosing |
| **excluded** | `top` 0000-0060; `top` 1680-1710; `side` 0090-0120; all of TeamTrack's own `test` split | gates fitted / degenerate / abuts a scored window / our scored `top` clips live inside it |
| **train** | the rest of TeamTrack's `train` split, both sports | |
| **valid** | TeamTrack's `val` split, both sports | checkpoint selection only, and the same two matches as `scored` |

Four properties the self-check asserts, so they cannot rot: no scored clip is ever train
or valid; a scored, gate-fitted, degenerate or buffer clip can never reach the tiler
however many clips arrive; no train/valid clip overlaps **or abuts** a scored window in
the same match; and the tile window always contains the ball and stays inside the frame.

The second of those replaced a stronger one. Until the fetch ran, the self-check asserted
*"nothing on disk is eligible"* — which was the scoping finding, and which the fetch
falsified by design. An assertion that encodes a transient state is a check that fails
for the right reason at the wrong time; the invariant that outlives it is the one now in
its place.

@M 126,101,110,101 :: .venv/bin/python model/validation/ball_clean.py --demo
   clips indexed, eligible under the rule, on disk, and eligible-on-disk.

## 4. The builder, and the set it produced

Tiles are **640 px crops at native scale**, which is what `ball_cmp` arm B feeds the
detector (`detectors._spans`, tile 640, overlap 0.10) — `T` and the random-offset window
are imported from `ball_train.py` rather than re-declared, so the clean rebuild changes
the data and nothing else. Boxes are TeamTrack's own, clipped to the tile by
`ball_train.clip_box`.

One deliberate departure, declared: **`NEG_KEEP = 4`**, one ball-free tile per four with
a ball. N48's review found every DFL training tile contained a ball, so nothing in that
run could see a false positive on empty pitch — and false positives are what N49–N51 lost
to (@C 36.6,22.5 :: `n48-ball-detector-ab.md` §Next, branch
`picker-path` 267f934 — arm B candidates per frame against arm A's).

### 4a. The real build: 82 train clips, 19 valid, 20,307 tiles

@M 16509,13227,3282,82,3798,3042,756,19 :: .venv/bin/python model/validation/ball_clean.py tiles
   train tiles / boxes / empty tiles / clips, then the same four for valid. Every
   eligible clip, every 5th ball-annotated frame, seed 0, `NEG_KEEP` 4, tile 640 at
   native scale. n=1 configuration, nothing tuned.

| | train | valid |
|---|---|---|
| clips | 82 | 19 |
| tiles | 16,509 | 3,798 |
| tiles with a ball | 13,227 | 3,042 |
| ball-free tiles | 3,282 | 756 |
| box w p5/p50/p95 px | 6 / 14 / 20 | 9 / 14 / 29 |
| box h p5/p50/p95 px | 6 / 12 / 20 | 9 / 14 / 24.9 |
| median box area, share of tile | 0.0004126 | 0.0004785 |
| ball centre in tile, x p5/p50/p95 | 0.183 / 0.500 / 0.816 | 0.184 / 0.516 / 0.815 |
| boxes cut by the tile edge | 29 | 2 |

@M 6.0,14.0,20.0,0.0004126,0.183,0.5,0.816,29,9.0,29.0,0.0004785,2 :: .venv/bin/python model/validation/ball_clean.py tiles
   train box w p5/p50/p95, median box area as a share of the tile, ball-centre x
   p5/p50/p95 and edge-cut boxes; then valid box w p5 and p95, its area share and its
   edge-cut boxes. Same run as above.

Ball-free tiles are @D 0.199 :: 3282/16509 of train, which is the 1-in-5 `NEG_KEEP = 4`
asks for. The centre distribution spans the declared `OFFSET` band with a median at 0.5,
so "the ball is in the middle of the tile" is not learnable. Edge-cut boxes are
@D 0.0022 :: 29/13227 of train boxes - a ball close enough to the frame border that the
window clamped; `ball_train.clip_box` clips the box rather than shifting or dropping it.

**The drop share, and its reason.** @D 0.00053 :: 7/13234 - seven wanted frames of
13,234 became no tile, all in `train`, none in `valid`:

@M 13234,7,3042,0 :: .venv/bin/python model/validation/ball_clean.py tiles
   frames wanted and frames dropped, train then valid, from each split's `drops` block.
   `drops` is accumulated by the builder, not reconstructed afterwards.

All seven are the **same annotation artefact**, not a hard case the builder quietly
refused: six `soccer_top` clips carry a ball row for frame 900 (one of them for 905 too)
while their video holds exactly 900 frames, indexed 0-899. The CSV annotates past the
end of the clip. Nothing else was dropped - box count equals wanted minus these seven in
both splits, which is the arithmetic the tag above prints.

### 4b. Composition: 53% top-down, 47% side panorama - and the side is the one that matters

`soccer_side` at 6500x1000 is a stitched panorama, which is what **our own footage looks
like**; `soccer_top` at 3840x2160 is a near-nadir drone view, which is not.

@M 8759,7010,40,7750,6217,42,1975,1823 :: .venv/bin/python model/validation/ball_clean.py tiles
   train `soccer_top` tiles / boxes / clips, then train `soccer_side` tiles / boxes /
   clips, then valid tiles for each. Same run.

| | top-down 4K | side panorama |
|---|---|---|
| train tiles | 8,759 (40 clips) | 7,750 (42 clips) |
| valid tiles | 1,975 (9 clips) | 1,823 (10 clips) |
| box w p5/p50/p95 px (train) | 11 / 14 / 19 | 6 / 10 / 25 |
| median box area, share of tile | 0.0004785 | 0.0002417 |

@M 11.0,14.0,19.0,0.0004785,6.0,10.0,25.0,0.0002417 :: .venv/bin/python model/validation/ball_clean.py tiles
   train box w p5/p50/p95 and median area share for `soccer_top`, then for
   `soccer_side`. Same run.

**The composition is not the problem the question anticipated.** The split is
@D 0.531 :: 8759/16509 top-down,
@D 0.469 :: 7750/16509 side, near enough even, and valid is
@D 0.520 :: 1975/3798 top. No weighting is needed to stop the model being trained on the
wrong geometry, because it is not.

### 4c. Two geometries, one nano: what I expect, and which half the product cares about

**They are not two disjoint problems — one is the narrow mode inside the other.** The
top-down box width runs 11–19 px p5–p95 and the side panorama's runs 6–25 px, so
top-down sits **entirely inside** side's range. The set is one continuous size
distribution with a spike in the middle, not a bimodal mixture, and the model never has
to represent a size that only one half supplies.

@M 11.0,19.0,6.0,25.0 :: .venv/bin/python model/validation/ball_clean.py tiles
   train box width p5 and p95 for `soccer_top`, then for `soccer_side`. Same build.

**So my expectation is better scale robustness, not mediocrity at both — and it is a
prediction, not a measurement.** The mechanism: RF-DETR resolves a 6 px and a 25 px ball
at different pyramid strides, and the side clips supply *real* scale variation (a ball
near and far under one camera) where `ball_train`'s DFL run had `multi_scale=False` and a
single broadcast scale. Real scale variation is the better version of the augmentation
that run switched off.

**The risk is not scale, it is the box regression head**, and it is N50's defect in a new
costume. The two halves differ in appearance as well as size — nadir drone against a
stitched panorama with seams and edge chromatic fringing (`README.md` §1) — and a head
that learns one geometry's box convention and applies it to the other is exactly how the
DFL nano came to draw @C 3.2 :: `n50-ball-size-gate.md` §6
(branch `picker-path` 267f934) times the ball's silhouette on a ball at rest. §6's two
box-size tags must therefore be read **per sport**, not pooled.

**Which half matters: the side panorama, decisively.** Our own footage is a stitched
panorama from the same University of Tsukuba rig (`README.md` §1 records
`soccer_side` as that venue), and the target setup has one panoramic camera.
A 4K nadir drone is not a format that setup has or will have. So **half of this training
set, and half of the scored population, is a geometry the product will never see.**

That already shows in N48's own arm B, which pools two quite different numbers:

@C 0.6744,0.7556 :: `ball_clean.py variance` in the working repository (the `.cmpB` input is not distributed)
   arm B's mean per-clip `oracle` for the three `soccer_top` clips and the three
   `soccer_side` clips. Equal frame counts, so these pool to N48's published 0.7150
   (@D 0.7150 :: (0.6744 + 0.7556) / 2). The `.cmpB` caches, a replay; not distributed (see above).

**Pre-registered, before the clean arm exists:** every clean-arm result reports the
**side-only** figure beside the pooled one, because side is the product-relevant
geometry. Beside it, never instead of it — dropping the top-down clips would move the
headline from 0.7150 to 0.7556 on arm B alone, and a filter that raises a number is the
change [the working rules](../README.md#working-rules) say to distrust most. The pooled number stays the ruled-on one; the
side number is what anyone reasoning about the product should quote.

## 5. What would have to be true for the clean model to be comparable to the blocked one

N48's numbers are on record — @C 0.1972 :: `n48-ball-detector-ab.md`
(branch `picker-path` 267f934), arm B minus arm C, pooled `oracle`, six clips, first 300
frames, conf 0.05, 20 px gate. **That is not a target the clean model can be held to**
without four things, and the fourth is not obtainable here:

1. **The same ruled-on delta.** B − C, not B − A. Arm C (stock weights at tile 640) is
   already built and its caches are committed, so the clean arm re-uses them unchanged.
2. **The same population and gate.** `ball_cmp.CLIPS`, first 300 frames, conf 0.05, 20 px,
   `ball.BallPicker` unmodified, each clip's own calibration — and the same statement of
   what that population contains: a third of it is a dead ball
   (`n49-ball-on-the-paint.md`, branch `ball-clip-gap` f26108d).
3. **A matched training budget.** N48's arm B trained on 916 train tiles for 12 epochs at
   nano/640 (`ball_train.py`'s rule). A clean run with ten times the tiles that wins has
   measured the tile count, not the licence. Match the count and the schedule, or run both
   and say which varied.
4. **The confound that cannot be removed.** The DFL model was trained **out of domain**
   (broadcast) and scored on TeamTrack. The clean model trains **in domain** — the same two
   matches, the same cameras, minutes away from the scored windows. A clean win is
   therefore evidence of domain match first and data quality second, and the two are
   inseparable on this data. Say it on the page, or the comparison is the wrong-protocol
   failure in [the working rules](../README.md#working-rules)' list, one more time.

**The honest framing of the clean run, decided now:** it is judged against **arm C**, not
against arm B. The question is "does a shippable detector beat the shipped one at matched
tile size", pass bar +0.05 on pooled `oracle` as before. Whether it also beats the blocked
arm B is reportable, never ruled on.

## 6. The box-size defect, and how to find out whether the clean rebuild inherits it

`n50-ball-size-gate.md` §6 measured, on `top 1710-1740`:

@C 1.30,1.81,1.79,4.12 :: `n50-ball-size-gate.md` (branch
`picker-path`, 267f934) — the ball's own silhouette, the annotator's box, the stock
detector's box and the DFL-trained nano's box, each as a multiple of `_expected_px`.
The stock detector reproduces the annotator; the fine-tuned nano draws
@D 3.2 :: 4.12 / 1.30 [n50 §6, cited above] times the ball's silhouette on a ball at rest, which is
what `SIZE_RATIO`'s 3.0 ceiling rejects.

**The hypothesis this rebuild tests for free:** the box regression learned the *label
convention of the DFL re-export*, not the ball. TeamTrack's own boxes are
@D 1.39 :: 1.81 / 1.30 [n50 §6, cited above] times the silhouette, which is inside the shipped gate.
Train on TeamTrack boxes and the model should reproduce TeamTrack's convention.

**What to measure, pre-registered, two tags and no new instrument:**

1. `ball_gate.py --size` on the clean arm's candidate cache: median true-ball box over
   `_expected_px`, and the share inside `(0.4, 3.0)`, per calibrated clip, split by play
   state (open play / at rest) exactly as N50 split it. **Pass** = the at-rest ratio sits
   within the shipped bounds and near the stock detector's, on all three calibrated clips.
2. `ball_gate.py --expected`: the clean model's box over the ball's `blob_px` silhouette.
   **Pass** = near TeamTrack's own 1.39, not the nano's 3.2.

Both commands exist on branch `picker-path` and are replays over a candidate cache — no
training and no re-detection beyond building the clean arm's cache once.

**What this cannot settle.** The direct test — are the DFL export's label boxes oversized
— is **foreclosed**: the data was deleted under B4 and re-fetching it would re-breach the
grant. So "the convention came from the labels" stays an inference, and the measurement
above can only show whether the clean model has the defect, not prove where the old one
got it.

## 7. The training budget I propose — NOT RUN

Nothing here has been trained. This is the budget, written down before the machine is
asked for, so that what runs can be checked against what was intended.

**The problem the budget has to solve is §5.3.** The clean train set is
@D 18.0 :: 16509/916 times the DFL arm's 916 tiles. Run the same 12 epochs over it and
any win mixes the data SOURCE with the data VOLUME, which is the N48 tile-size confound
again in a different costume. So two arms, and **the matched one is the comparison**:

| arm | train tiles | epochs | what it answers |
|---|---|---|---|
| **D1, matched** | 916, sampled clip-stratified from the 82 at seed 0 | 12 | is MIT data as good as DFL data *at equal volume* — the arm ruled on |
| **D2, full** | all 16,509 | 6 | what the whole clean set gives — the product candidate, reported, not ruled on |

Everything else is `ball_train.train` unchanged: RF-DETR **nano at resolution 640**,
batch 4 x grad-accum 4, lr 1e-4, `multi_scale=False`, no early stopping, and the
checkpoint rfdetr keeps as best on VALID (`checkpoint_best_ema`). Valid is the same
3,798 tiles for both arms, so checkpoint selection is not a third variable.

D1 drops @D 0.931 :: 1 - 916/13227 of the train boxes on purpose. That is a filter, and
[The working rules](../README.md#working-rules)' rule about filters applies: it is declared here as a matched-budget control,
its drop share is stated, and it is never quoted as the better number if it happens to
win.

**Wall clock, and every figure below is an estimate rather than a measurement.**

@C 9 :: `model/validation/ball_train.py` docstring — minutes per epoch at 916 tiles on
this Mac, for the DFL run.

@D 0.59 :: (9 * 60) / 916 — seconds per tile, if the rate carries over.
@D 162 :: 16509 * 0.59 / 60 — minutes per epoch for D2 at that rate.
@D 16.2 :: 162 * 6 / 60 — hours for D2's six epochs.
@D 1.8 :: 9 * 12 / 60 — hours for D1's twelve epochs at 916 tiles.

**Measure the first epoch before committing to the rest.** That per-tile rate comes from
a different tile mix on a machine in a different state, so a scaled docstring figure is
a derivation, not a run — which is why it is tagged as one and must not be quoted as a
measured training time.

The pass rule is already fixed in §5: judged against **arm C** (stock weights at tile
640), bar +0.05 on pooled `oracle` over `ball_cmp.CLIPS`. **D1 is the arm the rule rules
on. D2 is reported beside it and never ruled on.** That is decided here, before either
runs, because the one thing that must not happen is choosing the ruled-on arm after
seeing which one won.

### 7a. What each outcome would MEAN — written before either arm exists

Two arms, one bar, four outcomes. The reading of each is fixed now:

| | D2 (full) passes | D2 fails |
|---|---|---|
| **D1 (matched) passes** | **1. Clean data works, and volume does not hurt.** The licence problem is solved on this population: TeamTrack matches DFL at equal volume, and the full set is a product candidate. Strongest outcome. | **2. A SCHEDULE failure, not a data failure.** D2 sees 18x the tiles in half the epochs at the same lr — fewer passes per tile and a different effective schedule. "More clean data is worse" is the wrong reading and must not be written. Action: fix D2's epochs/lr and re-run D2 only; D1 stands as the result. |
| **D1 fails** | **3. The clean set needs VOLUME to match DFL.** At equal volume TeamTrack loses — plausible, since DFL is broadcast diversity and TeamTrack is two matches. The ruled-on verdict is **NOT SUPPORTED**, and D2's pass is reported as what the full set gives, not promoted to the headline. Do not swap the ruled-on arm. | **4. The clean rebuild does not beat stock@640 here.** Then the blocked model's @C 0.1972 :: `n48-ball-detector-ab.md` (branch `picker-path` 267f934) is not reproducible from legal data, and the next move is a third licensed source or a different architecture — **not another sweep**. Three pre-registered picker negatives already exist (N49–N51); a fourth clean negative here would be a real finding, not a failure to iterate. |

Three commitments that go with the table:

- **The pair is two configurations.** Anything phrased as "the better of D1 and D2"
  carries `n=2`, because both are scored on the same held-out population. The bar
  applies to D1 alone precisely so that D2's existence does not turn the run into a
  two-point search over the test clips.
- **Outcome 2 is the one most likely to be told as a story.** A D2 that underperforms
  has an obvious tempting narrative ("the extra clips are noisier") and an obvious dull
  one (it trained for less time per tile). The dull one is the default reading and only
  a matched-epoch D2 may displace it.
- **Both arms report the side-only figure beside the pooled one** (§4c), because half
  the scored population is a geometry the product will never see.

## 7b. RESULT — D1 trained, scored, SUPPORTED (2026-09-23)

D1 ran: 916 clip-stratified tiles, 12 epochs, RF-DETR nano at 640, the budget in §7
unchanged. Scored as `ball_cmp` arm D against the same six clips, same gate, same
picker as N48's arms.

    @M 0.8822,0.4706,0.3644,86.2 :: .venv/bin/python model/validation/ball_cmp.py --show
       arm D pooled oracle / shipped picker / D - C / candidates per frame. The bar is
       +0.05 on pooled oracle against arm C, committed 2026-09-20 in d0856ec and fixed
       on D1 as the ruled-on arm in bb38497, both before D1 existed -> SUPPORTED.
       1800 gt-present frames over 6 clips, conf 0.05, tile 640, gate 20 px.
       n=1 configuration; D2 has not run, so the pair is n=1, not the n=2 §7 proposes.
       READ 86.2 BESIDE THE VERDICT: arm C emits 34.5 candidates a frame and arm D
       2.5x that, and `oracle` - is the ball anywhere in the list - can only rise with
       the length of the list. No precision term is in this bar. That is N48's
       tile-size confound in a third costume, on the arm carrying the verdict.
       The same-match caveat under the D-B tag below applies here too: D1's checkpoint
       was SELECTED on a valid split from the same two matches these clips come from.

    @M 0.1672,0.0650 :: .venv/bin/python model/validation/ball_cmp.py --show
       D - B on oracle and on the shipped picker. NOT a data-source comparison: D1
       trained on other minutes of the SAME two matches these clips come from and B
       has never seen either venue, D1 saw 187 ball-free tiles to B's 0, and the two
       chose checkpoints on different valid splits. The command prints all three
       caveats beside the number.

    @D 27.1 :: 325 / 12 [wall minutes from out/ball_clean/d1.log's first and last
       stamps, 12 epochs] — minutes per epoch, against §7's assumed 9. The budget
       there is wrong by 3x on this machine; D2 at 16,509 tiles is not a 3 h job.

The curve plateaus at epoch 4-5 (val mAP50 0.6973 at 4, 0.7021 at 11), so the
12-epoch budget was not the binding constraint. **D2 has NOT run**, so this page's
§7a four-outcome table is still unresolved and the pair is n=1, not n=2.

Where the remaining loss sits, on arm D's own candidates: the detector never
proposes the ball on 11.8% of frames (72% of those are balls AT REST), and the
picker loses another 41.2% where a correct candidate was in the list. The picker is
now the larger half, which it was not under arms A or B.

## 8. Next, in order

1. ~~**Fetch.**~~ DONE — 101 clips, 2.10 GB, MIT, in `data/footage/soccertrack/`.
2. ~~**Build the tiles.**~~ DONE — §4a. They live in the durable store at
   `validation/out/ball_clean/tiles/{train,valid}`, beside `out/ball/`, **not** in a
   temporary scratch directory: that is how the 2026-08-29 ball checkpoint was lost
   (`ball-detector-trained-not-yet-scored` (internal note)).
   `ball_clean.py tiles` now re-prints that build instead of re-cutting it; `tiles
   rebuild` forces the ~35 minutes of 4K decode.
3. ~~**Train D1**~~ DONE 2026-09-23, §7b: 12 epochs, 5.42 h, SUPPORTED against arm C.
   **D2 still HELD** - and re-cost before it starts: the measured rate is 27.1 min an
   epoch against §7's assumed 9, so 16,509 tiles x 6 epochs is not the job §7 priced.
4. `ball_cmp` with the clean checkpoint as a fourth arm, ruled on against **C**.
5. §6's two box-size tags, on the clean arm's cache.
6. A third licensed source, for the only question TeamTrack can never answer (§1).

## What this page does NOT say

- **D2 was not trained and not scored**, so nothing here is a statement about what
  the FULL clean set gives; §7b is D1, the matched arm, alone.
- Nothing about the product: `jobs.py` still passes no `ball=`, so the pipeline runs no
  ball at all (`ball-numbers-measured-at-the-wrong-conf` (internal note)).
- Nothing about whether the clean model will beat anything. It does not exist.
- Input dropped, stated: the build dropped @D 0.00053 :: 7/13234 of wanted train frames,
  all of them annotation rows past the last frame of their video (§4a). The census reads
  every on-disk clip and every annotated frame of each.
- Nothing about the box-size defect. §6 says what to measure; no clean model exists to
  measure it on.
