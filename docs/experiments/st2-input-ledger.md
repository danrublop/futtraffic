# G-1: the input ledger

**What every number in `st2-sprint.md` is conditioned on, measured on one half, on
one day, against one ground truth.** Run 2026-09-07 on `118575_1st` (a TRAIN match)
— `data/jobs/118575_full_1st` + `model/validation/soccertrack/118575_1st.tracks.json.gz`.

Before this, the plan's input figures came from six experiments at six different
times on different data. This is the first time they have been reported together.

## Detection — `assoc_eval.py`, 15,000 frames

```
SHIPPED (cbiou)   recall 0.628   precision 0.691
as first published  recall 0.652   precision 0.669   <- FALLBACK tracker, see below
```

**Recall by width band (a depth proxy) is the finding:**

```
band   -2     -1      0      1      2      3      4      5      6      7      8      9
rec   0.00  0.011  0.148  0.320  0.497  0.726  0.645  0.801  0.948  0.987  0.971  0.859
      <---------------- FAR SIDE ---------------->        <------ NEAR SIDE ------>
```

A **6.7x gradient across the pitch.** Events on the far side largely have no actor
to attribute to. This is the camera-placement problem already recorded in
`far-side-jitter-is-camera-placement-not-calibration` (internal note), now quantified as recall.

## Tracking and identity — `assoc_eval.py`, `contam_eval.py`

```
IDF1                    0.155
pred tracks               796   for 22 players
  attributed              384   of which impure 322
fragments per player     47.3
id switches             2,775   in 15,000 frames (10 min)
mean track life          16.2 s
contamination            33.2%  per detection
```

**16.2 s of track life against a 1.6 s median inter-event gap** is the one
encouraging number here: a track survives ~10 events, so *local* attribution has
something to hold onto even though *player-level* identity does not exist.

## Position — `dist_eval.py`

```
coverage (mean)              0.70
distance ratio, matched      1.139   median   (IQR 1.082-1.234)
distance ratio, raw          0.795
peak speed   GT max 9.72 m/s   ours max 29.36 m/s
             179 / 384 fragments peak above MAX_SPEED 9.5
orphan ids                     23   emitting 1.42 km for nobody
```

**Nearly half our fragments contain a physically impossible jump.**

**And the ruler is blunt.** GT player positions are on the same **1.05 x 0.68 m
integer grid** as the ball (verified: every x an exact multiple of 1.05, every y of
0.68; 120 unique x over 1.59 M rows). Quantisation alone contributes
**~0.36 m RMS** (1.05/sqrt(12) in x, 0.68/sqrt(12) in y). Any positional-accuracy
claim below ~0.4 m on this data is measuring the grid, not the system — and the
"1.010 m on players" figure has that floor inside it. `oracle_eval.py --gt-smooth`
already existed to remove this; nothing else accounts for it.

## Team — `team_eval.py`

```
detection-level accuracy    0.697    (home 0.690 / away 0.703)
track-level accuracy        0.930    over 200 scored tracks
```

**This contradicts the 15.4% track-level figure quoted in `model/engine/events.py:415`
and repeated throughout `st2-sprint.md`.** Most likely the two score different
populations — 200 scoreable tracks here versus all 796 including short fragments —
but that is a guess. **Resolve it before P2 treats team as a corrupted input**;
the plan's §P2 currently predicts team will dominate, on the strength of a number
that may not mean what it says.

## Ball — `ball_eval.py`, 900 gt-present frames

```
detector never proposed it   718  (79.8%)
detector had it, picker lost  43  ( 4.8%)
picker right                 139  (15.4%)
oracle ceiling                     20.2%
centre control                      0.3%
```

**The ball does not exist on this footage.** 80% of the loss is the detector, not
the picker. Excluding it from the model's channels (§P1) is not a preference.

## What this does to the plan

| capability | verdict |
|---|---|
| **Event spotting** | Plausibly survives. It reads the configuration of bodies, and 65% recall still carries most of the shape — **except on the far side**, where 15% recall means the actors are simply absent. |
| **Local attribution** (which track touched it) | Plausibly survives. 16.2 s track life vs 1.6 s event gap. |
| **Player-level anything over a match** | Does not survive. 47 fragments/player, IDF1 0.155. Consistent with the product's existing refusal. |
| **Ball-derived features** | Dead, as already assumed. |

**The biggest consequence is for P2.** The plan frames pipeline-vs-GSR as a
"tracking tax" — a delta. It is not a delta: the model expects 22 coherent players
and would be handed **796 fragments with 33% contamination**. That is a different
input, not a degraded one. P2 needs a fragmentation-tolerant formulation (variable
N with a mask, no persistent player identity) or it needs tracking fixed first.

**P1 on GSR is unaffected and still worth running** — it establishes the ceiling,
and the ceiling is what tells us whether to spend the next month on the event model
or on the tracker.

## Not yet measured

- Only **one half of one train match**. 117092_1st/2nd and 117093_1st (a VAL match)
  have GT tracks converted and are downloading; the detector pass is ~4 h/half.
- **No external comparison.** All three SoccerTrack v2 leaderboards (MOT, GSR, BAS)
  are empty, so these numbers have no published yardstick.

---

## G1 addendum: the attribution floor, measured

`118575` p0, nearest GT track to the **GT ball** at the labelled frame, actor
resolved BAS `player_id` -> `player_nodes.back_number` -> GT `(team, jersey)`
(team side recovered by jersey-set overlap, 22/22 exact).

```
scored 987 of 1062 events   coverage 92.9%
ACTOR correct   878/987 =  89.0%      chance 1-of-22 = 4.5%
TEAM  correct   898/987 =  91.0%      chance 1-of-11 given team = 9.1%

PASS      440  91.8%      TACKLE   19  42.1%
DRIVE     411  91.7%      BLOCK    14  50.0%
HIGH PASS  62  91.9%      OUT      13   7.7%
CROSS      15 100.0%      SHOT     10  70.0%
```

Independently reproduces the adversarial review's 89.4%.

**Independently recomputed 2026-09-09: HOTA 0.5212 / AssA 0.5227, exact.** Its
box set is a strict subset of the UNMASKED conf-0.3 run (containment 1.0000), so
the withdrawn `on_pitch` bug never touched it. A from-scratch rebuild with
different matching gives 0.5430 / 0.5323. **"Association, not detection, is the
constraint" stands** - the headroom is ~2x either way.

**But the oracle is NOT realisable, and this was never written down.** A pure
relabelling of our tracklets double-covers players: two boxes claim one identity
in the same frame, and TrackEval refuses it. Resolving those collisions costs
9-10% of boxes. So **0.52 is an upper bound a merging stitcher structurally
cannot reach** - it is the ceiling of a different operation than the one we run.

**Two consequences, and they point opposite ways.**

**(a) Given a correct spot, attribution is close to solved by geometry alone** -
89% against a 4.5% floor, with no model. So the plan's framing was wrong:
**spotting is the problem, attribution is not.** §2's claim that "attribution to a
track id is a question this repo has never measured" is retired.

**(b) That 89% is an ORACLE-BALL number.** It uses `ball/*.npz`, which the model
does not get: our own ball is 15.4% correct at ship confidence and the detector
never proposes it on 79.8% of frames. So 89% is the **ceiling for the geometric
route**, not a baseline we already have. The honest ball-free floors - random
1-of-22 (4.5%), random 1-of-11 given team (9.1%), possession-chain persistence,
and PathCRF's ball-free path - still need measuring, and one of those is what a
ball-free model must actually beat.

**(c) The class pattern is independent support for the P3 scoping.** The classes
that collapse - TACKLE 42%, BLOCK 50%, OUT 8%, HEADER 0/1 - are exactly the
contested-contact set predicted to need pixels, and they collapse even with a
perfect ball. Geometry does not fail at them for want of resolution; it fails
because the information is not in the ground plane.

---

## STC2025: the first comparable number this stack has ever had

`model/validation/st2_mot.py`, 2026-09-08. Official challenge split (128057 + 132831),
4-minute `mot/clips`, TrackEval HOTA, mean of the two sequences. Our shipped
CoreML detector + C-BIoU tracker on the raw panorama, **no remap, no calibration,
conf 0.5** - i.e. no dataset-specific help.

```
       seq    HOTA    DetA    AssA    LocA    IDF1   IDSW
    128057   0.369   0.661   0.206   0.872   0.342    280
    132831   0.369   0.504   0.271   0.859   0.350    260
      MEAN   0.369   0.576   0.237   0.866   0.346    540

published   ByteTrack 0.42 | Deep-EIoU 0.54 | GTATrack (winner) 0.60
winner      DetA 0.76  AssA 0.47  LocA 0.84  IDSW 331.5

*IDSW was previously printed as 540, which is the SUM of the two sequences while
 every other cell in that row is a mean - and the winner's 331.5 is a mean too
 (the .5 gives it away). The like-for-like figure is **270, which BEATS the
 winner's 331.5**. A favourable number was being reported as a bad one.
```

**We sit 0.05 below the ByteTrack baseline.** Consistency check on the two
training sequences: 117093 0.338, 128058 0.348 - the same picture, and AssA is
pinned at 0.196-0.271 across all four.

**The decomposition is the finding, and it is unambiguous.** HOTA = sqrt(DetA x AssA):

```
today                            sqrt(0.576 x 0.237) = 0.369
fix detection to the winner's    sqrt(0.760 x 0.237) = 0.424   +0.055
fix association to the winner's  sqrt(0.576 x 0.470) = 0.520   +0.151
```

**Association is worth ~3x what detection is worth.** And `LocA 0.866` BEATS the
winner's 0.84 - when we find a player the box is well placed. So the stack is not
broadly broken: localisation is good, detection is mediocre, association is the
constraint.

Three independent routes now agree: IDF1 0.155 in the ledger, AssA 0.237 here, and
the challenge winner's own ablation ladder (0.42 -> 0.54 -> 0.60) which moved AssA
0.26 -> 0.38 -> 0.47 while DetA barely moved (0.74 -> 0.76).

**Open, and cheap to test: the remap may be hurting.** IDF1 is 0.346 here with NO
remap against 0.155 on the remapped full panorama in the ledger above. Different
sequence and different pipeline, so not attributable yet - but it is one A/B on one
clip, and if it holds it is a large free win.

---

## The association ceiling is not the link cost — measured 2026-09-08

A day spent on `stitch`'s cost function, ending in a negative result worth more
than the +0.039 HOTA it also produced.

**What moved.** baseline 0.369 -> stitch on a height-derived scale 0.376 ->
velocity coasting 0.389 -> conf 0.3 0.394 -> `on_pitch` mask 0.408.

**What did not.** AssA: 0.237 -> 0.266. Against an oracle of **0.523** (merging
our own tracklets by their true identity). Everything after coasting was DetA.

**Four candidate explanations, all tested, all refuted:**

| hypothesis | test | result |
|---|---|---|
| the cost ranks joins badly | label all 2,948 candidate joins with GT, score every feature | coast 0.692, team 0.681, dist 0.678, gap 0.637, box height 0.573, **direction 0.483**. Team-gated coast = 0.779; a grid over everything else = 0.781. Geometry exhausted. |
| appearance would separate what geometry cannot | OSNet/SportsMOT on GT crops, height-matched, >=5 s apart | **AUC 0.799 vs a 0.578 control** - it works. As a link cost 0.761 vs 0.724. But **end to end +0.004 HOTA**. |
| the window is too tight | sweep `max_gap_s` 10/20/30/40/60/120/240, with and without appearance | 10 s is the peak. Longer merges more and merges wrong: AssA 0.241 -> 0.219 at 60 s. |
| true joins are blocked by temporal overlap | count same-player tracklet pairs that overlap | only 9-14%, and agglomeration is transitive, so a spanning route exists anyway |

**So a better link cost does not fix association here, geometric or learned.**
The remaining suspect is the search, not the score: greedy cheapest-first
agglomeration over a 7%-base-rate candidate set at AUC 0.78 makes wrong merges
early, and every wrong merge builds a component that attracts more of them. That
is a known failure mode of greedy agglomerative clustering and it wants a global
or cardinality-constrained formulation, not a better pairwise number.

**Two things not to repeat.** OC-SORT's OCM (direction consistency) is worth
+3.6 HOTA on the benchmark it was published on and **0.483 AUC - nothing - here**;
it was measured before it was built. And "appearance is dead on this footage" is
true of generic encoders and false of sports-trained ReID - see
`appearance-reid-is-dead-on-this-footage` (internal note), now qualified.

---

## Held out: does any of this generalise, or was it fitted to the test pair?

Every configuration decision on 2026-09-08 was scored on **128057 + 132831**, the
benchmark test pair - `max_gap_s`, COAST on/off, the mask margin, conf, the
appearance weight, the gap/slack relaxation. Roughly **24 scored configurations,
best kept**. That is threshold-tuning on test, the exact failure §3 of the sprint
plan warns about, and the published cautionary tale for this task is a class going
0.256 test -> 0.056 challenge from precisely it.

So the frozen configuration was re-scored on **118576, 118578 and 132877** - three
matches that no decision touched. Never scored, never swept against, never looked
at. No further tuning.

```
                         HOTA    DetA    AssA    LocA    IDF1
HELD OUT  (118576, 118578, 132877)
  raw                   0.353   0.541   0.232   0.850   0.331
  frozen config         0.373   0.540   0.258   0.850   0.408
  delta                +0.020          +0.026          +0.077

TEST PAIR (128057, 132831 - tuned against ~24 times)
  raw                   0.374   0.590   0.237   0.862   0.360
  frozen config         0.394   0.590   0.263   0.862   0.408
  delta                +0.020          +0.026          +0.048
```

**The delta is identical to three decimals**, and IDF1 gains MORE on the held-out
matches. The changes generalise.

Why the tuning did not inflate it: most of the 24 sweeps landed back on the
shipped defaults, and the two changes that survived - stitching on a
height-derived scale, and velocity coasting in `_link` - are mechanisms rather
than fitted constants. The one genuinely fitted constant, the `on_pitch` margin,
is exactly the one that turned out to be a coordinate-frame bug and is withdrawn.

Absolute HOTA is lower on held-out (0.373 vs 0.394), but so is the raw baseline
(0.353 vs 0.374). That is match difficulty, not overfitting.

**What this does NOT test.** All ten matches are the same University of Tsukuba
panoramic rig. This shows the changes are not fitted to two clips; it says nothing
about a different camera. `HeightScale` in particular assumes a touchline camera
at the halfway line seeing 105 m across and 68 m deep, and will silently produce a
wrong metre scale on a corner or elevated mount. The only cure for that is a
second camera family with ground truth.
