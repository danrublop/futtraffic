# N40: the association cost is not the lever — the detector under occlusion is

**2026-09-17, branch `n30-operator-clicks`.** Protocols: [`model/validation/n40_association.py`](../../model/validation/n40_association.py)
(committed at `dd38069` before any score) and [`model/validation/n40_occlusion_recall.py`](../../model/validation/n40_occlusion_recall.py).
Track B step 1 of `plan-parallel-tracks.md`. Follows `model/engine/stitch.py`'s own docstring and
`association-not-detection-clears-the-leaderboard (internal note)`.
**Development, seven same-install clips; the link test on three of them (DEV); nothing reached the held-out four.**

## 1. Two physically-motivated link costs, measured before anything was built

The shipped stitcher joins fragments cheapest-first on a pitch-plane miss distance after coasting each end along its own
velocity. Two costs followed from its own diagnosis (the nearest fragment to an exit is usually the wrong one, because
tracks end under another player; pitch-plane distance amplifies far-side pixel noise ten times): V1 divides the miss by
the noise it should have if the join were right (box-bottom noise through the camera's local metres-per-pixel, plus an
acceleration term); V2 lets an exit ride the player who occluded it to the entry's frame. Scored first at the LINK level:
every feasible exit-entry pair on the three DEV matches whose two tracklets carry a known ground-truth identity, AUC as a
same-player ranker.

| cost | AUC |
|---|---|
| V0 shipped miss distance | 0.8278 |
| V1 noise-normalised, with the acceleration term | 0.5059 |
| V2 occluder-carried coast | 0.8272 |
| V3 both | 0.5074 |
| post-hoc, declared: V1 without the acceleration term (5th cost) | 0.8167 |

    @M 0.8278,0.5059,0.8272,0.5074,2513,300 :: .venv/bin/python model/validation/n40_association.py --show

2,513 pairs, 300 same-player, n = 5 costs. The review (2026-09-18) noted the pairs include 792 opposite-team pairs that
`stitch()`'s team gate would never consider; on the team-compatible subset the AUCs are V0 0.8334, V1 0.4971, V2 0.8318,
V3 0.4983 - the same verdict. It also found that 38 of the 300 true same-player pairs (12.7%) carry OPPOSITE team
labels, i.e. the team label blocks one true join in eight: a larger association lever than either cost, and the one
thing on this page worth a follow-up. **Neither idea goes to the end-to-end level or to the held-out matches**;
by the protocol's own rule the shipped cost stays. Dividing the miss by anything that grows with the gap is what
`stitch.py` already found for implied speed, only worse; the occluder's path adds nothing the exit's own coast did not
have. Geometry is exhausted for this decision, as the stitcher's docstring said.

## 2. Where the fragments actually come from

`model/validation/n40_occlusion_recall.py`: the product's own detections (coreml, conf 0.5, stride 2, the cached frames of the
seven clips) against the ground-truth boxes. A GT box is OCCLUDED when another GT box covers at least 35% of it,
ISOLATED when nothing touches it; depth thirds by the foot's image row. `lapse` is the share of a player's frames in that
class where the raw track (no stitching) changes id or has no detection.

| class | GT boxes | detector recall (IoU >= 0.5) | raw-track id lapse per frame |
|---|---|---|---|
| occluded, far third | 14,408 | 0.493 | 0.633 |
| occluded, middle third | 23,169 | 0.614 | 0.533 |
| occluded, near third | 14,405 | 0.685 | 0.462 |
| isolated, far third | 108,336 | 0.878 | 0.242 |
| isolated, middle third | 100,019 | 0.972 | 0.068 |
| isolated, near third | 112,368 | 0.970 | 0.047 |

    @M 14408,0.4928,0.6332,23169,0.6140,0.5332,14405,0.6846,0.4617,108336,0.8775,0.2416,100019,0.9716,0.0675,112368,0.9703,0.0471 :: .venv/bin/python model/validation/n40_occlusion_recall.py

- **Half of occluded players are not detected at all** (0.49-0.68 against 0.88-0.97 isolated), and where there is no
  box there is nothing to associate. Occluded frames are 14% of player-frames and they are where tracks end: the id
  lapses on 46-63% of them against 5-7% of isolated frames in the middle and near thirds.
- The far third is weaker even isolated (0.878, and a 24% lapse rate): small players, 20-40 px tall.
- The unverified audit figure in an internal note (0.307 vs 0.904) was the right shape and the wrong numbers; these are
  the measured ones, on seven clips, with the command above.
- **They are detected weakly, not missed** (one detector pass on 118575 cached down to conf 0.05, `--conf`):

| conf floor | occluded far | occluded mid | occluded near | isolated far | isolated mid | isolated near | non-GT boxes per frame |
|---|---|---|---|---|---|---|---|
| 0.50 (ships) | 0.308 | 0.711 | 0.740 | 0.640 | 0.989 | 0.964 | 2.21 |
| 0.40 | 0.464 | 0.783 | 0.787 | 0.773 | 0.992 | 0.974 | 2.54 |
| 0.30 | 0.580 | 0.800 | 0.804 | 0.878 | 0.993 | 0.980 | 3.62 |
| 0.20 | 0.664 | 0.805 | 0.811 | 0.917 | 0.993 | 0.984 | 7.08 |
| 0.10 | 0.705 | 0.806 | 0.822 | 0.929 | 0.993 | 0.988 | 23.68 |

    @M 0.308,0.711,0.740,0.640,0.989,0.964,2.21,0.464,0.580,0.800,0.878,3.62,0.664,0.917,7.08,0.705,23.68 :: .venv/bin/python model/validation/n40_occlusion_recall.py --conf

  The ground truth holds the 22 players and no officials, so about 2 of the non-GT boxes per frame at 0.5 are the
  referees; 0.3 adds about 1.4 boxes per frame of real false positives or duplicates and nearly doubles far-side
  occluded recall (0.31 to 0.58) and lifts isolated far recall from 0.64 to 0.88. Below 0.3 the boxes bought per
  false positive collapse. This is 118575 only; the seven-clip pooled row at 0.5 is the table above. Whether 0.3
  is a better PRODUCT setting is a separate, pre-registered question (fragments, HOTA with both DetRe and DetPr,
  and distance/HSR - false boxes that form tracks inflate distance), not a threshold to move on a recall table.

## 3. What this changes in the plan

Track B step 1 (association cost) is closed as NOT SUPPORTED at the link level. The fragments are, first, a detection
problem: the next work is detection under occlusion and on the far third, measured by exactly these two recalls, before
any further association work. Candidate moves, in order of cost: a lower confidence floor if recall at 0.3-0.1 shows the
boxes exist; inference at a larger input size or on crowd crops; a detector fine-tuned on occluded ST2 crops, which the
dataset's full-box ground truth makes possible without hand-labelling.
