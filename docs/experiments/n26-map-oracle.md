# N26: the map's own oracle — a correction through the operator's 21 clicks removes the HSR over-read

> **Corrected 2026-09-14 after independent review of the result.** The verdict stands. Three
> things this page first said do not: that the correction removes all of the excess — part of
> that is movement scale, to which HSR is highly sensitive (§4); that the truth arm carries its
> own excess from the feed's grid — the grid deflates the other arms instead (§4); and nothing
> about 132877, whose ground-truth feed turns out to be mirrored (§4). The measurements are
> unchanged.

**2026-09-14, branch `full-pipeline`.** Protocol in the header of
[`model/validation/dist_eval.py`](../../model/validation/dist_eval.py), committed in `5e65e06` and
**revised before its first run** in `381400a` after an independent review: the first
version's round-trip check would have voided a clip on grid arithmetic, its reference
inverse added frame-to-frame noise no other arm carried, and its hull switch made a
corrected player jump metres in one step. No arm was scored before the revision.
Results in `model/validation/out/stc2025/n26_results.json` and `n26.log`, gitignored.

[`n23-detector-or-map.md`](n23-detector-or-map.md) put the HSR over-read on the step
from image to metres, and `n25-speed-estimator.md` found the
speed estimator does not hold it. `n12-calibration.md` had
already found the clicked map 8.27 m off where players stand, because one homography
cannot describe a stitched panorama. The research answer is a projective map plus a
smooth correction (`research-farfield-engineering.md`
§8.2, research-competitors-vendors (not published) §8.2). N26
ran our detections through three maps — what ships, the same map plus a thin-plate
spline through its residuals at the 21 clicks it was solved from, and ST2's own chain
inverted — against true positions on the same frames, with a control arm that pushes
the truth through ST2's chain and back.

## 1. The result

    @M 0.0143,0.0005,True,True,True,0.9901,1.1156,1.0049,loo_disagree 0,MAP + CLICKS CARRY IT :: .venv/bin/python -c "import json;v=json.load(open('model/validation/out/stc2025/n26_results.json'))['verdict'];print(v['control'],v['refmap_dropped_max'],v['ruler_held'],v['valid'],v['tps21_valid'],v['tps21_inside'],v['share_hsr']['refmap'],v['share_hsr']['tps21'],'loo_disagree',sum(x!=v['follows'] for x in v['leave_one_out'].values()),v['follows'])"
       held fixed: N22 and N23's nine clips and .p21 tracks, stride-2 frames, settle 0, one
       owner map, fragment crediting, the shipped Kalman, N6's calibration and its 21 perfect
       clicks; n=3 maps and one control arm, nothing tuned.

    @C 1.0049,MAP + CLICKS CARRY IT :: model/validation/out/stc2025/n26_results.json,
       written by `.venv/bin/python model/validation/dist_eval.py --n26` **THE PRODUCER.**
       Cited, not measured, and the reason is RE-DONE WORK, not the clock: the producer's
       result is this file, the file is on disk, and re-running the producer on every push
       re-derives what it already wrote. That is the defect; the durations below are
       supporting detail, not the justification.
       @REC 156.5 :: model/validation/claims.log, the quiet-box serial run of 2026-09-20 -
          seconds of wall, one claim, INSIDE the gate's 600 s cap. It is not structurally
          uncheckable; it passes alone and is at risk only under contention.
       Re-run when the protocol changes, by hand. The artefact carries `protocol`.
       THE NUMBERS ARE STILL CHECKED: the measured tag below re-reads them out of that artefact on every run,
       in under a second. If the file goes stale or is rebuilt to different values, that tag goes red.
    @M 1.0049,MAP + CLICKS CARRY IT :: .venv/bin/python -c "import json;v=json.load(open('model/validation/out/stc2025/n26_results.json'))['verdict'];print(v['share_hsr']['tps21'],v['follows'])"
       tps21's share of the HSR excess, and what follows from it,
       read straight out of the artefact above - no re-computation, and nothing dropped.

- **Valid.** The ground truth did not move between arms. The control — true positions
  pushed into pixels through ST2's chain and read back — reads within this share of the
  excess of the truth itself, against an allowed 0.1:
  `@D 0.0143 :: abs(1.6838 - 1.6549) / (3.6776 - 1.6549)`
  ST2's chain dropped at most 0.0005 of a clip's rows beyond what the shipped map drops.
  The correction returned every click exactly and did not fold on any clip.
- **MAP + CLICKS CARRY IT**, and not fragile: left out one at a time, each of the nine
  clips gives the same verdict. ST2's chain removes more than the whole HSR excess of our
  detections over true positions:
  `@D 1.1156 :: (3.6776 - 1.4211) / (3.6776 - 1.6549)`
  and the correction through the 21 clicks removes all of it:
  `@D 1.0049 :: (3.6776 - 1.6449) / (3.6776 - 1.6549)`

## 2. Every arm

    @M 3.6776,4.5751,1.3443,1.6449,2.4077,1.2018,1.4211,2.0601,1.1902,1.6838,2.0258,1.2546,1.6549,1.9828,1.2392 :: .venv/bin/python -c "import json;a=json.load(open('model/validation/out/stc2025/n26_results.json'))['arms'];print(*[a[k][m] for k in ('drop','tps21','refmap','feedref','position') for m in ('pooled_hsr_ratio','pooled_sprint_ratio','pooled_ratio_matched')])"

| arm | HSR ratio | sprint ratio | distance per covered metre |
|---|---|---|---|
| drop — our detections through the shipped map | 3.6776 | 4.5751 | 1.3443 |
| tps21 — the same map, corrected through its 21 clicks | 1.6449 | 2.4077 | 1.2018 |
| refmap — ST2's own chain, inverted smoothly | 1.4211 | 2.0601 | 1.1902 |
| feedref — the control: truth through ST2's chain and back | 1.6838 | 2.0258 | 1.2546 |
| position — true positions | 1.6549 | 1.9828 | 1.2392 |

Beside the rule, not ruled on: the share of the sprint excess the correction removes,
then ST2's chain's; and the same two for distance per covered metre.

    @M 0.8361,0.9702,1.3559,1.4662 :: .venv/bin/python -c "import json;v=json.load(open('model/validation/out/stc2025/n26_results.json'))['verdict']['share_beside'];print(v['sprints']['tps21'],v['sprints']['refmap'],v['distance']['tps21'],v['distance']['refmap'])"

## 3. Per clip

    @M 7.9013,3.661,3.1595,2.2041,2.1353,3.4814,1.5285,1.2425,1.5392,1.514,3.7183,1.4571,1.4207,1.6323,1.6052,3.1059,2.6604,2.3285,2.4467,2.4123,1.9347,1.0636,1.026,1.3831,1.3527,2.3153,1.1311,0.972,1.4374,1.424,4.2131,1.5307,1.2077,2.0087,1.9765,3.6289,1.4723,1.1287,1.614,1.5907,4.5793,1.6112,1.5667,1.5503,1.5272 :: .venv/bin/python -c "import json;p=json.load(open('model/validation/out/stc2025/n26_results.json'))['per_clip_hsr_ratio'];print(*[p[a][c] for c in sorted(p['drop']) for a in ('drop','tps21','refmap','feedref','position')])"

HSR ratio per clip:

| clip | drop | tps21 | refmap | feedref | position |
|---|---|---|---|---|---|
| 117092 | 7.9013 | 3.661 | 3.1595 | 2.2041 | 2.1353 |
| 117093 | 3.4814 | 1.5285 | 1.2425 | 1.5392 | 1.514 |
| 118575 | 3.7183 | 1.4571 | 1.4207 | 1.6323 | 1.6052 |
| 118576 | 3.1059 | 2.6604 | 2.3285 | 2.4467 | 2.4123 |
| 118577 | 1.9347 | 1.0636 | 1.026 | 1.3831 | 1.3527 |
| 118578 | 2.3153 | 1.1311 | 0.972 | 1.4374 | 1.424 |
| 128057 | 4.2131 | 1.5307 | 1.2077 | 2.0087 | 1.9765 |
| 128058 | 3.6289 | 1.4723 | 1.1287 | 1.614 | 1.5907 |
| 132877 | 4.5793 | 1.6112 | 1.5667 | 1.5503 | 1.5272 |

## 4. What it shows, and how far to trust it

- **The speed over-read is the clicked map.** Same detections, same tracker, same filter:
  pooled over the nine clips, a map that bends where one homography cannot takes HSR
  down to the level of true positions on the same frames. Clip by clip, the correction
  through the 21 clicks removes this share of each clip's excess — at least 0.642
  everywhere, 0.97 or more on seven of nine, and all of it or more on five; 117092 and
  118576 keep a quarter to over a third of theirs:

      @M 0.735,0.993,1.07,0.642,1.497,1.329,1.199,1.058,0.972,tps21_le_position 5 :: .venv/bin/python -c "import json;p=json.load(open('model/validation/out/stc2025/n26_results.json'))['per_clip_hsr_ratio'];C=sorted(p['drop']);sh=lambda a,c:round((p['drop'][c]-p[a][c])/(p['drop'][c]-p['position'][c]),3);print(*[sh('tps21',c) for c in C],'tps21_le_position',sum(p['tps21'][c]<=p['position'][c] for c in C))"
         per clip in the order of §3, then the number of clips where the corrected map reads
         at or below true positions.

  It takes most of the sprint excess and not all of it.
- **Corrected after review: those shares are read at each map's own movement scale, and HSR
  is several times as sensitive to that scale as distance is.** Along long stretches of one
  player's run, the shipped map lengthens movement against the feed on every clip, while the
  corrected map and ST2's chain read it slightly short of the feed on most clips — the
  corrected map inherits the chain's scale, because its perfect clicks are the chain's own
  pixels. Matched to the feed's movement scale, the correction still removes most of the
  pooled excess and still clears the rule, ST2's chain removes all of it, and per clip the
  correction removes less than all of the excess on every clip but one; 118576 keeps the
  most, with the movement scale right, which is a limit of what 21 clicks can bend. Review
  checks on the same tracks, not registered; N27 registers a movement-scale check.
- **Corrected after review: the better maps read below "true positions" for a different
  reason than this page first gave.** It said the truth arm carries its own excess from the
  feed's 1.05 m × 0.68 m grid. Smoothing the feed leaves the truth arm where it was on the two
  clips checked. The grid inflates the ground-truth HSR every ratio divides by, which deflates
  the arms built from detections and not the truth arm; that, and the maps reading movement
  short, is what puts them below true positions. Review checks, not registered.
- **A data fault the review found: on 132877 the joined ground-truth feed is mirrored top to
  bottom** against ST2's own boxes and chain, and matches every player once mirrored back.
  Speeds and distances do not change under a mirror, and repeating that clip with the mirror
  undone barely moves its HSR share; its sprint counts do move, and any position-at-feet use
  of its feed is wrong. The join passes its own gate. Recorded, not fixed here.
- **The control arm cannot see a movement-scale gap between box feet and the feed.** The
  truth pushed through ST2's chain reads back at the feed's scale by construction, while
  boxes on the same players read slightly short through the same chain.
- **Position at players improves with the map**, measured by the review at our detections on
  several clips: the shipped map sits several metres from the truth, the corrected map much
  closer, ST2's chain closer still. Not N12's metric and not registered.
- **It is a ceiling, not an operator.** The 21 clicks are perfect: each landmark's pixel
  from ST2's own chain. A spline through every click passes through a real operator's
  errors too, and N12's finding that click care does not matter was about the single
  homography, not this.
- **Position at a player's feet is not measured here**, only speed. An earlier thin-plate
  spline in this repo improved landmarks and made player positions worse
  (`tps-correction-is-a-tail-fix` (internal note)) — on the vendor's remap chain, with
  a smoothing chosen on landmarks. Different chain and different settings, so neither
  result transfers to the other.
- **Reported, not ruled:** the corrected map stretches the pitch locally by up to 2.42
  (p90 at most 1.91 on any clip), and this share of rows falls outside the clicks' hull,
  where the spline extrapolates:
  `@D 0.0099 :: 1 - 0.9901`

      @M 0.287,1.91,2.42,False,True :: .venv/bin/python -c "import json;c=list(json.load(open('model/validation/out/stc2025/n26_results.json'))['tps21_checks'].values());print(round(min(x['min_det'] for x in c),3),round(max(x['stretch_p90'] for x in c),2),round(max(x['stretch_max'] for x in c),2),any(x['folds'] for x in c),max(x['exact_m'] for x in c)<1e-6)"
         the smallest Jacobian determinant; the largest p90 and max stretch; folds on any clip;
         every click returned to 1e-6 m.

- **Scope:** nine ST2 clips from one kind of rig — a fixed stitched panorama — on the
  `.p21` tracks.

## 5. What it changes

- **The correction goes into the product next**, as the rule says and as its own change:
  scored at player feet on N12's metric, with noisy clicks, and on speed at what ships
  (`.p21id`, through the product path), before anyone reads it as a shipped number. That
  protocol has to check movement scale against the truth before it reads any speed gain,
  score against ground truth with its grid staircase removed, and leave 132877 out until
  its join is fixed.
- **It is the first measured lever on the shipped HSR over-read.** Nothing about the
  detector, the tracker or the estimator changed.
- **N23's "map" for HSR is now a direct oracle result**, and N25's "not the estimator"
  is consistent with it.
- **Sprints are not done:** the correction leaves part of the sprint excess, and N25
  found the sprint count moves with the estimator on ground truth itself.
