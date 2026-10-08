# N23: which part of "position" inflates speed — the detector or the map

> **Corrected 2026-09-13 after independent review.** Three things this page first said were
> wrong or unscoped: the detector result was stated beyond the population it covers (§3),
> §4's explanation for the sprint disagreement is refuted, and the distance verdict rests on
> one reading of the guard (§2). The measurements are unchanged; what they support is below.

**2026-09-13, branch `full-pipeline`.** Protocol in the header of
[`model/validation/dist_eval.py`](../../model/validation/dist_eval.py), committed in `74721c3`
before the first run. Results in `model/validation/out/stc2025/n23_results.json` and
`n23.log`, gitignored.

[`n22-what-inflates-distance.md`](n22-what-inflates-distance.md) put about three
quarters of the sprint and HSR over-count on position errors and could not say
whose: its position oracle replaced the detector's foot points and the clicked
map's geometry at once. N23 replaces them one at a time on N21's tracks. The
steps telescope — our detections, then MOT ground-truth feet through our map,
then ST2's position feed through its own vendor chain and our map, then true
metres — so the three shares sum to N22's position share exactly.

**Disclosed in the pre-registration commit.** Before committing, a mechanical
check on 117093 looked at inputs and scored no arm: every owned detection had a
MOT foot and a valid vendor pixel, our feet sat a median 1.53 px from the MOT
feet, and the MOT feet sat a median 17.33 px from the vendor-projected feed. The
rule was written before that check and not changed after it.

## 1. The four arms

    @M 4.5751,3.6776,1.3443,4.6438,3.6378,1.3267,3.5322,3.74,1.39,1.9828,1.6549,1.2392,-0.0192,0.3109,0.4334,INCONCLUSIVE,0.0149,-0.0382,0.7787,MAP,0.0511,-0.1839,0.438 :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n23_results.json'));a,v=r['arms'],r['verdict'];M=('pooled_sprint_ratio','pooled_hsr_ratio','pooled_ratio_matched');print(*[(k,*[a[k][m] for m in M]) for k in a]);print(*[(m,v[m]['detector_share'],v[m]['annotation_share'],v[m]['map_share'],v[m]['driver']) for m in M])"
       held fixed: N21's nine clips and .p21 tracks, stride-2 frames, N6's calibration,
       the shipped settle, one owner map for every arm; n=4 configurations, nothing tuned.

    @C 4.6438,3.5322,1.9828,INCONCLUSIVE,MAP :: model/validation/out/stc2025/n23_results.json,
       written by `.venv/bin/python model/validation/dist_eval.py --n23` **THE PRODUCER.**
       Cited, not measured, and the reason is RE-DONE WORK, not the clock: the producer's
       result is this file, the file is on disk, and re-running the producer on every push
       re-derives what it already wrote. That is the defect; the durations below are
       supporting detail, not the justification.
       @REC 94.9 :: model/validation/claims.log, the quiet-box serial run of 2026-09-20 -
          seconds of wall, one claim, INSIDE the gate's 600 s cap. It is not structurally
          uncheckable; it passes alone and is at risk only under contention.
       Re-run when the protocol changes, by hand. The artefact carries `protocol`.
       THE NUMBERS ARE STILL CHECKED: the measured tag below re-reads them out of that artefact on every run,
       in under a second. If the file goes stale or is rebuilt to different values, that tag goes red.
    @M 4.6438,3.5322,1.9828,INCONCLUSIVE,MAP :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n23_results.json'));a,v=r['arms'],r['verdict'];print(a['gtpix']['pooled_sprint_ratio'],a['feedpix']['pooled_sprint_ratio'],a['position']['pooled_sprint_ratio'],v['pooled_sprint_ratio']['driver'],v['pooled_hsr_ratio']['driver'])"
       MOT-feet / vendor-projected / true-position sprint ratios, then the driver on
       sprints and on HSR,
       read straight out of the artefact above - no re-computation, and nothing dropped.

| arm | sprint ratio | HSR ratio | distance per covered metre |
|---|---|---|---|
| drop — our detections | 4.5751 | 3.6776 | 1.3443 |
| MOT feet through our map | 4.6438 | 3.6378 | 1.3267 |
| vendor-projected feed through our map | 3.5322 | 3.74 | 1.39 |
| true positions | 1.9828 | 1.6549 | 1.2392 |

## 2. The verdict, as the rule reads it

- **Sprints — the decision metric — INCONCLUSIVE.** The annotation share, the
  part that moves when one ground truth is swapped for the other, is too large
  a part of the position share to split the rest:
  `@D 0.3109 :: (4.6438 - 3.5322) / (4.5751 - 1)`
  against a guard of a third of the position share:
  `@D 0.2417 :: 0.7251 / 3`
- **HSR — MAP.** The map carries this share of the HSR over-read:
  `@D 0.7787 :: (3.74 - 1.6549) / (3.6776 - 1)`
  with the two ground truths agreeing to within 0.0382 of it.
- **Distance — INCONCLUSIVE on the committed code, MAP on the protocol's text.** The guard's
  code compares the annotation share's MAGNITUDE, and distance's is negative, -0.1839; the
  protocol text says only "more than a third". Read signed, the guard does not fire and the
  map share, 0.438, clears two thirds of the position share:
  `@D 0.2035 :: 0.3053 * 2 / 3`
  The page reports the code's verdict and both readings; the decision metric, sprints, is
  INCONCLUSIVE either way. N22's distance verdict is itself corrected — see its page.

## 3. What holds, and on what population

**For detections matched to a ground-truth box at IoU >= 0.5, the foot point is not the
position problem.** Swapping those detections' feet for the MOT boxes' changes each metric
by almost nothing: a detector share of -0.0192 on sprints, 0.0149 on HSR,
0.0511 on distance. Review found it a real null rather than one by construction — the swap
does remove detector-only frame-to-frame error (a single-clip check, not registered) — but it
is scoped:

- every arm drops the detections that match no box, this share of the sampled boxes:

      @M 0.0922 :: .venv/bin/python -c "import json;a=json.load(open('model/validation/out/stc2025/n23_results.json'))['arms'];print(sorted({v['pooled_unattributed_frac'] for v in a.values()}))"

- false detections, boxes too badly placed to pass the gate, and players never detected
  are not tested at all;
- for sprints the rule returned INCONCLUSIVE, so the detector share there is an observation
  beside the verdict, not the verdict.

## 4. What is unresolved

The two ground truths — ST2's MOT boxes and ST2's own position feed — disagree by a median
17.33 px on 117093, and swapping one for the other moves the sprint count by a third of its
excess. **This page first explained that as frame-to-frame jitter that per-frame boxes carry
and the feed lacks. Review refuted it:** a single-clip check found the feed — a staircase on a
1% grid — jumpier frame to frame than the MOT boxes, not smoother, and smoothing our own foot
points left the sprint count where it was. Neither check is registered. What the annotation
share reflects is open; the suspicion is how the filter and the sprint counter treat a
quantised reference. Resolving it needs a reference both ground truths agree on.

## 5. What it changes

- Position work for **HSR** goes to the map — the clicked calibration's pixel-to-
  metre geometry — and not to the detector's foot points.
- Position work for **sprints** stays unresolved (§4), and smoothing foot points is unlikely to
  be the lever. For matched detections the foot point is not it either (§3).
- **"Map" means our map against ST2's own projection** — its nearest-pixel table lookup
  included — not against the pitch.
- **Distance** is not N22's identity verdict any more: that verdict measured how the scorer
  credits fragments (see [`n22-what-inflates-distance.md`](n22-what-inflates-distance.md)).
