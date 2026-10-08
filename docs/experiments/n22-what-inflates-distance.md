# N22: what inflates distance and sprints at what ships

> **Corrected 2026-09-13 after independent review.** As first published, this page said
> identity errors drive about half of the distance over-read and pointed the next project at
> association. They do not make anyone run further. The identity oracle leaves our metres
> where they were and changes how much ground truth the scorer credits as covered — see §3.
> The rule's verdict is reported as computed, and it measures the scorer. Separately, the
> identity arm split fragments less often than its protocol states; that was fixed in
> `aeed20a`, and every number below is the re-run.

**2026-09-13, branch `full-pipeline`.** Protocol in the header of
[`model/validation/dist_eval.py`](../../model/validation/dist_eval.py), committed in `a6fbcb2`
before the first run, with a dated review note under it. Results in
`model/validation/out/stc2025/n22_results.json`, gitignored.

`n21-product-now.md` found the product as it ships reading
more distance per covered metre than ground truth, and counting sprints far above
it. N22 replaces one error at a time with the truth on N21's `.p21` tracks, with
one owner map computed on the shipped rows and held fixed in every arm: identity
(each fragment split wherever its true owner changes), position (each detection
moved to its owner's true position), and both. All three are compared with a
drop arm that removes the detections no ground-truth box owns, because no truth
exists to put in their place.

## 1. The five arms

    @M 1.3451,4.4612,3.6436,1.3443,4.5751,3.6776,1.1714,3.6818,3.2591,1.2392,1.9828,1.6549,1.0554,1.1023,1.2439,0.5022,0.3053,IDENTITY,0.2499,0.7251,POSITION,0.1563,0.7554 :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n22_results.json'));a,v=r['arms'],r['verdict'];M=('pooled_ratio_matched','pooled_sprint_ratio','pooled_hsr_ratio');print(*[(k,*[a[k][m] for m in M]) for k in a]);print(*[(m,v[m]['identity_share'],v[m]['position_share'],v[m]['driver']) for m in M])"
       held fixed: N21's nine clips and .p21 tracks, stride-2 frames, N6's calibration,
       the shipped settle, one owner map for every arm; n=5 configurations, nothing tuned.

    @M 88141.4,65565.7,88733.2,75752.3,81249.0,79947.5 :: .venv/bin/python -c "import json;a=json.load(open('model/validation/out/stc2025/n22_results.json'))['arms'];print(*[(k,round(a[k]['pooled_ours_m'],1),round(a[k]['pooled_gt_matched_m'],1)) for k in ('drop','identity','position','both')])"

    @C 1.1714,1.2392,1.0554,IDENTITY,POSITION :: model/validation/out/stc2025/n22_results.json,
       written by `.venv/bin/python model/validation/dist_eval.py --n22` **THE PRODUCER.**
       Cited, not measured, and the reason is RE-DONE WORK, not the clock: the producer's
       result is this file, the file is on disk, and re-running the producer on every push
       re-derives what it already wrote. That is the defect; the durations below are
       supporting detail, not the justification.
       @REC 117.4 :: model/validation/claims.log, the quiet-box serial run of 2026-09-20 -
          seconds of wall, one claim, INSIDE the gate's 600 s cap. It is not structurally
          uncheckable; it passes alone and is at risk only under contention.
       Re-run when the protocol changes, by hand. The artefact carries `protocol`.
       THE NUMBERS ARE STILL CHECKED: the measured tag below re-reads them out of that artefact on every run,
       in under a second. If the file goes stale or is rebuilt to different values, that tag goes red.
    @M 1.1714,1.2392,1.0554,IDENTITY,POSITION :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n22_results.json'));a,v=r['arms'],r['verdict'];print(a['identity']['pooled_ratio_matched'],a['position']['pooled_ratio_matched'],a['both']['pooled_ratio_matched'],v['pooled_ratio_matched']['driver'],v['pooled_sprint_ratio']['driver'])"
       identity / position / both distance-per-covered-metre, then the driver of the
       matched ratio and of the sprint ratio,
       read straight out of the artefact above - no re-computation, and nothing dropped.

| arm | distance per covered metre | our metres | covered GT metres | sprint ratio | HSR ratio |
|---|---|---|---|---|---|
| shipped | 1.3451 | | | 4.4612 | 3.6436 |
| drop — the baseline | 1.3443 | 88,141.4 | 65,565.7 | 4.5751 | 3.6776 |
| identity oracle | 1.1714 | 88,733.2 | 75,752.3 | 3.6818 | 3.2591 |
| position oracle | 1.2392 | 81,249.0 | 65,565.7 | 1.9828 | 1.6549 |
| both | 1.0554 | 79,947.5 | 75,752.3 | 1.1023 | 1.2439 |

Every oracle arm drops this share of the sampled boxes — no ground-truth box owns them:

    @M 0.0922 :: .venv/bin/python -c "import json;a=json.load(open('model/validation/out/stc2025/n22_results.json'))['arms'];print(sorted({v['pooled_unattributed_frac'] for v in a.values()}))"

## 2. What the rule returns

The rule names a driver when one oracle removes at least half of the drop arm's
excess over 1.0 and the other does not.

- **Distance per covered metre: IDENTITY, by 0.0022.**
  `@D 0.5022 :: (1.3443 - 1.1714) / (1.3443 - 1)`
  against the position oracle's
  `@D 0.3053 :: (1.3443 - 1.2392) / (1.3443 - 1)`
- **Sprints and HSR: POSITION**, 0.7251 and 0.7554, with identity at 0.2499 and 0.1563.

## 3. What the identity verdict measures

**Not path length.** Against the drop arm, the identity oracle moves our own
metres up, not down:
`@D +0.67% :: 88733.2 / 88141.4 - 1`
and moves the ground truth the scorer credits as covered up by far more:
`@D +15.54% :: 75752.3 / 65565.7 - 1`
The whole identity share comes from that denominator. The scorer credits each
fragment whole to its majority owner and counts that fragment's frames as
covered for that owner alone, so the frames a fragment spends on somebody else
are covered for nobody. Splitting by true owner hands those frames back, and the
ratio falls because more ground truth is counted — not because less distance was
run.

Two things follow:

- **"Distance per covered metre" is partly a property of the scorer.** N10's
  `ratio_matched` — 1.3451 at what ships in N21 — inherits the same crediting, so
  it is not a clean physical over-read. N10's `ratio_raw`, our metres against all
  ground-truth metres, does not use the crediting.
- **Identity errors still matter, for a different reason.** They put one
  player's distance in another player's row. They do not lengthen the total.

**Position does change path length.** The position oracle shortens our metres by
`@D -7.82% :: 81249.0 / 88141.4 - 1`
and with crediting corrected — the identity arm as baseline — position removes
`@D 0.6768 :: (1.1714 - 1.0554) / (1.1714 - 1)`
of what remains. That baseline is not the one the protocol registered, so it is a
finding here, not a verdict.

## 4. How far to trust the verdicts

- **The distance verdict is fragile as well as mislabelled.** Review found that
  leaving out one clip flipped it to NEITHER ALONE on the first run; after the fix
  the pooled share is closer to the bar still. Do not act on it.
- **What the "both" arm leaves** — distance 1.0554, sprints 1.1023, HSR 1.2439 —
  is what neither oracle touches. Duplicate ids on one player and the filter's
  wake at fragment starts are the suspects; neither is tested here. N24's review
  adds the scorer itself as a suspect: this arm is ground truth on our own frames,
  and ground truth is not credited at frames our steps jump (`n24-crediting.md` §3).
- **The drop arm's own effect**: removing unowned detections moves pooled sprints
  from 4.4612 to 4.5751. This page first said every dropped detection is a gap the
  filter must recover from; that was not tested, and review found clips where
  dropping lowers the count. Why pooled sprints rise is not established.

## 5. What it changes

- **No project follows from the distance verdict.** It measures crediting.
- **Position is what lengthens our paths and inflates sprints and HSR.**
  [`n23-detector-or-map.md`](n23-detector-or-map.md) splits it: not the foot point
  of a matched detection; the map for HSR; unresolved for sprints.
- **Association work** fixes whose row a distance lands in, not how much distance
  there is.
