# N27: the 21-click correction in the product — feet land far closer; DO NOT SHIP, under a speed rule even true positions fail

> **Corrected 2026-09-14 after independent review of the result.** The verdict and every registered
> number stand. The speed reading does not:
> - **Rule C could not be passed by any map.** True positions through the product's own fragments
>   fail it too (§4).
> - **Withdrawn: "for speed, the map is not the lever once movement scale is right."** On the same
>   fragments at matched scale, the correction removes about half of what true positions would (§4).
> - **The restitch is not neutral.** It carries about half of sigma 0's matched gap (§4).
> - **The matched share depends on the scale statistic,** by about ±0.2 in share. C fails under
>   every choice (§5).
> - **"The truth reads box feet a few percent short" is ST2's own chain disagreeing with the feed.**
>   The corrected map, fitted to the chain's pixels, follows the chain (§3, §5).
> - **Smaller:** B misses by more at 5 px of click noise; "a third on every clip" holds only pooled;
>   N26 required removing the feed's grid staircase and N27 did not, without saying so (§2, §3, §5).

**2026-09-14, branch `full-pipeline`.** Protocol in the header of
[`model/validation/dist_eval.py`](../../model/validation/dist_eval.py), committed in `429a0d2` and
**revised before its first run** in `100b7bb` after an independent review. The first version
could not replay a detection cache under another calibration, would have voided itself on folds
at heavy click noise, and read one noisy seed and no leave-one-clip-out. No arm was scored before
the revision. Results in `model/validation/out/stc2025/n27_results.json` and `n27.log`, gitignored.
Re-running needs the `.p27*` outputs removed first: `dist_eval.py --n27` refuses to reuse replays.

[`n26-map-oracle.md`](n26-map-oracle.md) found that a thin-plate spline through the shipped
homography's residuals at its own 21 clicks takes the HSR over-read down. Its review added that
part of that is movement scale. N26's rule sent the correction into the product as its own change,
scored at players' feet, with noisy clicks, and on speed at what ships. N27 is that change:
`TpsCorrected` in [`model/engine/calibrators.py`](../../model/engine/calibrators.py), projection type
`perspective_tps`, loaded by `app/server/jobs.py`. The UI still writes `perspective`.

## 1. The result

    @M 0.1272,0.051,0.122,0.0837,0.0804,0.0624,True,False,False,False :: .venv/bin/python -c "import json;v=json.load(open('model/validation/out/stc2025/n27_results.json'))['verdict'];print(v['scale_deviation']['ship0'],v['scale_deviation']['tps0'],*v['share_matched'].values(),v['valid'],v['A'],v['B'],v['C'])"
       held fixed: N21's nine clips less 132877, stride-2 frames, N21's `.p21` detections replayed
       through track.run and jobs.apply_identity, N6's calibration and its clicks, N12's click noise;
       n=1 correction configuration, nothing tuned. Printed: the median |slope - 1| of the shipped
       and corrected maps, the matched HSR share at sigma 0 and at sigma 5 seeds 0-2, then valid and
       parts A, B, C.

- **Valid.**
  - Every click set solved and every slope is finite.
  - The corrected Jacobian is its own.
  - The ground truth held across arms, every replay carried `.p21`'s detections, and no
    output existed before the run.
- **DO NOT SHIP**, failing all three parts, in very different ways:
  - **A, position,** fails only on the p90 tail at click noise of 10–20 px.
  - **B, movement scale,** misses its bound by a thousandth with perfect clicks, and by more with
    noisy ones (§3).
  - **C, speed,** fails decisively: with movement scale matched to the feed, the correction removes
    `@D 0.122 :: (2.029 - 1.9035) / (2.029 - 1)`
    of the shipped HSR excess over the truth, against a required 0.5, and no seed or
    left-out clip comes near. **Corrected after review:** true positions through the same
    fragments fail this bar too (§4), so C's failure says nothing about the correction.

## 2. A: position at players' feet

Pooled metres from MOT box feet to the joined feed. For sigma 0 and each sigma-5 seed: median,
p90 and far-half median, shipped then corrected:

    @M 7.49,2.41,10.36,7.45,7.42,2.52,7.33,2.87,10.41,9.04,7.22,3.3,7.54,2.74,10.46,8.56,7.52,3.15,7.54,2.83,10.5,8.07,7.55,3.29 :: .venv/bin/python -c "import json;p=json.load(open('model/validation/out/stc2025/n27_results.json'))['position'];print(*[round(p[k][m],2) for k in ('0/0','5/0','5/1','5/2') for m in ('ship','tps','ship_p90','tps_p90','ship_far','tps_far')])"

| click noise | median, shipped | median, corrected | p90, shipped | p90, corrected | far half, shipped | far half, corrected |
|---|---|---|---|---|---|---|
| 0 px | 7.49 | 2.41 | 10.36 | 7.45 | 7.42 | 2.52 |
| 5 px, seed 0 | 7.33 | 2.87 | 10.41 | 9.04 | 7.22 | 3.3 |
| 5 px, seed 1 | 7.54 | 2.74 | 10.46 | 8.56 | 7.52 | 3.15 |
| 5 px, seed 2 | 7.54 | 2.83 | 10.5 | 8.07 | 7.55 | 3.29 |

With perfect clicks the corrected median is this share of the shipped one:
`@D 0.3218 :: 2.41 / 7.49`

Per clip at sigma 0, shipped then corrected median:

    @M 11.18,4.7,7.16,2.59,8.0,2.0,6.78,1.86,7.4,1.92,6.52,3.79,6.85,2.31,7.28,2.55 :: .venv/bin/python -c "import json;p=json.load(open('model/validation/out/stc2025/n27_results.json'))['per_clip_position'];print(*[round(p[c]['0/0'][m],2) for c in sorted(p) for m in ('ship','tps')])"

| clip | shipped | corrected |
|---|---|---|
| 117092 | 11.18 | 4.7 |
| 117093 | 7.16 | 2.59 |
| 118575 | 8.0 | 2.0 |
| 118576 | 6.78 | 1.86 |
| 118577 | 7.4 | 1.92 |
| 118578 | 6.52 | 3.79 |
| 128057 | 6.85 | 2.31 |
| 128058 | 7.28 | 2.55 |

Where A fails: the pooled p90, shipped then corrected, at 10 px then 20 px of noise, seeds 0–2.
Then how many of the 24 click sets at each of those noise levels switched the correction off, and
how many at 5 px or less:

    @M 10.32,10.79,10.75,8.59,10.52,9.23,11.31,12.72,10.46,10.79,10.84,11.07,fallback 2,fallback 2 11,fallback 2 11 0 :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n27_results.json'));p,fb=r['position'],r['fallback'];print(*[round(p[k][m],2) for k in ('10/0','10/1','10/2','20/0','20/1','20/2') for m in ('ship_p90','tps_p90')],'fallback',*[sum(fb[c][k] for c in fb for k in fb[c] if t(k)) for t in (lambda k:k.startswith('10/'),lambda k:k.startswith('20/'),lambda k:float(k.split('/')[0])<=5)])"

- **Every clip is better, at sigma 0 and at every sigma-5 seed.** The pooled median and the far
  half are better at every noise level. No correction switched off at 5 px or less. Pooled, the
  corrected median is about a third of the shipped one. Per clip at sigma 0 it ranges over the
  smallest and largest of these shares, 118578 the largest:

      @M 0.25,0.581 :: .venv/bin/python -c "import json;p=json.load(open('model/validation/out/stc2025/n27_results.json'))['per_clip_position'];r=[p[c]['0/0']['tps']/p[c]['0/0']['ship'] for c in p];print(round(min(r),3),round(max(r),3))"

- **The tail is not.** At 10 px on one seed and at 20 px on every seed, the corrected p90 is above
  the shipped p90. The correction also switches itself off on a growing share of click sets.
  `n12-calibration.md` §3 called 20 px far beyond what an operator would
  miss by.
- **The two clips that keep the most error are where ST2's own chain and the feed disagree most**
  at the same feet, in median metres:
  `@C 4.634, 2.652 :: model/validation/out/stc2025/n27_results.json chain_vs_feed_m`
  The corrected map is fitted to the chain's pixels and follows the chain there (§3).

## 3. B: movement scale

Least-squares slope of 4 s chord displacement through the map against the feed's, on the product's
own tracks at sigma 0, per clip, shipped then corrected:

    @M 1.228,1.059,1.176,0.974,1.18,0.957,1.041,1.002,1.083,0.932,1.111,0.926,1.134,0.932,1.12,0.972 :: .venv/bin/python -c "import json;s=json.load(open('model/validation/out/stc2025/n27_results.json'))['slope'];print(*[round(s[a][c],3) for c in sorted(s['ship0']) for a in ('ship0','tps0')])"

| clip | shipped | corrected |
|---|---|---|
| 117092 | 1.228 | 1.059 |
| 117093 | 1.176 | 0.974 |
| 118575 | 1.18 | 0.957 |
| 118576 | 1.041 | 1.002 |
| 118577 | 1.083 | 0.932 |
| 118578 | 1.111 | 0.926 |
| 128057 | 1.134 | 0.932 |
| 128058 | 1.12 | 0.972 |

The shipped map lengthens movement on every clip. The corrected map sits inside [0.90, 1.10] on every
clip, and its median deviation from 1 is 0.051 against a bound of 0.05. It reads movement short on
six of eight. The protocol set the bound loose because "the truth itself reads box feet a few
percent short".

**Corrected after review.** That is ST2's own chain reading box-feet movement short against the feed.
The corrected map, whose perfect clicks are the chain's own pixels, follows the chain. With 5 px of
click noise the miss is larger, and one clip leaves the band. Median |slope − 1| for the three
sigma-5 seeds, then the clips outside [0.90, 1.10] on any of them:

    @M 0.0634,0.0575,0.0634,128057 :: .venv/bin/python -c "import json,numpy as np;S=json.load(open('model/validation/out/stc2025/n27_results.json'))['slope'];A=('tps5_0','tps5_1','tps5_2');print(*[round(float(np.median([abs(x-1) for x in S[a].values()])),4) for a in A],*sorted({q for a in A for q,x in S[a].items() if not 0.9<=x<=1.1}))"

## 4. C: speed at what ships

Pooled over the eight clips, shipped then corrected at sigma 0: HSR ratio at each map's own scale,
HSR ratio with movement scale matched, sprint ratio at own scale, distance per covered metre:

    @M 3.5995,2.029,4.4133,1.3381,1.6577,1.9035,2.5668,1.2095 :: .venv/bin/python -c "import json;d=json.load(open('model/validation/out/stc2025/n27_results.json'))['detail'];f=lambda a,k,i=0:sum(v[k][i] for v in d[a].values());print(*[round(f(a,k)/f(a,k,1),4) for a in ('ship0','tps0') for k in ('hsr_own','hsr_matched','sprints_own','distance_own')])"

| map | HSR ratio, own scale | HSR ratio, scale matched (ruled) | sprint ratio, own scale | distance per covered metre |
|---|---|---|---|---|
| shipped, `.p21id` | 3.5995 | 2.029 | 4.4133 | 1.3381 |
| corrected, `.p27id` | 1.6577 | 1.9035 | 2.5668 | 1.2095 |

Leaving out each clip in turn at sigma 0, then the two diagnostics beside the rule. The first is
the corrected map on `.p21id` without a restitch; the second is `.p27id` read with the homography's
noise model:

    @M 0.0769,0.1252,0.2073,0.1398,0.1513,0.1166,0.0696,0.0718,1.8424,1.8972 :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n27_results.json'));print(*r['verdict']['leave_one_out'].values(),r['diagnostics']['tps0_no_restitch'],r['diagnostics']['tps0_H_noise'])"

- **At each map's own scale the correction looks like a fix.** HSR falls from about 3.6 times the
  truth to about 1.7 times, removing
  `@D 0.747 :: (3.5995 - 1.6577) / (3.5995 - 1)`
  of the excess; sprints and distance per covered metre fall with it.
- **Almost all of that own-scale drop is movement scale.** Matching scale multiplies the shipped
  ratio by
  `@D 0.5637 :: 2.029 / 3.5995`
  and the corrected ratio by
  `@D 1.1483 :: 1.9035 / 1.6577`
  and leaves both maps about twice the truth. Movement scale is itself a property of the map.

**Corrected after review: the ruled bar was out of reach, and the map matters beyond movement scale.**
True positions put into the same replays' fragments (N22's position oracle), pooled over the eight
clips, on the shipped replays then the corrected ones:

    @M 1.6634,1.7369 :: .venv/bin/python -c "import sys;sys.path[:0]=['model/validation','model/engine'];import dist_eval as D;f=lambda v:[sum(x) for x in zip(*[(r['speed']['our_hsr_m'],r['speed']['gt_matched_hsr_m']) for r in [D.evaluate_stc(q,v,oracle='position') for q in D.N27_SEQS]])];print(*[round(a/b,4) for a,b in map(f,('p21id','p27id'))])"  #slow

- **Rule C could not be passed.** By C's own bar, true positions remove only this share of the
  shipped excess:
  `@D 0.3553 :: (2.029 - 1.6634) / (2.029 - 1)`
  The rest of the excess over 1.0 is what this pipeline's fragments read even with perfect positions.
  C asked a map to remove what no map can.
- **Against that floor, the correction does real work beyond movement scale.** On the same fragments,
  in the no-restitch diagnostic, it removes this share of what true positions would:
  `@D 0.5104 :: (2.029 - 1.8424) / (2.029 - 1.6634)`
  After its own restitch it removes
  `@D 0.3433 :: (2.029 - 1.9035) / (2.029 - 1.6634)`
- **The restitch is not neutral.** It carries this share of sigma 0's matched gap between the maps:
  `@D 0.487 :: (1.9035 - 1.8424) / (2.029 - 1.9035)`
  and the corrected replays' own true-position floor sits above the shipped replays'.

## 5. How far to trust it

- **Corrected after review: the matched share is only known to about ±0.2.** It moves with the
  statistic used to match scale: the least-squares slope that was ruled, the median chord-length
  ratio reported beside it, or a reference slope a few percent either side of 1. C fails under every
  choice, and "almost all of the own-scale drop is movement scale" holds under every choice. This
  replaces the common-factor bound this section first gave. The scaling is also not exactly neutral
  for the filter: it scales positions and measurement noise but not the Kalman's initial covariance
  or process noise. That effect is small and slightly favours the correction. Review checks, not
  registered.
- **Corrected after review: what the page called the truth reading short is ST2's chain disagreeing
  with the feed.** Through ST2's own chain, box-feet movement reads a few percent short of the feed on
  most clips, and the corrected map lands far nearer the chain than the feed. Its perfect clicks are
  the chain's own landmark pixels, so where chain and feed disagree the correction follows the chain
  by construction. A parallel session's provenance check (`model/validation/camera_models.py --provenance`,
  uncommitted) finds ST2's 118575 calibration was fitted on the same keypoints the UI's 21 landmarks
  match, so no clicks in this repo are independent of the chain yet. How the feed itself was produced
  is not documented. Review checks, not registered.
- **Not disclosed until review: the feed's grid staircase was not removed.** N26 required it
  ([`n26-map-oracle.md`](n26-map-oracle.md) §5). By N26's review the grid inflates the truth's HSR,
  so removing it would raise both ratios. Its size here is not measured.
- **N26 and N27 agree once both are read against the fragments' floor.** N26 read each map at its own
  scale against true positions through the same fragments, and found the correction removes all of
  the excess above them. At its own scale N27 finds the same; at matched scale, about half on the same
  fragments.
- **Scope:** eight ST2 clips from one kind of rig, a fixed stitched panorama. Sigma 0 is N6's
  calibration; noise is N12's synthetic click noise, not a real operator.

## 6. What it changes

- **Not shipped.** `TpsCorrected` stays in `model/engine/calibrators.py` and `app/server/jobs.py` still
  accepts `perspective_tps`, but nothing writes it. It is the measured alternative, like N25's
  filter.
- **Withdrawn after review: "for speed, the map is not the lever once movement scale is right."**
  Above the pipeline's own true-position floor, the correction removes about half of the excess on the
  same fragments at matched scale, and all of it at its own scale. The floor itself, HSR about 1.7
  times the truth with perfect positions through the product's fragments, is what a speed measurement
  that is not about the map has to explain. It should be sized against that floor, not against 1.0.
- **For position, the correction cuts error at feet to about a third pooled, and a quarter to three
  fifths per clip, at click noise up to 5 px.** A position test needs clicks that do not come from
  ST2's chain, scoring against both the feed and the chain, per clip, with the p90 tail at heavy noise
  and the movement scale inside it.
- **Every speed arm should carry a movement-scale measurement from now on.** The largest HSR effect
  N25–N27 found ran through it.
