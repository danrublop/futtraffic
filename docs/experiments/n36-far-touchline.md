# N36: the far touchline — registered to the paint, and what the paint then says about the camera

**2026-09-17, branch `n30-operator-clicks`.** Test and rule: [`model/validation/n36_far_touchline.py`](../../model/validation/n36_far_touchline.py),
committed at `fd9132a` before the fix was written. Fix: `native=` in [`model/engine/autocal.py`](../../model/engine/autocal.py) `search`.
Probes: [`model/validation/n36_probes/`](../../model/validation/n36_probes). **Reviewed 2026-09-17 by three independent agents
(protocol, code, status); their corrections are folded in below and marked CORRECTION.** Follows
`research-optimal-calibration.md` (Q3: the far side's error is bias, not noise) and
[`n34-zero-click.md`](n34-zero-click.md). **Development, ONE install, seven clips of one pitch through one camera. The
registered fix FAILED its rule; nothing in the product's default path changed. The one configuration that passes the rule
(section 3, one global cubic in the vertical law) was found AFTER the rule, on the same clips, so it is a candidate.**

Every number in section 2 is printed by `--show`, which reads what `--before`, `--after`, `--after --width 67` and
`--after --warp` wrote to `model/validation/out/stc2025/n36_*.json` (each a run of about eight minutes over the seven clips'
40-frame medians, pitch 105 x 68 unless said, zero-click cameras of `n34_autocal.json` as the start).

## 1. The test, written first

Three measurements, none of which reads the position feed:

- **T, terminations.** A line sample constrains a camera only across the line, so where the halfway line ENDS is
  never fitted. T is the row where it ends minus the row of the camera's far touchline, in px. CORRECTION: that
  independence holds only while the model's far touchline sits on the boards. The halfway line ends AT the far
  touchline's paint, so once that paint is the fit target T is the fit's own residual there (the review measured T
  minus the touch_far residual at 0.1-0.25 px on all seven clips). Under the rigid camera T measures something real
  - the model cannot reach the paint; under any model that can, "T within 1 px" says only that the optimiser converged.
- **N, N32's five far references** (`n32-depth-reference.md`): three throw-in takers and two
  players at the far grass edge, reviewed blind; the signed depth error of the map at their feet. CORRECTION: N is
  T in metres (0.6 m per px at the far touchline) read at three players' box bottoms, from two clips, against a truth
  band 1.1 m wide (N32 takes -0.3 for feet in [-1.0, +0.1]). It can tell 2 m from 0; it cannot tell 0.02 from 0.5.
- **P, held-out interior paint**: the far side must not be bought with the interior.

Rule: T within 1 px on 6 of 7 clips, |median N| at most 1.0 m, P no more than 0.05 m worse on 6 of 7 clips.

## 2. What happened

| configuration | T (px) | clips within 1 px | N median (m) | P (m) | clips P no worse | far-touchline samples on paint | rule |
|---|---|---|---|---|---|---|---|
| before: the perimeter rule that ships | 3.76 to 4.37 | 0 of 7 | 2.15 | 0.207 to 0.272 | — | 43 to 64 of about 210 | — |
| **the registered fix** (as scored: configuration 2, the STRONGEST grass-flanked ridge; the registered "first" ridge took grass texture and never reached a score) | 2.27 to 2.80 | 0 of 7 | 1.19 | 0.259 to 0.326 | 3 of 7 | 157 to 209 | **FAIL** |
| exploratory: the same on a 67 m wide pitch | 2.15 to 5.23 | 0 of 7 | 1.07 | 0.394 to 0.543 | 0 of 7 | 167 to 210 | FAIL |
| exploratory: the same at 68 m, then N31's column warp freed | 0.92 (the one negative) to 0.84 | 7 of 7 | 0.02 | 0.188 to 0.474 | 5 of 7 | 167 to 210 | FAIL (P) |
| CORRECTION, found by the review: one global cubic in the vertical law, kappa = -0.4 (`--after --kappa -0.4`) | 0.13 to 0.71 | 7 of 7 | 0.40 | 0.210 to 0.375 | 6 of 7 | 169 to 210 | passes the letter of the rule |
| the same at kappa -0.2 / -0.6 | 1.11 to 1.71 / -1.61 to 0.41 | 0 of 7 / 6 of 7 | 0.76 / -0.85 | 0.195 to 0.273 / 0.224 to 0.422 | 7 of 7 / 5 of 7 | | FAIL / FAIL |

    @M 3.76,4.37,2.15,0.207,0.272,paint 43,43 to 64 :: .venv/bin/python model/validation/n36_far_touchline.py --show
    @M 2.27,2.80,1.19,0.259,0.326,157,209 :: .venv/bin/python model/validation/n36_far_touchline.py --show
    @M 5.23,1.07,0.394,0.543 :: .venv/bin/python model/validation/n36_far_touchline.py --show
    @M -0.92,0.84,0.02,0.188,0.474 :: .venv/bin/python model/validation/n36_far_touchline.py --show
    @M 0.13,0.71,0.40,0.210,0.375,169,210 :: .venv/bin/python model/validation/n36_far_touchline.py --show
    @M 1.11,1.71,0.76,0.195,0.273,-1.61,0.41,-0.85,0.224,0.422 :: .venv/bin/python model/validation/n36_far_touchline.py --show

**How many configurations saw these clips.** The fix itself: first-ridge (registered; never scored), strongest-ridge
(scored as "registered"), and the three constants FLANKS_PX, PERIM_RADIUS, GRASS_MIN settled while making it work. Then
67 m; the warp at four shrinkages (`warp_ridge_sweep.py`: P 5/7, 5/7, 5/7 and 1/7 at 2.0) and at degree 3; the width
scan (13 widths, T printed for each); the kappa scan (9 values, T printed for each); and the three kappa runs above.
CORRECTION: the page first said n = 9; counting every configuration whose T on these clips was read, it is at least
30. No configuration chosen before the data passes. **The old perimeter rule stays the default.** The kappa = -0.4
row passes the rule's letter, with one parameter, but it was found by the review after everything else had been read
off these seven clips, and T no longer measures anything independent for a model that reaches the paint (section 1).
It is the candidate for a test on data it has not seen - not a result.

What the fix did do is what it was for: the far touchline went from a quarter of its samples on something to nearly
all of them on paint, and the three throw-in takers, who stand on the line by Law 15, moved from 2.15, 2.87 and 0.48 m
inside the pitch to 0.94, 0.02 and 0.02 m under the warped camera (0.40, 0.56 and 0.89 m under kappa -0.4) - which,
given N's 1.1 m truth band, says "on the line to within about half a metre", not "to 0.02 m".

## 3. Why a rigid camera cannot finish the job: three explanations, two refuted

With the paint found, the rigid spherical camera still leaves the far touchline 2.3-2.8 px from where the halfway
line ends, and pays for what it gains there with the interior. Something in the MODEL is wrong at the top of the
frame. Out there one pixel is 0.055 m of HEIGHT (range over focal length), so centimetres matter.

    @D 0.0555 :: 84/1513   [range to the far touchline in m, over the camera's px per radian; both from n34-zero-click.md]


**The pitch is narrower than 68 m — not supported.** The halfway line's two ends and its two crossings of the centre
circle are collinear; rays to collinear points lie in a plane, and position along the line is `y0 + D tan(theta -
theta0)` in the in-plane angle. Four rows, four unknowns, one of them the width; the camera's position, height and tilt
never enter, only its intrinsics. It reads 66.90 m
(from 66.63 to 67.19; seven clips of one pitch, not seven measurements; 0.6 m per px of the far end and 1.1 m per px
of the far circle crossing). But it assumes the row is linear in elevation, which is the very thing in doubt: the
review re-read the same four rows under a cubic law and got 68 m at kappa about -0.6, 67.5 m at -0.4. CORRECTION: the
page first said "the registration contradicts 67 m" from the `--width 67` run; that run re-settled from a camera
converged at 68 m and two clips wandered (118575's principal point moved 550 px, 128057's length to 103.5 m), and
`width_scan.py` shows the paint cannot rank widths at all (hits and RMS flat from 64 to 69 m). Width and a vertical
correction at the top of the frame cannot be told apart from this one view, and the warp does not separate them
either (a per-column affine map in v leaves a cross-ratio unchanged: the review re-read 66.5-67.1 m through the warped
camera). **An outside measurement of the width decides it** - which is what the pitch-identification pipeline is for
(`research-pitch-size-sources.md`).

    @M 66.90,66.63,67.19 :: .venv/bin/python model/validation/n36_probes/cross_ratio_width.py

**A uniform crown — refuted; a crown confined to the middle — untestable from here.** A straight line images to a great circle whatever the camera; a line across a
crowned pitch arches out of it. Fitting `z = -c (y/34)^2` to the out-of-plane angle of both goal lines (goal mouths
excluded) gives c between 0.001 and
0.022 m on the right goal line of all seven clips and
at most 0.054 m on the noisier left one. CORRECTION: the same estimator reads -0.05 to -0.09 m on the penalty-area fronts and +0.06 to +0.07 m on the halfway
line (where a crown cannot show), so its systematic floor is about 0.07 m, not centimetres; the review's injection
test shows it recovers a 0.14-0.20 m crown, which is what the far-side offset would need, so a crown that reaches the
goal lines is excluded with power. A crown only in the middle of the pitch cannot be tested from this view (the only
paint there is the halfway line, in the camera's own vertical plane). Goal-line arch RMS is 0.6-1.2 px on the right
and up to 2.8 px on the left.

    @M -0.001,-0.022,-0.054 :: .venv/bin/python model/validation/n36_probes/crown_great_circle.py

**The panorama is cylindrical, not spherical — refuted. A cubic the OTHER way — the review's finding.** With
`v = cv + fv (e + kappa e^3)`, kappa = 1/3 (a cylinder to third order) raises the line RMS from 1.450
to 1.770 px on 117093 and loses a hundred samples. CORRECTION: the page first dismissed negative kappa as "no
better"; the same probe's rows show kappa -0.4 GAINS 42 samples, lowers the 8 px RMS, takes T from 2.30 to 0.34 px and
halves the left/right asymmetry, and under the committed test (table above) kappa = -0.4 passes T, N and P with one
parameter where the twelve-coefficient warp fails P. It is now the leading explanation of the top-of-frame term: the
panorama's rows are not quite linear in elevation. Both the `_fwd` and `_inv` of `model/engine/autocal.py` carry it as
`VERT_CUBIC` (0 by default: the product is unchanged).

    @M 1.450,1.770 :: .venv/bin/python model/validation/n36_probes/vertical_kappa.py 117093

**Each lens has its own small vertical correction — what survives.** The rigid fit's residuals are not symmetric:
the far edge of the left penalty area sits +0.8 px and the right one -1.5 px (`vertical_kappa.py`, kappa 0, clip
117093 only). That is
column-dependent, which is what N31's warp `a(u) + b(u)(v - v0)` models
(`research-unstitching-panorama.md` Q1). N31 found the warp unhelpful - but N31's
far touchline was on the boards, so the warp had nothing true to bend toward at the top of the frame. Given the paint,
it takes T inside a pixel (which, for a model that reaches the paint, is convergence, not evidence) and the throw-in
takers to within half a metre of the line. Kappa does the same with one parameter and a smaller hold-out cost.

Where the warp's hold-out loss comes from (`model/validation/n36_probes/warp_holdout_families.py`, n = 2 warp degrees, both
on clips already seen): held out, the FAR edges of the penalty areas improve under the warp on all seven clips (on
118577 from 0.83 to 0.39 m), while the centre circle and the fronts of the penalty areas get worse (the circle on
118575 from 0.11 to 0.81 m). CORRECTION: the page first blamed twelve unconstrained coefficients. The review's reading fits better: the warp
splits the far-side discrepancy between the far touchline and the circle crossing (residuals of -0.3 to -0.8 px at the
far end and +0.3 to +0.7 px at the far circle crossing under the warped camera), so the circle is where it pays. A
degree-3 warp does not cure that and gives back T. Either way the warp is no longer the candidate; kappa is.

    @M 0.83,0.39,0.11,0.81 :: .venv/bin/python model/validation/n36_probes/warp_holdout_families.py

## 4. Players' feet against the feed

`model/validation/n36_probes/feet_vs_feed.py`: MOT box feet on N31's development frames (f % 20 == 10, n = 46097
feet pooled over seven clips, 0 non-finite dropped - the script prints it), split along and across the camera ray
because the feed's depth is itself in question (`audit-n33.md`).

| camera | median (m) | far half (m) | across the ray, median (m) | across, far half (m) | along the ray, signed % of range |
|---|---|---|---|---|---|
| zero-click as it ships (n34) | 2.28 | 2.84 | 0.36 | 0.41 | 3.91 short |
| paint rule, rigid | 2.05 | 2.50 | 0.35 | 0.40 | 3.53 short |
| paint rule + warp | 2.14 | 2.57 | 0.28 | 0.29 | 3.85 short |

    @M 46097,2.28,2.84,0.36,0.41,-3.91 :: .venv/bin/python model/validation/n36_probes/feet_vs_feed.py
    @M 2.05,2.50,0.35,0.40,-3.53 :: .venv/bin/python model/validation/n36_probes/feet_vs_feed.py
    @M 2.14,2.57,0.28,0.29,-3.85 :: .venv/bin/python model/validation/n36_probes/feet_vs_feed.py

Across the ray, where the feed's depth question cannot reach, the warped camera is the best of the three and the far
half stops being worse than the near. Along the ray nothing moved: every camera reads players about 4% of range
nearer than the feed, near half (3.2-3.7%) as well as far. So the 4% was never our far-touchline bias. CORRECTION:
it is NOT a constant fraction of range either - under the shipping camera -1.65% within 25 m, -3.35% at 25-50 m,
-4.12% at 50-70 m, -4.15% beyond 70 m (under native+warp -2.42, -3.79, -4.00, -3.55). A feed scaled about the
camera's foot would give the same percentage at every range; a growing one is what a camera height or vertical-law
error looks like. So the gap is at least partly ours, and the sentence this page first carried - "more weight on the
feed being scaled" - is withdrawn. What decides ownership is still what `audit-n33.md` asked for and
nobody has run: players on known marks at restarts, n >= 20.

    @M -1.65,-3.35,-4.12,-4.15,-2.42,-3.79,-4.00,-3.55 :: .venv/bin/python model/validation/n36_probes/feet_vs_feed.py

## 5. Corrections to earlier pages

- `n32-depth-reference.md`'s extension took an edge-stander's truth as y = 0 because "on
  this install the far touchline is painted at the grass edge". Where this page finds the touchline there is grass
  beyond it, so that truth does not hold in general and the two edge-standers carry none. CORRECTION: the page
  first said the -2.9 to -4.0 m read for one of them "is probably right" and did not mention the other, which reads
  +1.3 m INSIDE under the warp at a column where the review finds no grass-flanked ridge at all (board to row 156,
  grass from 158, foot at 159). Neither is evidence for anything; both are dropped. N's median rests on the three
  throw-ins.
- `research-optimal-calibration.md` Q3 called the far side's error association
  alone. Association is about 1.5 px of the 4 (T 3.8-4.4 before, 2.3-2.8 after); the rest is a vertical term the
  rigid equirect camera does not have.

## 6. What this changes, and what it does not

- The product is unchanged. `search`, `register` and `_settle` take `native=`; `VERT_CUBIC` is 0; without them the
  code is what it was (the review diffed the default path bit for bit against fd9132a on three clips), and
  `model/engine/autocal.py`'s self-check now carries the board-beside-the-paint case the old rule fails.
- The candidate for the next pre-registered test is **paint rule + kappa = -0.4** (one parameter), ahead of the column
  warp. It has been seen on all seven clips and it was found after the rule, so these clips cannot confirm it, and T
  cannot be its instrument. What can: a second rig; restarts on a full match (kick-off positions are known by Law);
  and the width from outside, which would also settle section 3's first question.
- Width is NOT measured by this page. The cross-ratio reading is published as a disagreement, not as a width.
- Timing: `--before`/`--after` take minutes here because the held-out interior test re-registers five times per
  clip; the probes take seconds.
