# Pitch mapping, explained

**Branch:** `tracker-association`. **Written:** 2026-09-09.
**Audience:** an engineer who has to change this and needs to know what is load-bearing.

---

## 0. How to read the numbers in this document

This repo has shipped hallucinated numbers before ([`docs/ledgers/ledger-claims.md`](../ledgers/ledger-claims.md) is the
list). So every number here carries one of three tags:

| tag | meaning |
|---|---|
| **`@RAN`** | I executed the command shown in this session and pasted its output. |
| **`@REC`** | Recorded in the repo at the `file:line` given. **Not** re-measured here. A docstring number is `@REC`, never measured. |
| **`@DER`** | Arithmetic on `@RAN` values, with the arithmetic shown. |

Everything untagged is an argument, not a fact. Section 6 is the list of things I
could **not** verify.

All commands below assume `cd fut-traffic` and the repo venv
(`.venv/bin/python` — there is no bare `python` on this box).

---

## 1. There are TWO distinct pitch-mapping paths

They do not share a solver, they do not share a coordinate convention, and only
one of them can be reached from the UI.

| | **A. Product path** | **B. SoccerTrack / STC chain** |
|---|---|---|
| entry point | `app/frontend/src/lib/pitch.ts:380` → `POST /api/matches/{id}/calibration` (`app/server/app.py:309`) → `app/server/jobs.py:422` | `model/validation/st2_overlay.py:56`, `model/validation/st2_align.py:75` |
| what is fitted | one 3×3 image→pitch homography (optionally per-strip, optionally + bow, optionally + k1) | **nothing.** The vendor's `H` and `mapx` are loaded from disk and used as-is |
| data | 4–21 operator clicks on one still frame | `{seq}_homography.npy` + `{seq}_mapx.npy` shipped by SoccerTrack v2 |
| DoF | 8 per homography; ×`n_strips` if panoramic; +`deg+1` if bow; +1 if k1 | 0 fitted by us |
| direction | image px → pitch metres | pitch metres → panorama px (forward), KD-tree for the inverse |
| shipped? | **yes**, this is the only one an operator can produce | validation only — never loaded by `track.py`/`stats.py` |

### 1.A The product path, in execution order

```
operator clicks           app/frontend/src/app/setup/calibrate-step.tsx:64  (clicks state)
   -> solveClicks()       app/frontend/src/lib/pitch.ts:380
       -> fit()           app/frontend/src/lib/pitch.ts:297     weighted DLT, MAD outlier loop
   -> calibrationPayload  app/frontend/src/lib/pitch.ts:404     {projection_type, pitch, H, points, rms_error_m, outliers}
   -> POST                app/server/app.py:309
       -> jobs.load_calibration   app/server/jobs.py:422        THE gate; re-loads with the worker's own call
           -> calibrators.load    model/engine/calibrators.py:1039
               -> from_dict       model/engine/calibrators.py:993
   -> track.py:826  cal = calibrators.load(calib)
       -> cal.undistort_frame(frame)   model/engine/calibrators.py:371 (remap) / :232 (identity)
       -> detector -> boxes -> box bottom-centre is the ground point
       -> make_prior  model/engine/track.py:491 -> sceneprior.camera_from_homography  model/engine/sceneprior.py:60
   -> stats.player_stats -> kalman_ground  model/engine/stats.py:271
       -> homography_jacobian  model/engine/stats.py:250   (R in metres, from cal.H)
```

**What is fitted, from what, how many DoF.** `PerspectiveCalibrator`
(`model/engine/calibrators.py:266`) wraps `pitch.Calibration.solve`
(`model/engine/pitch.py:122`): a single 3×3, 8 DoF, solved by
`cv2.findHomography(src_px, dst_metres, LMEDS)` when there are ≥5 points and by
the exact 4-point method otherwise (`model/engine/pitch.py:133`). `src` is pixels,
`dst` is metres, so **H maps image → pitch directly**; there is no separate
inverse stored.

**Assumptions it makes**, and where each is recorded as already false:

- the pitch is a plane ([`docs/ledgers/assumptions.md:26`](../ledgers/assumptions.md) A3, **UNVERIFIED** `@REC`)
- the image is a flat perspective render ([`docs/ledgers/assumptions.md:27`](../ledgers/assumptions.md) A4,
  **FALSIFIED** on our only real footage `@REC`)
- the camera is rigidly fixed ([`docs/ledgers/assumptions.md:24`](../ledgers/assumptions.md) A1, VERIFIED over 30 s
  only `@REC`)
- square pixels / principal point at centre — required by
  `sceneprior.camera_from_homography` (`model/engine/sceneprior.py:66`), and this is
  why `load_maps`' `yscale` knob is deprecated in favour of `crop`+`scale`
  (`model/engine/calibrators.py:100-107`)

**Where the product path is inverted:** see §3.

### 1.B The SoccerTrack chain

```
pitch metres (corner origin)
  --H (vendor 3x3)-->  REMAPPED canvas px
  --round to int, mapx[v,u] lookup-->  RAW panorama px
```

Both halves are in `model/validation/st2_overlay.py:83-101` and
`model/validation/st2_align.py:80-85`. `mapx` is OpenCV fixed-point `CV_16SC2`, so
`mapx[v,u]` is the **source** panorama pixel feeding remapped pixel `(u,v)` —
i.e. the table natively runs remapped→raw, which is the direction this chain
wants.

The trap, stated three times in the repo and worth restating: **applying `H`
alone is not a calibration failure, it is a coordinate-frame error.** `H` lands
on the remapped canvas, where the pitch is a thin band.

```
@RAN  .venv/bin/python  (script in transcript; loads {seq}_homography.npy + {seq}_mapx.npy)
128057: pitch 105.0x68.0  cond(H) 2.024e+06
   remapped canvas: x 570..3397  y 414..626  -> 213 px for 68.0 m
   panorama px    : 1000/1000 in table; y 151..724 -> 573 px
132831: cond(H) 1.466e+13
   remapped canvas: y 1003..1003  -> 0 px for 68.0 m       <- genuinely degenerate
```

The 213 px figure quoted in `model/validation/st2_metres.py:11` and
`context-pitch-mapping-pivot (not published)` is **confirmed**. 132831's degeneracy
(the whole pitch collapsing to one point) is **confirmed**.

Fit quality of the two variants, on 118575, which is the only match shipping a
keypoints file:

```
@RAN  (65 published keypoints, data/footage/soccertrack_calib_118575/118575_keypoints.json)
RANSAC pitch->RAW panorama: median 103.4 px  inliers 11/65
LSQ    pitch->RAW panorama: median  46.4 px  p90 362.4
vendor chain H+mapx       : median   7.8 px  p90  25.2
```

This reproduces `model/validation/st2_align.py:14` (103 px, 11/65) and `:22`
(median 7.8 px) exactly. Note the 103 px is the **RANSAC** number; plain
least-squares on the raw panorama is 46.4 px median with a 362 px p90 — the
distinction is not made anywhere in the docstrings.

**Nothing in the product loads this chain.** `app/server/jobs.py:453` accepts only
`perspective` and `remap`. `RemapCalibrator` is our *reimplementation* of the
same idea (remap the frame, then one homography of our own) and explicitly
declines to consume a vendor `H` (`model/engine/calibrators.py:319-327`).

### 1.C Line-based calibration code (`model/engine/pancheck.py`)

`points_from_lines` (`model/engine/pancheck.py:407`) is the **only line-based
calibration code in the repo**: it total-least-squares-refits Hough segments to
the paint under them (`:378`), intersects every constant-x line with every
constant-y line, and emits `"(x, y)"` literal-metre clicks that
`pitch.resolve_landmarks` already accepts. It is not wired to anything.

---

## 2. The product path in detail

### 2.1 How many landmarks does the UI collect, and which

**21 named landmarks.** `LANDMARK_COUNT = 21` at `app/frontend/src/lib/pitch.ts:101`;
the list is `app/frontend/src/lib/pitch.ts:67-99` and must stay in lockstep with
`model/engine/pitch.py:36-69` or the server rejects the export with "unknown
landmarks". Human labels at `app/frontend/src/lib/pitch.ts:43-65`.

```
corner_bl  corner_tl                    pitch corners, left goal line
pen_l_bl   pen_l_tl                     penalty box, on the goal line
pen_l_br   pen_l_tr                     penalty box, outer face (x = 16.5 m)
goal_l_br  goal_l_tr                    6-yard box, outer face (x = 5.5 m)
half_b     half_t     centre            halfway line ends + centre spot
circle_b   circle_t                     centre circle top/bottom (y = cy +- 9.15)
goal_r_bl  goal_r_tl                    6-yard box, right
pen_r_bl   pen_r_tl   pen_r_br  pen_r_tr    penalty box, right
corner_br  corner_tr                    pitch corners, right goal line
```

Five of the seven dimensions are laws-of-the-game constants and the operator
types only length and width (`model/engine/pitch.py:22-30`,
`app/frontend/src/app/setup/calibrate-step.tsx:556-568`). Defaults are **118×72 yd**
(`app/frontend/src/lib/pitch.ts:33`), *not* 105×68 — deliberate, because NCAA fields
are not FIFA-sized and hardcoding 105×68 is `@REC` a "4–6 % systematic error on
every distance" (`model/engine/pitch.py:5`).

**The structural weakness of this set, and it is measured.** The 21 landmarks
sit in exactly three x-clusters — left box, halfway, right box — because that is
where a pitch has paint, leaving ~36 m gaps either side of centre. `@REC`
`model/engine/pitch.py:86-91`: on 118575 the 21 named landmarks give **6.69 m
held-out**, which is no better than a single homography's 6.53 m; the same
match's 65 dataset keypoints (42 of them sampled along the touchlines at 5.5 m
spacing) reach **0.84 m**. Those 65 are **not clickable** — nobody can pick out
"x = 5.5 m along the touchline" — so the operator flow cannot produce them.

### 2.2 How the homography is solved

Two solvers exist and they are *not* the same code:

| | browser preview | server / worker |
|---|---|---|
| where | `app/frontend/src/lib/pitch.ts:236` `solveW` | `model/engine/pitch.py:134` (perspective) / `model/engine/calibrators.py:48` `_fit_weighted` (panoramic) |
| method | Hartley-normalised weighted DLT, 8×8 normal equations, Gauss–Jordan, `h33 = 1` | `cv2.findHomography(..., LMEDS)` / SVD null vector of the weighted DLT |
| outlier rule | median+6·MAD, ≤3 iterations, bails if >n/5 rejected (`pitch.ts:312-344`) | LMEDS mask (perspective) / same median+MAD loop (`calibrators.py:455-471`) |

`app/frontend/src/lib/pitch.ts:232-235` names this discrepancy itself and says the
two "agree to well under a centimetre on clean input". The posted payload carries
both `H` and `points`, so the server can always re-solve; `from_dict` chooses to
**trust the stored H** rather than refit (`model/engine/calibrators.py:1012`).

The `H` in the payload is row-major image-px → pitch-metres
(`app/frontend/src/lib/pitch.ts:286`, `:414`).

### 2.3 Is there a distortion model?

**Not in the shipped operator flow.** Three exist in `calibrators.py` and none
is reachable from the UI:

1. **`UndistortWrapper`** (`model/engine/calibrators.py:645`) — `cv2.undistortPoints`
   with a real `K` + `distCoeffs`, applied **before** the inner projection
   (`:689`). `get_calibrator` builds it only when `dist_coeffs=` is passed
   (`:985`); nothing in `app/server/` or `app/frontend/` passes it.
2. **`estimate_k1`** (`model/engine/calibrators.py:526`) — label-free plumb-line
   recovery of one radial coefficient from detected pitch lines. Self-checked
   against planted distortion `@RAN` (`calibrators.py ok (k1 recovered +0.073 as
   the correction for a planted -0.080)`). **Never scored against real player positions.**
3. **`BowWrapper`** (`model/engine/calibrators.py:702`) — a `deg`-degree polynomial in
   image column that shifts the image row, plus one linear y term (`:860-867`).
   Fitted jointly with the homography by `least_squares` (`:906`), guarded by a
   5-fold held-out gate that refuses to exist unless it beats the plain
   homography by ≥20 % (`MIN_GAIN = 0.8`, `:832`, `:914-949`).
   **`DO NOT ENABLE THIS ON 118575`** is the first line of its docstring, and §4
   explains why.

`PanoramicCalibrator` is *also* a distortion model in effect — a piecewise
homography absorbing cylindrical bend — and it is **refused by the pipeline**
(`app/server/jobs.py:453`, and the Save button is disabled for it at
`app/frontend/src/app/setup/calibrate-step.tsx:617`, with the reason shown verbatim
to the operator at `:577-590`).

### 2.4 What happens when landmarks are insufficient or degenerate

Every gate, in the order a bad click set hits them:

| condition | where | behaviour |
|---|---|---|
| < 4 clicks | `app/frontend/src/lib/pitch.ts:387` | `solveClicks` returns `null`, Save disabled, UI says "Place at least four landmarks that are not collinear" (`calibrate-step.tsx:635`) |
| < 4 clicks (server) | `model/engine/pitch.py:125` | `ValueError: need >=4 landmarks, got N` |
| exactly 4 | `model/engine/pitch.py:133` | falls back from LMEDS to the exact method — **no outlier rejection at all at n=4** |
| collinear | `model/engine/pitch.py:136` | `cv2.findHomography` returns `None` → `ValueError("homography failed - are the points collinear?")` |
| collinear (weighted DLT) | `model/engine/calibrators.py:68` | `abs(H[2,2]) < 1e-12` → `ValueError("degenerate homography")` |
| unknown landmark name | `model/engine/pitch.py:106` | raises. Deliberate: silently dropping a typo'd click is a calibration that looks fine (`calibrators.py:1387`) |
| singular H that still "solves" | `app/server/jobs.py:475` | `det(H) < 1e-12` → refused. This is the one bad calibration that raises nowhere else: it collapses the frame to one pitch point and every distance comes out a confident zero |
| non-perspective/remap type | `app/server/jobs.py:453` | 400 at POST, with the operator-facing paragraph passed through verbatim |
| `points: {}` on save | `model/engine/calibrators.py:259` | **warning only, not an error.** A point-free file is permanently unscorable by `calcheck` — this is how the shipped 118575 calibration sat unscorable |
| < `min_pts` for panoramic | `model/engine/calibrators.py:429` | `ValueError("...with fewer, every strip fits the same points and this is just a single homography wearing a hat")` |
| > n/5 landmarks look wrong | `model/engine/calibrators.py:465`, `pitch.ts:340` | **stop rejecting.** "If a fifth of the landmarks look wrong, the MODEL is wrong, not the clicks" — report the ugly number rather than deleting the evidence |
| camera unrecoverable from H | `model/engine/sceneprior.py:146` | raises if recovered height is outside 1–200 m; `track.make_prior` catches, prints, and runs unfiltered |

There is **no scale check on the click set itself**, and this is documented as a
known blind spot: `model/engine/calcheck.py:274-302` asserts that a click set taken
from a pitch 3× too small scores "good" on held-out error. The only guard is
`calcheck.height_sanity` (`:127`), which reads box heights the calibration never
saw and asks what human height they imply. `@REC` `model/engine/calcheck.py:59-61`: on
118575's 31 121 real detections the shipped calibration implies **1.76 m**.
It fires at ~1.4× error, not at 10 % (`:66-69`).

---

## 3. Every place the mapping is inverted (pixel → metres, or metres → pixel)

The forward and inverse directions have completely different error, so it matters
which one each site uses. **The measured asymmetry:**

```
@RAN  .venv/bin/python model/validation/st2_overlay.py --check --seq 128057
128057: drawn vs painted, signed perpendicular, 4 frames
       marking   signed   |med|     p90  unmatched
       outline     +0.0     3.0    17.5      32.8%
       halfway     -1.5     1.5     4.0       1.2%
 centre circle     +5.0     7.0    13.0      29.8%
     penalty L     +3.0     3.0    13.0      11.3%
     penalty R     +0.0     5.0    22.0      22.9%
    goal box L     +1.0     1.0    17.1       2.6%
    goal box R     +2.0     4.0    12.0       0.0%
           ALL     +0.0     3.0    16.0
```

```
@RAN  .venv/bin/python model/validation/st2_metres.py --seq 128057
128057: pixel -> metres, 20 frames, 440 instants, 114933 grid points
  overall            median  1.26 m   p90  5.26 m
  farthest quartile  median  3.23 m   p90  9.46 m
  25-50%             median  1.60 m   p90  5.48 m
  50-75%             median  1.08 m   p90  2.96 m
  nearest quartile   median  0.74 m   p90  1.63 m
```

Both tables in `context-pitch-mapping-pivot (not published)` §3.1 reproduce exactly.

**A detail the pivot doc does not state and which changes the reading:**
`st2_metres.run()` inverts **ground-truth MOT boxes**, not our detections —
`gt_feet()` at `model/validation/st2_metres.py:86` opens `/…/st2/mot/{seq}.txt`. So
1.26 m / 3.23 m contain **zero detector error**. They are mapping + ground-point
convention + truth quantisation only.

### The inversion sites

| # | site | direction | mechanism | notes |
|---|---|---|
| 1 | `model/engine/pitch.py:147` `Calibration.to_pitch` | px → m | **forward matrix apply.** `H` is *already* image→pitch, so this is not an inverse at all | the product's main measurement path |
| 2 | `model/engine/calibrators.py:496` `PanoramicCalibrator.to_pitch` | px → m | forward apply of `n_strips` homographies, then **linear blend in image x** between the two bracketing strip centres, clamped outside the clicked span (`:507`) | continuous by construction — a hard switch would step every player's velocity at a seam |
| 3 | `model/engine/overlay.py:291` `pitch_lines` | m → px | **`np.linalg.inv(cal.H)`, a matrix inverse.** Sign pinned on the centre spot first (`:300`), `w>0` checked to drop points behind the camera (`:311`) | **this is the product renderer and it does NOT densify.** See the bug in §4.6 |
| 4 | `app/frontend/src/lib/pitch.ts:361` `toImage` | m → px | **quasi-Newton, 8 iterations.** Seeded from the global `inv3(H)`, then steps by the pitch-space residual through the local strip inverse | the comment at `:357` says blending the inverses is *not* the inverse of the blended forward map — "lands ~19 px out on a stitched frame" |
| 5 | `model/validation/st2_metres.py:71` `inverse()` | px → m | **KD-tree (`scipy.spatial.cKDTree`) over a 0.25 m forward-evaluated grid.** No closed form exists because of the `mapx` lookup | returns the *nearest grid point*, so its resolution floor is the grid step **and** the pixel quantisation — measured in §4.2 |
| 6 | `model/engine/calibrators.py:173` `remap_points` | raw px → remapped px | **nearest table entry + one exact Newton step through a local linear model, then a forward re-check to `tol_px=0.5`** | refuses uncovered points rather than snapping to an edge (`:209`). Self-verified `@RAN` at 0.012 px median / 0.066 px max over 200 pixels |
| 7 | `model/engine/sceneprior.py:70` `camera_from_homography` | — | `np.linalg.inv(H_img2pitch)` to get `G` (metres→px), then Zhang's single-homography decomposition | feeds the height gate and the pitch mask |
| 8 | `model/engine/calcheck.py:313` `_selfcheck` | m → px | `np.linalg.inv(good.H)` | synthetic detections only |
| 9 | `model/engine/stats.py:250` `homography_jacobian` | — | analytic **d(metres)/d(pixels)** — a *derivative* of the forward map, not an inverse | this is what makes the Kalman's `R` anisotropic |
| 10 | `model/engine/ball.py:397` | m → px | `np.linalg.inv(cal.H)` | ball path rendering |
| 11 | `model/validation/dist_eval.py:103`, `model/validation/oracle_eval.py:141` | m → px | `np.linalg.inv(H)` | evaluation only |
| 12 | `model/engine/pancheck.py:144` | m → px | `np.linalg.inv(cal.H)` for probe placement | measurement instrument |

**So: five distinct inversion mechanisms** — matrix inverse (3, 7, 8, 10, 11,
12), strip-blend forward apply (2), quasi-Newton (4), KD-tree (5), and
Newton-through-a-lookup-table (6). Only #5 and #6 handle a stitched frame
honestly; #3 does not, and it is the one the product uses to draw the check
picture.

---

## 4. Error budget

Geometry of the reference clip, measured, because every magnitude below scales
with it:

```
@RAN  camera recovered from 128057's remapped-canvas H (sceneprior.camera_from_homography)
  f = 446 px   height = 7.26 m   at pitch (75.1, 97.8) m
  i.e. ~30 m OUTSIDE the near touchline, 7.3 m up.

@RAN  depth resolution at x = 52.5 m, walking y from far (0) to near (68):
   y m  remapped row  panorama row  m per remapped row  m per panorama row
     0         421.9         155.0               1.855                 (int-quantised)
    16         432.3         188.0               1.246                 0.667
    32         449.0         244.0               0.743                 0.286
    48         479.4         342.0               0.370                 0.125
    64         553.6         590.0               0.126                 0.038
```

**14.7× compression across the depth range on the remapped canvas.** That single
number drives most of the table below.

```
@RAN  metres per pixel on the SHIPPED 118575 calibration, on its own 11878x870 canvas
       (SVD of stats.homography_jacobian over 602 on-pitch samples)
  major 0.028-0.616   minor 0.004-0.057
    far  y  0-17  median major 0.567 m/px   minor 0.049 m/px
    y      17-34  median major 0.333        minor 0.037
    y      34-51  median major 0.181        minor 0.024
    near y 51-68  median major 0.072 m/px   minor 0.011 m/px
```

This independently reproduces an internal note's recorded "0.53 m per pixel on
the worst band" (0.567 measured). It does **not** reproduce
`model/engine/stats.py:116`'s "0.45 m/px near, 2.07 m/px far" — see §6.

### The budget

| # | source | near side | far side | provenance |
|---|---|---|
| 1 | **Homography / model fit residual, held out** | — | — | `@RAN` `calcheck.py --real`, table below. Single homography **2.50 m** leave-one-out on 118575; region splits 4.66–14.12 m |
| 2 | **Stitching residual not removed by the remap** | 3.44 px rms | 1.54 px rms | `@RAN` `calibrators.py --real`: touchline bow, raw → remapped, near 49.00 → 3.44, far 17.43 → 1.54 |
| 3 | **`mapx` lookup / pixel quantisation** | **0.00 m** median | **1.00 m** median, 1.50 p90, 1.75 max | `@RAN`, §4.2 below — a hard floor no fit can beat |
| 4 | **Box-bottom-as-ground-point (foot in the air)** | **1.55 m** at 0.30 m lift | **4.15 m** at 0.30 m lift | `@DER` from the measured camera pose, §4.3 |
| 5 | **Detector box jitter** | σ_v 1.39 px → **0.053 m** | σ_v 1.45 px → **0.72 m** | `@RAN` σ, §4.4; metres `@DER` at the measured m/px |
| 6 | **Camera placement / depth compression** | 0.072 m/px | 0.567 m/px | `@RAN` above. **This is the multiplier on 3 and 5, not an independent term** |
| 7 | **Truth quantisation (ST2 `loc`)** | ~1.05 m in x, ~0.68 m in y | same | `@REC` `model/validation/st2_metres.py:45-47`. Contributes ~0.35 m of the 1.26 m median |

**Which dominate.**

- **Near side (0.74 m measured, `@RAN`):** truth quantisation (7, ~0.35 m) and
  the ground-point convention (4, up to 1.55 m at peak foot lift) are the whole
  story. Detector jitter (5) is 0.05 m — irrelevant. Pixel quantisation (3) is
  literally zero.
- **Far side (3.23 m measured, `@RAN`):** the ground-point convention (4, up to
  4.15 m) and the model residual (1) dominate, with pixel quantisation (3, 1.0 m)
  as a hard floor underneath them. Detector jitter (5, 0.72 m) is real but small
  in quadrature against 3.23 m.
- **Detector quality is not the far-side constraint.** This is the third
  independent measurement in this repo to say so, and the cleanest: the 1.26 m /
  3.23 m figures were produced from *ground-truth boxes*, so the detector
  contributed nothing to them.

### 4.1 Model fit residual, held out (reproduced exactly)

```
@RAN  cd engine && ../.venv/bin/python calcheck.py --real
held-out error, metres, 65 landmarks of calibration.json
split                   |   single homography    |      + bow deg 4       |      + bow deg 5       |  panoramic, 12 strips  |   thin-plate spline
                        |   med    p90     max   |   med    p90     max   |   med    p90     max   |   med    p90     max   |   med    p90     max
leave-one-out           |    2.50    2.50    6.01|    0.42    0.42    1.83|    0.48    0.48    1.78|    0.54    0.54  350.83|    0.62    0.62    5.77
random 1-in-5 x8        |    2.38    4.40    6.32|    0.43    1.11    2.87|    0.47    0.99    1.97|    0.58    1.46  507.78|    0.64    2.71   11.18
touchlines -> the rest  |    4.66    6.59    7.44|    1.00    2.74    2.87|    1.04    2.83    3.03|   10.81   53.41  174.93|   23.53   28.70   29.47
every 2nd touchline pt  |    2.51    4.05    4.57|    0.44    1.53    1.58|    0.44    1.38    1.81|    0.56    2.19    2.99|    0.40    1.17    1.26
x band 35..70 m out     |    4.77    5.82    6.86|    0.52    1.84    2.13|    0.53    1.78    2.09|   11.49   76.20  238.97|    1.17    2.28    3.38
near half -> far half   |   14.12   19.60   21.60|    6.82    8.14    8.46|    2.78    3.73    4.74|    7.47   11.62   17.19|   20.51   27.54   30.86
left half -> right      |    9.53   17.64   20.24|   44.23   64.58   73.42|    3.82    9.66   10.24|    3.23    8.41    8.96|   19.48   41.35   59.40
```

Read the **region** rows, not the random ones. Both the spline and the dense
panoramic fit win at random and blow up (23.5 m, 10.8 m, 350 m max) the moment
you ask for the middle of the pitch — which is the only part players stand on.
This is the trap `context-pitch-mapping-pivot (not published)` and an internal note
"Landmark hold-out selects the wrong population" both name: 42 of the 65
keypoints sit on two image rows.

And the whole-frame numbers:

```
@RAN  cd engine && ../.venv/bin/python calibrators.py --real
inversion self-check  dest->src->dest over 200 pixels: median 0.012 px  max 0.066 px
inversion witness     vendor H(world) vs our inverse(raw kp): median 3.9 px over a 2927x188 px span
source px per remapped px over the pitch interior:
  vertical    min 1.45 median 2.91 max 3.17
  horizontal  min 0.42 median 2.47 max 3.17
pitch band rows 390..690 of 1080 (300 rows, 27.8% of the canvas)  x2.9 isotropic -> 11878x870

position error, metres                               median      p95      max
single homography on RAW panorama (fit residual)       6.17    11.72    13.61
single homography on RAW panorama (leave-one-out)      6.57    12.62    14.89
REMAP then homography (fit residual)                   2.36     4.72     5.53
REMAP then homography (leave-one-out)                  2.50     4.99     6.01
REMAP + pitch-band crop, then homography (fit residual)     2.36     4.72     5.52
REMAP + pitch-band crop, then homography (leave-one-out)     2.50     4.99     6.01

  camera from the vendor  canvas 4096x1080: f=  469.3 px (= 0.1146 x frame width)  height= 7.69 m  at pitch (64.0, 91.0) m
  camera from the cropped canvas 11878x870: f= 1361.4 px (= 0.1146 x frame width)  height= 7.69 m  at pitch (63.9, 91.0) m

touchline bow (deviation from a straight line, px):
  far  (pitch y=0) raw       rms  17.43  max  34.06
  far  (pitch y=0) remapped  rms   1.54  max   3.39
  near (pitch y=68) raw       rms  49.00  max  82.41
  near (pitch y=68) remapped  rms   3.44  max   6.30

fit error implied by pure annotation scatter (Monte Carlo, 30 draws):
sigma, remapped px              vendor canvas   cropped canvas
  1.5                                 0.69 m          0.69 m
  2.5                                 1.26 m          1.20 m
  3.5                                 1.83 m          1.88 m
```

The Monte-Carlo row is the one people misread: at σ = 2.5 remapped px of pure
*annotation* scatter, the fit reads 1.26 m. So of the 2.50 m leave-one-out,
roughly half is the dataset's own hand, not our geometry
(`model/engine/calibrators.py:309-317`).

The crop is free: 2.36 / 4.72 / 5.52 versus 2.36 / 4.72 / 5.53 uncropped, and the
recovered focal is the same **fraction** of frame width (0.1146) on both canvases
— which is the isotropy assertion at `model/engine/calibrators.py:1586-1591`.

### 4.2 Lookup quantisation — a term nothing in the repo names

`model/validation/st2_overlay.py:93-94` rounds the remapped coordinate to an integer
before the `mapx` lookup. `st2_metres.inverse()` builds its KD-tree from that
same rounded forward map. So the inverse's resolution is bounded by how much
pitch one panorama pixel covers:

```
@RAN  0.25 m grid over 128057, pushed forward, grouped by the panorama pixel it lands on
128057: 114933 grid points at 0.25 m -> 74053 distinct panorama pixels (1.6 share a pixel)
  metre spread of the pitch points sharing ONE panorama pixel:
    far  y 0-17 m    n=  7766  median 1.00 m  p90 1.50 m  max 1.75 m
    y 17-34          n= 14669  median 0.25 m  p90 0.75 m  max 1.00 m
    y 34-51          n= 22797  median 0.00 m  p90 0.25 m  max 0.50 m
    near y 51-68     n= 28400  median 0.00 m  p90 0.00 m  max 0.50 m
```

**At the far touchline one panorama pixel is a median 1.00 m of pitch.** No
choice of model, fit, or refinement can produce a far-side inverse better than
that from this chain. It accounts for roughly a third of the measured 3.23 m far
quartile on its own.

### 4.3 Box-bottom-as-ground-point

`model/engine/groundpoint.py:8` gives the mechanism: `ground error ≈ d_horiz · h /
H_camera`. Using the camera pose I recovered above (7.26 m up, at pitch (75.1,
97.8) m) and `FOOT_LIFT_M = 0.30` from `model/engine/groundpoint.py:66`:

```
@DER  d_horiz * 0.30 / 7.26
  far touchline y=0    d_horiz 100.3 m ->  4.15 m
  centre       y=34    d_horiz  67.7 m ->  2.80 m
  near touchline y=68  d_horiz  37.4 m ->  1.55 m
```

This is a **peak**, not a median: 0.30 m is the top of the |sin| foot-lift model,
and a median over a gait cycle is smaller. But even at a third of it the far-side
term is ~1.4 m, and the near/far ratio (2.7×) is close to the measured error
ratio (4.4×). `groundpoint.py`'s conclusion — "mount at 6 m or higher and the box
bottom is inside the gate" — is about **total distance**, where the oscillation
averages out. It says nothing about **instantaneous position**, which is what the
event model needs. [`docs/ledgers/assumptions.md:29`](../ledgers/assumptions.md) (A5) already flags the whole curve as
synthetic.

### 4.4 Detector box jitter — measured here for the first time in this geometry

`stats.SIGMA_PX = 1.0` (`model/engine/stats.py:208`) is a guess;
[`docs/ledgers/assumptions.md:68`](../ledgers/assumptions.md) (D1) marks it **UNVERIFIED**, and
[`docs/ledgers/ledger-claims.md:34`](../ledgers/ledger-claims.md) records that the "5.70 px measured" claim was
withdrawn as unreproducible (`measure_sigma.py` matches 0 of 858 GT boxes as
invoked, and its real provenance is a nadir drone clip at 60.2 % recall).

So I measured it on the target geometry: our shipped tracks against 128057's MOT
ground truth, box-bottom to box-bottom, Hungarian-paired, gated at half a GT box
height.

```
@RAN  128057 .c03s vs MOT GT, 120 frames, 2260 matched box-bottoms
  u (along image): median +0.25 px  robust sigma 1.11  p95|.| 3.31
  v (image row)  : median +0.44 px  robust sigma 1.42  p95|.| 3.79
  farthest quartile (small boxes)  n= 563  sigma_v 1.45 px  sigma_u 1.22 px  median GT h 41 px
  nearest quartile (big boxes)     n= 565  sigma_v 1.39 px  sigma_u 1.07 px  median GT h 85 px
```

Three things fall out:

1. **σ ≈ 1.4 px, not 5.7 px.** The shipped `SIGMA_PX = 1.0` is ~1.4× low, not
   5.7× low, on the geometry the product actually targets.
2. **It is flat in depth** (1.45 far vs 1.39 near) — box-bottom noise in *pixels*
   does not care how far away the player is. All the far-side blow-up comes from
   the m/px multiplier, not from the detector getting worse.
3. Caveat: the ≤0.5·h gate excludes badly-localised boxes, so this is a
   **flattering** σ, in exactly the way `model/engine/stats.py:186` warns about.

### 4.5 What is NOT measured

| term | status | cheapest experiment |
|---|---|---|
| **Pitch crown / non-planarity (A3)** | **NOT MEASURED.** No code path detects it | Fit a plane to the 65 118575 keypoints in 3D using the recovered camera, and look at the residual's spatial pattern. If the crown is real the residual is a smooth dome, not noise. ~1 hour, no new data. |
| **True foot-lift distribution on real boxes** | **NOT MEASURED.** `FOOT_LIFT_M = 0.30` is an idealised model constant | Take 128057's GT boxes for one player over 200 frames, project the box bottom, and look at the **spectrum** of the residual against a smoothed path. A peak at ~3 Hz is stride oscillation and its amplitude is the answer. ~2 hours, data already on disk. |
| **`SIGMA_PX` without the matching gate** | **PARTIAL** (§4.4 is gated) | Re-run §4.4 at gate = 1.5·h and 3·h and report the σ curve. 10 minutes. |
| **128057 / 132831 forward accuracy in metres** | **NOT MEASURED.** Only 118575 ships keypoints | `st2_metres` already gives the inverse. For the forward direction on these clips the only reference is paint, and `st2_overlay --check` already does it in px. Converting 3.0 px to metres needs the m/px field, which §4 now has: 3.0 px × 0.567 m/px = **1.70 m at the far touchline**, 3.0 × 0.072 = **0.22 m near**. `@DER`, on the 118575 Jacobian, so treat as an order of magnitude. |
| **Whether the 2.50 m generalises off 118575** | **NOT MEASURED.** One match ships keypoints; `context-pitch-mapping-pivot (not published)` §4 records that 117092, 118575 and 118577 have never been tracked at all | Run `st2_metres` on the other STC sequences that have a usable `H` (7 of 10, per the cond numbers). ~30 min. |
| **k1 against real player positions** | **NOT MEASURED**, and `model/engine/calibrators.py:567-573` says so | Fit k1 on 128057, wrap the ST2 chain in `undistorted()`, re-run `st2_metres`. ~1 hour. This is the only distortion model in the repo that has never been scored where it counts. |

### 4.6 A live drawing bug in the product renderer

`model/validation/st2_overlay.py:106` `densify()` exists because `pitch_model` returns
a touchline as its **two end points**, and on a stitched panorama a straight line
images as a curve, so the chord cuts across it — `@REC` `st2_overlay.py:113-115`:
120 px above the painted touchline at halfway on 128057.

**`model/engine/overlay.py:264` `pitch_lines` has no densify.** It walks
`pitch_model(P)` (`:289`) and projects the raw vertices (`:304-307`). On a
`remap` calibration the frame really is flattened first, so the chord error is
small — but it is the same code path, and `pitch_lines` is what the product uses
as its "calibration check with nothing between you and it" (`:267`). If a
panoramic or bowed calibration ever reaches it, the drawn pitch will be wrong in
a way that looks exactly like a bad homography and is not one.

Fix is one line: reuse the same subdivision. The two functions were deliberately
split (`model/engine/overlay.py:241`) so that `st2_overlay` could share `pitch_model`;
`densify` should move next to it.

---

## 5. Upgrade paths already hinted at in the code

Ranked by (measured value) ÷ (cost). Each row names the code that already exists.

### 5.1 Line-based calibration — the one the pivot doc calls "the current work"

**What exists.** `model/engine/lines.py` is the detection half: `grass_mask` (`:103`) →
`paint_mask` (`:141`) → `segments` (`:155`) → `merge` (`:205`) → `find` (`:243`).
No model, no weights. `model/engine/pancheck.py:378` `refit` total-least-squares-fits a
segment to the hundreds of paint pixels beneath it; `:407` `points_from_lines`
intersects the two line families and emits `"(x, y)"` literal-metre clicks that
`pitch.resolve_landmarks` (`model/engine/pitch.py:104`) already parses.

**What is missing.** The **correspondence** step — deciding which detected line
is the halfway line. `model/engine/lines.py:46-48` says so outright: "that
correspondence step is the hard half and lives above this."

**What it buys.**
- It is the only route to the dense, x-spread landmark set the panorama needs.
  `@REC` `model/engine/pitch.py:86-93`: 21 clicked landmarks give 6.69 m; 42 touchline
  samples at 5.5 m spacing give 0.84 m. The line fit can emit exactly that
  format.
- It removes the operator click step entirely (`model/engine/lines.py:20`).
- It is the prerequisite for the off-pitch gate. ~~`@REC`
  `context-pitch-mapping-pivot (not published)` §2.5 measured 0.393 → **0.417**
  (+0.024 HOTA) from one foot-y band cut~~ — **CORRECTED 2026-09-10:** that was
  the hand-picked `0.55·H` band on n=2, chosen after seeing the answer, and it is
  withdrawn. The gate was then built from geometry (`model/validation/st2_onpitch.py`,
  variant `.c03s_op`) and measures `@V` **+0.0122 mean HOTA over ten sequences,
  7/7 improved, p = 0.008** (`@D` +0.0152 held out) — and `@M` **4/4, p = 0.062
  → DOES NOT SHIP on AssA**, the declared primary metric. Roughly half the
  advertised gain, and it does not clear the repo's own bar on the metric that
  bar names.

**What it costs.** The correspondence solver. `model/engine/lines.py:26-35` already
scoped the geometry: four lines is not the gate, "four lines in two non-parallel
families spanning the frame" is — 0.85 m held out at k=6 with a goal line visible, 20.8 m without.
Estimate: days, not hours.

### 5.2 Densify `model/engine/overlay.py:pitch_lines`

**Cost:** one line. **Buys:** the product's own calibration-check picture stops
lying on any curved frame. See §4.6.

### 5.3 Set `SIGMA_PX` from §4.4 — but re-tune `ACCEL_Q` with it

`@RAN` σ_v = 1.42 px against the shipped 1.0. `model/engine/stats.py:138` records that
`(ACCEL_Q, SIGMA_PX)` is a **one-parameter family** — the gain depends only on
`q/σ` — so changing σ alone is equivalent to changing q, and
`model/engine/stats.py:194` warns that doing it without re-tuning "would just move the
bias". **Cost:** the σ change is free; the re-tune is a sweep. **Buys:** removes
D1 from [`docs/ledgers/assumptions.md`](../ledgers/assumptions.md) and lets D3 (the anisotropic R) finally be
exercised on a real homography.

### 5.4 The bow — read the warning, then do not enable it

`model/engine/calibrators.py:702` `BowWrapper`. `@RAN` (§4.1): 2.50 m → **0.42 m**
leave-one-out, 4.66 → 1.00 on touchlines→rest, 4.77 → 0.52 on the x-band split.
Those are excellent.

**And it is worse against real players.** `@REC` `model/engine/calibrators.py:715-720`:

```
GT y band     players      homography    bow deg 4
0-17 m           715          1.837 m      1.384 m   <- bow wins here
17-34 m         3795          1.370 m      1.839 m
34-51 m         3074          0.752 m      1.472 m
51-68 m         1082          0.595 m      0.950 m
overall         8679          1.010 m      1.507 m
```

`@REC` `:722`: end to end it takes stitched IDF1 from 0.1032 to 0.0898.
The reason is the sampling: 42 of the 65 landmarks sit on two image rows, so the
hold-out selects for touchline accuracy and pays for it where the players are.
`@RAN` confirms the *symptom* independently — `left half -> right` is **44.23 m**
for bow deg 4 against 9.53 m for the plain homography. Extrapolating a clamped
degree-4 polynomial off the end of its data is a cliff (`:812-819`).

**The one thing worth keeping:** the homography's real error against players is
`@REC` **1.010 m**, not the 2.50 m the landmark hold-out implies. Calibration is
~2.5× better than the landmark number suggests — which is also why neither the
recall ceiling nor the speed spikes moved when it was "improved"
(`model/engine/calibrators.py:736-740`).

### 5.5 Score `estimate_k1` against real positions

`model/engine/calibrators.py:567-573` names its own gap: nothing has scored k1
against real player positions. But
**128057 has labels** — ST2's per-frame `loc`, which `st2_metres` already reads.
**Cost:** ~1 hour (fit k1, wrap with `undistorted()`, re-run `st2_metres`).
**Buys:** either a physically-motivated 1-parameter correction that survives the
test the bow failed, or a clean retirement of the idea. Either is worth an hour.

### 5.6 The stated "right answer": the rig's own cylindrical parameters

`model/engine/calibrators.py:416-419` (panoramic) and `:826-827` (bow) both say the
same thing: get the rig's cylindrical parameters, `cv2.remap` the frame flat,
then use ONE homography — "at which point deg=0 and this class becomes a no-op."
`RemapCalibrator` (`:293`) is that answer already built; what is missing is the
tables for a *user's* camera. `:322-327` states the acceptance criterion: it
needs "nothing from the vendor but the remap tables (or a checkerboard, or the
fisheye coefficients)."

**Cost:** a vendor conversation, not engineering. **Buys:** `@RAN` 6.17 m → 2.36 m,
and touchline bow 49.00 → 3.44 px near / 17.43 → 1.54 px far.

### 5.7 Widen the pipeline gate to `remap` — already done, check it stayed done

`app/server/jobs.py:434-441` documents that widening is a one-line change **in three
places** (`load_calibration`, `stats.build_match`, `track.run`) and "do not do one
without the others." `@RAN` I confirmed `app/server/jobs.py:453` now accepts
`("perspective", "remap")` and `model/engine/track.py:826` loads via
`calibrators.load`. **`panoramic` stays refused on purpose** — its global `.H` is
the 6.17 m single-fit its own docstring warns about, and `stats.py` would
differentiate that `H` for measurement noise and produce plausible wrong numbers.

### 5.8 The `sigma_scale` knob, and why it is not a free win

`model/engine/calibrators.py:475-487`: `sigma_scale=1.0` is right for the 21 clustered
operator landmarks (they must borrow across the 36 m gaps) and **wrong** for a
dense set, where borrowing reimposes a global fit — `@REC` on 118575's 42
touchline samples, 5-fold held out, 1.0 gives 3.53 m and 0.25 gives 0.84 m.
`from_keypoints` (`:1051`) switches automatically at `DENSE_LANDMARKS = 40`
(`:1048`).

But `:1064-1074` then tells you not to believe the 0.84 m, and `@RAN` §4.1
confirms it: the dense panoramic fit reads **10.81 m median / 174.93 m max** on
the touchlines→interior split. **This knob is not an upgrade path. It is the
spline failure in a different hat.**

---

## 6. What I could NOT verify

Listed so nobody quotes them as measured.

1. **`model/engine/stats.py:116` "0.45 m/px near, 2.07 m/px far, vs 0.06 m/px
   across-pitch."** `@RAN` on the shipped 118575 calibration on its own
   11878×870 canvas I get **0.072 m/px near / 0.567 m/px far (major), 0.011 /
   0.049 (minor)**. Scaling to the vendor 4096×1080 canvas (÷2.9) gives 0.209 /
   1.644. Neither matches 0.45 / 2.07. The same pair appears at
   `model/engine/track.py:673` and [`docs/ledgers/assumptions.md:70`](../ledgers/assumptions.md). **Which canvas and which
   calibration produced 0.45/2.07 is not stated anywhere.** Do not quote it until
   that is pinned.

2. **[`docs/ledgers/assumptions.md:27`](../ledgers/assumptions.md) (A4) "median 5.35 m, p95 15.2 m, max 17.0 m" for a
   single homography on the raw panorama.** `@RAN` `calibrators.py --real` gives
   **6.17 / 11.72 / 13.61** (fit residual) and 6.57 / 12.62 / 14.89
   (leave-one-out), and `model/engine/calibrators.py:301` says 6.17 / 11.7.
   [`docs/ledgers/ledger-claims.md:54`](../ledgers/ledger-claims.md) quotes **both** numbers in the same row as if they
   were the same measurement. They are not. Two of the three
   (`README.md`-sourced 5.35, archived-stack-doc-sourced, not published) are unreproduced.

3. **`RemapCalibrator`'s "2.36 m is mostly the keypoints, not the geometry."**
   The Monte-Carlo evidence is `@RAN` real (σ=2.5 px → 1.26 m), but it is an
   *upper bound argument*, not a decomposition. Nobody has measured the
   dataset's actual annotation σ; 2.5 px is inferred from the post-remap bow.

4. **`model/engine/calibrators.py:715-720`, the bow-vs-ground-truth table (1.010 m
   overall).** `@REC` only. It needs the 118575 GT player positions and a full
   re-projection; I verified the *related* symptom (the left→right split blowing
   up to 44.23 m) but not the table itself. The 1.010 m is the single most
   consequential number in this document and it is `@REC`, not `@RAN`.

5. **`model/engine/calcheck.py:59-61`, "implied 1.76 m over 31 121 real detections."**
   `@REC` only — needs a 189 MB detection cache. The self-check passes `@RAN`
   but it is circular by its own admission (`model/engine/calcheck.py:318-323`).

6. **Whether any operator has ever produced a calibration through the UI on real
   footage.** Every calibration on disk that I inspected came from
   `calibrators.py --write-calib`, not from clicks. `@RAN` the shipped
   `data/jobs/118575_full_1st/calibration.json` is `projection_type: "remap"`
   with 65 points, `maps_crop: [390, 690]`, `maps_scale: 2.9`,
   `rms_error_m: 2.845` — i.e. the *dataset's* keypoints, not clicks. **The 21-
   landmark operator path's end-to-end accuracy on real footage has never been
   measured.** The nearest thing is `@REC` `model/engine/pitch.py:86-88`: 21 named
   landmarks reach 6.69 m held out on 118575, "no better than a single
   homography".

7. **Pitch non-planarity, real foot-lift amplitude, and `SIGMA_PX` without the
   matching gate.** See the table in §4.5 for the cheapest experiment for each.

---

## 7. One-paragraph summary for someone who reads nothing else

The product solves **one 3×3 homography from up to 21 operator clicks**
(`model/engine/pitch.py:122`), maps image px → pitch metres directly, and has no
distortion model enabled. Three richer models exist and are all off: piecewise
(`PanoramicCalibrator`, refused by the pipeline at `app/server/jobs.py:453`), radial
(`UndistortWrapper`, never wired), and a row-bow (`BowWrapper`, off because it
wins on landmarks and **loses** against real player positions). The stitched-
panorama path is a different chain entirely — vendor `H` to a remapped canvas,
then a `mapx` lookup — and it is inverted by KD-tree, not by matrix inverse,
because no closed form exists. **The forward direction measures 3.0 px against
paint; the inverse measures 1.26 m median and 3.23 m in the far quartile, from
ground-truth boxes with zero detector error.** The far-side blow-up is 14.7×
depth compression multiplying three things: the ground-point convention (up to
4.15 m at peak foot lift, the largest single term), the model residual (2.50 m
held out), and a 1.00 m hard floor from pixel quantisation. Detector jitter is
σ ≈ 1.4 px and flat in depth — it is not the constraint. The upgrade path with
the most already built is **line-based calibration**: `model/engine/lines.py` detects,
`model/engine/pancheck.py:407` converts lines to dense literal-metre landmarks, and
only the correspondence step is missing.
