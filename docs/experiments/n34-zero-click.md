# N34: zero-click calibration — the camera found from the paint, no dots

**2026-09-17, branch `n30-operator-clicks`.** Code: [`model/validation/n34_autocal.py`](../../model/validation/n34_autocal.py),
on top of [`model/validation/n31_linereg.py`](../../model/validation/n31_linereg.py). Follows
[`n31-line-registration.md`](n31-line-registration.md) (the registration, and why the operator's dots
were the largest error) and `n33-heights.md` with its review
`audit-n33.md` (the square-pixel prior; what the feed's 4% is). The product goal it
serves: upload the match and it calibrates itself. **Development, ONE camera install, no ground
truth used to build or to select anything; not yet independently reviewed.**

> **CORRECTED 2026-09-28.** `model/validation/out/stc2025/n34_autocal.json` is this experiment's record and predates the length fit (`autocal._settle`, section 6); the shipped code no longer reproduces it. Anything that wants "the camera the product runs" reads [`model/validation/fixtures/calib/golden.json`](../../model/validation/fixtures/calib/golden.json) (re-frozen with current code, guarded by `model/validation/check.py calib`), not this file.

## 1. The idea

The operator's 21 dots only ever told the registration roughly where the camera was. A fixed
panoramic rig does not need telling: it hangs a few metres up, a few metres behind a touchline, near
halfway, looking across. That is a small box, and the pitch is a known drawing.

1. **Frame.** A 40-frame median: players gone, paint left.
2. **Paint, by shape.** A multi-scale Hessian ridge filter on min(B,G,R), kept only inside the
   largest grass region, in long thin strokes that are lines or shallow arcs, and cut off below the
   near touchline (the longest stroke) - a club's name painted beyond it would otherwise pull the
   search toward a camera lying on the grass. Colour cannot do this job: a line 1-3 px wide stays
   GREEN, because it blends with the turf, so a whiteness test keeps the near touchline and the
   halfway line and loses the circle and both boxes.
3. **Coarse search** over the mount - field of view, tilt, height, distance behind the near
   touchline, offset along it, pan: 21,875 physical spherical cameras with square angular pixels.
   The score is two-way: DRAWING -> PAINT in pixels (is every line of the pitch on paint?) and
   PAINT -> DRAWING in metres through the camera's inverse (is the paint on the pitch explained by a
   line?). One way alone is won by a degenerate camera that folds the drawing onto one stroke. The
   grid is scored against a 200 px basin, because at grid spacing the node beside the truth otherwise
   loses to a stranger.
4. **Polish and register.** The best 40 nodes are polished (Nelder-Mead, bounded mount, roll and
   equator row freed), the best 6 go to `n31_linereg.register` with NO dots and the square-pixel
   prior, at 25, 12, 8, 5, 3 px, repeated while it still gains paint.
5. **Acceptance without ground truth:** N31's refusal rule (paint hits, line RMS), camera 2-20 m up,
   and every family of the drawing - touchlines, goal lines, halfway, boxes, circle - on paint.
6. **One camera per install.** A match the search refuses starts again from the camera of the other
   matches of its install, under the same registration and the same acceptance.

Two defects of the first attempt, both silent, both found by asking whether the TRUE camera scored
well: the paint mask was N31's ridge map, which fires on grass texture and covered 61% of the frame,
so every camera scored zero; and the distance transform was taken on a mask shrunk by area
averaging, which deletes a 1-2 px stroke, so every camera scored the truncation.

## 2. What it does

    @M direct 6,via_install 1,refused 1,0.12,0.16,0.43 :: .venv/bin/python -c "import json,sys,numpy as np;sys.path[:0]=['model/validation','model/engine'];import n31_linereg as L;a=json.load(open('model/validation/out/stc2025/n34_autocal.json'));b=json.load(open(L.OUT_B));g=np.stack(np.meshgrid(np.linspace(3,102,12),np.linspace(3,65,8)),-1).reshape(-1,2);ok=[q for q in a if a[q]['camera']];d={q:np.linalg.norm(L._inv(np.array(a[q]['camera']),L._fwd(np.array(b[q]['camera']),g))-g,axis=1) for q in ok};m=[float(np.median(v)) for v in d.values()];print('direct',sum(1 for q in ok if a[q].get('via')!='install'),'via_install',sum(1 for q in ok if a[q].get('via')=='install'),'refused',sum(1 for q in a if not a[q]['camera']),round(min(m),2),round(max(m),2),round(max(float(v.max()) for v in d.values()),2))"
       held fixed: the eight N30 clips' 40-frame medians and their 105 x 68 m pitch size - NOT their
       dots; n34_autocal.py and n31_linereg.py as committed with this page. Clips calibrated from
       nothing, clips calibrated from their install, clips refused; then, against the CLICK-STARTED
       registration of the same clip (n31b), the same pixels read through both cameras over a 12 x 8
       pitch grid: lowest and highest per-clip median metres, and the largest single disagreement.
       No ground truth read. ONE install.

| clip | zero-click | camera centre (m), f (px/rad) | paint hits of 1,351 | vs click-started |
|---|---|---|---|---|
| 117093 | from nothing | 51.23, 83.01, 7.52 · 1512 | 843 | 0.12 m median, 0.32 max |
| 118575 | from nothing | 51.22, 83.07, 7.55 · 1514 | 802 | 0.16 m, 0.43 |
| 118576 | from nothing | 51.23, 83.05, 7.54 · 1514 | 828 | 0.12 m, 0.33 |
| 118577 | from nothing | 51.22, 83.06, 7.55 · 1514 | 815 | 0.13 m, 0.35 |
| 118578 | from nothing | 51.22, 83.01, 7.52 · 1513 | 822 | 0.16 m, 0.42 |
| 128058 | from nothing | 51.23, 82.99, 7.51 · 1512 | 870 | 0.12 m, 0.27 |
| 128057 | **refused** from nothing (292 hits, a 4.8 m camera); accepted from its install | 51.23, 83.02, 7.52 · 1513 | 849 | 0.14 m, 0.30 |
| 117092 | **refused**: no green pitch to find | - | - | - |

Seven matches, no clicks, the same physical camera each time - 7.5 m up, 15 m behind the near
touchline, a metre left of halfway - and a map within 0.12-0.16 m of the one that started from 21
dots. The search takes about 40 s a clip on a laptop CPU. On a synthetic panorama through a mount the
grid does not contain, the recovered map is 0.008 m from the truth at the median (`--demo` asserts
it, and asserts that a frame with no pitch, or a grey one, is refused).

**Against the feed** - a diagnostic, since ST2's feed carries its own ~4% offset away from the
camera (`audit-n33.md` finding 4): the zero-click cameras read exactly as N31's
click-started ones did.
`@C 2.28 m pooled median; 0.36 m median ACROSS the camera ray; -1.8% to -4.2% of range ALONG it :: inline script, N34 session, n34_autocal.json cameras on n31_score's feet frames, seven clips`
Across the ray, where the feed has no such bias and is quantised to about a metre, the map is
0.29-0.54 m from it at every range.

## 3. What it does not do yet

- **117092 - night, frost, a two-lens rig with a visible seam** (its median frame shows a step where
  the halves meet, and no penalty areas at all). The pitch REGION is found by greenness; there is none.
  Refused upstream, correctly. It needs a pitch segmenter that is not a colour test, and the seam says
  this rig is NOT one spherical camera.
- **Unknown pitch size.** Every clip was given 105 x 68 m. Length is pinned by the paint; WIDTH, which
  lies along the camera ray, is not (`audit-n33.md` finding 6). A new venue needs its
  size from the club or a wider search.
- **The far touchline** is as weak as it was: faint paint a few pixels from white boards. 1-3 m there.
- **One install.** Nothing here has seen a different vendor, mount, or a pitch without painted boxes.
- ~~It lives in `validation/`~~ **In the engine since 2026-09-17**: [`model/engine/autocal.py`](../../model/engine/autocal.py),
  `calibrators.SphericalCamera`, and `app/server/jobs.auto_calibrate` - see §5.

## 4. Next

1. The research commissioned beside this (zero-click initialisation on the sphere - a Hough transform
   over great circles, rotation from two vanishing directions - and what vendors do) decides whether
   the brute-force search stays or is replaced by the closed-form route.
2. Into the engine: `median frame -> autocal -> per-install cache -> Camera.to_pitch / jacobian`, with
   the refusal surfacing as "we could not calibrate this match" and the operator's dots as the
   fallback, not the default.
3. A second rig.

## 5. In the product

`app/server/jobs._pipeline` now tries, in order: **no clicks** (`auto_calibrate`: this match's own paint,
started from the rig's last automatic camera when the same club, ground and frame size have one), then
the older copy of a hand-clicked calibration from the same rig, then a human. Only footage at least
1.9 times as wide as it is tall is tried - the model is one spherical camera and a 16:9 single-lens
feed is a pinhole. `FUTTRAFFIC_AUTOCAL=0` turns it off.

What it writes is `projection_type: "spherical"` (`calibrators.SphericalCamera`): positions from its own
inverse map, metres per pixel from its Jacobian (which `stats.kalman_ground` already reads), the
camera's place from `camera_centre_m` (identity's facing-away gate), the pitch's outline in the image
from `pitch_polygon_px` - walked along the curved touchlines, which `track.make_prior` now prefers - and
an `H` that is only a stand-in for code that wants "some 3 x 3". It assumes 105 x 68 m and says so
(`auto.pitch_assumed`); a refusal is recorded on the job with its reason and the job parks for the
operator's dots exactly as before. The setup screen shows the found pitch drawn on the frame, and a
hand calibration posted afterwards replaces it.

Through the real job code, on an isolated data directory (an inline script in the N34 session, cited
not re-runnable): a first
match calibrated from nothing in 43 s; a second at the same ground - the one the search refuses on its
own - from its install in 11 s; the night match refused, reason attached. Self-checks:
`model/engine/autocal.py`, `model/engine/calibrators.py`, `model/engine/track.py`, `app/server/jobs.py`. The validation
scripts import the engine's code rather than keep a copy, and reproduce their stored cameras to 0.0.

## 6. The pitch's size: length measured, width asked for

The 105 x 68 m assumption is gone for LENGTH. The penalty areas, goal areas and centre circle are
regulation, so only one pitch length lets the circle AND the boxes AND the goal lines sit on paint at
once. It cannot be refined into (a metre of length moves a goal line some twenty pixels, outside any
search radius) and it cannot be read off one fit (at the wrong length the robust fit bends the CAMERA
to reach the boxes and gives up the circle - 3 circle samples of 115 on paint, on a synthetic 110.5 m
pitch fitted as 105). So `autocal._settle` scans it - 90 to 120 m, registering at each, growing the
pitch about its centre spot - keeps the length where the most of the drawing finds paint, and refines
it jointly with the camera. `model/engine/autocal.py`'s self-check draws a 110.5 m pitch, searches it as 105,
and requires 110.5 back.

`@C 105.01,105.25,0.035 :: inline script, N34 session: autocal._settle from each of the seven same-install clips' zero-click cameras - lowest and highest measured length in metres, and the fit's own 1-sigma; no ground truth read`

Seven matches, seven independent measurements, a quarter of a metre apart: this pitch is 105.1 m long.

WIDTH is not measured and cannot be from here: it lies along the camera's line of sight
(`audit-n33.md` finding 6). It is the one number a venue owes us.
`POST /api/matches/{id}/pitch {"width": m}` records it on the job, every later match at the same club
and ground takes it from there (`jobs.venue_pitch`), an automatic calibration made under the old width
is dropped and the match re-queued, and the setup screen shows "Length 105.2 m, measured - Width 68 m,
ASSUMED" with a box to type it. Outside sources for it (OpenStreetMap and others) are being researched
separately.

## 7. End to end: the product under the zero-click calibration

[`model/validation/n35_end_to_end.py`](../../model/validation/n35_end_to_end.py) - a DIAGNOSTIC declared before it
ran: clip 118575, one configuration per arm, nothing tuned, one install. SHIP is the product as it
ships (N21's `.p21id`, the clicked homography); AUTO is the same path - fresh detection, the scene
prior drawn from the spherical camera's own pitch outline, the tracker, the metric stitch, identity -
under the camera found with no clicks.

    @M 8.0,2.33,1.1802,0.9519,1.117,0.947,2.317,0.828 :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n35_e2e.json'));print(*[r[a][k] for k in ('pos_median_m',) for a in ('SHIP','AUTO')],r['SHIP']['slope']['all'],r['AUTO']['slope']['all'],r['SHIP']['distance_over_floor'],r['AUTO']['distance_over_floor'],r['SHIP']['hsr_over_floor'],r['AUTO']['hsr_over_floor'])"
       held fixed: 118575 (ST2 train split), the product config, stride-2 frames; n=1 per arm. SHIP then
       AUTO: median metres from our own boxes' feet to the feed; chord slope; distance per covered metre
       over its floor; high-speed running over its floor. Rows with no IoU owner are excluded:
       9.5% (SHIP) and 10.9% (AUTO). ONE clip, ONE install.

| 118575 | position median / p90 | slope all (x, y) | distance over floor | HSR over floor | ids |
|---|---|---|---|---|---|
| SHIP, clicked homography | 8.00 / 9.60 m | 1.180 (1.212, 1.136) | 1.117 | 2.32 | 48 |
| AUTO, zero clicks | 2.33 / 3.82 m | 0.952 (0.949, 0.957) | 0.947 | 0.83 | 44 |

The floor is TRUE positions through the same fragments, so "over floor" isolates the map. The feed
itself reads about 4% long (`audit-n33.md` finding 4), so a CORRECT map is expected at
about 0.96 on slope, a few percent under 1 on distance, and - HSR's elasticity to scale being about 4.5 -
around 0.83 on HSR. AUTO reads 0.952, 0.947 and 0.83. That agreement is an inference from a bias whose
owner is "probably the feed", not a measurement of truth; what is measured is that the shipped map reads
movement 18% long and unevenly (x 21%, y 14%), and the zero-click one reads it 5% under the feed and
evenly. The fragments' own floor also fell (distance 1.395 to 1.280, HSR 1.61 to 1.36, 48 ids to 44): a
stitch that works in metres works better when the metres are right.

### The other six clips (extension declared at `8d13bbc` before it ran: same configuration, nothing tuned)

`model/validation/n35_end_to_end.py --seq <clip> --cal / --build`, then `--summary`. AUTO's calibration on each clip is the
camera `n34_autocal.py --fit` found from that clip's own paint. Rows with no IoU owner are excluded from every figure:
6-15% (SHIP), 10-11% (AUTO). All seven are ONE camera on ONE pitch - seven samples of the players, not seven tests
of the calibration.

| clip | position median m | distance over floor | HSR over floor | ids | slope all (far, near) |
|---|---|---|---|---|---|
| 118575 | 8.0 -> 2.33 | 1.117 -> 0.947 | 2.317 -> 0.828 | 48 -> 44 | 1.180 (1.033, 1.277) -> 0.952 (0.947, 0.954) |
| 117093 | 7.72 -> 2.46 | 1.0818 -> 0.9437 | 2.299 -> 0.695 | 73 -> 61 | 1.176 (1.065, 1.244) -> 0.953 (0.945, 0.959) |
| 118576 | 7.75 -> 2.04 | 0.9508 -> 0.939 | 1.288 -> 0.833 | 56 -> 51 | 1.041 (1.068, 1.029) -> 0.973 (0.952, 0.985) |
| 118577 | 7.44 -> 2.33 | 1.0436 -> 0.9165 | 1.43 -> 0.792 | 52 -> 53 | 1.083 (1.048, 1.133) -> 0.945 (0.946, 0.946) |
| 118578 | 6.49 -> 2.45 | 1.0243 -> 0.9513 | 1.626 -> 0.831 | 56 -> 54 | 1.111 (1.013, 1.151) -> 0.950 (0.949, 0.956) |
| 128057 | 7.0 -> 2.04 | 1.0669 -> 0.912 | 2.131 -> 0.617 | 57 -> 58 | 1.134 (1.049, 1.177) -> 0.937 (0.943, 0.933) |
| 128058 | 7.63 -> 2.17 | 1.0533 -> 0.942 | 2.281 -> 0.751 | 60 -> 55 | 1.120 (1.026, 1.173) -> 0.961 (0.964, 0.962) |
| **median** | 7.630 -> 2.330 | 1.053 -> 0.942 | 2.131 -> 0.792 | | 1.120 -> 0.952 |

    @M 7.72,2.46,7.75,2.04,7.44,6.49,2.45,7.0,7.63,2.17,1.0818,0.9437,0.9508,0.939,1.0436,0.9165,1.0243,0.9513,1.0669,0.912,1.0533,0.942 :: .venv/bin/python model/validation/n35_end_to_end.py --summary
    @M 2.299,0.695,1.288,0.833,1.43,0.792,1.626,0.831,2.131,0.617,2.281,0.751,7.630,2.330,1.053,1.120,0.952 :: .venv/bin/python model/validation/n35_end_to_end.py --summary

118575 was not a lucky clip. On every one of the seven the zero-click map reads position three times nearer the
feed, movement 3-6% under the feed and THE SAME far and near (the shipped map's far and near slopes differ by up
to 0.24; the zero-click map's by at most 0.033), and distance 5-9% under its floor. HSR swings from a 1.3-2.3x over-read to a 0.62-0.83x
under-read: nearer 1 on all seven, still the least trustworthy figure, and the one a wrong feed scale moves most.
What these clips cannot say is how much of the remaining 5% is the feed's
([`n36-far-touchline.md`](n36-far-touchline.md) section 4: the along-ray gap is uniform near and far and survived the
far-touchline fix).

**Not done:** width from outside (`research-pitch-size-sources.md`: OSM has no usable
outline for this ground), a second rig, an independent review of N34, N35 and N36.

## 8. Review of 2026-09-17: what the numbers above do and do not say

Three independent agents reviewed N34, N35 and N36 (protocol, code, status). They reproduced every number in
section 7 from the track files. Their corrections, which stand over anything above that reads differently:

- **"Over floor" is not "against the feed".** Section 7's headline ratios divide by a tracking floor. Raw, against the
  feed, the product under the zero-click camera still reads distance per covered metre 1.120-1.221 (median 1.183) and
  high-speed running 1.110-2.026 (median 1.199); under the shipped camera 1.226-1.558 and 1.935-4.213. The map fix
  removed the map's share of the over-read; what is left is the fragments', and more calibration work will not move it.
  "Distance under-reads" and "HSR under-reads" are not things to repeat.

    @M 1.120,1.221,1.183,1.110,2.026,1.199,1.226,1.558,1.935,4.213 :: .venv/bin/python model/validation/n35_raw.py

- **The 8 m to 2.3 m is entirely the map.** Crossing the pixel sets with the two cameras, SHIP's own tracks through
  the zero-click camera read 2.33 m and slope 0.952 on 118575, and the zero-click tracks through the homography 8.00 m
  and 1.18 (the review's run, not a committed check). The arms share one pipeline configuration (same detector, conf,
  stride, prior, stitch, identity).
- **The remaining ~2.3 m is not shown to be the feed's.** A pure 4% scale about the camera's foot accounts for 2.0-2.3 m
  of it, and 0.7-0.9 m is left (0.4 m of that is the feed's own 1 m grid). But the along-ray gap grows with range
  (about -1.7% within 25 m, -4.2% beyond 70 m: [n36-far-touchline.md](n36-far-touchline.md) section 4), which a scaled
  feed would not do and a camera-height or vertical-law error would, and our residual is largest in the far-right
  corner where the right goal line has 0-13 of 136 samples on paint on every clip. Ownership needs the restart
  protocol audit-n33.md asked for; it has not run.
- **From nothing is 6 of 7, not 7 of 7.** 128057 calibrated only through the install fallback; from nothing a false
  camera 4.4-4.8 m off won its basin. On two other clips such false cameras polish to within 0.02 of the true objective
  and score 274-332 hits against the 300 gate. `accept()` pools line families by name, so "every family on paint" is
  met with the right goal line at 0 hits. The search grid has nodes near this install's true mount and the lettering
  cut was built for this ground: the constants were shaped on the one install they are tested on. The only other rig
  (117092) was refused before the method started, so nothing is known about a second camera, ground, vendor, night or
  worn paint.
- **HSR over floor is a dubious ratio.** HSR's elasticity to scale is 3.9-6.6 per clip here (median 5.1), not one
  number; per clip the zero-click HSR sits 0.84-1.14 of a pure-scale expectation. That is +-15% of unexplained scatter,
  not a systematic under-read. (The review's numbers; not a committed check.)
- The seven .p35 calibration files used a 105.0 m pitch, not the 105.1 m section 6 measures (0.1%).
- Section 7's own measured tags re-run a printer of stored JSON; they show the files exist, not that the scoring reproduces.

**Safe to say, always with "one install, seven four-minute clips, against ST2's feed":** the camera was found from the
paint with no clicks on seven daylight clips (six from nothing) and refused the night two-lens rig; positions read
2.0-2.5 m median from the feed and 0.3 m across the ray; movement scale within 4-6% of the feed and even near and far;
the shipped clicked homography reads 6.5-8 m and movement 4-18% long. **Not to say:** "0.12-0.16 m" as an accuracy (it
is the convergence of one registration from two starts); "the 2.3 m is the feed's"; anything implying a second rig;
that distance or sprinting now under-read.
