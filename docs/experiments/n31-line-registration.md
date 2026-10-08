# N31: the error was in the dots — register the camera to the painted lines

**2026-09-17, branch `n30-operator-clicks`.** Follows `n30-operator-maps.md`
and its review `audit-n30.md`. Code:
[`model/validation/n31_scale.py`](../../model/validation/n31_scale.py) (scale validator),
[`model/validation/n31_linereg.py`](../../model/validation/n31_linereg.py) (line registration),
[`model/validation/n31_score.py`](../../model/validation/n31_score.py) (the scoring protocol, committed before it
ran). Research behind it: `research-unstitching-panorama.md`
(the spherical family; Carr, Sheikh & Matthews' project-and-search) and
`research-gp-residual-maps.md`.

§1–§3 were written BEFORE the line-registered cameras were scored against any ground truth. §4 is
the score, appended after.

---

## 1. What the scale validator said

N30's review asked why every map, ST2's own chain included, reads movement several percent short of
the feed. [`n31_scale.py`](../../model/validation/n31_scale.py) splits N27's chord slope by pitch axis and
by region, on the maps and rows N30 already scored — a diagnostic, nothing selected.

    @M 0.9429,0.9543,0.9275,0.9366,0.9479,0.945,0.9325,0.9421 :: .venv/bin/python -c "import json;p=json.load(open('model/validation/out/stc2025/n31_scale.json'))['pooled']['SPH10'];print(*[p[k] for k in ('all','x','y','far','near','left','mid','right')])"
       held fixed: N30's frozen clicks and its SPH10 arm, `p21idn30` rows, stride 2, 4 s chords with
       at least 5 m of true movement; pooled over 8 clips. `oracle`-style drop: rows with no IoU
       owner are excluded, 6.4% to 15.5% per clip (`drop_share` in the same file).

The spherical map's under-read is the SAME number in both axes, both halves and all three thirds.
The fisheye's is not (x 0.923, y 0.968), the spline's is not (left 1.053, mid 0.863), and the
homography's runs from 0.848 to 1.335 (`pooled` in `n31_scale.json`). A uniform factor is not a
projection-family error, not a crowned pitch (that flips sign near to far) and not a pan-angle
effect. Three explanations were then tested and two fell:

- **Noise in the feed diluting the regression:** no. The forward and reverse slopes bracket the
  same 0.93–0.95 at every chord length from 5 to 30 m, and it does not rise on long chords.
  `@C 0.9348,0.9564 at 5-10 m; 0.9087,0.9123 at 20-30 m :: inline script, N31 session, FISH11 exports on 118575, 118577, 128058`
- **The pitch is not 105 x 68:** no. Pooling the seven same-install clips' clicks under the
  spherical form and freeing the size gives a LARGER pitch, which would push the slope the other way.
  `@C 108.42 +- 0.59, 68.94 +- 0.17 :: inline script, N31 session, equirect10 pooled over 7 clips with length and width free`
- **The feed is a magnified copy of our map about one centre:** yes, as a description. Fitting
  feed = C + s (ours - C) per clip takes the median error at feet from 2.5–3.7 m to 1.1–1.6 m on all
  seven same-install clips, with s between 1.048 and 1.085 and C on the camera's side of the pitch.
  `@C 1.0480 to 1.0847; 1.11 to 1.62 m after :: inline script, N31 session, FISH11 exports, N30's feet frames`
  But the feed's own density of play stops at the far touchline, not 4–5 m beyond it, so it is
  OUR maps that pull far players toward the camera, not the feed that pushes them away.

## 2. Why: the dots

A map that pulls the far side in has a far touchline in the wrong place. Looking at 118575's frame:
`half_b` is in the stand; the far corner dots sit at the top of the flag poles, not their feet; the
far edge of the centre circle is 8 px off the paint. Read carefully off 8x ruled crops of a
40-frame median (players gone), 19 of the 21 landmarks could be re-placed
([`model/validation/n31_reread_118575.json`](../../model/validation/n31_reread_118575.json); two left-box points
were unreadable among other sports' markings):

    @M n 19,6.5,19.3,9.2,5.9 :: .venv/bin/python -c "import json,numpy as np;a=json.load(open('model/validation/n31_reread_118575.json'))['points'];b=json.load(open('model/validation/n30_clicks/118575.calibration.json'))['points'];d=np.array([np.hypot(a[n][0]-b[n][0],a[n][1]-b[n][1]) for n in a]);far=np.array([a[n][1]<270 for n in a]);print('n',len(d),round(float(np.median(d)),1),round(float(d.max()),1),round(float(np.median(d[far])),1),round(float(np.median(d[~far])),1))"
       landmarks re-read, then px from the operator's dot to the re-read position: median, max,
       median on the far side (image row < 270), median on the near side. The re-reading is one
       person's, good to about 2 px, and is not ground truth.

Nine pixels on the far side is 4–5 m: a pixel there is about half a metre
(`far-side-jitter-is-camera-placement-not-calibration` (internal note)). On those re-read
landmarks the families separate cleanly, with no ground truth involved:

`@C 3.2 px, 0.30 m median; 7.3 px, 1.57 m median :: inline script, N31 session: equirect10 then fisheye11 fitted to the 19 re-read landmarks, fit RMS and the landmarks' own error in metres`

against 6.8 px / 1.22 m and 9.9 px / 2.29 m on the operator's dots for the same 19. Two things follow.

- **The family is spherical**, as `research-unstitching-panorama.md`
  found from line collinearity. N30 could not see it because both families were fitted to the same
  displaced dots.
- **The dots repeat to 3 px across matches because the screen's guesses repeat, not because they are
  right.** The calibration screen seeds every landmark from the current HOMOGRAPHY fit, the operator
  accepts a guess with a touch, and a 4096-wide frame shown about 1400 wide makes 9 px look like 3.
  A guess from a map that is metres wrong on a panorama, accepted, is that map's error saved as the
  operator's evidence. "Systematic model error" in
  `n30-operator-maps.md` §3 was partly this, as the GP report warned:
  operator bias and model defect look identical from the dots alone.

**The one development probe that read ground truth, disclosed.** On 118575 only, on N30's feet
frames, equirect10 fitted to the re-read landmarks:

`@C 3.51 -> 1.72 m median at feet, p90 5.52 -> 3.02, slope 0.927 -> 0.962 :: inline script, N31 session; operator dots then re-read landmarks; n=3 forms on top of N30's 5; ST2's chain reads 1.20 m on the same feet`

Half the error at feet, and half the under-read, were the dots. One clip, one annotator.

## 3. The fix that does not need a careful operator

No operator places a dot to a pixel on that screen, and none should have to: the pitch has thousands
of painted pixels and 21 dots. [`n31_linereg.py`](../../model/validation/n31_linereg.py): a median frame, a
ridge map of the paint, a spherical camera from the dots under a robust loss, then project the
pitch's lines and circle, search along each sample's normal for the paint, and refit on the normal
component - at 12, 8, 5 and 3 px. It reads no ground truth. It refuses, and says so, when too few
samples find paint or the lines do not close.

    @M ok 7,1122,1186,0.64,0.68,117092,187 :: .venv/bin/python -c "import json,numpy as np;r=json.load(open('model/validation/out/stc2025/n31_linereg.json'));ok=[q for q in r if not r[q]['refused']];print('ok',len(ok),min(r[q]['hits'] for q in ok),max(r[q]['hits'] for q in ok),round(min(r[q]['line_rms_px'] for q in ok),2),round(max(r[q]['line_rms_px'] for q in ok),2),[q for q in r if r[q]['refused']],r['117092']['hits'])"
       held fixed: N30's frozen clicks as the start, 40-frame median per clip, n31_linereg.py at
       7386fc5 with its constants as committed; of 1,307 model samples. Clips registered, fewest and
       most paint hits, lowest and highest line RMS px; the clip refused, and its hits.

Three checks that need no ground truth. All three are cited from inline scripts run in the N31
session, not re-runnable tags:

- on a synthetic panorama with far dots pushed 10–18 px up the image, the click-only camera is
  2.28 m out at the median and the registered one 0.00 m (`--demo` asserts it);
- against the hand re-read landmarks on 118575 - independent of the operator's dots and of this
  code - the registered camera lands at a median of 3.1 px and 0.37 m, the click-only one at 5.5 px
  and 1.70 m;
- the seven same-install cameras, each registered from a different match, agree on a pitch grid to
  a median of 0.9 px (2.1 px click-only), and put the camera at the same point to 0.2 m.

**117092 is refused** - 187 hits. Its dots start the camera too far from the paint for a 12 px
search. It is the other rig, and nothing here is evidence about it.

## 4. The score

Protocol and rule in [`n31_score.py`](../../model/validation/n31_score.py)'s docstring, committed in `3e41d49`
before the run. **NOT SUPPORTED: L1, L3 and L4 pass, L2 fails.**

    @M 2.37,3.18,2.92,1.41,0.9506,0.9351,1 of 7,NOT SUPPORTED :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n31_score.json'));p=r['pooled'];print(*[round(p[a]['median'],2) for a in ('LINE','SPH10','FISH11','CHAIN')],r['pooled_slope']['LINE']['all'],round(r['signed_distance_over_floor']['LINE'],4),r['verdict']['near_chain'],r['verdict']['verdict'])"
       held fixed: N30's frozen clicks, the line-registered cameras of n31_linereg.py at 7386fc5, MOT
       box feet on frames with f % 20 == 10 against the feed, `p21idn30` rows for scale and distance
       (6.4% to 15.5% of rows per clip have no IoU owner and are excluded); pooled over the SEVEN
       clips LINE is scored on - ONE camera install; n=1 new configuration beside three of N30's.
       Pooled median m at feet for LINE, SPH10, FISH11 and ST2's chain; LINE's pooled slope; its
       signed distance over the floor; clips within 0.5 m of the chain; the verdict.

| 7 clips pooled | LINE | SPH10 | FISH11 | SHIP | CHAIN |
|---|---|---|---|---|---|
| median m at feet | 2.37 | 3.18 | 2.92 | 7.12 | 1.41 |
| p90 | 3.74 | 5.03 | 5.87 | 9.23 | 6.48 |
| far-half median | 2.95 | 3.99 | 3.35 | 6.89 | 1.82 |
| slope, all | 0.9506 | 0.9329 | 0.9309 | 1.1255 | 0.9543 |
| signed distance over floor | 0.9351 | 0.9141 | 0.9223 | 1.0490 | - |
| HSR at matched scale over floor | 0.981 | 0.971 | 0.992 | 1.157 | - |

- **L1 passes:** LINE is below both click-only camera maps on all seven clips (2.26–2.70 m against
  2.78–3.65 and 2.45–3.60).
- **L2 fails, 1 of 7:** it is within 0.5 m of ST2's chain only on 118578, where the chain itself
  reads 2.64 m. Elsewhere the chain reads 1.13–1.52 m and LINE 2.28–2.70. It is also short of the
  hand re-read fit on 118575 (1.72 m, §2).
- **L3 passes:** pooled slope nearer 1 than either rival, and uniform - x 0.9596, y 0.9391, far
  0.9489, near 0.9536. It now reads what ST2's own chain reads (0.9543): the remaining ~5% is common
  to every landmark-anchored map, the dataset authors' included, so it is the reference's or the
  box-foot point's, not this camera's.
- **L4 passes:** signed distance 0.9351 of the floor against 0.9141 and 0.9223 - still short.
- **117092 refused**, scored by nothing.
- At matched scale every camera-family map reads HSR within 3% of what true positions read through
  the same fragments; the homography reads 16% over. Reported, not ruled.

**Why L2 failed, found without ground truth and before the run finished** (three clips' medians had
printed). Against the hand re-read landmarks on 118575 the registered camera's FAR landmarks sit
3–5 px higher in the image - `half_b` −4.7, `corner_br` −3.9, `circle_b` −3.4 px - and the same
pixels read through the two cameras differ by 1.1 m in depth at the far touchline, falling smoothly
to nothing at the near one. Row 150 on that frame is the white band of the perimeter boards; the
far touchline is at about 155 and faint. The third failure the protocol wrote down in advance -
"the far touchline registered to the boards, not the paint" - is the one that happened.
`@C -4.7,-3.9,-3.4 px; 1.1 m :: inline script, N31 session, n31_linereg's 118575 camera against equirect10 on model/validation/n31_reread_118575.json`

## 5. After the score: what the remaining error is made of

Three things were done after §4, all on the development set, none of them a new score at feet.

**A perimeter rule, which barely mattered.** On the four perimeter lines the search now walks from
inside the pitch outward and takes the first ridge with grass at its inner foot
(`n31_linereg.py --fit`; `--v1` reproduces what §4 scored). On a synthetic frame with a bright
board 6 px beyond the far touchline it is the difference between 0.67 m and 0.01 m (`--demo`
asserts it). On the real clips it moved the far touchline by about a pixel and the far-side map by
half a metre: the rigid spherical camera cannot bend to the real far touchline, which is nearly flat
across the middle third where the model wants it curved. That is what each lens's own small
vertical correction looks like (`research-unstitching-panorama.md`
Q1, Gaddam et al. 2014) - nothing on the near side, metres on the far one.

**A column warp, and a test of it that needs no ground truth.** `_fwd`/`_inv` take an optional
smooth warp of the row by column, a(u) + b(u)(v - v0), degree 5, ridge-penalised toward the rigid
camera, closed-form to invert. The test: hide one family of lines from the fit, then measure how
far the map lands from that family's paint, in metres through the map's own local scale. It is the
only ground-truth-free measure of a map BETWEEN its constraints, and its population is the pitch
interior, where players are.

    @M 7 of 7,True :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n31c_logo.json'))['_go'];print(r['warp_better_on'],r['go'])"
       held fixed: the seven same-install clips, N30's frozen clicks as the start, 40-frame medians,
       six held-out families (circle; halfway; penalty-box fronts; six-yard fronts; penalty-box side
       lines; far touchline), rigid against warped; n=2 configurations; GO rule written in
       `n31_linereg.logo` before it ran. No ground truth read.

Held-out interior paint lands 0.03-0.45 m from the rigid map and a little closer under the warp; the
held-out FAR TOUCHLINE lands 1.05-3.29 m out (rigid) and 1.00-2.85 m (warped) (`n31c_logo.json`).
So: between its constraints, in the interior, the line-registered map is within half a metre of the
paint. The far touchline is where it is wrong, and where the paint itself is hardest to tell from
the boards. The warped cameras have NOT been scored at feet.

**The error against the feed is almost entirely along the camera ray.** For §4's LINE, pooled over
the seven clips, decomposed about the camera's ground point:

`@C 0.34,0.47,-1.9%,-3.1%,-3.9%,-4.1%,-4.4% :: inline script, N31 session, n31_linereg.json cameras on n31_score's feet frames: lowest and highest mean absolute error ACROSS the ray in metres over five range bands, then the mean error ALONG the ray as a share of range at 15-30, 30-45, 45-60, 60-75 and 75-100 m from the camera`

Laterally the map agrees with a feed quantised to about a metre as well as anything can. In depth it
puts every player a near-constant 4% closer to the camera than the feed does - a similarity about the
camera's foot, which is also exactly what a uniform chord slope of 0.95 on both axes and both halves
is (§4, L3), and what §1 found as "the feed is a magnified copy of our map". A constant fraction of
range is what a reference point at a different HEIGHT does: about 0.3 m under a 6.8 m camera.
`@D 0.28 :: 6.8 * (1 - 1 / 1.043)`

Which side is right is NOT settled. For the feed: ST2's chain agrees with it to 1.41 m, and at
kick-off the feed puts the nearest player exactly on the centre spot in 4 matches of 7 (offsets 0.0,
0.0, 0.0, 0.0, -0.7, -2.0, -2.7 m; a 4% magnification predicts -2.0 every time). For the map: its
held-out interior paint, including the centre circle, lands within half a metre, and the feed's
density of play runs 3-4 m past the far touchline and the left goal line but stops within 1-2 m of
the near touchline and the right goal line. Both readings are cited from inline scripts run in the
N31 session over the seven matches' raw `tracker_box_data.xml`, not re-runnable tags.
Until an independent physical reference exists - a surveyed point, or a player known to be standing
on a line - "metres at feet against the feed" cannot separate a better map from a map that shares
the feed's convention, below about 2 m. L2's comparison with the chain is confounded by exactly that.

## 6. Where this leaves the work

- **Established, without ground truth:** the operator's far-side dots were the largest single error;
  the family is spherical; line registration removes the dependence on dot precision and lands
  interior paint within half a metre; seven matches of one install give the same camera to a pixel.
- **Established, development set, one install:** line registration beats both click-only camera
  maps on every clip (2.37 m against 2.92 and 3.18 pooled, p90 3.74 against 5.87 and 5.03) and
  reads movement as ST2's own chain does.
- **Not established:** that it is within any stated distance of the truth. The reference disagrees
  with it by a 4% radial factor whose owner is unknown.
- **Refused, and therefore unknown:** 117092, the other rig.
- **Product, done:** the setup screen's placing lens (`calibrate-step.tsx`). **Owed:** guesses seeded
  from a spherical fit instead of a homography; a "these dots do not fit a camera" warning; pooling an
  install's matches into one camera (§3's 0.9 px agreement says it would work); the registration
  itself in `engine/`, which nothing here has earned yet.
- **Next measurement worth making:** an independent depth reference on this install - players
  standing on known lines at restarts (kick-offs, corners, goal kicks taken from the six-yard line),
  read off the video, scored against both the map and the feed. It needs no new footage and it is
  the only thing that can say who owns the 4%.

**Done, 2026-09-17:** that measurement is `n32-depth-reference.md`. UNRESOLVED -
five far references where its rule needed eight - with a lead: on the far touchline the larger share
of the disagreement is this map's.
