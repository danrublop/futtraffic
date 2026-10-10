# data/footage/ — real fixed-camera soccer video

> **Start here.** No video ships with this repository. The footage is
> [SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2) (CC BY 4.0): download it and set
> `ST2_DIR` (default `~/Downloads/st2`). A few scripts also read the 118575 first half and its
> calibration bundle from this folder, as `118575_panorama_1st_half.mp4` and
> `soccertrack_calib_118575/` (both from the SoccerTrack v2 Drive mirror below). Ball clips come
> from TeamTrack (MIT) via `model/validation/ball_clean.py fetch`, into `soccertrack/`.
>
> Everything below is the original acquisition log, kept because other documents cite its
> sections (§1–§4).

Everything here was downloaded from a public, **commercially-usable** source.
Licence was treated as a gate: anything non-commercial, NDA-gated, or of unclear
provenance was rejected and is recorded below under "Rejected".

Nothing in this directory came from SoccerNet or any re-upload of it.

## TL;DR

Three findings that change the pipeline:

1. **It is a stitched panorama, not a perspective view.** One homography on the raw frame
   has **median 46 px / max 428 px** error against the dataset's own keypoints. Wrong model.
2. **The fix ships with the dataset.** `soccertrack_calib_118575/` has OpenCV remap tables.
   Correct order is `cv2.remap(...)` **then** homography — verified, see §2.
3. **Players are 8–115 px tall, not 20–50.** Our assumption was wrong by ~2x at both ends.

---

## 1. SoccerTrack v2 — PRIMARY SOURCE

- **Project:** https://atomscott.github.io/SoccerTrack-v2/
- **Repo:** https://github.com/AtomScott/SoccerTrack-v2
- **Data licence:** **CC BY 4.0** — explicitly permits commercial use with attribution.
  Verbatim from `LICENSE-DATA`: *"for any purpose, even commercially"*.
  Source code in the repo is MIT (separate licence).
- **Attribution required.** Cite the SoccerTrack v2 paper and link the CC BY 4.0 licence.
- **Content:** University of Tsukuba amateur matches, fixed multi-camera array at the
  halfway line, stitched to a 4K panorama. 10 matches, both halves each.

### Files used (from the SoccerTrack v2 Drive mirror)

| File | Source URL | Res | fps | Duration | Projection |
|---|---|---|---|---|---|
| `117093_calibrated_panorama_1st_half.mp4` | Drive `videos/117093/` (id `1CA-EFfgaXnclsE288I_DLOPcvbtpkgHz`) | 4096x1080 | 25 | 1.88 s (47 frames) | **stitched panorama (calibrated)** — decodes OK |
| `118575_panorama_1st_half.mp4` | Drive `videos/118575/` (id `1-FZNCqVSx8YleM2E7UK90-SAdNAlTeSD`) | 4096x1080 | 25 | 47.9 min (71,850 frames) | **stitched panorama — COMPLETE and decodes.** 3,550,087,937 B. Note the GT is 72,101 frames, 251 longer; the offset is in `soccertrack_calib_118575/118575_padding_info.csv`. |
| `st2_demo-gsr_and_bas.mp4` | `raw.githubusercontent.com/AtomScott/SoccerTrack-v2/main/docs/assets/demo-gsr_and_bas.mp4` | 4096x1080 | 25 | 30 s (750 f) | **stitched panorama**, GSR/BAS overlays burned in |
| `st2_demo-tracking.mp4` | same repo, `docs/assets/demo-tracking.mp4` | 1280x338 | 25 | 30 s | downscaled panorama + tracking overlay |
| `st2_demo-minimap.mp4` | same repo, `docs/assets/demo-minimap.mp4` | 874x546 | 25 | 30 s | derived 2D minimap, not camera footage |
| `st2_panorama-day.jpg` | same repo, `docs/assets/panorama-day.jpg` | 900x376 | — | still | clean panorama, no overlays |
| `st2_panorama-night.jpg` | same repo, `docs/assets/panorama-night.jpg` | 900x376 | — | still | clean panorama, no overlays |

Drive mirror root: https://drive.google.com/drive/folders/1N2Qx2qkFgRtpbHitl2Vh6sLVYGgqkWwn

### Why 118575 and 117093 matter

`model/validation/soccertrack/` holds the ground-truth tracks for `118575_1st` (the other
halves are rebuilt from the dataset by `validate.py --cache`). The videos above are **the same
matches**, so there is video AND ground-truth pitch positions for the same footage.

`118575_1st.tracks.json.gz`: 72,101 frames @ 25.0 fps = **48.07 min**, 22 track IDs,
positions in metres, x from -14.70 to 110.25, y from -8.84 to 74.80 (pitch plus
off-pitch margin). Frame ids start at **3000001**, so the working hypothesis is
`video_frame_0 -> f=3000001`. The video is 25 fps, matching the track fps exactly.

**Pitch coordinate system (derived from the data):** a standard **105 x 68 m** pitch with
the origin at a corner, x along the length, y across. 99.49% of all positions fall inside
`0<=x<=105` and 99.39% inside `0<=y<=68`; the excursions are throw-ins, keepers behind the
goal line, and staff on the touchline. So the calibration target rectangle is
`(0,0)-(105,68)` in metres — not a normalised or centre-origin frame.

---

## 2. Projection: it is a PANORAMA, not a perspective view

**This is settled, not a guess.** See `st2_panorama-day.jpg`, one of the
images in the SoccerTrack v2 repository's assets folder:

- Touchlines bow into pronounced arcs — the near touchline sags at image centre and
  rises at both ends; the far touchline bows the opposite way.
- Black lens/bowtie-shaped stitch mask at top and bottom corners.
- Aspect ratio 4096:1080 = **3.79:1**.

**Consequence: a single homography is mathematically wrong for this footage.**
Straight world lines are not straight in the image, so no plane-to-plane projective
map can be correct across the frame.

Also: **image row does not map monotonically to pitch depth.** Because the pitch bows,
a player near the near touchline appears near the image *bottom* at centre-x but near the
image *top* at the left and right ends. Any "lower in frame = closer to camera" heuristic
will break at the panorama edges.

### The "calibrated" variant does not fully fix it

Match 117093 also ships `117093_calibrated_panorama_*.mp4`. In it the pitch is far
flatter, but the content band is **still visibly bowed**. Non-black content occupies only
**3963 x 406 px inside the 4096x1080 canvas** (~9.8:1) — the rest is black padding.
So "calibrated" means per-camera lens undistortion + stitch, giving a cylindrical
projection — **not** a plane-rectified image. A single homography is still not valid.

The repo's own `docs/calibration.md` describes the pipeline as *undistortion and
perspective correction* followed by *"computes the homography matrix for coordinate
projection"* — i.e. homography is applied only **after** undistortion, never to the raw
panorama. Our pipeline must do the same, or use a piecewise/cylindrical model.

### The fix is shipped with the dataset — `soccertrack_calib_118575/`

The Drive mirror's `raw/118575/` folder contains the upstream calibration artefacts, all
downloaded to `soccertrack_calib_118575/` (same CC BY 4.0 licence):

| File | What it is |
|---|---|
| `118575_mapx.npy` | `(1080, 4096, 2)` **int16** |
| `118575_mapy.npy` | `(1080, 4096)` **uint16** |
| `118575_homography.npy` | `(3,3)` float64 |
| `118575_camera_intrinsics.npz` | `K`, `D` (4x1 → **fisheye** model), `Knew`, `rvecs`, `tvecs`, `rms` |
| `118575_keypoints.json` | 65 pitch-metre → panorama-pixel correspondences |
| `118575_player_nodes.csv`, `118575_12_class_events.json`, `118575_tracker_box_metadata.xml` | annotations |

The `mapx`/`mapy` dtype pair (`CV_16SC2` + `CV_16UC1`) is exactly OpenCV's fixed-point
remap format from `cv2.convertMaps`. **So the intended pipeline is:**

```
frame -> cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR) -> apply homography -> pitch metres
```

The homography is only valid **after** the remap. We do not need to derive our own
rectification — it ships with the match.

**The remap tables load and work.** `cv2.remap` on a synthetic regular grid succeeds and
turns the grid into a clean perspective fan — all lines stay **straight** and converge to a
single vanishing point. That is the signature of a genuine
projective rectification, i.e. after the remap the image behaves like a pinhole camera and a
single homography becomes the correct model. The maps sample source rows from -1210 to 2197,
well outside the 1080-row frame, which is why rectified output has large black margins.

**Verified, not assumed:** the shipped `H` was tested against the 65 keypoints in all four
directions on the *raw* panorama (`H` and `inv(H)`, metres→px and px→metres). Every
direction is wildly off (best median 109.5, others 587 / 1961 / 3408). So `H` does **not**
live in raw-panorama space — it operates on the remapped image, confirming the order above.

**`118575_padding_info.csv` — timing/alignment metadata:**

```
Event Period,Period Order,Video,Padding,Start Match Time,End Match Time
FIRST_HALF, 0, 657952,  8436000, 0,       2874000
SECOND_HALF,1, 657952, 12002000, 2700000, 5637000
```

Times are milliseconds. The first half is 0 → 2,874,000 ms = **47.9 min**, which matches the
48.07 min of `118575_1st.tracks.json.gz` to within ~250 frames. `Padding` is the offset of
each half inside one continuous source recording (8436 s vs 12002 s → a 59.4 min gap between
half starts, i.e. a 47.9 min half plus ~11.5 min of half-time — self-consistent). The
per-half mp4 we downloaded is already split, so padding is most likely already applied;
**verify this before trusting frame alignment.**

### Hard numbers proving a single homography is wrong on the raw panorama

Fitting one homography to all 65 shipped pitch keypoints (`cv2.findHomography`, pitch
metres → raw panorama pixels) gives:

| | px |
|---|---|
| mean reprojection error | **116** |
| median | **46** |
| p95 | **410** |
| max | **428** |

Players are 8–115 px tall, so the median error is roughly one whole player and the tail is
several. Straight touchlines are measurably curved in the raw panorama: the near touchline
(y=68 m) bows **169 px** across its length, the far touchline (y=0 m) bows **63 px**.
This is not a tuning problem — it is the wrong model.

---

## 3. Measured player pixel heights

Measured by HSV grass segmentation + connected components, with a vertical
morphological close (players otherwise fragment into shirt/shorts/legs blobs), then
cross-checked by eye on 2x zoomed crops.

**Raw panorama, 4096x1080:**

| Where | Player height (px) |
|---|---|
| Far side (beyond halfway, upper band) | **20–30** |
| Mid-pitch | **30–50** |
| Near side / near touchline | **70–115** (a referee near the near touchline measured **~93 px**) |
| Overall median | **~30** |

**Calibrated panorama, content band 3963x406:** median **24 px**, p10 **10**, p90 **78**,
full range 7–83.

> **Our 20–50 px assumption was wrong.** The real spread is roughly **8 px to 115 px**,
> a ~14x scale range within a single frame. Detector input size, anchor/stride choice, and
> any minimum-box filter need to cover that whole range. Small far-side players near 8–20 px
> are the hard case and will be missed by a detector tuned for 20–50 px.

---

## 4. Camera geometry

- **Fixed** camera array — no pan, no zoom, no cuts. Verified, not assumed: phase
  correlation of a static background strip between frame 0 and frame 740 of
  `st2_demo-gsr_and_bas.mp4` gives a global shift of **(-0.014, +0.017) px**. The camera
  is rigid to well under a pixel over 30 s, so background subtraction and a
  once-per-match calibration are both viable.
- **At the halfway line — confirmed from the shipped extrinsics.** Decomposing
  `rvecs`/`tvecs` in `118575_camera_intrinsics.npz` (`C = -R^T t`) puts the camera centre at
  **x = 51.1 m** on a 105 m pitch (centre = 52.5 m), and about **18 m back from the near
  touchline**. The camera rig and an operator are visible at the bottom centre of the raw
  panorama, on a scaffold/tower.
- **Camera height: roughly 15–20 m, and this is an estimate, not a published figure.**
  The shipped extrinsics do *not* give a trustworthy height: `rms` reprojection error is
  **10.0 px** and `K` has `fy = 80,777` against `fx = 1,628`, i.e. a wildly anisotropic
  synthetic camera. A pinhole model simply does not describe a cylindrical stitch, so its
  decomposed translation should not be read as a physical height. The 15–20 m figure comes
  instead from the player-size ratio: players are ~93 px at the near touchline and ~25 px at
  the far one, so far/near distance ratio ≈ 3.7; with the touchlines 68 m apart that puts the
  near touchline ~25 m (slant) away, which with the ~18 m setback implies ~17 m of height.
  **If anything depends on camera height, measure it — do not trust this number.**
- **The entire pitch is in frame**, both goals included, plus several metres of run-off,
  the technical areas, and the far-side treeline/stands. The ground-truth x range
  (-14.7 to 110.25 m) confirms coverage well beyond the 105 m touchline.

---

## 5. Rejected sources (licence or content gate)

| Source | Why rejected |
|---|---|
| `archive.org/details/snafuinfinitysoccer-boys_110424-vs-ludlow` (CC0) | Licence fine, **content wrong**: not a fixed wide camera. It is an operator-driven pan/zoom camera that follows the ball, including face close-ups. Whole pitch never in frame. |
| `archive.org/details/soccer_boys_vsConcordCarlisle_110212` (Public Domain Mark 1.0) | Same: panning/zooming ball-following camera, spectators crossing frame. |
| Alfheim / Tromsø (`datasets.simula.no/alfheim/`) | Genuinely fixed panoramic array with 20 Hz ZXY ground truth, but terms state *"The data may only be used for non-commercial research purposes"*. **Fails the commercial gate.** Not downloaded. |
| `huggingface.co/datasets/atomscott/soccertrack-v2` | Returns **HTTP 401** for the repo page, API, and all file paths. Not publicly readable. Not circumvented. |
| TeamTrack (Kaggle) | Requires a Kaggle account. Not circumvented. |
| `huggingface.co/datasets/tartotarto/soccernet-500-videos` | Unauthorised re-upload of NDA'd SoccerNet video. **Never touched.** |
