# footage2/ — hunting a single fixed PERSPECTIVE camera

> **Start here.** This folder holds TeamTrack clips (MIT), the ball detector's data.
> `model/validation/ball_clean.py fetch` downloads them here one member at a time, using the
> committed `teamtrack_zip_index.json` (§6). Nothing else is needed from this page.
>
> Below is the original search log, kept because other documents cite its sections. Perspective,
> handball and stock-footage cameras were explored and are out of scope for this repository.

Goal for this directory: real soccer video from **one fixed perspective camera** (a single
planar homography is the correct model), because `stats.build_match` accepts a perspective
calibration and rejects panoramic ones, and the only real match we had (`data/footage/`,
SoccerTrack v2) is a 3-camera stitch.

Nothing here overlaps `data/footage/`. SoccerTrack v2 was not re-fetched.
`huggingface.co/datasets/tartotarto/soccernet-500-videos` was never touched.
No YouTube scraping.

---

## TL;DR

**Found: a genuine single fixed perspective camera showing the whole pitch — TeamTrack
`soccer_top`.** 4K, MIT-licensed, 6 x 30 s clips on disk, with image-space bounding-box
ground truth. It is a hovering drone at nadir rather than a tripod, but it drifts only
**2.7–10.1 px per 30 s clip** and its pitch lines are **straight to 0.63–0.75 px RMS**,
which is the signature of a real pinhole projection. A single homography is the right model
for it. This is the first real video on disk that the production path can actually be run on.

**Not found: a side-on tripod camera on a whole pitch under a commercial licence.** Every
side-on fixed rig in public research data is a stitched panorama (SoccerTrack v2, TeamTrack
`soccer_side`); every side-on public-archive item is an operator-driven pan/zoom rig; and
every free stock clip moves. See §5 for what a human would have to do.

| Rank | File(s) | Projection | Static? | Whole pitch? |
|---|---|---|---|---|
| **1** | `soccer_top__*.mp4` (6 clips, 3 min) | **perspective (nadir)** | 2.7–10.1 px / 30 s | **yes, plus margins, both goals** |
| 2 | `soccer_side__*.mp4` (3 clips, 90 s) | stitched panorama | **0.0–0.1 px / 30 s** | yes |
| 3 | `pexels_28588421`, `pexels_28588274`, `pexels_15425039` | perspective | **no — 46–536 px** | ~60–75 % |
| — | `handball_side__*.mp4` | fisheye, not soccer | 0.0 px / 30 s | n/a |
| — | 12 other `pexels_*.mp4` | perspective | no, all moving | mostly no |

Disk: 1.1 GB used here, 316 GB free.

---

## 1. Full table

`drift` = displacement of the image centre under a RANSAC-fitted similarity transform
between frame 0 and 5 later frames (ORB features, `cv2.estimateAffinePartial2D`). Phase
correlation was tried first and abandoned: on grass with moving players its response
collapses to ~0.00 and the reported shifts are noise. ORB+RANSAC reports 1200–2900 inliers
on the static clips, so its near-zero readings are trustworthy.

Every file below was verified to decode with
`cv2.VideoCapture(...).isOpened()` **and** a successful first-frame read. There are **no
failed downloads and no non-decoding files** in this directory.

### TeamTrack — `soccer_top` (THE FIND)

| Path | Res | fps | Dur | Size | Drift over clip | Projection |
|---|---|---|---|---|---|---|
| `soccer_top__train__videos__D_20220220_1_0000_0030.mp4` | 3840x2160 | 29.97 | 30.03 s | 35.2 MB | **9.7 px** | perspective |
| `soccer_top__train__videos__D_20220220_1_0030_0060.mp4` | 3840x2160 | 29.97 | 30.03 s | 32.2 MB | **10.1 px** | perspective |
| `soccer_top__test__videos__D_20220220_1_1680_1710.mp4` | 3840x2160 | 29.97 | 30.03 s | 26.1 MB | **6.1 px** | perspective |
| `soccer_top__test__videos__D_20220220_1_1710_1740.mp4` | 3840x2160 | 29.97 | 30.03 s | 26.9 MB | **2.7 px** | perspective |
| `soccer_top__test__videos__D_20220220_1_1740_1770.mp4` | 3840x2160 | 29.97 | 30.03 s | 28.9 MB | **5.5 px** | perspective |
| `soccer_top__test__videos__D_20220220_1_1770_1800.mp4` | 3840x2160 | 29.97 | 30.03 s | 27.8 MB | **5.2 px** | perspective |

Matching per-clip bbox ground truth: `soccer_top__*__annotations__*.csv` (one per video).

- **Source:** TeamTrack, `teamtrack.zip` inside
  https://drive.google.com/drive/folders/1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ
  (repo https://github.com/AtomScott/TeamTrack , mirror
  https://www.kaggle.com/datasets/atomscott/teamtrack ).
- **Licence: MIT.** Stated on both the GitHub repo and the Kaggle dataset page. MIT permits
  commercial use; it requires the copyright notice and permission notice be retained, so
  ship the MIT text with any redistribution.
- **Zip members were range-extracted, not downloaded whole.** `teamtrack.zip` is 8.60 GB;
  its ZIP64 central directory was fetched from the tail with an HTTP `Range` request, then
  each wanted member was pulled by byte range and inflated. See §6 to get more clips.
- **Projection: perspective, near-nadir.** Judged three ways:
  1. **Line straightness.** Fitting a straight line to the touchlines of frame 0 of
     `D_20220220_1_1680_1710`: top touchline **RMS 0.75 px, max deviation 2.07 px** over an
     2865 px span; bottom touchline **RMS 0.63 px, max deviation 1.78 px** over 2860 px.
     A quadratic fit finds a bow of only 0.4–1.1 px. Straight world lines are straight in
     the image, which a cylindrical stitch cannot do.
     (For contrast, `data/footage/`'s SoccerTrack v2 panorama bows its near touchline by 169 px.)
  2. **Aspect ratio 16:9**, not the 3.8:1 / 6.5:1 of a stitch. No stitch seams, no
     bowtie mask, no chromatic fringing at the frame edges.
  3. **Isotropic scale.** The pitch spans ~2912 px across 105 m (27.7 px/m) and ~1904 px
     across 68 m (28.0 px/m). Equal scale in both axes is what a near-nadir pinhole view
     gives; a cylindrical panorama would not.
- **Static?** Hovering drone, not a tripod. 2.7–10.1 px of centre drift per 30 s clip,
  with 900–1670 RANSAC inliers, and scale held to 1.0000 ± 0.0008 (no zoom). Different
  clips are at slightly different poses, so calibrate per clip, not once for all six.
- **Pitch coverage: the whole pitch**, both goals, all four corners, plus several metres of
  run-off on every side and the surrounding ball-catch cages. Nothing is cropped.
- **Player pixel heights (from the GT `bb_height` column, 19,800 boxes per clip):**
  median **24–40 px**, p5 **20–36 px**, p95 **31–50 px**, full range **16–64 px**.
  Note how tight that is — about a **2x** scale spread across the frame, against roughly
  **14x** in the SoccerTrack v2 panorama. A nadir view makes every player the same size,
  which makes it much friendlier to a fixed-scale detector.
- Evidence frame: `frames/soccer_top__test__videos__D_20220220_1_1680_1710.jpg`.

### TeamTrack — `soccer_side` (panorama, but perfectly rigid + bbox GT)

| Path | Res | fps | Dur | Size | Drift over clip |
|---|---|---|---|---|---|
| `soccer_side__train__videos__F_20200220_1_0000_0030.mp4` | 6500x1000 | 25 | 30.0 s | 10.1 MB | **0.04 px** |
| `soccer_side__train__videos__F_20200220_1_0030_0060.mp4` | 6500x1000 | 25 | 30.0 s | 7.9 MB | **0.01 px** |
| `soccer_side__train__videos__F_20200220_1_0060_0090.mp4` | 6500x1000 | 25 | 30.0 s | 9.3 MB | **0.06 px** |

Same MIT licence and same source zip as above. Matching `soccer_side__*__annotations__*.csv`.

- **Projection: stitched panorama.** Aspect **6.50:1**. Touchlines visibly bow (near
  touchline sags at image centre, rises at both ends). Chromatic fringing at the extreme
  left and right edges is the tell-tale of per-camera lens undistortion before the stitch.
  A line fit to a far-touchline band gives **max deviation 88 px** over a 5895 px span
  (caveat: that band was chosen by eye and may pick up off-pitch content, so treat 88 px as
  an order-of-magnitude figure — the visual bow in `frames/…F_20200220_1_0000_0030.jpg` is
  the primary evidence). **A single homography is wrong here, same as SoccerTrack v2.**
- **Static: yes, absolutely.** 0.0–0.1 px of drift over 30 s with 2400–2950 inliers. This
  is the most rigid camera in either footage directory.
- Whole pitch in frame, both goals, plus the far treeline and the near touchline crowd.
- **Player pixel heights (from GT):** median **33–44 px**, p5 **20–28 px**, p95 **60–97 px**,
  full range **11–123 px**.
- **Why it is here anyway:** it is the same University of Tsukuba venue as `data/footage/`, and
  unlike `data/footage/` it ships **image-space bounding boxes**, which `data/footage/` is still
  missing (`118575_tracker_box_data.xml` never downloaded). Useful for detector work even
  though the geometry path stays broken.

### TeamTrack — `handball_side` (not soccer; kept as a reference fixed fisheye)

| Path | Res | fps | Dur | Size | Drift |
|---|---|---|---|---|---|
| `handball_side__test__videos__2nd_fisheye_690-720.mp4` | 3300x450 | 29.97 | 30.06 s | 13.0 MB | **0.02 px** |

MIT, same zip. Filename says `fisheye`; aspect is 7.33:1 so it is a cropped/stitched wide
strip, not a plain perspective frame. Downloaded only to test whether TeamTrack's indoor
splits were single-camera perspective — they are not obviously so, and it is a handball
court, not a pitch. Not useful for this task; delete if it is in the way.

### Pexels — free stock, commercially usable, but **every single clip moves**

All from https://www.pexels.com/ under the **Pexels License** (free for commercial use, no
attribution required, https://www.pexels.com/license/ ). Downloaded via
`https://www.pexels.com/download/video/<id>/`. All decode.

| File | Page | Res | fps | Dur | Size | Max drift | Verdict |
|---|---|---|---|---|---|---|---|
| `pexels_28588421.mp4` | pexels.com/video/…-28588421/ | 1920x1080 | 25 | 8.92 s | 10.8 MB | **145.7 px** (+15 % zoom) | perspective, elevated, ~70 % of pitch — but drone, zooming |
| `pexels_28588274.mp4` | …-28588274/ | 1920x1080 | 25 | 13.88 s | 16.5 MB | **535.6 px** | perspective, elevated, ~60 % of pitch — steady left pan |
| `pexels_15425039.mp4` | …-15425039/ | 3840x2160 | 29.97 | 6.84 s | 22.4 MB | **46.4 px** | perspective, elevated, ~65 % of pitch — slow drone creep |
| `pexels_29853053.mp4` | …-29853053/ | 3840x2160 | 29.97 | 5.24 s | 114.7 MB | 282.7 px | drone, partial pitch |
| `pexels_15449351.mp4` | …-15449351/ | 1280x720 | 59.94 | 20.02 s | 7.7 MB | 491.9 px | steady pan, behind goal |
| `pexels_12718608.mp4` | …-12718608/ | 2560x1440 | 29.97 | 22.89 s | 36.6 MB | 45.4 px (+21 % zoom) | drone nadir, whole 5-a-side pitch, but descending |
| `pexels_11918917.mp4` | …-11918917/ | 1920x1080 | 60 | 13.10 s | 10.0 MB | 986.7 px | handheld phone from a stand |
| `pexels_3560017.mp4` | …-3560017/ | 3840x2160 | 25 | 9.00 s | 24.8 MB | 3049.9 px | handheld, pitch cropped |
| `pexels_20120710.mp4` | …-20120710/ | 3840x2160 | 30 | 13.67 s | 39.3 MB | 345.8 px | ground level, no pitch lines |
| `pexels_11474932.mp4` | …-11474932/ | 3840x2160 | 29.97 | 18.72 s | 59.0 MB | 3095.1 px | handheld, dust pitch |
| `pexels_15448985.mp4` | …-15448985/ | 1920x1080 | 59.94 | 34.53 s | 25.3 MB | 697.0 px | ground level, close |
| `pexels_15462750.mp4` | …-15462750/ | 1920x1080 | 59.94 | 37.54 s | 27.2 MB | 1047.4 px | ground level, empty stadium |
| `pexels_33717035.mp4` | …-33717035/ | 3840x2160 | 29.97 | 50.68 s | 358.4 MB | 1954.4 px | B&W, handheld, roaming |
| `pexels_36835896.mp4` | …-36835896/ | 1920x1080 | 29.97 | 36.24 s | 38.7 MB | 957.8 px | ground level, roaming |
| `pexels_37179914.mp4` | …-37179914/ | 2160x3840 | 59.94 | 14.83 s | 164.6 MB | 1106.1 px | **vertical** phone video |

Player pixel heights for the three usable-geometry ones (HSV grass segmentation + vertical
close, approximate): `28588421` far **~28 px** / near **~96 px**, 10th–90th pct 16–114 px;
`28588274` far **~39 px** / near **~87 px**, 12–91 px; `15425039` **10–39 px** but that
number is contaminated by the hillside town in the background — treat it as unreliable.

**Conclusion on stock video:** it was worth trying and it was a dead end. Stock footage is
shot to look good, which means it always moves. 15 clips, 15 moving cameras. The lowest
drift among them (46 px) is still ~1.5 player-heights, and it came with a zoom.

Evidence frames for all of the above are in `frames/` (one JPEG per mp4, sampled a third of
the way in, downscaled to 1400 px on the long edge).

---

## 2. How each judgement was made

- **Perspective vs panorama.** Primary test: fit a straight line to a touchline and report
  RMS and max deviation, plus a quadratic fit to quantify bow. Secondary: frame aspect
  ratio (a stitch is 3.8:1 to 7.3:1; a single camera is 16:9), presence of stitch seams,
  edge chromatic fringing, and a bowtie/black mask at the corners. Tertiary for the nadir
  view: equal px/m along and across the pitch.
- **Static or not.** ORB (4000 features) between frame 0 and 5 evenly spaced later frames,
  matched with a crossCheck BFMatcher, then `cv2.estimateAffinePartial2D` with RANSAC at a
  2 px threshold; reported number is the displacement of the image centre under the fitted
  transform, and the scale term is checked separately to catch zoom. Inlier counts are
  reported so a near-zero reading can be distinguished from a failed fit — every static
  claim above rests on 900+ inliers.
- **Player pixel heights.** Where ground truth exists (all TeamTrack clips), taken directly
  from the `bb_height` rows of the annotation CSV with BALL columns excluded — 16,500 to
  19,800 boxes per clip, so these are exact, not estimated. For the Pexels clips, HSV grass
  segmentation with a vertical morphological close, which is approximate.

---

## 3. Licence record

| Asset | Licence | Commercial use | Source |
|---|---|---|---|
| All `soccer_top__*`, `soccer_side__*`, `handball_side__*` | **MIT** | yes (retain notice) | github.com/AtomScott/TeamTrack , kaggle.com/datasets/atomscott/teamtrack , Drive folder `1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ` |
| All `pexels_*.mp4` | **Pexels License** | yes, no attribution required | pexels.com/license/ |

Both licences clear the commercial gate. TeamTrack's MIT covers the repository and, per
both the GitHub repo and the Kaggle dataset page, the dataset; there is no separate,
stricter data licence file the way SoccerTrack v2 has `LICENSE-DATA`. If TeamTrack video is
going to be redistributed rather than just used internally, have someone confirm that with
the maintainers, and cite the TeamTrack paper regardless.

---

## 4. Rejected — checked and not used

| Source | Licence | Why rejected |
|---|---|---|
| `archive.org/details/pcac-boys-soccer-st.-petersburg-vs.-lakewood-1997.01.24` | **CC0** | Licence is fine and it is a full 91.7 min game (543 MB mp4 / 3.19 GB mkv). **Content wrong:** sampled frames at t=1800 s and t=3000 s show an operator zooming to a full-frame close-up of the goalkeeper, then pulling out. Same pan/zoom ball-following rig as the two items already rejected in `data/footage/`. Whole pitch never in frame. Only 3 frames were pulled; the file was not downloaded. |
| `archive.org/details/20240307-r` "Random Soccer/Football Match" | PDM 1.0 | 3.2 h, 7.8 GB. Description is *"Just some soccer/football game I found sitting on a DVD-RW"* — the uploader stamped Public Domain Mark on content they did not create. **Unclear provenance, fails the licence gate.** Not downloaded. |
| ~90 further archive.org soccer items (`snafuinfinity*` Belmont HS, `soccer_*_vs*-CL19`) | PDM / CC0 | Same Belmont HS Sports TV operator-driven camera already rejected in `data/footage/`. Not re-tested individually. |
| The rest of archive.org's CC-licensed soccer (WCT*, WHS*, DoubleACS*, MMU*, WUHS*) | **CC BY-NC-\*** | Non-commercial. Fails the gate. Not downloaded. |
| SoccerTrack v2 `raw/*` folders | CC BY 4.0 | Checked specifically for **pre-stitch per-camera video** — there is none. Every `raw/<match>/` folder holds only calibration and annotation artefacts (`mapx/mapy/homography/intrinsics/keypoints/player_nodes/events/tracker_box`), and every `videos/<match>/` folder holds only `*_panorama_*.mp4`. All 10 matches checked. No perspective video exists in that dataset. |
| Pixabay | Pixabay Content License (would pass) | **HTTP 403 to both `curl` and the fetch tool**, on the search pages and the API without a key. Not circumvented. A human with a browser, or a free Pixabay API key, could search it; given Pexels' 15/15 failure rate the expected yield is low. |
| Coverr | — | Search returns HTTP 404 and the only extractable assets are AI-generated clips. No real match footage. |
| Mixkit | Mixkit License (would pass) | 18 soccer clips total, all short close-ups and slow motion. No wide pitch shots. |
| Wikimedia Commons | CC BY-SA / CC0 | 36 total hits for soccer match video; almost all are FIFA **Beach Soccer** anthem/celebration clips of a few seconds. No fixed-camera match footage. |
| SoccerTrack v1 (`kaggle.com/datasets/atomscott/soccertrack`) | **unverified** | Its "wide-view" split was the best remaining lead for a single fisheye camera (priority 2). The Kaggle page is JS-rendered and returns no licence string to a non-browser client, and the GitHub README (now "SportsLabKit") documents the library, not the dataset licence. **Licence could not be established, so nothing was downloaded.** See §5. |
| ISSIA-CNR | **unverified** | Not reachable to check within this session's budget. Historically distributed on request and research-only. Not downloaded. |
| Alfheim / Tromsø | non-commercial | Already rejected in `data/footage/`. Unchanged. |
| `huggingface.co/datasets/tartotarto/soccernet-500-videos` | NDA re-upload | **Never touched.** |

Web search budget for the session was exhausted at the first call, so all discovery above
was done by hitting known APIs and endpoints directly (archive.org advancedsearch, Google
Drive folder listings, Kaggle/GitHub raw, Pexels download endpoint). That mainly cost
breadth on the long tail of research datasets — see §5.

---

## 5. What a human would still have to do

Ranked by expected value.

1. **Establish the SoccerTrack v1 licence.** Open
   https://www.kaggle.com/datasets/atomscott/soccertrack in a browser and read the licence
   box. If it is CC BY / CC0 / MIT, download the **wide-view** split: it is reported to be a
   single wide-angle/fisheye camera rather than a stitch, which would be a priority-2 hit
   (undistort, then one homography) and, unlike the nadir drone, would be a genuine side-on
   view. If it is CC BY-NC, stop — it fails the gate.
2. **Get more TeamTrack `soccer_top` if 3 minutes is not enough.** 60 clips exist
   (42 train / 9 val / 9 test, 30 s each = **30 min** of continuous match). Extract them
   without downloading the 8.6 GB zip using the recipe in §6.
3. **Ask the University of Tsukuba group for the pre-stitch camera feeds.** The SoccerTrack
   v2 rig is 3 fixed cameras that are stitched into the panorama we already have. Each raw
   feed is by definition a single fixed perspective camera on a real match, and the group
   already publishes under CC BY 4.0. They are not on the Drive mirror; an email or a GitHub
   issue on `AtomScott/SoccerTrack-v2` is the cheapest possible route to exactly the asset
   this task was after.
4. **ISSIA-CNR.** Six fixed 1080p perspective cameras is precisely the target geometry.
   Find the current distribution point and read the terms; if it is research-only it fails
   the gate, but that has to be confirmed rather than assumed.
5. **Record 5 minutes yourself.** A phone on a tripod on a stand at any local ground, at the
   halfway line, high enough to see both goals. Zero licence risk, exactly the geometry the
   production path wants, and probably faster than any of the above. This is a serious
   suggestion, not a joke — the search above establishes that the asset essentially does not
   exist in public commercially-licensed form.

Do **not** bother with more free stock video. Fifteen clips from the largest CC0-style
library all moved; the medium is fundamentally the wrong one.

---

## 6. Getting more TeamTrack clips without the 8.6 GB zip

`teamtrack.zip` (Drive id `12KOMgl9BFirmPVHbbUTUWacc9vLG94er`, 8,598,559,786 bytes) serves
`Range` requests. Its ZIP64 central directory sits at offset 8,598,363,774, is 195,914
bytes, and lists 1,310 members. The already-parsed member table (name, offset, compressed
size, uncompressed size, method) is saved here as **`teamtrack_zip_index.json`** — no need
to re-derive it. To regenerate: fetch the last 2 MB of the file and parse the ZIP64 EOCD.

```python
# /tmp/zget.py — extracts one member by byte range
import json, struct, subprocess, zlib
ID  = "12KOMgl9BFirmPVHbbUTUWacc9vLG94er"
URL = f"https://drive.usercontent.google.com/download?id={ID}&export=download&confirm=t"
ents = {e[0]: e for e in json.load(open('footage2/teamtrack_zip_index.json'))}

def rng(a, b):
    return subprocess.run(["curl","-s","-m","900","-r",f"{a}-{b}",URL],
                          capture_output=True).stdout

def get(name, dest):
    _, off, csz, usz, meth = ents[name]
    hdr = rng(off, off + 63)                       # local file header
    assert hdr[:4] == b'PK\x03\x04'
    nlen, elen = struct.unpack('<HH', hdr[26:30])  # local extra len != central extra len
    data = rng(off + 30 + nlen + elen, off + 30 + nlen + elen + csz - 1)
    assert len(data) == csz
    if meth == 8:
        data = zlib.decompressobj(-15).decompress(data)
    assert len(data) == usz
    open(dest, 'wb').write(data)

get("teamtrack/soccer_top/val/videos/D_20220220_1_1320_1350.mp4", "out.mp4")
```

Clip naming is `D_20220220_1_<start>_<end>` for `soccer_top` and `F_2020/20220220_1_<start>_<end>`
for `soccer_side`, in seconds, contiguous in 30 s steps. Each video has a same-named `.csv`
under `annotations/` in the same split. Splits are by time (verified against the zip index):

| Split | `soccer_top` | `soccer_side` |
|---|---|---|
| train | 42 clips, 0–1260 s | 46 clips, 0–1380 s |
| val | 9 clips, 1260–1530 s | 10 clips, 1380–1680 s |
| test | 9 clips, 1530–1800 s | 10 clips, 1680–1980 s |

**Quirk to know about:** `soccer_side` train filenames use two different date stamps,
`F_20200220_1_*` for 0–870 s and `F_20220220_1_*` for 900–1380 s. It is the same match; the
`2020` is a typo upstream. Getting it wrong is the most likely reason a range extract fails.

If Google Drive starts answering with *"Too many users have viewed or downloaded this
file recently"*, that is the same anonymous bandwidth quota that blocked `data/footage/`'s
118575 download. It clears in hours; signing in to any Google account in a browser uses a
separate per-user quota.
