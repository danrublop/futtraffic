# Third-party notices

This project is licensed under Apache-2.0 (see [`LICENSE`](../LICENSE) and
[`NOTICE`](../NOTICE) at the repo root). A built copy of the app bundles the
third-party work listed here, each under its own licence, and each notice is
reproduced in full below because that is what those licences require of a
distribution - not as a courtesy. AGPL components are excluded by design.

Two things this file is NOT:

- It is not the attribution for player headshots the app may fetch at runtime.
  Those are per-image, mostly CC BY-SA, and their credit is stored with each
  fetched image and served at `GET /api/players/photo-credits`; no photographs
  are distributed with this repository. A licence that requires attribution
  *at the point of display* cannot be satisfied by a file in a repository.
- It is not a substitute for the licence texts that ship inside the application
  bundle. `cv2/LICENSE-3RD-PARTY.txt` in particular must stay where PyInstaller
  puts it.

---

## OpenCV and the media libraries inside it

**This is the LGPL component of a built app.** OpenCV's own code
is Apache-2.0, but the `opencv-python-headless` wheel redistributes third-party
media binaries inside its `cv2/` directory, and those are LGPL-2.1. PyInstaller
freezes that whole directory into the app (`app/desktop/backend.spec` lists `cv2` in
`hiddenimports`), so the obligation travels with every built copy distributed.

- OpenCV: Apache-2.0 - https://github.com/opencv/opencv
- LGPL-2.1, in `cv2/` on every platform: **FFmpeg** (https://ffmpeg.org)
- LGPL-2.1, additionally in `cv2/` on macOS: libbluray, libgnutls, libnettle,
  libhogweed, libintl, libmp3lame, libp11, librtmp, libsoxr, libtasn1

These libraries are dynamically loaded and are not statically linked into any
other component, so LGPL-2.1 section 6 is satisfied by keeping them
replaceable. Their source is available from the projects above, and the
corresponding object files can be obtained from the copyright holder of this
project so that a user may relink the application against a modified version.

The verbatim texts - Apache-2.0, LGPL-2.1, and every other licence in the wheel -
are in `LICENSE-3RD-PARTY.txt`, which ships beside the `cv2` module inside the
application. It must not be removed from the bundle.

Note for the web deployment only: `sharp` (Apache-2.0) and its native backend
`@img/sharp-libvips-*` (LGPL-3.0-or-later) are optional dependencies of Next.js
used by the image optimizer. This app uses no `next/image`, so the cleanest
answer is to exclude `sharp` from the deployment bundle, at which point this
paragraph does not apply. The desktop build is a static export and never
includes it.

---

## Geist and Geist Mono

The UI face and the readout face, loaded through `next/font/google` in
`app/frontend/src/app/layout.tsx`. Next.js DOWNLOADS AND SELF-HOSTS these files into
the build output, so they are redistributed with a built app even though they
appear in no `package.json` and in no lockfile - which is exactly why a
dependency scanner will not find them.

- Source: https://github.com/vercel/geist-font
- Licence: SIL Open Font License 1.1 (reproduced under Inter, below - it is the
  same licence text)

"Geist" and "Geist Mono" are Reserved Font Names: a modified version may not
use them.

---

## Inter Variable (@fontsource-variable/inter 5.3.0)

The UI body face, imported once at `app/frontend/src/app/globals.css`.

- Source: https://github.com/rsms/inter
- Licence: SIL Open Font License 1.1

```
Copyright 2016 The Inter Project Authors (https://github.com/rsms/inter) Inter-Italic[opsz,wght].ttf: Copyright 2016 The Inter Project Authors (https://github.com/rsms/inter)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
http://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```

---

## lucide-react 1.28.0

Every icon in the interface. Two copyright holders, so two notices.

- Source: https://github.com/lucide-icons/lucide
- Licence: ISC, with an MIT grant covering the icons inherited from Feather

```
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

---

The following Lucide icons are derived from the Feather project:

airplay, alert-circle, alert-octagon, alert-triangle, aperture, arrow-down-circle, arrow-down-left, arrow-down-right, arrow-down, arrow-left-circle, arrow-left, arrow-right-circle, arrow-right, arrow-up-circle, arrow-up-left, arrow-up-right, arrow-up, at-sign, calendar, cast, check, chevron-down, chevron-left, chevron-right, chevron-up, chevrons-down, chevrons-left, chevrons-right, chevrons-up, circle, clipboard, clock, code, columns, command, compass, corner-down-left, corner-down-right, corner-left-down, corner-left-up, corner-right-down, corner-right-up, corner-up-left, corner-up-right, crosshair, database, divide-circle, divide-square, dollar-sign, download, external-link, feather, frown, hash, headphones, help-circle, info, italic, key, layout, life-buoy, link-2, link, loader, lock, log-in, log-out, maximize, meh, minimize, minimize-2, minus-circle, minus-square, minus, monitor, moon, more-horizontal, more-vertical, move, music, navigation-2, navigation, octagon, pause-circle, percent, plus-circle, plus-square, plus, power, radio, rss, search, server, share, shopping-bag, sidebar, smartphone, smile, square, table-2, tablet, target, terminal, trash-2, trash, triangle, tv, type, upload, x-circle, x-octagon, x-square, x, zoom-in, zoom-out

The MIT License (MIT) (for the icons listed above)

Copyright (c) 2013-present Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## flag-icons 7.5.0

Country flags for the nationality picker on the player profile
(`app/frontend/src/lib/player-facts.ts`, drawn by `Flag` in
`app/frontend/src/components/data/player-editor.tsx`, which is also where the
stylesheet is imported).

- Source: https://github.com/lipis/flag-icons
- Licence: MIT

```
The MIT License (MIT)

Copyright (c) 2013 Panayiotis Lipiridis

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## TeamTrack

Derivatives of TeamTrack's bounding-box ground truth are committed to this
repository and are read at test time - this is redistribution, not just
measurement:

- `model/validation/soccer_top__*.tracks.json.gz`
- `model/validation/ball_cands_soccer_{top,side}__*.json.gz`
- `model/validation/ball_extents_soccer_{top,side}__*.json.gz`
- `model/validation/calib_soccer_top__*.json`, `model/validation/teamtrack_soccer_top.calib.json`

Converted from the dataset's annotation CSVs by `model/validation/teamtrack_tracks.py`.

**Since 2026-09-20 the obligation is larger than these derivatives.** 101 further
TeamTrack clips (`soccer_top` and `soccer_side`, 30 s each, with their annotation
CSVs) were fetched into `data/footage/soccertrack/` by
`model/validation/ball_clean.py fetch`, and they are the training set for a ball
detector intended to **ship** (`docs/experiments/n52-clean-ball-data.md`). The video itself is
not redistributed - `data/` is gitignored - but **weights trained on it are a
derivative**, so this notice must travel with any built application, not only with
the repository. That is the reason this source was chosen over the DFL re-export,
whose terms do not permit redistribution: MIT permits it, provided the notice goes along.
The attribution is also recorded in [`NOTICE`](../NOTICE).

- Source: https://github.com/AtomScott/TeamTrack , dataset zip in Drive folder
  `1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ` (member table committed as
  `data/footage/soccertrack/teamtrack_zip_index.json`; what was fetched is recorded
  in `data/footage/soccertrack/teamtrack_fetch.json`).
- Licence: MIT (dataset), which requires the
  copyright and permission notice be retained in any redistribution.
- Credit: Scott et al., TeamTrack.
- Modifications: clips cut into 640 px crops at native scale with the dataset's own
  ball boxes (`model/validation/ball_clean.py tiles`); annotations converted to this
  project's JSON schema.

---

## SoccerTrack v2 - University of Tsukuba

Two artefacts derived from SoccerTrack v2 are committed to this repository:

- `model/validation/soccertrack/118575_1st.tracks.json.gz` - ground-truth tracks.
- `model/validation/nvidia_fixture.npz` - **14 WebP-encoded image tiles decoded out of
  `118575_panorama_1st_half.mp4`**, the detector regression fixture read by
  `model/engine/nvidia.py`. These are frames of the source match video, not a statistic
  derived from it.

Because the second is imagery, this credit must travel with it wherever it goes:
a detection overlay pasted into a slide or a web page carries the same
obligation as the file. The attribution is also recorded in [`NOTICE`](../NOTICE).

- Source: https://atomscott.github.io/SoccerTrack-v2/
- Licence: CC BY 4.0 - https://creativecommons.org/licenses/by/4.0/
  (the source code in that repository is MIT, separately)
- Credit: Scott et al., SoccerTrack v2, University of Tsukuba, CC BY 4.0.
- Modifications: frames cropped to 1024x870 tiles and re-encoded to WebP at
  576 px; tracks converted to this project's JSON schema.

---

## ssac26 vision layer ("Wide Open Gazes")

Vendored in `data/vendor/ssac26/` with its LICENSE retained.

- Source: https://github.com/USSoccerFederation/ssac26_visual_exploratory_behavior
- Licence: BSD 3-Clause
- Credit: J. Bekkers / UnravelSports.
- The attribution is also recorded in [`NOTICE`](../NOTICE).

---

## Machine-learning models

Weights carry their own licence, separate from the code that trains or serves
them.

**RF-DETR Medium** (`rfdetr-medium-576.mlpackage`, exported to CoreML) -
Apache-2.0, Roboflow. Nano/Small/Medium/Large only: XL and 2XL ship under
Roboflow PML 1.0 and are refused at runtime by a variant whitelist in
`model/engine/detectors.py`, which raises rather than downloading them.

**Ball detector** (`model/validation/out/ball/run/checkpoint_best_ema.pth`, 2026-09-19)
- RF-DETR Nano (Roboflow, Apache-2.0) finetuned on
`martinjolif/football-ball-detection`. That dataset tags itself CC BY 4.0 but is a
Roboflow re-export of the **DFL Bundesliga Data Shootout** clips - six of its fourteen
clip ids are `model/validation/train_roles.py`'s own hold-out tuples. This entry is a
refusal, not a credit: **the checkpoint is not distributed**, because DFL data is
competition-use-only. It is wired into no app path; the training copies were deleted
on 2026-09-19 and the code that would re-fetch them raises.

**RoleNet** (`role_clf.pt`) - ResNet-18 (torchvision, BSD-3-Clause) finetuned on
labelled crops from `martinjolif/football-player-detection`, which is **CC BY
4.0** and re-exports Roboflow's football-players-detection. Attribution is
required wherever the model is distributed:

> Training data: martinjolif/football-player-detection (CC BY 4.0), derived from
> Roboflow football-players-detection.

Deliberately NOT derived from SoccerNet: that footage is NDA-gated and
research-only however a re-uploader tagged it.

**YuNet** (`face_yunet.onnx`) - face detector from the OpenCV Model Zoo, fetched
at first use by `app/server/headshots.py` and used only to centre the crop of
publicly sourced photographs. It performs no identification or matching.
Upstream: https://github.com/opencv/opencv_zoo. The licence is believed to be
MIT; the download is a bare `.onnx` with no licence file travelling with it, so
this should be confirmed against the Zoo's per-model licence before the model is
ever bundled rather than fetched.

---

## PyInstaller

Used to freeze the FastAPI backend into the desktop application. GPL-2.0, with
an explicit Bootloader Exception that permits embedding the bootloader in an
executable under any licence:

> In addition to the permissions in the GNU General Public License, the authors
> give you unlimited permission to link or embed compiled bootloader and related
> files into combinations with other programs, and to distribute those
> combinations without any restriction coming from the use of those files.

The embedded bootloader is unmodified.

---

## certifi and tqdm

Both redistributed unmodified under the Mozilla Public License 2.0 (tqdm is dual
MPL-2.0 / MIT). MPL is file-level copyleft: it reaches the files it covers and
imposes nothing on this project's own source. Source is at
https://github.com/certifi/python-certifi and https://github.com/tqdm/tqdm, and
a copy of MPL-2.0 ships in the application bundle.

---

## Apache-2.0 components

Redistributed unmodified: PyTorch and torchvision, transformers,
huggingface_hub, tokenizers, safetensors, RF-DETR, trackers, supervision's
Apache-licensed parts, class-variance-authority, TypeScript. None of them ships
a NOTICE file, so Apache-2.0 section 4(d) adds nothing beyond this listing and a
copy of the licence.

---

## The permissive long tail

Several hundred further npm and PyPI packages are redistributed under MIT, ISC,
BSD-2-Clause, BSD-3-Clause, 0BSD, Blue Oak 1.0.0, Python-2.0 and CC-BY-4.0. Each
requires its copyright notice and licence text accompany the distribution. That
listing should be generated at build time (`license-checker`, `pip-licenses`)
rather than maintained by hand here, and shipped as `licenses/` in the bundle.
