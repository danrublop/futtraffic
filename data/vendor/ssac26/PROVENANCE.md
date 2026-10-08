# `ssac26` — the vision layer from *Wide Open Gazes*

Upstream: <https://github.com/USSoccerFederation/ssac26_visual_exploratory_behavior>
at `9c4b37c` ("submission"), committed 2025-12-05, read 2026-09-11.

Paper: Bekkers, J. *Wide Open Gazes: Quantifying Visual Exploratory Behavior in
Soccer with Pose Enhanced Positional Data.* Unpublished manuscript, 2026 — the
bibtex in the upstream notebooks says `@unpublished`, so **there is no published
paper to check the code against**. Everything here is read off the code.

## Licence

`LICENSE` is the BSD 3-Clause text verbatim, plus an attribution paragraph and an
explicit redistribution grant. GitHub's API reports `NOASSERTION` for it — that is
the detector failing on the appended paragraphs and on the copyright line, which
still carries its template brackets (`Copyright (c) 2025 [UnravelSports]`). Read
`LICENSE`; it is BSD-3, it is commercial-safe, and it requires the notice be kept,
which is why the file is here. Same lesson as
`licence-audit (not published)` §2.5: `NOASSERTION` means
*classify it yourself*, not *unlicensed*.

## What is here, and what was changed

| file | upstream | change |
|---|---|---|
| `vision.py` | `src/vision.py` | `from .plot import plot_matrix` and the `plot()` method that used it are removed — they pull `mplsoccer` and `matplotlib`, which this repo does not depend on. Nothing else is touched. |
| `helpers.py` | `src/helpers.py` | verbatim |
| `LICENSE` | `LICENSE` | verbatim |
| `ssac26_angles.csv` | `data/ssac26.parquet` | re-encoded as CSV (this repo has no parquet reader) and trimmed to the columns the model consumes. Constant columns are in the header comment. Floats rounded to 4 dp. |

Not vendored, because nothing here uses them: `src/dataset.py`, `src/plot.py`,
`src/colors.py`, `src/schema.py` (they need `polars`, `pandera`, `mplsoccer`,
`matplotlib`), the two notebooks, and `assets/`.

## What the upstream repo does NOT publish

The paper's results rest on four things that are not in it:

1. **The pose estimation.** `head_angle`, `shoulders_angle` and `hips_angle`
   arrive pre-computed in the data. No model, no code, no accuracy figure.
2. **Pitch control and pitch value.** The README combines vision maps with both;
   neither is in `src/`.
3. **The XGBoost classifier, the ablation and the SHAP analysis** behind Figure 2.
4. **The dataset.** `data/ssac26.parquet` is **116 frames — 4.6 s of one match**
   (`game_id d9d1d8e0`, period 2, 23 tracked objects, one Argentina–Colombia
   possession). It is a demo for the animation, not the study corpus.

So what is published is the vision-map *formulae*, in runnable form, plus one clip
to watch them on. That is still the useful half for us: the formulae are the part
we would otherwise have to re-derive, and
[`model/validation/vision_eval.py`](../../../model/validation/vision_eval.py) uses the clip as the
only pose ground truth in this repo.

## One upstream behaviour to know before you reuse this

`smoothing_multiplier` is a render-resolution knob for the field of view and a
*physical* one for occlusion: `__post_init__` scales `human_width` by it,
`_apparent_width_vec` uses the scaled width in metre space, and `_occlusion_vec`
divides it back out when placing the shoulders. The two halves of the occlusion
model therefore disagree at any multiplier but 1, and an occluder's angular size
grows with the plot's resolution. Measured, and why `model/validation/vision_eval.py`
runs at multiplier 1:
`research-visual-exploratory-behavior (not published)` §1.
