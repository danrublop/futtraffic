# N43: a far-third tile pass nearly doubles far-side detection — and fails its gate for two fixable reasons

**2026-09-18, branch `n30-operator-clicks`.** Protocol [`model/validation/n43_far_tiles.py`](../../model/validation/n43_far_tiles.py),
committed at `7db51ad` before any pass (the plan was re-cut once BEFORE running, disclosed in the docstring). Three
dev clips; the held-out four were not run because the L1 gate was not cleared. Follows
[`n40-association.md`](n40-association.md), [`n41-conf.md`](n41-conf.md), [`n42-merge.md`](n42-merge.md);
the mechanism is `model/engine/detectors.geometry_tiles`' far pass, never before measured on the raw panorama.

## What ran

Same detector (RF-DETR Medium, CoreML, 576 input, conf 0.5), same merge, same pitch prior, stride 2. T0 = the shipped
uniform 1024 grid (5 tiles per 4096x1080 frame). T1 = the same 5 tiles plus 15 tiles of about 200 px over the far
third of the pitch (a player there lands ~96 px in the model input instead of ~20): 4x the detector cost per frame,
and in wall-clock about 2.5 h per four-minute clip on this machine against 8 min for T0.

## Recall by class, three dev clips (IoU >= 0.5 against the MOT boxes, after the prior)

| | occluded far | occluded mid | occluded near | isolated far | isolated mid | isolated near | non-GT boxes / frame |
|---|---|---|---|---|---|---|---|
| T0 shipped | 0.4285 | 0.5614 | 0.7085 | 0.8146 | 0.9621 | 0.966 | 2.651 |
| T1 far pass | 0.771 | 0.7282 | 0.6338 | 0.9923 | 0.9554 | 0.9183 | 12.793 |

    @M 0.4285,0.5614,0.7085,0.8146,0.9621,0.966,2.651,0.771,0.7282,0.6338,0.9923,0.9554,0.9183,12.793 :: .venv/bin/python model/validation/n43_far_tiles.py --show

**Gate: T1 stops at L1** (far recalls rise by far more than the 0.05 asked, but non-GT boxes per frame rise by 10.1
against the +0.5 allowed). No product run, no held-out run.

## What the extra and the lost boxes are (118575, `--why`)

- Of the 7.15 boxes per frame T1 adds, **43% are true players the shipped grid missed**, 8% are near-miss duplicates of
  a player (a second box at a different scale that fails IoU 0.5), and **49% match no ground-truth box - and their
  bottom rows sit at median row 155, on the far touchline itself.** The ground truth holds the 22 players only; people
  standing on or just behind the far line (assistant referee, substitutes, spectators) are inside the prior's 3 m
  margin and the magnified tiles now see them. They are not hallucinated boxes; they are the wrong people.
- T1 also LOSES 0.65 shipped boxes per frame, at bottom rows 200-333: the far strip ends near row 300, and where a
  far tile and a uniform tile both see a player at the strip's edge the merge keeps the magnified tile's differently
  cut box, which then misses IoU 0.5 with the ground truth. That is the near/mid recall drop in the table (0.966 ->
  0.918 near isolated): a cross-scale merge artefact, not a detection loss.

    @M 7.15,0.43,0.08,0.49,155,0.65 :: .venv/bin/python model/validation/n43_far_tiles.py --why 118575

## Reading

This is the first lever since N40 that moves the number that matters: far-third occluded recall 0.43 -> 0.77 and
isolated 0.81 -> 0.99. It is not shippable as run, for two reasons that are engineering rather than physics: the
pitch prior's margin admits the touchline crowd once they are detectable (a margin of ~0.5 m beyond the far line for
far-tile boxes, or the officials' own class, would keep the players and drop the crowd), and the merge does not know
which scale to trust at a strip boundary (prefer the uniform tile's box wherever both tiles see the whole player).
Each is a follow-up with its own pre-registration; neither is a tuning of this one. The compute cost (4x tiles, 15x
wall-clock here) is the other price, and it says the far pass is the measurement of what resolution buys, and the
fine-tune at the product's own tile geometry (`plan-detector-finetune.md`) is the way to
buy most of it without the tiles.

**Not shown:** the product-path metrics (HOTA, distance, HSR) under the far pass - by the protocol they are not run
when L1 fails; and anything about the held-out clips.
