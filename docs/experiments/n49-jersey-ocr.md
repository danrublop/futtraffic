# N49: the product writes jersey numbers, and nothing has ever scored them

2026-09-20, branch `jersey-ocr-score`. Protocol: the docstring of
[`model/validation/jersey_eval.py`](../../model/validation/jersey_eval.py), committed
before the run. Follows
`jersey-ocr-measured-back-gate-is-the-fix` (internal note),
`jersey-ocr-four-models-all-fail` (internal note) and
`jersey-numbers-come-from-the-operator` (internal note),
which this page corrects.

## 1. What actually ships, traced today

`app/server/jobs.py:1177` calls `_try_identity` on every job. It imports
`identity` and hands the job's `tracks.raw.json` to `apply_identity`
(`jobs.py:1009`), which builds the strategy from
`os.environ.get("FUTTRAFFIC_IDENTITY", "assisted")` — nothing sets that variable
anywhere in the repo, so the default is what runs.

| stage | what ships | file:line |
| --- | --- | --- |
| strategy | `AssistedIdentity`, `min_conf=0.6`, `ocr="auto"`, **no roster passed** | `jobs.py:1014` |
| backend | `get_ocr_backend("auto")` tries paddle then doctr; paddleocr is absent, **python-doctr 1.1.0 is installed**, so `recognition_predictor` (recognition only, no text detector) runs | `identity.py:400`, `identity.py:431` |
| crop | `torso_box`: the 20–50 % body band of the DETECTOR's box, head excluded, taken on the FLATTENED canvas (`cal.undistort_frame`) | `identity.py:339`, `identity.py:552` |
| which crops | best `samples_per_track=60` sightings per track, ranked facing-away first then box height; then `legible()` (torso ≥ 12 px, aspect, sharpness ≥ 40, nearest neighbour ≥ 18 px) | `identity.py:517`, `identity.py:371` |
| vote | per MERGED tracklet: winner's share of the confidence mass × `min(1, count/3)`; a chest read is weighted `FRONT_WEIGHT=0.45` | `identity.py:674`, `identity.py:640` |
| gate | group keeps its number only if `confidence >= 0.6`; two groups of one team voting the same number → the less confident is dropped | `identity.py:815`, `identity.py:839` |
| write | `x["jersey"] = p.number`, `x["jersey_conf"] = conf` on **every detection of the group** | `jobs.py:1029` |

**Is a wrong OCR number distinguishable from an operator-typed one downstream?**
Mostly no. `PlayerId.source` (`"ocr"` / `"roster"` / `"manual"`) exists and is
**dropped at `jobs.py:1029`** — only the number and a float survive.
`jersey_conf` reaches exactly one consumer, `app.py:509`'s track digest, and
dies there: `stats.py:679` reads `jersey` alone into `match.json`, so the report
tables, the film overlay, the notepad and `/season`'s `by_match.jersey` render
`#9` identically whether a human typed it or docTR voted it off a 7 px crop.
Two consequences worth naming: `player-view.tsx:158` prints "Read by jersey OCR"
over operator-typed numbers too, and `_derive_tracks` (`jobs.py:955`) overwrites
`jersey` on a correction but leaves the **old** `jersey_conf`, so after an
operator fixes a wrong read the digest reports the discarded guess's confidence
as the confidence of the human's number.

The season/profile layer is the one place that got it right: `jobs.py:2303` and
`jobs.py:1813` accept only bindings with `source == "manual"`, and in the
shipped configuration `jobs.py:1014` passes no roster at all, so
`_resolve_roster`/`_persist` early-return and an OCR number acquires **no**
`player_id` and no name. A wrong number therefore mislabels a match page, not a
season profile.

## 2. Pre-registration

Fixed and committed before the first run; the full text is the harness
docstring. In short:

- **population** — 118575 1st half panorama, frames 15,000–29,999 (minutes
  10:00–20:00). **Drop: 79.1 % of the half's frames are not scanned**:

      @D 0.791 :: 1 - 15000 / 71850

  Stated direction of the bias: fewer sightings
  per merged group than a full half, and `vote_number` discounts by read count,
  so **coverage here is a floor**; precision has no obvious direction. The
  window is 10 minutes because the full half costs ~25–40 min of decode, remap
  and OCR and the run budget is ~20.
- **tracks** — the product job's own `tracks.raw.json`, with the detector's box
  `w/h` rejoined from its `dets.cache.json` on the foot point (exact to 2e-4 px,
  `team_eval.build`'s join). Without the box, `torso_box` uses its broadcast
  height fallback, which its own docstring says is wrong in both directions on a
  panorama — so an unjoined run would score a crop geometry the product does not
  use.
- **code path** — `jobs.apply_identity`, the function a job runs, unmodified.
  n=1 configuration.
- **truth** — SoccerTrack's own `jersey` field, 22 ids, one constant number each
  (5 and 15 both wear 14, one per team). Join: feet → `cal.to_pitch`, greedy
  nearest GT player inside **2.0 m** per frame, modal GT id over the group.
  A group needs ≥ 20 matched detections and ≥ 60 % agreement or it is
  **NO-TRUTH** and leaves precision — counted and reported, never dropped
  silently. Officials have no GT row, so a number written on one lands there.
- **metrics** — P1 group precision, P2 detection-weighted precision, C1 player
  coverage (of 22), C2 share of detections carrying a jersey, E error profile
  (digits, box height, distance from camera), S the min_conf cost curve at
  0.6/0.7/0.8/0.9 recomputed from one run's cached votes through the product's
  own `vote_number` and `_drop_duplicates`.
- **the bar** — "safe to show a user" is **P1 ≥ 0.90 and P2 ≥ 0.90 with
  C1 ≥ 8 of 22**. A number with no provenance on any surface but the setup grid
  relabels a physical profile when it is wrong, and retyping it is cheap; under
  9 in 10 right the proposal costs more operator attention than it saves.

## 3. Result — the shipped gate writes a wrong number more often than a right one

One run, 741 s, 2026-09-20. Every number below is one command over its cached
artefacts in `model/validation/out/n49_jersey/`:

    @M 0.4375,0.6768,6,11,0.0732 :: .venv/bin/python model/validation/jersey_eval.py --score
       118575 1st half, frames 15000-29999, jobs.apply_identity/assisted, docTR
       1.1.0, min_conf 0.6, the job's remap calibration, n=1 configuration.
       Drop: 79.1% of the half's frames not scanned; 0 detections lost to the
       box join; 5 of 21 written groups have no GT player and leave precision.

| | value | what it is |
| --- | --- | --- |
| groups | 681 (from 893 tracklets) | one identity decision each |
| **written at min_conf 0.6** | **21** | 3.1 % of groups get a number |
| of those, no GT player | 5 (23.8 %) | officials and stubs; excluded from P1/P2 |
| **P1 group precision** | **0.4375** (7 of 16) | more wrong than right |
| **P2 detection-weighted** | **0.6768** | the long tracks are the right ones |
| **C1 players read correctly** | **6 of 22** | 11 of 22 got a number of some kind |
| players given a WRONG number | 8 | a real player, mislabelled |
| **— of which, cleanly right** | **3** | only-right / both / only-wrong = 3 / 3 / 5 |
| C2 detections carrying a jersey | 0.0732 | 7.3 % of the window's rows |
| legible acceptance | 0.3233 (10,974 of 33,947 crops) | the heuristic, on real footage |
| reads that parsed to a number | 2,072 | 1,820 of them (87.8 %) facing away |

**THE TWO COUNTS OVERLAP, and this page did not say so until a review of
2026-09-20.** 6 correct and 8 wrong reads as 14 distinct players, but only 11 got a
number at all: three players hold a correct group **and** a wrong one in the same
match. So the sentence the evidence supports is **3 of 22 players are cleanly right**,
3 more carry two conflicting numbers, and 5 are simply wrong — not "6 right, 8 wrong".
The partition is printed now rather than left to be inferred.

@M 3,3,5,11 :: .venv/bin/python model/validation/jersey_eval.py --score
   `players_partition.only_right`, `.both_a_right_and_a_wrong`, `.only_wrong`, then
   `C1_players_any_number`; the three sum to it. Same run, window and min_conf 0.6 as
   the table above. n=1 configuration.

**The whole error is single digits.** Of the 16 judged writes, 9 are a
one-digit number and **0 of 9 are right**; 7 are two-digit and **7 of 7 are
right**. The wrong numbers are 1×2, 2×2, 3×2, 4×2, 9×1 — and **4 of the 9 wrong
writes are one digit OF the true number**: "1" for 18, "2" for 24, "9" for 39,
"1" for 10.

    @M 0.0,1.0,4 :: .venv/bin/python model/validation/jersey_eval.py --score
       `digits.right_when_1_digit`, `digits.right_when_2_digit`,
       `wrong_write_is_a_digit_of_the_true_number`; same run and window as above.

This is the failure `jersey-ocr-four-models-all-fail` (internal note) predicted from the other side: at a ~25 px number band a general recogniser returns half a number,
or a fold, and `_as_jersey` accepts it because one digit is a legal shirt.

**Wrong reads are not low-confidence reads.** Group 1350 wrote `4` over a
2,151-detection track of player 10 at **confidence 1.0**; group 1516 wrote `1`
over player 10 at 1.0; group 1584 wrote `3` over player 12 at 1.0. Confidence is
the winner's share of the vote mass — a track whose reads agree on the same
wrong digit scores 1.0.

Error profile: wrong writes sit on smaller, further players — mean box height
**42.9 px** wrong vs **61.2 px** right, mean distance from the camera **71.2 m**
wrong vs **53.4 m** right.

    @M 42.9,61.2,71.2,53.4 :: .venv/bin/python model/validation/jersey_eval.py --score
       `box_h_px`, `camera_m`; camera from the calibration's own centre, means
       over each group's GT-matched positions.

**Verdict against the pre-registered bar (P1 ≥ 0.90, P2 ≥ 0.90, C1 ≥ 8):
FAILS on all three.**

### The four options, each with its measured cost

    @M 0.4375,0.5,0.5455,0.5 :: .venv/bin/python model/validation/jersey_eval.py --score
       `min_conf_curve`, the P1 column below; one run's cached votes through
       the product's own vote_number and _drop_duplicates, NOT four runs and
       NOT a selection.

| min_conf | groups written | P1 | P2 | players right | detections with a jersey |
| --- | --- | --- | --- | --- | --- |
| **0.6 (ships)** | 21 | 0.4375 | 0.6768 | 6 | 0.0732 |
| 0.7 | 17 | 0.5000 | 0.6928 | 6 | 0.0710 |
| 0.8 | 14 | 0.5455 | 0.7310 | 5 | 0.0553 |
| 0.9 | 13 | 0.5000 | 0.6837 | 4 | 0.0471 |

1. **Keep as is** — 8 of 22 players carry a wrong number (5 of them wrong and nothing
   else, 3 alongside a correct group), only 3 are cleanly right, and nothing downstream
   marks any of it as a guess (§1). Fails the bar.
2. **Raise `min_conf`** — buys at most **+0.11 P1** and gives up a third of the
   writes; at 0.9 precision falls back to 0.50 with two fewer players. The gate
   cannot work, because the wrong reads arrive at confidence 1.0.
3. **Back-facing crops only** — **already effectively in place**: the sampling
   rule ranks facing-away first, and **87.8 % of the reads the product spends
   are of a back** (`identity_report.ocr_reads_facing_away` / `ocr_reads`). A
   hard gate can remove at most the remaining 12.2 %. That bound is an INFERENCE from
   the read shares, not a measurement — options 2 and 4 were scored and this one was
   not, although `_curve` already re-votes the cached votes and could have scored it.
   Treat "it cannot fix a 0.44" as the expectation it justifies, not as a result.

       @M 1820,2072 :: .venv/bin/python model/validation/jersey_eval.py --score
          reads that parsed to a jersey, and how many were of a player the
          facing estimate called away from the camera.

4. **Switch the write off** — costs the 6 of 22 players (7 groups) the OCR got
   right in this window — of whom only 3 are cleanly right — whose longest groups
   carry 1,238–3,509 detections.

### Recommended: keep the read, gate the WRITE to two digits

POST HOC, stated as a proposal and not as a result — this arm was chosen after
seeing which writes were wrong, so it is the next run's hypothesis, not this
run's finding. On this window it writes 7 numbers, **all 7 correct**: P1 1.00,
P2 1.00, 6 of 22 players, **0 players mislabelled**. Its cost is structural and
known: **4 of 22 players wear a single digit**, so they can never be proposed,
and `identity.py` already caps that ceiling at 18 of 22.

    @M 7,1.0,1.0,6,0.1818 :: .venv/bin/python model/validation/jersey_eval.py --score
       `post_hoc_two_digit_only` (writes, P1, P2, players correct) and
       `digits.true_1_digit_share`; post hoc, same single run.

That clears the precision half of the bar and still misses the coverage half
(6 < 8) — on a window that was pre-registered as a coverage FLOOR, so the test
is a full-half run. **Until that runs, the product should not show an
OCR-proposed number as if a human typed it.** Two changes, in this order:

1. Carry `PlayerId.source` through `jobs.py:1029` and render it. A guess that
   says it is a guess costs an operator a glance; a guess that does not costs a
   physical profile.
2. Land the two-digit write gate behind its own pre-registration, and re-run
   this harness on a second match (118576/118577 have the same GT shape) before
   quoting either number as a property of the method rather than of 118575.

### What this run cannot say

- **One match, one 10-minute window, 16 judged writes.** 0-of-9 versus 7-of-7 is
  a strong split on a small sample; nothing here bounds it on another kit, camera
  or number font.
- The tracks are the 2026-08-16 job's, not a tracker re-run at today's config —
  identity consumes `tracks.raw.json`, so this is the right INPUT, but a newer
  tracker would fragment differently, and fragmentation sets how many reads a
  group gets to vote with.
- Two groups both wrote 55 and both were right: `_drop_duplicates` keys on
  `(team, number)`, so a team-label disagreement between two fragments of one
  player keeps both. Not scored here; it is the association problem, not OCR's.
