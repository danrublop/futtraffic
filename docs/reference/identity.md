# Identity

**Summary.** `tracks.raw.json` and the video in; `tracks.id.json` out (tracklets merged into groups,
optional OCR shirt numbers), then `tracks.json` after operator corrections are replayed.

Detection, association and stitching are [the detection and tracking reference](detection-tracking.md).
Accuracy lives in the ledgers (`ledger-vision.md`,
`ledger-output.md`, [`assumptions.md`](../ledgers/assumptions.md));
history and measurements on the evidence page.
`engine-2026-09 (not published)` and `pipeline-2026-09 (not published)` point here.

<a id="entry-point"></a>
## Entry point

`app/server/jobs.py :: _try_identity(d: Path, st: dict, cal)`, called from `_pipeline` after
tracking when `tracks.id.json` is absent and `st["identity_done"]` is unset. It calls
`app/server/jobs.py :: apply_identity(identity, tracks, video, cal)`, which builds
`model/engine/identity.py :: get_identity_strategy(name, cal=cal)` and calls
`.assign(tracks, video_path)`. `model/validation/st2_product.py` and `check.py identity` score this
same function.

### Order

1. `apply_identity` resolves the strategy name from env `FUTTRAFFIC_IDENTITY` (default in the
   `os.environ.get` call: `assisted`). Only `cal` is passed: **no roster, no match id**.
2. `AssistedIdentity.assign`: `scan_video` (one decode pass, OCR votes) when the video exists;
   `merge_tracklets`; one decision per merged group (`vote_number`); `_drop_duplicates`; the
   `min_conf` gate; one-digit OCR numbers withheld.
3. `apply_identity` rewrites each detection ([Outputs](#contract)).
4. `_try_identity` writes `tracks.id.json`, `identity_report`, and `identity_note` when the run
   failed or degraded; `_pipeline` sets `identity_done`.
5. `_pipeline` runs `_derive_tracks` when any input is newer than `tracks.json`.

`POST /api/matches/{jid}/retry` (`app/server/app.py :: retry`) clears `identity_done`, so a failed
identity stage is re-attempted; a finished one left `tracks.id.json` and is not re-run.

Corrections: `app/server/app.py :: post_identity` (`POST /api/matches/{jid}/identity`) ->
`jobs.py :: apply_corrections(jid, payload)` -> `_rebind` (roster) and `_derive_tracks`, then
re-queues the job.

<a id="merge"></a>
### Merge

`identity.py :: merge_tracklets(by_id, fps, cal=None, max_gap_s=MAX_GAP_S)`. Positions are feet
projected to pitch metres by `cal.to_pitch` (`_positions`). `_pair_cost` refuses a pair across
teams, with a gap `<= 0` or above `max_gap_s`, or an implied speed above `stats.py :: MAX_SPEED`;
otherwise the cost is a weighted mean of speed and gap. `_mutual_best` keeps a pair only if each is
the other's best, under `MAX_COST`, and beats the runner-up by `MERGE_MARGIN` in both directions.
Up to `MAX_PASSES` passes, one merge per tracklet per pass. Group id = smallest member id. No
appearance term; `_selfcheck_no_appearance` fails if one returns
(decision).

<a id="ocr"></a>
### Jersey OCR

`scan_video` ranks each track's sightings by `facing_away` (displacement against the camera centre,
in pitch metres), then box height, and keeps `samples_per_track`; a track with no box height is
sampled evenly across its length. It crops `torso_box` (the `TORSO_TOP`..`TORSO_BOT` body band,
**head excluded**: never cropped, embedded or stored) on the **flattened canvas**
(`cal.undistort_frame`), filters with `legible`, reads with `get_ocr_backend` (paddle, then docTR
recognition-only; `None` if neither imports) and parses with `_as_jersey` (one or two digits,
never zero). A chest read is weighted `FRONT_WEIGHT`. `vote_number` votes per merged group: the
winner's share of confidence mass, discounted below `MIN_OCR_VOTES` reads. A group keeps its number
only at `confidence >= min_conf`, an OCR number only when it has two digits, and only when no more confident group of the
same team voted the same number.

<a id="corrections"></a>
### Corrections

`POST /api/matches/{jid}/identity` takes one JSON object:

| key | type | meaning |
|---|---|---|
| `assign` | `{str tid: {jersey?, team?, player?} \| null}` | per track id, replaced whole; `null` clears it |
| `merge` | `[[keep, drop], ...]` | `drop` becomes `keep` |
| `split` | `[{"track": tid, "at_frame": f}, ...]` | frames `>= f` of `tid` get a new id |
| `exclude` | `[tid, ...]` | detections dropped (referees, ball kids, phantom boxes) |
| `undo` | `[tid, ...]` | remove every correction naming `tid` (below) |
| `operator_seconds` | number | added to the job's total |

`apply_corrections` accumulates into `corrections.json` rather than replacing:

- **assign** by key; a `null` value is stored and replays as nothing.
- **merge** de-duplicated. A merge whose `keep` already resolves to its `drop` (a self-merge, or one
  closing a cycle) is **refused** with `ValueError`. `_merge_map` also skips such an edge on replay,
  for a file saved before the refusal existed.
- **split** de-duplicated on `(track, at_frame)`. Each new split is stored with a **stable
  `new_id`**, minted above every id the base file or any stored split can hold, and `members`: the
  track plus every id merged into it at that moment, so a split after a merge moves the merged-in
  frames too. A split before any tracks file exists is refused. A legacy entry without `new_id`
  replays with the old positional id (base max + 1 + list index).
- **exclude** as a sorted set.
- **undo** removes, for each id, its `assign` entry, its `exclude` entry, every merge naming it on
  either side, and every split whose `track` or `new_id` is that id (so undoing a split's minted
  id puts its frames back on the original track). `_rebind` unbinds the id in every roster file.

Then `apply_corrections` writes `corrections.json` atomically, calls `_rebind`, adds
`operator_seconds`, sets `st["corrections"]`, and, when `tracks.raw.json` exists, runs
`_derive_tracks`, deletes `match.json` and re-queues the job (to `calibrating` without a
calibration). It returns `status_view`.

`_derive_tracks` replays `corrections.json` onto `tracks.id.json` (else `tracks.raw.json`) from
scratch; the base file is never mutated. Per detection, in order:

1. **split**: the latest split of the id at or before the frame wins; a split of a split chains.
2. **merge**: `drop -> keep`, chains flattened.
3. **exclude**: the detection is dropped.
4. **assign**, on the final id: `jersey` sets `jersey`, sets `jersey_source` to `operator` and
   removes `jersey_conf`; a non-null `team` replaces the model's.

<a id="roster"></a>
### Roster

`model/engine/roster.py :: Roster`: one team's squad plus `match_id -> track_id -> {player_id,
confidence, source}`, one file per team. `_rebind` writes an `assign` entry's `player` as a
`manual` binding **only into the job's own team roster** (`st["team"]`, else
`DEFAULT_TEAM`); an assignment without a `player`, and every `undo` id, is unbound in every roster
file. `player_id` is the name slug (`_slug`, accent-folding), else `<team-slug>-<number>`.
`tracks_for(player_id)` is the season join.

## Inputs

- `tracks.raw.json` ([detection-tracking contract](detection-tracking.md#contract)): feet `x, y`
  and box `w, h` in **image pixels of the detector's canvas** (flattened on remap footage), `f` the
  original frame index, `fps`, `team`.
- The uploaded video (raw frames; flattened here through `cal`). Absent: merge only, no OCR.
- `cal`: the job's loaded calibration (`to_pitch`: canvas px to pitch metres, `H`, optional
  `undistort_frame`, `camera_centre_m`). Required by signature.
- `corrections.json` ([schema above](#corrections)) and `DATA_DIR/rosters/*.json`.

<a id="contract"></a>
## Outputs

| file / key | written by | content |
|---|---|---|
| `tracks.id.json` | `_try_identity` | the raw contract plus per detection: `raw_id` (int, pre-merge id), `id` rewritten to the group id, and when a number passed: `jersey` (int), `jersey_conf` (float 0-1, three places), `jersey_source` (`ocr` on the job path; the strategy's `source`); `team` |
| `st["identity_report"]` | `_try_identity` | `strategy`, `tracklets_before/after`, `groups`, `one_digit_withheld`, `tracks_assigned`, `tracks_needing_human`, `groups_needing_human`; when scanned `crops_attempted`, `crops_legible`, `legible_frame_rate`, `ocr_backend`, `ocr_reads`, `ocr_reads_facing_away`; `warnings` only when degraded |
| `st["identity_note"]` | `_try_identity` | the exception text when identity failed or was unavailable, or the joined `warnings` of a degraded run (no OCR backend, camera probe failed) |
| `tracks.json` | `_derive_tracks` | corrected tracks, same schema; an operator's number carries `jersey_source: "operator"` and no `jersey_conf`; deletes `tracks_summary.json` |
| `corrections.json` | `apply_corrections` | `assign {str tid: {jersey?, team?, player?} \| null}`, `merge [[keep, drop]]`, `split [{track, at_frame, new_id, members}]`, `exclude [tid]`, `updated` (epoch s) |
| `st["operator_seconds"]`, `st["corrections"]` | `apply_corrections` | cumulative operator seconds; count of assign + merge + split entries |
| `DATA_DIR/rosters/<team-slug>.json` | `Roster.save` (atomic) | `team`, `season`, `players {id: Player}`, `bindings {match: {str tid: {player_id, confidence, source}}}` |

`PlayerId.source` is one of `manual`, `roster`, `ocr`, `auto`, `unassigned`. Downstream,
`players.jersey` is tier `prov` with caveat `jersey_ocr_unvalidated`
(`model/engine/stats.py :: METRICS`, `CAVEATS`).

<a id="config"></a>
## Config

| what | where | how it reaches the run |
|---|---|---|
| strategy | `app/server/jobs.py :: apply_identity` (env `FUTTRAFFIC_IDENTITY`) | env, else the `get` default |
| calibration | `app/server/jobs.py :: _pipeline` -> `_try_identity(d, st, cal)` | explicit, required |
| roster, match id | `model/engine/identity.py :: AssistedIdentity.__init__` (`roster`, `match_id`) | **not passed**: identity-side roster reads and writes are off on the job path |
| number gate | `identity.py :: AssistedIdentity.__init__` (`min_conf`); `AutoIdentity.__init__` | signature default |
| one-digit withholding | `identity.py :: AssistedIdentity.assign` | code |
| OCR backend | `identity.py :: AssistedIdentity.__init__` (`ocr`) -> `get_ocr_backend`, `_OCR_BACKENDS` | first that imports |
| merge window | `identity.py :: MAX_GAP_S` (via `max_gap_s`) | module constant |
| merge gates | `identity.py :: W_SPEED`, `W_GAP`, `MAX_COST`, `MERGE_MARGIN`, `MAX_PASSES`; `model/engine/stats.py :: MAX_SPEED` | module constants |
| sightings per track | `identity.py :: scan_video` (`samples_per_track`) | signature default (not passed) |
| facing | `identity.py :: facing_away` (`span`, stand-still distance in the body) | signature default |
| crop | `identity.py :: TORSO_TOP`, `TORSO_BOT`, `HEIGHT_PX_FAR`, `HEIGHT_PX_NEAR`; width factor in `torso_box` | box `w, h` when present |
| legibility | `identity.py :: MIN_TORSO_PX`, `TORSO_ASPECT`, `MIN_SHARPNESS`, `MIN_NEIGHBOUR_PX` | module constants |
| vote | `identity.py :: MIN_OCR_VOTES`, `FRONT_WEIGHT` | module constants |
| fps fallback | `identity.py :: AssistedIdentity.assign` | when the tracks file has no `fps` |
| corrections body limit | `app/server/app.py :: MAX_JSON` | module constant |
| roster location | `app/server/jobs.py :: _roster_dir`, `_roster_path` | `DATA_DIR` |

`model/validation/shipped_config.py` does not resolve this stage; `check.py identity` records the
resolved values in its golden's `config`.

<a id="errors-and-refusals"></a>
## Errors and refusals

- **Never fatal.** `_try_identity` catches `Exception` and `SystemExit` (from `scan_video` on an
  unreadable video) and a failed `import identity`; it records `identity_note`, `_pipeline` sets
  `identity_done`, and the job continues on `tracks.raw.json`.
- **Degraded, not silent.** No OCR backend, or a failed camera probe (facing gate off), finishes
  the run and puts a `warnings` entry in the report and in `identity_note`.
- Without `cal`, merges run in pixel space and `MAX_SPEED` refuses every one; `_try_identity` takes
  `cal` as a required argument and the jobs self-check asserts the strategy receives it
  (evidence).
- `get_identity_strategy` raises `ValueError` for an unknown name (caught as above).
- A group under `min_conf`, a one-digit OCR number, or the less confident of two same-team groups
  voting one number is returned unassigned (`number=None`) for the operator.
- `apply_corrections` raises `ValueError` for a self or cyclic merge and for a split with no tracks
  file; malformed entries raise `KeyError`/`TypeError`. `post_identity` answers all three with 400,
  also a non-object body, bad JSON, or a key of the wrong type; 413 over `MAX_JSON`.
- `Roster.bind` raises `KeyError` for a `player_id` not in the squad; `_rebind` skips unknown
  players and unreadable roster files silently. `_roster_path` raises `ValueError` for a team with
  an empty slug.

<a id="limits"></a>
## Limits

History and measurements: Evidence: known limits.

- **Roster seeding is off on the job path.** `apply_identity` passes no roster (`roster=None`), so
  an OCR number gets no `player_id` or name. Only `_rebind` writes the roster, from operator
  assignments. Corrected 2026-09-29: even with a roster, confirmed bindings from earlier matches
  would not seed a new match (`confirmed()` is per match id); what a roster adds is naming by
  squad shirt number, and that lookup (`Roster.by_number`) has no team check, so an opponent in
  one of our numbers would get our player's name. Pre-registered, not run:
  N71.
<a id="ocr-precision"></a>
- **OCR precision is low and measured once.** At the shipped path N49 scored
  `@C 0.4375 :: docs/experiments/n49-jersey-ocr.md, group precision (one window of 118575, n=1)`,
  with every one-digit write wrong at high confidence, so no `min_conf` separates them; one-digit
  numbers are therefore withheld, and players whose true number has one digit never get one from
  OCR ([N49](../experiments/n49-jersey-ocr.md)).
- **One install, one match of OCR evidence.** Every OCR and facing figure was measured on 118575
  (one camera, one kit pairing); nothing scores the field at a second install. The registry flags
  it `jersey_ocr_unvalidated`.
<a id="welds"></a>
- Under heavy correlated fragmentation the merger has welded two players; tightening the gates did
  not remove it (evidence).
- After stitching the merge is close to a no-op on the scored sequences
  (N8-N11, N21,
  decision).
- `facing_away` is a velocity proxy: wrong for backpedalling or sideways players, blind to turning
  on the spot; `span` counts detections, not frames.
- Without box `w, h` the crop height is a linear-in-row model ([assumptions](../ledgers/assumptions.md)).
- **FIXED 2026-09-29: undo of a split's `new_id`** now removes that split
  ([corrections](#corrections)); a legacy split without a stored `new_id` still matches on `track` only.
- Two humans with one name share one `player_id` unless the caller passes one explicitly;
  `by_number` is first-match-wins (evidence).
- `Roster.bind` has no guard against overwriting a `manual` binding; callers check `confirmed()`.
- A split POST parses the base tracks file a second time to mint its id (`ponytail` in
  `apply_corrections`).

<a id="check"></a>
## How to check

| command | covers | does not cover |
|---|---|---|
| `.venv/bin/python model/validation/check.py identity --fast` | `apply_identity` at the shipped defaults on the track fixture's replay (selection, crops, legibility, vote, `min_conf`, one-digit rule) and a merge-only replay, exact against the golden; plus the three self-checks below | the decode, docTR itself, and pixels of a crop whose box did not move |
| `.venv/bin/python model/validation/check.py identity --real` | the same on the clip's pixels with docTR; re-records the crop/OCR lookup | jersey truth (its ST2 diagnostic is non-gating) |
| `.venv/bin/python model/validation/check.py identity --prove` | plants gate, weight, sort and crop mutations that must FAIL | correctness of the golden itself |
| `.venv/bin/python model/engine/identity.py` | facing sign, sighting ranking, even sampling without `h`, occlusion rejoin, crossing refusal, team block, appearance guard, strategies, vote, one-digit withholding, missing-backend warning, roster join | real footage; OCR |
| `.venv/bin/python model/engine/roster.py` | slugging, id fallbacks, bindings, persistence, pinned limitations | the job path |
| `.venv/bin/python app/server/jobs.py` | identity best effort on an unreadable video, `cal` reaches the strategy, degraded warnings reach `identity_note`, `/retry` clears `identity_done`, corrections merge/split/undo/assign/exclude, self/cyclic merge refusal, stable split ids, undo by a split's `new_id`, two splits of one track, split after merge, `_rebind` own-team binding and unbind | OCR accuracy |
| `.venv/bin/python model/engine/identity.py <tracks.json> -v <video> -c <calibration.json>` | prints the assisted report on one file | scoring |
| `model/validation/jersey_eval.py` | N49's protocol: the product's written numbers against SoccerTrack jerseys | other installs |
| `model/validation/st2_product.py` | `apply_identity` on ST2 sequences at what ships | jersey truth |

<a id="evidence"></a>
## Evidence

All on the evidence page:
known limits,
design stance,
synthetic merge purity,
appearance term,
OCR,
facing and sighting selection,
crop geometry and flattening,
calibration is required,
roster ids.
Decision: appearance term.
Experiments: [N49 jersey OCR](../experiments/n49-jersey-ocr.md),
N8-N11, N21.
Plan: identity scope (superseded).
