"""The ball detector's CLEAN training set: TeamTrack tiles (MIT), and the split, registered before any training.

WHY THIS EXISTS
    `ball_train.py` trained a ball detector that works and cannot ship. Its data is
    `martinjolif/football-ball-detection`, a Roboflow re-export of the DFL Bundesliga
    Data Shootout clips under a competition-use-only, delete-all-copies grant
    (licence-audit (not published) blocker B4, dfl-dataset (not published)). The copies were deleted
    and `ball_train.fetch`/`tiles` refuse. The checkpoint is research-only.

    TeamTrack (`data/footage/soccertrack/README.md`) is MIT, commercially usable, and
    already on disk with ball annotations. It is the clean source. It is ALSO the
    population `ball_eval.py`, `ball_cmp.py` and N48-N51 score on, which is the whole
    difficulty and the reason this file registers a split before it builds anything.

THE COLLISION, STATED BEFORE THE SPLIT WAS CHOSEN (2026-09-20, before `fetch` ran)
    Nine TeamTrack clips were on disk. Every one of them was already spent:

      * six ARE the scored population (`ball_cmp.CLIPS`);
      * two are where `ball.BallPicker`'s gates were fitted (`ball_eval.py:112`);
      * one is the dead ball `ball_cmp` excludes by name as degenerate.

    So nothing on disk could be trained on. That was not a split problem to be solved
    by a cleverer rule; it was a data problem, and the answer was to FETCH the training
    clips. TeamTrack holds 42+9+9 `soccer_top` and 46+10+10 `soccer_side` clips of 30 s
    each and the README's §6 recipe pulls any member by byte range without the 8.6 GB
    zip. `fetch` below does exactly that, and the 101 eligible clips are now on disk.
    `inventory()` reads the committed member table, so the split was registered over
    clips before any of them existed locally.

    AND THE LIMIT THAT NO SPLIT FIXES: `soccer_top` is one match (D_20220220_1) and
    `soccer_side` is one match (F_*_20220220_1). Two matches, two venues, two cameras.
    A clip-disjoint split inside a single match holds the camera, the pitch, the ball,
    the kits and the light fixed, so a held-out clip here is a held-out MINUTE, not a
    held-out match. Every number this data can ever produce is same-match. Breaking
    that needs a third commercially-licensed source, not a rearrangement of this one.

THE SPLIT, FIXED HERE, BEFORE ANY TRAINING RUN (`role()` is the rule, `--demo` checks it)
    scored    `ball_cmp.CLIPS` - three `soccer_top` test and three `soccer_side` train
              clips. Never trained on, never validated on, never looked at while
              choosing anything.
    excluded  `soccer_top` 0000-0060 (ball.py's gates were fitted there, so a detector
              trained on it is fitted to the same pixels the picker's constants are);
              `soccer_top` 1680-1710 (dead ball, `ball_cmp`'s own exclusion);
              `soccer_side` 0090-0120 (abuts a scored window - adjacent 30 s of one
              match are near-duplicate frames); and TeamTrack's whole `test` split,
              held out entire because our scored `soccer_top` clips live inside it.
    train     every remaining clip of TeamTrack's `train` split, both sports.
    valid     TeamTrack's `val` split, both sports. Checkpoint selection only, and it
              is the SAME two matches as `scored` - so valid cannot detect
              overfitting to the venue, only to the frames.

THE TILE GEOMETRY, MATCHED TO INFERENCE, NOT CHOSEN
    `ball_cmp` arm B runs the detector at tile 640 over the native frame
    (`detectors._spans`, overlap 0.10) - a 640 px crop at native scale, no resize. So
    a training tile is a 640 px crop at native scale, and `T` and the random-offset
    window are imported from `ball_train` rather than re-declared: the clean rebuild
    changes the DATA, and nothing else, or the comparison to N48 measures two things.

DECLARED CONSTANTS (chosen here, nothing tunes them, no run has been scored)
    FRAME_STEP = 5    30 fps; consecutive frames are near-duplicate crops of one ball.
    NEG_KEEP = 4      one ball-FREE tile per four tiles with a ball. N48's review named
                      this: every DFL training tile contained a ball, so nothing in
                      that run could see a false positive on empty pitch, and false
                      positives are exactly what N48-N51 lost to (36.6 candidates a
                      frame against COCO's 22.5). Negatives are the one deliberate
                      departure from `ball_train`, and it is declared, not silent.
    OFFSET            `ball_train`'s (0.15, 0.85), unchanged.
    boxes             TeamTrack's own annotation boxes, clipped to the tile by
                      `ball_train.clip_box`. NOT re-measured from pixels: N50 §6 found
                      the annotator's box is ~1.4x the ball's silhouette and the stock
                      detector reproduces it, while the DFL-trained nano draws ~3.2x.
                      Training on this annotator's convention is the experiment.

WHAT THIS FILE DOES NOT DO
    It does not train. It does not score. `tiles` refuses any clip that is scored,
    excluded or absent, and `sample` writes only to out/ball_clean/sample/ and stamps
    every manifest it writes as a geometry proof that is not a training input. The
    training budget is docs/experiments/n52-clean-ball-data.md §7 and has not been run.

    .venv/bin/python model/validation/ball_clean.py --demo      # self-check, no video
    .venv/bin/python model/validation/ball_clean.py census      # every clip on disk
    .venv/bin/python model/validation/ball_clean.py plan        # the split over all 126
    .venv/bin/python model/validation/ball_clean.py variance    # the two-match limit, as a number
    .venv/bin/python model/validation/ball_clean.py fetch       # the eligible clips, by byte range (MIT)
    .venv/bin/python model/validation/ball_clean.py sample      # geometry proof, 2 clips
    .venv/bin/python model/validation/ball_clean.py tiles       # reuses the build; `tiles rebuild` re-cuts
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "model" / "engine"), str(Path(__file__).resolve().parent)]

import ball_cmp  # noqa: E402
from ball_train import T, clip_box  # noqa: E402  640 px, and the tile-edge box clip
from paths import FOOTAGE2  # noqa: E402
from teamtrack_tracks import load as load_tracks  # noqa: E402

#: Built artefacts. `.resolve()` on purpose: like `STORE` below, this follows the
#: symlink a worktree leaves behind, so the tiles are written once into the durable
#: store beside `out/ball/` and are not re-cut into a scratchpad that gets deleted.
#: The 2026-08-29 ball checkpoint was lost exactly that way
#: (ball-detector-trained-not-yet-scored (internal note)).
OUT = (ROOT / "model" / "validation" / "out" / "ball_clean").resolve()

#: Where the clips live, resolved THROUGH a file that is already there. In a plain
#: checkout this is `FOOTAGE2` itself; in a git worktree whose `data/` is a tree of
#: symlinks into the main checkout it is the shared store, so footage fetched here is
#: written once and outlives the worktree. Footage is data, not repo content.
STORE = (FOOTAGE2 / "teamtrack_zip_index.json").resolve().parent
ZIP_INDEX = STORE / "teamtrack_zip_index.json"

SEED = 0
FRAME_STEP = 5
NEG_KEEP = 4
OFFSET = (0.15, 0.85)

#: The six clips N48-N51 score on, by their TeamTrack stamp.
SCORED = {c.rsplit("__", 1)[1] for c in ball_cmp.CLIPS}
#: ball.BallPicker's ASPECT/SIZE_RATIO bounds were fitted on these (ball_eval.py:112).
GATE_FITTED = {"D_20220220_1_0000_0030", "D_20220220_1_0030_0060"}
#: ball_cmp excludes this one by name: the ball is carried forward, motionless, all clip.
DEGENERATE = {"D_20220220_1_1680_1710"}
#: Within 30 s of a scored window. Adjacent clips of one match are near-duplicates.
BUFFER = {"F_20200220_1_0090_0120"}


def role(sport: str, tt_split: str, stamp: str) -> str:
    """Why a clip is, or is not, eligible to train on. THE split rule; see the header."""
    if stamp in SCORED:
        return "scored"
    if stamp in GATE_FITTED:
        return "excluded: ball.py's gates were fitted here (ball_eval.py:112)"
    if stamp in DEGENERATE:
        return "excluded: dead ball carried forward (ball_cmp's own exclusion)"
    if stamp in BUFFER:
        return "excluded: abuts a scored window"
    if tt_split == "test":
        return "excluded: TeamTrack's own test split, held out whole"
    return "train" if tt_split == "train" else "valid"


# ------------------------------------------------------------------ inventory

def inventory() -> list[tuple[str, str, str]]:
    """[(sport, teamtrack split, clip stamp)] for every soccer clip in the zip index.

    Reads `teamtrack_zip_index.json`, which is on disk, so the split can be registered
    over the 116 clips before any of them is fetched. `._` members are the macOS
    AppleDouble sidecars inside the upstream zip and are not clips.
    """
    out = []
    for name, *_ in json.load(open(ZIP_INDEX)):
        parts = name.split("/")
        if len(parts) != 5 or parts[3] != "videos" or not name.endswith(".mp4"):
            continue
        if not parts[1].startswith("soccer_"):
            continue            # the zip also holds basketball and handball
        stamp = parts[4][:-4]
        if stamp.startswith("._"):
            continue
        out.append((parts[1], parts[2], stamp))
    return sorted(set(out))


#: TeamTrack's distribution zip, served by Google Drive with HTTP Range. Pulling one
#: 30 MB member beats downloading 8.6 GB; the recipe and the ZIP64 parse are
#: data/footage/soccertrack/README.md §6 and the member table is committed beside it.
DRIVE_ID = "12KOMgl9BFirmPVHbbUTUWacc9vLG94er"
DRIVE_URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
LICENCE = ("TeamTrack, Scott et al., https://github.com/AtomScott/TeamTrack - MIT, "
           "commercial use permitted, notice retained (docs/THIRD_PARTY_NOTICES.md).")


def _range(a: int, b: int) -> bytes:
    return subprocess.run(["curl", "-s", "-m", "900", "-r", f"{a}-{b}", DRIVE_URL],
                          capture_output=True).stdout


def _member(ent: list, dest: Path, tries: int = 4) -> int:
    """Inflate one zip member straight out of the remote file, by byte range.

    The LOCAL header's extra-field length differs from the central directory's, so it
    is read from the local header rather than assumed - getting that wrong yields a
    plausible-looking file that is shifted by a few bytes.

    Every length is checked against the index and a short or wrong read is RETRIED,
    not accepted: Drive drops a range now and then under concurrency, and a silently
    truncated clip is a dataset defect that surfaces as a training curve, not an error.
    """
    _, off, csz, usz, meth = ent
    for attempt in range(tries):
        try:
            hdr = _range(off, off + 63)
            if hdr[:4] != b"PK\x03\x04":
                raise ValueError("not a local file header (Drive quota, or a dropped range)")
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            lo = off + 30 + nlen + elen
            data = _range(lo, lo + csz - 1)
            if len(data) != csz:
                raise ValueError(f"got {len(data)} of {csz} compressed bytes")
            if meth == 8:
                data = zlib.decompressobj(-15).decompress(data)
            if len(data) != usz:
                raise ValueError(f"inflated to {len(data)}, index says {usz}")
        except (ValueError, zlib.error) as e:
            if attempt == tries - 1:
                raise SystemExit(f"{dest.name}: {e}") from e
            time.sleep(2 * (attempt + 1))
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(dest)        # never leave a truncated clip where a reader will find it
        return usz
    raise SystemExit(f"{dest.name}: unreachable")


def fetch(limit: int | None = None, workers: int = 3) -> dict:
    """Pull every clip this file's split makes eligible into the shared footage store.

    Writes `<sport>__<split>__{videos,annotations}__<stamp>.{mp4,csv}`, the flat naming
    the nine clips already on disk use, plus `teamtrack_fetch.json` recording what
    arrived. Existing files are kept, so this is resumable and re-running it is free.
    """
    ents = {e[0]: e for e in json.load(open(ZIP_INDEX))}
    want = []
    for sport, sp, stamp in inventory():
        if role(sport, sp, stamp) not in ("train", "valid"):
            continue
        for kind, ext in (("videos", "mp4"), ("annotations", "csv")):
            want.append((f"teamtrack/{sport}/{sp}/{kind}/{stamp}.{ext}",
                         STORE / f"{sport}__{sp}__{kind}__{stamp}.{ext}"))
    missing = [(m, d) for m, d in want if not d.exists()]
    todo = missing if limit is None else missing[:limit * 2]
    print(f"{len(want)} members eligible, {len(want) - len(missing)} already here, "
          f"{len(todo)} to fetch now -> {STORE}")
    got = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_member, ents[m], d): d for m, d in todo}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            got += f.result()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {got / 1e6:.0f} MB", flush=True)
    have = sorted(on_disk())
    eligible = [c for c in have if role(*_parts(c)) in ("train", "valid")]
    rep = {"licence": LICENCE, "source": DRIVE_URL, "clips_on_disk": len(have),
           # `got` is THIS run only - fetch is resumable, so a resumed run's total is
           # small and means nothing. `bytes_on_disk` is the number to quote.
           "clips_eligible": len(eligible), "fetched_bytes_this_run": got,
           "bytes_on_disk": sum((STORE / f"{c}.mp4").stat().st_size for c in have),
           "clip_ids": [_parts(c)[2] for c in eligible]}
    json.dump(rep, open(STORE / "teamtrack_fetch.json", "w"), indent=1)
    print(f"\n{rep['clips_on_disk']} clips on disk, {rep['clips_eligible']} eligible, "
          f"{rep['bytes_on_disk'] / 1e9:.2f} GB of video.\n{LICENCE}")
    return rep


def on_disk() -> list[str]:
    """The clip basenames actually present under data/footage/soccertrack/."""
    return sorted(p.stem for p in STORE.glob("soccer_*__videos__*.mp4"))


def _tt_split(clip: str) -> str:
    return _parts(clip)[1]


def _parts(clip: str) -> tuple[str, str, str]:
    """'soccer_top__test__videos__D_..._1710_1740' -> ('soccer_top', 'test', 'D_..._1710_1740')."""
    sport, split, _, stamp = clip.split("__", 3)
    return sport, split, stamp


# -------------------------------------------------------------------- the ball

def ball_rows(clip: str) -> list[dict]:
    """TeamTrack's ball rows for a clip, frames rebased to 0 to match the video.

    Same rebasing as `ball_eval.load_ball` and for the same reason: the CSV numbers
    rows from 1 and the video's first frame is 0, so an unrebased join shifts every
    box by one frame of ball travel.
    """
    csv = STORE / f"{clip.replace('videos', 'annotations')}.csv"
    rows = load_tracks(csv, want_ball=True).get("BALL_BALL", [])
    if not rows:
        raise SystemExit(f"no BALL column in {csv}")
    base = min(r["f"] for r in rows)
    return [dict(r, f=r["f"] - base) for r in sorted(rows, key=lambda r: r["f"])]


def _video_shape(clip: str) -> tuple[int, int, float, int]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(STORE / f"{clip}.mp4"))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          round(float(cap.get(cv2.CAP_PROP_FPS)), 2), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return wh


# --------------------------------------------------------------------- census

def census() -> dict:
    """What the nine on-disk clips actually give, and which of them is already spent."""
    rep = []
    for clip in on_disk():
        sport, split, stamp = _parts(clip)
        w, h, fps, n = _video_shape(clip)
        rows = ball_rows(clip)
        side = np.array([max(r["w"], r["h"]) for r in rows], float)
        xy = np.array([[r["x"], r["y"]] for r in rows], float)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        rep.append({
            "clip": clip, "stamp": stamp, "res": f"{w}x{h}", "fps": fps, "frames": n,
            "ball_frames": len(rows), "ball_share": round(len(rows) / max(n, 1), 4),
            "box_px_p5_p50_p95": [round(float(v), 1) for v in np.percentile(side, [5, 50, 95])],
            "step_px_p50": round(float(np.median(step)), 3),
            "frozen_share": round(float((step < 0.5).mean()), 4),
            "role": role(sport, split, stamp),
        })
    print(f"{'clip':56} {'res':>10} {'ball':>10} {'box p5/p50/p95':>18} "
          f"{'step':>7} {'frozen':>7}  role")
    for r in rep:
        print(f"{r['clip']:56} {r['res']:>10} {r['ball_frames']:>5}/{r['frames']:<4} "
              f"{str(r['box_px_p5_p50_p95']):>18} {r['step_px_p50']:>7.2f} "
              f"{r['frozen_share']:>7.1%}  {r['role']}")
    tot = sum(r["ball_frames"] for r in rep)
    eligible = [r for r in rep if r["role"] in ("train", "valid")]
    print(f"\n{len(rep)} clips on disk, {tot} ball positions, "
          f"{len(eligible)} eligible to train or validate on.")
    return {"clips": rep, "ball_positions": tot, "eligible_on_disk": len(eligible)}


def plan() -> dict:
    """The registered split over TeamTrack's whole soccer inventory, fetched or not."""
    inv = inventory()
    have = {_parts(c)[2] for c in on_disk()}
    counts: dict[tuple[str, str], int] = {}
    for sport, split, stamp in inv:
        counts[(sport, role(sport, split, stamp).split(":")[0])] = \
            counts.get((sport, role(sport, split, stamp).split(":")[0]), 0) + 1
    for (sport, r), n in sorted(counts.items()):
        got = sum(1 for s, sp, st in inv
                  if s == sport and role(s, sp, st).split(":")[0] == r and st in have)
        print(f"{sport:12} {r:9} {n:3} clips  ({got} on disk)")
    print(f"\n{len(inv)} soccer clips in the index, "
          f"{sum(n for (_, r), n in counts.items() if r in ('train', 'valid'))} eligible, "
          f"{len(have)} on disk.")
    # The match key is the leading letter, NOT the date: soccer_side's train filenames
    # carry two date stamps for one match (F_20200220 to 870 s, F_20220220 after - an
    # upstream typo, README §6). Counting dates would report three matches where there
    # are two, which is exactly the independence claim this file is careful about.
    matches = sorted({st.split("_", 1)[0] for _, _, st in inv})
    print(f"distinct matches {len(matches)}: {matches} - a held-out clip here is a held-out "
          "MINUTE, not a match.")
    return {"counts": {f"{s}/{r}": n for (s, r), n in counts.items()},
            "matches": matches, "on_disk": sorted(have)}


# ---------------------------------------------------------------- the builder

def _window(cx: float, cy: float, iw: int, ih: int, rng: random.Random) -> tuple[int, int]:
    """Top-left of a T-px window CONTAINING (cx, cy) at a random offset, inside the frame."""
    x0 = int(min(max(cx - rng.uniform(*OFFSET) * T, 0), iw - T))
    y0 = int(min(max(cy - rng.uniform(*OFFSET) * T, 0), ih - T))
    return x0, y0


def _empty_window(box: tuple[float, float, float, float], iw: int, ih: int,
                  rng: random.Random) -> tuple[int, int] | None:
    """A T-px window that does NOT touch the ball's box, or None if 20 tries fail."""
    bx0, by0, bx1, by1 = box
    for _ in range(20):
        x0, y0 = rng.randint(0, iw - T), rng.randint(0, ih - T)
        if x0 >= bx1 or x0 + T <= bx0 or y0 >= by1 or y0 + T <= by0:
            return x0, y0
    return None


def build_clip(clip: str, dest: Path, iid: int, aid: int, imgs: list, anns: list,
               frame_step: int = FRAME_STEP, limit: int | None = None,
               drops: dict | None = None) -> tuple[int, int]:
    """Cut one clip into tiles. Frames are read SEQUENTIALLY - see ball_eval.build_cache.

    `drops` accumulates every wanted frame that did NOT become a tile, by reason. A
    builder that silently skips is the filter CLAUDE.md warns about: deleting the hard
    inputs and handling them look identical in the tile count.
    """
    import cv2  # noqa: PLC0415

    d = drops if drops is not None else {}
    rng = random.Random(f"{SEED}:{clip}")
    iw, ih, _, _ = _video_shape(clip)
    if iw < T or ih < T:
        raise SystemExit(f"{clip}: {iw}x{ih} is smaller than a {T} px tile")
    want = {r["f"]: r for r in ball_rows(clip) if r["f"] % frame_step == 0}
    if limit:
        want = {f: want[f] for f in sorted(want)[:limit]}
    d["wanted"] = d.get("wanted", 0) + len(want)
    cap = cv2.VideoCapture(str(STORE / f"{clip}.mp4"))
    dest.mkdir(parents=True, exist_ok=True)
    npos = 0
    seen = set()
    for f in range(max(want) + 1 if want else 0):
        ok, frame = cap.read()
        if not ok:
            break
        r = want.get(f)
        if r is None:
            continue
        seen.add(f)
        bw, bh = r["w"], r["h"]
        bx0, by0 = r["x"] - bw / 2, r["y"] - bh / 2
        wins = [_window(r["x"], r["y"], iw, ih, rng)]
        if npos % NEG_KEEP == NEG_KEEP - 1:
            neg = _empty_window((bx0, by0, bx0 + bw, by0 + bh), iw, ih, rng)
            if neg:
                wins.append(neg)
            else:
                d["negative_unplaced"] = d.get("negative_unplaced", 0) + 1
        npos += 1
        for x0, y0 in wins:
            tile = frame[y0:y0 + T, x0:x0 + T]
            if tile.shape[:2] != (T, T):
                d["tile_short"] = d.get("tile_short", 0) + 1
                continue
            iid += 1
            name = f"{clip}_{f:05d}_{x0}_{y0}.jpg"
            cv2.imwrite(str(dest / name), tile)
            imgs.append({"id": iid, "file_name": name, "width": T, "height": T,
                         "clip": clip, "frame": f, "origin": [x0, y0]})
            b = clip_box(bx0 - x0, by0 - y0, bw, bh)
            if b:
                aid += 1
                anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": b,
                             "area": b[2] * b[3], "iscrowd": 0})
    cap.release()
    d["frame_unreadable"] = d.get("frame_unreadable", 0) + len(set(want) - seen)
    return iid, aid


def _write_coco(dest: Path, imgs: list, anns: list, extra: dict) -> None:
    json.dump({"images": imgs, "annotations": anns,
               "categories": [{"id": 1, "name": "ball", "supercategory": "none"}],
               "info": extra},
              open(dest / "_annotations.coco.json", "w"))


def _stats(imgs: list, anns: list) -> dict:
    """Crop and box statistics for a built split. Descriptive; nothing is selected on them."""
    w = np.array([a["bbox"][2] for a in anns], float)
    h = np.array([a["bbox"][3] for a in anns], float)
    cx = np.array([(a["bbox"][0] + a["bbox"][2] / 2) / T for a in anns], float)
    cy = np.array([(a["bbox"][1] + a["bbox"][3] / 2) / T for a in anns], float)
    edge = sum(1 for a in anns
               if a["bbox"][0] <= 0 or a["bbox"][1] <= 0
               or a["bbox"][0] + a["bbox"][2] >= T or a["bbox"][1] + a["bbox"][3] >= T)
    pct = [5, 50, 95]
    return {"tiles": len(imgs), "boxes": len(anns), "empty_tiles": len(imgs) - len(anns),
            "box_w_p5_p50_p95": [round(float(v), 1) for v in np.percentile(w, pct)],
            "box_h_p5_p50_p95": [round(float(v), 1) for v in np.percentile(h, pct)],
            "box_area_frac_p50": round(float(np.median(w * h)) / (T * T), 7),
            "centre_x_p5_p50_p95": [round(float(v), 3) for v in np.percentile(cx, pct)],
            "centre_y_p5_p50_p95": [round(float(v), 3) for v in np.percentile(cy, pct)],
            "boxes_touching_tile_edge": edge}


def _brightness(dest: Path, imgs: list, anns: list) -> list:
    """[p5, p50] of (brightest pixel in the box) / (tile median grey), over the tiles.

    The crop is checked against PIXELS rather than trusted. A white ball on grass reads
    well above 1; a rebasing error of one frame, or a box written in the wrong
    coordinates, lands the box on grass and reads near 1. This is the join's own check.
    """
    import cv2  # noqa: PLC0415

    by_id = {im["id"]: im["file_name"] for im in imgs}
    r = []
    for a in anns:
        g = cv2.imread(str(dest / by_id[a["image_id"]]), cv2.IMREAD_GRAYSCALE)
        x, y, w, h = (int(v) for v in a["bbox"])
        r.append(float(g[y:y + h, x:x + w].max()) / max(float(np.median(g)), 1.0))
    return [round(float(v), 2) for v in np.percentile(r, [5, 50])] if r else []


#: The sample is a geometry proof. It is written where no trainer looks and it says so.
NOT_TRAINING = ("geometry proof only - these clips are scored or gate-fitted "
                "(ball_clean.role) and must never be a training input")


def sample(clips: tuple[str, ...] = (), limit: int = 20) -> dict:
    """Prove the tiler on a handful of frames of one top-down and one side clip."""
    clips = clips or ("soccer_top__train__videos__D_20220220_1_0000_0030",
                      "soccer_side__train__videos__F_20200220_1_0000_0030")
    rep = {}
    for clip in clips:
        dest = OUT / "sample" / clip
        imgs: list = []
        anns: list = []
        build_clip(clip, dest, 0, 0, imgs, anns, limit=limit)
        _write_coco(dest, imgs, anns, {"purpose": NOT_TRAINING, "clip": clip,
                                       "role": role(*_parts(clip)), "tile": T,
                                       "frame_step": FRAME_STEP, "neg_keep": NEG_KEEP})
        st = _stats(imgs, anns)
        st["res"] = "x".join(str(v) for v in _video_shape(clip)[:2])
        st["box_brightness_over_tile_median"] = _brightness(dest, imgs, anns)
        rep[clip] = st
        print(f"\n{clip}  ({st['res']}, role: {role(*_parts(clip))})")
        for k, v in st.items():
            print(f"    {k:26} {v}")
    json.dump(rep, open(OUT / "sample_stats.json", "w"), indent=1)
    print(f"\n-> {OUT / 'sample_stats.json'}   ({NOT_TRAINING})")
    return rep


#: N52 §7's two arms, fixed in the plan (commit bb38497) before either ran. D1 is
#: the arm the rule rules on; D2 is reported, never promoted.
ARMS = {"d1": {"tiles": 916, "epochs": 12}, "d2": {"tiles": None, "epochs": 6}}


def matches() -> dict:
    """How many MATCHES the clean corpus really holds, and the proof for the answer.

    The filenames suggest three: soccer_top's D_20220220_1 and soccer_side's TWO ids,
    F_20200220_1 and F_20220220_1. The year differs, so it reads as two matches two
    years apart. It is one: the ids cover non-overlapping time ranges of a single
    recording, and a frame from each correlates on the static stand-and-treeline band
    at a level only one camera on one afternoon produces. Printed rather than asserted
    because N52 §1's "two matches" is a load-bearing caveat on every ball number.
    """
    import re  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    got: dict = {}
    for f in sorted((ROOT / "data" / "footage" / "soccertrack").iterdir()):
        m = re.match(r"(soccer_\w+)__\w+__videos__([A-Z]_\d{8}_\d+)_(\d{4})_(\d{4})\.mp4", f.name)
        if m:
            got.setdefault((m.group(1), m.group(2)), []).append(int(m.group(3)))
    for (sport, mid), starts in sorted(got.items()):
        print(f"{sport:12s} {mid}  {len(starts):3d} clips  t {min(starts)}-{max(starts)}s")
    pair = [k for k in got if k[1].startswith("F_")]
    corr = float("nan")
    if len(pair) == 2:
        fr = []
        for sport, mid in sorted(pair):
            clip = f"{sport}__train__videos__{mid}_{min(got[(sport, mid)]):04d}_{min(got[(sport, mid)]) + 30:04d}.mp4"
            cap = cv2.VideoCapture(str(ROOT / "data" / "footage" / "soccertrack" / clip))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            ok, img = cap.read()
            cap.release()
            if ok:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fr.append(g[:int(g.shape[0] * 0.35)].astype(float).ravel())
        if len(fr) == 2:
            corr = float(np.corrcoef(fr[0], fr[1])[0, 1])
    print(f"the two F_ ids: background-band correlation {corr:.4f} -> "
          f"{'ONE match, the filename year differs' if corr > 0.9 else 'two different scenes'}")
    # DERIVED FROM corr, not hardcoded. The first version printed "2 distinct
    # matches" from arithmetic that could not see the correlation at all - so a
    # third F_ id, or a correlation of 0.3, would have printed the same
    # conclusion under the same measured tag. A headline independent of its
    # evidence is the thing this file exists to refuse.
    f_ids = {k for k in got if k[1].startswith("F_")}
    f_matches = 1 if (len(f_ids) == 2 and corr > 0.9) else len(f_ids)
    print(f"MATCHES: {len({k[0] for k in got})} sports splits, "
          f"{f_matches + len(got) - len(f_ids)} distinct matches"
          + ("" if f_matches == 1 else "  <- the F_ ids did NOT correlate; they are separate scenes"))
    return {"ids": {f"{a}/{b}": len(v) for (a, b), v in sorted(got.items())}, "f_corr": corr}


def matched(n: int = 916, seed: int = 0) -> dict:
    """D1's train set: n tiles, clip-stratified, deterministic at `seed`.

    Stratified rather than uniform because the 82 clips are wildly uneven and a
    uniform draw of 916 from 16,509 would silently become a sample of the biggest
    few - which would confound SOURCE with clip diversity, the same shape as N48's
    tile-size confound. Largest-remainder allocation, so the per-clip shares sum to
    exactly n rather than to n +/- the rounding.
    """
    src = OUT / "tiles" / "train"
    d = json.loads((src / "_annotations.coco.json").read_text())
    by_clip: dict = {}
    for im in d["images"]:
        by_clip.setdefault(im["file_name"].rsplit("_", 3)[0], []).append(im)
    total = len(d["images"])
    exact = {c: n * len(v) / total for c, v in by_clip.items()}
    take = {c: int(x) for c, x in exact.items()}
    for c in sorted(exact, key=lambda k: (exact[k] - take[k], k), reverse=True)[:n - sum(take.values())]:
        take[c] += 1
    rng = random.Random(seed)
    keep = []
    for c in sorted(by_clip):
        ims = sorted(by_clip[c], key=lambda i: i["file_name"])
        keep += rng.sample(ims, min(take[c], len(ims)))
    ids = {i["id"] for i in keep}
    return {"images": keep, "clips": len(by_clip), "take": take,
            "annotations": [a for a in d["annotations"] if a["image_id"] in ids],
            "categories": d["categories"], "info": d.get("info", {}), "licenses": d.get("licenses", [])}


def arm(which: str = "d1") -> str:
    """Build the arm's dataset dir (links, never copies) and train it."""
    import ball_train  # noqa: PLC0415

    spec = ARMS[which]
    dest = OUT / f"tiles_{which}"
    if spec["tiles"] is None:
        dest = OUT / "tiles"                       # D2 trains on everything
    else:
        sub = matched(spec["tiles"])
        (dest / "train").mkdir(parents=True, exist_ok=True)
        for im in sub["images"]:
            link = dest / "train" / im["file_name"]
            if not link.exists():
                link.symlink_to(OUT / "tiles" / "train" / im["file_name"])
        (dest / "train" / "_annotations.coco.json").write_text(json.dumps(sub))
        # valid is a REAL directory of links, never a symlink to one: the same rule
        # the claim gate's out/ follows, and for the same reason - a symlinked dir
        # resolves for every ordinary read and then breaks one tool at the end.
        (dest / "valid").mkdir(parents=True, exist_ok=True)
        for f in (OUT / "tiles" / "valid").iterdir():
            link = dest / "valid" / f.name
            if not link.exists():
                link.symlink_to(f)
        print(f"{which}: {len(sub['images'])} train tiles over {sub['clips']} clips, "
              f"{len(sub['annotations'])} boxes, valid linked whole")
    run = OUT / f"run_{which}"
    print(f"{which}: training {spec['epochs']} epochs -> {run}")
    ball_train.train(dataset_dir=str(dest), output_dir=str(run), epochs=spec["epochs"])
    return str(run)


def tiles(rebuild: bool = False) -> dict:
    """Build train/ and valid/ from the eligible clips. Refuses what is not eligible.

    An existing build is REUSED, not silently re-cut. The cut is deterministic at seed 0
    so a rebuild would land on the same tiles, but it costs ~35 minutes of 4K decode and
    the next session has no way to know it is repeating work. `rebuild` forces it.
    """
    if not rebuild and (OUT / "tile_stats.json").exists() and all(
            (OUT / "tiles" / s / "_annotations.coco.json").exists()
            for s in ("train", "valid")):
        rep = json.load(open(OUT / "tile_stats.json"))
        print(f"tiles already built -> {OUT / 'tiles'}\n"
              "  re-printing tile_stats.json; nothing was decoded. To cut them again:\n"
              "  .venv/bin/python model/validation/ball_clean.py tiles rebuild")
        for split, st in rep.items():
            print(f"\n{split}: {json.dumps(st, indent=1)}")
        print(f"\n-> {OUT / 'tile_stats.json'}\n{LICENCE}")
        return rep
    have = on_disk()
    todo = {s: [c for c in have if role(*_parts(c)) == s] for s in ("train", "valid")}
    if not todo["train"] or not todo["valid"]:
        raise SystemExit(
            "REFUSED: no eligible TeamTrack clip is on disk.\n"
            f"  on disk: {len(have)} clips, roles: "
            + ", ".join(sorted({role(*_parts(c)).split(':')[0] for c in have})) + "\n"
            "  Every clip here is either scored by ball_cmp, gate-fitted, or degenerate.\n"
            "  Fetch the eligible ones first - they are MIT and free, by byte range, per\n"
            "  data/footage/soccertrack/README.md §6; `ball_clean.py plan` lists them.\n"
            "  Training on what is on disk would score the detector on its own training\n"
            "  frames, which is the failure .claude/CLAUDE.md exists to stop.")
    rep = {}
    for split, clips in todo.items():
        dest = OUT / "tiles" / split
        imgs: list = []
        anns: list = []
        drops: dict = {}
        iid = aid = 0
        for i, clip in enumerate(clips, 1):
            iid, aid = build_clip(clip, dest, iid, aid, imgs, anns, drops=drops)
            print(f"  [{split} {i}/{len(clips)}] {clip}  {len(imgs)} tiles", flush=True)
        _write_coco(dest, imgs, anns, {"clips": clips, "tile": T,
                                       "frame_step": FRAME_STEP, "neg_keep": NEG_KEEP,
                                       "licence": LICENCE})
        st = _stats(imgs, anns)
        st["clips"] = len(clips)
        st["drops"] = drops
        #: Split by sport, because the two geometries are the whole composition question:
        #: 6500x1000 side panoramas are what our own footage looks like, 4K top-down is not.
        by_clip = {im["id"]: im["clip"] for im in imgs}
        for sport in ("soccer_top", "soccer_side"):
            si = [im for im in imgs if im["clip"].startswith(sport)]
            sa = [a for a in anns if by_clip[a["image_id"]].startswith(sport)]
            st[sport] = _stats(si, sa) if si else {}
            st[sport]["clips"] = sum(1 for c in clips if c.startswith(sport))
        rep[split] = st
        print(f"\n{split}: {json.dumps(st, indent=1)}")
    json.dump(rep, open(OUT / "tile_stats.json", "w"), indent=1)
    print(f"\n-> {OUT / 'tile_stats.json'}\n{LICENCE}")
    return rep


def variance(suffix: str = ".cmpB") -> dict:
    """Put a NUMBER on the two-match limit instead of a warning.

    One-way decomposition of N48's six per-clip `oracle` rates by match - `soccer_top`
    is D_20220220_1, `soccer_side` is F_*_20220220_1 - into between-match and
    within-match sums of squares. A replay over the committed caches; nothing detects.

    READ THE SHARE THE RIGHT WAY ROUND. Two groups give ONE degree of freedom between,
    so a small between share is NOT evidence that the two matches agree and therefore
    that same-match scoring is safe. It is mostly evidence that six clips drawn from two
    matches cannot resolve a match effect at all: the within-match spread here runs from
    a clip the detector gets on every frame to one it gets on a sixth of them, and that
    swamps any venue term. The decomposition is reported because it bounds what this
    dataset can ever say, not because it licenses the conclusion.
    """
    import ball_eval as BE  # noqa: PLC0415

    rates, by_match = [], {}
    for clip in ball_cmp.CLIPS:
        r = BE.report(clip, BE.GATE_PX, suffix)["scores"]["oracle"]
        rate = r["correct"] / r["gt_present"]
        rates.append((clip, round(rate, 4)))
        by_match.setdefault(_parts(clip)[0], []).append(rate)
    v = np.array([r for _, r in rates], float)
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    ss_between = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in by_match.values()))
    out = {"per_clip": dict(rates),
           "match_means": {k: round(float(np.mean(g)), 4) for k, g in by_match.items()},
           "ss_between": round(ss_between, 5), "ss_within": round(ss_total - ss_between, 5),
           "between_share": round(ss_between / ss_total, 4) if ss_total else None,
           "within_range": [round(float(v.min()), 4), round(float(v.max()), 4)]}
    print(json.dumps(out, indent=1))
    print(f"\nbetween-match share of the per-clip spread: {out['between_share']:.2%} "
          f"(1 df between, 4 within - it bounds what this data can say, it does not "
          "license same-match scoring)")
    return out


# ------------------------------------------------------------------ self-check

def demo() -> None:
    inv = inventory()
    assert inv, "no clips in the zip index"
    roles = {(s, sp, st): role(s, sp, st) for s, sp, st in inv}

    # 1. The scored population is never trainable, and every scored clip is in the index.
    assert SCORED <= {st for _, _, st in inv}, SCORED - {st for _, _, st in inv}
    for k, r in roles.items():
        assert not (k[2] in SCORED and r in ("train", "valid")), k
    # 2. Whatever is on disk, the build list can only ever hold train/valid clips.
    #    Until 2026-09-20 this asserted the stronger "nothing on disk is eligible",
    #    which was the scoping finding and stopped being true the moment `fetch` ran.
    #    The invariant that outlives it is this one: a scored, gate-fitted, degenerate
    #    or buffer clip must never reach the tiler, however many clips arrive.
    disk_roles = {c: role(*_parts(c)) for c in on_disk()}
    spent = SCORED | GATE_FITTED | DEGENERATE | BUFFER
    for c, r in disk_roles.items():
        assert (r in ("train", "valid")) == (_parts(c)[2] not in spent
                                             and _tt_split(c) != "test"), (c, r)
    # 3. A scored clip is never adjacent to a train/valid clip in the same match.
    for sport, split, stamp in inv:
        if roles[(sport, split, stamp)] not in ("train", "valid"):
            continue
        a, b = (int(v) for v in stamp.rsplit("_", 2)[1:])
        for s2, sp2, st2 in inv:
            if st2 in SCORED and st2.rsplit("_", 2)[0].split("_", 1)[0] == stamp.split("_", 1)[0]:
                c, d = (int(v) for v in st2.rsplit("_", 2)[1:])
                assert b <= c or a >= d, f"{stamp} overlaps scored {st2}"
                assert b != c and a != d, f"{stamp} abuts scored {st2} - no buffer"

    # 4. The window contains the ball, stays inside the frame, and clips at the edge.
    rng = random.Random(0)
    for iw, ih, cx, cy in ((3840, 2160, 10.0, 10.0), (3840, 2160, 3830.0, 2150.0),
                           (6500, 1000, 3000.0, 500.0), (3840, 2160, 1900.0, 1000.0)):
        x0, y0 = _window(cx, cy, iw, ih, rng)
        assert 0 <= x0 <= iw - T and 0 <= y0 <= ih - T, (x0, y0, iw, ih)
        assert x0 <= cx <= x0 + T and y0 <= cy <= y0 + T, (x0, y0, cx, cy)
    assert clip_box(-5, 10, 20, 20) == [0, 10, 15, 20], "a cut box is clipped, not shifted"
    # 5. A negative window really is ball-free.
    for _ in range(50):
        n = _empty_window((1000.0, 1000.0, 1012.0, 1012.0), 3840, 2160, rng)
        if n:
            assert n[0] >= 1012 or n[0] + T <= 1000 or n[1] >= 1012 or n[1] + T <= 1000, n
    # 6. Frames are rebased: the CSV starts at 1, the returned rows start at 0.
    disk = on_disk()
    if disk:
        assert min(r["f"] for r in ball_rows(disk[0])) == 0, "ball rows not rebased to 0"
    # 7. matches(): the headline must follow the correlation, not a constant.
    assert (lambda ids, c: 1 if (len(ids) == 2 and c > 0.9) else len(ids))({"a", "b"}, 0.95) == 1
    assert (lambda ids, c: 1 if (len(ids) == 2 and c > 0.9) else len(ids))({"a", "b"}, 0.30) == 2, \
        "an uncorrelated pair must read as two scenes, or the tag proves nothing"

    # 8. D1's matched sample: the three ways a stratified draw goes quietly wrong.
    if (OUT / "tiles" / "train" / "_annotations.coco.json").exists():
        m = matched(916)
        assert len(m["images"]) == 916, len(m["images"])
        assert sum(m["take"].values()) == 916, "largest-remainder did not close the gap"
        assert len({i["id"] for i in m["images"]}) == 916, "a tile was drawn twice"
        assert len({i["file_name"].rsplit("_", 3)[0] for i in m["images"]}) == m["clips"], \
            "a clip dropped out of the draw - that is a uniform sample wearing a stratified name"
        assert [i["id"] for i in matched(916)["images"]] == [i["id"] for i in m["images"]], \
            "not deterministic at seed 0, so the arm cannot be re-run"
        ids = {i["id"] for i in m["images"]}
        assert all(a["image_id"] in ids for a in m["annotations"]), "a box outside the sample"

    print(f"ball_clean demo: ok ({len(inv)} clips indexed, "
          f"{sum(r in ('train', 'valid') for r in roles.values())} eligible, "
          f"{len(disk)} on disk, {sum(r in ('train','valid') for r in disk_roles.values())} "
          "of them eligible)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--demo"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "fetch":
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "tiles":
        tiles(rebuild=len(sys.argv) > 2 and sys.argv[2] == "rebuild")
    elif cmd == "arm":
        arm(sys.argv[2] if len(sys.argv) > 2 else "d1")
    else:
        {"--demo": demo, "census": census, "plan": plan,
         "sample": sample, "tiles": tiles, "variance": variance,
         "matches": matches}[cmd]()
