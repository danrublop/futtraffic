"""Can an appearance embedding tell two players on the SAME TEAM apart?

WHY THIS EXISTS
    Contamination is the binding constraint on per-player output: measured
    against SoccerTrack ground truth, 15.1% of detections sit under a raw
    C-BIoU id whose dominant true player is somebody else, and stitching takes
    that to 31.0% even with the team gate refusing cross-bank merges. Every
    remaining error is WITHIN a team, so no team-level signal can touch it -
    `stitch.py` blocks opposing merges already, and `track.assign_teams` is
    91.1% accurate on the tracks the gate reads.

    What is missing is a signal that says WHICH PLAYER. This file measures
    whether one exists on this footage before any of it is wired anywhere,
    because the honest answer might be no: `stitch.py` measured kit colour at
    AUC 0.724 for the same-player question, which is barely above chance, and
    the reason is physical - the two players you most need to separate are
    wearing the same shirt.

WHAT IS AND IS NOT ALLOWED HERE
    Torso crops only, head EXCLUDED - `teamembed.shirt_crop` takes rows 0.2-0.5
    of the box, which is the shirt and neither the face nor the legs. No face
    recognition and no gait, ever, at any accuracy. That is a standing product
    constraint and not a tuning knob.

    Nothing here attributes an event, a touch, a pass or a caption to a player.
    This is association: deciding that two fragments of tracking are the same
    person, which is the same decision the tracker already makes every frame
    from position alone.

THE CONFOUND THIS FILE IS BUILT AROUND
    Two fragments of one player are usually SECONDS apart. Two different
    players sampled at random are usually MINUTES apart. Floodlight, cloud and
    white balance all drift over a 48-minute half, so an embedding that encodes
    nothing but "when was this frame" would score well on a naive positive/
    negative split - and would then fail in production, where the pairs a
    stitcher actually asks about are all short-gap.

    So negatives are drawn to MATCH the positives' time-gap distribution, in
    log-spaced bins. `--no-match` runs it the naive way for comparison; the gap
    between the two numbers is the size of the confound, and on this footage it
    is not small.

THE ANSWER, MEASURED: NO. DO NOT WIRE AN APPEARANCE EMBEDDING.
    7,568 torso crops over 400 frames of 118575's first half, all 22 GT players,
    negatives drawn same-team and gap-matched. AUC on "are these two crops the
    same player":

        method                     overall   0-30px  30-50px  50-80px  80+px
        colour histogram (free)     0.6074    0.570    0.618    0.565   0.564
        SigLIP torso (780 MB)       0.6113    0.664    0.634    0.550   0.540
        ---- controls ----
        BOX HEIGHT ALONE            0.5860      -        -        -       -
        FRAME INDEX ALONE           0.5008      -        -        -       -

    Box height carries no appearance at all. It is one number saying how far
    away the player is. It scores 0.586, so of SigLIP's 0.611 essentially ALL
    of it is depth: the marginal value of 780 MB of vision weights over "how
    big is the box" is about 0.025 AUC, and a free colour histogram matches it.

    The second tell is the per-band row. Both methods get WORSE as the crop gets
    BIGGER - SigLIP 0.634 at 30-50 px down to 0.540 at 80+. Real appearance
    identity improves with resolution. A number that degrades as the evidence
    improves is not reading the evidence; it is reading depth, and the near-side
    bands are where depth stops discriminating.

    The frame-index control at 0.5008 is the harness clearing itself: the
    gap-matching removed the temporal confound completely, so the weak result is
    not an artefact of the sampling.

    WHY, PHYSICALLY. The two players this has to separate wear the same shirt,
    the same shorts and the same socks, at 24-50 px of torso, on a 4096-wide
    panorama covering a 105 m pitch. There is no appearance left to read once
    the kit is shared - which is also why the TEAM question ("any team" row in
    the output, 0.66-0.68) scores higher than the identity question on the same
    embeddings. The pipeline already answers the team question at 91.1%.

    WHAT WOULD ACTUALLY MOVE IT: source pixels, not a bigger model. Jersey-number
    OCR needs an order of magnitude more resolution than the torso has here.
    Motion and geometry are not exhausted the way appearance now is.

    So per-player identity is not recoverable from appearance on this footage,
    and combined with the scouting-feed result - a ball-following camera hides
    ~43% of players at any instant - the honest product statement is that
    per-player physical output needs a higher-resolution fixed-wide source than
    either input we have. Team-level output is what these inputs support.

READ THE PER-BAND ROWS, NOT THE HEADLINE
    A far-side shirt crop is ~24 px tall and a near-side one is 70-115. Those
    are different problems and averaging them hides the only version of this
    that could work. Bands are on the SMALLER box of the pair, because a pair
    is only as legible as its worse half.

    python model/validation/reid_eval.py --build          # sample crops (slow, once)
    python model/validation/reid_eval.py                  # score every method
    python model/validation/reid_eval.py --selfcheck
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))

import calibrators  # noqa: E402

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")
VIDEO = os.path.join(ROOT, "data/footage/118575_panorama_1st_half.mp4")
CROPS = os.path.join(JOB, "reid_eval.crops.npz")

#: Crop side fed to the embedding. 96 because SigLIP resizes to 224 anyway and
#: the source torso is 24-115 px tall - caching bigger would store upsampling.
SIZE = 96

#: How close a detection must be to a ground-truth player to inherit its id.
#: Same gate as assoc_eval, for the same reason: looser would score the
#: calibration, tighter would score it the other way.
GATE_M = 2.0


def load_owner(job: str = JOB, gt_path: str = GT, tracks: str = "tracks.cbiou.json",
               stride: int = 10) -> tuple[dict[int, str], dict[str, str]]:
    """Dominant ground-truth player for each track, and each player's team.

    A track that swaps between two players gets its MAJORITY owner. That is the
    right call for this file: a pair drawn from two mostly-A fragments is a
    same-player pair, and the minority frames are noise on both sides of the
    comparison rather than a different question.
    """
    raw = json.load(gzip.open(gt_path))["detections"]
    base = min(r["f"] for r in raw)
    per_frame: dict[int, dict[str, np.ndarray]] = collections.defaultdict(dict)
    team: dict[str, str] = {}
    for r in raw:
        per_frame[r["f"] - base][str(r["id"])] = np.array([r["x"], r["y"]])
        team[str(r["id"])] = r["team"]

    cal = calibrators.load(os.path.join(job, "calibration.json"))
    rows = json.load(open(os.path.join(job, tracks)))["detections"]
    by_frame: dict[int, list] = collections.defaultdict(list)
    for d in rows:
        by_frame[d["f"]].append(d)

    vote: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for f in range(0, max(by_frame) + 1, stride):
        g, p = per_frame.get(f), by_frame.get(f)
        if not g or not p:
            continue
        gid = list(g)
        G = np.array([g[k] for k in gid])
        xy = cal.to_pitch(np.array([[r["x"], r["y"]] for r in p]))
        d = np.linalg.norm(xy[:, None, :] - G[None, :, :], axis=2)
        j = d.argmin(1)
        for i, r in enumerate(p):
            if d[i, j[i]] < GATE_M:
                vote[int(r["id"])][gid[j[i]]] += 1
    return {t: c.most_common(1)[0][0] for t, c in vote.items() if c}, team


def build(n_frames: int = 400, job: str = JOB, video: str = VIDEO, out: str = CROPS,
          tracks: str = "tracks.cbiou.json") -> str:
    """Cache one torso crop per tracked detection, with its box height and owner.

    Height is stored because it is the only thing that decides whether this can
    work: it is the difference between 24 px of shirt and 115.
    """
    import cv2  # noqa: PLC0415

    sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
    import teamembed  # noqa: PLC0415

    owner, team = load_owner(job, tracks=tracks)
    cal = calibrators.load(os.path.join(job, "calibration.json"))
    cache = json.load(open(os.path.join(job, "dets.cache.json")))["frames"]
    rows = json.load(open(os.path.join(job, tracks)))["detections"]
    by_frame: dict[int, list] = collections.defaultdict(list)
    for d in rows:
        by_frame[d["f"]].append(d)

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(0, total - 1, n_frames).astype(int)

    fs, tids, gids, hs, crops = [], [], [], [], []
    for k, f in enumerate(picks):
        rs, boxes = by_frame.get(int(f)), cache.get(str(int(f)))
        if not rs or not boxes:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, raw = cap.read()
        if not ok:
            continue
        frame = cal.undistort_frame(raw)
        B = np.array([[(b["xyxy"][0] + b["xyxy"][2]) / 2, b["xyxy"][3]] for b in boxes])
        for r in rs:
            if int(r["id"]) not in owner:
                continue        # no ground-truth identity, so no pair to make
            j = int(np.hypot(*(np.array([r["x"], r["y"]]) - B).T).argmin())
            if np.hypot(*(np.array([r["x"], r["y"]]) - B[j])) > 0.01:
                continue        # no box for this track this frame
            x1, y1, x2, y2 = (int(v) for v in boxes[j]["xyxy"])
            c = teamembed.shirt_crop(frame, (x1, y1, x2, y2), size=SIZE)
            if c is None:
                continue
            fs.append(int(f))
            tids.append(int(r["id"]))
            gids.append(owner[int(r["id"])])
            hs.append(y2 - y1)
            crops.append(c)
        if k % 50 == 0:
            print(f"  frame {f} ({k}/{len(picks)})  {len(fs)} crops", flush=True)
    cap.release()
    # REFUSE to write an empty cache over a good one. cv2.VideoCapture fails
    # SILENTLY on a missing file, so the frame loop collected nothing and the
    # unconditional savez below then overwrote an irreplaceable 16 MB / 160 MB
    # cache with ~1 KB of nothing, printed "0 crops", and exited 0. That is the
    # worst shape a harness can have: it does not merely report success while
    # testing nothing, it destroys the evidence on the way past.
    if not fs:
        raise SystemExit(
            f"collected 0 crops from {video!r} - refusing to overwrite {out!r}. "
            "The capture opened empty (a missing or unreadable video reports no "
            "error), so there is nothing to cache.")
    np.savez_compressed(
        out, f=np.array(fs, np.int32), tid=np.array(tids, np.int32),
        gid=np.array(gids), h=np.array(hs, np.float32),
        crop=np.array(crops, np.uint8),
        team=np.array([team[g] for g in gids]))
    print(f"{len(fs)} crops, {len(set(gids))} GT players -> {out}")
    return out


# ---------------------------------------------------------------------------
# Embeddings. Every one returns L2-normalised rows, so cosine is a dot product.

def _norm(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def embed_hist(crops: np.ndarray, bins: int = 8) -> np.ndarray:
    """Joint hue/saturation/value histogram of the torso. The free baseline.

    This is what the pipeline already has, in a richer readout than the mean
    Lab colour `track.team_by_shirt` takes. If SigLIP cannot beat it there is
    no case for 780 MB of weights.
    """
    import cv2  # noqa: PLC0415

    out = []
    for c in crops:
        hsv = cv2.cvtColor(c[:, :, ::-1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3,
                            [0, 180, 0, 256, 0, 256])
        out.append(hist.ravel())
    return _norm(np.array(out, np.float32))


def embed_siglip(crops: np.ndarray, batch: int = 64) -> np.ndarray:
    """SigLIP vision tower on the torso crop. Apache-2.0 weights."""
    sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
    import teamembed  # noqa: PLC0415

    return _norm(teamembed.embed(list(crops), batch=batch))


METHODS = {"hist": embed_hist, "siglip": embed_siglip}


# ---------------------------------------------------------------------------

def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive outranks a random negative.

    Rank-based rather than a threshold sweep, so it is exact and needs no bins.
    Ties get half credit, which is what makes a constant embedding score 0.5
    instead of 1.0 - and a constant embedding is a real failure mode here.
    """
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # Average ranks within ties.
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def pairs(z, rng: np.random.Generator, n: int = 40000, match_gap: bool = True,
          same_team_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs: positives (same GT player) and negatives (different).

    `same_team_only` is the whole point. A negative drawn across the two banks
    is a question the team gate in `stitch.py` already answers, so scoring on
    those measures a capability the pipeline HAS and inflates every number. The
    open problem is two players in identical shirts.

    `match_gap` resamples the negatives so their frame-gap histogram matches the
    positives'. Without it, positives are seconds apart and negatives minutes,
    and the embedding gets credit for reading the floodlights.
    """
    f, gid, team = z["f"], z["gid"], z["team"]
    by_player: dict[str, np.ndarray] = {g: np.flatnonzero(gid == g) for g in np.unique(gid)}
    players = [g for g, ix in by_player.items() if len(ix) >= 2]

    pos = []
    for _ in range(n):
        g = players[rng.integers(len(players))]
        i, j = rng.choice(by_player[g], 2, replace=False)
        pos.append((i, j))
    pos_arr = np.array(pos)
    gaps = np.abs(f[pos_arr[:, 0]] - f[pos_arr[:, 1]])

    # Bin the positive gaps, then draw each negative from the same bin.
    edges = np.unique(np.quantile(gaps, np.linspace(0, 1, 9)))
    want = np.histogram(gaps, edges)[0] if match_gap else None

    neg: list[tuple[int, int]] = []
    tries = 0
    target = list(want) if match_gap else [n]
    for b in range(len(target)):
        got = 0
        while got < target[b] and tries < n * 60:
            tries += 1
            g1 = players[rng.integers(len(players))]
            cand = [g for g in players if g != g1 and (not same_team_only or
                    team[by_player[g][0]] == team[by_player[g1][0]])]
            if not cand:
                continue
            g2 = cand[rng.integers(len(cand))]
            i = by_player[g1][rng.integers(len(by_player[g1]))]
            j = by_player[g2][rng.integers(len(by_player[g2]))]
            if match_gap:
                d = abs(int(f[i]) - int(f[j]))
                if not (edges[b] <= d <= edges[b + 1]):
                    continue
            neg.append((i, j))
            got += 1
    return pos_arr, np.array(neg)


def score(z, X: np.ndarray, rng: np.random.Generator, match_gap: bool = True,
          same_team_only: bool = True) -> dict:
    """AUC overall and by the SMALLER box height of the pair."""
    p, n = pairs(z, rng, match_gap=match_gap, same_team_only=same_team_only)
    if not len(n):
        return {"AUC": float("nan"), "n_pos": len(p), "n_neg": 0, "bands": {}}
    sp = (X[p[:, 0]] * X[p[:, 1]]).sum(1)
    sn = (X[n[:, 0]] * X[n[:, 1]]).sum(1)
    out = {"AUC": round(_auc(sp, sn), 4), "n_pos": len(p), "n_neg": len(n), "bands": {}}
    h = z["h"]
    hp = np.minimum(h[p[:, 0]], h[p[:, 1]])
    hn = np.minimum(h[n[:, 0]], h[n[:, 1]])
    for lo, hi in ((0, 30), (30, 50), (50, 80), (80, 10000)):
        a, b = (hp >= lo) & (hp < hi), (hn >= lo) & (hn < hi)
        if a.sum() > 50 and b.sum() > 50:
            out["bands"][f"{lo}-{hi if hi < 10000 else '+'} px"] = {
                "AUC": round(_auc(sp[a], sn[b]), 4), "n": int(a.sum())}
    return out


def evaluate(crops: str = CROPS, methods=("hist", "siglip"), seed: int = 0) -> dict:
    if not os.path.exists(crops):
        raise SystemExit(f"{crops} missing - run: python model/validation/reid_eval.py --build")
    z = np.load(crops, allow_pickle=False)
    res = {}
    for m in methods:
        X = METHODS[m](z["crop"])
        rng = np.random.default_rng(seed)
        res[m] = {
            "same team, gap-matched": score(z, X, rng, True, True),
            "same team, NOT matched": score(z, np.array(X), np.random.default_rng(seed),
                                            False, True),
            "any team, gap-matched": score(z, np.array(X), np.random.default_rng(seed),
                                           True, False),
        }
    return res


def _selfcheck() -> None:
    """The scorer and the confound control, on synthetic data."""
    # _auc: a perfect separation is 1.0, a reversed one 0.0, identical 0.5.
    assert _auc(np.array([1.0, 2, 3]), np.array([-1.0, 0])) == 1.0
    assert _auc(np.array([-1.0, 0]), np.array([1.0, 2, 3])) == 0.0
    assert _auc(np.array([1.0, 1, 1]), np.array([1.0, 1])) == 0.5, "ties must score 0.5"

    # A synthetic set: 6 players, 3 per team, each with a distinct constant
    # torso colour. An embedding of that colour must separate them perfectly.
    rng = np.random.default_rng(0)
    n_each = 30
    gid, team, f, crop = [], [], [], []
    for p in range(6):
        for k in range(n_each):
            gid.append(str(p))
            team.append("home" if p < 3 else "away")
            f.append(int(rng.integers(0, 70000)))
            c = np.zeros((SIZE, SIZE, 3), np.uint8)
            c[:, :, p % 3] = 40 + 30 * p
            c[:, :, (p + 1) % 3] = 200 - 20 * p
            crop.append(c)
    z = {"gid": np.array(gid), "team": np.array(team), "f": np.array(f),
         "h": np.full(len(gid), 60.0), "crop": np.array(crop)}
    X = embed_hist(z["crop"])
    r = score(z, X, np.random.default_rng(0))
    assert r["AUC"] > 0.99, r

    # ...and an embedding carrying NO identity must score chance, not 1.0. This
    # is the guard that matters: every metric in this repo that lied did so by
    # scoring well on a degenerate input.
    flat = np.ones((len(gid), 4), np.float32) / 2.0
    r2 = score(z, flat, np.random.default_rng(0))
    assert abs(r2["AUC"] - 0.5) < 1e-6, r2

    # The gap-matcher must actually match, and this is the arrangement where it
    # bites: two clusters of players 50,000 frames apart, all on one team. Every
    # positive is inside a cluster and small-gap; a NAIVE negative is
    # cross-cluster half the time and therefore ~50,000 frames wide.
    z["team"] = np.full(len(gid), "home")
    f2: list[int] = []
    for p in range(6):
        base = 0 if p < 3 else 50_000
        f2 += list(base + rng.integers(0, 300, n_each))
    z["f"] = np.array(f2)

    pp, nn = pairs(z, np.random.default_rng(0), n=2000, match_gap=True)
    assert len(nn) > 100, f"gap matching produced almost no negatives: {len(nn)}"
    gp = np.abs(z["f"][pp[:, 0]] - z["f"][pp[:, 1]])
    gn = np.abs(z["f"][nn[:, 0]] - z["f"][nn[:, 1]])
    assert gn.max() < 1000, f"a matched negative crossed the clusters: {gn.max()}"
    assert abs(np.median(gp) - np.median(gn)) < 200, (np.median(gp), np.median(gn))

    # ...and without matching the confound is visible, which is what makes the
    # --no-match column worth printing.
    _, naive = pairs(z, np.random.default_rng(0), n=2000, match_gap=False)
    gnaive = np.abs(z["f"][naive[:, 0]] - z["f"][naive[:, 1]])
    assert np.median(gnaive) > 10_000, np.median(gnaive)

    # same_team_only must never emit a cross-team negative.
    _, nn2 = pairs(z, np.random.default_rng(0), n=800, same_team_only=True)
    assert all(z["team"][i] == z["team"][j] for i, j in nn2), "cross-team negative leaked"
    print("reid_eval.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--methods", default="hist,siglip")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.build:
        build(a.frames)
    else:
        print(json.dumps(evaluate(methods=tuple(a.methods.split(","))), indent=2))
