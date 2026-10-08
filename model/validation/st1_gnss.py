"""SoccerTrack v1's GNSS: player positions that come from no camera - and whether they can serve
as truth for the wide-view video recorded with them.

DATA - Kaggle `atomscott/soccertrack`, LGPL-3.0 on the dataset page (research-data-hunt (not published)
§3.2), fetched 2026-09-14 with the kaggle CLI into ~/Downloads/soccertrack-v1 (FUTTRAFFIC_ST1
overrides), 64 files:
    GNSS/G_20200220_1_<s0>_<s1>.csv                    30 chunks of 30 s, session 2020-02-20 #1
    wide_view/annotations/F_20200220_1_<s0>_<s1>.csv   boxes in the fisheye wide view, same chunks
    wide_view/videos/F_20200220_1_0000_0030.mp4        the first chunk only: 6500 x 1000, 25 fps
    gnss_keypoints.json, fisheye_keypoints.json        65 pitch points (x, y) m -> [lon, lat] / pixel
    GNSS rows: time of day, then Lat, Lon per (TeamID, PlayerID), two teams of 11. Box rows: frame
    1..750, then bb_height, bb_left, bb_top, bb_width per (TeamID, PlayerID), then BALL.
    Read while writing the loader, so a description and not a test: GNSS times sit on a 0.1 s
    grid, some repeated (the first row is kept) and a few steps missing, with no blank cells.
    Not fetched: the drone video and its 2022 session, and the rest of the wide-view videos.

METRES
    Latitude/longitude -> east/north metres, equirectangular about the keypoints' centroid, then one
    least-squares affine map onto the pitch frame through the 65 keypoints. Its singular values are
    the pitch's scale against the ground: a pitch whose labels are the wrong size reads away from 1,
    and so does every speed measured in it.

USABLE AS TRUTH - written 2026-09-14, before any of it was computed:
    1. GEOREFERENCED: the affine fits the 65 keypoints to RMS <= 1.0 m, and both singular values are
       within 0.02 of 1.
    2. ON THE VIDEO, WITHOUT FITTING THE PLAYERS: GNSS positions go into the wide view through a
       thin-plate spline on fisheye_keypoints.json - the dataset's own keypoints, never the players -
       on one video frame per second of every chunk. The time offset is the one in a +-3.0 s scan at
       0.1 s that minimises the median pixel distance from each player's projected position to that
       player's own box feet, pooled over the chunks; every chunk's own best offset must sit within
       0.2 s of it; and at it, Hungarian on box feet must return the given (TeamID, PlayerID) for at
       least 90% of player-frames.
    Either failing -> NOT USABLE, and which. Because the map is not fitted to players, a mirrored or
    renumbered feed shows up here as cost: the failure st2_team_gt's free fit could not see.

FOUND AFTER THE CHECK - EXPLORATORY, 2026-09-14, not registered
    `--check` read NOT USABLE: as the files state them, the GNSS never lands on its boxes. A search
    without labels over the whole session then found two faults on the video side, not in the GNSS:
    the wide view runs on a 30 fps time base though it is encoded and named at 25 - chunk s0 starts
    s0 * 25/30 s into the GNSS session and its frame i (i - 1)/30 s after that - and the boxes'
    PlayerIDs are the GNSS ones shifted by two. `--diagnose` recomputes both: per chunk, on the 30 fps
    base, the offset that minimises the median distance of a label-free Hungarian match, the
    permutation solved from the chunk's summed distances at it, and how often single frames agree with
    that permutation. It was found by searching all 30 chunks, so it cannot confirm itself, and there
    is no second GNSS session to confirm it on.
    REVIEW, same day: a statistic needing no labels - the players' mean position over every frame
    against the GNSS - confirms the 30 fps base. Chunks 30-90 s carry a second, box-side numbering, a
    third fault. With all three undone the criteria above still fail, so the GNSS is not shown to be
    truth. soccertrack-v1-gnss (not published) carries it.

    .venv/bin/python model/validation/st1_gnss.py --check       # writes model/validation/out/st1_gnss.json
    .venv/bin/python model/validation/st1_gnss.py --diagnose    # writes model/validation/out/st1_gnss_diagnose.json
    .venv/bin/python model/validation/st1_gnss.py --selfcheck
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ST1 = os.environ.get("FUTTRAFFIC_ST1", os.path.expanduser("~/Downloads/soccertrack-v1"))
SESSION = "20200220_1"
OUT = os.path.join(ROOT, "model/validation/out/st1_gnss.json")
DIAGNOSE_OUT = os.path.join(ROOT, "model/validation/out/st1_gnss_diagnose.json")
FPS = 25.0
GNSS_STEP_S = 0.1
R_EARTH_M = 6371008.8

#: USABLE AS TRUTH, as the docstring states it.
GEOREF_RMS_M = 1.0
SCALE_TOL = 0.02
OFFSET_SCAN_S = 3.0
OFFSET_AGREE_S = 0.2
AGREEMENT_MIN = 0.90


def _files(folder: str, prefix: str) -> list[str]:
    fs = sorted(glob.glob(f"{ST1}/{folder}/{prefix}_{SESSION}_*.csv"))
    if not fs:
        raise SystemExit(f"no {folder}/{prefix}_{SESSION}_*.csv under {ST1} - fetch it with "
                         "`kaggle datasets download atomscott/soccertrack -f <path>` (see the docstring)")
    return fs


def _keypoints(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(pitch metres, values) from one of the dataset's 65-point files."""
    d = json.load(open(f"{ST1}/{name}"))
    return (np.array([[float(v) for v in k.strip("()").split(",")] for k in d], float),
            np.array(list(d.values()), float))


def load_gnss() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """(seconds from the first row, lat/lon T x 22 x 2, [(team, player)]), repeated times dropped."""
    labels, times, vals = None, [], []
    for f in _files("GNSS", "G"):
        with open(f) as fh:
            head = [next(fh).rstrip("\n").split(",") for _ in range(4)]
            cols = range(1, len(head[0]), 2)
            assert all(head[2][c] == "Lat" and head[2][c + 1] == "Lon" for c in cols), f
            lab = [(int(head[0][c]), int(head[1][c])) for c in cols]
            labels = labels or lab
            assert lab == labels, f"{f}: player columns differ from the first chunk"
            for line in fh:
                c = line.rstrip("\n").split(",")
                h, m, s = c[0].split(":")
                times.append(int(h) * 3600 + int(m) * 60 + float(s))
                vals.append([float(v) if v else np.nan for v in c[1:1 + 2 * len(lab)]])
    t = np.array(times)
    assert np.all(np.diff(t) >= 0), "GNSS rows out of time order"
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep] - t[0], np.array(vals)[keep].reshape(-1, len(labels), 2), labels


def _enu(lonlat: np.ndarray, origin: np.ndarray) -> np.ndarray:
    k = np.pi / 180.0 * R_EARTH_M
    return np.stack([(lonlat[..., 0] - origin[0]) * k * np.cos(np.radians(origin[1])),
                     (lonlat[..., 1] - origin[1]) * k], -1)


def _affine(pitch: np.ndarray, lonlat: np.ndarray) -> dict:
    """Ground -> pitch metres by least squares: the map, its RMS on the points, its scale."""
    origin = lonlat.mean(0)
    X = np.c_[_enu(lonlat, origin), np.ones(len(pitch))]
    M, *_ = np.linalg.lstsq(X, pitch, rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((X @ M - pitch) ** 2, 1))))
    return {"rms_m": rms, "scale": np.linalg.svd(M[:2], compute_uv=False).tolist(),
            "to_pitch": lambda ll: np.c_[_enu(np.asarray(ll, float), origin), np.ones(len(ll))] @ M}


def georef() -> dict:
    return _affine(*_keypoints("gnss_keypoints.json"))


def positions() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict]:
    """(seconds, pitch metres T x 22 x 2, labels, georeference) for the whole session."""
    t, P, labels = load_gnss()
    g = georef()
    return t, g["to_pitch"](P[..., ::-1].reshape(-1, 2)).reshape(P.shape), labels, g


def load_boxes(path: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """(frames, box feet F x 22 x 2 in pixels, [(team, player)]) for one wide-view chunk, BALL dropped."""
    with open(path) as fh:
        head = [next(fh).rstrip("\n").split(",") for _ in range(4)]
        rows = [line.rstrip("\n").split(",") for line in fh]
    cols: dict[tuple[int, int], dict[str, int]] = {}
    for c in range(1, len(head[0])):
        if head[0][c] not in ("", "BALL"):
            cols.setdefault((int(head[0][c]), int(head[1][c])), {})[head[2][c]] = c
    labels = sorted(cols)
    A = np.array([[float(v) if v else np.nan for v in r[:len(head[0])]] for r in rows])
    feet = np.stack([np.stack([A[:, cols[k]["bb_left"]] + A[:, cols[k]["bb_width"]] / 2,
                               A[:, cols[k]["bb_top"]] + A[:, cols[k]["bb_height"]]], -1)
                     for k in labels], 1)
    return A[:, 0].astype(int), feet, labels


def fisheye_map():
    from scipy.interpolate import RBFInterpolator  # noqa: PLC0415

    pitch, px = _keypoints("fisheye_keypoints.json")
    return RBFInterpolator(pitch, px, kernel="thin_plate_spline", smoothing=0.0)


def _grid(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Positions on the 0.1 s grid, NaN where a step is missing."""
    g = np.full((int(round(t[-1] / GNSS_STEP_S)) + 1, *xy.shape[1:]), np.nan)
    g[np.round(t / GNSS_STEP_S).astype(int)] = xy
    return g


def _scan(grid: np.ndarray, frame_t: np.ndarray, feet: np.ndarray, fn, offsets) -> list[dict]:
    """Per offset: pixel distances from each player's projected position to its OWN box, and how
    often Hungarian on the box feet returns the given label. Columns of `grid` and `feet` share
    one label order."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    out = []
    for d in offsets:
        dist, agree, n = [], 0, 0
        for k, ft in enumerate(frame_t):
            i = int(round((ft + d) / GNSS_STEP_S))
            if not 0 <= i < len(grid):
                continue
            if not np.isfinite(grid[i]).all():
                continue
            px = fn(grid[i])
            ok = np.isfinite(feet[k]).all(1)
            dist += np.linalg.norm(px[ok] - feet[k][ok], axis=1).tolist()
            ib = np.where(ok)[0]
            r, c = linear_sum_assignment(np.linalg.norm(px[:, None] - feet[k][ib][None], axis=2))
            agree += int(np.sum(r == ib[c]))
            n += len(r)
        out.append({"dist": dist, "agree": agree, "n": n})
    return out


def check() -> dict:
    t, xy, glab, g = positions()
    grid, fn = _grid(t, xy), fisheye_map()
    offsets = np.round(np.arange(-OFFSET_SCAN_S, OFFSET_SCAN_S + 1e-9, GNSS_STEP_S), 3)
    pooled = [{"dist": [], "agree": 0, "n": 0} for _ in offsets]
    chunks = {}
    for f in _files("wide_view/annotations", "F"):
        frames, feet, blab = load_boxes(f)
        assert sorted(glab) == blab, f"{f}: box labels differ from GNSS labels"
        s0 = int(re.search(r"_(\d{4})_\d{4}\.csv$", f).group(1))
        pick = np.arange(0, len(frames), int(FPS))
        order = [glab.index(b) for b in blab]
        sc = _scan(grid[:, order], s0 + (frames[pick] - 1) / FPS, feet[pick], fn, offsets)
        med = [np.median(s["dist"]) if s["dist"] else np.inf for s in sc]
        chunks[os.path.basename(f)] = {"best_offset_s": float(offsets[int(np.argmin(med))]), "scan": sc}
        for p, s in zip(pooled, sc):
            p["dist"] += s["dist"]
            p["agree"] += s["agree"]
            p["n"] += s["n"]
    med = [float(np.median(p["dist"])) if p["dist"] else np.inf for p in pooled]
    b = int(np.argmin(med))
    best = float(offsets[b])
    spread = max(abs(c["best_offset_s"] - best) for c in chunks.values())
    agreement = pooled[b]["agree"] / max(pooled[b]["n"], 1)
    fails = []
    if not (g["rms_m"] <= GEOREF_RMS_M and all(abs(s - 1.0) <= SCALE_TOL for s in g["scale"])):
        fails.append("GEOREFERENCED")
    if not (spread <= OFFSET_AGREE_S + 1e-9 and agreement >= AGREEMENT_MIN):
        fails.append("ON THE VIDEO")
    steps = int(round(t[-1] / GNSS_STEP_S)) + 1
    res = {
        "session_s": float(t[-1]), "players": len(glab), "gnss_rows": len(t), "grid_steps": steps,
        "grid_present": len(t) / steps,
        "georef_rms_m": g["rms_m"], "georef_scale": g["scale"],
        "offset_s": best, "offset_spread_s": spread, "median_px_at_offset": med[b],
        "agreement": agreement, "player_frames": pooled[b]["n"],
        "chunk_offsets_s": {k: c["best_offset_s"] for k, c in chunks.items()},
        "chunk_agreement_at_offset": {k: c["scan"][b]["agree"] / max(c["scan"][b]["n"], 1)
                                      for k, c in chunks.items()},
        "verdict": "USABLE" if not fails else "NOT USABLE: " + ", ".join(fails),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("chunk_")}, indent=1))
    return res


def diagnose() -> dict:
    """EXPLORATORY - the two video-side faults, chunk by chunk. See FOUND AFTER THE CHECK."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    t, xy, glab, _g = positions()
    grid, fn = _grid(t, xy), fisheye_map()
    px = fn(np.nan_to_num(grid).reshape(-1, 2)).reshape(grid.shape)
    seen = np.isfinite(grid).all((1, 2))
    offsets = np.round(np.arange(-8.0, 8.0 + 1e-9, GNSS_STEP_S), 3)
    chunks = {}
    for f in _files("wide_view/annotations", "F"):
        frames, feet, blab = load_boxes(f)
        s0 = int(re.search(r"_(\d{4})_\d{4}\.csv$", f).group(1))
        pick = np.arange(0, len(frames), 15)
        ft, F = s0 * 25.0 / 30.0 + (frames[pick] - 1) / 30.0, feet[pick]

        def costs(d):
            i = np.round((ft + d) / GNSS_STEP_S).astype(int)
            ok = (i >= 0) & (i < len(grid)) & np.isfinite(F).all((1, 2))
            ok[ok] &= seen[i[ok]]
            return [np.linalg.norm(px[i[k]][:, None] - F[k][None], axis=2) for k in np.where(ok)[0]]

        free = [np.median([np.median(C[linear_sum_assignment(C)]) for C in costs(d)] or [np.inf]) for d in offsets]
        b = int(np.argmin(free))
        Cs = costs(offsets[b])
        r, c = linear_sum_assignment(np.mean(Cs, 0))
        chunks[os.path.basename(f)] = {
            "offset_s": float(offsets[b]), "label_free_px": float(free[b]),
            "perm": [[list(glab[i]), list(blab[j])] for i, j in zip(r, c)],
            "agreement": float(np.mean([np.mean(linear_sum_assignment(C)[1] == c) for C in Cs]))}
    perms = collections.Counter(json.dumps(v["perm"]) for v in chunks.values())
    top, n_top = perms.most_common(1)[0]
    pairs = json.loads(top)
    stat = lambda k: [float(np.min(v)), float(np.median(v)), float(np.max(v))] if (  # noqa: E731
        v := [c[k] for c in chunks.values()]) else None
    res = {"chunks": chunks, "distinct_permutations": len(perms), "chunks_on_most_common": n_top,
           "team_kept": all(g[0] == b[0] for g, b in pairs),
           "id_shift_mod_11": sorted({(b[1] - g[1]) % 11 for g, b in pairs}),
           "offset_s": stat("offset_s"), "label_free_px": stat("label_free_px"), "agreement": stat("agreement")}
    os.makedirs(os.path.dirname(DIAGNOSE_OUT), exist_ok=True)
    json.dump(res, open(DIAGNOSE_OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "chunks"}, indent=1))
    return res


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    # The georeference reads a pitch drawn 5% larger on the ground than its labels as 1/1.05.
    pitch = rng.uniform([0, 0], [105, 68], (65, 2))
    th = np.radians(35.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    origin = np.array([140.1075, 36.1024])
    for s in (1.0, 1.05):
        en = (pitch * s) @ R.T + [20.0, -15.0]
        k = np.pi / 180.0 * R_EARTH_M
        lonlat = np.c_[origin[0] + en[:, 0] / (k * np.cos(np.radians(origin[1]))), origin[1] + en[:, 1] / k]
        a = _affine(pitch, lonlat)
        assert a["rms_m"] < 0.01 and np.allclose(a["scale"], 1.0 / s, atol=2e-3), (s, a["rms_m"], a["scale"])
        assert np.allclose(a["to_pitch"](lonlat), pitch, atol=0.01)

    # A missing time lands on the grid as NaN, not as a shifted sample.
    t = np.array([0.0, 0.1, 0.2, 0.4])
    g = _grid(t, np.arange(4.0)[:, None, None] * np.ones((1, 1, 2)))
    assert g.shape[0] == 5 and np.isnan(g[3]).all() and g[4, 0, 0] == 3.0

    # The join: 22 walkers seen 1.3 s later by the camera. The scan finds the lag, the given labels
    # win Hungarian there, and a mirrored feed does not win it anywhere.
    fn = lambda p: np.asarray(p, float) * [40.0, 10.0] + [100.0, 50.0]  # noqa: E731
    n = 300
    xy = np.cumsum(rng.normal(0, 0.3, (n, 22, 2)), 0) + rng.uniform([10, 10], [95, 58], (1, 22, 2))
    grid = _grid(np.arange(n) * GNSS_STEP_S, xy)
    frame_t = np.arange(0.0, 25.0, 1.0)
    feet = np.stack([fn(grid[int(round((ft + 1.3) / GNSS_STEP_S))]) for ft in frame_t])
    offsets = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.1), 3)
    sc = _scan(grid, frame_t, feet, fn, offsets)
    b = int(np.argmin([np.median(s["dist"]) for s in sc]))
    assert abs(offsets[b] - 1.3) < 1e-6 and sc[b]["agree"] / sc[b]["n"] > 0.9, (offsets[b], sc[b]["agree"])
    mirrored = _scan(grid * [1.0, -1.0] + [0.0, 68.0], frame_t, feet, fn, offsets)
    assert max(s["agree"] / s["n"] for s in mirrored) < 0.5, "a mirrored feed joins the boxes"
    print("st1_gnss.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--diagnose", action="store_true", help="exploratory; see FOUND AFTER THE CHECK")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.check:
        check()
    elif a.diagnose:
        diagnose()
    else:
        ap.print_help()
