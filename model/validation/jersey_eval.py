"""N49: score the jersey numbers THE PRODUCT WRITES, against SoccerTrack's own.

WHY THIS EXISTS
    `app/server/jobs.py:_try_identity` runs the `assisted` strategy on every job.
    That calls `identity.get_ocr_backend("auto")`; paddleocr is absent, docTR
    1.1.0 is installed, so docTR recognition runs, and any track group whose
    vote clears `min_conf=0.6` has its number written into every one of its
    detections as `jersey` (`jobs.py:1029`) and shown to the user. NOTHING has
    ever measured how often that number is right. Two measurements exist and
    neither is this question: a best-crop back-gate study over a half (5 -> 14
    of 22 players, `jersey-ocr-measured-back-gate-is-the-fix` (internal note))
    and four recognisers on DFL crops (`jersey-ocr-four-models-all-fail.md`).
    Both score OCR. This scores THE PRODUCT'S WRITE.

PROTOCOL, fixed and committed before the first run (2026-09-20). n=1
configuration; nothing here is tuned, so there is no sweep for an argmax to
hide in. The min_conf curve at the end is a COST CURVE recomputed from one
run's cached votes, not a selection.

    population   SoccerTrack v2 match 118575, 1st half panorama
                 (`data/footage/118575_panorama_1st_half.mp4`, 71,850 frames,
                 4096x1080), frames 15,000-29,999 - a contiguous 10-minute
                 window, minutes 10:00-20:00 of the half.
                 DROP: 79.1% of the half's frames are not scanned. Direction of
                 the bias, stated before the run: a merged group gets fewer
                 sightings than it would over a whole half, and `vote_number`
                 discounts by read count, so COVERAGE here is a floor on a
                 full-match run. Precision has no obvious direction.
                 The first two minutes are excluded on team_eval.py's rule -
                 the teams swap ends and the sun moves, so the opening minutes
                 are not the half.
    tracks       `data/jobs/118575_full_1st/tracks.raw.json` - the tracker
                 output the identity stage actually consumes, from the job that
                 ran this match (rf-detr-medium, remap calibration, 2026-08-16).
                 The detector's box w/h is REJOINED from that job's
                 `dets.cache.json` on the foot point (team_eval.build's join,
                 exact to 2e-4 px). Today's `track.py` emits w/h and this file
                 predates it; without them `identity.torso_box` falls back to
                 its broadcast height model, which its own docstring says is
                 wrong in both directions on a flattened panorama. Detections
                 with no box are dropped and counted.
    code path    `jobs.apply_identity` itself - the function a job runs - with
                 FUTTRAFFIC_IDENTITY unset (`assisted`), min_conf 0.6, ocr
                 "auto" -> docTR 1.1.0 recognition, and the job's own
                 calibration. No roster is passed, exactly as jobs.py:1014
                 does not pass one.
    truth        `model/validation/soccertrack/118575_1st.tracks.json.gz`: 22
                 ids, per-detection METRE positions, one constant `jersey` per
                 id (ids 5 and 15 both wear 14, one per team). Frames rebased
                 to 0, as team_eval.load_gt does.
    join         track feet -> `cal.to_pitch`, greedy nearest GT player inside
                 2.0 m per frame (assoc_eval/team_eval's gate and reason: the
                 calibration's own residual is about a metre). A group's true
                 player is the modal GT id over its matched detections. A group
                 needs >= 20 matched detections and >= 60% agreement on one id,
                 or it is NO-TRUTH and is excluded from precision - counted and
                 reported, never silently dropped. Officials have no GT row at
                 all, so a number written on one lands here.
    metrics      P1  written-number precision per GROUP: of the groups that
                     come out with a jersey at conf >= 0.6, the share whose
                     number is the true player's.
                 P2  the same, DETECTION-WEIGHTED: a wrong number on a long
                     track corrupts more of the product than one on a stub.
                 C1  coverage: of the 22 GT players, how many are carried
                     CORRECTLY by at least one written group; and how many
                     receive a written number of any kind.
                 C2  share of the window's detections that carry a jersey.
                 E   error profile: 1- vs 2-digit, box height, and distance
                     from the camera in metres, correct vs wrong.
                 S   the min_conf cost curve at 0.6 / 0.7 / 0.8 / 0.9,
                     recomputed from the cached votes through the product's own
                     `vote_number` and `_drop_duplicates`.
    the bar      FIXED BEFORE THE RUN. "Safe to show a user" is P1 >= 0.90 AND
                 P2 >= 0.90 with C1 >= 8 of 22. Reason: the number is rendered
                 as an identity with no provenance on every surface but the
                 setup grid, a wrong one silently relabels a physical profile,
                 and retyping it is cheap - so under 9 in 10 right the proposal
                 costs more operator attention than it saves.

    CONFIG NOTE (2026-09-28): N49's published figures were measured with the appearance
    term ON (scan_video then returned embeddings too); that config was retired on 2026-09-20.
    --run today measures the current config and will not reproduce them; N49 was not re-run.

    .venv/bin/python model/validation/jersey_eval.py --selfcheck   # seconds
    .venv/bin/python model/validation/jersey_eval.py --run         # ~15 min
    .venv/bin/python model/validation/jersey_eval.py --score       # seconds
"""
import argparse
import collections
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assoc_eval  # noqa: E402 - GT positions, the 2 m gate and the per-frame match

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")
VIDEO = os.path.join(ROOT, "data/footage/118575_panorama_1st_half.mp4")
OUT = os.path.join(ROOT, "model/validation/out/n49_jersey")
LO, HI = 15000, 30000
TOTAL_FRAMES = 71850
GATE_M = assoc_eval.GATE_M       # 2.0 m, and its reason, live in assoc_eval
MIN_MATCHED, MIN_AGREE = 20, 0.6
CONFS = (0.6, 0.7, 0.8, 0.9)


# --- inputs ------------------------------------------------------------------

def build_tracks(lo: int = LO, hi: int = HI) -> tuple[dict, dict]:
    """Product tracks in the window, with the DETECTOR's box rejoined.

    The tracks file records the foot point only (box centre-x, box bottom-y),
    so the box comes back by joining to `dets.cache.json` on that point. Exact
    to 2e-4 px - the float32 rounding the tracks file went through - so a miss
    means no box for that track this frame, and it is dropped, not guessed.
    """
    dets = [d for d in json.load(open(f"{JOB}/tracks.raw.json"))["detections"]
            if lo <= d["f"] < hi]
    cache = json.load(open(f"{JOB}/dets.cache.json"))["frames"]
    by_frame: dict[int, list] = collections.defaultdict(list)
    for d in dets:
        by_frame[d["f"]].append(d)

    out, missed = [], 0
    for f, rows in by_frame.items():
        boxes = cache.get(str(f)) or []
        if not boxes:
            missed += len(rows)
            continue
        B = np.array([[(b["xyxy"][0] + b["xyxy"][2]) / 2, b["xyxy"][3]] for b in boxes])
        for r in rows:
            p = np.array([r["x"], r["y"]])
            j = int(np.hypot(*(p - B).T).argmin())
            if np.hypot(*(p - B[j])) > 0.01:
                missed += 1
                continue
            x1, y1, x2, y2 = boxes[j]["xyxy"]
            out.append({**r, "w": x2 - x1, "h": y2 - y1})
    out.sort(key=lambda d: (d["f"], d["id"]))
    stats = {"window": [lo, hi], "dets_in_window": len(dets),
             "dets_without_a_box": missed,
             "box_join_drop_share": round(missed / max(len(dets), 1), 4),
             "tracks": len({d["id"] for d in out})}
    return {"fps": 25.0, "detections": out}, stats


def load_gt(path: str = GT) -> tuple[dict, dict]:
    """frame -> {gt id: xy metres}, and {gt id: (jersey, team)}.

    The positions and the frame rebasing are `assoc_eval.load_gt`'s - the same
    file, the same 3,000,001 offset, one implementation. Only the per-id jersey
    and team, which nothing else needs, are read here.
    """
    per_frame, _ = assoc_eval.load_gt(path)
    who = {str(r["id"]): (int(r["jersey"]), r["team"])
           for r in json.load(gzip.open(path))["detections"]}
    return per_frame, who


# --- the run -----------------------------------------------------------------

def run() -> None:
    os.makedirs(OUT, exist_ok=True)
    # Nothing here writes into the real store, and a stray write would land in
    # the product's data dir - see data-dir-rebind-needs-functions (internal note).
    os.environ["FUTTRAFFIC_DATA"] = os.path.join(OUT, "_data")
    os.environ.pop("FUTTRAFFIC_IDENTITY", None)      # the default is what ships
    import identity                                  # noqa: PLC0415
    import jobs                                      # noqa: PLC0415

    t0 = time.time()
    tracks, stats = build_tracks()
    print(f"[{time.time() - t0:.0f}s] {stats}", flush=True)

    # Record what the OCR pass produced, pass it through untouched: the cost
    # curve below is then arithmetic on one run rather than four runs.
    real_scan = identity.scan_video

    def recording_scan(*a, **kw):
        votes, rep = real_scan(*a, **kw)      # (votes, report) since 2026-09-20; was (emb, votes, report)
        json.dump({"votes": {str(k): v for k, v in votes.items()}, "scan": rep},
                  open(f"{OUT}/votes.json", "w"))
        return votes, rep

    identity.scan_video = recording_scan
    cal = jobs.load_calibration(f"{JOB}/calibration.json")
    tracks, report = jobs.apply_identity(identity, tracks, VIDEO, cal)
    print(f"[{time.time() - t0:.0f}s] {json.dumps(report)}", flush=True)

    d = tracks["detections"]
    np.savez_compressed(
        f"{OUT}/written.npz",
        f=np.array([x["f"] for x in d], np.int32),
        group=np.array([x["id"] for x in d], np.int32),
        raw=np.array([x.get("raw_id", x["id"]) for x in d], np.int32),
        x=np.array([x["x"] for x in d], np.float64),
        y=np.array([x["y"] for x in d], np.float64),
        h=np.array([x["h"] for x in d], np.float32),
        jersey=np.array([-1 if x.get("jersey") is None else x["jersey"] for x in d],
                        np.int32),
        conf=np.array([x.get("jersey_conf") or 0.0 for x in d], np.float32),
        team=np.array([x.get("team") or "?" for x in d]))
    json.dump({"build": stats, "identity_report": report,
               "seconds": round(time.time() - t0, 1)},
              open(f"{OUT}/run.json", "w"), indent=2)
    print(f"[{time.time() - t0:.0f}s] wrote {OUT}/written.npz", flush=True)


# --- scoring -----------------------------------------------------------------

def match_to_gt(f: np.ndarray, xy: np.ndarray, gt: dict,
                gate: float = GATE_M) -> np.ndarray:
    """Per-detection GT id, or '' - `assoc_eval.greedy_match`, frame by frame.

    This used to be a fourth per-frame matcher, vectorised and GLOBAL-minimum
    greedy where the other three are GT-major. The two disagree only where two
    GT players contest one detection inside 2 m, but "only where" is not "never"
    and one of them had to be the harness's rule. The shared one wins: its
    behaviour is what assoc_eval's IDF1 and team_eval's accuracy were measured
    under, and re-running `--score` (seconds, off the cached written.npz) shows
    every recorded number in docs/experiments/n49-jersey-ocr.md unchanged.
    """
    out = np.array([""] * len(f), dtype=object)
    by_frame: dict[int, list[int]] = collections.defaultdict(list)
    for i, fr in enumerate(f):
        by_frame[int(fr)].append(i)
    for fr, idx in by_frame.items():
        g = gt.get(fr)
        if not g:
            continue
        for gid, i in assoc_eval.greedy_match(g, {i: xy[i] for i in idx}, gate).items():
            out[i] = gid
    return out


def _truth_per_group(z, gt: dict, cal) -> tuple[dict, dict]:
    """group -> (gt id | None, matched, total), and the per-detection gt ids."""
    xy = cal.to_pitch(np.stack([z["x"], z["y"]], 1))
    gid = match_to_gt(z["f"], xy, gt)
    per: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    total: collections.Counter = collections.Counter()
    for g, t in zip(z["group"], gid):
        total[int(g)] += 1
        if t:
            per[int(g)][t] += 1
    truth = {}
    for g, n in total.items():
        c = per.get(g, collections.Counter())
        m = sum(c.values())
        best, k = (c.most_common(1) + [(None, 0)])[0]
        ok = m >= MIN_MATCHED and k / max(m, 1) >= MIN_AGREE
        truth[g] = (best if ok else None, m, n)
    return truth, gid


def score() -> None:
    import calibrators                                # noqa: PLC0415
    import identity                                   # noqa: PLC0415

    z = np.load(f"{OUT}/written.npz")
    cal = calibrators.load(f"{JOB}/calibration.json")
    gt, who = load_gt()
    truth, gid = _truth_per_group(z, gt, cal)
    run = json.load(open(f"{OUT}/run.json"))
    votes = {int(k): [tuple(v) for v in vv]
             for k, vv in json.load(open(f"{OUT}/votes.json"))["votes"].items()}

    # group -> written number at the SHIPPED gate, straight off the detections
    first = {}
    for g, j, c in zip(z["group"], z["jersey"], z["conf"]):
        first.setdefault(int(g), (int(j), float(c)))
    dets_per_group = collections.Counter(int(g) for g in z["group"])
    cam = _camera_xy(cal)
    hm = collections.defaultdict(list)
    for g, h in zip(z["group"], z["h"]):
        hm[int(g)].append(float(h))

    rows = []
    for g, (num, conf) in sorted(first.items()):
        if num < 0:
            continue
        true_id, matched, n = truth[g]
        true_num = who[true_id][0] if true_id else None
        rows.append({"group": g, "written": num, "conf": round(conf, 3),
                     "true": true_num, "dets": n, "matched": matched,
                     "box_h": round(float(np.mean(hm[g])), 1),
                     "cam_m": _cam_dist(g, z, gid, gt, cam)})

    judged = [r for r in rows if r["true"] is not None]
    right = [r for r in judged if r["written"] == r["true"]]
    dw_r = sum(r["dets"] for r in right)
    dw_j = sum(r["dets"] for r in judged)
    carried = sum(n for g, n in dets_per_group.items() if first[g][0] >= 0)

    covered_ok = {r["true"] for r in right}
    covered_any = {r["true"] for r in judged}
    # C1 and players_given_a_wrong_number OVERLAP: a player can hold a correct
    # group AND a wrong one, so 6 + 8 > the 11 who got any number at all.
    # Reporting the two alone reads as 14 distinct players; these three keys are
    # a partition and sum to covered_any.
    covered_bad = {r["true"] for r in judged if r["written"] != r["true"]}

    out = {
        "population": run["build"], "identity_report": run["identity_report"],
        "frames_scanned_share": round((HI - LO) / TOTAL_FRAMES, 4),
        "groups_total": len(dets_per_group),
        "groups_written": len(rows),
        "groups_written_no_truth": len(rows) - len(judged),
        "P1_group_precision": _r(len(right), len(judged)),
        "P2_detection_weighted": _r(dw_r, dw_j),
        "C1_players_correct": len(covered_ok),
        "C1_players_any_number": len(covered_any),
        "players_partition": {"only_right": len(covered_ok - covered_bad),
                              "both_a_right_and_a_wrong": len(covered_ok & covered_bad),
                              "only_wrong": len(covered_bad - covered_ok)},
        "C2_detections_with_a_jersey": _r(carried, len(z["f"])),
        "digits": {
            "written_1_digit": _r(sum(r["written"] < 10 for r in judged), len(judged)),
            "right_when_1_digit": _r(sum(r["written"] == r["true"] for r in judged
                                         if r["written"] < 10),
                                     sum(r["written"] < 10 for r in judged)),
            "right_when_2_digit": _r(sum(r["written"] == r["true"] for r in judged
                                         if r["written"] >= 10),
                                     sum(r["written"] >= 10 for r in judged)),
            "true_1_digit_share": _r(sum(v[0] < 10 for v in who.values()), len(who)),
        },
        "box_h_px": {"right": _mean([r["box_h"] for r in right]),
                     "wrong": _mean([r["box_h"] for r in judged
                                     if r["written"] != r["true"]])},
        "camera_m": {"right": _mean([r["cam_m"] for r in right if r["cam_m"]]),
                     "wrong": _mean([r["cam_m"] for r in judged
                                     if r["written"] != r["true"] and r["cam_m"]])},
        "wrong_numbers": collections.Counter(
            r["written"] for r in judged if r["written"] != r["true"]).most_common(),
        "min_conf_curve": _curve(votes, z, truth, who, identity),
        # POST HOC, not pre-registered: the arm the error profile suggests.
        # Reported as a proposal that needs its own run on another match, never
        # as a result - it is chosen after seeing which writes were wrong.
        "post_hoc_two_digit_only": {
            "groups_written": sum(r["written"] >= 10 for r in rows),
            "P1": _r(sum(r["written"] == r["true"] for r in judged if r["written"] >= 10),
                     sum(r["written"] >= 10 for r in judged)),
            "P2": _r(sum(r["dets"] for r in right if r["written"] >= 10),
                     sum(r["dets"] for r in judged if r["written"] >= 10)),
            "players_correct": len({r["true"] for r in right if r["written"] >= 10}),
            "players_wrong": len({r["true"] for r in judged
                                  if r["written"] != r["true"] and r["written"] >= 10}),
        },
        "players_given_a_wrong_number": len(covered_bad),
        "wrong_write_is_a_digit_of_the_true_number": sum(
            str(r["written"]) in str(r["true"]) for r in judged
            if r["written"] != r["true"]),
        "rows": rows,
    }
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


def _curve(votes, z, truth, who, identity) -> list[dict]:
    """The product's own tail (vote -> duplicate drop -> gate) at four gates."""
    members = collections.defaultdict(set)
    team = {}
    for g, r, t in zip(z["group"], z["raw"], z["team"]):
        members[int(g)].add(int(r))
        team.setdefault(int(g), str(t))
    dets = collections.Counter(int(g) for g in z["group"])

    out = []
    for mc in CONFS:
        strat = identity.AssistedIdentity(min_conf=mc)
        decided = {}
        for g, mem in members.items():
            num, conf = identity.vote_number([v for m in mem for v in votes.get(m, [])])
            decided[g] = identity.PlayerId(team[g], num, None, None, conf,
                                           "ocr" if num else "unassigned", g)
        decided = strat._drop_duplicates(decided)   # noqa: SLF001 - the product's own step
        written = {g: p for g, p in decided.items()
                   if p.assigned and p.confidence >= mc}
        judged = [(g, p) for g, p in written.items() if truth[g][0] is not None]
        right = [(g, p) for g, p in judged if p.number == who[truth[g][0]][0]]
        out.append({
            "min_conf": mc, "groups_written": len(written),
            "no_truth": len(written) - len(judged),
            "P1": _r(len(right), len(judged)),
            "P2": _r(sum(dets[g] for g, _ in right),
                     sum(dets[g] for g, _ in judged)),
            "players_correct": len({truth[g][0] for g, _ in right}),
            "detections_with_a_jersey": _r(sum(dets[g] for g in written), len(z["f"])),
        })
    return out


def _camera_xy(cal):
    import sceneprior                                  # noqa: PLC0415
    own = getattr(cal, "camera_centre_m", None)
    if own is not None:
        return np.array(own[:2], float)
    try:
        return np.array(sceneprior.camera_from_homography(
            cal.H, (11878, 870)).centre_m[:2])
    except Exception:                                  # noqa: BLE001
        return None


def _cam_dist(g, z, gid, gt, cam):
    """Mean distance of this group's MATCHED GT positions from the camera."""
    if cam is None:
        return None
    d = [float(np.hypot(*(gt[int(f)][t] - cam)))
         for f, gg, t in zip(z["f"], z["group"], gid)
         if int(gg) == g and t and int(f) in gt]
    return round(float(np.mean(d)), 1) if d else None


def _r(a, b):
    return None if not b else round(a / b, 4)


def _mean(v):
    return round(float(np.mean(v)), 1) if v else None


# --- self-check --------------------------------------------------------------

def selfcheck() -> None:
    """The two pieces of logic that can silently produce a wrong number."""
    # 1. the greedy per-frame GT match: nearest inside the gate, one GT id at
    #    most once per frame, and nothing matched outside it.
    f = np.array([0, 0, 1])
    xy = np.array([[10.0, 10.0], [10.5, 10.0], [99.0, 60.0]])
    gt = {0: {"a": np.array([10.1, 10.0]), "b": np.array([30.0, 10.0])},
          1: {"a": np.array([10.0, 10.0])}}
    got = list(match_to_gt(f, xy, gt))
    assert got[0] == "a" and got[1] == "" and got[2] == "", got

    # 2. the truth rule: a group under the agreement or the count floor has no
    #    truth, and must never enter precision as a wrong answer.
    class _Cal:
        @staticmethod
        def to_pitch(p):
            return p
    n = MIN_MATCHED
    z = {"f": np.arange(2 * n), "x": np.full(2 * n, 10.0), "y": np.full(2 * n, 10.0),
         "group": np.r_[np.zeros(n, int), np.ones(n, int)]}
    gt2 = {i: {"a" if i < n + n // 2 else "b": np.array([10.0, 10.0])}
           for i in range(2 * n)}
    truth, _ = _truth_per_group(z, gt2, _Cal)
    assert truth[0][0] == "a", truth
    assert truth[1][0] is None, truth      # 50/50 split -> no truth, not a guess
    print("jersey_eval.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.score:
        score()
    if a.selfcheck or not (a.run or a.score):
        selfcheck()
