"""databallpy's tracking-data quality check, ported - and the two holes closed.

WHERE IT COMES FROM
    `databallpy/data_parsers/tracking_data_parsers/utils/_quality_check_tracking_data.py`
    (MIT, (c) 2023 Alexander Oonk & Daan Grob,
    https://github.com/Alek050/databallpy). Thresholds are theirs, not invented
    here: 12 m/s as the fastest speed recorded in football, 50 m/s as the hardest
    shot, 0.5% of a player's play time, 1% of frames, a 1 s gap, 7 players a side,
    30 s. The arithmetic is rewritten against our own data shapes; nothing is
    copied. See research-data-interchange (not published) §2.

WHY THIS FILE EXISTS AT ALL
    This repo clamps and carries on. `stats.MAX_SPEED` caps a speed at 9.5 m/s
    and `api://product.speed.clamped_frac` says half of all fragments hit it;
    nothing anywhere REFUSES a track file. A gate that refuses is the instrument
    every postmortem in docs/ asks for and none of them built.

THE TWO HOLES, BOTH CLOSED HERE
    1. **Their own criterion 6 never fires.** The upstream docstring promises
       "there is a gap in the realistic player velocity for at least 1 second for
       at least one player"; `_check_player_velocity` computes
       `max_sequences_invalid_frames` and then returns without ever warning on
       it. That is the failure the internal note
       `docs-guard-must-have-a-runnable-check` is about, in somebody else's repo. It is restored as C6 below.

    2. **Feeding this our product speeds would make C5 incapable of failing.**
       `MAX_SPEED = 9.5` is BELOW the 12.0 m/s realism threshold, so every
       clamped speed passes by construction. C5 and C6 therefore run on
       `speed_series(..., method="raw")` - unfiltered and unclamped - and that
       choice is the whole reason the check can fail at all.

WHAT WE CANNOT EVALUATE, SAID ON EVERY JOB RATHER THAN PASSED OVER
    Five of the nine criteria are about the ball. `app/server/jobs.py` has run the
    ball since 66fba82 (2026-09-23), but its rows are PIXEL CENTRES of the picked box,
    not ground positions (ball.Ball: a ball in flight is not on the ground), so no
    velocity or gap in metres can be taken from them, and nothing here evaluates them.
    Those five report `not_evaluated` - never `ok`.
    The same goes for the "while the ball is alive" mask every upstream criterion
    is computed under: we have no ball status, so ours run over the whole clip,
    which is STRICTER for the presence criteria (breaks and stoppages count
    against us) and is recorded as `scope` on each row.

    A verdict of `ok` is therefore unreachable today. That is not a bug in the
    gate; it is the gate reporting what the product is.

TELEPORTS AND GAPS ARE DIFFERENT FAILURES
    Upstream differentiates naively, so a re-acquisition after an occlusion
    appears as an impossible velocity. With `api://product.distance.coverage` at
    0.70 that would fire on every job for the wrong reason. C5 and C6 count only
    steps at the track's own cadence - `speed_series`'s `real` mask - and the
    share of steps dropped as gaps is reported beside them as `gap_step_share`.
    A tracker that teleports and a tracker that loses people are two different
    problems and they get two different numbers.

WHAT IT DISCRIMINATES, MEASURED - AND WHAT IT DOES NOT
    Run across N19's four scene-prior arms on 118575, the presence criteria and
    `gap_step_share` order them the way HOTA does, using NO ground truth at all:
    the two arms N19 condemned read worst, the two it liked read best. That is
    the case for having a gate that needs no benchmark.

    C5 does not discriminate. On real tracker output it fails EVERY arm - our
    raw, unfiltered speed estimate crosses 12 m/s too often for a 0.5%-of-play-
    time rule to separate anything. A criterion that fails everything is exactly
    as useless as one that cannot fail, so it is reported as measured rather than
    quietly retuned: upstream's threshold is kept, and the upgrade path is a
    FILTERED-but-unclamped series, which does not exist today because
    `kalman_ground` clamps inside itself.

    And the gate sees recall, not precision. It cannot tell N19's v1 (no prior,
    more recall, worse precision) from v2 (polygon only, the arm that won) - it
    ranks v1 first. It separates broken from plausible. It does not replace the
    benchmark. Numbers and the run: research-data-interchange (not published) §2.1.

    .venv/bin/python model/engine/quality.py <tracks.json> <calibration.json>
    .venv/bin/python model/engine/quality.py --demo
"""
import json
import sys

import numpy as np

# Upstream's thresholds, one name each so a change to any of them is a diff.
MAX_PLAYER_MS = 12.0     # fastest player speed recorded in football
MAX_BALL_MS = 50.0       # hardest shot
PLAYER_BAD_SHARE = 0.005  # of one player's own play time
BALL_BAD_SHARE = 0.01    # of all frames
GAP_S = 1.0              # a run of unrealistic values this long is a gap
MIN_IDS_PER_SIDE = 7     # fewer than this on a side is "data not available"
MISSING_S = 30.0         # ... for longer than this, and the file is refused

NO_BALL = "ball rows are pixel centres, not ground positions; not evaluated"


def _rows_by_id(dets):
    out: dict = {}
    for d in dets:
        out.setdefault(d["id"], []).append(d)
    return out


def _longest_run(mask) -> int:
    """Longest run of True. Upstream's `_max_sequence_invalid_frames`."""
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def _row(cid, status, value, threshold, note, scope="whole clip"):
    return {"id": cid, "status": status, "value": value,
            "threshold": threshold, "note": note, "scope": scope}


def check(data: dict, cal, fps: float | None = None) -> dict:
    """The nine criteria over one loaded tracks file. Never raises on bad data.

    `data` is `{fps, detections:[{f,id,team,x,y}]}` as model/engine/track.py writes it;
    `cal` is anything with `.to_pitch` and `.H` (calibrators.load or Calibration).
    """
    import stats  # noqa: PLC0415 - stats imports pitch; keep the cycle out of load

    fps = float(fps or data.get("fps") or 25.0)
    dets = data.get("detections") or []
    by_id = _rows_by_id(dets)
    frames = sorted({int(d["f"]) for d in dets})
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    step_s = step / fps

    # C5 / C6 - player velocity, on the UNCLAMPED series (see the module docstring).
    over, worst_share, worst_gap_s, gap_steps, real_steps, skipped = 0, 0.0, 0.0, 0, 0, 0
    teleports = 0
    for track in by_id.values():
        if len(track) < 3:
            skipped += 1
            continue
        try:
            v, dt, real, _ = stats.speed_series(track, cal, fps, method="raw")
        except Exception:  # a degenerate track must not take the whole gate down
            skipped += 1
            continue
        gap_steps += int((~real).sum())
        real_steps += int(real.sum())
        v_real = v[real]
        if v_real.size == 0:
            skipped += 1
            continue
        bad = v_real > MAX_PLAYER_MS
        teleports += int(bad.sum())
        share = float(bad.mean())
        over += share > PLAYER_BAD_SHARE
        worst_share = max(worst_share, share)
        worst_gap_s = max(worst_gap_s, _longest_run(bad) * float(np.median(dt[real])))

    # C7 / C8 - is there a team on the pitch at all. Counted in ids per frame,
    # because ids are what we have: a fragmented tracker can read BOTH short
    # (the far band is missing) and long (one player, three ids), and only the
    # short direction is what upstream's criterion is about.
    per_frame: dict = {}
    for d in dets:
        t = d.get("team")
        if t in ("home", "away"):
            per_frame.setdefault((int(d["f"]), t), set()).add(d["id"])
    thin = {"home": 0, "away": 0}
    med = {"home": 0.0, "away": 0.0}
    for side in ("home", "away"):
        counts = np.array([len(per_frame.get((f, side), ())) for f in frames]) if frames else np.zeros(0)
        thin[side] = int((counts < MIN_IDS_PER_SIDE).sum())
        med[side] = float(np.median(counts)) if counts.size else 0.0

    n_ids = len(by_id)
    rows = [
        _row("C1_ball_missing", "not_evaluated", None, f"<{BALL_BAD_SHARE:.0%} of frames", NO_BALL),
        _row("C2_ball_gap", "not_evaluated", None, f"<{GAP_S} s", NO_BALL),
        _row("C3_ball_velocity", "not_evaluated", None, f"<{MAX_BALL_MS} m/s", NO_BALL),
        _row("C4_ball_velocity_gap", "not_evaluated", None, f"<{GAP_S} s", NO_BALL),
        _row("C5_player_velocity",
             "fail" if over else "ok", {"players_over": over, "of": n_ids,
                                        "worst_share": round(worst_share, 5)},
             f"no player over {MAX_PLAYER_MS} m/s for more than {PLAYER_BAD_SHARE:.1%} of play time",
             "unclamped: stats.MAX_SPEED is below the threshold, so the product "
             "speeds cannot fail this"),
        _row("C6_player_velocity_gap",
             "fail" if worst_gap_s >= GAP_S else "ok", round(worst_gap_s, 2),
             f"<{GAP_S} s", "upstream computes this and never warns on it; restored here"),
        _row("C7_home_present",
             "fail" if thin["home"] * step_s > MISSING_S else "ok",
             {"seconds": round(thin["home"] * step_s, 1), "median_ids": med["home"]},
             f"<{MISSING_S} s under {MIN_IDS_PER_SIDE} ids",
             "ids, not players - a fragmented tracker can read long as well as short"),
        _row("C8_away_present",
             "fail" if thin["away"] * step_s > MISSING_S else "ok",
             {"seconds": round(thin["away"] * step_s, 1), "median_ids": med["away"]},
             f"<{MISSING_S} s under {MIN_IDS_PER_SIDE} ids", "as C7"),
        _row("C9_ball_present", "not_evaluated", None, f"<{MISSING_S} s", NO_BALL),
    ]

    status = {r["status"] for r in rows}
    verdict = "fail" if "fail" in status else ("incomplete" if "not_evaluated" in status else "ok")
    return {
        "verdict": verdict,
        "criteria": rows,
        "n_ids": n_ids, "n_frames": len(frames), "fps": fps, "frame_step": step,
        "tracks_skipped": skipped,
        # Beside C5/C6 on purpose: a high share here means the file is full of
        # holes, which is a different complaint from an impossible velocity.
        "gap_step_share": round(gap_steps / max(gap_steps + real_steps, 1), 4),
        # NOT a criterion, and the reason is worth the line. C5 needs 0.5% of a
        # player's steps and C6 needs a run of 1 s, so a SINGLE impossible jump -
        # which is exactly what an id switch looks like - passes both. Id
        # switching is this repo's dominant failure
        # (api://product.identity.contaminated), so the count is reported raw and
        # nobody gets to set a threshold on it without measuring one first.
        "teleport_steps": teleports,
        "source": "databallpy _quality_check_tracking_data (MIT), ported",
    }


def check_files(tracks_path: str, calib_path: str) -> dict:
    import calibrators  # noqa: PLC0415
    with open(tracks_path) as fh:
        data = json.load(fh)
    return check(data, calibrators.load(calib_path))


def _fake(n_frames=1200, step=2, fps=25.0, ids_per_side=11):
    """A clean file: everyone walking at 1 m/s, identity homography so px == m."""
    dets = []
    for side, base in (("home", 0), ("away", 100)):
        for k in range(ids_per_side):
            for f in range(0, n_frames, step):
                dets.append({"f": f, "id": base + k, "team": side, "jersey": k + 1,
                             "x": 10.0 + k * 2 + (f / fps) * 1.0, "y": 10.0 + k})
    return {"fps": fps, "detections": dets}


def demo():
    """Self-check: every evaluable criterion is planted and must actually fail."""
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from pitch import Calibration, Pitch  # noqa: PLC0415
    cal = Calibration(pitch=Pitch(), H=np.eye(3))  # px == metres

    clean = check(_fake(), cal)
    ok = {r["id"]: r["status"] for r in clean["criteria"]}
    assert ok["C5_player_velocity"] == "ok", clean
    assert ok["C6_player_velocity_gap"] == "ok", clean
    assert ok["C7_home_present"] == "ok" and ok["C8_away_present"] == "ok", clean
    assert all(ok[c] == "not_evaluated" for c in
               ("C1_ball_missing", "C2_ball_gap", "C3_ball_velocity",
                "C4_ball_velocity_gap", "C9_ball_present")), clean
    # The ball rows make `ok` unreachable, and that is the point of the gate.
    assert clean["verdict"] == "incomplete", clean

    # C5: teleports at cadence (jumps, not gaps), on more than 0.5% of one
    # player's steps - which takes SIX of them in 600. That number is the
    # criterion's real sensitivity and the assert below pins it: one jump is not
    # enough, and `teleport_steps` in the result is why that matters here.
    d = _fake()
    for r in d["detections"]:
        if r["id"] == 0 and r["f"] % 100 == 0 and 100 <= r["f"] <= 600:
            r["x"] += 60.0
    got = check(d, cal)
    assert {r["id"]: r["status"] for r in got["criteria"]}["C5_player_velocity"] == "fail", got
    assert got["verdict"] == "fail", got

    one = _fake()
    for r in one["detections"]:
        if r["id"] == 0 and r["f"] >= 600:
            r["x"] += 60.0
    got1 = check(one, cal)
    assert {r["id"]: r["status"] for r in got1["criteria"]}["C5_player_velocity"] == "ok", got1
    assert got1["teleport_steps"] >= 1, got1  # seen, counted, and below the rule

    # C6: one player held above the threshold for well over a second.
    d = _fake()
    for r in d["detections"]:
        if r["id"] == 1 and 400 <= r["f"] <= 500:
            r["x"] += (r["f"] - 400) * 1.2      # 1.2 m per 2 frames = 15 m/s
    got = check(d, cal)
    by = {r["id"]: r for r in got["criteria"]}
    assert by["C6_player_velocity_gap"]["status"] == "fail", by["C6_player_velocity_gap"]
    assert by["C6_player_velocity_gap"]["value"] >= GAP_S, by["C6_player_velocity_gap"]

    # C7: home drops to six ids for 40 s at 25 fps.
    d = _fake()
    d["detections"] = [r for r in d["detections"]
                       if not (r["team"] == "home" and r["id"] >= 6 and r["f"] >= 200)]
    got = check(d, cal)
    by = {r["id"]: r for r in got["criteria"]}
    assert by["C7_home_present"]["status"] == "fail", by["C7_home_present"]
    assert by["C8_away_present"]["status"] == "ok", by["C8_away_present"]

    # The clamp trap, asserted rather than described: the product's own speed
    # estimate tops out below the threshold this gate tests against, so if C5
    # were ever switched to it the check would be decoration.
    import stats  # noqa: PLC0415
    assert stats.MAX_SPEED < MAX_PLAYER_MS, (stats.MAX_SPEED, MAX_PLAYER_MS)

    # A file with no detections must not pass; it must report, not crash.
    empty = check({"fps": 25.0, "detections": []}, cal)
    assert empty["verdict"] in ("fail", "incomplete"), empty

    print("quality.py ok")


def _print(res: dict) -> None:
    print(f"verdict {res['verdict'].upper()}  ids {res['n_ids']}  frames {res['n_frames']} "
          f"@ fps {res['fps']} step {res['frame_step']}  gap_step_share {res['gap_step_share']}"
          + (f"  tracks_skipped {res['tracks_skipped']}" if res["tracks_skipped"] else ""))
    for r in res["criteria"]:
        mark = {"ok": " ok ", "fail": "FAIL", "not_evaluated": " -- "}[r["status"]]
        print(f"  [{mark}] {r['id']:24s} {str(r['value']):46s} {r['threshold']}")
        print(f"           {r['note']}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif len(sys.argv) >= 3:
        _print(check_files(sys.argv[1], sys.argv[2]))
    else:
        raise SystemExit(__doc__.strip().splitlines()[-2].strip())
