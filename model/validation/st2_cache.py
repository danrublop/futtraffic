"""GT-track position cache for the ST2 event model - sprint P1 step 6.

One `.npz` per match under model/validation/out/st2_events/, holding per-frame
pitch-metre positions of every player the vendor's dense feed tracks, plus the BAS
labels aligned to those frames. Split, drop rule, receiver rule and G0's pass rule
are fixed in st2_index.py's PROTOCOL block, committed before this file existed.

WHERE THE POSITIONS COME FROM
    raw/<m>_tracker_box_data.xml via st2_team_gt.dense_loc - the vendor's own
    tracking, not a detector. Corner-origin metres, x along the 105 m length.
    `loc` is printed to two decimals of the pitch, so every position sits on a
    1.05 m x 0.68 m grid: a tie between two players is a real outcome, not noise.

THE CLOCK
    BAS `position` and the feed's `matchTime` are one cumulative match clock, and
    st2_index keeps period k's clock as t_s + k*45:00. So a label lands on the feed
    frame whose matchTime is nearest, per period. frameNumber restarts at 251 each
    half and is NOT the clock; 132877 ships no <period> lines in its metadata, so
    matchTime is read off every frame rather than from there.

THE THREE TRAPS, handled here and nowhere else
    team     BAS `team` is left/right and flips at half time. Side (home/away) is
             taken from the roster per player id; BAS team is kept only to count
             disagreements.
    mirror   132877's feed runs y the wrong way (st2-132877-feed-is-mirrored (internal note)).
             It is read with y -> 1 - y, gated on
             st2_team_gt.feed_mirrored, and the cache records `unmirrored`. Any
             other mirrored feed is refused. 132831 is excluded by EVENT_SPLIT.
             ONLY THE PLAYERS ARE MIRRORED: flipping the feed's ball too put it a
             median 6.35 m from the nearest player on 132877 (2.04 m unflipped;
             117093 reads 2.21 m), and put the labelled actor ~30 m from it.
             `build_match` now asserts ball-to-nearest-player < BALL_NEAR_M.
    ball     the feed carries a ball too. It is cached ONLY as the reference point
             of st2_eval's nearest-player floor - never a model channel (§P1).
             The feed parks an unknown ball at exactly (0.50, 0.50); that value
             is stored as NaN.

    .venv/bin/python model/validation/st2_cache.py --build     # all 9, ~2 min
    .venv/bin/python model/validation/st2_cache.py --drops     # per-match landed/dropped
    .venv/bin/python model/validation/st2_cache.py --g0        # contact sheets, val
    .venv/bin/python model/validation/st2_cache.py --offsets   # per-period label lag
    .venv/bin/python model/validation/st2_cache.py --demo
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st2_team_gt as G  # noqa: E402
from st2_index import EVENT_SPLIT, HALF_MS, OUT as INDEX  # noqa: E402

ROOT = G.ROOT
OUT = os.path.join(ROOT, "model/validation/out/st2_events")
PERIODS = ("FIRST_HALF", "SECOND_HALF")
FRAME_MS = 40
BALL_NEAR_M = 3.0     # median ball-to-nearest-player a coherent feed stays under
_CLOCK = re.compile(rb'<frame matchTime="([-0-9]+)" frameNumber="([0-9]+)" '
                    rb'eventPeriod="([A-Z_]+)"')
_BALL = re.compile(rb'<ball playerId="ball" loc="\[([-0-9.]+), ?([-0-9.]+)\]"')


def _clock_ball(m):
    """{(period, frameNumber): (matchTime, ball (x, y) normalised or None)}."""
    out, key = {}, None
    with open(f"{G.RAW}/{m}/{m}_tracker_box_data.xml", "rb") as fh:
        for line in fh:
            s = line.lstrip()
            if s[:6] == b"<frame":
                c = _CLOCK.match(s)
                if c:
                    key = (c.group(3).decode(), int(c.group(2)))
                    out[key] = (int(c.group(1)), None)
            elif s[:5] == b"<ball" and key:
                b = _BALL.match(s)
                if b:
                    out[key] = (out[key][0], (float(b.group(1)), float(b.group(2))))
    return out


def _pitch(m):
    t = open(f"{G.RAW}/{m}/{m}_tracker_box_metadata.xml").read()
    w, h = re.search(r'<pitch width="([0-9.]+)" height="([0-9.]+)"', t).groups()
    return float(w), float(h)


def build_match(m, index=INDEX):
    """-> dict of arrays for one match. Refuses a mirrored feed it cannot fix."""
    assert EVENT_SPLIT[m] != "excluded", f"{m} is excluded by EVENT_SPLIT"
    mirrored = G.feed_mirrored(G.load()[m])
    if mirrored and m != "132877":
        raise SystemExit(f"{m}: mirrored feed and no reviewed un-mirroring for it")
    L, CB = G.dense_loc(m), _clock_ball(m)
    R = G.roster(m)["players"]
    lx, ly = _pitch(m)

    def metres(xy, flip=mirrored):
        x, y = xy
        return (x * lx, (1.0 - y if flip else y) * ly)

    out = {"match": m, "split": EVENT_SPLIT[m], "unmirrored": mirrored,
           "pitch": np.array([lx, ly], np.float32)}
    for k, per in enumerate(PERIODS):
        fns = sorted(f for p, f in L if p == per)
        assert fns == list(range(fns[0], fns[-1] + 1)), f"{m} {per}: frameNumber has gaps"
        ids = sorted({i for f in fns for i in L[(per, f)]})
        slot = {i: j for j, i in enumerate(ids)}
        xy = np.full((len(fns), len(ids), 2), np.nan, np.float32)
        ball = np.full((len(fns), 2), np.nan, np.float32)
        t_ms = np.array([CB[(per, f)][0] for f in fns], np.int64)
        assert (np.diff(t_ms) == FRAME_MS).all(), f"{m} {per}: matchTime is not 40 ms steps"
        for r, f in enumerate(fns):
            for i, v in L[(per, f)].items():
                xy[r, slot[i]] = metres(v)
            b = CB[(per, f)][1]
            if b and b != (0.5, 0.5):     # the feed's parking spot for an unknown ball
                ball[r] = metres(b, flip=False)     # the feed's ball is NOT mirrored
        # The ball rides with SOME player nearly always. If it does not, the ball
        # and the players are in different frames - 132877's first build, with the
        # ball flipped along with the players, read 6.35 m here against 2.21 m on
        # 117093.
        near = np.nanmedian(np.nanmin(np.linalg.norm(xy - ball[:, None], axis=2), axis=1))
        assert near < BALL_NEAR_M, f"{m} {per}: ball-to-nearest-player median {near:.2f} m"
        out[f"p{k}_ball_near_m"] = np.float32(near)
        out[f"p{k}_xy"], out[f"p{k}_ball"], out[f"p{k}_t_ms"] = xy, ball, t_ms
        out[f"p{k}_frame_number"] = np.array(fns, np.int32)
        out[f"p{k}_pid"] = np.array(ids, np.int64)
        out[f"p{k}_side"] = np.array([{"home": 0, "away": 1}.get(R.get(i, {}).get("side"), -1)
                                      for i in ids], np.int8)
        out[f"p{k}_jersey"] = np.array([int(R[i]["back_number"]) if i in R and
                                        str(R[i]["back_number"]).isdigit() else -1
                                        for i in ids], np.int16)
    out.update(_events(m, out, index, R))
    _label_offsets(out)
    return out


def _events(m, c, index, R):
    """Every BAS row of the match, sorted by (period, clock), with its landing."""
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, index))) if r["match"] == m]
    rows.sort(key=lambda r: (int(r["period"]), float(r["t_s"])))
    ev = {k: [] for k in ("period", "t_s", "label", "pid", "bas_team", "row",
                          "actor", "side", "recv_pid", "recv_event", "completed", "drop")}
    for n, r in enumerate(rows):
        k, pid = int(r["period"]), int(r["player_id"]) if r["player_id"].isdigit() else -1
        drop, row, actor, side = "", -1, -1, -1
        if r["no_footage"] == "1":
            drop = "no_footage"
        elif r["clock_negative"] == "1":
            drop = "clock_negative"
        elif pid < 0:
            drop = "no_actor_id"
        else:
            t = c[f"p{k}_t_ms"]
            row = int(round((round(float(r["t_s"]) * 1000) + k * HALF_MS - t[0]) / FRAME_MS))
            if not 0 <= row < len(t):
                drop, row = "outside_feed", -1
            else:
                ids = list(c[f"p{k}_pid"])
                actor = ids.index(pid) if pid in ids else -1
                if actor < 0 or np.isnan(c[f"p{k}_xy"][row, actor, 0]):
                    drop, actor = "actor_not_on_pitch", -1
                else:
                    side = int(c[f"p{k}_side"][actor])
        # RECEIVER: the next label, iff it is a DRIVE by someone else, same period.
        rp, re_, comp = -1, -1, -1
        nxt = rows[n + 1] if n + 1 < len(rows) else None
        if (r["label"] == "PASS" and nxt and nxt["label"] == "DRIVE"
                and nxt["period"] == r["period"] and nxt["player_id"].isdigit()
                and nxt["player_id"] != r["player_id"] and pid >= 0):
            rp, re_ = int(nxt["player_id"]), n + 1
            a, b = R.get(pid, {}).get("side"), R.get(rp, {}).get("side")
            comp = int(a == b) if a and b else -1
        for key, v in (("period", k), ("t_s", float(r["t_s"])), ("label", r["label"]),
                       ("pid", pid), ("bas_team", r["team"]), ("row", row), ("actor", actor),
                       ("side", side), ("recv_pid", rp), ("recv_event", re_),
                       ("completed", comp), ("drop", drop)):
            ev[key].append(v)
    return {f"ev_{k}": np.array(v) for k, v in ev.items()}


LABEL_SHIFT = 25        # frames: the one-second lag (st2_index PROTOCOL, LABEL OFFSET)
LABEL_MARGIN = 0.05


def _near_share(c, k, rows, acts, thr=1.5):
    n = len(c[f"p{k}_t_ms"])
    g = (rows >= 0) & (rows < n)
    d = np.linalg.norm(c[f"p{k}_xy"][rows[g], acts[g]] - c[f"p{k}_ball"][rows[g]], axis=1)
    return float(np.nanmean(d < thr)) if g.any() else 0.0


def _label_offsets(c):
    """Shift each period's landed labels by LABEL_SHIFT iff the passers sit at the
    ball clearly more often one second later. In place; records the choice."""
    c["ev_drop"] = c["ev_drop"].astype("<U24")
    for k in (0, 1):
        per = (c["ev_period"] == k) & (c["ev_drop"] == "")
        sel = per & (c["ev_label"] == "PASS")
        s0 = _near_share(c, k, c["ev_row"][sel], c["ev_actor"][sel])
        s1 = _near_share(c, k, c["ev_row"][sel] + LABEL_SHIFT, c["ev_actor"][sel])
        off = LABEL_SHIFT if s1 - s0 >= LABEL_MARGIN else 0
        c[f"p{k}_label_offset"] = np.array([off, round(s0 * 1000), round(s1 * 1000)], np.int32)
        if off:
            r = c["ev_row"][per] + off
            gone = r >= len(c[f"p{k}_t_ms"])
            c["ev_row"][per] = np.where(gone, -1, r)
            idx = np.where(per)[0][gone]
            c["ev_drop"][idx] = "outside_feed"
            c["ev_actor"][idx] = -1


def stress():
    """Is the LABEL OFFSET decision robust, and does G0 rule (b) still do work?

    WHY THIS EXISTS. Rule (b) tests that the median actor-to-ball distance
    minimises near shift 0 AFTER the offset has been applied - and the offset was
    chosen by maximising how often the actor is within 1.5 m of that same ball,
    over the same events. Same functional, same ball, same rows: passing (b) after
    the shift is close to tautological and st2-p1-step6 (not published) now says so.

    What is NOT circular is whether the DECISION was robust, so that is measured
    here: how decisively each period chose, whether the 1.5 m threshold or the
    choice of PASS as the probe class could flip it, and where the post-correction
    sweep actually lands per period rather than pooled.

    Train and val only - the test split's periods are not counted or printed.

        .venv/bin/python model/validation/st2_cache.py --stress
    """
    ms = sorted(x for x, s in EVENT_SPLIT.items() if s in ("train", "val"))
    THRS, PROBES = (1.0, 1.5, 2.0, 3.0), ("PASS", "DRIVE")
    margins, same_thr, same_cls, argmins, n = [], 0, 0, [], 0
    for m in ms:
        c = load(m)
        for k in (0, 1):
            n += 1
            off = int(c[f"p{k}_label_offset"][0])
            s0, s1 = (int(x) / 1000 for x in c[f"p{k}_label_offset"][1:])
            margins.append(abs(s1 - s0))
            per = (c["ev_period"] == k) & (c["ev_drop"] == "")
            base = {}
            for probe in PROBES:
                sel = per & (c["ev_label"] == probe)
                # ev_row is ALREADY shifted, so undo it to re-decide from scratch
                r0, a = c["ev_row"][sel] - off, c["ev_actor"][sel]
                for thr in THRS:
                    d = LABEL_SHIFT if (_near_share(c, k, r0 + LABEL_SHIFT, a, thr)
                                        - _near_share(c, k, r0, a, thr)) >= LABEL_MARGIN else 0
                    base[(probe, thr)] = d
            same_thr += all(base[("PASS", t)] == base[("PASS", 1.5)] for t in THRS)
            same_cls += all(base[("DRIVE", t)] == base[("PASS", t)] for t in THRS)
            # post-correction sweep, PER PERIOD (shift_sweep pools all of them)
            rows, acts = c["ev_row"][per], c["ev_actor"][per]
            best, nn = (1e9, 0.0), len(c[f"p{k}_t_ms"])
            for sft in np.round(np.arange(-2.0, 2.01, 0.2), 1):
                r = rows + int(round(sft * 1000 / FRAME_MS))
                g = (r >= 0) & (r < nn)
                d = np.linalg.norm(c[f"p{k}_xy"][r[g], acts[g]] - c[f"p{k}_ball"][r[g]], axis=1)
                best = min(best, (float(np.nanmedian(d)), float(sft)))
            argmins.append(best[1])
    print(f"LABEL OFFSET stress - {n} periods, train+val only (test not counted)")
    print(f"  decision margin |s1-s0| vs the {LABEL_MARGIN} rule: min {min(margins):.3f}, "
          f"median {float(np.median(margins)):.3f}  -> the margin is INERT, "
          f"every period decides by far more")
    print(f"  same decision at thresholds {THRS} m: {same_thr}/{n} periods")
    print(f"  DRIVE-only agrees with PASS-only: {same_cls}/{n} periods")
    print(f"  post-correction per-period sweep argmin: max |s| "
          f"{max(abs(x) for x in argmins):.2f} s over {len(argmins)} periods")


# ---------------------------------------------------------------- roles
# The LISTED position from the vendor's metadata XML, folded to four lines. Used
# by st2_train's G3B role arm (its PROTOCOL addendum, 2026-09-20).
#
# THIS IS NOT A ROLE CLASSIFIER. `model/engine/rolenet.py` is a different thing -
# player / goalkeeper / referee, for detection - and it has never been run on ST2
# footage, so there is no per-frame role output for these matches and none is
# invented here. This is the team sheet: one static line per player.
#
# `ponytail: read from the XML rather than stored in the .npz, because rebuilding
# the cache is off the table on this branch (a push gate is reading out/st2_events).
# Upgrade path is one p{k}_role array in build_match the next time it is rebuilt.`
ROLE_LINES = ("GK", "DEF", "MID", "FWD")
_ROLE_OF = {"GK": 0,
            "CB": 1, "LB": 1, "RB": 1, "LWB": 1, "RWB": 1,
            "CDM": 2, "CM": 2, "CAM": 2, "LM": 2, "RM": 2,
            "LW": 3, "RW": 3, "CF": 3}


def roles(m, pids):
    """-> (N,) index into ROLE_LINES per slot, -1 where the vendor lists none.

    -1 becomes an all-zero one-hot, which is an honest "unknown" and not a
    fifth line. Measured 2026-09-20: 472 of 472 tracked slots over the nine
    cached matches carry a position, so -1 does not fire on this data - the
    branch exists because the next match may not."""
    R = G.roster(m)["players"]
    return np.array([_ROLE_OF.get(R.get(int(i), {}).get("position") or "", -1)
                     for i in pids], np.int8)


def path(m):
    return os.path.join(OUT, f"{m}.npz")


def load(m):
    p = path(m)
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing - run: python model/validation/st2_cache.py --build")
    return dict(np.load(p, allow_pickle=False))


def build():
    os.makedirs(OUT, exist_ok=True)
    for m in sorted(EVENT_SPLIT):
        if EVENT_SPLIT[m] == "excluded":
            print(f"{m}: EXCLUDED by EVENT_SPLIT, not cached")
            continue
        c = build_match(m)
        np.savez_compressed(path(m), **c)
        print(f"{m} {c['split']:5s} -> {path(m)}" + ("  (feed UN-MIRRORED)" if c["unmirrored"] else ""),
              flush=True)
    drops()


def drops():
    """Landed vs dropped labels, per match and reason. The drop share is the number.

    Also totals how many PERIODS the label-offset rule shifts. st2-p1-step6 (not published)
    cites the video read as the evidence that the LABELS are early rather than the
    feed, and that read covers three periods - so the count it covers out of the
    count it does not has to be printed, not inferred from nine printed pairs.
    """
    shifted = shifted_nontest = 0
    print(f"\n{'match':7s} {'split':5s} {'labels':>6s} {'landed':>6s} {'dropped':>7s} {'share':>6s}  reasons"
          "   (132831 excluded)")
    for m in sorted(EVENT_SPLIT):
        if EVENT_SPLIT[m] == "excluded":
            continue
        c = load(m)
        n_shift = sum(int(c[k][0]) != 0 for k in ("p0_label_offset", "p1_label_offset"))
        shifted += n_shift
        shifted_nontest += n_shift if EVENT_SPLIT[m] != "test" else 0
        if EVENT_SPLIT[m] == "test":
            # THE TEST SPLIT SHOWS ITS LABEL OFFSET AND NOTHING ELSE. The offset
            # is protocol - LABEL OFFSET is applied to every split as label
            # cleaning and has to be printed to be checkable - but landed,
            # receiver and completed counts are properties of 128057's events,
            # and publishing those is a peek at the test set. `offsets()` already
            # refuses the test split; this is the same guard on the same data.
            print(f"{m:7s} {'test':5s} {'':>6s} {'':>6s} {'':>7s} {'':>6s}  "
                  f"counts WITHHELD (test split)"
                  f"                    label offset P1/P2 "
                  f"{int(c['p0_label_offset'][0])}/{int(c['p1_label_offset'][0])} frames")
            continue
        d = c["ev_drop"]
        n, bad = len(d), int((d != "").sum())
        why = {str(r): int((d == r).sum()) for r in sorted(set(d)) if r}
        side_off = int(((c["ev_side"] >= 0) & (c["ev_side"] != _bas_side(c))).sum())
        passes = (c["ev_label"] == "PASS") & (d == "")
        rec = passes & (c["ev_recv_pid"] >= 0)
        comp = float((c["ev_completed"][rec] == 1).mean())
        print(f"{m:7s} {EVENT_SPLIT[m]:5s} {n:6d} {n - bad:6d} {bad:7d} {bad / n:6.3f}  {why}"
              f"  bas-team-disagrees {side_off}  receiver {int(rec.sum())}/{int(passes.sum())}"
              f" completed {comp:.3f}  label offset P1/P2 "
              f"{int(c['p0_label_offset'][0])}/{int(c['p1_label_offset'][0])} frames")
    # VIDEO_CHECKED: the periods st2-p1-step6 (not published)'s sheet read actually covers.
    # A constant, because it records a human pass, not something this file computes.
    print(f"\nperiods shifted {shifted}, of which non-test {shifted_nontest}; "
          f"video-checked {len(VIDEO_CHECKED)} ({', '.join(VIDEO_CHECKED)}); the others "
          f"rest on the offset functional alone")


#: The three shifted periods the 2026-09-19 sheet read covers (st2-p1-step6 (not published)).
#: Not computed here: it is the record of a human pass, and the sheets are not committed.
VIDEO_CHECKED = ("117093 P1", "118576 P2", "128058 P1")


def _bas_side(c):
    """BAS left/right -> home/away per period, by majority of that period's landed rows.
    Only used to COUNT disagreements; side itself comes from the roster."""
    out = np.full(len(c["ev_side"]), -1)
    for k in (0, 1):
        sel = (c["ev_period"] == k) & (c["ev_side"] >= 0)
        left_home = np.mean(c["ev_side"][sel & (c["ev_bas_team"] == "left")] == 0) > 0.5
        lr = c["ev_bas_team"] == "left"
        out[(c["ev_period"] == k) & lr] = 0 if left_home else 1
        out[(c["ev_period"] == k) & ~lr] = 1 if left_home else 0
    return out


# ---------------------------------------------------------------- G0
def shift_sweep(c, shifts=np.round(np.arange(-2.0, 2.01, 0.2), 1)):
    """Median actor-to-ball distance (m) with every landed label moved by s seconds."""
    ok = (c["ev_drop"] == "")
    out = {}
    for s in shifts:
        d = []
        for k, row, a in zip(c["ev_period"][ok], c["ev_row"][ok], c["ev_actor"][ok]):
            r = row + int(round(s * 1000 / FRAME_MS))
            if 0 <= r < len(c[f"p{k}_t_ms"]):
                d.append(np.linalg.norm(c[f"p{k}_xy"][r, a] - c[f"p{k}_ball"][r]))
        out[float(s)] = float(np.nanmedian(d))
    return out


def offsets():
    """EXPLORATORY, not in the protocol: per period, the label shift (-2..+3 s, one
    frame steps, n=126) at which the most PASS actors stand within 1.5 m of the
    ball. The ball leaves the passer at release, so this peaks at the release
    frame. Train + val only - test is not opened."""
    for m in sorted(x for x, s in EVENT_SPLIT.items() if s in ("train", "val")):
        c, line = load(m), []
        for k in (0, 1):
            sel = (c["ev_drop"] == "") & (c["ev_label"] == "PASS") & (c["ev_period"] == k)
            rows, acts, n = c["ev_row"][sel], c["ev_actor"][sel], len(c[f"p{k}_t_ms"])
            best = (-1.0, 0)
            for f in range(-50, 76):
                r = rows + f
                g = (r >= 0) & (r < n)
                d = np.linalg.norm(c[f"p{k}_xy"][r[g], acts[g]] - c[f"p{k}_ball"][r[g]], axis=1)
                best = max(best, (float(np.nanmean(d < 1.5)), f))
            line.append(f"P{k + 1} {best[1] * FRAME_MS / 1000:+.2f}s ({best[0]:.2f})")
        print(f"{m} {EVENT_SPLIT[m]:5s} " + "  ".join(line))


def g0(m, n=24, seed=0):
    """24 labelled events drawn round-robin across classes and both halves, plotted
    on the pitch at their label frame: home blue, away red, ball black, actor ringed,
    receiver ringed green. A human reads it; the shift sweep is printed beside it."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    c = load(m)
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(c["ev_drop"] == "")
    by = {}
    for i in rng.permutation(ok):
        by.setdefault((str(c["ev_label"][i]), int(c["ev_period"][i])), []).append(int(i))
    pick, keys = [], sorted(by)
    while len(pick) < n and any(by.values()):
        for kk in keys:
            if by[kk] and len(pick) < n:
                pick.append(by[kk].pop())
    pick.sort(key=lambda i: (c["ev_period"][i], c["ev_t_s"][i]))
    lx, ly = c["pitch"]
    fig, axs = plt.subplots(6, 4, figsize=(20, 22))
    for ax, i in zip(axs.flat, pick):
        k, row, a = int(c["ev_period"][i]), int(c["ev_row"][i]), int(c["ev_actor"][i])
        xy, side, b = c[f"p{k}_xy"][row], c[f"p{k}_side"], c[f"p{k}_ball"][row]
        ax.add_patch(plt.Rectangle((0, 0), lx, ly, fill=False, color="0.6"))
        ax.axvline(lx / 2, color="0.8", lw=0.8)
        for s, col in ((0, "tab:blue"), (1, "tab:red")):
            ax.scatter(*xy[side == s].T, s=18, c=col)
        ax.scatter(*b, marker="x", s=60, c="k", zorder=5)
        ax.scatter(*xy[a], s=160, facecolors="none", edgecolors="k", lw=2)
        rp = int(c["ev_recv_pid"][i])
        if rp >= 0 and rp in c[f"p{k}_pid"]:
            j = list(c[f"p{k}_pid"]).index(rp)
            ax.scatter(*xy[j], s=160, facecolors="none", edgecolors="tab:green", lw=2)
            ax.annotate("", xy[j], xy[a], arrowprops=dict(arrowstyle="->", color="tab:green"))
        dist = np.linalg.norm(xy[a] - b)
        ax.set_title(f"{c['ev_label'][i]}  P{k + 1} {c['ev_t_s'][i]:.1f}s  "
                     f"#{c[f'p{k}_jersey'][a]} {'HA'[side[a]]}  ball {dist:.1f} m", fontsize=10)
        ax.set_xlim(-3, lx + 3)
        ax.set_ylim(ly + 3, -3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    sw = shift_sweep(c)
    best = min(sw, key=sw.get)
    fig.suptitle(f"G0 {m} ({EVENT_SPLIT[m]}{', feed UN-MIRRORED' if c['unmirrored'] else ''})  "
                 f"actor ringed black, receiver green, ball x.  median actor-ball "
                 f"{sw[0.0]:.2f} m at shift 0, min {sw[best]:.2f} m at {best:+.1f} s", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(OUT, f"G0_alignment_{m}.png")
    fig.savefig(p, dpi=60)
    print(f"{m}: -> {p}")
    print("  shift(s) " + " ".join(f"{s:+.1f}:{v:.2f}" for s, v in sw.items()))
    print(f"  argmin {best:+.1f} s -> rule (b) {'PASS' if abs(best) <= 0.4 else 'FAIL'}")


def demo():
    """The landing arithmetic and the receiver rule, on a synthetic match."""
    t = np.arange(0, 10_000, FRAME_MS, dtype=np.int64) + HALF_MS - 33   # half 2's -33 ms start
    c = {"p1_t_ms": t, "p1_pid": np.array([7, 9]), "p1_side": np.array([0, 1], np.int8),
         "p1_xy": np.zeros((len(t), 2, 2), np.float32)}
    c["p1_xy"][:, 1] = np.nan                       # player 9 never on the pitch
    rows = [("1", "4.0", "PASS", "7"), ("1", "5.4", "DRIVE", "9"), ("1", "6.0", "PASS", "9"),
            ("1", "11.0", "PASS", "7")]
    tmp = os.path.join(OUT, "_demo_index.csv")
    os.makedirs(OUT, exist_ok=True)
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match", "period", "t_s", "label", "team", "player_id", "no_footage",
                    "clock_negative"])
        for p, ts, lab, pid in rows:
            w.writerow(["x", p, ts, lab, "left", pid, "0", "0"])
    try:
        e = _events("x", c, tmp, {7: {"side": "home"}, 9: {"side": "away"}})
    finally:
        os.remove(tmp)
    assert e["ev_row"][0] == round((4000 + 33) / FRAME_MS), e["ev_row"]      # 101, not 100
    assert list(e["ev_drop"]) == ["", "actor_not_on_pitch", "actor_not_on_pitch", "outside_feed"]
    assert e["ev_recv_pid"][0] == 9 and e["ev_completed"][0] == 0      # home -> away = lost
    assert e["ev_recv_pid"][2] == -1                 # PASS followed by a PASS has no receiver
    # LABEL OFFSET: passers reach the ball exactly one second after their labels
    n = 200
    o = {"ev_period": np.zeros(4, int), "ev_label": np.array(["PASS"] * 4),
         "ev_drop": np.array(["", "", "", ""]), "ev_row": np.array([10, 40, 80, 190]),
         "ev_actor": np.zeros(4, int), "p0_t_ms": np.arange(n) * FRAME_MS,
         "p0_xy": np.full((n, 1, 2), 30.0, np.float32), "p0_ball": np.zeros((n, 2), np.float32),
         "p1_t_ms": np.arange(n) * FRAME_MS, "p1_xy": np.zeros((n, 1, 2), np.float32),
         "p1_ball": np.zeros((n, 2), np.float32)}
    for r in (10, 40, 80):
        o["p0_xy"][r + LABEL_SHIFT, 0] = 0.0
    _label_offsets(o)
    assert o["p0_label_offset"][0] == LABEL_SHIFT and o["p1_label_offset"][0] == 0
    assert list(o["ev_row"]) == [35, 65, 105, -1] and o["ev_drop"][3] == "outside_feed", o["ev_row"]
    # ROLE MAP: every listed position folds to a line in range, and a position
    # the vendor does not list is -1 (an all-zero one-hot), never quietly a 0 -
    # which would read as "goalkeeper" for every unknown player.
    assert set(_ROLE_OF.values()) == set(range(len(ROLE_LINES))), _ROLE_OF
    assert _ROLE_OF.get("") is None and _ROLE_OF.get("None") is None
    print("st2_cache demo ok: clock landing, drop reasons, receiver + outcome, label offset, "
          "role lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--drops", action="store_true")
    ap.add_argument("--g0", action="store_true")
    ap.add_argument("--offsets", action="store_true")
    ap.add_argument("--stress", action="store_true", help="is the offset decision robust?")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.drops:
        drops()
    elif a.offsets:
        offsets()
    elif a.stress:
        stress()
    elif a.g0:
        for m in sorted(x for x, s in EVENT_SPLIT.items() if s == "val"):
            g0(m)
    else:
        demo()
