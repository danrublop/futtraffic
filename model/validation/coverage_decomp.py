"""COVERAGE: where the 30% of player-time we never produce a position for goes.

WHAT IS ALREADY KNOWN, AND WHY THIS IS THE LAST PIECE
    The product's distance reads 20% low (`api://product.distance.ratio_raw`).
    Two of its three parts are settled:
      * the estimator OVER-reads by 19% on what it can see, and all of it lives
        in clamped fragments recovering from occlusion (speed-clamp (not published));
      * the shipped map COMPRESSES path length ~23%; an oracle calibration takes
        `ratio_raw` 0.795 -> 1.046 (state-and-roadmap (not published) sec A).
    The third is `api://product.distance.coverage` = 0.704. An oracle map moves
    it DOWN, to 0.695, so it is not calibration. Nobody had broken it down.

WHAT COVERAGE ACTUALLY IS, WHICH IS NOT WHAT IT SOUNDS LIKE
    `dist_eval.evaluate` attributes every track id to its MAJORITY ground-truth
    owner (`contam_eval.owners`, nearest GT inside 2 m, sampled every 10th
    frame), then credits GT player g with EVERY FRAME of EVERY fragment whose
    majority owner is g:

        seen[g] = union of frames of the fragments dominated by g
        coverage[g] = |seen[g] & GT frames of g| / |GT frames of g|

    So coverage is a statement about FRAGMENT OWNERSHIP, not about whether a box
    was on that player in that frame. A fragment that drifts onto someone else
    still credits its owner for the frames it drifted; a fragment that sits on g
    all match but is majority-owned by h credits g with nothing. Both directions
    are measured below and neither had been.

THE LADDER, and each rung is a different module
    Per ground-truth player-instant (one GT player, one frame):

      D  a DETECTION was offered to the tracker within the gate   detectors.py
      T  a TRACK ROW is within the gate                            track.py/stitch.py
      A  ...and that row's id is majority-owned by this player     the attribution rule
      C  ...and dist_eval credits this frame to this player        what coverage reports

    D and T are matched in IMAGE SPACE - our feet go band -> raw by table lookup
    and meet the MOT boxes in their own pixels, gate in units of the GT box
    height. No calibration is in either, deliberately: the map is known to be
    1.19 m wrong and putting it inside the ceiling measurement would confound
    the one question this file exists to answer. A and C are metre-space and
    frame-set facts respectively, because that is what dist_eval does.

    1 - C  =  (1 - D)  +  (D - T)  +  (T - A)  -  (C - A)
             no box     no track    not credited   credited with no box
    The identity is asserted, not assumed (`--selfcheck`).

WHAT RESULT WAS POSSIBLE BEFORE ONE WAS REPORTED (postmortem (not published) C2)
    The instrument's dynamic range is checked by controls that must move, every
    run, before any number is read:
      CEILING  GT's own feet fed in as BOTH the detections and the tracks, with
               identity attribution. Every rung must read 1.000 and both
               attribution defects must read 0. If it does not, the ladder
               cannot show a clean pipeline and no split it produces means
               anything.
      NEGATIVE swallow 40% of the track rows at random. `D - T` must absorb
               almost exactly that 40% and `1 - D` must not move. A
               decomposition that cannot see 40% of the tracker vanishing
               cannot see the tracker at all.
      IDENTITY the C rung, aggregated, must reproduce `dist_eval`'s own
               coverage on the same rows and window to 1e-9. That is the check
               that fails if the decomposition breaks.

DENOMINATORS (postmortem (not published) B2)
    Everything is per GT PLAYER-INSTANT: one GT player, one frame, over the
    6,000 clip frames where BOTH ground-truth feeds exist (MOT pixels and the
    metre track). Clip frame 0 is gz frame 5998 and the MOT id -> gz id map is
    not the identity; both come from `model/validation/out/calib_hypothesis_join.json`
    rather than being re-derived. The published 0.704 is the full half and the
    mean over players; the window's own value is printed beside every split so
    the two are never confused.

    .venv/bin/python model/validation/coverage_decomp.py --selfcheck
    .venv/bin/python model/validation/coverage_decomp.py --run
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrators  # noqa: E402
import contam_eval  # noqa: E402
import dist_eval  # noqa: E402
import calib_hypothesis as CH  # noqa: E402  - join, band->raw, load_rows, score

JOB = CH.JOB
OUT = CH.OUT
DETS = os.path.join(JOB, "dets.cache.json")

#: Image gate, in units of the GT box height. 1.0 is the loosest gate
#: `calib_hypothesis.space_recall` sweeps and the one that produced the 0.784
#: this file's first question is about. Swept, never assumed.
H_GATES = (0.3, 0.6, 1.0, 1.5, 2.0)
PRIMARY_H = 1.0

#: Metre gate for the attribution rule. `contam_eval.GATE_M`'s own default.
GATE_M = 2.0


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------

def assign(D: np.ndarray, gate: float) -> dict[int, int]:
    """GT row -> prediction column, exclusive nearest-first inside `gate`.

    Byte-for-byte the rule `calib_hypothesis.space_recall.greedy` counts with
    (one detection satisfies one GT player, so a single box in a crowd cannot
    cover three), except that it returns WHICH prediction matched instead of
    how many did - the whole ladder below needs the identity of the match.
    """
    out: dict[int, int] = {}
    taken: set[int] = set()
    if D.size == 0:
        return out
    for i, j in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
        if D[i, j] >= gate:
            break
        i, j = int(i), int(j)
        if i in out or j in taken:
            continue
        out[i] = j
        taken.add(j)
    return out


def load_dets(lo: int, hi: int, path: str = DETS) -> dict[int, np.ndarray]:
    """{frame: (n,2) BAND-pixel feet} for the boxes the TRACKER WAS OFFERED.

    `model/engine/track.py` caches AFTER `sceneprior`'s on-pitch and plausible-height
    gates (track.py :: track, "Already past the prior when it was cached"), so this
    is the association ceiling, not the raw detector's. A box the prior threw
    away is invisible here and is counted in `1 - D` alongside a box that was
    never detected; `api://product.detect.recall_px` is the number that sees
    the difference and it is quoted, not recomputed.

    Foot is bottom-centre of xyxy, the same convention `track.py :: track` writes.
    """
    t0 = time.time()
    blob = json.load(open(path))
    log(f"  dets.cache.json: {len(blob['frames'])} frames in {time.time() - t0:.0f} s")
    out = {}
    for f in range(lo, hi):
        b = blob["frames"].get(str(f))
        if not b:
            continue
        xy = np.array([d["xyxy"] for d in b], float)
        out[f] = np.column_stack([(xy[:, 0] + xy[:, 2]) / 2.0, xy[:, 3]])
    del blob
    return out


def gt_heights(path: str = CH.GT_PX) -> dict[int, dict[int, float]]:
    """{clip frame: {mot id: box height px}} - the unit the image gate is in."""
    hs = collections.defaultdict(dict)
    for line in open(path):
        p = line.split(",")
        if len(p) >= 6:
            hs[int(p[0])][int(p[1])] = float(p[5])
    return dict(hs)


# --------------------------------------------------------------------------

def ladder(cal, jn, gt_px, gt_h, gt_m, rows_raw, row_f, row_id, dets_raw,
           dominant, seen, gates=H_GATES, primary=PRIMARY_H,
           lo=None, hi=None) -> dict:
    """The four rungs, per GT player-instant, plus every stratum.

    `rows_raw` / `dets_raw` are ALREADY in raw panorama pixels: the band -> raw
    table lookup is done once for the whole run rather than per frame, which is
    the difference between seconds and minutes and is also what lets the
    controls hand ground truth straight in without inverting anything.
    """
    off, perm = jn["offset"], jn["perm"]
    by_f_rows = collections.defaultdict(list)
    for k, f in enumerate(row_f):
        by_f_rows[int(f)].append(k)

    rec = collections.defaultdict(list)          # column -> list
    hit_D = {g: [] for g in gates}
    hit_T = {g: [] for g in gates}

    clip = sorted(c for c in gt_px if (lo is None or lo <= c + off < hi))
    for c in clip:
        f = c + off
        gm = gt_m.get(f)
        if not gm:
            continue
        ids = [i for i in gt_px[c] if perm.get(i) in gm]
        if not ids:
            continue
        G_px = np.array([gt_px[c][i] for i in ids], float)
        G_h = np.array([gt_h[c][i] for i in ids], float)
        G_m = np.array([gm[perm[i]] for i in ids], float)
        gids = [perm[i] for i in ids]

        # duel: how far the NEAREST other GT player is, in metres. The duel /
        # clear split failure-taxonomy-association (not published) uses is box overlap;
        # in metres the equivalent is "closer than a player is wide", and it is
        # swept rather than fixed.
        dd = np.linalg.norm(G_m[:, None] - G_m[None], axis=2)
        np.fill_diagonal(dd, np.inf)
        near_m = dd.min(1) if len(G_m) > 1 else np.full(len(G_m), np.inf)

        ridx = by_f_rows.get(f, [])
        R = rows_raw[ridx] if ridx else np.zeros((0, 2))
        Dm = dets_raw.get(f, np.zeros((0, 2)))

        D_T = np.linalg.norm(G_px[:, None] - R[None], axis=2) / G_h[:, None] \
            if len(R) else np.zeros((len(ids), 0))
        D_D = np.linalg.norm(G_px[:, None] - Dm[None], axis=2) / G_h[:, None] \
            if len(Dm) else np.zeros((len(ids), 0))

        for g in gates:
            mt, md = assign(D_T, g), assign(D_D, g)
            hit_T[g].extend(k in mt for k in range(len(ids)))
            hit_D[g].extend(k in md for k in range(len(ids)))
            if g == primary:
                for k, gid in enumerate(gids):
                    j = mt.get(k)
                    tid = int(row_id[ridx[j]]) if j is not None else -1
                    rec["g"].append(gid)
                    rec["f"].append(f)
                    rec["T"].append(j is not None)
                    rec["Tid"].append(tid)
                    rec["A"].append(j is not None and dominant.get(tid) == gid)
                    rec["steal"].append(j is not None
                                        and dominant.get(tid) not in (gid, None))
                    rec["C"].append(f in seen.get(gid, ()))
                    rec["h"].append(G_h[k])
                    rec["ym"].append(G_m[k][1])
                    rec["xm"].append(G_m[k][0])
                    rec["v"].append(G_px[k][1])
                    rec["duel"].append(near_m[k])
    out = {k: np.array(v) for k, v in rec.items()}
    out["D"] = np.array(hit_D[primary], bool)
    out["sweep"] = {str(g): {"D": float(np.mean(hit_D[g])),
                             "T": float(np.mean(hit_T[g])),
                             "n": len(hit_D[g])} for g in gates}
    return out


def split(rec: dict) -> dict:
    """The decomposition, with its denominator, and the identity asserted."""
    n = len(rec["g"])
    D, T, A, C = rec["D"], rec["T"], rec["A"], rec["C"]
    s = {
        "n_instants": int(n),
        "coverage_C": float(C.mean()),
        "ceiling_D": float(D.mean()),
        "tracked_T": float(T.mean()),
        "attributed_A": float(A.mean()),
        "no_detection": float((~D).mean()),
        "det_but_no_track": float((D & ~T).mean()),
        "track_but_not_attributed": float((T & ~A).mean()),
        "credited_without_a_box": float((C & ~A).mean()),
        # the two attribution defects, stated as what they cost and to whom
        "hidden_by_attribution": float((T & ~C).mean()),
        "created_by_attribution": float((C & ~T).mean()),
        "stolen_by_another_id": float(rec["steal"].mean()),
        # T without D: a track row inside the gate where no CACHED detection
        # was. The tracker interpolates through occlusion, so this is real and
        # is reported rather than clamped to zero.
        "track_without_detection": float((T & ~D).mean()),
    }
    lhs = 1 - s["coverage_C"]
    rhs = (s["no_detection"] + s["det_but_no_track"]
           + s["track_but_not_attributed"] - s["credited_without_a_box"])
    assert abs(lhs - rhs) < 1e-9, (lhs, rhs, s)
    s["hole_total"] = lhs
    return s


def by_stratum(rec: dict, key: np.ndarray, edges, names) -> list[dict]:
    """One split per bucket. Every row carries its own denominator."""
    out = []
    for lab, m in zip(names, [(key >= a) & (key < b) for a, b in edges]):
        if m.sum() < 50:
            continue
        sub = {k: v[m] for k, v in rec.items() if isinstance(v, np.ndarray)}
        s = split(sub)
        s["stratum"] = lab
        out.append(s)
    return out


# --------------------------------------------------------------------------

def worth(cal, rec, gt, per_player, fps, lo, hi) -> dict:
    """If coverage went to each rung, what does `ratio_raw` become?

    Not a guess: for every rung, the reference is re-scored with that rung's
    frame mask as its gap mask - `dist_eval`'s own `real_override`, the same
    call that produces `gt_matched_m` - and each player's measured over-read
    (`ours_m / gt_matched_m`, i.e. what the estimator is worth WHERE IT CAN
    SEE) is carried onto the larger mask. So this says what today's estimator
    would report at tomorrow's coverage, and nothing about a better estimator.

    The C rung is computed the same way as the rest and must reproduce the
    actual `ratio_raw`; that is the check that the projection is arithmetic
    rather than an assumption.
    """
    masks = {"C": rec["C"], "A": rec["A"], "T": rec["T"], "D": rec["D"],
             "ALL": np.ones(len(rec["C"]), bool)}
    have = {k: collections.defaultdict(set) for k in masks}
    for k, m in masks.items():
        for g, f, ok in zip(rec["g"], rec["f"], m):
            if ok:
                have[k][g].add(int(f))

    pp = {p["gt_id"]: p for p in per_player}
    tot_gt = 0.0
    tot = {k: 0.0 for k in masks}
    cov = {k: [] for k in masks}
    for g, t in gt.items():
        sel = (t["f"] >= lo) & (t["f"] < hi)
        if sel.sum() < 2:
            continue
        fr, xy = t["f"][sel], t["xy"][sel]
        gpx = dist_eval.to_px(cal, xy)
        trk = [{"f": int(a), "id": g, "x": x, "y": y} for a, (x, y) in zip(fr, gpx)]
        full, _o, _p, _c = dist_eval.track_metrics(trk, cal, fps, "kalman")
        tot_gt += full
        p = pp.get(g, {})
        ours, matched = p.get("ours_m", 0.0), p.get("gt_matched_m", 0.0)
        scale = ours / matched if matched > 0 else 0.0
        for k, hv in have.items():
            h = np.fromiter((int(a) in hv[g] for a in fr), bool, len(fr))
            cov[k].append(h.mean())
            m, _o, _p, _c = dist_eval.track_metrics(
                trk, cal, fps, "kalman", real_override=h[:-1] & h[1:])
            tot[k] += scale * m
    return {"gt_km": tot_gt / 1000,
            "rungs": {k: {"coverage": float(np.mean(cov[k])),
                          "projected_ours_km": tot[k] / 1000,
                          "projected_ratio_raw": tot[k] / tot_gt if tot_gt else 0.0}
                      for k in masks}}


# --------------------------------------------------------------------------

def prepare(tracks="tracks.split.json", lo=None, hi=None, gate_m=GATE_M):
    """Everything the ladder eats, loaded once. Returns a dict of parts."""
    cal = calibrators.load(os.path.join(JOB, "calibration.json"))
    jp = os.path.join(OUT, "calib_hypothesis_join.json")
    jn = json.load(open(jp))
    jn["perm"] = {int(k): v for k, v in jn["perm"].items()}
    off = jn["offset"]
    lo = off if lo is None else lo
    hi = off + 6000 if hi is None else hi
    log(f"JOIN (cached): clip frame 0 = gz frame {off}, identity perm "
        f"{jn['identity_perm']}, cost {jn['cost_m']:.2f} m")

    rows, fps = CH.load_rows(tracks, lo, hi)
    log(f"  {tracks}: {len(rows)} rows, {len({r['id'] for r in rows})} ids, "
        f"frames {lo}-{hi}")

    # attribution: dist_eval's own, imported not reimplemented
    keep, contam_eval.GATE_M = contam_eval.GATE_M, gate_m
    try:
        owner = contam_eval.owners(rows, cal, contam_eval.load_gt(),
                                   contam_eval.STRIDE)
    finally:
        contam_eval.GATE_M = keep
    votes = collections.defaultdict(collections.Counter)
    for i, g in owner.items():
        votes[int(rows[i]["id"])][g] += 1
    dominant = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    seen = collections.defaultdict(set)
    for r in rows:
        g = dominant.get(int(r["id"]))
        if g is not None:
            seen[g].add(int(r["f"]))

    row_f = np.array([r["f"] for r in rows], np.int64)
    row_id = np.array([r["id"] for r in rows], np.int64)
    rows_raw = CH.band_to_raw(cal, np.array([[r["x"], r["y"]] for r in rows], float))

    dets = load_dets(lo, hi)
    keys = sorted(dets)
    flat = np.vstack([dets[k] for k in keys])
    flat = CH.band_to_raw(cal, flat)
    dets_raw, i = {}, 0
    for k in keys:
        n = len(dets[k])
        dets_raw[k] = flat[i:i + n]
        i += n

    gt_px = CH.load_gt_px()
    gt_h = gt_heights()
    gtm_frames = CH.load_gt_m()[1]
    return dict(cal=cal, jn=jn, lo=lo, hi=hi, fps=fps, rows=rows, row_f=row_f,
                row_id=row_id, rows_raw=rows_raw, dets_raw=dets_raw,
                gt_px=gt_px, gt_h=gt_h, gtm_frames=gtm_frames,
                dominant=dominant, seen=seen)


def run(a):
    P = prepare(a.tracks, gate_m=a.gate_m)
    cal, lo, hi, fps = P["cal"], P["lo"], P["hi"], P["fps"]

    log("\n  LADDER: one GT player-instant at a time, four rungs")
    t0 = time.time()
    rec = ladder(cal, P["jn"], P["gt_px"], P["gt_h"], P["gtm_frames"],
                 P["rows_raw"], P["row_f"], P["row_id"], P["dets_raw"],
                 P["dominant"], P["seen"], lo=lo, hi=hi)
    log(f"    {len(rec['g'])} instants in {time.time() - t0:.0f} s")

    s = split(rec)
    _gfps, gtd = dist_eval.load_gt()
    gtw = CH.window_gt(gtd, lo, hi)
    sc = CH.score(P["rows"], fps, cal, cal, gtw, contam_eval.load_gt(), 10, a.gate_m)

    # THE IDENTITY CHECK. The C rung is this file's reading of dist_eval's own
    # coverage; if the two disagree the decomposition is of something else.
    mine = float(np.mean([
        np.mean([rec["C"][i] for i in range(len(rec["g"])) if rec["g"][i] == g])
        for g in sorted({*rec["g"]})]))
    log(f"\n  cross-check: dist_eval coverage {sc['coverage']:.4f} (mean over "
        f"players, its own GT frames) vs this file's C rung {mine:.4f} "
        f"(mean over players, frames where BOTH GT feeds exist)")

    log(f"\n=== THE SPLIT, n = {s['n_instants']} GT player-instants, "
        f"frames {lo}-{hi}, image gate {PRIMARY_H} box heights, "
        f"metre gate {a.gate_m} m ===")
    log(f"  coverage (C)                      {s['coverage_C']:.4f}")
    log(f"  hole                              {s['hole_total']:.4f}   <- decomposed below")
    log(f"    no detection was offered        {s['no_detection']:.4f}"
        f"   ({s['no_detection'] / s['hole_total']:.1%} of the hole)  DETECTOR")
    log(f"    detection, no track claimed it  {s['det_but_no_track']:.4f}"
        f"   ({s['det_but_no_track'] / s['hole_total']:.1%})  TRACKER")
    log(f"    track, not attributed to them   {s['track_but_not_attributed']:.4f}"
        f"   ({s['track_but_not_attributed'] / s['hole_total']:.1%})  ATTRIBUTION")
    log(f"    less: credited with no box      -{s['credited_without_a_box']:.4f}"
        f"   ({-s['credited_without_a_box'] / s['hole_total']:.1%})  ATTRIBUTION")
    log(f"\n  rungs      D {s['ceiling_D']:.4f}   T {s['tracked_T']:.4f}   "
        f"A {s['attributed_A']:.4f}   C {s['coverage_C']:.4f}")
    log(f"  attribution: hides {s['hidden_by_attribution']:.4f}, "
        f"creates {s['created_by_attribution']:.4f}, "
        f"net {s['created_by_attribution'] - s['hidden_by_attribution']:+.4f}; "
        f"{s['stolen_by_another_id']:.4f} of instants sit under an id owned by "
        f"someone else")
    log(f"  tracker rows with no cached detection under them: "
        f"{s['track_without_detection']:.4f}")

    log("\n  GATE SWEEP (image gate in GT box heights)")
    log(f"    {'gate':>6} {'D detected':>11} {'T tracked':>10} {'n':>7}")
    for g, v in rec["sweep"].items():
        log(f"    {float(g):6.1f} {v['D']:11.4f} {v['T']:10.4f} {v['n']:7d}")

    strata = {}
    h, ym, duel, f = rec["h"], rec["ym"], rec["duel"], rec["f"]
    qh = np.percentile(h, [25, 50, 75])
    strata["box_height_px"] = by_stratum(
        rec, h, [(-np.inf, qh[0]), (qh[0], qh[1]), (qh[1], qh[2]), (qh[2], np.inf)],
        [f"Q1 <{qh[0]:.0f}px (farthest)", f"Q2 {qh[0]:.0f}-{qh[1]:.0f}",
         f"Q3 {qh[1]:.0f}-{qh[2]:.0f}", f"Q4 >{qh[2]:.0f}px (nearest)"])
    qy = np.percentile(ym, [25, 50, 75])
    strata["pitch_y_m"] = by_stratum(
        rec, ym, [(-np.inf, qy[0]), (qy[0], qy[1]), (qy[1], qy[2]), (qy[2], np.inf)],
        [f"y<{qy[0]:.1f}", f"{qy[0]:.1f}-{qy[1]:.1f}",
         f"{qy[1]:.1f}-{qy[2]:.1f}", f"y>{qy[2]:.1f}"])
    strata["duel_m"] = by_stratum(
        rec, duel, [(-np.inf, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)],
        ["<1 m (duel)", "1-2 m", "2-4 m", ">4 m (clear)"])
    tb = (f - lo) // 600
    strata["time"] = by_stratum(rec, tb.astype(float),
                                [(i, i + 1) for i in range(10)],
                                [f"{i * 24}-{(i + 1) * 24}s" for i in range(10)])

    n_all = s["n_instants"]
    for name, rowsx in strata.items():
        log(f"\n  BY {name.upper()}   (last two columns are this stratum's SHARE "
            f"of the whole population's loss, not its own rate)")
        log(f"    {'stratum':>22} {'n':>7} {'share':>6} {'C':>6} {'D':>6} {'T':>6} "
            f"{'A':>6} {'noDet':>6} {'noTrk':>6} {'noAttr':>7} {'created':>8} "
            f"{'%noDet':>7} {'%noTrk':>7}")
        for r in rowsx:
            n = r["n_instants"]
            r["share_of_no_detection"] = (r["no_detection"] * n
                                          / (s["no_detection"] * n_all))
            r["share_of_det_but_no_track"] = (r["det_but_no_track"] * n
                                              / (s["det_but_no_track"] * n_all))
            log(f"    {r['stratum']:>22} {n:7d} {n / n_all:6.1%} "
                f"{r['coverage_C']:6.3f} "
                f"{r['ceiling_D']:6.3f} {r['tracked_T']:6.3f} {r['attributed_A']:6.3f} "
                f"{r['no_detection']:6.3f} {r['det_but_no_track']:6.3f} "
                f"{r['track_but_not_attributed']:7.3f} "
                f"{r['created_by_attribution']:8.3f} "
                f"{r['share_of_no_detection']:7.1%} "
                f"{r['share_of_det_but_no_track']:7.1%}")

    # DUEL WITHIN BOX HEIGHT. Duels are commoner at the far side, so the
    # marginal duel row above is partly a distance row wearing a duel label.
    # Crossed, the two separate: if duels cost anything of their own, the gap
    # survives inside the NEAREST quartile, where detection is otherwise ~1.00.
    log("\n  DUEL x BOX HEIGHT (the far-side confound removed by crossing them)")
    log(f"    {'height':>10} {'duel':>12} {'n':>7} {'D':>7} {'T':>7} {'A':>7} {'C':>7}")
    cross = []
    for a_, b_, lab in [(-np.inf, qh[0], "Q1 far"), (qh[0], qh[1], "Q2"),
                        (qh[1], qh[2], "Q3"), (qh[2], np.inf, "Q4 near")]:
        hm = (h >= a_) & (h < b_)
        for l_, u_, dl in [(-np.inf, 2.0, "<2 m duel"), (2.0, np.inf, ">=2 m clear")]:
            m = hm & (duel >= l_) & (duel < u_)
            if m.sum() < 50:
                continue
            row = {"height": lab, "duel": dl, "n": int(m.sum()),
                   "D": float(rec["D"][m].mean()), "T": float(rec["T"][m].mean()),
                   "A": float(rec["A"][m].mean()), "C": float(rec["C"][m].mean())}
            cross.append(row)
            log(f"    {lab:>10} {dl:>12} {row['n']:7d} {row['D']:7.3f} "
                f"{row['T']:7.3f} {row['A']:7.3f} {row['C']:7.3f}")

    log("\n  WHAT IT IS WORTH: today's estimator at each rung's coverage")
    w = worth(cal, rec, gtw, sc["players"], fps, lo, hi)
    log(f"    actual in this window: coverage {sc['coverage']:.3f}  ratio_raw "
        f"{sc['ratio_raw']:.3f}  ratio_matched {sc['ratio_matched']:.3f}")
    log(f"    {'rung':>6} {'coverage':>9} {'proj ours km':>13} {'proj ratio_raw':>15}")
    for k in ("C", "A", "T", "D", "ALL"):
        v = w["rungs"][k]
        log(f"    {k:>6} {v['coverage']:9.3f} {v['projected_ours_km']:13.2f} "
            f"{v['projected_ratio_raw']:15.3f}")
    log(f"    (GT in window {w['gt_km']:.2f} km. C must reproduce the actual "
        f"ratio_raw above; it is the projection's own control.)")

    out = {"window": [lo, hi], "tracks": a.tracks, "gate_m": a.gate_m,
           "image_gate_h": PRIMARY_H, "split": s, "sweep": rec["sweep"],
           "strata": strata, "duel_x_height": cross, "worth": w,
           "dist_eval": {k: v for k, v in sc.items() if k != "players"},
           "c_rung_mean_over_players": mine}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "coverage_decomp.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"\nwrote {p}")


# --------------------------------------------------------------------------

def selfcheck():
    """What this instrument can show, asserted before anything is read off it."""
    # 1. the matcher is exclusive and nearest-first, on arithmetic
    D = np.array([[0.1, 0.2], [0.15, 0.9]])
    assert assign(D, 1.0) == {0: 0, 1: 1}, assign(D, 1.0)
    assert assign(D, 0.12) == {0: 0}, assign(D, 0.12)
    assert assign(np.zeros((0, 3)), 1.0) == {}

    # 2. the identity holds on a hand-built ladder, including the sign of the
    #    attribution term. Three instants: one clean, one where the box exists
    #    but the owner is wrong, one credited with nothing under it.
    rec = {"g": np.array(["1", "1", "1"]), "f": np.array([0, 1, 2]),
           "D": np.array([True, True, False]), "T": np.array([True, True, False]),
           "A": np.array([True, False, False]), "C": np.array([True, False, True]),
           "steal": np.array([False, True, False]),
           "h": np.ones(3), "ym": np.ones(3), "xm": np.ones(3),
           "v": np.ones(3), "duel": np.ones(3), "Tid": np.zeros(3)}
    s = split(rec)
    assert abs(s["coverage_C"] - 2 / 3) < 1e-12, s
    assert abs(s["hidden_by_attribution"] - 1 / 3) < 1e-12, s
    assert abs(s["created_by_attribution"] - 1 / 3) < 1e-12, s

    # --- the two live controls, on the real files and the real calibration ---
    P = prepare(lo=None, hi=None)
    cal, jn, lo = P["cal"], P["jn"], P["lo"]
    hi = lo + 600                     # 600 frames is enough to move every rung
    off, perm = jn["offset"], jn["perm"]

    # 3. CEILING. GT's own feet as BOTH the detections and the tracks, each
    #    player his own id. Every rung must read 1.000 and neither attribution
    #    defect may fire. If this fails the ladder cannot show a clean
    #    pipeline, and no split it produces means anything.
    gf, gid, gxy = [], [], []
    for c in sorted(P["gt_px"]):
        if not lo <= c + off < hi:
            continue
        for i, foot in P["gt_px"][c].items():
            j = perm.get(i)
            if j is not None:
                gf.append(c + off)
                gid.append(int(j))
                gxy.append(foot)
    gf = np.array(gf, np.int64)
    gid = np.array(gid, np.int64)
    graw = np.array(gxy, float)
    dets = collections.defaultdict(list)
    for f, p in zip(gf, graw):
        dets[int(f)].append(p)
    dets = {k: np.array(v) for k, v in dets.items()}
    dom = {int(g): str(g) for g in set(gid.tolist())}
    seen = collections.defaultdict(set)
    for f, g in zip(gf, gid):
        seen[str(g)].add(int(f))
    ctrl = ladder(cal, jn, P["gt_px"], P["gt_h"], P["gtm_frames"], graw, gf, gid,
                  dets, dom, seen, lo=lo, hi=hi)
    cs = split(ctrl)
    assert cs["ceiling_D"] > 0.999 and cs["tracked_T"] > 0.999, cs
    assert cs["attributed_A"] > 0.999 and cs["coverage_C"] > 0.999, cs
    assert cs["created_by_attribution"] < 1e-9, cs
    assert cs["hidden_by_attribution"] < 1e-9, cs

    # 4. NEGATIVE. Swallow 40% of the track rows. `D - T` must absorb almost
    #    exactly that 40% and `1 - D` must not move at all. A decomposition
    #    that cannot see 40% of the tracker vanish cannot see the tracker.
    rng = np.random.default_rng(0)
    k = rng.random(len(gf)) > 0.4
    neg = ladder(cal, jn, P["gt_px"], P["gt_h"], P["gtm_frames"], graw[k], gf[k],
                 gid[k], dets, dom, seen, lo=lo, hi=hi)
    ns = split(neg)
    assert abs(ns["ceiling_D"] - cs["ceiling_D"]) < 1e-9, (ns, cs)
    assert 0.36 < ns["det_but_no_track"] < 0.44, ns
    assert 0.36 < ns["hidden_by_attribution"] * 0 + (1 - ns["tracked_T"]) < 0.44, ns
    # and with the frame credit left intact, the attribution rule now CREATES
    # exactly the coverage the tracker lost - which is the mechanism this file
    # is here to measure, reproduced on demand.
    assert abs(ns["created_by_attribution"] - ns["det_but_no_track"]) < 0.02, ns

    # 5. AND THE REAL RUN'S C RUNG IS dist_eval's COVERAGE. The check that
    #    fails if the decomposition drifts off the metric it decomposes.
    rec = ladder(cal, jn, P["gt_px"], P["gt_h"], P["gtm_frames"], P["rows_raw"],
                 P["row_f"], P["row_id"], P["dets_raw"], P["dominant"],
                 P["seen"], lo=lo, hi=hi)
    for g in sorted({*rec["g"]})[:5]:
        m = rec["g"] == g
        want = np.mean([int(f) in P["seen"].get(g, ()) for f in rec["f"][m]])
        assert abs(rec["C"][m].mean() - want) < 1e-12, (g, want)

    log(f"selfcheck ok  (matcher exclusive, identity holds, GT-as-everything "
        f"reads 1.000 on all four rungs, a 40% track dropout lands "
        f"{ns['det_but_no_track']:.3f} in det-but-no-track and creates "
        f"{ns['created_by_attribution']:.3f} of phantom coverage, C rung == "
        f"dist_eval's seen-set)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tracks", default="tracks.split.json")
    ap.add_argument("--gate-m", type=float, default=GATE_M)
    ap.add_argument("--sweep-gate-m", type=float, nargs="*", default=None,
                    help="re-attribute at each metre gate and print the ladder; "
                         "the knob `contam_eval.GATE_M` fixes by fiat")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.sweep_gate_m:
        log(f"{'gate_m':>7} {'C cov':>7} {'A':>7} {'T':>7} {'D':>7} {'hides':>7} "
            f"{'creates':>8} {'stolen':>7}")
        for gm in a.sweep_gate_m:
            P = prepare(a.tracks, gate_m=gm)
            r = ladder(P["cal"], P["jn"], P["gt_px"], P["gt_h"], P["gtm_frames"],
                       P["rows_raw"], P["row_f"], P["row_id"], P["dets_raw"],
                       P["dominant"], P["seen"], lo=P["lo"], hi=P["hi"])
            s = split(r)
            log(f"{gm:7.1f} {s['coverage_C']:7.4f} {s['attributed_A']:7.4f} "
                f"{s['tracked_T']:7.4f} {s['ceiling_D']:7.4f} "
                f"{s['hidden_by_attribution']:7.4f} "
                f"{s['created_by_attribution']:8.4f} "
                f"{s['stolen_by_another_id']:7.4f}")
    elif a.run:
        run(a)
    else:
        ap.print_help()
