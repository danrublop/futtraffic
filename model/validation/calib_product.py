"""N12: the operator's 21-click calibration, scored WHERE PLAYERS STAND.

WHY THIS EXISTS
    `model-map (not published)` §3 lists the product's calibration row as UNSCORED on
    the shipped path. The product's map is the operator clicking up to 21 pitch
    landmarks in the UI and `app/frontend/src/lib/pitch.ts` solving one homography
    from them. N6 measured its FIT residual - ~6 m RMS on ST2's stitched
    panoramas even with perfect clicks - and that residual is a property of the
    21 landmark positions, 8 of which sit on a touchline and 4 on a goal line.
    Nothing has ever asked what the same map does at a player's feet.

    That distinction has cost this repo a result twice
    (`landmark-holdout-selects-the-wrong-population`,
    `landmark-holdout-is-covariate-shift`, internal notes): on 118575
    a model chosen on landmarks was a 6x win there and a LOSS against real
    player positions. So the population here is player feet, and the landmarks
    the map was fitted on appear nowhere in the score.

--------------------------------------------------------------------------------
PROTOCOL, fixed 2026-09-12 BEFORE the first run, not after seeing a number
--------------------------------------------------------------------------------
population   Every ground-truth player foot in ST2's MOT ground truth
             (`{seq}.txt`), foot = (x + w/2, y + h) in RAW CLIP PIXELS - the
             same pixels `app/server/jobs.py`'s tracker sees, because the product
             runs no remap on this footage. Sampled every FRAME_EVERY = 25 clip
             frames (1 s at 25 fps) across the whole 4-minute clip, so ~240
             frames x 22 players ~ 5,280 instants a sequence. NOT landmarks,
             and no landmark is in this set.

sequences    The nine ST2 sequences that have a `.p05.calibration.json` on
             disk - the UI's own solver run headless on perfect simulated
             clicks by `model/validation/st2_product.py::calibrate`. 132831 is
             excluded upstream and the reason is the benchmark's, not ours: its
             pitch->pixel chain puts all 16 in-frame landmarks on one pixel.

reference    ST2's own chain, `pitch metres --H--> remapped --mapx--> clip px`
             (`st2_overlay.projector`), INVERTED on a GRID_M = 0.1 m grid with
             a KD-tree, exactly as `model/validation/st2_metres.inverse` does at
             0.25 m. The grid is extended GRID_MARGIN_M = 5 m past every
             touchline so a throw-in taker is mapped rather than clamped.

             THIS REFERENCE IS NOT TRUTH AND THE HEADLINE IS NOT ABSOLUTE.
             The same chain measures median 1.40 m against ST2's own player
             positions on 118575 (`api://calib.metres.118575.overall`, n=440).
             The headline below is therefore DISAGREEMENT WITH THE VENDOR MAP,
             which is a lower bound on the clicked map's absolute error: the
             reference's own error adds to it, roughly in quadrature away from
             the click points. `--absolute` measures the absolute version
             directly on the two sequences whose clip-frame offset into ST2's
             XML is known (118575, 128057, `st2_metres.OFFSET`), pairing GT
             boxes to ST2 `loc` with the REFERENCE map and then scoring BOTH
             maps on that one pairing - so a 6 m map cannot mis-pair its way to
             a better number, which is the failure `st2_metres` warns about.

bands        Four 17 m strips of pitch width on the REFERENCE y, declared here
             before any number: [0,17) [17,34) [34,51) [51,68). The near/far
             split is the two low strips against the two high ones. WHICH END
             IS FAR IS MEASURED, NOT ASSUMED: every band prints its median GT
             box height, and the far band is the one with the shortest boxes.
             (The repo's convention, `detection-tracking-evidence (not published)#evidence-far-touchline`, is band 0 = far
             touchline; this reports the evidence rather than inheriting it.)

metric       Median and p90 metres of error, overall and per band, plus the
             share of feet within GATES = 0.5 / 1 / 2 / 4 m - the same four
             thresholds `st2_metres` prints, so these are comparable to the
             registered `calib.metres.*` rows and to nuScenes' ground-plane
             distance matching. HEADLINE_GATE = 2 m is the stated gate.

drop share   One filter only: a foot whose nearest reference grid point is more
             than MAX_GRID_PX = 8 px away is outside the mapped region (more
             than 5 m off the pitch, or off the remap table) and is DROPPED.
             Its share is printed per sequence and stored in the artefact. No
             hull gate, no confidence gate, no band selection - the retraction
             in `calibration-is-not-the-distance-error` (internal note)
             was a hull gate quietly eating 14.1% of the detections at the far
             touchline, which is exactly where this map is worst.

held out     NOTHING IS FITTED HERE, so there is no train/test split to keep and
             claiming one would be theatre. What IS held out is the population:
             the map is fitted on 21 landmarks and scored on ~5,280 player feet
             that share no point with them. That is the hold-out the two
             memories above say is the one that matters.

             The clicks are PERFECT simulated clicks - the landmark's true pixel
             from ST2's own chain. That is a CEILING on an operator, not a
             sample of one. The sweep below is the honest half of the question.

n            HEADLINE n=1 configuration per sequence: one calibration, already
             on disk, nothing tuned, no argmax anywhere.
             SWEEP n = 6 sigmas x 3 seeds = 16 click-sets a sequence (sigma 0
             has one draw), 144 solver calls over the nine. Nothing is selected
             from the sweep either; it is reported as a curve.

sweep        `--sweep`: each click is displaced by an isotropic Gaussian of
             SIGMAS = 0, 1, 2, 5, 10, 20 px and the jittered set is re-solved by
             the SAME solver the UI ships (`app/frontend/scripts/solve-clicks.mts`),
             then re-scored on the same feet. Reported: median/p90 metres and
             the solver's own `rms_error_m` - the number the UI shows the
             operator - at every sigma, plus the rank correlation between the
             two across all 144 calibrations. That correlation is the only
             thing that could ever key a product rule on calibration quality.

--------------------------------------------------------------------------------
WHAT A FAILING RESULT LOOKS LIKE - all four written before the run
--------------------------------------------------------------------------------
 1. THE RUN IS INVALID, and no number leaves it, if any of:
      * a sigma-0 re-solve does not reproduce the on-disk `.p05.calibration
        .json` H bit for bit (`--demo` asserts this on 128057). "Reproduced"
        means diffed against the golden, not "the keys look right"
        (`reproduced-means-diffed-against-the-golden`, internal note);
      * the negative control misses: the reference map displaced by a known
        3.0 m must measure 3.0 m (`--demo`);
      * the drop share exceeds 10% on any sequence - at that point the filter
        has selected the population rather than measured it.

 2. THE N12 READING. The clicked calibration is a metric blocker iff its median
    error on GT feet is >= 2 m on at least 5 of the 9 sequences. IF THE MEDIAN
    IS UNDER 2 m the finding is the opposite one and it is just as useful: the
    ~6 m fit RMS is then a landmark-population artefact - the residual lives at
    the corners and the goal lines, where nobody plays - the map is usable where
    players stand, and §3's "the clicked calibration is the gap" has to be
    re-read as a detection/association story rather than a metric one.

 3. THE SWEEP READING. Operator click precision is a product lever iff the
    median at sigma = 5 px is at least 1 m worse than at sigma = 0. If the curve
    is flat, telling operators to click more carefully buys nothing and the
    ceiling is the single-homography model itself, which no click can fix.

 4. THE GATE READING. `rms_error_m` is usable as a calibration-quality signal
    iff |Spearman rho| >= 0.5 against the measured median across the 144
    calibrations. Below that, the number the UI shows the operator does not
    predict what the map does to a player and must not gate anything.

--------------------------------------------------------------------------------
    .venv/bin/python model/validation/calib_product.py            # headline, ~1 min
    .venv/bin/python model/validation/calib_product.py --sweep    # + perturbation
    .venv/bin/python model/validation/calib_product.py --absolute # vs ST2 `loc`
    .venv/bin/python model/validation/calib_product.py --demo     # self-check, ~5 s

Artefact: model/validation/out/n12_calib_product.json (model/validation/out/ is gitignored).
Results and their @M tags: n12-calibration (not published).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import st2_mot as T  # noqa: E402
from st2_overlay import pitch_of, projector  # noqa: E402

#: The nine ST2 sequences with a UI-solved calibration on disk. 132831 has none:
#: its ST2 chain collapses all 16 in-frame landmarks onto one pixel.
SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132877")
FRAME_EVERY = 25          # clip frames between sampled instants (1 s at 25 fps)
GRID_M = 0.1              # reference inverse grid step, metres
GRID_MARGIN_M = 5.0       # grid extends this far past every touchline
MAX_GRID_PX = 8.0         # foot farther than this from the grid -> outside the map
BANDS = ((0.0, 17.0), (17.0, 34.0), (34.0, 51.0), (51.0, 68.0))
GATES = (0.5, 1.0, 2.0, 4.0)
HEADLINE_GATE = 2.0
SIGMAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
SEEDS = (0, 1, 2)
OUT = os.path.join(ROOT, "model/validation/out/n12_calib_product.json")


def reference(seq):
    """clip px -> (pitch metres, pixel residual). ST2's own chain, inverted.

    The chain has no closed-form inverse (`st2_metres` explains why a direct
    cv2.invert(H) is the trap), so the forward map is evaluated on a GRID_M grid
    and inverted with a KD-tree. The residual comes back so the caller can drop
    feet that fall outside the mapped region instead of silently clamping them
    to its edge.

    ITS RESOLUTION IS THE FOOTAGE'S, NOT THE GRID'S. `projector` rounds to a
    whole remap-table pixel, so every grid point inside one image pixel is a
    plateau and the tree returns an arbitrary member of it. At the far touchline
    one image row is 0.666 m of pitch
    (`landmark-holdout-selects-the-wrong-population`, internal note),
    so the reference carries up to ~+-0.33 m of pick noise THERE and far less
    near the camera. Noise, not bias, and two orders below the residual this is
    built to measure - but it is the floor, and no smaller GRID_M moves it.
    """
    P = pitch_of(seq)
    fn = projector(seq, T._wh(seq))
    gx = np.arange(-GRID_MARGIN_M, P.length + GRID_MARGIN_M + 1e-9, GRID_M)
    gy = np.arange(-GRID_MARGIN_M, P.width + GRID_MARGIN_M + 1e-9, GRID_M)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    px, ok = fn(G)
    G = G[ok]
    tree = cKDTree(px[ok])

    def inv(uv):
        d, i = tree.query(np.asarray(uv, float))
        return G[i], d
    return inv, len(G)


def gt_feet(seq, every=FRAME_EVERY):
    """(N,2) foot pixels and (N,) box heights, every `every`th clip frame."""
    rows = []
    with open(f"{T.GT_SRC}/{seq}.txt") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 6 or int(p[0]) % every:
                continue
            x, y, w, h = (float(v) for v in p[2:6])
            rows.append((x + w / 2.0, y + h, h))
    A = np.array(rows, float)
    return A[:, :2], A[:, 2]


def score(ref_xy, pred_xy, box_h):
    """The whole metric, in one place so the self-check can plant a known error.

    `ref_xy` is where the reference map puts each foot, `pred_xy` where the map
    under test puts it. Both (N,2) pitch metres, already drop-filtered.
    """
    dv = pred_xy - ref_xy
    d = np.linalg.norm(dv, axis=1)
    # OFFSET vs WARP, because they cost different things downstream. A rigid
    # translation of the whole pitch leaves every DISTANCE and speed untouched
    # and ruins every zone; what survives removing the best constant offset is
    # the part that also bends distances. Splitting it is the difference between
    # "possession zones are wrong" and "distance covered is wrong".
    off = dv.mean(0)
    warp = np.linalg.norm(dv - off, axis=1)
    out = {"n": int(len(d)), "median_m": float(np.median(d)),
           "p90_m": float(np.percentile(d, 90)),
           "offset_m": float(np.hypot(*off)),
           "offset_xy_m": [float(off[0]), float(off[1])],
           "median_after_offset_m": float(np.median(warp)),
           "p90_after_offset_m": float(np.percentile(warp, 90)),
           "within": {f"{g:g}m": float((d < g).mean()) for g in GATES},
           "bands": {}}
    for lo, hi in BANDS:
        m = (ref_xy[:, 1] >= lo) & (ref_xy[:, 1] < hi)
        out["bands"][f"{lo:g}-{hi:g}"] = {
            "n": int(m.sum()),
            "median_m": float(np.median(d[m])) if m.any() else None,
            "p90_m": float(np.percentile(d[m], 90)) if m.any() else None,
            # The evidence for which end is far, so nothing has to be assumed.
            "median_box_px": float(np.median(box_h[m])) if m.any() else None,
        }
    near = ref_xy[:, 1] >= 34.0
    for lab, m in (("y_lt_34", ~near), ("y_ge_34", near)):
        out[lab] = {"n": int(m.sum()),
                    "median_m": float(np.median(d[m])) if m.any() else None,
                    "p90_m": float(np.percentile(d[m], 90)) if m.any() else None,
                    "median_box_px": float(np.median(box_h[m])) if m.any() else None}
    return out


def _feet_and_ref(seq):
    """-> (uv, box_h, ref_xy, drop_share, n_grid) with the drop filter applied."""
    inv, n_grid = reference(seq)
    uv, box_h = gt_feet(seq)
    ref_xy, resid = inv(uv)
    keep = resid <= MAX_GRID_PX
    return uv[keep], box_h[keep], ref_xy[keep], float(1.0 - keep.mean()), n_grid


def clicks_of(seq):
    """The perfect simulated clicks `st2_product.calibrate` feeds the UI solver."""
    P = pitch_of(seq)
    lm = P.landmarks()
    px, ok = projector(seq, T._wh(seq))(list(lm.values()))
    return P, {n: [float(u), float(v)] for n, (u, v), k in zip(lm, px, ok) if k}


def solve(pitch, clicks):
    """The PRODUCT's solver, headless. Not model/engine/pitch.py: on this footage its
    LMEDS throws out most of a set of perfect clicks (N6). -> payload or None."""
    r = subprocess.run(
        ["npx", "vite-node", "--config", "vitest.config.mts", "scripts/solve-clicks.mts"],
        cwd=os.path.join(ROOT, "app", "frontend"), check=True, capture_output=True,
        text=True, input=json.dumps({"pitch": pitch.__dict__, "clicks": clicks}))
    payload = json.loads(r.stdout)
    return None if "error" in payload else payload


def _to_pitch(payload, uv):
    import calibrators  # noqa: PLC0415
    return calibrators.from_dict(payload).to_pitch(uv)


def run(seqs=SEQS):
    """The headline: the shipped calibration on GT feet, per sequence."""
    import calibrators  # noqa: PLC0415
    res = {}
    for seq in seqs:
        uv, box_h, ref_xy, drop, n_grid = _feet_and_ref(seq)
        path = os.path.join(T.WORK, f"{seq}.p05.calibration.json")
        cal = calibrators.load(path)
        s = score(ref_xy, cal.to_pitch(uv), box_h)
        s["drop_share"] = drop
        s["fit_rms_m"] = float(json.load(open(path))["rms_error_m"])
        s["n_clicks"] = len(json.load(open(path))["points"])
        s["ref_on_pitch"] = float((
            (ref_xy[:, 0] >= 0) & (ref_xy[:, 0] <= 105)
            & (ref_xy[:, 1] >= 0) & (ref_xy[:, 1] <= 68)).mean())
        res[seq] = s
        b = s["bands"]
        print(f"{seq}  n={s['n']:5d} drop={100 * drop:4.1f}%  fit_rms={s['fit_rms_m']:.2f} m"
              f"  median={s['median_m']:6.2f} m  p90={s['p90_m']:6.2f} m"
              f"  <2m={100 * s['within']['2m']:5.1f}%"
              f"  offset={s['offset_m']:5.2f} m  warp={s['median_after_offset_m']:5.2f} m")
        print("      bands " + "  ".join(
            f"{k}: {v['median_m']:.2f}/{v['p90_m']:.2f} m (box {v['median_box_px']:.0f}px n={v['n']})"
            if v["median_m"] is not None else f"{k}: -" for k, v in b.items()))
    q = lambda k, f=None: float(np.median(  # noqa: E731
        [(f(v) if f else v[k]) for v in res.values()]))
    med, blockers = q("median_m"), sum(
        v["median_m"] >= HEADLINE_GATE for v in res.values())
    print(f"\nover {len(res)} sequences: median of medians {med:.2f} m, "
          f"p90 of p90s {q('p90_m'):.2f} m, within 2 m {100 * q('', lambda v: v['within']['2m']):.1f}%, "
          f"within 4 m {100 * q('', lambda v: v['within']['4m']):.1f}%, "
          f"max drop share {100 * max(v['drop_share'] for v in res.values()):.2f}%")
    print(f"far (y<34) {q('', lambda v: v['y_lt_34']['median_m']):.2f} m vs "
          f"near (y>=34) {q('', lambda v: v['y_ge_34']['median_m']):.2f} m; "
          f"offset {q('offset_m'):.2f} m and warp after removing it "
          f"{q('median_after_offset_m'):.2f} m")
    print(f"{blockers}/{len(res)} sequences at or above the {HEADLINE_GATE:g} m gate "
          f"-> N12 reading: {'BLOCKER' if blockers >= 5 else 'NOT A BLOCKER'}")
    return {"headline": res, "n_ge_gate": blockers,
            "median_of_medians_m": med}


def sweep(seqs=SEQS):
    """Perturbed clicks: how much worse is a real click than a perfect one."""
    rows = []
    for seq in seqs:
        uv, box_h, ref_xy, drop, _ = _feet_and_ref(seq)
        P, clean = clicks_of(seq)
        names = list(clean)
        base = np.array([clean[n] for n in names], float)
        for sigma in SIGMAS:
            for seed in (SEEDS if sigma else SEEDS[:1]):
                rng = np.random.default_rng([int(seq), seed, int(sigma * 10)])
                jit = base + (rng.normal(0.0, sigma, base.shape) if sigma else 0.0)
                payload = solve(P, {n: list(map(float, p)) for n, p in zip(names, jit)})
                if payload is None:
                    rows.append({"seq": seq, "sigma_px": sigma, "seed": seed,
                                 "solved": False})
                    continue
                s = score(ref_xy, _to_pitch(payload, uv), box_h)
                rows.append({"seq": seq, "sigma_px": sigma, "seed": seed,
                             "solved": True, "fit_rms_m": payload["rms_error_m"],
                             "outliers": len(payload["outliers"]),
                             "median_m": s["median_m"], "p90_m": s["p90_m"],
                             "within_2m": s["within"]["2m"],
                             "far_median_m": s["y_lt_34"]["median_m"],
                             "near_median_m": s["y_ge_34"]["median_m"],
                             "drop_share": drop})
        print(f"  {seq} done ({sum(r['seq'] == seq for r in rows)} click-sets)")
    ok = [r for r in rows if r["solved"]]
    print(f"\nsigma  n  fit_rms_m      median_m      p90_m     <2m    outliers")
    table = {}
    for sigma in SIGMAS:
        g = [r for r in ok if r["sigma_px"] == sigma]
        if not g:
            continue
        f = lambda k: float(np.median([r[k] for r in g]))  # noqa: E731
        table[f"{sigma:g}"] = {"n": len(g), "fit_rms_m": f("fit_rms_m"),
                               "median_m": f("median_m"), "p90_m": f("p90_m"),
                               "within_2m": f("within_2m"),
                               "outliers": f("outliers")}
        t = table[f"{sigma:g}"]
        print(f"{sigma:5g} {t['n']:3d}   {t['fit_rms_m']:7.2f}      {t['median_m']:7.2f}   "
              f"{t['p90_m']:8.2f}  {100 * t['within_2m']:5.1f}%  {t['outliers']:6.1f}")
    from scipy.stats import spearmanr  # noqa: PLC0415
    rho = float(spearmanr([r["fit_rms_m"] for r in ok],
                          [r["median_m"] for r in ok]).statistic)
    d5 = table["5"]["median_m"] - table["0"]["median_m"]
    print(f"\nsigma 5 px costs {d5:+.2f} m of median -> click precision is "
          f"{'A LEVER' if d5 >= 1.0 else 'NOT the lever'}")
    print(f"Spearman rho(fit_rms_m, median_m) = {rho:.3f} over n={len(ok)} "
          f"calibrations -> the UI's residual "
          f"{'CAN' if abs(rho) >= 0.5 else 'CANNOT'} gate calibration quality")
    return {"sweep": table, "rows": rows, "rho_rms_vs_median": rho,
            "sigma5_minus_sigma0_m": float(d5), "n_calibrations": len(ok)}


def absolute(seqs=("118575", "128057"), cals=None):
    """Absolute error against ST2's own `loc`, on the reference map's pairing.

    The reference map is ~1.4 m off itself, so the headline above is a lower
    bound. This is the direct measurement, and it exists on exactly the two
    scorable sequences whose clip-frame offset into the XML is established
    (st2_metres.OFFSET; 132831's third entry has no calibration). PAIRING IS
    MADE BY THE REFERENCE MAP AND THEN FROZEN - a 6 m map allowed to run its own
    Hungarian assignment can mis-pair its way to a flattering number, which is
    the understatement st2_metres warns about in its own docstring.

    `cals` maps seq -> any object with `to_pitch(uv)`, to score a map other than
    the shipped `.p05` calibration on the same pairing (camera_models.py).
    """
    import calibrators  # noqa: PLC0415
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    import st2_metres  # noqa: PLC0415
    out = {}
    for seq in seqs:
        P = pitch_of(seq)
        inv, _ = reference(seq)
        cal = cals[seq] if cals else calibrators.load(os.path.join(T.WORK, f"{seq}.p05.calibration.json"))
        feet = st2_metres.gt_feet(seq)
        frames = list(range(0, max(feet) + 1, 300))
        truth = st2_metres.st2_truth(seq, frames)
        ref_d, cal_d, drop = [], [], [0, 0]
        for f, Tr in truth.items():
            if f not in feet or not len(Tr):
                continue
            Tr = np.c_[Tr[:, 0] * P.length, Tr[:, 1] * P.width]
            uv = np.array([(x, y) for x, y, _ in feet[f]])
            R, resid = inv(uv)
            keep = resid <= MAX_GRID_PX
            drop[0] += int((~keep).sum()); drop[1] += len(keep)
            R, uv = R[keep], uv[keep]
            if not len(R):
                continue
            C = cal.to_pitch(uv)
            D = np.linalg.norm(R[:, None] - Tr[None], axis=2)
            r, c = linear_sum_assignment(D)      # pairing from the REFERENCE map
            ref_d += [D[a, b] for a, b in zip(r, c)]
            cal_d += [float(np.linalg.norm(C[a] - Tr[b])) for a, b in zip(r, c)]
        ref_d, cal_d = np.array(ref_d), np.array(cal_d)
        out[seq] = {"n": int(len(ref_d)), "frames": len(truth),
                    "drop_share": drop[0] / max(drop[1], 1),
                    "reference_median_m": float(np.median(ref_d)),
                    "reference_p90_m": float(np.percentile(ref_d, 90)),
                    "clicked_median_m": float(np.median(cal_d)),
                    "clicked_p90_m": float(np.percentile(cal_d, 90)),
                    "clicked_within_2m": float((cal_d < 2.0).mean())}
        o = out[seq]
        print(f"{seq}  n={o['n']}  frames={o['frames']}  drop={100 * o['drop_share']:.1f}%\n"
              f"    reference (ST2's own chain) median {o['reference_median_m']:.2f} m  "
              f"p90 {o['reference_p90_m']:.2f} m\n"
              f"    CLICKED   (the product)     median {o['clicked_median_m']:.2f} m  "
              f"p90 {o['clicked_p90_m']:.2f} m  <2m {100 * o['clicked_within_2m']:.1f}%")
    return {"absolute": out}


def demo():
    """Self-check, including the negative control the protocol promised."""
    seq = "128057"
    inv, n_grid = reference(seq)
    assert n_grid > 100_000, n_grid

    # Round trip: metres -> pixels -> metres. The tolerance is the FOOTAGE's,
    # not the grid's: one image row is 0.666 m at the far touchline, so a
    # far-side point comes back up to that far away and a near-side one exactly.
    # The exact check is the pixel residual - the inverse must land on a grid
    # point whose pixel IS the queried pixel.
    fn = projector(seq, T._wh(seq))
    pts = np.array([[52.5, 34.0], [20.0, 10.0], [90.0, 60.0]])
    px, ok = fn(pts)
    assert ok.all()
    back, resid = inv(px)
    assert np.abs(back - pts).max() <= 1.0, back - pts
    assert resid.max() <= MAX_GRID_PX, resid

    uv, box_h, ref_xy, drop, _ = _feet_and_ref(seq)
    assert 1000 < len(uv), len(uv)
    assert drop < 0.10, f"drop share {drop:.3f} selects the population"

    # POSITIVE CONTROL: the reference scored against itself is exactly zero.
    z = score(ref_xy, ref_xy, box_h)
    assert z["median_m"] == 0.0 and z["p90_m"] == 0.0, z

    # NEGATIVE CONTROL: a map displaced by a known 3 m must measure 3 m, in
    # every band. A metric that cannot see a planted error cannot see a real
    # one, and this is the assertion that fails if `score` ever drifts.
    planted = score(ref_xy, ref_xy + np.array([3.0, 0.0]), box_h)
    assert abs(planted["median_m"] - 3.0) < 1e-9, planted["median_m"]
    assert abs(planted["p90_m"] - 3.0) < 1e-9, planted["p90_m"]
    assert planted["within"]["2m"] == 0.0 and planted["within"]["4m"] == 1.0, planted
    # ...and the offset/warp split must call a planted TRANSLATION all offset.
    assert abs(planted["offset_m"] - 3.0) < 1e-9, planted["offset_m"]
    assert planted["median_after_offset_m"] < 1e-9, planted["median_after_offset_m"]
    for k, v in planted["bands"].items():
        assert v["median_m"] is None or abs(v["median_m"] - 3.0) < 1e-9, (k, v)
    # ...and a displacement ACROSS the bands must move the band populations,
    # so the band split is reading the reference and not a constant.
    shifted = score(ref_xy + np.array([0.0, 20.0]), ref_xy, box_h)
    assert shifted["bands"]["0-17"]["n"] != planted["bands"]["0-17"]["n"]

    # The drop filter must actually fire on a foot that is off the map.
    _, far = inv(np.array([[-5000.0, -5000.0]]))
    assert far[0] > MAX_GRID_PX, far

    # REPRODUCED MEANS DIFFED AGAINST THE GOLDEN: a sigma-0 re-solve has to give
    # back the H that is on disk, bit for bit. If this fails, every sweep row is
    # measuring a different click set from the headline.
    pitch, clicks = clicks_of(seq)
    payload = solve(pitch, clicks)
    golden = json.load(open(os.path.join(T.WORK, f"{seq}.p05.calibration.json")))
    assert payload is not None and payload["H"] == golden["H"], "sigma-0 re-solve drifted"
    assert payload["rms_error_m"] == golden["rms_error_m"]

    # And the far band is the one with the SHORT boxes - the claim the bands
    # rest on, measured rather than inherited from detection-tracking-evidence (not published)#evidence-far-touchline.
    real = score(ref_xy, _to_pitch(golden, uv), box_h)
    assert real["y_lt_34"]["median_box_px"] < real["y_ge_34"]["median_box_px"], real
    print(f"ok  ({seq}: {n_grid} grid points, {len(uv)} feet, drop {100 * drop:.1f}%, "
          f"planted 3 m read back as {planted['median_m']:.3f} m, "
          f"sigma-0 re-solve == golden H, far band box "
          f"{real['y_lt_34']['median_box_px']:.0f} px < near "
          f"{real['y_ge_34']['median_box_px']:.0f} px)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="perturbed clicks too")
    ap.add_argument("--absolute", action="store_true", help="vs ST2 `loc`, 2 seqs")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
        sys.exit(0)
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if a.sweep:
        old.update(sweep())
    elif a.absolute:
        old.update(absolute())
    else:
        old.update(run())
    with open(OUT, "w") as fh:
        json.dump(old, fh, indent=1)
    print(f"\n-> {OUT}")
