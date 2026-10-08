"""What the SSAC26 vision layer costs when it is fed OUR inputs instead of a pose vendor's.

WHY THIS EXISTS
    `data/vendor/ssac26/vision.py` turns (x, y, speed, head angle) for one player plus
    (x, y, shoulder angle) for every other player into a top-down map of what that
    player can see. It is arithmetic: no model, no weights, no training. Dropped
    into this repo it would run today - except for one input. **We have no head
    angle and no shoulder angle, and nothing in this repo estimates either.**

    So the question is not "can we run it" but "what does it cost to run it on
    what we can actually produce". This file answers that, in the model's own
    output units, against the only pose ground truth we hold: the 4.6 s clip the
    upstream repo publishes as an animation demo.

THE PROTOCOL, FIXED BEFORE THE SWEEP RAN
    Population. Every (observer, target) player pair, every frame of
    `data/vendor/ssac26/ssac26_angles.csv`: 116 frames of ONE possession in ONE match,
    23 tracked objects, 2,373 player-frames carrying head, shoulder and run angle.
    This is one clip. It is not a split, there is nothing held out, and no number
    from it generalises to a match, a team or a league. It is used because it is
    the only pose-labelled football data in this repo, not because it is enough.

    Reference (what every arm is scored against). The upstream model with the
    vendor's own head angle, every player present, positions exactly as annotated.
    NOT truth about human vision - truth about what the published method outputs.

    Arms. Each replaces exactly one input with something this repo could plausibly
    produce, all else held at the reference:
      A1_run       head angle <- direction of travel. The free proxy: we have
                   velocity for every track today, and nothing else.
      A2_shoulder  head angle <- the player's own shoulder angle. Stands in for
                   "estimate torso orientation from the crop", which is a smaller
                   ask of a vision model than head yaw.
      A3_pos1m     positions perturbed by isotropic N(0, 1 m). Our calibration
                   where players stand, on the good path
                   (calibration-gate-is-open-at-1m-on-players (internal note)).
      A4_pos6m     the same at N(0, 6 m) - the clicked calibration the product
                   actually ships on a stitched panorama (model-map (not published) N6).
      A5_drop30    30% of the OTHER players deleted, standing in for the coverage
                   hole (api://product.distance.coverage). Occluders that are not
                   detected cannot occlude.
      A6_stack     A2 + A3 + A5 together: the best stack this repo could build
                   if someone shipped a torso-orientation estimator tomorrow.
    A1 and A2 were run alone before the sweep was written and reproduce inside it.

    Metric. For each pair, the arm's visibility value at the target's own grid
    cell against the reference's, and:
      disagree@t   share of pairs where (ref > t) and (arm > t) disagree - "would
                   we say this player was seen". Both t = 0.1 and t = 0.5 are
                   reported; n=2 thresholds, neither chosen after the fact.
      area ratio   sum of the arm's map over sum of the reference's map - whether
                   the arm thinks the player sees more or less pitch overall.
    Stochastic arms (A3-A6) run 3 seeds; the table pools them.

    DROP SHARE, because rule 5's corollary exists. A5 and A6 delete 30% of the
    targets, so they are scored on ~70% of the pairs the other arms are - and the
    deleted pairs are not free, they are rows the product would simply not have.
    The table prints the pair count for every arm for exactly this reason: A5's
    low disagreement is a statement about the pairs that SURVIVED, and reads as
    good news only if you ignore the ones that did not.

WHAT THE ANGLE TABLE IS FOR, AND THE TRAP IN IT
    `--angles` compares the three orientations against each other. Run it before
    anything else: in this data `head_angle` is ANTI-PARALLEL to `shoulders_angle`,
    `hips_angle` and `run_angle` - a constant pi offset that upstream documents
    nowhere. Subtract it and the head-to-shoulder disagreement collapses from
    near-antiparallel to single-digit degrees - `--angles` prints both, and the
    values are tagged in research-visual-exploratory-behavior (not published) §1, not
    repeated here where nothing would re-run them.
    Anyone who implements this from the repo without noticing will feed
    the occlusion model a torso pointing backwards, and the occlusion rectangle is
    symmetric under a pi rotation, so **it will not look broken**.

    That near-collinearity may be a property of the pose vendor rather than of
    football: a model that regresses head yaw off the torso cannot disagree with
    it. There is no way to tell from this clip, and it is the one finding here
    that most needs a second data source before it is built on.

    .venv/bin/python model/validation/vision_eval.py --angles
    .venv/bin/python model/validation/vision_eval.py --arms     # ~80 s
    .venv/bin/python model/validation/vision_eval.py --demo     # self-check

See research-visual-exploratory-behavior (not published) for what to do about all of this.
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "data", "vendor"))

from ssac26.vision import Vision  # noqa: E402
from ssac26.helpers import get_grid  # noqa: E402

CSV = os.path.join(ROOT, "data", "vendor", "ssac26", "ssac26_angles.csv")

# The pitch the upstream Dataset draws on: src/helpers.py PITCH_LENGTH/PITCH_WIDTH,
# and x_adj/y_adj in the published file are in that frame, not the centred x/y.
PITCH_L, PITCH_W = 100, 64
THRESHOLDS = (0.1, 0.5)
SEEDS = 3
DROP = 0.30
SIGMA_GOOD, SIGMA_SHIPPED = 1.0, 6.0


def wrap(a):
    """Angle difference into (-pi, pi]. Everything here is circular."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def load(path=CSV):
    """Rows as dicts, floats parsed, blanks as None. `#` lines are the header note."""
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if k in ("frame_id", "player_id", "jersey_number"):
                d[k] = int(v) if v else None
            elif k in ("team_id",):
                d[k] = v
            elif k in ("is_imputed", "is_on_ball"):
                d[k] = v == "True"
            else:
                d[k] = float(v) if v else None
        out.append(d)
    return out


def angles(rows):
    """Median/p90 absolute difference between the three published orientations.

    Returns a list of (label, n, median_deg, p90_deg, share_over_60deg).
    """
    ok = [r for r in rows if r["team_id"] != "ball" and r["head_angle"] is not None]
    g = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in ok])  # noqa: E731
    head, sh, hip, run = g("head_angle"), g("shoulders_angle"), g("hips_angle"), g("run_angle")
    pairs = [
        ("|head - shoulders|", head, sh),
        ("|head - (shoulders+pi)|", head, sh + np.pi),
        ("|head - (hips+pi)|", head, hip + np.pi),
        ("|head - (run+pi)|", head, run + np.pi),
        ("|shoulders - run|", sh, run),
    ]
    out = []
    for label, a, b in pairs:
        m = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(np.degrees(wrap(a - b)))[m]
        out.append((label, int(m.sum()), float(np.median(d)),
                    float(np.percentile(d, 90)), float(np.mean(d > 60))))
    return out


def by_speed(rows, quartiles=4):
    """|head - (run+pi)| by speed quartile - where the free proxy fails worst."""
    ok = [r for r in rows if r["team_id"] != "ball" and r["head_angle"] is not None
          and r["run_angle"] is not None]
    d = np.abs(np.degrees(wrap(np.array([r["head_angle"] for r in ok])
                               - np.array([r["run_angle"] for r in ok]) - np.pi)))
    v = np.array([r["v"] for r in ok])
    qs = np.quantile(v, np.linspace(0, 1, quartiles + 1))
    out = []
    for i in range(quartiles):
        sel = (v >= qs[i]) & (v <= qs[i + 1])
        out.append((float(qs[i]), float(qs[i + 1]), int(sel.sum()),
                    float(np.median(d[sel])), float(np.mean(d[sel] > 60))))
    return out


def _map(x, y, angle, speed, x2s, y2s, a2s):
    v = Vision(x=x, y=y, head_angle=angle, speed=speed,
               pitch_length=PITCH_L, pitch_width=PITCH_W)
    with np.errstate(invalid="ignore", divide="ignore"):
        # upstream `_normalize` divides by the map's max, which is 0 for an empty
        # wedge. Silenced here, COUNTED by the caller - not the same thing.
        g = v.compute(x2s=x2s, y2s=y2s, shoulder_angles=a2s)
    # A wedge that falls entirely off the grid normalises 0/0. Rare (2 of 2,373
    # player-frames in the run arm) and counted by the caller, not hidden.
    return np.nan_to_num(g), bool(np.isfinite(g).all())


def _cell(x, y):
    """Target's grid cell, through the model's OWN indexing so it cannot drift."""
    return (min(max(get_grid(y, PITCH_W, PITCH_W), 0), PITCH_W - 1),
            min(max(get_grid(x, PITCH_L, PITCH_L), 0), PITCH_L - 1))


ARMS = ("A1_run", "A2_shoulder", "A3_pos1m", "A4_pos6m", "A5_drop30", "A6_stack")


def arms(rows, seeds=SEEDS, drop=DROP):
    """The sweep. Returns {arm: dict(pairs, disagree@t..., ratio_med, ratio_iqr)}."""
    frames = {}
    for r in rows:
        frames.setdefault(r["frame_id"], []).append(r)
    acc = {a: [] for a in ARMS}
    mass = {a: [] for a in ARMS}
    n_ref = degenerate = 0
    for fid in sorted(frames):
        ps = [r for r in frames[fid] if r["team_id"] != "ball"
              and r["head_angle"] is not None and r["shoulders_angle"] is not None
              and r["run_angle"] is not None]
        if len(ps) < 3:
            continue
        for i, p in enumerate(ps):
            others = [q for j, q in enumerate(ps) if j != i]
            x2 = np.array([q["x_adj"] for q in others])
            y2 = np.array([q["y_adj"] for q in others])
            a2 = np.array([q["shoulders_angle"] for q in others])
            ref, fin = _map(p["x_adj"], p["y_adj"], p["head_angle"], p["v"], x2, y2, a2)
            n_ref += 1
            degenerate += not fin
            idx = [_cell(q["x_adj"], q["y_adj"]) for q in others]
            vr = np.array([ref[r, c] for r, c in idx])

            def add(arm, g, keep=None):
                vo = np.array([g[r, c] for r, c in idx]) if keep is None else \
                     np.array([g[r, c] for (r, c), k in zip(idx, keep) if k])
                base = vr if keep is None else vr[keep]
                acc[arm].append(np.c_[base, vo])
                mass[arm].append((ref.sum(), g.sum()))

            add("A1_run", _map(p["x_adj"], p["y_adj"], p["run_angle"] + np.pi,
                               p["v"], x2, y2, a2)[0])
            add("A2_shoulder", _map(p["x_adj"], p["y_adj"], p["shoulders_angle"] + np.pi,
                                    p["v"], x2, y2, a2)[0])
            rng = np.random.default_rng(fid * 100 + i)
            for arm, sig in (("A3_pos1m", SIGMA_GOOD), ("A4_pos6m", SIGMA_SHIPPED)):
                for _ in range(seeds):
                    dx, dy = rng.normal(0, sig, (2, len(others) + 1))
                    add(arm, _map(p["x_adj"] + dx[-1], p["y_adj"] + dy[-1],
                                  p["head_angle"], p["v"], x2 + dx[:-1], y2 + dy[:-1], a2)[0])
            for _ in range(seeds):
                keep = rng.random(len(others)) > drop
                if keep.sum() < 2:
                    continue
                add("A5_drop30", _map(p["x_adj"], p["y_adj"], p["head_angle"], p["v"],
                                      x2[keep], y2[keep], a2[keep])[0], keep)
                dx, dy = rng.normal(0, SIGMA_GOOD, (2, int(keep.sum()) + 1))
                add("A6_stack", _map(p["x_adj"] + dx[-1], p["y_adj"] + dy[-1],
                                     p["shoulders_angle"] + np.pi, p["v"],
                                     x2[keep] + dx[:-1], y2[keep] + dy[:-1], a2[keep])[0], keep)

    out = {"_ref": {"player_frames": n_ref, "degenerate_maps": degenerate}}
    for a in ARMS:
        z = np.vstack(acc[a])
        ref_v, arm_v = z[:, 0], z[:, 1]
        m = np.array(mass[a])
        ratio = m[:, 1] / np.maximum(m[:, 0], 1e-9)
        row = {"pairs": len(z),
               "ratio_med": float(np.median(ratio)),
               "ratio_iqr": (float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75)))}
        for t in THRESHOLDS:
            row[f"disagree@{t}"] = float(np.mean((ref_v > t) != (arm_v > t)))
            row[f"seen_ref@{t}"] = float(np.mean(ref_v > t))
            row[f"seen_arm@{t}"] = float(np.mean(arm_v > t))
        out[a] = row
    return out


def print_angles(rows):
    print(f"ANGLES  one clip, {len(rows)} rows, 116 frames, "
          f"game d9d1d8e0 period 2 - NOT a split, nothing held out")
    print(f"  {'pair':26s} {'n':>6s} {'median':>8s} {'p90':>8s} {'>60deg':>8s}")
    for label, n, med, p90, over in angles(rows):
        print(f"  {label:26s} {n:6d} {med:8.1f} {p90:8.1f} {over:8.3f}")
    print("  |head - (run+pi)| by speed quartile  (the free proxy, where it fails)")
    for lo, hi, n, med, over in by_speed(rows):
        print(f"    v [{lo:4.1f},{hi:4.1f}] m/s  n={n:5d}  median {med:6.1f}  >60deg {over:.3f}")


def print_arms(res):
    r = res["_ref"]
    print(f"ARMS  reference = upstream model, vendor head angle, all players, exact positions")
    print(f"      {r['player_frames']} player-frames, {r['degenerate_maps']} degenerate reference maps, "
          f"{SEEDS} seeds on the stochastic arms")
    print(f"  {'arm':12s} {'pairs':>8s} {'dis@0.1':>8s} {'dis@0.5':>8s} "
          f"{'seen_ref':>9s} {'seen_arm':>9s} {'area_med':>9s} {'area_IQR':>16s}")
    for a in ARMS:
        d = res[a]
        print(f"  {a:12s} {d['pairs']:8d} {d['disagree@0.1']:8.3f} {d['disagree@0.5']:8.3f} "
              f"{d['seen_ref@0.1']:9.3f} {d['seen_arm@0.1']:9.3f} {d['ratio_med']:9.3f} "
              f"  ({d['ratio_iqr'][0]:.3f},{d['ratio_iqr'][1]:.3f})")
    kept = res["A5_drop30"]["pairs"] / res["A1_run"]["pairs"] / SEEDS
    print(f"  DROP SHARE: A5/A6 score {kept:.3f} of the pairs the other arms do. "
          f"The missing {1 - kept:.3f} are not errors, they are rows the product would not have.")


def demo():
    """Self-check. Each assert fails on a specific way this file could be wrong."""
    # 1. the sampler indexes the model's own grid, not a transposed copy of it.
    g, _ = _map(50.0, 32.0, 0.0, 0.0, np.array([60.0]), np.array([32.0]), np.array([0.0]))
    r, c = _cell(60.0, 32.0)
    assert g.shape == (PITCH_W, PITCH_L), g.shape
    assert g[r, c] > 0.5, f"target dead ahead at 10 m should be seen, got {g[r, c]}"

    # 2. facing away is not visible - if this passes with the head reversed, the
    #    angle never reached the model.
    g2, _ = _map(50.0, 32.0, np.pi, 0.0, np.array([60.0]), np.array([32.0]), np.array([0.0]))
    assert g2[r, c] == 0.0, f"target behind the observer must be invisible, got {g2[r, c]}"

    # 3. the pi offset is real in the published data, and the correction is the
    #    right sign. This is the finding the whole file rests on.
    rows = load()
    tbl = {label: med for label, _, med, _, _ in angles(rows)}
    assert tbl["|head - shoulders|"] > 150, tbl
    assert tbl["|head - (shoulders+pi)|"] < 15, tbl
    assert tbl["|head - (shoulders+pi)|"] < tbl["|head - (run+pi)|"], tbl

    # 4. a dropped target is not silently scored as invisible: the pair vanishes.
    res = arms(rows[:24 * 20], seeds=1, drop=0.5)
    assert res["A5_drop30"]["pairs"] < res["A1_run"]["pairs"], res
    assert res["A2_shoulder"]["pairs"] == res["A1_run"]["pairs"], res

    # 5. an arm that changes nothing must disagree with nothing. Without this the
    #    whole table could be measuring the sampler instead of the input swap.
    ident = arms(rows[:24 * 20], seeds=1, drop=0.0)
    assert ident["A5_drop30"]["disagree@0.1"] == 0.0, ident["A5_drop30"]
    print("vision_eval.py ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--angles", action="store_true", help="the orientation table only")
    ap.add_argument("--arms", action="store_true", help="the input-substitution sweep (~80 s)")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--demo", action="store_true", help="self-check")
    a = ap.parse_args(argv)
    if a.demo:
        demo()
        return 0
    rows = load()
    if a.angles or not a.arms:
        print_angles(rows)
    if a.arms or not a.angles:
        print_arms(arms(rows, seeds=a.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
