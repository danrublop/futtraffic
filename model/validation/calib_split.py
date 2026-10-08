"""Split the +23% calibration win between the HOMOGRAPHY and the GROUND POINT.

WHAT IS ALREADY SETTLED (`model/validation/calib_hypothesis.py`, docs §A)
    Substituting an oracle pixel->metre map at the calibration step takes
    per-detection metre recall 0.575 -> 0.753 and `ratio_raw` 0.795 -> 1.046.

    `@M` **THE SECOND NUMBER IS RETRACTED (2026-09-10).** `calib_hypothesis.Oracle`
    extrapolated outside its own training hull; gated, `ratio_raw` reads 0.811,
    not 1.046. This file's `span` at :785 is therefore ~15x smaller than when the
    98.4/1.6 split was published, and `orcf` at :763 still calls the ORACLE
    UNGATED - pass `base=shipped_map(cal)(p)` before quoting the share again.
    The recall figure and the residual decomposition are unaffected: both are
    scored on the held-out GT points, which are inside the hull by construction.
    The shipped map compresses path length by ~23%.

    But the oracle is a local regression from BOX-BOTTOM pixels to ANNOTATED
    ground position, so it absorbs two different errors at once and cannot say
    which module owns them:

      HOMOGRAPHY CHAIN   model/engine/calibrators.py. Error is a function of where on
                         the ground plane you are.
      GROUND POINT       model/engine/groundpoint.py + the detector. Where in the box
                         we call the foot. A fixed VERTICAL PIXEL offset.

WHY "ROW EFFECT vs POSITION EFFECT" IS NOT A TEST, AS WRITTEN
    On a fixed camera the image row and the cross-pitch coordinate are one
    variable. `--design` measures the correlation on this footage rather than
    asserting it. A regression of the residual on row and on position therefore
    CANNOT come out two ways along that axis; running it as stated would have
    produced a confident number for an unidentifiable model. (postmortem (not published)
    C2.)

    TWO THINGS ARE FREE, AND THEY ARE THE TEST:

    1. THE PIXEL REPARAMETERISATION. A ground-point error is a fixed vertical
       displacement IN PIXELS. Push the metre residual back through the shipped
       map's local Jacobian and it becomes an equivalent pixel displacement
       (du, dv). A ground-point error is then a CONSTANT (0, c) everywhere,
       whatever the row; a homography error is not, and generally has a du.
       The nuisance correlation is gone because the model is constant in the
       parameterisation, not fitted against it.

    2. THE ALONG-TOUCHLINE DIRECTION. At a fixed image row the panorama column
       sweeps the whole pitch length, so residual structure in pitch X at fixed
       row is separable from every row effect. `--design` reports the spread.

    WHAT REMAINS UNIDENTIFIABLE, STATED UP FRONT: a homography error that
    happens to depend ONLY on cross-pitch distance is indistinguishable from a
    row-varying ground-point offset. Control D below plants exactly that and
    shows the harness calling it a ground-point term. Only the CONSTANT term and
    the along-X structure are clean.

THE THIRD TERM, WHICH THE ORACLE NEVER SAW
    The oracle is fitted on GT box bottoms, so it swallows any offset between
    the GT box bottom and the true ground contact. It does NOT swallow the
    offset between OUR detector's box bottom and the GT box bottom - that error
    survives every map change, and no calibration work can touch it.
    `--boxbias` measures it, signed, by row. context-pitch-mapping-pivot (not published)
    §8.2 lists it as unmeasured and worth 1.2 m at the far side at 2 px.

    §8.2 says measuring it "is a re-run, not an analysis" because tracks.* carry
    no w/h. That is true of tracks.*; `dets.cache.json` carries full xyxy boxes
    in band space for all 71,850 frames, and band->raw is a table lookup.

    python model/validation/calib_split.py --design     # identifiability + controls
    python model/validation/calib_split.py --selfcheck  # the controls, as asserts
    python model/validation/calib_split.py --boxbias    # our boxes vs GT boxes
    python model/validation/calib_split.py --run        # decomposition + ratio ladder
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
import calib_hypothesis as CH  # noqa: E402

JOB = CH.JOB
OUT = CH.OUT
DETS = os.path.join(JOB, "dets.cache.json")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# the join and the oracle, reused verbatim - re-deriving them is how you get a
# different answer to the same question

def load_join() -> dict:
    p = os.path.join(OUT, "calib_hypothesis_join.json")
    jn = json.load(open(p))
    jn["perm"] = {int(k): v for k, v in jn["perm"].items()}
    assert jn["offset"] == 5998, jn["offset"]
    assert not jn["identity_perm"], "the id map is a permutation, not identity"
    return jn


def heldout(jn: dict, test_every: int = 5):
    """The oracle's OWN held-out points: second temporal half, every 5th frame.

    `build_oracle` does not return them, so the split rule is repeated here -
    and then asserted against the count `build_oracle` reported, which is what
    makes it a reuse rather than a re-derivation.
    """
    px, (_fps, gm) = CH.load_gt_px(), CH.load_gt_m()
    off, perm = jn["offset"], jn["perm"]
    frames = sorted(px)
    cut = frames[len(frames) // 2]
    U, X, F, I = [], [], [], []
    for c in [f for f in frames if f >= cut][::test_every]:
        g = gm.get(c + off)
        if not g:
            continue
        for i, foot in px[c].items():
            j = perm.get(i)
            if j in g:
                U.append(foot)
                X.append(g[j])
                F.append(c + off)
                I.append(j)
    return (np.array(U, float), np.array(X, float),
            np.array(F, int), np.array(I))


# --------------------------------------------------------------------------
# the local Jacobians: metres per band pixel, and raw pixels per band pixel

def jacobians(cal, band: np.ndarray, h: float = 1.0):
    """(N,2,2) d(metres)/d(band px) and (N,2,2) d(raw px)/d(band px).

    Central differences. `to_pitch` is a homography and `band_to_raw` a bilinear
    table, so both are smooth at 1 px and a one-sided difference would carry a
    curvature term for no reason.
    """
    def d(fn):
        return np.stack([(fn(band + [h, 0]) - fn(band - [h, 0])) / (2 * h),
                         (fn(band + [0, h]) - fn(band - [0, h])) / (2 * h)],
                        axis=2)                      # [:, out_axis, in_axis]
    Jh = d(lambda p: np.asarray(cal.to_pitch(p), float))
    Jg = d(lambda p: CH.band_to_raw(cal, p))
    return Jh, Jg


def to_pixels(Jh, Jg, r):
    """Metre residual -> the equivalent displacement in RAW panorama pixels.

    r = Jh @ d_band  =>  d_band = Jh^-1 r  =>  d_raw = Jg @ d_band.
    """
    d_band = np.linalg.solve(Jh, r[:, :, None])
    return (Jg @ d_band)[:, :, 0], d_band[:, :, 0]


def raw_dirs(Jh, Jg):
    """Metres moved per ONE raw pixel of u and of v, as (N,2,2) columns."""
    Ji = np.linalg.inv(Jg)                     # d(band)/d(raw)
    return Jh @ Ji                             # d(metres)/d(raw)


# --------------------------------------------------------------------------
# the models. Each is "a pixel offset of some shape"; what none of them can
# express is the homography term.

def fit_offset(r, D, groups=None, axes=(1,)):
    """Least-squares pixel offset explaining the metre residual r.

    r   (N,2) metre residual, oracle minus shipped
    D   (N,2,2) d(metres)/d(raw px); D[:, :, k] is the metre direction of a
        unit raw-pixel step along axis k (0 = u, 1 = v)
    groups  per-point integer bin, or None for one global offset
    axes    which pixel axes the model is allowed to move (1 = vertical only,
            which is the foot-lift shape; (0,1) also allows a horizontal
            box-centre bias)

    Returns (coefficients per group per axis, fitted metre field, R^2 on the
    metre residual). R^2 is on the SUM OF SQUARED METRES, both components, which
    is the currency the +23% is eventually expressed in.
    """
    A = np.stack([D[:, :, k] for k in axes], axis=2)      # (N,2,len(axes))
    g = np.zeros(len(r), int) if groups is None else groups
    ng = int(g.max()) + 1
    coef = np.zeros((ng, len(axes)))
    fit = np.zeros_like(r)
    for k in range(ng):
        m = g == k
        if m.sum() < len(axes) + 1:
            continue
        Ak = A[m].reshape(-1, len(axes))                  # (2n, len(axes))
        c, *_ = np.linalg.lstsq(Ak, r[m].reshape(-1), rcond=None)
        coef[k] = c
        fit[m] = (A[m] @ c)
    ss = float((r ** 2).sum())
    return coef, fit, 1.0 - float(((r - fit) ** 2).sum()) / ss


def qbins(x, k=10):
    """Equal-count bins. Equal-width bins on a perspective camera put 90% of
    the points in one bin and then report its mean as a trend."""
    e = np.quantile(x, np.linspace(0, 1, k + 1))
    e[-1] += 1e-9
    return np.clip(np.searchsorted(e, x, "right") - 1, 0, k - 1), e


# --------------------------------------------------------------------------
# map variants, all evaluated the same way: raw pixel -> metres

def shipped_map(cal):
    def fn(raw):
        return np.asarray(cal.to_pitch(CH.raw_to_band(cal, raw)), float)
    return fn


def raw_offset(coef, edges=None):
    """A ground-point correction: raw pixels -> the offset to add to them.

    `coef` is (nbins, 1) vertical-only or (nbins, 2) with a horizontal term;
    `edges` are the interior row edges, or None for one global offset.
    """
    coef = np.atleast_2d(coef)
    def off(raw):
        b = (np.zeros(len(raw), int) if edges is None else
             np.clip(np.searchsorted(edges, raw[:, 1], "right") - 1,
                     0, len(coef) - 1))
        d = coef[b]
        o = np.zeros_like(raw)
        o[:, 1] = d[:, -1]
        if d.shape[1] == 2:
            o[:, 0] = d[:, 0]
        return o
    return off


def shifted_map(cal, coef, edges=None, base=None):
    """A map with a ground-point correction bolted on and NOTHING else changed:
    shift the raw pixel, then run `base` (the shipped chain by default). This is
    the only thing `model/engine/groundpoint.py` can actually deliver.
    """
    off = raw_offset(coef, edges)
    base = base or shipped_map(cal)
    def fn(raw):
        raw = np.asarray(raw, float)
        return base(raw + off(raw))
    return fn


def relabel_through(cal, fn, rows, chunk=100000):
    """Our detections re-expressed so the shipped calibrator reads `fn`.

    Same device as `calib_hypothesis.relabel`: only the MAP changes, ids,
    frames, fragments and the estimator are all held fixed.
    """
    uv = np.array([[r["x"], r["y"]] for r in rows], float)
    xy = np.empty((len(uv), 2))
    for i in range(0, len(uv), chunk):
        xy[i:i + chunk] = fn(CH.band_to_raw(cal, uv[i:i + chunk]))
    px = dist_eval.to_px(cal, xy)
    return [{**r, "x": float(u), "y": float(v)} for r, (u, v) in zip(rows, px)]


# --------------------------------------------------------------------------
# our detector's boxes against the GT's, in raw panorama pixels

def load_gt_boxes(path=CH.GT_PX):
    """{clip frame: {mot_id: (x1, y1, x2, y2)}} in raw panorama pixels."""
    out = collections.defaultdict(dict)
    for line in open(path):
        p = line.split(",")
        if len(p) < 6:
            continue
        x, y, w, h = (float(v) for v in p[2:6])
        out[int(p[0])][int(p[1])] = (x, y, x + w, y + h)
    return dict(out)


def band_box_to_raw(cal, boxes: np.ndarray) -> np.ndarray:
    """(N,4) band xyxy -> (N,4) raw xyxy, by mapping the four corners.

    The band is a horizontal stretch of the raw panorama, so a small box's
    corners stay corners; the bounding box of the four mapped corners is the
    honest conversion and it is what the IoU match runs on. Only the BOTTOM row
    is used for the bias number, and the bottom edge maps to a near-horizontal
    curve of at most a fraction of a pixel of sag over a 20 px box.
    """
    c = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]],
                  boxes[:, [0, 3]], boxes[:, [2, 3]]], axis=1)   # (N,4,2)
    r = CH.band_to_raw(cal, c.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.c_[r[:, :, 0].min(1), r[:, :, 1].min(1),
                 r[:, :, 0].max(1), r[:, :, 1].max(1)]


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def box_bias(cal, jn, every=25, iou_gate=0.3, conf=0.0, match="iou"):
    """Signed (ours - GT) box-bottom row, by raw row band. §8.2's open number.

    Matching is by IoU on the WHOLE box, not by foot distance: matching on the
    quantity being measured would drive the answer to zero. `match="centre"`
    re-runs it on horizontal-centre distance alone, which cannot see the
    vertical offset at all, as the control on that choice.
    """
    gtb = load_gt_boxes()
    d = json.load(open(DETS))["frames"]
    off = jn["offset"]
    rows_v, dv, du, gh, ious = [], [], [], [], []
    n_gt = n_det = 0
    for c in sorted(gtb)[::every]:
        det = d.get(str(c + off))
        if not det:
            continue
        ours = np.array([b["xyxy"] for b in det
                         if b["cls"] == "person" and b["conf"] >= conf], float)
        if not len(ours):
            continue
        ids = sorted(gtb[c])
        G = np.array([gtb[c][i] for i in ids], float)
        O = band_box_to_raw(cal, ours)
        n_gt += len(G)
        n_det += len(O)
        if match == "iou":
            S = iou(G, O)
            order = np.argsort(-S, axis=None)
            ok = lambda s: s >= iou_gate
        else:                       # horizontal box-centre distance, in GT heights
            S = -np.abs((G[:, [0, 2]].mean(1)[:, None]
                         - O[:, [0, 2]].mean(1)[None, :])) / (G[:, 3] - G[:, 1])[:, None]
            order = np.argsort(-S, axis=None)
            ok = lambda s: s >= -iou_gate
        tg, td = set(), set()
        for i, j in zip(*np.unravel_index(order, S.shape)):
            if not ok(S[i, j]):
                break
            if i in tg or j in td:
                continue
            tg.add(int(i))
            td.add(int(j))
            rows_v.append(G[i, 3])
            dv.append(O[j, 3] - G[i, 3])
            du.append(O[j, [0, 2]].mean() - G[i, [0, 2]].mean())
            gh.append(G[i, 3] - G[i, 1])
            ious.append(float(iou(G[i:i + 1], O[j:j + 1])[0, 0]))
    return {"row": np.array(rows_v), "dv": np.array(dv), "du": np.array(du),
            "gt_h": np.array(gh), "iou": np.array(ious),
            "n_gt": n_gt, "n_det": n_det}


# --------------------------------------------------------------------------
# the decomposition

def decompose(cal, oracle_fn, uv_raw, nbins=10, label="oracle"):
    """Oracle-minus-shipped, resolved into pixel-offset models.

    Everything is reported against ONE denominator: the total squared metre
    residual between the two maps at these points. That is the quantity the
    +23% is made of, so a fraction of it is a fraction of the finding.
    """
    band = CH.raw_to_band(cal, uv_raw)
    m_ship = np.asarray(cal.to_pitch(band), float)
    r = oracle_fn(uv_raw) - m_ship
    Jh, Jg = jacobians(cal, band)
    D = raw_dirs(Jh, Jg)
    d_raw, _d_band = to_pixels(Jh, Jg, r)

    gv, ev = qbins(uv_raw[:, 1], nbins)          # image row
    gx, _ex = qbins(m_ship[:, 0], nbins)         # pitch X, along the touchline

    out = {"label": label, "n": len(r),
           "resid_median_m": float(np.median(np.linalg.norm(r, axis=1))),
           "resid_ss_m2": float((r ** 2).sum()),
           "dv_raw_median": float(np.median(d_raw[:, 1])),
           "dv_raw_iqr": [float(np.percentile(d_raw[:, 1], 25)),
                          float(np.percentile(d_raw[:, 1], 75))],
           "du_raw_median": float(np.median(d_raw[:, 0])),
           "du_raw_iqr": [float(np.percentile(d_raw[:, 0], 25)),
                          float(np.percentile(d_raw[:, 0], 75))],
           "m_per_raw_px_v_median": float(np.median(
               np.linalg.norm(D[:, :, 1], axis=1)))}

    models = {
        "const_v": fit_offset(r, D, None, (1,)),
        "const_uv": fit_offset(r, D, None, (0, 1)),
        "perrow_v": fit_offset(r, D, gv, (1,)),
        "perrow_uv": fit_offset(r, D, gv, (0, 1)),
        "perX_v": fit_offset(r, D, gx, (1,)),
    }
    out["r2"] = {k: v[2] for k, v in models.items()}
    out["const_v_px"] = float(models["const_v"][0][0, 0])
    out["const_uv_px"] = [float(models["const_uv"][0][0, 0]),
                          float(models["const_uv"][0][0, 1])]
    out["perrow_v_px"] = models["perrow_v"][0][:, 0].tolist()
    out["row_edges"] = ev.tolist()
    out["perX_v_px"] = models["perX_v"][0][:, 0].tolist()

    # FLATNESS IS THE DISCRIMINATOR, not R^2. A ground-point error is a fixed
    # vertical pixel offset, so re-fitting it independently in each row band and
    # in each pitch-X band must return THE SAME NUMBER every time. R^2 alone is
    # too generous: the planted controls show a constant offset absorbing 65% of
    # a planted homography error, because both mostly point across the pitch.
    for k, key in (("perrow_v", "row"), ("perX_v", "X")):
        c = models[k][0][:, 0]
        out[f"flat_{key}_span_px"] = float(c.max() - c.min())
        out[f"flat_{key}_std_px"] = float(c.std())

    # ALONG-X STRUCTURE AT FIXED ROW: the one axis on which a homography error
    # cannot hide behind a row effect. Residual left by the per-row model, then
    # binned by pitch X - if the per-row model were the whole story this is
    # noise with no trend.
    res = r - models["perrow_v"][1]
    out["perrow_resid_by_X"] = [
        float(np.median(np.linalg.norm(res[gx == k], axis=1)))
        for k in range(nbins)]
    _c, _f, r2x = fit_offset(res, D, gx, (0, 1))
    out["r2_X_on_perrow_resid"] = r2x
    out["_d_raw"], out["_r"], out["_D"] = d_raw, r, D
    out["_row"], out["_gv"], out["_ev"] = uv_raw[:, 1], gv, ev
    out["_m_ship"] = m_ship
    return out


def report_decomp(o):
    log(f"  n = {o['n']} held-out GT feet")
    log(f"  oracle - shipped: median {o['resid_median_m']:.3f} m, "
        f"total {o['resid_ss_m2']:.0f} m^2")
    log(f"  as an equivalent RAW-PIXEL displacement: "
        f"dv {o['dv_raw_median']:+.2f} px (IQR {o['dv_raw_iqr'][0]:+.2f} "
        f"{o['dv_raw_iqr'][1]:+.2f}), du {o['du_raw_median']:+.2f} px "
        f"(IQR {o['du_raw_iqr'][0]:+.2f} {o['du_raw_iqr'][1]:+.2f})")
    log(f"  1 raw pixel of v is {o['m_per_raw_px_v_median']:.3f} m of pitch here")
    log(f"  {'model':<12} {'params':>7} {'R^2 on the metre residual':>26}")
    for k in ("const_v", "const_uv", "perrow_v", "perrow_uv", "perX_v"):
        p = {"const_v": 1, "const_uv": 2, "perrow_v": 10,
             "perrow_uv": 20, "perX_v": 10}[k]
        log(f"  {k:<12} {p:>7} {o['r2'][k]:>26.3f}")
    log(f"  best constant vertical offset: {o['const_v_px']:+.2f} raw px")
    log(f"  FLATNESS (a real foot offset refits to the SAME px in every band):")
    log(f"    by row   span {o['flat_row_span_px']:.2f} px, "
        f"sd {o['flat_row_std_px']:.2f}")
    log(f"    by pitch X  span {o['flat_X_span_px']:.2f} px, "
        f"sd {o['flat_X_std_px']:.2f}")
    log(f"  per-row offsets (px), far -> near: " +
        " ".join(f"{c:+.1f}" for c in o["perrow_v_px"]))
    log(f"  after the per-row model, residual still left, by pitch X (m): " +
        " ".join(f"{v:.2f}" for v in o["perrow_resid_by_X"]))
    log(f"  and a per-X pixel offset explains {o['r2_X_on_perrow_resid']:.3f} "
        f"of what the per-row model left -> that part cannot be a row effect")


# --------------------------------------------------------------------------
# planted controls: what this harness can and cannot come back with

def controls(cal, uv_raw, nbins=10):
    """Four synthetic 'oracles' with known composition. The harness must call
    A a ground point, C a homography, and it must FAIL to tell D apart."""
    ship = shipped_map(cal)

    def A(raw):                       # pure ground point: 3 raw px of foot lift
        return ship(raw + np.array([0.0, 3.0]))

    def B(raw):                       # pure homography: rigid 2 m along X
        return ship(raw) + np.array([2.0, 0.0])

    def C(raw):                       # homography error that varies ALONG X
        m = ship(raw)
        return m + np.stack([np.zeros(len(m)), 0.03 * m[:, 0]], 1)

    def D(raw):                       # homography error that varies with Y ONLY
        m = ship(raw)                 # -- the ambiguous case, by construction
        return m + np.stack([np.zeros(len(m)), 0.05 * m[:, 1]], 1)

    return {k: decompose(cal, f, uv_raw, nbins, label=k)
            for k, f in (("A_groundpoint_3px", A), ("B_homography_shiftX", B),
                         ("C_homography_varies_in_X", C),
                         ("D_homography_varies_in_Y", D))}


# --------------------------------------------------------------------------

def design(cal, jn):
    """State what result was possible BEFORE reporting one. postmortem.md C2."""
    te_u, te_x, _f, _i = heldout(jn)
    band = CH.raw_to_band(cal, te_u)
    m = np.asarray(cal.to_pitch(band), float)
    log("IDENTIFIABILITY, measured on this footage, not assumed")
    for name, a, b in (("image row v  vs  pitch Y (across)", te_u[:, 1], m[:, 1]),
                       ("image row v  vs  pitch X (along)", te_u[:, 1], m[:, 0]),
                       ("image col u  vs  pitch X (along)", te_u[:, 0], m[:, 0])):
        log(f"  {name:<36} r = {np.corrcoef(a, b)[0, 1]:+.3f}")
    log(f"  raw rows spanned by GT feet: {te_u[:, 1].min():.0f} - "
        f"{te_u[:, 1].max():.0f}  ({te_u[:, 1].max() - te_u[:, 1].min():.0f} px "
        f"for {m[:, 1].max() - m[:, 1].min():.0f} m of pitch depth)")
    g, _e = qbins(te_u[:, 1], 10)
    sp = [float(np.ptp(m[g == k, 0])) for k in range(10)]
    log(f"  pitch X spanned WITHIN each row decile (m): " +
        " ".join(f"{v:.0f}" for v in sp))
    log("  => row and cross-pitch position are ONE variable (r above). A "
        "regression of the\n     residual on row against a regression on "
        "position cannot come out two ways.")
    log("  => but every row decile still spans most of the pitch LENGTH, so "
        "structure in X\n     at fixed row is free of any row effect. That, and "
        "the pixel reparameterisation,\n     are the test.\n")

    log("PLANTED CONTROLS: can the decomposition come out both ways?")
    cs = controls(cal, te_u)
    log(f"  {'control':<28} {'R2 const_v':>10} {'dv px':>7} "
        f"{'row span':>9} {'X span':>8} {'X left over':>12}")
    for k, o in cs.items():
        log(f"  {k:<28} {o['r2']['const_v']:>10.3f} {o['const_v_px']:>+7.2f} "
            f"{o['flat_row_span_px']:>9.2f} {o['flat_X_span_px']:>8.2f} "
            f"{o['r2_X_on_perrow_resid']:>12.3f}")
    log("  READ IT ON FLATNESS, NOT R^2. Control C shows a constant foot offset "
        "absorbing\n  0.65 of a planted HOMOGRAPHY error, because both point "
        "mostly across the pitch.\n  What only a real foot offset does is refit "
        "to the same pixel value everywhere:\n  A's per-row and per-X spans are "
        "~0.05 px, every homography control's are metres\n  of pixels.")
    log("  WHAT STAYS UNIDENTIFIABLE: control D is a homography error depending "
        "only on\n  cross-pitch distance. It is separable from a CONSTANT foot "
        "offset (its per-row\n  coefficients run 0.4 -> 31 px) but NOT from a "
        "ROW-VARYING one - which is exactly\n  the fix "
        "docs/research/research-farfield-engineering.md §2.5 proposes. So the constant "
        "term\n  and the along-X term are clean; a per-row correction is not "
        "attributable.")
    return cs


def selfcheck():
    cal = calibrators.load(os.path.join(JOB, "calibration.json"))
    jn = load_join()
    te_u, te_x, _f, _i = heldout(jn)

    # 1. THE HELD-OUT SET IS THE ORACLE'S OWN. `build_oracle` reported 13,200
    #    test points; if this split rule has drifted the decomposition is being
    #    run on points the oracle was fitted on and its residual is meaningless.
    assert len(te_u) == 13200, len(te_u)

    # 2. THE JACOBIAN IS THE MAP'S. A first-order model of a 1 px step must
    #    reproduce the map's own answer; if it does not, every pixel number
    #    below is arithmetic on the wrong linearisation.
    band = CH.raw_to_band(cal, te_u[:2000])
    Jh, Jg = jacobians(cal, band)
    step = np.tile([0.4, 0.7], (len(band), 1))
    lin = (Jh @ step[:, :, None])[:, :, 0]
    tru = (np.asarray(cal.to_pitch(band + step), float)
           - np.asarray(cal.to_pitch(band), float))
    # 1 cm, against a GT quantised at 0.30 m and a residual measured in metres.
    # What is left is the homography's own second-order term over a sub-pixel
    # step; it is 0.6% of the step's displacement at the far side.
    assert np.abs(lin - tru).max() < 1e-2, np.abs(lin - tru).max()

    # 3. THE PIXEL REPARAMETERISATION INVERTS. A known raw-pixel shift, pushed
    #    to metres and back, must return itself. This is the whole method.
    shift = np.array([0.0, 2.5])
    r = shipped_map(cal)(te_u[:2000] + shift) - np.asarray(cal.to_pitch(band), float)
    d_raw, _ = to_pixels(Jh, Jg, r)
    assert np.abs(np.median(d_raw[:, 1]) - 2.5) < 0.05, np.median(d_raw[:, 1])
    assert abs(np.median(d_raw[:, 0])) < 0.05, np.median(d_raw[:, 0])

    # 4. THE CONTROLS. A planted ground-point error must be attributed to the
    #    ground point and a planted along-X homography error must NOT be - and
    #    both have to be reachable, or the run below is a null read.
    cs = controls(cal, te_u[:4000])
    a, b, c, d = (cs["A_groundpoint_3px"], cs["B_homography_shiftX"],
                  cs["C_homography_varies_in_X"], cs["D_homography_varies_in_Y"])
    assert a["r2"]["const_v"] > 0.98, a["r2"]
    assert abs(a["const_v_px"] - 3.0) < 0.1, a["const_v_px"]
    assert a["flat_row_span_px"] < 0.3 and a["flat_X_span_px"] < 0.3, a
    assert b["r2"]["const_v"] < 0.2, b["r2"]
    #    Flatness, not R^2: a constant offset absorbs 0.65 of control C on R^2
    #    alone, so R^2 is not allowed to be the discriminator anywhere.
    assert c["flat_row_span_px"] > 2.0 and c["flat_X_span_px"] > 2.0, c
    assert c["r2_X_on_perrow_resid"] > 0.5, c["r2_X_on_perrow_resid"]
    # 5. AND THE LIMIT IS REAL, not a claim. D is a homography error that
    #    depends only on cross-pitch distance. It must be separable from a
    #    CONSTANT foot offset and NOT from a row-varying one. If a future edit
    #    makes D look separable, the decomposition has started to believe
    #    something it cannot see and this assert is the thing that says so.
    assert d["flat_row_span_px"] > 5.0, d
    assert d["r2"]["perrow_v"] - d["r2"]["const_v"] > 0.4, d["r2"]

    # 6. BOX MATCHING. Our band boxes must land inside the raw panorama and
    #    overlap the GT boxes at all - a space error here reads as a huge bias.
    bb = box_bias(cal, jn, every=500)
    assert len(bb["dv"]) > 100, len(bb["dv"])
    assert bb["iou"].mean() > 0.4, bb["iou"].mean()

    log(f"selfcheck ok  (held-out set is the oracle's 13,200; Jacobian "
        f"linearises to <1e-2 m; a planted 2.5 px shift reads back "
        f"{np.median(d_raw[:, 1]):.3f} px; control A refits flat to "
        f"{a['flat_row_span_px']:.2f} px span, control C to "
        f"{c['flat_row_span_px']:.1f} px so it is NOT a foot offset, control D "
        f"stays unidentifiable against a per-row model "
        f"{d['r2']['perrow_v']:.2f} vs {d['r2']['const_v']:.2f}; "
        f"{len(bb['dv'])} box matches at mean IoU {bb['iou'].mean():.2f})")


def run_boxbias(cal, jn, out=None):
    log("\nTHE THIRD TERM: our detector's boxes against the GT's, raw pixels")
    log("  (the oracle never saw this - it is fitted on GT feet, so a bias "
        "between\n   OUR box bottom and the GT's survives every map change)")
    res = {}
    for match, mgate in (("iou", 0.3), ("iou", 0.5), ("centre", 0.25)):
        bb = box_bias(cal, jn, every=25, match=match, iou_gate=mgate)
        band = CH.raw_to_band(cal, np.c_[np.full(len(bb["row"]), 2048.0),
                                         bb["row"]])
        # metres per raw vertical pixel, at each matched foot
        Jh, Jg = jacobians(cal, band)
        mpp = np.linalg.norm(raw_dirs(Jh, Jg)[:, :, 1], axis=1)
        k = f"{match}@{mgate}"
        res[k] = {"n": int(len(bb["dv"])), "n_gt": bb["n_gt"],
                  "n_det": bb["n_det"],
                  "dv_median": float(np.median(bb["dv"])),
                  "dv_mean": float(bb["dv"].mean()),
                  "dv_iqr": [float(np.percentile(bb["dv"], 25)),
                             float(np.percentile(bb["dv"], 75))],
                  "dv_sem": float(bb["dv"].std(ddof=1) / np.sqrt(len(bb["dv"]))),
                  "du_median": float(np.median(bb["du"])),
                  "m_median": float(np.median(bb["dv"] * mpp))}
        log(f"\n  match {k}: {len(bb['dv'])} matches of {bb['n_gt']} GT boxes "
            f"({100 * len(bb['dv']) / bb['n_gt']:.1f}%), {bb['n_det']} of ours")
        log(f"    signed dv (ours - GT), median {np.median(bb['dv']):+.3f} raw px "
            f"(IQR {np.percentile(bb['dv'], 25):+.2f} "
            f"{np.percentile(bb['dv'], 75):+.2f}), mean {bb['dv'].mean():+.3f} "
            f"+- {bb['dv'].std(ddof=1) / np.sqrt(len(bb['dv'])):.3f}; "
            f"du median {np.median(bb['du']):+.3f} px")
        log(f"    in metres of pitch: median {np.median(bb['dv'] * mpp):+.3f} m")
        if match == "iou" and mgate == 0.3:
            g, e = qbins(bb["row"], 7)
            log(f"    {'raw row band':>16} {'n':>7} {'GT box h':>9} "
                f"{'dv px':>8} {'dv m':>8} {'|dv| px':>8}")
            rows = []
            for i in range(7):
                m = g == i
                if m.sum() < 20:
                    continue
                rows.append({"lo": float(e[i]), "hi": float(e[i + 1]),
                             "n": int(m.sum()),
                             "gt_h": float(np.median(bb["gt_h"][m])),
                             "dv_px": float(np.median(bb["dv"][m])),
                             "dv_m": float(np.median((bb["dv"] * mpp)[m])),
                             "abs_dv_px": float(np.median(np.abs(bb["dv"][m])))})
                log(f"    {e[i]:7.0f}-{e[i + 1]:<8.0f} {m.sum():7d} "
                    f"{np.median(bb['gt_h'][m]):9.1f} "
                    f"{np.median(bb['dv'][m]):+8.2f} "
                    f"{np.median((bb['dv'] * mpp)[m]):+8.3f} "
                    f"{np.median(np.abs(bb['dv'][m])):8.2f}")
            res[k]["by_row"] = rows
    if out is not None:
        out["box_bias"] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--boxbias", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tracks", default="tracks.split.json")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--bins", type=int, default=10)
    # 10000 is the oracle §A reports (0.549 / 1.186 / 96.2% held out) and the
    # one `calib_hypothesis.selfcheck` asserts. It is not a taste: it is the
    # only setting on the sweep that is DISTANCE-NEUTRAL. At smoothing 1.0 the
    # same map reads 1.098 on GT's own feet - a tenth of the +23% would be the
    # interpolator's own wiggle, and the ladder below would be measuring it.
    ap.add_argument("--smoothing", type=float, default=10000.0)
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck()

    cal = calibrators.load(os.path.join(JOB, "calibration.json"))
    jn = load_join()
    log(f"JOIN (cached): clip frame 0 = gz frame {jn['offset']}, "
        f"perm is identity? {jn['identity_perm']}, cost {jn['cost_m']:.2f} m")

    out = {"join": jn}
    if a.design or a.run:
        log("\n=== DESIGN: what can this test come back with? ===")
        cs = design(cal, jn)
        out["controls"] = {k: {kk: vv for kk, vv in v.items()
                               if not kk.startswith("_")} for k, v in cs.items()}
    if a.boxbias and not a.run:
        run_boxbias(cal, jn, out)
        json.dump(out, open(os.path.join(OUT, "calib_split.json"), "w"), indent=1)
        return
    if not a.run:
        return

    log("\n=== THE ORACLE, refitted exactly as calib_hypothesis fits it ===")
    orc, ostat = CH.build_oracle(cal, jn, smoothing=a.smoothing)
    neu = CH.oracle_distance_control(cal, orc, jn, every=10)
    ostat.update(neu)
    log(f"    DISTANCE NEUTRALITY {neu['distance_neutrality']:.3f} "
        f"(GT's own feet through the oracle vs the GT metres)")
    # A map that adds path length of its own lands directly on `ratio_raw`, and
    # the ladder below is a ladder in `ratio_raw`. Refuse to build it on a map
    # that is not neutral - `calib_hypothesis.selfcheck` uses the same window.
    assert 0.95 < neu["distance_neutrality"] < 1.05, neu
    out["oracle"] = ostat
    te_u, te_x, _f, _i = heldout(jn)
    assert len(te_u) == ostat["n_test"], (len(te_u), ostat["n_test"])

    log("\n=== 1. THE RESIDUAL DECOMPOSITION (oracle minus shipped) ===")
    o = decompose(cal, lambda p: orc(p), te_u, a.bins, "oracle")
    report_decomp(o)
    out["decomp"] = {k: v for k, v in o.items() if not k.startswith("_")}

    # Sanity: how far is each model from the GT itself, not just from the
    # oracle? A model can explain the map difference and still be no better on
    # the ground, if the oracle is wrong where the model is right.
    log("\n  ACCURACY AGAINST GT, same held-out points (median / p90 / <2 m):")
    Jh, Jg = jacobians(cal, CH.raw_to_band(cal, te_u))
    D = raw_dirs(Jh, Jg)
    m_ship = o["_m_ship"]
    r = o["_r"]
    maps = {"shipped": m_ship,
            "shipped + const dv": m_ship + fit_offset(r, D, None, (1,))[1],
            "shipped + per-row dv": m_ship + fit_offset(r, D, o["_gv"], (1,))[1],
            "oracle": m_ship + r}
    acc = {}
    for k, v in maps.items():
        e = np.linalg.norm(v - te_x, axis=1)
        acc[k] = {"median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)),
                  "within_2m": float((e < 2).mean())}
        log(f"    {k:<22} {np.median(e):6.3f}  {np.percentile(e, 90):6.3f}  "
            f"{100 * (e < 2).mean():5.1f}%")
    out["accuracy"] = acc

    bb = run_boxbias(cal, jn, out)

    log("\n=== 3. THE SAME SPLIT IN THE CURRENCY OF THE FINDING: ratio_raw ===")
    lo, hi = jn["offset"], jn["offset"] + 6000
    rows, fps = CH.load_rows(a.tracks, lo, hi)
    _g, gtd = dist_eval.load_gt()
    gtw = CH.window_gt(gtd, lo, hi)
    gtf = contam_eval.load_gt()
    log(f"  {a.tracks}, frames {lo}-{hi}, {len(rows)} detections, "
        f"{len({r['id'] for r in rows})} ids, {len(gtw)} GT players")

    cv = np.atleast_2d(o["const_v_px"])
    pr = np.array(o["perrow_v_px"])[:, None]
    ev = np.array(o["row_edges"][1:-1])
    # The measured box-bottom bias, as a correction: our foot sits dv px lower
    # than the GT's, so subtract it. This is the THIRD term - the one the oracle
    # never saw - and these two rungs are what it is worth in ratio_raw.
    br = bb["iou@0.3"]["by_row"]
    bc = np.array([[-r["dv_px"]] for r in br])
    be = np.array([r["hi"] for r in br[:-1]])
    orcf = lambda p: orc(p)
    variants = {
        "shipped": None,
        "ground point: const dv": shifted_map(cal, cv),
        "ground point: per-row dv": shifted_map(cal, pr, ev),
        "shipped - box bias": shifted_map(cal, bc, be),
        "oracle (both)": orcf,
        "oracle - box bias": shifted_map(cal, bc, be, base=orcf),
    }
    log(f"\n  {'map':<26} {'coverage':>9} {'ratio_raw':>10} {'ours km':>9}")
    lad = {}
    base = None
    for k, fn in variants.items():
        t0 = time.time()
        rr = rows if fn is None else relabel_through(cal, fn, rows)
        s = CH.score(rr, fps, cal, cal, gtw, gtf, a.stride, a.gate)
        lad[k] = {kk: vv for kk, vv in s.items() if kk != "players"}
        if base is None:
            base = s["ratio_raw"]
        log(f"  {k:<26} {s['coverage']:9.3f} {s['ratio_raw']:10.3f} "
            f"{s['ours_km']:9.2f}   ({time.time() - t0:.0f} s)")
    # the share column, now that the oracle end point is known
    span = lad["oracle (both)"]["ratio_raw"] - base
    log(f"\n  share of the shipped->oracle gap in ratio_raw ({base:.3f} -> "
        f"{lad['oracle (both)']['ratio_raw']:.3f}):")
    for k, v in lad.items():
        log(f"    {k:<26} {(v['ratio_raw'] - base) / span:6.1%}")
    out["ladder"] = lad
    out["ladder_share"] = {k: (v["ratio_raw"] - base) / span for k, v in lad.items()}

    p = os.path.join(OUT, "calib_split.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"\nwrote {p}")


if __name__ == "__main__":
    main()
