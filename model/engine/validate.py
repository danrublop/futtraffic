"""Accuracy validation harness: pipeline numbers vs a trusted reference.

The whole phase-1 deliverable is one sentence — "our distance and speed numbers
land within X% of a trusted reference" — so this file is the thing that
produces it. Three rules are baked into the code rather than left to
discipline, because all three are easy to violate by accident and none of the
violations look like a bug:

1. **Metrics are reported separately.** Total distance averages over ~135k
   frames and tolerates noise. Peak speed is an extreme-value statistic on the
   same jitter. Reporting them as one "distance and speed" figure hides an
   order of magnitude of difference in achievable accuracy.
2. **Headline figures come from held-out matches only.** Tuning smoothing or
   thresholds on the same matches you report on produces a fitted number, not a
   measured one, and it will not survive a skeptical reader. Anything computed
   on the tuning split is labelled FITTED and is structurally barred from being
   the headline.
3. **Agreement, not a bare percentage.** The reference device has its own error
   (GPS is ~1-2% on total distance and considerably worse on instantaneous peak
   speed at 10Hz). A claim of "within 3%" against an instrument with 2% error is
   a claim about the pair, so we report Bland-Altman limits of agreement and
   state the reference error next to every number.

Ground-truth sources, in order of how much a reader should trust them:

- **GPS vests** (Catapult / STATSports export) — the reference the published
  claim needs. Slot in via ``--gt metrics.csv``; nothing else changes.
- **SoccerTrack v2 GSR** (CC BY 4.0, per-frame pitch coordinates in metres) —
  free, available today, and good enough for a first internal error figure.
  Two caveats stated in every report it produces: it is 4K panoramic footage
  against our 1080p perspective product camera, and because both sides run
  through the same metric estimator, the estimator's own bias cancels. That
  makes SoccerTrack a lower bound on error, not the published claim.
"""
from dataclasses import dataclass, field
import argparse
import gzip
import json
import math
import os
import random

import numpy as np

import stats
from pitch import Calibration, Pitch

# name -> (target MAE %, gates phase 1). From the design doc's Accuracy Targets
# table. Kept here rather than in a config file because changing a target is a
# decision someone should have to justify in a diff.
METRICS: dict[str, tuple[float, bool]] = {
    "distance_m": (5.0, True),
    "minutes": (5.0, True),
    "hsr_m": (15.0, False),
    "sprints": (20.0, False),
    "peak_speed_ms": (10.0, False),
}
IDSW_TARGET = 2.0  # switches per player per match, gating
MIN_PLAYER_MATCHES = 60  # doc: no figure is published below this
IDSW_RADIUS_M = 2.0  # association gate for the ID-switch count
IDSW_HOLD = 12  # ~0.5s at 25fps; a swap must stick this long to count as a switch

# High-speed running shares stats.SPRINT_SPEED deliberately. Two thresholds for
# "fast" in one product is how HSR and sprint counts end up disagreeing about
# the same run.
HSR_SPEED = stats.SPRINT_SPEED

# Matches build_match's own floor so the two sides count the same players.
MIN_MINUTES = stats.MIN_MINUTES   # imported, never re-declared

METRE_CAL = Calibration(pitch=Pitch(105.0, 68.0), H=np.eye(3))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

# Was a local copy of stats.player_stats' first half. Deleted: switching the
# default estimator from boxcar to Kalman left the copy behind, so HSR came off a
# different velocity series than distance for the same track. stats.speed_series
# is now the single source and the assert in _demo() pins them together.
_speed_series = stats.speed_series


def player_metrics(track: list[dict], cal: Calibration, fps: float,
                   method: str = "kalman") -> dict | None:
    """One track -> the five per-player-match numbers under test."""
    ps = stats.player_stats(track, cal, fps, method)
    if ps is None or ps.minutes < MIN_MINUTES:
        return None
    v, dt, real, _xy = _speed_series(track, cal, fps, method)
    fast = real & (v >= HSR_SPEED)
    return {
        "distance_m": ps.distance_m,
        "minutes": ps.minutes,
        "hsr_m": float(np.sum(v[fast] * dt[fast])),
        "sprints": float(ps.sprints),
        "peak_speed_ms": ps.top_speed_ms,
        "team": ps.team,
        "jersey": ps.jersey,
    }


def match_metrics(tracks: dict, cal: Calibration, fps: float,
                  method: str = "kalman") -> dict[str, dict]:
    out = {}
    for tid, tr in tracks.items():
        m = player_metrics(tr, cal, fps, method)
        if m is not None:
            out[str(tid)] = m
    return out


def id_switches(gt: dict, meas: dict, radius: float = IDSW_RADIUS_M,
                cal: Calibration = METRE_CAL, hold_frames: int = IDSW_HOLD) -> dict[str, int]:
    """MOT-style IDSW: how often a GT player's identity changes hands.

    Gating metric, and the only one that cannot be computed from aggregates —
    it needs per-frame positions on both sides. Optimal assignment inside
    `radius` per frame; a GT player whose matched measured track id changes and
    *stays* changed for `hold_frames` scores one switch.
    """
    from scipy.optimize import linear_sum_assignment  # already a numpy-stack dep

    gk, gw, gxy, gspan = _frame_index(gt, cal)
    _, mw, mxy, mspan = _frame_index(meas, cal)
    held: dict[int, int] = {}          # GT player -> currently committed measured id
    pend: dict[int, list] = {}         # GT player -> [candidate id, frames held]
    switches = {k: 0 for k in gk}
    for frame in sorted(set(gspan) & set(mspan)):
        gs, ge = gspan[frame]
        ms, me = mspan[frame]
        d = np.linalg.norm(gxy[gs:ge, None, :] - mxy[None, ms:me, :], axis=2)
        # Globally optimal rather than greedy: with 22 players inside a few
        # metres of each other at a corner, greedy nearest-neighbour invents
        # switches that never happened.
        for gi, mi in zip(*linear_sum_assignment(np.where(d > radius, 1e6, d))):
            if d[gi, mi] > radius:
                continue
            key, mid = int(gw[gs + gi]), int(mw[ms + mi])
            if held.get(key) == mid or key not in held:
                held[key] = mid
                pend.pop(key, None)
                continue
            # Hysteresis. A one-frame flicker that reverts is not a switch an
            # operator would ever see or correct; counting it makes this metric
            # measure association noise instead of tracking quality.
            p = pend.get(key)
            pend[key] = [mid, p[1] + 1] if p and p[0] == mid else [mid, 1]
            if pend[key][1] >= hold_frames:
                switches[gk[key]] += 1
                held[key] = mid
                pend.pop(key)
    return switches


def _frame_index(tracks: dict, cal: Calibration):
    """(track keys, per-detection key index, pitch xy, frame -> [start, end)).

    Everything is projected in one call and grouped by a sorted frame column.
    Doing this per frame instead costs ~67k cv2 round-trips per half, which is
    the difference between seconds and an hour.
    """
    keys = list(map(str, tracks))
    f = np.fromiter((p["f"] for tr in tracks.values() for p in tr), np.int64)
    who = np.fromiter((i for i, tr in enumerate(tracks.values()) for _ in tr), np.int32)
    xy = np.array([[p["x"], p["y"]] for tr in tracks.values() for p in tr], dtype=np.float64)
    if not len(f):
        return keys, who, xy, {}
    order = np.argsort(f, kind="stable")
    f, who, xy = f[order], who[order], cal.to_pitch(xy[order])
    cuts = np.flatnonzero(np.diff(f)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(f)]))
    span = {int(f[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
    return keys, who, xy, span


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

# One definition, in stats, because validate imports stats and not the reverse.
# build_match needs it too: every ground-truth half in model/validation/ is gzipped.
from stats import _open  # noqa: E402 - grouped with the loaders it serves


def _stream_array(path: str, key: str):
    """Yield objects from a top-level array without loading the file.

    A SoccerTrack GSR half is ~2.7GB of pretty-printed JSON; json.load on one
    needs tens of GB of RAM. This is the reason a streaming reader exists here
    at all rather than a one-line json.load.
    """
    dec = json.JSONDecoder()
    with _open(path) as f:
        buf, needle = "", f'"{key}"'
        while needle not in buf:
            chunk = f.read(1 << 20)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index(needle):]
        # Position, not slicing. Re-slicing the buffer after every record makes
        # this quadratic, which on a 2.7GB half is the difference between 40
        # seconds and never finishing.
        i = buf.index("[") + 1
        while True:
            while i < len(buf) and buf[i] in " \n\r\t,":
                i += 1
            if i < len(buf) and buf[i] == "]":
                return
            try:
                obj, i = dec.raw_decode(buf, i)
            except ValueError:
                chunk = f.read(1 << 20)
                if not chunk:
                    return
                buf = buf[i:] + chunk  # compact only when we actually need more
                i = 0
                continue
            yield obj
            if i > (1 << 22):
                buf, i = buf[i:], 0


def load_soccertrack(path: str) -> tuple[dict, float]:
    """SoccerTrack v2 / SoccerNet GSR half -> tracks in tracks.json shape.

    GT pitch coordinates have their origin at the centre circle; pitch.py puts
    it at the bottom-left corner. Shifting here means one convention downstream.
    Referees and "other" are dropped: a referee runs further than most players,
    so scoring one as a player poisons the distance figure at both ends.
    """
    tracks: dict[str, list[dict]] = {}
    for a in _stream_array(path, "annotations"):
        attrs = a.get("attributes") or {}
        if attrs.get("role") not in ("player", "goalkeeper"):
            continue
        bp = a.get("bbox_pitch")
        if not bp or bp.get("x_bottom_middle") is None:
            continue
        tid = str(a["track_id"])
        jersey = attrs.get("jersey")
        tracks.setdefault(tid, []).append({
            "f": int(a["image_id"]),
            "id": tid,
            "team": {"left": "home", "right": "away"}.get(attrs.get("team"), "unknown"),
            "jersey": int(jersey) if jersey not in (None, "") else None,
            "x": float(bp["x_bottom_middle"]) + 52.5,
            "y": float(bp["y_bottom_middle"]) + 34.0,
        })
    return tracks, 25.0  # GSR annotation rate is the source video rate


def load_tracks(path: str) -> tuple[dict, float]:
    with _open(path) as f:
        data = json.load(f)
    tracks: dict[str, list[dict]] = {}
    for d in data["detections"]:
        tracks.setdefault(str(d["id"]), []).append(d)
    return tracks, float(data["fps"])


def load_metrics_csv(path: str) -> dict[str, dict[str, dict]]:
    """Generic aggregate ground truth. This is the GPS-vest slot.

    Columns: match_id, player_id, then any of distance_m, minutes, hsr_m,
    sprints, peak_speed_ms. Missing columns are simply not compared, so a vest
    export that reports four of the five needs no code change.
    """
    import csv
    out: dict[str, dict[str, dict]] = {}
    with _open(path) as f:
        for row in csv.DictReader(f):
            rec = {k: float(row[k]) for k in METRICS if row.get(k) not in (None, "")}
            out.setdefault(row["match_id"], {})[row["player_id"]] = rec
    return out


def load_match_json(path: str) -> dict[str, dict]:
    """stats.build_match output -> per-player aggregates, as the product reports them."""
    with _open(path) as f:
        m = json.load(f)
    return {str(p["id"]): {"distance_m": p["distance_km"] * 1000.0,
                           "minutes": p["minutes"],
                           "sprints": float(p["sprints"]),
                           # None where stats.py refused it: MAX_SPEED bound on
                           # that track, so the published number was the clamp.
                           # nan, not 0.0 - a refused peak is missing, and 0.0
                           # would score as a player who never moved.
                           "peak_speed_ms": (p["top_speed_kmh"] / 3.6
                                             if p["top_speed_kmh"] is not None
                                             else float("nan"))}
            for p in m["players"]}


def load_any(path: str, cal: Calibration, method: str = "kalman") -> tuple[dict[str, dict], dict | None, float]:
    """(aggregate metrics, per-frame tracks or None, fps). Sniffs the format."""
    if path.endswith((".csv", ".csv.gz")):
        raise ValueError("CSV holds many matches; use load_metrics_csv directly")
    with _open(path) as f:
        head = f.read(4096)
    if '"annotations"' in head or '"im_dir"' in head:
        tracks, fps = load_soccertrack(path)
    elif '"detections"' in head:
        tracks, fps = load_tracks(path)
    elif '"players"' in head:
        return load_match_json(path), None, 0.0
    else:
        raise ValueError(f"unrecognised ground-truth/measurement format: {path}")
    return match_metrics(tracks, cal, fps, method), tracks, fps


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

@dataclass
class Result:
    metric: str
    target: float
    gates: bool
    n: int
    mae_pct: float
    bias_pct: float
    loa_lower: float
    loa_upper: float
    verdict: str
    skipped_zero_gt: int = 0
    fitted: bool = False
    skipped_refused: int = 0   # measured side was nan - a refusal, not a reading
    # The four statistics sports science actually expects. Without these a
    # reviewer or a club's performance staff will not accept the claim, and
    # none of them exist as a named function in any R or Python package - the
    # field's real toolchain is Hopkins' spreadsheets plus hand-rolled stats.
    tee_pct: float = float("nan")      # typical error of estimate, as CV%
    ccc: float = float("nan")          # Lin's concordance correlation
    std_bias: float = float("nan")     # standardised bias, Hopkins scale
    calib_slope: float = float("nan")  # so a user can correct our output
    calib_intercept: float = float("nan")
    tee_rating: str = "n/a"            # Hoppe et al.: good <5%, moderate <10%


def band(mae_pct: float) -> str:
    """The doc's three bands. The middle one exists because the previous plan
    had a 5-10% no-man's-land where the target was missed but no action fired."""
    if mae_pct <= 5.0:
        return "PASS"
    if mae_pct <= 10.0:
        return "MARGINAL"
    return "FAIL"


def bland_altman(gt: np.ndarray, meas: np.ndarray) -> tuple[float, float, float]:
    """Percentage bias and 95% limits of agreement.

    Reported instead of a bare MAE because MAE folds sign away: a pipeline that
    is 4% high on every player and one that is randomly +-4% both score 4%, and
    only the first is fixable with a calibration constant.
    """
    mean = (gt + meas) / 2.0
    diff = np.where(mean != 0, (meas - gt) / np.where(mean == 0, 1, mean) * 100.0, 0.0)
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def agreement_stats(g: np.ndarray, m: np.ndarray) -> dict:
    """TEE, Lin's CCC, standardised bias, and the calibration equation.

    This is the reporting format the optical-vs-GPS validation literature uses
    (Hoppe 2018, Linke 2018/2020). TEE is the headline: the residual SD from
    regressing criterion on practical measure, expressed as a CV. Ratings are
    Hoppe et al.'s scale - good <5%, moderate 5-10%, poor >=10%.

    Note FIFA's EPTS programme certifies instantaneous position and velocity
    only, NOT total distance, HSR or sprint counts, and publishes no numeric
    thresholds at all. There is no external bar here; this scale is the field's
    convention, not a certification.
    """
    n = len(g)
    # A degenerate spread makes the regression meaningless, not just noisy:
    # if every measured value is identical there is no slope to fit, and polyfit
    # returns a poorly-conditioned answer with a warning rather than refusing.
    if n < 3 or np.ptp(m) < 1e-9 or np.ptp(g) < 1e-9:
        return {}
    # Calibration: regress criterion on measured, so a user can correct us.
    slope, intercept = np.polyfit(m, g, 1)
    resid = g - (slope * m + intercept)
    # ddof=2 - two parameters were fitted.
    see = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
    tee_pct = float(see / np.mean(g) * 100.0) if np.mean(g) else float("nan")

    # Lin's CCC: correlation AND agreement. Pearson r cannot see a constant
    # offset, so a pipeline reading 10% high scores r=1.0 and is still wrong.
    vg, vm = g.var(ddof=0), m.var(ddof=0)
    cov = float(((g - g.mean()) * (m - m.mean())).mean())
    denom = vg + vm + (g.mean() - m.mean()) ** 2
    ccc = float(2 * cov / denom) if denom else float("nan")

    diff = m - g
    pooled = float(np.sqrt((vg + vm) / 2.0))
    std_bias = float(diff.mean() / pooled) if pooled else float("nan")

    rating = "good" if tee_pct < 5 else "moderate" if tee_pct < 10 else "poor"
    return {"tee_pct": tee_pct, "ccc": ccc, "std_bias": std_bias,
            "calib_slope": float(slope), "calib_intercept": float(intercept),
            "tee_rating": rating}


def score(pairs: list[tuple[float, float]], metric: str, fitted: bool = False) -> Result:
    """pairs are (ground truth, measured) for one metric across player-matches."""
    target, gates = METRICS[metric]
    nonzero = [(g, m) for g, m in pairs if g != 0]
    skipped = len(pairs) - len(nonzero)
    # A nan on the MEASURED side is a refusal, not a reading: load_match_json
    # emits one for every clamp-refused peak, which on 118575 is 179 of 384
    # rows. Left in, a single nan propagated through mean/std and turned the
    # whole metric into "n/a" with a FAIL verdict, while `n` still counted it -
    # so the harness reported a failure it had not measured. Counted, not
    # scored, and surfaced separately from a zero-GT skip.
    usable = [(g, m) for g, m in nonzero if math.isfinite(m)]
    refused = len(nonzero) - len(usable)
    if not usable:
        return Result(metric, target, gates, 0, float("nan"), float("nan"),
                      float("nan"), float("nan"), "NO DATA", skipped, fitted,
                      skipped_refused=refused)
    g = np.array([p[0] for p in usable], dtype=np.float64)
    m = np.array([p[1] for p in usable], dtype=np.float64)
    # MAE and bias MUST be the same statistic with and without the sign, or the
    # table is unreadable. They were not: MAE divided by GT (unbounded, so one
    # player with a tiny GT dominates the mean) while Bland-Altman divides by the
    # pair mean (bounded at +-200%). Side by side they looked like two different
    # phenomena and produced "sprints bias -150%", which is arithmetically
    # impossible for a count and was pure normaliser mismatch. Both now come off
    # the Bland-Altman diff array, which is the one the limits of agreement use.
    mean = (g + m) / 2.0
    diff = np.where(mean != 0, (m - g) / np.where(mean == 0, 1, mean) * 100.0, 0.0)
    mae = float(np.mean(np.abs(diff)))
    bias, lo, hi = bland_altman(g, m)
    # Report-only metrics have their own targets and do not get the 5/10 bands,
    # which are defined for the gating metrics.
    verdict = band(mae) if gates else ("MET" if mae <= target else "MISSED")
    return Result(metric, target, gates, len(usable), mae, bias, lo, hi, verdict,
                  skipped, fitted, skipped_refused=refused,
                  **agreement_stats(g, m))


def score_idsw(counts: list[int], fitted: bool = False) -> Result:
    if not counts:
        return Result("id_switches", IDSW_TARGET, True, 0, float("nan"), float("nan"),
                      float("nan"), float("nan"), "NO DATA", 0, fitted)
    a = np.array(counts, dtype=np.float64)
    mean = float(a.mean())
    # Same 1x/2x structure as the 5%/10% bands, on the switch target.
    verdict = "PASS" if mean < IDSW_TARGET else "MARGINAL" if mean < 2 * IDSW_TARGET else "FAIL"
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return Result("id_switches", IDSW_TARGET, True, len(a), mean, mean,
                  mean - 1.96 * sd, mean + 1.96 * sd, verdict, 0, fitted)


@dataclass
class Split:
    """One match split's worth of paired observations."""
    name: str
    fitted: bool
    pairs: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    idsw: list[int] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)

    def add(self, metric: str, gt: float, meas: float) -> None:
        self.pairs.setdefault(metric, []).append((gt, meas))

    def results(self) -> dict[str, Result]:
        out = {k: score(v, k, self.fitted) for k, v in self.pairs.items()}
        if self.idsw:
            out["id_switches"] = score_idsw(self.idsw, self.fitted)
        return out

    @property
    def n(self) -> int:
        return max((len(v) for v in self.pairs.values()), default=0)


def compare(gt: dict[str, dict], meas: dict[str, dict], mapping: dict | None,
            split: Split, match_id: str) -> None:
    """Pair players within one match and accumulate into `split`."""
    mapping = mapping or {}
    for tid, mm in meas.items():
        key = mapping.get(tid, tid)
        gm = gt.get(key)
        if gm is None:
            continue
        for metric in METRICS:
            if metric in gm and metric in mm:
                split.add(metric, float(gm[metric]), float(mm[metric]))
    split.matches.append(match_id)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _degrade_params(config: dict | None) -> dict[str, float] | None:
    """Parse the `degrade` string into numbers, for deciding which metrics the
    degrade model was capable of perturbing at all.

    Returns None when the run was NOT in degrade mode - the measured side then
    came from real measurement files, whose frame indices and identities differ
    from GT for real reasons, so every metric was genuinely evaluated. Only the
    synthetic path can be vacuous. Erring this way on an unparseable string is
    deliberate: wrongly marking a real result NOT EVALUATED hides a genuine
    measurement, which is the worse of the two mistakes.
    """
    raw = (config or {}).get("degrade")
    if not raw or raw == "none":
        return None
    out: dict[str, float] = {}
    for kv in str(raw).split(","):
        k, _, v = kv.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            return None          # can't prove it was vacuous, so don't claim it
    return out


def build_report(held: Split, train: Split, reference_error: float,
                 source: str, caveats: list[str], config: dict | None = None) -> dict:
    """Assemble the report. Refusing a headline is a normal outcome here.

    `config` records WHAT PRODUCED THIS REPORT. It is not optional bookkeeping:
    q, sigma_px, the estimator and the degrade level were process globals on
    `stats` and appeared nowhere in the output, so `model/validation/tune_runs/sweep.md`
    sat on disk reading like a measurement of the shipped product when it was in
    fact q=16, sigma_px=0.5 and 11 px of injected noise - 32x the shipped q/sigma
    ratio and the worst cell of the sweep. An unattributable report is worse than
    no report, because it gets quoted.
    """
    # Both computed ONCE. Split.results() builds fresh Result objects on every
    # call, so the "NOT EVALUATED" marking below would be silently discarded if
    # the dump at the bottom called it again.
    headline = held.results() if held.n else None
    fitted = train.results() if train.n else None
    status = "OK"
    if headline is None:
        status = ("REFUSED: no held-out matches. Every figure below was computed on "
                  "the tuning split and is FITTED, not measured.")
    elif held.n < MIN_PLAYER_MATCHES:
        status = (f"UNDERPOWERED: n={held.n} player-matches, below the n>={MIN_PLAYER_MATCHES} "
                  "the design doc requires before any figure is published.")

    def dump(res: dict[str, Result]) -> dict:
        return {k: {kk: (None if isinstance(vv, float) and math.isnan(vv) else vv)
                    for kk, vv in r.__dict__.items()} for k, r in res.items()}

    # A metric the degrade model provably cannot perturb has not been tested, and
    # must not be counted as a pass. With drop=0 the measured frame index is
    # byte-identical to GT, so `minutes` is comparing a number with itself; with
    # swap=0 the same is true of identities and `id_switches`. Both duly reported
    # "MAE 0.0%, PASS", which made 2 of the 3 gating metrics in the headline
    # vacuous while reading as corroboration. Called out rather than hidden: the
    # metric stays in the table, with a verdict that says what happened.
    deg = _degrade_params(config)
    vacuous = {} if deg is None else {
        "minutes": deg.get("drop", 0.0) == 0.0,
        "id_switches": deg.get("swap", 0.0) == 0.0,
    }
    for res in (headline, fitted):
        if not res:
            continue
        for name, is_vacuous in vacuous.items():
            if is_vacuous and name in res:
                res[name].verdict = "NOT EVALUATED"

    # Only genuinely evaluated gates decide the overall gate. Otherwise a run
    # that tested nothing at all would report PASS.
    gating = [r for r in (headline or {}).values()
              if r.gates and r.verdict != "NOT EVALUATED"]
    return {
        "status": status,
        "config": config or {},
        "source": source,
        "reference_error_pct": reference_error,
        "n_player_matches": held.n,
        "n_required": MIN_PLAYER_MATCHES,
        "held_out_matches": sorted(set(held.matches)),
        "train_matches": sorted(set(train.matches)),
        "headline": dump(headline) if headline else None,
        "fitted_train_split": dump(fitted) if fitted else None,
        "gate": ("NOT EVALUATED" if not gating else
                 "FAIL" if any(r.verdict == "FAIL" for r in gating) else
                 "MARGINAL" if any(r.verdict == "MARGINAL" for r in gating) else "PASS"),
        "caveats": caveats,
    }


_BANDS = ("- **<=5% PASS** — ship the claim.\n"
          "- **5-10% MARGINAL** — one time-boxed iteration on smoothing, calibration and "
          "outlier rejection, then re-decide. Not an automatic pass and not an automatic pivot.\n"
          "- **>10% FAIL** — the single-camera premise is wrong at 1080p.\n")


def _table(res: dict, gating: bool) -> str:
    rows = ["| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |",
            "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if bool(r["gates"]) != gating:
            continue
        # ID switches are a count, not a percentage. Printing a "%" next to it
        # would be the exact bundling this file exists to prevent.
        pct = "" if k == "id_switches" else "%"
        target = (f"<{r['target']:g} per player" if k == "id_switches"
                  else f"<={r['target']:g}%")
        mae = "n/a" if r["mae_pct"] is None else f"{r['mae_pct']:.1f}{pct}"
        loa = ("n/a" if r["loa_lower"] is None
               else f"{r['loa_lower']:+.1f}{pct} to {r['loa_upper']:+.1f}{pct}")
        bias = "n/a" if r["bias_pct"] is None else f"{r['bias_pct']:+.1f}{pct}"
        name = "ID switches (mean/player)" if k == "id_switches" else k
        rows.append(f"| {name} | {target} | {r['n']} | {mae} | {bias} | {loa} | "
                    f"**{r['verdict']}** |")
    return "\n".join(rows) + "\n" if len(rows) > 2 else "_none_\n"


def to_markdown(rep: dict, label: str) -> str:
    """The artifact a club actually reads. Honest beats flattering here: a reader
    who knows the field will look for the caveats, and not finding them is worse
    than finding them."""
    out = [f"# Accuracy validation — {label}", ""]
    if rep["headline"] is None:
        out += ["> ## NO HEADLINE FIGURE",
                f"> {rep['status']}",
                "> Nothing in this report may be quoted as a measured accuracy claim.", ""]
    else:
        out += [f"**Overall gate: {rep['gate']}** "
                f"(n = {rep['n_player_matches']} player-matches, "
                f"held-out matches: {', '.join(rep['held_out_matches']) or 'none'})", ""]
        if rep["status"] != "OK":
            out += [f"> **{rep['status']}**", ""]

    out += [f"Reference: **{rep['source']}**, own error stated at "
            f"**±{rep['reference_error_pct']:g}%**. Every figure below is agreement "
            "between two imperfect instruments, not error against truth — the limits "
            "of agreement column is the honest read.", ""]

    # Directly under the headline, not in an appendix. A report that does not say
    # what produced it gets quoted as if it were the shipped product.
    cfg = rep.get("config") or {}
    if cfg:
        out += ["**Configuration that produced this report** — a figure here describes "
                "THIS configuration and no other:", "",
                "| setting | value |", "|---|---|"]
        out += [f"| `{k}` | `{v}` |" for k, v in cfg.items()]
        out += [""]

    if rep["headline"]:
        out += ["## Gating metrics (held-out)", "", _table(rep["headline"], True), "",
                "## Report-only metrics (held-out)", "", _table(rep["headline"], False), "",
                "### Bands", "", _BANDS]

    if rep["fitted_train_split"]:
        out += ["## Tuning split — FITTED, NOT A CLAIM", "",
                "These matches were available for smoothing and threshold tuning. Any "
                "number here is fitted to the data it is measured on. It is shown only "
                "so the gap against the held-out figures is visible; quoting it would be "
                "a measurement error, not a rounding one.", "",
                f"Matches: {', '.join(rep['train_matches']) or 'none'}", "",
                _table(rep["fitted_train_split"], True),
                _table(rep["fitted_train_split"], False), ""]

    out += ["## Caveats", ""] + [f"- {c}" for c in rep["caveats"]]
    out += [f"- n >= {rep['n_required']} player-matches is required before publishing; "
                f"this report has {rep['n_player_matches']}."
                + ("" if rep["n_player_matches"] >= rep["n_required"] else " **Not met.**"),
            "- Peak speed is an extreme-value statistic and is report-only by design. "
            "A 10% agreement target against a reference whose own peak-speed error is "
            "5-10% is inside the instrument's noise and may not be evaluable at all.",
            ""]
    return "\n".join(out)


def write_report(rep: dict, out_dir: str, label: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in label).strip("-").lower()
    jp, mp = os.path.join(out_dir, f"{slug}.json"), os.path.join(out_dir, f"{slug}.md")
    with open(jp, "w") as f:
        json.dump(rep, f, indent=2)
    with open(mp, "w") as f:
        f.write(to_markdown(rep, label))
    return jp, mp


# --------------------------------------------------------------------------
# degradation (sensitivity study, not a pipeline measurement)
# --------------------------------------------------------------------------

def degrade(tracks: dict, noise: float = 0.0, drop: float = 0.0,
            swap: float = 0.0, seed: int = 0) -> dict:
    """Synthesize a measured side from ground truth with a known error budget.

    This measures the *stats layer's* sensitivity to tracker error, which is
    the curve the README asks for before the tracker exists: it says how good
    detection and association have to be for the distance target to be
    reachable. It is NOT a measurement of our pipeline and any report built
    from it says so.
    """
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    # Draw in bulk. Per-detection npr.normal() calls on a 1.5M-detection half
    # cost more than every other step in this file combined.
    out = {}
    for tid, tr in tracks.items():
        keep = npr.random(len(tr)) >= drop if drop else np.ones(len(tr), bool)
        kept = [dict(p) for p, k in zip(tr, keep) if k]
        if noise:
            for p, (dx, dy) in zip(kept, npr.normal(0, noise, (len(kept), 2))):
                p["x"] += float(dx)
                p["y"] += float(dy)
        out[tid] = kept
    if swap:
        ids = [t for t in out if len(out[t]) > 2]
        for _ in range(int(round(swap * len(ids)))):
            a, b = rng.sample(ids, 2)
            fa = [p["f"] for p in out[a]]
            cut = fa[len(fa) // 2]
            # Swap the tails: exactly what an ID switch does to two tracks.
            ta = [p for p in out[a] if p["f"] >= cut]
            tb = [p for p in out[b] if p["f"] >= cut]
            out[a] = [p for p in out[a] if p["f"] < cut] + tb
            out[b] = [p for p in out[b] if p["f"] < cut] + ta
            for tid in (a, b):
                for p in out[tid]:
                    p["id"] = tid
    return out


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def _cache_tracks(tracks: dict, fps: float, path: str) -> None:
    dets = [p for tr in tracks.values() for p in tr]
    with gzip.open(path, "wt") as f:
        json.dump({"fps": fps, "detections": dets}, f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt", nargs="+", required=True,
                    help="ground truth: SoccerTrack GSR json, tracks json(.gz), or one metrics csv")
    ap.add_argument("--measured", nargs="*", default=[],
                    help="pipeline output: tracks json(.gz), match.json, or one metrics csv. "
                         "Omit with --degrade to synthesize from ground truth.")
    ap.add_argument("--map", help="json {match_id: {measured_track_id: gt_player_id}}")
    ap.add_argument("--held-out", default="",
                    help="comma-separated match ids reserved for reporting. Without this "
                         "no headline figure is produced.")
    ap.add_argument("--reference-error", type=float, default=1.5,
                    help="the reference device's own error, %% (GPS total distance ~1-2%%)")
    ap.add_argument("--degrade", default="",
                    help="synthesize measured from GT, e.g. 'noise=0.5,drop=0.1,swap=0.1'")
    ap.add_argument("--calib", help="calibration.json; default is identity (GT already in metres)")
    ap.add_argument("--out", default="model/validation")
    ap.add_argument("--label", default="validation")
    ap.add_argument("--source", default="SoccerTrack v2 GSR (per-frame pitch metres, CC BY 4.0)")
    ap.add_argument("--cache", action="store_true",
                    help="write a compact tracks.json.gz next to each GT file and exit")
    # The GT side has always run through the same Kalman as the measured side,
    # which cancels the estimator's own bias and makes an ABSOLUTE distance error
    # unmeasurable - it can only ever report agreement between two filtered
    # copies of the same track. `raw` scores the reference as unfiltered
    # sum-of-deltas path length, which is what "26% low" has to be low AGAINST.
    # It is bandwidth-dependent: state the rate it was computed at.
    ap.add_argument("--gt-estimator", default="kalman", choices=["kalman", "raw", "boxcar"],
                    help="how the REFERENCE side's aggregates are computed")
    # Tuning knobs, so a sweep does not have to edit stats.py between runs.
    ap.add_argument("--q", type=float, help="override stats.ACCEL_Q")
    ap.add_argument("--sigma-px", type=float, help="override stats.SIGMA_PX")
    a = ap.parse_args(argv)

    if a.q is not None:
        stats.ACCEL_Q = a.q
    if a.sigma_px is not None:
        stats.SIGMA_PX = a.sigma_px
    # A directory of per-clip calibrations, not one file: a hovering drone sits
    # at a different pose in every clip, so one shared H silently warps the
    # others. Falls back to the single-file form, and to identity (GT already in
    # metres) when no calibration is given at all.
    cal_dir = a.calib if a.calib and os.path.isdir(a.calib) else None
    cal = Calibration.load(a.calib) if a.calib and not cal_dir else METRE_CAL
    held_ids = {s.strip() for s in a.held_out.split(",") if s.strip()}
    deg = dict(kv.split("=") for kv in a.degrade.split(",")) if a.degrade else {}

    held, train = Split("held-out", False), Split("train", True)
    caveats: list[str] = []

    for path in a.gt:
        match_id = os.path.basename(path).split(".")[0]
        if cal_dir:
            cal = Calibration.load(os.path.join(cal_dir, f"calib_{match_id}.json"))
        gt_metrics, gt_tracks, fps = load_any(path, cal, a.gt_estimator)
        if a.cache:
            out = os.path.join(os.path.dirname(path), match_id + ".tracks.json.gz")
            _cache_tracks(gt_tracks, fps, out)
            print(f"{match_id}: {len(gt_tracks)} tracks -> {out}")
            continue

        if deg:
            meas_tracks = degrade(gt_tracks, float(deg.get("noise", 0)),
                                  float(deg.get("drop", 0)), float(deg.get("swap", 0)))
            meas_metrics = match_metrics(meas_tracks, cal, fps)
        else:
            mpath = next((m for m in a.measured if match_id in os.path.basename(m)), None)
            if mpath is None:
                print(f"no measurement file for {match_id}, skipping")
                continue
            meas_metrics, meas_tracks, _ = load_any(mpath, cal)

        split = held if match_id in held_ids else train
        compare(gt_metrics, meas_metrics, (json.load(open(a.map)) if a.map else {}).get(match_id),
                split, match_id)
        if gt_tracks and meas_tracks:
            split.idsw += list(id_switches(gt_tracks, meas_tracks, cal=cal).values())
        print(f"{match_id}: {len(gt_metrics)} GT players, {len(meas_metrics)} measured "
              f"-> {split.name}")

    if a.cache:
        return 0

    if a.gt_estimator != "kalman":
        caveats.append(f"Reference aggregates use the **{a.gt_estimator}** estimator, not the "
                       "product's Kalman, so this is an ABSOLUTE error against a stated-"
                       "bandwidth path length rather than agreement between two filtered "
                       "copies. Sum-of-deltas path length grows with sample rate; the figure "
                       "is only meaningful next to the rate it was computed at.")
    if deg:
        caveats.append(f"**Measured side is synthetic** ({a.degrade}): ground truth degraded "
                       "with a known error budget. This is a sensitivity study of the stats "
                       "layer, NOT a measurement of the tracking pipeline, which has never "
                       "been run on real footage.")
    if "soccertrack" in a.source.lower():
        caveats += [
            "SoccerTrack v2 is 4K panoramic footage; the product camera is 1080p "
            "perspective. The domain gap is real and unquantified.",
            "Both sides run through the same metric estimator (stats.py smoothing, speed "
            "clamp, gap rules), so the estimator's own bias cancels. These figures are a "
            "lower bound on error. The published claim needs GPS-vest reference data, "
            "where nothing cancels.",
        ]
    caveats.append("Player pairing is by track id (or an explicit --map). Identity is "
                   "unbuilt in the pipeline, so a real run needs the mapping supplied.")

    rep = build_report(held, train, a.reference_error, a.source, caveats, config={
        "accel_q": stats.ACCEL_Q,
        "sigma_px": stats.SIGMA_PX,
        "distance_window_s": stats.DISTANCE_WINDOW_S,
        "gt_estimator": a.gt_estimator,
        "degrade": a.degrade or "none",
        "calibration": ("per-match dir" if cal_dir else a.calib or "IDENTITY (GT in metres)"),
    })
    jp, mp = write_report(rep, a.out, a.label)
    print(f"\n{rep['status']}\ngate: {rep['gate']}\nwrote {jp} and {mp}")
    return 0


# --------------------------------------------------------------------------

def _demo() -> None:
    """Known-error cases in, known verdicts out."""
    # The duplication in _speed_series must reproduce stats.player_stats exactly,
    # or HSR is computed off a different track than every other metric.
    n = int(20 * 25)
    track = [{"f": i, "id": 1, "team": "home", "x": 5.0 * (i / 25.0), "y": 34.0}
             for i in range(n + 1)]
    t = np.array([p["f"] / 25.0 for p in track])
    v, dt, real, xy = _speed_series(track, METRE_CAL, 25.0)
    m = player_metrics(track, METRE_CAL, 25.0)
    # Distance is the windowed path length off `xy`, NOT sum(v*dt) - see
    # stats.DISTANCE_WINDOW_S. This assert is what stops the two drifting apart.
    assert abs(stats.path_length(xy, t, real) - m["distance_m"]) < 1e-9
    assert m["hsr_m"] == 0.0, m["hsr_m"]  # 5 m/s is under the 5.5 threshold

    # On a straight constant-speed run the window estimator must be exact: no
    # curvature to lose to the chord, so 5 m/s for 20 s is 100 m however it is
    # sliced. A window bug shows up here as a few percent, silently, on the one
    # case where there is a closed-form answer.
    assert abs(m["distance_m"] - 100.0) < 1e-6, m["distance_m"]

    # 10000m GT, 10400m measured. 400 over a PAIR MEAN of 10200 is 3.922%, not
    # 4.000% - MAE now uses the Bland-Altman denominator so that it and the bias
    # are one statistic with and without the sign. The old GT denominator gave
    # 4.000% here and was unbounded, which is what let one small-GT player carry
    # a whole metric (and produced a -150% bias on a count).
    r = score([(10000.0, 10400.0)] * 30, "distance_m")
    assert abs(r.mae_pct - 400.0 / 10200.0 * 100.0) < 1e-9, r.mae_pct
    assert abs(r.mae_pct - abs(r.bias_pct)) < 1e-9, (r.mae_pct, r.bias_pct)
    assert r.verdict == "PASS", r.verdict
    assert r.gates and r.target == 5.0

    # 12% -> FAIL, and the middle band actually fires at 7%.
    assert score([(10000.0, 11200.0)] * 30, "distance_m").verdict == "FAIL"
    assert score([(10000.0, 10700.0)] * 30, "distance_m").verdict == "MARGINAL"

    # Bias must survive: a systematic +4% is a different problem from random +-4%,
    # and MAE alone cannot tell them apart.
    assert r.bias_pct > 3.5, r.bias_pct
    mixed = score([(10000.0, 10400.0), (10000.0, 9600.0)] * 15, "distance_m")
    # Neither is exactly 4.0 / 0.0: Bland-Altman divides by the pair mean, so
    # +400 on a 10200 mean (+3.9216%) and -400 on a 9800 mean (-4.0816%) do not
    # cancel. That asymmetry is the method, not a bug - and now that MAE shares
    # the denominator it inherits the same skew, which is the point: one
    # statistic, with and without the sign.
    hi, lo = 400.0 / 10200.0 * 100.0, 400.0 / 9800.0 * 100.0
    assert abs(mixed.mae_pct - (hi + lo) / 2.0) < 1e-9, mixed.mae_pct
    assert abs(mixed.bias_pct - (hi - lo) / 2.0) < 1e-9, mixed.bias_pct
    assert mixed.loa_lower < -3.5 and mixed.loa_upper > 3.5

    # Zero-GT rows (a player with no sprints) are skipped, not scored as infinite.
    z = score([(0.0, 1.0), (4.0, 5.0)], "sprints")
    assert z.n == 1 and z.skipped_zero_gt == 1, z

    # A headline computed on the tuning split must be refused outright.
    train = Split("train", True)
    for _ in range(80):
        train.add("distance_m", 10000.0, 10400.0)
    rep = build_report(Split("held-out", False), train, 1.5, "synthetic", [])
    assert rep["headline"] is None, "reported a headline from the tuning split"
    assert rep["status"].startswith("REFUSED"), rep["status"]
    assert rep["gate"] == "NOT EVALUATED", rep["gate"]
    assert rep["fitted_train_split"]["distance_m"]["fitted"] is True
    md = to_markdown(rep, "self-check")
    assert "NO HEADLINE FIGURE" in md and "FITTED, NOT A CLAIM" in md

    # A gate the degrade model cannot move is NOT a pass. With drop=0 the
    # measured frame index IS the GT frame index, so `minutes` compares a number
    # with itself; reporting that as PASS is how a run that tested nothing reads
    # as corroboration.
    held = Split("held-out", False)
    for _ in range(80):
        held.add("distance_m", 10000.0, 10400.0)
        held.add("minutes", 90.0, 90.0)
    cfg = {"degrade": "noise=3"}                       # drop absent => drop == 0
    r2 = build_report(held, Split("train", True), 1.5, "synthetic", [], cfg)
    assert r2["headline"]["minutes"]["verdict"] == "NOT EVALUATED", r2["headline"]["minutes"]
    assert r2["headline"]["distance_m"]["verdict"] == "PASS"
    assert r2["gate"] == "PASS", r2["gate"]            # decided by distance alone

    # ...but a run that only had vacuous gates must not claim a pass at all.
    only = Split("held-out", False)
    for _ in range(80):
        only.add("minutes", 90.0, 90.0)
    assert build_report(only, Split("train", True), 1.5, "syn", [], cfg)["gate"] == "NOT EVALUATED"

    # Real measurement files (no degrade) can move everything, so nothing is
    # marked - otherwise the honest path gets its results suppressed.
    r3 = build_report(held, Split("train", True), 1.5, "synthetic", [], {"degrade": "none"})
    assert r3["headline"]["minutes"]["verdict"] == "PASS", r3["headline"]["minutes"]
    assert _degrade_params({"degrade": "drop=0.2,swap=1"}) == {"drop": 0.2, "swap": 1.0}
    assert _degrade_params({"degrade": "garbage"}) is None

    # With a held-out split the same numbers do produce a headline, but n<60 is
    # called out rather than quietly published.
    held = Split("held-out", False)
    for _ in range(30):
        held.add("distance_m", 10000.0, 10400.0)
    held.matches.append("m1")
    small = build_report(held, train, 1.5, "synthetic", [])
    assert small["status"].startswith("UNDERPOWERED"), small["status"]
    assert small["gate"] == "PASS" and small["headline"]["distance_m"]["fitted"] is False
    for _ in range(40):
        held.add("distance_m", 10000.0, 10400.0)
    assert build_report(held, train, 1.5, "synthetic", [])["status"] == "OK"

    # ID switches: a clean pairing scores zero, a mid-track swap scores one each.
    a_tr = {"1": [{"f": i, "id": "1", "x": 10.0 + i * 0.1, "y": 10.0} for i in range(100)],
            "2": [{"f": i, "id": "2", "x": 60.0 - i * 0.1, "y": 50.0} for i in range(100)]}
    assert set(id_switches(a_tr, a_tr).values()) == {0}
    swapped = degrade(a_tr, swap=0.5, seed=1)  # 0.5 * 2 tracks = one swap event
    assert sum(id_switches(a_tr, swapped).values()) == 2, id_switches(a_tr, swapped)
    assert score_idsw([0, 1, 1, 0]).verdict == "PASS"
    assert score_idsw([3, 3, 3]).verdict == "MARGINAL"
    assert score_idsw([9, 9]).verdict == "FAIL"

    # Degradation must actually move the number it claims to move.
    gt_tracks = {str(t): [{"f": i, "id": str(t), "x": 20.0 + 3.0 * math.sin(i / 40.0 + t),
                           "y": 30.0 + 3.0 * math.cos(i / 55.0 + t)} for i in range(1500)]
                 for t in range(4)}
    clean = match_metrics(gt_tracks, METRE_CAL, 25.0)
    noisy = match_metrics(degrade(gt_tracks, noise=0.5), METRE_CAL, 25.0)
    # This assert used to require that noise INFLATE distance, which encoded the
    # old boxcar pathology as expected behaviour: summing position deltas ratchets,
    # so every jitter added length and none removed it. The Kalman integrates
    # filtered velocity and has no ratchet, so it correctly failed that assert.
    # What must still hold is that the degradation MOVES the number - a no-op
    # degrade would make every sensitivity study meaningless - and that the
    # response stays bounded rather than exploding.
    # Test the two properties separately, because measuring degrade() through a
    # filter built to reject it conflates "the degrader is broken" with "the
    # filter works". Check the perturbation on RAW positions...
    degraded = degrade(gt_tracks, noise=0.5)
    moved = [abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
             for k in gt_tracks for a, b in zip(gt_tracks[k], degraded[k])]
    assert np.mean(moved) > 0.1, f"degrade() barely moved raw positions: {np.mean(moved)}"
    # ...and separately that the filtered distance stays bounded. The old boxcar
    # path inflated ~90% here; the ratchet coming back would show up as a blowup.
    ratios = [noisy[k]["distance_m"] / clean[k]["distance_m"] for k in clean]
    assert all(r < 1.5 for r in ratios), f"noise inflated distance - ratchet is back: {ratios}"

    print("validate.py ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    _demo()
