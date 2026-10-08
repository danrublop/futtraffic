"""N28: plausibility without ground truth - does a professional league's top speed flag a wrong
speed estimate, and leave the truth alone?

WHY
    N21 knows the product's speeds are too high only because ST2 has ground truth; footage a user
    uploads has none. Top speed has a published range: in SkillCorner's open A-League 2024/25
    aggregates the fastest player's PSV-99 is 8.97 m/s (research-physical-metrics-and-joins (not published)
    §6). A run whose typical player is faster than that is wrong somewhere upstream, and seeing that
    takes no truth. N28 asks whether the check sees what N21 measured without flagging the truth.

PROTOCOL - committed 2026-09-14, before any arm below was computed except the one marked SEEN
    PSV-99, SkillCorner's peak speed: for one player, the 99th percentile (numpy's default linear
    interpolation) of their observed speed samples above 4.0 m/s. A player with under 1.0 s of such
    samples has none. A source's statistic is the median PSV-99 over its players.
    PRIOR: psv99 / 3.6 over every row of SkillCorner's aus1league_physicalaggregates_20242025.csv -
    406 rows, as the research page measured them. A source is IMPLAUSIBLE iff its median PSV-99 is above the
    prior's maximum: its typical player outruns the fastest player in a professional league.

    RULED - N27's eight clips (N22's nine without 132877), over each clip's 4 minutes:
      truth    ST2's own per-row speed field over the clip's joined window, per joined player -
               neither our estimator nor our map.
      product  what ships: `.p21id` tracks, N6's calibration (`.p05`), `stats.build_match`'s
               population (per track id, MIN_MINUTES, officials out), speed from
               `stats.speed_series` on observed steps with MAX_SPEED lifted. That velocity is what
               HSR and sprints are thresholded on; the cap only lowers readings above 9.5 m/s, and
               the app refuses a top speed wherever it binds, so the capped reading (`shipped`) is
               reported, not ruled.
      truth IMPLAUSIBLE          -> FLAGS THE TRUTH: the check is void on 4-minute clips
      else product IMPLAUSIBLE   -> CATCHES THE PRODUCT
      else                       -> BLIND TO THE PRODUCT
    Beside the rule: the verdict with each clip left out (FRAGILE if any differs), per-clip
    statistics, and each source's share of players above the prior's maximum.

    REPORTED, NOT RULED
      match  ST2's speed field over each whole match, all ten, players with at least 10 minutes in
             the two halves. SEEN in an exploratory check before this protocol - 132831 reads
             implausible (st2-132877-feed-is-mirrored (internal note)) - so it cannot test the rule.
      gnss   SoccerTrack v1's GNSS session in pitch metres (model/validation/st1_gnss.py), speed from a
             1.0 s central difference on the 0.1 s grid: no camera anywhere in it.

    .venv/bin/python model/validation/plausibility.py --n28        # writes model/validation/out/n28_results.json
    .venv/bin/python model/validation/plausibility.py --selfcheck
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
import stats  # noqa: E402

PRIOR_CSV = os.path.expanduser(
    "~/Downloads/skillcorner-opendata/data/aggregates/aus1league_physicalaggregates_20242025.csv")
OUT = os.path.join(ROOT, "model/validation/out/n28_results.json")
PSV_FLOOR_MS = 4.0
PSV_MIN_S = 1.0
FEED_FPS = 25.0
MATCH_MIN_S = 600.0
ARMS = ("truth", "product", "shipped")

_FRAME = re.compile(rb'<frame matchTime="[-0-9]+" frameNumber="([0-9]+)" eventPeriod="([A-Z_]+)"')
_SPEED = re.compile(rb'<player playerId="([0-9]+)" loc="\[[^\]]*\]" speed="([-0-9.naN]+)"')


def psv99(v, dt) -> float | None:
    """SkillCorner's PSV-99 over one player's speed samples, or None under 1 s above the floor."""
    v = np.asarray(v, float)
    fast = np.isfinite(v) & (v > PSV_FLOOR_MS)
    if float(np.sum(np.broadcast_to(np.asarray(dt, float), v.shape)[fast])) < PSV_MIN_S - 1e-9:
        return None
    return float(np.percentile(v[fast], 99))


def prior() -> dict:
    if not os.path.exists(PRIOR_CSV):
        raise SystemExit(f"{PRIOR_CSV} missing - SkillCorner's open data "
                         "(research-skillcorner-opendata (not published))")
    p = np.array([float(r["psv99"]) / 3.6 for r in csv.DictReader(open(PRIOR_CSV))])
    return {"rows": len(p), "median": float(np.median(p)), "max": float(p.max())}


def summary(values, pri: dict) -> dict:
    v = np.asarray(values, float)
    med = float(np.median(v)) if len(v) else None
    return {"players": len(v), "median": med,
            "above_max": float(np.mean(v > pri["max"])) if len(v) else None,
            "implausible": med is not None and med > pri["max"]}


def feed_speeds(seq: str, window: tuple[str, int, int] | None = None) -> dict[str, np.ndarray]:
    """{ST2 player id: samples of the feed's own speed field} over one period's frame window, or
    over both halves of the whole match."""
    import st2_team_gt as G  # noqa: PLC0415

    per: dict[str, list] = collections.defaultdict(list)
    keep = False
    with open(f"{G.RAW}/{seq}/{seq}_tracker_box_data.xml", "rb") as fh:
        for line in fh:
            s = line.lstrip()
            if s[:6] == b"<frame":
                m = _FRAME.match(s)
                keep = bool(m) and (m.group(2) in (b"FIRST_HALF", b"SECOND_HALF") if window is None else
                                    m.group(2) == window[0].encode() and window[1] <= int(m.group(1)) < window[2])
            elif keep and s[:7] == b"<player":
                m = _SPEED.match(s)
                if m:
                    per[m.group(1).decode()].append(float(m.group(2)))
    return {k: np.array(v) for k, v in per.items()}


def _track_speeds(track: list[dict], cal, fps: float) -> dict | None:
    """One track in `build_match`'s population -> {"product": lifted, "shipped": capped} observed
    (speeds, dt), or None where build_match would drop it."""
    ps = stats.player_stats(track, cal, fps)
    if ps is None or ps.minutes < stats.MIN_MINUTES or ps.team == "official":
        return None
    v, dt, real, _ = stats.speed_series(track, cal, fps)
    out = {"shipped": (v[real], dt[real])}
    keep, stats.MAX_SPEED = stats.MAX_SPEED, np.inf
    try:
        v, dt, real, _ = stats.speed_series(track, cal, fps)
    finally:
        stats.MAX_SPEED = keep
    out["product"] = (v[real], dt[real])
    return out


def product_speeds(seq: str) -> dict[str, list[tuple]]:
    import calibrators  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415

    cal = calibrators.load(f"{D.STC}/{seq}.p05.calibration.json")
    blob = json.load(open(f"{D.STC}/{seq}.p21id.tracks.json"))
    by_id: dict[int, list] = collections.defaultdict(list)
    for d in blob["detections"]:
        by_id[d["id"]].append(d)
    got = [s for tr in by_id.values() if (s := _track_speeds(tr, cal, float(blob["fps"]))) is not None]
    return {a: [s[a] for s in got] for a in ("product", "shipped")}


def _central_1s(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """|p(t + 0.5 s) - p(t - 0.5 s)| per player on the 0.1 s grid; NaN where either end is missing."""
    import st1_gnss as S  # noqa: PLC0415

    g = S._grid(t, xy)
    half = int(round(0.5 / S.GNSS_STEP_S))
    return np.linalg.norm(g[2 * half:] - g[:-2 * half], axis=2) / (2 * half * S.GNSS_STEP_S)


def n28_verdict(res: dict) -> dict:
    pri, C = res["prior"], sorted(res["clips"])

    def pooled(arm, clips):
        return summary([x for c in clips for x in res["clips"][c][arm]], pri)

    def rule(clips):
        return ("FLAGS THE TRUTH" if pooled("truth", clips)["implausible"] else
                "CATCHES THE PRODUCT" if pooled("product", clips)["implausible"] else
                "BLIND TO THE PRODUCT")

    follows = rule(C)
    loo = {c: rule([x for x in C if x != c]) for c in C}
    return {**{a: pooled(a, C) for a in ARMS},
            "per_clip": {c: {a: summary(res["clips"][c][a], pri) for a in ARMS} for c in C},
            "leave_one_out": loo, "fragile": any(v != follows for v in loo.values()),
            "follows": follows}


def run_n28() -> dict:
    import dist_eval as D  # noqa: PLC0415
    import st1_gnss as S  # noqa: PLC0415
    import st2_team_gt as G  # noqa: PLC0415

    pri = prior()
    print(f"prior: {pri}")
    J = G.load()
    clips = {}
    for seq in D.N27_SEQS:
        j = J[seq]
        feed = feed_speeds(seq, (j["period"], j["offset"], j["offset"] + j["clip_frames"]))
        truth = [psv99(feed[p], 1.0 / FEED_FPS) for p in sorted({str(x) for x in j["mot_to_player"].values()})
                 if p in feed]
        ours = product_speeds(seq)
        clips[seq] = {"truth": [x for x in truth if x is not None],
                      **{a: [x for v, dt in ours[a] if (x := psv99(v, dt)) is not None] for a in ("product", "shipped")}}
        print(f"  {seq}: " + "  ".join(f"{a} {summary(clips[seq][a], pri)}" for a in ARMS), flush=True)

    match = {}
    for seq in G.SEQS:
        feed = feed_speeds(seq)
        vals = [psv99(v, 1.0 / FEED_FPS) for v in feed.values() if len(v) >= MATCH_MIN_S * FEED_FPS]
        match[seq] = summary([x for x in vals if x is not None], pri)
        print(f"  match {seq}: {match[seq]}", flush=True)

    t, xy, labels, g = S.positions()
    v = _central_1s(t, xy)
    gnss = {**summary([x for k in range(v.shape[1]) if (x := psv99(v[:, k], S.GNSS_STEP_S)) is not None], pri),
            "georef_rms_m": g["rms_m"], "georef_scale": g["scale"]}
    print(f"  gnss: {gnss}")

    res = {"prior": pri, "clips": clips, "match": match, "gnss": gnss}
    res["verdict"] = n28_verdict(res)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    vd = res["verdict"]
    print(f"N28 truth {vd['truth']}  product {vd['product']}  shipped {vd['shipped']}")
    print(f"N28 {vd['follows']}{'  FRAGILE' if vd['fragile'] else ''}")
    return res


def _selfcheck() -> None:
    import dist_eval as D  # noqa: PLC0415

    fps = 25.0
    # PSV-99 is the fast tail, and a second of it is the floor.
    jog, sprint = np.full(1500, 3.0), np.full(50, 8.0)
    assert psv99(np.r_[jog, sprint], 1 / fps) == 8.0
    assert psv99(np.r_[jog, sprint[:20]], 1 / fps) is None, "0.8 s above 4 m/s is not a PSV-99"
    assert psv99(np.r_[jog, sprint[:20]], np.full(1520, 0.1)) == 8.0, "the floor is time, not samples"

    # The rule, on medians whose answers are arithmetic.
    pri = {"rows": 3, "median": 7.0, "max": 9.0}

    def res(truth, product):
        return {"prior": pri, "clips": {c: {"truth": truth[i], "product": product[i], "shipped": product[i]}
                                        for i, c in enumerate("ab")}}
    assert n28_verdict(res([[7, 8], [7]], [[10, 10], [6, 10]]))["follows"] == "CATCHES THE PRODUCT"
    assert n28_verdict(res([[10, 10], [9.5]], [[10], [10]]))["follows"] == "FLAGS THE TRUTH"
    assert n28_verdict(res([[7], [7]], [[8], [8]]))["follows"] == "BLIND TO THE PRODUCT"
    one = n28_verdict(res([[7], [7]], [[10, 10, 10], [8]]))
    assert one["follows"] == "CATCHES THE PRODUCT" and one["fragile"], one["leave_one_out"]

    # The product is read above the cap, and its capped reading is kept apart.
    got = _track_speeds([dict(r, team="home") for r in D._walk(400, fps, 11.0)], D._Flat, fps)
    assert np.max(got["product"][0]) > stats.MAX_SPEED + 1.0, np.max(got["product"][0])
    assert np.max(got["shipped"][0]) <= stats.MAX_SPEED + 1e-9
    assert _track_speeds([dict(r, team="official") for r in D._walk(400, fps, 11.0)], D._Flat, fps) is None

    # GNSS speed: 6 m/s along x on the 0.1 s grid with one step missing.
    t = np.delete(np.arange(0, 5.0, 0.1), 20)
    xy = np.stack([6.0 * t, np.zeros_like(t)], 1)[:, None, :]
    v = _central_1s(t, xy)
    assert np.nanmax(np.abs(v - 6.0)) < 1e-9 and np.isnan(v[:, 0]).sum() == 2, v[:, 0]

    if os.path.exists(PRIOR_CSV):
        p = prior()
        assert p["rows"] == 406 and round(p["max"], 2) == 8.97, p
    print("plausibility.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n28", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.n28:
        run_n28()
    else:
        ap.print_help()
