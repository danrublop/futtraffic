"""N58: the ball's position RELATIVE TO THE PLAYERS as a likelihood term.

    .venv/bin/python model/validation/n58_player_prior.py --fit      # the prior, from TRAIN clips only
    .venv/bin/python model/validation/n58_player_prior.py --score    # the arms
    .venv/bin/python model/validation/n58_player_prior.py --demo

WHY THIS RUNS, AND WHY IT IS THE LAST IDEA STANDING
    FOUR pre-registered PICKER changes have lost: N49 on arm B, N50's re-fitted
    size gate, N51's joint arm, and N49 §5b on arm D. (N56's penalty-area pass is
    not one of them and an earlier draft of this line counted it: it is a
    detector-side re-threshold scored on player recall over N46's TEST clips,
    sharing no population, cache, baseline or metric with the picker family - and
    it ROSE and missed a bar, where all four picker arms fell below their own
    baselines.)
    N57 then removed the explanation I had proposed for all of them - detector
    confidence is not flat or inverted, it is a strong ranker (AUC 0.8895 pooled),
    so `log(conf)` was pointing the right way and the losses are four independent
    negatives rather than one sign error.

    What N57 leaves is arithmetic, not a mystery: AUC asks whether one true beats
    one decoy; an argmax must beat ALL of them, and arm D emits 86.2 candidates a
    frame. The lever is therefore either fewer candidates or a signal that is
    NEITHER confidence NOR motion, so that it multiplies rather than duplicates
    what is already there. The 22 tracked bodies are the only such signal on disk,
    and nothing in `ball.BallPicker` knows where any of them are.

    Wei et al. (CVIU 2013) track players first and infer possession; Maksai et al.
    (CVPR 2016) model the ball's state as in-possession or free flight, so that a
    disappearance is EXPLAINED by the state rather than charged as a miss.

THE FEATURE, fixed here
    For a candidate c and the set of player boxes P in the same frame:
        d(c) = min over p in P of || centre(c) - foot(p) || / height(p)
    A body-height-normalised distance to the nearest player's ground point. It is
    scale-free on purpose - these six clips span 4K top-down and 6500x1000 side
    views with no shared calibration, and a metric distance is unavailable on the
    F clips (`ball_eval._clip_cal` returns None for them).

THE FIT SET, which is the part that decides whether this is honest
    p(d | ball) is estimated on the 82 TeamTrack clips whose `ball_clean.role()`
    is "train" - GT ball and GT players, never scored by ball_cmp, disjoint from
    the six scored clips by construction and ASSERTED to be. Nothing about the
    scored clips is used to build the term. No parameter is fitted on the data the
    verdict is read from.

RULE, FIXED BEFORE THE SCORED RUN (committed with this file, nothing run yet)
    population  ball_cmp.CLIPS, the committed `.cmpD` candidate caches, every GT
                frame, reported for BOTH of N57's populations (dead-ball and
                in-play) and pooled.
    arms        SHIPPED   ball.BallPicker untouched, pooled 0.4706. The PRODUCT's
                          picker, kept for context - NOT the bar.
                CONF      argmax(conf) with the scenery, on-pitch and size gates
                          REMOVED, pooled 0.5167. **THIS IS THE BASELINE THE BAR
                          IS ANCHORED TO.** An earlier draft anchored to SHIPPED,
                          and since a signal-free argmax already scores 0.5167,
                          a PRIOR landing on the old 0.5206 bar would have read
                          as "the player prior works" while 99% of it came from
                          deleting three gates. A control that nearly clears the
                          bar without the treatment is not a control.
                PRIOR     the identical gates, then argmax of
                          log(conf) + log p(d | ball), with p from --fit.
                ORACLE-P  the same, with GT player boxes instead of detected ones.
                          The CEILING on this idea, never shippable, reported so
                          that a PRIOR failure can be told from a player-tracking
                          failure.
                oracle/centre unchanged, as ball_cmp defines them.
    players     for PRIOR: the product's own detector, coreml conf 0.5 tile 1024,
                the shipped configuration - not GT, or the arm measures nothing
                the product could do.
    PASS        pooled PRIOR >= CONF + 0.05 = 0.5667 on the six clips together - the bar
                N49 §5b used on this metric (N56's +0.15 was a DIFFERENT metric,
                one player's recall at IoU 0.5, and an earlier draft of this line
                miscited it) - AND no population may fall by more than 0.01
                against SHIPPED. A gain pooled over a loss on one population is
                not a gain; N57 split them for this reason.
    HEADROOM, PER POPULATION, STATED BEFORE THE RUN because N56's bar was set at
                95% of a ceiling already committed in the repo and nobody checked.
                Arm D, 20 px gate: dead shipped 0.0117 / oracle 0.7433 over 600
                frames; in-play 0.7000 / 0.9517 over 1200. So 59% of the pooled
                headroom sits in the dead balls - the population this very feature
                cannot move, since a resting ball is near nobody (n55). A pooled
                +0.05 therefore demands roughly +0.075 from the in-play clips
                alone, and the per-population floor is INERT on the dead side
                where shipped is 7 frames of 600. This is disclosed, not fixed:
                if the arm passes it will be an in-play result and must be read
                as one.
    VOID        (a) the fit clips intersect the scored clips;
                (b) `centre` scores over 1%;
                (c) the arms do not score identical frames;
                (d) SHIPPED does not reproduce ball_cmp's 0.4706.
    n           n=1 configuration. One feature, one normalisation, one fit set,
                no weight between the two terms (they are added as log
                likelihoods, which is the only combination that needs no
                parameter). If a weight is ever introduced, that n moves.
    REPORT      `wrong` and `silent` per arm, as N49 §5b required: a picker that
                answers less often buys correctness with silence, and on this
                metric silence is a failure.
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
OUT = os.path.join(ROOT, "model", "validation", "out", "ball_clean", "n58_prior.json")
#: The published pooled `shipped` for each candidate cache, by suffix. .cmpD is D1 at the
#: RFDETRNano DEFAULT input of 384 - which is what every ball number in this repo was
#: measured at - and .cmpDr is the same checkpoint at the 640 it was TRAINED at
#: (docs/experiments/n65-every-ball-number-was-measured-at-384.md).
BASELINE = {".cmpD": 0.4706, ".cmpDr": 0.5850}

BINS = np.linspace(0.0, 8.0, 33)      # body heights; beyond 8 is "far from everyone"
FLOOR = 1e-4                           # so an empty bin cannot make a candidate impossible


def _players(clip, frame, gt_players):
    return gt_players.get(frame, [])


def feature(cx, cy, players):
    """Body-height-normalised distance to the nearest player's ground point."""
    best = np.inf
    for x, y, h in players:
        if h <= 0:
            continue
        best = min(best, float(np.hypot(cx - x, cy - y) / h))
    return best


def fit(limit=None):
    """p(d | ball) from TRAIN clips only. GT ball, GT players, no detector."""
    import ball_clean as B

    store = B.STORE if hasattr(B, "STORE") else None
    clips = [c for c in B.on_disk() if B.role(*B._parts(c)) == "train"]
    if limit:
        clips = clips[:limit]
    ds = []
    for clip in clips:
        rows = B.ball_rows(clip)
        csv = str(store / f"{clip.replace('videos', 'annotations')}.csv")
        tr = B.load_tracks(csv)
        # REBASE THE PLAYERS. ball_rows() rebases ball frames to 0 and
        # teamtrack_tracks.load does not, so a raw join pairs the ball at frame k
        # with the players at k-1 and drops frame 0 entirely - the exact bug
        # ball_rows' own docstring warns about, reintroduced at the call site.
        base = min((r["f"] for v in tr.values() for r in v), default=0)
        by_f = collections.defaultdict(list)
        for v in tr.values():
            for r in v:
                by_f[r["f"] - base].append((r["x"], r["y"], r["h"]))
        for r in rows:
            p = by_f.get(r["f"], [])
            if p:
                # ball_rows' x,y are ALREADY the box centre (want_ball=True sets
                # y = top + h/2). Adding w/2 and h/2 again put the point on the
                # bottom-right corner of the ball's own box.
                ds.append(feature(r["x"], r["y"], p))
    ds = np.array([d for d in ds if np.isfinite(d)])
    # AN EMPTY FIT IS THE FAILURE THIS ARM CANNOT SURVIVE, and it had no guard:
    # with every player height 0 the feature was inf everywhere, the histogram was
    # uniform, log p(d) was a constant, and argmax(log conf + const) is argmax(log
    # conf) - so the PRIOR arm would have scored EXACTLY the baseline and N58
    # would have published its own idea as a null result.
    if ds.size < 1000:
        raise SystemExit(f"VOID: the fit set has {ds.size} usable ball frames. A "
                         f"uniform prior makes PRIOR identical to the baseline by "
                         f"construction, which is a null result dressed as a test.")
    hist, _ = np.histogram(np.clip(ds, BINS[0], BINS[-1] - 1e-9), bins=BINS, density=False)
    p = (hist + 1) / (hist.sum() + len(hist))
    json.dump({"clips": len(clips), "n": int(ds.size), "bins": BINS.tolist(),
               "p": p.tolist(), "median_d": float(np.median(ds)) if ds.size else None},
              open(OUT, "w"), indent=1)
    print(f"fit on {len(clips)} TRAIN clips, {ds.size} ball frames with players, "
          f"median d = {np.median(ds):.2f} body heights -> {os.path.relpath(OUT, ROOT)}")
    return p


def logp(d, p):
    i = int(np.clip(np.digitize(d, BINS) - 1, 0, len(p) - 1))
    return float(np.log(max(p[i], FLOOR)))


PCACHE = os.path.join(ROOT, "model", "validation", "out", "ball_clean", "n58_players")


def player_cache(clip, n_frames=None, rebuild=False):
    """{frame: [(foot_x, foot_y, height)]} from the PRODUCT's player detector.

    coreml, conf 0.5, tile 1024 - `jobs.py`'s configuration, because an arm run at a
    threshold the product does not ship measures a configuration nobody has
    (registry-measures-configs-that-do-not-ship (internal note)). Cached for the same
    reason ball_eval caches candidates: the detector is the bill, the picker is free.

    The point is the BOX BOTTOM-CENTRE, which is what teamtrack_tracks.load calls x,y
    and therefore what `fit()` measured p(d | ball) against. A centre here and a foot
    there would make the scored feature a different quantity from the fitted one, and
    the arm would be measuring that mismatch.
    """
    import gzip  # noqa: PLC0415
    os.makedirs(PCACHE, exist_ok=True)
    path = os.path.join(PCACHE, f"{clip}.json.gz")
    if os.path.exists(path) and not rebuild:
        with gzip.open(path, "rt") as fh:
            d = json.load(fh)
        return {int(k): [tuple(t) for t in v] for k, v in d["frames"].items()}, d
    import cv2  # noqa: PLC0415
    import detectors  # noqa: PLC0415
    import ball_eval as BE  # noqa: PLC0415
    video = str(BE.FOOTAGE2 / f"{clip}.mp4")
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = total if n_frames is None else min(n_frames, total)
    det = detectors.get_detector("coreml", conf=0.5, tile=1024)
    out = {}
    for f in range(want):
        ok, frame = cap.read()
        if not ok:
            break
        rows = []
        for d in det.detect(frame):
            if d["cls"] not in detectors.PLAYER_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in d["xyxy"])
            rows.append((round((x1 + x2) / 2, 2), round(y2, 2), round(y2 - y1, 2)))
        out[f] = rows
        if f % 50 == 0:
            print(f"  {clip[-9:]} frame {f}/{want}, {len(rows)} players", flush=True)
    meta = {"clip": clip, "detector": "coreml", "conf": 0.5, "tile": 1024,
            "n_frames": len(out), "frames": {str(k): v for k, v in out.items()}}
    with gzip.open(path, "wt") as fh:
        json.dump(meta, fh)
    return out, meta


def gt_player_cache(clip):
    """{frame: [(foot_x, foot_y, height)]} from ground truth, for ORACLE-P.

    Rebased exactly as fit() rebases: ball_rows() zeroes the ball's frames and
    load_tracks does not, so a raw join is off by one and drops frame 0.
    """
    import ball_clean as B  # noqa: PLC0415
    csv = str(B.STORE / f"{clip.replace('videos', 'annotations')}.csv")
    tr = B.load_tracks(csv)
    base = min((r["f"] for v in tr.values() for r in v), default=0)
    by_f = collections.defaultdict(list)
    for v in tr.values():
        for r in v:
            by_f[r["f"] - base].append((r["x"], r["y"], r["h"]))
    return by_f


def prior_pred(cache, p, players):
    """argmax of log(conf) + log p(d | ball) over the plausible candidates.

    The gates are the same as CONF's - `plausible()` only - so PRIOR minus CONF is
    the prior term and nothing else.
    """
    import ball as ballmod  # noqa: PLC0415
    H = cache["frame_h"]
    out = {}
    for f in sorted(cache["frames"]):
        keep = [r for r in cache["frames"][f] if ballmod.plausible(tuple(r[:4]), H)]
        if not keep:
            out[f] = None
            continue
        pl = players.get(f, [])
        # No late-binding defaults: `sc` would be called inside this iteration either
        # way, and the defaults read to lint_truth as a constant rebound per frame.
        b = max(keep, key=lambda r: np.log(max(r[4], 1e-9))
                + logp(feature((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, pl), p))  # noqa: B023
        out[f] = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return out


def decoy(suffix=".cmpD"):
    """POST-HOC, NO BAR: does d separate the ball from the decoys at all?

    Added after score() lost, and labelled as post-hoc because it is. It is not a new
    arm and it does not revise the verdict - it asks what the rule could not: the fit
    estimated p(d | ball) and the PRIOR arm added log p(d | ball) to log(conf) as
    though that were a likelihood RATIO. A ratio needs p(d | decoy) too, and nothing
    in this file ever estimated it.

    AUC is exact, by ranks (Mann-Whitney), so there is no sample and no seed.
    """
    import ball as ballmod  # noqa: PLC0415
    import ball_cmp as C  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    bd, dd = [], []
    for clip in C.CLIPS:
        cache = B.load_cache(clip + suffix)
        frames = sorted(cache["frames"])
        det, _m = player_cache(clip, n_frames=max(frames) + 1)
        gt, _wh, _n = B.load_ball(clip)
        H = cache["frame_h"]
        for f in frames:
            g = gt.get(f)
            if g is None:
                continue
            pl = det.get(f, [])
            for r in cache["frames"][f]:
                if not ballmod.plausible(tuple(r[:4]), H):
                    continue
                cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
                d = feature(cx, cy, pl)
                if not np.isfinite(d):
                    continue
                (bd if np.hypot(cx - g[0], cy - g[1]) <= B.GATE_PX else dd).append(d)
    b, dc = np.array(bd), np.array(dd)
    # exact AUC of "closer to a player" as ball evidence, ties at 0.5
    both = np.concatenate([-b, -dc])
    order = np.argsort(both, kind="stable")
    ranks = np.empty(len(both), float)
    sb = both[order]
    i = 0
    while i < len(sb):
        j = i
        while j + 1 < len(sb) and sb[j + 1] == sb[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    auc = (ranks[:len(b)].sum() - len(b) * (len(b) + 1) / 2) / (len(b) * len(dc))
    print(f"  true ball  n={len(b)}  median d {np.median(b):.2f}  "
          f"p10 {np.percentile(b, 10):.2f}  p90 {np.percentile(b, 90):.2f}")
    print(f"  decoys     n={len(dc)}  median d {np.median(dc):.2f}  "
          f"p10 {np.percentile(dc, 10):.2f}  p90 {np.percentile(dc, 90):.2f}")
    print(f"  AUC of 'closer to a player' as ball evidence: {auc:.4f}  "
          f"(0.5 = no information; N57 measured log conf at 0.8895)")
    return 0


def score(suffix=".cmpD", build=False):
    """The rule in this file's docstring, run. n=1 configuration."""
    import ball_clean as BC  # noqa: PLC0415
    import ball_cmp as C  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    import n57_conf_evidence as N57  # noqa: PLC0415

    if not os.path.exists(OUT):
        raise SystemExit("no prior on disk; run --fit first")
    pr = json.load(open(OUT))
    p = np.array(pr["p"])

    # VOID (a): the fit set may not touch the scored clips.
    fitc = {c for c in BC.on_disk() if BC.role(*BC._parts(c)) == "train"}
    if fitc & set(C.CLIPS):
        print(f"  VOID (a): {len(fitc & set(C.CLIPS))} fit clip(s) are scored clips")
        return 1
    print(f"  VOID (a) ok: {len(fitc)} fit clips, 0 of them scored "
          f"(role() calls the six 'scored' and also excludes what abuts them)")

    pop = {c: v[0] for c, v in N57.split().items()}
    arms = ("shipped", "conf", "prior", "oracle-p", "oracle", "centre")
    acc = {a: collections.defaultdict(int) for a in arms}
    per = {a: {} for a in arms}
    for clip in C.CLIPS:
        cache = B.load_cache(clip + suffix)
        frames = sorted(cache["frames"])
        gt, _wh, _n = B.load_ball(clip)
        gt_here = {f: gt[f] for f in frames if f in gt}
        # max(frames)+1, not len(frames): player_cache's loop is `for f in range(want)`,
        # so a COUNT is only the right range end when the frames start at 0 and are
        # contiguous. A resumed cache covering 100-399 would mine players for 0-299 and
        # leave a hundred frames playerless.
        want = max(frames) + 1
        det_pl, _meta = player_cache(clip, n_frames=want, rebuild=bool(build))
        preds = {"shipped": B.predict(cache, "shipped", gt_here, None),
                 "conf": B.predict(cache, "conf", gt_here, None),
                 "oracle": B.predict(cache, "oracle", gt_here, None),
                 "centre": B.predict(cache, "centre", gt_here, None),
                 "prior": prior_pred(cache, p, det_pl),
                 "oracle-p": prior_pred(cache, p, gt_player_cache(clip))}
        # VOID (c) USED TO COMPARE sorted(preds[a]) TO frames. Every predictor iterates
        # sorted(cache["frames"]) and assigns a key per frame, so that comparison was
        # identically False and could not fail whatever broke. The quantity that mattered
        # is whether the PLAYER boxes cover the scored frames: prior_pred does
        # players.get(f, []), and a frame with no player row yields d = inf -> the tail
        # bin -> a per-frame constant, so PRIOR silently collapses to CONF there and the
        # arm is biased toward the null with no count reported.
        missing = [f for f in frames if not det_pl.get(f)]
        if missing:
            print(f"  VOID (c): {len(missing)} of {len(frames)} frames on {clip[-9:]} "
                  f"have no detected player, so PRIOR = CONF on them")
            return 1
        for a in arms:
            assert sorted(preds[a]) == frames, (a, len(preds[a]), len(frames))
        for a in arms:
            sc_ = B.score(gt_here, preds[a], frames, B.GATE_PX)
            per[a][clip] = sc_
            for k in ("correct", "gt_present", "wrong", "silent"):
                acc[a][k] += sc_[k]

    ctr = acc["centre"]["correct"] / acc["centre"]["gt_present"]
    print(f"  VOID (b) centre {ctr:.4f}")
    if ctr > 0.01:
        print("  VOID: the centre control scores over 1%")
        return 1
    sh = acc["shipped"]["correct"] / acc["shipped"]["gt_present"]
    # VOID (d) IS CONFIGURATION-SPECIFIC. It exists to catch a harness that is not
    # reproducing the known baseline, and the known baseline depends on the INPUT
    # RESOLUTION - which N65 found had never been passed: .cmpD is D1 at 384 (the default)
    # and .cmpDr is the same checkpoint at the 640 it was trained at, worth +0.1144 on
    # this very number. A single hardcoded 0.4706 would refuse the corrected
    # configuration as if it were drift.
    want = BASELINE.get(suffix)
    if want is None:
        print(f"  VOID (d): no published baseline for suffix {suffix}")
        return 1
    print(f"  VOID (d) shipped {sh:.4f} vs {want} (the published baseline for {suffix})")
    if abs(sh - want) > 5e-4:
        print(f"  VOID: shipped does not reproduce {want}")
        return 1

    dead = [c for c in C.CLIPS if pop.get(c) == "dead"]
    play = [c for c in C.CLIPS if pop.get(c) != "dead"]

    def pooled(a, clips):
        c = sum(per[a][k]["correct"] for k in clips)
        n = sum(per[a][k]["gt_present"] for k in clips)
        return c / n if n else 0.0

    print(f"\n  n=1 configuration; prior fitted on {pr['clips']} clips, "
          f"{pr['n']} ball frames, median d {pr['median_d']:.2f} body heights")
    print("  arm        pooled     dead   in-play    wrong  silent")
    for a in arms:
        print(f"  {a:10s} {pooled(a, C.CLIPS):.4f}   {pooled(a, dead):.4f}   "
              f"{pooled(a, play):.4f}    {acc[a]['wrong']:5d}  {acc[a]['silent']:5d}")

    conf, pri = pooled("conf", C.CLIPS), pooled("prior", C.CLIPS)
    bar = conf + 0.05
    d_ok = pooled("prior", dead) - pooled("shipped", dead) >= -0.01
    p_ok = pooled("prior", play) - pooled("shipped", play) >= -0.01
    print(f"\n  PRIOR {pri:.4f} vs bar CONF+0.05 = {bar:.4f}  (delta {pri - bar:+.4f})")
    print(f"  dead {pooled('prior', dead):.4f} vs shipped {pooled('shipped', dead):.4f}   "
          f"in-play {pooled('prior', play):.4f} vs {pooled('shipped', play):.4f}   floor -0.01")
    ok = pri >= bar and d_ok and p_ok
    print(f"  RULE: pooled >= CONF+0.05 AND neither population below -0.01 -> "
          f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"  ORACLE-P {pooled('oracle-p', C.CLIPS):.4f} - the ceiling with GT players, "
          f"so a PRIOR loss under it is this feature, not player tracking")
    return 0


def demo():
    """The feature and the lookup, which is all the arithmetic there is here."""
    # a candidate ON a player's feet is at distance 0; one a body-height away is 1
    assert feature(100.0, 200.0, [(100.0, 200.0, 50.0)]) == 0.0
    assert abs(feature(100.0, 250.0, [(100.0, 200.0, 50.0)]) - 1.0) < 1e-9
    # the NEAREST player wins, and a zero-height box cannot divide by zero
    assert abs(feature(0.0, 0.0, [(300.0, 0.0, 50.0), (100.0, 0.0, 50.0)]) - 2.0) < 1e-9
    assert feature(0.0, 0.0, [(10.0, 0.0, 0.0)]) == np.inf, "a zero-height box must not be used"
    assert feature(0.0, 0.0, []) == np.inf, "no players means no evidence, not distance zero"
    # the lookup is finite everywhere, including outside the binned range
    p = np.full(len(BINS) - 1, 1.0 / (len(BINS) - 1))
    for d in (-1.0, 0.0, 4.0, 1e6, np.inf):
        assert np.isfinite(logp(d, p)), d
    # AND THE REAL LOADER, because the synthetic players above carry heights by
    # construction and so could never have caught the defect that mattered: the
    # player branch of teamtrack_tracks.load did not emit w/h at all, so every
    # body had height 0, feature() returned inf for all of them, and the fit set
    # was empty. A fixture that supplies the field under test cannot test it.
    import ball_clean as _B  # noqa: PLC0415
    _clips = [c for c in _B.on_disk() if _B.role(*_B._parts(c)) == "train"]
    assert _clips, "no TRAIN clips on disk: the synthetic half alone cannot catch the " \
                   "empty-fit-set bug this block exists for"
    if _clips:
        _tr = _B.load_tracks(str(_B.STORE / f"{_clips[0].replace('videos', 'annotations')}.csv"))
        _h = [r["h"] for v in _tr.values() for r in v]
        assert _h and min(_h) > 0, "player rows carry no usable height; the fit would be empty"
        _pf = {r["f"] for v in _tr.values() for r in v}
        _bf = {r["f"] for r in _B.ball_rows(_clips[0])}
        assert min(_pf) - min(_bf) in (0, 1), \
            f"unexpected ball/player frame offset {min(_pf) - min(_bf)}; the join would be wrong"
    print("n58 demo: feature, lookup and the REAL loader's player rows ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--fit" in a:
        fit()
    elif "--decoy" in a:
        raise SystemExit(decoy())
    elif "--score" in a:
        # REFUSE, do not fall through to demo(). The usage block advertises
        # --score; before this it printed "demo ok" and exited 0, so the one
        # command a reader would run to check for a result returned success
        # having measured nothing.
        raise SystemExit(score(suffix=".cmpDr" if "--at640" in a else ".cmpD",
                               build="--rebuild" in a))
    else:
        demo()
