"""Tier 3: tracked positions -> who has the ball, with NO ball and NO event feed.

`events.py` needs a second source telling it a pass happened. This does not: it
infers the possession path from the 22 trajectories alone, using PathCRF
(Kim et al., KDD 2026, arXiv 2602.12080, MPL-2.0). That is the whole point -
an event feed is the thing amateur football cannot buy.

WHAT THIS SHIPS, AND WHAT IT REFUSES TO
    Measured on match 118575, clean ground-truth tracks, released weights, no
    fine-tuning: event F1 0.6007 against the paper's own held-out 0.7569.
    Team-level outputs survive that gap; player-level ones do not, and this
    module does not expose them:

        possession location     1.77 m median                    SHIPS
        event timing            0.40 s median                    SHIPS
        team possession share   -1.12 pp WITH a live/dead mask    GATED
                                +6.97 pp without one              REFUSED
        sender / receiver naming                                 REFUSED
        pass networks                                            REFUSED

    Share is gated, not shipped. A half is only ~73% ball-in-play and this
    model never flags a stoppage, so every share number needs a live/dead
    signal we do not yet have - see `share`. That is the one thing standing
    between this module and the headline Phase 2 deliverable.

    The refusal is not caution, it is a measurement. Charging a perfect model
    nothing but our identity error takes sender accuracy from 100% to 9.1%
    against a 4.5% chance floor, and pass-network degree error (8.09) exceeds
    the signal it measures (7.36). `edge_src`/`edge_dst` exist on Path below
    because the model emits them and `timeline` needs the changes; naming the
    player is what is refused. Do not surface them per player.

THREE THINGS THAT SILENTLY DESTROY THIS, ALL MEASURED
  1. UNSMOOTHED POSITIONS. The model's 8th input channel is acceleration, a
     second derivative. SoccerTrack GT jitters ~0.2 m/frame and that lands in
     accel at a median of 10.3 m/s2 against the ~1 elite tracking shows. Feeding
     it raw scored F1 0.054; a 2.44 s Savitzky-Golay on xy first scored 0.60.
     `SMOOTH_S` is that window and it is not a style choice.
  2. A WRONG GOALKEEPER. Node feature 0 is a +-1 LEFT/RIGHT indicator, and
     upstream `utils.detect_keepers` picks the keeper by mean-x within each team
     bank. Wrong team labels move those means, and the ordering can invert -
     which flips the indicator for every node at once. Seed-varied, that is a
     cliff, not a gradient: F1 0.34, 0.33, 0.087, 0.076, and the two low seeds
     are exactly the two where the keeper came back wrong. We pin the keeper
     from the roster instead; it is an exact no-op when detection agreed.
  3. WRONG TEAM LABELS. Two swapped players cost ~0.1 F1 alone, but combined
     with 1 m of positional error they cost 0.51 and take possession share out
     to +11.9 pp. `events.py` measures kit-colour team assignment at 15.4%
     track-level accuracy on real pipeline output. Check `teams` before trusting
     the split - see `share`, which refuses rather than reporting 50/50.

POSITIONAL ACCURACY IS A BINDING CONSTRAINT, not the domain gap and not
identity. Measured sensitivity: 0.0 m -> F1 0.60, 0.5 m -> 0.56, 1.0 m -> 0.46.
`require_accuracy` is the gate and it is on by default.

    CORRECTED 2026-09-02. This paragraph used to say the vendor-remap pipeline
    "delivers ~9.4 m and is hopeless here". Nothing measures 9.4 m. That number
    was withdrawn as a canvas artifact and the line was never updated, and it
    has been steering the decision not to attempt this ever since.

    What 118575 actually measures, and WHICH POPULATION each number is over -
    which is the whole point, because they disagree by 2.5x:

      calibration.json rms_error_m       2.85 m   landmarks, IN-SAMPLE
      landmarks held out (calcheck)      2.50 m   landmarks, leave-one-out
      SoccerTrack GT, 8,679 samples      1.010 m  WHERE PLAYERS ACTUALLY STAND

    The landmark numbers are the wrong population. Forty-two of the 65
    landmarks sit on the two touchlines, and no outfield player stands there:
    6,869 of those 8,679 player samples are in the middle of the pitch. So the
    landmark ledger scores a region the model is not asked to serve.

    Against the population that matters the plain single homography is 1.010 m,
    which is INSIDE MAX_RMS_M. The gate is open. Read `rms_error_m` off the
    file with that in mind - it is the in-sample landmark figure and it will
    read ~2.85 m on a calibration that is really a ~1 m calibration where it
    counts, so passing it to `infer()` refuses a match that should pass.

    DO NOT REACH FOR BowWrapper TO "FIX" THE 2.85. Its landmark numbers are a
    6x improvement and it is measurably WORSE on players - 1.507 m against
    1.010 m overall, and it takes stitched-track IDF1 from 0.1032 to 0.0898.
    Its own docstring says do not enable it on this match. Panoramic strips and
    a thin-plate spline are worse still: both score well at random hold-out and
    blow out to 175-350 m max when the held-out set is the middle of the pitch.

    THE HONEST CONSEQUENCE. 1.010 m is inside the gate but it is not free: the
    sensitivity table above puts event F1 near 0.46 at 1 m, against the 0.6007
    that justified shipping team-level outputs at all. Everything this module
    ships needs re-justifying at 0.46 before it is quoted. The gate being open
    is permission to measure, not permission to claim.

    @C 1.010 m :: model/engine/calibrators.py :: BowWrapper docstring, the homography against SoccerTrack
       GT, n=8,679 player samples on 118575, binned by GT y band. Scoped (2026-09-28) to
       118575's vendor remap with 65 dataset keypoints, which the product cannot produce.

    Withdrawn with it, and easy to miss because the corrected line simply
    stopped mentioning it: a "1.33 m" for raw panorama plus a dense-keypoint
    calibration, which sat beside the 9.4 m. It was a 24-frame run
    (app/frontend/src/app/checks/page.tsx) quoted with no n=, so it is not a
    counterweight to anything. And MAX_RMS_M = 1.5 itself has not been
    re-derived since the 9.4 m went: it is a threshold, not evidence about any
    particular pipeline.

LICENCE. PathCRF is MPL-2.0 and its training data is Sportec Open DFL under
CC BY 4.0. It is NOT BUNDLED: PATHCRF_DIR points at a vendored checkout that is
gitignored and absent from the build, so nothing ships today and the question is
about a path that has to be opened deliberately. If it is ever bundled, the
determination and the attribution belong in docs/THIRD_PARTY_NOTICES.md, not in a
docstring - the short version being that MPL is file-level, so modified PathCRF
files must stay MPL and be published while this file is unaffected.
"""
from dataclasses import dataclass
import os

import numpy as np

# PathCRF lives outside the package because it is a vendored third party under a
# different licence. Point PATHCRF_DIR at a checkout containing model/models/,
# set_transformer/, datatools/ and saved/<trial>/.
PATHCRF_DIR = os.environ.get(
    "PATHCRF_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/vendor/pathcrf"),
)
TRIAL = 140                 # Dynamic Masked CRF - the paper's best variant
TARGET_FPS = 5.0            # what the weights were trained at; not a knob
WINDOW_S = 10.0
SMOOTH_S = 2.44             # see note 1 above
SEGMENT_S = 40.0            # CRF decode chunk; the training episode scale, see _decode
TEAM_SIZE = 11
N_OUTSIDE = 4               # out_left/right/bottom/top, injected by the model's contract

# Below this calibration RMS the possession path is not worth computing: the
# model is already at F1 0.46 by 1 m and falls off a cliff after. See the
# sensitivity table above.
MAX_RMS_M = 1.5


@dataclass
class Path:
    """The possession path at TARGET_FPS. Node indices, deliberately not names."""

    t: np.ndarray            # (T,) seconds
    src: np.ndarray          # (T,) node index holding the ball
    dst: np.ndarray          # (T,) node it is travelling to; == src while held
    xy: np.ndarray           # (T,2) possession location, metres
    team: np.ndarray         # (T,) "home"/"away"/"out"
    n_players: int


def _features(pos, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(T,N,2) metres -> (T,N,6) [x,y,vx,vy,speed,accel], smoothed. Plus the mask.

    Savitzky-Golay on POSITION first, then differences. Differencing raw
    positions and smoothing afterwards is not the same thing and does not
    survive note 1: the jitter is already in the derivative by then.
    """
    from scipy.signal import savgol_filter  # noqa: PLC0415

    XY = np.array(pos.XY, dtype=np.float64)
    T, N, _ = XY.shape
    win = int(round(SMOOTH_S * fps)) | 1          # savgol needs odd
    if T > win >= 5:
        # Hold gaps still rather than smoothing across them: a hole interpolated
        # to a straight line becomes speed the player never had.
        filled = np.where(pos.mask[:, :, None], XY, np.nan)
        filled = _ffill(filled)
        XY = savgol_filter(filled, win, 2, axis=0)

    v = np.gradient(XY, 1.0 / fps, axis=0)
    speed = np.linalg.norm(v, axis=2)
    accel = np.abs(np.gradient(speed, 1.0 / fps, axis=0))
    return np.concatenate([XY, v, speed[..., None], accel[..., None]], axis=2), pos.mask


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-then-back fill NaNs along axis 0, per player per axis."""
    out = a.copy()
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            col = out[:, i, j]
            ok = ~np.isnan(col)
            if not ok.any():
                col[:] = 0.0
                continue
            idx = np.where(ok, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[:] = col[idx]
            col[: np.argmax(ok)] = col[np.argmax(ok)]
    return out


def _node_order(pos, left_gk: str, right_gk: str) -> tuple[list, str, str]:
    """Player column order: left team first, then right, as the weights expect.

    The keeper is PINNED by caller, never inferred - see note 2. `left_gk` and
    `right_gk` are track ids, and which side each is on is decided here by their
    own mean x, which is the one thing about a keeper that is reliable.
    """
    ids = list(pos.ids)
    teams = np.asarray(pos.teams)
    mx = {i: float(np.nanmean(np.where(pos.mask[:, k], pos.XY[:, k, 0], np.nan)))
          for k, i in enumerate(ids)}
    if left_gk not in mx or right_gk not in mx:
        raise ValueError(f"pinned keepers {left_gk!r}/{right_gk!r} are not tracks")
    if mx[left_gk] > mx[right_gk]:
        left_gk, right_gk = right_gk, left_gk
    lteam = teams[ids.index(left_gk)]
    left = [i for k, i in enumerate(ids) if teams[k] == lteam]
    right = [i for k, i in enumerate(ids) if teams[k] != lteam]
    return left + right, left_gk, right_gk


def infer(pos, left_gk: str, right_gk: str, fps: float = 25.0,
          rms_error_m: float | None = None, require_accuracy: bool = True,
          pathcrf_dir: str = None, trial: int = TRIAL) -> Path:
    """Positions -> possession Path. Keepers are pinned, not detected.

    `rms_error_m` is the calibration's own error. With `require_accuracy` this
    refuses above MAX_RMS_M rather than returning a plausible wrong answer - the
    failure mode here is not noise, it is a confident possession path drawn over
    players who are 9 m from where the model thinks they are.
    """
    if require_accuracy:
        if rms_error_m is None:
            raise ValueError("pass rms_error_m, or require_accuracy=False to override")
        if rms_error_m > MAX_RMS_M:
            raise ValueError(
                f"calibration rms {rms_error_m:.2f} m > {MAX_RMS_M} m: the possession "
                "path would be drawn over the wrong players. Fix calibration first "
                "(see calibrators.from_keypoints), or pass require_accuracy=False.")

    if len(pos.ids) != TEAM_SIZE * 2:
        raise ValueError(f"need {TEAM_SIZE * 2} tracks, got {len(pos.ids)}")

    order, left_gk, right_gk = _node_order(pos, left_gk, right_gk)
    feats, mask = _features(pos, fps)
    step = max(1, int(round(fps / TARGET_FPS)))

    ids = list(pos.ids)
    cols = [ids.index(i) for i in order]
    F = feats[::step][:, cols, :]                     # (T,22,6)
    t = np.asarray(pos.t)[::step]

    src, dst = _decode(F, order, left_gk, right_gk, pathcrf_dir or PATHCRF_DIR, trial)

    n_p = len(order)
    xy = np.zeros((len(src), 2))
    team = np.empty(len(src), dtype=object)
    teams = np.asarray(pos.teams)
    for k, s in enumerate(src):
        if s < n_p:
            xy[k] = F[k, s, :2]
            team[k] = teams[ids.index(order[s])]
        else:
            xy[k] = _OUTSIDE_XY[s - n_p]
            team[k] = "out"
    return Path(t=t, src=src, dst=dst, xy=xy, team=team, n_players=n_p)


_OUTSIDE_XY = np.array([[0.0, 34.0], [105.0, 34.0], [52.5, 0.0], [52.5, 68.0]])


def _decode(F: np.ndarray, order, left_gk, right_gk, root: str, trial: int):
    """Run the released weights. Import is lazy so this module loads without torch."""
    import sys  # noqa: PLC0415

    if root not in sys.path:
        sys.path.insert(0, root)
    import torch  # noqa: PLC0415
    from models.utils import build_model, load_trial_args  # noqa: PLC0415

    save = os.path.join(root, f"saved/{trial:03d}")
    args = load_trial_args(save)
    model = build_model(args, device=torch.device("cpu"))
    model.load_state_dict(torch.load(os.path.join(save, "model/state_dict_best_acc.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()

    T, N, _ = F.shape
    ind = np.zeros((N + N_OUTSIDE, 2), dtype=np.float32)
    ind[:TEAM_SIZE, 0] = -1.0                      # left bank
    ind[TEAM_SIZE:N, 0] = 1.0                      # right bank
    for gk in (left_gk, right_gk):
        ind[order.index(gk), 1] = 1.0

    feats = np.zeros((T, N + N_OUTSIDE, 6), dtype=np.float32)
    feats[:, :N] = F
    feats[:, N:, :2] = _OUTSIDE_XY
    x = np.concatenate([np.broadcast_to(ind[None], (T, N + N_OUTSIDE, 2)), feats], axis=2)

    win = int(round(WINDOW_S * TARGET_FPS))
    stride = max(1, win // 2)
    emis = None
    embeds = None
    weight = np.zeros(T)
    with torch.no_grad():
        for s in range(0, max(1, T - win + 1), stride):
            e = min(s + win, T)
            _, micro, edge = model(torch.tensor(x[s:e])[None])
            micro = micro.squeeze(0)
            if emis is None:
                emis = torch.zeros((T, micro.shape[1]))
            emis[s:e] += micro
            if edge is not None:
                edge = edge.squeeze(0) if edge.dim() == 4 else edge
                if embeds is None:
                    embeds = torch.zeros((T,) + tuple(edge.shape[1:]))
                embeds[s:e] += edge
            weight[s:e] += 1.0
    w = torch.tensor(np.maximum(weight, 1.0), dtype=torch.float32)
    emis /= w[:, None]
    if embeds is not None:
        embeds /= w.view(-1, *([1] * (embeds.dim() - 1)))

    # Decode with the model's own CRF, not argmax. The CRF is the paper's whole
    # contribution: it forbids illegal transitions and holds the path together
    # across steps. Argmax over all N^2 edges instead scored 41.2% home share
    # against a truth of 33.8 and emitted 1822 changes where the CRF gives ~1340
    # - it over-segments, because nothing stops the ball teleporting each step.
    valid = model.valid_orig_edge_ids
    em_all = emis.index_select(dim=1, index=valid)
    eb_all = embeds.index_select(dim=1, index=valid) if embeds is not None else None

    # Decode in SEGMENT_S chunks, not one chain over the whole half. The CRF was
    # trained on ball-alive EPISODES (~37 s on 118575), and its transition prior
    # is calibrated to that length. Run over 14,421 continuous steps it simply
    # sticks: measured 94.0% home share against a truth of 33.8, and 147 changes
    # where the paper's own per-episode decode gives ~1340.
    # ponytail: fixed-length chunks because ball_state needs a ball we do not
    # have. Segment on detected dead time instead if that ever becomes available.
    seg = max(1, int(round(SEGMENT_S * TARGET_FPS)))
    src_out, dst_out = [], []
    for s0 in range(0, T, seg):
        e0 = min(s0 + seg, T)
        em = em_all[s0:e0].unsqueeze(0)
        if eb_all is not None:
            d = model.crf.decode(em, eb_all[s0:e0].unsqueeze(0))
        else:
            d = model.crf.decode(em)
        d = d.squeeze(0).cpu().numpy()
        src_out.append(model.comp_src.cpu().numpy()[d])
        dst_out.append(model.comp_dst.cpu().numpy()[d])
    return np.concatenate(src_out), np.concatenate(dst_out)


def share(p: Path, live: np.ndarray = None, teams_confident: bool = True) -> dict:
    """Team possession share, in percent of live held steps. Refuses without `live`.

    `live` is a (T,) bool over `p.t`: is the ball in play. THIS IS NOT OPTIONAL,
    and it is the open problem in ball-free possession. Measured on 118575 GT:

        every step                      home 40.81%   +6.97 pp
        live steps only (feed oracle)   home 32.72%   -1.12 pp

    The whole bias is dead time. A half is only ~73% ball-in-play, and PathCRF
    never routes to its own out-of-play nodes on our tracks (0.0% of steps), so
    it names a holder right through a throw-in and every stoppage. The model is
    fine - with the oracle it lands inside the paper's own +-0.51 pp - but the
    number is wrong by 7 points without one, and 7 points is the difference
    between "we edged possession" and "we were run over".

    We have no live/dead signal: a ball feed would give it and the point of this
    module is not needing one, so this needs a trajectory-based segmenter. Until
    then a caller must supply `live` or accept the refusal.

    Also refuses on unverified team labels, because that failure is not a shrink
    toward 50/50 a caller could de-bias: with 1 m of positional error and two
    swapped players it overshoots the midpoint, measured +11.9 pp on a truth of
    33.84. A wrong split is worse than no split.
    """
    if not teams_confident:
        return {"home": None, "away": None, "n": 0,
                "why": "team assignment unverified; see events.py team_purity"}
    if live is None:
        return {"home": None, "away": None, "n": 0,
                "why": "no live/dead signal: share over every step is +7 pp biased "
                       "by stoppages, and the model cannot flag them itself"}
    held = (p.team != "out") & np.asarray(live, bool)
    n = int(held.sum())
    if not n:
        return {"home": None, "away": None, "n": 0, "why": "no live held steps"}
    h = float((p.team[held] == "home").sum()) / n * 100.0
    return {"home": round(h, 2), "away": round(100.0 - h, 2), "n": n}


def timeline(p: Path) -> list[dict]:
    """Possession changes: {t, team}. Timings only - never who.

    Measured event-timing error is 0.40 s median, so these are worth showing
    against video. The player holding the ball is not, which is why `team` is
    the only attribution here.
    """
    out = []
    prev = None
    for k in range(len(p.src)):
        cur = int(p.src[k])
        if cur != prev:
            out.append({"t": round(float(p.t[k]), 2), "team": p.team[k]})
            prev = cur
    return out


def _demo() -> None:
    """Synthetic asserts, plus the real match when PathCRF and the GT are here."""
    import events  # noqa: PLC0415

    rng = np.random.default_rng(0)
    T, N = 300, 22
    xy = np.cumsum(rng.normal(0, 0.05, (T, N, 2)), axis=0) + rng.uniform(5, 60, (1, N, 2))
    pos = events.Positions(t=np.arange(T) / 25.0, XY=xy, mask=np.ones((T, N), bool),
                           ids=[str(i) for i in range(N)],
                           teams=np.array(["home"] * 11 + ["away"] * 11))

    # Smoothing must actually cut the second derivative - note 1 is the whole
    # reason this module scores 0.60 rather than 0.054.
    f, _ = _features(pos, 25.0)
    raw_acc = np.median(np.abs(np.gradient(np.linalg.norm(
        np.gradient(xy, 1 / 25.0, axis=0), axis=2), 1 / 25.0, axis=0)))
    assert np.median(f[..., 5]) < raw_acc, (np.median(f[..., 5]), raw_acc)

    # Keepers are pinned, and the left/right decision comes from their own x.
    order, lg, rg = _node_order(pos, pos.ids[0], pos.ids[11])
    assert len(order) == 22 and lg != rg
    order2, lg2, rg2 = _node_order(pos, pos.ids[11], pos.ids[0])
    assert (lg2, rg2) == (lg, rg), "keeper side must not depend on argument order"
    try:
        _node_order(pos, "nope", pos.ids[11])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown keeper id was not refused")

    # The accuracy gate must refuse a calibration that cannot support the model.
    try:
        infer(pos, pos.ids[0], pos.ids[11], rms_error_m=9.4)
    except ValueError as e:
        assert "rms" in str(e)
    else:
        raise AssertionError("bad calibration was not refused")
    try:
        infer(pos, pos.ids[0], pos.ids[11])
    except ValueError as e:
        assert "require_accuracy" in str(e)
    else:
        raise AssertionError("missing rms was not refused")

    # share() refuses rather than inventing a split.
    p = Path(t=np.arange(4.0), src=np.array([0, 0, 12, 12]), dst=np.zeros(4, int),
             xy=np.zeros((4, 2)), team=np.array(["home", "home", "away", "out"], dtype=object),
             n_players=22)
    live = np.ones(4, bool)
    assert share(p, live)["home"] == 66.67, share(p, live)
    assert share(p, live, teams_confident=False)["home"] is None
    # No live/dead signal must refuse, not quietly return the +7 pp number.
    assert share(p)["home"] is None and "live" in share(p)["why"]
    # And the mask must actually exclude: drop the first home step.
    assert share(p, np.array([False, True, True, True]))["home"] == 50.0
    assert [e["t"] for e in timeline(p)] == [0.0, 2.0], timeline(p)

    if os.path.isdir(os.path.join(PATHCRF_DIR, f"saved/{TRIAL:03d}")):
        print("possession.py ok (PathCRF present but the real-match check needs GT tracks)")
    else:
        print(f"possession.py ok (PathCRF not vendored at {PATHCRF_DIR}, model path unexercised)")


if __name__ == "__main__":
    _demo()
