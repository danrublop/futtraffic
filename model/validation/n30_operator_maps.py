"""N30: which pitch map ships for POSITION AND DISTANCE on stitched panoramas, on an operator's own clicks.

PROTOCOL, COMMITTED BEFORE ANY ARM IS SCORED ON THE OPERATOR'S CLICKS.
Revised before commit after review by the N27 session (NaN handling, D's population, rule-code
pinning, P3's reference, the margin's anchor, null metrics, freeze checks, loc disclosure,
confirmation tolerance). No arm had been scored on any real input when those revisions were made.

WHY
    N27 (docs/experiments/n27-tps-in-product.md) scored the 21-click spline correction on click sets generated
    from ST2's own chain. `camera_models.py --provenance` showed the only human clicks in the repo
    (118575's keypoints) are also the chain's input. So every position figure for any map built from
    clicks agrees with the chain partly by construction. N27 also bundled speed into its ship rule,
    and its review showed no map could pass that rule (N27 §4). N30 asks the position-and-distance
    question alone, on clicks nobody fitted a reference with, and puts the spline and three camera
    models side by side. Research behind the arms: research-stitched-panorama-calibration (not published).

HELD-OUT INPUT
    clicks       one operator (the user), in the product's own calibration screen (frontend setup page,
                 the UI's solver), on an isolated instance (FUTTRAFFIC_DATA=model/validation/out/
                 operator_clicks/data, API :8010; jobs carry upload_complete=False so no worker can
                 start). The operator did not look at ST2 overlays, the data/footage/ keypoint files, or any
                 result. One pass per clip; only a plain mis-click (a wrong landmark) may be corrected.
                 The fit residual the UI shows is the only feedback. No score of any map on these clicks
                 may be computed before the freeze below.
    the screen   Disclosed because it is the instrument. The clicks are taken on the calibration screen
                 as of the commit that carries this paragraph, which seeds every unplaced landmark as a
                 dashed GUESS - from the current fit once four points are down, from a generic side-on
                 view before that - and turns a guess into a click when the operator grabs it. So a
                 landmark the operator agrees with is accepted rather than placed from scratch, which is
                 the automatic-first shape every vendor in research-competitors-vendors (not published) §8.4
                 ships. A guess is never saved on its own: the payload holds only what the operator
                 grabbed or placed. The operator was also shown a schematic of the 21 landmark NAMES on
                 a drawn pitch (no footage, no pixel positions), because the screen's labels call the
                 far touchline "bottom" - a product defect, recorded here as what the operator knew.
                 REVISED 2026-09-17, before any click was saved on this instance (0 of 8 jobs held a
                 calibration.json) and before any arm was scored: the screen now DEFAULTS the venue to
                 105 x 68 m (it was 118 x 72 yd, which the operator had been told to overwrite by hand),
                 folds the landmark list shut, offers "Next match" after a save, and draws a guess as a
                 solid grey dot rather than a dashed ring. Where guesses come from, the solver and the
                 payload are unchanged: only a dot the operator grabbed or placed is saved. An earlier 117092 click set (2026-09-14, the screen before the guesses, saved at
                 the default 107.9 x 65.8 m) is DISCARDED, not frozen: it is a different instrument.
                 Nothing was scored on it.
    DEVIATION    Recorded 2026-09-17, after the clicks and before the freeze; no arm had been scored and
                 the operator had seen no result. "One pass per clip" was broken on two clips. 117092 and
                 128057 were first saved with 20 landmarks each; the operator reopened both to add the
                 missing one (goal_l_br, pen_l_tl) and, asked afterwards, said they also REFINED dots
                 already on the right landmark - not mis-click corrections. The only feedback on screen
                 was the UI's fit RMS, which ROSE on both (8.65 -> 9.46 m, 7.38 -> 7.61 m: freeze.json
                 holds the second figure of each, first_pass/ the first). The dot is drawn about 18 px in
                 radius on these frames and snaps its centre to the pointer when grabbed, so 128057's
                 moves are all inside one dot; 117092's are not. Landmarks added, landmarks moved and the
                 largest move in px, per clip:
                 @M "['goal_l_br']" 18,205.47,"['pen_l_tl']" 10,14.73 :: .venv/bin/python -c "import json;C='model/validation/n30_clicks';[print(s,*(lambda a,b:(sorted(set(b)-set(a)),sum(a[n]!=b[n] for n in a),round(max(max(abs(a[n][0]-b[n][0]),abs(a[n][1]-b[n][1])) for n in a),2)))(json.load(open(f'{C}/first_pass/{s}.calibration.json'))['points'],json.load(open(f'{C}/{s}.calibration.json'))['points'])) for s in ('117092','128057')]"
                    held fixed: the two clips' first saved calibration.json against the frozen one.
                 DECIDED BEFORE THE FREEZE, by the operator: the SECOND-pass clicks are the frozen input on
                 both clips - they are the calibration this screen would have shipped. The first-pass
                 files are committed under model/validation/n30_clicks/first_pass/ for audit and are scored by
                 NOTHING: scoring both and keeping the better is the sweep rule 6 exists to stop. The other
                 six clips are one pass each, byte-identical between the snapshot and the freeze.
                 ADDED AFTER THE RUN, 2026-09-17, from the independent review (audit-n30 (not published) §1): the
                 choice of pass decides the verdict, and that was knowable from the clicks alone before
                 the freeze. On 117092's FIRST-pass clicks every camera arm's fit RMS is over
                 MAX_FIT_RMS_PX (58.6-61.9 px against 44.1-44.6 on the frozen clicks; arms_for, no ground
                 truth) and TPS folds on both passes, so P5 fails for every candidate and the rule's
                 output is DO NOT SHIP. The FISH11 verdict and its confirmation are therefore conditional
                 on the second pass, on the one clip from a different rig. n30-operator-maps (not published) §2.
    clips        N27's eight: 117092 117093 118575 118576 118577 118578 128057 128058. 132877 and 132831
                 are excluded (mirrored feeds). A clip the operator did not calibrate is excluded and
                 listed; fewer than 6 clips -> VOID.
    freeze       `--freeze` refuses any job whose calibration was inherited. It then copies each
                 clip's calibration.json into model/validation/n30_clicks/, records the UI's own fit RMS per
                 clip and each calibration.json's mtime in freeze.json, and writes SHA256SUMS over the click files, freeze.json and each
                 clip's `.p21id` tracks. model/validation/n30_clicks/ is committed before `--run`.

ARMS (n = 5 maps; 4 candidates against what ships; one configuration each; nothing tuned on these clicks)
    SHIP         calibrators.from_dict(payload) exactly as the UI wrote it.
    TPS          from_dict({**payload, "projection_type": "perspective_tps"}): model/engine/calibrators.py
                 TpsCorrected as committed, with smoothing 0 and the solver's kept clicks. Its fold
                 switch-off is a refusal.
    SPH10        camera_models.fit_form("equirect10", ...)
    FISH11       camera_models.fit_form("fisheye11_fx=fy", ...)
    FISH12       camera_models.fit_form("fisheye12", ...)
                 Each camera arm is fitted to the payload's clicks minus the solver's outliers, in raw
                 clip pixels against pitch.resolve_landmarks metres, by the committed LM from 8 pose
                 initialisations, best fit RMS kept. REFUSED on a clip if the fit fails, if its fit RMS
                 exceeds 50 px, or if more than 1% of that clip's scored MOT feet or of its `.p21id` rows
                 map to a non-finite position.
    refusal      A refused arm fails P5 and gets null metrics on that clip, never another arm's numbers.
    DISCLOSED    The three camera forms were picked from probes on the vendor's clicks
                 (camera_models.py --real, 5 forms plus the product's two maps). That choice is not
                 held out; the clicks are.

METRICS (per clip, then pooled)
    A position   dist_eval._feet_pixels: MOT box feet on clip frames divisible by 20, against the joined
                 feed at that frame and player. Median, p90, far-half median (feed y < 34), over the
                 feet finite under that arm. Beside it, not ruled: each arm's median against ST2's chain
                 at the same feet (dist_eval._refmap). The chain is a REFERENCE, not truth. Per clip,
                 the chain's own median against the feed is printed and clips over 2 m are flagged
                 (N27 read 117092 at 4.63 m and 118578 at 2.65 m).
    A2 loc       calib_product.absolute(cals=...) on 118575 and 128057. Reported, not ruled.
    POPULATION   for B, D and speed: `.p21id` rows (N21's product tracks) minus every row that is
                 non-finite under ANY non-refused arm, written as `{seq}.p21idn30.tracks.json`. All
                 arms and both floors read that one file. The share of rows removed is reported per
                 clip. A removed row inside a fragment is a missed frame to the scorer's gap rule, as
                 in every other variant.
    B scale      dist_eval._chord_slope on that population: the 4 s chord slope against the feed, with
                 the median chord-length ratio beside it. The same slope is computed for ST2's chain
                 (_refmap) on the same rows, as the reference scale.
    D distance   dist_eval.evaluate_stc(q, "p21idn30", oracle="drop", our_cal=arm), NO restitch: the map's
                 own effect on the product's tracks. oracle="drop" removes the rows with no IoU owner,
                 as FLOOR's oracle does, so arms and floor score the same rows. Distance per covered
                 metre = sum ours_m / sum gt_matched_m, pooled.
                 FLOOR = the same with oracle="position" (true positions through the same fragments).
                 An arm's distance error is |its ratio - FLOOR|, not |ratio - 1|.
    speed        HSR pooled at own scale, with its floor taken from the same oracle="position" calls.
                 NOT RULED.

RULE (fixed now)
    An arm X PASSES iff all of:
      P1  on every scored clip, X's median at feet against the feed is below SHIP's;
      P2  pooled median, p90 and far-half median of X are not above SHIP's;
      P3  X's chord slope is within [0.90, 1.10] on every clip, AND its median over clips of |s - 1| is
          at most 0.05 above the chain's own median |s_chain - 1| on the same rows. The reason for
          measuring against the chain: box feet read short of the feed even through ST2's chain
          (the N27 review measured a median slope of 0.9557). An absolute 0.05 bound would rule on
          that disagreement, not on the map;
      P4  pooled |distance per covered metre - FLOOR| of X is not above SHIP's;
      P5  X is refused on no clip.
    CHOICE
      - No arm passes: DO NOT SHIP. The map stays SHIP.
      - Only TPS passes: SHIP TPS.
      - A camera arm passes: it replaces TPS only if its pooled median at feet against the feed is at
        least 0.25 m below TPS's, AND below TPS's on all but at most one scored clip. If TPS failed, the
        camera arm is compared against SHIP instead. Otherwise TPS ships (or nothing, if TPS failed).
        The margin is about twice N27's spread in TPS's pooled median across three 5 px click-noise
        seeds, highest and lowest `@C 2.87,2.74 :: docs/experiments/n27-tps-in-product.md §2`
        a spread of `@D 0.13 :: 2.87 - 2.74`
        "All but one" is 7 of 8, a one-sided sign test `@D 0.0352 :: 9 / 256` under no difference. On 6
        scored clips it is 5 of 6, which is weaker (`@D 0.109 :: 7 / 64`), and the result must say so.
      - More than one camera arm clears that: the one with FEWEST parameters ships (SPH10 < FISH11 <
        FISH12). Never the lowest score.
    CONFIRMATION, before anything reaches a user
      P1 and P2 do not depend on tracks. P3 and P4 do. The chosen arm is replayed through the product
      path (track.run with its calibration, then jobs.apply_identity, as N27 C did) on the scored clips,
      and P3-P4 are recomputed on the replayed tracks. Any failure -> DO NOT SHIP. A camera arm first
      needs a product calibrator (a `remap` exported from the fit). That export counts only if its
      `to_pitch` reproduces `Fitted.to_pitch` at the scored feet to a median of 0.1 m or better. The
      export is written after these results; the tolerance is fixed now.
    CONFIRMATION, AS RUN
      Written 2026-09-17, after --run's verdict and before any replay. The paragraph above fixed the
      tolerance and the pass rule; this fixes what it left open, and records one departure from it.
      export     NOT a dense remap, which is what the paragraph above says. The chosen arm is a camera,
                 so it is exported as one: calibrators.FisheyeCamera, projection_type
                 perspective_fisheye - the operator's payload plus the fit's 12 numbers, detection still
                 on the raw frame. A remap would re-detect every clip on flattened frames, and so
                 confirm a different product from the one --run scored. The check is unchanged: the
                 export's to_pitch against Fitted.to_pitch at D._feet_pixels, median at most 0.1 m on
                 every clip, else VOID. engine's inverse is written apart from camera_models', so the
                 check can fail.
      arms       two replays per clip, as N27 C ran them: `.p21`'s detections (D._n27_cache) through
                 ST.track then ST.identify, once under the operator's payload as saved (SHIP, `.p30s` ->
                 `.p30sid`) and once under the export (`.p30f` -> `.p30fid`). `.p21id` is NOT SHIP here:
                 it was tracked under N6's simulated clicks, not the operator's. n=1 configuration per
                 arm, nothing tuned. Both arms keep the boxes N6's pitch polygon gated, as N27 disclosed.
      validity   _check_freeze passes; --run's verdict names a camera arm; no `.p30*` output exists
                 before the run; every replay's detections equal `.p21`'s row for row; the ground
                 truth is identical between the two arms on every clip.
      P3         as in the RULE, X's slope from D._chord_slope on `p30fid`, the chain's on the same rows.
      P4         each arm against ITS OWN floor, because each replay has its own fragments:
                 |X ratio - FLOOR(p30fid)| <= |SHIP ratio - FLOOR(p30sid)|, pooled, oracle="drop" for
                 the ratios and oracle="position" for the floors, as in D.
      reported   SHIP's slopes; HSR at own scale and its floor, per arm; the share of replayed rows
                 where the export fell back to the homography. Not ruled.
      verdict    CONFIRMED iff valid, P3 and P4. Anything else is DO NOT SHIP.

VALIDITY (VOID unless)
    SHA256SUMS is committed in HEAD and every file matches it; model/validation/n30_operator_maps.py,
    model/validation/camera_models.py, model/validation/dist_eval.py and model/engine/calibrators.py have no uncommitted
    change (the rule that runs is the committed rule); every payload loads; the ground truth (gt_m, gt
    HSR) is identical across all arms per clip; at least 6 clips are scored.

WHAT FAILING LOOKS LIKE, written first
    - SHIP beats everything on a clip where the operator's clicks are poor: P1 fails there. Report that
      clip's UI fit RMS from freeze.json; never drop the clip.
    - Camera arms beat TPS against the chain but not against the feed. That is the chain-family bias the
      N27 review found (the spline follows the chain). Rule on the feed only.
    - The feed may itself have been produced through ST2's chain; nothing documented says how `loc`
      was made. If it was, a fisheye form resembling the chain agrees with the feed partly by
      construction, and ruling on the feed does not remove that. A fisheye arm winning by a small
      margin over SPH10 or TPS is the reading this would produce.
    - A camera fit that lands well at feet but reads a slope near 0.95 on every clip is inheriting the
      chain's scale disagreement with the feed. P3 allows it; report it.

    .venv/bin/python model/validation/n30_operator_maps.py --freeze   # after the clicks, before --run
    .venv/bin/python model/validation/n30_operator_maps.py --run      # ~15 min
    .venv/bin/python model/validation/n30_operator_maps.py --confirm  # after --run: the chosen arm, replayed
    .venv/bin/python model/validation/n30_operator_maps.py --demo     # self-check, no clicks needed
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(ROOT, "model", "engine")]

import camera_models as CM  # noqa: E402

SEQS = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")
DATA = os.path.join(ROOT, "model/validation/out/operator_clicks/data")
CLICKS = os.path.join(ROOT, "model/validation/n30_clicks")
SUMS = os.path.join(CLICKS, "SHA256SUMS")
OUT = os.path.join(ROOT, "model/validation/out/stc2025/n30_results.json")
PINNED = ("model/validation/n30_operator_maps.py", "model/validation/camera_models.py",
          "model/validation/dist_eval.py", "model/engine/calibrators.py")
CAMERA_ARMS = {"SPH10": "equirect10", "FISH11": "fisheye11_fx=fy", "FISH12": "fisheye12"}  # fewest params first
ARMS = ("SHIP", "TPS", *CAMERA_ARMS)
VARIANT = "p21idn30"
MAX_FIT_RMS_PX = 50.0
MAX_NAN_SHARE = 0.01
MIN_CLIPS = 6
MARGIN_M = 0.25
SLOPE_BAND = (0.90, 1.10)
SLOPE_OVER_CHAIN = 0.05



def _git_env():
    """verify_claims.git_env - the environment MINUS every GIT_*.

    Without it an exported GIT_DIR (a hook, or any shell started inside one)
    BEATS cwd=, so this guard reads the OTHER tree's HEAD, where this path is
    usually clean - returncode 0, "registered", green. It fails SHORT, in the
    direction that serves an unregistered run as a registered one.
    a-green-gate-is-not-evidence-it-looked (internal note)
    """
    import sys as _sys  # noqa: PLC0415
    _sys.path.insert(0, os.path.join(ROOT, "model", "validation"))
    import verify_claims as _V  # noqa: PLC0415
    return _V.git_env()


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def freeze():
    import dist_eval as D  # noqa: PLC0415
    jobs = json.load(open(os.path.join(DATA, "jobs_by_seq.json")))
    if os.path.exists(SUMS):
        raise SystemExit(f"{SUMS} exists - the clicks are already frozen; refusing to overwrite")
    os.makedirs(CLICKS, exist_ok=True)
    lines, missing, record = [], [], {}
    for seq in SEQS:
        jd = os.path.join(DATA, "jobs", jobs[seq])
        st = json.load(open(os.path.join(jd, "job.json"))) if os.path.exists(os.path.join(jd, "job.json")) else {}
        if st.get("calibration_inherited_from"):
            raise SystemExit(f"{seq}: calibration was inherited, not clicked - refusing to freeze")
        src = os.path.join(jd, "calibration.json")
        if not os.path.exists(src):
            missing.append(seq)
            continue
        dst = os.path.join(CLICKS, f"{seq}.calibration.json")
        shutil.copyfile(src, dst)
        pay = json.load(open(dst))
        record[seq] = {"ui_fit_rms_m": pay.get("rms_error_m"), "clicked": len(pay.get("points", {})),
                       "outliers": pay.get("outliers"), "clicked_at_mtime": os.path.getmtime(src)}
        lines.append(f"{_sha(dst)}  {seq}.calibration.json")
        lines.append(f"{_sha(os.path.join(D.STC, f'{seq}.p21id.tracks.json'))}  tracks:{seq}.p21id.tracks.json")
    with open(os.path.join(CLICKS, "freeze.json"), "w") as fh:
        json.dump({"clips": record, "not_calibrated": missing}, fh, indent=1)
    lines.append(f"{_sha(os.path.join(CLICKS, 'freeze.json'))}  freeze.json")
    with open(SUMS, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(record, indent=1), f"\nnot calibrated: {missing or 'none'}")
    print("commit model/validation/n30_clicks/ before --run")


def _check_freeze():
    head = subprocess.run(["git", "show", "HEAD:model/validation/n30_clicks/SHA256SUMS"], cwd=ROOT,
                          capture_output=True, text=True)
    if head.returncode:
        raise SystemExit("VOID: SHA256SUMS is not committed")
    if head.stdout != open(SUMS).read():
        raise SystemExit("VOID: SHA256SUMS differs from HEAD")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *PINNED], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit(f"VOID: uncommitted change in {PINNED} - the rule that runs must be the committed one")
    import dist_eval as D  # noqa: PLC0415
    seqs = []
    for line in filter(None, head.stdout.split("\n")):
        h, name = line.split("  ")
        path = (os.path.join(D.STC, name[len("tracks:"):]) if name.startswith("tracks:")
                else os.path.join(CLICKS, name))
        if _sha(path) != h:
            raise SystemExit(f"VOID: {name} changed since the freeze")
        if name.endswith(".calibration.json"):
            seqs.append(name.split(".")[0])
    if len(seqs) < MIN_CLIPS:
        raise SystemExit(f"VOID: {len(seqs)} clips frozen, need {MIN_CLIPS}")
    return seqs


def arms_for(payload, wh):
    """-> ({arm: calibrator or None}, {arm: refusal reason or None}, {arm: fit rms px})."""
    import calibrators as CAL  # noqa: PLC0415
    from pitch import resolve_landmarks  # noqa: PLC0415
    cals, why, rms = {}, {}, {}
    cals["SHIP"], why["SHIP"] = CAL.from_dict(payload), None
    tps = CAL.from_dict({**payload, "projection_type": "perspective_tps"})
    cals["TPS"], why["TPS"] = tps, (None if tps.active else "switched off")
    pitch = cals["SHIP"].pitch
    out = set(payload.get("outliers") or [])
    names = [n for n in payload["points"] if n not in out]
    metres = resolve_landmarks(pitch, {n: payload["points"][n] for n in names})
    obj = np.array([metres[n] for n in names], float)
    img = np.array([payload["points"][n] for n in names], float)
    for arm, form in CAMERA_ARMS.items():
        b = CM.fit_form(form, obj, img, wh, pitch)
        if b is None:
            cals[arm], why[arm] = None, "fit failed"
            continue
        rms[arm] = round(b[0], 2)
        cals[arm], why[arm] = CM.Fitted(form, b[1]), (f"fit rms {b[0]:.1f} px" if b[0] > MAX_FIT_RMS_PX else None)
    return cals, why, rms


def common_rows(uv, cals, why):
    """Refuse arms whose non-finite share of `uv` exceeds MAX_NAN_SHARE (mutates `why`), then
    -> (keep mask: finite under EVERY non-refused arm, removed share)."""
    keep = np.ones(len(uv), bool)
    for a, c in cals.items():
        if c is None or why[a]:
            continue
        fin = np.isfinite(np.asarray(c.to_pitch(uv), float).reshape(-1, 2)).all(1)
        if (~fin).mean() > MAX_NAN_SHARE:
            why[a] = f"non-finite share {(~fin).mean():.3f} of track rows"
            continue
        keep &= fin
    return keep, float(1 - keep.mean())


def verdict(clips, pooled, slope, chain_slope, dist, floor, refused, arm_names=("TPS", *CAMERA_ARMS)):
    """The RULE and CHOICE above, on already-computed numbers. Pure, so --demo can plant answers.
    A refused arm's clip carries None, and every comparison against None fails."""
    med = lambda arm, q: clips[q][arm]["median"]  # noqa: E731
    lt = lambda a, b: a is not None and b is not None and a < b  # noqa: E731
    qs = list(clips)
    chain_dev = float(np.median([abs(chain_slope[q] - 1) for q in qs]))
    passes, parts = {}, {}
    for x in arm_names:
        sl = [slope[x].get(q) for q in qs]
        p = {"P1": all(lt(med(x, q), med("SHIP", q)) for q in qs),
             "P2": pooled[x] is not None and all(pooled[x][k] <= pooled["SHIP"][k] for k in ("median", "p90", "far_median")),
             "P3": (all(s is not None and SLOPE_BAND[0] <= s <= SLOPE_BAND[1] for s in sl)
                    and float(np.median([abs(s - 1) for s in sl])) <= chain_dev + SLOPE_OVER_CHAIN),
             "P4": dist[x] is not None and abs(dist[x] - floor) <= abs(dist["SHIP"] - floor),
             "P5": not any(refused[x][q] for q in qs)}
        parts[x], passes[x] = p, all(p.values())
    base = "TPS" if passes["TPS"] else "SHIP"
    need = len(qs) - 1
    winners = [a for a in CAMERA_ARMS if a in passes and passes[a]
               and pooled[a]["median"] <= pooled[base]["median"] - MARGIN_M
               and sum(lt(med(a, q), med(base, q)) for q in qs) >= need]
    choice = winners[0] if winners else ("TPS" if passes["TPS"] else None)
    return {"parts": parts, "passes": passes, "chain_median_abs_slope_dev": chain_dev,
            "wins_needed": f"{need} of {len(qs)}", "ships": choice or "nothing (SHIP stays)"}


class _Chain:
    """ST2's chain as a calibrator for the reference slope: non-finite where it places nothing."""

    def __init__(self, seq):
        import dist_eval as D  # noqa: PLC0415
        self.inv = D._refmap(seq)

    def to_pitch(self, uv):
        X, placed = self.inv(np.asarray(uv, float).reshape(-1, 2))
        X = np.asarray(X, float).copy()
        X[~np.asarray(placed, bool)] = np.nan
        return X


def run():
    import calib_product as CP  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    seqs = _check_freeze()
    clips, slope, chain_slope, refused = {}, {a: {} for a in ARMS}, {}, {a: {} for a in ARMS}
    errs = {a: {"all": [], "far": []} for a in ARMS}
    dsum, hsum = {a: [0.0, 0.0] for a in ARMS}, {a: [0.0, 0.0] for a in ARMS}
    fl, hfl, chain, removed, rms_all, loc = [0.0, 0.0], [0.0, 0.0], {}, {}, {}, {}
    for q in seqs:
        payload = json.load(open(os.path.join(CLICKS, f"{q}.calibration.json")))
        cals, why, rms_all[q] = arms_for(payload, CM.T._wh(q))
        # position at MOT feet
        uv, G = D._feet_pixels(q)
        far = G[:, 1] < 34.0
        Xr, placed = D._refmap(q)(uv)
        chain[q] = round(float(np.median(np.linalg.norm(Xr[placed] - G[placed], axis=1))), 3)
        X = {a: (np.asarray(c.to_pitch(uv), float).reshape(-1, 2) if c is not None else None) for a, c in cals.items()}
        for a in CAMERA_ARMS:
            if X[a] is not None and not why[a] and (~np.isfinite(X[a]).all(1)).mean() > MAX_NAN_SHARE:
                why[a] = f"non-finite share {(~np.isfinite(X[a]).all(1)).mean():.3f} of MOT feet"
        # one population for B, D and speed
        blob = json.load(open(f"{D.STC}/{q}.p21id.tracks.json"))
        rows = blob["detections"]
        keep, removed[q] = common_rows(np.array([[r["x"], r["y"]] for r in rows], float), cals, why)
        with open(f"{D.STC}/{q}.{VARIANT}.tracks.json", "w") as fh:
            json.dump({**blob, "detections": [r for r, k in zip(rows, keep) if k]}, fh)
        chain_slope[q] = D._chord_slope(q, VARIANT, _Chain(q))[0]
        fr = D.evaluate_stc(q, VARIANT, oracle="position")
        fl = [fl[0] + fr["squad"]["ours_m"], fl[1] + fr["squad"]["gt_matched_m"]]
        hfl = [hfl[0] + fr["speed"]["our_hsr_m"], hfl[1] + fr["speed"]["gt_matched_hsr_m"]]
        ruler, clips[q] = None, {}
        for a in ARMS:
            refused[a][q] = why[a]
            if why[a] or cals[a] is None:
                clips[q][a] = {"median": None, "refused": why[a]}
                continue
            e = np.linalg.norm(X[a] - G, axis=1)
            ok = np.isfinite(e)
            errs[a]["all"].append(e[ok])
            errs[a]["far"].append(e[ok & far])
            s, n, lr = D._chord_slope(q, VARIANT, cals[a])
            slope[a][q] = s
            r = D.evaluate_stc(q, VARIANT, oracle="drop", our_cal=cals[a])
            dsum[a] = [dsum[a][0] + r["squad"]["ours_m"], dsum[a][1] + r["squad"]["gt_matched_m"]]
            hsum[a] = [hsum[a][0] + r["speed"]["our_hsr_m"], hsum[a][1] + r["speed"]["gt_matched_hsr_m"]]
            g = (r["squad"]["gt_m"], r["speed"]["gt_hsr_m"])
            if ruler is not None and g != ruler:
                raise SystemExit(f"VOID: {q}: the ground truth moved between arms")
            ruler = g
            clips[q][a] = {"median": float(np.median(e[ok])), "p90": float(np.percentile(e[ok], 90)),
                           "far_median": float(np.median(e[ok & far])),
                           "vs_chain_median": float(np.median(np.linalg.norm((X[a] - Xr)[placed & ok], axis=1))),
                           "slope": s, "chords": n, "chord_length_ratio": lr,
                           "distance_per_covered_m": r["squad"]["ours_m"] / r["squad"]["gt_matched_m"],
                           "refused": None}
        if q in ("118575", "128057"):
            loc[q] = {a: (CP.absolute(seqs=(q,), cals={q: cals[a]})["absolute"][q] if not why[a] and cals[a] else None)
                      for a in ARMS}
    pooled = {a: ({"median": float(np.median(np.concatenate(errs[a]["all"]))),
                   "p90": float(np.percentile(np.concatenate(errs[a]["all"]), 90)),
                   "far_median": float(np.median(np.concatenate(errs[a]["far"])))} if errs[a]["all"] else None)
              for a in ARMS}
    dist = {a: (dsum[a][0] / dsum[a][1] if dsum[a][1] else None) for a in ARMS}
    floor = fl[0] / fl[1]
    v = verdict(clips, pooled, slope, chain_slope, dist, floor, refused)
    res = {"seqs": seqs, "clips": clips, "pooled": pooled, "slope": slope, "chain_slope": chain_slope,
           "distance_per_covered_m": dist, "distance_floor": floor,
           "hsr_own_scale": {a: (hsum[a][0] / hsum[a][1] if hsum[a][1] else None) for a in ARMS},
           "hsr_floor": hfl[0] / hfl[1], "rows_removed_share": removed, "chain_vs_feed_m": chain,
           "chain_over_2m": [q for q in chain if chain[q] > 2], "fit_rms_px": rms_all, "loc": loc,
           "verdict": v, "protocol": "model/validation/n30_operator_maps.py docstring",
           "rule_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip()}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("pooled", "distance_per_covered_m", "distance_floor", "hsr_floor",
                                          "chain_slope", "rows_removed_share", "chain_vs_feed_m", "verdict")},
                     indent=1, default=str))
    print(f"-> {OUT}")


CONFIRM_OUT = os.path.join(ROOT, "model/validation/out/stc2025/n30_confirm.json")
EXPORT_TOL_M = 0.1


def export(payload, fitted):
    """The chosen camera arm as a product calibration file's contents."""
    return {**payload, "projection_type": "perspective_fisheye", "camera": [float(v) for v in fitted.q]}


def confirm_verdict(valid, slope, chain_slope, ratio, floor):
    """CONFIRMATION, AS RUN, on already-computed numbers. Pure, so --demo can plant answers."""
    qs = list(chain_slope)
    chain_dev = float(np.median([abs(chain_slope[q] - 1) for q in qs]))
    sl = [slope[q] for q in qs]
    p3 = (all(SLOPE_BAND[0] <= s <= SLOPE_BAND[1] for s in sl)
          and float(np.median([abs(s - 1) for s in sl])) <= chain_dev + SLOPE_OVER_CHAIN)
    p4 = abs(ratio["X"] - floor["X"]) <= abs(ratio["SHIP"] - floor["SHIP"])
    return {"valid": bool(valid), "P3": bool(p3), "P4": bool(p4), "chain_median_abs_slope_dev": chain_dev,
            "verdict": "CONFIRMED" if valid and p3 and p4 else "DO NOT SHIP"}


def confirm():
    import glob  # noqa: PLC0415
    import calibrators as CAL  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    seqs = _check_freeze()
    arm = json.load(open(OUT))["verdict"]["ships"]
    if arm not in CAMERA_ARMS or CAMERA_ARMS[arm].split("_")[0][:7] != "fisheye":
        raise SystemExit(f"--run chose {arm!r}: this confirmation exports a fisheye camera arm and nothing else")
    stale = sorted(f for q in seqs for f in glob.glob(ST._p(q, ".p30*")))
    if stale:
        raise SystemExit(f"{len(stale)} N30 replay outputs already exist (e.g. {stale[0]}): remove them and re-run")
    key_rows = lambda path: sorted((int(r["f"]), round(float(r["x"]), 3), round(float(r["y"]), 3))  # noqa: E731
                                   for r in json.load(open(path))["detections"])
    tags = {"SHIP": ".p30s", "X": ".p30f"}
    cals, export_m, fell_back, same_dets = {"SHIP": {}, "X": {}}, {}, {}, True
    for q in seqs:
        payload = json.load(open(os.path.join(CLICKS, f"{q}.calibration.json")))
        fitted, why, _rms = arms_for(payload, CM.T._wh(q))
        if why[arm] or fitted[arm] is None:
            raise SystemExit(f"VOID: {q}: {arm} is refused here ({why[arm]}), and --run did not refuse it")
        base_rows = key_rows(ST._p(q, ".p21.tracks.json"))
        for name, d in (("SHIP", payload), ("X", export(payload, fitted[arm]))):
            cfile = ST._p(q, f"{tags[name]}.calibration.json")
            with open(cfile, "w") as fh:
                json.dump(d, fh)
            cals[name][q] = CAL.load(cfile)
            if name == "X":
                uv, _G = D._feet_pixels(q)
                export_m[q] = float(np.median(np.linalg.norm(cals["X"][q].to_pitch(uv) - fitted[arm].to_pitch(uv), axis=1)))
                if not export_m[q] <= EXPORT_TOL_M:
                    raise SystemExit(f"VOID: {q}: the export is {export_m[q]:.3f} m from the fit it exports")
            D._n27_cache(q, tags[name], cfile)
            ST.track(q, cfile, tag=tags[name])
            ST.identify(q, cfile, src=tags[name], dst=f"{tags[name]}id")
            same_dets = same_dets and key_rows(ST._p(q, f"{tags[name]}.tracks.json")) == base_rows
        rows = json.load(open(ST._p(q, ".p30fid.tracks.json")))["detections"]
        fell_back[q] = float((~np.isfinite(fitted[arm].to_pitch(np.array([[r["x"], r["y"]] for r in rows], float))).all(1)).mean())
    slope, chain_slope, ruler = {"SHIP": {}, "X": {}}, {}, True
    sums = {n: {"ours": 0.0, "gt": 0.0, "hsr": 0.0, "gt_hsr": 0.0, "fl": 0.0, "fl_gt": 0.0, "fl_hsr": 0.0,
                "fl_gt_hsr": 0.0} for n in tags}
    for q in seqs:
        gt = None
        for name, tag in tags.items():
            v = f"{tag[1:]}id"
            slope[name][q] = D._chord_slope(q, v, cals[name][q])[0]
            if name == "X":
                chain_slope[q] = D._chord_slope(q, v, _Chain(q))[0]
            r = D.evaluate_stc(q, v, oracle="drop", our_cal=cals[name][q])
            f = D.evaluate_stc(q, v, oracle="position")
            g = (r["squad"]["gt_m"], r["speed"]["gt_hsr_m"])
            ruler, gt = ruler and gt in (None, g), g
            for k, x in (("ours", r["squad"]["ours_m"]), ("gt", r["squad"]["gt_matched_m"]),
                         ("hsr", r["speed"]["our_hsr_m"]), ("gt_hsr", r["speed"]["gt_matched_hsr_m"]),
                         ("fl", f["squad"]["ours_m"]), ("fl_gt", f["squad"]["gt_matched_m"]),
                         ("fl_hsr", f["speed"]["our_hsr_m"]), ("fl_gt_hsr", f["speed"]["gt_matched_hsr_m"])):
                sums[name][k] += x
    ratio = {n: s["ours"] / s["gt"] for n, s in sums.items()}
    floor = {n: s["fl"] / s["fl_gt"] for n, s in sums.items()}
    v = confirm_verdict(same_dets and ruler, slope["X"], chain_slope, ratio, floor)
    res = {"arm": arm, "seqs": seqs, "export_vs_fit_m": export_m, "same_detections": same_dets,
           "ground_truth_held": ruler, "slope": slope, "chain_slope": chain_slope,
           "distance_per_covered_m": ratio, "distance_floor": floor,
           "hsr_own_scale": {n: s["hsr"] / s["gt_hsr"] for n, s in sums.items()},
           "hsr_floor": {n: s["fl_hsr"] / s["fl_gt_hsr"] for n, s in sums.items()},
           "fell_back_to_homography_share": fell_back, "verdict": v,
           "rule_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip()}
    with open(CONFIRM_OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    print(f"-> {CONFIRM_OUT}")


def demo():
    """The rule on planted numbers, and the NaN population on a planted map."""
    # the confirmation's rule: each arm against its own floor, the slope against the chain
    cs, ok = {f"c{i}": 0.956 for i in range(8)}, {f"c{i}": 0.94 for i in range(8)}
    near = {"X": 1.18, "SHIP": 1.33}
    assert confirm_verdict(True, ok, cs, near, {"X": 1.25, "SHIP": 1.25})["verdict"] == "CONFIRMED"
    assert not confirm_verdict(True, ok, cs, near, {"X": 1.40, "SHIP": 1.30})["P4"]      # its own floor, not SHIP's
    assert not confirm_verdict(True, {**ok, "c0": 0.89}, cs, near, {"X": 1.25, "SHIP": 1.25})["P3"]
    assert confirm_verdict(False, ok, cs, near, {"X": 1.25, "SHIP": 1.25})["verdict"] == "DO NOT SHIP"
    qs = [f"c{i}" for i in range(8)]

    def planted(meds):
        clips = {q: {a: {"median": meds[a]} for a in ARMS} for q in qs}
        pooled = {a: {"median": meds[a], "p90": meds[a] + 2, "far_median": meds[a]} for a in ARMS}
        return clips, pooled
    slope = {a: {q: 0.97 for q in qs} for a in ARMS}
    chain_slope = {q: 0.956 for q in qs}
    refused = {a: {q: None for q in qs} for a in ARMS}
    dist = {a: 1.25 for a in ARMS}
    dist["SHIP"] = 1.40
    clips, pooled = planted({"SHIP": 7.5, "TPS": 2.4, "SPH10": 2.0, "FISH11": 1.9, "FISH12": 2.1})
    v = verdict(clips, pooled, slope, chain_slope, dist, 1.20, refused)
    assert v["ships"] == "SPH10" and v["wins_needed"] == "7 of 8", v   # fewest parameters, not lowest score
    clips, pooled = planted({"SHIP": 7.5, "TPS": 2.4, "SPH10": 2.3, "FISH11": 2.3, "FISH12": 2.3})
    assert verdict(clips, pooled, slope, chain_slope, dist, 1.20, refused)["ships"] == "TPS"   # inside the margin
    clips[qs[3]]["SPH10"]["median"] = 2.5                                   # loses one clip to TPS
    clips[qs[4]]["SPH10"]["median"] = 2.5                                   # and a second: 6 of 8 is not enough
    pooled["SPH10"]["median"] = 2.0
    assert verdict(clips, pooled, slope, chain_slope, dist, 1.20, refused)["ships"] == "TPS"
    clips[qs[3]]["TPS"]["median"] = 9.0                                     # TPS loses a clip to SHIP
    v = verdict(clips, pooled, slope, chain_slope, dist, 1.20, refused)
    assert not v["parts"]["TPS"]["P1"] and v["ships"] == "SPH10", v        # so camera arms face SHIP
    # P3 is measured against the chain: 0.956 everywhere must not fail a map that matches it
    assert verdict(clips, pooled, {a: {q: 0.956 for q in qs} for a in ARMS}, chain_slope, dist, 1.20,
                   refused)["parts"]["TPS"]["P3"] is True
    slope["SPH10"][qs[0]] = 0.85
    assert not verdict(clips, pooled, slope, chain_slope, dist, 1.20, refused)["parts"]["SPH10"]["P3"]
    refused["FISH11"][qs[1]] = "fit failed"
    clips[qs[1]]["FISH11"] = {"median": None}
    slope["FISH11"].pop(qs[1])
    p = verdict(clips, pooled, slope, chain_slope, dist, 1.20, refused)["parts"]["FISH11"]
    assert not p["P5"] and not p["P1"] and not p["P3"], p                 # a refusal fails, never crashes
    # one non-finite row under one arm: removed for everyone, nobody refused, P4 inputs stay finite
    class Map:
        def __init__(self, bad): self.bad = bad
        def to_pitch(self, uv):
            X = np.asarray(uv, float) / 40.0
            X[self.bad] = np.nan
            return X
    uv = np.arange(400, dtype=float).reshape(200, 2)
    why = {"SHIP": None, "TPS": None, "SPH10": None}
    keep, share = common_rows(uv, {"SHIP": Map([]), "TPS": Map([]), "SPH10": Map([17])}, why)
    assert keep.sum() == 199 and not keep[17] and abs(share - 0.005) < 1e-12 and not any(why.values()), (share, why)
    keep, share = common_rows(uv, {"SHIP": Map([]), "SPH10": Map(list(range(5)))}, why := {"SHIP": None, "SPH10": None})
    assert why["SPH10"] and keep.all(), why                                 # 2.5% refuses the arm instead
    print("n30 demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    modes = {"freeze": freeze, "run": run, "confirm": confirm, "demo": demo}
    for m in modes:
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    modes[next(m for m in modes if getattr(a, m))]()
