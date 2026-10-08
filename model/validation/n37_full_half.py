"""N37 on ONE FULL HALF with OUR detector's boxes: the feet-on-the-paint depth reference gets a real sample. Protocol first.

WHY
    docs/experiments/n37-feet-on-lines.md: seven four-minute clips with human-drawn boxes gave 12 events, 8 at the far touchline,
    against a rule that asks for 30 from >= 3 clips. plan-parallel-tracks (not published) Track A item 1: run the same instrument
    on a full half, with the product's own detector, so the rule gets its n. Nothing in the instrument changes: the same
    locator (model/validation/n37_feet_on_lines.py: paint_rows, ridge_at, line_pixels, DEPTH_LINES - IMPORTED, not copied), the
    same three arms, the same rule. What changes is the population, and it is written down here before anything runs.

POPULATION, fixed before the first frame is decoded
    match/half    118575, FIRST half: ~/Downloads/118575_panorama_1st_half.mp4 (data/footage/README.md:
                  the ST2 distribution's own panorama, 4096x1080, 25 fps, 71850 frames). The MOT clip 118575.mp4 is
                  frames 6000..11999 of it: clip frame 0 and half frame 6000 are pixel-identical (mean |diff| 0.0; the
                  neighbours read 3.9), printed by --score as the image half of the join proof.
    frames        every 5th frame of the half (V = 0, 5, ..., 71845): 14370 detected frames.
    detector      the product's: model/engine/track.run, detector='coreml', conf at track.run's default 0.5, no scene prior
                  (calib=None, as model/validation/st2_product.raw_detect runs it). The prior is a post-filter on boxes with a
                  3 m margin outside the pitch (calibrators.pitch_polygon_px), so it cannot remove a touchline stander
                  unless a map is > 3 m out; leaving it off keeps the population independent of the maps under test.
                  Detections are cached through track.run's own --dets-cache mechanism (keyed by original frame index),
                  never re-implemented. cls == 'person' only.
    foot          box bottom centre ((x1 + x2) / 2, y2), as everywhere in this repo.
    standing      N37 asked for a foot still (<= 1.5 px in u and in v per step) over >= 10 consecutive even frames
                  (18 frames spanned, "0.8 s"). At stride 5 the equivalent is >= 5 consecutive sampled frames (20 frames
                  spanned = 0.8 s), <= 1.5 px per step. Detections carry no identity, so a stand is a chain: each
                  sampled frame's box is linked to the nearest box of the previous sampled frame within 1.5 px in both
                  axes (greedy, nearest first, one-to-one). A chain shorter than 5 samples is not a stand. Two stands
                  at the same spot (<= 3 px) with a gap <= 10 samples (2 s) are ONE event: a detector that jitters
                  2 px once splits a throw-in in two, and N37 already counted one player three times. n_raw is printed
                  beside n. A stand's foot is the median (u, v) over its samples; an event's over its stands.
    long stands   N37's ">= 25 even frames" (50 frames): here >= 11 samples (50 frames).
    labelled      as N37: the nearest depth line of the SHIP camera's drawing within 15 px of the foot (LABEL_PX).
    located       as N37 revision 1: the paint found ALONG the line by model/engine/autocal.search on the clip's 40-frame
                  median (model/validation/out/stc2025/118575.median40.png - the camera is fixed and the half's frame 6000 is
                  the clip's frame 0, so it is the same picture of the same paint), interpolated at the foot's column
                  between samples within 20 px either side; the ridge re-found within 4 rows must be within 1 px of it
                  and, on far lines, <= 3 px half-width; the foot is ON the line iff |v - ridge| <= max(1 px, half-width).
    truth         the line's y in the 105 x 68 frame (DEPTH_LINES).

ARMS  (nothing fitted, nothing tuned, all on disk; model/validation/n37_feet_on_lines.ARMS)
    SHIP    the zero-click camera the product runs: model/validation/out/stc2025/n34_autocal.json["118575"]
    KAPPA   the N36 candidate, VERT_CUBIC -0.4: model/validation/out/stc2025/n36_after_k-0.4.json["118575"]
    FEED    ST2's position feed for the WHOLE half, joined to the video as below

THE JOIN, and how it is proved (MOT ids exist only inside the clip)
    The clip join (model/validation/st2_team_gt.py, 118575: period FIRST_HALF, offset 6251, cost 36 px, vendor chain agreeing on
    22 of 22 ids) says clip frame f is XML frameNumber 6251 + f. The image check says clip frame f is half frame 6000 + f.
    So half frame V is XML frameNumber V + 251 in FIRST_HALF (the 251-frame pad of 118575_padding_info.csv; matchTime at
    frameNumber 6251 is 239967 ms = 40 ms * 6000 - 33 ms, i.e. the feed sample sits 33 ms before the video frame, which is
    nothing for a player standing still and is not corrected). --score PROVES it on the clip window: the full-half reader
    at V = 6000 + f, for f in the clip's stride-10 frames, must return exactly dist_eval.stc_gt's metres for every MOT id
    (max |diff| printed; the run is VOID unless it is 0). That check pins the arithmetic; the image check pins the video.
    Outside the window the half is one video at one rate, and the XML's FIRST_HALF frameNumbers run to 72351 (past the
    video's 72100): nothing is missing.

FEED IDENTITY BY MATCHING, its risk, and the pre-registered choice
    Without MOT ids the feed player standing at an event is found by geometry: the event's foot through the SHIP camera
    (metres), then the nearest feed player per sample frame, within a GATE; the event's player is the mode over its
    frames and must hold >= half of them; ambiguous frames (second-nearest within 2 m of the nearest) count for nobody.
    The map is used for MATCHING only; the arms' numbers never use the match.
    THE RISK, stated first: the hypothesis under test is a 3-4 m disagreement along depth between the maps and the feed
    at the far touchline. A 3 m gate REJECTS exactly the events where that disagreement is present and keeps the ones
    where the feed agrees with the map - it selects the answer. So the verdict uses a 6 m gate (covers both of
    n36 section 4's hypotheses, whose predicted gaps are <= 4.5 m), the 2 m ambiguity margin guards the neighbour, and
    the 3 m gate is printed beside it as the check: if the two gates disagree, the 3 m one is the biased one.
    Match rates, ambiguous shares and unmatched shares are printed per gate.

CHECKS, printed with the result because each could fake it
    bottom bias   our boxes vs the MOT ground-truth boxes on the clip window (V = 6000 + f, f % 10 == 0, 600 frames),
                  matched greedy by IoU >= 0.5: median (our y2 - GT y2) in px, overall and for GT feet the SHIP camera
                  puts in the far third (y < 22.67 m), with n matched and the GT recall at IoU 0.5. A systematic +1 px
                  is 0.6 m at the far touchline, and it would move all three arms together (they read the same foot).
    drop shares   chains -> stands (>= 5 samples) -> merged -> labelled -> paint found both sides -> ridge confirmed
                  -> ON the line (= events); each count printed.
    feed y <= 0   share of far-touchline events whose matched feed y is <= 0 (a feed clipped to the pitch cannot read a
                  player outside it).
    windows       events per 4-minute window of the half (the "clips" of a single half, for spread; NOT the rule's clips).
    long subset   the same medians on events with >= 11 samples.
                  REVISION 1 (2026-09-17, after --score, no arm number moved): the long subset was taken over
                  feed-MATCHED events, so a 30-sample event without a feed match fell out of it; it is over all events
                  of the line now, the FEED median over those with a match. --why was added at the same time.

RULE, N37's, unchanged.  e = arm's y minus the line's y per event; per line the median, n and windows.
      OWNER = SHIP/KAPPA  iff at the far touchline n >= 30 events from >= 3 CLIPS, |median e_map| <= 0.5 m and |median e_FEED| >= 1.5 m
      OWNER = FEED        iff the same n, |median e_FEED| <= 0.5 m and |median e_map| >= 1.5 m
      anything else: UNRESOLVED, medians printed regardless.
    ONE half is ONE clip. The clips criterion CANNOT be met by this run and the rule therefore returns UNRESOLVED
    whatever the medians say; this run supplies n and the direction at one install, and the doc says so.
    Predictions, written first, are N37's: feed scaled 4% about the camera's foot -> e_FEED about +3.3 m at the far
    touchline with e_SHIP near 0; map height/vertical-law error -> e_SHIP about -3 m there with e_FEED near 0. N37's twelve
    events read FEED -2.4, SHIP +1.6, KAPPA +0.1 (sign: + is inside the pitch from that line).

    .venv/bin/python model/validation/n37_full_half.py --detect   # the detector pass -> model/validation/out/stc2025/n37_full_118575_h1.dets.cache.json (resumable)
    .venv/bin/python model/validation/n37_full_half.py --events   # stands -> events on paint; reads NO position -> ...h1.events.json
    .venv/bin/python model/validation/n37_full_half.py --score    # join proof, matching, arms, rule, checks -> model/validation/out/stc2025/n37_full_118575_h1.json
    .venv/bin/python model/validation/n37_full_half.py --show     # print the stored result (the tokens the doc quotes)
    .venv/bin/python model/validation/n37_full_half.py --why      # ADDED AFTER --score (2026-09-17), a diagnostic that changes no number above:
                                                            # per event, the nearest feed player through SHIP and how many the feed carries;
                                                            # and N37's own GT far events on this clip: sampled frames vs frames our detector
                                                            # had a box within 8 px of the GT foot, with our v - GT v per detected frame;
                                                            # and every ridge-confirmed candidate's foot - paint offset per line (why 88 of 90 fell)
    .venv/bin/python model/validation/n37_full_half.py --demo     # synthetic: chaining finds a still box and not a walker; two split stands merge
"""
import collections
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
from n37_feet_on_lines import ARMS, DEPTH_LINES, LABEL_PX, STC, line_pixels, paint_rows, ridge_at  # noqa: E402
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

SEQ, PERIOD, PAD, CLIP0, CLIP_N = "118575", b"FIRST_HALF", 251, 6000, 6000
VIDEO = "~/Downloads/118575_panorama_1st_half.mp4"
XML = _ST2 + f"/raw/{SEQ}/{SEQ}_tracker_box_data.xml"
MOT = _ST2 + f"/mot/{SEQ}.txt"
TAG = os.path.join(STC, f"n37_full_{SEQ}_h1")
CACHE, EVENTS, OUT = TAG + ".dets.cache.json", TAG + ".events.json", TAG + ".json"
STRIDE, STILL_PX, MIN_STILL, LONG, MERGE_PX, MERGE_GAP = 5, 1.5, 5, 11, 3.0, 10
GATES, AMBIG_M, MIN_FAR, MIN_CLIPS, CLIPS_HERE, FAR_THIRD_M = (6.0, 3.0), 2.0, 30, 3, 1, 68.0 / 3



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


def detect():
    import track  # noqa: PLC0415
    track.run(VIDEO, TAG + ".tracks.json", detector="coreml", stride=STRIDE, dets_cache=CACHE)


def feet_by_frame():
    """{V: [(u, v, y2)]} from the detection cache, persons only."""
    raw = json.load(open(CACHE))
    if not raw.get("complete"):
        raise SystemExit(f"{CACHE} is not complete; run --detect again (it resumes)")
    return {int(k): [((b["xyxy"][0] + b["xyxy"][2]) / 2.0, b["xyxy"][3]) for b in v if b.get("cls") == "person"]
            for k, v in raw["frames"].items()}


def chains(feet, still=STILL_PX, stride=STRIDE):
    """Link each frame's boxes to the previous sampled frame's within `still` px in both axes -> [[(V, u, v), ...]]."""
    done, open_ = [], []
    for V in sorted(feet):
        cur = [(V, float(u), float(v)) for u, v in feet[V]]
        pairs = sorted((max(abs(c[1] - r[-1][1]), abs(c[2] - r[-1][2])), i, j) for i, r in enumerate(open_)
                       if r[-1][0] == V - stride for j, c in enumerate(cur))
        used_r, used_c, nxt = set(), set(), []
        for d, i, j in pairs:
            if d <= still and i not in used_r and j not in used_c:
                used_r.add(i), used_c.add(j)
                open_[i].append(cur[j])
                nxt.append(open_[i])
        done += [r for i, r in enumerate(open_) if i not in used_r]
        open_ = nxt + [[cur[j]] for j in range(len(cur)) if j not in used_c]
    return done + open_


def merge(stands):
    """Stands at one spot (<= MERGE_PX) with a gap <= MERGE_GAP samples are one event -> [[stand, ...]]."""
    out = []
    for s in sorted(stands, key=lambda s: s[0][0]):
        u, v = np.median([p[1] for p in s]), np.median([p[2] for p in s])
        for ev in out:
            last = ev[-1]
            lu, lv = np.median([p[1] for p in last]), np.median([p[2] for p in last])
            if 0 < s[0][0] - last[-1][0] <= MERGE_GAP * STRIDE and abs(u - lu) <= MERGE_PX and abs(v - lv) <= MERGE_PX:
                ev.append(s)
                break
        else:
            out.append([s])
    return out


def _scene():
    import cv2  # noqa: PLC0415
    import autocal as A  # noqa: PLC0415
    bgr = cv2.imread(os.path.join(STC, f"{SEQ}.median40.png"))
    cam = np.array(json.load(open(os.path.join(STC, ARMS["SHIP"])))[SEQ]["camera"], float)
    return A.native_map(bgr), A.grass_map(bgr), line_pixels(cam, bgr.shape[1]), paint_rows(bgr, cam)


def locate(u, v, scene):
    """N37 rev 1's label-and-locate for one foot -> (stage reached, label, label px, ridge row, half-width)."""
    native, grass, lines, paint = scene
    label, dist = None, LABEL_PX
    for name, L in lines.items():
        dd = float(np.min(np.hypot(L[:, 0] - u, L[:, 1] - v)))
        if dd < dist:
            label, dist = name, dd
    if label is None:
        return "merged", None, None, None, None
    pu, pv = paint[label]
    left, right = pu[pu <= u], pu[pu >= u]
    if not len(left) or not len(right) or u - left[-1] > 20 or right[0] - u > 20:
        return "labelled", label, dist, None, None
    v_paint = float(np.interp(u, pu, pv))
    r = ridge_at(native, grass, u, v_paint, radius=4.0)
    if r is None or abs(r[0] - v_paint) > 1.0 or (label.endswith("far") and r[1] > 3.0):
        return "paint_both_sides", label, dist, None, None
    return ("on_line" if abs(v - r[0]) <= max(1.0, r[1]) else "ridge_confirmed"), label, dist, r[0], r[1]


def candidates():
    """Every merged event through locate() -> [(stage, label, label px, ridge row, half, u, v, frames, n stands)]."""
    scene = _scene()
    ch = chains(feet_by_frame())
    stands = [c for c in ch if len(c) >= MIN_STILL]
    out = []
    for ev in merge(stands):
        pts = [p for s in ev for p in s]
        u, v = float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts]))
        out.append((*locate(u, v, scene), u, v, [int(p[0]) for p in pts], len(ev)))
    return len(ch), len(stands), out


STAGES = ("merged", "labelled", "paint_both_sides", "ridge_confirmed", "on_line")


def events_cmd():
    n_ch, n_st, cand = candidates()
    drop = collections.Counter(chains=n_ch, stands=n_st)
    for c in cand:
        for st in STAGES[:STAGES.index(c[0]) + 1]:
            drop[st] += 1
    events = [{"frames": fr, "stands": ns, "u": round(u, 2), "v": round(v, 2), "line": label, "ridge_row": round(row, 2),
               "ridge_half_px": round(half, 2), "label_px": round(dist, 1)}
              for stage, label, dist, row, half, u, v, fr, ns in cand if stage == "on_line"]
    json.dump({"drop": dict(drop), "events": events}, open(EVENTS, "w"), indent=1)
    print("drop shares:", dict(drop))
    print(len(events), "events:", {k: sum(e["line"] == k for e in events) for k in DEPTH_LINES}, "->", EVENTS)


def feed(frames):
    """{V: {playerId: (x_m, y_m)}} for the wanted half frames, from the FIRST_HALF XML at frameNumber V + PAD."""
    import st2_team_gt as G  # noqa: PLC0415
    want = {V + PAD: V for V in frames}
    out, cur = {}, None
    with open(XML, "rb") as fh:
        for line in fh:
            s = line.lstrip()
            if s[:6] == b"<frame":
                m = G._FRAME.match(s)
                cur = None
                if m and m.group(2) == PERIOD and int(m.group(1)) in want:
                    cur = out.setdefault(want[int(m.group(1))], {})
            elif cur is not None and s[:7] == b"<player":
                m = G._PLAYER.match(s)
                if m:
                    cur[int(m.group(1))] = (float(m.group(2)) * 105.0, float(m.group(3)) * 68.0)
    return out


def join_proof(F):
    """The full-half reader at V = CLIP0 + f must equal dist_eval.stc_gt at f for every MOT id -> (max |diff| m, n)."""
    import cv2  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    import st2_team_gt as G  # noqa: PLC0415
    gt, p_of = D.stc_gt(SEQ, D.STC_EVERY), G.load()[SEQ]["mot_to_player"]
    worst, n = 0.0, 0
    for mid, g in gt.items():
        for f, xy in zip(g["f"].tolist(), g["xy"]):
            if f % 10 == 0 and CLIP0 + f in F and int(p_of[mid]) in F[CLIP0 + f]:
                worst = max(worst, float(np.abs(np.array(F[CLIP0 + f][int(p_of[mid])]) - xy).max()))
                n += 1
    H, C = cv2.VideoCapture(VIDEO), cv2.VideoCapture(_ST2 + f"/mot/clips/{SEQ}.mp4")
    c0 = C.read()[1].astype(int)
    img = {}
    for V in (CLIP0 - 1, CLIP0, CLIP0 + 1):
        H.set(cv2.CAP_PROP_POS_FRAMES, V)
        img[V] = round(float(np.abs(H.read()[1].astype(int) - c0).mean()), 3)
    return worst, n, img


def bottom_bias(cam):
    """Our boxes vs MOT boxes on the clip window: median (our y2 - GT y2) px, overall and far third; recall at IoU 0.5."""
    import autocal as A  # noqa: PLC0415
    raw = {int(k): np.array([b["xyxy"] for b in v if b.get("cls") == "person"]).reshape(-1, 4) for k, v in json.load(open(CACHE))["frames"].items()}
    gtb = collections.defaultdict(list)
    for line in open(MOT):
        p = line.split(",")
        if len(p) >= 6 and int(p[0]) % 10 == 0:
            x, y, w, h = map(float, p[2:6])
            gtb[int(p[0])].append((x, y, x + w, y + h))
    diffs, far, n_gt, n_gt_far = [], [], 0, 0
    for f, g in gtb.items():
        g, d = np.array(g), raw.get(CLIP0 + f)
        if d is None:
            continue
        gy = A._inv(cam, np.column_stack([(g[:, 0] + g[:, 2]) / 2, g[:, 3]]))[:, 1]
        n_gt += len(g)
        n_gt_far += int((gy < FAR_THIRD_M).sum())
        if not len(d):
            continue
        ix1, iy1 = np.maximum(g[:, None, 0], d[None, :, 0]), np.maximum(g[:, None, 1], d[None, :, 1])
        ix2, iy2 = np.minimum(g[:, None, 2], d[None, :, 2]), np.minimum(g[:, None, 3], d[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
        iou = inter / (area(g)[:, None] + area(d)[None, :] - inter)
        used_g, used_d = set(), set()
        for k in np.argsort(-iou, axis=None):
            i, j = divmod(int(k), iou.shape[1])
            if iou[i, j] < 0.5:
                break
            if i in used_g or j in used_d:
                continue
            used_g.add(i), used_d.add(j)
            diffs.append(d[j, 3] - g[i, 3])
            if gy[i] < FAR_THIRD_M:
                far.append(d[j, 3] - g[i, 3])
    return {"n_matched": len(diffs), "median_px": round(float(np.median(diffs)), 2), "recall_iou05": round(len(diffs) / n_gt, 3),
            "far_n_matched": len(far), "far_median_px": round(float(np.median(far)), 2) if far else None,
            "far_recall_iou05": round(len(far) / n_gt_far, 3) if n_gt_far else None, "gt_boxes": n_gt, "gt_boxes_far": n_gt_far}


def match(ev, F, cam, gate):
    """The feed player at this event under `gate` -> (playerId, feed_y, n_frames_used) or a reason string."""
    import autocal as A  # noqa: PLC0415
    xy = A._inv(cam, np.array([[ev["u"], ev["v"]]]))[0]
    votes, ys = collections.Counter(), collections.defaultdict(list)
    for V in ev["frames"]:
        P = F.get(V, {})
        if len(P) < 2:
            continue
        ids = list(P)
        d = np.hypot(*(np.array([P[i] for i in ids]) - xy).T)
        o = np.argsort(d)
        if d[o[0]] > gate:
            votes["far"] += 1
        elif d[o[1]] - d[o[0]] < AMBIG_M:
            votes["ambiguous"] += 1
        else:
            votes[ids[o[0]]] += 1
            ys[ids[o[0]]].append(P[ids[o[0]]][1])
    if not votes:
        return "no_feed"
    pid, n = votes.most_common(1)[0]
    if pid in ("far", "ambiguous") or n * 2 < sum(votes.values()):
        return "ambiguous" if pid == "ambiguous" or votes["ambiguous"] >= votes["far"] else "far"
    return pid, float(np.median(ys[pid])), n


def score():
    import autocal as A  # noqa: PLC0415
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n37_full_half.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the protocol must be committed before any position is read")
    events = json.load(open(EVENTS))["events"]
    cams = {a: np.array(json.load(open(os.path.join(STC, f)))[SEQ]["camera"], float) for a, f in ARMS.items()}
    F = feed(sorted({V for e in events for V in e["frames"]} | set(range(CLIP0, CLIP0 + CLIP_N, 10))))
    worst, n_join, img = join_proof(F)
    print(f"join proof: max |full-half feed - stc_gt| = {worst} m over {n_join} (id, frame) pairs; clip frame 0 vs half frames {img}")
    if worst > 0.0 or img[CLIP0] > 0.0:
        raise SystemExit("VOID: the full-half join does not reproduce the clip join")
    res = {"join_proof": {"max_diff_m": worst, "pairs": n_join, "image_mean_absdiff": img},
           "bottom_bias": bottom_bias(cams["SHIP"]), "events": len(events), "gates": {}}
    rows = []
    for e in events:
        y = DEPTH_LINES[e["line"]]
        row = {**e, "truth": y, "n_samples": len(e["frames"]), "window": e["frames"][0] // (240 * 25)}
        for a, cam in cams.items():
            A.VERT_CUBIC = -0.4 if a == "KAPPA" else 0.0
            row[f"{a}_y"] = float(A._inv(cam, np.array([[e["u"], e["v"]]]))[0, 1])
        A.VERT_CUBIC = 0.0
        for g in GATES:
            m = match(e, F, cams["SHIP"], g)
            row[f"feed_{g:g}"] = None if isinstance(m, str) else m[1]
            row[f"why_{g:g}"] = m if isinstance(m, str) else "matched"
        rows.append(row)
    res["rows"] = rows
    for g in GATES:
        G = {"matched": sum(r[f"why_{g:g}"] == "matched" for r in rows), "unmatched": collections.Counter(r[f"why_{g:g}"] for r in rows if r[f"why_{g:g}"] != "matched"),
             "lines": {}}
        for name, y in DEPTH_LINES.items():
            sel = [r for r in rows if r["line"] == name]
            fed = [r for r in sel if r[f"feed_{g:g}"] is not None]
            lng = [r for r in sel if r["n_samples"] >= LONG]          # revision 1: was `fed`, which hid a 30-sample event with no feed match
            med = lambda rs, k: round(float(np.median([r[k] - y for r in rs if r[k] is not None])), 2) if any(r[k] is not None for r in rs) else None  # noqa: E731
            G["lines"][name] = {"n": len(sel), "n_raw": sum(r["stands"] for r in sel), "n_feed": len(fed), "windows": len({r["window"] for r in sel}),
                                "med_SHIP": med(sel, "SHIP_y"), "med_KAPPA": med(sel, "KAPPA_y"), "med_FEED": med(fed, f"feed_{g:g}"),
                                "n_long": len(lng), "long_SHIP": med(lng, "SHIP_y"), "long_KAPPA": med(lng, "KAPPA_y"), "long_FEED": med(lng, f"feed_{g:g}")}
        far = [r for r in rows if r["line"] == "touch_far" and r[f"feed_{g:g}"] is not None]
        G["far_feed_at_or_outside_zero_share"] = round(float(np.mean([r[f"feed_{g:g}"] <= 0.0 for r in far])), 3) if far else None
        L = G["lines"]["touch_far"]
        verdict = f"UNRESOLVED (clips: {CLIPS_HERE} of {MIN_CLIPS}; far events: {L['n_feed']} of {MIN_FAR})"
        if L["n_feed"] >= MIN_FAR and CLIPS_HERE >= MIN_CLIPS:
            ms, mf = abs(L["med_SHIP"]), abs(L["med_FEED"])
            verdict = "MAP" if ms <= 0.5 and mf >= 1.5 else "FEED" if mf <= 0.5 and ms >= 1.5 else "UNRESOLVED"
        G["owner_of_the_4pct"] = verdict
        res["gates"][f"{g:g}"] = G
    res["events_per_window"] = dict(sorted(collections.Counter(r["window"] for r in rows).items()))
    res["drop"] = json.load(open(EVENTS))["drop"]
    json.dump(res, open(OUT, "w"), indent=1)
    show(res)


def show(res=None):
    res = res or json.load(open(OUT))
    print("drop shares:", res["drop"], "| events per 4-min window:", res["events_per_window"])
    b = res["bottom_bias"]
    print(f"bottom bias (our y2 - GT y2, px): median {b['median_px']} over {b['n_matched']} matched, recall {b['recall_iou05']} of {b['gt_boxes']} GT boxes; "
          f"far third: median {b['far_median_px']} over {b['far_n_matched']}, recall {b['far_recall_iou05']} of {b['gt_boxes_far']}")
    for g, G in res["gates"].items():
        print(f"--- feed matched through SHIP, gate {g} m: {G['matched']} of {res['events']} events matched, unmatched {dict(G['unmatched'])}")
        print(f"{'line':11s} {'n':>4s} {'raw':>4s} {'feed':>4s} win  {'SHIP':>6s} {'KAPPA':>6s} {'FEED':>6s}   (long: n, SHIP, KAPPA, FEED)")
        for name, L in G["lines"].items():
            print(f"{name:11s} {L['n']:4d} {L['n_raw']:4d} {L['n_feed']:4d} {L['windows']:3d}  {str(L['med_SHIP']):>6s} {str(L['med_KAPPA']):>6s} {str(L['med_FEED']):>6s}"
                  f"   ({L['n_long']}, {L['long_SHIP']}, {L['long_KAPPA']}, {L['long_FEED']})")
        print("far-touchline events with feed y <= 0:", G["far_feed_at_or_outside_zero_share"], "| OWNER of the 4%:", G["owner_of_the_4pct"])
    print("->", OUT)


def why():
    import autocal as A  # noqa: PLC0415
    res, cam, feet = json.load(open(OUT)), np.array(json.load(open(os.path.join(STC, ARMS["SHIP"])))[SEQ]["camera"], float), feet_by_frame()
    F = feed(sorted({V for r in res["rows"] for V in r["frames"]}))
    for r in res["rows"]:
        xy = A._inv(cam, np.array([[r["u"], r["v"]]]))[0]
        P = F[r["frames"][len(r["frames"]) // 2]]
        d = sorted(float(np.hypot(*(np.array(p) - xy))) for p in P.values())
        print(f"event {r['line']} frames {r['frames'][0]}-{r['frames'][-1]} n {r['n_samples']} SHIP xy {xy.round(2).tolist()}: "
              f"nearest feed player {d[0]:.1f} m, second {d[1]:.1f} m, feed carries {len(P)} players")
    off = collections.defaultdict(list)
    for stage, label, _d, row, _h, _u, v, _f, _n in candidates()[2]:
        if stage in ("ridge_confirmed", "on_line"):
            off[label].append(round(v - row, 1))
    for label, o in sorted(off.items()):
        print(f"ridge-confirmed candidates on {label}: {len(o)}, foot - paint px sorted: {sorted(o)}")
    for e in [e for e in json.load(open(os.path.join(STC, "n37_events.json"))) if e["seq"] == SEQ]:
        Vs = [CLIP0 + f for f in e["frames"] if (CLIP0 + f) % STRIDE == 0]
        dv = [round(min((abs(u - e["u"]), v - e["v"]) for u, v in feet.get(V, []) if abs(u - e["u"]) < 8 and abs(v - e["v"]) < 15)[1], 1)
              for V in Vs if any(abs(u - e["u"]) < 8 and abs(v - e["v"]) < 15 for u, v in feet.get(V, []))]
        print(f"N37 GT event {e['line']} mot {e['mot']} at ({e['u']}, {e['v']}): sampled {len(Vs)} frames, our detector has a box on {len(dv)}, our v - GT v {dv}")


def demo():
    F = {V: [(100.0 + 0.2 * (V // 5), 50.0), (300.0 + 4.0 * (V // 5), 50.0)] for V in range(0, 100, 5)}
    F[50] = [F[50][1]]                                     # the still box is missed once: two stands, one event
    ch = chains(F)
    st = [c for c in ch if len(c) >= MIN_STILL]
    assert len(st) == 2 and all(abs(c[0][1] - 100) < 5 for c in st), [(len(c), c[0]) for c in ch]
    assert len(merge(st)) == 1                            # split by one missed frame, merged back
    print("n37 full-half demo ok: a walker never chains into a stand; a stand split by one missed detection is one event")


if __name__ == "__main__":
    a = sys.argv[1:]
    (detect if "--detect" in a else events_cmd if "--events" in a else score if "--score" in a else show if "--show" in a else why if "--why" in a else demo)()
