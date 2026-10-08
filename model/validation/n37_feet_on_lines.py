"""N37: feet on the paint. A map-free depth reference from players standing ON painted lines - the protocol, first.

WHY
    Our map and ST2's position feed disagree by about 4% of range along the camera ray (docs/experiments/n36-far-touchline.md
    section 4), and nothing so far says who owns it. N32 tried throw-in takers by hand: three far ones. The author's
    idea (2026-09-17): find the lines, then find players whose feet are ON them, with its own logic, not the tracker.
    A foot pixel that sits on a line's paint in the IMAGE is truth about one pitch coordinate that neither map made:
    the player is on that line to within its width. Every line parallel to the touchlines gives a depth at a known
    range: far touchline (y = 0), the far penalty-area edges (13.84), the far goal-area edges (24.84), the near ones
    (43.16, 54.16), the near touchline (68). Between them they sample the whole depth of the pitch.

WHAT COUNTS AS AN EVENT, decided before any position is read
    population   the seven same-install clips; the MOT ground-truth boxes (human-drawn, every frame), foot = bottom
                 centre of the box. Our own detector is not used: this is a test of the maps, not of detection.
    labelled     by the zero-click camera ONLY as a label: the nearest depth line of the drawing within 15 px of the
                 foot, in the image. Lines are never closer than 25 px to each other where this applies, so the label
                 cannot be wrong by a line; the camera's own error (a few px) cannot change which line is nearest.
    located      by the PAINT, not the camera: on the clip's 40-frame median (players gone), the line's own paint as
                 model/engine/autocal.search finds it along the whole line (the N36 grass-flanked rule for the perimeter,
                 the ridge map for interior lines), within 8 px of the zero-click camera's line; its row at the
                 foot's column is interpolated between the two nearest found samples, which must lie within 20 px
                 either side. The ridge's half-width at that column comes from the profile and must be <= 3 px on
                 the far lines. The foot is ON the line iff |foot row - paint row| <= max(1.0 px, half-width).
                 REVISION 1 (2026-09-17, after the first --events run and before any position was read into a
                 verdict): the first definition took the strongest grass-flanked ridge within 12 px of the FOOT, and
                 on 118575 it took ridges 6-8 m inside the pitch (all three arms agreed on that depth), which are not
                 paint. The paint is found along the LINE now, not around the foot. n = 2 event definitions.
    standing     the same MOT id's foot moves <= 1.5 px between consecutive even frames for >= 10 of them (0.8 s).
                 A stand is ONE event whatever its length: its foot, its feed position and its map positions are the
                 medians over its frames. Walking across a line at the sampled instant is not an event.
    truth        the line's pitch coordinate y. Uncertainty: the line is 0.12 m wide and the box bottom is a pixel
                 or two: 0.1 m near, 0.6-1 m at the far touchline per event, unbiased either way.

ARMS  (nothing fitted, nothing tuned, all on disk)
    SHIP    the zero-click camera the product runs: model/validation/out/stc2025/n34_autocal.json
    KAPPA   the N36 candidate, paint rule + VERT_CUBIC -0.4: model/validation/out/stc2025/n36_after_k-0.4.json
    FEED    ST2's position feed at the stand's frames (dist_eval.stc_gt)

RULE, fixed now.  e = arm's y minus the line's y, per event; per line the median over events, with n = events and the
    number of clips they come from. The FAR TOUCHLINE decides ownership of the 4%, because that is where 4% is 3 m:
      OWNER = SHIP/KAPPA  iff, with n >= 30 events from >= 3 clips there, |median e_map| <= 0.5 m and |median e_FEED| >= 1.5 m
      OWNER = FEED        iff |median e_FEED| <= 0.5 m and |median e_map| >= 1.5 m   (the map is pulled in)
      anything else, or fewer events: UNRESOLVED, and every line's medians are printed regardless.
    Predictions, written first (the two hypotheses of n36 section 4): if the feed is a 4% similarity about the camera's
    foot, e_FEED runs from about +0.6 m at the near touchline to +3.3 m at the far one with e_SHIP near 0 everywhere;
    if the map has a height/vertical-law error, e_SHIP runs from about -0.3 m near to -3 m far with e_FEED near 0.
    Checks printed with the result, because each could fake it: the share of far-touchline events whose feed y is
    exactly 0 or negative (a feed clipped to the pitch cannot read a player outside it); the events per clip; and the
    same medians with only stands of >= 25 frames.

    .venv/bin/python model/validation/n37_feet_on_lines.py --events    # finds and lists the events; reads NO position
    .venv/bin/python model/validation/n37_feet_on_lines.py --score     # the arms on those events -> model/validation/out/stc2025/n37_score.json
    .venv/bin/python model/validation/n37_feet_on_lines.py --demo      # synthetic: a foot on a 1 px line beside a board is an event; 3 px off is not
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
SEQS = ("117093", "118575", "118576", "118577", "118578", "128057", "128058")
EVENTS = os.path.join(STC, "n37_events.json")
OUT = os.path.join(STC, "n37_score.json")
ARMS = {"SHIP": "n34_autocal.json", "KAPPA": "n36_after_k-0.4.json"}
LABEL_PX, LOCATE_PX, STILL_PX, MIN_STILL, MIN_FAR, MIN_CLIPS = 15.0, 12.0, 1.5, 10, 30, 3
# depth lines: y of each, in the 105 x 68 drawing's frame (y = 0 is the far touchline)
DEPTH_LINES = {"touch_far": 0.0, "pen_far": 13.84, "six_far": 24.84, "six_near": 43.16, "pen_near": 54.16, "touch_near": 68.0}



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


def line_pixels(cam, width, length=105.0):
    """Each depth line as image points (N,2), x from 0 to `length` at 0.1 m, through `cam`."""
    import autocal as A  # noqa: PLC0415
    xs = np.arange(0.0, length + 0.05, 0.1)
    out = {}
    for name, y in DEPTH_LINES.items():
        if name.startswith("pen"):
            keep = (xs <= 16.5) | (xs >= length - 16.5)
        elif name.startswith("six"):
            keep = (xs <= 5.5) | (xs >= length - 5.5)
        else:
            keep = np.ones_like(xs, bool)
        out[name] = A._fwd(cam, np.column_stack([xs[keep], np.full(keep.sum(), y)]), float(width))
    return out


def ridge_at(native, grass, u, v, radius=LOCATE_PX):
    """The strongest thin grass-flanked ridge on the vertical through (u, v) within `radius` rows -> (row, half-width) or None."""
    import autocal as A  # noqa: PLC0415
    s = np.arange(-radius, radius + 0.25, 0.5)
    rows = v + s
    P = map_coordinates(native, [rows, np.full_like(rows, u)], order=1)
    G = map_coordinates(grass, [rows, np.full_like(rows, u)], order=1)
    best = None
    for j in range(1, len(s) - 1):
        if not (P[j] >= P[j - 1] and P[j] > P[j + 1]):
            continue
        for w in A.FLANKS_PX:
            f = int(round(w / 0.5))
            if f <= j < len(s) - f and G[j - f] > A.GRASS_MIN and G[j + f] > A.GRASS_MIN:
                c = P[j] - 0.5 * (P[j - f] + P[j + f])
                if c >= A.MIN_PEAK and (best is None or c > best[0]):
                    best = (c, j, f)
    if best is None:
        return None
    c, j, f = best
    side = 0.5 * (P[j - f] + P[j + f])
    a, b, d = P[j - 1], P[j], P[j + 1]
    centre = rows[j] + 0.5 * 0.5 * (a - d) / (a - 2 * b + d - 1e-9)
    half = P >= side + 0.5 * c
    lo, hi = j, j
    while lo > 0 and half[lo - 1]:
        lo -= 1
    while hi < len(s) - 1 and half[hi + 1]:
        hi += 1
    return float(centre), float(0.5 * (s[hi] - s[lo]) + 0.25)


def stands(feet):
    """{id: [(frames, u, v)]}: runs of >= MIN_STILL consecutive even frames where the foot moved <= STILL_PX."""
    by_id = {}
    for f in sorted(feet):
        if f % 2:
            continue
        for pid, (u, v) in feet[f].items():
            by_id.setdefault(pid, []).append((f, u, v))
    out = []
    for pid, seq in by_id.items():
        run = [seq[0]]
        for cur in seq[1:]:
            prev = run[-1]
            if cur[0] == prev[0] + 2 and abs(cur[1] - prev[1]) <= STILL_PX and abs(cur[2] - prev[2]) <= STILL_PX:
                run.append(cur)
            else:
                if len(run) >= MIN_STILL:
                    out.append((pid, run))
                run = [cur]
        if len(run) >= MIN_STILL:
            out.append((pid, run))
    return out


def paint_rows(bgr, cam):
    """{line: (u sorted, v) of the paint found ALONG each depth line by model/engine/autocal.search}."""
    import autocal as A  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415
    X, T, grp = A.model_samples(Pitch(105.0, 68.0), 0.25)
    fam = {"touch_far": ("touch_far",), "touch_near": ("touch_near",), "pen_far": ("pen_l_far", "pen_r_far"), "pen_near": ("pen_l_near", "pen_r_near"),
           "six_far": ("six_l_far", "six_r_far"), "six_near": ("six_l_near", "six_r_near")}
    out_ = np.array([A.OUTWARD.get(g, (0.0, 0.0)) for g in grp])
    hit, d, _n = A.search(A.ridge_map(bgr), cam, X, T, 8.0, grass=A.grass_map(bgr), outward=out_, native=A.native_map(bgr))
    rows = {}
    for name, gs in fam.items():
        m = hit & np.isin(grp, gs)
        o = np.argsort(d[m, 0])
        rows[name] = (d[m, 0][o], d[m, 1][o])
    return rows


def find_events(seq):
    import autocal as A  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    bgr = cv2.imread(os.path.join(STC, f"{seq}.median40.png"))
    native, grass = A.native_map(bgr), A.grass_map(bgr)
    cam = np.array(json.load(open(os.path.join(STC, ARMS["SHIP"])))[seq]["camera"], float)
    lines, paint = line_pixels(cam, bgr.shape[1]), paint_rows(bgr, cam)
    events = []
    for pid, run in stands(D.G_mot_feet(seq)):
        u, v = float(np.median([r[1] for r in run])), float(np.median([r[2] for r in run]))
        label, dist = None, LABEL_PX
        for name, pts in lines.items():
            dd = np.min(np.hypot(pts[:, 0] - u, pts[:, 1] - v))
            if dd < dist:
                label, dist = name, dd
        if label is None:
            continue
        pu, pv = paint[label]
        left, right = pu[pu <= u], pu[pu >= u]
        if not len(left) or not len(right) or u - left[-1] > 20 or right[0] - u > 20:
            continue                                            # no paint found on both sides of the foot's column
        v_paint = float(np.interp(u, pu, pv))
        r = ridge_at(native, grass, u, v_paint, radius=4.0)
        if r is None or abs(r[0] - v_paint) > 1.0 or (label.endswith("far") and r[1] > 3.0):
            continue
        centre, half = r
        if abs(v - centre) <= max(1.0, half):
            events.append({"seq": seq, "mot": int(pid), "frames": [int(x[0]) for x in run], "u": round(u, 2), "v": round(v, 2),
                           "line": label, "ridge_row": round(centre, 2), "ridge_half_px": round(half, 2), "label_px": round(float(dist), 1)})
    return events


def events_cmd():
    allev = []
    for q in SEQS:
        ev = find_events(q)
        allev += ev
        print(q, len(ev), "events:", {k: sum(e["line"] == k for e in ev) for k in DEPTH_LINES}, flush=True)
    json.dump(allev, open(EVENTS, "w"), indent=1)
    print(len(allev), "events ->", EVENTS)


def score():
    import autocal as A  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n37_feet_on_lines.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the protocol must be committed before any position is read")
    events = json.load(open(EVENTS))
    cams = {a: json.load(open(os.path.join(STC, f))) for a, f in ARMS.items()}
    rows = []
    for q in SEQS:
        gt = D.stc_gt(q, D.STC_EVERY)
        for e in [e for e in events if e["seq"] == q]:
            g = gt.get(str(e["mot"]))
            if g is None:
                continue
            fr = g["f"].tolist()
            idx = [fr.index(f) for f in e["frames"] if f in fr]
            if len(idx) < MIN_STILL // 2:
                continue
            truth = DEPTH_LINES[e["line"]]
            row = {**e, "n_frames": len(idx), "feed_y": float(np.median(g["xy"][idx, 1])), "truth": truth}
            for a in ARMS:
                A.VERT_CUBIC = -0.4 if a == "KAPPA" else 0.0
                row[f"{a}_y"] = float(A._inv(np.array(cams[a][q]["camera"], float), np.array([[e["u"], e["v"]]]))[0, 1])
            A.VERT_CUBIC = 0.0
            rows.append(row)
    res = {"events": len(rows), "lines": {}}
    for name, y in DEPTH_LINES.items():
        sel = [r for r in rows if r["line"] == name]
        long = [r for r in sel if r["n_frames"] >= 25]
        res["lines"][name] = {"n": len(sel), "clips": len({r["seq"] for r in sel}),
                              **{f"med_{a}": (round(float(np.median([r[f"{a}_y"] - y for r in sel])), 2) if sel else None) for a in (*ARMS, "feed")},
                              "n_long": len(long),
                              **{f"long_{a}": (round(float(np.median([r[f"{a}_y"] - y for r in long])), 2) if long else None) for a in (*ARMS, "feed")}}
    far = [r for r in rows if r["line"] == "touch_far"]
    res["far_feed_at_or_outside_zero_share"] = round(float(np.mean([r["feed_y"] <= 0.0 for r in far])), 3) if far else None
    res["events_per_clip"] = {q: sum(r["seq"] == q for r in rows) for q in SEQS}
    L = res["lines"]["touch_far"]
    verdict = "UNRESOLVED (too few)"
    if L["n"] >= MIN_FAR and L["clips"] >= MIN_CLIPS:
        ms, mf = abs(L["med_SHIP"]), abs(L["med_feed"])
        verdict = "MAP" if ms <= 0.5 and mf >= 1.5 else "FEED" if mf <= 0.5 and ms >= 1.5 else "UNRESOLVED"
    res["owner_of_the_4pct"] = verdict
    res["rows"] = rows
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{'line':11s} {'n':>4s} clips  {'SHIP':>7s} {'KAPPA':>7s} {'FEED':>7s}   (median signed y error, m; long stands: n, SHIP, KAPPA, FEED)")
    for name, L in res["lines"].items():
        print(f"{name:11s} {L['n']:4d} {L['clips']:5d}  {str(L['med_SHIP']):>7s} {str(L['med_KAPPA']):>7s} {str(L['med_feed']):>7s}   ({L['n_long']}, {L['long_SHIP']}, {L['long_KAPPA']}, {L['long_feed']})")
    print("far-touchline events with feed y <= 0:", res["far_feed_at_or_outside_zero_share"], "| per clip:", res["events_per_clip"])
    print("OWNER of the 4%:", verdict, "->", OUT)


def demo():
    import autocal as A  # noqa: PLC0415
    img = np.full((200, 200, 3), (40, 110, 40), np.uint8)
    img[:96, :] = (90, 90, 95)                 # stand
    img[96:100, :] = (250, 250, 250)           # board, 4 rows, ending 4 px above the line
    img[104, :] = (160, 200, 160)              # 1 px of faint far paint on grass
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    r = ridge_at(A.native_map(img), A.grass_map(img), 100.0, 105.0)
    assert r is not None and abs(r[0] - 104.0) < 0.5 and r[1] <= 1.5, r
    assert abs(105.0 - r[0]) <= max(1.0, r[1])          # a foot 1 px off a 1 px line is on it
    assert abs(108.0 - r[0]) > max(1.0, r[1])           # 4 px off is not
    still = stands({f: {1: (50.0 + 0.1 * f, 40.0), 2: (50.0 + 3.0 * f, 40.0)} for f in range(0, 60, 2)})
    assert [pid for pid, _ in still] == [1], still      # a walker is never an event
    print("n37 demo ok: the paint beside a board is found and the board is not; a walker is not a stand")


if __name__ == "__main__":
    (events_cmd if "--events" in sys.argv else score if "--score" in sys.argv else demo)()
