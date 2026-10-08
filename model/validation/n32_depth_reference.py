"""N32: who owns the 4%? A throw-in taker stands on the touchline - a depth reference neither map nor feed made.

PROTOCOL, COMMITTED BEFORE ANY POSITION OF ANY THROW-IN TAKER IS READ.

WHY
    N31 (docs/experiments/n31-line-registration.md §5) left a line-registered spherical map and ST2's position
    feed agreeing to 0.34-0.47 m ACROSS the camera ray and disagreeing by a near-constant 4% of range
    ALONG it: the map puts every player closer to the camera than the feed does. Interior paint held
    out of the fit says the map is right to half a metre; ST2's chain and four kick-offs of seven say
    the feed is. Metres-at-feet against the feed cannot separate them. A law of the game can: the
    thrower "has part of each foot on the touchline or on the ground outside the touchline" (Law 15).
    On the far touchline, about 84 m from this camera, 4% is about 3.4 m.

POPULATION
    clips        the seven same-install clips N31 scored (117092 is refused by the registration).
    events       `throwIn` rows of raw/<seq>_player_nodes.csv in the clip's period, whose event_time
                 maps (nearest tracker frame by matchTime) to a clip frame; the thrower is the row's
                 player_id, carried to a MOT id by st2_team_gt's join. No choice of player is made here.
    instant      f* = the event's clip frame minus 10 (0.4 s before the annotated release, feet
                 planted), rounded down to even. The MOT box and the feed position must both exist at f*.
    REVIEW, BLIND TO EVERY ESTIMATE   `--candidates` writes one crop per event at f* with the thrower's
                 MOT box drawn, and a list that carries NO position from any source. Each crop is then
                 accepted or rejected BY EYE, before `--score` exists to be run, on three fixed grounds:
                 (a) it is not a throw-in, or the boxed player is not the thrower;
                 (b) the thrower's feet are hidden;
                 (c) the feet are visibly NOT at the touchline at f* (already stepped in, or standing
                     well back) - the law is the truth only while the feet are on the line.
                 Accepted keys and every rejection's ground go in model/validation/n32_valid.json, committed
                 before `--score`.

TRUTH   far touchline: feet y in [-1.0, +0.1] m, taken as -0.3; near: [67.9, 69.0], taken as 68.3.
        Which touchline: far if the feed's y < 34, near otherwise (4% cannot move a player across the pitch).

ARMS (nothing fitted, nothing tuned; every one already exists on disk)
    FEED     the joined feed at f* (dist_eval.stc_gt).
    LINE     model/validation/out/stc2025/n31_linereg.json through the MOT box's foot - the cameras N31 scored.
    CHAIN    ST2's own calibration through the same foot. Reported, not ruled.

RULE, fixed now, on FAR throw-ins only (near ones are reported: 4% of 15 m is inside the truth's tolerance)
    With e = estimated y minus -0.3, pooled over clips:
      OWNER = LINE  iff |median e_LINE| <= 1.0 m and |median e_FEED| >= 2.0 m   (the feed is pushed away)
      OWNER = FEED  iff |median e_FEED| <= 1.0 m and |median e_LINE| >= 2.0 m   (the map is pulled in)
      anything else, or fewer than 8 accepted far throw-ins: UNRESOLVED.
    Predictions written first. If the map is right and the feed is magnified 4.3% about the camera's
    foot (y = 83.3): e_FEED about -3.6, e_LINE about 0. If the feed is right: e_LINE about +3.4, e_FEED about 0.

EXTENSION, written 2026-09-17 AFTER the blind review and BEFORE any position was read
    The clips are four minutes each: 11 throw-ins had a box at f*, 9 passed review, and only 3 of those
    are on the FAR touchline (model/validation/n32_valid.json) - the rule cannot reach 8 on throw-ins alone.
    A second source of the same truth, defined with no map and no feed:
    edge-standers   on clip frames with f % 50 == 0, a MOT box in the far part of the image whose foot
                 pixel (x + w/2, y + h) lies within 2.0 px of the GRASS EDGE in its own column - the
                 first row, scanning down that column of the clip's 40-frame median, where
                 n31_linereg.grass_map stays positive for 6 rows. On this install the far touchline is
                 painted at the grass edge (the run-off beyond it is boards). One per clip, player and
                 250 frames. `--edge` writes their crops and a list with no positions.
    review       by eye, blind as above: accept only a player visibly STANDING with feet at the far
                 touchline; reject if the box bottom is not the feet, the player is airborne or
                 sliding, or the grass edge there is not the touchline. Accepted keys join
                 n32_valid.json under "edge", committed before `--score`.
    truth        y = 0.0 for an edge-stander (on the line; 2 px is about 1 m out there).
    rule         unchanged, on far throw-ins and edge-standers POOLED, each against its own truth.
                 Both sources are also printed apart.

    .venv/bin/python model/validation/n32_depth_reference.py --candidates   # crops + list, no positions
    .venv/bin/python model/validation/n32_depth_reference.py --edge         # the extension's crops + list
    .venv/bin/python model/validation/n32_depth_reference.py --score        # needs model/validation/n32_valid.json committed
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

SEQS = ("117093", "118575", "118576", "118577", "118578", "128057", "128058")
CROPS = os.path.join(ROOT, "model/validation/out/stc2025/n32_crops")
CAND = os.path.join(ROOT, "model/validation/out/stc2025/n32_candidates.json")
VALID = os.path.join(ROOT, "model/validation/n32_valid.json")
OUT = os.path.join(ROOT, "model/validation/out/stc2025/n32_score.json")
LEAD_FRAMES, TRUTH, MIN_FAR = 10, {"far": -0.3, "near": 68.3}, 8
_FR = re.compile(rb'<frame matchTime="(-?[0-9]+)" frameNumber="([0-9]+)" eventPeriod="([A-Z_]+)"')



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


def events(seq):
    """-> [{key, f_event, f, mot}] for throw-ins inside the clip. Carries no position."""
    import st2_team_gt as G  # noqa: PLC0415
    j = G.load()[seq]
    t, fn = [], []
    with open(f"{G.RAW}/{seq}/{seq}_tracker_box_data.xml", "rb") as fh:
        for line in fh:
            m = _FR.search(line)
            if m and m.group(3).decode() == j["period"]:
                t.append(int(m.group(1)))
                fn.append(int(m.group(2)))
    t, fn = np.array(t), np.array(fn)
    order = np.argsort(t)
    t, fn = t[order], fn[order]
    p2m = {int(v): int(k) for k, v in j["mot_to_player"].items()}
    out = []
    for r in csv.DictReader(open(f"{G.RAW}/{seq}/{seq}_player_nodes.csv")):
        if r["event_period"] != j["period"] or "throwIn" not in r["filtered_event_types"].split():
            continue
        i = int(np.clip(np.searchsorted(t, int(r["event_time"])), 1, len(t) - 1))
        i = i if abs(t[i] - int(r["event_time"])) < abs(t[i - 1] - int(r["event_time"])) else i - 1
        f_event = int(fn[i]) - int(j["offset"])
        f = (f_event - LEAD_FRAMES) // 2 * 2
        if not 0 <= f < int(j["clip_frames"]) or int(r["player_id"]) not in p2m:
            continue
        out.append({"key": f"{seq}:{r['id']}", "seq": seq, "f_event": f_event, "f": f, "mot": p2m[int(r["player_id"])]})
    return out


def candidates():
    import cv2  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    os.makedirs(CROPS, exist_ok=True)
    allc = []
    for q in SEQS:
        boxes = {}
        for line in open(f"{os.path.expanduser('~')}/Downloads/st2/mot/{q}.txt"):
            p = line.split(",")
            boxes[(int(p[0]), int(p[1]))] = tuple(float(v) for v in p[2:6])
        cap = cv2.VideoCapture(f"{ST.CLIPS}/{q}.mp4")
        for e in events(q):
            b = boxes.get((e["f"], e["mot"]))
            if b is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, e["f"])
            ok, fr = cap.read()
            if not ok:
                continue
            x, y, w, h = b
            cx, cy = int(x + w / 2), int(y + h)
            x0, y0 = max(0, cx - 110), max(0, cy - 70)
            crop = fr[y0:y0 + 110, x0:x0 + 220].copy()
            cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), (0, 255, 255), 1)
            big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            cv2.putText(big, f"{e['key']} f={e['f']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(CROPS, e["key"].replace(":", "_") + ".jpg"), big)
            allc.append(e)
    with open(CAND, "w") as fh:
        json.dump(allc, fh, indent=1)
    print(len(allc), "throw-in candidates with a MOT box at f* ->", CROPS)
    print({q: sum(c["seq"] == q for c in allc) for q in SEQS})


def grass_edge(bgr):
    """Per column, the first row from the top where grass holds for 6 rows; -1 where it never does."""
    import n31_linereg as R  # noqa: PLC0415
    g = R.grass_map(bgr) > 0
    run = np.ones_like(g)
    for k in range(6):
        run[:-k or None] &= g[k:]
    run[-6:] = False
    first = run.argmax(0)
    return np.where(run.any(0), first, -1)


def edge_candidates():
    import cv2  # noqa: PLC0415
    import n31_linereg as R  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    os.makedirs(CROPS, exist_ok=True)
    allc = []
    for q in SEQS:
        edge = grass_edge(R._median_cached(q))
        last = {}
        cap = cv2.VideoCapture(f"{ST.CLIPS}/{q}.mp4")
        rows = [l.split(",") for l in open(f"{os.path.expanduser('~')}/Downloads/st2/mot/{q}.txt")]
        for p in rows:
            f, mid = int(p[0]), int(p[1])
            if f % 50 or f - last.get(mid, -10**9) < 250:
                continue
            x, y, w, h = (float(v) for v in p[2:6])
            cx, fy = int(round(x + w / 2)), y + h
            if not (0 <= cx < len(edge)) or edge[cx] < 0 or fy > 300 or abs(fy - edge[cx]) > 2.0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            last[mid] = f
            x0, y0 = max(0, cx - 60), max(0, int(fy) - 60)
            crop = fr[y0:y0 + 80, x0:x0 + 120].copy()
            cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), (0, 255, 255), 1)
            big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            key = f"{q}:edge:{mid}:{f}"
            cv2.putText(big, key, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(CROPS, key.replace(":", "_") + ".jpg"), big)
            allc.append({"key": key, "seq": q, "f": f, "mot": mid})
    with open(CAND.replace("candidates", "edge_candidates"), "w") as fh:
        json.dump(allc, fh, indent=1)
    print(len(allc), "edge-stander candidates ->", CROPS, {q: sum(c["seq"] == q for c in allc) for q in SEQS})


def verdict(e_line, e_feed):
    if min(len(e_line), len(e_feed)) < MIN_FAR:
        return "UNRESOLVED (too few)"
    a, b = abs(float(np.median(e_line))), abs(float(np.median(e_feed)))
    return "LINE" if a <= 1.0 and b >= 2.0 else "FEED" if b <= 1.0 and a >= 2.0 else "UNRESOLVED"


def score():
    import dist_eval as D  # noqa: PLC0415
    import n30_operator_maps as N  # noqa: PLC0415
    import n31_linereg as R  # noqa: PLC0415
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n32_depth_reference.py", "model/validation/n32_valid.json"],
                      cwd=ROOT, env=_git_env()).returncode or subprocess.run(["git", "cat-file", "-e", "HEAD:model/validation/n32_valid.json"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the protocol and the reviewed list must be committed before any position is read")
    vj = json.load(open(VALID))
    valid = set(vj["accepted"])
    edge_ok = set(vj.get("edge", {}).get("accepted", []))
    reg = json.load(open(R.OUT))
    rows = []
    for q in SEQS:
        feet, gt, chain = D.G_mot_feet(q), D.stc_gt(q, D.STC_EVERY), N._Chain(q)
        for e in events(q):
            if e["key"] not in valid:
                continue
            uv = feet.get(e["f"], {}).get(e["mot"])
            g = gt.get(str(e["mot"]))
            if uv is None or g is None or e["f"] not in set(g["f"].tolist()):
                rows.append({**e, "dropped": "no box or no feed at f*"})
                continue
            fy = float(g["xy"][g["f"].tolist().index(e["f"])][1])
            side = "far" if fy < 34 else "near"
            ly = float(R._inv(np.array(reg[q]["camera"], float), np.array([uv], float))[0, 1])
            cy = float(chain.to_pitch(np.array([uv], float))[0, 1])
            rows.append({**e, "side": side, "e_feed": fy - TRUTH[side], "e_line": ly - TRUTH[side], "e_chain": cy - TRUTH[side]})
    for q in SEQS:
        feet, gt, chain = D.G_mot_feet(q), D.stc_gt(q, D.STC_EVERY), N._Chain(q)
        for key in sorted(k for k in edge_ok if k.startswith(q + ":")):
            mid, f = int(key.split(":")[2]), int(key.split(":")[3])
            uv, g = feet.get(f, {}).get(mid), gt.get(str(mid))
            if uv is None or g is None or f not in set(g["f"].tolist()):
                rows.append({"key": key, "dropped": "no box or no feed at f"})
                continue
            fy = float(g["xy"][g["f"].tolist().index(f)][1])
            ly = float(R._inv(np.array(reg[q]["camera"], float), np.array([uv], float))[0, 1])
            cy = float(chain.to_pitch(np.array([uv], float))[0, 1])
            rows.append({"key": key, "seq": q, "f": f, "mot": mid, "side": "far", "source": "edge",
                         "e_feed": fy - 0.0, "e_line": ly - 0.0, "e_chain": cy - 0.0})
    far = [r for r in rows if r.get("side") == "far"]
    near = [r for r in rows if r.get("side") == "near"]
    med = lambda rs, k: (round(float(np.median([r[k] for r in rs if np.isfinite(r[k])])), 2) if rs else None)  # noqa: E731
    res = {"accepted": len(valid), "scored": len(far) + len(near), "dropped": [r for r in rows if "dropped" in r],
           "far": {"n": len(far), **{k: med(far, k) for k in ("e_feed", "e_line", "e_chain")}},
           "near": {"n": len(near), **{k: med(near, k) for k in ("e_feed", "e_line", "e_chain")}},
           "owner": verdict([r["e_line"] for r in far], [r["e_feed"] for r in far]), "rows": rows}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    for r in far:
        print(f"  far  {r['key']:22s} f={r['f']:5d}  feed {r['e_feed']:+5.1f}   line {r['e_line']:+5.1f}   chain {r['e_chain']:+5.1f}")
    print(json.dumps({k: res[k] for k in ("accepted", "scored", "far", "near", "owner")}, indent=1), "\n->", OUT)


def demo():
    assert verdict([0.2] * 9, [-3.5] * 9) == "LINE" and verdict([3.3] * 9, [0.1] * 9) == "FEED"
    assert verdict([1.6] * 9, [-1.9] * 9) == "UNRESOLVED" and verdict([0.0] * 5, [-3.5] * 5).startswith("UNRESOLVED")
    print("n32 demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g_ = ap.add_mutually_exclusive_group(required=True)
    for m_ in ("candidates", "edge", "score", "demo"):
        g_.add_argument(f"--{m_}", action="store_true")
    a_ = ap.parse_args()
    (candidates if a_.candidates else edge_candidates if a_.edge else score if a_.score else demo)()
