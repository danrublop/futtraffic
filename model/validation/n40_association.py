"""N40: the tracklet-association cost - protocol before code, split before scores.

WHY
    Association is the product's binding constraint (plan-parallel-tracks (not published) Track B step 1): 44-61 fragments
    per four-minute clip for 22 players, HOTA 0.369 against the winner's 0.60 (the '0.42 floor' was ByteTrack on the winner's fine-tuned detector - docs/experiments/n45-winner-boxes.md), and an oracle merge of
    the SAME tracklets already beats the winner (association-not-detection-clears-the-leaderboard (internal note)).
    model/engine/stitch.py joins fragments cheapest-first on a pitch-plane miss distance, and its own docstring says where
    it stops: the nearest fragment to an exit is usually the WRONG one, because tracks end where another player
    stands on top of them; appearance is dead (same kit, 24-50 px); pitch-plane distance amplifies far-side pixel
    noise by up to ten times. Two costs follow from that diagnosis and neither has been measured:

    V1  NORMALISED miss: the same miss distance divided by what it should be if the join were right - box-bottom
        noise through the calibration's local metres-per-pixel at each end (sigma_px = 1.5 px, the Jacobian's
        vertical scale), plus an unmodelled-acceleration term A * gap^2 / 2 with A = 2 m/s^2. A 2 m miss is noise
        on the far touchline and a different player near the camera; today they cost the same.
    V2  OCCLUDER-CARRIED coast: while a player is hidden, they are where the player hiding them is. For an exit at
        frame f, the occluders are the tracklets alive within +-2 frames of f and within 2 m; the exit is coasted
        along each occluder's own path to the entry's frame, and the miss is the smallest of own-coast and
        occluder-carried. Same gate, same greedy loop.
    V3  both.
    V0  the shipped cost, unchanged: the control, and the reproduction check.

POPULATION
    The seven same-install STC clips, the product's own detections (model/validation/out/stc2025/<seq>.p35.dets.cache.json,
    coreml, conf 0.5, stride 2) replayed through the product's tracker with stitching OFF (`.p40raw`), then
    model/engine/track.stitch_tracks' exact path (split -> stitch -> apply -> team re-vote) with the cost swapped in.
    Calibration for the metres: the clip's zero-click camera (`.p35.calibration.json`), the frozen baseline map.
    Ground truth: the STC MOT boxes; TrackEval HOTA/DetA/AssA/IDF1/IDSW through model/validation/st2_mot.py.

    SPLIT, fixed now.  DEV  = 118575, 118576, 117093 (everything is looked at here).
                       HELD = 118577, 118578, 128057, 128058 (looked at ONCE, by one configuration).
    All seven are one install; the split is by match, not by camera. It guards against tuning to three matches'
    crowds, not against a second rig.

TWO LEVELS
    L1  link discrimination, DEV only. Every feasible exit->entry pair under the shipped gate (gap <= 10 s, speed
        <= 9 m/s + 4 m slack) between split tracklets whose ground-truth identity is known (>= 80% of a tracklet's
        boxes match one GT id at IoU >= 0.5; both tracklets known). Label: same GT id or not. Score: AUC of each
        cost as a same-player ranker, pooled over DEV, with n pairs and the same-player share. Constants above are
        not swept; n = 4 costs.
    L2  end to end. The shipped cost and the best L1 cost by AUC (ONE candidate; if two are within 0.01 AUC the
        simpler one) run through the full stitch on DEV; the candidate then runs ONCE on HELD.

RULE, fixed now. The candidate REPLACES the shipped cost iff, on HELD (mean of four clips unless said):
      AssA  >= shipped + 0.02;   IDF1 not lower than shipped on >= 3 of 4 clips;   IDSW <= 1.10 x shipped;
      DetA identical to the shipped run on every clip (stitching relabels; it cannot move a box - a change here is a bug).
    Anything else: NOT SUPPORTED, the shipped cost stays, and the numbers are published anyway.
    Reproduction check, before anything is believed: V0 through this harness must equal the product's `.p35` stitched
    tracks on TrackEval to three decimals on every clip; if it does not, the harness is wrong and nothing is scored.

    .venv/bin/python model/validation/n40_association.py --build       # raw tracklets for the seven clips (~20 s each)
    .venv/bin/python model/validation/n40_association.py --l1          # link AUCs on DEV -> model/validation/out/stc2025/n40_l1.json
    .venv/bin/python model/validation/n40_association.py --l2 V2       # end to end, DEV; --l2 V2 --held for the one HELD run
    .venv/bin/python model/validation/n40_association.py --show        # what the stored runs say, and the rule
    .venv/bin/python model/validation/n40_association.py --demo
"""
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
DEV = ("118575", "118576", "117093")
HELD = ("118577", "118578", "128057", "128058")
GT_MOT = _ST2 + "/mot"
SIGMA_PX, ACCEL, OCC_FRAMES, OCC_M, PURITY = 1.5, 2.0, 2, 2.0, 0.8
FPS = 25.0




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

# ----------------------------------------------------------------------------- the costs
def _local_scale(cal, uv):
    """metres per pixel of vertical image motion at a pixel: the position estimate's own noise gain."""
    J = cal.jacobian(np.asarray(uv, float).reshape(1, 2))[0]          # d(pitch)/d(px), 2x2
    return float(np.linalg.norm(J[:, 1]))


def make_link(variant, cal, tl, fps, px=None):
    """A drop-in for stitch._link for one variant. `tl` is the tracklet dict (needed by V2 for the occluders);
    `px` = {tracklet id: {0: entry pixel, -1: exit pixel}} for V1's local scale (_Tracklet has __slots__)."""
    import stitch as S  # noqa: PLC0415
    base = S._link
    if variant == "V0":
        return base
    ends = {tid: (t.f[-1], t.xy[-1]) for tid, t in tl.items()}
    scale = {}                                                       # per tracklet end: m/px, from the pixel cache

    def sig(t, end):
        key = (t.id, end)
        if key not in scale:
            uv = (px or {}).get(t.id, {}).get(end)
            scale[key] = SIGMA_PX * (_local_scale(cal, uv) if uv is not None else 0.3)
        return scale[key]

    def carried(a, b, gap_s):
        """Smallest miss when a's exit rides an occluder to b's entry frame."""
        best = None
        fa, xa = ends[a.id]
        for o in tl.values():
            if o.id in (a.id, b.id) or o.start > fa + OCC_FRAMES or o.end < fa - OCC_FRAMES:
                continue
            k = int(np.searchsorted(o.f, fa))
            k = min(max(k, 0), len(o.f) - 1)
            if float(np.hypot(*(o.xy[k] - xa))) > OCC_M:
                continue
            j = int(np.searchsorted(o.f, b.start)) - 1                # the occluder's last sample before b starts
            j = min(max(j, 0), len(o.f) - 1)
            pos = o.xy[j] + S._vel(o, -1, fps) * max(0.0, (b.start - o.f[j]) / fps) if o.end < b.start else o.xy[j]
            m = float(np.hypot(*(b.xy[0] - pos)))
            best = m if best is None or m < best else best
        return best

    def link(a, b, fps_, max_speed, slack_m, max_gap_s):
        c = base(a, b, fps_, max_speed, slack_m, max_gap_s)
        if c is None:
            return None
        gap_s = (b.start - a.end) / fps_
        if variant in ("V2", "V3"):
            m = carried(a, b, gap_s)
            if m is not None and m < c:
                c = m
        if variant in ("V1", "V3"):
            s = float(np.sqrt(sig(a, -1) ** 2 + sig(b, 0) ** 2 + (0.5 * ACCEL * gap_s ** 2) ** 2))
            c = c / max(s, 1e-6)
        return c
    return link


# ----------------------------------------------------------------------------- data
def build():
    import st2_product as ST  # noqa: PLC0415
    for q in DEV + HELD:
        print(ST.track(q, ST._p(q, ".p35.calibration.json"), tag=".p40raw", stitch=False, cache_from=".p35"), flush=True)


def _rows(q):
    return json.load(open(os.path.join(STC, f"{q}.p40raw.tracks.json")))["detections"]


def _cal(q):
    import calibrators as C  # noqa: PLC0415
    return C.load(os.path.join(STC, f"{q}.p35.calibration.json"))


def _split_tracklets(q):
    """The product's split, then tracklets in metres with their pixel endpoints attached."""
    import stitch as S  # noqa: PLC0415
    rows = S.split(_rows(q), _cal(q), fps=FPS)
    tl = S.tracklets(rows, _cal(q))
    by = {}
    for r in rows:
        by.setdefault(int(r["id"]), []).append(r)
    px = {}
    for tid in tl:
        rs = sorted(by[tid], key=lambda r: r["f"])
        px[tid] = {0: np.array([rs[0]["x"], rs[0]["y"]], float), -1: np.array([rs[-1]["x"], rs[-1]["y"]], float)}
    return rows, tl, px


def gt_identity(q, rows):
    """{tracklet id: (majority GT id, purity)} by IoU >= 0.5 against the MOT ground truth."""
    gt = {}
    for line in open(f"{GT_MOT}/{q}.txt"):
        f, i, x, y, w, h = line.split(",")[:6]
        gt.setdefault(int(f), []).append((int(i), float(x), float(y), float(w), float(h)))
    votes = {}
    for r in rows:
        best, bi = 0.0, None
        x0, y0, x1, y1 = r["x"] - r["w"] / 2, r["y"] - r["h"], r["x"] + r["w"] / 2, r["y"]
        for gi, gx, gy, gw, gh in gt.get(int(r["f"]), []):
            iw, ih = min(x1, gx + gw) - max(x0, gx), min(y1, gy + gh) - max(y0, gy)
            if iw <= 0 or ih <= 0:
                continue
            iou = iw * ih / (r["w"] * r["h"] + gw * gh - iw * ih)
            if iou > best:
                best, bi = iou, gi
        votes.setdefault(int(r["id"]), Counter())[bi if best >= 0.5 else None] += 1
    out = {}
    for tid, c in votes.items():
        gid, n = c.most_common(1)[0]
        out[tid] = (gid, n / sum(c.values()))
    return out


# ----------------------------------------------------------------------------- L1
def _auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(s)                                             # low cost = predicted same
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos, neg = y.sum(), (~y).sum()
    return float((ranks[~y].sum() - neg * (neg + 1) / 2) / (pos * neg)) if pos and neg else float("nan")   # P(cost_same < cost_diff)


def l1():
    import stitch as S  # noqa: PLC0415
    res, pooled = {}, {v: ([], []) for v in ("V0", "V1", "V2", "V3")}
    for q in DEV:
        rows, tl, px = _split_tracklets(q)
        ident = gt_identity(q, rows)
        known = {tid for tid, (g, p) in ident.items() if g is not None and p >= PURITY}
        links = {v: make_link(v, _cal(q), tl, FPS, px) for v in pooled}
        n, same = 0, 0
        ids = sorted(tl)
        for i in ids:
            for j in ids:
                if i == j or tl[i].end >= tl[j].start or i not in known or j not in known:
                    continue
                c0 = links["V0"](tl[i], tl[j], FPS, S.MAX_SPEED, S.SLACK_M, S.MAX_GAP_S)
                if c0 is None:
                    continue
                lab = ident[i][0] == ident[j][0]
                n, same = n + 1, same + int(lab)
                for v in pooled:
                    c = links[v](tl[i], tl[j], FPS, S.MAX_SPEED, S.SLACK_M, S.MAX_GAP_S)
                    pooled[v][0].append(c); pooled[v][1].append(lab)
        res[q] = {"pairs": n, "same": same, "tracklets": len(tl), "known": len(known)}
        print(q, res[q], flush=True)
    res["AUC"] = {v: round(_auc(*pooled[v]), 4) for v in pooled}
    res["pairs_pooled"], res["same_pooled"] = len(pooled["V0"][0]), int(sum(pooled["V0"][1]))
    json.dump(res, open(os.path.join(STC, "n40_l1.json"), "w"), indent=1)
    print("AUC on DEV links:", res["AUC"], "pairs", res["pairs_pooled"], "same", res["same_pooled"])


# ----------------------------------------------------------------------------- L2
def stitched_rows(q, variant):
    """model/engine/track.stitch_tracks' exact path with the cost swapped: split -> stitch -> apply -> team re-vote."""
    import stitch as S  # noqa: PLC0415
    rows, tl, px = _split_tracklets(q)
    labels = {int(d["id"]): d["team"] for d in rows}
    keep = S._link
    S._link = make_link(variant, _cal(q), tl, FPS, px)
    try:
        rows = S.apply(rows, S.stitch(tl, FPS, teams=labels))
    finally:
        S._link = keep
    votes = {}
    for d in rows:
        votes.setdefault(int(d["id"]), Counter())[d["team"]] += 1
    for d in rows:
        d["team"] = votes[int(d["id"])].most_common(1)[0][0]
    return rows


def l2(variant, held=False):
    import st2_mot as M  # noqa: PLC0415
    seqs = HELD if held else DEV
    if held and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n40_association.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the held-out run must be made by the committed protocol")
    tag = f".p40{variant.lower()}"
    for q in seqs:
        out = os.path.join(STC, f"{q}{tag}.tracks.json")
        if not os.path.exists(out):
            src = json.load(open(os.path.join(STC, f"{q}.p40raw.tracks.json")))
            json.dump({**src, "detections": stitched_rows(q, variant)}, open(out, "w"))
    M.VARIANT = tag
    M.stage_gt(seqs)
    M.stage_pred(seqs)
    res = M.run(seqs)
    for q in seqs:
        res[q]["ids"] = len({r["id"] for r in json.load(open(os.path.join(STC, f"{q}{tag}.tracks.json")))["detections"]})
    if variant == "V0" and not held:
        # reproduction: the product's own stitched tracks through the same scorer
        M.VARIANT = ".p35"
        M.stage_gt(seqs); M.stage_pred(seqs)
        gold = M.run(seqs)
        for q in seqs:
            for k in ("HOTA", "DetA", "AssA", "IDF1"):
                assert abs(gold[q][k] - res[q][k]) < 5e-4, (q, k, gold[q][k], res[q][k])
        print("reproduction OK: V0 through this harness == the product's .p35 tracks on TrackEval")
    json.dump(res, open(os.path.join(STC, f"n40_l2_{variant}_{'held' if held else 'dev'}.json"), "w"), indent=1)


def show():
    import glob  # noqa: PLC0415
    p = os.path.join(STC, "n40_l1.json")
    if os.path.exists(p):
        r = json.load(open(p))
        print("L1 AUC on DEV links:", r["AUC"], "| pairs", r["pairs_pooled"], "same-player", r["same_pooled"])
    runs = {os.path.basename(f)[7:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(STC, "n40_l2_*.json")))}
    for name, r in runs.items():
        seqs = [q for q in r]
        print(f"{name:10s} " + "  ".join(f"{q}: HOTA {r[q]['HOTA']:.3f} DetA {r[q]['DetA']:.3f} AssA {r[q]['AssA']:.3f} IDF1 {r[q]['IDF1']:.3f} IDSW {r[q]['IDSW']:.0f} ids {r[q].get('ids', '?')}" for q in seqs)
              + f"  | mean AssA {np.mean([r[q]['AssA'] for q in seqs]):.4f} IDF1 {np.mean([r[q]['IDF1'] for q in seqs]):.4f} IDSW {np.mean([r[q]['IDSW'] for q in seqs]):.1f}")
    base, cand = runs.get("V0_held"), [v for k, v in runs.items() if k.endswith("_held") and not k.startswith("V0")]
    if base and cand:
        c = cand[0]
        a_ok = np.mean([c[q]["AssA"] for q in HELD]) >= np.mean([base[q]["AssA"] for q in HELD]) + 0.02
        i_ok = sum(c[q]["IDF1"] >= base[q]["IDF1"] for q in HELD) >= 3
        s_ok = np.mean([c[q]["IDSW"] for q in HELD]) <= 1.10 * np.mean([base[q]["IDSW"] for q in HELD])
        d_ok = all(abs(c[q]["DetA"] - base[q]["DetA"]) < 5e-4 for q in HELD)
        print("RULE on HELD:", "AssA +0.02" if a_ok else "AssA short", "| IDF1 3 of 4" if i_ok else "| IDF1 short",
              "| IDSW ok" if s_ok else "| IDSW up", "| DetA identical" if d_ok else "| DetA MOVED (bug)",
              "->", "REPLACES the shipped cost" if a_ok and i_ok and s_ok and d_ok else "NOT SUPPORTED")


def demo():
    """V1 ranks a far-side 2 m miss as cheaper than a near-side one; V2 rides an occluder."""
    import stitch as S  # noqa: PLC0415

    class Cal:
        def jacobian(self, uv):
            u, v = np.asarray(uv, float)[0]
            m = 0.6 if v < 300 else 0.05                                # far rows: 0.6 m/px; near rows: 0.05
            return np.array([[[0.02, 0.0], [0.0, m]]])

    pxd = {}

    def T(tid, f0, f1, x0, y0, x1, y1, v):
        pxd[tid] = {0: np.array([100.0, v]), -1: np.array([100.0, v])}
        return S._Tracklet(tid, np.arange(f0, f1 + 1), np.column_stack([np.linspace(x0, x1, f1 - f0 + 1), np.linspace(y0, y1, f1 - f0 + 1)]))
    far_a, far_b = T(1, 0, 20, 50, 5, 50, 5, 150.0), T(2, 30, 50, 52, 5, 52, 5, 150.0)      # 2 m apart, far side
    near_a, near_b = T(3, 0, 20, 50, 60, 50, 60, 900.0), T(4, 30, 50, 52, 60, 52, 60, 900.0)  # 2 m apart, near side
    tl = {t.id: t for t in (far_a, far_b, near_a, near_b)}
    v0, v1 = make_link("V0", Cal(), tl, FPS, pxd), make_link("V1", Cal(), tl, FPS, pxd)
    assert abs(v0(far_a, far_b, FPS, 9, 4, 10) - v0(near_a, near_b, FPS, 9, 4, 10)) < 1e-9
    assert v1(far_a, far_b, FPS, 9, 4, 10) < v1(near_a, near_b, FPS, 9, 4, 10)
    # occluder: a ends at (50,30) where o stands; o walks to (60,30) by frame 50; b appears there
    a, b = T(5, 0, 20, 50, 30, 50, 30, 500.0), T(6, 50, 60, 60, 30, 60, 30, 500.0)
    o = T(7, 20, 49, 50, 30, 60, 30, 500.0)
    tl2 = {t.id: t for t in (a, b, o)}
    c0, c2 = make_link("V0", Cal(), tl2, FPS, pxd)(a, b, FPS, 9, 4, 10), make_link("V2", Cal(), tl2, FPS, pxd)(a, b, FPS, 9, 4, 10)
    assert c0 > 5 and c2 < 0.5, (c0, c2)
    assert abs(_auc([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0]) - 1.0) < 1e-9 and abs(_auc([0.4, 0.3, 0.2, 0.1], [1, 1, 0, 0])) < 1e-9
    print("n40 demo ok: V1 normalises far-side noise, V2 rides the occluder, AUC is oriented low-cost = same")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--build" in a:
        build()
    elif "--l1" in a:
        l1()
    elif "--l2" in a:
        l2(a[a.index("--l2") + 1], held="--held" in a)
    elif "--show" in a:
        show()
    else:
        demo()
