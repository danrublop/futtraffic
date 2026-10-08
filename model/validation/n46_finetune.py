"""N46: does fine-tuning our detector on this rig's own footage close the gap the STC2025 winner closed? Protocol first.

WHY
    docs/experiments/n45-winner-boxes.md: on the winner's published boxes our unchanged tracker and stitch score HOTA 0.5149 on
    128057 against our shipped 0.3836 and their pipeline's 0.5583 - the detector is 75% of the gap. Their detector was
    fine-tuned on this rig's training clips (research-stc2025-winners (not published)). This does the same for OURS (RF-DETR
    Medium, Apache-2.0) on the same data (SoccerTrack v2, CC BY 4.0), in the product's own tile geometry
    (model/validation/n46_dataset.py), trained on a PC (model/validation/n46_pc/) and judged here, on the Mac, through the product path.

ARMS - every arm's boxes come from model/validation/n46_pc/detect.py: the product's tiled detector at stride 2, every tile's
    boxes saved at conf >= 0.2 BEFORE the merge; on import the product's conf 0.5, its merge (model/engine/detectors._merge at
    0.55) and its pitch prior (the clip's frozen zero-click calibration) are applied - exactly the product at 0.5.
    STOCK   the COCO-pretrained Medium through that PyTorch path. The control: the fine-tune is judged against the SAME
            inference path, so a CoreML-vs-PyTorch difference cannot pass for a training gain.
    FT0 FT1 FT2   three seeds of the same recipe (model/validation/n46_pc/train.py), run records beside the weights.
    SHIP    the shipped CoreML `.p35` caches - reported beside STOCK to show the export's own drift; never judged.

SPLIT (model/validation/n46_dataset.py): TRAIN 117092 117093 118575 132831 132877; VALID 118576; TEST 118577 118578 128057
    128058. One club and one ground on both sides (118578 repeats 117092's pairing); 128057, the challenge clip with
    teams found nowhere else, is printed on its own and against the winner.

ORDER, enforced in code
    1. VALID: STOCK and the three seeds, recall table only. The MEDIAN SEED by the mean of occluded-far, occluded-mid
       and isolated-far recall on VALID is written to model/validation/n46_choice.json - which must be COMMITTED before any
       TEST file is imported. Nothing on TEST picks anything.
    2. TEST, once: STOCK and all three seeds imported and scored; the rule reads the chosen seed; the other two are the
       spread, published beside it.

RULE, fixed now, chosen seed vs STOCK on TEST (means of four clips unless said)
    detection   occluded far >= STOCK + 0.15;  isolated far >= STOCK + 0.08;  occluded and isolated mid and near each
                >= STOCK - 0.01;  non-GT boxes per frame <= STOCK + 0.5   (N40's recall classes, IoU >= 0.5, "shaped")
                AND occluded far FOUND >= STOCK + 0.10, where "found" is one-to-one greedy matching at IoU >= 0.3 - added
                before any training because the review measured stock's occluded-far recall at 0.551 (IoU 0.5) vs 0.747
                (IoU 0.3) on VALID: a model could clear +0.15 by learning the annotators' box shape on players stock
                already finds. Recall is POOLED over the four clips (counts summed), not a mean of per-clip recalls.
    product     HOTA >= STOCK + 0.015 (the measured run-to-run spread, benchmark-status (not published));  DetPr >= STOCK - 0.020
                on every clip;  IDSW <= 1.15 x STOCK;  no-owner share <= STOCK + 0.02.  Distance and HSR raw REPORTED,
                not judged (their spread is 0.07 and 0.18 on four-minute clips).
    Secondary, VALID only, REPORTED not judged: every arm's recall also at conf 0.4 and 0.3 (the saved 0.2 floor replayed;
    n = 3 thresholds), because fine-tuning moves a model's confidence calibration and the rule holds both arms at 0.5.
    A fine-tune that fails at 0.5 and would pass at 0.4 is a second, pre-registered experiment, not a re-read of this one.
    Anything else: NOT SUPPORTED - the shipped weights stay, the numbers are published with n = 1 recipe, 3 seeds.
    Read the product line with this in mind: the ground truth has no officials and the fine-tune learns them as
    background, so part of any DetPr / non-GT / HOTA gain is referees no longer drawn, not players found; the recall lines
    are what say players were found.
    PASS does not ship anything on its own: shipping needs the CoreML export to reproduce the PyTorch boxes (a separate,
    pre-registered check) and says "this ground" until a second rig is tested.

WHAT COULD FOOL IT, written first
    leakage through one club's kits; the ground truth's missing officials rewarding a model that stops seeing people;
    the dataset's box convention (amodal or visible) deciding what "recall" means; a detector that gains recall by
    drawing boxes on grass (DetPr and non-GT guard it); a checkpoint that reloads with scores under 0.5 (train.py's
    reload_check); the PyTorch path differing from the CoreML path (STOCK vs SHIP printed).

    .venv/bin/python model/validation/n46_finetune.py --import DIR          # raw detections from the PC -> product caches (prior applied)
    .venv/bin/python model/validation/n46_finetune.py --valid               # recall table on VALID, writes model/validation/n46_choice.json
    .venv/bin/python model/validation/n46_finetune.py --test                # product path + both rules on TEST (needs the choice committed)
    .venv/bin/python model/validation/n46_finetune.py --show
    .venv/bin/python model/validation/n46_finetune.py                       # self-check
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
VALID, TEST = ("118576",), ("118577", "118578", "128057", "128058")
CHOICE = os.path.join(ROOT, "model/validation/n46_choice.json")
ARMS = ("stock", "ft0", "ft1", "ft2")
CONF, MERGE_THR = 0.5, 0.55                    # the product's detector threshold and TiledDetector's merge_thr
REPLAY = {"c40": 0.4, "c30": 0.3}               # secondary VALID-only thresholds, reported beside the rule
CLASSES = ("occluded_far", "occluded_mid", "occluded_near", "isolated_far", "isolated_mid", "isolated_near")



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


def _committed(path):
    rel = os.path.relpath(path, ROOT)
    return (subprocess.run(["git", "cat-file", "-e", f"HEAD:{rel}"], cwd=ROOT, env=_git_env()).returncode == 0
            and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, env=_git_env()).returncode == 0)


def import_raw(src):
    """<seq>.<tag>.raw.json from detect.py -> <seq>.p46<tag>.dets.cache.json with the product's prior applied and a
    header that model/engine/track.run will replay for detector 'rfdetr' at conf 0.5."""
    import calibrators as C  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    import detectors as D  # noqa: PLC0415
    for p in sorted(glob.glob(os.path.join(src, "*.raw.json"))):
        raw = json.load(open(p))
        seq, tag = raw["seq"], raw["tag"]
        if seq in TEST and tag.startswith("ft"):
            if not _committed(CHOICE):
                raise SystemExit(f"VOID: {p} is a TEST clip; commit model/validation/n46_choice.json (from --valid) first")
            want = json.load(open(CHOICE))["weights_sha256"].get(tag)
            if want != raw.get("weights_sha256"):
                raise SystemExit(f"VOID: {p} was made by weights {str(raw.get('weights_sha256'))[:12]}, but VALID chose {tag} = {str(want)[:12]}")
        # frames_read/frame_count exist from the review-fixed detect.py on; a clip cut short by a decode error is refused
        if "frame_count" in raw and not raw.get("max_frames"):
            assert raw["frames_read"] >= raw["frame_count"] - 5, f"{p}: read {raw['frames_read']} of {raw['frame_count']} frames"
        assert raw["conf"] <= min(CONF, *REPLAY.values()) and raw["merged"] is False and raw["stride"] == 2 and raw["tile"] == 1024 and raw["overlap"] == D.OVERLAP, raw.get("conf")
        cal = C.load(os.path.join(STC, f"{seq}.p35.calibration.json"))
        poly = np.asarray(cal.pitch_polygon_px(), np.float32)
        inside = lambda b: cv2.pointPolygonTest(poly, (float((b[0] + b[2]) / 2), float(b[3])), False) >= 0  # noqa: E731
        head = json.load(open(os.path.join(STC, f"{seq}.p35.dets.cache.json")))
        head = {k: head[k] for k in ("video", "start", "calib", "calib_sha", "settings")}
        head["settings"] = {**head["settings"], "detector": "rfdetr", "conf": 0.5}
        for suffix, conf in (("", CONF), *(REPLAY.items() if seq in VALID else ())):
            frames = {}
            for f, v in raw["frames"].items():
                kept = D._merge([dict(d) for d in v if d["conf"] >= conf], MERGE_THR)     # the product: threshold per tile, then merge
                frames[f] = [{**d, "cls": "person" if d.get("cls") == "player" else d.get("cls")} for d in kept if inside(d["xyxy"])]
            out = os.path.join(STC, f"{seq}.p46{tag}{suffix}.dets.cache.json")
            json.dump({**head, "settings": {**head["settings"], "conf": conf}, "complete": True, "weights_sha256": raw.get("weights_sha256"),
                       "frames": frames}, open(out, "w"))
            print(seq, tag, conf, "frames", len(frames), "boxes/frame %.2f" % (sum(len(v) for v in frames.values()) / max(1, len(frames))), "->", os.path.basename(out))


def recall(tag, seqs):
    import n40_occlusion_recall as R  # noqa: PLC0415
    cnt = {k: [0, 0] for k in CLASSES}
    fp = nf = 0
    found = [0, 0]
    for q in seqs:
        gt = R._gt(q)
        frames = {int(k): [d["xyxy"] for d in v if d.get("cls") in ("person", "player")]
                  for k, v in json.load(open(os.path.join(STC, f"{q}.{tag}.dets.cache.json")))["frames"].items()}
        feet = [g[2] + g[4] for fr in gt.values() for g in fr]
        th = np.percentile(feet, [33.3, 66.7])
        for f, boxes in frames.items():
            if f not in gt:
                continue
            nf += 1
            xyxy = [(x, y, x + w, y + h) for _i, x, y, w, h in gt[f]]
            for g, (_gi, x, y, w, h) in zip(xyxy, gt[f]):
                cover = max((max(0, min(g[2], o[2]) - max(g[0], o[0])) * max(0, min(g[3], o[3]) - max(g[1], o[1])) / (w * h) for o in xyxy if o is not g), default=0.0)
                cls = "occluded" if cover >= R.OCC else "isolated" if cover == 0.0 else None
                if cls is None:
                    continue
                k = f"{cls}_{'far' if y + h < th[0] else 'mid' if y + h < th[1] else 'near'}"
                cnt[k][0] += 1
                cnt[k][1] += any(R._iou(g, tuple(b)) >= 0.5 for b in boxes)
            fp += sum(1 for b in boxes if not any(R._iou(g, tuple(b)) >= 0.5 for g in xyxy))
            # FOUND: one-to-one greedy matching at IoU >= 0.3, so one box cannot find two overlapping players
            pairs = sorted(((R._iou(g, tuple(b)), gi, bi) for gi, g in enumerate(xyxy) for bi, b in enumerate(boxes)), reverse=True)
            used_g, used_b = set(), set()
            for iou, gi, bi in pairs:
                if iou < 0.3:
                    break
                if gi not in used_g and bi not in used_b:
                    used_g.add(gi); used_b.add(bi)
            for gi, (g, (_i, x, y, w, h)) in enumerate(zip(xyxy, gt[f])):
                cover = max((max(0, min(g[2], o[2]) - max(g[0], o[0])) * max(0, min(g[3], o[3]) - max(g[1], o[1])) / (w * h) for o in xyxy if o is not g), default=0.0)
                if cover >= R.OCC and y + h < th[0]:
                    found[0] += 1
                    found[1] += gi in used_g
    return {**{k: round(v[1] / max(1, v[0]), 4) for k, v in cnt.items()}, "non_gt_per_frame": round(fp / max(1, nf), 3),
            "occluded_far_found": round(found[1] / max(1, found[0]), 4)}


def per_id(tag, seqs):
    """Recall per GT id, per clip. The class metrics above pool over players.

    A detector can lose one player completely and move the pooled number by less
    than a percentage point: on 128057 ft0 lost GT id 12 from 0.980 to 0.352 while
    every other id IMPROVED, and the clip's pooled recall still rose +0.005. Pooled
    recall cannot see that, and neither can any of the five class gates, because the
    lost player is one of 22 in a class of thousands of instances.
    """
    import n40_occlusion_recall as R  # noqa: PLC0415
    out = {}
    for q in seqs:
        gt = R._gt(q)
        frames = {int(k): [d["xyxy"] for d in v if d.get("cls") in ("person", "player")]
                  for k, v in json.load(open(os.path.join(STC, f"{q}.{tag}.dets.cache.json")))["frames"].items()}
        cnt = {}
        for f, boxes in frames.items():
            if f not in gt:
                continue
            for gi, x, y, w, h in gt[f]:
                c = cnt.setdefault(gi, [0, 0])
                c[0] += 1
                c[1] += any(R._iou((x, y, x + w, y + h), tuple(b)) >= 0.5 for b in boxes)
        out[q] = {str(k): round(v[1] / max(1, v[0]), 4) for k, v in sorted(cnt.items())}
    return out


#: PER-ID FLOOR. No GT id may lose more than this much recall against stock.
#: PRE-REGISTERED FOR THE NEXT FINE-TUNE, NOT FOR ft0: it was written on
#: 2026-09-22, AFTER ft0's TEST was scored and after the id-12 collapse was found,
#: so it cannot judge ft0 - a rule written to fit a result it has already seen
#: judges nothing. For ft0 the number is REPORTED beside the verdict and the ship
#: decision is the author's. From the next fine-tune on it is a gate.
PER_ID_FLOOR = -0.10


def worst_ids(res, arm, n=5):
    """The n largest per-id recall LOSSES against stock, across the TEST clips."""
    s, f = res.get("per_id", {}).get("stock", {}), res.get("per_id", {}).get(arm, {})
    rows = [(f[q][i] - s[q][i], q, i, s[q][i], f[q][i])
            for q in s if q in f for i in s[q] if i in f[q]]
    return sorted(rows)[:n]


def choose_score(r):
    return (r["occluded_far"] + r["occluded_mid"] + r["isolated_far"]) / 3.0


def median_seed(res):
    """The seed whose VALID choice score is the median of the three fine-tuned arms."""
    seeds = sorted((a for a in res if a.startswith("ft")), key=lambda a: choose_score(res[a]))
    return seeds[1] if len(seeds) == 3 else None


def valid():
    res = {arm: recall(f"p46{arm}", VALID) for arm in ARMS if os.path.exists(os.path.join(STC, f"{VALID[0]}.p46{arm}.dets.cache.json"))}
    for arm, r in res.items():
        print(arm, r, "choice score %.4f" % choose_score(r))
    replay = {f"{arm}{sfx}": recall(f"p46{arm}{sfx}", VALID) for arm in res for sfx in REPLAY
              if os.path.exists(os.path.join(STC, f"{VALID[0]}.p46{arm}{sfx}.dets.cache.json"))}
    for k, r in replay.items():
        print("  secondary", k, r)
    chosen = median_seed(res)
    if chosen:
        shas = {a: json.load(open(os.path.join(STC, f"{VALID[0]}.p46{a}.dets.cache.json"))).get("weights_sha256") for a in res}
        choice = {"chosen_seed": chosen, "by": "median of mean(occluded_far, occluded_mid, isolated_far) recall on VALID 118576",
                  "weights_sha256": shas, "valid": res, "valid_secondary_conf": replay}
        json.dump(choice, open(CHOICE, "w"), indent=1)
        print("chosen (median) seed:", chosen, "-> commit", os.path.relpath(CHOICE, ROOT), "before importing TEST")


def _track(tag, seq):
    import st2_product as ST  # noqa: PLC0415
    out = os.path.join(STC, f"{seq}.{tag}.tracks.json")
    if not os.path.exists(out):
        cal = ST._p(seq, ".p35.calibration.json")
        code = ("import sys; sys.path.insert(0, 'model/engine'); import track; "
                f"track.run({f'{ST.CLIPS}/{seq}.mp4'!r}, {out!r}, calib={cal!r}, detector='rfdetr', stride=2, "
                f"dets_cache={os.path.join(STC, f'{seq}.{tag}.dets.cache.json')!r})")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
        ST.identify(seq, cal, src=f".{tag}", dst=f".{tag}id")


def test():
    import n41_conf as N41  # noqa: PLC0415
    if not _committed(CHOICE) or not _committed(os.path.abspath(__file__)):
        raise SystemExit("VOID: the protocol and model/validation/n46_choice.json must be committed before TEST is scored")
    chosen = json.load(open(CHOICE))["chosen_seed"]
    res = {"chosen": chosen, "recall": {"ship": recall("p35", TEST)}, "product": {}}
    # An arm with no TEST detections is a MISSING SEED, not a crash. The rule is
    # ruled on the chosen seed and publishes the spread across seeds beside it, so a
    # run without every seed still yields the verdict - it just cannot yet print the
    # spread, and must say so rather than dying on a FileNotFoundError forty minutes
    # into tracking (2026-09-20: it did exactly that, on ft1, after scoring stock and
    # the chosen arm).
    have = [a for a in ARMS if all(os.path.exists(os.path.join(STC, f"{q}.p46{a}.dets.cache.json")) for q in TEST)]
    res["arms_missing"] = [a for a in ARMS if a not in have]
    if chosen not in have:
        raise SystemExit(f"VOID: the chosen seed {chosen} has no TEST detections")
    if res["arms_missing"]:
        print("SPREAD PENDING - no TEST detections for:", " ".join(res["arms_missing"]),
              "(the verdict rules on", chosen + "; the seed spread cannot be published until these run)")
    res["per_id"] = {"ship": per_id("p35", TEST)}
    for arm in have:
        res["recall"][arm] = recall(f"p46{arm}", TEST)
        res["per_id"][arm] = per_id(f"p46{arm}", TEST)
        for q in TEST:
            _track(f"p46{arm}", q)
        if arm != "stock":
            N41.ARMS = {"C50": ("p46stock", "p46stockid"), "C30": (f"p46{arm}", f"p46{arm}id")}
            out = os.path.join(STC, f"n46_{arm}_test.json")
            N41.score("held", out=out)
            res["product"][arm] = json.load(open(out))
    json.dump(res, open(os.path.join(STC, "n46_test.json"), "w"), indent=1)
    show()


def verdict(res):
    arm = res["chosen"]
    s, f = res["recall"]["stock"], res["recall"][arm]
    # ISOLATED-FAR, AMENDED 2026-09-20 by the author. As drafted this read
    # `+ 0.08`, and stock scores 0.9294 on TEST and 0.9461 on VALID, so the bar was
    # 1.0094 and 1.0261 - above 1.0, unsatisfiable by any detector that ever existed.
    # It was a drafting error, not a standard. RAISED BEFORE TEST WAS SCORED (the
    # VALID table showed it on 2026-09-20, commit 0af9d01, with TEST unopened) and
    # amended after, so the correction cannot be a response to the result. The class
    # is near-saturated for both arms, so the property worth gating is "does not
    # regress": isolated_far >= stock, which the original text was reaching for and
    # mis-stated. `det_as_drafted` below keeps the original bar and is reported
    # beside the ruling verdict, always, so the amendment is visible in the output
    # rather than buried in this comment.
    det = [f["occluded_far"] >= s["occluded_far"] + 0.15, f["isolated_far"] >= s["isolated_far"],
           all(f[k] >= s[k] - 0.01 for k in ("occluded_mid", "occluded_near", "isolated_mid", "isolated_near")),
           f["non_gt_per_frame"] <= s["non_gt_per_frame"] + 0.5,
           f["occluded_far_found"] >= s["occluded_far_found"] + 0.10]
    det_as_drafted = list(det)
    det_as_drafted[1] = f["isolated_far"] >= s["isolated_far"] + 0.08
    p = res["product"][arm]
    m = lambda a, k: float(np.mean([p[a][q][k] for q in TEST]))  # noqa: E731
    prod = [m("C30", "HOTA") >= m("C50", "HOTA") + 0.015, all(p["C30"][q]["DetPr"] >= p["C50"][q]["DetPr"] - 0.020 for q in TEST),
            m("C30", "IDSW") <= 1.15 * m("C50", "IDSW"), m("C30", "no_owner") <= m("C50", "no_owner") + 0.02]
    return det, prod, all(det) and all(prod), det_as_drafted, all(det_as_drafted) and all(prod)


def show():
    p = os.path.join(STC, "n46_test.json")
    if os.path.exists(CHOICE):
        c = json.load(open(CHOICE))
        print("VALID:", {a: round(choose_score(r), 4) for a, r in c["valid"].items()}, "chosen", c["chosen_seed"])
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    for arm, r in res["recall"].items():
        print(f"TEST recall {arm:5s}", " ".join(f"{k} {v}" for k, v in r.items()))
    for arm, pr in res["product"].items():
        m = lambda a, k: float(np.mean([pr[a][q][k] for q in TEST]))  # noqa: E731
        print(f"TEST product {arm}: " + " ".join(f"{k} {m('C50', k):.4f}->{m('C30', k):.4f}" for k in ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "IDF1", "IDSW", "dist_raw", "hsr_raw", "no_owner")))
        q = "128057"
        print(f"   128057 alone: STOCK {pr['C50'][q]['HOTA']:.4f} -> {arm} {pr['C30'][q]['HOTA']:.4f} (shipped 0.3836, winner's pipeline 0.5583, docs/experiments/n45-winner-boxes.md)")
    if res.get("per_id"):
        for d, q, i, sv, fv in worst_ids(res, res["chosen"]):
            print(f"   per-id {q} id{i:<3} stock {sv:.4f} -> {fv:.4f} ({d:+.4f})"
                  + ("  BELOW THE FLOOR - reported, not ruling, see PER_ID_FLOOR" if d < PER_ID_FLOOR else ""))
    det, prod, ok, det_drafted, ok_drafted = verdict(res)
    print("RULE:", "detection", det, "product", prod, "->", "SUPPORTED (then the CoreML export check)" if ok else "NOT SUPPORTED")
    # The amendment is printed every time, never only when it changes the answer -
    # a correction that shows up solely in the run it rescues is indistinguishable
    # from moving a goalpost. See verdict()'s ISOLATED-FAR note for the dates.
    print("AS DRAFTED (isolated_far >= stock + 0.08, unsatisfiable: needs %.4f):" % (res["recall"]["stock"]["isolated_far"] + 0.08),
          "detection", det_drafted, "->", "SUPPORTED" if ok_drafted else "NOT SUPPORTED")


def show_per_id(arm=None):
    """Per-id recall from the caches on disk: stock vs every seed present.

    EVERY seed, not just the chosen one, because "is this seed noise or the
    training set" is the only question a single arm cannot answer - and on
    2026-09-24 the answer was the training set: id 12 collapses on all three.
    """
    have = [a for a in ARMS if a != "stock"
            and all(os.path.exists(os.path.join(STC, f"{q}.p46{a}.dets.cache.json")) for q in TEST)]
    stock = per_id("p46stock", TEST)
    if len(have) > 1:
        # ONCE per arm. The first cut of this called per_id and recall inside the
        # f-strings and took minutes; a claim the gate has to run is not the place
        # for an accidental O(arms^2).
        ids_ = {a: per_id(f"p46{a}", TEST) for a in have}
        ngt_ = {a: recall(f"p46{a}", TEST)["non_gt_per_frame"] for a in ("stock", *have)}
        # Only ids EVERY arm has. per_id derives its id set from the frames in
        # that arm's cache, so a cache built with --max-frames yields a different
        # set and indexing stock's ids into it raises. worst_ids() already guards
        # this; the first cut of this block did not.
        shared = [(q, i) for q in stock for i in stock[q]
                  if all(i in ids_[a].get(q, {}) for a in have)]
        if not shared:
            raise SystemExit("no id is present in every arm - the caches cover different frames")
        _, wq, wi = sorted(((stock[q][i] - min(ids_[a][q][i] for a in have), q, i)
                            for q, i in shared), reverse=True)[0]
        print(f"SEED SPREAD on the worst id ({wq} id{wi}): stock {stock[wq][wi]:.4f} -> "
              + ", ".join(f"{a} {ids_[a][wq][wi]:.4f}" for a in have))
        print("   all seeds losing the same id is a TRAINING SET answer, not seed noise.")
        # "non-GT" is a box matching NO GT box at IoU >= 0.5. It is NOT "a person
        # the GT lacks": review measured 20.3% of stock's overlapping a GT player
        # at 0 < IoU < 0.5. Print what is measured; the interpretation lives in
        # n53-per-id-collapse (not published) where it can be argued with.
        print("   detector boxes matching no GT box at IoU>=0.5, per frame: "
              + f"stock {ngt_['stock']:.3f} -> " + ", ".join(f"{a} {ngt_[a]:.3f}" for a in have))
    arm = arm or json.load(open(CHOICE))["chosen_seed"]
    # reuse the pass already made above rather than repeating it - the comment on
    # that block says ONCE per arm and the first cut did it twice anyway.
    chosen = ids_[arm] if len(have) > 1 and arm in ids_ else per_id(f"p46{arm}", TEST)
    res = {"per_id": {"stock": stock, arm: chosen}}
    for d, q, i, sv, fv in worst_ids(res, arm, n=8):
        print(f"{q} id{i:<3} stock {sv:.4f}  {arm} {fv:.4f}  delta {d:+.4f}"
              + ("   BELOW THE FLOOR" if d < PER_ID_FLOOR else ""))
    for q in TEST:
        sv = np.mean(list(res["per_id"]["stock"][q].values()))
        fv = np.mean(list(res["per_id"][arm][q].values()))
        print(f"{q} mean over ids: stock {sv:.4f} -> {arm} {fv:.4f} ({fv - sv:+.4f})")
    # THE TABLE IS WRITTEN, because the inputs are not in git. The ft1/ft2
    # detection caches are 47 MB and gitignored; their durable source is release
    # n46-stage2-2 (sha256 in n53-per-id-collapse (not published)), not this machine. So
    # the small derived artefact is tracked and the tags cite it, exactly as the
    # nine stc2025 result files are - a number whose run nobody else can reach is
    # the failure this repo exists to refuse, and 47 MB in git is not the fix.
    out = os.path.join(STC, "n46_per_id.json")
    json.dump({"per_id": {"stock": stock, **({a: ids_[a] for a in ids_} if len(have) > 1 else
                                             {arm: res["per_id"][arm]})},
               "non_gt_per_frame": ngt_ if len(have) > 1 else None,
               "source": "release n46-stage2-2 + n46-stage2-1; caches gitignored (47 MB)"},
              open(out, "w"), indent=1, sort_keys=True)
    print(f"-> {os.path.relpath(out, ROOT)}")
    ids = [d for d, *_ in worst_ids(res, arm, n=10**6)]
    print(f"worst per-id delta {min(ids):+.4f} against a floor of {PER_ID_FLOOR:+.2f}; "
          f"{sum(1 for d in ids if d < PER_ID_FLOOR)} of {len(ids)} ids below it")
    return res


def demo():
    """The rule on made-up numbers: a clear gain passes; a gain bought with precision fails; the choice is the median."""
    base = {k: 0.5 for k in CLASSES} | {"non_gt_per_frame": 2.0, "occluded_far_found": 0.7}
    good = base | {"occluded_far": 0.7, "isolated_far": 0.6, "non_gt_per_frame": 1.0, "occluded_far_found": 0.85}
    prod_ok = {"C50": {q: {"HOTA": 0.36, "DetPr": 0.77, "IDSW": 300, "no_owner": 0.10} for q in TEST},
               "C30": {q: {"HOTA": 0.40, "DetPr": 0.80, "IDSW": 290, "no_owner": 0.09} for q in TEST}}
    assert verdict({"chosen": "ft1", "recall": {"stock": base, "ft1": good}, "product": {"ft1": prod_ok}})[2]
    # The amendment must not smuggle in a pass: a model that REGRESSES isolated_far
    # still fails, under both the amended bar and the drafted one.
    regress = good | {"isolated_far": base["isolated_far"] - 0.01}
    v = verdict({"chosen": "ft1", "recall": {"stock": base, "ft1": regress}, "product": {"ft1": prod_ok}})
    assert not v[2] and not v[4], "a regression on isolated_far must fail both readings"
    # PER-ID: a collapse on one player, with every other player improving, is
    # exactly the shape the class gates cannot see - assert the reader finds it.
    sid = {"128057": {str(i): 0.90 for i in range(1, 23)} | {"12": 0.98}}
    fid = {"128057": {str(i): 0.94 for i in range(1, 23)} | {"12": 0.35}}
    w = worst_ids({"per_id": {"stock": sid, "ft9": fid}}, "ft9")
    assert w[0][2] == "12" and round(w[0][0], 2) == -0.63, w[0]
    assert w[0][0] < PER_ID_FLOOR and w[1][0] > PER_ID_FLOOR, "only the collapsed id is below the floor"
    pooled = sum(fid["128057"].values()) / 22 - sum(sid["128057"].values()) / 22
    assert pooled > 0, "the fixture must be one the pooled mean calls an improvement"

    bad = json.loads(json.dumps(prod_ok))
    bad["C30"]["128057"]["DetPr"] = 0.70
    assert not verdict({"chosen": "ft1", "recall": {"stock": base, "ft1": good}, "product": {"ft1": bad}})[2]
    assert not verdict({"chosen": "ft1", "recall": {"stock": base, "ft1": good | {"non_gt_per_frame": 3.0}}, "product": {"ft1": prod_ok}})[2]
    shape_only = good | {"occluded_far_found": 0.72}                       # +0.20 "shaped" but the same players "found"
    assert not verdict({"chosen": "ft1", "recall": {"stock": base, "ft1": shape_only}, "product": {"ft1": prod_ok}})[2]
    mk = lambda v: {k: v for k in CLASSES} | {"non_gt_per_frame": 1.0}  # noqa: E731
    assert median_seed({"stock": mk(0.9), "ft0": mk(0.2), "ft1": mk(0.5), "ft2": mk(0.3)}) == "ft2"   # stock never chosen
    assert median_seed({"ft0": mk(0.2), "ft1": mk(0.5)}) is None                                        # two seeds: no choice
    print("n46 demo ok: the rule passes a real gain, fails a precision-bought one and a false-positive one; the median seed is chosen")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--import" in a:
        import_raw(a[a.index("--import") + 1])
    elif "--valid" in a:
        valid()
    elif "--test" in a:
        test()
    elif "--show" in a:
        show()
    elif "--per-id" in a:
        show_per_id()
    else:
        demo()
