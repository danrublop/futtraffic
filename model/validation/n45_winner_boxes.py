"""N45: our association on the STC2025 winner's boxes - how much of 0.59 vs 0.37 is the detector? Protocol first.

WHY
    research-stc2025-winners (not published): GTATrack's repository ships its final track files AND its raw detector output for
    the challenge pair (128057, 132831). Scored under our own harness their tracks read HOTA 0.5875 / DetA 0.7631 /
    AssA 0.4528 (every frame; 0.5874 at our stride-2 staging - the protocol is not flattering us). Their detector,
    fine-tuned twice on this rig's training matches with pseudo-labels, at 1280/1920 px full frame with flip/scale
    test-time augmentation, emits 21-22 boxes per frame at recall 0.95-0.98 and 0.2-0.6 non-GT boxes per frame; ours
    at what ships emits ~0.88-0.90 recall and ~2.2 non-GT boxes (the officials). Their tracker is Deep-EIoU with a
    SportsMOT OSNet and an occlusion re-match, plus GTA-Link offline. The one experiment that separates the two
    shares without a GPU: run OUR online tracker and OUR stitch on THEIR boxes and score both arms on the same frames.

ARMS, the challenge pair, GT every frame and every 2nd frame (both printed)
    THEIRS   their answer files as published (already measured: 0.5875).
    OURS-ON-THEIRS  their raw detections (conf >= 0.6 as they left them; officials already absent) written as a
             detection cache in the product's format, replayed through model/engine/track.run at stride 2 with the
             product's C-BIoU tracker, then the product's split + team-gated stitch. 128057 has a zero-click
             calibration (the metric stitch as the product runs it; NOT the scene prior - a replayed cache is treated as already past
             it, model/engine/track.py; measured afterwards, the prior would have removed 101 of 63,043 boxes, 0.16%); 132831 has none (a 3840x1504 canvas
             the zero-click search has not been run on), so it runs with no prior and no stitch - the tracker alone -
             and the page says so beside its number.
    OURS-ON-OURS  the shipped product on 128057 (`.p35`, n41/n42's C50 columns) for the same-clip comparison; on
             132831 the shipped `.c03s` pair-scoring run (our-tracker-scores-0369-hota (internal note), no calibration).
    Their detections are used ONLY as a research measurement: the model behind them is AGPL and its weights are
    unpublished; nothing from this arm can ship.

READING, written first. On 128057 with a calibration:
    HOTA >= 0.50 on their boxes  -> the detector is most of the gap; the fine-tune plan is confirmed as the first move
                                    and the association question shrinks to what their tracker adds on top.
    HOTA <= 0.42                 -> association is the gap even on excellent boxes; the appearance/ReID question has to
                                    be reopened despite appearance-reid-is-dead-on-this-footage (internal note).
    in between                   -> both, in the proportion the DetA and AssA columns show.
    n = 1 configuration per arm; nothing is tuned; the pair is the challenge's, fixed before this was written.

    .venv/bin/python model/validation/n45_winner_boxes.py --caches     # their detections -> <seq>.p45win.dets.cache.json
    .venv/bin/python model/validation/n45_winner_boxes.py --track      # our tracker (+ stitch where calibrated) -> <seq>.p45win.tracks.json
    .venv/bin/python model/validation/n45_winner_boxes.py --score      # TrackEval, every frame and every 2nd, all arms -> n45.json
    .venv/bin/python model/validation/n45_winner_boxes.py --show
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
WIN = os.path.join(STC, "winner")
PAIR = ("128057", "132831")
CALIB = {"128057": os.path.join(STC, "128057.p35.calibration.json"), "132831": None}


def caches():
    import st2_mot as M  # noqa: PLC0415
    for q in PAIR:
        frames = {}
        for line in open(os.path.join(WIN, f"{q}_yolo")):
            p = line.replace(",", " ").split()
            if len(p) < 7:
                continue
            f = int(p[0]) - 1                                       # their files are 1-based; the product's caches are 0-based
            x, y, w, h, c = map(float, p[2:7])
            frames.setdefault(str(f), []).append({"xyxy": [x, y, x + w, y + h], "conf": c, "cls": "person"})
        # the header the product's replay checks: what track.run would stamp for detector coreml, conf 0.5, its gates
        wh = M._wh(q)
        head = {"video": f"{q}.mp4", "start": 0, "calib": CALIB[q], "calib_sha": None,
                "settings": {"prior_gates": ["pitch"] if CALIB[q] else None, "conf": 0.5, "detector": "coreml", "ball": False,
                             "overlap": __import__("detectors").OVERLAP, "tile": 1024}}
        if CALIB[q]:
            import track as T  # noqa: PLC0415
            head["calib_sha"] = T._calib_digest(CALIB[q])
        json.dump({**head, "complete": True, "frames": frames}, open(os.path.join(STC, f"{q}.p45win.dets.cache.json"), "w"))
        print(q, "frames", len(frames), "boxes per frame %.2f" % (sum(len(v) for v in frames.values()) / len(frames)), "size", wh)


def track_all():
    import st2_product as ST  # noqa: PLC0415
    for q in PAIR:
        out = os.path.join(STC, f"{q}.p45win.tracks.json")
        if os.path.exists(out):
            continue
        code = ("import sys; sys.path.insert(0, 'model/engine'); import track; "
                f"track.run({f'{ST.CLIPS}/{q}.mp4'!r}, {out!r}, calib={CALIB[q]!r}, detector='coreml', stride=2, "
                f"dets_cache={os.path.join(STC, f'{q}.p45win.dets.cache.json')!r})")
        import subprocess  # noqa: PLC0415
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def score():
    import st2_mot as M  # noqa: PLC0415
    res = {}
    for every in (1, 2):
        # THEIRS: answer files, staged directly
        M.VARIANT = f".p45theirs_e{every}"
        M.stage_gt(list(PAIR), every=every)
        d = os.path.join(M._root("trackers"), M.TRACKER, "data")
        os.makedirs(d, exist_ok=True)
        for q in PAIR:
            with open(os.path.join(WIN, f"answer_{q}.txt")) as fh, open(os.path.join(d, f"{q}.txt"), "w") as out:
                for line in fh:
                    p = line.replace(",", " ").split()
                    if len(p) >= 6 and (int(p[0]) - 1) % every == 0:
                        out.write(f"{p[0]},{int(float(p[1]))},{p[2]},{p[3]},{p[4]},{p[5]},1,-1,-1,-1\n")
        res[f"THEIRS_e{every}"] = M.run(list(PAIR))
    M.VARIANT = ".p45win"
    M.stage_gt(list(PAIR), every=2)
    M.stage_pred(list(PAIR))
    res["OURS_ON_THEIRS_e2"] = M.run(list(PAIR))
    for q in PAIR:
        res["OURS_ON_THEIRS_e2"][q]["ids"] = len({r["id"] for r in json.load(open(os.path.join(STC, f"{q}.p45win.tracks.json")))["detections"]})
    json.dump(res, open(os.path.join(STC, "n45.json"), "w"), indent=1)
    show()


def show():
    r = json.load(open(os.path.join(STC, "n45.json")))
    ship = json.load(open(os.path.join(STC, "n41_held.json")))["C50"]["128057"]
    print("arm                 clip     HOTA   DetA   DetRe  DetPr  AssA   IDF1   IDSW   ids")
    for arm, per in r.items():
        for q, m in per.items():
            print(f"{arm:19s} {q}  {m['HOTA']:.4f} {m['DetA']:.4f} {m['DetRe']:.4f} {m['DetPr']:.4f} {m['AssA']:.4f} {m['IDF1']:.4f} {m['IDSW']:6.0f} {m.get('ids', '')}")
    print(f"{'OURS_ON_OURS (.p35)':19s} 128057  {ship['HOTA']:.4f} {ship['DetA']:.4f} {ship['DetRe']:.4f} {ship['DetPr']:.4f} {ship['AssA']:.4f} {ship['IDF1']:.4f} {ship['IDSW']:6.0f} {ship['ids']}")
    a, b = r["OURS_ON_THEIRS_e2"]["128057"]["HOTA"], ship["HOTA"]
    t = r["THEIRS_e2"]["128057"]["HOTA"]
    print(f"128057: shipped {b:.4f} -> our association on their boxes {a:.4f} -> their pipeline {t:.4f}: detector share {a - b:+.4f} of {t - b:+.4f} ({100 * (a - b) / (t - b):.0f}%)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--caches" in a:
        caches()
    elif "--track" in a:
        track_all()
    elif "--score" in a:
        score()
    else:
        show()
