"""Does the fine-tuned ball detector raise the CEILING over the stock one? A/B, same frames.

WHY
    `ball_train.py` scored @C 0.9147 :: model/validation/evidence/ball_test.json mAP50 on
    held-out TILES of broadcast footage (n=1 run; its rule forbids re-scoring test, so it can
    never carry a re-runnable tag and the artefact is committed instead; valid sat at 0.824, BELOW the bar
    test passed, on 159 tiles from two matches). That
    is not a claim about our footage or our pipeline. The question a product decision
    rests on is the one `ball_eval.py` already asks: over SoccerTrack's own ball truth,
    how often is the true ball among the detector's candidates at all (`oracle`), and
    how often does the shipped picker then choose it (`shipped`).

    `oracle` is the ceiling: no picker can beat its own detector's candidate list. The
    lost 2026-08-29 model was never taken this far, which is why the tile mAP in the
    memory note has sat unused for three weeks.

RULE, FIXED BEFORE THE RUN (committed with this file)
    population  the six TeamTrack clips (MIT, data/footage/soccertrack/README.md; this
                repo has long mislabelled them SoccerTrack v1) whose gates the picker
                was NOT fitted on:
                soccer_top test 1710-1740, 1740-1770, 1770-1800 and all three
                soccer_side. EXCLUDED and why: both soccer_top train clips (the gates
                in ball.py were fitted on them, ball_eval.py:112) and soccer_top test
                1680-1710 (a dead ball carried forward on every frame, degenerate).
    frames      the FIRST 300 of each clip, sequentially, identical for both arms
                (build_cache reads sequentially; seeking poisons a 20 px gate). The
                share of each clip's annotated frames that leaves out is reported
                with every number - a rate over a subset is not a rate over the clip.
    arm A       `coreml`, the shipped detector (rfdetr-medium-576, tile 1024), ball
                class from COCO. This is the only ball the product could have today.
    arm B       `rfdetr` nano at tile 640 with ball_train.py's valid-selected
                checkpoint. Tile 640 because that is what it was trained at; behind
                the default 1024 the ball shrinks by a third and the measurement is
                of the tiling, not the model. variant="nano" for the same reason:
                a nano checkpoint loaded as medium returns nothing.
    held fixed  conf 0.05 both arms (ball_eval.CONF - deliberately under the shipped
                0.5, to separate "never proposed" from "chosen wrongly"), 20 px gate
                at 4K, ball.BallPicker unmodified, each clip's own calibration.
    metric      pooled over the six clips, as ball_eval pools: frames where the
                picker's answer is within the gate, over frames with a GT ball.
    CONFOUND, found in review 2026-09-19, NOT fixed by this rule: A runs tile 1024 and B tile
                640, so the arms differ in the weights AND in how large the ball is in the model's
                input. On `top test 1710-1740`, stock weights at tile 640 score oracle 132/300
                where the same weights at 1024 score 45/300 - tiling alone is worth more than
                this rule's whole bar. Arm C is the control that separates them; until it runs,
                any delta here is "fine-tuned@640 vs shipped@1024" - the product question, but
                NOT a claim about the weights.
    SECOND CONFOUND, named in review 2026-09-20 and NOT removable: arm C is `coreml`
                rfdetr-MEDIUM-576 and arm B is `rfdetr` NANO, so B - C changes the weights,
                the architecture AND the inference backend together. It cannot be separated
                here - a nano checkpoint loaded as medium returns nothing, and there is no
                fine-tuned medium - so B - C is "a ball-specific fine-tune at the same tile
                size", never "the weights' own delta". Say it that way wherever it is quoted.
    PASS        >= +0.05 on pooled `oracle`, and from 2026-09-19 the delta judged is B - C,
                the fine-tune's own delta at a FIXED tile size, whenever arm C has been
                built; B - A (which mixes it with the tiling) is still printed, never ruled
                on. Anything less and the fine-tune has not moved the ceiling on OUR footage,
                whatever it scores on broadcast tiles.
    also report candidates per frame, per arm. The A -> B rise was read as the fine-tune
                emitting more decoys; arm C is the control on that too.
    also report `shipped`, `any` and the `centre` control per arm, and the frames with
                no candidate at all. `centre` over 1% voids the run (ball_eval header).
    NOT decided here: whether jobs.py should run a ball at all. That needs a second
                detector pass per frame, and its cost is not measured yet.

    .venv/bin/python model/validation/ball_cmp.py --build   # both arms, ~6 clips x 300 frames
    .venv/bin/python model/validation/ball_cmp.py --show    # the comparison off the caches
    .venv/bin/python model/validation/ball_cmp.py --demo    # self-check
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ball_eval as B  # noqa: E402

CLIPS = [
    "soccer_top__test__videos__D_20220220_1_1710_1740",
    "soccer_top__test__videos__D_20220220_1_1740_1770",
    "soccer_top__test__videos__D_20220220_1_1770_1800",
    "soccer_side__train__videos__F_20200220_1_0000_0030",
    "soccer_side__train__videos__F_20200220_1_0030_0060",
    "soccer_side__train__videos__F_20200220_1_0060_0090",
]
FRAMES = 300
MARGIN = 0.05
ARMS = {"A": dict(suffix=".cmpA", detector="coreml", weights=None, tile=None, variant=None),
        # C: the control the review demanded - A's weights at B's tile size.
        "C": dict(suffix=".cmpC", detector="coreml", weights=None, tile=640, variant=None),
        "B": dict(suffix=".cmpB", detector="rfdetr", variant="nano", tile=640,
                  weights=os.path.join(B.ROOT, "model", "validation", "out", "ball",
                                       "run", "checkpoint_best_ema.pth")),
        # D: N52's clean arm. Same INFERENCE config as B - detector, nano variant,
        # 640 tile - and the same 916 training tiles. That is all demo() can assert,
        # and it is NOT enough to call D - B a comparison of data sources: D1 trained
        # on the same two matches these clips come from and B never saw them, D1 had
        # 187 ball-free tiles to B's 0, and the two chose checkpoints on different
        # valid splits. show() prints those caveats beside the number. D - C is what
        # N52 §7 pre-registered (bar +0.05 on pooled oracle), because C is the stock
        # control at the same tile size.
        "D": dict(suffix=".cmpD", detector="rfdetr", variant="nano", tile=640,
                  weights=os.path.join(B.ROOT, "model", "validation", "out", "ball_clean",
                                       "run_d1", "checkpoint_best_ema.pth")),
        # E1: N64's arm. Same INFERENCE config as D - detector, nano, 640 - and D1's
        # 916 tiles PLUS 916 mined hard negatives. It was TRAINED on CUDA under
        # emulated bf16 while D1 trained on MPS, so E1 - D differs in device and
        # precision as well as tiles; docs/n64-*.md reports that rather than hiding it,
        # and the D1' control is pre-registered in n64_ball_negatives.py.
        "E1": dict(suffix=".cmpE1", detector="rfdetr", variant="nano", tile=640,
                   weights=os.path.join(B.ROOT, "model", "validation", "out", "ball_clean",
                                        "run_e1", "checkpoint_best_ema.pth")),
        # N67's pair, both trained on the PC under cuda/fp16/seed 0 and both scored at the
        # 640 they were TRAINED at - n64.pair()'s VOID (f) refuses any cache built at the
        # RFDETRNano default of 384, which is what cost E1 its attribution.
        "D1P": dict(suffix=".cmpD1P", detector="rfdetr", variant="nano", tile=640,
                    resolution=640,
                    weights=os.path.join(B.ROOT, "model", "validation", "out", "ball_clean",
                                         "run_d1p", "checkpoint_best_ema.pth")),
        "E2": dict(suffix=".cmpE2", detector="rfdetr", variant="nano", tile=640,
                   resolution=640,
                   weights=os.path.join(B.ROOT, "model", "validation", "out", "ball_clean",
                                        "run_e2", "checkpoint_best_ema.pth"))}
OUT = os.path.join(B.ROOT, "model", "validation", "out", "ball", "cmp.json")


def build():
    for arm, kw in ARMS.items():
        for clip in CLIPS:
            if os.path.exists(B.cache_path(clip + kw["suffix"])):
                print(f"=== arm {arm}  {clip}: cache exists, kept "
                      "(delete it to rebuild - these are N48's committed evidence)")
                continue
            print(f"\n=== arm {arm}  {clip}", flush=True)
            # `resolution` IS PASSED. ARMS["D1P"]/["E2"] carry resolution=640 and build()
            # did not forward it, so `ball_cmp --build` would have produced 384 caches for
            # arms whose whole point is the trained geometry - and n64.pair()'s VOID (f)
            # depends on it. The on-disk D1P/E2 caches are correct only because they were
            # built through ball_eval directly. A caller's default silently beating the
            # callee's is a-callers-default-silently-beats-the-callees (internal note).
            B.build_cache(clip, n_frames=FRAMES, conf=B.CONF, detector=kw["detector"],
                          weights=kw["weights"], tile=kw["tile"], variant=kw["variant"],
                          suffix=kw["suffix"], resolution=kw.get("resolution"))


def show(write: bool = False):
    res = {}
    LABEL = {"A": "stock COCO @1024 (what ships)", "C": "stock COCO @640 (tiling control)",
             "B": "fine-tuned nano @640 (DFL data - research only, B4)",
             "D": "fine-tuned nano @640 (CC BY 4.0 SoccerTrack data, N52 D1)",
             "E1": "D1 + 916 mined hard negatives (N64; trained bf16/CUDA, D1 was MPS)",
             "D1P": "D1's 916 tiles retrained on the PC (N67 control: cuda/fp16/seed 0)",
             "E2": "D1P + 53 TRUE negatives at MINE_CONF 0.30 (N64 E2 v2)"}
    for arm, kw in ARMS.items():
        if not all(os.path.exists(B.cache_path(c + kw["suffix"])) for c in CLIPS):
            print(f"\n################ ARM {arm}: no cache, skipped (run --build)")
            continue
        print(f"\n################ ARM {arm}: {kw['detector']} {LABEL[arm]}")
        rows = [B.report(c, B.GATE_PX, kw["suffix"]) for c in CLIPS]
        pooled, gt, n = {}, 0, 0
        # "conf" IS POOLED NOW, and its absence was load-bearing. It is
        # argmax(conf) with the scenery, on-pitch and size gates REMOVED, and
        # ball_eval printed it per clip while this table - the one every picker
        # verdict was read from - left it out. Pooled on arm D it beats the
        # shipped picker, so the largest measured picker lever is deleting gates,
        # and five experiments were read from a table that could not show it.
        for m in ("shipped", "conf", "oracle", "any", "centre"):
            right = sum(r["scores"][m]["correct"] for r in rows)
            pooled[m] = {"correct": right,
                         "rate": round(right / sum(r["scores"][m]["gt_present"] for r in rows), 4)}
        gt = sum(r["scores"]["oracle"]["gt_present"] for r in rows)
        n = sum(r["n"] for r in rows)
        res[arm] = {"pooled": pooled, "gt_present": gt, "frames_scored": n,
                    "no_cand": sum(r["no_cand"] for r in rows),
                    # candidates per frame. N51 read the A->B rise as the
                    # fine-tune's doing; arm C says most of it is the tile size,
                    # so the number has to be printed per ARM or it misleads.
                    "cand_per_frame": round(sum(r["n_cand"] for r in rows) / n, 1)}
    # "identical frames in both arms" is the protocol's claim; make it the code's.
    # build_cache flushes every 100 frames, so a killed build leaves a shorter cache
    # that still loads, and pooling rates across arms of different length is silent.
    arms = [a for a in res if isinstance(res[a], dict) and "pooled" in res[a]]
    assert len({res[a]["frames_scored"] for a in arms}) == 1 and \
        len({res[a]["gt_present"] for a in arms}) == 1, \
        {a: (res[a]["frames_scored"], res[a]["gt_present"]) for a in arms}
    d = round(res["B"]["pooled"]["oracle"]["rate"] - res["A"]["pooled"]["oracle"]["rate"], 4)
    res["delta_oracle"] = d
    res["void_centre"] = max(v["pooled"]["centre"]["rate"] for v in res.values()
                             if isinstance(v, dict) and "pooled" in v) > 0.01
    if "C" in res:
        res["delta_tiling_only"] = round(
            res["C"]["pooled"]["oracle"]["rate"] - res["A"]["pooled"]["oracle"]["rate"], 4)
        res["delta_oracle_vs_control"] = round(
            res["B"]["pooled"]["oracle"]["rate"] - res["C"]["pooled"]["oracle"]["rate"], 4)
        # B - C: the fine-tune at a FIXED tile size. NOT "the weights' own delta" -
        # B is nano/rfdetr and C is medium-576/coreml, an unremovable confound.
        d = res["delta_oracle_vs_control"]
    # N52's arm. Its bar was committed in bb38497, before D1 existed, and it reads
    # D - C: the clean fine-tune against the stock control at the SAME tile size.
    # D - B is the one comparison in this file where only the training DATA differs,
    # since D and B share detector, variant and tile (asserted in demo()).
    if "D" in res:
        res["n52_delta_vs_C"] = round(
            res["D"]["pooled"]["oracle"]["rate"] - res["C"]["pooled"]["oracle"]["rate"], 4)
        res["n52_verdict"] = ("SUPPORTED" if res["n52_delta_vs_C"] >= MARGIN
                              else "NOT SUPPORTED")
        res["delta_clean_vs_dfl"] = round(
            res["D"]["pooled"]["oracle"]["rate"] - res["B"]["pooled"]["oracle"]["rate"], 4)
        res["delta_clean_vs_dfl_shipped"] = round(
            res["D"]["pooled"]["shipped"]["rate"] - res["B"]["pooled"]["shipped"]["rate"], 4)

    # VOID is a refusal, not a field nobody reads.
    res["ruled_on"] = ("B - C (fine-tune at fixed tile size)" if "C" in res
                       else "B - A (fine-tune AND tiling)")
    res["verdict"] = ("VOID: the centre control scores" if res["void_centre"]
                      else ("SUPPORTED" if d >= MARGIN else "NOT SUPPORTED")
                      + ("" if "C" in res else " (PROVISIONAL: arm C, the tiling control, "
                         "has not been built - this delta is not a claim about the weights)"))
    # --show PRINTS; only --write saves cmp.json. A write here made every verify_claims run
    # dirty a committed evidence file, and verify_claims' own self-check then failed in the
    # push gate (2026-09-28).
    if write:
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n{'':10s} {'oracle':>8s} {'shipped':>8s} {'conf':>8s} {'any':>8s} {'centre':>8s}"
          f" {'no_cand':>8s} {'gt':>7s} {'cand/frm':>9s}")
    # EVERY ARM IN res, not a hardcoded four. E1 was computed, included in the cross-arm
    # frames assert and in void_centre, written to cmp.json - and never printed in the
    # table these verdicts are read from. That invisibility is the same defect the `conf`
    # column comment above was written about: an arm the table cannot show is an arm
    # nobody compares.
    for arm in [a for a in ARMS if a in res]:
        p = res[arm]["pooled"]
        print(f"arm {arm:6s} " + " ".join(f"{p[m]['rate']:8.4f}" for m in
                                          ("oracle", "shipped", "conf", "any", "centre"))
              + f" {res[arm]['no_cand']:8d} {res[arm]['gt_present']:7d}"
              + f" {res[arm]['cand_per_frame']:9.1f}")
    if "C" in res:
        print(f"delta tiling only (C-A) {res['delta_tiling_only']:+.4f}   "
              f"delta fine-tune @fixed tile (B-C) {res['delta_oracle_vs_control']:+.4f}   "
              f"delta mixed (B-A) {res['delta_oracle']:+.4f}")
    if "D" in res:
        print(f"N52 clean arm: delta vs control (D-C) {res['n52_delta_vs_C']:+.4f} "
              f"(bar {MARGIN:+.2f}) -> {res['n52_verdict']}")
        print(f"   clean vs DFL: oracle (D-B) {res['delta_clean_vs_dfl']:+.4f}, "
              f"shipped picker {res['delta_clean_vs_dfl_shipped']:+.4f}")
        print("   NOT 'clean data beats DFL data': D1 trained on other minutes of "
              "the SAME two matches these clips come from, B has never seen either "
              "venue. Within-venue vs cross-venue, and D1 also saw 187 ball-free "
              "tiles to B's 0 and chose its checkpoint on a different valid split.")
    print(f"\ndelta ruled on [{res['ruled_on']}] {d:+.4f} (bar {MARGIN:+.2f}) -> {res['verdict']}"
          )
    print(f"frames {res['A']['frames_scored']} of {FRAMES} x {len(CLIPS)} requested; "
          f"first {FRAMES} frames of each clip only")
    return res


def demo():
    assert not (set(CLIPS) & {c for c in B.ALL_CLIPS if "train__videos__D" in c
                              or "1680_1710" in c}), "fitted and degenerate clips excluded"
    assert len(CLIPS) == 6 and set(CLIPS) <= set(B.ALL_CLIPS), CLIPS
    assert ARMS["B"]["tile"] == 640 and ARMS["B"]["variant"] == "nano", \
        "a nano checkpoint at another tile size measures the tiling, not the model"
    assert len({a["suffix"] for a in ARMS.values()}) == len(ARMS), "the arms must not share a cache"
    # D must differ from B in WEIGHTS ONLY, or "clean data vs DFL data" is measuring
    # something else - the N48 tile-size confound in its third costume.
    assert {k: v for k, v in ARMS["D"].items() if k not in ("weights", "suffix")} == \
           {k: v for k, v in ARMS["B"].items() if k not in ("weights", "suffix")}, \
        "arm D differs from B in more than the training data"
    assert ARMS["C"]["weights"] is None and ARMS["C"]["tile"] == ARMS["B"]["tile"], \
        "arm C is A's weights at B's tile size, or it controls nothing"
    print("ball_cmp demo: population, arms and cache separation ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true", help="with --show: also save cmp.json")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.show:
        show(write=a.write)
    else:
        demo()
