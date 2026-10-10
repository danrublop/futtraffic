"""N64: fewer decoys, by training D1 on the tiles it gets WRONG.

    .venv/bin/python model/validation/n64_ball_negatives.py            # self-check
    .venv/bin/python model/validation/n64_ball_negatives.py --mine     # find the hard negatives (local, no GPU)
    .venv/bin/python model/validation/n64_ball_negatives.py --build    # the E1 dataset dir, for the PC
    .venv/bin/python model/validation/n64_ball_negatives.py --score    # the arms (refuses until weights exist)

WHY THIS IS THE ONLY BALL LEVER LEFT
    SEVEN pre-registered picker changes have lost: N49 arm B, N50's re-fitted size
    gate, N51's joint arm, N49 sec5b on arm D, N59's gate subsets under
    leave-one-clip-out, and N58's player prior twice over - with detected players AND
    with ground-truth ones. N58's post-hoc diagnostic says why, and it bounds every
    future picker idea of that shape: over 3,313 true-ball and 151,373 decoy
    candidates, "closer to a player" separates them at AUC 0.5466, against log conf's
    0.8893 on the same gated population. That bounds THAT FEATURE - one scalar, distance
    to the nearest player - and not every possible per-candidate signal; an earlier
    version of this line said "there is no unused per-candidate signal on this footage",
    which is wider than N58 measured.

    N55 named the two exits from a data-association problem in target-like clutter:
    a better per-candidate likelihood, or FEWER CANDIDATES. The first is now measured
    shut. This file takes the second, and it is a detector change, not a picker one:
    arm D1 emits about 86 candidates a frame, and the picker's job is to find one ball
    among them. Cutting the decoy count is the only way left to make that easier.

WHAT THE ARM IS
    D1 trained on 916 clip-stratified tiles of the 16,509 in the clean corpus
    (`ball_clean.matched`). So about 15,593 tiles it has NEVER SEEN are on disk as
    640 px images, already cut, already annotated - and the ones where D1 fires at a
    box that matches no annotation are hard negatives in the only sense that matters:
    this model, on this footage, calls them a ball.

    E1 = D1's 916 tiles + N_NEG tiles mined that way, with their OWN true annotations
    (usually none). Same architecture, same resolution, same hyperparameters, same
    epoch count as D1 - `ball_train.train`, unchanged - so E1 minus D1 is the added
    tiles and nothing else. Mining is LOCAL; only the training needs the GPU.

DECLARED CONSTANTS, fixed here before anything is mined
    N_NEG = 916        one hard negative per positive tile. Chosen as 1:1 rather than
                       tuned, because a ratio swept against the scored clips would
                       make the bar meaningless and a ratio swept against valid would
                       still need its own held-out story. If E1 passes, the ratio is
                       the first thing a follow-up should sweep - with its own n=.
    MINE_CONF = 0.30   a candidate counts as a false positive above this. NOT the
                       shipped 0.5: the point is to catch what the model believes
                       moderately, since those are what survive the picker's argmax
                       once the true ball is absent.
    MINE_IOU = 0.10    a fired box "matches" an annotation above this IoU, so a
                       merely-near miss still counts as a false positive.

THE FIT/EVAL BOUNDARY, which is the part that can silently invalidate this
    Mining reads ONLY tiles whose clip has `ball_clean.role() == "train"`. That
    excludes the six scored clips, the two clips ball.py's gates were fitted on, the
    degenerate one, and the clip that abuts a scored window - `role()` already encodes
    all four and this file re-asserts the intersection rather than trusting it. The
    verdict is read on `ball_cmp.CLIPS` through `ball_eval`, untouched.

RULE, FIXED BEFORE THE SCORED RUN (committed with this file, nothing trained yet)
    population  ball_cmp.CLIPS, 1800 gt-present frames, gate 20 px, N57's dead and
                in-play split, scored through ball_eval exactly as D1 was.
    arms        D1     the shipped-candidate baseline: pooled shipped 0.4706,
                       oracle 0.8822, about 86 candidates a frame.
                E1     D1's tiles plus the mined negatives, nothing else changed.
    PASS        pooled `shipped` >= 0.4706 + 0.05 = 0.5206  - the bar N49 sec5b used on
                this metric - AND `oracle` no lower than 0.8822 - 0.01, AND neither
                population's `shipped` more than 0.01 below D1's. All four, or it does
                not ship.
    WHY THE ORACLE FLOOR IS THE LOAD-BEARING HALF
                Training on negatives is a filter, and a filter that improves a metric
                by deleting hard inputs is indistinguishable from one that solves them
                (docs/README.md#working-rules). Here the deleted input would be the BALL: teach the model
                hard enough that white round things are not balls and the decoy count
                falls, the picker's job gets easier, and `shipped` can rise while the
                true ball stops being proposed at all. `oracle` is the share of frames
                where SOME candidate is the ball, so it is exactly the recall this
                trade would spend. A pass with oracle down is not a pass.
    REPORT      candidates per frame for both arms - the MECHANISM, not the verdict -
                and `wrong` and `silent` per arm, because a picker that answers less
                often buys correctness with silence.
    n           n=1 configuration. One mining rule, one ratio, one threshold, one
                epoch count. Any sweep over these moves n and needs held-out
                selection, which 6 clips cannot support well (N59 measured that:
                8 subsets in-sample +0.0627, held-out -0.0256).
ADDENDUM, 2026-09-26: WRITTEN AFTER E1 TRAINED AND BEFORE IT WAS SCORED
    The PC reported that E1 trained on CUDA under EMULATED bf16, because `amp_dtype` is
    not in `ball_train.HP` and rfdetr's default "auto" resolves per device. D1's own
    out/ball_clean/run_d1/training_config.json records `device: "mps"` and `amp_dtype:
    "auto"`. So E1 minus D1 differs in DEVICE BACKEND and PRECISION as well as in the
    added tiles. That is the N48 confound in a new place, and it was found by the
    machine that ran the arm - not by the rule above, which pinned nine hyperparameters
    into one dict precisely so a kit could not drift, and then left the tenth to
    resolve itself differently on every card.

    WHAT STILL STANDS. The PASS rule is an ABSOLUTE threshold on E1's own pooled score,
    >= 0.5206, not a paired difference. It still decides E1, is read unchanged, and no
    number in it moves.

    WHAT DOES NOT. The ATTRIBUTION, in both directions. A pass could be the precision,
    and a failure could equally be the precision, since emulated bf16 carries less
    mantissa than the path MPS would have taken. So E1's verdict cannot be reported as
    a finding about hard negatives, whichever way it lands.

    THE CONTROL, pre-registered here before E1's score exists
        D1' = D1's own 916 tiles (out/ball_clean/tiles_d1, the same directory, not
        re-sampled), retrained on the SAME PC, same card, same library versions, with
        `amp_dtype` PINNED to whatever E1 resolved to instead of left as "auto", so
        E1 minus D1' is the added tiles and nothing else.
        PAIRED PASS   E1 pooled `shipped` >= D1' pooled `shipped` + 0.05 AND E1 `oracle`
                      >= D1' `oracle` - 0.01, on the same six clips through the same
                      harness. The same shape as the committed rule, against a
                      same-device baseline instead of the MPS one.
        The two verdicts are reported SEPARATELY and neither replaces the other: the
        committed rule says whether E1 is good enough, the paired one says whether the
        tiles are why. A claim about hard negatives needs both.
        n stays 1 per verdict. Nothing is selected here - D1' is a baseline, not a
        candidate - and if D1' and D1 disagree, that difference is a measurement of the
        backend and is reported as one rather than averaged away.

E2 AND D1', REGISTERED 2026-09-26. VERSION 2, AND VERSION 1 IS WITHDRAWN UNRUN.
    V1 (commit 121c512) said E2 = 916 positives + 916 mined negatives at MINE_CONF 0.30,
    and gated the mechanism on candidates per frame falling 10%. A review found both
    halves wrong before either arm trained:

    (i) MINE_CONF NEVER APPLIED. `detectors._keep` routes by class NAME; D1's only class
        is "ball", so every box took the ball branch and was compared against the module
        default BALL_CONF = 0.05 instead of the conf= that was passed. The first mining
        ran at 0.05 while `n64_mined.json` recorded 0.3. At the DECLARED 0.30 the pool of
        truly-negative firing tiles is 53, not the 1,762 the page published - so V1's 1:1
        arm was reachable only because the code ignored its own constant.
    (ii) THE MECHANISM GATE MEASURED THE WRONG THING. SNIPER (NeurIPS 2018) reports
        proposal recall IDENTICAL with and without mined negative chips (AR 65.4 / AR50
        93.2 both ways) while detection AP moves +2.7: recall does not account for false
        positives, so a correct negative-mining arm can leave candidate COUNT flat. A
        gate demanding a 10% count drop can fail a working intervention and pass a
        threshold artefact.

    V1's ratio was also 50% background images against Ultralytics' documented 0-10% and
    COCO's ~1%, i.e. 5-50x outside practice - and enforcing the declared threshold moves
    it INSIDE that range on its own. The bug and the literature point the same way.

    D1'   D1's own 916 tiles (out/ball_clean/tiles_d1, rebuilt from matched() and DIFFED
          against it), retrained on the PC with amp_dtype and seed pinned.
    E2    those 916 plus EVERY truly-negative tile clearing MINE_CONF, taken top-first by
          the confidence D1 gave the false positive. Not a ratio: the threshold decides
          the count, and the count is reported with the result. Same card, same pinned
          precision and seed as D1'.

    PASS, all three, and the first is the one V1 got wrong
      MECHANISM   FALSE CANDIDATES PER FRAME AT MATCHED RECALL, <= 0.90 x D1'. Each arm's
                  conf threshold is chosen so both reach the SAME oracle recall (the
                  lower arm's ceiling); at that operating point, count the candidates
                  that are NOT the ball. This is the quantity the picker actually fights
                  and it cannot be moved by emitting more boxes, which is exactly how E1
                  passed V1's column while its raw count rose 22.7%.
      OUTCOME     E2 pooled `shipped` >= D1' + 0.05, AND the paired cluster-bootstrap
                  95% interval excludes zero. The interval is part of the bar because
                  E1 cleared an absolute threshold by 0.0016 - 2.88 frames of 1800 - and
                  V1 had no interval to refuse it with. Bootstrap resamples CLIPS, not
                  frames: frames within a clip are correlated, and N59 measured the
                  clip-level swing at 0.09 on this very corpus.
      GUARD       E2 `oracle` >= D1' `oracle` - 0.01, so decoys are not bought with the
                  ball.
    SESOI         +0.05 pooled, declared here. With 6 clips a clip-level sign-flip
                  permutation has 2^6 = 64 assignments, so the smallest attainable
                  two-sided p is 2/64 = 0.031: one shot, no multiplicity budget. A null
                  from this design reads "no evidence", never "evidence of no effect".
    REPORT        D1' against D1 as its own line - the MPS-to-CUDA backend and precision
                  change measured alone - plus leave-one-clip-out on the paired delta,
                  and the raw candidate count beside the matched-recall one so the two
                  can be seen to disagree.
    n             n=1 per verdict, one seed per arm. The seed is pinned but a single draw
                  is not a noise floor; n46_finetune runs three for the player fine-tune
                  and this budget does not.
    E2 VOID       Restored 2026-09-27: the V2 rewrite above dropped this list, so the rule
                  had PASS clauses and no refusals for two days while the code carried the
                  guards anyway. Each is implemented in `negatives()` or `pair()`.
                  (a) ANY E2 tile carries an annotation - asserted PER TILE, not counted in
                      aggregate, which is exactly how V1's guard failed;
                  (b) `centre` scores over 1%;
                  (c) D1' and E2's training_config.json disagree on device, amp_dtype or
                      seed, since then the pair is confounded as E1 was;
                  (d) either checkpoint's sha256 is missing from the report;
                  (e) no truly-negative tile clears MINE_CONF;
                  (f) either arm's candidate cache was not built at the resolution the
                      checkpoint was TRAINED at. N65 measured the default's cost - pooled
                      shipped 0.4706 against 0.5850 and 86.2 candidates a frame against
                      71.7 - and E1's pass did not survive being re-measured. A cache
                      predating the `resolution` field records None, which means 384.

    VOID        (a) any mined tile's clip is not role()=="train";
                (b) `centre` scores over 1%;
                (c) D1 does not reproduce pooled shipped 0.4706 and oracle 0.8822
                    under this harness, so the comparison has no baseline;
                (d) the E1 weights' sha256 is not recorded beside the number
                    (reproduced-means-diffed-against-the-golden (internal note));
                (e) fewer than N_NEG/2 hard negatives are found, which would make E1
                    a differently-sized D1 rather than the arm described here.
"""
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

#: THE __main__ GUARD IS LOAD-BEARING, NOT STYLE. `num_workers: 4` makes Lightning
#: spawn workers, and on Windows spawn re-imports __main__, so a module-level
#: `.train()` re-enters training recursively and the script cannot run at all
#: (rfdetr #428, unfixed upstream). The first E1 kit shipped without it; the PC added
#: it to get the run to start, and the fix belongs here so the next kit carries it.
_SCRIPT = (
    '"' * 3 + """Train on the PC. Every setting comes from n64_spec.json, which the Mac
wrote from ball_train.HP - do not retype, raise or disable any of them.

    python train_e1.py
""" + '"' * 3 + """
import json
import os

from rfdetr import RFDETRNano

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    s = json.load(open(os.path.join(HERE, "n64_spec.json")))
    kw = dict(s["hp"])
    if s.get("amp_dtype"):
        kw["amp_dtype"] = s["amp_dtype"]   # PINNED, never resolved per device
    RFDETRNano(resolution=s["resolution"]).train(
        dataset_dir=HERE, output_dir=s["output_dir"], epochs=s["epochs"], **kw)
    print("REPORT BACK the device and amp_dtype in " + s["output_dir"]
          + "/training_config.json - auto resolves differently per card and that "
          "difference confounds the arm.")


if __name__ == "__main__":
    main()
""")

N_NEG = 916
MINE_CONF = 0.30
MINE_IOU = 0.10
MINED = os.path.join(ROOT, "model", "validation", "out", "ball_clean", "n64_mined.json")


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _labelled():
    """{tile file name: True if it carries at least one ball box}.

    One place, because mine(), negatives() and audit() each re-derived it and the whole
    N64 failure was three call sites disagreeing about what "negative" means.
    """
    import ball_clean as BC  # noqa: PLC0415
    d = json.load(open(os.path.join(str(BC.OUT), "tiles", "train",
                                    "_annotations.coco.json")))
    wb = {a["image_id"] for a in d["annotations"]}
    return {im["file_name"]: (im["id"] in wb) for im in d["images"]}


def unseen():
    """Tiles in the clean corpus that D1 never trained on, with their annotations.

    `matched(916, seed=0)` is deterministic, so D1's own training set is recoverable
    exactly rather than remembered - the difference between a held-out claim and a
    hopeful one.
    """
    import ball_clean as BC  # noqa: PLC0415
    src = os.path.join(str(BC.OUT), "tiles", "train")
    d = json.load(open(os.path.join(src, "_annotations.coco.json")))
    # D1's SIZE, not N_NEG. They are both 916 by coincidence, and the docstring names
    # the 1:1 ratio as the first thing a follow-up should sweep - at which point
    # N_NEG=458 would have silently re-admitted 458 of D1's own training tiles to the
    # "never seen" pool and halved D1's data inside the arm.
    used = {im["file_name"] for im in BC.matched(BC.ARMS["d1"]["tiles"], seed=0)["images"]}
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    return src, [(im, by_img.get(im["id"], [])) for im in d["images"]
                 if im["file_name"] not in used], d


def mine(limit=None):
    """Tiles where D1 fires on something that is not an annotated ball."""
    import cv2  # noqa: PLC0415

    import ball_clean as BC  # noqa: PLC0415
    import ball_cmp as C  # noqa: PLC0415
    import detectors as D  # noqa: PLC0415
    src, pool, _d = unseen()
    if limit:
        pool = pool[:limit]
    # VOID (a), asserted before a single tile is read rather than reported after.
    # role() encodes FOUR exclusions - scored, gate-fitted, degenerate, and abuts-a-
    # scored-window - and the first version of this check tested only the first, by
    # substring. A check narrower than its own name is the defect this file keeps
    # finding; ask role() itself, per tile.
    bad = [im["file_name"] for im, _ in pool
           if BC.role(*BC._parts(im["clip"])) != "train"]
    if bad:
        raise SystemExit(f"VOID (a): {len(bad)} pool tiles are not role()=='train', "
                         f"e.g. {bad[:2]}")
    del C  # only imported for the old substring check
    w = os.path.join(str(BC.OUT), "run_d1", "checkpoint_best_ema.pth")
    # BOTH thresholds, AND the predicate below checks it again. `conf=` alone was dead:
    # detectors._keep routes by CLASS NAME, D1's only class is "ball", so every box took
    # the ball branch and was compared against the module default BALL_CONF = 0.05. The
    # first mining ran at 0.05 while recording 0.3, which is the same defect as a VOID
    # measuring a proxy - a declared constant guarding a quantity it never reached.
    # lint_truth's DEFAULT TRAP has no rule for "passed to a callee that thresholds it
    # elsewhere", so the threshold is enforced here, where it is read.
    det = D.RFDETRDetector(conf=MINE_CONF, ball_conf=MINE_CONF, variant="nano",
                           weights=w, ball=True)
    assert det._det_conf >= MINE_CONF, (det._det_conf, MINE_CONF)
    hits = []
    for i, (im, anns) in enumerate(pool):
        img = cv2.imread(os.path.join(src, im["file_name"]))
        if img is None:
            continue
        gt = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
               a["bbox"][1] + a["bbox"][3]) for a in anns]
        worst = 0.0
        for b in det.detect(img):
            if float(b["conf"]) < MINE_CONF:      # the predicate owns its own threshold
                continue
            box = tuple(float(v) for v in b["xyxy"])
            if max((_iou(box, g) for g in gt), default=0.0) < MINE_IOU:
                worst = max(worst, float(b["conf"]))
        if worst > 0:
            hits.append({"file_name": im["file_name"], "conf": round(worst, 4),
                         "boxes": len(anns)})
        if i % 500 == 0:
            print(f"  {i}/{len(pool)} tiles, {len(hits)} with a false positive", flush=True)
    hits.sort(key=lambda h: (-h["conf"], h["file_name"]))
    json.dump({"n_pool": len(pool), "n_hits": len(hits), "conf": MINE_CONF,
               "iou": MINE_IOU, "weights_sha256": _sha(w), "hits": hits},
              open(MINED, "w"), indent=1)
    print(f"\n  {len(hits)} of {len(pool)} unseen tiles carry a false positive at "
          f"conf >= {MINE_CONF}; top conf {hits[0]['conf'] if hits else 0}")
    print(f"  -> {os.path.relpath(MINED, ROOT)}")
    # VOID (e) COUNTS THE NEGATIVES, not the firing tiles. The first version counted
    # `hits` - every tile with a false positive, 12,001 of them - against a threshold
    # meant for tiles that hold NO BALL, of which there were 1,762 at the conf the run
    # actually used and 53 at the conf it declared. It passed while what it protected
    # was already broken.
    lab = _labelled()
    neg = [h for h in hits if not lab[h["file_name"]]]
    print(f"  of those, TRULY NEGATIVE (no annotation at all): {len(neg)}")
    if len(neg) < 1:
        print("  VOID (e): no truly-negative firing tiles at all")
        return 1
    return 0


def _sha(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def negatives(n=None):
    """The top `n` false-firing tiles that hold NO annotation. E2's added tiles.

    The predicate E1's mining lacked: `not anns`, per tile, not a count in aggregate.
    """
    lab = _labelled()
    m = json.load(open(MINED))
    # THE MINED FILE'S OWN THRESHOLD IS NOT TRUSTED. It records `conf`, and the first
    # run recorded 0.3 while the detector ran at 0.05; so re-threshold the per-tile
    # confidences the file stores and take only what clears MINE_CONF now.
    hits = [h for h in m["hits"] if h["conf"] >= MINE_CONF]
    neg = [h["file_name"] for h in hits if not lab[h["file_name"]]]
    if n is None:
        # V2: the THRESHOLD decides the count, not a ratio. Every truly-negative tile
        # that clears MINE_CONF, capped at N_NEG so the arm can never exceed 1:1.
        if not neg:
            raise SystemExit(f"VOID (e): no truly-negative tiles at conf >= {MINE_CONF}")
        return neg[:N_NEG]
    if len(neg) < n:
        raise SystemExit(f"VOID (e): {len(neg)} truly-negative tiles at conf >= "
                         f"{MINE_CONF} (file says it mined at {m['conf']}), need {n}")
    return neg[:n]


def build(which="e1"):
    """The arm's dataset dir: D1's 916 tiles + its added ones, by link."""
    import ball_clean as BC  # noqa: PLC0415
    if not os.path.exists(MINED):
        raise SystemExit("no mined set; run --mine first")
    mined = json.load(open(MINED))
    src = os.path.join(str(BC.OUT), "tiles", "train")
    d = json.load(open(os.path.join(src, "_annotations.coco.json")))
    keep_names = {im["file_name"] for im in BC.matched(BC.ARMS["d1"]["tiles"], seed=0)["images"]}
    if which == "d1p":
        # D1' ADDS NOTHING. Rebuilt from matched(916, seed=0) rather than shipping the
        # old tiles_d1 directory, and then DIFFED against it: a baseline that is only
        # believed to be D1's tiles is not a baseline
        # (reproduced-means-diffed-against-the-golden (internal note)).
        have = {f for f in os.listdir(os.path.join(str(BC.OUT), "tiles_d1", "train"))
                if f.endswith(".jpg")}
        if have != keep_names:
            raise SystemExit(f"VOID: rebuilt D1' differs from tiles_d1 by "
                             f"{len(have ^ keep_names)} file(s)")
        print(f"  D1' reproduces tiles_d1 exactly: {len(have)} files, zero difference")
    else:
        keep_names |= (set(negatives()) if which == "e2"
                       else {h["file_name"] for h in mined["hits"][:N_NEG]})
    imgs = [im for im in d["images"] if im["file_name"] in keep_names]
    ids = {im["id"] for im in imgs}
    anns = [a for a in d["annotations"] if a["image_id"] in ids]
    dest = os.path.join(str(BC.OUT), f"tiles_{which}")
    os.makedirs(os.path.join(dest, "train"), exist_ok=True)
    for im in imgs:
        link = os.path.join(dest, "train", im["file_name"])
        if not os.path.exists(link):
            os.symlink(os.path.join(src, im["file_name"]), link)
    json.dump({"images": imgs, "annotations": anns, "categories": d["categories"],
               "info": d.get("info", {}), "licenses": d.get("licenses", [])},
              open(os.path.join(dest, "train", "_annotations.coco.json"), "w"))
    # valid: real directory of links, never a symlinked dir - ball_clean.arm's rule.
    os.makedirs(os.path.join(dest, "valid"), exist_ok=True)
    vsrc = os.path.join(str(BC.OUT), "tiles", "valid")
    for f in sorted(os.listdir(vsrc)):
        link = os.path.join(dest, "valid", f)
        if not os.path.exists(link):
            os.symlink(os.path.join(vsrc, f), link)
    empt = sum(1 for im in imgs if im["id"] not in {a["image_id"] for a in anns})
    print(f"{which.upper()}: {len(imgs)} train tiles ({len(keep_names)} named, {empt} with no box), "
          f"{len(anns)} boxes, valid linked whole -> {os.path.relpath(dest, ROOT)}")
    print(f"  train it with ball_train.train(dataset_dir='{dest}', "
          f"output_dir='{os.path.join(str(BC.OUT), 'run_' + which)}', epochs="
          f"{BC.ARMS['d1']['epochs']})  # D1's epoch count, unchanged")
    return 0


#: The precision the kit pins when the caller does not say. NOT "auto": sm_75 (Turing,
#: the training PC's RTX 2060 SUPER) has no bf16 tensor cores, torch's
#: is_bf16_supported() counts EMULATION and returns True, and rfdetr's "auto" therefore
#: picks bf16-mixed - roughly 2x the step time of fp16 on that card and the worst of the
#: three mantissas for run-to-run reproducibility. model/validation/n46_pc/train.py has
#: said `amp_dtype = "fp16" if cuda and cap[0] < 8 else "auto"` since N46, with the
#: comment "Turing (7.x) has no native bfloat16; name fp16 explicitly rather than trust
#: auto". The knowledge was in the repo for this same machine; this kit started fresh
#: and inherited the confound instead.
AMP = "fp16"


def _added(which):
    """How many tiles this arm ADDS over D1's 916 - measured off the built directory."""
    import ball_clean as BC  # noqa: PLC0415
    d = os.path.join(str(BC.OUT), f"tiles_{which}", "train")
    if not os.path.isdir(d):
        return None
    n = len([f for f in os.listdir(d) if f.endswith(".jpg")])
    return n - BC.ARMS["d1"]["tiles"]


def pack(which="e1", amp_dtype=AMP):
    """The PC kit: the E1 tiles with their links resolved, and the SETTINGS AS A FILE.

    The settings are written from `ball_train.HP`, `T` and `EPOCHS` rather than retyped
    on the other machine. E1 minus D1 has to be the added tiles and nothing else, and a
    second copy of nine hyperparameters on a Windows box is how that stops being true -
    N48's +0.3311 was tile size, not a fine-tune
    (n48-ab-was-confounded-with-tile-size (internal note)).
    """
    import subprocess  # noqa: PLC0415

    import ball_clean as BC  # noqa: PLC0415
    import ball_train as BT  # noqa: PLC0415
    dest = os.path.join(str(BC.OUT), f"tiles_{which}")
    if not os.path.isdir(dest):
        raise SystemExit(f"no tiles_{which}; run --build {which} first")
    if which in ("d1p", "e2") and not amp_dtype:
        raise SystemExit("VOID (c): a paired arm may not ship an unpinned amp_dtype - "
                         "'auto' resolved to emulated bf16 on the PC and to MPS's path "
                         "on this Mac, which is how E1 lost its attribution")
    spec = {"resolution": BT.T, "epochs": BC.ARMS["d1"]["epochs"], "hp": BT.HP,
            # AMP is the tenth setting HP forgot. rfdetr defaults amp_dtype to "auto",
            # which resolved to emulated bf16 on the PC's Turing card while D1 trained on
            # MPS, so E1 minus D1 carries a precision difference nobody chose. None means
            # unpinned, which is what E1 ran under; the control passes what E1 resolved
            # to, so D1' matches it.
            "amp_dtype": amp_dtype,
            "dataset_dir": f"tiles_{which}", "output_dir": f"run_{which}",
            "baseline": {"arm": "d1", "weights_sha256": _sha(os.path.join(
                str(BC.OUT), "run_d1", "checkpoint_best_ema.pth"))},
            # THE BASE WEIGHTS TOO. RFDETRNano downloads rf-detr-nano.pth into
            # ~/.roboflow PER MACHINE, and run_d1's config records this Mac's local
            # path. If the two boxes hold different files the arms start from different
            # initialisations and nothing downstream shows it.
            "pretrain_sha256": _sha(os.path.expanduser(
                "~/.roboflow/models/rf-detr-nano.pth")),
            # n_neg WAS THE CAP, NOT THE COUNT. The PC read `n_neg: 916` off this spec and
            # found the arms differ by 53 tiles - correctly. V2 lets the THRESHOLD decide
            # the count, so the cap is nearly always the wrong number to report, and a
            # spec that misstates the intervention is how E1's story went wrong. Both are
            # recorded now, and the count is derived from the built directory rather than
            # from a constant.
            "mine_conf": MINE_CONF, "n_neg_cap": N_NEG,
            "n_added": _added(which),
            "mined": json.load(open(MINED)) if os.path.exists(MINED) else None}
    if spec["mined"]:
        spec["mined"].pop("hits", None)          # the file list is 12k lines; counts are the claim
    open(os.path.join(dest, "n64_spec.json"), "w").write(json.dumps(spec, indent=1))
    open(os.path.join(dest, "train_e1.py"), "w").write(_SCRIPT)
    zp = os.path.join(str(BC.OUT), f"n64_{which}.zip")
    if os.path.exists(zp):
        os.remove(zp)
    # zip WITHOUT -y, so every symlink is stored as its contents: the PC has no repo to
    # resolve them against.
    subprocess.run(["zip", "-rq", zp, f"tiles_{which}"], cwd=str(BC.OUT), check=True)
    mb = os.path.getsize(zp) / 1e6
    print(f"  {os.path.relpath(zp, ROOT)}  {mb:.0f} MB, links resolved")
    print(f"  epochs {spec['epochs']}, resolution {spec['resolution']}, hp {spec['hp']}")
    print(f"  on the PC: unzip, then `python train_e1.py`, then send back "
          f"run_{which}/checkpoint_best_ema.pth, metrics.csv AND training_config.json "
          f"- the config is what VOID (c) checks, so a run without it cannot be paired")
    return 0


def audit():
    """WHAT THE MINED SET ACTUALLY IS, which is not what the rule above called it.

    Added after E1 scored, because the mechanism column went the wrong way: candidates
    per frame ROSE 22.7%. "Hard negative" means a tile the model fires on that holds no
    ball. This ranked the unseen pool by false-positive confidence and never required
    the tile to be negative - and the corpus is ball-CENTRED crops, so most tiles have
    a ball in them. The counts below are the arm's real composition.
    """
    import ball_clean as BC  # noqa: PLC0415
    lab = _labelled()
    _src, pool, _d = unseen()
    pool_empty = sum(1 for im, anns in pool if not anns)
    hits = json.load(open(MINED))["hits"]
    neg_hits = [h for h in hits if not lab[h["file_name"]]]
    rows = []
    for label, names in (("D1's 916", [im["file_name"] for im in
                                       BC.matched(N_NEG, seed=0)["images"]]),
                         ("mined 916", [h["file_name"] for h in hits[:N_NEG]])):
        wb = sum(1 for n in names if lab[n])
        rows.append((label, wb, len(names) - wb))
        print(f"  {label}: {wb} with a ball box, {len(names) - wb} empty")
    print(f"  unseen pool {len(pool)} tiles, {pool_empty} with NO ball "
          f"({100 * pool_empty / len(pool):.1f}%)")
    print(f"  tiles D1 false-fires on: {len(hits)}; of those TRULY NEGATIVE: "
          f"{len(neg_hits)}")
    # WHICH THRESHOLD THESE COUNTS ARE AT. The file records `conf`, and the first mining
    # recorded 0.3 while the detector actually ran at BALL_CONF 0.05 - so the counts
    # above describe 0.05, which is what E1 was built from. The enforced figure is the
    # one E2 uses, and the two must never be quoted as each other.
    at_conf = [h for h in hits if h["conf"] >= MINE_CONF]
    at_neg = [h for h in at_conf if not lab[h["file_name"]]]
    print(f"  ^ those are at the threshold the first mining ACTUALLY used "
          f"(min hit conf {min(h['conf'] for h in hits):.2f}); E1 was built from them")
    print(f"  at the ENFORCED MINE_CONF {MINE_CONF}: {len(at_conf)} firing, "
          f"{len(at_neg)} truly negative = {100 * len(at_neg) / N_NEG:.1f}% of {N_NEG} "
          f"(E2 v2 uses these)")
    print(f"  so at the ENFORCED threshold a 1:1 arm needs {N_NEG} and is "
          f"{'buildable' if len(at_neg) >= N_NEG else 'NOT buildable'}; E2 v2 therefore "
          f"takes all {len(at_neg)} rather than a ratio")

    # SAME-MATCH SHARE. ball_clean.py:29-35 says it outright - soccer_top is ONE match
    # and soccer_side is ONE match, so a held-out clip here is a held-out MINUTE and
    # "every number this data can ever produce is same-match". An arm that adds other
    # minutes of the matches it is scored on is not adding independent data.
    import collections  # noqa: PLC0415
    import ball_cmp as C  # noqa: PLC0415
    scored_m = {c.rsplit("__", 1)[1].rsplit("_", 2)[0] for c in C.CLIPS}
    added = [h["file_name"] for h in hits[:N_NEG]]
    same = sum(1 for f in added if any(m in f for m in scored_m))
    print(f"  scored matches {sorted(scored_m)}")
    print(f"  of the {len(added)} added tiles, {same} ({100 * same / max(len(added), 1):.1f}%) "
          f"are other minutes of those same matches")

    # THE DETECTOR-LEVEL COMPARISON, which needs no GPU and no new run: both arms wrote
    # a metrics.csv against a valid split this asserts is identical. It is the closest
    # thing on disk to "is E1 a better detector", and it is far smaller than the oracle
    # change - which is monotone in candidate-list length and so cannot answer that.
    import csv  # noqa: PLC0415
    va = {}
    for arm in ("d1", "e1"):
        q = os.path.join(str(BC.OUT), f"tiles_{arm}", "valid", "_annotations.coco.json")
        m = os.path.join(str(BC.OUT), f"run_{arm}", "metrics.csv")
        if not (os.path.exists(q) and os.path.exists(m)):
            return 0
        d = json.load(open(q))
        va[arm] = (len(d["images"]), len(d["annotations"]), list(csv.DictReader(open(m))))
    assert va["d1"][:2] == va["e1"][:2], ("the valid splits differ", va["d1"][:2], va["e1"][:2])
    print(f"  valid split identical: {va['d1'][0]} images, {va['d1'][1]} boxes")
    for arm in ("d1", "e1"):
        rows = [r for r in va[arm][2] if r.get("val/ema_mAP_50")]
        best = max(rows, key=lambda r: float(r["val/ema_mAP_50"]))
        print(f"  {arm} best val/ema_mAP_50 {float(best['val/ema_mAP_50']):.4f} "
              f"at epoch {best['epoch']}, F1 {float(best['val/F1']):.4f}")
    return 0


def score():
    """The committed rule, run. Two arms, four VOID checks, one absolute bar."""
    import collections  # noqa: PLC0415

    import ball_clean as BC  # noqa: PLC0415
    import ball_cmp as C  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    import n57_conf_evidence as N57  # noqa: PLC0415

    w = os.path.join(str(BC.OUT), "run_e1", "checkpoint_best_ema.pth")
    sha = _sha(w)
    if not sha:
        raise SystemExit("no E1 weights at out/ball_clean/run_e1/checkpoint_best_ema.pth")
    # VOID (d): the number and the weights that made it never travel separately.
    print(f"  E1 weights sha256 {sha}")

    pop = {c: v[0] for c, v in N57.split().items()}
    arms = {"D1": ".cmpD", "E1": ".cmpE1"}
    # EVERY ARM OVER THE SAME FRAMES. ball_cmp.show() asserts this and says why:
    # build_cache flushes every 100 frames, so a killed build leaves a SHORTER cache
    # that still loads, and pooling rates across arms of different length is silent.
    # E2's mechanism gate is a RATIO of cands/frame between two arms, which is exactly
    # the quantity a short cache corrupts - a build stopping early on the hardest clip
    # would lower cands/frame toward the gate and raise shipped at the same time.
    caches = {a: {c: B.load_cache(c + suf) for c in C.CLIPS} for a, suf in arms.items()}
    for c in C.CLIPS:
        sets = {a: tuple(sorted(caches[a][c]["frames"])) for a in arms}
        if len(set(sets.values())) != 1:
            print(f"  VOID: arms scored different frames on {c[-9:]}: "
                  f"{ {a: len(v) for a, v in sets.items()} }")
            return 1
    picks = ("shipped", "conf", "oracle", "centre")
    per = {a: {} for a in arms}
    cands = {}
    for a, suf in arms.items():
        n_c = n_f = 0
        for clip in C.CLIPS:
            cache = caches[a][clip]
            frames = sorted(cache["frames"])
            gt, _wh, _n = B.load_ball(clip)
            gt_here = {f: gt[f] for f in frames if f in gt}
            per[a][clip] = {m: B.score(gt_here, B.predict(cache, m, gt_here, None),
                                      frames, B.GATE_PX) for m in picks}
            n_c += sum(len(cache["frames"][f]) for f in frames)
            n_f += len(frames)
        cands[a] = n_c / n_f

    def pooled(a, m, clips):
        c = sum(per[a][k][m]["correct"] for k in clips)
        n = sum(per[a][k][m]["gt_present"] for k in clips)
        return c / n if n else 0.0

    dead = [c for c in C.CLIPS if pop.get(c) == "dead"]
    play = [c for c in C.CLIPS if pop.get(c) != "dead"]

    # VOID (b) and (c), before any comparison is read.
    ctr = pooled("E1", "centre", C.CLIPS)
    print(f"  VOID (b) centre {ctr:.4f}")
    if ctr > 0.01:
        print("  VOID: the centre control scores over 1%")
        return 1
    d_ship, d_or = pooled("D1", "shipped", C.CLIPS), pooled("D1", "oracle", C.CLIPS)
    print(f"  VOID (c) D1 shipped {d_ship:.4f} vs 0.4706, oracle {d_or:.4f} vs 0.8822")
    if abs(d_ship - 0.4706) > 5e-4 or abs(d_or - 0.8822) > 5e-4:
        print("  VOID: D1 does not reproduce its own numbers, so there is no baseline")
        return 1

    print(f"\n  arm   cands/frame   shipped     dead   in-play    oracle     conf   "
          f"wrong  silent")
    for a in arms:
        ws = sum(per[a][k]["shipped"]["wrong"] for k in C.CLIPS)
        si = sum(per[a][k]["shipped"]["silent"] for k in C.CLIPS)
        print(f"  {a:4s}  {cands[a]:11.1f}   {pooled(a, 'shipped', C.CLIPS):.4f}   "
              f"{pooled(a, 'shipped', dead):.4f}   {pooled(a, 'shipped', play):.4f}    "
              f"{pooled(a, 'oracle', C.CLIPS):.4f}   {pooled(a, 'conf', C.CLIPS):.4f}   "
              f"{ws:5d}   {si:5d}")

    e_ship, e_or = pooled("E1", "shipped", C.CLIPS), pooled("E1", "oracle", C.CLIPS)
    bar, floor = 0.4706 + 0.05, 0.8822 - 0.01
    d_ok = pooled("E1", "shipped", dead) - pooled("D1", "shipped", dead) >= -0.01
    p_ok = pooled("E1", "shipped", play) - pooled("D1", "shipped", play) >= -0.01
    print(f"\n  E1 shipped {e_ship:.4f} vs bar {bar:.4f} ({e_ship - bar:+.4f})")
    print(f"  E1 oracle  {e_or:.4f} vs floor {floor:.4f} ({e_or - floor:+.4f})")
    print(f"  dead {pooled('E1', 'shipped', dead):.4f} vs D1 "
          f"{pooled('D1', 'shipped', dead):.4f}   in-play "
          f"{pooled('E1', 'shipped', play):.4f} vs {pooled('D1', 'shipped', play):.4f}"
          f"   floor -0.01 each")
    ok = e_ship >= bar and e_or >= floor and d_ok and p_ok
    print(f"  RULE (committed): shipped >= {bar:.4f} AND oracle >= {floor:.4f} AND "
          f"neither population below -0.01 -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"  candidates per frame {cands['D1']:.1f} -> {cands['E1']:.1f} "
          f"({100 * (cands['E1'] / cands['D1'] - 1):+.1f}%), the arm's MECHANISM")
    print("  ATTRIBUTION IS NOT AVAILABLE from this table: E1 trained on CUDA/bf16 and "
          "D1 on MPS, so\n  the paired D1' control pre-registered in this file's "
          "addendum is what says whether the\n  tiles are the reason.")
    return 0


def _hits(cache, gt, frames, gate):
    """Per frame: (candidate confs, whether each is the ball). One pass, reused by both
    the recall curve and the false-candidate count, so they cannot disagree."""
    import numpy as np  # noqa: PLC0415
    out = {}
    for f in frames:
        g = gt.get(f)
        row = cache["frames"][f]
        cf = np.array([r[4] for r in row], float)
        if g is None:
            out[f] = (cf, np.zeros(len(row), bool), False)
            continue
        cx = np.array([(r[0] + r[2]) / 2 for r in row])
        cy = np.array([(r[1] + r[3]) / 2 for r in row])
        out[f] = (cf, np.hypot(cx - g[0], cy - g[1]) <= gate, True)
    return out


def _recall_at(h, t):
    """Share of gt-present frames where SOME candidate at conf >= t is the ball."""
    n = ok = 0
    for cf, is_ball, present in h.values():
        if not present:
            continue
        n += 1
        ok += bool((is_ball & (cf >= t)).any())
    return ok / n if n else 0.0


def _fc_at(h, t):
    """False candidates per frame at conf >= t - every candidate that is NOT the ball."""
    tot = n = 0
    for cf, is_ball, _present in h.values():
        n += 1
        tot += int(((cf >= t) & ~is_ball).sum())
    return tot / n if n else 0.0


def matched(h_a, h_b, steps=200):
    """Both arms at the SAME oracle recall, then their false-candidate counts.

    The threshold is not a free parameter: each arm's is solved for the common recall
    target, which is the lower arm's own ceiling. That is what makes the comparison
    immune to "emit more boxes" - the intervention E1 passed V1's gate with.
    """
    import numpy as np  # noqa: PLC0415
    target = min(_recall_at(h_a, 0.0), _recall_at(h_b, 0.0))
    out = []
    for h in (h_a, h_b):
        grid = np.linspace(0.0, 1.0, steps + 1)
        ok = [t for t in grid if _recall_at(h, t) >= target - 1e-12]
        t = max(ok) if ok else 0.0
        out.append((t, _recall_at(h, t), _fc_at(h, t)))
    return target, out


def _boot(per_clip, clips, reps=10000, seed=0):
    """Paired cluster bootstrap over CLIPS. Frames inside a clip are correlated, and
    N59 measured the clip-level swing on this corpus at 0.09; resampling frames would
    give an interval that is narrow and wrong."""
    import numpy as np  # noqa: PLC0415
    rng = np.random.default_rng(seed)
    idx = np.arange(len(clips))
    reps_out = []
    for _ in range(reps):
        pick = rng.choice(idx, size=len(clips), replace=True)
        ca = sum(per_clip[clips[i]]["a"][0] for i in pick)
        na = sum(per_clip[clips[i]]["a"][1] for i in pick)
        cb = sum(per_clip[clips[i]]["b"][0] for i in pick)
        nb = sum(per_clip[clips[i]]["b"][1] for i in pick)
        if na and nb:
            reps_out.append(cb / nb - ca / na)
    q = np.percentile(reps_out, [2.5, 97.5])
    return float(q[0]), float(q[1])


def res(arm="D"):
    """N65: is every ball number in this repo measured at the WRONG input resolution?

    RULE, fixed before the cache is built
        `RFDETRNanoConfig().resolution` is 384 and NO ball call site has ever passed
        `resolution`; `run_d1/training_config.json` records the checkpoint was TRAINED at
        640. So the 640 tile is resized to 384 at inference - a further 1.67x shrink of a
        ball that is already a few pixels across - and `ball_cmp`'s "tile 640 because that
        is what it was trained at" plus `build_cache`'s "saw the ball at NATIVE scale
        inside 640 px tiles" both describe something that does not happen.
    V1 OF THIS RULE WAS VACUOUS AND IS WITHDRAWN. It tested `oracle` on ONE clip,
    ball_cmp.CLIPS[0] - which is the DEAD-BALL clip, where the ball is stationary and
    visible and some candidate is within 20 px on every frame. `oracle` read 1.0000 at
    BOTH resolutions: the test statistic was at its ceiling and could not move, so
    "EXISTING NUMBERS STAND" measured nothing. Same failure as N61's first bucket, which
    was clip-wide min/max and so could not put anything outside it.

    The statistics that COULD move did: `conf` 0.4467 -> 0.6767 and `shipped` 0.0100 ->
    0.0433 on that clip, at 79.4 -> 82.8 candidates a frame.

    RULE V2, fixed before any further cache is built
        POPULATION  ALL SIX clips of ball_cmp.CLIPS, pooled, 1800 gt-present frames, gate
                    20 px, D1's checkpoint, tile 640. Pooled because a per-clip statistic
                    saturates on the dead clips, which is exactly what voided V1.
        VERDICT     if pooled `conf` OR pooled `oracle` moves by more than 0.01 in either
                    direction, every ball number in this repo was measured at an input
                    resolution the checkpoint was not trained at, and none may be quoted
                    as a property of the detector until it is re-measured at 640.
        WHY BOTH    `oracle` is the detector's ceiling over its own candidates and answers
                    "is the ball findable at all"; `conf` is plain argmax and answers
                    "is it findable FIRST", which is what the picker consumes and what
                    moved by 0.23 on the one clip measured. Using only the ceiling is how
                    V1 tested a saturated number.
        VOID        (a) the two caches do not cover the same frames;
                    (b) pooled `oracle` at 384 is above 0.99, i.e. the ceiling statistic
                        is saturated over the whole population too - then this design
                        cannot answer the question and says so instead of passing.
        `shipped` is reported beside them and is not the test: it also depends on the
        picker, and on the dead clips the gates dominate it.
    """
    import ball_cmp as C  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    import ball_train as BT  # noqa: PLC0415

    kw = C.ARMS[arm]
    suf = kw["suffix"] + "r"        # ".cmpDr", ".cmpE1r", ...
    picks = ("shipped", "conf", "oracle")
    base = kw["suffix"]
    acc = {sx: {m: [0, 0] for m in picks} for sx in (base, suf)}
    cands = {sx: [0, 0] for sx in (base, suf)}
    for clip in C.CLIPS:
        if not os.path.exists(B.cache_path(clip + suf)):
            print(f"  building {clip[-9:]} at resolution {BT.T} (trained), "
                  f"tile {kw['tile']}", flush=True)
            B.build_cache(clip, n_frames=C.FRAMES, conf=B.CONF, detector=kw["detector"],
                          weights=kw["weights"], tile=kw["tile"], variant=kw["variant"],
                          suffix=suf, resolution=BT.T)
        seen = {}
        for sx in (base, suf):
            cache = B.load_cache(clip + sx)
            frames = sorted(cache["frames"])
            seen[sx] = frames
            gt, _wh, _n = B.load_ball(clip)
            gt_here = {f: gt[f] for f in frames if f in gt}
            for m in picks:
                sc = B.score(gt_here, B.predict(cache, m, gt_here, None), frames,
                             B.GATE_PX)
                acc[sx][m][0] += sc["correct"]
                acc[sx][m][1] += sc["gt_present"]
            cands[sx][0] += sum(len(v) for v in cache["frames"].values())
            cands[sx][1] += len(frames)
        if seen[base] != seen[suf]:      # VOID (a)
            print(f"  VOID (a): the caches differ in frames on {clip[-9:]}")
            return 1

    def rate(sx, m):
        c, n = acc[sx][m]
        return c / n if n else 0.0

    for label, sx in ((f"arm {arm} at 384 (as measured all along)", base),
                      (f"arm {arm} at {BT.T} (as trained)", suf)):
        print(f"  {label:28s} " + "  ".join(f"{m} {rate(sx, m):.4f}" for m in picks)
              + f"  cands/frame {cands[sx][0] / cands[sx][1]:.1f}  "
                f"n={acc[sx]['oracle'][1]}")
    if rate(base, "oracle") > 0.99:      # VOID (b)
        print(f"  VOID (b): pooled oracle at 384 is {rate(base, 'oracle'):.4f}, "
              f"saturated - this design cannot answer the question")
        return 1
    d_conf = rate(suf, "conf") - rate(base, "conf")
    d_orc = rate(suf, "oracle") - rate(base, "oracle")
    print(f"  pooled deltas: conf {d_conf:+.4f}, oracle {d_orc:+.4f}, "
          f"shipped {rate(suf, 'shipped') - rate(base, 'shipped'):+.4f}")
    hit = abs(d_conf) > 0.01 or abs(d_orc) > 0.01
    print(f"  RULE V2: |conf| or |oracle| > 0.01 -> "
          f"{'RE-MEASURE every ball number at 640' if hit else 'EXISTING NUMBERS STAND'}")
    return 0


def pair(a="d1p", b="e2"):
    """The V2 rule, run: D1' against E2, three gates, VOID (c) and (d) in code."""
    import ball_clean as BC  # noqa: PLC0415
    import ball_cmp as C  # noqa: PLC0415
    import ball_eval as B  # noqa: PLC0415
    import ball_train as BT2  # noqa: PLC0415 - T is the resolution VOID (f) requires

    suf = {a: f".cmp{a.upper()}", b: f".cmp{b.upper()}"}
    # VOID (d): both checkpoints hashed, beside the number, before anything is read.
    sha = {}
    for arm in (a, b):
        w = os.path.join(str(BC.OUT), f"run_{arm}", "checkpoint_best_ema.pth")
        sha[arm] = _sha(w)
        if not sha[arm]:
            raise SystemExit(f"VOID (d): no weights for {arm} at {w}")
        print(f"  {arm} weights sha256 {sha[arm]}")
    # VOID (c): the two runs must agree on device and precision, or the pair is
    # confounded exactly as E1 was - and E1's own run never sent this file back.
    cfg = {}
    for arm in (a, b):
        q = os.path.join(str(BC.OUT), f"run_{arm}", "training_config.json")
        if not os.path.exists(q):
            raise SystemExit(f"VOID (c): {arm} sent no training_config.json, so device "
                             f"and amp_dtype cannot be compared")
        d = json.load(open(q))
        cfg[arm] = (d["model_config"].get("device"),
                    d["train_config"].get("amp_dtype"),
                    d["train_config"].get("seed"))
    if cfg[a] != cfg[b]:
        print(f"  VOID (c): {a} ran {cfg[a]}, {b} ran {cfg[b]} (device, amp_dtype, seed)")
        return 1
    print(f"  VOID (c) ok: both ran device={cfg[a][0]} amp_dtype={cfg[a][1]} "
          f"seed={cfg[a][2]}")

    per, hh = {}, {}
    seen_frames = {}
    for arm in (a, b):
        for clip in C.CLIPS:
            cache = B.load_cache(clip + suf[arm])
            # VOID (f), and NOTHING in this repo read cache["resolution"] until now:
            # ball_eval wrote the field and no code read it back, so the clause was prose.
            if cache.get("resolution") != BT2.T:
                print(f"  VOID (f): {arm}'s cache for {clip[-9:]} was built at resolution "
                      f"{cache.get('resolution')}, not the trained {BT2.T}")
                return 1
            # and the cross-arm frame-set check score() has, for the reason score() gives:
            # a killed build leaves a SHORTER cache that still loads, and the mechanism gate
            # is a RATIO between arms, which is exactly what a short cache corrupts.
            seen_frames.setdefault(clip, {})[arm] = tuple(sorted(cache["frames"]))
            frames = sorted(cache["frames"])
            gt, _wh, _n = B.load_ball(clip)
            gt_here = {f: gt[f] for f in frames if f in gt}
            sc = B.score(gt_here, B.predict(cache, "shipped", gt_here, None), frames,
                         B.GATE_PX)
            orc = B.score(gt_here, B.predict(cache, "oracle", gt_here, None), frames,
                          B.GATE_PX)
            ctr = B.score(gt_here, B.predict(cache, "centre", gt_here, None), frames,
                          B.GATE_PX)
            per.setdefault(clip, {}).setdefault("ctr", [0, 0])
            per[clip]["ctr"][0] += ctr["correct"]
            per[clip]["ctr"][1] += ctr["gt_present"]
            per.setdefault(clip, {})[("a" if arm == a else "b")] = (
                sc["correct"], sc["gt_present"], orc["correct"])
            hh.setdefault(arm, {})[clip] = _hits(cache, gt_here, frames, B.GATE_PX)

    def pooled(k, i=0):
        c = sum(per[cl][k][i] for cl in C.CLIPS)
        n = sum(per[cl][k][1] for cl in C.CLIPS)
        return c / n if n else 0.0

    # VOID (b): the control, which pair() did not score at all while the rule said it did.
    cn = sum(per[cl]["ctr"][0] for cl in C.CLIPS) / max(
        sum(per[cl]["ctr"][1] for cl in C.CLIPS), 1)
    print(f"  VOID (b) centre {cn:.4f}")
    if cn > 0.01:
        print("  VOID (b): the centre control scores over 1%")
        return 1
    for cl, byarm in seen_frames.items():
        if len(set(byarm.values())) != 1:
            print(f"  VOID: arms scored different frames on {cl[-9:]}: "
                  f"{ {k: len(v) for k, v in byarm.items()} }")
            return 1
    sa, sb = pooled("a"), pooled("b")
    oa, ob = pooled("a", 2), pooled("b", 2)
    # KEYED BY (clip, frame), NOT frame. build_cache numbers every clip's frames
    # `for f in range(want)`, so all six clips share the keys 0..299 and a flat {f: v}
    # keeps only the LAST clip written - matched() then solved its thresholds and counted
    # false candidates on ONE clip of six while every label said "pooled 1800". Measured:
    # the collided version reported 48.8 -> 28.4 (-41.8%); keyed correctly it is
    # 54.25 -> 43.42 (-20.0%). Deterministic, so verify_claims re-confirmed the wrong
    # number every time: a green measured tag is not evidence that the population is what
    # the label says it is.
    flat = {arm: {(cl, f): v for cl in C.CLIPS for f, v in hh[arm][cl].items()}
            for arm in (a, b)}
    for arm in (a, b):
        assert len(flat[arm]) == sum(len(hh[arm][cl]) for cl in C.CLIPS), \
            f"{arm}: {len(flat[arm])} pooled keys from " \
            f"{sum(len(hh[arm][cl]) for cl in C.CLIPS)} frames - keys collided"
    target, ((ta, ra, fa), (tb, rb, fb)) = matched(flat[a], flat[b])
    # ONE LOAD PER CLIP, not one per frame. The same comprehension shape in n59 cost
    # 2,431s against a 300s gate timeout and blocked a push; 33.6s after hoisting.
    raw = {}
    for arm in (a, b):
        n_c = n_f = 0
        for cl in C.CLIPS:
            cc = B.load_cache(cl + suf[arm])
            n_c += sum(len(v) for v in cc["frames"].values())
            n_f += len(cc["frames"])
        raw[arm] = n_c / n_f
    lo, hi = _boot(per, list(C.CLIPS))

    print(f"\n  matched oracle recall {target:.4f}: {a} at conf {ta:.3f} -> "
          f"{fa:.1f} false cands/frame; {b} at conf {tb:.3f} -> {fb:.1f}")
    print(f"  raw candidates/frame {raw[a]:.1f} -> {raw[b]:.1f} (the number V1 gated on)")
    print(f"  shipped {sa:.4f} -> {sb:.4f} (delta {sb - sa:+.4f}, bar +0.05)")
    print(f"  paired cluster bootstrap 95% CI over {len(C.CLIPS)} clips: "
          f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"  oracle  {oa:.4f} -> {ob:.4f} (floor {oa - 0.01:.4f})")
    for held in C.CLIPS:
        rest = [c for c in C.CLIPS if c != held]
        d = (sum(per[c]["b"][0] for c in rest) / sum(per[c]["b"][1] for c in rest)
             - sum(per[c]["a"][0] for c in rest) / sum(per[c]["a"][1] for c in rest))
        print(f"    without {held[-9:]}: delta {d:+.4f}")
    mech = fb <= 0.90 * fa
    out = (sb - sa) >= 0.05 and lo > 0
    guard = ob >= oa - 0.01
    print(f"  MECHANISM {'PASS' if mech else 'FAIL'}  OUTCOME "
          f"{'PASS' if out else 'FAIL'}  GUARD {'PASS' if guard else 'FAIL'}")
    print(f"  RULE V2 (all three) -> {'SUPPORTED' if (mech and out and guard) else 'NOT SUPPORTED'}")
    return 0


def demo():
    """The mining predicate, which is the only arithmetic here that can be wrong."""
    # IoU: a box against itself is 1, a disjoint pair is 0, and half-overlap is 1/3
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    # the predicate: a fired box counts as a false positive only if it matches NOTHING
    gt = [(0.0, 0.0, 10.0, 10.0)]
    assert max((_iou((0, 0, 10, 10), g) for g in gt), default=0.0) >= MINE_IOU
    assert max((_iou((90, 90, 99, 99), g) for g in gt), default=0.0) < MINE_IOU
    # AND an empty tile makes every fired box a false positive, which is the point
    assert max((_iou((0, 0, 10, 10), g) for g in []), default=0.0) < MINE_IOU
    # D1's own tiles are recoverable exactly, so "unseen" means unseen
    import ball_clean as BC  # noqa: PLC0415
    a = {im["file_name"] for im in BC.matched(N_NEG, seed=0)["images"]}
    b = {im["file_name"] for im in BC.matched(N_NEG, seed=0)["images"]}
    assert a == b and len(a) == N_NEG, (len(a), len(b))
    # THE KIT MAY NOT HOLD ITS OWN COPY OF THE SETTINGS. If a spec has been packed,
    # every value in it must still equal ball_train's, or E1 was trained at settings
    # D1 was not and the arm measures the settings.
    import ball_clean as _BC  # noqa: PLC0415
    import ball_train as _BT  # noqa: PLC0415
    # EVERY packed spec, not just tiles_e1 - the two that matter (d1p, e2) were the two
    # this check did not look at, and E2's VOID (c) hangs on them agreeing.
    import glob  # noqa: PLC0415
    _specs = sorted(glob.glob(os.path.join(str(_BC.OUT), "tiles_*", "n64_spec.json")))
    # A SPEC WHOSE ARM HAS ALREADY TRAINED IS A RECORD, NOT AN INSTRUCTION. tiles_e1's
    # spec documents what E1 actually ran - HP without a seed, amp_dtype unpinned - and
    # rewriting it to match today's HP would falsify the one file that says why E1's
    # attribution is gone. Pending kits must match HP; trained ones must not be touched.
    _amps = set()
    for _f in _specs:
        _arm = os.path.basename(os.path.dirname(_f)).replace("tiles_", "")
        if os.path.isdir(os.path.join(str(_BC.OUT), f"run_{_arm}")):
            continue
        _s = json.load(open(_f))
        assert _s["hp"] == _BT.HP, (_f, _s["hp"], _BT.HP)
        assert _s["resolution"] == _BT.T and _s["epochs"] == _BC.ARMS["d1"]["epochs"], _f
        if os.path.basename(os.path.dirname(_f)) in ("tiles_d1p", "tiles_e2"):
            _amps.add(_s.get("amp_dtype"))
    assert None not in _amps, "a control arm's spec leaves amp_dtype unpinned"
    assert len(_amps) <= 1, f"the paired arms pin different precision: {_amps}"
    # THE MATCHED-RECALL MECHANISM METRIC, on synthetic arms where the answer is known.
    # The decoys OUTRANK the ball here, which is the regime that matters: the shipped
    # picker is right on 47% of frames, so the ball is not top-1 on most of them. To
    # reach the recall target an arm must drop its threshold below the decoys and admit
    # them, and the count of what it admits is the thing the picker fights.
    import numpy as _np  # noqa: PLC0415
    def _arm(n_decoy):
        return {f: (_np.array([0.5] + [0.9] * n_decoy),
                    _np.array([True] + [False] * n_decoy), True) for f in range(20)}
    _lean, _fat = _arm(1), _arm(4)
    _t, ((_ta, _ra, _fa), (_tb, _rb, _fb)) = matched(_lean, _fat)
    assert abs(_ra - _rb) < 1e-9 and _ra == 1.0, (_ra, _rb)
    assert (_fa, _fb) == (1.0, 4.0), (_fa, _fb)
    assert _fb > _fa, "more decoys at equal recall must read worse"
    # a genuinely cleaner arm - same recall, decoys gone - and the 0.90x gate can pass
    _clean = {f: (_np.array([0.5]), _np.array([True]), True) for f in range(20)}
    _t2, ((_, _, _f1), (_, _, _f2)) = matched(_lean, _clean)
    assert _f2 == 0.0 and _f2 <= 0.90 * _f1, (_f1, _f2)
    # recall is monotone non-increasing in the threshold, which is what makes the solve
    assert _recall_at(_lean, 0.0) >= _recall_at(_lean, 0.6) >= _recall_at(_lean, 0.95)
    assert _recall_at(_lean, 0.6) == 0.0   # above the ball, below the decoy: nothing
    # AND the raw count cannot tell these apart the same way: it rises with box count
    # whatever the ranking, which is how E1 passed V1's column while getting noisier.
    assert _fc_at(_fat, 0.0) == 4.0 > _fc_at(_lean, 0.0) == 1.0
    # the cluster bootstrap resamples CLIPS: identical clips -> zero-width interval
    _pc = {f"c{i}": {"a": (40, 100, 90), "b": (50, 100, 90)} for i in range(6)}
    _lo, _hi = _boot(_pc, [f"c{i}" for i in range(6)])
    assert abs(_lo - 0.10) < 1e-12 and abs(_hi - 0.10) < 1e-12, (_lo, _hi)
    # and a table where one clip carries the whole delta must give a WIDE interval,
    # which is the property that would have refused E1's 0.0016
    _pc2 = {f"c{i}": {"a": (40, 100, 90), "b": (40 + (60 if i == 0 else 0), 100, 90)}
            for i in range(6)}
    _lo2, _hi2 = _boot(_pc2, [f"c{i}" for i in range(6)])
    assert _lo2 <= 0.0 < _hi2, (_lo2, _hi2)

    # THE PREDICATE THAT REPLACED THE BROKEN VOID (e) MUST ITSELF FAIL WHEN BROKEN.
    # A mutation review deleted `negatives()`'s no-ball filter - the exact
    # bug, re-planted - and demo() stayed green, because nothing here called it.
    import ball_clean as _BC2  # noqa: PLC0415
    if os.path.exists(MINED) and (_BC2.OUT / "tiles" / "train" / "_annotations.coco.json").exists():
        _lab = _labelled()
        _n = negatives(8)
        assert _n and not [f for f in _n if _lab[f]], \
            "negatives() returned a tile that holds a ball - VOID (e) counts the wrong thing again"
        try:
            negatives(10 ** 7)
        except SystemExit as e:
            assert "VOID (e)" in str(e), e
        else:
            raise AssertionError("the negative-pool floor does not refuse")

    # THE GENERATED SCRIPT MUST PARSE AND MUST GUARD __main__. The first kit's script
    # could not run on Windows at all, and nothing here would have noticed.
    import ast  # noqa: PLC0415
    ast.parse(_SCRIPT)
    assert 'if __name__ == "__main__":' in _SCRIPT and "def main():" in _SCRIPT
    print("n64_ball_negatives.py ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--mine" in a:
        raise SystemExit(mine())
    if "--build" in a:
        i = a.index("--build")
        raise SystemExit(build(a[i + 1] if len(a) > i + 1 and not a[i + 1].startswith("-")
                               else "e1"))
    if "--pack" in a:
        i = a.index("--pack")
        rest = [x for x in a[i + 1:] if not x.startswith("-")]
        raise SystemExit(pack(rest[0] if rest else "e1",
                              rest[1] if len(rest) > 1 else AMP))
    if "--audit" in a:
        raise SystemExit(audit())
    if "--res" in a:
        i = a.index("--res")
        raise SystemExit(res(a[i + 1] if len(a) > i + 1 and not a[i + 1].startswith("-")
                             else "D"))
    if "--pair" in a:
        raise SystemExit(pair())
    if "--score" in a:
        raise SystemExit(score())
    raise SystemExit(demo() or 0)
