"""N63: how much of the collapsed class the fine-tune's TRAINING SET actually contained.

WHY THIS AND NOT ANOTHER PROBE
    N53 found one goalkeeper (128057 id 12) falling 0.9803 -> 0.3517 across all three
    fine-tune seeds while the clip mean rose (n53-per-id-collapse (not published)). N62 then
    ran the detector at conf 0.001 and found the box IS proposed on 0.983-1.000 of the
    frames the product loses, scored 0.0401 against 0.8940 for labelled outfield
    players on the same frames: the damage is in CLASSIFICATION, not proposal
    (n62-classified-down (not published)). Two explanations survive inside that branch and N62
    could not separate them:

      LABEL POLICY   n46_dataset labels "the 22 players and NO officials", so referees,
                     assistants, benches and staff are background. Its own docstring
                     states the risk: "a model that learns 'this referee kit is
                     background' may also suppress a player in a similar kit on another
                     ground." N61 measured that under half of what the fine-tune
                     suppressed can be an official, so this is not the whole story.
      COMPOSITION    the training set simply carried too few examples of this visual
                     class - an isolated figure in a third kit, far from the ball - for
                     the model to hold it as a positive.

    Three cheaper explanations are already dead and are not retested here: conf (N53
    [conf]: at the raw floor 0.20 the keeper reaches 0.509, and N62's own probe - a
    DIFFERENT run, conf floor 0.001 over 60 of 1,937 frames - puts his p90 at 0.2769;
    both under the shipped 0.5), tiling (N53 [bands]), keepers as a class (N53 [keepers]:
    the extreme-x pair GAINS on the other three TEST clips, +0.042/+0.104/+0.026, and
    only 128057's pair loses, -0.292) and kit darkness (N53 [darkness]: TRAIN carries
    DARKER torsos than this keeper - 117092 id 11 at V 67, 118575 id 13 at 103, keeper
    115 - and id 6 on this very clip is as dark at 115 and gains +0.013).

    COMPOSITION is the one survivor that a count can address, and it needs no GPU and
    no model: the built training set is on disk with every box it was trained on. This
    attributes each of those boxes to a ground-truth track id and asks how many were
    goalkeepers.

PASS RULE, fixed here before the first run, and it decides in both directions
    POPULATION  the TRAIN split of model/validation/out/n46_dataset as BUILT - its
                _annotations.coco.json, not a replay - which is what the three seeds
                trained on. VALID and TEST are not opened.
    PER-CAPITA  A = the keeper proxy's share of GT boxes on exactly the frames the
                build sampled. Measured, not assumed to be 2/22: a keeper cropped off
                the canvas or lost to VISIBLE_MIN never had a per-capita share to lose.
    KEEPER PROXY  N53's definition, reused unchanged and not re-tuned: per clip, the
                two track ids with the most extreme mean box-centre x over that clip's
                whole ground truth.
    B           the keeper proxy's share of the built TRAIN annotations.

      B <= A/3                          DEPLETED. The builder removes keepers, so
                                        composition is a live cause and the GPU day
                                        arm is to keep keeper tiles.
      B >= 2*A/3 and count >= FLOOR_N   PRESENT. Rarity is not available as an
                                        explanation; the surviving branch is label
                                        policy, and the GPU arm is the ignore-region
                                        relabel of N60, not more keeper crops.
      otherwise                         UNRESOLVED, and neither arm is justified here.

    VOID unless every built TRAIN annotation is claimed by exactly one GT id and the
    recomputed total equals manifest.json's own box count. An id attribution that
    silently drops boxes would move both shares in the same direction and the verdict
    with them (reproduced-means-diffed-against-the-golden (internal note)).

WHAT IT CANNOT SETTLE, stated before the number exists
    Not "this KIT was unseen". The model saw keepers in five other kits, none of them
    this one, so a high count kills "keepers are rare" and leaves "this appearance is
    new" untouched. On 128057 BOTH teams' kits are unseen - its teams appear nowhere in
    TRAIN - and only the keeper collapses, which is why kit alone was never the leading
    story; but this probe does not measure kit.

    .venv/bin/python model/validation/n63_train_composition.py            # self-check
    .venv/bin/python model/validation/n63_train_composition.py --count    # the run
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "engine"), os.path.join(ROOT, "model", "validation")]

import n46_dataset as N46  # noqa: E402 - tile_boxes, so the cut is the build's own, never a copy

#: 2000 keeper TILE-INSTANCES. Above it, "the model never saw one" is not an available
#: explanation. An earlier comment converted this to "about thirteen minutes on screen"
#: at a flat 2.5 Hz, which is wrong twice: build_split scales the step per clip (30, 10,
#: 10, 20, 10) so there is no single rate, and these are box instances duplicated across
#: overlapping tiles rather than frames. The number is declared here so it cannot be
#: moved after the count; it is not a duration.
FLOOR_N = 2000


def gt_with_ids(seq):
    """{frame: [(id, x0, y0, x1, y1)]} - gt_boxes() without throwing the id away."""
    out = collections.defaultdict(list)
    for line in open(f"{N46.MOT}/{seq}.txt"):
        f, i, x, y, w, h = line.split(",")[:6]
        x, y, w, h = float(x), float(y), float(w), float(h)
        out[int(f)].append((int(i), x, y, x + w, y + h))
    return out


def proxy_ids(gt):
    """N53's keeper proxy: the two most extreme mean-centre-x ids in the clip."""
    mx = collections.defaultdict(list)
    for rows in gt.values():
        for gi, x0, _y0, x1, _y1 in rows:
            mx[gi].append((x0 + x1) / 2)
    order = sorted(mx, key=lambda k: float(np.mean(mx[k])))
    return {order[0], order[-1]}


def attribute(split="train", root=N46.OUT):
    """Each built annotation -> the GT id whose cut box equals it. Exact, or VOID."""
    coco = json.load(open(os.path.join(root, split, "_annotations.coco.json")))
    by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(tuple(a["bbox"]))
    per_clip, unclaimed = {}, 0
    for seq in sorted({im["seq"] for im in coco["images"]}):
        gt = gt_with_ids(seq)
        gks = proxy_ids(gt)
        keep = tot = gt_keep = gt_tot = 0
        for im in coco["images"]:
            if im["seq"] != seq:
                continue
            tile, rows = tuple(im["tile"]), gt[im["frame"]]
            want = collections.Counter(by_img.get(im["id"], []))
            for gi, *box in rows:
                gt_tot += 1
                gt_keep += gi in gks
                for b in N46.tile_boxes([tuple(box)], tile):   # the build's own cut
                    if want[tuple(b)]:
                        want[tuple(b)] -= 1
                        tot += 1
                        keep += gi in gks
            unclaimed += sum(want.values())
        per_clip[seq] = {"boxes": tot, "keeper": keep, "gt_boxes": gt_tot, "gt_keeper": gt_keep,
                         "ids": sorted(gks)}
    return per_clip, unclaimed


def _verdict(a, b, n):
    """The rule, in ONE place. demo() used to re-implement this expression, so flipping
    the threshold here left demo() green - the check exercised its own copy."""
    return ("DEPLETED" if b <= a / 3 else
            "PRESENT" if b >= 2 * a / 3 and n >= FLOOR_N else "UNRESOLVED")


def count():
    man = json.load(open(os.path.join(N46.OUT, "manifest.json")))
    golden = man["counts"]["train"]["boxes"]
    per_clip, unclaimed = attribute("train")
    tot = sum(c["boxes"] for c in per_clip.values())
    keep = sum(c["keeper"] for c in per_clip.values())
    gt_tot = sum(c["gt_boxes"] for c in per_clip.values())
    gt_keep = sum(c["gt_keeper"] for c in per_clip.values())
    for s, c in sorted(per_clip.items()):
        print(f"  {s}  ids {c['ids']}  built {c['keeper']}/{c['boxes']} = "
              f"{c['keeper'] / max(c['boxes'], 1):.4f}   GT {c['gt_keeper']}/{c['gt_boxes']} = "
              f"{c['gt_keeper'] / max(c['gt_boxes'], 1):.4f}")
    print(f"\n  attributed {tot} of manifest {golden}, {unclaimed} annotation(s) unclaimed")
    if tot != golden or unclaimed:
        print("  VOID: the attribution is not one-to-one with the built set")
        return 1
    a, b = gt_keep / gt_tot, keep / tot
    verdict = _verdict(a, b, keep)
    print(f"  A (GT per-capita) {a:.4f}   B (built) {b:.4f}   B/A {b / a:.3f}   n {keep}")
    print(f"  RULE: <=A/3 DEPLETED, >=2A/3 and n>={FLOOR_N} PRESENT -> {verdict}")
    return 0


def demo():
    # tile_boxes is the build's; the attribution only has to find it again
    tile = (0, 0, 1024, 1024)
    box = (100.0, 100.0, 140.0, 220.0)
    cut = N46.tile_boxes([box], tile)
    assert len(cut) == 1, cut
    # the proxy picks the extremes of mean x, and only two of them
    gt = {0: [(1, 0, 0, 10, 10), (2, 500, 0, 510, 10), (3, 900, 0, 910, 10)]}
    assert proxy_ids(gt) == {1, 3}, proxy_ids(gt)
    # a verdict at each end of the fixed rule, so the rule is exercised, not just stated
    for b, a, n, want in ((0.01, 0.09, 9e9, "DEPLETED"), (0.08, 0.09, 9e9, "PRESENT"),
                          (0.05, 0.09, 9e9, "UNRESOLVED"), (0.08, 0.09, 1, "UNRESOLVED")):
        assert _verdict(a, b, n) == want, (b, a, n, _verdict(a, b, n), want)
    print("n63_train_composition.py ok")


if __name__ == "__main__":
    raise SystemExit(count() if "--count" in sys.argv else (demo() or 0))
