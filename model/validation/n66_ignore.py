"""N66: ignore regions for RF-DETR, which the library does not have.

    .venv/bin/python model/validation/n66_ignore.py            # self-check: the mask works and can fail
    .venv/bin/python model/validation/n66_ignore.py --source   # is the forked upstream function unchanged?

WHY
    N46's fine-tune lost one player - 128057 id 12, a goalkeeper, recall 0.9803 -> 0.3517
    on all three seeds while the clip mean rose (n53-per-id-collapse (not published)). N62 showed
    the detector PROPOSES him on 0.983-1.000 of the frames the product loses and scores him
    0.0401 against labelled outfielders' 0.8940, so it is classification. N63 killed rarity:
    keepers were 9.9% of training, above per-capita. What is left is a conjunction - an
    appearance TRAIN never carried, inside a policy that labels every person who is not one
    of the 22 as background (n63-keepers-were-in-the-training-set (not published)).

    The textbook fix for that disease is to stop supervising the uncovered people as
    negatives and mark them IGNORE instead (n60-sparse-annotation (not published)). RF-DETR cannot:

      @C 347 :: .venv/lib/python3.14/site-packages/rfdetr/datasets/coco.py, the line that drops iscrowd==1 at load
      @C 313-322 :: .venv/lib/python3.14/site-packages/rfdetr/models/criterion.py, loss_labels' negative weights

    The loader DELETES every `iscrowd=1` annotation, and `loss_labels` builds a negative
    weight for EVERY query-class cell and overwrites only the matched ones. So the obvious
    encoding - pseudo-label the uncovered people, flag them crowd - deletes the box and
    leaves the pixels as background, which is bit-for-bit today's policy.

HOW, AND WHY IT IS NOT A FORK OF THE PACKAGE
    Ignore regions ride as a SENTINEL ANNOTATION CLASS inside `boxes`/`labels`, not as a
    parallel field. `coco.py` only rejects an unknown `category_id` when `cat2label` is set,
    and otherwise passes it through - so a second category in the COCO json arrives as
    label 1 and, being an ordinary box, is cropped, flipped and resized correctly by every
    transform in the pipeline for free. A parallel field would have to be filtered in step
    with `keep` in `datasets/kornia_transforms.py`, and the first augmentation nobody
    checked would silently misalign it.

    Then ONE function is replaced at run time from this module - no vendored copy of the
    package, no patch file to re-apply:

      forward()      strips the sentinel targets into a stash before the matcher sees them,
                     so they never reach an index, a class one-hot or the num_boxes
                     denominator, and calls the real forward on what is left.
      loss_labels()  the upstream body, plus: for every query NOT matched to a target whose
                     PREDICTED box lies mostly inside a stashed region, zero the negative
                     weight. The query stops being told "there is nothing here".

    IoA, not IoU - intersection over the QUERY's area. A small prediction sitting inside a
    large ignore region has low IoU with it and must still be ignored, which is the whole
    case: the regions are people and the predictions are parts of them.

THE DRIFT GUARD, because a vendored function is a copy and copies rot
    `loss_labels` is forked, so `patch()` refuses unless the upstream source still hashes to
    SRC_SHA. An upstream edit then fails loudly at patch time instead of silently training
    against a stale copy of someone else's loss. `--source` prints both hashes.

    It also refuses unless `ia_bce_loss` is on, because that is the only branch forked here
    and it is the branch RF-DETR Nano and Medium actually use (`run_d1/training_config.json`
    records `ia_bce_loss: true`).

WHAT THIS FILE DOES NOT DO
    It does not train anything and it is not an experiment. It is the mechanism arm A needs,
    with a check that the mechanism is real. The scored arm - a pseudo-labelled dataset, a
    pre-registered bar, and the PER_ID_FLOOR = -0.10 already waiting in n46_finetune.py -
    comes after this, and its rule will be committed before it runs.
"""
import hashlib
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

#: The sentinel category.
#:
#: THE OUT-OF-RANGE PROPERTY IS CONDITIONAL, AND AN EARLIER VERSION OF THIS COMMENT STATED IT
#: AS A FACT. It claimed "num_classes stays 1, so this label is out of range ON PURPOSE". It
#: does not stay 1 by itself: rfdetr's `_detect_num_classes_for_training` counts the COCO
#: json's `categories`, and `_align_num_classes_from_dataset` auto-aligns before the module is
#: built unless the caller set num_classes explicitly. Measured on the two datasets -
#: out/n46_dataset resolves to 1 and out/n67_ignore to 2 - so an unpinned IG run gets a
#: SECOND logit column, the sentinel becomes a valid index, and the arm differs from its
#: baseline by a head shape as well as by the regions.
#:
#: So the property holds only if the caller PINS num_classes. Any training entry point using
#: these regions must pass num_classes explicitly and assert it, and `assert_pinned()` below
#: is what it calls to check.
IGNORE_LABEL = 1

#: The per-instance fields a target dict carries. forward() filters exactly these and
#: asserts nothing else is per-instance - it used to filter "every tensor whose first dim
#: happens to equal the label count", which also truncated `size` and `orig_size` (both
#: shape (2,)) on any tile with exactly 2 annotations, and `image_id` (shape (1,)) on a tile
#: holding the sentinel alone. Latent, because the criterion reads none of them - and a
#: coincidence is not a field list.
PER_INSTANCE = ("boxes", "labels", "area", "iscrowd", "masks", "keypoints")

#: How much of a PREDICTED box must lie inside an ignore region before that query stops
#: being supervised as background. 0.5 is the anchor-based convention for ignore regions.
IOA = 0.5

#: sha256 of the upstream loss_labels this module forked. Set by --source on first run.
SRC_SHA = "4dbc2a5c6f976ed6f67ca9a73f77526bba26db5158a0a852a3b7c37e805fa650"

_ORIG: dict = {}


def _sha_of(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def source():
    """Print the upstream hash beside the pinned one. The fork's tripwire."""
    from rfdetr.models.criterion import SetCriterion
    got = _sha_of(SetCriterion.loss_labels)
    print(f"  upstream loss_labels sha256 {got}")
    print(f"  pinned in this file        {SRC_SHA}")
    print("  MATCH" if got == SRC_SHA else "  DIFFERS - the fork is stale, re-read it before patching")
    return 0 if got == SRC_SHA else 1


def _ioa(pred_xyxy, ign_xyxy):
    """Intersection over the PREDICTED box's area, (Q, N)."""
    import torch
    q = pred_xyxy[:, None, :]
    g = ign_xyxy[None, :, :]
    lt = torch.maximum(q[..., :2], g[..., :2])
    rb = torch.minimum(q[..., 2:], g[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area = ((q[..., 2] - q[..., 0]) * (q[..., 3] - q[..., 1])).clamp(min=1e-9)
    return inter / area


def _ignore_mask(self, outputs):
    """(B, Q) bool: queries whose predicted box lies mostly inside a stashed region."""
    import torch
    from rfdetr.utilities import box_ops
    stash = getattr(self, "_n66_ignore", None)
    logits = outputs["pred_logits"]
    mask = torch.zeros(logits.shape[:2], dtype=torch.bool, device=logits.device)
    if not stash or "pred_boxes" not in outputs:
        return mask
    for b, ign in enumerate(stash):
        if ign is None or ign.numel() == 0:
            continue
        pred = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][b].detach())
        gi = box_ops.box_cxcywh_to_xyxy(ign.to(pred.device).detach())
        mask[b] = (_ioa(pred, gi) >= IOA).any(dim=1)
    return mask


def patch():
    """Replace SetCriterion.forward and loss_labels. Refuses on drift."""
    import torch
    import torch.nn.functional as F
    from rfdetr.models.criterion import SetCriterion
    from rfdetr.utilities import box_ops

    if _ORIG:
        return
    got = _sha_of(SetCriterion.loss_labels)
    if got != SRC_SHA:
        raise SystemExit(f"VOID: upstream loss_labels is {got}, not the forked {SRC_SHA}. "
                         f"Re-read it and re-pin before training against a stale copy.")
    _ORIG["forward"] = SetCriterion.forward
    _ORIG["loss_labels"] = SetCriterion.loss_labels

    def forward(self, outputs, targets, num_boxes=None):
        stash, clean = [], []
        for t in targets:
            lab = t.get("labels")
            if lab is None or lab.numel() == 0:
                stash.append(None)
                clean.append(t)
                continue
            ig = lab == IGNORE_LABEL
            stash.append(t["boxes"][ig] if ig.any() else None)
            if not ig.any():
                clean.append(t)
                continue
            keep = ~ig
            # EXACTLY the per-instance fields, by NAME. A field left unfiltered would
            # misalign against boxes/labels, which is the failure the sentinel-class design
            # exists to avoid; a field filtered by accident - because its first dim happened
            # to equal the annotation count - is the mirror of that and was the earlier bug.
            nt = dict(t)
            for k in PER_INSTANCE:
                v = t.get(k)
                if torch.is_tensor(v) and v.shape[:1] == lab.shape:
                    nt[k] = v[keep]
            # and nothing ELSE may be per-instance, or it is silently left misaligned
            extra = [k for k, v in t.items()
                     if k not in PER_INSTANCE and torch.is_tensor(v)
                     and v.ndim and v.shape[0] == lab.shape[0] and lab.shape[0] > 2]
            assert not extra, (f"unfiltered per-instance field(s) {extra} would misalign "
                               f"against boxes/labels - add them to PER_INSTANCE")
            clean.append(nt)
        self._n66_ignore = stash
        try:
            return _ORIG["forward"](self, outputs, clean, num_boxes)
        finally:
            self._n66_ignore = None

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        if not self.ia_bce_loss:
            raise SystemExit("VOID: n66 forked only the ia_bce_loss branch, and it is off")

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        alpha = self.focal_alpha
        gamma = 2
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        iou_targets, _ = box_ops.elementwise_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )
        pos_ious = iou_targets.clone().detach()
        prob = src_logits.sigmoid()
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = prob**gamma

        pos_ind = list(idx)
        pos_ind.append(target_classes_o)

        t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
        t = torch.clamp(t, 0.01).detach()

        pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
        neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)

        # ------------------------------------------------ N66, and this is the whole change
        ign = _ignore_mask(self, outputs)
        if ign.any():
            ign = ign.clone()
            ign[idx] = False            # a MATCHED query is supervised, ignore or not
            neg_weights = neg_weights * (~ign).unsqueeze(-1).to(neg_weights.dtype)
        # -------------------------------------------------------------------------------

        loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
        loss_ce = loss_ce.sum() / num_boxes
        return {"loss_ce": loss_ce}

    SetCriterion.forward = forward
    SetCriterion.loss_labels = loss_labels


def assert_pinned(model, dataset_dir):
    """Refuse to train unless num_classes was PINNED, and pinned to the baseline's value.

    The whole arm rests on the IG run differing from FT by the ignore regions and nothing
    else. rfdetr auto-aligns num_classes from the json's `categories`, and the IG json
    declares two, so an unpinned run silently trains a two-logit head against a one-logit
    baseline. This is VOID (d) made runnable.
    """
    from rfdetr.detr import RFDETR
    got = RFDETR._detect_num_classes_for_training(dataset_dir, use_grouppose_keypoints=False)
    pinned = getattr(model, "model_config", None)
    pinned = getattr(pinned, "num_classes", None) if pinned is not None else None
    if pinned is None or pinned != 1:
        raise SystemExit(f"VOID (d): num_classes is {pinned}, not pinned to 1. The IG dataset "
                         f"declares {got} categories and rfdetr would auto-align to it, so "
                         f"the arm would differ from FT by a head shape as well as by the "
                         f"ignore regions.")
    print(f"  num_classes pinned to {pinned}; the dataset would have auto-aligned to {got}")
    return pinned


def unpatch():
    from rfdetr.models.criterion import SetCriterion
    if not _ORIG:
        return
    SetCriterion.forward = _ORIG.pop("forward")
    SetCriterion.loss_labels = _ORIG.pop("loss_labels")
    _ORIG.clear()


def demo():
    """The mask must change the loss, and it must change it ONLY where it should."""
    import torch
    from rfdetr.models.criterion import SetCriterion

    # one image, three queries, one real target matched to query 0.
    # query 1's box sits inside the ignore region; query 2's does not.
    outputs = {"pred_logits": torch.full((1, 3, 1), 2.0),
               "pred_boxes": torch.tensor([[[0.10, 0.10, 0.05, 0.05],
                                            [0.50, 0.50, 0.04, 0.04],
                                            [0.90, 0.90, 0.04, 0.04]]])}
    real = {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.10, 0.10, 0.05, 0.05]])}
    ignore_box = torch.tensor([[0.50, 0.50, 0.20, 0.20]])
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    crit = SetCriterion.__new__(SetCriterion)
    crit.ia_bce_loss, crit.focal_alpha = True, 0.25

    base = _ORIG.get("loss_labels") or SetCriterion.loss_labels
    before = base(crit, outputs, [real], indices, torch.tensor(1.0))["loss_ce"]

    patch()
    try:
        # with a region over query 1
        crit._n66_ignore = [ignore_box]
        after = SetCriterion.loss_labels(crit, outputs, [real], indices, torch.tensor(1.0))["loss_ce"]
        # with no region at all, the forked body must reproduce upstream EXACTLY
        crit._n66_ignore = None
        same = SetCriterion.loss_labels(crit, outputs, [real], indices, torch.tensor(1.0))["loss_ce"]
        assert torch.allclose(same, before, atol=1e-6), (float(same), float(before))

        assert after < before - 1e-6, ("an ignore region must REMOVE negative loss",
                                       float(after), float(before))
        # and it must remove only query 1's share: query 2 is outside the region, so the
        # drop equals exactly one query's negative contribution, not two.
        crit._n66_ignore = [torch.tensor([[0.50, 0.50, 0.20, 0.20],
                                          [0.90, 0.90, 0.20, 0.20]])]
        both = SetCriterion.loss_labels(crit, outputs, [real], indices, torch.tensor(1.0))["loss_ce"]
        assert both < after - 1e-6, (float(both), float(after))
        assert abs((before - after) - (after - both)) < 1e-5, \
            ("each ignored query must remove the same amount", float(before), float(after), float(both))

        # a region over the MATCHED query must change nothing: it is supervised either way
        crit._n66_ignore = [torch.tensor([[0.10, 0.10, 0.20, 0.20]])]
        matched = SetCriterion.loss_labels(crit, outputs, [real], indices, torch.tensor(1.0))["loss_ce"]
        assert torch.allclose(matched, before, atol=1e-6), (float(matched), float(before))

        # IoA, not IoU: a TINY prediction inside a LARGE region is ignored, and its IoU
        # with that region is far below the 0.5 threshold, so IoU would have missed it.
        from rfdetr.utilities import box_ops
        tiny = box_ops.box_cxcywh_to_xyxy(torch.tensor([[0.50, 0.50, 0.01, 0.01]]))
        big = box_ops.box_cxcywh_to_xyxy(torch.tensor([[0.50, 0.50, 0.40, 0.40]]))
        assert float(_ioa(tiny, big)[0, 0]) == 1.0
        inter = 0.01 * 0.01
        iou = inter / (0.01 * 0.01 + 0.40 * 0.40 - inter)
        assert iou < 0.01, iou

        # forward() must strip the sentinel targets so they never reach the matcher
        seen = {}

        def fake_forward(self, o, t, n=None):
            # RECORD BOXES AND AREA TOO. Recording only labels let a forward() that filters
            # `labels` and leaves `boxes`/`area` misaligned pass this check - which is exactly
            # the failure PER_INSTANCE exists to prevent, surviving the test written to
            # prevent it.
            seen["labels"] = [x["labels"].tolist() for x in t]
            seen["boxes"] = [x["boxes"].tolist() for x in t]
            seen["area"] = [x["area"].tolist() for x in t]
            return {}

        _ORIG["forward"], keep = fake_forward, _ORIG["forward"]
        mixed = {"labels": torch.tensor([0, IGNORE_LABEL, 0]),
                 "boxes": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2],
                                        [0.9, 0.9, 0.1, 0.1]]),
                 "area": torch.tensor([1.0, 2.0, 3.0]),
                 # NOT per-instance, and shape (2,) - the real coco.py sets both of these,
                 # and the old shape-coincidence filter truncated them on any tile with
                 # exactly two annotations
                 "orig_size": torch.tensor([720, 1280]),
                 "size": torch.tensor([720, 1280])}
        SetCriterion.forward(crit, outputs, [mixed])
        assert seen["labels"] == [[0, 0]], seen
        # every per-instance field must have dropped the SAME row, not just labels
        assert len(seen["boxes"][0]) == 2 and len(seen["area"][0]) == 2, seen
        assert torch.allclose(torch.tensor(seen["boxes"][0]),
                              torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]),
                              atol=1e-6), seen["boxes"]
        assert torch.allclose(torch.tensor(seen["area"][0]), torch.tensor([1.0, 3.0])), \
            seen["area"]
        # and a field that is NOT per-instance must be left alone, not truncated by a
        # first-dimension coincidence
        assert mixed["orig_size"].tolist() == [720, 1280], mixed["orig_size"]
        assert crit._n66_ignore is None, "the stash must be cleared after forward"
        _ORIG["forward"] = keep
    finally:
        unpatch()
    print("n66_ignore.py ok - the ignore mask removes negative loss, only where it should")


if __name__ == "__main__":
    if "--source" in sys.argv[1:]:
        raise SystemExit(source())
    raise SystemExit(demo() or 0)
