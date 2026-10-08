"""Team assignment from a SigLIP embedding of the SHIRT region. Alternative to
`track.assign_teams`, which clusters the mean Lab colour of the same region.

Measured on `data/jobs/crop600` (600 frames of the real 118575 panorama, 1389
crops sampled at stride 10, 31 SoccerTrack-GT-matched tracks, the two officials
excluded by hand because that GT holds 22 PLAYERS and no crew). "gap" is
`median(away width_m) - median(home width_m)` out of `stats.shape_series`, the
flagship relative-shape number:

    method                          track acc   detection acc   width gap
    ground truth                          -             -        +8.88 m
    kit colour, 600 frames  (track.py)  83.9%         86.2%      +8.21 m   92%
    kit colour, same 1389 crops         80.6%         78.4%     +15.09 m  170%
    SigLIP FULL crop + kmeans           54.8%         57.8%      refused
    SigLIP FULL crop + umap3 + kmeans   51.6%         48.7%      refused
    SigLIP SHIRT crop + umap3 + kmeans  93.5%         98.7%      +8.88 m  100%

Read the middle rows before adopting this. Roboflow's published recipe -
embed the WHOLE detection box, UMAP to 3, k-means to 2 - scores at chance here
and produces banks so lopsided that `stats.shape_series` refuses to emit a
sample. On a 4096x1080 panorama the median player box is 25x53 px and most of
it is grass, sock and hair; the first principal component of the full-box
embedding carries 28.3% of the variance and correlates 0.32 with team, while
the best of the top twelve reaches 0.57. There is no team axis for k-means to
find. Cropping to the shirt first is what makes the embedding work, and that is
the same prior `track.team_by_shirt` already encodes - this module buys a better
READOUT of the shirt, not a new signal.

NOT WIRED IN, AND NOW THERE IS A NUMBER FOR WHY. Nothing imports this module.
Scored against SoccerTrack's per-detection team truth over the whole 118575
first half (`model/validation/team_eval.py`, 300 frames spread across the half, 4,101
detections matched to a GT player) this loses to plain kit colour:

    SigLIP shirt crops + umap3 + kmeans   67.8-69.4% detections, 89-90% tracks
    kit colour (track.team_by_shirt)      70.7%      detections, 93.0% tracks

(The kit-colour row is its value when this was written; the current, registered
value is api://team.accuracy.detection, benchmark (not published) §4.)

The table above still stands - it was measured on 600 frames of the same clip,
against the kit-colour readout AS IT THEN WAS, which clustered the mean Lab
colour and scored 78.3% per detection. That readout has since been replaced
(chromaticity plus a within-cluster metric, see `track._kmeans`), and the free
method now beats the 780 MB one. The three SigLIP rows here also move 1.6
points between seeds, where kit colour is seeded and deterministic.

WHAT WOULD STILL BE WORTH THE WEIGHTS: the goalkeeper. A keeper wears a third kit by
regulation, so kit-colour clustering cannot place him with his team by
construction - on this clip it puts BOTH keepers in the wrong bank, and
`stats._goalkeepers` then refuses on both, which is what drags the shipped gap
off truth. SigLIP places them from shape and context, not from kit colour:
8 of 12 seeds recover BOTH keepers and 11 of 12 recover at least one, and the
seeds that find both are exactly the ones that reproduce the true gap.

UMAP is load-bearing and is NOT interchangeable with PCA: on the same shirt
embeddings, k-means direct scores 52.8% and PCA-3 50.0%.

FEED IT ENOUGH CROPS. UMAP is a neighbourhood method and it is the crop count,
not the seed, that decides whether this works. At 464 crops (~1 per track per
second) 3 of 12 seeds collapsed to chance and `shape_series` refused to emit a
sample at all; at 1389 crops (~3 per track per second) 0 of 12 did, track
accuracy tightened to 87.1-93.5% and the gap held at 92-124% of truth. A single
run at a thin sampling rate is not a reading.

Licences, all permissive and all verified: SigLIP weights Apache-2.0, model card
`google/siglip-base-patch16-224`; transformers Apache-2.0; torch BSD-3-Clause;
umap-learn BSD-3-Clause (pulls numba, llvmlite, pynndescent, scikit-learn, all
BSD). Nothing here is AGPL and nothing here is a person-ReID model trained on
Market-1501 or DukeMTMC - see README on why those are barred.

Every heavy import is lazy, exactly like the rfdetr and OCR backends: the core
pipeline, `track.py` and this file's own clustering self-check all run with
none of them installed.
"""
import argparse

import cv2
import numpy as np

MODEL = "google/siglip-base-patch16-224"
# The same window `track.team_by_shirt` samples. Not a tuned
# hyper-parameter - it is the definition of "shirt" both methods share, which is
# what makes the two rows of the table above a like-for-like comparison.
SHIRT = (0.2, 0.5)


def shirt_crop(frame: np.ndarray, box: tuple[int, int, int, int],
               size: int = 96) -> np.ndarray | None:
    """Shirt region of one box, as a square RGB crop. None if the box is empty.

    Clamped the same way `team_by_shirt` is, and for the same reason: a negative
    index silently samples the far side of the image, which here would embed
    somebody else's shirt.
    """
    x1, y1, x2, y2 = box
    h = y2 - y1
    top, bot = max(0, y1 + int(SHIRT[0] * h)), max(0, y1 + int(SHIRT[1] * h))
    c = frame[top:bot, max(0, x1):max(0, x2)]
    if c.size == 0:
        return None
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()


def embed(crops: list[np.ndarray], batch: int = 32, device: str | None = None) -> np.ndarray:
    """(N, 768) mean-pooled SigLIP vision features. RGB uint8 crops in.

    The VISION tower only - `SiglipVisionModel`, not `SiglipModel` - so the text
    encoder is never built. Measured 51.5 crops/s on an M3 via MPS.
    """
    import torch  # noqa: PLC0415
    from transformers import SiglipImageProcessor, SiglipVisionModel  # noqa: PLC0415

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = SiglipImageProcessor.from_pretrained(MODEL)
    model = SiglipVisionModel.from_pretrained(MODEL).to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch):
            px = proc(images=crops[i:i + batch], return_tensors="pt")["pixel_values"].to(device)
            out.append(model(pixel_values=px).last_hidden_state.mean(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def assign_teams(embeddings: dict[int, list[np.ndarray]], seed: int = 0) -> dict[int, str]:
    """{track id: [shirt embedding, ...]} -> {track id: "home" | "away"}.

    Per-DETECTION clustering with a per-track majority vote, not clustering of
    per-track means: a track that is half occluded contributes a mean that sits
    between the two kits, and one such track in a bank of eleven is enough to
    move `stats.shape_series` by metres (see module docstring).

    NO "official" CLASS, unlike `track.assign_teams`. This is a real regression
    against that function and the reason this module does not simply replace it:
    k=2 must put the referee somewhere, and on this clip it puts him in a team.
    A referee inside a bank is a phantom high-distance profile - `track._by_kit`
    explains why that is worse than a missing player. Pair this with the
    kit-colour referee test, or gate on it, before shipping it into a job.
    """
    ids = [i for i, v in embeddings.items() if v]
    if len(ids) < 2:
        return {i: "home" for i in ids}
    import umap  # noqa: PLC0415

    X = np.concatenate([np.asarray(embeddings[i], np.float32) for i in ids])
    owner = np.concatenate([np.full(len(embeddings[i]), k) for k, i in enumerate(ids)])
    # 3 components, k-means after: k-means on the 768-d vectors, or on their top
    # 3 principal components, both score at chance here (52.8% / 50.0%). The
    # non-linearity is doing the work, so this is not a dimension-reduction
    # convenience that PCA could stand in for.
    Z = np.ascontiguousarray(
        umap.UMAP(n_components=3, random_state=seed).fit_transform(X), np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    lab = cv2.kmeans(Z, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)[1].ravel()

    # Majority vote per track, then the larger cluster is "home" - the same
    # arbitrary-but-stable convention `track.assign_teams` uses. Both labels are
    # cluster ids, not identities; nothing downstream may assume which is which.
    votes = np.array([np.bincount(lab[owner == k], minlength=2).argmax()
                      for k in range(len(ids))])
    big = 0 if (votes == 0).sum() >= (votes == 1).sum() else 1
    return {i: ("home" if v == big else "away") for i, v in zip(ids, votes)}


def _selfcheck() -> None:
    """Clustering on synthetic vectors, plus the crop geometry. No weights, no
    network - the embedding half is checked separately by `--real`, which needs
    the model on disk."""
    rng = np.random.default_rng(0)

    # Shirt window: must sample the SHIRT, never the shorts or the grass. A
    # frame striped by height makes that testable without a model.
    frame = np.zeros((100, 40, 3), np.uint8)
    frame[20:50] = (0, 0, 255)                     # the shirt band, boxes 0-100
    c = shirt_crop(frame, (0, 0, 40, 100))
    assert c is not None and c.shape == (96, 96, 3), None if c is None else c.shape
    assert (c[:, :, 0] > 200).all() and (c[:, :, 2] < 40).all(), "not the shirt band (or not RGB)"
    # A box running off the frame edge must clamp, never wrap to the far side.
    assert shirt_crop(frame, (-10, 0, 5, 100)) is not None
    assert shirt_crop(frame, (0, 0, 0, 100)) is None, "empty crop must be None, not garbage"

    # Two kits, 11 tracks each, 8 detections each. The vote must beat per-track
    # means, so give three tracks a half-occluded run whose MEAN lands between
    # the kits while every individual detection stays on the right side.
    A, B = rng.normal(0, 1, 32), rng.normal(0, 1, 32)
    emb: dict[int, list[np.ndarray]] = {}
    for k in range(11):
        emb[k] = [A * 6 + rng.normal(0, 1, 32) for _ in range(8)]
        emb[100 + k] = [B * 6 + rng.normal(0, 1, 32) for _ in range(8)]
    for k in (0, 1, 2):
        emb[k] = [A * 6 + rng.normal(0, 1, 32) for _ in range(5)] + \
                 [A * 6 - (A - B) * 2 + rng.normal(0, 1, 32) for _ in range(3)]
    out = assign_teams(emb)
    assert len(out) == 22, len(out)
    one = {out[k] for k in range(11)}
    two = {out[100 + k] for k in range(11)}
    assert len(one) == 1 and len(two) == 1 and one != two, (sorted(one), sorted(two))
    assert sum(v == "home" for v in out.values()) == 11, out

    # Degenerate inputs must degrade, never raise: a job with one track, or with
    # a track that got no usable crop at all, still has to produce a labelling.
    assert assign_teams({}) == {}
    assert assign_teams({7: [A]}) == {7: "home"}
    assert assign_teams({7: [A], 8: []}) == {7: "home"}
    print("teamembed.py ok")


def _real() -> None:
    """End to end against the weights: shapes, and that two kits really separate.

    Skipped by the default self-check because it needs ~780 MB of weights on
    disk. It is the only check that would catch a mean-pool over the wrong axis,
    or a BGR/RGB swap surviving into the processor.
    """
    rng = np.random.default_rng(0)
    crops, emb = [], {}
    for k, kit in enumerate([(220, 40, 40), (40, 40, 220)]):
        for j in range(8):
            img = np.full((96, 96, 3), 70, np.uint8)
            img[:] = kit
            img = np.clip(img.astype(int) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
            crops.append(img)
            emb.setdefault(k, []).append(len(crops) - 1)
    E = embed(crops)
    assert E.shape == (16, 768), E.shape
    assert np.isfinite(E).all()
    a, b = E[:8].mean(0), E[8:].mean(0)
    within = float(np.linalg.norm(E[:8] - a, axis=1).mean())
    between = float(np.linalg.norm(a - b))
    print(f"  within-kit {within:.2f}  between-kit {between:.2f}")
    assert between > within, "the embedding does not separate two flat kits"
    print("teamembed.py --real ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", action="store_true",
                   help="also run the check that loads the SigLIP weights")
    a = p.parse_args()
    _selfcheck()
    if a.real:
        _real()
