"""Player, goalkeeper or referee, from the crop. The third signal on that question.

WHY A THIRD ONE
    `track.assign_teams` asks it from KIT COLOUR and `roles.py` asks it from
    MOTION. Both fail on the same person, and the failure is structural rather
    than a tuning miss:

      colour  clusters kits per match, so it can only see someone who matches
              NEITHER team. Measured on a DFL clip: the on-field referee in
              yellow scores 4.9 team-MADs out where a real player in green
              scores 4.4. Half a MAD apart - no threshold separates them. It
              also cannot tell a referee from a GOALKEEPER, since a keeper
              contrasts with both teams by exactly the same rule.
      motion  roles.py's own docstring refuses the case: "NOT referee. A
              referee's distance and speed profile overlaps a box-to-box
              midfielder's."

    This asks what a person LOOKS like, which is the question the other two are
    not asking, and it is the one that answers the referee.

WHAT IT IS
    ResNet-18 (torchvision, BSD-3) finetuned on labelled crops from
    `martinjolif/football-player-detection` (CC BY 4.0; cut from the DFL
    Bundesliga Data Shootout clips, not SoccerNet). A classifier over crops,
    not a detector: the boxes are unchanged and each gets a name. OFF on the job
    path (track.run roles=False). Measured accuracy, including WITHDRAWN
    figures: detection-tracking-evidence (not published)#evidence-roles

WHERE IT IS KNOWN TO BE WRONG
    On a DFL clip, three real players in a dark green kit drew a referee vote at
    0.58-0.70 while the actual referee drew 1.00 and a touchline official 0.86.
    Dark kits look like officials to it. That gap is why the caller votes over a
    whole track and requires a MAJORITY SHARE rather than trusting one crop -
    see `vote`.

    ponytail: one 48x96 crop, no context. It cannot see that the "referee" it
    found is wearing the same kit as ten other people on the pitch, which is the
    one thing colour clustering does know. Upgrade path is to feed the colour
    gate's per-track distance in as a second input, not to grow this model.
"""

from __future__ import annotations

import os
from collections import Counter

import cv2
import numpy as np

#: Crop size the model was trained at. Changing either number invalidates the
#: weights - the network has a fixed flatten.
W, H = 48, 96
#: The 10% pad the training crops carried. Shorts and socks are as distinctive
#: as the shirt, and a box clipped tight to the torso throws that away.
PAD = 0.1
NAMES = ("goalkeeper", "player", "referee")
#: Share of a track's crops that must agree before the name is used. Chosen on one
#: DFL clip's tracks (the referee and a touchline official clear it; three dark-kit
#: players do not); no seeded measurement supports it. detection-tracking-evidence (not published)#evidence-roles
MIN_SHARE = 0.8
#: ...and a track this short has no vote worth counting.
MIN_VOTES = 25

WEIGHTS = os.environ.get("FT_ROLE_MODEL",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "models", "role_clf.pt"))

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class RoleNet:
    """Crops in, class names out. Constructed only when the weights exist."""

    def __init__(self, weights: str | None = None):
        import torch  # noqa: PLC0415 - heavy, and this class is optional
        import torchvision  # noqa: PLC0415

        path = weights or WEIGHTS
        if not os.path.exists(path):
            raise SystemExit(
                f"role model not found: {path}\n"
                "  train one with model/validation/train_roles.py, or set $FT_ROLE_MODEL")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = torchvision.models.resnet18()
        m.fc = torch.nn.Linear(512, len(NAMES))
        m.load_state_dict(ck["state"])
        # MPS is a real speed-up here and a real risk of taking a nine-hour run
        # down with it, so a failure to move falls back to CPU rather than
        # raising - the same bargain detectors.py makes with fp16.
        self.dev = "cpu"
        if torch.backends.mps.is_available():
            try:
                m = m.to("mps")
                self.dev = "mps"
            except Exception:            # noqa: BLE001
                m = m.to("cpu")
        self._torch = torch
        self.model = m.eval()
        self.names = tuple(ck.get("names", NAMES))

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """[(name, probability), ...], one per crop, in order."""
        if not crops:
            return []
        a = np.stack(crops)[..., ::-1].astype(np.float32) / 255.0    # BGR -> RGB
        a = (a - _MEAN) / _STD
        t = self._torch.from_numpy(a.transpose(0, 3, 1, 2).copy()).to(self.dev)
        with self._torch.no_grad():
            p = self.model(t).softmax(1).cpu().numpy()
        return [(self.names[int(i)], float(row[i])) for row, i in zip(p, p.argmax(1))]


def crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """One training-shaped crop, or None when the box has no pixels."""
    x1, y1, x2, y2 = box
    px, py = PAD * (x2 - x1), PAD * (y2 - y1)
    c = frame[max(0, int(y1 - py)):int(y2 + py), max(0, int(x1 - px)):int(x2 + px)]
    if c.size == 0:
        return None
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA)


def vote(counts: dict[int, Counter]) -> dict[int, str]:
    """Per-track class votes -> the name to use, for the tracks sure enough.

    A track that does not clear MIN_SHARE gets NO entry rather than its
    plurality: the caller treats a missing name as "no opinion" and falls back
    to kit colour, and on this model a weak plurality is exactly the dark-kit
    player it gets wrong.
    """
    out = {}
    for tid, c in counts.items():
        n = sum(c.values())
        if n < MIN_VOTES:
            continue
        name, k = c.most_common(1)[0]
        if k / n >= MIN_SHARE:
            out[tid] = name
    return out


def _selfcheck() -> None:
    """The vote must be a MAJORITY and a track must be long enough to have one.

    Both numbers are what stop the measured failure - dark-kit players drawing a
    0.58-0.70 referee vote - from deleting real players from a squad.
    """
    strong = {1: Counter({"referee": 90, "player": 10})}
    weak = {2: Counter({"referee": 60, "player": 40})}
    short = {3: Counter({"referee": 10})}
    assert vote(strong) == {1: "referee"}, vote(strong)
    assert vote(weak) == {}, "a 0.60 plurality was trusted"
    assert vote(short) == {}, "a 10-detection track was trusted"
    # Exactly at the threshold counts as agreement, or 0.8 means 0.81.
    assert vote({4: Counter({"referee": 80, "player": 20})}) == {4: "referee"}

    # The crop is the trained shape whatever the box, and a box off the frame
    # edge returns None instead of a wrong-sized array.
    img = np.zeros((200, 200, 3), np.uint8)
    assert crop(img, (50, 50, 90, 150)).shape == (H, W, 3)
    assert crop(img, (300, 300, 340, 400)) is None
    print("rolenet.py ok")


if __name__ == "__main__":
    _selfcheck()
