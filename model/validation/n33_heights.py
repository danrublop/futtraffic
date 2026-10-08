"""N33: how tall does each side say the players are? A depth test that needs no second calibration.

WHY
    N31/N32 left the line-registered map and ST2's feed 4% apart along the camera ray, the far
    touchline aside. A box has a TOP as well as a bottom. With a physical camera - height and vertical
    scale pinned by the goalposts and a square-pixel prior (model/validation/n33_posts_118575.json; the
    plane alone leaves them free, and pinning them moves the ground map by under 0.2 m) - the head's
    ray passes over any claimed ground position at a definite height. Claim the right position and
    that height is the player's; claim one 4% too far and it is
        h - 0.04 (H - h)  =  about 0.23 m short under a 7.5 m camera.
    Nobody's calibration is used but the camera's own; the feed and the map are both only CLAIMS
    about where the feet are.

WHAT IT PRINTS  per range band from the camera, pooled over the seven same-install clips, on the MOT
    boxes of frames with f % 20 == 10: the median implied height under the MAP's position (the same
    camera, foot ray to the ground) and under the FEED's. A DIAGNOSTIC: no map is scored or chosen.

PREDICTIONS, written before it ran
    A box is a posture, not a stature: expect a median near 1.6-1.75 m for adult players, and the
    SAME at every range for whichever side is right. If the map is right the feed's reads ~0.1 m low
    near and ~0.25 m low far. If the feed is right the map's reads that much HIGH, rising with range.
    One pixel of box looseness is worth about 0.05 m at 80 m and cannot make 0.25.

    .venv/bin/python model/validation/n33_heights.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
SEQS = ("117093", "118575", "118576", "118577", "118578", "128057", "128058")
BANDS = ((15, 30), (30, 45), (45, 60), (60, 75), (75, 100))
OUT = os.path.join(ROOT, "model/validation/out/stc2025/n33_heights.json")


def rays(q, uv):
    """World-frame unit-free ray directions and the camera centre, equirect10."""
    import camera_models as CM  # noqa: PLC0415
    R = CM._rot(q[4:7])
    a, e = (uv[:, 0] - q[2]) / q[0], (uv[:, 1] - q[3]) / q[1]
    return np.column_stack([np.sin(a), np.tan(e), np.cos(a)]) @ R, -R.T @ q[7:10]


def height_over(q, head_uv, ground_xy):
    """Height at which the head's ray passes over `ground_xy`."""
    d, C = rays(q, head_uv)
    lam = np.linalg.norm(ground_xy - C[:2], axis=1) / np.linalg.norm(d[:, :2], axis=1)
    z = C[2] + lam * d[:, 2]
    return z * np.sign(C[2])                     # the camera is above the pitch whichever way the fit's z points


def main():
    import dist_eval as D  # noqa: PLC0415
    import n30_operator_maps as N  # noqa: PLC0415
    import n31_linereg as R  # noqa: PLC0415
    acc = {b: {"map": [], "feed": []} for b in BANDS}
    cams = {}
    for q in SEQS:
        P, _n, obj, img = R.clicks_of(q)
        bgr = R._median_cached(q)
        cam, rep = R.register(R.ridge_map(bgr), R.fit_clicks(obj, img, N.CM.T._wh(q), P), P, obj, img,
                              grass=R.grass_map(bgr), square=1.0)
        d, C = rays(cam, np.zeros((1, 2)))
        cams[q] = {"camera": cam.tolist(), "height_m": float(abs(C[2])), "f": float(abs(cam[0])), "refused": rep["refused"]}
        boxes = {}
        for line in open(f"{os.path.expanduser('~')}/Downloads/st2/mot/{q}.txt"):
            p = line.split(",")
            boxes[(int(p[0]), int(p[1]))] = tuple(float(v) for v in p[2:6])
        for g, t in D.stc_gt(q, D.STC_EVERY).items():
            for fr, xy in zip(t["f"].tolist(), t["xy"]):
                b = boxes.get((fr, int(g))) if fr % 20 == 10 else None
                if b is None:
                    continue
                x, y, w, h = b
                foot, head = np.array([[x + w / 2, y + h]]), np.array([[x + w / 2, y]])
                pm = R._inv(cam, foot)
                if not np.isfinite(pm).all():
                    continue
                rng = float(np.linalg.norm(pm[0] - C[:2]))
                for lo, hi in BANDS:
                    if lo <= rng < hi:
                        acc[(lo, hi)]["map"].append(float(height_over(cam, head, pm)[0]))
                        acc[(lo, hi)]["feed"].append(float(height_over(cam, head, np.array([xy], float))[0]))
        print(q, "camera height", round(cams[q]["height_m"], 2), "m, f", round(cams[q]["f"], 1), "px/rad", flush=True)
    res = {"cameras": cams, "bands": {}}
    print("\nrange from camera     boxes    implied height, MAP's position    FEED's position    feed - map")
    for b in BANDS:
        m, f = np.array(acc[b]["map"]), np.array(acc[b]["feed"])
        res["bands"][f"{b[0]}-{b[1]}"] = {"n": int(len(m)), "map_median_m": float(np.median(m)), "feed_median_m": float(np.median(f)),
                                          "map_iqr": [float(np.percentile(m, 25)), float(np.percentile(m, 75))]}
        print(f"  {b[0]:3d}-{b[1]:3d} m          {len(m):6d}        {np.median(m):5.2f} m                    {np.median(f):5.2f} m           {np.median(f) - np.median(m):+5.2f}")
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
