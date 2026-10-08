"""A video to LOOK at: every test clip with the zero-click camera's pitch drawn over the moving footage.

No number comes out of this; it is for eyes. Per clip, eight seconds:
    top      the whole panorama, the pitch as the zero-click camera (n34_autocal.json - what the product runs) puts it, RED
    bottom   the far side at the frame's own pixels (where a pixel is half a metre), the same RED lines one pixel wide, and
             in CYAN the N36 candidate (paint rule + column warp, n36_after_warp.json - NOT shipped, not confirmed)
    The lines blink off for half a second in every two so the paint under them can be seen.
A clip the calibration REFUSED is shown bare, with the reason.

    .venv/bin/python model/validation/n34_review_video.py      # -> model/validation/out/stc2025/n34_review.mp4
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
OUT_W, TOP_H, STRIP = 2560, 675, (768, 90, 360)           # strip: first column, first row, rows - native pixels
SECONDS, FPS, STEP = 8, 25, 2


def main():
    import autocal as A  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415
    ship = json.load(open(os.path.join(STC, "n34_autocal.json")))
    cand = json.load(open(os.path.join(STC, "n36_after_warp.json"))) if os.path.exists(os.path.join(STC, "n36_after_warp.json")) else {}
    raw = os.path.join(STC, "n34_review.raw.mp4")
    H = 40 + TOP_H + STRIP[2]
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, H))
    X = A.model_samples(Pitch(105.0, 68.0), 0.05)[0]
    for seq in sorted(ship):
        cap = cv2.VideoCapture(f"{ST.CLIPS}/{seq}.mp4")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        px = {}
        for name, cams in (("ship", ship), ("cand", cand)):
            c = (cams.get(seq) or {}).get("camera")
            if c is not None:
                Xc = A.model_samples(Pitch(float(cams[seq].get("length_m") or 105.0), 68.0), 0.05)[0] if name == "cand" else X
                p = np.round(A._fwd(np.array(c, float), Xc, float(W))).astype(int)
                px[name] = p[np.isfinite(p).all(1)]
        why = ship[seq].get("why") or ship[seq].get("refused")
        for i in range(SECONDS * FPS if "ship" in px else 3 * FPS):
            for _ in range(STEP):
                ok, f = cap.read()
            if not ok:
                break
            on = (i % (2 * FPS)) < int(1.5 * FPS)
            top, strip = f.copy(), f[STRIP[1]:STRIP[1] + STRIP[2], STRIP[0]:STRIP[0] + OUT_W].copy()
            if on:
                for name, colour in (("cand", (255, 255, 0)), ("ship", (0, 0, 255))):
                    if name not in px:
                        continue
                    p = px[name]
                    if name == "ship":
                        m = (p[:, 0] >= 1) & (p[:, 0] < W - 1) & (p[:, 1] >= 1) & (p[:, 1] < f.shape[0] - 1)
                        for du, dv in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):          # fat enough to survive the downscale
                            top[p[m, 1] + dv, p[m, 0] + du] = colour
                    s = p - [STRIP[0], STRIP[1]]
                    m = (s[:, 0] >= 0) & (s[:, 0] < OUT_W) & (s[:, 1] >= 0) & (s[:, 1] < STRIP[2])
                    strip[s[m, 1], s[m, 0]] = colour
            frame = np.zeros((H, OUT_W, 3), np.uint8)
            frame[40:40 + TOP_H] = cv2.resize(top, (OUT_W, TOP_H), interpolation=cv2.INTER_AREA)
            frame[40 + TOP_H:] = strip
            label = (f"{seq}   RED: zero-click camera, as the product runs it ({ship[seq].get('hits')} of {ship[seq].get('samples')} drawing points on paint, found by {'its install' if ship[seq].get('via') == 'install' else 'search'})"
                     f"   CYAN (far strip): N36 candidate, not shipped" if "ship" in px else f"{seq}   REFUSED - no camera: {why}")
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "far side, the frame's own pixels", (12, 40 + TOP_H + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(frame)
        cap.release()
        print(seq, "drawn" if "ship" in px else f"refused: {why}", flush=True)
    vw.release()
    out = os.path.join(STC, "n34_review.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)
    os.remove(raw)
    print("->", out)


if __name__ == "__main__":
    main()
