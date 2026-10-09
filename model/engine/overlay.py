"""Draw the tracker's output back onto the footage it came from.

There is no annotated video anywhere in this pipeline: track.py writes JSON and
nothing renders it. That is fine for the product - the app plays the original
footage and draws its own overlays in the browser - but it leaves nobody able to
LOOK at what the tracker did over a real half, which is the only way some kinds
of failure become obvious.

What it draws, and why each choice:

  - An ELLIPSE on the ground under the feet, not a box. track.py stores
    `x=(x1+x2)/2, y=y2` - the bottom-centre of the box, because the homography
    maps the ground plane and a centroid floats above it. The ellipse is drawn
    at that point and sized from the box width `w`, so it grows with the player
    as the camera does: a fixed-radius marker looks correct near and absurd far.
    Tracks written before `w`/`h` existed fall back to a nominal box, which is
    the only thing that can be done without inventing one.

  - The TRACK ID in a badge above the head, filled in the TEAM colour. Team
    colour on the marker and the id in the text: a player who changes number
    across a passage of play is an id switch, and the number says so as plainly
    as a hue did - the first full-half run produced 3,973 tracks for ~25 bodies,
    and the distinct-id counter in the HUD is what shows you that.

  - The ball as a triangle above it, in one colour that is neither team's.
    `track.py --ball` writes them; without it there is no ball key and nothing
    is drawn.

  - Optionally the PITCH MODEL (`--lines`), projected back through the
    calibration. This is the check with nothing between you and it: the drawn
    halfway line either lands on the painted one or the geometry is wrong. It
    earned its keep immediately, and twice. On the DFL job it showed that
    `calibration_f000.json` is simply WRONG - its halfway line sits 13 degrees
    and ~65 px off the painted one, on every frame including the one it was
    solved on, because it was carried back from frame 749 by registration and
    the carry-back failed. Nothing else in the pipeline noticed: it reports an
    rms of 0.73 m, computed against its own carried-back points.
    `calibration_f749.json`, solved on real paint, lands within 0.5-4.2 px on
    every frame checked.

    Then, with the right calibration, it showed the pan. `--pan` registers each
    frame to the frame the calibration was solved on and prints what the camera
    movement costs in metres:

        1 s   0.07 m      8 s   31.78 m
        2 s   1.71 m     20 s   16.83 m
        4 s  12.51 m     30 s   40.55 m

    Read that as the error in every position on this footage, not as a drawing
    artefact - this clip is unusable for measurement beyond about two seconds
    either side of its calibration frame. The DRAWING is what --pan corrects;
    tracks.json is still on the one fixed homography.

  - Optionally a MINIMAP (`--map`): the same foot points pushed through
    `cal.to_pitch` and drawn on a pitch from above. The markers above are in
    pixels and look correct whatever the calibration does; this panel is where a
    bad homography becomes visible, because it puts players in the stands.

  - Frames go through the CALIBRATION'S OWN undistort_frame, not a remap
    rebuilt here. The detections are in remapped pitch-band space (11878x870
    for this match); anything that reproduces that transform by hand will drift
    from it, and a drifted overlay is worse than none because it looks right.

Usage:
    python model/engine/overlay.py VIDEO TRACKS CALIB OUT [--start S] [--dur S]
                                [--width PX] [--fps N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calcheck  # noqa: E402
import pancheck  # noqa: E402
import calibrators  # noqa: E402

# Team ring colours, BGR. Officials get their own so a yellow ring is never
# mistaken for a third team.
TEAM_BGR = {
    "home": (235, 235, 235),
    "away": (246, 109, 47),
    "official": (60, 200, 246),
    None: (140, 140, 140),
}
# "unknown" is what track.py writes for a track whose kit colour never resolved
# - it is the common case on a bad crop, not an edge case, and a KeyError here
# would kill the render at whatever frame it first appears.
UNKNOWN_BGR = TEAM_BGR[None]
BALL_BGR = (60, 220, 250)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# What a box is assumed to be when the tracks predate w/h. Roughly a player at
# mid-distance in the 118575 remapped band.
NOMINAL_WH = (26.0, 62.0)


def marker(img, x: int, y: int, w: float, h: float, label: str,
           colour: tuple[int, int, int]) -> None:
    """One player: an ellipse on the ground at (x, y) and a badge above the head.

    The ellipse is open at the top (-45..235 deg, the broadcast convention) so a
    player standing in it is not sliced in half by their own marker. Everything
    is drawn twice, black first: over grass, white kit and a painted line all in
    the same frame, a single-stroke marker disappears against one of them.
    """
    ax = max(6, int(round(w / 2)))
    ay = max(3, int(round(ax * 0.35)))
    cv2.ellipse(img, (x, y), (ax, ay), 0, -45, 235, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (ax, ay), 0, -45, 235, colour, 2, cv2.LINE_AA)
    if not label:
        return                      # ring only - see render()'s `label` argument

    fs = min(0.9, max(0.35, ax / 26))
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, 1)
    pad = max(2, int(round(th * 0.4)))
    bw, bh = tw + 2 * pad, th + 2 * pad
    # Above the HEAD, so h is doing real work: at the far touchline a player is
    # a third the height of one near, and a fixed offset would leave the badge
    # floating in the stands or printed across their chest.
    x0, y0 = x - bw // 2, int(round(y - h)) - bh - 3
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), colour, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), 1, cv2.LINE_AA)
    # Dark text on a light kit colour, light on a dark one - "home" is nearly
    # white, so one fixed text colour is unreadable for half the pitch.
    ink = (0, 0, 0) if sum(colour) > 460 else (255, 255, 255)
    cv2.putText(img, label, (x0 + pad, y0 + pad + th), FONT, fs, ink, 1, cv2.LINE_AA)


def ball_marker(img, x: int, y: int, size: int = 11) -> None:
    """A triangle pointing down at the ball, sitting just above it.

    Above, not on: the ball is 9 px across on this footage and any marker drawn
    over it hides the thing you are checking.
    """
    pts = np.array([[x - size, y - 2 * size], [x + size, y - 2 * size],
                    [x, y - size // 2]], np.int32)
    cv2.fillConvexPoly(img, pts, BALL_BGR, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def minimap(cal, rows: list[dict], ball: list[dict], w: int) -> np.ndarray:
    """The pitch, seen from above, with everyone on it. -> a BGR panel `w` wide.

    THIS IS THE MEASUREMENT, NOT A DECORATION. Every metre this product reports
    is computed from `cal.to_pitch(foot_point)`, and until now nothing rendered
    that projection - the overlay drew markers in PIXELS, which look right
    whatever the calibration does. A bad homography puts players in the stands
    here while the pixel markers still sit neatly on their feet, so this panel
    is the one place a calibration failure is visible rather than merely
    downstream of everything.

    Same geometry and the same convention as the app's own map
    (app/frontend/src/components/pitch.tsx, whose `flipY` is this `W - y`): pitch y
    is measured FROM THE FAR TOUCHLINE, so y=0 is far and y=W is near, and the
    flip puts the NEAR touchline at the TOP of the panel.

    That means the map is mirrored top-to-bottom against the camera view above
    it - the benches you see along the bottom of the frame appear along the top
    of the map. It is deliberate, because this has to agree with the map in the
    app; do not "fix" it. Verified on the DFL clip rather than taken from the
    comment: the four touchline staff project to y = 68.9-76.2 m and the far-side
    players to y = 10.8-20.2 m.
    """
    L, W = cal.pitch.length, cal.pitch.width
    pad = 3.0
    k = w / (L + 2 * pad)                        # pixels per metre
    h = int(round((W + 2 * pad) * k))
    img = np.full((h, w, 3), (30, 62, 34), np.uint8)   # darker than any grass in shot

    def P(x, y):
        return int(round((x + pad) * k)), int(round((W - y + pad) * k))

    line = (210, 210, 210)
    cv2.rectangle(img, P(0, 0), P(L, W), line, 1, cv2.LINE_AA)
    cv2.line(img, P(L / 2, 0), P(L / 2, W), line, 1, cv2.LINE_AA)
    cv2.circle(img, P(L / 2, W / 2), int(round(cal.pitch.centre_circle_r * k)), line, 1, cv2.LINE_AA)
    pd, pw = cal.pitch.penalty_depth, cal.pitch.penalty_width
    gd, gw = cal.pitch.goal_box_depth, cal.pitch.goal_box_width
    for x0, x1 in ((0, pd), (L - pd, L)):
        cv2.rectangle(img, P(x0, W / 2 - pw / 2), P(x1, W / 2 + pw / 2), line, 1, cv2.LINE_AA)
    for x0, x1 in ((0, gd), (L - gd, L)):
        cv2.rectangle(img, P(x0, W / 2 - gw / 2), P(x1, W / 2 + gw / 2), line, 1, cv2.LINE_AA)

    if rows:
        # ONE call, not one per row: to_pitch on a panorama rig walks a remap
        # table, and per-detection calls made this the slowest thing on screen.
        pts = cal.to_pitch(np.array([[r["x"], r["y"]] for r in rows], float))
        for r, (x, y) in zip(rows, pts):
            # Off the map is DRAWN AT THE EDGE, never dropped: a player the
            # calibration puts in the car park is the failure this panel exists
            # to show, and silently skipping him hides it.
            u, v = P(min(max(x, -pad), L + pad), min(max(y, -pad), W + pad))
            colour = TEAM_BGR.get(r.get("team"), UNKNOWN_BGR)
            if r.get("team") == "official":
                # A square, not a disc. Officials ARE on this map deliberately -
                # they are the population every team total excludes, so seeing
                # where they stand is how a mislabelled player gets caught - but
                # at 3 px a hue alone does not separate them from a third team.
                cv2.rectangle(img, (u - 4, v - 4), (u + 4, v + 4), (0, 0, 0), -1, cv2.LINE_AA)
                cv2.rectangle(img, (u - 3, v - 3), (u + 3, v + 3), colour, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (u, v), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (u, v), 3, colour, -1, cv2.LINE_AA)
    if ball:
        # The ball's pitch position is only meaningful while it is ON the
        # ground (ball.Ball), so it is drawn hollow - a marker you are meant to
        # trust less than the players around it.
        b = cal.to_pitch(np.array([[ball[0]["x"], ball[0]["y"]]], float))[0]
        u, v = P(min(max(b[0], -pad), L + pad), min(max(b[1], -pad), W + pad))
        cv2.circle(img, (u, v), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 3, BALL_BGR, 1, cv2.LINE_AA)
    return img


def _solved_on(calib: str) -> int | None:
    """The video frame a calibration was solved on, if it says.

    Written by the line-labelling path as `source.frame`. Absent on a
    hand-clicked calibration, where the caller has to say - and where guessing
    would be worse than asking, because the whole point of the reference frame
    is that the geometry is known to be right on it.
    """
    try:
        with open(calib) as fh:
            return json.load(fh).get("source", {}).get("frame")
    except Exception:                        # noqa: BLE001
        return None


def pitch_model(P) -> list[list[tuple[float, float]]]:
    """The painted markings, in CORNER-ORIGIN metres (x 0..length, y 0..width).

    Split out of `pitch_lines` so a projection that is not `cal.H` - the
    SoccerTrack chain in model/validation/st2_overlay.py is one - draws the same
    pitch. Two copies of these dimensions is two chances to mistake the goal
    area for the penalty box, which is exactly the error calcheck.py exists to
    catch and which looks self-consistent when you make it.
    """
    L, W = P.length, P.width
    cy = W / 2
    box = lambda x0, x1, h: [(x0, cy - h / 2), (x1, cy - h / 2),      # noqa: E731
                             (x1, cy + h / 2), (x0, cy + h / 2), (x0, cy - h / 2)]
    circle = [(L / 2 + P.centre_circle_r * np.cos(t), cy + P.centre_circle_r * np.sin(t))
              for t in np.linspace(0, 2 * np.pi, 64)]
    return [
        [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],          # touchlines and goal lines
        [(L / 2, 0), (L / 2, W)],                          # halfway
        circle,
        box(0, P.penalty_depth, P.penalty_width),
        box(L - P.penalty_depth, L, P.penalty_width),
        box(0, P.goal_box_depth, P.goal_box_width),
        box(L - P.goal_box_depth, L, P.goal_box_width),
    ]


def densify(poly, step_m=0.5):
    """Subdivide a polyline in METRES before projecting it.

    `pitch_model` gives a touchline as its two END POINTS, which is all a
    straight-line rig needs. This panorama is STITCHED, so the image of a
    straight pitch line is a curve, and a segment drawn between two correctly
    projected corners cuts a chord across it - 120 px above the painted
    touchline at halfway on 128057, converging to zero at both ends. That reads
    exactly like a bad calibration and is not one: the halfway line's own
    endpoint, projected on its own, lands on the painted touchline.

    The centre circle already came with 64 vertices, which is why it was the one
    marking that looked right. This gives every line the same treatment.
    """
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(2, int(np.ceil(d / step_m)) + 1)
        out.extend(zip(np.linspace(a[0], b[0], n)[:-1],
                       np.linspace(a[1], b[1], n)[:-1]))
    out.append(tuple(poly[-1]))
    return out


def pitch_lines(cal, wh: tuple[int, int]) -> list[np.ndarray]:
    """The pitch model pushed BACK into the image. -> polylines in pixels.

    This is the calibration check with nothing between you and it. Everything
    else the overlay draws is downstream of the homography and will look
    plausible whatever it does; a drawn halfway line either sits on the painted
    halfway line or it does not, and if it does not, no number in the report is
    worth anything.

    `to_pitch` runs image -> metres, so this runs its INVERSE, which is where
    the two traps live:

      - BEHIND THE CAMERA. A homography happily maps a point behind the lens to
        a perfectly ordinary-looking pixel. The far goal on a touchline rig is
        routinely such a point, so the homogeneous w is checked and those
        vertices are dropped; without that you get a confident line across the
        crowd.
      - A PANORAMIC calibration fits a strip homography per band and `cal.H` is
        the global fit only (calibrators.PanoramicCalibrator). These lines are
        therefore APPROXIMATE on that rig - correct enough to see a gross
        error, not to measure one. `to_pitch` itself uses the strips and is
        unaffected.
      - A STRAIGHT LINE IS NOT STRAIGHT IN THE IMAGE, and `pitch_model` gives a
        touchline as its two end points. Drawing a segment between two
        correctly projected corners cuts a chord across the curve - measured at
        120 px above the painted touchline at halfway on a stitched panorama,
        converging to zero at both ends, which is indistinguishable from a bad
        homography by eye. `densify` subdivides in METRES first, so the
        curvature is sampled rather than assumed away.
    """
    P = cal.pitch
    L, W = P.length, P.width
    model = pitch_model(P)

    Hi = np.linalg.inv(np.asarray(cal.H, np.float64))
    # NORMALISE THE SIGN FIRST. A homography is defined only up to scale, so H
    # and -H are the same projection with every homogeneous w negated, and both
    # are produced in practice - the two calibrations in this one job differ.
    # The w>0 test below is therefore meaningless until the sign is pinned, and
    # getting it wrong does not misplace a line, it silently discards ALL of
    # them: this drew a full pitch for one calibration and nothing whatsoever
    # for the other. The centre spot is in front of the camera for any football
    # calibration that means anything, so it is what fixes the sign.
    if (np.array([P.length / 2, W / 2, 1.0]) @ Hi.T)[2] < 0:
        Hi = -Hi
    w, h = wh
    # A camera model (SphericalCamera) projects through itself; its `H` is only a
    # flat fit to it and misses a stitched panorama's near touchline by ~30 px on
    # 118575, which reads as a bad calibration and is not one.
    to_image = getattr(cal, "to_image", None)
    out = []
    for poly in model:
        pts = np.asarray(densify(poly), np.float64)
        if to_image is not None:
            px = np.asarray(to_image(pts), np.float64).reshape(-1, 2)
            front = np.isfinite(px).all(1)            # the model returns NaN behind the lens
        else:
            q = np.c_[pts, np.ones(len(pts))] @ Hi.T
            px = calcheck._project(Hi, pts)
            front = q[:, 2] > 1e-9
        # w <= 0 is behind the camera; the generous pixel bound then drops the
        # ones that are merely far outside the frame, while keeping vertices
        # just off-frame so the segment crossing the edge still gets drawn.
        ok = front & np.isfinite(px).all(1) & \
             (np.abs(px[:, 0]) < 20 * w) & (np.abs(px[:, 1]) < 20 * h)
        run: list = []
        for keep, xy in zip(ok, px):
            if keep:
                run.append(xy)
            elif len(run) > 1:
                out.append(np.array(run))
                run = []
            else:
                run = []
        if len(run) > 1:
            out.append(np.array(run))
    return out


#: Metres a ball may cover between two frames before the segment joining them
#: is a lie rather than a path. A hard strike is ~35 m/s, so 1.4 m/frame at
#: 25 fps; 2.5 leaves room for a fast clearance, a high frame gap and the
#: calibration's own error, and still refuses the 40 m single-frame jumps the
#: picker produces when it flickers between candidates.
MAX_BALL_M_PER_FRAME = 2.5


def ball_trail(img, seen: list[tuple[int, np.ndarray, np.ndarray | None]], f: int,
               span: int, max_gap: int = 3) -> None:
    """The ball's recent path, fading out behind it.

    BROKEN ACROSS GAPS, never bridged. The picker reports nothing on the frames
    it is not sure about (ball.BallPicker), and joining across those frames
    would draw a straight line through an occlusion the ball did not travel in a
    straight line through - inventing a path is how a gap becomes fake distance.
    A gap longer than `max_gap` frames starts a new stroke, so the breaks in the
    trail ARE the frames the picker declined to answer, and are worth seeing.

    ALSO BROKEN WHERE THE IMPLIED SPEED IS IMPOSSIBLE, when the metres are
    known. The picker is memoryless by measurement (ball.BallPicker: every
    temporal term made it worse), so it will happily answer one side of the
    pitch on one frame and the other side on the next. That is the right
    behaviour for a picker and the wrong thing to JOIN UP: a segment implying
    40 m in a frame draws a pass nobody made. The physics that does not help
    choose the ball does help refuse to draw one.
    """
    prev_f = prev_p = prev_m = None
    for g, p, m in seen:
        fast = (prev_m is not None and m is not None and
                float(np.hypot(*(m - prev_m))) > MAX_BALL_M_PER_FRAME * max(g - prev_f, 1))
        if prev_p is not None and 0 < g - prev_f <= max_gap and not fast:
            age = (f - g) / max(span, 1)                  # 0 newest, 1 oldest
            k = max(0.0, 1.0 - age)
            cv2.line(img, tuple(np.int32(prev_p)), tuple(np.int32(p)),
                     tuple(int(c * k) for c in BALL_BGR),
                     max(1, int(round(1 + 2 * k))), cv2.LINE_AA)
        prev_f, prev_p, prev_m = g, p, m


def _interpolate(rows: list[dict], fill: int) -> list[dict]:
    """Carry each track across the frames the STRIDE skipped.

    A strided run emits rows on every Nth frame carrying their ORIGINAL frame
    numbers - track.py's docstring is emphatic that renumbering would break four
    other readers - so at `--stride 2` HALF the frames of a render have no
    detections at all and every marker blinks at 12.5 Hz. That flicker is the
    render's, not the tracker's.

    Linear across a gap of at most `fill` missing frames. A longer gap is left
    empty on purpose: it is an occlusion or a lost track, and drawing through it
    invents a path exactly the way ball_trail refuses to.

    These rows are DRAWN AND NEVER MEASURED. Nothing downstream reads them and
    they carry `_fill` so a dumped frame can be told apart from a detected one.
    """
    by_id: dict[int, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    out = list(rows)
    for trk in by_id.values():
        trk.sort(key=lambda d: d["f"])
        for a, b in zip(trk, trk[1:]):
            gap = b["f"] - a["f"]
            if not 1 < gap <= fill + 1:
                continue
            for f in range(a["f"] + 1, b["f"]):
                k = (f - a["f"]) / gap
                row = dict(a)
                row["f"], row["_fill"] = f, True
                for key in ("x", "y", "w", "h"):
                    if key in a and key in b:
                        row[key] = a[key] + k * (b[key] - a[key])
                out.append(row)
    return out


def by_frame(tracks_path: str, f0: int, f1: int,
             fill: int = 0) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """(players, ball) for [f0, f1), each grouped by frame.

    Streams the filter rather than building the whole index: the full-half file
    is 158 MB / 1.55M detections and a 90 s clip wants 2,250 frames of it.
    `ball` is empty for tracks written without `track.py --ball`.

    `fill` closes the stride gaps, and is off by default so that anything
    reading raw rows out of here still gets raw rows.
    """
    with open(tracks_path) as fh:
        data = json.load(fh)
    rows = [r for r in data.get("detections", ()) if f0 <= r["f"] < f1]
    balls: dict[int, list[dict]] = {}
    for r in data.get("ball", ()):
        if f0 <= r["f"] < f1:
            balls.setdefault(r["f"], []).append(r)
    if fill:
        rows = _interpolate(rows, fill)
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["f"], []).append(r)
    return out, balls


def render(
    video: str,
    tracks: str,
    calib: str,
    out: str,
    start_s: float = 0.0,
    dur_s: float | None = None,
    width: int = 2400,
    fps_out: float | None = None,
    map_frac: float = 0.0,
    lines: bool = False,
    pan: bool = False,
    ref: int | None = None,
    trail: int = 0,
    label: str = "id",
    fill: int = 0,
) -> dict:
    cal = calibrators.load(calib)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    f0 = int(round(start_s * fps))
    f1 = total if dur_s is None else min(total, f0 + int(round(dur_s * fps)))
    if f0 >= f1:
        raise SystemExit(f"empty range: frames {f0}..{f1} of {total}")

    dets, balls = by_frame(tracks, f0, f1, fill)

    # Probe the flattened size on one blank frame — same trick track.py uses to
    # fail fast on missing tables rather than 70,000 frames in.
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fh, fw = cal.undistort_frame(np.zeros((h0, w0, 3), np.uint8)).shape[:2]
    scale = width / fw
    ow, oh = width, max(2, int(round(fh * scale)) // 2 * 2)

    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), fps_out or fps, (ow, oh))
    if not vw.isOpened():  # avc1 is absent on some OpenCV builds
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps_out or fps, (ow, oh))
    if not vw.isOpened():
        raise SystemExit(f"cannot open a writer for {out}")

    # Computed once in REFERENCE pixels - the frame the calibration was solved
    # on. Without --pan they are drawn as-is, which is correct only for a
    # bolted-down camera; with it they are carried onto each frame.
    model = pitch_lines(cal, (fw, fh)) if lines else []
    sift = pancheck._sift() if pan else None
    probes = pancheck.pitch_probes(cal, (fw, fh)) if pan else None
    drift = float("nan")

    # THE REFERENCE IS THE FRAME THE CALIBRATION WAS SOLVED ON, and the file
    # records which one that is. Registering to the first RENDERED frame instead
    # is the mistake that hides a bad calibration: it makes the lines
    # self-consistent with whatever geometry the clip starts on, so they follow
    # the camera beautifully while sitting a dozen pixels off the paint, and the
    # drift number then reads near zero at the start of every clip whatever the
    # calibration does. On this DFL job the two frames are 30 s apart and the
    # difference is 13 degrees of halfway line.
    ref_gray = None
    hom: dict[int, np.ndarray] = {}
    if pan and (model or trail):
        want = ref if ref is not None else _solved_on(calib)
        if want is None:
            want = f0
        probe = cv2.VideoCapture(video)
        probe.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, rf = probe.read()
        probe.release()
        if not ok:
            raise SystemExit(f"cannot read reference frame {want} of {video}")
        ref_gray = cv2.cvtColor(cal.undistort_frame(rf), cv2.COLOR_BGR2GRAY)
        print(f"  registering to frame {want}, where {os.path.basename(calib)} was solved",
              flush=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    seen_tracks: set[int] = set()
    drawn = 0
    t0 = time.perf_counter()

    for f in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        flat = cal.undistort_frame(frame)
        img = cv2.resize(flat, (ow, oh), interpolation=cv2.INTER_AREA)

        # REGISTER to the reference frame. The calibration is one 3x3 solved on
        # one frame; the moment an operator pans, it describes a camera that no
        # longer exists. pancheck does this on the STATIC scene - stands,
        # hoardings, paint - with the players outliered away, so nothing here is
        # fitted to the calibration it is checking.
        grid = [np.round(q * scale).astype(np.int32) for q in model]
        if pan and ref_gray is not None:
            H_now, _ = pancheck.register(ref_gray, cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), sift)
            hom[f] = H_now
            if H_now is not None:
                grid = [np.round(pancheck._warp(H_now, q) * scale).astype(np.int32)
                        for q in model]
                # ...and say IN METRES how wrong the geometry now is. Lines that
                # follow the camera while the numbers do not is exactly the
                # "looks right, is wrong" this repo keeps being bitten by, so
                # the cost of the drift is stated on the frame that has it.
                drift = float(np.median(pancheck.drift_m(cal, H_now, probes)))

        # UNDER the players, so a line never hides the thing being checked.
        for poly in grid:
            cv2.polylines(img, [poly], False, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(img, [poly], False, (80, 240, 255), 1, cv2.LINE_AA)

        for d in dets.get(f, ()):
            x, y = int(round(d["x"] * scale)), int(round(d["y"] * scale))
            tid = d["id"]
            seen_tracks.add(tid)
            lab = "" if label == "none" else str(d.get("jersey") or tid)
            marker(img, x, y,
                   d.get("w", NOMINAL_WH[0]) * scale, d.get("h", NOMINAL_WH[1]) * scale,
                   lab, TEAM_BGR.get(d.get("team"), UNKNOWN_BGR))
            drawn += 1

        if trail:
            # Each past position is carried into THIS frame's pixels when --pan
            # is on. Without that the trail is drawn where the ball was on the
            # sensor, not where it was on the pitch, and on a panning camera
            # those differ by ~7 px a frame - a second of trail would lag the
            # ball by half its own length and read as a tracking error.
            pts = []
            for g in range(max(f0, f - trail), f + 1):
                for b in balls.get(g, ()):
                    q = np.array([[b["x"], b["y"]]], np.float32)
                    if pan and g in hom and f in hom and hom[g] is not None \
                            and hom[f] is not None:
                        q = pancheck._warp(hom[f] @ np.linalg.inv(hom[g]), q)
                    # Metres from the UNWARPED position: the speed test compares
                    # two consecutive frames, where the camera has moved a few
                    # pixels, and that is nothing beside the jumps it is there
                    # to catch.
                    m = cal.to_pitch(np.array([[b["x"], b["y"]]], float))[0]
                    pts.append((g, q[0] * scale, m))
            ball_trail(img, pts, f, trail)

        for b in balls.get(f, ()):
            ball_marker(img, int(round(b["x"] * scale)), int(round(b["y"] * scale)))

        if map_frac:
            # Bottom-left, over the near touchline: on a fixed wide rig the
            # bottom corners are crowd or advertising, and the near touchline is
            # the one part of the frame with nothing to occlude.
            m = minimap(cal, dets.get(f, []), balls.get(f, []), int(ow * map_frac))
            mh, mw = m.shape[:2]
            y0 = oh - mh - 12
            # Blended, not pasted: at 15% of a 1920 frame the panel is 288 px
            # wide and would hide a player standing behind it.
            roi = img[y0:y0 + mh, 12:12 + mw]
            img[y0:y0 + mh, 12:12 + mw] = cv2.addWeighted(m, 0.9, roi, 0.1, 0)
            cv2.rectangle(img, (12, y0), (12 + mw, y0 + mh), (0, 0, 0), 1, cv2.LINE_AA)

        # Clock and a live distinct-track count: the count climbing during one
        # continuous passage of play IS the fragmentation, stated in a number
        # while you watch it happen.
        hud = f"{f / fps:7.1f}s   frame {f}   {len(dets.get(f, ())):2d} on pitch   {len(seen_tracks)} distinct ids so far"
        if pan and np.isfinite(drift):
            hud += f"   camera drift {drift:5.2f} m"
        cv2.putText(img, hud, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, hud, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)

        if f > f0 and (f - f0) % 250 == 0:
            done = f - f0
            print(f"  {done}/{f1 - f0} frames  {done / (time.perf_counter() - t0):.1f} fps", flush=True)

    cap.release()
    vw.release()
    res = {
        "frames": f1 - f0,
        "detections_drawn": drawn,
        "distinct_tracks": len(seen_tracks),
        "out": out,
        "size": [ow, oh],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(res), flush=True)
    return res


def _selfcheck() -> None:
    """The marker must scale with the box and stay inside the frame.

    A marker sized off a constant is the failure this file exists to avoid, so
    the check is that a wide box really does draw a wider ellipse - and that a
    player at the top of the frame does not push their badge off it, which is
    what a plain `y - h` would do at the far touchline.
    """
    def painted(w, h, y=200):
        img = np.zeros((240, 320, 3), np.uint8)
        marker(img, 160, y, w, h, "7", (240, 120, 40))
        return int((img.any(axis=2)).sum())

    assert painted(60, 140) > painted(20, 48), "ellipse does not scale with the box"

    # Badge clipped off the top: OpenCV silently drops it, so the check is that
    # SOMETHING is still drawn - the ellipse - rather than that nothing crashed.
    img = np.zeros((240, 320, 3), np.uint8)
    marker(img, 160, 12, 26, 62, "7", (240, 120, 40))
    assert img.any(), "nothing drawn for a player at the top edge"

    # The ball sits ABOVE its point, never on it.
    img = np.zeros((240, 320, 3), np.uint8)
    ball_marker(img, 160, 120)
    ys = np.nonzero(img.any(axis=2))[0]
    assert ys.max() <= 120, (ys.min(), ys.max())

    # Text ink flips with the fill, or half the pitch is unreadable.
    light = np.zeros((240, 320, 3), np.uint8)
    marker(light, 160, 200, 26, 62, "7", (235, 235, 235))
    assert light.any()

    # The minimap's Y IS FLIPPED, the same way app/frontend/src/components/pitch.tsx
    # flips it. Pitch y is measured from the FAR touchline, so y=W is the NEAR
    # one and must land at the TOP of the panel. Getting this backwards mirrors
    # the two teams and looks entirely plausible, which is why it is asserted
    # rather than eyeballed.
    class _Cal:
        H = np.array([[2.0, 0.1, -30.0], [0.05, 2.0, -20.0], [0.0004, 0.0009, 1.0]])
        pitch = type("P", (), {"length": 105.0, "width": 68.0, "penalty_depth": 16.5,
                               "penalty_width": 40.32, "goal_box_depth": 5.5,
                               "goal_box_width": 18.32, "centre_circle_r": 9.15})()

        @staticmethod
        def to_pitch(pts):        # pixels are already metres in this stub
            return np.asarray(pts, float)

    nearside = minimap(_Cal(), [{"x": 52.5, "y": 68.0, "team": "home"}], [], 300)
    farside = minimap(_Cal(), [{"x": 52.5, "y": 0.0, "team": "home"}], [], 300)
    empty = minimap(_Cal(), [], [], 300)
    row_near = np.nonzero((nearside != empty).any(axis=2))[0].mean()
    row_far = np.nonzero((farside != empty).any(axis=2))[0].mean()
    assert row_near < row_far, f"pitch y is not flipped: near={row_near} far={row_far}"

    # THE SIGN OF H IS NOT LOAD-BEARING. H and -H are the same projection, and
    # both turn up in real calibration files; a pitch_lines that only handles
    # one of them draws nothing at all for the other, which reads as "--lines
    # is broken" rather than as a sign bug. Both must give the same picture.
    class _Signed(_Cal):
        H = -np.array([[2.0, 0.1, -30.0], [0.05, 2.0, -20.0], [0.0004, 0.0009, 1.0]])

    class _Plain(_Cal):
        H = -_Signed.H

    a = pitch_lines(_Plain(), (1920, 1080))
    b = pitch_lines(_Signed(), (1920, 1080))
    assert len(a) == len(b) == 7, (len(a), len(b))
    assert all(np.allclose(x, y) for x, y in zip(a, b)), "H and -H drew different pitches"

    # THE TRAIL MUST BREAK WHERE THE PICKER WENT QUIET. Bridging a gap draws a
    # straight line through an occlusion, which is a path the ball did not take
    # and distance nobody measured.
    img = np.zeros((200, 400, 3), np.uint8)
    run = [(i, np.array([50.0 + 10 * i, 100.0]), None) for i in range(5)]
    ball_trail(img, run, 4, 25)
    joined = int((img.any(axis=2)).sum())
    assert joined > 0, "a continuous run drew nothing"

    img2 = np.zeros((200, 400, 3), np.uint8)
    gapped = [(0, np.array([50.0, 100.0]), None), (20, np.array([350.0, 100.0]), None)]
    ball_trail(img2, gapped, 20, 25)
    assert not img2.any(), "the trail bridged a 20-frame gap"

    # ...and it fades: the segment nearest `f` must be brighter than the oldest.
    img3 = np.zeros((200, 400, 3), np.uint8)
    ball_trail(img3, [(i, np.array([50.0 + 10 * i, 100.0]), None) for i in range(20)], 19, 20)
    # Sampled INSIDE the strokes, not at the lit extremes: the outermost lit
    # columns are anti-aliasing fringe a pixel past the line's end, and they are
    # dim on every segment whatever its age.
    cols = img3[:, :, 2].max(axis=0)
    old, new = int(cols[70:90].max()), int(cols[200:230].max())
    assert new > old, f"the trail does not fade with age: old={old} new={new}"

    # AN IMPOSSIBLE SPEED IS NOT DRAWN. Consecutive frames, so no gap to catch
    # it - only the metres say this cannot have happened, and a picker with no
    # memory produces exactly this when it flickers between candidates.
    img4 = np.zeros((200, 400, 3), np.uint8)
    ball_trail(img4, [(0, np.array([50.0, 100.0]), np.array([10.0, 30.0])),
                      (1, np.array([350.0, 100.0]), np.array([70.0, 30.0]))], 1, 25)
    assert not img4.any(), "drew a 60 m single-frame pass"
    # ...and a plausible 1 m step on the same two frames IS drawn.
    img5 = np.zeros((200, 400, 3), np.uint8)
    ball_trail(img5, [(0, np.array([50.0, 100.0]), np.array([10.0, 30.0])),
                      (1, np.array([60.0, 100.0]), np.array([11.0, 30.0]))], 1, 25)
    assert img5.any(), "refused a 1 m step"

    # A point off the pitch is clamped to the edge, never dropped - it is the
    # calibration failure this panel exists to make visible.
    off = minimap(_Cal(), [{"x": -400.0, "y": 300.0, "team": "away"}], [], 300)
    assert (off != empty).any(), "an off-pitch point was silently dropped"

    print("overlay.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("tracks", nargs="?")
    ap.add_argument("calib", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--dur", type=float, default=None, help="seconds to render (default: to the end)")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--lines", action="store_true",
                    help="draw the PITCH MODEL back onto the video through the "
                         "calibration. If the drawn halfway line does not sit on the "
                         "painted one, no number in the report is worth anything")
    ap.add_argument("--pan", action="store_true",
                    help="carry the drawn lines onto each frame by registering it to "
                         "the reference frame, and show the resulting error IN METRES. "
                         "Only the DRAWING is corrected - every metre in tracks.json "
                         "is still on the fixed homography")
    ap.add_argument("--label", default="id", choices=("id", "none"),
                    help="what rides above each player: the track id, or nothing")
    ap.add_argument("--fill", type=int, default=2, metavar="FRAMES",
                    help="interpolate a track across at most this many frames the "
                         "stride skipped; 0 disables. The default closes a "
                         "--stride 2 or 3 run, which otherwise flickers at 12.5 Hz")
    ap.add_argument("--trail", type=int, default=0, metavar="FRAMES",
                    help="draw the ball's path over the last N frames, fading out. "
                         "Broken across the frames the picker declined to answer - "
                         "the gaps are real and are not bridged")
    ap.add_argument("--ref", type=int, default=None, metavar="FRAME",
                    help="frame the calibration is correct on, for --pan to register "
                         "against. Defaults to the calibration's own source.frame")
    ap.add_argument("--map", type=float, default=0.0, metavar="FRAC",
                    help="draw the pitch map inset, this fraction of the frame width "
                         "(0.22 is a good start). It plots cal.to_pitch of every "
                         "foot point - the projection every metre is computed from")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck or not a.video:
        _selfcheck()
    else:
        render(a.video, a.tracks, a.calib, a.out, a.start, a.dur, a.width, a.fps,
               a.map, a.lines, a.pan, a.ref, a.trail, a.label, a.fill)
