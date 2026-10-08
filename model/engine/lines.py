"""Find the pitch markings in a frame, so geometry can be re-anchored per frame.

SCOPE NOTE: written for panning footage, which is out of scope for this repository.
It stays because calibrators.py (lens bow) and pancheck.py use its paint and grass
masks on panoramic frames. The measurements below were taken on a third-party
scouting clip that is not distributed here.

WHY THIS EXISTS
    Everything upstream assumes ONE homography, solved once from four clicked
    corners and reused for the whole match. That holds for a bolted-down camera
    and breaks for everything else - a scouting feed pans, a broadcast pans and
    zooms - and it breaks SILENTLY, projecting every frame after the first
    through geometry that is no longer true.

    The markings are the fix. They are the one thing in the scene whose real
    dimensions are known exactly (pitch.py), painted in high contrast, and
    present in every frame. Re-anchoring to them each frame removes the drift a
    frame-to-frame alignment would accumulate over a half, and it is the same
    machinery that would eventually remove the four-click onboarding tax.

WHAT IT FINDS, MEASURED ON THE DFL SCOUTING FEED
    30 frames spread across a 30 s clip, default parameters: a MEDIAN OF 6
    merged lines, and at least the four a homography needs in 100% of them.
    That number is only true since the open kernel was sized to the frame (see
    segments()); with the old fixed 3x3 it was a median of 2 and 10%.

    Six is not six USABLE lines. Typically it is the two touchlines, the halfway
    line, one penalty-box edge and one or two fragments of hoarding or shadow,
    and several are mutually parallel on the pitch, which is a weaker constraint
    than the count suggests: three parallel lines are concurrent at a vanishing
    point, so they do not span a homography between them. Measured on this clip,
    the geometry visible in a MIDFIELD frame (two touchlines, halfway line, one
    box edge -> a 2x3 grid of intersections) gives a homography that is stable
    in-sample (0.66 m rms) and falls apart held out - 20.8 m leave-one-out
    median. A frame that also shows the GOAL LINE (a 3x3 grid) holds up: 0.85 m
    held out at k=6, 1.57 m leave-one-out. So the real gate is not "four lines",
    it is "four lines in two non-parallel families spanning the frame", and on a
    ball-following camera that means the frames near a goal, not the ones in
    midfield.

    Not yet solved, and not pretended otherwise: the centre circle (an ellipse
    under perspective, invisible to a line transform) and the broadcast
    watermark, which is white, on the grass, and straight. It is also why the
    published work uses a learned keypoint detector instead - a network can
    place the centre-circle vertices from partial evidence, and Hough cannot see
    an arc at all.

WHAT THIS FILE IS, AND IS NOT
    It is the DETECTION half: pixels in, line segments out, in image
    coordinates. It does not solve a homography and it does not decide which
    detected line is the halfway line - that correspondence step is the hard
    half and lives above this.

    No model, no weights, no training data. A green field with white lines on it
    is one of the few things in this pipeline that classical vision genuinely
    does better than a network, and a detector we would have to train is a
    detector we would have to annotate first.

THE ORDER OF OPERATIONS MATTERS
    Mask the grass FIRST, then look for white inside it. Doing it the other way
    finds the stands, the goal frame, the advertising boards and the white kit -
    all of which are whiter than the paint and none of which are on the ground
    plane. On the DFL feed the crowd fills the top third of the frame, so this
    is not a refinement, it is the difference between working and not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Grass in HSV. Wide, because floodlit night football, low winter sun and a
# worn January pitch are all still green, and the mask only has to separate
# pitch from crowd rather than judge the turf.
GRASS_LO = np.array([35, 50, 50], np.uint8)
GRASS_HI = np.array([85, 255, 255], np.uint8)

#: Paint is brighter and less saturated than the grass around it. Applied only
#: INSIDE the grass mask, so "bright and grey" means paint rather than a shirt.
PAINT_S_MAX = 90
PAINT_V_MIN = 130


@dataclass
class Segment:
    """One detected line segment in image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle(self) -> float:
        """Direction in degrees, folded to [0, 180) - a line has no arrowhead."""
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def grass_mask(bgr: np.ndarray) -> np.ndarray:
    """The playing surface. Largest green component, holes filled.

    Largest COMPONENT rather than every green pixel: a green advertising board
    or a strip of grass behind the goal is also green, and only one green region
    is the pitch. Holes are closed because the players standing on it are not
    grass, and leaving them open would cut the lines they stand near.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    # OPEN BEFORE the component step, and CLOSE only after it. The order is the
    # whole correctness of this function, and getting it wrong is not subtle.
    #
    # Closing first was the original bug: a 25x25 close bridged the trees behind
    # the stand to the pitch through the advertising hoardings, so "the largest
    # green component" became pitch + trees + crowd. Measured on the DFL frame,
    # that put 22.5% of the CROWD band inside the mask, and the line detector
    # then found its strongest lines on the hoardings rather than the paint.
    # Opening first severs those thin bridges, so the component step sees the
    # pitch alone: crowd band 22.5% -> 0.1%, pitch band still 99%.
    #
    # The SMALL close first is the third term, and it is not symmetric with the
    # big one: the painted lines themselves cut the green into rectangles, so
    # without it "the largest green component" can be a single cell of the grid
    # rather than the pitch. 7x7 bridges paint (3-5 px wide) and nothing else -
    # measured, it takes a synthetic pitch from 29% covered to 100% while moving
    # the real crowd band only 0.1% -> 0.5%. Real footage hid this because
    # anti-aliased paint never fully severs the mask; a clean synthetic one does.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    # Now it is safe to fill the holes the players punch in the surface.
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def paint_mask(bgr: np.ndarray, grass: np.ndarray) -> np.ndarray:
    """White paint, but only where there is grass to paint on."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, PAINT_V_MIN], np.uint8),
                        np.array([180, PAINT_S_MAX, 255], np.uint8))
    # Erode the grass before intersecting. Without this the mask's OWN boundary
    # survives as a bright edge and Hough traces it as a line - measured on the
    # DFL frame as three strong "lines" running along the advertising hoardings
    # where the pitch ends. Those are the edge of the mask, not paint on grass.
    # The touchline sits a little inside that boundary and survives the erode.
    inner = cv2.erode(grass, np.ones((9, 9), np.uint8))
    return cv2.bitwise_and(white, inner)


def segments(bgr: np.ndarray, min_len_frac: float = 0.04,
             max_gap_frac: float = 0.01) -> list[Segment]:
    """Line segments on the painted markings.

    Lengths are FRACTIONS of the frame width, not pixels: this has to run on a
    1920-wide scouting feed and an 11878-wide panorama band, and a threshold in
    pixels would mean two different things.
    """
    h, w = bgr.shape[:2]
    grass = grass_mask(bgr)
    paint = paint_mask(bgr, grass)
    # Thin the paint to its skeleton-ish core before Hough: a touchline is
    # several pixels wide, and Hough on a thick band returns one line per EDGE.
    #
    # The kernel scales DOWN with width and is CAPPED. Both halves are load-
    # bearing and each was found by breaking the other.
    #
    # Scaling down: a fixed 3x3 was silently fatal below ~3000 px, because paint
    # only 2 px wide does not survive a 3x3 open - the markings were erased
    # before Hough ever saw them. Over 30 frames of the 1920-wide DFL scouting
    # feed a 3x3 open left a MEDIAN OF 2 merged lines and reached the four a
    # homography needs in 10% of frames; at k=1 it is a median of 6 and 100%.
    #
    # The cap: paint is a few pixels wide in EVERY source, so the ratio that
    # rescues a 1920-wide frame over-erodes a wide one. This runs on the pitch-
    # band crop - 11878 px after the 2.9x isotropic upscale, not the 4096 px
    # native frame - and uncapped that asks for an 8x8 open. Measured on the
    # real band, frame 20000:
    #
    #     kernel 1x1   106 merged lines
    #     kernel 3x3    47      (what shipped, and what still ships)
    #     kernel 8x8     1      <- uncapped
    #
    # One line on the only format we can currently calibrate. The lesson is the
    # one this file already knows in min_len_frac: measure on the image the
    # PIPELINE passes, not the one the decoder hands you.
    k = min(3, max(1, round(w / 1400)))
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    edges = cv2.Canny(paint, 40, 120, apertureSize=3)

    found = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=60,
        minLineLength=int(w * min_len_frac), maxLineGap=int(w * max_gap_frac))
    if found is None:
        return []
    # HoughLinesP returns (N,1,4) on some OpenCV builds and (N,4) on others.
    # reshape rather than index, so the caller does not care which one is here.
    return [Segment(*map(float, s)) for s in np.asarray(found).reshape(-1, 4)]


def merge(segs: list[Segment], angle_tol: float = 3.0,
          offset_tol: float = 12.0) -> list[Segment]:
    """Collapse segments that lie on the same infinite line.

    Hough returns both edges of every painted band plus every gap-crossing
    fragment, so a single touchline arrives as a dozen segments. Grouping by
    (angle, perpendicular offset from the origin) is the cheap way to put them
    back together; the merged segment spans the extremes of its group.
    """
    out: list[Segment] = []
    used = [False] * len(segs)
    for i, a in enumerate(segs):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        ta = np.radians(a.angle)
        # Perpendicular distance from the origin to a's infinite line.
        na = np.array([-np.sin(ta), np.cos(ta)])
        da = float(na @ [a.x1, a.y1])
        for j, b in enumerate(segs[i + 1:], i + 1):
            if used[j]:
                continue
            dang = abs(a.angle - b.angle)
            dang = min(dang, 180.0 - dang)           # 179 deg and 1 deg are 2 apart
            if dang > angle_tol:
                continue
            if abs(float(na @ [b.x1, b.y1]) - da) > offset_tol:
                continue
            group.append(b)
            used[j] = True
        pts = np.array([[s.x1, s.y1] for s in group] + [[s.x2, s.y2] for s in group])
        d = np.array([np.cos(ta), np.sin(ta)])       # project onto the direction
        t = pts @ d
        out.append(Segment(*pts[int(t.argmin())], *pts[int(t.argmax())]))
    return out


def find(bgr: np.ndarray, min_len_frac: float = 0.04) -> list[Segment]:
    """Detected pitch lines, merged, longest first."""
    return sorted(merge(segments(bgr, min_len_frac)), key=lambda s: -s.length)


def _selfcheck() -> None:
    """A synthetic pitch: green ground, two white lines, known geometry."""
    img = np.zeros((400, 800, 3), np.uint8)
    img[:] = (60, 140, 70)                                  # BGR grass
    cv2.line(img, (400, 20), (400, 380), (245, 245, 245), 4)   # halfway
    cv2.line(img, (40, 350), (760, 350), (245, 245, 245), 4)   # a touchline

    g = grass_mask(img)
    assert g.mean() > 200, "grass mask should cover a grass frame"
    p = paint_mask(img, g)
    assert 0 < p.mean() < 60, f"paint should be a small bright minority, got {p.mean()}"

    segs = find(img)
    assert len(segs) >= 2, f"expected both lines, got {len(segs)}"
    angs = sorted(s.angle for s in segs[:2])
    # One near-horizontal, one near-vertical, in whatever order.
    assert angs[0] < 15 or angs[0] > 165, angs
    assert 75 < angs[1] < 105, angs

    # Paint OUTSIDE the grass must be ignored — this is the whole ordering rule.
    crowd = img.copy()
    crowd[:60] = (200, 200, 200)                            # a white stand
    g2 = grass_mask(crowd)
    assert g2[:60].mean() < 40, "the stand must not be inside the grass mask"

    # merge() must fold the two edges of one thick band into one line.
    band = np.zeros((200, 400, 3), np.uint8); band[:] = (60, 140, 70)
    cv2.line(band, (20, 100), (380, 100), (245, 245, 245), 6)
    assert len(find(band)) <= 2, "a single band should not explode into many lines"

    # THIN paint at BROADCAST width must survive the open. This is the case the
    # fixed 3x3 kernel erased: at 1920 a touchline is ~2 px, and a 3x3 open
    # deletes it outright - the detector then returned 2 lines per frame on the
    # DFL feed and could not have calibrated anything. Full width, real
    # thickness, because a 400 px toy frame cannot fail this way.
    hd = np.zeros((1080, 1920, 3), np.uint8); hd[:] = (60, 140, 70)
    cv2.line(hd, (960, 100), (1000, 1000), (245, 245, 245), 2)   # 2 px, near-vertical
    cv2.line(hd, (100, 800), (1850, 860), (245, 245, 245), 2)    # 2 px, near-horizontal
    assert len(find(hd)) >= 2, f"2 px paint at 1920 wide was erased: {len(find(hd))}"

    # ...and the other end, which the ratio broke while fixing the first. The
    # pipeline runs this on the 11878 px pitch band, where an UNCAPPED w/1400
    # asks for an 8x8 open and takes the real panorama from 47 lines to 1. Paint
    # there is ~6 px, so a wide frame with 6 px paint is the honest test.
    # 870 rows, because that is the band's real height and min_len_frac is a
    # fraction of WIDTH: at 11878 the floor is 475 px, so a line has to be long
    # in absolute terms before it counts at all.
    wide = np.zeros((870, 11878, 3), np.uint8); wide[:] = (60, 140, 70)
    cv2.line(wide, (200, 435), (11578, 522), (245, 245, 245), 6)
    cv2.line(wide, (5939, 20), (5999, 850), (245, 245, 245), 6)
    assert len(find(wide)) >= 2, f"6 px paint at 11878 wide was erased: {len(find(wide))}"

    print("lines.py ok")


if __name__ == "__main__":
    _selfcheck()
