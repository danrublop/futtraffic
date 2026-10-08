"""How big is this pitch? Three sources, one rule, and every answer names where it came from.

    LENGTH  is measured from the paint (autocal: the penalty areas and the circle are regulation, so only
            one length puts them all on paint).
    WIDTH   cannot be measured from one side-on camera - it trades against a vertical correction at the top
            of the frame (docs/experiments/n36-far-touchline.md section 3) - so it comes from outside:
              1. somebody who knows it typed it;
              2. OpenStreetMap's outline of the pitch, IF its length agrees with the painted length. About
                 half of OSM's pitch polygons trace the paint (width within a metre or two) and half trace
                 the grass or the fence (3-22 m too big), and nothing in the polygon says which - but the
                 painted length does (research-pitch-size-sources (not published), 23 venues);
              3. assumed 68 m, SAID to be assumed.
            That precedence lives where the product keeps a pitch size - app/server/jobs.py venue_pitch and
            app.py set_pitch - not in a second copy here; this module measures and gates.

The lookup is by coordinates, never by name: the OSM geocoder's usage policy rules it out for an app like
this one. Overpass is a shared free service - one request per lookup, made only when somebody asks, an identifying User-Agent with
no person in it, and a failure is said, never a blocked match.
Data (c) OpenStreetMap contributors, ODbL: anything shown from here carries ATTRIBUTION beside it, and an
OSM-derived width is stored in its own field, never blended into a typed one.

    .venv/bin/python model/engine/pitchsize.py                    # self-check, no network
    .venv/bin/python model/engine/pitchsize.py 51.6043 -0.0664    # pitches within 300 m of a point
"""
import json
import math
import sys
import urllib.parse
import urllib.request

import numpy as np

IFAB = {"L": (90.0, 120.0), "W": (45.0, 90.0)}
GATE_M = 1.5        # ponytail: one threshold, picked on 23 venues AFTER seeing them (n=1); re-fit on a fresh sample before trusting it
OVERPASS = "https://overpass-api.de/api/interpreter"    # ponytail: the public instance; a paid or own instance if lookups ever matter at volume
USER_AGENT = "futtraffic-pitchsize/1 (desktop app; one lookup per venue)"
ATTRIBUTION = "Pitch outline © OpenStreetMap contributors, ODbL - openstreetmap.org/copyright"


def _is_pitch(L, W):
    return IFAB["L"][0] <= L <= IFAB["L"][1] and IFAB["W"][0] <= W <= IFAB["W"][1] and L > W


def gate(osm_length, paint_length):
    """-> (usable, why). The OSM width is believed only when the OSM LENGTH matches the paint's."""
    if paint_length is None:
        return False, "no painted length to check the outline against"
    d = osm_length - paint_length
    if abs(d) <= GATE_M:
        return True, f"its length is within {GATE_M} m of the painted {paint_length:.1f} m"
    return False, (f"{d:.1f} m longer than the painted {paint_length:.1f} m: it outlines the grass, not the lines" if d > 0
                   else f"{-d:.1f} m shorter than the painted {paint_length:.1f} m: another pitch, or a bad trace")


def measure(latlon):
    """[(lat, lon), ...] of a polygon -> (length m, width m, fill). The minimum-area rectangle on a local
    tangent plane with the WGS84 radii of curvature at the polygon's own latitude - a sphere of 6371 km is
    0.3 m out on a pitch's length at 52 N. `fill` = polygon area / rectangle area."""
    lat, lon = np.radians(np.asarray(latlon, float)).T
    a, e2 = 6378137.0, 0.00669437999014
    s = 1 - e2 * math.sin(lat.mean()) ** 2
    M, N = a * (1 - e2) / s ** 1.5, a / math.sqrt(s)
    P = np.column_stack([(lon - lon.mean()) * N * math.cos(lat.mean()), (lat - lat.mean()) * M])
    import cv2  # noqa: PLC0415
    (_c, (w, h), _ang) = cv2.minAreaRect(P.astype(np.float32))
    area = 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))
    return float(max(w, h)), float(min(w, h)), float(area / max(w * h, 1e-9))


def osm_pitches(lat, lon, paint_length=None, radius_m=300, timeout=25):
    """Football-pitch-sized `leisure=pitch` outlines within `radius_m` of a point, nearest first:
    [{way, length, width, fill, distance_m, name, surface, usable, why}]. Never picks one: two pitches side by
    side are the normal case at a training ground. None when the service could not be reached (it is free and
    often busy) - which is not the same as [] "no pitch is mapped here"."""
    q = f"[out:json][timeout:{timeout}];way(around:{int(radius_m)},{float(lat):.6f},{float(lon):.6f})[leisure=pitch];out tags geom;"
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": USER_AGENT})
    try:
        elements = json.load(urllib.request.urlopen(req, timeout=timeout + 5)).get("elements", [])
    except Exception:  # noqa: BLE001
        return None
    out = []
    for e in elements:
        g, tags = e.get("geometry") or [], e.get("tags") or {}
        if len(g) < 4 or ("sport" in tags and "soccer" not in tags["sport"]):
            continue
        L, W, fill = measure([(p["lat"], p["lon"]) for p in g])
        if not _is_pitch(L, W):
            continue
        clat, clon = np.mean([p["lat"] for p in g]), np.mean([p["lon"] for p in g])
        dist = 111320.0 * math.hypot(clat - lat, (clon - lon) * math.cos(math.radians(lat)))
        ok, why = gate(L, paint_length)
        out.append({"way": int(e["id"]), "length": round(L, 2), "width": round(W, 2), "fill": round(fill, 3), "distance_m": round(dist),
                    "name": tags.get("name"), "surface": tags.get("surface"), "usable": ok, "why": why})
    return sorted(out, key=lambda r: r["distance_m"])


def _demo():
    # a 105 x 68 rectangle laid out at 52 N, turned 30 degrees, comes back to the centimetre
    lat0, lon0, th = 52.0, 0.1, math.radians(30)
    a, e2 = 6378137.0, 0.00669437999014
    s = 1 - e2 * math.sin(math.radians(lat0)) ** 2
    M, N = a * (1 - e2) / s ** 1.5, a / math.sqrt(s)
    ring = [(x * math.cos(th) - y * math.sin(th), x * math.sin(th) + y * math.cos(th)) for x, y in ((0, 0), (105, 0), (105, 68), (0, 68))]
    L, W, fill = measure([(lat0 + math.degrees(y / M), lon0 + math.degrees(x / (N * math.cos(math.radians(lat0))))) for x, y in ring])
    assert abs(L - 105) < 0.02 and abs(W - 68) < 0.02 and fill > 0.999, (L, W, fill)
    # the gate, on venues the research measured
    assert gate(105.03, 105.0)[0]                                   # Tottenham: traced on the paint
    ok, why = gate(102.82, 100.2)
    assert not ok and "grass" in why, why                           # Craven Cottage: traced on the grass
    assert not _is_pitch(128.17, 93.27)                             # Tsukuba's polygon is the whole turf area
    assert not gate(105.0, None)[0]                                 # no paint: an outline is a suggestion only
    print("pitchsize ok: a 105 x 68 m outline at 52 N measured as %.2f x %.2f; the length gate tells paint from grass" % (L, W))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        for row in osm_pitches(float(sys.argv[1]), float(sys.argv[2])) or ["lookup failed or nothing mapped"]:
            print(row)
        print(ATTRIBUTION)
    else:
        _demo()
