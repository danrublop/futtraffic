"""Seed player headshots from Wikipedia.

    python headshots.py "Full Name" "Another Full Name"
    python headshots.py --file squad.txt
    python headshots.py --list                 # what is in the store already

WHY WIKIPEDIA and not a football API. Every feed that carries player photographs
- FotMob, API-Football, Transfermarkt - carries them under a licence that covers
the subscriber and not whoever the subscriber hands the file to, and a scouting
report is a document that gets MAILED. Wikipedia's lead images come off Wikimedia
Commons under CC-BY-SA or public domain: free to redistribute, on the condition
that the credit travels with the picture. So the credit is stored beside the
photo (credits.json) rather than thrown away, and the report prints it.

It is also the only one of them that needs no key, which matters for a thing a
club runs on its own laptop.

THE CROP is the part that is not obvious. A lead image is whatever photograph
somebody uploaded: a portrait, a bust, or a wide shot of a man running. Cropping
a square off the top gets the first two right and the third badly wrong - the
head is somewhere in the upper left of an action shot, not the middle. So this
runs a face detector and widens the box to head-and-shoulders, which is the
framing every scouting report uses, and falls back to a top-anchored square when
it finds nothing.

The detector is YuNet, through the OpenCV already in the tree - no new
dependency, only its weights (232 KB, MIT, fetched once into model/models/). OpenCV 5
dropped the Haar cascades that used to ship inside the wheel, so "the classifier
is bundled" is no longer true and the file has to come from somewhere.

The store is jobs.write_photo, so what lands here is exactly what an upload
would land - same path, same magic-number check, same everything.
"""
from __future__ import annotations

import _path  # noqa: F401  - puts model/engine/ on sys.path

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO

import jobs

API = "https://en.wikipedia.org/w/api.php"
YUNET = "face_yunet.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# Wikimedia asks for a real agent with a way to be contacted. An anonymous
# python-urllib gets rate-limited and deserves to be.
AGENT = "futtraffic-headshots/0.1 (club scouting tool; contact: local install)"
SIZE = 128          # what the block draws at 38 px, with room for a retina screen
JPEG_Q = 82

_TAGS = re.compile(r"<[^>]*>")


def slug(s: str) -> str:
    """roster.py's own id rule, so a photo fetched by NAME lands on the id the
    roster would have minted for that name. Copied rather than imported: this
    script must run without the roster package on the path."""
    folded = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def _get(params: dict) -> dict:
    """One API call, with the one retry Wikimedia's rate limiter asks for.

    A squad done one name at a time gets 429 after about five, which is exactly
    right of them and the reason `lookup` below asks for fifty at once. The
    retry is for the times a batch still lands on a busy minute."""
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for wait in (0, 5, 20):
        if wait:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed host
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
    raise RuntimeError("Wikipedia is rate-limiting; try again in a minute")


def _bytes(url: str) -> bytes:
    """One file off Wikimedia, with the same backoff. The IMAGE host limits
    separately from the API host, and a squad is thirty-eight files in a row -
    which is exactly the shape a limiter is looking for."""
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for wait in (0, 3, 15, 45):
        if wait:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - Wikimedia
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
    raise RuntimeError("Wikimedia is rate-limiting the images")


BATCH = 50  # what the API takes in one `titles=`


def _resolved(query: dict) -> dict[str, str]:
    """asked-for title -> the title the API actually answered with.

    Both hops matter and they chain: "Pipo Ferro" is NORMALISED to "Pipo Ferro"
    and then REDIRECTED to "Andres Ferro", and half a squad is redirects.
    """
    hop = {}
    for key in ("normalized", "redirects"):
        for r in query.get(key, []):
            hop[r["from"]] = r["to"]
    out = {}
    for name in list(hop) + list(query.get("pages", {})):
        seen, t = set(), name
        while t in hop and t not in seen:
            seen.add(t)
            t = hop[t]
        out[name] = t
    return out


def _search(name: str) -> dict | None:
    """The best-matching page with a lead image, for a name that is not a title.

    `generator=search` runs the query and hands the hits straight to
    pageimages, so this is one call rather than a search followed by a lookup.
    "footballer" is in the query because half these surnames are also a
    politician, and the top hit for a bare surname is not the player.
    """
    res = _get({
        "action": "query", "generator": "search", "gsrsearch": f"{name} footballer",
        "gsrlimit": 3, "prop": "pageimages", "piprop": "thumbnail|name",
        "pithumbsize": SIZE * 4,
    })
    hits = list(res.get("query", {}).get("pages", {}).values())
    hits.sort(key=lambda p: p.get("index", 99))
    # EVERY word of the name has to be in the title, accents folded off both
    # sides. Without this the top hit for "Nicolas Ibanez" is Sebastián
    # Domínguez and for "Lucas Moro" it is Lucas Prado - a search ranks by
    # relevance and a surname is relevant. A scouting report carrying the wrong
    # man's face is worse than one carrying a silhouette, so a hit that is not
    # clearly him is not a hit.
    want = set(slug(name).split("-"))
    for p in hits:
        if not (p.get("thumbnail") and p.get("pageimage")):
            continue
        if want <= set(slug(p.get("title", "")).split("-")):
            return p
    return None


def lookup(names: list[str]) -> dict[str, dict]:
    """{asked-for name: lead image + the credit that has to travel with it}.

    BATCHED, fifty at a time. One name per request is what a rate limiter is
    for, and a squad is thirty-eight names; this is two calls.
    """
    found: dict[str, dict] = {}
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        res = _get({
            "action": "query", "prop": "pageimages", "piprop": "thumbnail|name",
            "pithumbsize": SIZE * 4, "titles": "|".join(chunk), "redirects": 1,
        })
        query = res.get("query", {})
        pages = {p.get("title"): p for p in query.get("pages", {}).values()}
        hop = _resolved(query)
        want = {}
        for name in chunk:
            page = pages.get(hop.get(name, name))
            if page and page.get("thumbnail") and page.get("pageimage"):
                want[name] = page

        # A name that resolved to nothing gets ONE search. `titles=` is an
        # exact match after case and underscore folding and nothing else, so
        # "Nicolas Ibanez" simply is not "Nicolás Ibáñez" - and a coach
        # types the first one. The search index folds accents and ranks, which
        # is the whole reason to fall back to it rather than ask people to type
        # diacritics.
        for name in chunk:
            if name in want:
                continue
            hit = _search(name)
            if hit:
                want[name] = hit
                pages[hit["title"]] = hit

        # The credits, batched the same way. A file page per photo, and the
        # licence condition is that the credit travels with the picture.
        files = sorted({f"File:{p['pageimage']}" for p in want.values()})
        meta: dict[str, dict] = {}
        for j in range(0, len(files), BATCH):
            got = _get({
                "action": "query", "prop": "imageinfo", "iiprop": "extmetadata",
                "titles": "|".join(files[j:j + BATCH]),
            })
            for p in got.get("query", {}).get("pages", {}).values():
                for info in p.get("imageinfo", []):
                    # Keyed on the NORMALISED title. `pageimage` carries
                    # underscores and the API answers with spaces, so keying on
                    # the raw string missed every credit and left a squad of
                    # photographs with no photographer - which is the one field
                    # the licence actually requires.
                    meta[p["title"].replace("_", " ")] = info.get("extmetadata", {})

        for name, page in want.items():
            file_title = f"File:{page['pageimage']}"
            ext = meta.get(file_title.replace("_", " "), {})
            clean = lambda k: _TAGS.sub("", str(ext.get(k, {}).get("value", ""))).strip()  # noqa: E731
            found[name] = {
                "url": page["thumbnail"]["source"],
                "page": "https://en.wikipedia.org/wiki/"
                        + urllib.parse.quote(page["title"].replace(" ", "_")),
                "file": "https://commons.wikimedia.org/wiki/"
                        + urllib.parse.quote(file_title.replace(" ", "_")),
                "artist": clean("Artist"),
                "licence": clean("LicenseShortName"),
                "title": page["title"],
            }
    return found


def _yunet_path():
    """The weights, fetched once. `paths.MODELS` is where every other model in
    this project lives, so a club that mirrors that folder gets this too."""
    from paths import MODELS  # noqa: PLC0415

    p = MODELS / YUNET
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(YUNET_URL, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed host
            p.write_bytes(r.read())
    return p


def head_crop(data: bytes) -> bytes:
    """Head and shoulders, square, as JPEG.

    The detector is the point. A lead image is whatever somebody uploaded, and a
    fixed crop off the top is right for a portrait and wrong for the wide action
    shots that are half of them. When it finds a face the box is widened to
    1.9x - a face box is the face, and a report frames the shoulders too - and
    pushed down a little, because a head centred in its own box sits high once
    the chest is in frame.
    """
    import cv2  # noqa: PLC0415 - heavy, and only this path needs it
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img = Image.open(BytesIO(data)).convert("RGB")
    w, h = img.size
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Detected on a bounded copy and scaled back: a 5000 px lead image is a
    # second of inference for a box that is only ever used at 128 px.
    k = min(1.0, 640 / max(w, h))
    small = cv2.resize(bgr, (max(1, int(w * k)), max(1, int(h * k)))) if k < 1 else bgr
    det = cv2.FaceDetectorYN.create(str(_yunet_path()), "", (small.shape[1], small.shape[0]),
                                    score_threshold=0.6)
    _, faces = det.detect(small)
    faces = [] if faces is None else list(faces)
    if faces:
        # The biggest face: a squad photo cropped to a bystander is worse than
        # no crop at all, and the subject is the largest face in a lead image.
        fx, fy, fw, fh = (max(faces, key=lambda f: f[2] * f[3])[:4] / k)
        side = int(max(fw, fh) * 1.9)
        cx = int(fx + fw / 2)
        cy = int(fy + fh / 2 + fh * 0.25)
    else:
        # No face: the upper middle, which is where a head is in a portrait and
        # the least wrong guess in anything else.
        side = min(w, h)
        cx, cy = w // 2, int(side * 0.5 + h * 0.05)
    side = max(16, min(side, w, h))
    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, cy - side // 2))
    out = img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)
    buf = BytesIO()
    out.save(buf, "JPEG", quality=JPEG_Q, optimize=True)
    return buf.getvalue()


def credits_path():
    return jobs.photos_dir() / "credits.json"


def load_credits() -> dict:
    try:
        return json.loads(credits_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_credit(pid: str, credit: dict) -> None:
    """The licence condition, kept with the file rather than in a shell history.
    A photo whose credit is lost is a photo that cannot legally be printed."""
    all_credits = load_credits()
    all_credits[pid] = credit
    credits_path().parent.mkdir(parents=True, exist_ok=True)
    credits_path().write_text(json.dumps(all_credits, indent=1, sort_keys=True), encoding="utf-8")


def fetch(names: list[str], force: bool = False) -> tuple[list[tuple[str, str]], list[str]]:
    """Fetch a whole list. Returns (done, missing): (player id, credit line) per
    name that worked, and the names Wikipedia had no picture for.

    The two failures are told apart on purpose. "No lead image" is a fact about
    the name and the answer is a better one; a rate limit is a fact about the
    minute and the answer is to run it again. Reporting both as "not found"
    sent me looking for spellings that were already right.

    IDEMPOTENT: a player who already has a photo AND a photographer on it is
    skipped, so re-running to fill in the four that a rate limit ate does not
    re-download the thirty-four that worked.
    """
    have = load_credits()
    todo = [
        n for n in names
        if force or not (jobs.has_photo(slug(n)) and have.get(slug(n), {}).get("artist"))
    ]
    found = lookup(todo) if todo else {}
    out: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in names:
        pid = slug(name)
        hit = found.get(name)
        if not pid or not hit:
            if name in todo:
                missing.append(name)
            continue
        try:
            raw = _bytes(hit["url"])
        except Exception as e:  # noqa: BLE001 - one bad file must not stop a squad
            print(f"  !! {name}: {e}")
            continue
        jobs.write_photo(pid, head_crop(raw))
        save_credit(pid, {k: hit[k] for k in ("page", "file", "artist", "licence", "title")})
        out.append((pid, f"{hit['title']} - {hit['artist'] or 'unknown'} "
                         f"({hit['licence'] or 'see file page'})"))
        time.sleep(0.4)  # polite, and cheaper than the backoff it avoids
    return out, missing


def _selfcheck() -> None:
    """No network. The two things that broke in development, pinned.

        python headshots.py --selfcheck
    """
    # 1. The title chain. "Pipo Ferro" is NORMALISED and then REDIRECTED, and a
    #    resolver that follows one hop finds half a squad.
    hop = _resolved({
        "normalized": [{"from": "pipo ferro", "to": "Pipo Ferro"}],
        "redirects": [{"from": "Pipo Ferro", "to": "Andres Ferro"}],
        "pages": {"1": {"title": "Andres Ferro"}},
    })
    assert hop["pipo ferro"] == "Andres Ferro", hop
    # A loop must not hang: a redirect pair that points at each other exists.
    loop = _resolved({"redirects": [{"from": "A", "to": "B"}, {"from": "B", "to": "A"}],
                      "pages": {}})
    assert loop["A"] in ("A", "B")

    # 2. The crop. A face detector that finds nothing must still return a
    #    square JPEG rather than raise - a squad where one photograph is a
    #    landscape crowd shot is the normal case, not the exception.
    from io import BytesIO as _B  # noqa: PLC0415

    from PIL import Image as _I  # noqa: PLC0415

    buf = _B()
    _I.new("RGB", (600, 900), (90, 120, 90)).save(buf, "PNG")
    out = head_crop(buf.getvalue())
    assert out[:3] == b"\xff\xd8\xff", "not a JPEG"
    assert _I.open(_B(out)).size == (SIZE, SIZE), _I.open(_B(out)).size

    assert slug("Ángel Del Río") == "angel-del-rio"
    assert slug("") == ""

    # 3. The search guard. A ranked search answers a surname with whoever is
    #    most famous, and a report with the wrong man's face on it is worse
    #    than one with a silhouette.
    matches = lambda name, title: set(slug(name).split("-")) <= set(slug(title).split("-"))  # noqa: E731
    assert matches("Nicolas Gonzalez", "Nicolás González (footballer, born 1998)")
    assert matches("Eduardo Salva", "Eduardo Salva")
    assert not matches("Nicolas Ibanez", "Sebastián Ibáñez")
    assert not matches("Lucas Moro", "Lucas Prado")
    print("headshots.py ok")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--file", help="one name per line")
    ap.add_argument("--list", action="store_true", help="what the store holds")
    ap.add_argument("--force", action="store_true", help="re-fetch what is already there")
    ap.add_argument("--selfcheck", action="store_true", help="offline checks")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0

    if args.list:
        for pid, c in sorted(load_credits().items()):
            print(f"{pid:26} {c.get('title','')} - {c.get('artist','?')} ({c.get('licence','?')})")
        return 0

    names = list(args.names)
    if args.file:
        names += [ln.strip() for ln in open(args.file, encoding="utf-8") if ln.strip()]
    if not names:
        ap.error("give some names, or --file")

    got, missing = fetch(names, force=args.force)
    for pid, line in got:
        print(f"  ok {pid:26} {line}")
    for name in missing:
        print(f"  -- {name}: no lead image on Wikipedia (try the exact article title)")
    print(f"{len(got)} fetched, {len(names)} asked for, into {jobs.photos_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
