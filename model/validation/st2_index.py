"""SoccerTrack v2 BAS labels -> one flat, video-aligned table. Run me before training.

WHY THIS IS NOT A THREE-LINE json.load
    `docs/format-bas.md` in the upstream repo describes a file that is NOT the one
    shipped. Shipped is `match_id`/`fps`/`actions` with UPPERCASE labels, numeric
    `player_id`, no `visibility`, and `position` on a single cumulative match clock
    - not the doc's `UrlLocal`/`annotations`, Title-Case, per-half clock. Code to
    the data, never to that doc.

THE HALF TAG LIES; POSITION DOES NOT
    132831 carries rows tagged "1 - 95:57". Three matches carry rows with no half
    prefix at all ("90:01"). So `gameTime` cannot assign a period.

    Nor can `position // 45:00`. Every period's clock overruns its band by up to
    five minutes of stoppage - half 1 reaches 2,999 s - so a plain band split files
    stoppage into the next period. Nor does file order alone work: three matches
    append a tail of late rows to the first block with no backwards jump.

    What IS reliable is the shape of the file. Rows come in contiguous runs, one
    run per period, and a run ends either with a backwards jump in `position` or -
    for the tails - with a silence of tens of minutes, where real play never has a
    gap over ~1 min. Cut on both, then read each run's period off where it STARTS,
    which always lands on a 45:00 mark (measured: 2,699-2,703 s, 5,400.1-5,401.6 s).
    Stoppage rides out to the end of its own run and is never reassigned.

    Two matches (117093, 128057) have neither signal - their halves abut smoothly,
    so the whole file is one run. They are caught by the fact that a real period
    never spans more than ~50 min: a run wider than that is split at the 45:00 mark
    it crosses. Matches WITH stoppage past the mark were already cut by the jump,
    so this second pass never touches them.

THREE MATCHES HAVE A THIRD PERIOD (offset 90:00)
    117092 (1,055 rows), 132831 (719), 132877 (451) - 9% of the dataset. A tail of
    those (48 / 28 / 20) is APPENDED to the first block instead of forming its own,
    monotonically, so the backwards-jump rule alone misses them - a second pass on
    the clock catches them. 117092 and
    132831 ship ~90-minute "half" videos (6.7 GB vs everyone else's 3.4), so each
    file holds two periods and those rows DO have footage, at a within-file offset
    this script cannot know. 132877's videos are normal length, so its 431 rows are
    orphans. Both cases are flagged, never silently dropped.

`ponytail: period offset is k * 45:00 and the within-video offset for the
double-length files is unknown until someone eyeballs a kickoff. Upgrade path is
one measured offset per period in PERIOD_VIDEO_OFFSET_S.`
"""
import csv
import glob
import json
import os
import sys

HALF_MS = 2_700_000          # every period's clock starts at k * 45:00
PERIOD_MAX_MS = 3_000_000    # no real period runs past 50 min
RUN_GAP_MS = 600_000         # a silence this long ends a run; real play never pauses 10 min
# Every match ships exactly two videos, so period 2+ has no footage anywhere.
PERIODS_WITH_VIDEO = 2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BAS_GLOB = os.path.join(_ROOT, "data/soccertrack_v2/bas__*.json")
OUT = os.path.join(_ROOT, "data/soccertrack_v2/bas_index.csv")

# THE EVENT-MODEL SPLIT. Owner decision 2026-09-18, fixed; not ST2's official split.
# The official val (132877) and test (132831) are the two matches whose tracking
# feed is mirrored (st2-132877-feed-is-mirrored (internal note)). So: 132877 stays
# in val, read UN-MIRRORED (y -> 1 - y, gated on st2_team_gt.feed_mirrored) and
# flagged; 132831 is EXCLUDED - its feed is mirrored against an unusable chain and
# its speeds are implausible - and that exclusion is stated beside every number;
# test is 128057 alone and is scored once, at the end, never by P1 development.
# Numbers on this split are NOT comparable to the ST2 paper's official split.
EVENT_SPLIT = {
    "117092": "train", "118575": "train", "118576": "train",
    "118577": "train", "118578": "train", "128058": "train",
    "117093": "val", "132877": "val",
    "128057": "test",
    "132831": "excluded",
}

# PROTOCOL for sprint P1 step 6 (GT-track cache, G0, nearest-player floor).
# Committed BEFORE st2_cache.py exists or any cache is built.
#
#   CACHE     positions come from the vendor's dense feed only
#             (raw/<m>_tracker_box_data.xml via st2_team_gt.dense_loc) - no
#             detector. The feed's own ball is cached as the REFERENCE POINT of
#             the nearest-player floor and nothing else; it is never a model
#             channel (sprint §P1).
#   LANDS     a labelled event lands iff it is in period 0 or 1, its clock is not
#             negative, a feed frame exists within 40 ms (one frame) of its match
#             clock, and its labelled actor id is on the pitch in that frame.
#             Everything else is DROPPED and counted per match, per reason.
#   RECEIVER  a PASS's receiver is the next event's actor iff that event is a
#             DRIVE by a different player in the same period; completed iff the
#             two are on the same side (home/away from the roster, never BAS
#             left/right, which flips at half time).
#   G0 ALIGNED on a val match iff BOTH
#             (a) the 24-frame contact sheet shows the labelled actor at the
#                 feed's ball on a human read, and
#             (b) shifting every label by s in -2.0..+2.0 s, step 0.2 s (n=21
#                 shifts), the median actor-to-ball distance is smallest at
#                 |s| <= 0.4 s.
#   FLOOR     nearest-player: on val, over landed events, name the on-pitch
#             player nearest the feed's ball at the label frame. A tie of k
#             players containing the actor scores 1/k. Reported beside chance
#             (mean 1/n on-pitch) per match, for PASS, DRIVE and all classes.
#             Attribution is only quoted as a delta over this. Test not scored.
#
# ADDENDUM 2026-09-19, committed before any cache is rebuilt with it. G0 failed:
# in 10 of 16 train+val periods the labels peak ~1 s early against the feed, and
# the STC clips (joined to the feed within 25 px of the MOT boxes, so feed = video
# clock) show the release ~1 s AFTER the label in those periods and AT it in the
# 118575 control - an exploratory eyeball, 6 events. So the LABELS are early.
#   LABEL OFFSET  per (match, period), over landed PASSes: share of actors within
#             1.5 m of the feed ball at the label frame (s0) and 25 frames later
#             (s1). Offset = +25 frames iff s1 - s0 >= 0.05, else 0; the rest of
#             the protocol then reads the shifted row. Applied to EVERY split,
#             test included: it is label cleaning from the GT feed, never a model
#             score, and the per-period offsets are printed by --drops.
#             A shifted row past the feed's end is dropped as outside_feed.
#   G0 again  on the corrected labels, rule (b) above, unchanged.


def split_check(split=EVENT_SPLIT):
    """The split is disjoint, complete and exactly the recorded split decision."""
    by = {}
    for m, s in split.items():
        assert s in ("train", "val", "test", "excluded"), (m, s)
        by.setdefault(s, set()).add(m)
    assert by["val"] == {"117093", "132877"}, by
    assert by["test"] == {"128057"}, by
    assert by["excluded"] == {"132831"}, by
    assert len(by["train"]) == 6 and len(split) == 10, by
    bas = {os.path.basename(f).split("__")[1] for f in glob.glob(BAS_GLOB)}
    assert not bas or bas == set(split), bas ^ set(split)
    return by


def periods(actions):
    """Period index per row, in file order. Cut into runs on a backwards jump or a
    10-minute silence; a run's period is the 45:00 mark it starts on."""
    starts, prev = [0], None
    for i, a in enumerate(actions):
        p = int(a["position"])
        if prev is not None and (p < prev or p - prev > RUN_GAP_MS):
            starts.append(i)
        prev = p
    starts.append(len(actions))

    # a real period is under 50 min; a wider run is two periods that abutted
    # smoothly, so cut it where it crosses the mark.
    split = []
    for lo, hi in zip(starts, starts[1:]):
        base = round(int(actions[lo]["position"]) / HALF_MS)
        cut = lo
        while cut < hi:
            mark = (base + 1) * HALF_MS
            nxt = next((i for i in range(cut, hi) if int(actions[i]["position"]) >= mark), hi)
            if int(actions[hi - 1]["position"]) - int(actions[cut]["position"]) <= PERIOD_MAX_MS:
                nxt = hi
            split.append((cut, nxt, base))
            cut, base = nxt, base + 1

    out = []
    for lo, hi, k in split:
        out += [k] * (hi - lo)
    return out


def index_match(path):
    d = json.load(open(path))
    mid = str(d["match_id"])
    fps = float(d["fps"])
    acts = d["actions"]
    rows = []
    for a, k in zip(acts, periods(acts)):
        t_ms = int(a["position"]) - k * HALF_MS
        rows.append({
            "match": mid,
            "period": k,                       # 0,1 = the two halves; 2+ = extra
            "t_s": round(max(t_ms, 0) / 1000, 3),
            "frame": round(max(t_ms, 0) / 1000 * fps),
            "label": a["label"],
            "team": a["team"],
            "player_id": a.get("player_id") or "",
            "video": f"{mid}_panorama_{'1st' if k == 0 else '2nd'}_half.mp4"
                     if k < PERIODS_WITH_VIDEO else "",
            # the flag that decides whether a row is trainable
            "no_footage": int(k >= PERIODS_WITH_VIDEO),
            "clock_negative": int(t_ms < 0),
        })
    return rows


def main():
    files = sorted(glob.glob(BAS_GLOB))
    if not files:
        sys.exit(f"no BAS files at {BAS_GLOB} - see soccertrack-v2 on HuggingFace")
    rows = [r for f in files for r in index_match(f)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    usable = [r for r in rows if not r["no_footage"]]
    print(f"{len(rows)} rows, {len(files)} matches -> {OUT}")
    print(f"  trainable              {len(usable)}")
    print(f"  no footage             {sum(r['no_footage'] for r in rows)}"
          f"   (third periods of 117092 / 132831 / 132877)")
    print(f"  clamped to t=0         {sum(r['clock_negative'] for r in rows)}")


def demo():
    """Self-check: every period must land inside a plausible video, which neither
    the raw cumulative `position` nor the `gameTime` half tag does."""
    split_check()
    try:     # a one-key mutation must be caught, or the check is decoration
        split_check({**EVENT_SPLIT, "132831": "test"})
        raise RuntimeError("split_check accepted 132831 in test")
    except AssertionError:
        pass
    rows = [r for f in sorted(glob.glob(BAS_GLOB)) for r in index_match(f)]
    assert len(rows) == 23663, len(rows)

    by = {}
    for r in rows:
        assert 0 <= r["t_s"] <= 3000, r        # no period clock past 50 min
        by.setdefault((r["match"], r["period"]), []).append(r["t_s"])

    shape = {}
    for (m, k), ts in by.items():
        assert max(ts) < 3000, (m, k, max(ts))
        shape.setdefault(m, set()).add(k)
    # seven ordinary matches, three with a third period
    assert {m for m, ks in shape.items() if ks == {0, 1}} == {
        "118575", "118576", "118577", "118578", "128058", "117093", "128057"}, shape
    assert {m for m, ks in shape.items() if ks == {0, 1, 2}} == {
        "117092", "132831", "132877"}, shape
    # the flags actually flag something
    assert sum(r["no_footage"] for r in rows) == 2225, "three orphaned third periods"
    assert sum(1 for r in rows if not r["no_footage"]) == 21438
    print(f"demo ok: {len(rows)} rows, {len(by)} periods, every clock under 50 min")


if __name__ == "__main__":
    (demo if "--demo" in sys.argv else main)()
