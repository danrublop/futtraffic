"""N71: does handing the identity stage the team's roster name players right WITHOUT writing wrong names? Protocol first.

WHY
    app/server/jobs.py apply_identity builds the strategy with cal only, so AssistedIdentity runs with roster=None
    and match_id "match" (docs/reference/identity.md, Limits: "Roster seeding is off on the job path"). The question
    put to this experiment: would passing the job's team roster, carrying the operator's confirmed bindings from
    earlier matches, improve naming without writing more wrong names?

WHAT A ROSTER CHANGES IN THE CODE (model/engine/identity.py AssistedIdentity.assign, roster.py; read, not assumed)
    1. Seeding: confirmed(self.match_id) - manual bindings FOR THIS MATCH ID ONLY. A binding confirmed in an earlier
       match is keyed by that match's id and that match's track ids, so it has NO path into a new match. Bindings
       from earlier matches therefore do not seed anything; the premise of the question does not hold in the code,
       and this page does not pretend it does. Checked as an invariant below (R1 with vs without the source
       bindings must be identical), not assumed.
    2. _resolve_roster: every group whose OCR vote produced a number gets roster.by_number(number) - name,
       player_id, source "roster". The NUMBER is unchanged, and by_number has NO TEAM GATE: an opponent group
       reading a number that is in our squad gets our player's name. by_number is first-match-wins.
    3. _persist: those machine bindings are written into the roster (the caller owns save()).
    So the only cross-match channel is the SQUAD (player id, name, shirt number), which an operator's confirmed
    history builds. That is what the arms vary. Numbers written, merges, min_conf and the one-digit rule are
    identical across arms by construction (checked below). apply_identity today writes number/conf/source only,
    never name or player_id, so shipping this also needs apply_identity to write them; the experiment scores the
    strategy's returned PlayerId, which is what that wiring would write.

ARMS (n=3 configurations; nothing else is tried, nothing is swept)
    R0  roster=None, match_id "match" - shipped.
    R1  a CORRECT roster. Players = team T's on-pitch players in the OTHER ST2 matches T plays in
        (st2_team_gt mot_to_player x team side), player_id "st2-<ST2 id>" (explicit, so name slugs cannot collide),
        name from the metadata. Number = the target match's TEAM-SHEET number (metadata back_number; None if the
        player is not on that sheet): the list an operator confirms before kick-off. It says who wears what,
        never which track is whom, so it carries none of the scored mapping. Plus a manual binding (source
        match, MOT id -> player) per source player, standing in for the operator's confirmations. match_id = the
        target clip. --inputs asserts no present player's number resolves to another man under R1.
        Why R1 is not the roster history would build: numbers change between matches. With the number each
        player LAST WORE in a source match instead (GT only, no pipeline output):
    @M 43 of 55,stale 8,maps-elsewhere 5,27 of 55,stale 7,maps-elsewhere 8 :: .venv/bin/python model/validation/n71_roster_seed.py --inputs
       held fixed: DEV then HELD units, R1's players with history numbers; target team-T players who are in the
       roster, whose roster number is stale, and whose number by_number resolves to someone else. n=0 arms run.
        So a wrong squad number is not hypothetical: it occurs naturally on both splits. R2 plants exactly one.
    R2  R1 with ONE wrong squad number planted, by a rule fixed now:
          P = the team-T target player whose two-digit GT number R0 writes on a group whose modal truth is P
              (highest vote confidence first); if R0 carries none, the target team-T player with a two-digit
              number and the smallest ST2 id.
          X = the roster player other than P with the smallest ST2 id.
          X.number := P's target number, and X moves to the front of the squad so by_number returns X.
        This is the product risk: the roster names the wrong man under a number OCR reads. "bit" = R0 carried
        P's number correctly, so the plant is exercised; a plant that does not bite tests nothing and is counted.
    R1b (INVARIANT, not an arm) R1 without the source bindings. Must equal R1 exactly, else VOID.

HELD FIXED
    The stored shipped-config tracks {seq}.p35.tracks.json (N69 replayed A0 on all seven clips and got these files
    back byte-for-byte; n69-tracker-frame-rate (not published)), the frozen N34 map {seq}.p35.calibration.json,
    the clip st2_product.CLIPS/{seq}.mp4, and jobs.apply_identity itself with FUTTRAFFIC_IDENTITY unset (assisted,
    min_conf 0.6, ocr auto -> docTR). The ONLY difference between arms is the roster and match_id injected into
    get_identity_strategy. scan_video runs ONCE per clip and its result is replayed to every arm (it never reads
    the roster); R0's per-detection jersey column must equal R1's and R2's, else VOID.

POPULATION AND SPLIT - units are (clip, team) with the team in at least one other ST2 match
    Target clips need a .p35 artefact (seven). Teams recur across matches (9834 in five, 30798 in three, 31424,
    31425, 34317 in two each), so every roster comes from DIFFERENT MATCHES than the frames it is scored on:
    no frame, track or GT row of the target clip enters its roster (asserted: the target is never a source).
    Sources may be any ST2 match, including 117092 / 132831 / 132877 (no .p35; GT names and numbers only).
      DEV  (N69's DEV clips; looked at freely)   118575x31424, 118575x31425, 118576x9834, 118576x31424, 117093x9834
      HELD (N69's HELD clips; scored ONCE)       118577x31425, 118578x9834, 118578x30798, 128058x9834, 128058x34317
    n=5 units each. DROPPED, and why: 128057 (both teams, 10581 and 34316, play no other ST2 match, so no roster
    can exist); the other team on 117093 (9701) and 118577 (31790) for the same reason. Not a hard-case filter:
    the drop is structural and decided by the team sheet, before any pipeline output. HELD is held out for THIS
    knob: no roster has ever been passed anywhere; its R0 identity outputs were produced by st2_product (N21).
    Tuning: the OCR/identity constants were shaped on 118575 (N49), which is DEV.

TRUTH AND METRICS (per unit, per arm; the join is check.py identity's _gt_diag, reused not re-invented)
    join     each output detection -> nearest MOT GT foot at its frame within 30 px L1; a group's truth is its
             modal MOT id if it has >= 20 hits and >= 60% share, else NO-TRUTH. MOT id -> ST2 player id and side
             by st2_team_gt. Pooled joined-detection share is printed; under 0.5 the run is VOID (spaces differ).
    PR   players named right: distinct team-T target players carried by >= 1 group whose truth is that player
         and whose player_id is theirs. Denominator: team-T players among the clip's 22 GT ids.
    GN   groups named (player_id set).   RG  groups named right.   precision = RG / GN.
    WN   wrong names = GN - RG, split into: CROSS (truth is the other team), SAME (truth is another team-T
         player), NOTRUTH (unverifiable: counted as wrong, because a name on it persists all the same).
    PROP (R2) wrong groups named X in R2 minus the same in R1.
    context, not in the rule: R0's groups carrying a team-T player's correct number, and R0's wrong numbers.
    R0 names nobody by construction (PR = GN = 0), so "R1 names more than R0" is trivially true on any name
    and is not a clause; the clauses are about whether the names it adds are right.

RULE, fixed now. Pooled over the 5 HELD units unless stated.
    SHIP roster seeding iff all of:
      (a) PR(R1) >= 5                              - at least one player per unit on average, or it is inert
      (b) precision(R1) >= 0.90                    - N49's "safe to show a user" bar, applied to names
      (c) CROSS(R1) == 0                           - no opponent carries one of our names (by_number has no team
                                                     gate; one such name puts an opponent in a season profile)
      (d) R2 bit on >= 3 of 5 units AND PROP <= K on EVERY unit where it was planted, K = 1
    K = 1 because _drop_duplicates keeps one group per (pipeline team, number): a wrong squad number can
    mislabel at most two groups per clip, the one that wears the number (unavoidable once the roster is wrong)
    and one across the team split. K = 1 refuses the second. No cross-match cascade is possible (point 1);
    R1b checks that.
    NO EFFECT     iff (a) fails.
    NOT SUPPORTED otherwise ((a) holds, (b), (c) or (d) fails). Only SHIP licenses wiring the roster in, and
    then apply_identity must also write name/player_id (see WHAT A ROSTER CHANGES).
    VOID          if an invariant fails (jersey column differs across arms, R1b != R1, merge groups differ) or
                  the pooled join share < 0.5.
    Deterministic replays: the thresholds are product bars, not noise bands. Five units give no significance
    test worth the name; (d)'s count is a consistency rule.

RUN (phase 2; one decode + docTR pass per clip, 7 clips; arms are in-process replays - ~5-10 min a clip is an
estimate, not timed, so ~35-70 min for both splits)
    .venv/bin/python model/validation/n71_roster_seed.py --inputs     # GT only, seconds; no pipeline
    caffeinate -i -s .venv/bin/python model/validation/n71_roster_seed.py --run dev
    caffeinate -i -s .venv/bin/python model/validation/n71_roster_seed.py --run held   # ONCE; refuses if uncommitted
    .venv/bin/python model/validation/n71_roster_seed.py --show
    .venv/bin/python model/validation/n71_roster_seed.py --demo       # the rule's own self-check

Page: n71-roster-seeding (not published)
"""
import collections
import copy
import functools
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, p) for p in ("model/validation", "model/engine", "app/server")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
DEV = (("118575", "31424"), ("118575", "31425"), ("118576", "9834"), ("118576", "31424"), ("117093", "9834"))
HELD = (("118577", "31425"), ("118578", "9834"), ("118578", "30798"), ("128058", "9834"), ("128058", "34317"))
K = 1
GATE_PX, JOIN_MIN_HITS, MIN_SHARE = 30, 20, 0.6        # check.py _gt_diag's join


@functools.cache
def _gt():
    import st2_team_gt as T  # noqa: PLC0415
    return T, T.load()


@functools.cache
def _ros(seq):
    return _gt()[0].roster(seq)


def _num(v):
    return int(v) if str(v).isdigit() else None


def present(seq, team):
    """{ST2 id: GT number} of team `team` among the clip's 22 MOT ids."""
    d = _gt()[1][seq]
    side = _ros(seq)["sides"][team]
    return {int(pid): _num(d["jersey"][m]) for m, pid in d["mot_to_player"].items() if d["team"][m] == side}


def sources(team, target):
    return [s for s in _gt()[0].SEQS if s != target and team in _ros(s)["sides"]]


def build_roster(team, target, bind=True, numbers="sheet"):
    """R1's roster. numbers="sheet": the target match's team-sheet number (None if not on it) - the arm.
    numbers="history": the number last worn in a source match - what --inputs reports, never an arm."""
    from roster import Roster  # noqa: PLC0415
    d = _gt()[1]
    srcs = sources(team, target)
    assert target not in srcs
    sheet = {pid: _num(p["back_number"]) for pid, p in _ros(target)["players"].items() if p["team_id"] == team}
    r = Roster(team=team)
    for s in srcs:
        side, R = _ros(s)["sides"][team], _ros(s)["players"]
        for m, pid in d[s]["mot_to_player"].items():
            if d[s]["team"][m] != side:
                continue
            p = R[int(pid)]
            num = _num(p["back_number"]) if numbers == "history" else sheet.get(int(pid))
            r.add(number=num, name=p["name"], player_id=f"st2-{pid}")
            if numbers == "sheet" and num is None:
                r.players[f"st2-{pid}"].number = None     # add() never clears a number; a later source must
            if bind:
                r.bind(s, int(m), f"st2-{pid}", 1.0, "manual")
    return r


def choose_plant(r1, pres, truth, r0num):
    """R2's rule (docstring). -> (P, X, number, bit) or None."""
    carried = sorted(((c, g) for g, (n, c) in r0num.items()
                      if truth.get(g) in pres and n is not None and n >= 10 and n == pres[truth[g]]), reverse=True)
    if carried:
        P, bit = truth[carried[0][1]], True
    else:
        two = [p for p, n in pres.items() if n is not None and n >= 10]
        if not two:
            return None
        P, bit = min(two), False
    X = min((k for k in r1.players if k != f"st2-{P}"), key=lambda k: int(k[4:]), default=None)
    return (P, X, pres[P], bit) if X else None


def plant(r, X, number):
    r = copy.deepcopy(r)
    r.players[X].number = number
    r.players = {X: r.players[X], **{k: v for k, v in r.players.items() if k != X}}
    return r


def truth_of(seq, out):
    """group id -> ST2 id (or absent = NO-TRUTH); and the joined-detection share."""
    T, d = _gt()
    feet, hit = T.mot_feet(seq), collections.defaultdict(collections.Counter)
    n = joined = 0
    for x in out["detections"]:
        n += 1
        cand = [(abs(u - x["x"]) + abs(v - x["y"]), m) for m, (u, v) in feet.get(int(x["f"]), {}).items()]
        if cand and min(cand)[0] <= GATE_PX:
            joined += 1
            hit[int(x["id"])][min(cand)[1]] += 1
    tr = {}
    for g, c in hit.items():
        m, k = c.most_common(1)[0]
        if k >= JOIN_MIN_HITS and k >= MIN_SHARE * sum(c.values()):
            tr[g] = int(d[seq]["mot_to_player"][str(m)])
    return tr, joined, n


def metrics(ids, truth, pres, X=None):
    named = {p.track_group: p.player_id for p in ids.values() if p.player_id}
    right = {g for g, pid in named.items() if g in truth and f"st2-{truth[g]}" == pid}
    wrong = set(named) - right
    cross = sum(1 for g in wrong if g in truth and truth[g] not in pres)
    notruth = sum(1 for g in wrong if g not in truth)
    return {"present": len(pres), "PR": len({truth[g] for g in right}), "GN": len(named), "RG": len(right),
            "WN": len(wrong), "CROSS": cross, "NOTRUTH": notruth, "SAME": len(wrong) - cross - notruth,
            "wrongX": sum(1 for g in wrong if named[g] == X) if X else 0}


def _sig(ids):
    return {t: (p.number, p.player_id, p.source, p.track_group) for t, p in ids.items()}


def run_clip(seq, teams):
    import identity  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    if "jobs" not in sys.modules:           # throwaway store, check.py _jobs()'s rule
        import tempfile  # noqa: PLC0415
        os.environ["FUTTRAFFIC_DATA"] = tempfile.mkdtemp(prefix="n71-")
    import jobs  # noqa: PLC0415
    os.environ.pop("FUTTRAFFIC_IDENTITY", None)
    with open(os.path.join(STC, f"{seq}.p35.tracks.json")) as fh:
        base = json.load(fh)
    cal = jobs.load_calibration(os.path.join(STC, f"{seq}.p35.calibration.json"))
    video = f"{ST.CLIPS}/{seq}.mp4"
    memo, real_scan, real_get = {}, identity.scan_video, identity.get_identity_strategy

    def scan(*a, **k):
        if "v" not in memo:
            memo["v"] = real_scan(*a, **k)
        return copy.deepcopy(memo["v"])

    def arm(roster):
        box = {}

        def get(name, **kw):
            s = real_get(name, **({"roster": roster, "match_id": seq} if roster is not None else {}), **kw)
            a = s.assign
            s.assign = lambda t, v=None: box.setdefault("ids", a(t, v))
            return s
        identity.scan_video, identity.get_identity_strategy = scan, get
        try:
            out, rep = jobs.apply_identity(identity, copy.deepcopy(base), video, cal)
        finally:
            identity.scan_video, identity.get_identity_strategy = real_scan, real_get
        return out, box["ids"]

    jersey = lambda o: [(x["id"], x.get("jersey")) for x in o["detections"]]  # noqa: E731
    out0, ids0 = arm(None)
    truth, joined, n = truth_of(seq, out0)
    r0num = {}
    for x in out0["detections"]:
        if x.get("jersey") is not None:
            r0num[int(x["id"])] = (x["jersey"], x["jersey_conf"])
    res = {"joined": joined, "dets": n, "units": {}}
    for team in teams:
        pres = present(seq, team)
        r1 = build_roster(team, seq)
        out1, ids1 = arm(copy.deepcopy(r1))
        _, ids1b = arm(build_roster(team, seq, bind=False))
        pl = choose_plant(r1, pres, truth, r0num)
        u = {"sources": sources(team, seq), "squad": len(r1.players), "void": [],
             "r0": {"carried": len({truth[g] for g, (nn, _c) in r0num.items() if g in truth and truth[g] in pres
                                    and nn == pres[truth[g]]}),
                    "wrong_numbers": sum(1 for g, (nn, _c) in r0num.items()
                                         if g in truth and truth[g] in pres and nn != pres[truth[g]])},
               "plant": None}
        if jersey(out1) != jersey(out0):
            u["void"].append("jersey column R1 != R0")
        if _sig(ids1) != _sig(ids1b):
            u["void"].append("R1b (no source bindings) != R1")
        X = None
        if pl:
            P, X, number, bit = pl
            out2, ids2 = arm(plant(r1, X, number))
            if jersey(out2) != jersey(out0):
                u["void"].append("jersey column R2 != R0")
            u["plant"] = {"P": P, "X": X, "number": number, "bit": bit}
            u["r2"] = metrics(ids2, truth, pres, X)
        u["r1"] = metrics(ids1, truth, pres, X)
        res["units"][team] = u
    return res


def run(which):
    import protocol  # noqa: PLC0415
    import verify_claims as V  # noqa: PLC0415
    me = "model/validation/n71_roster_seed.py"
    dst = os.path.join(STC, f"n71_{which}.json")
    if which == "held":
        env = V.git_env()
        tracked = not subprocess.run(["git", "ls-files", "--error-unmatch", me], cwd=ROOT, env=env,
                                     capture_output=True).returncode
        if not tracked or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", me], cwd=ROOT, env=env).returncode:
            raise SystemExit("VOID: the held-out run must be made by the committed protocol")
        if os.path.exists(dst):
            raise SystemExit(f"HELD is scored once and {dst} exists")
    units = HELD if which == "held" else DEV
    by = collections.defaultdict(list)
    for seq, team in units:
        by[seq].append(team)
    res = {"protocol": protocol.protocol_hash(__doc__), "clips": {}}
    for seq, teams in by.items():
        res["clips"][seq] = run_clip(seq, teams)
        with open(dst, "w") as fh:
            json.dump(res, fh, indent=1)
    show()


def _flat(r):
    return [u for c in r["clips"].values() for u in c["units"].values()]


def verdict(r):
    us = _flat(r)
    joined = sum(c["joined"] for c in r["clips"].values()) / max(1, sum(c["dets"] for c in r["clips"].values()))
    if joined < 0.5 or any(u["void"] for u in us):
        return "VOID", []
    s = lambda k: sum(u["r1"][k] for u in us)  # noqa: E731
    planted = [u for u in us if u["plant"]]
    ok = [s("PR") >= 5,
          s("GN") > 0 and s("RG") / s("GN") >= 0.90,
          s("CROSS") == 0,
          sum(u["plant"]["bit"] for u in planted) >= 3
          and all(u["r2"]["wrongX"] - u["r1"]["wrongX"] <= K for u in planted)]
    if all(ok):
        return "SHIP", ok
    if not ok[0]:
        return "NO EFFECT", ok
    return "NOT SUPPORTED", ok


def show():
    for which in ("dev", "held"):
        p = os.path.join(STC, f"n71_{which}.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            r = json.load(fh)
        print(f"== {which.upper()} (protocol {r['protocol']}) ==")
        for seq, c in r["clips"].items():
            print(seq, f"joined {c['joined']}/{c['dets']}")
            for team, u in c["units"].items():
                print(" ", team, "sources", u["sources"], "squad", u["squad"], "R0", u["r0"], "R1", u["r1"],
                      "plant", u["plant"], "R2", u.get("r2"), "VOID" if u["void"] else "", *u["void"])
        if which == "held":
            v, ok = verdict(r)
            print("RULE:", dict(zip("abcd", ok)), "->", v)


def inputs():
    """GT only: what each unit's R1 roster holds, before any pipeline output exists."""
    for name, units in (("DEV", DEV), ("HELD", HELD)):
        tot = collections.Counter()
        for seq, team in units:
            pres, r = present(seq, team), build_roster(team, seq, numbers="history")
            r1 = build_roster(team, seq)
            assert not any(n is not None and (q := r1.by_number(n)) and q.player_id != f"st2-{pid}"
                           for pid, n in pres.items()), "R1's sheet roster maps a number to the wrong man"
            ours = {int(k[4:]): p.number for k, p in r.players.items()}
            row = {"present": len(pres), "known": sum(1 for pid in pres if pid in ours),
                   "stale": sum(1 for pid, n in pres.items() if pid in ours and ours[pid] != n),
                   "clash": sum(1 for pid, n in pres.items()
                                if n is not None and (q := r.by_number(n)) and q.player_id != f"st2-{pid}")}
            tot.update(row)
            print(seq, team, f"sources {len(sources(team, seq))} squad {len(r.players)} present {row['present']} "
                  f"in-roster {row['known']} stale-number {row['stale']} number-maps-to-someone-else {row['clash']}")
        print(f"{name} total: in-roster {tot['known']} of {tot['present']}, stale {tot['stale']}, "
              f"maps-elsewhere {tot['clash']}")


def demo():
    from identity import PlayerId  # noqa: PLC0415
    pres = {1: 10, 2: 23}
    truth = {100: 1, 200: 2, 300: 9}                          # 9 = other team
    ids = {100: PlayerId("0", 10, "a", "st2-1", .9, "roster", 100),
           200: PlayerId("0", 23, "b", "st2-7", .9, "roster", 200),
           300: PlayerId("1", 10, "a", "st2-1", .9, "roster", 300),
           400: PlayerId("1", 44, "c", "st2-5", .9, "roster", 400)}
    m = metrics(ids, truth, pres, X="st2-1")
    assert (m["PR"], m["GN"], m["RG"], m["WN"], m["CROSS"], m["SAME"], m["NOTRUTH"], m["wrongX"]) == \
        (1, 4, 1, 3, 1, 1, 1, 1), m

    def mk(pr, gn, rg, cross, bit=(1, 1, 1, 0, 0), prop=(0, 1, 0, 1, 0)):
        us = {str(i): {"void": [], "r1": {"PR": pr[i], "GN": gn[i], "RG": rg[i], "CROSS": cross[i], "wrongX": 0},
                       "plant": {"bit": bool(bit[i])}, "r2": {"wrongX": prop[i]}} for i in range(5)}
        return {"clips": {"c": {"joined": 9, "dets": 10, "units": us}}}
    good = mk((2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (0,) * 5)
    assert verdict(good)[0] == "SHIP"
    assert verdict(mk((1, 1, 1, 1, 0), (1,) * 5, (1,) * 5, (0,) * 5))[0] == "NO EFFECT"                  # (a)
    assert verdict(mk((2, 1, 1, 1, 0), (4, 1, 1, 1, 0), (2, 1, 1, 1, 0), (0,) * 5))[0] == "NOT SUPPORTED"  # (b)
    assert verdict(mk((2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (0, 0, 0, 0, 0),
                      prop=(0, 2, 0, 0, 0)))[0] == "NOT SUPPORTED"                                            # (d) K
    assert verdict(mk((2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (0,) * 5,
                      bit=(1, 1, 0, 0, 0)))[0] == "NOT SUPPORTED"                                             # (d) untested
    bad = mk((2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (2, 1, 1, 1, 0), (0,) * 5)
    bad["clips"]["c"]["joined"] = 1
    assert verdict(bad)[0] == "VOID"
    print("n71 rule ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--run" in a:
        run(a[a.index("--run") + 1])
    elif "--inputs" in a:
        inputs()
    elif "--demo" in a:
        demo()
    else:
        show()
