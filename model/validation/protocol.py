"""Pre-registration the repository can prove, instead of pre-registration it asserts.

WHY
    Rule 4 of CLAUDE.md is "define success criteria BEFORE running". Every
    producer in this repo says it did: "PASS RULE, fixed before running". On
    2026-09-12 a reviewer checked, and git could prove it for exactly one of the
    day's measurements. Every other pass rule landed in the SAME commit as the
    number it judged, so "before" was an honour system - and one decision's
    bootstrap seed was fixed in the same commit as the result it decided.
    harness-design (not published) §E.

WHAT IT PROVES
    For a pass rule and the result it judges, the first commit that introduced
    each, by `git log -S` on the file that holds it. Then:

      PROVEN        the rule's commit is a strict ancestor of the result's, AND the
                    rule's paragraph reads the same at both commits
      RULE_CHANGED  the rule came first, but the paragraph holding it was edited
                    before the result landed - order proves nothing about a rule
                    that was loosened after the run (fails the gate)
      SAME_COMMIT   they landed together - consistent with honesty, provable by
                    nothing, and the state of almost everything today
      AFTER         the rule landed after the result: it was written to fit
      MISSING       a needle was not found, so nothing was checked

    SAME_COMMIT is not an accusation. It is the absence of proof, which is the
    whole distinction rule 4 needs and git can supply for free. It FAILS the gate
    for a result page added on or after REGISTERED_FROM; older pages are
    grandfathered by date, the same rule `unregistered()` applies.

WHY THE PARAGRAPH AND NOT THE FILE
    "unchanged between rule and result" was measured both ways on the registry
    (2026-09-29): a whole-file `git diff rule..result -- file` is non-empty for 17
    of the 32 PROVEN entries, because a rule usually lives in the producer's
    docstring and the producer is then written. That test fails the honest case
    and would be bypassed. The needle's paragraph (blank-line delimited,
    whitespace-normalised through protocol_hash) is the rule text itself, and
    editing it is exactly the "loosened after the result" shape.

WHAT IT CANNOT PROVE
    That nobody ran the producer, looked, and then committed the rule. A rule
    committed first is evidence of order, not of blindness. The protocol hash
    exists for the stronger form: a run records `protocol_hash()` of the text it
    ran under, so an artefact carries which rule it was judged against.

    .venv/bin/python model/validation/protocol.py          # report on the registered measurements
    .venv/bin/python model/validation/protocol.py --demo
"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "validation"))

import verify_claims as V  # noqa: E402 - git_env, reused and never copied

# Each entry: the pass rule (file, distinctive text) and the result it judges
# (file, distinctive text). Text is matched with `git log -S`, so it must be
# specific enough to appear exactly when the rule or the result did.
REGISTRY = [
    ("N11 sprints/HSR", "model/validation/dist_eval.py", "N11 PASSES",
     "docs/experiments/n8-n11-downstream.md", "4.2337"),
    ("N12 calibration", "model/validation/calib_product.py", "click precision is a product lever",
     "docs/experiments/n12-calibration.md", "8.27"),
    ("N8 identity merge", "model/validation/contam_eval.py", "the claim under test is N6's",
     "docs/experiments/n8-n11-downstream.md", "37.3%"),
    ("N17 conf", "model/validation/st2_product.py", "MOVE the product to conf 0.3 iff",
     "docs/experiments/n17-conf.md", "0.2109"),
    ("tile overlap revert", "model/engine/detsweep.py", "PASS RULE, and it decides in BOTH directions",
     "docs/experiments/tile-overlap.md", "0.0756"),
    ("N21 product as it ships", "model/validation/st2_product.py", "N21 is VOID",
     "docs/experiments/n21-product-now.md", "0.3605"),
    ("N7 metric stitch", "model/validation/st2_product.py", "N7 is VOID",
     "docs/experiments/n21-product-now.md", "332.4"),
    ("N22 identity or position", "model/validation/dist_eval.py", "IDENTITY points the next project",
     "docs/experiments/n22-what-inflates-distance.md", "0.3053"),
    ("N23 detector or map", "model/validation/dist_eval.py", "DETECTOR points position work at foot points",
     "docs/experiments/n23-detector-or-map.md", "0.7787"),
    ("N24 crediting", "model/validation/dist_eval.py", "RUNNING `ratio_matched` at most 1.05",
     "docs/experiments/n24-crediting.md", "0.0326"),
    ("N25 speed estimator", "model/validation/dist_eval.py",
     "ESTIMATOR puts the filter in the product's speed series next",
     "docs/experiments/n25-speed-estimator.md", "-0.1768"),
    ("N26 map oracle", "model/validation/dist_eval.py", "REVISED 2026-09-14 BEFORE THE FIRST RUN",
     "docs/experiments/n26-map-oracle.md", "1.0049"),
    ("N27 correction in the product", "model/validation/dist_eval.py", "SHIP FOR STITCHED PANORAMAS iff all of",
     "docs/experiments/n27-tps-in-product.md", "0.1272"),
    ("N28 plausibility", "model/validation/plausibility.py", "-> BLIND TO THE PRODUCT",
     "docs/experiments/n28-plausibility.md", "8.6992"),
    # The G0 entry's values are the floor BEFORE the label offset, and step 6 now tags
    # them as cited on our own past. The rule that produced the live numbers is the LABEL
    # OFFSET addendum, so it gets its own entry - without it protocol.py would keep
    # proving a withdrawn result pre-registered.
    ("ST2 P1 step 6 G0 + floor (pre-offset)", "model/validation/st2_index.py",
     "G0 ALIGNED on a val match iff", "docs/experiments/st2-p1-step6.md", "0.774,0.529,0.663"),
    ("ST2 P1 step 6 label offset", "model/validation/st2_index.py", "LABEL OFFSET",
     "docs/experiments/st2-p1-step6.md", "0.888,0.912"),
    ("ST2 P1 step 7 G2 overfit", "model/validation/st2_train.py", "PASS      mAP@1s > 0.80 on that same half",
     "docs/experiments/st2-p1-step7.md", "0.970,1134,0.0035"),
    ("ST2 P1 step 7 attribution", "model/validation/st2_train.py",
     "is ATTRIBUTED CORRECTLY iff that", "docs/experiments/st2-p1-step7.md", "1129,1134,0.998,0.878,0.045"),
    ("N51 joint gate+path", "model/validation/ball_gate.py",
     "THE JOINT ARM", "docs/experiments/n51-joint-gate-and-path.md", "0.3344"),
    ("N50 ball size gate", "model/validation/ball_gate.py",
     "LEAVE-ONE-CLIP-OUT over the", "docs/experiments/n50-ball-size-gate.md", "0.3772"),
    ("N49 ball path", "model/validation/ball_path.py",
     "pooled `path` >= pooled `shipped` + 0.05", "docs/experiments/n49-ball-path.md", "0.3856"),
    ("N49 arm D path", "model/validation/ball_path.py",
     "SECOND RULE, ARM D", "docs/experiments/n49-ball-path.md", "-0.0106"),
    ("N56 penalty-area pass", "model/validation/n56_goal_roi.py",
     "the worst id's recall rises by >= +0.15", "docs/experiments/n56-goal-roi.md", "+0.1021"),
    ("N57 dead-ball split", "model/validation/n57_conf_evidence.py",
     "dead-ball if the GT ball's mean speed", "docs/experiments/n57-confidence-is-evidence.md", "0.341"),
    ("N57 confidence AUC", "model/validation/n57_conf_evidence.py",
     "AUC >= 0.55", "docs/experiments/n57-confidence-is-evidence.md", "0.8895"),
    ("ST2 P1 step 8 G3 budget", "model/validation/st2_train.py",
     "BUDGET IS EQUAL OPTIMISER STEPS, NOT EQUAL EPOCHS",
     "docs/experiments/st2-p1-step8.md", "12,170294,28.5"),
    # N61's buckets and its page landed together, so this entry reads SAME_COMMIT and
    # is meant to: the audit is descriptive, and the thing worth pre-registering about
    # it was that the bucket edges could not be tuned after the counts were seen.
    ("N61 suppressed-box buckets", "model/validation/n61_unlabelled_audit.py",
     "THE BUCKETS, fixed here so the counting cannot be tuned afterwards",
     "docs/experiments/n61-what-was-suppressed.md", "19.8"),
    ("N62 proposed or classified down", "model/validation/n62_query_probe.py",
     "share >= 0.80  CLASSIFIED DOWN",
     "docs/experiments/n62-classified-down.md", "0.0401"),
    ("N63 keeper composition", "model/validation/n63_train_composition.py",
     "B >= 2*A/3 and count >= FLOOR_N", "docs/experiments/n63-keepers-were-in-the-training-set.md", "7816"),
    ("N59 gate subsets", "model/validation/n59_gates.py",
     "the held-out total beats `shipped` by >= +0.05 pooled",
     "docs/experiments/n59-gates-earn-their-place.md", "0.4450"),
    ("N58 player prior", "model/validation/n58_player_prior.py",
     "pooled PRIOR >= CONF + 0.05", "docs/experiments/n58-player-prior.md", "0.2944"),
    # N64 is PROVEN on order and still does not support its own hypothesis: the rule
    # landed first, the arm cleared the bar, and the page records that the intervention
    # tested was not the one the rule described. Pre-registration proves order, not
    # that the treatment was what the rule called it.
    ("N64 hard-negative arm E1", "model/validation/n64_ball_negatives.py",
     "pooled `shipped` >= 0.4706 + 0.05 = 0.5206",
     "docs/experiments/n64-e1-passed-the-bar-on-a-different-experiment.md", "0.5222"),
    # N54 is another session's page and I would not invent a rule commit for it - but it
    # HAS one, in its own text, and rule and result landed together, so the honest entry
    # reads SAME_COMMIT and does not fail. Registering it is what lets UNREGISTERED
    # become blocking above.
    # N65's rule V1 was vacuous and V2 supersedes it; the entry names V2's text, which is
    # what the number was judged against. V1 is recorded on the page, not in the registry.
    ("N64 E2 paired arm", "model/validation/n64_ball_negatives.py",
     "MECHANISM   FALSE CANDIDATES PER FRAME AT MATCHED RECALL",
     "docs/experiments/n67-fewer-candidates-does-not-help-the-picker.md", "43.4"),
    ("N65 input resolution", "model/validation/n64_ball_negatives.py",
     "if pooled `conf` OR pooled `oracle` moves by more than 0.01",
     "docs/experiments/n65-every-ball-number-was-measured-at-384.md", "0.5850"),
    ("N54 near-touchline arms", "docs/experiments/n54-near-profile.md",
     "wins if lower by >= 0.25 px", "docs/experiments/n54-near-profile.md", "0.25 px"),
    ("N69 tracker frame rate", "model/validation/n69_frame_rate.py",
     "mean HOTA delta >= +0.010", "docs/experiments/n69-tracker-frame-rate.md", "0.3639,0.3697"),
    ("N68 128057 length fixes", "docs/experiments/n68-128057-length.md",
     "Pass rule for testing any of them (pre-registered here, before any is written)",
     "docs/experiments/n68-128057-length.md", "**n=3 fix arms tried.**"),
    ("N70 class-aware merge", "model/validation/n70_class_merge.py",
     "SHIP M1 iff ALL of", "docs/experiments/n70-class-aware-merge.md", "0.3124,0.3366,+0.0242"),
]


def _git(*args, cwd=ROOT) -> subprocess.CompletedProcess:
    """git in `cwd`, and ONLY in `cwd`.

    A pre-push hook runs with GIT_DIR (and GIT_INDEX_FILE, GIT_WORK_TREE...)
    exported, and those BEAT cwd: on 2026-09-19 this file's demo, run from the
    push gate, committed its seven specimen commits onto the branch that was
    checked out instead of into its own temp repo. The repo is the directory,
    never the ambient environment.
    """
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          env=V.git_env())


def _names(path: str, cwd=ROOT) -> list[str]:
    # Every name the file has had. `-S` does not follow renames (and `--follow`
    # with `-S` finds nothing), so a moved file would date its rule to the move.
    names = _git("log", "--follow", "--name-only", "--format=", "--", path, cwd=cwd).stdout.splitlines()
    return list(dict.fromkeys([path, *names]))


def first_commit(path: str, needle: str, cwd=ROOT) -> str | None:
    out = _git("log", "-S", needle, "--reverse", "--format=%H", "--", *_names(path, cwd),
               cwd=cwd).stdout.split()
    return out[0] if out else None


def rule_text(commit: str, path: str, needle: str, cwd=ROOT) -> list[str]:
    """Hashes of every paragraph holding `needle` in the rule file at `commit`.

    Empty when the needle is gone - which then differs from the rule commit's
    non-empty list, so a deleted rule is a changed rule.
    """
    for p in _names(path, cwd):
        r = _git("show", f"{commit}:{p}", cwd=cwd)
        if r.returncode == 0 and needle in r.stdout:
            return [protocol_hash(b) for b in re.split(r"\n\s*\n", r.stdout) if needle in b]
    return []


def added(rel: str, cwd=ROOT, fmt="%ad") -> str:
    """When `rel` was first added, following renames: a page moved into a docs
    subfolder is otherwise "added" by the move and loses its grandfathering."""
    out = _git("log", "--follow", "--diff-filter=A", f"--format={fmt}", "--date=short",
               "--", rel, cwd=cwd).stdout
    return (out.splitlines() or [""])[-1].strip()


def experiment_pages(root=ROOT) -> list[Path]:
    """docs/**/nNN-*.md at any depth, the memory store excepted."""
    docs = Path(root) / "docs"
    return sorted(p for p in docs.rglob("n[0-9]*.md") if p.relative_to(docs).parts[0] != "memory")


def preregistered(rule_path, rule_needle, result_path, result_needle, cwd=ROOT) -> tuple[str, str | None, str | None]:
    rule = first_commit(rule_path, rule_needle, cwd)
    result = first_commit(result_path, result_needle, cwd)
    if rule is None or result is None:
        return "MISSING", rule, result
    if rule == result:
        return "SAME_COMMIT", rule, result
    if _git("merge-base", "--is-ancestor", rule, result, cwd=cwd).returncode == 0:
        if rule_text(rule, rule_path, rule_needle, cwd) != rule_text(result, rule_path, rule_needle, cwd):
            return "RULE_CHANGED", rule, result
        return "PROVEN", rule, result
    return "AFTER", rule, result


FAILS = {"AFTER", "MISSING", "RULE_CHANGED"}

#: RECORDED, NOT HIDDEN (author decision, 2026-09-29). The five entries whose
#: pre-registration git cannot prove, found the day RULE_CHANGED and the
#: SAME_COMMIT cut-off landed. Each is reported as NOT PROVEN (recorded) and does
#: not block; its verdict stands as weaker evidence, and each page carries a
#: dated banner saying so. An entry must match EXACTLY - same verdict, same
#: rule and result commits - or the run fails: a record that no longer
#: describes the repo is a stale claim, and this list must not rot into a
#: blanket exemption. Nothing is added here without the same review.
NOT_PROVABLE = {
    "N49 ball path": ("RULE_CHANGED", "rule paragraph reworded (a derived tag removed) before the result",
                      "441c1eb", "03de368"),
    "N64 hard-negative arm E1": ("RULE_CHANGED", "VOID (a)-(e) list no longer in the rule paragraph at the result",
                                 "106ef74", "2e84020"),
    "N64 E2 paired arm": ("RULE_CHANGED", "E2 VOID list added to the rule paragraph before the result",
                          "e9eb4e3", "5eefa62"),
    "N61 suppressed-box buckets": ("SAME_COMMIT", "buckets and counts landed in one commit, after REGISTERED_FROM",
                                   "30d9113", "30d9113"),
    "N54 near-touchline arms": ("SAME_COMMIT", "rule and result landed in one commit, on REGISTERED_FROM",
                                "363e88a", "363e88a"),
}


def recorded(label: str, status: str, rule, result, table=None) -> str | None:
    """None (not listed), "RECORDED" (listed and still exactly true) or "STALE"."""
    table = NOT_PROVABLE if table is None else table
    if label not in table:
        return None
    verdict, _why, r, d = table[label]
    ok = status == verdict and (rule or "").startswith(r) and (result or "").startswith(d)
    return "RECORDED" if ok else "STALE"


def blocks(status: str, result_path: str, cwd=ROOT) -> bool:
    """Does this verdict fail the gate? SAME_COMMIT does for a page born on or
    after REGISTERED_FROM - the registry existed then, so proof was available."""
    if status == "SAME_COMMIT":
        born = added(result_path, cwd)
        return bool(born) and born >= REGISTERED_FROM
    return status in FAILS


def protocol_hash(text: str) -> str:
    """What a run stamps into its artefact: the rule it was judged against."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:12]


def demo():
    def commit(repo, name, body, msg):
        (repo / name).write_text(body)
        _git("add", name, cwd=repo)
        _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", msg, cwd=repo)

    with tempfile.TemporaryDirectory() as t:
        repo = Path(t)
        _git("init", "-q", cwd=repo)

        # rule first, result later -> PROVEN
        commit(repo, "rule.py", "PASS RULE: ships iff x > 1\n", "rule")
        commit(repo, "doc.md", "result 1.42\n", "result")
        assert preregistered("rule.py", "ships iff", "doc.md", "1.42", repo)[0] == "PROVEN"

        # rule and result together -> SAME_COMMIT, never PROVEN
        (repo / "rule2.py").write_text("PASS RULE: keep iff y\n")
        (repo / "doc2.md").write_text("result 0.77\n")
        _git("add", "rule2.py", "doc2.md", cwd=repo)
        _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "both", cwd=repo)
        assert preregistered("rule2.py", "keep iff", "doc2.md", "0.77", repo)[0] == "SAME_COMMIT"

        # result first, rule written afterwards to fit it -> AFTER
        commit(repo, "doc3.md", "result 0.31\n", "result first")
        commit(repo, "rule3.py", "PASS RULE: good iff z < 0.4\n", "rule fitted after")
        assert preregistered("rule3.py", "good iff", "doc3.md", "0.31", repo)[0] == "AFTER"

        # THE LOOSENED RULE: committed first, edited before the result landed.
        # Order alone reads PROVEN; the paragraph test must refuse it.
        commit(repo, "rule4.py", "PASS RULE: ships iff w > 0.05\n", "rule4")
        commit(repo, "rule4.py", "PASS RULE: ships iff w > 0.01\n", "loosen rule4")
        commit(repo, "doc4.md", "result 0.03\n", "result4")
        assert preregistered("rule4.py", "ships iff", "doc4.md", "0.03", repo)[0] == "RULE_CHANGED", \
            "a rule loosened between its commit and the result's still reads PROVEN"
        # ...and an unrelated edit elsewhere in the file does not
        commit(repo, "rule5.py", "PASS RULE: ok iff v > 2\n", "rule5")
        commit(repo, "rule5.py", "PASS RULE: ok iff v > 2\n\ndef producer(): pass\n", "code")
        commit(repo, "doc5.md", "result 2.9\n", "result5")
        assert preregistered("rule5.py", "ok iff", "doc5.md", "2.9", repo)[0] == "PROVEN"

        # SAME_COMMIT blocks for a page born on/after REGISTERED_FROM, not before
        assert blocks("SAME_COMMIT", "doc2.md", repo), "a new SAME_COMMIT page passed the gate"
        (repo / "old.md").write_text("result 0.5\n")
        _git("add", "old.md", cwd=repo)
        _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--date=2026-09-01T00:00:00",
             "-m", "old", cwd=repo)
        assert not blocks("SAME_COMMIT", "old.md", repo), "a grandfathered page blocked"
        assert blocks("RULE_CHANGED", "old.md", repo) and not blocks("PROVEN", "old.md", repo)

        # NOT_PROVABLE: listed and exact reports without blocking; unlisted still
        # blocks; a listed entry whose condition changed is STALE, never silent.
        tab = {"X": ("RULE_CHANGED", "why", "aaaaaaa", "bbbbbbb")}
        assert recorded("Y", "RULE_CHANGED", "aaaaaaa1", "bbbbbbb1", tab) is None, "an unlisted case was excused"
        assert recorded("X", "RULE_CHANGED", "aaaaaaa1", "bbbbbbb1", tab) == "RECORDED"
        assert recorded("X", "PROVEN", "aaaaaaa1", "bbbbbbb1", tab) == "STALE", "a record that rotted passed"
        assert recorded("X", "RULE_CHANGED", "ccccccc1", "bbbbbbb1", tab) == "STALE", "commits moved, record kept"

        # a needle that is not there checks nothing, and says so
        assert preregistered("rule.py", "no such rule", "doc.md", "1.42", repo)[0] == "MISSING"

        # moving the rule's file later must not re-date the rule to the move
        (repo / "sub").mkdir()
        _git("mv", "rule.py", "sub/rule.py", cwd=repo)
        _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "move", cwd=repo)
        assert preregistered("sub/rule.py", "ships iff", "doc.md", "1.42", repo)[0] == "PROVEN"

        # an experiment page moved into a docs subfolder is still found, and still
        # dated by its first add, not by the move
        (repo / "docs" / "memory").mkdir(parents=True)
        commit(repo, "docs/n1-x.md", "result 2.5\n", "add n1")
        commit(repo, "docs/memory/n2-note.md", "note\n", "memory note")
        (repo / "docs" / "experiments").mkdir()
        _git("mv", "docs/n1-x.md", "docs/experiments/n1-x.md", cwd=repo)
        _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "move n1", cwd=repo)
        assert [p.relative_to(repo).as_posix() for p in experiment_pages(repo)] == \
            ["docs/experiments/n1-x.md"], "the experiment scan missed a subfolder"
        assert added("docs/experiments/n1-x.md", repo, "%s") == "add n1", "a move re-dated the page"

    # the hash ignores reflowed whitespace but not a changed rule
    assert protocol_hash("ships iff  x > 1") == protocol_hash("ships iff\nx > 1")
    assert protocol_hash("ships iff x > 1") != protocol_hash("ships iff x > 2")
    print("protocol.py ok")


#: An entry is added when the experiment has a RESULT: the registry maps a rule
#: commit to a result commit, and a pending arm has no second half - entering one
#: early reports MISSING and fails the gate for an experiment that is behaving
#: correctly. N58 and N59 are pre-registered and unscored; they enter when scored,
#: and `unregistered()` below is what stops that being forgotten.
#: Experiment pages from this date on must appear in REGISTRY. Older ones are
#: grandfathered BY DATE rather than by a baseline file: the registry was not kept
#: for most of them, and a check that fails on forty pages is a check nobody runs.
#: The date is the day this was written, so it grandfathers nothing new.
REGISTERED_FROM = "2026-09-23"


def unregistered() -> list[str]:
    """Experiment pages added recently with no REGISTRY entry.

    The registry went green on 18 PROVEN while saying nothing at all about N56,
    N57 and N58, because an experiment nobody registers is invisible to it - the
    green gate that did not look, arriving through the registry instead of
    through GIT_DIR (a-green-gate-is-not-evidence-it-looked (internal note)).
    """
    known = {dp for _l, _rp, _rn, dp, _dn in REGISTRY}
    out = []
    for f in experiment_pages():
        rel = f.relative_to(ROOT).as_posix()
        if rel in known:
            continue
        born = added(rel)
        if not (born and born >= REGISTERED_FROM):
            continue
        # A PAGE WITH NO MEASURED TAG IS NOT AN EXPERIMENT. N55 is a reading of
        # other people's results and states plainly that it measures nothing; it
        # has no rule to pre-register and demanding one would teach people to
        # write fake rules to get green.
        import verify_claims as V  # noqa: PLC0415
        if not any(c.file == rel and c.runnable for c in V.scan()):
            continue
        out.append(f"{rel} (added {born})")
    return out


def main() -> int:
    if "--demo" in sys.argv:
        demo()
        return 0
    counts, failed = {}, 0
    for label, rp, rn, dp, dn in REGISTRY:
        status, rule, result = preregistered(rp, rn, dp, dn)
        counts[status] = counts.get(status, 0) + 1
        rec = recorded(label, status, rule, result)
        bad = rec == "STALE" or (rec is None and blocks(status, dp))
        failed += bad
        short = lambda c: c[:7] if c else "-"  # noqa: E731
        note = {"RECORDED": "  NOT PROVEN (recorded: " + NOT_PROVABLE.get(label, ("", ""))[1] + ")",
                "STALE": "  STALE RECORD: NOT_PROVABLE no longer matches - update or remove it"}.get(rec, "")
        print(f"{'!!' if bad else '  '}{status:12s} {label:22s} rule {short(rule)}  result {short(result)}{note}")
    print("\n" + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
          + f"; {failed} blocking")
    miss = unregistered()
    for m in miss:
        print(f"  UNREGISTERED  {m} - add it to REGISTRY or it proves nothing")
    if miss:
        print(f"{len(miss)} experiment page(s) since {REGISTERED_FROM} carry no "
              f"pre-registration entry. A registry that does not list an experiment "
              f"is green about everything except the experiment.")
    # AFTER is a rule fitted to its result; MISSING means the registry is stale; and
    # UNREGISTERED now FAILS too. It used to print and exit 0 inside a pre-push block
    # that treats this file as blocking, so docs/experiments/n54-near-profile.md sat as a permanent
    # warning - and a warning that is always on is not a gate, it is furniture that the
    # next unregistered page hides behind.
    return 1 if failed or miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
