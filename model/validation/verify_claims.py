"""Three characters in front of every number, and a checker that re-runs the M ones.

WHY THIS EXISTS
    Seven errors shipped into commits and docs on 2026-09-08. They look like seven
    different mistakes and they are one: nothing in the prose distinguishes a number
    this repo produced from a number it read somewhere from a number it computed on
    top of one of those. All three get written in the same voice, and the voice is
    confident.

        an accusation from a research summary, relayed as fact
        one project's hyperparameters attributed to another's run
        arithmetic on a boolean flag, quoted as a config value
        one clip's recall carried into a claim about a different clip
        a published table read backwards
        a SUM reported in a row of means
        24 passes over the test set, best one reported as a finding

    Every one of them is legible the moment the sentence has to say where the number
    came from. None of them is legible without it.

--------------------------------------------------------------------------------
THE CONVENTION
--------------------------------------------------------------------------------
One line. Sigil, value, `::`, source. It goes in markdown, in docstrings, in commit
messages, in a table cell, anywhere.

    @M <value[,value...]> :: <shell command>   what was held fixed
    @C <value>            :: <primary source you opened>   what the number is OF
    @D <value>            :: <the literal arithmetic>   [@M/@C ids it consumes]

    The command ENDS at the first run of two spaces outside quotes; what follows
    on the line is prose (cut_cmd). A command that fails `sh -n` is SYNTAX and
    never runs; one that exits non-zero is ERROR whatever it printed. An integer
    of under 3 digits must be labelled in the output (`n=7`, `hits: 7`) or the
    tag declares the label printed before it: `hits=7`, `of 7`, `IDSW 29`, or
    `"a, b ->" 7` quoted when it holds a comma (split_label). A `-` right after a
    digit is a range (`1.7-5.0` prints 5.0); an integer matches `N.000`
    (token_present). Commas outside double quotes always separate values.

    M  MEASURED   a command, in this repo, today, that prints that value.
    C  CITED      read from an external paper, repo, or leaderboard.
    D  DERIVED    arithmetic or inference on top of the above.

FOUR MORE SIGILS EXIST, AND THE CHECKER WAS BLIND TO ALL FOUR
    They were invented in the docs after this file was written, and by
    2026-09-10 they had been used 460 times between them:

        @V    384   "verified by a command reproduced inline"   = @M
        @RAN   38   "I executed this and pasted the output"     = @M
        @REC   30   "recorded at file:line, NOT re-measured"    = @C on our own past
        @DER    8   "arithmetic on @RAN values"                 = @D

    They are the three sigils above under different names, so they are accepted
    as aliases rather than deprecated: the parser meets the docs where they are.
    `@V` is the one that mattered. It reads to a reader as STRONGER than @M -
    "verified" - and it was the only sigil the tooling had never heard of, so
    the 384 places most likely to be trusted were the 384 places nothing looked.

SIX RULES, AND EVERY ONE OF THEM IS A HABIT, NOT A FEATURE

 1. A number on a shipped surface carries exactly one sigil. No sigil, the number
    does not exist; delete it or tag it. This is the whole mechanism. The rest is
    bookkeeping.

    AND WHEN YOU WRITE ABOUT A MARKER, NAME IT IN WORDS RATHER THAN TYPING IT.
    "a bare measured marker", not the marker itself. A sigil in a sentence about
    sigils is still a sigil to the parser, and bare_markers.py's ratchet will
    refuse the push - correctly, because nothing downstream can tell prose from a
    claim. On 2026-09-20 that happened three times in one day, to three different
    people, every one of them writing documentation OF THIS RULE; one blocked a
    push twice. The people most likely to trip the ratchet are the ones
    explaining it, which is worth a line here rather than a lesson each time.
    Where a specimen genuinely has to appear, mark it `#example` as the worked
    examples below do, and the scanner will parse it without executing it.

 2. @M's source is a command that runs from the repo root and prints the value.
    If it is not runnable it is not @M. A number from a run whose command is gone
    is @C on your own past self, and it should say so.

 3. @C names the artefact you actually opened - a URL, a file path in a vendored
    repo, a table number. A number from a research summary is @C on THE SUMMARY,
    never on the paper it summarises. Failure 1 was a @C-on-a-summary written as
    if it were a @C-on-a-repo, about named people.

 4. @D writes the arithmetic out literally and names its inputs. `sqrt(0.760 *
    0.237)` is a claim you can check by eye; "the decomposition suggests" is not.
    A @D is never stronger than its weakest input: @D over a @C is a @C. Failure 3
    was @D on a boolean printed as @C, and failure 6 was a @D (a sum) sitting in a
    column of @Ms (means).

 5. THE SOURCE LINE NAMES WHAT WAS HELD FIXED - clip, split, code path, config,
    detector, confidence. A number is not a property of a function. It is a
    property of a run. Failures 2, 4 and the on_pitch bug below are all one
    sentence: the run identity was not written down, so the number travelled.

    COROLLARY, AND IT IS THE ONE THAT MATTERS MOST HERE: if the run DROPPED
    INPUT, the tag says how much. A score that goes up because the pipeline
    stopped being asked about the hard cases is not a result, and it is
    indistinguishable from a real one in every metric. Any tag whose command
    filters, masks, gates or thresholds carries the share it removed.

 6. @M on a held-out, test or benchmark split states `n=<configurations tried>`.
    See TUNING, at the bottom.

--------------------------------------------------------------------------------
WORKED EXAMPLE: TODAY'S `on_pitch`, WHICH DELETED 45% OF THE PITCH
--------------------------------------------------------------------------------
What was written (model/engine/stitch.py, `on_pitch`, since withdrawn):

    Measured on the STC2025 pair, 132831 spends **64 of its 195 tracklets on
    things that never match any player** ...

        margin   0     2     4     6     10    15(off)
        HOTA     -   0.400 0.405 0.408 0.401 0.394

    6 m is the peak and the curve is flat either side of it

Under the convention it is two tags and they do not fit on the page together:

    @M 0.408 :: .venv/bin/python model/validation/st2_mot.py --seq 128057 --margin 6  #example
       cal=HeightScale, i.e. the UNCALIBRATED path. n=6 margins, on the same
       held-out split this number is reported from.
    @M 64,195 :: .venv/bin/python model/validation/st2_mot.py --seq 132831 --tracklets  #example
       cal=HeightScale.

    (`#example` because the function is withdrawn and the command no longer
     exists; the checker never runs a tag marked that way.)

Three separate things then catch it, and any one of them is enough:

  * Rule 5 forces `cal=HeightScale` onto the page. The function ships inside
    `stitch_tracks`, which only runs WITH a real calibration - and `pitch.py` is
    corner-origin, x in 0..105, where `abs(x) < 52.5 + margin` keeps 56 m of a
    105 m pitch. The tag names the frame the number was taken in, so the mismatch
    between the frame it was measured in and the frame it ships in is visible in
    the diff instead of buried in a code path nobody re-read.

  * Rule 6 forces `n=6`. "6 m is the peak and the curve is flat either side of it"
    then reads as what it is: the argmax of a 6-point sweep worth +0.014 over a
    0.394 baseline, selected on the split it is reported on. That sentence does
    not survive being written next to its own n.

  * Rule 5's corollary forces the drop share onto the page, and this is the one
    that gets to the bottom of it. The mask kept 110 of 200 rows and truncated x
    at 56 m of a 105 m pitch. Written down, "+0.014 HOTA, dropped 45% of the
    pitch" is not a mixed result - it is a self-refuting one, because the 45% it
    dropped is the far side, which is where every player this stack cannot track
    already lives. THE +0.014 WAS THE BUG. The score rose because the benchmark
    stopped asking about the players that were being got wrong. Rebuilt
    correctly, on the frame it actually ships in, the margin is worth ~0.000.

    That is the general shape and it is worth more than the specific case: a
    filter that improves a metric is the single most dangerous kind of change,
    because deleting hard inputs and solving hard inputs look identical
    downstream. The drop share is the only thing that separates them, and it
    costs one clause.

  * Rule 2 makes the number re-runnable, and re-running it against the fixed code
    prints 0.392, not 0.408 - a LOSS against the 0.394 baseline it was sold as
    beating. That is exactly what was found when it was rebuilt correctly, one
    day later and by hand, instead of by a checker in 20 seconds.

The 64/195 tag drifts too: rebuilt correctly the mask cuts 0.0% on 132831, the
sequence it was built for. Two tags, both red, before the commit.

--------------------------------------------------------------------------------
THE CHECKER
--------------------------------------------------------------------------------
    .venv/bin/python model/validation/verify_claims.py          # fast claims; manual, no hook runs this form
    .venv/bin/python model/validation/verify_claims.py --all    # including #slow; pre-push, heavy pushes only
    .venv/bin/python model/validation/verify_claims.py --list    # every tag, run nothing
    .venv/bin/python model/validation/verify_claims.py --history # the tuning odometer
    .venv/bin/python model/validation/verify_claims.py --all -j1 # the old serial loop

`--jobs`/`-j` defaults to 4 and claims that share state never overlap whatever it
is. See RUNNING THEM IN PARALLEL, below the parser, for what a "resource" is and
why the floor is a chain of claims that cannot be split, not the serial total
over N. The two recorded tags in that block are the shape it was designed against.

Scans every git-tracked text file for @M tags, runs each command from the repo
root, and requires every recorded value to appear as a whole number-token in the
output. Substring matching on purpose: it couples to nothing, it needs no output
contract, and when a script changes its printing the claim goes red and gets
re-recorded, which is correct.

`ponytail: whole-token substring match. A false PASS needs a coincidental
occurrence AND a real drift in the same output. If that ever happens the upgrade
is a labelled capture group, not a parser.`

--------------------------------------------------------------------------------
THE DAY IT PRINTED `8/8 @M verified` AND MEANT NOTHING (2026-09-10)
--------------------------------------------------------------------------------
audit-claims (not published) §A: `--list` found ten tags, and all ten were the SEED
CLAIMS block below. Every document written in this file's voice since it landed
- 441 `@M` markers across docs/ - used the sigil as a PREFIX DECORATION:

    `@M` mean HOTA **+0.0148**, mean AssA **+0.0210** over ten sequences

No `::`, no command, so the parser cannot see it and nothing re-runs it. The
checker was green, honestly green, and green over its own docstring.

THE TEMPTING FIX IS THE WRONG ONE. It is easy to widen the parser until it
swallows that line: pull the bolded float out as the value, guess the command
from a nearby code fence. Do not. A parser that GUESSES which number a marker
refers to produces false greens, and a false green is the one output this file
exists to prevent - it is strictly worse than the silence it replaces, because
somebody acts on it. `@M` on a sentence with no command in it is not an
under-parsed claim. It is not a claim. Rule 2 already says so: if it is not
runnable it is not @M.

So the grammar did not move. Three things around it did:

  * THE CENSUS. Every bare marker - a sigil with no `value :: command` after it
    - is counted and printed next to the verified count. `11 tags verified,
    1,197 markers verify nothing` is the true state of the repo, and it is the
    sentence that makes the conversion job visible instead of invisible. The
    number only goes down by converting docs, which is the work.

  * THE ALIASES. See above: `@V`/`@RAN`/`@REC`/`@DER` now parse.

  * THE VACUITY GUARD, which is the one that would have caught this. If every
    runnable tag in the repo lives inside this file, the run is VACUOUS and
    exits non-zero with a banner instead of a count. A checker whose only
    subject is itself is not passing; it is not running. `_selfcheck` asserts
    that state is detected in both directions, and asserts against the REAL
    repo that at least one runnable tag exists outside this file - so if the
    docs ever lose their tags again, the self-check goes red on the way in.

WHAT TO WRITE INSTEAD, for the doc author holding a prefix marker. This:

    `@M` HOTA **0.364** over ten sequences

becomes this, and the only new work is remembering which command it was:

    @M 0.364 :: .venv/bin/python model/validation/status.py --variant .c03s  #example
       ten ST2 sequences, .c03s, TrackEval MOT17. held-out split, n=1.
       (`#example` because 0.364 is a made-up number in a doc ABOUT the
        convention; the real one is @M 0.393,0.356 in audit-claims (not published) §M.)

--------------------------------------------------------------------------------
TUNING (rule 6)
--------------------------------------------------------------------------------
Failure 7 was a config scored ~24 times against the TEST set and the best of the
24 reported as the headline. Nothing anywhere recorded that it was a best-of-24:
not the commit, not the doc, not a log, not a file on disk. The number is not
wrong - it is unqualified, which on a 24-way max over a noisy metric is worse,
because a best-of-24 on a spread of +-0.01 buys about +0.02 for free.

That is the thing to design against, and no checker can see it after the fact:
the runs left no trace. Two things make it visible, and the first one is doing
the work:

  * THE HABIT. `n=` is part of the tag. Writing `n=24` next to a +0.014 is
    self-limiting in a way that no amount of tooling is, because you have to type
    it and then look at it. `n=1` is a result. `n=24` on a 0.014 delta is a
    sweep, and it goes next to the baseline's own run-to-run spread or it is not
    a finding.

  * THE ODOMETER. Every verification appends to model/validation/claims.log (JSONL,
    committed). `--history` prints, per command, how many times it ran and how
    many DISTINCT values have ever been recorded for it. A claim that has been
    re-recorded eleven times is a knob, and the log says so in the diff.

The odometer only sees claims that went through a tag. A sweep run by hand outside
the tag system is invisible to it, and that is the honest limit: the odometer
catches the tuning you wrote down, rule 6 catches the tuning you did.

--------------------------------------------------------------------------------
SEED CLAIMS
--------------------------------------------------------------------------------
Five real claims, copied verbatim out of the files named. They live here so this
prototype has something to check; adopting the convention means moving each tag to
the line it describes and deleting this block. The scanner will find them either
way - it does not know this file is special.

@M 0.3522,11.5,1352,0.6764 :: .venv/bin/python model/engine/stitch.py --eval  #slow
   118575, first 9,000 frames, cbiou tracklets, no calibration change. IDF1 /
   fragments-per-player / id switches / precision, read off the STITCHED row.
   Copied from model/engine/stitch.py's module docstring (moved 2026-09-28 to
   detection-tracking-evidence (not published)#evidence-stitching) and ledger-vision (not published):41.

   `@M` RE-RECORDED 2026-09-10 from 0.3548,11.3,1348: IDF1 -0.7%, fragments
   +0.2, switches +4, precision UNCHANGED at 0.6764. Precision holding exactly
   while the association terms move a fraction of a percent is the signature of
   a stitching-side change, not a detection one. Small enough that it could be
   run-to-run variation; large enough that the gate is right to refuse the old
   number. If it moves again without a stitch.py change, that is a
   nondeterminism bug and it should be chased then.

@M 2.50,4.66,14.12 :: .venv/bin/python model/engine/calcheck.py --real
   65 published keypoints of data/jobs/118575_full_1st/calibration.json, SINGLE
   HOMOGRAPHY column only. leave-one-out median / touchlines->rest median / near
   half->far half median. Copied from ledger-vision (not published):34.

@M 33.2 :: .venv/bin/python model/validation/contam_eval.py  #slow
   118575 full half, shipped config: 45 m/s split ON, team gate ON. Percent of
   attributed detections under an id whose dominant true player is someone else.
   Copied from ledger-output (not published):17 and pipeline-2026-09 (not published)'s opening scope note.

@M 139,15.4%,20.2% :: .venv/bin/python model/validation/ball_eval.py
   soccer_top D_20220220 0000-0030, detector coreml at conf 0.05, 20 px gate, 900
   frames. picker-right count / picker-right share / oracle ceiling. Copied from
   docs/experiments/st2-input-ledger.md:84-85.
   NOTE THE TAG IS NOT THE SENTENCE: docs/experiments/st2-input-ledger.md:147 says this is
   "15.4% correct at ship confidence" and the command above says conf 0.05, where
   ship is 0.5. The tag reproduces; the sentence around it is still false. See
   WHAT THIS DOES NOT CATCH.

@M 0.137,0.583 :: .venv/bin/python model/validation/st2_eval.py --demo
   st2 val split, seed 0. chance DRIVE AP / shift-copy DRIVE AP. The floors that
   P1's gate has to clear. Was 0.123,0.575 until 2026-09-19: the LABEL OFFSET rule
   (st2-p1-step6 (not published)) moved the floors onto the corrected label clock and this
   seed - the checker's own - was not updated with them, so the gate would have
   rejected the very branch that moved them. shift-copy is GT-FED, a gate floor and
   not a deployable baseline (st2-p1-step7 (not published)'s correction).

And the two other sigils, for shape - neither is run by anything:

@C 331.5 :: STC2025 leaderboard, IDSW column, MEAN over 2 sequences
   Our like-for-like figure is a mean too. Failure 6 was our SUM printed beside it.
@D 0.424 :: sqrt(0.760 * 0.237) [@M st2-deta, @M st2-assa]
   HOTA if detection were fixed to the winner's and association left alone.

--------------------------------------------------------------------------------
WHAT THIS DOES NOT CATCH
--------------------------------------------------------------------------------
It re-runs commands and diffs numbers. It cannot read. A tag that reproduces
perfectly can still sit under a false sentence - the ball_eval claim above is a
live example in this very docstring, and no checker will ever find it. A cited
tag is unverifiable by construction: a labelling discipline, whose whole value is
that a reader can see which is which. A derived tag's arithmetic IS checked, by
derived.py (pre-push), which evaluates the literal expression; its inputs are not. That is the point: the sigil is the
mechanism, the checker is only the part of it that can be automated.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "model" / "validation" / "claims.log"

ME = "model/validation/verify_claims.py"

# The three sigils, plus the four aliases the docs invented after this file
# landed. Grouped by what the checker can DO with them, which is the only
# distinction the code cares about: run it, or don't.
RUNNABLE = {"M", "V", "RAN"}    # a command exists; re-run it and diff the number
CITED = {"C", "REC"}            # read from somewhere - external, or our own past
DERIVED = {"D", "DER"}          # arithmetic on the above
# Longest alternative first, or `@DER` parses as `@D` and silently loses a
# sigil. `\b` after, or `@MEOW` parses as `@M`.
SIGILS = "DER|REC|RAN|M|C|D|V"

# sigil, value(s), `::`, source. Leading junk (markdown bullets, table pipes,
# comment hashes, docstring indent) is eaten so the tag works anywhere.
# The value must START WITH A DIGIT OR SIGN. That is what stops this file's own
# `@M <value> :: <shell command>` grammar line from being parsed as a claim and
# then executed - which it was, the first time it ran.
# A value may also be written `label=value` - the DECLARED ANCHOR for a small
# integer (see token_present); the label is a name, so the grammar line's
# `<value...>` still does not parse.
# A value may also START with a declared label (`of 7`, `"a, b" 7`): an
# unquoted label must be followed by a number that ENDS the value, so prose
# like `@M means nothing :: here` still does not parse.
TAG = re.compile(rf"@({SIGILS})\b\s+((?:[-+.\d]|[A-Za-z_][\w.-]*="
                 r'|"[^":\n]*"\s*[-+]?\.?\d'
                 r"|[A-Za-z_][^:,\"\n]*?\s[-+]?\.?\d[\d.]*%?(?=\s*(?:,|::)))"
                 r"[^:]*?)\s*::\s*(.+?)\s*$")
# The same sigils with NOTHING after them. 1,200-odd of these live in docs/ and
# each one is a number nobody can re-run. Counting them is the whole reporting
# change: an unconverted claim used to be invisible, and is now a number that
# only goes down when somebody does the conversion.
MARKER = re.compile(rf"@({SIGILS})\b")
HELD_OUT = re.compile(r"\b(held.?out|test split|benchmark|--split test)\b", re.I)
TEXT = {".py", ".md", ".txt", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".sh"}


def git_env(base=None) -> dict:
    """The environment minus every GIT_* variable. THE REPO IS THE DIRECTORY.

    A pre-push hook, and any agent shell started inside one, exports GIT_DIR /
    GIT_INDEX_FILE / GIT_WORK_TREE, and those BEAT `cwd=`. `git ls-files` then
    lists the OTHER tree's index - measured here, 731 paths instead of 765 -
    so the scanner walks a file set that is not the one being pushed and prints
    a GREEN count over it. That is the vacuous-green failure this file exists to
    refuse, arriving through the environment instead of through the docs.

    protocol.py learned this on 2026-09-19 (9b4a01b) when its self-check
    committed seven specimen commits onto the checked-out branch; the fix was
    local to that file and every other gate still inherits the variables. This
    is the same fix, in the module the other scanners already import from.
    """
    return {k: v for k, v in (base or os.environ).items() if not k.startswith("GIT_")}


def head(short: bool = True, cwd=None) -> tuple[str, bool]:
    """(HEAD sha, is the tree dirty) for the repo this file lives in.

    A PROVENANCE stamp, and it was written out four times - here, api.py,
    status.py and train_roles.py - each with its own idea of what to do when git
    is not there. It always answers: an unstamped row is better than a row
    stamped with another tree's HEAD, and a crash in a stamp must never take
    down the thing being stamped.
    """
    def g(*args):
        try:
            return subprocess.run(["git", *args], cwd=cwd or ROOT, env=git_env(),
                                  capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:   # noqa: BLE001 - a stamp never fails its caller
            return ""
    sha = g("rev-parse", *(["--short"] if short else []), "HEAD")
    return sha or "unknown", bool(g("status", "--porcelain"))


def cut_cmd(source: str) -> str:
    """The command ends where the prose begins: at the first run of TWO OR MORE
    spaces (or a tab) OUTSIDE quotes.

    Why that rule, from the live corpus on 2026-09-29: 51 of 445 runnable tags
    put "what was held fixed" on the tag line after the command, always after
    two or more spaces (`--show   gate 6 m block; ...`), and not one command
    needs two spaces outside quotes. Before this the prose was PART OF THE
    COMMAND: passed to the script as arguments, or - n37-full-half (not published):82,
    whose prose holds a `(` - a shell syntax error whose message
    `sh: line 0: ...` supplied the recorded value "0". Quotes are respected so
    `python -c "a;  b"` survives; a backslash escapes the next character.
    """
    q, i = None, 0
    while i < len(source):
        ch = source[i]
        if ch == "\\" and q != "'":
            i += 2
            continue
        if q:
            q = None if ch == q else q
        elif ch in "'\"":
            q = ch
        elif ch == "\t" or source.startswith("  ", i):
            return source[:i].strip()
        i += 1
    return source.strip()


_SYNTAX: dict[str, bool] = {}


def syntax_error(cmd: str) -> bool:
    """`sh -n`: does the command even parse? Never executes it. Cached per command."""
    if cmd not in _SYNTAX:
        _SYNTAX[cmd] = subprocess.run(["sh", "-n", "-c", cmd], capture_output=True).returncode != 0
    return _SYNTAX[cmd]


class Claim:
    def __init__(self, sigil, values, source, file, line, extra):
        self.sigil, self.source, self.file, self.line = sigil, source, file, line
        # commas separate values, except inside a quoted label (split_label)
        self.values = [v.strip() for v in re.findall(r'(?:"[^"]*"|[^,])+', values) if v.strip()]
        self.slow = "#slow" in source
        # `#example` = a tag shown as illustration, in a doc ABOUT the convention.
        # Never run. Documentation of a convention always contains specimens of it,
        # and a specimen that gets executed makes the checker useless in its own docs.
        self.example = "#example" in source
        self.cmd = cut_cmd(re.split(r"#slow|#example", source)[0])
        self.extra = extra          # the indented continuation lines under the tag

    @property
    def key(self):
        return f"{self.file}:{self.line}"

    @property
    def runnable(self):
        """@M and its aliases. Everything downstream branches on this, not on
        the literal sigil, so adding an alias is one entry in one set."""
        return self.sigil in RUNNABLE

    @property
    def mine(self):
        """A tag inside this file. Seeds and specimens - never evidence that the
        checker is checking anything."""
        return self.file.endswith(ME)

    def context(self):
        return self.source + " " + " ".join(self.extra)


def scan(census: collections.Counter | None = None) -> list[Claim]:
    """Every tag in every git-tracked text file, plus untracked ones.

    Untracked included on purpose: a claim written five minutes ago in a file not
    yet `git add`ed is exactly the claim most likely to be wrong.

    Pass a Counter as `census` to also collect the BARE markers - sigils with no
    `value :: command` after them. Same pass over the same files, because the
    two numbers are only meaningful next to each other: `11 verified` is a
    triumph or a rounding error depending entirely on whether the second number
    is 30 or 1,200.
    """
    def ls(*args):
        return subprocess.run(["git", "ls-files", *args], cwd=ROOT, env=git_env(),
                              capture_output=True, text=True).stdout.splitlines()

    claims = []
    for rel in ls() + ls("--others", "--exclude-standard"):
        p = ROOT / rel
        if p.suffix not in TEXT or not p.is_file():
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            m = TAG.search(line)
            if census is not None:
                # every sigil on the line, minus the one the tag consumed.
                for s in MARKER.findall(line):
                    census[s] += 1
                if m:
                    census[m.group(1)] -= 1
            if not m:
                continue
            # continuation: the indented lines under the tag, up to the next
            # blank line. That is where rule 5's "what was held fixed" lives.
            extra, j = [], i
            while j < len(lines) and lines[j].strip() and not TAG.search(lines[j]):
                if lines[j][:1] in " \t":
                    extra.append(lines[j].strip())
                    j += 1
                else:
                    break
            claims.append(Claim(m.group(1), m.group(2), m.group(3), rel, i, extra))
    return claims


def touches_tag(diff: str) -> bool:
    """Does this unified diff change any line INSIDE a tag block?

    THE HOLE THIS CLOSES. `.githooks/pre-push` decides whether to run the heavy
    gates with a grep for a sigil on a changed line, `^[-+].*@<sigil>`, over
    the diff. That sees the tag LINE and
    nothing else - and a tag is two parts. The second part is the indented
    continuation under it, which is where rule 5 lives: the clip, the split,
    the conf, the detector, and rule 6's `n=`. Edit only those and the diff
    carries no sigil on a changed line, touches no engine path, and the push is
    classified as prose - skipping verify_claims entirely, its rule-6 lint with
    it, and api --verify and mutate too.

    That is not a hypothetical shape. This repo's headline failure is `conf
    0.05 quoted for a stack that ships at 0.5` - docs/README.md#working-rules names it - and
    rewriting `conf 0.05` to `conf 0.5` under an unchanged tag is exactly this
    edit. The VALUE never moves, so re-running the command would not catch it
    either; what has to change is that the push stops counting as prose.

    The block rule is `scan()`'s own, on purpose: a tag line opens a block, and
    an indented non-blank line that is not itself a tag continues it. Context
    lines are read, +/- lines are the test, so a continuation edit fires even
    when the tag line above it is untouched. It needs enough context to reach
    the tag from the changed line - the hook passes --unified=25.

    `ponytail: chosen over the other candidate - running the fast
    verify_claims on doc-only pushes - because that one still executes ~280
    commands (not fast) and changes what the gate MEANS on a doc push. This
    asks one question a person can check at 3am: did you change a line under a
    tag?`
    """
    for hunk in re.split(r"^@@.*$", diff, flags=re.M)[1:]:
        inside = False
        for raw in hunk.splitlines():
            if raw[:1] not in "+- ":
                break                      # past the hunk body, into the next header
            body = raw[1:]
            if TAG.search(body):
                inside = True
            elif not body.strip() or body[:1] not in " \t":
                inside = False             # a blank line, or back to column 0
            if inside and raw[:1] in "+-":
                return True
    return False


#: An integer this short is in every output by accident - line numbers, dates,
#: exit codes, `(1 of 3)`. It must be ANCHORED to count as present.
SMALL_INT = re.compile(r"[-+]?\d{1,2}")
DECLARED = re.compile(r"([A-Za-z_][\w.-]*)=(.+)")
#: `"label" value` and `label value`: the literal text the output prints right
#: before the value. The value is the LAST word and must be a number, or the
#: whole token is read as a literal string (`2.4 s`, `NOT SUPPORTED`).
QUOTED = re.compile(r'"([^"]+)"\s*(\S+)')
SPACED = re.compile(r"(\S.*?)\s+(\S+)")
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)%?")
#: The left edge of a number: not after a digit or `.`, and not after a `-`
#: that is a SIGN. A `-` right after a digit is a RANGE separator - `1.7-5.0`
#: prints the value 5.0, not -5.0 - so the value after it stands alone.
LEFT = r"(?<![\d.])(?:(?<=\d-)|(?<!-))"
#: An integer claim also matches the same integer printed with only zeros
#: after the point (`291.0000`); `291.5` is a different number.
INT_ZEROS = r"(?:\.0+)?"


def split_label(value: str) -> tuple[str | None, str, str]:
    """(label, separator pattern, value) - the label a tag DECLARES, if any.

      `"ids 44.000->" 59`  quoted: any text, commas allowed; label then value
      `name=7`              the label, then `=`, `:` (optionally after a quote) or whitespace
      `of 7`, `IDSW 291`    unquoted: every word but the last is the label
    Labels are matched whitespace-normalised; the unquoted form needs a
    NUMERIC last word, so `2.4 s` stays the literal string it always was.
    """
    if (m := QUOTED.fullmatch(value)) and NUMBER.fullmatch(m[2]):
        return m[1], r"\s*", m[2]
    if (m := DECLARED.fullmatch(value)):
        return m[1], r"""(?:['"]?\s*[=:]\s*|\s+)""", m[2]
    if (m := SPACED.fullmatch(value)) and NUMBER.fullmatch(m[2]):
        return m[1], r"\s*", m[2]
    return None, "", value


def token_present(value: str, text: str) -> bool:
    """Is `value` printed in `text` as a WHOLE number, anchored if it is short?

    Whole number: not preceded by a digit, `.` or a SIGN `-` (so `0.56` is not
    found in `-0.56` nor `11.3` in `111.3`), and not followed by a digit,
    `.<digit>` or an exponent (`0.354` not in `0.3548`, `1.5` not in `1.5e-3`).
    A `-` directly after a digit is a range separator, not a sign: `5.0` IS in
    `1.7-5.0` (LEFT). An integer also matches itself printed with only zeros
    after the point: `291` is in `291.0000`, not in `291.5` (INT_ZEROS).

    ANCHORED: an integer of fewer than 3 digits must sit after a label in the
    output - `name=7`, `name: 7`, `'far': 2` - or the tag must DECLARE the
    anchor (split_label). Three spellings, all inside the value list:

        name=7              `name`, then `=`, `:` or whitespace, then 7
        of 7 / IDSW 291     the words before the last are the literal text
                            printed immediately before the value
        "a, b ->" 7         quoted, when the label holds a comma (the value
                            separator) - the quotes keep it one value

    A declared label is whitespace-normalised and must start at a word
    boundary. The tag grammar has no colon inside the value list (`::` ends
    it), so a label that prints `hits: 7` is written `hits=7`. `line 1` in a
    traceback and `line 0` in a shell error were both reading as recorded values.

    THE COMMA IS ALWAYS A VALUE SEPARATOR IN A TAG (Claim splits on it, outside
    double quotes), so a thousands-grouped value cannot be written with one:
    `1,352` is the two values `1` and `352`. Joining digit groups instead would
    split 42 live multi-value tags whose neighbours happen to look like
    `12,345` (2026-09-29, --list). Write `1352`. A tag that meant 1,352 and
    wrote it that way now fails loudly, because its `1` is an unanchored small
    integer.
    """
    label, sep, value = split_label(value)
    zeros = INT_ZEROS if re.fullmatch(r"[-+]?\d+", value) else ""
    body = LEFT + re.escape(value) + zeros + r"(?!\d|\.\d|[eE][+-]?\d)"
    if label:
        words = label.split()
        pat = (r"(?<!\w)" if re.match(r"\w", words[0]) else "") + \
            r"\s+".join(map(re.escape, words)) + sep + body
    elif SMALL_INT.fullmatch(value):
        pat = r"""[\w'"\])]\s*[=:]\s*""" + body
    else:
        pat = body
    return re.search(pat, text) is not None


def unanchored(c: "Claim") -> list[str]:
    """The small integers in a tag that only an output label can anchor."""
    return [v for v in c.values if SMALL_INT.fullmatch(v)]


def report_census(census: collections.Counter) -> str:
    """One line: how many sigils in this repo verify nothing.

    It prints next to the verified count, always, and it is deliberately the
    larger number. `11/11 @M verified` invites the reading "the claims are
    checked". `11/11 @M verified; 1,197 bare markers verify nothing` invites the
    correct one, which is that the convention is 1% adopted. The gap is the
    backlog, and a backlog you cannot see is a backlog nobody burns down.
    """
    n = sum(census.values())
    if not n:
        return "no bare markers - every sigil in the repo is a real tag"
    top = ", ".join(f"{c} @{s}" for s, c in census.most_common())
    return (f"{n} BARE markers ({top}) carry no `value :: command` and verify "
            f"nothing - prefix decoration, not claims")


def vacuous(claims: list[Claim]) -> str | None:
    """Is this run about to verify nothing, in green? Reason, or None.

    THIS IS THE BUG THIS FILE ACTUALLY HAD, and it is worth being precise about
    why a green count hid it for two days. Nothing was broken. The parser was
    right, the matcher was right, the five seed claims genuinely re-ran and
    genuinely reproduced, and `8/8 @M verified` was a true statement. It was
    just a statement about this file's own docstring, and the docstring is the
    one artefact in the repo that was never in doubt.

    A test suite that only tests its own fixtures is the same failure, and the
    instruments audit found five checks in this repo that could not fail. The
    shape is always this: the check is real, its SUBJECT quietly went away.

    So the count is not allowed to stand on its own. Two states are refused:

      * no runnable tag anywhere - nothing to do, so nothing is being checked;
      * every runnable tag inside this file - the checker is its own subject.

    Both exit non-zero with a banner instead of a number. Loud, because the
    quiet version of this is what shipped.
    """
    live = [c for c in claims if c.runnable and not c.example]
    if not live:
        return "there is not one runnable tag in the whole repo"
    outside = [c for c in live if not c.mine]
    if not outside:
        return (f"all {len(live)} runnable tags are inside {ME} - the checker's "
                "only subject is its own seed block")
    return None


def banner(reason: str) -> None:
    print("\n" + "!" * 78)
    print("VACUOUS RUN - THIS CHECKER IS VERIFYING NOTHING")
    print(f"  {reason}.")
    print("  A green count here would be true and worthless. See the docstring,")
    print("  'THE DAY IT PRINTED 8/8 @M verified AND MEANT NOTHING'.")
    print("!" * 78)


def judge(c: Claim, blob: str, rc: int) -> tuple[str, list[str]]:
    """The verdict for ONE claim against an output that is already captured.

    Split out of run() so that N claims quoting different numbers from the SAME
    command are judged from ONE execution. 98 of 400 runnable claims share a
    command with another, worth 753 s of re-running per --all; `ball_cmp.py
    --show` alone was executed ten times for ten tags reading ten numbers out of
    the same table. Judging is pure: same blob, same verdict, so this cannot
    change an answer - only how many times the machine produces it.
    """
    missing = [v for v in c.values if not token_present(v, blob)]
    # A NEGATIVE returncode is a signal, not an exit status - see run().
    if rc < 0:
        return "KILLED", missing
    # A FAILED COMMAND NEVER VERIFIES ANYTHING, whatever it printed. It used to
    # pass when the value was "in the output" - and the output of a failure is
    # a traceback or a shell error, full of line numbers: `python -c "...+undefined"`
    # printed `line 1` and verified a recorded 1.
    if rc != 0:
        return "ERROR", missing or ["(exit status %d)" % rc]
    return ("OK" if not missing else "DRIFT"), missing


def run(c: Claim, timeout: int) -> tuple[str, list[str], float]:
    """One claim: run_group of one, so the SYNTAX refusal cannot be bypassed."""
    return run_group([c], timeout)[0][1]


def run_group(cs: list[Claim], timeout: int) -> list[tuple[Claim, tuple]]:
    """Execute a shared command ONCE and judge every claim that quotes it.

    The whole group carries the same wall time - which is honest: it IS one run,
    and pretending each claim paid for it separately is what the log said before.
    """
    # SYNTAX: a command that does not parse is refused BEFORE it runs. Its
    # error text is not output to be searched for a number.
    if syntax_error(cs[0].cmd):
        return [(c, ("SYNTAX", c.values, 0.0)) for c in cs]
    t0 = time.time()
    try:
        # shell=True on purpose: the command is prose written by this repo's own
        # author in this repo's own tracked files, at the same trust level as the
        # code around it, and claims want pipes and redirection. There is no
        # untrusted input here; if a tag ever comes from outside the repo, this
        # is the line that has to change.
        r = subprocess.run(cs[0].cmd, cwd=ROOT, shell=True, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        took = time.time() - t0
        return [(c, ("TIMEOUT", c.values, took)) for c in cs]
    blob, took = r.stdout + r.stderr, time.time() - t0
    return [(c, (*judge(c, blob, r.returncode), took)) for c in cs]


# -------------------------------------------------------------------------------
# AN INCOMPLETE TREE IS NOT A DRIFT, AND IT USED TO PRINT AS ONE
# -------------------------------------------------------------------------------
# 2026-09-20, a second session ran this gate in a fresh worktree and 49 claims
# failed in about a tenth of a second each. Every one printed the same ERROR and
# the same `NOT IN OUTPUT: <value>` that a number which genuinely moved prints.
# Nothing had moved. The worktree simply did not have the footage, the jobs or
# the caches, because those are gitignored and a new worktree does not get them.
#
# That is the same failure as the SIGTERMed run reported as drift, in a different
# costume, and it costs the same thing: the reader is taught to distrust the
# number instead of the tree. The first question a red gate has to answer is
# "did anything actually get measured", and a tree that cannot run the command
# measured nothing.
#
# So the paths are checked ONCE, BEFORE the first claim, and a missing one stops
# the run with a list of what to link. It is deliberately a list of literal
# paths rather than anything clever: a check that tries to infer what a claim
# needs would itself be a thing that can be wrong, and the failure it exists to
# catch is somebody forgetting to run `ln -s`.


#: What a claim needs on disk that git does not carry. All gitignored, so a
#: fresh worktree has none of them and every claim that opens one fails fast.
REQUIRED = (
    ".venv",
    # the detector weights: six claims lost to these being absent, 2026-09-20
    "model/models",
    # the click solver several claims shell out to through `npx vite-node`;
    # two more claims lost to this one, and its absence reads as a CalledProcess
    # -Error deep in a traceback rather than as a missing tree
    "app/frontend/node_modules",
    "data/jobs", "data/chats", "data/demo", "data/notes", "data/renders",
    "data/scratch", "data/soccertrack_v2",
    "data/match.db", "data/presets.json", "data/footage/soccertrack",
)
#: Directories git tracks but whose heavy children it does not: present but
#: EMPTY is the unlinked state, and `exists()` alone would call it fine.
REQUIRED_NONEMPTY = ("data/footage", "data/vendor")


def preflight(root: Path | None = None) -> list[str]:
    """What this tree is missing. Empty list = it can run. Never runs a claim."""
    root = ROOT if root is None else root
    bad = [p for p in REQUIRED if not (root / p).exists()]
    bad += [p for p in REQUIRED_NONEMPTY
            if not (root / p).is_dir() or not any((root / p).iterdir())]
    out = root / "model" / "validation" / "out"
    # THE ONE RULE THAT IS NOT JUST `exists`. A symlink at `out` resolves for
    # every ordinary read and then breaks api.py's self-check, so the tree looks
    # complete right up until one gate fails for a reason unrelated to its
    # subject. Real directory, symlinked children: that is the working layout.
    if not out.is_dir():
        bad.append("model/validation/out (absent)")
    elif out.is_symlink():
        bad.append("model/validation/out (a symlink - must be a REAL directory "
                   "whose children are links, or api.py --verify breaks)")
    return bad


#: Where the evidence a claim reads actually lives. Walked twice per run.
TREE = ("model/validation/out", "model/models")


def tree_paths(root: Path | None = None) -> set[str]:
    """Every file under the artefact tree and the weights, right now."""
    root = ROOT if root is None else root
    found = set()
    for d in TREE:
        for base, _, files in os.walk(root / d):
            found.update(os.path.join(base, f) for f in files)
    return found


def tree_moved(before: set[str]) -> list[str]:
    """Of the files that existed when this run STARTED, which are gone now?

    THE THIRD CAUSE THAT PRINTS AS DRIFT. Tonight has now produced three, and
    they are indistinguishable in the output: an incomplete worktree, contention
    against the per-claim timeout, and this one - somebody moving an artefact
    while a run is reading it. Ten claims failed mid-run to a move and every one
    of them passes now; nothing had drifted, the ground had moved.

    ONLY DISAPPEARANCES COUNT, and that is the whole design. Claims write into
    out/ constantly - that is what a producer does - so "some file changed" fires
    on every healthy run and would be worthless. A file that was there when the
    run began and is NOT there when it ends was taken away by something other
    than this run, which is exactly the event worth naming.
    """
    return sorted(p for p in before if not os.path.exists(p))


def cited_dirty() -> list[str]:
    """Tracked artefacts under out/ whose file on disk differs from the commit.

    WHICH COPY IS AUTHORITATIVE, since 2026-09-20 there are two. Nine small
    result files under model/validation/out are tracked, because the tags that
    quote them are citations now and a cited file inside a gitignored directory
    is evidence on one machine's disk.

    THE COMMITTED COPY IS THE EVIDENCE OF RECORD. It is what a reviewer reads in
    a diff and what a fresh clone gets. The DISK copy is what the readers
    actually open, so the gate's verdicts are about the disk copy - and when the
    two differ, a green gate is green about numbers that are not in git.

    That divergence has exactly one cause: somebody re-ran a producer and did not
    commit what it wrote. Which is legitimate - that is how the artefact is meant
    to be updated - so this does not fail the run. It says so, loudly, because
    the alternative is a push whose evidence of record is a file nobody updated.

    Derived from `git ls-files`, not from a hardcoded list: track a tenth
    artefact tomorrow and it is covered without editing this function.
    """
    # EVERY line git reports counts, with no status filtering, and that is
    # deliberate rather than lazy. Ignored files are not reported at all without
    # `--ignored`, so the only paths that can appear here are the tracked nine
    # and anything the .gitignore negations re-included. Modified, deleted and
    # untracked are all the same fact about those: the file on disk is not the
    # file in the commit. A filter for `??` here would be a branch no test can
    # reach, which this file has a banner about.
    r = subprocess.run(["git", "status", "--porcelain", "--",
                        "model/validation/out"], cwd=ROOT, env=git_env(),
                       capture_output=True, text=True)
    return sorted(l[3:].strip() for l in r.stdout.splitlines() if l.strip())


def diverged(dirty: list[str]) -> None:
    print("\n" + "-" * 78)
    print("CITED ARTEFACTS DIFFER FROM THEIR COMMITTED COPIES")
    for p in dirty:
        print(f"  modified, uncommitted  {p}")
    print("  The tags above were verified against the files ON DISK. The evidence")
    print("  of record is the COMMITTED copy, and it no longer says what these do.")
    print("  A producer was re-run and its output not committed. Commit it, or")
    print("  `git checkout` these back, before the numbers go anywhere.")
    print("-" * 78)


def moved(gone: list[str], failed: int) -> None:
    print("\n" + "!" * 78)
    print("THE ARTEFACT TREE CHANGED UNDER THIS RUN - VERDICTS ABOVE ARE SUSPECT")
    print(f"  {len(gone)} file(s) present at the start of this run are gone now:")
    for p in gone[:8]:
        print(f"    {os.path.relpath(p, ROOT)}")
    if len(gone) > 8:
        print(f"    ... and {len(gone) - 8} more")
    if failed:
        print(f"  {failed} claim(s) failed above. Re-run before believing any of them:")
        print("  a claim that read a file while it was being moved did not measure a")
        print("  number that moved. This is the third cause tonight to print the same")
        print("  words a real drift prints - the others are an incomplete worktree and")
        print("  contention against the per-claim timeout.")
    else:
        print("  Nothing failed, so nothing was harmed - but the run was not isolated.")
    print("!" * 78)


def incomplete(missing: list[str]) -> None:
    print("\n" + "!" * 78)
    print("INCOMPLETE WORKTREE - NOTHING WAS MEASURED, SO NOTHING DRIFTED")
    for p in missing:
        print(f"  missing  {p}")
    print("  These are gitignored, so a fresh worktree does not get them. Link")
    print("  them from the main checkout and run again. Without them ~49 claims")
    print("  fail in a tenth of a second each, printing the same ERROR and the")
    print("  same 'NOT IN OUTPUT' a real drift prints. This is not that.")
    print("!" * 78)


# -------------------------------------------------------------------------------
# RUNNING THEM IN PARALLEL, AND WHY THE SCHEDULE IS NOT THE INTERESTING PART
# -------------------------------------------------------------------------------
# `--all` ran long enough that it got backgrounded, and a backgrounded push gets
# reaped - which is how "a claim drifted" was printed twice over a run that had
# been SIGTERMed. The fix for that was gate.sh telling KILLED from failed. This
# is the other half: make the run short enough that nobody backgrounds it.
#
# @REC 294,78.7,0.30,34.8,401.1 :: model/validation/claims.log, the block at sha
#    1e8af8f, 2026-09-20. Claims run / minutes of wall / median claim s / p90 s /
#    longest claim s. NOT re-measured here, and not a measured tag: a duration
#    is a property of the machine that ran it, so re-running this would record a
#    knob, not a finding. It is read once, to shape the schedule.
#
# So the COUNT is irrelevant and the TAIL is everything: the schedule is
# longest-first (LPT) off the durations the log already records, or the 400 s
# claim lands last and four workers wait for it.
#
# THE CORRECTNESS BAR IS NOT THE SPEEDUP. A verdict must not depend on how many
# workers there are. These commands share state - `dist_eval.stc_gt` does a
# read-modify-write of out/n10_stc_gt.json with no lock, every N-series module
# truncates one fixed out/ artefact per run, n47_bench reloads-and-rewrites
# n47_l1.json - so two of them at once is a torn read, a lost update, or a
# JSONDecodeError that reads as ERROR. A gate that fails differently at -j4 than
# at -j1 is worse than a slow gate.
#
# So claims declare RESOURCES and two claims that share one never overlap. The
# resource is derived, not hand-listed, because a hand-list goes stale the first
# time a claim is added:
#
#   * every model/{validation,engine}/<mod>.py the command runs, or imports by
#     name after a sys.path.insert - the `python -c "import dist_eval as D"`
#     form is a third of the claim set and names no path;
#   * every model/validation/out/ artefact the command names, mapped through
#     OWNER to the module that WRITES it, so a producer and its readers land in
#     one chain (dist_eval.py --n26 and the four claims that read
#     out/stc2025/n26_results.json);
#   * "gpu" for anything that drives the detector, CoreML or MPS. Oversubscribing
#     MPS does not corrupt anything, it just makes every arm slower, and slower
#     is a per-claim TIMEOUT, which is a verdict change - so serialising the
#     whole family is bought, not assumed: it is smaller than the chain below.
#
# Resources are EXCLUSIVE - no reader/writer distinction, so two readers of one
# artefact also queue. `ponytail: the readers are a third of a second each; a
# reader-writer lock would buy nothing and is a second thing to get wrong.`
#
# WHAT THIS COSTS, and it is the honest headline - the speedup has a hard floor
# that is NOT the worker count:
#
# @REC 1273,1193,614 :: the same claims.log block, grouped by `resources()` as
#    written below, 2026-09-20. Seconds in the dist_eval chain / in the st2_train
#    chain / in every gpu claim summed. The two chains run alongside each other
#    and neither can be split, so the first of them is the floor: no number of
#    workers goes under it, and the gpu family fits underneath for free.
#
# --jobs 4 reaches that floor; more does not. 4 is also the memory budget:
# `dist_eval.stc_gt`'s own docstring says an st2 loader costs ~800 MB of RSS and
# this machine has 16 GiB, so a worker count chosen for speed alone would start
# trading TIMEOUTs for swap.
#
# --jobs 1 is not "the parallel path with one worker". It is the original loop,
# in document order, untouched - so the thing the gate has always done stays
# available exactly as it was, and _selfcheck diffs the other path against it.
#
# WHAT IT ACTUALLY BOUGHT, measured 2026-09-22 on the merged main (4a8be31), the
# two arms back to back on an otherwise idle machine, same scope, same tree:
#
# @M 41.8,13.7,366 :: .venv/bin/python model/validation/verify_claims.py --timing
#    scope --all, --timeout 600, --gpu-jobs 1 (the mutex) in both arms. Serial
#    minutes / parallel minutes / claims, and BOTH ARMS RETURNED 366 OK, which is
#    the pre-registered correctness bar - the verdict set may not depend on the
#    worker count. Read back out of claims.log rather than re-timed: a duration is
#    a property of the machine that ran it, so the tag pins the record, it does not
#    re-measure the knob.
#
# An earlier run of this pair was reported as 30.7 -> 12.7 min and is NOT what the
# log says. Both arms had run with --no-log, so no row existed and the numbers
# lived only in a terminal; the serial arm re-measures at 41.8. That is why log()
# now stamps jobs, wall and scope on every row, and why rows without those fields
# are skipped by --timing instead of being read as jobs=1.


# Module stems this repo owns, read off disk rather than listed, so a new module
# is covered the day it lands.
def _known_modules() -> set[str]:
    out = set()
    for d in ("model/validation", "model/engine"):
        with contextlib.suppress(OSError):
            out |= {f[:-3] for f in os.listdir(ROOT / d) if f.endswith(".py")}
    return out


# out/ artefact stem -> the module that writes it. Grepped 2026-09-20; only the
# ones a claim actually NAMES need an entry, because a module is already its own
# resource. Anything unlisted falls back to the path itself, which over-groups
# (two readers queue) and never under-groups.
OWNER = {
    "n10_stc_gt": "dist_eval", "n12_calib_product": "calib_product",
    "n28_results": "plausibility", "n21_results": "st2_product",
    "n7_results": "st2_product", "p05_results": "st2_product",
    "st1_gnss": "st1_gnss", "st1_gnss_diagnose": "st1_gnss",
    **{f"n{n}_results": "dist_eval" for n in range(22, 28)},
}

# Modules that run a detector, a CoreML model or a torch device. Deliberately
# generous: a false entry costs a few queued seconds, a missing one costs a
# TIMEOUT that only appears under load.
GPU = {
    "detsweep", "detectors", "track", "stitch", "quality",
    "ball_cmp", "ball_eval", "ball_train", "contam_eval", "vision_eval",
    "bodypose_eval", "coreml_check", "reid_eval", "team_eval", "assoc_eval",
    "oracle_eval", "n37_full_half", "n40_occlusion_recall", "n40_boot",
    "n41_conf", "n42_merge", "n43_far_tiles", "n43b_far_fixes", "n44_baseline",
    "n46_equiv", "n46_coreml", "n46_finetune", "n46_dataset", "n47_bench",
}


def resources(cmd: str, known: set[str] | None = None) -> frozenset[str]:
    """What this command touches. Two claims sharing one never run together."""
    known = _known_modules() if known is None else known
    mods = {m for m in known
            if re.search(rf"model/(?:validation|engine)/{m}\.py"
                         rf"|\bimport\s+{m}\b|\bfrom\s+{m}\s+import\b", cmd)}
    res = set(mods)
    for p in re.findall(r"model/validation/out/([\w./\-]+)", cmd):
        res.add(OWNER.get(p.split("/")[-1].split(".")[0], "out:" + p))
    if mods & GPU:
        res.add("gpu")
    return frozenset(res)


def recorded_secs(path: Path | None = None) -> dict[str, float]:
    """cmd -> the duration of its LAST recorded run. The schedule's only input.

    Last rather than max: the max over a command's history is the run that was
    contended, and scheduling off it inflates the tail it is meant to place.
    """
    path = LOG if path is None else path
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            with contextlib.suppress(ValueError, KeyError):
                e = json.loads(line)
                out[e["cmd"]] = float(e.get("secs", 0.0))
    return out


#: How much slower a long claim runs with three others beside it, on the machine
#: this gate runs on. NOT a safety factor picked to feel right - measured, and it
#: is the one hazard the schedule CREATES instead of inheriting:
#:
#: @REC 363.2,599.3,600 :: the -j4 transcript of 2026-09-20 against the same
#:    claim's serial row in claims.log. `st2_train.py --g2 --device cpu --epochs
#:    150 --configs 2` took 363.2 s alone and 599.3 s beside three other claims,
#:    against a 600 s budget. It passed by seven tenths of a second, and had it
#:    not, the gate would have printed TIMEOUT for a claim that did not move -
#:    a verdict that depends on the worker count, which is the exact thing the
#:    resource exclusion exists to prevent.
#:
#: The cause is the hardware, so it does not go away by scheduling better: an
#: Apple M3 has four performance cores and four efficiency cores, and the fourth
#: concurrent multi-threaded claim lands on the slow half. 2.0 rounds the
#: measured 1.65 up rather than fitting it.
CONTENTION = 2.0


def tight_claims(todo: list[Claim], timeout: int, secs: dict) -> list[Claim]:
    """Claims whose recorded duration leaves no room for that. Warn, never skip.

    A refusal here would be a gate that declines to run, which is worse than a
    slow one; and a timeout raised automatically would be this file quietly
    deciding what "did not finish" means. Both belong to the person reading it.
    """
    return [c for c in todo if secs.get(c.cmd, 0.0) * CONTENTION > timeout]


def capacity(resource: str, gpu_jobs: int) -> int:
    """How many claims may hold this resource at once. One, except maybe "gpu".

    Every resource but "gpu" is a mutex and stays one: they stand for a file
    somebody writes, and two writers is the data race the whole scheme exists to
    prevent. "gpu" is different in kind - it stands for a machine resource that
    is SHARED, not corrupted, by two users. Oversubscribing MPS does not produce
    a wrong number, it produces a slower one, and slow only becomes wrong if it
    crosses the per-claim timeout.

    That is why this is a knob and the others are not, and why the knob defaults
    to 1: the mutex was chosen when the gpu family cost 614 s under a 1273 s
    binding chain, i.e. when it was free. After the 2026-09-20 conversions it IS
    the binding chain, so whether it can be relaxed is now worth measuring - and
    measuring, not arguing, because the failure it risks is a verdict that
    depends on scheduling.

    PRE-REGISTERED PASS RULE, committed before the run that judges it so
    protocol.py can prove the order rather than take my word (the repo has five
    measurements where it could only say SAME_COMMIT). Population: every claim
    whose resources() contains "gpu", 71 at the time of writing. Metric:
    per-claim wall seconds, and the full verdict set. Cap: the 600 s
    .githooks/pre-push passes.

        ACCEPT --gpu-jobs 2 as the default ONLY IF BOTH:
          A1  every gpu-family claim finishes UNDER 300 s, half the cap. The
              longest measured serially is 77.8 s, so A1 allows 3.85x inflation
              against the 1.65x measured at -j4 - generous, and well short of
              the 7.7x that would actually breach the cap.
          A2  the verdict set is IDENTICAL, claim for claim, to the -j1 run.

        REVERT to the mutex if A1 fails for any claim or A2 fails anywhere. A
              faster floor bought with one flaky claim is not a trade this repo
              makes: the mutex exists so a verdict cannot depend on scheduling,
              which is worth more than the minutes.

        REPORT EITHER WAY the per-claim durations of the whole gpu family, so
              the next reader sees the headroom instead of trusting a verdict.

    n=1 configuration tried. If a second capacity is ever measured, that n moves
    and the rule above stops being a pre-registration and becomes a sweep.
    """
    return gpu_jobs if resource == "gpu" else 1


def verify(todo: list[Claim], timeout: int, jobs: int = 1,
           emit=None, secs=None, gpu_jobs: int = 1) -> list[dict]:
    """Run every claim exactly once and return one row each, in claim order.

    jobs=1 is the original loop verbatim. Above that, longest-first under the
    resource exclusion above. `emit(claim, status, missing, secs)` prints as
    results land, from THIS thread - the workers only run subprocesses.
    """
    rows: list[dict] = []

    def finish(c, res):
        status, missing, took = res
        if emit:
            emit(c, status, missing, took)
        rows.append({"cmd": c.cmd, "key": c.key, "expect": c.values,
                     "status": status, "secs": round(took, 1)})

    # ONE EXECUTION PER DISTINCT COMMAND. Claims are scheduled as GROUPS from
    # here down: a group is every claim quoting one command, and it is the unit
    # that runs, holds resources and gets timed. Two claims reading two numbers
    # out of one table are one run, not two.
    groups: dict[str, list[Claim]] = {}
    for c in todo:
        groups.setdefault(c.cmd, []).append(c)

    if jobs <= 1:
        for g in groups.values():
            for c, res_ in run_group(g, timeout):
                finish(c, res_)
        # Claim order, not command order: rows are read positionally elsewhere.
        pos = {c.key: i for i, c in enumerate(todo)}
        rows.sort(key=lambda r: pos[r["key"]])
        return rows

    known = _known_modules()
    res = {c.key: resources(c.cmd, known) for c in todo}
    secs = recorded_secs() if secs is None else secs
    for c in tight_claims(todo, timeout, secs):
        print(f"  NOTE  {c.key} last ran in {secs[c.cmd]:.0f}s against a "
              f"--timeout {timeout}; at -j{jobs} a claim can take "
              f"{CONTENTION:g}x that. A TIMEOUT here is contention, not drift.",
              file=sys.stderr)
    # A claim with no recorded duration sorts FIRST, not last: an untimed claim
    # is usually a new one, and a new 400 s claim discovered at the end of the
    # run is exactly the tail this schedule exists to avoid.
    # The schedulable unit is the GROUP, keyed by its command. Its resources and
    # its recorded duration are the command's, which they always were - res and
    # secs are both keyed by cmd underneath.
    pend = sorted(groups.values(), key=lambda g: -secs.get(g[0].cmd, math.inf))
    order = {c.key: i for i, c in enumerate(todo)}
    # A COUNT, not a set: every resource still admits one holder except "gpu",
    # which admits gpu_jobs. With gpu_jobs=1 this is exactly the set behaviour.
    held: collections.Counter = collections.Counter()
    futs: dict[cf.Future, list] = {}
    with cf.ThreadPoolExecutor(jobs) as pool:
        while pend or futs:
            while len(futs) < jobs:
                nxt = next((g for g in pend
                            if all(held[r] < capacity(r, gpu_jobs)
                                   for r in res[g[0].key])), None)
                if nxt is None:
                    break       # everything left is blocked; wait for a finish
                pend.remove(nxt)
                held.update(res[nxt[0].key])
                futs[pool.submit(run_group, nxt, timeout)] = nxt
            # With nothing running, `held` is all zeros and the loop above must
            # have accepted a claim. If it did not, a resource was taken and
            # never given back - and `cf.wait` on an empty set returns at once,
            # so the honest name for what happens next is a SPIN, not a wait.
            # This is a guard rather than a comment because that is the whole
            # lesson of this file: a gate that hangs is worse than one that
            # fails, since a hang gets killed and a kill reads as drift.
            if pend and not futs:
                raise RuntimeError(
                    f"scheduler stuck: {len(pend)} groups pending, none "
                    f"runnable, nothing running. Held: {dict(+held)}. A "
                    f"resource was acquired and not released.")
            for f in cf.wait(futs, return_when=cf.FIRST_COMPLETED).done:
                g = futs.pop(f)
                held.subtract(res[g[0].key])
                for c, res_ in f.result():
                    finish(c, res_)
    rows.sort(key=lambda r: order[r["key"]])
    return rows


def log(rows: list[dict], jobs: int = 1, wall: float = 0.0, scope: str = "") -> None:
    """Append one row per claim, each stamped with the RUN's shape.

    jobs/wall/scope are properties of the run, not of the claim, and they are
    repeated on every row so a row is readable alone. A parallel run's wall time
    is not the sum of its rows' secs - that gap is what --jobs is for, and
    without wall on the row there is nothing to compare serial against.
    """
    sha, dirty = head()
    run = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a") as f:
        for r in rows:
            f.write(json.dumps({"t": run, "git": sha, "dirty": dirty,
                                "jobs": jobs, "wall": round(wall, 1),
                                "scope": scope, **r}) + "\n")


def timing(path: Path | None = None) -> int:
    """Every logged run, one line: scope, jobs, wall, claims, verdicts.

    This is the artefact behind any sentence about how long the gate takes. A
    timing reported from a terminal that has scrolled away is not a measurement;
    a row here is, and it names the scope and the jobs it was taken at, because
    wall seconds without both are not comparable to anything.
    """
    path = LOG if path is None else path
    if not path.exists():
        print("no model/validation/claims.log yet - nothing has been verified")
        return 1
    runs: dict[str, list[dict]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "wall" in r:                      # rows written before this field
            # jobs/wall/scope are IN the key: `t` is one-second resolution and
            # stamped at the end of a run, so two runs that finish in the same
            # second merged into one line - the serial arm swallowing the parallel
            # arm's claims and printing a sum of claim seconds larger than its own
            # wall, with no complaint. Several sessions share this repo.
            runs[f'{r["t"]}|{r["git"]}|{r["jobs"]}|{r["wall"]}|{r["scope"]}'].append(r)
    if not runs:
        print("no run in claims.log carries a wall time - re-run without --no-log")
        return 1
    for stamp, rows in sorted(runs.items()):
        t, sha = stamp.split("|")[:2]
        kinds = collections.Counter(r["status"] for r in rows)
        slowest = max(rows, key=lambda r: r["secs"])
        print(f'{t}  {sha:8s} scope={rows[0]["scope"]:4s} jobs={rows[0]["jobs"]} '
              f'wall={rows[0]["wall"]/60:6.1f} min  {len(rows):3d} claims  '
              f'{", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))}')
        print(f'{"":33s}sum of claim secs {sum(r["secs"] for r in rows)/60:.1f} min, '
              f'slowest {slowest["secs"]/60:.1f} min: {slowest["cmd"][:60]}')
    return 0


def history() -> int:
    """The tuning odometer. Distinct recorded values per command = knob turns."""
    if not LOG.exists():
        print("no model/validation/claims.log yet - nothing has been verified")
        return 0
    per = collections.defaultdict(list)
    for line in LOG.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            per[e["cmd"]].append(e)
    print(f"{'runs':>5} {'values':>7}  command")
    for cmd, es in sorted(per.items(), key=lambda kv: -len(kv[1])):
        vals = {tuple(e["expect"]) for e in es}
        flag = "  <-- re-recorded; a claim that moves this often is a knob" \
            if len(vals) > 2 else ""
        print(f"{len(es):5d} {len(vals):7d}  {cmd}{flag}")
        if len(vals) > 1:
            for v in sorted(vals):
                last = max(e["t"] for e in es if tuple(e["expect"]) == v)
                print(f"{'':13}  {last}  {','.join(v)}")
    return 0


def lint(claims: list[Claim], verbose: bool = True) -> int:
    """Rule 6, and it is the only rule a machine can check at all."""
    bad = [c for c in claims
           if c.runnable and HELD_OUT.search(c.context()) and "n=" not in c.context()]
    for c in bad:
        if verbose:
            print(f"  n= MISSING  {c.key}  held-out/test claim with no configuration "
                  f"count: {c.cmd}")
    return len(bad)


def main(argv=None, stdin=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="include #slow claims")
    ap.add_argument("--list", action="store_true", help="print every tag, run nothing")
    ap.add_argument("--history", action="store_true", help="the tuning odometer")
    ap.add_argument("--timing", action="store_true",
                    help="every logged run: scope, jobs, wall seconds, verdicts")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--gpu-jobs", type=int, default=1,
                    help="how many detector/CoreML/MPS claims may run at once. "
                         "1 (the default) is a mutex, which is what shipped; >1 "
                         "is under measurement, see capacity().")
    ap.add_argument("--jobs", "-j", type=int, default=4,
                    help="claims to run at once (1 = the original serial loop). "
                         "Claims that share a resource never overlap whatever "
                         "this is; see RUNNING THEM IN PARALLEL above.")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--touches-tag", action="store_true",
                    help="read a unified diff on stdin; print HEAVY if it "
                         "changes a line inside a tag block (the hook trigger)")
    a = ap.parse_args(argv)

    if a.touches_tag:
        # Prints the WORD, and the hook branches on the word, never on the exit
        # status: a traceback also exits 1, and "the checker crashed" must not
        # be indistinguishable from "no tag was touched". That is a trigger
        # that disarms itself silently, which is the whole subject here.
        hit = touches_tag((stdin or sys.stdin).read())
        print("HEAVY" if hit else "LIGHT")
        return 0 if hit else 1

    if a.timing:
        return timing()
    if a.history:
        return history()

    census = collections.Counter()
    claims = scan(census)
    why = vacuous(claims)

    if a.list:
        for c in claims:
            print(f"@{c.sigil:3s} {c.key:44s} {','.join(c.values):28s} {c.cmd}")
        print(f"\n{len(claims)} tags: " + ", ".join(
            f"{n} @{s}" for s, n in sorted(
                collections.Counter(c.sigil for c in claims).items())))
        print(report_census(census))
        live = [c for c in claims if c.runnable and not c.example]
        syn = [c for c in live if syntax_error(c.cmd)]
        small = [c for c in live if unanchored(c)]
        for c in syn:
            print(f"  SYNTAX  {c.key}  {c.cmd[:70]}")
        print(f"{len(syn)} runnable tag(s) fail `sh -n` (SYNTAX, never run); "
              f"{len(small)} carry a <3-digit integer that needs an output label "
              f"or a declared `name=` anchor")
        if why:
            banner(why)
        return 1 if (why or syn) else 0

    # BEFORE THE FIRST CLAIM, never after: the whole point is that no claim
    # gets to print a verdict about a tree that cannot run it.
    if (missing := preflight()):
        incomplete(missing)
        return 1

    todo = [c for c in claims if c.runnable and not c.example]
    skipped = [c for c in todo if c.slow and not a.all]
    todo = [c for c in todo if not (c.slow and not a.all)]

    def emit(c, status, missing, secs):
        print(f"{status:7s} {secs:5.1f}s  {c.key}")
        print(f"              {c.cmd}")
        if status != "OK":
            print(f"              recorded {','.join(c.values)} - "
                  f"NOT IN OUTPUT: {','.join(missing)}")

    before = tree_paths()
    t0 = time.monotonic()
    rows = verify(todo, a.timeout, a.jobs, emit, gpu_jobs=a.gpu_jobs)
    wall = time.monotonic() - t0
    bad = sum(r["status"] != "OK" for r in rows)
    gone = tree_moved(before)

    if not a.no_log and rows:
        log(rows, jobs=a.jobs, wall=wall, scope="all" if a.all else "fast")

    print()
    # WHY THE FAILURES ARE COUNTED BY KIND. The push hook printed "a claim
    # drifted" for every non-zero exit, and on 2026-09-19 it printed it twice
    # for a run that had been SIGTERMed and once for a command that could never
    # finish inside the budget. Neither is drift, and the sentence sent a person
    # hunting for a number that had not moved. The exit status stays 1 for all
    # of them - an unverified claim is unverified - but the reader is told which.
    kinds = collections.Counter(r["status"] for r in rows if r["status"] != "OK")
    if kinds:
        print("FAILED: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
        if kinds["TIMEOUT"]:
            print(f"  TIMEOUT is not drift: the command did not finish inside "
                  f"--timeout {a.timeout}s. If it never can, it is not a measured "
                  f"tag (rule 2). Cite the artefact it writes, name the producer in "
                  f"prose (st2-p1-step8 (not published), 1e8af8f).")
        if kinds["SYNTAX"]:
            print("  SYNTAX: the command does not parse under `sh -n` and was never "
                  "run. Usually prose after it on the tag line - the command ends at "
                  "the first run of two spaces (cut_cmd).")
        if kinds["ERROR"]:
            print("  ERROR: the command exited non-zero. A failed run verifies "
                  "nothing, even if the number appears in its output.")
        if kinds["KILLED"]:
            print("  KILLED is not drift: the command was terminated by a signal "
                  "(a backgrounded push reaped, an OOM, a Ctrl-C). Nothing was "
                  "measured. Re-run in the foreground before touching a number.")
    if gone:
        moved(gone, bad)
    if (dirty := cited_dirty()):
        diverged(dirty)
    bad += lint(claims)
    n_c = sum(c.sigil in CITED for c in claims)
    n_d = sum(c.sigil in DERIVED for c in claims)
    ok = len(todo) - sum(r["status"] != "OK" for r in rows)
    print(f"{ok}/{len(todo)} @M verified"
          + (f", {len(skipped)} #slow skipped (--all)" if skipped else "")
          + f"; {n_c} @C and {n_d} @D are not machine-checkable by construction")
    # The count never ships alone. See report_census().
    print(report_census(census))
    outside = sum(c.runnable and not c.example and not c.mine for c in todo + skipped)
    print(f"{outside} of those runnable tags live outside {ME}; "
          f"{len(todo) + len(skipped) - outside} are its own seeds and specimens")
    if why:
        banner(why)
        return 1
    return 1 if bad else 0


def _selfcheck() -> None:
    """The parser and the token matcher, which is where this can be wrong.

    Adversarially: every assertion here is a mutation that would make the checker
    LIE - report OK on a drifted number - not one that would make it crash.
    """
    c = Claim("M", "0.3548,11.3", "python x.py --eval  #slow", "f.py", 1, [])
    assert c.values == ["0.3548", "11.3"] and c.slow and c.cmd == "python x.py --eval"
    assert Claim("M", "1", "x  #example", "f", 1, []).example

    # The grammar line in this file's own docstring must NOT parse as a claim.
    # It did, and the checker executed `<shell command>   what was held fixed`.
    assert not TAG.search("    @M <value[,value...]> :: <shell command>   fixed")
    # (marked #example so the scanner does not execute this file's own fixture)
    assert TAG.search("| **IDF1 0.3548** | @M 0.3548 :: model/engine/stitch.py --eval #example |")

    # A drifted value must not be found inside a longer number. This is the whole
    # correctness of the matcher: plain `in` reports OK for all four of these.
    assert token_present("11.3", "frag  11.3  switch")
    assert not token_present("11.3", "frag 111.3 switch"), "matched a longer number"
    assert not token_present("11.3", "frag 11.35 switch"), "matched a longer number"
    assert not token_present("0.354", "IDF1 0.3548"), "matched a prefix"
    assert not token_present("2.50", "12.50"), "matched a suffix"
    assert token_present("15.4%", "shipped 139 15.4% 515")

    # THE 2026-09-29 REVIEW'S HOLES. Specimens are ASSEMBLED (`at + "M"`) so the
    # scanner does not find them as tags in this file. Each assertion below
    # fails on the code before the fix it names.
    at = "@"
    # (a) the command ends where the prose begins - n37-full-half (not published):82
    _n37 = (f"    {at}M 0 :: .venv/bin/python model/validation/n37_full_half.py --show   "
            "events matched to a feed player, either gate (\"unmatched {'far': 2}\")")
    _m = TAG.search(_n37)
    _c37 = Claim(_m.group(1), _m.group(2), _m.group(3), "docs/x.md", 82, [])
    assert _c37.cmd == ".venv/bin/python model/validation/n37_full_half.py --show", \
        f"prose after the command was run as part of it: {_c37.cmd!r}"
    assert syntax_error(_m.group(3)), "the uncut n37 line parses - the specimen is not the hazard"
    assert cut_cmd("python -c \"a;  b\"  prose") == "python -c \"a;  b\"", "a quoted run was cut"
    # (b) a command that does not parse is SYNTAX, and is never executed
    _sent = os.path.join(tempfile.gettempdir(), f"vc_syn_{os.getpid()}")
    with contextlib.suppress(OSError):
        os.remove(_sent)
    assert run(Claim("M", "1.5", f"touch {_sent}; echo 1.5 (", "f", 1, []), 30)[0] == "SYNTAX"
    assert not os.path.exists(_sent), "a command that fails sh -n was executed"
    # (c) rc != 0 is NEVER OK, whatever the output contains
    assert judge(_c37, "sh: line 0: syntax error near unexpected token `('", 2)[0] == "ERROR", \
        "the n37 shell error verified its recorded 0 off `line 0`"
    assert judge(Claim("M", "1.5", "x", "f", 1, []), "value: 1.5", 1)[0] == "ERROR", \
        "a failed command verified a value it printed"
    _undef = Claim("M", "1", '.venv/bin/python -c "print(1+undefined)"', "f", 1, [])
    assert run(_undef, 60)[0] == "ERROR", "a NameError verified a recorded 1 off `line 1`"
    # (d) whole numbers: sign, trailing fraction and exponent are part of the number
    assert not token_present("0.56", "delta -0.56"), "matched a sign-flipped number"
    assert token_present("-0.56", "delta -0.56")
    assert not token_present("1.5", "lr 1.5e-3"), "matched a mantissa"
    assert not token_present("11.3", "v 11.3.4"), "matched a dotted prefix"
    # (e) a <3-digit integer must be anchored by a label, or declare one
    assert not token_present("1", 'File "<string>", line 1, in <module>'), \
        "an unanchored small integer matched a traceback line number"
    assert token_present("2", "unmatched {'far': 2}") and token_present("9", "n=9")
    assert token_present("12", "hits: 12") and token_present("100", "count 100")
    assert token_present("hits=2", "hits 2") and not token_present("hits=2", "miss 2")
    assert TAG.search(f"{at}M hits=2 :: x").group(2) == "hits=2", "a declared anchor did not parse"
    # (f) the comma always separates: 1,352 is two values, and the `1` fails loudly
    assert Claim("M", "1,352", "x", "f", 1, []).values == ["1", "352"]
    assert not token_present("1", "total 1,352")
    # (g) a `-` after a digit is a range, not a sign - n50-ball-size-gate (not published):105
    assert token_present("5.0", "D_1710_1740      1.7-5.0       48"), "a range hyphen read as a sign"
    assert not token_present("-5.0", "1.7-5.0") and not token_present("5.0", "range -5.0")
    # (h) an integer matches itself with only zeros after the point - n41-conf.md:29
    assert token_present("291", "IDSW 291.0000->339.0000") and token_present("339", "291.0000->339.0000")
    assert not token_present("291", "IDSW 291.5") and not token_present("1.5", "v 1.50")
    # (i) a declared label: the literal text printed right before the value
    assert token_present("IDSW 29", "IDF1 0.4 IDSW 29 ids 5") and not token_present("ids 29", "IDSW 29 ids 5")
    assert token_present("of 7", "on 5 of 7 clips") and not token_present("of 7", "proof 7")
    assert token_present("shipped 45", "  shipped     45   15.0%") and token_present("ids 44", "ids 44.000->59")
    assert token_present('"22,6.76," 35', "22,6.76,35,10.85") and not token_present('"22,6.76," 35', "22,6.97,35")
    assert Claim("M", '"a, b" 7,of 3', "x", "f", 1, []).values == ['"a, b" 7', "of 3"], "a quoted comma split"
    assert token_present("2.4 s", "median 2.4 s") and not token_present("2.4 s", "median 2.4 m"), \
        "a non-numeric last word is not a label form - the whole value stays literal"
    assert TAG.search(f"{at}M of 7, IDSW 291 :: x").group(2) == "of 7, IDSW 291", "a label value did not parse"
    assert TAG.search(f'{at}M "a, b" 7 :: x').group(2) == '"a, b" 7'
    assert not TAG.search(f"{at}M means nothing :: here"), "prose parsed as a labelled value"

    # timing(): a run written without the run's shape must not be read as one
    # taken at jobs=1. Rows predating the field are SKIPPED, not defaulted - a
    # serial/parallel comparison assembled from rows that never recorded which
    # they were is the exact claim this reader exists to make impossible.
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "claims.log"
        old = {"t": "2026-09-20T11:20:36", "git": "1e8af8f", "dirty": True,
               "cmd": "x", "key": "k", "expect": ["1"], "status": "OK", "secs": 3.0}
        log_path.write_text(json.dumps(old) + "\n")
        assert timing(log_path) == 1, "read a pre-wall row as a timed run"
        new = {**old, "t": "2026-09-22T06:00:00", "jobs": 4, "wall": 762.0, "scope": "all"}
        log_path.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n")
        assert timing(log_path) == 0

        # TWO RUNS THAT FINISH IN THE SAME SECOND ARE TWO RUNS. `t` is
        # one-second resolution, so keying on it alone merged the serial arm and
        # the parallel arm into one line: one wall time, the other's claims
        # counted into it, and a sum of claim seconds larger than the wall it was
        # printed beside. Several sessions share this repo, so the collision is
        # not hypothetical.
        serial = {**old, "t": "2026-09-22T07:00:00", "jobs": 1, "wall": 2507.0, "scope": "all"}
        par = {**serial, "jobs": 4, "wall": 822.0}
        log_path.write_text("\n".join(json.dumps(r) for r in (serial, par)) + "\n")
        with contextlib.redirect_stdout(io.StringIO()) as _t:
            timing(log_path)
        assert _t.getvalue().count("2026-09-22T07:00:00") == 2, \
            "two runs in one second collapsed into one line:\n" + _t.getvalue()

    # ONE EXECUTION PER COMMAND, asserted by counting executions rather than by
    # timing them. Three claims quoting three numbers from one command must run
    # it ONCE - 98 of this repo's 400 runnable claims share a command, worth
    # 753 s of re-running per --all before this. Both paths, because the serial
    # loop is the reference the parallel one is diffed against.
    with tempfile.TemporaryDirectory() as _d:
        _mark = os.path.join(_d, "runs")
        _cmd = f"echo x >> {_mark}; echo 1.5 2.5 n=9"
        _cs = [Claim("M", v, _cmd, "f", i, []) for i, v in enumerate(("1.5", "2.5", "9"))]
        for _jobs in (1, 4):
            open(_mark, "w").close()
            _rows = verify(_cs, 30, jobs=_jobs)
            _n = len(open(_mark).read().split())
            assert _n == 1, f"jobs={_jobs}: one command, three claims, {_n} executions"
            assert [r["status"] for r in _rows] == ["OK"] * 3, _rows
            assert [r["key"] for r in _rows] == [c.key for c in _cs], \
                f"jobs={_jobs}: rows came back out of claim order: {[r['key'] for r in _rows]}"
        # ...and a shared command that FAILS must fail every claim quoting it,
        # not just the first: judging is per claim, execution is per command.
        _bad = [Claim("M", "1.5", "echo nope", "f", 9, []), Claim("M", "2.5", "echo nope", "f", 10, [])]
        assert [r["status"] for r in verify(_bad, 30, jobs=4)] == ["DRIFT", "DRIFT"]

    # Rule 6: a held-out claim without n= is an error, with n= is not. Getting
    # this backwards is silent, so it is asserted in both directions.
    held = Claim("M", "0.4", "cmd", "f", 1, ["held-out split, n=6 margins"])
    assert lint([held], verbose=False) == 0
    assert lint([Claim("M", "0.4", "cmd", "f", 1, ["held-out split"])], False) == 1
    assert lint([Claim("C", "0.4", "cmd", "f", 1, ["held-out split"])], False) == 0

    # The four aliases must parse, and `@DER` must not be read as `@D` - the
    # alternation order is the only thing standing between those two, and
    # getting it wrong loses a sigil silently, which is how we got here.
    # (every fixture below carries #example, or the scanner finds these very
    #  lines and executes them - the same trap the docstring grammar line set.)
    assert TAG.search("@V 0.055 :: st2_ab.py --a .c03 --b .c03s  #example")
    assert TAG.search("@RAN 8/10 :: x  #example")
    assert TAG.search("@REC 1.5 :: docs/x.md:12  #example")
    assert TAG.search("@DER 0.42 :: 0.6 * 0.7  #example").group(1) == "DER", \
        "@DER read as @D - fix the alternation order in SIGILS"
    assert not TAG.search("@MEOW 1 :: x"), "\\b missing: @MEOW parsed as @M"
    assert Claim("V", "1", "x", "f", 1, []).runnable, "@V must be runnable"
    assert not Claim("REC", "1", "x", "f", 1, []).runnable, "@REC is not measured"

    # THE FAILURE THIS FILE ACTUALLY HAD, asserted in BOTH directions. A guard
    # that only ever returns None is exactly as useless as no guard, and this
    # repo has already shipped five checks that could not fail.
    self_only = [Claim("M", "1", "true", ME, 1, [])]
    elsewhere = Claim("M", "1", "true", "docs/x.md", 1, [])
    assert vacuous([]), "an empty claim set must be vacuous"
    assert vacuous(self_only), "a self-only claim set must be vacuous"
    assert vacuous([Claim("C", "1", "x", "docs/x.md", 1, [])]), "no @M is vacuous"
    assert vacuous(self_only + [Claim("M", "1", "true  #example", "docs/x.md", 1, [])]), \
        "an #example outside this file is a specimen, not a subject"
    assert vacuous(self_only + [elsewhere]) is None, "a real outside tag is not vacuous"

    # The scanner must actually find the tags in this file, or every run is a
    # vacuous pass - the failure mode that made four of five self-checks useless.
    census = collections.Counter()
    all_claims = scan(census)
    found = [c for c in all_claims if c.mine]
    assert len(found) >= 6, f"scanner found {len(found)} tags in its own docstring"
    # All seven sigils appear in this file - three as seed claims, four as the
    # alias fixtures above. If one drops out, the alternation lost it.
    assert {c.sigil for c in found} == set(RUNNABLE | CITED | DERIVED), \
        f"a sigil went missing: {sorted({c.sigil for c in found})}"
    assert any("stitch.py --eval" in c.cmd for c in found)
    assert sum(c.example for c in found) == 12, \
        f"an #example specimen is runnable ({sum(c.example for c in found)} marked)"

    # AND AGAINST THE REAL REPO. This is the assertion that goes red the day the
    # docs lose their tags again - the regression the whole edit exists to stop.
    # It is deliberately a claim about the world and not about a fixture, which
    # means it can rot, and its rotting IS the signal.
    assert vacuous(all_claims) is None, \
        "VACUOUS: nothing outside this file carries a runnable tag - " \
        "the checker has regressed to checking itself"
    outside = [c for c in all_claims if c.runnable and not c.example and not c.mine]
    assert len(outside) >= 5, f"only {len(outside)} runnable tags outside this file"

    # A KILLED run must not be reported as a claim that moved. Both directions:
    # a real drift must still read DRIFT, or this just renames every failure.
    killed = Claim("M", "1.23", "sh -c 'kill -TERM $$'", "f", 1, [])
    assert run(killed, 30)[0] == "KILLED", "a SIGTERMed command was not read as killed"
    assert run(Claim("M", "1.23", "echo 9.99", "f", 1, []), 30)[0] == "DRIFT", \
        "a real drift stopped reading DRIFT"
    assert run(Claim("M", "1.23", "echo 1.23", "f", 1, []), 30)[0] == "OK"
    assert run(Claim("M", "1.23", "sh -c 'echo 1.23; exit 3'", "f", 1, []), 30)[0] == "ERROR", \
        "a non-zero exit verified a value because it was in the output - rc != 0 is never OK"
    assert run(Claim("M", "1.23", "sleep 5", "f", 1, []), 1)[0] == "TIMEOUT"

    # GIT_DIR beats cwd=. The scanner must walk THIS tree whatever the ambient
    # environment says, or a gate run from inside a hook prints a green count
    # over a file set that is not the one being pushed.
    # AN UNREADABLE GIT_DIR, and the reason it is not another real checkout is
    # worth writing down, because the first version of this check tried that
    # and PASSED under the mutation.
    #
    # `scan()` reads `ls-files` AND `ls-files --others --exclude-standard`.
    # Point GIT_DIR at a second checkout and the first call returns that tree's
    # index while the second, run from this cwd, returns every local file that
    # index does not contain - so the union BACKFILLS and the scan goes long,
    # not short. Measured 2026-09-20 against the ui-fixes checkout:
    # 731 vs 767 tracked, but 1234 vs 767 once `--others` is added.
    #
    # So verify_claims, bare_markers and derived are masked by their own
    # backfill. The scanner with no `--others` fallback is NOT:
    # `shipped_config.check_constants` reads `git ls-files "*.py"` alone and
    # sees 161 files instead of 166 - five config constants nothing looks at,
    # in green. That is the live exposure, and it is why this strips GIT_* in
    # every scanner rather than only where a test can reach it.
    _saved = os.environ.get("GIT_DIR")
    _want = len(all_claims)
    os.environ["GIT_DIR"] = "/nonexistent/.git"
    try:
        _under = scan()
        # Counts as well as emptiness: any change to the scanned file set is a
        # run whose subject is not the tree being pushed.
        assert [c for c in _under if c.mine], \
            "an ambient GIT_DIR emptied the scan - git_env() is not reaching ls-files"
        assert len(_under) == _want, \
            f"an ambient GIT_DIR changed the scanned file set: {len(_under)} " \
            f"tags against {_want} - the gate would run green over another tree"
    finally:
        os.environ.pop("GIT_DIR") if _saved is None else os.environ.update(GIT_DIR=_saved)
    # The unit property, and it comes SECOND on purpose. It proves git_env()
    # filters; only the end-to-end scan above proves the filtering reaches
    # ls-files, and dropping `env=` from that one subprocess is the realistic
    # regression. A check whose first assertion is the weaker one reports the
    # weaker failure, which is how a guard gets trusted for the wrong reason.
    assert "GIT_DIR" not in git_env({"GIT_DIR": "/elsewhere/.git", "PATH": "/bin"})
    assert git_env({"GIT_DIR": "/x", "PATH": "/bin"}) == {"PATH": "/bin"}

    # THE CONTINUATION-LINE HOLE, both directions. A diff that changes only the
    # line saying what was held fixed must set HEAVY; ordinary prose must not,
    # or the trigger degrades to "always run the hour-long gate", which is how
    # a gate gets bypassed - the failure this whole audit is about.
    _cont = ("@@ -1,5 +1,5 @@\n"
             " @M 0.364 :: model/validation/status.py --variant .c03s  #example\n"
             "-   ten sequences, conf 0.05, held-out split, n=1.\n"
             "+   ten sequences, conf 0.5, held-out split, n=1.\n"
             " \n text\n")
    _tagline = ("@@ -1,2 +1,2 @@\n"
                "-@M 0.364 :: cmd  #example\n"
                "+@M 0.400 :: cmd  #example\n")
    _prose = ("@@ -1,4 +1,4 @@\n @M 0.364 :: cmd  #example\n"
              "    held fixed: n=1.\n \n-some prose\n+other prose\n")
    # The OLD trigger - the hook's grep, written out in Python - must MISS the
    # continuation edit and still catch the tag-line edit. Without this pair the
    # test below proves only that the new check fires, not that it closes a hole.
    # built from SIGILS, not a literal - the hook greps for any sigil, and a
    # second copy of the alternation here is the one thing this file forbids.
    _old_grep = lambda d: bool(  # noqa: E731
        re.search(rf"^[-+].*@({SIGILS})\b\s", d, re.M))
    assert not _old_grep(_cont), \
        "the fixture carries a sigil on a changed line - the old grep would " \
        "have caught it, so this is not testing the hole"
    assert _old_grep(_tagline), "the old grep must still catch a changed tag line"
    assert touches_tag(_cont), \
        "a continuation-line edit did not set HEAVY - the conf 0.05/0.5 shape"
    assert touches_tag(_tagline), "a changed tag line did not set HEAVY"
    assert not touches_tag(_prose), \
        "an edit outside every tag block set HEAVY - the trigger is now always on"
    assert not touches_tag(""), "an empty diff set HEAVY"
    # The hook branches on the printed word. Both must print, and differ, or the
    # fail-closed `case` in .githooks/pre-push is a coin flip.
    for _d, _want in ((_cont, "HEAVY"), (_prose, "LIGHT")):
        _out = io.StringIO()
        with contextlib.redirect_stdout(_out):
            main(["--touches-tag"], stdin=io.StringIO(_d))
        assert _out.getvalue().strip() == _want, \
            f"--touches-tag printed {_out.getvalue()!r}, wanted {_want}"

    # PARALLEL MUST NOT CHANGE A VERDICT. Asserted on a planted fixture, run
    # BOTH ways, because "it was faster and still green" is not evidence: the
    # only question is whether -j4 and -j1 disagree, and a fixture with nothing
    # but passing claims in it cannot answer that. So the fixture carries one of
    # each verdict the gate can print - OK, DRIFT, KILLED, TIMEOUT - and the two
    # runs are compared as a whole map, not as a count.
    #
    # The two `#lockstep` claims are the mutual-exclusion test and they are the
    # reason this is a real check rather than a smoke test. Both name
    # `model/validation/dist_eval.py`, so `resources()` puts them in one chain;
    # each then creates a lock file, holds it, and refuses to print its value if
    # the other already holds it. If the exclusion ever breaks they go DRIFT at
    # -j4 and stay OK at -j1, which is exactly the failure being guarded - a
    # verdict that depends on the worker count. `mkdir` because it is the one
    # atomic test-and-set every shell has.
    _lk = os.path.join(tempfile.gettempdir(), f"vc_selfcheck_{os.getpid()}")
    _excl = (f"sh -c 'mkdir {_lk} 2>/dev/null || exit 0; sleep 0.4; "
             f"rmdir {_lk}; echo 4.20'")   # one space before the `#` below: two would end the command
    _fix = [Claim("M", "1.23", "echo 1.23", "fix.md", 1, []),
            Claim("M", "1.23", "echo 9.99", "fix.md", 2, []),
            Claim("M", "1.23", "sh -c 'kill -TERM $$'", "fix.md", 3, []),
            Claim("M", "1.23", "sleep 9", "fix.md", 4, []),
            Claim("M", "4.20", f"{_excl} # model/validation/dist_eval.py", "fix.md", 5, []),
            Claim("M", "4.20", f"{_excl} # model/validation/dist_eval.py", "fix.md", 6, []),
            Claim("M", "0.5", "echo 0.5", "fix.md", 7, [])]
    _want = {"fix.md:1": "OK", "fix.md:2": "DRIFT", "fix.md:3": "KILLED",
             "fix.md:4": "TIMEOUT", "fix.md:5": "OK", "fix.md:6": "OK",
             "fix.md:7": "OK"}
    with contextlib.suppress(OSError):
        os.rmdir(_lk)
    _ser = verify(_fix, timeout=2, jobs=1)
    _par = verify(_fix, timeout=2, jobs=4, secs={})
    assert {r["key"]: r["status"] for r in _ser} == _want, \
        f"the serial fixture is not what it claims to be: {_ser}"
    assert {r["key"]: r["status"] for r in _par} == _want, \
        f"-j4 changed a verdict: {[(r['key'], r['status']) for r in _par]}"
    assert [r["key"] for r in _par] == [r["key"] for r in _ser], \
        "rows came back out of claim order - the log would not be comparable"
    assert len(_par) == len(_fix), "a claim was dropped by the parallel path"
    # THE TRACKED-ARTEFACT DIVERGENCE CHECK. Against the real repo, both ways:
    # a clean tree reports nothing, and a cited artefact edited on disk is
    # reported by path. Uses the repo because the whole question is what git
    # thinks, and a mock git would only be testing the mock.
    assert not cited_dirty(), \
        f"cited artefacts already differ from their commits: {cited_dirty()}"
    _cited = subprocess.run(["git", "ls-files", "--", "model/validation/out"],
                            cwd=ROOT, env=git_env(), capture_output=True,
                            text=True).stdout.splitlines()
    if _cited:                      # only once the artefacts are actually tracked
        _f = ROOT / _cited[0]
        _orig = _f.read_bytes()
        try:
            _f.write_bytes(_orig + b"\n")
            assert _cited[0] in cited_dirty(), \
                "a cited artefact was edited on disk and the run did not say so " \
                "- the gate would verify numbers that are not the ones in git"
        finally:
            _f.write_bytes(_orig)
        assert not cited_dirty(), "the fixture did not restore the artefact"

    # AND THAT main() SAYS SO. Third call site guarded this way tonight, for the
    # third reason the transcript can be misleading - here the numbers verify
    # fine and the evidence of record is simply not what was verified.
    _real_cd, _real_scan3 = cited_dirty, scan
    globals()["cited_dirty"] = lambda: ["model/validation/out/stc2025/n22_results.json"]
    globals()["scan"] = lambda census=None: [
        Claim("M", "1.11", "echo 1.11", "docs/x.md", 1, [])]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as _o:
            main(["--no-log"])
    finally:
        globals()["cited_dirty"], globals()["scan"] = _real_cd, _real_scan3
    assert "CITED ARTEFACTS DIFFER" in _o.getvalue(), \
        "a cited artefact diverged from its commit and the transcript never " \
        "said so - the push would carry evidence nobody updated"
    assert "n22_results.json" in _o.getvalue(), \
        "the divergence notice did not name which artefact"

    # THE MOVED-TREE GUARD, and the direction that matters most is the NEGATIVE
    # one: a guard that fires when a producer writes a file would fire on every
    # healthy run, and a banner nobody can trust is worse than no banner.
    with tempfile.TemporaryDirectory() as _d:
        _t = Path(_d)
        for _p in TREE:
            (_t / _p).mkdir(parents=True)
        _keep = _t / TREE[0] / "n22_results.json"
        _keep.write_text("{}")
        _w = _t / TREE[1] / "weights.pt"
        _w.write_text("x")
        _before = tree_paths(_t)
        assert len(_before) == 2, _before

        # a claim WRITING a new artefact, and REWRITING an existing one: normal
        (_t / TREE[0] / "n99_results.json").write_text("{}")
        _keep.write_text('{"rewritten": true}')
        assert not tree_moved(_before), \
            "an ordinary producer write set the moved-tree banner - it would fire " \
            "on every healthy run and stop meaning anything"

        # a file taken away mid-run: the incident
        _w.unlink()
        assert tree_moved(_before) == [str(_w)], \
            "a file that vanished mid-run was not reported - ten claims failed to " \
            "exactly this and every one of them read as drift"

    # THE INCOMPLETE-TREE GUARD, on real directories rather than a mock,
    # because every rule in it is a question about the filesystem and a mock
    # would only be testing this function against itself. Built complete, then
    # broken one way at a time - a guard that only ever sees the good tree is
    # the vacuous check this file already has a banner about.
    with tempfile.TemporaryDirectory() as _d:
        _t = Path(_d)
        for _p in REQUIRED:
            (_t / _p).parent.mkdir(parents=True, exist_ok=True)
            (_t / _p).mkdir() if "." not in Path(_p).name else (_t / _p).touch()
        for _p in REQUIRED_NONEMPTY:
            (_t / _p).mkdir(parents=True, exist_ok=True)
            (_t / _p / "linked").touch()
        (_t / "model" / "validation" / "out").mkdir(parents=True)
        assert not preflight(_t), f"a complete tree failed preflight: {preflight(_t)}"

        # 1. a gitignored path simply absent - the fresh-worktree case
        (_t / "data" / "match.db").unlink()
        assert "data/match.db" in preflight(_t), \
            "a missing gitignored path did not stop the run - 49 false drifts"
        (_t / "data" / "match.db").touch()
        assert not preflight(_t)

        # 2. a tracked directory present but EMPTY - the half-linked case that
        #    `exists()` alone calls fine
        (_t / "data" / "vendor" / "linked").unlink()
        assert any("data/vendor" in m for m in preflight(_t)), \
            "an empty data/vendor passed - exists() is not enough here"
        (_t / "data" / "vendor" / "linked").touch()
        assert not preflight(_t)

        # 3. out as a SYMLINK - resolves for every ordinary read, then breaks
        #    api.py's self-check. This is the rule that is not just `exists`.
        (_t / "model" / "validation" / "out").rmdir()
        (_t / "elsewhere").mkdir()
        (_t / "model" / "validation" / "out").symlink_to(_t / "elsewhere")
        assert (_t / "model" / "validation" / "out").is_dir(), \
            "the fixture is not the hazard: this symlink does not resolve"
        assert any("symlink" in m for m in preflight(_t)), \
            "a symlinked model/validation/out passed - api.py --verify breaks"

    # AND THAT main() ACTUALLY CONSULTS IT, BEFORE THE FIRST CLAIM. Checking
    # preflight() in isolation proves the rules; it does not prove the call is
    # still there, and deleting the call is the realistic regression - it is
    # three lines in a function nobody re-reads. So the claim that runs writes
    # a sentinel: if the file exists afterwards, a claim was executed against a
    # tree this file had already declared unrunnable.
    _sentinel = os.path.join(tempfile.gettempdir(), f"vc_pre_{os.getpid()}")
    with contextlib.suppress(OSError):
        os.remove(_sentinel)
    _real_scan, _real_pre = scan, preflight
    globals()["scan"] = lambda census=None: [
        Claim("M", "1", f"touch {_sentinel}", "docs/x.md", 1, [])]
    globals()["preflight"] = lambda root=None: ["data/jobs"]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as _o:
            _rc = main(["--no-log"])
    finally:
        globals()["scan"], globals()["preflight"] = _real_scan, _real_pre
    assert not os.path.exists(_sentinel), \
        "a claim RAN on a tree preflight had already refused - the guard is " \
        "after the loop, or main() no longer calls it"
    assert _rc == 1, f"an incomplete tree did not refuse the push (rc={_rc})"
    assert "INCOMPLETE WORKTREE" in _o.getvalue(), \
        "the refusal did not say the tree was incomplete, so it reads as drift"
    # No VERDICT line of any kind, which is the stronger statement: the banner
    # is allowed to say "nothing DRIFTED", a per-claim row is not.
    assert not re.search(r"^(OK|DRIFT|ERROR|TIMEOUT|KILLED)\s", _o.getvalue(), re.M), \
        "an incomplete tree printed a per-claim verdict - the exact confusion " \
        "this guard exists to end"

    # AND THAT main() REPORTS IT. Same reasoning as the preflight call-site
    # check: the rule being right does not put the banner in the transcript, and
    # the banner IS the product - without it the run prints failures and names
    # innocent tags, which is the incident.
    _real_tp, _real_scan2 = tree_paths, scan
    globals()["tree_paths"] = lambda root=None: {"/nonexistent/moved-away.json"}
    globals()["scan"] = lambda census=None: [
        Claim("M", "9.99", "echo 1.11", "docs/x.md", 1, [])]   # fails on purpose
    try:
        with contextlib.redirect_stdout(io.StringIO()) as _o:
            main(["--no-log"])
    finally:
        globals()["tree_paths"], globals()["scan"] = _real_tp, _real_scan2
    _txt = _o.getvalue()
    assert "ARTEFACT TREE CHANGED" in _txt, \
        "a file vanished mid-run and the transcript never said so - the failures " \
        "above it read as drift, which is the incident this exists to end"
    assert "moved-away.json" in _txt, "the banner did not name what went missing"
    assert "DRIFT" in _txt, \
        "the fixture no longer produces a failing claim, so this proves nothing " \
        "about a moved tree being reported ALONGSIDE failures"

    # AND AGAINST THE TREE THIS IS RUNNING IN, which is the assertion that goes
    # red in the worktree that is actually missing something.
    assert not preflight(), \
        f"this tree cannot run the gate; missing {preflight()}"

    # The contention warning, both directions - a warning that cannot fire is
    # the same decoration as a bare sigil.
    _t = Claim("M", "1", "slow-cmd", "f", 1, [])
    assert tight_claims([_t], 600, {"slow-cmd": 363.2}), \
        "the claim that came within 0.7 s of a TIMEOUT at -j4 is not flagged"
    assert not tight_claims([_t], 600, {"slow-cmd": 100.0})
    assert not tight_claims([_t], 600, {}), "an unrecorded claim cannot be judged"

    # THE GPU SEMAPHORE, and what is being tested is that the knob CHANGES
    # CONCURRENCY - not that it is a good idea, which only a measurement on a
    # quiet box can say. Two claims in the gpu family, each refusing to print
    # its value while the other holds the lock file: under the mutex they never
    # overlap and both read OK; at capacity 2 they do overlap and one goes
    # DRIFT. Both directions, because a knob wired to nothing would pass the
    # first assertion on its own.
    _gl = os.path.join(tempfile.gettempdir(), f"vc_gpu_{os.getpid()}")
    # TWO DIFFERENT gpu MODULES on purpose. Two claims naming the same module
    # are serialised by that module's own mutex, so a same-module fixture stays
    # OK at any gpu capacity and would have "passed" while proving nothing -
    # which is exactly what it did on the first attempt.
    _gx = (lambda mod:
           f"sh -c 'mkdir {_gl} 2>/dev/null || exit 0; sleep 0.4; rmdir {_gl}; "
           f"echo 5.50' # model/validation/{mod}.py")
    _gfix = [Claim("M", "5.50", _gx("n47_bench"), "g.md", 1, []),
             Claim("M", "5.50", _gx("n46_equiv"), "g.md", 2, [])]
    with contextlib.suppress(OSError):
        os.rmdir(_gl)
    assert {"gpu"} == resources(_gx("n47_bench")) & resources(_gx("n46_equiv")), \
        "the two fixture claims must share gpu AND NOTHING ELSE, or the module " \
        "mutex serialises them and the gpu capacity is never exercised"
    _mutex = {r["key"]: r["status"] for r in verify(_gfix, 5, 4, secs={}, gpu_jobs=1)}
    assert _mutex == {"g.md:1": "OK", "g.md:2": "OK"}, \
        f"the gpu mutex let two gpu claims overlap: {_mutex}"
    with contextlib.suppress(OSError):
        os.rmdir(_gl)
    _sem = {r["key"]: r["status"] for r in verify(_gfix, 5, 4, secs={}, gpu_jobs=2)}
    assert sorted(_sem.values()) == ["DRIFT", "OK"], \
        f"--gpu-jobs 2 did not actually raise concurrency: {_sem} - the knob is " \
        "wired to nothing and the measurement would compare a run with itself"
    with contextlib.suppress(OSError):
        os.rmdir(_gl)
    assert capacity("gpu", 2) == 2 and capacity("dist_eval", 2) == 1, \
        "only gpu is a semaphore; every other resource stands for a file"

    # and the resource derivation itself, in both directions.
    _k = _known_modules()
    assert "dist_eval" in resources("model/validation/dist_eval.py --n26", _k)
    assert "dist_eval" in resources(
        ".venv/bin/python -c \"import dist_eval as D;print(D.x)\"", _k), \
        "a `python -c` import names no path - it must still claim its module"
    assert "dist_eval" in resources(
        "python -c \"json.load(open('model/validation/out/stc2025/n26_results.json'))\"", _k), \
        "a reader of n26_results.json must share dist_eval's chain (OWNER)"
    assert "gpu" in resources("model/validation/n47_bench.py --l1", _k)
    assert not resources("git log -1 --format=%cs 7c2cd35", _k), \
        "a claim that touches nothing must be free to run beside anything"
    assert _known_modules() >= {"dist_eval", "st2_train", "api"}, \
        "module discovery is not reading model/validation - every claim would be free"
    # The census must see the docs. If this hits zero the bare-marker count has
    # stopped counting, and the backlog goes invisible again.
    assert census["V"] > 20, f"census sees {census['V']} bare @V; the docs have hundreds"
    assert sum(census.values()) > len(all_claims), "census is not seeing the docs"

    print("verify_claims.py self-check OK "
          f"({len(found)} tags in this file, {len(outside)} runnable tags outside it, "
          f"{sum(census.values())} bare markers, matcher rejects 4 near-misses)")


if __name__ == "__main__":
    sys.exit(_selfcheck() or 0 if "--selfcheck" in sys.argv else main())
