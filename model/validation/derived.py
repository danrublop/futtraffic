"""Execute the arithmetic behind a derived number instead of trusting its label.

WHY THIS EXISTS
    `verify_claims.py` re-runs every @M and says @D is "not machine-checkable by
    construction - a labelling discipline". But a @D is required to carry "the
    literal arithmetic", so it IS checkable. It was simply never run.

    And the error that prompted this was not a @D at all. On 2026-09-12 both
    model-map (not published) and n8-n11-downstream (not published) asserted, untagged:

        0.4731 ≈ 0.478 × 1.188   "to within a percent"

    0.478 × 1.188 is 0.5679 - twenty percent off, not one - and it sat on a
    shipped surface in two documents because no instrument in the repo looks at
    a derivation that forgot its sigil. A reviewer found it by doing the
    multiplication. So this does two things:

      1. every @D is evaluated and compared with its value at the PRINTED
         precision - "0.2075" must land within 0.00005, "46.6%" within 0.05
         points - and a mismatch fails;
      2. every UNTAGGED approximate identity in a document - a number, `≈`, an
         arithmetic expression - is evaluated too, and fails if it misses by
         more than APPROX_REL. One that holds is still reported: it is a
         derivation with no sigil, which rule 1 says does not exist.

WHAT IT REFUSES TO DO
    Pass what it cannot evaluate. A @D whose source is prose, or cut off, or
    whose value is a comparison rather than a number, is reported UNEVALUABLE
    and counted - never OK. An evaluator that quietly skips what it cannot parse
    is a check that cannot fail, and R1 found exactly that in model/engine/quality.py
    on the same day this was written.

    AND THE COUNT IS A RATCHET. Counted-but-passing made UNEVALUABLE the one
    status a new derived tag could take and still go green, so a derivation written as
    prose passed the gate exactly like arithmetic that holds. UNEVALUABLE_BASELINE
    is the stock on 2026-09-29; a rise fails, a fall is lowered by hand here.

    It never calls eval(). Expressions are parsed with `ast` and walked by hand,
    admitting only numeric literals, + - * / **, unary +/-, parentheses and the
    functions in FUNCS. Anything else - a name, an attribute, a call to
    something not on the list - is rejected, so a document cannot execute code
    by containing it.

HOW AN EXPRESSION IS FOUND
    @D sources are written inline in prose and wrap across lines:
    `(@D 2.4% :: 0.52 /\\n21.99) and 0.60 of ...` #example. Source is joined with the
    next line, then the LONGEST leading prefix that parses as an allowed
    expression containing at least one operation is taken. A bare number is
    never accepted as a derivation: if `0.52` is the only thing that parses,
    the expression was truncated, and that is UNEVALUABLE, not a mismatch.

    .venv/bin/python model/validation/derived.py          # scan the repo; exit 1 on any failure
    .venv/bin/python model/validation/derived.py --demo   # self-check
"""
import ast
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "validation"))

import verify_claims as V  # noqa: E402 - git_env, reused and never copied

FUNCS = {"sqrt": math.sqrt, "abs": abs, "min": min, "max": max,
         "log": math.log, "exp": math.exp, "round": round}
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
       ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
       ast.Pow: lambda a, b: a ** b}
UNARY = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a}

# `≈` is approximate by definition, so it is held to a relative band rather than
# to its printed digits. Declared, and deliberately loose: it exists to catch a
# claim that is WRONG, like the 20% one above, not to litigate rounding.
APPROX_REL = 0.05

TEXT = {".md", ".py", ".txt", ".rst", ".mts", ".ts", ".tsx"}
TAG_D = re.compile(r"@(?:D|DER)\s+([-+.\d][^:]*?)\s*::\s*(.*)$")
APPROX = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*≈\s*(.+)$")
SIGIL = re.compile(r"@(?:M|C|D|V|RAN|REC|DER)\b")
VALUE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(%|x)?")


def _normalise(s: str) -> str:
    # Cut at `#`: ast.parse reads it as a comment and would otherwise display
    # the prose after it as if it were part of the arithmetic.
    s = s.split("#", 1)[0]
    return (s.replace("×", "*").replace("÷", "/").replace("−", "-")
             .replace("^", "**").replace("`", " "))


def _walk(node):
    """Evaluate an allowed AST node, or raise ValueError. No eval() anywhere."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_walk(node.left), _walk(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_walk(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in FUNCS and not node.keywords:
        return FUNCS[node.func.id](*[_walk(a) for a in node.args])
    raise ValueError(f"not allowed: {ast.dump(node)[:60]}")


def _has_operation(node) -> bool:
    return any(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call)) for n in ast.walk(node))


def longest_expression(src: str):
    """(text, value) for the longest evaluable leading prefix, or (None, None).

    A prefix that is only a number is refused: it means the arithmetic was cut
    off, and treating `0.52` as the derivation of `2.4%` would report a false
    mismatch instead of the truth, which is that nothing was evaluated.
    """
    s = _normalise(src)
    for end in range(len(s), 0, -1):
        cand = s[:end].strip().rstrip(",")
        if not cand:
            continue
        try:
            tree = ast.parse(cand, mode="eval").body
        except (SyntaxError, ValueError):
            continue
        if not _has_operation(tree):
            continue
        try:
            value = _walk(tree)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        return cand, float(value)
    return None, None


def parse_value(text: str):
    """(number, unit, decimals) for '0.2075' / '+46.6%' / '1.23x', else None."""
    m = VALUE.fullmatch(text.strip())
    if not m:
        return None
    num, unit = m.group(1), m.group(2) or ""
    decimals = len(num.split(".")[1]) if "." in num else 0
    return float(num), unit, decimals


def holds(result: float, number: float, unit: str, decimals: int) -> bool:
    """Does `result` print as `number` at the precision `number` was written?"""
    scale = 100.0 if unit == "%" else 1.0
    return abs(result * scale - number) <= 0.5 * 10 ** (-decimals) + 1e-9


def check_d(value_text: str, source: str) -> dict:
    """One @D. status is OK, MISMATCH or UNEVALUABLE - never silently OK."""
    parsed = parse_value(value_text)
    if parsed is None:
        return {"status": "UNEVALUABLE", "why": f"value {value_text!r} is not a number"}
    expr, result = longest_expression(source)
    if expr is None:
        return {"status": "UNEVALUABLE", "why": "no evaluable arithmetic in the source"}
    number, unit, decimals = parsed
    ok = holds(result, number, unit, decimals)
    shown = result * (100.0 if unit == "%" else 1.0)
    return {"status": "OK" if ok else "MISMATCH", "expr": expr,
            "evaluates_to": round(shown, max(decimals + 2, 4)),
            "claimed": value_text.strip()}


def check_approx(lhs_text: str, rhs_source: str) -> dict | None:
    """One untagged `a ≈ expr`. None if the right-hand side is not arithmetic."""
    expr, result = longest_expression(rhs_source)
    if expr is None:
        return None
    lhs = float(lhs_text)
    rel = abs(result - lhs) / abs(lhs) if lhs else abs(result)
    return {"status": "FALSE" if rel > APPROX_REL else "UNTAGGED",
            "expr": expr, "evaluates_to": round(result, 4), "claimed": lhs_text,
            "off_by": f"{100 * rel:.1f}%"}


def _files():
    def ls(*a):
        # env=git_env(): see verify_claims.git_env - GIT_DIR beats cwd=.
        return subprocess.run(["git", "ls-files", *a], cwd=ROOT, env=V.git_env(),
                              capture_output=True, text=True).stdout.splitlines()
    for rel in ls() + ls("--others", "--exclude-standard"):
        p = ROOT / rel
        if p.suffix in TEXT and p.is_file():
            yield rel, p


def scan(files=None) -> list[dict]:
    rows = []
    for rel, path in (files if files is not None else _files()):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            # `#example` is verify_claims' marker for a specimen - a tag in a
            # document ABOUT tags, or a planted case in a self-check - and the
            # same line is exempt here for the same reason.
            if "#example" in line:
                continue
            m = TAG_D.search(line)
            if m:
                r = check_d(m.group(1), m.group(2) + " " + nxt)
                rows.append({"kind": "@D", "where": f"{rel}:{i + 1}", **r})
                continue
            # Untagged approximate identities - prose only. A line that already
            # carries a sigil is the tagged path's business, not this one's.
            if path.suffix != ".md" or SIGIL.search(line):
                continue
            a = APPROX.search(line)
            if a:
                r = check_approx(a.group(1), a.group(2) + " " + nxt)
                if r:
                    rows.append({"kind": "≈", "where": f"{rel}:{i + 1}", **r})
    return rows


FAILING = {"MISMATCH", "FALSE"}
#: The UNEVALUABLE count `derived.py | tail -1` printed on 2026-09-29. Only goes down.
UNEVALUABLE_BASELINE = 31


def failures(rows, baseline: int = UNEVALUABLE_BASELINE) -> int:
    """Rows that fail outright, plus 1 if the UNEVALUABLE stock rose."""
    n_unev = sum(r["status"] == "UNEVALUABLE" for r in rows)
    rose = n_unev > baseline
    if rose:
        print(f"RATCHET     {n_unev} UNEVALUABLE derived tags against a baseline of {baseline}: a new "
              f"derivation that cannot be evaluated is a claim nothing checks. Write it as "
              f"arithmetic, or tag it as cited.")
    return sum(r["status"] in FAILING for r in rows) + rose


def report(rows) -> int:
    for r in rows:
        if r["status"] == "OK":
            continue
        detail = (f"{r.get('claimed', '')!s:>10}  ←  {r.get('expr', '')} = {r.get('evaluates_to', '')}"
                  if "expr" in r else r.get("why", ""))
        extra = f"  (off by {r['off_by']})" if "off_by" in r else ""
        print(f"{r['status']:11s} {r['kind']:2s} {r['where']}\n            {detail}{extra}")
    by = {}
    for r in rows:
        by[(r["kind"], r["status"])] = by.get((r["kind"], r["status"]), 0) + 1
    print("\n" + ", ".join(f"{n} {k} {s}" for (k, s), n in sorted(by.items())))
    return failures(rows)


def demo():
    """Every status is produced, and each failing one is shown to fail."""
    # 1. an identity that holds, at printed precision
    assert check_d("0.2075", "1 - 217954/275034")["status"] == "OK"
    assert check_d("+46.6%", "(74826 - 51057) / 51057 |")["status"] == "OK"
    assert check_d("1.23x", "0.478 / 0.390, against the")["status"] == "OK"

    # 2. THE CASE THIS EXISTS FOR, written as the @D it should have been
    bad = check_d("0.4731", "0.478 × 1.188")
    assert bad["status"] == "MISMATCH", bad

    # 3. precision is the printed precision, both ways
    assert check_d("0.2075", "0.20754")["status"] == "UNEVALUABLE"   # bare number
    assert check_d("0.2075", "0.2075 * 1")["status"] == "OK"
    assert check_d("0.2075", "0.2081 * 1")["status"] == "MISMATCH"

    # 4. a wrapped source joins the next line rather than failing on "0.52 /"
    assert check_d("2.4%", "0.52 / 21.99) and 0.60 of")["status"] == "OK"
    truncated = check_d("2.4%", "0.52 /")
    assert truncated["status"] == "UNEVALUABLE", truncated   # not a false MISMATCH

    # 5. prose trailing the arithmetic is not swallowed into it
    assert check_d("0.3%", "0.07 / 22.00 - where the polygon")["status"] == "OK"

    # 6. what cannot be evaluated is never OK
    assert check_d("5.27 > 4.19", "mask_bias_pp vs floor")["status"] == "UNEVALUABLE"
    assert check_d("0.5", "the decomposition suggests")["status"] == "UNEVALUABLE"

    # 7. no code runs: a hostile source is rejected, not executed
    assert longest_expression("__import__('os').system('true')") == (None, None)
    assert check_d("1", "().__class__ + 1")["status"] == "UNEVALUABLE"

    # 8. untagged ≈ : the real 2026-09-12 sentence FAILS, an honest one does not
    n10 = check_approx("0.4731", "0.478 × 1.188` to within a percent")
    assert n10["status"] == "FALSE" and n10["off_by"] == "20.0%", n10
    assert check_approx("0.5", "1 / 2 exactly")["status"] == "UNTAGGED"

    # 9. the scanner end to end, on a planted file
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "x.md"
        # Assembled rather than written out: spelled contiguously, these lines
        # are claims in derived.py's own source and the repo scan would report
        # the planted bug as a real one - which it did, on the first run.
        d, approx = "@" + "D", "≈"
        f.write_text(f"good {d} 0.42 :: 0.6 * 0.7\n"
                     f"bad {d} 0.99 :: 0.6 * 0.7\n"
                     f"prose 0.4731 {approx} 0.478 × 1.188 here\n")
        rows = scan([("x.md", f)])
    got = sorted((r["kind"], r["status"]) for r in rows)
    assert got == [("@D", "MISMATCH"), ("@D", "OK"), ("≈", "FALSE")], got

    # 10. the UNEVALUABLE ratchet: at the baseline passes, one more fails
    import contextlib
    import io
    unev = [{"status": "UNEVALUABLE"}] * 3
    with contextlib.redirect_stdout(io.StringIO()):
        assert failures(unev, baseline=3) == 0
        assert failures(unev + [{"status": "UNEVALUABLE"}], baseline=3) == 1, \
            "a new UNEVALUABLE derived tag passed the gate - counted, never failed"
        assert failures([{"status": "OK"}], baseline=0) == 0
    print("derived.py ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(1 if report(scan()) else 0)
