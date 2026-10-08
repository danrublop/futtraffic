"""Every reference to a docs/ path in the repo: checked, and rewritten when docs move.

Two kinds of reference, and the same two rules serve --check, --rewrite and the census:

  bare   a repo-root path starting `docs/` anywhere in a text file: prose, code
         comments, docstrings, sigil tag sources and commands, JSON keys (the
         bare-marker baseline is keyed by path), hooks. Not preceded by a path
         character, so `../docs/x` and `app/frontend/docs/` are not matches.
         Followed by `*`, `<`, `{` or `[` it is a pattern (`docs/n68-*`), not a path.
  link   a relative markdown link or image `[t](target#anchor)` in a .md file,
         outside fenced code. Resolved against the linking file's directory.

A bare path may carry an anchor (`docs/<page>.md#<heading>`); it is checked like a
link's. A dead bare path that ends a line in `-` and continues on the next
(a path wrapped across comment lines) is reported with the joined path.

Allowlist: model/validation/doclinks_allow.txt, `file<TAB>target<TAB>reason`, for
references that are meant not to resolve (a specimen in a self-check, another
project's docs/ tree). An entry that matches nothing is itself reported.

Not covered, and listed by hand in the move plan: paths BUILT in code
(`ROOT / "docs" / ...`, `glob("docs/*.md")`), Obsidian `[[wikilinks]]` (they
resolve by basename, which a move keeps), and a bare basename in prose.

    .venv/bin/python model/validation/doclinks.py --check              # exit 1 on a dead reference
    .venv/bin/python model/validation/doclinks.py --rewrite MAP.tsv --dry-run
    .venv/bin/python model/validation/doclinks.py --rewrite MAP.tsv    # rewrite, then git mv
    .venv/bin/python model/validation/doclinks.py --split PAGE PART... [--dry-run]
    .venv/bin/python model/validation/doclinks.py --selfcheck

MAP.tsv: `old<TAB>new[<TAB>anything]`, repo-relative; `#` lines, a header row
starting `old`, and rows with old == new are ignored. A row `old.md#a<TAB>new.md#b`
maps one anchor and moves no file; it wins over the row for its page.

--split PAGE PART...: PAGE was split and some of its headings now live in the
PARTs, heading text kept. Every reference to PAGE#anchor whose anchor is no longer
in PAGE but is in exactly one PART is re-pointed there; nothing moves.
"""
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "validation"))
import verify_claims as V  # noqa: E402 - git_env and TEXT, reused and never copied

TEXT = V.TEXT | {".mts", ".js", ".mjs", ".html", ".css", ".tsv", ""}
BARE = re.compile(r"(?<![\w./-])(docs/[\w./-]*)(#[\w-]+)?")
LINK = re.compile(r"(!?\[)([^\]\n]*)(\]\()(<[^>\n]+>|[^)\s]+)((?:\s+\"[^\"\n]*\")?\))")
FENCE = re.compile(r"^\s*(```|~~~)")
HEAD = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
HTML_ID = re.compile(r"""\b(?:id|name)=["']([^"']+)["']""")
ALLOW = ROOT / "model" / "validation" / "doclinks_allow.txt"
WRAP_NEXT = re.compile(r"^[\s#*/]*([\w./-]+)")


def files(root):
    """(rel, path) for every tracked or untracked-not-ignored text file."""
    def ls(*a):
        return subprocess.run(["git", "ls-files", *a], cwd=root, env=V.git_env(),
                              capture_output=True, text=True).stdout.splitlines()
    for rel in sorted(set(ls() + ls("--others", "--exclude-standard"))):
        p = Path(root) / rel
        if p.suffix in TEXT and p.is_file() and not p.is_symlink() and p.stat().st_size < 5_000_000:
            yield rel, p


def bare_path(m, text):
    """The path a BARE match names, or None for a pattern or the bare directory."""
    p = m.group(1).rstrip(".-")
    if m.end(1) < len(text) and text[m.end(1)] in "*<{[":
        return None
    return p if p.rstrip("/") != "docs" else None


def bare_frag(m, p):
    """The anchor after a bare path, only when it follows the path directly."""
    return m.group(2)[1:] if m.group(2) and p == m.group(1) else ""


def unwrapped(line, nxt):
    """The path a line-final `docs/...-` names when joined to the next line's head."""
    m = list(BARE.finditer(line))
    if not m or not line.rstrip().endswith("-") or m[-1].end() != len(line.rstrip()):
        return None
    w = WRAP_NEXT.match(nxt)
    return (m[-1].group(0) + w.group(1)).rstrip(".") if w else None


def split_target(t):
    t = t[1:-1] if t.startswith("<") else t
    path, _, frag = t.partition("#")
    return path, frag


def resolve(src_rel, path):
    """Repo-relative path a relative link names, or None (URL, absolute, outside the repo)."""
    if not path or re.match(r"[a-zA-Z][\w+.-]*:", path) or path.startswith("/"):
        return None
    r = posixpath.normpath(posixpath.join(posixpath.dirname(src_rel), unquote(path)))
    return None if r.startswith("..") else r


def refs(rel, text):
    """Yield (line_no, kind, target_repo_rel, anchor) for every reference in one file."""
    md = rel.endswith(".md")
    fenced = False
    for i, line in enumerate(text.splitlines(), 1):
        if md and FENCE.match(line):
            fenced = not fenced
        # A bare match inside a link target is that link, reported once, as a link.
        spans = [m.span(4) for m in LINK.finditer(line)] if md and not fenced else []
        for m in BARE.finditer(line):
            p = bare_path(m, line)
            if p and not any(a <= m.start() < b for a, b in spans):
                yield i, "bare", p, bare_frag(m, p)
        if md and not fenced:
            for m in LINK.finditer(line):
                path, frag = split_target(m.group(4))
                if not path:
                    if frag:
                        yield i, "anchor", rel, frag
                    continue
                r = resolve(rel, path)
                if r:
                    yield i, "link", r, frag


def slug(h):
    """GitHub's heading anchor."""
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


def anchors(text):
    out, seen, fenced = set(), {}, False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        out.update(HTML_ID.findall(line))
        m = HEAD.match(line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def load_allow(path):
    """{(file, target): reason}. A line without a reason is refused, not ignored."""
    out = {}
    if not Path(path).exists():
        return out
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        assert len(cols) >= 3 and cols[2].strip(), f"{path}:{n}: needs file<TAB>target<TAB>reason"
        out[(cols[0].strip(), cols[1].strip())] = cols[2].strip()
    return out


def tracked(root):
    """Every path git tracks under `root`, plus every directory that holds one.

    A link resolves only if its target SHIPS. `exists()` on disk also accepts a
    page written and never `git add`ed, which is live here and dead in every
    clone - the push carries the link and not the page.
    """
    out = subprocess.run(["git", "ls-files"], cwd=root, env=V.git_env(),
                         capture_output=True, text=True).stdout.splitlines()
    dirs = {posixpath.dirname(f) for f in out}
    for d in list(dirs):
        while d:
            d = posixpath.dirname(d)
            dirs.add(d)
    return set(out) | dirs


def check(root=ROOT, allow=None):
    """[(file, line, message)] for every reference that does not resolve.

    Bare paths anywhere; links only when source or target is under docs/ (the
    frontend's own READMEs are not this tool's business). "Resolves" means the
    target is in the TRACKED set (see tracked()), not merely on this disk."""
    root = Path(root)
    live = tracked(root)
    allow = load_allow(root / ALLOW.relative_to(ROOT) if allow is None else allow)
    used = set()
    cache, bad = {}, []
    for rel, p in files(root):
        if rel == ALLOW.relative_to(ROOT).as_posix():   # it names dead paths by design
            continue
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        for ln, kind, tgt, frag in refs(rel, text):
            if kind in ("link", "anchor") and not (rel.startswith("docs/") or tgt.startswith("docs/")):
                continue
            if (rel, tgt) in allow:
                used.add((rel, tgt))
                continue
            t = root / tgt
            if posixpath.normpath(tgt) not in live:
                w = unwrapped(lines[ln - 1], lines[ln] if ln < len(lines) else "") if kind == "bare" else None
                hint = f" (wrapped across lines: {w} exists)" if w and posixpath.normpath(w) in live else ""
                why = "is not tracked by git" if t.exists() else "does not exist"
                bad.append((rel, ln, f"{kind}: {tgt} {why}{hint}"))
            elif frag and t.suffix == ".md":
                if tgt not in cache:
                    cache[tgt] = anchors(t.read_text(errors="replace"))
                if frag not in cache[tgt]:
                    bad.append((rel, ln, f"{kind}: {tgt}#{frag} - no such heading"))
    for f, tgt in sorted(set(allow) - used):
        bad.append((ALLOW.relative_to(ROOT).as_posix(), 0, f"allow: {f} -> {tgt} matched nothing"))
    return bad


def load_map(tsv):
    """(moves {old: new}, anchors {old#a: new#b}). Anchor rows move no file."""
    mp, am = {}, {}
    for line in Path(tsv).read_text().splitlines():
        cols = line.split("\t")
        if not line.strip() or line.startswith("#") or cols[0] == "old" or len(cols) < 2:
            continue
        old, new = cols[0].strip(), cols[1].strip()
        if old != new:
            if "#" in old:
                assert "#" in new, f"anchor row needs an anchor on both sides: {line!r}"
                am[old] = new
            else:
                mp[old] = new
    news = list(mp.values())
    dup = {n for n in news if news.count(n) > 1}
    assert not dup, f"two files map to one target: {sorted(dup)}"
    return mp, am


def split_map(page, parts, root=ROOT):
    """{page#a: part#a} for every anchor gone from PAGE and present in one PART."""
    root = Path(root)
    left = anchors((root / page).read_text()) if (root / page).exists() else set()
    am = {}
    for part in parts:
        for a in anchors((root / part).read_text()) - left:
            assert f"{page}#{a}" not in am, f"#{a} is in two parts: {am[f'{page}#{a}']} and {part}"
            am[f"{page}#{a}"] = f"{part}#{a}"
    return am


def rewrite_text(rel, text, mp, am=None):
    """(new text, n bare rewritten, n links rewritten). Pure; one pass, so a chain
    a->b, b->c in the map cannot rewrite twice. `am` maps old#anchor to new#anchor
    and wins over `mp` for that one anchor."""
    am = am or {}
    src_new = mp.get(rel, rel)
    n = {"bare": 0, "link": 0}

    def bare(m, s):
        p = bare_path(m, s)
        frag = bare_frag(m, p) if p else ""
        if p and frag and f"{p}#{frag}" in am:
            n["bare"] += 1
            return am[f"{p}#{frag}"] + m.group(0)[len(p) + 1 + len(frag):]
        if p and p in mp:
            n["bare"] += 1
            return mp[p] + m.group(0)[len(p):]
        return m.group(0)

    def link(m):
        text_part = BARE.sub(lambda b: bare(b, m.group(2)), m.group(2))
        raw = m.group(4)
        path, frag = split_target(raw)
        r = resolve(rel, path) if path else rel
        target = raw
        if r and frag and f"{r}#{frag}" in am:
            r, frag = am[f"{r}#{frag}"].split("#", 1)   # an anchor row names the FINAL path
            new = "" if r == src_new else posixpath.relpath(r, posixpath.dirname(src_new) or ".")
            new = new.replace(" ", "%20") if "%20" in path else new
            n["link"] += 1
            target = new + "#" + frag
            target = f"<{target}>" if raw.startswith("<") else target
        elif r and path and (r in mp or src_new != rel):
            new = posixpath.relpath(mp.get(r, r), posixpath.dirname(src_new) or ".")
            if "%20" in path:
                new = new.replace(" ", "%20")
            if posixpath.normpath(unquote(path)) != new:
                n["link"] += 1
                target = new + (("#" + frag) if frag else "")
                target = f"<{target}>" if raw.startswith("<") else target
        return m.group(1) + text_part + m.group(3) + target + m.group(5)

    md = rel.endswith(".md")
    out, fenced = [], False
    for line in text.split("\n"):
        if md and FENCE.match(line):
            fenced = not fenced
        if md and not fenced:
            # Links first, then bare paths outside the links already handled.
            parts, last = [], 0
            for m in LINK.finditer(line):
                parts.append(BARE.sub(lambda b, s=line[last:m.start()]: bare(b, s), line[last:m.start()]))
                parts.append(link(m))
                last = m.end()
            parts.append(BARE.sub(lambda b, s=line[last:]: bare(b, s), line[last:]))
            line = "".join(parts)
        else:
            line = BARE.sub(lambda b, s=line: bare(b, s), line)
        out.append(line)
    return "\n".join(out), n["bare"], n["link"]


def rewrite(tsv, root=ROOT, dry=True, am=None):
    """Rewrite every reference, then git mv. `tsv` is a map file or (moves, anchors)."""
    root = Path(root)
    mp, am0 = load_map(tsv) if not isinstance(tsv, tuple) else tsv
    am = {**am0, **(am or {})}
    missing = [o for o in mp if not (root / o).exists()]
    taken = [n for n in mp.values() if (root / n).exists() and n not in mp]
    assert not missing, f"map names files that do not exist: {missing}"
    assert not taken, f"targets already exist: {taken}"
    tot = {"files": 0, "bare": 0, "link": 0}
    per = []
    for rel, p in files(root):
        text = p.read_text(errors="replace")
        new, b, l = rewrite_text(rel, text, mp, am)
        if new != text:
            tot["files"] += 1
            tot["bare"] += b
            tot["link"] += l
            per.append((rel, b, l))
            if not dry:
                p.write_text(new)
    if not dry:
        for old, new in mp.items():
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["git", "mv", old, new], cwd=root, env=V.git_env(), capture_output=True)
            if r.returncode:   # untracked file: a plain move
                shutil.move(str(root / old), str(root / new))
    return tot, per, len(mp)


def selfcheck():
    """A tiny repo, one move, and the two outcomes that matter: rewritten, the tree
    is clean; moved WITHOUT the rewrite, check() says so."""
    d = "doc" + "s"   # spelt apart so this file's own scan finds no path in it
    tree = {
        f"{d}/a.md": f"# A\n\n## Some heading\n\n[b](b.md#b-head) [code](../code.py) [self](#some-heading)\n"
                     f"see `{d}/b.md` and {d}/n68-*\n```\n[not a link](nowhere.md)\n```\n",
        f"{d}/b.md": "# B\n\n## B head\n\n[a](a.md#some-heading)\n",
        "README.md": f"[a]({d}/a.md) and <img src=\"{d}/b.md\">\n",
        "code.py": f"# {d}/a.md#some-heading is the reference\nPATH = '{d}/b.md'\n",
    }
    movemap = f"old\tnew\n{d}/a.md\t{d}/x/a.md\n{d}/b.md\t{d}/y/b.md\n"

    def build(t):
        for k, v in tree.items():
            (Path(t) / k).parent.mkdir(parents=True, exist_ok=True)
            (Path(t) / k).write_text(v)
        subprocess.run(["git", "init", "-q"], cwd=t, env=V.git_env(), check=True)
        subprocess.run(["git", "add", "-A"], cwd=t, env=V.git_env(), check=True)

    tsv = Path(tempfile.mkdtemp()) / "map.tsv"   # outside the tree: its targets do not exist yet
    tsv.write_text(movemap)
    with tempfile.TemporaryDirectory() as t:
        build(t)
        assert check(t) == [], check(t)
        # ON DISK BUT NEVER `git add`ed: live here, dead in every clone.
        (Path(t) / f"{d}/untracked.md").write_text("# U\n")
        (Path(t) / "README.md").write_text(f"[u]({d}/untracked.md)\n")
        assert [m for *_, m in check(t)] == [f"link: {d}/untracked.md is not tracked by git"], \
            f"a link to an untracked page passed: {check(t)}"
        (Path(t) / f"{d}/untracked.md").unlink()
        (Path(t) / "README.md").write_text(f"[a]({d}/a.md#nope)\n")
        assert [m for *_, m in check(t)] == [f"link: {d}/a.md#nope - no such heading"], check(t)
        (Path(t) / "README.md").write_text(tree["README.md"])
        tot, _, n = rewrite(tsv, t, dry=True)
        assert n == 2 and tot["files"] == 4 and tot["bare"] == 4 and tot["link"] == 4, tot
        assert (Path(t) / f"{d}/a.md").exists(), "a dry run moved a file"
        rewrite(tsv, t, dry=False)
        assert check(t) == [], check(t)
        a = (Path(t) / f"{d}/x/a.md").read_text()
        assert "(../y/b.md#b-head)" in a and "(../../code.py)" in a and "(#some-heading)" in a, a
        assert f"`{d}/y/b.md`" in a and f"{d}/n68-*" in a and "(nowhere.md)" in a, a
        assert f"'{d}/y/b.md'" in (Path(t) / "code.py").read_text()
    with tempfile.TemporaryDirectory() as t:   # moved, not rewritten: must fail
        build(t)
        for old, new in load_map(tsv)[0].items():
            (Path(t) / new).parent.mkdir(parents=True, exist_ok=True)
            os.rename(Path(t) / old, Path(t) / new)
        subprocess.run(["git", "add", "-A"], cwd=t, env=V.git_env(), check=True)
        bad = check(t)
        assert len(bad) >= 4 and {f for f, *_ in bad} >= {"README.md", "code.py"}, bad
    shutil.rmtree(tsv.parent)

    # A page split in two, heading text kept: every reference to a heading that
    # left it is dead until the anchors are mapped, by --split or by anchor rows.
    split_tree = {
        f"{d}/p.md": "# P\n\n## Keep\n\n## Moved away\n\n[self](#moved-away)\n",
        f"{d}/q.md": "[p](p.md#moved-away) [k](p.md#keep)\n",
        "code.py": f"# {d}/p.md#moved-away and {d}/p.md#keep\n",
    }
    after = {f"{d}/p.md": "# P\n\n## Keep\n\n[self](#moved-away)\n",
             f"{d}/p-evidence.md": "# P evidence\n\n## Moved away\n"}

    def build2(t):
        tree = dict(split_tree)
        build_tree(t, tree)
        for k, v in after.items():
            (Path(t) / k).write_text(v)
        subprocess.run(["git", "add", "-A"], cwd=t, env=V.git_env(), check=True)

    def build_tree(t, tr):
        for k, v in tr.items():
            (Path(t) / k).parent.mkdir(parents=True, exist_ok=True)
            (Path(t) / k).write_text(v)
        subprocess.run(["git", "init", "-q"], cwd=t, env=V.git_env(), check=True)

    with tempfile.TemporaryDirectory() as t:
        build2(t)
        assert len(check(t)) == 3, check(t)   # q's link, p's self-link, code's bare path
        am = split_map(f"{d}/p.md", [f"{d}/p-evidence.md"], t)
        assert am[f"{d}/p.md#moved-away"] == f"{d}/p-evidence.md#moved-away" and f"{d}/p.md#keep" not in am, am
        rewrite(({}, am), t, dry=False)
        assert check(t) == [], check(t)
        assert "(p-evidence.md#moved-away) [k](p.md#keep)" in (Path(t) / f"{d}/q.md").read_text()
        assert "(p-evidence.md#moved-away)" in (Path(t) / f"{d}/p.md").read_text()
        assert f"{d}/p-evidence.md#moved-away and {d}/p.md#keep" in (Path(t) / "code.py").read_text()
    rows = (f"{d}/p.md\t{d}/r/p.md\n{d}/p-evidence.md\t{d}/r/p-evidence.md\n"
            f"{d}/p.md#moved-away\t{d}/r/p-evidence.md#moved-away\n")
    tsv = Path(tempfile.mkdtemp()) / "map.tsv"
    for with_anchor_row in (False, True):
        tsv.write_text(rows if with_anchor_row else rows.rsplit("\n", 2)[0] + "\n")
        with tempfile.TemporaryDirectory() as t:
            build2(t)
            rewrite(tsv, t, dry=False)
            assert (check(t) == []) == with_anchor_row, check(t)
    shutil.rmtree(tsv.parent)

    # A path wrapped across comment lines is named; an allowlisted one is not
    # reported, and an allowlist entry that matches nothing is.
    with tempfile.TemporaryDirectory() as t:
        build_tree(t, {f"{d}/p-evidence.md": "# E\n", "w.py": f"# see ({d}/p-\n#  evidence.md) and {d}/gone.md\n"})
        subprocess.run(["git", "add", "-A"], cwd=t, env=V.git_env(), check=True)
        msgs = [m for *_, m in check(t)]
        assert any(f"wrapped across lines: {d}/p-evidence.md exists" in m for m in msgs), msgs
        allow = Path(tempfile.mkdtemp()) / "allow.txt"
        allow.write_text(f"w.py\t{d}/gone.md\ta specimen\nw.py\t{d}/p\twrapped, on purpose\n"
                         f"x.py\t{d}/never.md\tstale\n")
        assert [m for *_, m in check(t, allow)] == [f"allow: x.py -> {d}/never.md matched nothing"], check(t, allow)
        shutil.rmtree(allow.parent)
    assert slug("Evidence: zero-click") == "evidence-zero-click"
    assert slug("`tracks.json` / `tracks.raw.json`") == "tracksjson--tracksrawjson"
    print("doclinks selfcheck OK")


def main(argv):
    if "--selfcheck" in argv:
        selfcheck()
        return 0
    if "--split" in argv:
        i = argv.index("--split")
        page, parts = argv[i + 1], [a for a in argv[i + 2:] if not a.startswith("--")]
        am = split_map(page, parts)
        for k, v in sorted(am.items()):
            print(f"{k}\t{v}")
        tot, per, _ = rewrite(({}, am), dry="--dry-run" in argv)
        for rel, b, l in per:
            print(f"{rel}\tbare {b}\tlink {l}")
        print(f"{'WOULD CHANGE' if '--dry-run' in argv else 'CHANGED'}: {len(am)} anchors, "
              f"{tot['files']} files, {tot['bare']} bare paths, {tot['link']} links")
        return 0
    if "--rewrite" in argv:
        tsv = argv[argv.index("--rewrite") + 1]
        dry = "--dry-run" in argv
        tot, per, n = rewrite(tsv, dry=dry)
        for rel, b, l in per:
            print(f"{rel}\tbare {b}\tlink {l}")
        print(f"{'WOULD CHANGE' if dry else 'CHANGED'}: {n} moves, {tot['files']} files, "
              f"{tot['bare']} bare paths, {tot['link']} links")
        return 0
    if "--check" in argv:
        bad = check()
        for f, ln, msg in bad:
            print(f"{f}:{ln}: {msg}")
        print(f"{len(bad)} dead reference(s)")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
