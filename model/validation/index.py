"""A generated map of every document in this repo, because 1.6 MB of research
with zero links between it is write-only storage.

State on 2026-09-10, which is why this exists: 48 files in `docs/`, 13 of them
`research-*.md` totalling ~700 KB, and **nine** wikilinks across the whole
directory. Nothing pointed into the research; nothing pointed out of it. The
work was saved and unfindable, which for retrieval purposes is not saved.

The index is GENERATED for the same reason every number here carries a sigil: a
hand-written index is a claim about the directory that nothing re-checks, and it
is wrong the first time a document lands without someone remembering it. This
walks the directory and reports what is actually there. Re-run it and it is
current. That is the entire design.

  title    the H1
  date     the first ISO date in the file's own header, else file mtime (marked)
  lead     the first prose line, which in this repo is reliably the thesis
  sections the H2s, but only for files over SECTION_THRESHOLD - a big document
           is exactly the one you cannot find your way around from a title

Memory notes carry frontmatter, so they are grouped by their declared `type`
and use their own `description` (the memory-note store is not published in this
copy, so that section is empty here). Its `MEMORY.md` is NOT regenerated
here: it is the auto-memory system's own context-loading index and it owns that
file. This one covers both stores.

    .venv/bin/python model/validation/index.py           # writes docs/INDEX.md
    .venv/bin/python model/validation/index.py --check   # exit 1 if stale (for hooks)
    .venv/bin/python model/validation/index.py --demo    # self-check
"""

import re
import sys
import datetime
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
OUT = DOCS / "INDEX.md"
SECTION_THRESHOLD = 30_000  # bytes; above this a title alone will not find it
LEAD_CHARS = 130

# Grouped by first folder under docs/, in STYLE (not published)'s order: top level
# first, archive last, any folder STYLE.md does not name just before archive.
# The folder IS the page type, so a stem rule ("audit" in the name) is gone.
FOLDERS = ["", "reference", "guides", "experiments", "research", "audits", "ledgers",
           "decisions", "plans", "datasets", "product"]


DATE_RE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
# Leading housekeeping clauses. Only the three that carry no thesis - "Scope:"
# and "Status:" often ARE the thesis here, so they stay.
META_RE = re.compile(r"^(as of|date|branch)\b[^.]*\.\s*", re.I)


def strip_md(s):
    """Emphasis and links off, so a lead line reads as prose in the index.

    HTML comments off too: they are directives, not prose. Copied as a lead,
    a doc's `<!-- api:strict -->` opted the whole INDEX into api.py's strict
    lint, which then blocked the push on numbers quoted from memory notes."""
    s = re.sub(r"<!--.*?-->", "", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]*)\*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def parse(text, size=0, mtime=None):
    """One document -> what the index needs. Pure; the caller does the I/O."""
    fm = {}
    m = FM_RE.match(text)
    if m:
        # Flat scan, indentation ignored: memory notes nest `type` under
        # `metadata:`, and skipping indented keys filed all 54 as "unfiled".
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                if v.strip():
                    fm[k.strip()] = v.strip()
        text = text[m.end():]

    lines = text.splitlines()
    title = ""
    body_at = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = strip_md(line[2:])
            body_at = i + 1
            break

    # The file's own stated date beats mtime, which only tracks last touch.
    head = "\n".join(lines[:20])
    dm = DATE_RE.search(head)
    if dm:
        date, dated = dm.group(1), True
    else:
        date = (mtime or datetime.date.today()).isoformat()
        dated = False

    lead = fm.get("description", "")
    if not lead:
        for line in lines[body_at:]:
            t = strip_md(line)
            # "- " and "* " are bullets; "**bold**" is how every
            # research doc opens its thesis. Do not confuse them.
            if not t or line.startswith(("#", "|", "```", ">",
                                          "- ", "* ", "---")):
                continue
            while True:
                t2 = META_RE.sub("", t)
                if t2 == t:
                    break
                t = t2
            if t:
                lead = t
                break
    if len(lead) > LEAD_CHARS:
        lead = lead[:LEAD_CHARS].rsplit(" ", 1)[0] + "…"

    sections = []
    if size > SECTION_THRESHOLD:
        sections = [strip_md(l[3:]) for l in lines if l.startswith("## ")]

    return {"title": title, "date": date, "dated": dated, "lead": lead,
            "sections": sections, "type": fm.get("type", "")}


def read(path):
    st = path.stat()
    mtime = datetime.date.fromtimestamp(st.st_mtime)
    d = parse(path.read_text(errors="replace"), st.st_size, mtime)
    d["path"] = path
    d["kb"] = round(st.st_size / 1024)
    return d


def group_of(rel):
    """First folder of a docs-relative path; "" for a top-level page."""
    return rel.split("/")[0] if "/" in rel else ""


def group_key(g):
    return (g == "archive", FOLDERS.index(g) if g in FOLDERS else len(FOLDERS), g)


def doc_pages(docs=DOCS):
    """Every page under docs/, at any depth, except the memory store and this index."""
    return sorted(p for p in docs.rglob("*.md")
                  if p.relative_to(docs).parts[0] != "memory" and p != docs / "INDEX.md")


def entry(d, rel):
    title = d["title"] or d["path"].stem
    stamp = d["date"] if d["dated"] else d["date"] + "*"
    out = [f"- **[{title}]({rel})** · {stamp} · {d['kb']} KB"]
    if d["lead"]:
        out.append(f"  {d['lead']}")
    if d["sections"]:
        out.append("  <details><summary>" + f"{len(d['sections'])} sections"
                   + "</summary>\n")
        for s in d["sections"]:
            out.append(f"  - {s}")
        out.append("  </details>")
    return "\n".join(out)


def build():
    docs = doc_pages()
    notes = sorted((DOCS / "memory").glob("*.md"))
    notes = [p for p in notes if p.name != "MEMORY.md"]

    buckets = {}
    for p in docs:
        rel = p.relative_to(DOCS).as_posix()               # "x.md", or "reference/x.md"
        buckets.setdefault(group_of(rel), []).append(entry(read(p), rel))

    by_type = {}
    for p in notes:
        d = read(p)
        by_type.setdefault(d["type"] or "unfiled", []).append(
            entry(d, f"memory/{p.name}"))

    L = [
        "# Index",
        "",
        f"Generated by `model/validation/index.py` on "
        f"{datetime.date.today().isoformat()}. Do not hand-edit — re-run it.",
        "",
        f"{len(docs)} documents and {len(notes)} memory notes. A `*` on a date "
        "means the file states none and this is its mtime. Big files list "
        "their sections; that is where the research actually is.",
        "",
    ]
    for g in sorted(buckets, key=group_key):
        L += [f"## {g + '/' if g else 'Top level'} ({len(buckets[g])})", ""]
        L += ["\n".join(buckets[g]), ""]

    if by_type:
        L += ["## Memory notes", "",
              "One fact per file, linked with `[[wikilinks]]`. "
              "`memory/MEMORY.md` is the loader's own index.", ""]
        for t in sorted(by_type):
            L += [f"### {t} ({len(by_type[t])})", "",
                  "\n".join(by_type[t]), ""]

    return "\n".join(L).rstrip() + "\n", len(docs), len(notes)


def demo():
    """Feeds literal markdown to the real parser. Asserts on its output, never
    on a second copy of the logic — a check that recomputes its subject is two
    implementations agreeing with each other (lint_truth check 2)."""
    d = parse("# Title here\n\n**As of 2026-09-10.** The `thesis` line.\n"
              "\n## One\n\n## Two\n", size=SECTION_THRESHOLD + 1)
    assert d["title"] == "Title here", d
    assert d["date"] == "2026-09-10" and d["dated"], d
    assert d["lead"] == "The thesis line.", d  # meta clause stripped
    assert d["sections"] == ["One", "Two"], d

    # Under threshold: sections suppressed, because a short file needs no map.
    assert parse("# T\n\n## One\n", size=10)["sections"] == []

    # Frontmatter description wins over the body's first line.
    d = parse("---\nname: x\nmetadata:\n  type: project\n---\n# T\n\nb\n")
    assert d["type"] == "project" and d["title"] == "T", d
    d = parse("---\ndescription: the summary\ntype: user\n---\n# T\n\nbody\n")
    assert d["lead"] == "the summary", d

    # No stated date -> mtime, and it must be flagged as such.
    # A line that is nothing BUT metadata falls through to the next one.
    d = parse("# T\n\n**Date:** 2026-01-01.\n\nthe real lead\n")
    assert d["lead"] == "the real lead", d
    # But "Scope:" carries meaning here and must survive.
    d2 = parse("# T\n\n**Scope:** strictly the model.\n")
    assert d2["lead"] == "Scope: strictly the model.", d2

    d = parse("# T\n\nbody\n", mtime=datetime.date(2020, 1, 2))
    assert d["date"] == "2020-01-02" and not d["dated"], d

    # A long lead is cut on a word boundary, not mid-word.
    d = parse("# T\n\n" + "word " * 100)
    assert d["lead"].endswith("…") and len(d["lead"]) <= LEAD_CHARS + 1, d

    assert strip_md("**a** `b` [c](u) [[d]]") == "a b c d"
    # A directive is not a lead: copied into the index it opted INDEX.md into
    # api.py's strict lint.
    d = parse("# T\n\n<!-- api:strict -->\n\nthe real lead\n")
    assert d["lead"] == "the real lead", d
    assert group_of("licence-audit.md") == "" and group_of("audits/x/y.md") == "audits"
    assert sorted(["archive", "zz", "", "research", "reference"], key=group_key) == \
        ["", "reference", "research", "zz", "archive"]
    # A page in a subfolder is indexed; memory/ and INDEX.md are not. The
    # pre-2026-09-28 glob("*.md") saw only the top level and would pass silently.
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for rel in ("top.md", "INDEX.md", "audits/a.md", "memory/m.md"):
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            (t / rel).write_text("# T\n")
        assert [p.relative_to(t).as_posix() for p in doc_pages(t)] == ["audits/a.md", "top.md"]
    print("index.py self-check: ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        text, nd, nn = build()
        if "--check" in sys.argv:
            cur = OUT.read_text() if OUT.exists() else ""
            # The date line changes daily; compare everything else.
            norm = lambda s: "\n".join(l for l in s.splitlines()
                                       if not l.startswith("Generated by"))
            if norm(cur) != norm(text):
                print("docs/INDEX.md is stale — run model/validation/index.py")
                sys.exit(1)
            print("docs/INDEX.md is current")
        else:
            OUT.write_text(text)
            print(f"wrote {OUT.relative_to(OUT.parent.parent)} "
                  f"({nd} docs, {nn} notes)")
