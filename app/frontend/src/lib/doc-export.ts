/** The document, on its own.
 *
 *  There is no download button and no server-side renderer, so PRINT IS THE
 *  EXPORT — everything a coach sends leaves through here. That makes this file
 *  the file format.
 *
 *  It does NOT print the app. The first attempt did: an `@media print` block
 *  that hid the shell and pinned the notepad to the page. It cannot work.
 *  `position: absolute` resolves against the nearest positioned ancestor, and
 *  the notepad sits several workspace panes deep, so the document printed
 *  offset and clipped down the right-hand edge. There is no CSS selector for
 *  "my ancestors", which is what that approach needs.
 *
 *  So the document is rebuilt as a standalone page instead: doctype, a paper
 *  stylesheet, the saved HTML, print. Nothing of the app is in it, which is
 *  also the point — this is the artefact, and it should be possible to hand
 *  someone the same bytes.
 */

import { MARGIN_DEFAULT, docMargin, docOrient } from "@/components/notepad/page-nodes";

/** The `.ft-*` rules, lifted from the running app rather than written twice.
 *
 *  One source of truth: globals.css already defines how a board, a table and a
 *  chart look, and a second hand-kept copy here would drift the first time
 *  somebody adjusted a hairline. Cross-origin sheets throw on `cssRules` and
 *  are skipped — everything of ours is same-origin.
 *
 *  Exported because the gallery builds one preview per card and the scan is
 *  the expensive half — the caller lifts it out of the loop.
 */
export function documentRules(): string {
  if (typeof document === "undefined") return "";
  const out: string[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    let rules: CSSRule[];
    try {
      rules = Array.from(sheet.cssRules ?? []);
    } catch {
      continue; // cross-origin, and never ours
    }
    for (const r of rules) {
      const text = r.cssText;
      // The document's own vocabulary, and nothing else. `.ft-controls` is the
      // editing chrome and is deliberately left behind.
      if (text.includes(".ft-") && !text.includes(".ft-controls")) out.push(text);
    }
  }
  return out.join("\n");
}

/** Paper. The app is dark by default and a dark theme printed is a solid black
 *  page and a dead cartridge, so the tokens are redefined for ink on white
 *  rather than the surfaces reused. The values are the light theme's, which is
 *  the one the product already designed for a bright surface. */
const PAPER = `
/* The tokens are NOT here - see .ft-paper in globals.css, which this document
   carries out with every other .ft- rule and wears on the article itself. Two
   copies of a palette is two palettes, and the one nobody looks at is the one
   that goes wrong. */
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fff; }
body {
  color: var(--ink-soft);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
/* LANDSCAPE, and tight margins.
 *
 *  A scouting report is landscape because its unit is a COLUMN, not a line: a
 *  MARGINS 11/12/12, not 12/12/13. That last millimetre is the sheet's margin
 *  of safety: the editor pins a sheet at 83em and the page held 83.24em, so a
 *  sheet filled to the line had under two pixels of room and two of them tipped
 *  onto a second page. 187 mm of usable height gives 84.1em against an 83em
 *  sheet - a line of slack, which is what a fold needs to be trustworthy.
 *
 *  page is two or three independent stacks side by side — a pitch and its
 *  caption here, a block of prose there, a run of tables down the right — and
 *  portrait gives you one column of any useful width or three unreadably
 *  narrow ones. 12mm rather than 18: their pages run nearly to the paper edge,
 *  because the density IS the format. A report that fits on four pages instead
 *  of eleven is a report somebody reads standing up.
 */
`;

/** The paper the document asked for, and the width that follows from it.
 *
 *  LANDSCAPE IS THE FORMAT and portrait is the exception, for the reason above
 *  the @page rule: a report's unit is a column. What changes with it is not
 *  only the sheet - a landscape page runs the FULL measure, because that is
 *  where the three-column layouts go, while a portrait page of prose keeps a
 *  reading column. That is the pairing the screen shows too (globals.css), so
 *  what a coach lays out is what comes out of the printer. */
const page = (orient: "portrait" | "landscape", m = MARGIN_DEFAULT[orient]) => {
  const w = orient === "portrait" ? 210 : 297;
  // The top inset was 1mm under the side margin in landscape and equal to it in
  // portrait. Keeping it EQUAL now: the difference was never explained anywhere
  // and an author who types 20 has asked for 20, not for 20 and 19.
  return (
    `@page { size: A4 ${orient}; margin: 0; }\n` +
    `.ft-doc { width: ${w}mm; max-width: none; padding: 0 ${m}mm; }\n` +
    `.ft-page { padding: calc(${m}mm + 1.2em) 1.4em 1.2em; margin-bottom: ${m}mm; }\n` +
    `.ft-head { padding-top: ${m}mm; }`
  );
};

/** After the lifted rules, not before. globals.css gives `.ft-doc` a border, a
 *  radius and a shadow so it reads as a sheet inside the app; on paper that is
 *  a box drawn around the page. Same for the scroll containers: `overflow-x:
 *  auto` is a scrollbar on screen and a guillotine in print, which is what was
 *  cutting "Milan Westbrook for Joaq" in half. */
const OVERRIDES = `
/* PAPER FURNITURE. The setup chip is an editing control and never prints; the
   break is the browser's own pagination doing the work, with the on-screen
   dashed rule and its label taken off so the page simply ends. */
.ft-pagesetup { display: none !important; }
/* A SHEET IS A SHEET. On paper it has no box drawn round it and no fixed
   height - the height IS the paper - so all that survives is the rule that the
   next one starts on a new one. A page whose content overflowed its box in the
   editor spills onto a second sheet here, which is the honest outcome: nothing
   is clipped, and it was visible before it was printed. */
.ft-page {
  /* BEFORE, not after. A break-after on every sheet also breaks after the LAST
     one, and the editor keeps a trailing paragraph beyond it - so every
     document printed with a blank final page. Breaking before each sheet has
     no such tail; the first one is excused below. */
  break-before: page;
  page-break-before: always;
  min-height: 0;
  height: auto;
  /* The SAME inset as the editor, so both measure the same line. It was zero
     here and 1.4em there, which made the screen quietly narrower than the page
     it was supposed to be. */
  padding: 1.2em 1.4em;
  margin: 0;
  border: 0;
  border-radius: 0;
  background: none;
  background-image: none;
  box-shadow: none;
}
/* A .ft-pagesetup is allowed between the two, and it has to be spelled out.
   It is hidden above with display:none, which removes it from the PAGE and not
   from the SIBLING CHAIN, so the adjacency below silently stopped matching the
   moment a document carried one. And a document carries one exactly when it
   asked for portrait, which is the case most likely to be a printed report: the
   first sheet lost its page-break-before:avoid, went to page two, and page one
   printed as a masthead over nothing. */
.ft-head + .ft-page,
.ft-head + .ft-pagesetup + .ft-page,
.ft-doc > .ft-page:first-child {
  break-before: auto;
  page-break-before: avoid;
}
/* The top inset is on the SHEET now, not on @page - see the page() rule - so
   the first sheet would pay it twice, once behind the masthead and once under
   it. The masthead is page one's top margin. */
.ft-head + .ft-page,
.ft-head + .ft-pagesetup + .ft-page { padding-top: 1.2em; }
.ft-pagebreak {
  break-after: page;
  page-break-after: always;
  border: 0;
  height: 0;
  margin: 0;
}
.ft-pagebreak::after { content: none; }
/* The running head: wordmark left, two-line title centred, hairline under.
   Same furniture every page of a scouting report carries. */
.ft-head {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin: 0 0 8pt;
  padding-bottom: 3pt;
  border-bottom: 0.5pt solid #c8ccd2;
}
.ft-head__mark {
  color: #1a7a3c;
  font-size: 9pt;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.ft-head__title {
  flex: 1 1 auto;
  text-align: right;
  color: var(--ink);
  font-size: 8pt;
  font-weight: 700;
  line-height: 1.25;
}
.ft-head__sub {
  display: block;
  text-align: right;
  color: var(--slate);
  font-size: 7pt;
  font-weight: 400;
}

.ft-doc {
  max-width: 46rem;
  /* No top margin: every sheet starts its own page, so this only ever ate
     forty pixels off the first one. */
  margin: 0 auto;
  /* none: the sheet is the measure, and .ft-page carries the inset. */
  padding: 0;
  background: none;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  /* THE DENSITY IS THE FORMAT.
   *
   *  A scouting report sets body text at about eight points and its section
   *  headings at ten — small enough that a column of prose, a pitch and a run
   *  of tables all fit across one landscape sheet, which is the only reason
   *  the three-column page works at all. At eleven points the same content is
   *  eleven pages and nobody reads past four. */
  /* THE ONE NUMBER PAPER HAS TO SET. Everything inside the document is sized
     in em in globals.css, against a body that is a fixed fraction of the
     measure - so the editor and the sheet are the same stylesheet, and this
     line is the only thing that differs between them. It has to be here
     because a printed page has no container for cqi to resolve against. */
  /* Kept, and now identical to the editor's - see globals.css. The document
     is 273 mm at 6.3pt in both places, so the line boxes round the same way and
     a sheet that fits on screen fits on paper. */
  font-size: 6.3pt;
  line-height: 1.32;
}
.ft-doc h1, .ft-doc h2, .ft-doc h3, .ft-doc h4 {
  color: #101317;
  font-weight: 700;
  line-height: 1.18;
  letter-spacing: -0.005em;
}
/* A heading that starts a column needs no space above it — and in a layout
   cell it always does, because the cell is where the column begins. */
.ft-doc h1:first-child, .ft-doc h2:first-child, .ft-doc h3:first-child { margin-top: 0; }
.ft-doc p { margin: 0; }

/* Figure titles and the provenance line under them. Theirs are small, bold and
   close to the thing they name; a 1rem title with a 0.55rem gap turns a column
   of four figures into a column of two. */
.ft-figure__title, .ft-chart__title { color: #101317; }
.ft-figure, .ft-figure-wrap, .ft-chart, .ft-chart-wrap,
/* A pitch is a FIGURE on a page, not the page. Uncapped it filled a whole
   landscape sheet — one board, nothing else — and because a board also
   avoids breaking inside, the two pages before it came out blank while the
   layout engine looked for somewhere it would fit. 105mm is about a third of
   the printable width, which is the size a scouting report draws one at. */
/* THE BOARD'S HEIGHT is what sets the page count, and it is not the width.
   The attacking crop stood up is TALLER than it is wide (78 x 85 in its own
   units), so at 105 mm across it is 115 mm down - and two of them in a column,
   which is how the source lays its four pictures out, is 230 mm on a sheet with
   185 mm of printable height. Capping the width caps the height with it. */
/* Same for anything else that can grow without limit. A figure taller than
   the page it must not break inside can never be placed at all. */
/* A CHART IS A FIGURE, and a figure is a column wide, not a page.
 *
 *  Left at 100% a chart drawn on a 640-unit viewBox came out 273mm across and
 *  a hundred tall, so two filled a sheet and eighteen took twenty pages —
 *  every one of them a picture alone on a landscape page with nothing beside
 *  it. That is not a report, it is a slide deck. 130mm is half the printable
 *  width, which puts several on a page and matches the size a scouting report
 *  draws one at.
 *
 *  A chart inside a layout cell is narrower still, because the cell caps it
 *  first — which is the right way round. */
/* !important for the same reason the pitch cap needs it: every chart svg is
   emitted with an inline max-width of 100% so it can never overflow whatever
   box it lands in, and inline beats a stylesheet. This is the one place that
   needs to say something narrower. */
/* A pitch-shaped chart is the same kind of object as a board and gets the same
   cap. Without it a shot map printed a page wide — legible, and the only thing
   on the sheet, which is not what a report does with one.

   !important because every pitch svg carries a max-width of 100% as an INLINE
   style, and inline beats a stylesheet. It is inline so that an svg can never
   overflow whatever box it lands in, which is worth keeping; this is the one
   place that needs to say something narrower.

   And the attribute selector beside the class, because THE EXPORT PRINTS THE
   STORED MARKUP — a frozen block was serialized when the note was last saved,
   so a class added today is not in a note written yesterday until something
   re-saves it. A meet-fitted preserveAspectRatio is what every pitch-shaped
   figure has and no other chart does: bars and lines take the default, and
   momentum is drawn with none so it can fill its width. */

/* TABLES, as a report draws them: no vertical rules and no outer box, a
   hairline under each row, and column labels in tiny grey type. The vertical
   rules only ever existed so the EDITOR could show you where a cell was. */
/* NO FONT-SIZE. It was 7.5pt against a 6.3pt body - tables nineteen per cent
   larger on paper than on screen - and that one line is most of why a section
   that fitted in the editor spilled two lines onto a second sheet. Everything
   sized lives in globals.css now, in em, and paper sets the body and nothing
   else. */
.ft-figure__table, .ft-chart__table, .ft-grid {
  border-collapse: collapse;
  width: 100%;
}
.ft-figure__table th, .ft-figure__table td,
.ft-chart__table th, .ft-chart__table td,
.ft-grid th, .ft-grid td {
  border: 0;
  border-bottom: 0.5pt solid #dfe3e8;
  /* padding comes from globals.css, in em against this body. */
  vertical-align: top;
}
.ft-figure__table thead th, .ft-chart__table thead th, .ft-grid th {
  background: none;
  color: var(--slate);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 0.5pt solid #b9bfc7;
}
/* The row label column, tinted — the one piece of fill their tables use.
   The :not(.ft-grid--layout) is load-bearing and cost a page of a long test
   report to find: a layout row's first CELL is a td too, so this tinted the
   whole left-hand column of every side-by-side page. The reset below could not
   win it either - ".ft-grid tbody tr > td:first-child" is two classes deep and
   ".ft-grid--layout > tbody > tr > td" is one. */
.ft-figure__table tbody tr > :first-child,
.ft-grid:not(.ft-grid--layout) tbody tr > td:first-child { background: #f4f5f7; }
.ft-grid--layout, .ft-grid--layout > tbody > tr > td,
.ft-grid--layout > tbody > tr > th { background: none; border: 0; }
/* Nothing scrolls on paper, so nothing may rely on scrolling to be readable. */
.ft-figure__scroll, .ft-chart__scroll { overflow: visible; }
.ft-figure__table, .ft-chart__table { min-width: 0; table-layout: auto; }
.ft-figure__table td, .ft-figure__table th,
.ft-chart__table td, .ft-chart__table th { white-space: normal; }
/* Figures keep their sticky header off paper, where it just paints twice. */
.ft-figure__table thead th { position: static; }
@media print {
  /* The running head: wordmark left, two-line title centred, hairline under.
   Same furniture every page of a scouting report carries. */
.ft-head {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin: 0 0 8pt;
  padding-bottom: 3pt;
  border-bottom: 0.5pt solid #c8ccd2;
}
.ft-head__mark {
  color: #1a7a3c;
  font-size: 9pt;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.ft-head__title {
  flex: 1 1 auto;
  text-align: right;
  color: var(--ink);
  font-size: 8pt;
  font-weight: 700;
  line-height: 1.25;
}
.ft-head__sub {
  display: block;
  text-align: right;
  color: var(--slate);
  font-size: 7pt;
  font-weight: 400;
}

.ft-doc { margin: 0; padding: 0; max-width: none; }
  /* Half a pitch on one sheet and half on the next is a figure nobody can
     read, and a heading at the foot of a page has nothing under it. */
  /* Figures keep together. A ROW OF COLUMNS does not: it is a page's worth of
     content by construction, and asking the engine not to break inside one is
     asking it to leave a page blank and try again lower down — which is what
     it did, twice, before this line was removed. Its cells still hold figures
     that individually avoid breaking, which is the granularity that matters. */
  .ft-pitch-wrap, .ft-pitch, .ft-figure, .ft-figure-wrap, .ft-chart,
  .ft-chart-wrap, .ft-stat {
    break-inside: avoid;
    page-break-inside: avoid;
  }
  /* A table is allowed to break across pages — a squad of thirty will not fit
     on one — but a ROW never splits down the middle, and the header repeats. */
  .ft-grid-wrap { overflow: visible; }
  /* :not(.ft-grid--layout) for the reason stated three comments up, which this
     line was quietly undoing: a layout ROW is also a .ft-grid tr, so pinning
     data rows pinned the columns too - and a row of columns that will not break
     leaves the page it did not fit on blank and starts again lower down. Three
     sheets of a long test report were a heading and white space. */
  .ft-grid:not(.ft-grid--layout) tr { break-inside: avoid; page-break-inside: avoid; }
  .ft-grid--layout, .ft-grid--layout > tbody, .ft-grid--layout > tbody > tr,
  .ft-grid--layout > tbody > tr > td {
    break-inside: auto;
    page-break-inside: auto;
  }
  .ft-grid thead { display: table-header-group; }
  /* The screen breakpoint stacks a row of columns on a narrow phone. A sheet
     of A4 is not a narrow phone, and a report whose figures un-column
     themselves on the way to the printer is the wrong document. */
  .ft-cols { flex-direction: row !important; }
  .ft-doc h1, .ft-doc h2, .ft-doc h3, .ft-doc h4 {
    break-after: avoid;
    page-break-after: avoid;
  }
  /* Boards and charts carry their colour inline; keep it rather than let the
     browser helpfully drain a green pitch to grey.
     .ft-grid__bar is in here for the same reason and is the easiest to miss:
     the comparison's share column is a background GRADIENT, and a browser
     printing without backgrounds drops it entirely rather than paling it. */
  /* EVERYTHING, not a list of three. "Background graphics" is OFF by default in
     Chrome's print dialogue, and in this document a fill is never decoration:
     the tinted label column, a group row's dark band, a hollow bar against a
     solid one. Naming individual selectors here meant the report printed from
     the dialogue looked nothing like the same report printed headless, which is
     a difference nobody should have to know about. */
  .ft-doc, .ft-doc *, svg, .ft-figure__split, .ft-grid__bar {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}
`;

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** The masthead. PAGE ONE ONLY, and that is a correction.
 *
 *  It was `position: fixed` so it would repeat on every sheet the way a
 *  report's running head does. Chrome does repeat a fixed element — and lays
 *  the flowing content out UNDERNEATH it, because the page's content box does
 *  not know the header is there. Printing the first real note put the header
 *  over the top two lines of every page after the first, and a top padding on
 *  the article only fixes page one.
 *
 *  The techniques that do work are both worse than the problem: `@page` margin
 *  boxes, which Chrome has not implemented, or wrapping the entire document in
 *  a table so `<thead>` repeats — which would nest every layout table inside a
 *  table and is a lot of structure to buy a line of grey text.
 *
 *  So it is a masthead: fixture, competition, score, once, at the top. The
 *  print dialogue adds page numbers if the reader wants them.
 */
const header = (title: string, sub: string) =>
  `<header class="ft-head">` +
  `<span class="ft-head__mark">Fut Traffic</span>` +
  `<span class="ft-head__title">${esc(title)}` +
  (sub ? `<span class="ft-head__sub">${esc(sub)}</span>` : "") +
  `</span></header>`;

/** The whole file: doctype, paper, the document. Self-contained by
 *  construction — no script, no fetch, no stylesheet to go and find. The board
 *  is inline SVG and the figures are real markup, so this opens the same in
 *  five years as it does today. */
export function documentHtml(
  bodyHtml: string,
  title: string,
  rules = documentRules(),
  sub = "",
): string {
  return (
    `<!doctype html>\n<html lang="en"><head><meta charset="utf-8">` +
    `<meta name="viewport" content="width=device-width,initial-scale=1">` +
    `<title>${esc(title)}</title>` +
    `<style>${PAPER}\n${rules}\n${OVERRIDES}\n${page(docOrient(bodyHtml), docMargin(bodyHtml))}</style></head>` +
    `<body><article class="ft-doc ft-paper">${header(title, sub)}${bodyHtml}</article></body></html>`
  );
}

/** The same page, without the masthead and without a title: what the gallery
 *  card draws, scaled down, inside a sandboxed frame.
 *
 *  A card shows the TOP OF THE NOTE AS IT ACTUALLY IS — its heading, its first
 *  figure, its columns — because that is what a coach recognises a document by,
 *  and no snippet of ours reproduces it. Rebuilt as a standalone page for the
 *  same reason the print export is: the styles that make a note look like a
 *  note live on `.ft-*` classes, and a frame has none of the app's stylesheet.
 *
 *  Paper rather than the app's dark surface, deliberately: a wall of little
 *  white pages reads as a shelf of documents, and it is what the note looks
 *  like when it leaves here.
 */
export function previewHtml(bodyHtml: string, rules = documentRules()): string {
  return (
    `<!doctype html><html lang="en"><head><meta charset="utf-8">` +
    `<style>${PAPER}\n${rules}\n${OVERRIDES}\n${page(docOrient(bodyHtml), docMargin(bodyHtml))}</style></head>` +
    `<body><article class="ft-doc ft-paper">${bodyHtml}</article></body></html>`
  );
}
