import { describe, expect, it } from "vitest";

import { documentHtml } from "./doc-export";

/** The print stylesheet, which IS the file format.
 *
 *  There is no download button and no server-side renderer, so everything a
 *  coach sends leaves through here. That makes a malformed rule invisible
 *  until somebody prints — and this file is edited by hand every time a block
 *  gains a class, which is how a stray regex once left `svg, .ft-figure__split,
 *  }` in it: valid TypeScript, silently broken CSS.
 *
 *  Runs under jsdom, where `document.styleSheets` is empty, so what is checked
 *  is the sheet this file WRITES rather than the app's own.
 */
describe("the print stylesheet", () => {
  // The whole page, as a coach would be handed it — paper palette, print
  // rules, and whatever the app's own sheet contributed. Asserting on the
  // assembled artefact rather than on one constant is the point: the bug this
  // guards against was a rule left dangling between two of them.
  const page = documentHtml("<p>x</p>", "Match", "");
  const css = page.slice(page.indexOf("<style>"), page.lastIndexOf("</style>"));

  it("is balanced, and never closes a brace it did not open", () => {
    let depth = 0;
    let worst = 0;
    for (const ch of css) {
      if (ch === "{") depth += 1;
      if (ch === "}") depth -= 1;
      worst = Math.min(worst, depth);
    }
    expect(depth).toBe(0);
    expect(worst).toBe(0);
  });

  it("has no rule with an empty or dangling selector", () => {
    // `svg, .ft-figure__split, }` parses as TypeScript and prints as nothing.
    expect(css).not.toMatch(/,\s*\{/);
    expect(css).not.toMatch(/,\s*\}/);
    expect(css).not.toMatch(/\{\s*\}/);
  });

  it("keeps the colours a browser would otherwise drain away", () => {
    // Boards, charts and the share column carry their colour inline or as a
    // background; printing without this drops them rather than paling them.
    expect(css).toContain("print-color-adjust: exact");
    expect(css).toContain("svg");
    expect(css).toContain(".ft-grid__bar");
  });

  it("styles the blocks a note can actually contain", () => {
    for (const cls of [".ft-grid", ".ft-figure", ".ft-chart", ".ft-pitch"]) {
      expect(css).toContain(cls);
    }
  });

  it("carries nothing for the ranked block that no longer exists", () => {
    // statTable was deleted; its print rules outlived it by several commits.
    expect(css).not.toContain(".ft-table__");
  });

  it("lets the masthead sit on the first sheet even behind a page setup", () => {
    // MEASURED BUG, 2026-09-02. `.ft-pagesetup` is display:none in the export,
    // which removes it from the page and NOT from the sibling chain, so a bare
    // `.ft-head + .ft-page` stopped matching the moment a document carried one.
    // A document carries one exactly when it asked for portrait — the case most
    // likely to be a printed report — and the symptom was page one printing as
    // a masthead over nothing, with every sheet pushed down by one.
    //
    // jsdom has no layout, so the rule cannot be exercised here; what is pinned
    // is that the selector still tolerates the node in between.
    expect(css).toContain(".ft-head + .ft-pagesetup + .ft-page");
    // And that the plain adjacency was not replaced by it.
    expect(css).toContain(".ft-head + .ft-page");
  });
});
