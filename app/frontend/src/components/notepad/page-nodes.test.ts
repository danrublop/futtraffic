import { describe, expect, it } from "vitest";

import { MARGIN_DEFAULT, docMargin, docOrient, marginOf, orientOf } from "./page-nodes";
import { safeSrc } from "./face-node";

describe("paper", () => {
  it("is landscape unless the document asks for portrait", () => {
    // Every note written before pages existed already PRINTED landscape, so
    // absent has to mean landscape or the whole library silently reflows.
    expect(docOrient("<h1>hi</h1>")).toBe("landscape");
    expect(docOrient('<div data-pagesetup="landscape"></div><p>x</p>')).toBe("landscape");
    expect(docOrient('<div data-pagesetup="portrait"></div><p>x</p>')).toBe("portrait");
  });

  it("takes nothing but the two words", () => {
    expect(orientOf(null)).toBe("landscape");
    expect(orientOf("")).toBe("landscape");
    expect(orientOf("PORTRAIT")).toBe("landscape"); // not a key, so the default
    expect(orientOf("portrait")).toBe("portrait");
  });
});

describe("a player's photograph", () => {
  it("takes the three formats a browser will certainly draw", () => {
    expect(safeSrc("data:image/png;base64,iVBORw0KGgo=")).toBe("data:image/png;base64,iVBORw0KGgo=");
    expect(safeSrc("data:image/jpeg;base64,/9j/4AAQ")).toBe("data:image/jpeg;base64,/9j/4AAQ");
    expect(safeSrc("/api/players/tomas-varga/photo")).toBe("/api/players/tomas-varga/photo");
  });

  it("takes nothing else, because this string is written into markup", () => {
    // An SVG data URI is the interesting one: it is an image everywhere else
    // and a document that can carry script here.
    for (const bad of [
      "data:image/svg+xml;base64,PHN2Zz4=",
      "javascript:alert(1)",
      "https://example.com/face.jpg",
      '/api/players/x/photo" onerror="alert(1)',
      "",
      null,
    ]) {
      expect(safeSrc(bad)).toBe("");
    }
  });
});

describe("the paper margin", () => {
  it("falls back to the paper's own default when the document says nothing", () => {
    // Every document written before the margin was authorable has no value for
    // it, and those documents did not mean "no margin".
    expect(docMargin('<div data-pagesetup="landscape"></div>')).toBe(MARGIN_DEFAULT.landscape);
    expect(docMargin('<div data-pagesetup="portrait"></div>')).toBe(MARGIN_DEFAULT.portrait);
    expect(docMargin("<p>a note with no setup at all</p>")).toBe(MARGIN_DEFAULT.landscape);
  });

  it("reads the one the document asked for", () => {
    expect(docMargin('<div data-pagesetup="portrait" data-margin="25"></div>')).toBe(25);
  });

  it("clamps to what a printer can actually reach", () => {
    // Under ~5mm most desktop printers crop without saying so; over 40mm on A4
    // there is no measure left to put a figure in.
    expect(marginOf(1)).toBe(5);
    expect(marginOf(200)).toBe(40);
    expect(marginOf(20)).toBe(20);
  });

  it("treats absent, zero and rubbish as absent rather than as zero", () => {
    for (const v of [null, undefined, "", 0, "wide"]) expect(marginOf(v)).toBe(0);
  });
});
