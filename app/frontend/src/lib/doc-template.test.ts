import { describe, expect, it } from "vitest";

import { starterDoc } from "./doc-template";
import type { MatchReport } from "./types";

/** The seed is no longer a standalone page with an inline chart — the notepad
 *  is a rich text editor now, and anything the editor's schema does not know is
 *  dropped on the first autosave. So what matters here is that the seed is made
 *  of blocks the editor can hold, and that the numbers it bakes in arrive with
 *  the tier the pipeline gave them. */

const report = (over: Partial<MatchReport> = {}): MatchReport =>
  ({
    meta: { home_name: "Real Sociedad B", away_name: "Amorebieta" },
    teams: {
      home: {
        distance_m: 108_400,
        top_speed_kmh: 31.2,
        sprints: 88,
        shape: { width_m: 41.3, depth_m: 29.8, defensive_line_m: 33.1, compactness_m2: 1230 },
      },
      away: {
        distance_m: 104_100,
        top_speed_kmh: 30.4,
        sprints: 81,
        shape: { width_m: 39.9, depth_m: 31.2, defensive_line_m: 30.6, compactness_m2: 1244 },
      },
    },
    // Players, because the seed's whole job is to bake numbers in and the
    // fixture used to hand it none — so "bakes in real numbers" asserted
    // attributes on an empty block and would have passed with no table at all.
    players: [
      { id: 1, jersey: 7, team: "home", distance_km: 11.4, top_speed_kmh: 30.1, sprints: 22, minutes: 90 },
      { id: 2, jersey: 9, team: "away", distance_km: 10.2, top_speed_kmh: 29.4, sprints: 18, minutes: 90 },
    ],
    ...over,
  }) as MatchReport;

describe("starterDoc", () => {
  it("seeds a board the editor can parse back", () => {
    const html = starterDoc("m1");
    expect(html).toContain("<div data-pitch=");
    // 22 players and a ball, as attribute-escaped JSON.
    expect(html).toContain("&quot;side&quot;:&quot;home&quot;");
    expect(html).toContain("&quot;side&quot;:&quot;ball&quot;");
  });

  it("carries no script, style or document chrome", () => {
    // Every one of these was in the old template and none survives a round
    // trip through the editor, so emitting them would only lose them later.
    const html = starterDoc("m1", report());
    for (const dead of ["<script", "<style", "<!doctype", "<html", "<canvas"]) {
      expect(html.toLowerCase()).not.toContain(dead);
    }
  });

  it("bakes in real numbers with their tier", () => {
    // The ranked table, not a row of lone numbers: /stat came off the menu
    // because a bare figure floating in a document is not a figure, and a seed
    // that opens with four of them teaches the shape the report does not have.
    const html = starterDoc("m1", report());
    // A REAL table, not a block with the rows hidden in an attribute: every
    // cell has to be somewhere the caret can go, or the first numbers a coach
    // meets are ones they cannot write beside.
    expect(html).toContain('<table class="ft-grid">');
    expect(html).toContain("<th><p>Player</p></th>");
    expect(html).toContain("<th><p>Distance</p></th>");
    // Ordered by the metric, the leader first, formatted with its unit.
    expect(html).toContain("<td><p>#7</p></td>");
    expect(html).toContain("11.40 km");
    expect(html.indexOf("#7")).toBeLessThan(html.indexOf("#9"));
    // Nothing of the old read-only block survives.
    expect(html).not.toContain("data-rows=");
  });

  it("opens as a page rather than a list", () => {
    // Every page of a scouting report is independent columns, and the export
    // prints landscape for exactly that. A seed that is one long stack teaches
    // the opposite shape and the notes come out portrait.
    const html = starterDoc("m1", report());
    expect(html).toContain('class="ft-grid ft-grid--layout"');
    expect(html.match(/ft-grid ft-grid--layout/g)).toHaveLength(1);
    // The board sits in the first column, the numbers in the last.
    // The three LAYOUT cells, which are the row's own children — counting
    // every <td> in the document would now catch the numbers table's cells too.
    const cells = html.slice(html.indexOf("<tr>")).split("</td>");
    expect(cells[0]).toContain("data-pitch=");
    expect(cells[2]).toContain('<table class="ft-grid">');
  });

  it("names the fixture when the report knows it", () => {
    expect(starterDoc("m1", report())).toContain("<h2>Real Sociedad B v Amorebieta</h2>");
    expect(starterDoc("m1")).toContain("<h2>Match notes</h2>");
    expect(starterDoc(null)).toContain("<h2>Notes</h2>");
  });

  it("says what to do when there is no match to pull from", () => {
    expect(starterDoc(null)).toContain("Pick a match");
    // And never seeds a lone stat block, which is no longer a thing you can
    // even insert.
    expect(starterDoc("m1")).not.toContain("data-field=");
    expect(starterDoc(null)).not.toContain("data-field=");
  });

  it("escapes a fixture name that carries markup", () => {
    const html = starterDoc(
      "m1",
      report({ meta: { home_name: "<script>x</script>", away_name: "B" } }),
    );
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });
});
