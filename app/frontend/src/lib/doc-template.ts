import { DEFAULT_MAGNETS } from "@/components/notepad/pitch-data";
import { buildTable, defaultTitle } from "@/components/notepad/stat-table";
import type { MatchReport } from "./types";

/** The starter document a first-open 404 seeds.
 *
 *  This used to be a whole standalone HTML page — inline <style>, an inline
 *  <script>, a canvas chart, and the report embedded as a JSON literal — because
 *  the notepad had no editor and rendered the doc in a sandboxed iframe. The
 *  notepad is a rich text editor now, so a page like that could not survive
 *  being loaded into it: TipTap parses to its schema and everything that is not
 *  a node it knows is dropped. The document would open, get autosaved, and lose
 *  its own chart.
 *
 *  So the seed is what the editor actually speaks: headings, paragraphs, a
 *  pitch block and a ranked table. The numbers are EMBEDDED rather than
 *  fetched — that part was never about the iframe:
 *
 *    - a document that phones home is broken the moment it leaves the app,
 *      which is the point of the thing (export to PDF, send to a player);
 *    - and notes should not silently change under the person who wrote them.
 *
 *  IT OPENS AS A PAGE, NOT A LIST. Every page of a scouting report is two or
 *  three independent columns — a pitch and its caption in one, the prose about
 *  how they play in the next, the numbers down the side — and the export prints
 *  landscape for exactly that. A seed that is one long stack teaches the
 *  opposite shape, and a coach who starts in a stack stays in one, so notes came
 *  out portrait on a landscape sheet. The row here is an ordinary layout table,
 *  which means the first thing somebody who works differently can do is delete
 *  a column or the whole row.
 */

const attr = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const text = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** A borderless row of columns, as the editor's own table markup. The class is
 *  what `layout` parses from, so it has to be exactly what renderHTML writes. */
const columns = (cells: string[]) =>
  `<table class="ft-grid ft-grid--layout" data-columns="${cells.length}"><tbody><tr>` +
  cells.map((c) => `<td>${c}</td>`).join("") +
  `</tr></tbody></table>`;

export function starterDoc(matchId: string | null, report?: unknown): string {
  const r = (report ?? null) as MatchReport | null;
  const meta = r?.meta;
  const title =
    meta?.home_name && meta?.away_name
      ? `${meta.home_name} v ${meta.away_name}`
      : matchId
        ? "Match notes"
        : "Notes";

  const table = buildTable({
    scope: "match",
    group: "player",
    metric: "distance_km",
    limit: 10,
    report: r,
    season: null,
  });
  // A REAL table, not the ranked-bars block this used to seed.
  //
  // That block was read-only: the first numbers a coach ever saw sat in a
  // figure with no caret in it, and the only way to put a note beside a
  // player's distance was to delete the whole thing and build a table by hand.
  // It is the same rows, in the markup `TableNode.parseHTML` reads back, so
  // every cell takes typing and the toolbar's rows/columns/merge all apply.
  //
  // What is lost is the little bar behind each value. A bar is a picture of a
  // proportion and this is a document you write in; `/visual` still draws one
  // for anyone who wants it, beside the table rather than inside it.
  const anySub = table.rows.some((r) => r.sub);
  const head = ["Player", ...(anySub ? ["Team"] : []), "Distance"];
  const cell = (tag: "th" | "td", v: string) => `<${tag}><p>${text(v)}</p></${tag}>`;
  const row = (cells: string[], tag: "th" | "td") =>
    `<tr>${cells.map((c) => cell(tag, c)).join("")}</tr>`;
  const provenance = [table.refusal ?? table.note].filter(Boolean).join(" ");
  const tableHtml = table.rows.length
    ? `<h4>${text(defaultTitle("match", "player", "distance_km"))}</h4>` +
      `<table class="ft-grid"><tbody>` +
      row(head, "th") +
      table.rows
        .map((r) => row([r.name, ...(anySub ? [r.sub] : []), r.value], "td"))
        .join("") +
      `</tbody></table>` +
      (provenance ? `<p>${text(provenance)}</p>` : "")
    : `<p>${text(table.refusal || table.note || "No tracked players in this match yet.")}</p>`;

  // `data-stripes="0"` spelled out. The parse reads ABSENT as on, so that a
  // board saved before stripes went off the default keeps them — which means a
  // seed that says nothing gets them too. The seed has to declare what it
  // wants rather than inherit a rule written for old documents.
  // "How they set up" over a hardcoded 4-4-2 is a CLAIM, and it was the first
  // thing every coach saw. The shape is a starting point to drag about, so the
  // seed says that instead of asserting a formation nobody measured.
  const board =
    `<div data-pitch="${attr(JSON.stringify(DEFAULT_MAGNETS))}" data-stripes="0" ` +
    `data-caption="Shape — drag the discs" ` +
    `data-note="${attr("A default 4-4-2 to draw over, not this team's shape. Use the board's “Average positions” to place the tracked players.")}"></div>`;

  // The headings are the skeleton of a scouting report, not decoration: a blank
  // page is the reason notes get written in a group chat instead.
  const shape = `<h3>Shape</h3>${board}`;
  const prose =
    `<h3>In possession</h3><p></p>` +
    `<h3>Out of possession</h3><p></p>` +
    `<h3>Set pieces</h3><p></p>`;
  const numbers = matchId
    ? `<h3>Numbers</h3>${tableHtml}`
    : `<h3>Numbers</h3><p>Pick a match to pull numbers in.</p>`;

  return (
    `<h2>${text(title)}</h2>` +
    columns([shape, prose, numbers]) +
    `<p>Type <code>/</code> for a table, a visual, a play or a board. ` +
    `<code>/columns</code> starts another row.</p>`
  );
}
