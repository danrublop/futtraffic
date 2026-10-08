/** A PDF OF THE DOCUMENT, WITHOUT THE BROWSER'S PRINT DIALOG.
 *
 *  Printing through Chrome's own dialog - which the panel used to offer as a
 *  second button, and no longer does - draws Chrome's furniture around
 *  whatever it prints: the date and the document title across the top, the page
 *  URL and "1/8" along the bottom. None of that is ours and NO PAGE CAN TURN IT
 *  OFF - it is a checkbox in the user's print settings ("Headers and footers"),
 *  and `@page { margin: 0 }` only persuades Chrome to skip it when the user has
 *  not asked for it. A document meant to be handed to a coach cannot depend on
 *  a setting in the reader's browser.
 *
 *  So the same render runs here instead, server-side, through the same
 *  Chromium that scripts/note-api.mts uses for writePdf - no dialog, no
 *  headers, no URL, and the paper the document actually asked for. The button
 *  in the notepad downloads the result.
 *
 *  NOT UNDER /api. next.config.ts rewrites /api/* to the FastAPI service, so a
 *  route handler there would never be reached.
 *
 *  The desktop build is `output: "export"`, which has no route handlers at all
 *  - there the Electron shell owns printing and `webContents.printToPDF` is the
 *  same job with no extra dependency. This is the web path.
 */
import { readFile } from "node:fs/promises";
import path from "node:path";

import { documentHtml } from "@/lib/doc-export";

/** Chromium has to start, and a big report has a lot of inline SVG in it. */
export const maxDuration = 60;

const CSS = path.join(process.cwd(), "src/app/globals.css");

export async function POST(req: Request) {
  let body: { html?: unknown; title?: unknown; sub?: unknown };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "expected JSON" }, { status: 400 });
  }
  const html = typeof body.html === "string" ? body.html : "";
  if (!html) return Response.json({ error: "expected {html: string}" }, { status: 400 });
  const title = typeof body.title === "string" ? body.title : "";
  const sub = typeof body.sub === "string" ? body.sub : "";

  let chromium;
  try {
    ({ chromium } = await import("playwright"));
  } catch {
    return Response.json(
      { error: "PDF export needs Playwright: npm i playwright && npx playwright install chromium" },
      { status: 501 },
    );
  }

  // TWO SEPARATE ABSENCES, and the second is the one a deploy actually hits.
  // The package resolving does not mean the BROWSER is on disk: Playwright puts
  // it in a per-user cache, which a container image built from `output:
  // "standalone"` will not have. Left to the generic handler below that comes
  // back as a 500 carrying a stack trace about a missing executable; said here
  // it comes back as the command to run.
  let browser;
  try {
    browser = await chromium.launch();
  } catch (e) {
    return Response.json(
      {
        error:
          "PDF export could not start Chromium — run `npx playwright install chromium` " +
          `on this host. (${e instanceof Error ? e.message.split("\n")[0] : String(e)})`,
      },
      { status: 501 },
    );
  }
  try {
    const page = await browser.newPage();
    // Two passes over one page, the same as note-api.mts: an empty document
    // wearing the app's stylesheet so there is a CSSOM to lift the .ft- rules
    // out of, then the real document with those rules baked into its <style>.
    // documentRules() needs a live CSSOM and returns "" without one, silently,
    // which is how this produces a correctly-paginated unstyled PDF if it is
    // ever called from Node directly.
    await page.setContent("<!doctype html><html><body></body></html>");
    await page.addStyleTag({ content: await readFile(CSS, "utf8") });
    const rules = await page.evaluate(() => {
      const out: string[] = [];
      for (const sheet of Array.from(document.styleSheets)) {
        let list: CSSRule[];
        try {
          list = Array.from(sheet.cssRules ?? []);
        } catch {
          continue;
        }
        for (const r of list) {
          const text = r.cssText;
          if (text.includes(".ft-") && !text.includes(".ft-controls")) out.push(text);
        }
      }
      return out.join("\n");
    });
    if (!rules.trim()) {
      return Response.json({ error: "no .ft-* rules found; the PDF would be unstyled" }, { status: 500 });
    }

    await page.setContent(documentHtml(html, title, rules, sub), { waitUntil: "load" });
    const pdf = await page.pdf({ preferCSSPageSize: true, printBackground: true });
    return new Response(new Uint8Array(pdf), {
      headers: {
        "content-type": "application/pdf",
        // `attachment` so it saves rather than opening in a viewer that would
        // add its own chrome back.
        "content-disposition": `attachment; filename="${(title || "document").replace(/["\\]/g, "")}.pdf"`,
        "content-length": String(pdf.length),
      },
    });
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  } finally {
    await browser.close();
  }
}
