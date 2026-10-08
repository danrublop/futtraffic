// FutTraffic main process.
//
// No HTTP server for the renderer. The UI is
// a static Next export served from disk through a privileged `futtraffic://`
// scheme; the one exception is `/api/*`, which the protocol handler proxies
// to the FastAPI sidecar (app/desktop/electron/backend-host.ts) so the frontend's
// same-origin fetches work unchanged - including raw multi-GB streamed uploads
// and Range requests on clips.
import { app, BrowserWindow, ipcMain, net, protocol } from "electron";
import fsSync from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";

import { backendPort, disposeBackend, ensureBackend } from "./backend-host";
import { DESKTOP_DIR, getStaticAppDir } from "./paths";

// Must run synchronously before app.ready.
protocol.registerSchemesAsPrivileged([
  {
    scheme: "futtraffic",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
]);

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".wasm": "application/wasm",
};

/** /api/* -> the FastAPI sidecar. Method, headers (Range, X-Upload-Offset)
 *  and the request body pass through untouched; the body is streamed, never
 *  buffered - a match upload can be tens of GB. */
async function handleApi(request: Request, pathname: string, search: string): Promise<Response> {
  let port: number;
  try {
    port = await ensureBackend();
  } catch (err) {
    return new Response(`futtraffic backend unavailable: ${err}`, { status: 502 });
  }
  const headers = new Headers(request.headers);
  headers.delete("host");
  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  return net.fetch(`http://127.0.0.1:${port}${pathname}${search}`, {
    method: request.method,
    headers,
    body: hasBody ? request.body : undefined,
    // Carry the renderer's abort through, so a dropped agent stream closes the
    // upstream request too. Belt and braces: the chat's Stop also POSTs
    // /api/agent/chats/{id}/stop, which does not depend on this.
    signal: request.signal,
    // @ts-expect-error - duplex is required for a streaming body and is
    // supported by Chromium's fetch; the DOM lib type lags behind.
    duplex: hasBody ? "half" : undefined,
  });
}

/** Static renderer bundle. Hand-rolled (not net.fetch file://) for the same
 *  reason any static handler is: traversal/symlink guards, directory index fallback,
 *  and manual Range/206 - Chromium's media stack fails without 206. */
async function handleStatic(pathname: string, rangeHeader: string | null): Promise<Response> {
  const baseDir = path.resolve(getStaticAppDir());
  const raw = decodeURIComponent(pathname).replace(/^\/+/, "") || "index.html";
  let filePath = path.resolve(baseDir, raw);

  if (!filePath.startsWith(baseDir + path.sep) && filePath !== baseDir) {
    return new Response("Forbidden", { status: 403 });
  }
  if (!path.extname(filePath)) {
    // Next export emits /checks/index.html for /checks.
    const asDir = path.join(filePath, "index.html");
    if (fsSync.existsSync(asDir)) filePath = asDir;
    else if (fsSync.existsSync(filePath + ".html")) filePath += ".html";
  }
  try {
    // Symlinks must not escape the bundle either.
    const real = await fsp.realpath(filePath);
    if (!real.startsWith(baseDir + path.sep)) {
      return new Response("Forbidden", { status: 403 });
    }
    const stat = await fsp.stat(real);
    const type =
      CONTENT_TYPES[path.extname(real).toLowerCase()] ?? "application/octet-stream";
    return serveFile(real, stat.size, type, rangeHeader);
  } catch {
    return new Response("Not found", { status: 404 });
  }
}

function serveFile(
  filePath: string,
  size: number,
  type: string,
  rangeHeader: string | null,
): Response {
  const match = rangeHeader ? /bytes=(\d*)-(\d*)/.exec(rangeHeader) : null;
  if (match && (match[1] || match[2])) {
    const start = match[1] ? parseInt(match[1], 10) : Math.max(0, size - parseInt(match[2], 10));
    const end = match[1] && match[2] ? Math.min(parseInt(match[2], 10), size - 1) : size - 1;
    if (start >= size || start > end) {
      return new Response("Range not satisfiable", {
        status: 416,
        headers: { "Content-Range": `bytes */${size}` },
      });
    }
    const stream = Readable.toWeb(
      fsSync.createReadStream(filePath, { start, end }),
    ) as ReadableStream;
    return new Response(stream, {
      status: 206,
      headers: {
        "Content-Type": type,
        "Content-Length": String(end - start + 1),
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Accept-Ranges": "bytes",
      },
    });
  }
  const stream = Readable.toWeb(fsSync.createReadStream(filePath)) as ReadableStream;
  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": type,
      "Content-Length": String(size),
      "Accept-Ranges": "bytes",
    },
  });
}

function registerProtocol(): void {
  protocol.handle("futtraffic", async (request) => {
    const url = new URL(request.url);
    if (url.hostname !== "app") return new Response("Not found", { status: 404 });
    if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
      return handleApi(request, url.pathname, url.search);
    }
    return handleStatic(url.pathname, request.headers.get("range"));
  });
}

/** The desktop half of the notepad's Print button. The static export has no
 *  /export/pdf route, so the renderer builds the finished document itself
 *  (documentHtml, with the rules lifted from its own live CSSOM) and this
 *  prints it in a hidden window - same Chromium, no dialog, no headers.
 *  JavaScript is off in that window: the note is user content. */
ipcMain.handle("print-pdf", async (_e, html: unknown) => {
  if (typeof html !== "string" || !html) throw new Error("expected a document");
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "futtraffic-pdf-"));
  const file = path.join(dir, "doc.html");
  const win = new BrowserWindow({ show: false, webPreferences: { javascript: false, sandbox: true } });
  try {
    await fsp.writeFile(file, html);
    await win.loadFile(file);
    return await win.webContents.printToPDF({ preferCSSPageSize: true, printBackground: true });
  } finally {
    win.destroy();
    void fsp.rm(dir, { recursive: true, force: true });
  }
});

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: "#121212",
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 12, y: 13 },
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  // FUTTRAFFIC_START_URL overrides the entry page (dev loops, spikes).
  mainWindow.loadURL(process.env.FUTTRAFFIC_START_URL ?? "futtraffic://app/index.html");
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  // A packaged build takes its icon from electron-builder.yml; `npm run dev`
  // would otherwise show Electron's own icon in the dock.
  if (!app.isPackaged) app.dock?.setIcon(path.join(DESKTOP_DIR, "icon.png"));
  registerProtocol();
  // Boot the backend alongside the window rather than waiting for the first
  // /api request - torch import is slow enough that laziness is felt.
  void ensureBackend().catch((err) => console.error("backend boot failed:", err));
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", () => disposeBackend());
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    disposeBackend();
    app.quit();
  }
});

export { backendPort };
