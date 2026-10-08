// FastAPI backend sidecar: lazy
// spawn, one persistent child, dispose on quit. Dreambyte speaks stdio
// JSON-RPC; our backend IS an HTTP service (the product API), so the child
// gets an ephemeral port and a readiness poll instead of a stdio protocol.
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";

import {
  getBackendBin,
  getDataDir,
  getDemoSeedDir,
  getDevPython,
  getHFCacheDir,
  getLogDir,
} from "./paths";

/** First run only: drop the bundled demo match into the data dir so a fresh
 *  install opens on a finished match, not an empty list. Skipped whenever the
 *  jobs dir already holds anything — deleting every match must not resurrect
 *  the demo. */
function seedDemo(dataDir: string): void {
  const seed = getDemoSeedDir();
  const jobsDir = path.join(dataDir, "jobs");
  try {
    if (!fs.existsSync(seed)) return;
    if (fs.existsSync(jobsDir) && fs.readdirSync(jobsDir).length > 0) return;
    fs.cpSync(seed, jobsDir, { recursive: true });
  } catch (err) {
    console.error("demo seed failed (non-fatal):", err);
  }
}

let child: ChildProcess | null = null;
let port: number | null = null;
let starting: Promise<number> | null = null;

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const address = srv.address();
      if (!address || typeof address === "string") {
        reject(new Error("no port from net.Server"));
        return;
      }
      srv.close(() => resolve(address.port));
    });
  });
}

function waitForDocs(p: number, tries = 180): Promise<void> {
  // torch import on cold boot is tens of seconds; 180 x 500ms = 90s budget.
  return new Promise((resolve, reject) => {
    const attempt = (left: number) => {
      if (child?.exitCode != null) {
        reject(new Error(`backend exited (${child.exitCode}) before serving`));
        return;
      }
      http
        .get(`http://127.0.0.1:${p}/docs`, (res) => {
          res.resume();
          resolve();
        })
        .on("error", () => {
          if (left <= 0) {
            reject(new Error("timeout waiting for backend /docs"));
            return;
          }
          setTimeout(() => attempt(left - 1), 500);
        });
    };
    attempt(tries);
  });
}

/** Spawn the backend if needed and return its port. Concurrent callers share
 *  one boot. */
export async function ensureBackend(): Promise<number> {
  if (port != null && child && child.exitCode == null) return port;
  if (starting) return starting;

  // Dev escape hatch: attach to a backend you started yourself (e.g. the repo
  // .venv running new code the frozen binary predates).
  const external = process.env.FUTTRAFFIC_BACKEND_PORT;
  if (external) {
    const p = parseInt(external, 10);
    await waitForDocs(p, 20);
    port = p;
    return p;
  }

  starting = (async () => {
    const p = await freePort();
    fs.mkdirSync(getLogDir(), { recursive: true });
    const out = fs.createWriteStream(`${getLogDir()}/backend.log`, { flags: "a" });
    seedDemo(getDataDir());

    const env = {
      ...process.env,
      PORT: String(p),
      FUTTRAFFIC_DATA: getDataDir(),
      HF_HOME: getHFCacheDir(),
    };
    const useFrozen = fs.existsSync(getBackendBin());
    const { cmd, args } = useFrozen ? { cmd: getBackendBin(), args: [] } : getDevPython();

    child = spawn(cmd, args, { env, stdio: ["ignore", "pipe", "pipe"] });
    child.stdout?.pipe(out);
    child.stderr?.pipe(out);
    child.on("close", (code) => {
      console.error(`futtraffic backend exited (${code}); see ${out.path}`);
      child = null;
      port = null;
      starting = null;
    });

    await waitForDocs(p);
    port = p;
    return p;
  })();

  try {
    return await starting;
  } finally {
    if (port == null) starting = null;
  }
}

export function backendPort(): number | null {
  return port;
}

export function disposeBackend(): void {
  if (child && child.exitCode == null) {
    try {
      child.kill("SIGKILL");
    } catch {
      /* already gone */
    }
  }
  child = null;
  port = null;
}
