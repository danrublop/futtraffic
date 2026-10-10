// Path resolution for the Electron shell. One place that knows where the
// renderer bundle, the backend binary, and the writable data dirs live, in
// dev and packaged modes alike.
import { app } from "electron";
import path from "node:path";

export const DESKTOP_DIR = path.join(__dirname, "..");
export const REPO_ROOT = path.join(DESKTOP_DIR, "..", "..");

/** Static renderer bundle (next build with FUTTRAFFIC_DESKTOP=1 → out/). */
export function getStaticAppDir(): string {
  return app.isPackaged
    ? path.join(__dirname, "..", "out") // inside asar
    : path.join(REPO_ROOT, "app", "frontend", "out");
}

/** Frozen FastAPI backend (app/desktop/scripts/build-backend.sh). */
export function getBackendBin(): string {
  return app.isPackaged
    ? path.join(process.resourcesPath, "backend", "futtraffic-api")
    : path.join(DESKTOP_DIR, "resources", "backend", "futtraffic-api");
}

/** Dev fallback when the frozen backend has not been built yet. */
export function getDevPython(): { cmd: string; args: string[] } {
  return {
    cmd: path.join(REPO_ROOT, ".venv", "bin", "python"),
    args: [path.join(REPO_ROOT, "app", "server", "app.py")],
  };
}

/** Match data (uploads, jobs, rosters). Packaged installs must not write into
 *  the read-only bundle; dev keeps using the repo's data/ so both workflows
 *  see the same matches. jobs.py honors FUTTRAFFIC_DATA. */
export function getDataDir(): string {
  return app.isPackaged
    ? path.join(app.getPath("userData"), "data")
    : path.join(REPO_ROOT, "data");
}

export function getLogDir(): string {
  return path.join(app.getPath("userData"), "logs");
}

/** RF-DETR downloads checkpoint weights on first use; they must land somewhere
 *  writable. */
export function getHFCacheDir(): string {
  return path.join(app.getPath("userData"), "hf-cache");
}

/** The demo match (Northside United v Riverside Athletic, synthetic) that ships with the app, seeded
 *  into the data dir on first run so a fresh install is not an empty list. */
export function getDemoSeedDir(): string {
  return app.isPackaged
    ? path.join(process.resourcesPath, "demo-job")
    : path.join(DESKTOP_DIR, "resources", "demo-job");
}
