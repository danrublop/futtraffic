// Build the desktop renderer: `next build` with FUTTRAFFIC_DESKTOP=1, which
// flips app/frontend/next.config.ts to output:"export" (and drops the /api
// rewrite - the futtraffic:// protocol handler proxies /api to the backend
// sidecar instead). Output lands in app/frontend/out/, which electron-builder
// picks up via a files mapping.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const desktopDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const frontendDir = path.join(path.dirname(desktopDir), "frontend");

const result = spawnSync("npm", ["run", "build"], {
  cwd: frontendDir,
  stdio: "inherit",
  env: {
    ...process.env,
    FUTTRAFFIC_DESKTOP: "1",
    NEXT_PUBLIC_FUTTRAFFIC_DESKTOP: "1",
  },
});
if (result.status !== 0) process.exit(result.status ?? 1);

if (!fs.existsSync(path.join(frontendDir, "out", "index.html"))) {
  console.error("next build produced no app/frontend/out/index.html");
  process.exit(1);
}
console.log("desktop renderer -> app/frontend/out/");
