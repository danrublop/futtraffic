// esbuild bundler for the main process and preload -> dist-electron/.
// The bundling rules: electron + node builtins
// external, application code bundled inline, CJS for the node platform.
//   node electron/build.mjs           one-shot
//   node electron/build.mjs --watch   rebuild on change
import esbuild from "esbuild";
import path from "node:path";
import { fileURLToPath } from "node:url";

const desktopDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const outdir = path.join(desktopDir, "dist-electron");

const nodeBuiltins = new Set([
  "assert", "buffer", "child_process", "cluster", "console", "constants",
  "crypto", "dgram", "dns", "domain", "events", "fs", "http", "http2",
  "https", "inspector", "module", "net", "os", "path", "perf_hooks",
  "process", "punycode", "querystring", "readline", "repl", "stream",
  "string_decoder", "sys", "timers", "tls", "tty", "url", "util", "v8",
  "vm", "worker_threads", "zlib",
]);

const externalizeBareSpecifiers = {
  name: "externalize-bare-specifiers",
  setup(build) {
    build.onResolve({ filter: /^[^./]/ }, (args) => {
      // On Windows an absolute entry path ("C:\...") also matches this
      // filter; an entry point can never be external.
      if (args.kind === "entry-point") return null;
      const root = args.path.split("/")[0];
      if (args.path === "electron" || nodeBuiltins.has(root) || root.startsWith("node:")) {
        return { external: true };
      }
      // No node_modules deps today; externalize anyway so a future dep is
      // resolved from node_modules at runtime rather than silently bundled.
      return { external: true };
    });
  },
};

const shared = {
  platform: "node",
  format: "cjs",
  target: "node22",
  bundle: true,
  sourcemap: true,
  outdir,
  plugins: [externalizeBareSpecifiers],
};

const entries = [
  path.join(desktopDir, "electron", "main.ts"),
  path.join(desktopDir, "electron", "preload.ts"),
];

if (process.argv.includes("--watch")) {
  const ctx = await esbuild.context({ ...shared, entryPoints: entries });
  await ctx.watch();
  console.log("esbuild watching electron/ -> dist-electron/");
} else {
  await esbuild.build({ ...shared, entryPoints: entries });
  console.log("dist-electron/ built");
}
