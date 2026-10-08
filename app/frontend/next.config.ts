import type { NextConfig } from "next";

// Where the FastAPI service lives. Next serves the UI and proxies /api to
// Python, so the browser sees a single origin and there is no CORS to set up.
// Works the same under `next dev` and `next start`.
const API = process.env.FUTTRAFFIC_API ?? "http://127.0.0.1:8000";

// FUTTRAFFIC_DESKTOP=1 (set by app/desktop/scripts/build-renderer.mjs) builds the
// static export the Electron shell serves over the futtraffic:// scheme.
// Rewrites do not exist in a static export — there the protocol handler in
// app/desktop/electron/main.ts proxies /api to the backend sidecar instead.
const DESKTOP = process.env.FUTTRAFFIC_DESKTOP === "1";

const nextConfig: NextConfig = DESKTOP
  ? {
      output: "export",
    }
  : {
      // Standalone = a self-contained server.js with only the runtime deps
      // traced in, for container-style deploys of the web app.
      output: "standalone",
      // /export/pdf renders the document through a real Chromium, because the
      // browser's own print dialog draws page numbers, a URL and a date over
      // whatever it prints and no page can turn that off. Playwright must not
      // be bundled: it resolves its driver and its browser by path at runtime,
      // and a bundled copy has neither.
      //
      // THE BROWSER BINARY IS NOT IN node_modules by default - Playwright puts
      // it in a per-user cache, which a container image built from this output
      // will not have. A deploy needs `npx playwright install --with-deps
      // chromium` in the image, or PLAYWRIGHT_BROWSERS_PATH=0 at install time
      // to keep it beside the package. Without it the route answers 501 with
      // that instruction rather than failing obscurely, so the rest of the app
      // still works.
      serverExternalPackages: ["playwright"],
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
      },
      // The workspace docs tab renders user-authored HTML in a sandboxed
      // iframe (sandbox="allow-scripts" → opaque origin). Its fetch of
      // /api/matches/{id}/match.json is cross-origin from there, so the API
      // answers carry a wildcard ACAO. The API is unauthenticated, read-mostly
      // local data — no credentials are ever attached from that origin.
      async headers() {
        return [
          {
            source: "/api/:path*",
            headers: [{ key: "Access-Control-Allow-Origin", value: "*" }],
          },
        ];
      },
    };

export default nextConfig;
