import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

/** The suite. Two environments, because most of it does not need a DOM.
 *
 *  It used to be node-only and `*.test.ts` only, which quietly meant every
 *  `.tsx` in the project was untestable BY CONFIGURATION — 4,300 lines of it in
 *  the notepad alone, including the three largest files in the feature. The
 *  pure logic was well covered and the components had no coverage at all, and
 *  nothing said so out loud.
 *
 *  `environmentMatchGlob` is gone in Vitest 4, so the split is done with
 *  projects: `.test.ts` keeps the fast node environment, `.test.tsx` gets
 *  jsdom. A component test costs about 20x a pure one to start up, and there is
 *  no reason to pay that for a function that takes numbers and returns numbers.
 *
 *  No @vitejs/plugin-react: tsconfig says `jsx: "react-jsx"`, the automatic
 *  runtime, which esbuild already compiles. The plugin exists for Fast Refresh,
 *  and a test run does not refresh.
 */
const alias = { "@": fileURLToPath(new URL("./src", import.meta.url)) };

export default defineConfig({
  resolve: { alias },
  test: {
    projects: [
      {
        // The `@/` alias tsconfig gives the app. doc-template.ts reaches into
        // components/ for the seed's board, and without this the suite cannot
        // resolve it.
        resolve: { alias },
        test: {
          name: "logic",
          environment: "node",
          include: ["src/**/*.test.ts"],
        },
      },
      {
        resolve: { alias },
        test: {
          name: "dom",
          environment: "jsdom",
          include: ["src/**/*.test.tsx"],
          setupFiles: ["./src/test/setup.ts"],
        },
      },
    ],
  },
});
