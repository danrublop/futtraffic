import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // THE SURFACES READ AND WRITE THEIR PARAMETERS THROUGH useSurfaceParams.
  // Each can be open in a dock tab beside the chat, where `?m=`/`?n=` belong to
  // the tab, not the page. A router call in here navigates the page the chat is
  // on - the match picker inside a note did exactly that. See
  // src/components/surface-params.tsx.
  {
    files: [
      "src/components/notepad/**",
      "src/components/data/**",
      "src/components/timeline/**",
      "src/components/video-panel.tsx",
      "src/components/doc-panel.tsx",
      "src/components/film-gallery.tsx",
      "src/components/match-picker.tsx",
      "src/hooks/**",
    ],
    ignores: ["**/*.test.ts", "**/*.test.tsx"],
    rules: {
      "no-restricted-imports": ["error", {
        paths: [{
          name: "next/navigation",
          message: "Surfaces can be docked: use useSurfaceParams() from @/components/surface-params.",
        }],
      }],
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
