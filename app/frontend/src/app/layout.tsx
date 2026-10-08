import type { Metadata, Viewport } from "next";
import { Suspense } from "react";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { AppShell } from "@/components/app-shell";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";

/* Two roles, and the boundary is strict.
 *
 *   UI/body  Inter Variable (loaded via @fontsource-variable/inter in
 *            globals.css) carries chrome, prose, and headings at 400-600.
 *            Geist stays loaded only as the next fallback in the chain.
 *   mono     every measured quantity, tabular, because a measurement is an
 *            instrument reading and a column of them has to line up. Geist
 *            Mono is this system's code face.
 *   display  the condensed wordmark-style headings (.font-condensed) keep
 *            Geist Display. */
const geist = Geist({
  variable: "--font-geist",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const display = Geist({
  variable: "--font-geist-display",
  subsets: ["latin"],
  weight: ["600"],
});

const mono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "futtraffic",
  description: "Per-player physical data from one fixed camera.",
};

/* `viewportFit: "cover"` is what makes env(safe-area-inset-*) report real
 * numbers — without it iOS letterboxes the page and the bottom tab bar sits
 * above a dead strip instead of under the home indicator.
 *
 * Deliberately NOT setting maximumScale/userScalable: this is read outdoors on
 * a phone and pinch-zoom is the one accessibility affordance that matters most
 * in that setting. */
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  colorScheme: "dark",
  themeColor: "#121212",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // Dark canvas by default: the surface ladder in globals.css is dark in
    // :root, and the `dark` class keeps the shadcn `dark:` custom variant
    // aligned. A `.light` class on <html> would flip to the light ladder.
    <html
      lang="en"
      className={`dark ${geist.variable} ${display.variable} ${mono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="bg-background text-foreground min-h-full">
        <TooltipProvider delayDuration={300}>
          {/* AppShell reads ?m= through useSearchParams, which must sit inside
              a Suspense boundary or every route opts into dynamic rendering. */}
          <Suspense fallback={null}>
            <AppShell>{children}</AppShell>
          </Suspense>
        </TooltipProvider>
        <Toaster position="bottom-right" />
      </body>
    </html>
  );
}
