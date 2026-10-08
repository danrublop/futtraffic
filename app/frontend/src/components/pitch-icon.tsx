import { cn } from "@/lib/utils";

/** The arena mark as a lucide-shaped icon, so it can sit anywhere an icon
 *  component is expected. The SVG is drawn in solid black; `invert` turns it
 *  chalk-white on the dark panel.
 *
 *  Its own file because two places need it now: the sidebar's Matches row and
 *  the database browser's Matches group. One subject gets one symbol wherever
 *  it appears, and the alternative was importing the whole app shell to borrow
 *  an `<img>`.
 */
export function PitchIcon({ className }: { className?: string }) {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src="/soccer-arena.svg"
      alt=""
      aria-hidden
      className={cn("size-4 shrink-0 invert opacity-70 select-none", className)}
    />
  );
}
