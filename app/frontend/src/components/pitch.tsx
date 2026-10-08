import { cn } from "@/lib/utils";

/** The pitch, as ~20 lines of SVG on an L x W viewBox. Ported from web/pitch.js.
 *
 *  Children are drawn in PITCH METRES, so a caller plots straight from
 *  avg_position. The one thing to remember: match.json's y runs from the far
 *  touchline up, and SVG's y runs down, so use `flipY` for anything you place.
 */
export function Pitch({
  length = 105,
  width = 68,
  className,
  children,
}: {
  length?: number;
  width?: number;
  className?: string;
  children?: React.ReactNode;
}) {
  const cy = width / 2;
  const line = "oklch(1 0 0 / 22%)";
  const box = (x: number, d: number, h: number) => (
    <rect x={x} y={cy - h / 2} width={d} height={h} />
  );
  return (
    <svg
      viewBox={`0 0 ${length} ${width}`}
      role="img"
      aria-label="Pitch diagram, attacking left to right"
      className={cn("bg-card h-auto w-full rounded-[var(--radius-md)]", className)}
    >
      <g fill="none" stroke={line} strokeWidth={0.3}>
        <rect x={0.15} y={0.15} width={length - 0.3} height={width - 0.3} />
        <line x1={length / 2} y1={0} x2={length / 2} y2={width} />
        <circle cx={length / 2} cy={cy} r={9.15} />
        {box(0, 16.5, 40.32)}
        {box(length - 16.5, 16.5, 40.32)}
        {box(0, 5.5, 18.32)}
        {box(length - 5.5, 5.5, 18.32)}
      </g>
      <g fill={line} stroke="none">
        <circle cx={length / 2} cy={cy} r={0.4} />
        <circle cx={11} cy={cy} r={0.4} />
        <circle cx={length - 11} cy={cy} r={0.4} />
      </g>
      {children}
    </svg>
  );
}

export const flipY = (y: number, width: number) => width - y;

/** Sampled positions as a soft cloud: one faint dot per sample under a blur.
 *  Occupancy does not integrate error the way distance does, so where a player
 *  spent time is more robust than how far he ran. */
export function Occupancy({
  points,
  width,
  color,
  id,
}: {
  points: [number, number][];
  width: number;
  color: string;
  id: string;
}) {
  // Never composite 13k circles: thin first, evenly, so the shape is kept.
  const step = Math.max(1, points.length / 1500);
  const thinned = Array.from({ length: Math.min(points.length, 1500) }, (_, i) =>
    points[Math.floor(i * step)],
  );
  if (!thinned.length) return null;
  return (
    <g filter={`url(#blur-${id})`}>
      <defs>
        <filter id={`blur-${id}`}>
          <feGaussianBlur stdDeviation={1.2} />
        </filter>
      </defs>
      {thinned.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={flipY(y, width)}
          r={2.4}
          fill={color}
          fillOpacity={0.07}
        />
      ))}
    </g>
  );
}
