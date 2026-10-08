"use client";

import { cn } from "@/lib/utils";

/** The furniture every `/` panel is built from.
 *
 *  Extracted from slash-panels.tsx when the panels split in two: `/table` and
 *  `/visual` are the same flow with a different last step, and two files each
 *  keeping their own Chip is how two halves of one menu end up looking like
 *  two different products.
 */

export function Panel({
  title,
  subtitle,
  note,
  action,
  children,
}: {
  /** A string is drawn as the header. A node replaces it — which is how the
   *  stepped panels put their breadcrumb trail there instead of printing the
   *  panel's name once as a title and again as the trail's first crumb. */
  title: React.ReactNode;
  subtitle?: string;
  /** Why there is nothing to add, when there is nothing to add. */
  note?: string;
  action?: { label: string; run: () => void; hint?: string };
  children?: React.ReactNode;
}) {
  return (
    <div className="flex w-[24rem] max-w-[80vw] flex-col gap-2 p-2.5">
      <div className="flex items-baseline gap-2">
        {typeof title === "string" ? (
          <span className="text-[11px] font-medium text-[var(--ink)]">{title}</span>
        ) : (
          title
        )}
        {subtitle && <span className="text-[10.5px] text-[var(--mute)]">{subtitle}</span>}
      </div>
      {children}
      {note && <Refusal>{note}</Refusal>}
      {action && (
        <div className="flex flex-col gap-1">
          <button
            type="button"
            onClick={action.run}
            className="h-7 rounded-[var(--radius-sm)] bg-[var(--accent)] text-[11.5px] font-medium text-[var(--accent-contrast,#fff)] transition-opacity hover:opacity-90"
          >
            {action.label}
          </button>
          {action.hint && (
            <span className="text-center text-[10px] text-[var(--mute)]">{action.hint}</span>
          )}
        </div>
      )}
    </div>
  );
}

export function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="w-14 shrink-0 text-[9.5px] leading-tight tracking-wide text-[var(--mute)] uppercase">
        {label}
      </span>
      {children}
    </div>
  );
}

export function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { id: string; label: string }[];
}) {
  return (
    <Row label={label}>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-6 min-w-0 flex-1 cursor-pointer rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1 text-[11px] text-[var(--ink)] outline-none hover:border-[var(--hairline-strong)]"
      >
        {options.map((o) => (
          <option key={o.id} value={o.id} className="bg-[var(--panel)] text-[var(--ink)]">
            {o.label}
          </option>
        ))}
      </select>
    </Row>
  );
}

export const Count = ({ n }: { n: number }) => (
  <span className="font-mono text-[10px] text-[var(--mute)]">{n}</span>
);

export const Refusal = ({ children }: { children: React.ReactNode }) => (
  <p className="text-[11px] leading-relaxed text-[var(--mute)]">{children}</p>
);

export function Chip({
  on,
  disabled,
  title,
  onClick,
  children,
}: {
  on?: boolean;
  disabled?: boolean;
  title?: string;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex h-5.5 items-center gap-1 rounded-[var(--radius-pill)] border px-1.5 py-0.5 text-[10.5px] whitespace-nowrap transition-colors",
        disabled
          ? "cursor-not-allowed border-[var(--hairline)] text-[var(--mute)] opacity-40"
          : on
            ? "border-[var(--hairline-strong)] bg-[var(--card)] text-[var(--ink)]"
            : "border-[var(--hairline)] text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
      )}
    >
      {children}
    </button>
  );
}

/** A search box over a long list. The team feed alone carries sixty metrics;
 *  chips for that many is a wall, and a plain `<select>` cannot multi-select. */
export function Find({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label={placeholder}
      className="h-7 w-full rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--input-bg)] px-2 text-[11.5px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
    />
  );
}
