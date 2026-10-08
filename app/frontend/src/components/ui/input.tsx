import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-8 w-full min-w-0 rounded-[var(--radius-md)] border border-[var(--border-input)] bg-[var(--input-bg)] px-3 py-1 text-[13px] text-[var(--ink)] transition-colors duration-[var(--dur-fast)] outline-none file:inline-flex file:h-6 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-[var(--mute)] focus-visible:border-[var(--accent)] focus-visible:ring-3 focus-visible:ring-[var(--accent-soft)] disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-[var(--panel-2)] disabled:text-[var(--stone)] disabled:opacity-60 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20",
        className
      )}
      {...props}
    />
  )
}

export { Input }
