import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "flex field-sizing-content min-h-16 w-full rounded-[var(--radius-md)] border border-[var(--border-input)] bg-[var(--input-bg)] px-3 py-2 text-[13px] text-[var(--ink)] transition-colors duration-[var(--dur-fast)] outline-none placeholder:text-[var(--mute)] focus-visible:border-[var(--accent)] focus-visible:ring-3 focus-visible:ring-[var(--accent-soft)] disabled:cursor-not-allowed disabled:bg-[var(--panel-2)] disabled:text-[var(--stone)] disabled:opacity-60 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
