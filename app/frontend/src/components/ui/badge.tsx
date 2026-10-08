import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "group/badge inline-flex h-5 w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-[var(--radius-pill)] border border-transparent px-2 py-0.5 text-[11px] font-medium tracking-[0.2px] whitespace-nowrap transition-all focus-visible:ring-2 focus-visible:ring-[var(--accent)] has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 aria-invalid:border-destructive aria-invalid:ring-destructive/20 [&>svg]:pointer-events-none [&>svg]:size-3!",
  {
    variants: {
      variant: {
        /* Filled badge: monochrome primary (the inverted action language). */
        default: "bg-primary text-primary-foreground [a]:hover:opacity-90",
        secondary:
          "bg-[var(--card)] text-[var(--ink-soft)] [a]:hover:bg-[var(--card-hover)]",
        destructive:
          "bg-[color-mix(in_srgb,var(--danger)_12%,transparent)] text-[var(--danger)] [a]:hover:bg-[color-mix(in_srgb,var(--danger)_20%,transparent)]",
        outline:
          "border-[var(--hairline-strong)] text-foreground [a]:hover:bg-[var(--card)]",
        ghost:
          "hover:bg-[var(--card)] hover:text-[var(--ink)]",
        link: "text-primary underline-offset-4 hover:underline",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
