import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "../../lib/cn";

const badgeVariants = cva(
  "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium tracking-tight",
  {
    variants: {
      variant: {
        default: "bg-muted text-muted-foreground",
        outline: "border border-border bg-background text-muted-foreground",
        good: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100",
        bad: "bg-rose-50 text-rose-700 ring-1 ring-rose-100",
        info: "bg-blue-50 text-blue-700 ring-1 ring-blue-100",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}


