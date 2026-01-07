import React, { useCallback, useMemo, useState } from "react";
import { cn } from "../lib/cn";

export type ToastLevel = "info" | "good" | "bad";

export type ToastItem = {
  id: string;
  title: string;
  message: string;
  level: ToastLevel;
};

export type ToastApi = {
  push: (title: string, message: string, level?: ToastLevel, ttlMs?: number) => void;
};

const ToastContext = React.createContext<ToastApi | null>(null);

function _id(): string {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function useToast(): ToastApi {
  const ctx = React.useContext(ToastContext);
  if (!ctx) {
    // Fallback no-op to avoid crashing if used outside provider.
    return { push: () => {} };
  }
  return ctx;
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);

  const push = useCallback((title: string, message: string, level: ToastLevel = "info", ttlMs: number = 8000) => {
    const id = _id();
    const toast: ToastItem = { id, title, message, level };
    setItems((prev) => [...prev, toast]);
    window.setTimeout(() => {
      setItems((prev) => prev.filter((t) => t.id !== id));
    }, Math.max(1500, ttlMs));
  }, []);

  const api = useMemo<ToastApi>(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <ToastContainer items={items} />
    </ToastContext.Provider>
  );
}

function ToastContainer({ items }: { items: ToastItem[] }) {
  return (
    <div className="fixed bottom-6 right-6 z-50 flex w-[360px] max-w-[92vw] flex-col gap-3" aria-live="polite" aria-relevant="additions">
      {items.map((t) => (
        <div
          key={t.id}
          className={cn(
            "rounded-lg bg-card p-4 shadow-card-soft ring-1 ring-border/60",
            t.level === "good" && "ring-emerald-200/70",
            t.level === "bad" && "ring-rose-200/80",
            t.level === "info" && "ring-blue-200/70"
          )}
        >
          <div className="text-sm font-semibold">{t.title}</div>
          <div className="mt-1 text-sm text-muted-foreground">{t.message}</div>
        </div>
      ))}
    </div>
  );
}

