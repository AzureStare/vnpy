export type RouteKey = "accounts" | "portfolio" | "reports" | "backtest" | "settings" | "monitor";

export const ROUTES: RouteKey[] = ["accounts", "portfolio", "reports", "backtest", "settings", "monitor"];

export function parseRouteFromHash(hash: string, opts: { isAdmin: boolean }): RouteKey {
  const { isAdmin } = opts;
  const m = (hash || "").match(/^#\/(\w+)/);
  const raw = (m?.[1] || "").toLowerCase();
  const fallback: RouteKey = isAdmin ? "accounts" : "portfolio";
  const route = (ROUTES as string[]).includes(raw) ? (raw as RouteKey) : fallback;
  if (route === "backtest" && !isAdmin) return "portfolio";
  if (route === "settings" && !isAdmin) return "portfolio";
  if (route === "monitor" && !isAdmin) return "portfolio";
  return route;
}

export function setRoute(route: RouteKey): void {
  window.location.hash = `#/${route}`;
}
