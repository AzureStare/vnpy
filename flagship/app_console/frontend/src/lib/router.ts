export type RouteKey = "paper" | "reports" | "backtest" | "settings";

export const ROUTES: RouteKey[] = ["paper", "reports", "backtest", "settings"];

export function parseRouteFromHash(hash: string, opts: { isAdmin: boolean }): RouteKey {
  const { isAdmin } = opts;
  const m = (hash || "").match(/^#\/(\w+)/);
  const raw = (m?.[1] || "").toLowerCase();
  const route = (ROUTES as string[]).includes(raw) ? (raw as RouteKey) : "paper";
  if (route === "backtest" && !isAdmin) return "paper";
  if (route === "settings" && !isAdmin) return "paper";
  return route;
}

export function setRoute(route: RouteKey): void {
  window.location.hash = `#/${route}`;
}
