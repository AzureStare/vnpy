import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ToastProvider, useToast } from "./components/Toast";
import { Layout } from "./components/Layout";
import { clearAuth, getRole, getToken, isAdminRole, UserRole } from "./lib/auth";
import { parseRouteFromHash, RouteKey, setRoute } from "./lib/router";
import { useInterval } from "./lib/interval";
import { fetchJson } from "./lib/api";
import { playCloseSound, playOpenSound } from "./lib/sound";
import { loadTradeAlertsState, saveTradeAlertsState, TradeAlertsState } from "./lib/trade_alerts";

import { PortfolioPage } from "./pages/PortfolioPage";
import { AccountsPage } from "./pages/AccountsPage";
import { BacktestPage } from "./pages/BacktestPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { MonitorPage } from "./pages/MonitorPage";

type OrdersSnapshot = {
  generated_at?: string;
  orders?: Array<Record<string, any>>;
};

function AppInner() {
  const toast = useToast();

  const token = useMemo(() => getToken(), []);
  const role = useMemo<UserRole>(() => getRole(), []);
  const isAdmin = useMemo(() => isAdminRole(role), [role]);

  const [route, setRouteState] = useState<RouteKey>(() => parseRouteFromHash(window.location.hash, { isAdmin }));
  const [alerts, setAlerts] = useState<TradeAlertsState>(() => loadTradeAlertsState());

  // Persist trade alerts state
  useEffect(() => {
    saveTradeAlertsState(alerts);
  }, [alerts]);

  // Require auth token
  useEffect(() => {
    if (!token) {
      window.location.href = "/login.html";
    }
  }, [token]);

  // Hash routing
  useEffect(() => {
    const onHash = () => setRouteState(parseRouteFromHash(window.location.hash, { isAdmin }));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [isAdmin]);

  useEffect(() => {
    // Normalize initial route
    const parsed = parseRouteFromHash(window.location.hash, { isAdmin });
    if (!window.location.hash || parsed !== route) {
      setRoute(parsed);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAdmin]);

  const onLogout = useCallback(() => {
    clearAuth();
    window.location.href = "/login.html";
  }, []);

  // ---------------- Global Trade Alerts (filled only) ----------------
  const notifyFilledOrders = useCallback(
    async (data: OrdersSnapshot) => {
      if (!alerts.enabled || alerts.muted) return;
      if (!data || !Array.isArray(data.orders)) return;

      const filled = data.orders.filter(
        (o: any) => o && o.id && o.filled_at && (o.status === "filled" || o.status === "partially_filled")
      );

      // First run after enabling: baseline silently
      if (!alerts.initialized) {
        const seen: Record<string, string> = { ...(alerts.seen || {}) };
        filled.slice(0, 200).forEach((o: any) => {
          seen[String(o.id)] = String(o.filled_at);
        });
        setAlerts((prev) => ({ ...prev, initialized: true, seen }));
        return;
      }

      const seen = { ...(alerts.seen || {}) };
      const newOnes: any[] = [];
      for (const o of filled.slice(0, 200)) {
        const id = String(o.id);
        const filledAt = String(o.filled_at);
        if (!seen[id] || seen[id] !== filledAt) {
          newOnes.push(o);
          seen[id] = filledAt;
        }
      }

      if (!newOnes.length) return;

      // Cap memory
      const keys = Object.keys(seen);
      if (keys.length > 1500) {
        keys
          .sort((a, b) => (seen[a] || "").localeCompare(seen[b] || ""))
          .slice(0, keys.length - 1200)
          .forEach((k) => delete seen[k]);
      }

      setAlerts((prev) => ({ ...prev, seen }));

      for (const o of newOnes.slice(0, 5)) {
        const symbol = String(o.symbol || "");
        const side = String(o.side || "");
        const qty = String(o.qty || "");
        const px = o.filled_avg_price != null ? String(o.filled_avg_price) : "";
        const title = `${side.toUpperCase()} ${symbol}`.trim();
        const body = `filled qty=${qty}${px ? ` avg_px=${px}` : ""}`;
        toast.push("Filled", `${title} ${body}`, "good", 8000);

        try {
          if (typeof Notification !== "undefined" && Notification.permission === "granted") {
            new Notification(title, { body });
          }
        } catch (_) {}

        try {
          if (side.toLowerCase() === "buy") {
            await playOpenSound();
          } else if (side.toLowerCase() === "sell") {
            await playCloseSound();
          } else {
            await playOpenSound();
          }
        } catch (_) {}
      }
    },
    [alerts.enabled, alerts.muted, alerts.initialized, alerts.seen, toast]
  );

  const pollOrdersForAlerts = useCallback(async () => {
    if (!alerts.enabled) return;
    try {
      const data = await fetchJson<OrdersSnapshot>("/data/orders.json");
      await notifyFilledOrders(data);
    } catch (_) {
      // ignore transient
    }
  }, [alerts.enabled, notifyFilledOrders]);

  useInterval(() => {
    void pollOrdersForAlerts();
  }, alerts.enabled ? 10_000 : null);

  return (
    <Layout route={route} isAdmin={isAdmin} role={role} onLogout={onLogout}>
      {route === "accounts" && <AccountsPage />}
      {route === "portfolio" && <PortfolioPage />}
      {route === "backtest" && isAdmin && <BacktestPage />}
      {route === "reports" && <ReportsPage />}
      {route === "monitor" && isAdmin && <MonitorPage />}
      {route === "settings" && isAdmin && <SettingsPage isAdmin={isAdmin} role={role} alerts={alerts} setAlerts={setAlerts} />}
    </Layout>
  );
}

export function App() {
  return (
    <ToastProvider>
      <AppInner />
    </ToastProvider>
  );
}
