import React from "react";
import { RouteKey, ROUTES, setRoute } from "../lib/router";
import { UserRole } from "../lib/auth";
import { cn } from "../lib/cn";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { TimeZones } from "./TimeZones";

const LABEL: Record<RouteKey, string> = {
  paper: "Paper",
  backtest: "Backtest",
  reports: "Reports",
  settings: "Settings",
};

export function Layout(props: {
  route: RouteKey;
  isAdmin: boolean;
  role: UserRole;
  onLogout: () => void;
  children: React.ReactNode;
}) {
  const { route, isAdmin, role, onLogout, children } = props;
  const tabs = ROUTES.filter((r) => (isAdmin ? true : r === "paper" || r === "reports"));

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-[1400px] px-6 py-6">
        <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="text-lg font-semibold tracking-tight">Flagship Ops Console</div>
              <Badge variant="outline">role={role}</Badge>
            </div>
            <div className="mt-1 text-sm text-muted-foreground">Paper trading dashboard</div>
          </div>
          <div className="flex items-center gap-2">
            <TimeZones />
            <Button variant="outline" onClick={onLogout}>
              Logout
            </Button>
          </div>
        </header>

        <nav className="mt-5 flex flex-wrap gap-2">
          {tabs.map((t) => (
            <Button
              key={t}
              variant="ghost"
              className={cn(
                "h-9 rounded-md px-3 text-sm font-medium",
                t === route
                  ? "bg-white shadow-sm ring-1 ring-border/70 hover:bg-white"
                  : "text-muted-foreground hover:bg-white/70 hover:text-foreground"
              )}
              onClick={() => setRoute(t)}
            >
              {LABEL[t]}
            </Button>
          ))}
        </nav>

        <main className="mt-6">{children}</main>
      </div>
    </div>
  );
}

