import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { useInterval } from "../lib/interval";
import { useToast } from "../components/Toast";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";

type BacktestStatus = {
  status?: string;
  message?: string;
  progress?: number;
  updated_at?: string;
  pid?: number;
  log_file?: string;
};

function todayIso(): string {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export function BacktestPage() {
  const toast = useToast();

  const [startDate, setStartDate] = useState<string>(() => {
    const d = new Date();
    d.setDate(d.getDate() - 30);
    return d.toISOString().slice(0, 10);
  });
  const [endDate, setEndDate] = useState<string>(() => todayIso());

  const [status, setStatus] = useState<BacktestStatus | null>(null);
  const isRunning = (status?.status || "") === "running";

  const refreshStatus = useCallback(async () => {
    try {
      const s = await fetchJson<BacktestStatus>("/api/backtest/status", { method: "GET" });
      setStatus(s);
    } catch (e: any) {
      toast.push("Backtest", String(e?.message || e), "bad");
    }
  }, [toast]);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  useInterval(
    () => {
      refreshStatus();
    },
    isRunning ? 2000 : 15000
  );

  const runBacktest = useCallback(async () => {
    try {
      toast.push("Backtest", "Submitting backtest job...", "info", 2500);
      await fetchJson("/api/backtest/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_date: startDate, end_date: endDate }),
      });
      await refreshStatus();
      toast.push("Backtest", "Started", "good", 4000);
    } catch (e: any) {
      toast.push("Backtest", String(e?.message || e), "bad");
    }
  }, [toast, startDate, endDate, refreshStatus]);

  const statusBadge = useMemo(() => {
    const s = status?.status || "unknown";
    return `${s}${status?.updated_at ? ` @ ${status.updated_at}` : ""}`;
  }, [status]);

  return (
    <div className="grid grid-cols-12 gap-6">
      <Card className="col-span-12">
        <CardHeader>
          <div>
            <CardTitle>Backtest Runner</CardTitle>
            <CardDescription>Trigger backtests and inspect status</CardDescription>
          </div>
          <Badge variant="outline">{statusBadge}</Badge>
        </CardHeader>

        <CardContent>
          <div className="flex flex-wrap items-center gap-2">
            <label className="text-sm text-muted-foreground">
              start&nbsp;
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="ml-2 h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </label>
            <label className="text-sm text-muted-foreground">
              end&nbsp;
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="ml-2 h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </label>
            <Button onClick={runBacktest} disabled={isRunning}>
              Run
            </Button>
            <Button variant="outline" onClick={refreshStatus}>
              Refresh
            </Button>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href="/data/report_latest.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Report (latest)
            </a>
          </div>

          <div className="mt-4 grid gap-1 text-sm text-muted-foreground">
            <div>status: {status?.status || "-"}</div>
            <div>message: {status?.message || "-"}</div>
            <div>progress: {typeof status?.progress === "number" ? `${Math.round(status.progress * 100)}%` : "-"}</div>
            <div>pid: {status?.pid ?? "-"}</div>
            <div>log_file: {status?.log_file ?? "-"}</div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

