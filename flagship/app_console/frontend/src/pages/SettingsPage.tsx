import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { clearAuth, UserRole } from "../lib/auth";
import { useToast } from "../components/Toast";
import { ensureAudioReady, playCloseSound, playOpenSound } from "../lib/sound";
import { TradeAlertsState } from "../lib/trade_alerts";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";

type UserOut = { username: string; role: string };

type DailyOpsStatus = {
  status?: string;
  message?: string;
  progress?: number;
  updated_at?: string;
  pid?: number;
  log_file?: string;
};

type ModelTrainStatus = {
  status?: string;
  message?: string;
  progress?: number;
  updated_at?: string;
  pid?: number;
  log_file?: string;
};

function inferSubdomainUrl(sub: string): string {
  const proto = window.location.protocol || "https:";
  const host = window.location.hostname || "";
  if (host.startsWith("app.")) {
    return `${proto}//${sub}.${host.slice(4)}`;
  }
  return `${proto}//${host}`;
}

export function SettingsPage(props: {
  isAdmin: boolean;
  role: UserRole;
  alerts: TradeAlertsState;
  setAlerts: (next: TradeAlertsState | ((prev: TradeAlertsState) => TradeAlertsState)) => void;
}) {
  const { isAdmin, alerts, setAlerts } = props;
  const toast = useToast();

  const [notifPerm, setNotifPerm] = useState<string>(() => (typeof Notification === "undefined" ? "unsupported" : Notification.permission));

  const [dailyOps, setDailyOps] = useState<DailyOpsStatus | null>(null);
  const [modelTrain, setModelTrain] = useState<ModelTrainStatus | null>(null);
  const [users, setUsers] = useState<UserOut[]>([]);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newRole, setNewRole] = useState<"admin" | "user">("user");

  useEffect(() => {
    if (typeof Notification === "undefined") return;
    setNotifPerm(Notification.permission);
  }, []);

  const refreshUsers = useCallback(async () => {
    if (!isAdmin) return;
    try {
      const data = await fetchJson<UserOut[]>("/api/users", { method: "GET" });
      setUsers(data);
    } catch (e: any) {
      toast.push("Users", String(e?.message || e), "bad");
    }
  }, [isAdmin, toast]);

  useEffect(() => {
    refreshUsers();
  }, [refreshUsers]);

  const loadDailyOpsStatus = useCallback(async () => {
    try {
      const st = await fetchJson<DailyOpsStatus>("/api/daily_ops/status", { method: "GET" });
      setDailyOps(st);
    } catch (_) {
      // ignore transient
    }
  }, []);

  useEffect(() => {
    void loadDailyOpsStatus();
  }, [loadDailyOpsStatus]);

  const loadModelTrainStatus = useCallback(async () => {
    if (!isAdmin) return;
    try {
      const st = await fetchJson<ModelTrainStatus>("/api/model/train/status", { method: "GET" });
      setModelTrain(st);
    } catch (_) {
      // ignore transient
    }
  }, [isAdmin]);

  useEffect(() => {
    void loadModelTrainStatus();
  }, [loadModelTrainStatus]);

  const runDailyOps = useCallback(async () => {
    if (!isAdmin) return;
    try {
      toast.push("Daily Ops", "Starting daily cycle...", "info", 3000);
      await fetchJson("/api/daily_ops/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: "v7" }),
      });
      toast.push("Daily Ops", "Started", "good", 4000);
      await loadDailyOpsStatus();
    } catch (e: any) {
      toast.push("Daily Ops", String(e?.message || e), "bad", 8000);
    }
  }, [isAdmin, toast, loadDailyOpsStatus]);

  const runModelTrain = useCallback(async () => {
    if (!isAdmin) return;
    try {
      toast.push("Model Training", "Starting model training...", "info", 3000);
      await fetchJson("/api/model/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: "v7" }),
      });
      toast.push("Model Training", "Started", "good", 4000);
      await loadModelTrainStatus();
    } catch (e: any) {
      toast.push("Model Training", String(e?.message || e), "bad", 8000);
    }
  }, [isAdmin, toast, loadModelTrainStatus]);

  const dailyOpsStatus = useMemo(() => {
    return String(dailyOps?.status || "idle");
  }, [dailyOps]);

  const dailyOpsBadgeVariant = useMemo(() => {
    if (dailyOpsStatus === "running") return "info" as const;
    if (dailyOpsStatus === "completed") return "good" as const;
    if (dailyOpsStatus === "failed") return "bad" as const;
    return "outline" as const;
  }, [dailyOpsStatus]);

  const dailyOpsRunning = dailyOpsStatus === "running";

  const modelTrainStatus = useMemo(() => {
    return String(modelTrain?.status || "idle");
  }, [modelTrain]);

  const modelTrainBadgeVariant = useMemo(() => {
    if (modelTrainStatus === "running") return "info" as const;
    if (modelTrainStatus === "completed") return "good" as const;
    if (modelTrainStatus === "failed") return "bad" as const;
    return "outline" as const;
  }, [modelTrainStatus]);

  const modelTrainRunning = modelTrainStatus === "running";

  const enableAlerts = useCallback(async () => {
    setAlerts((prev) => ({ ...prev, enabled: true }));
    if (typeof Notification !== "undefined" && Notification.permission === "default") {
      try {
        const p = await Notification.requestPermission();
        setNotifPerm(p);
      } catch (_) {}
    }
    toast.push("Trade Alerts", "enabled", "good");
  }, [setAlerts, toast]);

  const disableAlerts = useCallback(() => {
    setAlerts((prev) => ({ ...prev, enabled: false }));
    toast.push("Trade Alerts", "disabled", "info");
  }, [setAlerts, toast]);

  const toggleMute = useCallback(() => {
    setAlerts((prev) => ({ ...prev, muted: !prev.muted }));
  }, [setAlerts]);

  const testSound = useCallback(async () => {
    try {
      await ensureAudioReady();
      await playOpenSound();
      window.setTimeout(() => void playCloseSound(), 220);
      toast.push("Sound", "played (open/close)", "good", 3000);
    } catch (e: any) {
      toast.push("Sound", String(e?.message || e), "bad");
    }
  }, [toast]);

  const createUser = useCallback(async () => {
    if (!isAdmin) return;
    try {
      await fetchJson("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: newUsername, password: newPassword, role: newRole }),
      });
      setNewUsername("");
      setNewPassword("");
      setNewRole("user");
      toast.push("Users", "created", "good", 3000);
      await refreshUsers();
    } catch (e: any) {
      toast.push("Users", String(e?.message || e), "bad");
    }
  }, [isAdmin, newUsername, newPassword, newRole, toast, refreshUsers]);

  const deleteUser = useCallback(
    async (username: string) => {
      if (!isAdmin) return;
      try {
        await fetchJson(`/api/users/${encodeURIComponent(username)}`, { method: "DELETE" });
        toast.push("Users", `deleted ${username}`, "good", 3000);
        await refreshUsers();
      } catch (e: any) {
        toast.push("Users", String(e?.message || e), "bad");
      }
    },
    [isAdmin, toast, refreshUsers]
  );

  const logout = useCallback(() => {
    clearAuth();
    window.location.href = "/login.html";
  }, []);

  const alertStatus = useMemo(() => {
    return `enabled=${alerts.enabled ? "1" : "0"} muted=${alerts.muted ? "1" : "0"} notifications=${notifPerm}`;
  }, [alerts.enabled, alerts.muted, notifPerm]);

  const grafana = useMemo(() => inferSubdomainUrl("monitor"), []);
  const prom = useMemo(() => inferSubdomainUrl("metrics"), []);
  const db = useMemo(() => inferSubdomainUrl("db"), []);

  return (
    <div className="grid grid-cols-12 gap-6">
      <Card className="col-span-12">
        <CardHeader>
          <div>
            <CardTitle>Daily Ops</CardTitle>
            <CardDescription>Manual trigger for today’s daily cycle (admin only)</CardDescription>
          </div>
          <Badge variant={dailyOpsBadgeVariant}>
            {dailyOpsStatus}
            {dailyOps?.updated_at ? ` · ${dailyOps.updated_at}` : ""}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => void runDailyOps()} disabled={dailyOpsRunning}>
              {dailyOpsRunning ? "Running..." : "Run Daily Ops"}
            </Button>
            <Button variant="outline" onClick={() => void loadDailyOpsStatus()}>
              Refresh status
            </Button>
            {dailyOps?.log_file ? (
              <a className="text-sm font-medium text-primary hover:underline" href={`/data/${dailyOps.log_file}`} target="_blank" rel="noopener noreferrer">
                Open log
              </a>
            ) : null}
          </div>
          {dailyOps?.message ? <div className="mt-3 text-sm text-muted-foreground">{dailyOps.message}</div> : null}
        </CardContent>
      </Card>

      <Card className="col-span-12">
        <CardHeader>
          <div>
            <CardTitle>Model Training</CardTitle>
            <CardDescription>Manual model retrain only (admin only)</CardDescription>
          </div>
          <Badge variant={modelTrainBadgeVariant}>
            {modelTrainStatus}
            {modelTrain?.updated_at ? ` · ${modelTrain.updated_at}` : ""}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => void runModelTrain()} disabled={modelTrainRunning}>
              {modelTrainRunning ? "Running..." : "Train Model"}
            </Button>
            <Button variant="outline" onClick={() => void loadModelTrainStatus()}>
              Refresh status
            </Button>
            {modelTrain?.log_file ? (
              <a className="text-sm font-medium text-primary hover:underline" href={`/data/${modelTrain.log_file}`} target="_blank" rel="noopener noreferrer">
                Open log
              </a>
            ) : null}
            <a className="text-sm font-medium text-primary hover:underline" href="/data/model_metrics.json" target="_blank" rel="noopener noreferrer">
              model_metrics.json
            </a>
          </div>
          {modelTrain?.message ? <div className="mt-3 text-sm text-muted-foreground">{modelTrain.message}</div> : null}
        </CardContent>
      </Card>

      <Card className="col-span-12">
        <CardHeader>
          <div>
            <CardTitle>Monitoring</CardTitle>
            <CardDescription>Grafana / Prometheus / PGAdmin quick links (admin only)</CardDescription>
          </div>
          <Badge variant="outline">links</Badge>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <a
              className="inline-flex h-9 items-center rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              href={grafana}
              target="_blank"
              rel="noopener noreferrer"
            >
              Grafana
            </a>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href={prom}
              target="_blank"
              rel="noopener noreferrer"
            >
              Prometheus
            </a>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href={db}
              target="_blank"
              rel="noopener noreferrer"
            >
              PGAdmin
            </a>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href="/data/market_status.json"
              target="_blank"
              rel="noopener noreferrer"
            >
              market_status.json
            </a>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href="/data/orders.json"
              target="_blank"
              rel="noopener noreferrer"
            >
              orders.json
            </a>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href="/data/model_metrics.json"
              target="_blank"
              rel="noopener noreferrer"
            >
              model_metrics.json
            </a>
          </div>

          <div className="mt-4 text-xs text-muted-foreground">
            If you are not under <code className="font-mono">app.*</code> domain (local/dev), links fall back to current host.
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-12 lg:col-span-6">
        <CardHeader>
          <div>
            <CardTitle>Trade Alerts</CardTitle>
            <CardDescription>Browser notifications + sound on filled orders</CardDescription>
          </div>
          <Badge variant="outline">{alertStatus}</Badge>
        </CardHeader>

        <CardContent>
          <div className="flex flex-wrap gap-2">
            {!alerts.enabled ? (
              <Button onClick={() => void enableAlerts()}>Enable</Button>
            ) : (
              <Button variant="outline" onClick={disableAlerts}>
                Disable
              </Button>
            )}

            <Button variant="outline" onClick={toggleMute} disabled={!alerts.enabled}>
              {alerts.muted ? "Unmute" : "Mute"}
            </Button>

            <Button variant="outline" onClick={() => void testSound()}>
              Test sound
            </Button>
          </div>

          <div className="mt-4 text-xs text-muted-foreground">
            Alerts are triggered by new filled orders observed in <code className="font-mono">/data/orders.json</code>.
            <br />
            Sound playback may require a user gesture (use “Test sound” once after login).
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-12 lg:col-span-6">
        <CardHeader>
          <div>
            <CardTitle>Session</CardTitle>
            <CardDescription>Authentication & logout</CardDescription>
          </div>
          <Badge variant="outline">auth</Badge>
        </CardHeader>
        <CardContent>
          <Button variant="outline" onClick={logout}>
            Logout (to login.html)
          </Button>
        </CardContent>
      </Card>

      {isAdmin && (
        <Card className="col-span-12">
          <CardHeader>
            <div>
              <CardTitle>User Management</CardTitle>
              <CardDescription>Create/delete users (admin only)</CardDescription>
            </div>
            <Badge variant="outline">admin</Badge>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <input
                placeholder="username"
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                className="h-9 w-[220px] rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <input
                placeholder="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                type="password"
                className="h-9 w-[220px] rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <select
                value={newRole}
                onChange={(e) => setNewRole(e.target.value as any)}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
              <Button onClick={() => void createUser()} disabled={!newUsername || !newPassword}>
                Create
              </Button>
              <Button variant="outline" onClick={() => void refreshUsers()}>
                Refresh
              </Button>
            </div>

            <div className="mt-4 overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>username</TableHead>
                    <TableHead>role</TableHead>
                    <TableHead className="text-right"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {users.map((u) => (
                    <TableRow key={u.username}>
                      <TableCell className="font-medium">{u.username}</TableCell>
                      <TableCell className="text-muted-foreground">{u.role}</TableCell>
                      <TableCell className="text-right">
                        <Button variant="outline" size="sm" onClick={() => void deleteUser(u.username)}>
                          Delete
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {!users.length && (
                    <TableRow>
                      <TableCell colSpan={3} className="text-sm text-muted-foreground">
                        No users
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
