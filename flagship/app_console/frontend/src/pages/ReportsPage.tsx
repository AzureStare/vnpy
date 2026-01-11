import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { useToast } from "../components/Toast";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { getRole, isAdminRole } from "../lib/auth";

type TradeRecapIndex = {
  generated_at?: string;
  latest?: { file: string; trade_date?: string };
  recaps?: Array<{ file: string; trade_date?: string }>;
};

type TradeRecapJobStatus = {
  status?: string;
  message?: string;
  progress?: number;
  updated_at?: string;
  pid?: number;
  log_file?: string;
  trade_date?: string;
};

const REPORT_THEME_OVERRIDE_STYLE_ID = "reportThemeOverride";
const REPORT_THEME_OVERRIDE_CSS = `
:root{
  --bg: #f8fafc !important;
  --panel: #ffffff !important;
  --card: #ffffff !important;
  --border: #e2e8f0 !important;
  --text: #0f172a !important;
  --muted: #64748b !important;
  --muted2: #94a3b8 !important;
  --primary: #2563eb !important;
  --ok: #16a34a !important;
  --bad: #dc2626 !important;
}
html,body{background: var(--bg) !important; color: var(--text) !important;}
table{border-color: var(--border) !important;}
th{background: #f1f5f9 !important; color: var(--muted) !important;}
td{border-color: var(--border) !important;}
`;

export function ReportsPage() {
  const toast = useToast();

  const role = useMemo(() => getRole(), []);
  const isAdmin = useMemo(() => isAdminRole(role), [role]);

  const [idx, setIdx] = useState<TradeRecapIndex | null>(null);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [job, setJob] = useState<TradeRecapJobStatus | null>(null);

  const loadIndex = useCallback(async () => {
    try {
      const data = await fetchJson<TradeRecapIndex>("/data/trade_recap_index.json");
      setIdx(data);
      const nextFile = data?.latest?.file || null;
      setCurrentFile((prev) => prev || nextFile);
    } catch (e: any) {
      toast.push("Reports", `trade_recap_index.json not ready: ${String(e?.message || e)}`, "bad", 6000);
    }
  }, [toast]);

  const loadJobStatus = useCallback(async () => {
    if (!isAdmin) return;
    try {
      const data = await fetchJson<TradeRecapJobStatus>("/api/reports/trade_recap/status", { method: "GET" });
      setJob(data);
    } catch (_) {
      // ignore transient (e.g., not deployed yet)
    }
  }, [isAdmin]);

  const regenerateTradeRecap = useCallback(
    async (trade_date: string | undefined) => {
      if (!isAdmin) return;
      const d = String(trade_date || "").trim();
      if (!d) return;
      try {
        toast.push("Trade Recap", `Regenerating ${d}...`, "info", 3000);
        await fetchJson("/api/reports/trade_recap/regenerate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trade_date: d, strategy: "v7" }),
        });
        toast.push("Trade Recap", "Started", "good", 4000);
        await loadJobStatus();
      } catch (e: any) {
        toast.push("Trade Recap", String(e?.message || e), "bad", 8000);
      }
    },
    [isAdmin, toast, loadJobStatus]
  );

  useEffect(() => {
    loadIndex();
  }, [loadIndex]);

  useEffect(() => {
    void loadJobStatus();
  }, [loadJobStatus]);

  const iframeUrl = useMemo(() => {
    if (currentFile) return `/data/${currentFile}`;
    return "/data/trade_recap_latest.html";
  }, [currentFile]);

  const injectThemeIntoIframe = useCallback(() => {
    try {
      const iframe = document.querySelector("iframe[title='trade-recap-preview']") as HTMLIFrameElement | null;
      const doc = iframe?.contentDocument;
      if (!doc) return;

      const head = doc.head || doc.getElementsByTagName("head")[0];
      if (!head) return;

      let style = doc.getElementById(REPORT_THEME_OVERRIDE_STYLE_ID) as HTMLStyleElement | null;
      if (!style) {
        style = doc.createElement("style");
        style.id = REPORT_THEME_OVERRIDE_STYLE_ID;
        head.appendChild(style);
      }
      style.textContent = REPORT_THEME_OVERRIDE_CSS;
    } catch (_) {
      // cross-origin or transient load issues
    }
  }, []);

  const openLatest = useCallback(() => {
    setCurrentFile(idx?.latest?.file || null);
  }, [idx]);

  const jobStatus = useMemo(() => String(job?.status || "idle"), [job]);
  const jobBadgeVariant = useMemo(() => {
    if (jobStatus === "running") return "info" as const;
    if (jobStatus === "completed") return "good" as const;
    if (jobStatus === "failed") return "bad" as const;
    return "outline" as const;
  }, [jobStatus]);

  return (
    <div className="grid grid-cols-12 gap-6">
      <Card className="col-span-12 lg:col-span-5">
        <CardHeader>
          <div>
            <CardTitle>Reports</CardTitle>
            <CardDescription>Daily trade recap & backtest reports</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{idx?.generated_at || "-"}</Badge>
            {isAdmin ? <Badge variant={jobBadgeVariant}>{jobStatus}</Badge> : null}
          </div>
        </CardHeader>

        <CardContent>
          <div className="flex flex-wrap gap-2">
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href="/data/report_latest.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Backtest report
            </a>
            <a
              className="inline-flex h-9 items-center rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted/50"
              href="/data/trade_recap_latest.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Trade recap
            </a>
            <Button variant="outline" onClick={loadIndex}>
              Refresh index
            </Button>
            <Button onClick={openLatest} disabled={!idx?.latest?.file}>
              Open latest
            </Button>
          </div>

          {isAdmin ? (
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Button variant="outline" size="sm" onClick={() => void loadJobStatus()}>
                Refresh job
              </Button>
              {job?.log_file ? (
                <a className="text-sm font-medium text-primary hover:underline" href={`/data/${job.log_file}`} target="_blank" rel="noopener noreferrer">
                  Open log
                </a>
              ) : null}
              {job?.message ? <span>{job.message}</span> : null}
            </div>
          ) : null}

          <div className="mt-4 text-xs text-muted-foreground">
            source: <code className="font-mono">/data/trade_recap_index.json</code>
          </div>

          <div className="mt-3 overflow-hidden rounded-lg ring-1 ring-border/60">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[120px]">trade_date</TableHead>
                  <TableHead>file</TableHead>
                  <TableHead className="w-[84px] text-right"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(idx?.recaps || []).slice(0, 30).map((r) => (
                  <TableRow key={r.file}>
                    <TableCell className="tabular-nums">{r.trade_date || "-"}</TableCell>
                    <TableCell className="font-mono text-xs">{r.file}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="outline" size="sm" onClick={() => setCurrentFile(r.file)}>
                          Open
                        </Button>
                        {isAdmin ? (
                          <Button variant="outline" size="sm" onClick={() => void regenerateTradeRecap(r.trade_date)} disabled={!r.trade_date}>
                            Regenerate
                          </Button>
                        ) : null}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {!(idx?.recaps || []).length && (
                  <TableRow>
                    <TableCell colSpan={3} className="text-sm text-muted-foreground">
                      No recap index yet
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-12 lg:col-span-7">
        <CardHeader>
          <div>
            <CardTitle>Preview</CardTitle>
            <CardDescription>Inline viewer</CardDescription>
          </div>
          <Badge variant="outline">{currentFile || "trade_recap_latest.html"}</Badge>
        </CardHeader>
        <CardContent className="pt-2">
          <iframe
            title="trade-recap-preview"
            src={iframeUrl}
            onLoad={injectThemeIntoIframe}
            className="h-[76vh] w-full rounded-lg bg-white ring-1 ring-border/60"
          />
        </CardContent>
      </Card>
    </div>
  );
}

