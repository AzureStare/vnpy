import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { useToast } from "../components/Toast";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";

type TradeRecapIndex = {
  generated_at?: string;
  latest?: { file: string; trade_date?: string };
  recaps?: Array<{ file: string; trade_date?: string }>;
};

export function ReportsPage() {
  const toast = useToast();

  const [idx, setIdx] = useState<TradeRecapIndex | null>(null);
  const [currentFile, setCurrentFile] = useState<string | null>(null);

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

  useEffect(() => {
    loadIndex();
  }, [loadIndex]);

  const iframeUrl = useMemo(() => {
    if (currentFile) return `/data/${currentFile}`;
    return "/data/trade_recap_latest.html";
  }, [currentFile]);

  const openLatest = useCallback(() => {
    setCurrentFile(idx?.latest?.file || null);
  }, [idx]);

  return (
    <div className="grid grid-cols-12 gap-6">
      <Card className="col-span-12 lg:col-span-5">
        <CardHeader>
          <div>
            <CardTitle>Reports</CardTitle>
            <CardDescription>Daily trade recap & backtest reports</CardDescription>
          </div>
          <Badge variant="outline">{idx?.generated_at || "-"}</Badge>
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
                      <Button variant="outline" size="sm" onClick={() => setCurrentFile(r.file)}>
                        Open
                      </Button>
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
            className="h-[76vh] w-full rounded-lg bg-white ring-1 ring-border/60"
          />
        </CardContent>
      </Card>
    </div>
  );
}

