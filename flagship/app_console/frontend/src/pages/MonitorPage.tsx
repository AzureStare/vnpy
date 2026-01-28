import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";

type DayRow = {
  vt_symbol?: string;
  rank_pos?: number;
  signal_score?: number;
  close_t?: number | null;
  spy_close_t?: number | null;
  trail_ret_1d?: number | null;
  trail_ret_3d?: number | null;
  trail_ret_5d?: number | null;
  trail_excess_1d?: number | null;
  trail_excess_3d?: number | null;
  trail_excess_5d?: number | null;
  fwd_ret_1d?: number | null;
  fwd_ret_3d?: number | null;
  fwd_ret_5d?: number | null;
  fwd_excess_1d?: number | null;
  fwd_excess_3d?: number | null;
  fwd_excess_5d?: number | null;
};

type DayResponse = {
  trade_date?: string;
  windows?: number[];
  rows?: DayRow[];
  available_start?: string | null;
  available_end?: string | null;
};

type RangeRow = {
  bucket?: string;
  horizon_d?: number;
  ret_type?: string;
  metric?: string;
  mean?: number | null;
  p50?: number | null;
  win_rate?: number | null;
  n?: number;
};

type RangeResponse = {
  start?: string;
  end?: string;
  bucket_size?: number;
  windows?: number[];
  rows?: RangeRow[];
  available_start?: string | null;
  available_end?: string | null;
};

function fmtPct(v: number | null | undefined): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  return `${(v * 100).toFixed(2)}%`;
}

function toneClass(v: number | null | undefined): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "text-foreground";
  if (v > 0) return "text-emerald-700";
  if (v < 0) return "text-rose-700";
  return "text-foreground";
}

function todayIso(): string {
  const d = new Date();
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export function MonitorPage() {
  const [tradeDate, setTradeDate] = useState<string>("");
  const [dayResp, setDayResp] = useState<DayResponse | null>(null);
  const [dayLoading, setDayLoading] = useState<boolean>(false);
  const [dayError, setDayError] = useState<string | null>(null);

  const [rangeStart, setRangeStart] = useState<string>("");
  const [rangeEnd, setRangeEnd] = useState<string>("");
  const [bucketSize, setBucketSize] = useState<number>(10);
  const [rangeResp, setRangeResp] = useState<RangeResponse | null>(null);
  const [rangeLoading, setRangeLoading] = useState<boolean>(false);
  const [rangeError, setRangeError] = useState<string | null>(null);

  const [window, setWindow] = useState<number>(1);
  const [direction, setDirection] = useState<"trail" | "fwd">("trail");
  const [metric, setMetric] = useState<"ret" | "excess">("excess");

  const [rangeWindow, setRangeWindow] = useState<number>(1);
  const [rangeDirection, setRangeDirection] = useState<"trail" | "fwd">("trail");
  const [rangeMetric, setRangeMetric] = useState<"ret" | "excess">("excess");

  const loadDay = useCallback(async () => {
    setDayLoading(true);
    setDayError(null);
    try {
      const url = tradeDate ? `/api/monitor/ranking_returns/day?trade_date=${tradeDate}` : `/api/monitor/ranking_returns/day`;
      const resp = await fetchJson<DayResponse>(url);
      setDayResp(resp);
      const ws = resp?.windows || [];
      if (ws.length && !ws.includes(window)) {
        setWindow(ws[0]);
      }
      if (!tradeDate && resp?.trade_date) {
        setTradeDate(resp.trade_date);
      }
    } catch (e: any) {
      setDayResp(null);
      setDayError(String(e?.message || e));
    } finally {
      setDayLoading(false);
    }
  }, [tradeDate, window]);

  const loadRange = useCallback(async () => {
    setRangeLoading(true);
    setRangeError(null);
    try {
      const url =
        rangeStart && rangeEnd
          ? `/api/monitor/ranking_returns/range?start=${rangeStart}&end=${rangeEnd}&bucket=${bucketSize}`
          : `/api/monitor/ranking_returns/range?bucket=${bucketSize}`;
      const resp = await fetchJson<RangeResponse>(url);
      setRangeResp(resp);
      const ws = resp?.windows || [];
      if (ws.length && !ws.includes(rangeWindow)) {
        setRangeWindow(ws[0]);
      }
      if (!rangeStart && resp?.start) setRangeStart(resp.start);
      if (!rangeEnd && resp?.end) setRangeEnd(resp.end);
    } catch (e: any) {
      setRangeResp(null);
      setRangeError(String(e?.message || e));
    } finally {
      setRangeLoading(false);
    }
  }, [rangeStart, rangeEnd, bucketSize, rangeWindow]);

  useEffect(() => {
    void loadDay();
    void loadRange();
  }, [loadDay, loadRange]);

  const dayRows = useMemo(() => {
    const rows = Array.isArray(dayResp?.rows) ? dayResp?.rows : [];
    const key = `${direction}_${metric}_${window}d`;
    const sorted = [...rows].filter((r) => r?.vt_symbol);
    sorted.sort((a, b) => {
      const va = (a as Record<string, unknown>)[key];
      const vb = (b as Record<string, unknown>)[key];
      const na = typeof va === "number" ? va : null;
      const nb = typeof vb === "number" ? vb : null;
      if (na === null && nb === null) return 0;
      if (na === null) return 1;
      if (nb === null) return -1;
      return nb - na;
    });
    return sorted.slice(0, 50);
  }, [dayResp, direction, metric, window]);

  const rangeRows = useMemo(() => {
    const rows = Array.isArray(rangeResp?.rows) ? rangeResp?.rows : [];
    return rows
      .filter((r) => Number(r?.horizon_d) === Number(rangeWindow))
      .filter((r) => String(r?.ret_type) === rangeDirection)
      .filter((r) => String(r?.metric) === rangeMetric);
  }, [rangeResp, rangeWindow, rangeDirection, rangeMetric]);

  const windows = useMemo(() => {
    const ws = dayResp?.windows || rangeResp?.windows || [];
    return ws.length ? ws : [1, 3, 5];
  }, [dayResp?.windows, rangeResp?.windows]);

  const availableHint = useMemo(() => {
    const a = dayResp?.available_start || rangeResp?.available_start;
    const b = dayResp?.available_end || rangeResp?.available_end;
    if (!a || !b) return null;
    return `available: ${a} ~ ${b}`;
  }, [dayResp?.available_start, dayResp?.available_end, rangeResp?.available_start, rangeResp?.available_end]);

  return (
    <div>
      <div className="mb-6">
        <div className="text-xl font-semibold tracking-tight">Monitor</div>
        <div className="mt-1 text-sm text-muted-foreground">Ranking returns (AlphaLab close-close)</div>
        {availableHint ? <div className="mt-1 text-xs text-muted-foreground">{availableHint}</div> : null}
      </div>

      <div className="grid grid-cols-12 gap-6">
        <Card className="col-span-12 xl:col-span-7">
          <CardHeader>
            <div>
              <CardTitle>Daily View</CardTitle>
              <CardDescription>Top50 ranking returns for one trade_date</CardDescription>
            </div>
            <Badge variant="outline">{dayResp?.trade_date || "-"}</Badge>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <input
                type="date"
                value={tradeDate}
                onChange={(e) => setTradeDate(e.currentTarget.value)}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <Button variant="outline" size="sm" onClick={() => void loadDay()} disabled={dayLoading}>
                {dayLoading ? "Loading..." : "Refresh"}
              </Button>
              <select
                value={window}
                onChange={(e) => setWindow(Number(e.currentTarget.value))}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {windows.map((w) => (
                  <option key={w} value={w}>
                    {w}D
                  </option>
                ))}
              </select>
              <select
                value={direction}
                onChange={(e) => setDirection(e.currentTarget.value as "trail" | "fwd")}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="trail">Trailing</option>
                <option value="fwd">Forward</option>
              </select>
              <select
                value={metric}
                onChange={(e) => setMetric(e.currentTarget.value as "ret" | "excess")}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="excess">Excess</option>
                <option value="ret">Absolute</option>
              </select>
              {dayError ? <span className="text-xs text-rose-600">{dayError}</span> : null}
            </div>
            <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[64px]">Rank</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead className="text-right">Score</TableHead>
                    <TableHead className="text-right">Trail Ret</TableHead>
                    <TableHead className="text-right">Trail Excess</TableHead>
                    <TableHead className="text-right">Fwd Ret</TableHead>
                    <TableHead className="text-right">Fwd Excess</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {dayRows.map((r, idx) => (
                    <TableRow key={`${r.vt_symbol}_${idx}`}>
                      <TableCell className="text-xs tabular-nums">{r.rank_pos ?? idx + 1}</TableCell>
                      <TableCell className="font-medium">{r.vt_symbol || "-"}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {typeof r.signal_score === "number" ? r.signal_score.toFixed(4) : "-"}
                      </TableCell>
                      <TableCell className={["text-right tabular-nums", toneClass((r as any)[`trail_ret_${window}d`])].join(" ")}>
                        {fmtPct((r as any)[`trail_ret_${window}d`])}
                      </TableCell>
                      <TableCell
                        className={[
                          "text-right tabular-nums",
                          toneClass((r as any)[`trail_excess_${window}d`]),
                        ].join(" ")}
                      >
                        {fmtPct((r as any)[`trail_excess_${window}d`])}
                      </TableCell>
                      <TableCell className={["text-right tabular-nums", toneClass((r as any)[`fwd_ret_${window}d`])].join(" ")}>
                        {fmtPct((r as any)[`fwd_ret_${window}d`])}
                      </TableCell>
                      <TableCell
                        className={[
                          "text-right tabular-nums",
                          toneClass((r as any)[`fwd_excess_${window}d`]),
                        ].join(" ")}
                      >
                        {fmtPct((r as any)[`fwd_excess_${window}d`])}
                      </TableCell>
                    </TableRow>
                  ))}
                  {!dayRows.length && (
                    <TableRow>
                      <TableCell colSpan={7} className="text-sm text-muted-foreground">
                        No data
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        <Card className="col-span-12 xl:col-span-5">
          <CardHeader>
            <div>
              <CardTitle>Range Analysis</CardTitle>
              <CardDescription>Bucketed stats by rank</CardDescription>
            </div>
            <Badge variant="outline">
              {rangeResp?.start || "-"} → {rangeResp?.end || "-"}
            </Badge>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <input
                type="date"
                value={rangeStart}
                onChange={(e) => setRangeStart(e.currentTarget.value)}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <input
                type="date"
                value={rangeEnd}
                onChange={(e) => setRangeEnd(e.currentTarget.value)}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <input
                type="number"
                min={5}
                max={50}
                step={5}
                value={bucketSize}
                onChange={(e) => setBucketSize(Number(e.currentTarget.value))}
                className="h-9 w-[100px] rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <Button variant="outline" size="sm" onClick={() => void loadRange()} disabled={rangeLoading}>
                {rangeLoading ? "Loading..." : "Refresh"}
              </Button>
              <select
                value={rangeWindow}
                onChange={(e) => setRangeWindow(Number(e.currentTarget.value))}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {windows.map((w) => (
                  <option key={w} value={w}>
                    {w}D
                  </option>
                ))}
              </select>
              <select
                value={rangeDirection}
                onChange={(e) => setRangeDirection(e.currentTarget.value as "trail" | "fwd")}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="trail">Trailing</option>
                <option value="fwd">Forward</option>
              </select>
              <select
                value={rangeMetric}
                onChange={(e) => setRangeMetric(e.currentTarget.value as "ret" | "excess")}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="excess">Excess</option>
                <option value="ret">Absolute</option>
              </select>
              {rangeError ? <span className="text-xs text-rose-600">{rangeError}</span> : null}
            </div>
            <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Bucket</TableHead>
                    <TableHead className="text-right">Mean</TableHead>
                    <TableHead className="text-right">P50</TableHead>
                    <TableHead className="text-right">Win rate</TableHead>
                    <TableHead className="text-right">N</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rangeRows.map((r, idx) => (
                    <TableRow key={`${r.bucket}_${idx}`}>
                      <TableCell className="font-medium">{r.bucket}</TableCell>
                      <TableCell className={["text-right tabular-nums", toneClass(r.mean)].join(" ")}>{fmtPct(r.mean)}</TableCell>
                      <TableCell className={["text-right tabular-nums", toneClass(r.p50)].join(" ")}>{fmtPct(r.p50)}</TableCell>
                      <TableCell className={["text-right tabular-nums", toneClass(r.win_rate)].join(" ")}>{fmtPct(r.win_rate)}</TableCell>
                      <TableCell className="text-right tabular-nums">{r.n ?? "-"}</TableCell>
                    </TableRow>
                  ))}
                  {!rangeRows.length && (
                    <TableRow>
                      <TableCell colSpan={5} className="text-sm text-muted-foreground">
                        No data
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
