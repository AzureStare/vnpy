import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchJson } from "../lib/api";
import { useInterval } from "../lib/interval";
import { useToast } from "../components/Toast";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";

type PortfolioSnapshot = Record<string, unknown> & {
  generated_at?: string;
  account?: { cash?: number; equity?: number; buying_power?: number; status?: string };
  positions?: Array<{ symbol: string; qty: number; market_value?: number; avg_entry?: number; unrealized_pnl?: number }>;
};

type SelectionSnapshot = Record<string, unknown> & {
  generated_at?: string;
  as_of_date?: string;
  signal_date?: string;
  rows?: Array<{
    vt_symbol?: string;
    signal?: number;
    close_price?: number | null;
    adv_usd?: number | null;
    med_volume?: number | null;
    market_cap?: number | null;
  }>;
};

type OrdersSnapshot = Record<string, unknown> & {
  generated_at?: string;
  orders?: Array<{
    id?: string;
    symbol?: string;
    side?: string;
    qty?: number;
    filled_qty?: number;
    filled_avg_price?: number | null;
    status?: string;
    order_type?: string;
    submitted_at?: string | null;
    filled_at?: string | null;
    canceled_at?: string | null;
  }>;
};

type PerformanceSnapshot = Record<string, unknown> & {
  generated_at?: string;
  equity_series?: Array<{ date: string; equity: number }>;
};

type TradingControls = Record<string, unknown> & {
  disabled_vt_symbols?: string[];
  buy_exposure_multiplier?: number;
};

type AccountSummary = {
  account_id: string;
  display_name: string;
  data_base_path: string;
};

type AccountsResponse = {
  accounts?: AccountSummary[];
};

function safeNumber(v: unknown): number | null {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtMoney(v: unknown, maxFractionDigits: number = 2): string {
  const n = safeNumber(v);
  if (n === null) return "-";
  return n.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: maxFractionDigits });
}

function fmtNumber(v: unknown, maxFractionDigits: number = 2): string {
  const n = safeNumber(v);
  if (n === null) return "-";
  return n.toLocaleString(undefined, { maximumFractionDigits: maxFractionDigits });
}

function fmtPct(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "-";
  return `${(v * 100).toFixed(2)}%`;
}

function fmtUtcTs(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (!Number.isFinite(d.valueOf())) return String(iso);
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mi = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss} UTC`;
}

function sumNumbers(values: Array<number | null | undefined>): number {
  let s = 0;
  for (const v of values) {
    if (typeof v === "number" && Number.isFinite(v)) s += v;
  }
  return s;
}

function buildSparklinePath(values: number[], width: number, height: number): string | null {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / (values.length - 1);
  const points = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * height;
    return { x, y };
  });
  return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
}

export function PortfolioPage() {
  const toast = useToast();

  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>("");

  const [portfolio, setPortfolio] = useState<PortfolioSnapshot | null>(null);
  const [selection, setSelection] = useState<SelectionSnapshot | null>(null);
  const [orders, setOrders] = useState<OrdersSnapshot | null>(null);
  const [perf, setPerf] = useState<PerformanceSnapshot | null>(null);
  const [controls, setControls] = useState<TradingControls | null>(null);

  const prevRankRef = useRef<Map<string, number>>(new Map());
  const [rankDeltaBySymbol, setRankDeltaBySymbol] = useState<Record<string, number>>({});

  const disabledSet = useMemo(() => {
    const xs = controls?.disabled_vt_symbols || [];
    return new Set(xs.map((s) => String(s || "").trim()).filter(Boolean));
  }, [controls]);

  const buyExposure = useMemo(() => {
    const m = safeNumber(controls?.buy_exposure_multiplier);
    if (m === null) return 1.0;
    return Math.min(1.0, Math.max(0.0, m));
  }, [controls]);

  const [draftExposurePct, setDraftExposurePct] = useState<number>(() => Math.round(buyExposure * 100));
  const [exposureDirty, setExposureDirty] = useState<boolean>(false);
  const [savingExposure, setSavingExposure] = useState<boolean>(false);

  useEffect(() => {
    const currentPct = Math.round(buyExposure * 100);
    if (!exposureDirty) {
      setDraftExposurePct(currentPct);
      return;
    }
    if (draftExposurePct === currentPct) {
      setExposureDirty(false);
    }
  }, [buyExposure, exposureDirty, draftExposurePct]);

  const selectedAccount = useMemo(() => {
    if (!selectedAccountId) return null;
    return accounts.find((a) => a.account_id === selectedAccountId) || null;
  }, [accounts, selectedAccountId]);

  const dataBase = useMemo(() => {
    // Default behavior: use /data/* (existing single-account snapshots)
    const p = String(selectedAccount?.data_base_path || "/data").trim();
    return p || "/data";
  }, [selectedAccount]);

  const loadAccounts = useCallback(async () => {
    try {
      const data = await fetchJson<AccountsResponse>("/api/accounts", { method: "GET" });
      const xs = Array.isArray(data?.accounts) ? data.accounts : [];
      setAccounts(xs);
      if (!selectedAccountId) {
        // Prefer the first account from backend; fallback to an empty string (use /data).
        setSelectedAccountId(xs[0]?.account_id || "");
      }
    } catch (_) {
      setAccounts([]);
      if (!selectedAccountId) setSelectedAccountId("");
    }
  }, [selectedAccountId]);

  const loadAll = useCallback(async () => {
    try {
      const [pf, sel, od, pr] = await Promise.all([
        fetchJson<PortfolioSnapshot>(`${dataBase}/portfolio.json`),
        fetchJson<SelectionSnapshot>("/data/selection.json"),
        fetchJson<OrdersSnapshot>(`${dataBase}/orders.json`),
        fetchJson<PerformanceSnapshot>(`${dataBase}/performance.json`),
      ]);
      setPortfolio(pf);
      setSelection(sel);
      setOrders(od);
      setPerf(pr);

      // Trading controls (auth API): do not fail the whole refresh if this is transient.
      try {
        const ctl = await fetchJson<TradingControls>("/api/trading/controls", { method: "GET" });
        setControls(ctl);
      } catch (_) {}

      // Rank delta: compare current snapshot ordering vs previous snapshot (in-memory).
      const next = new Map<string, number>();
      const deltas: Record<string, number> = {};
      for (let i = 0; i < (sel?.rows || []).length; i++) {
        const sym = String(sel?.rows?.[i]?.vt_symbol || "").trim();
        if (!sym) continue;
        const rank = i + 1;
        next.set(sym, rank);
      }
      const prev = prevRankRef.current;
      for (const [sym, rank] of next.entries()) {
        const prevRank = prev.get(sym);
        deltas[sym] = typeof prevRank === "number" ? prevRank - rank : 0;
      }
      prevRankRef.current = next;
      setRankDeltaBySymbol(deltas);
    } catch (e: any) {
      toast.push("Portfolio", String(e?.message || e), "bad");
    }
  }, [toast, dataBase]);

  const refreshControls = useCallback(async () => {
    try {
      const ctl = await fetchJson<TradingControls>("/api/trading/controls", { method: "GET" });
      setControls(ctl);
    } catch (e: any) {
      toast.push("Trading Controls", String(e?.message || e), "bad", 6000);
    }
  }, [toast]);

  const toggleDisabled = useCallback(
    async (vt_symbol: string, nextDisabled: boolean) => {
      const sym = String(vt_symbol || "").trim();
      if (!sym) return;
      try {
        await fetchJson("/api/trading/controls/disabled", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vt_symbol: sym, disabled: Boolean(nextDisabled) }),
        });
        toast.push("Trading Controls", `${nextDisabled ? "disabled" : "enabled"} ${sym}`, "good", 3000);
        await refreshControls();
      } catch (e: any) {
        toast.push("Trading Controls", String(e?.message || e), "bad", 6000);
      }
    },
    [toast, refreshControls]
  );

  const setExposure = useCallback(
    async (multiplier: number) => {
      const m = Number(multiplier);
      if (!Number.isFinite(m)) return;
      try {
        await fetchJson("/api/trading/controls/exposure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ multiplier: m }),
        });
        toast.push("Trading Controls", `exposure ${(m * 100).toFixed(0)}%`, "good", 3000);
        await refreshControls();
      } catch (e: any) {
        toast.push("Trading Controls", String(e?.message || e), "bad", 6000);
      }
    },
    [toast, refreshControls]
  );

  const loadOrdersOnly = useCallback(async () => {
    try {
      const od = await fetchJson<OrdersSnapshot>(`${dataBase}/orders.json`);
      setOrders(od);
    } catch (_) {
      // ignore transient
    }
  }, [dataBase]);

  useEffect(() => {
    void loadAccounts();
    loadAll();
  }, [loadAccounts, loadAll]);

  // Regular refresh
  useInterval(() => {
    loadAll();
  }, 60_000);

  // Orders-only refresh for timely UI (alerts are handled globally in App.tsx)
  useInterval(() => {
    loadOrdersOnly();
  }, 10_000);

  const positions = portfolio?.positions || [];

  const leaderboard = useMemo(() => {
    const rows = selection?.rows || [];
    return rows
      .filter((r) => Boolean(r?.vt_symbol))
      .slice(0, 20)
      .map((r, idx) => ({
        rank: idx + 1,
        vt_symbol: String(r.vt_symbol || ""),
        signal: typeof r.signal === "number" ? r.signal : null,
        close_price: typeof r.close_price === "number" ? r.close_price : null,
        adv_usd: typeof r.adv_usd === "number" ? r.adv_usd : null,
      }));
  }, [selection]);

  const filledBuys = useMemo(() => {
    const all = orders?.orders || [];
    return all
      .filter((o) => String(o.side || "").toLowerCase() === "buy")
      .filter((o) => {
        const filledQty = safeNumber(o.filled_qty) ?? 0;
        const filledPx = safeNumber(o.filled_avg_price);
        const filledAt = o.filled_at;
        const status = String(o.status || "").toLowerCase();
        return (
          filledQty > 0 &&
          filledPx !== null &&
          Boolean(filledAt) &&
          (status === "filled" || status === "partially_filled")
        );
      })
      .map((o) => {
        const filledQty = safeNumber(o.filled_qty) ?? 0;
        const filledPx = safeNumber(o.filled_avg_price) ?? 0;
        return {
          symbol: String(o.symbol || ""),
          filled_at: String(o.filled_at || ""),
          qty: filledQty,
          px: filledPx,
          notional: filledQty * filledPx,
        };
      })
      .sort((a, b) => (a.filled_at < b.filled_at ? 1 : -1));
  }, [orders]);

  const BUY_PAGE_SIZE = 20;
  const [buyPage, setBuyPage] = useState<number>(0);
  const buyPageCount = useMemo(() => {
    const n = filledBuys.length;
    return Math.max(1, Math.ceil(n / BUY_PAGE_SIZE));
  }, [filledBuys.length]);

  useEffect(() => {
    setBuyPage((p) => {
      const maxPage = Math.max(0, buyPageCount - 1);
      return Math.min(Math.max(0, p), maxPage);
    });
  }, [buyPageCount]);

  const buyPageItems = useMemo(() => {
    const start = buyPage * BUY_PAGE_SIZE;
    const end = start + BUY_PAGE_SIZE;
    return filledBuys.slice(start, end);
  }, [filledBuys, buyPage]);

  const buySummary = useMemo(() => {
    type Agg = { symbol: string; qty: number; notional: number };
    const map = new Map<string, Agg>();
    for (const f of filledBuys) {
      if (!f.symbol) continue;
      const it = map.get(f.symbol) || { symbol: f.symbol, qty: 0, notional: 0 };
      it.qty += f.qty;
      it.notional += f.notional;
      map.set(f.symbol, it);
    }
    return Array.from(map.values())
      .map((x) => ({ ...x, vwap: x.qty > 0 ? x.notional / x.qty : 0 }))
      .sort((a, b) => b.notional - a.notional)
      .slice(0, 10);
  }, [filledBuys]);

  const perfSummary = useMemo(() => {
    const series = (perf?.equity_series || []).slice().sort((a, b) => (a.date < b.date ? -1 : 1));
    const last = series.at(-1);
    const prev = series.length >= 2 ? series.at(-2) : null;
    const weekAgo = series.length >= 8 ? series.at(-8) : series.length >= 2 ? series[0] : null;

    const equity = safeNumber(last?.equity);
    const prevEquity = safeNumber(prev?.equity);
    const weekEquity = safeNumber(weekAgo?.equity);

    const d1 = equity !== null && prevEquity !== null ? equity - prevEquity : null;
    const d1Pct = equity !== null && prevEquity !== null && prevEquity !== 0 ? (equity - prevEquity) / prevEquity : null;
    const d7 = equity !== null && weekEquity !== null ? equity - weekEquity : null;
    const d7Pct = equity !== null && weekEquity !== null && weekEquity !== 0 ? (equity - weekEquity) / weekEquity : null;

    const values = series.map((x) => x.equity).filter((x) => typeof x === "number" && Number.isFinite(x));
    return { points: values, equity, d1, d1Pct, d7, d7Pct, lastDate: last?.date || null };
  }, [perf]);

  const unrealizedPnl = useMemo(() => {
    return sumNumbers(positions.map((p) => safeNumber(p.unrealized_pnl)));
  }, [positions]);

  const currentExposurePct = useMemo(() => Math.round(buyExposure * 100), [buyExposure]);

  const saveExposure = useCallback(async () => {
    if (!exposureDirty) return;
    const pct = Number(draftExposurePct);
    if (!Number.isFinite(pct)) return;
    const m = Math.min(100, Math.max(0, pct)) / 100;
    setSavingExposure(true);
    try {
      await setExposure(m);
    } finally {
      setSavingExposure(false);
    }
  }, [draftExposurePct, exposureDirty, setExposure]);

  const sparklinePath = useMemo(() => {
    return buildSparklinePath(perfSummary.points, 240, 64);
  }, [perfSummary.points]);

  return (
    <div>
      <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xl font-semibold tracking-tight">Portfolio</div>
          <div className="mt-1 text-sm text-muted-foreground">Leaderboard, positions and daily performance</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={selectedAccountId || ""}
            onChange={(e) => setSelectedAccountId(e.target.value)}
            className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {(accounts || []).map((a) => (
              <option key={a.account_id} value={a.account_id}>
                {a.display_name}
              </option>
            ))}
            {!accounts.length ? <option value="">Default</option> : null}
          </select>
          <Button variant="outline" size="sm" onClick={() => void loadAll()}>
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Leaderboard */}
        <Card className="col-span-12 lg:col-span-7">
          <CardHeader>
            <div>
              <CardTitle>Leaderboard (Today)</CardTitle>
              <CardDescription>
                as_of_date={selection?.as_of_date || "-"} · signal_date={selection?.signal_date || "-"}
              </CardDescription>
            </div>
            <Badge variant="outline">{fmtUtcTs(selection?.generated_at)}</Badge>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[88px]">Rank</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead className="text-right">Score</TableHead>
                    <TableHead className="text-right">Close</TableHead>
                    <TableHead className="text-right">ADV($)</TableHead>
                    <TableHead className="text-right">Trade</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {leaderboard.map((r) => {
                    const delta = rankDeltaBySymbol[r.vt_symbol] || 0;
                    const deltaLabel = delta === 0 ? "—" : delta > 0 ? `↑${delta}` : `↓${Math.abs(delta)}`;
                    const deltaTone = delta === 0 ? "text-muted-foreground" : delta > 0 ? "text-emerald-700" : "text-rose-700";
                    const isDisabled = disabledSet.has(r.vt_symbol);
                    return (
                      <TableRow key={r.vt_symbol}>
                        <TableCell className="text-xs">
                          <span className="font-medium tabular-nums">{r.rank}</span>{" "}
                          <span className={["ml-1 tabular-nums", deltaTone].join(" ")}>{deltaLabel}</span>
                        </TableCell>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            <span>{r.vt_symbol}</span>
                            {isDisabled ? <Badge variant="bad">disabled</Badge> : null}
                          </div>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{fmtNumber(r.signal, 4)}</TableCell>
                        <TableCell className="text-right tabular-nums">
                          {r.close_price ? fmtMoney(r.close_price, 2) : "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{r.adv_usd ? fmtMoney(r.adv_usd, 0) : "-"}</TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant={isDisabled ? "outline" : "destructive"}
                            size="sm"
                            onClick={() => void toggleDisabled(r.vt_symbol, !isDisabled)}
                          >
                            {isDisabled ? "Enable" : "Disable"}
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                  {!leaderboard.length && (
                    <TableRow>
                      <TableCell colSpan={6} className="text-sm text-muted-foreground">
                        No selection data yet
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* Right column: trading controls + portfolio snapshot */}
        <div className="col-span-12 flex flex-col gap-6 lg:col-span-5">
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Exposure</CardTitle>
                <CardDescription>Buy-side multiplier (adjust then Save)</CardDescription>
              </div>
              <Badge variant="outline">Current {currentExposurePct}%</Badge>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-3">
                  <div className="w-[34px] text-right text-xs text-muted-foreground tabular-nums">0%</div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={draftExposurePct}
                    onChange={(e) => {
                      const v = Number(e.currentTarget.value);
                      setDraftExposurePct(v);
                      setExposureDirty(v !== currentExposurePct);
                    }}
                    className="h-2 w-full cursor-pointer accent-primary"
                    disabled={savingExposure}
                  />
                  <div className="w-[52px] text-right text-xs font-medium tabular-nums">{draftExposurePct}%</div>
                </div>

                <div className="flex items-center justify-end gap-2">
                  {exposureDirty ? (
                    <span className="text-xs text-muted-foreground">Unsaved</span>
                  ) : (
                    <span className="text-xs text-muted-foreground">Saved</span>
                  )}
                  <Button
                    variant={exposureDirty ? "default" : "outline"}
                    size="sm"
                    onClick={() => void saveExposure()}
                    disabled={!exposureDirty || savingExposure}
                  >
                    {savingExposure ? "Saving..." : "Save"}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div>
                <CardTitle>Portfolio</CardTitle>
                <CardDescription>Real-time positions + key metrics</CardDescription>
              </div>
              <Badge variant="outline">{fmtUtcTs(portfolio?.generated_at)}</Badge>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Equity</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(portfolio?.account?.equity)}</div>
                </div>
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Cash</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(portfolio?.account?.cash)}</div>
                </div>
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Buying power</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(portfolio?.account?.buying_power)}</div>
                </div>
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Unrealized PnL</div>
                  <div
                    className={[
                      "mt-1 text-lg font-semibold tabular-nums",
                      unrealizedPnl > 0 ? "text-emerald-700" : unrealizedPnl < 0 ? "text-rose-700" : "text-foreground",
                    ].join(" ")}
                  >
                    {fmtMoney(unrealizedPnl)}
                  </div>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between">
                <div className="text-sm font-medium">Holdings</div>
                <Badge variant="outline">{positions.length} positions</Badge>
              </div>

              <div className="mt-3 overflow-hidden rounded-lg ring-1 ring-border/60">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead className="text-right">Qty</TableHead>
                      <TableHead className="text-right">Mkt value</TableHead>
                      <TableHead className="text-right">uPnL</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {positions.slice(0, 10).map((p) => {
                      const upnl = safeNumber(p.unrealized_pnl) ?? 0;
                      return (
                        <TableRow key={p.symbol}>
                          <TableCell className="font-medium">{p.symbol}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmtNumber(p.qty, 0)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmtMoney(p.market_value, 0)}</TableCell>
                          <TableCell
                            className={[
                              "text-right tabular-nums",
                              upnl > 0 ? "text-emerald-700" : upnl < 0 ? "text-rose-700" : "text-foreground",
                            ].join(" ")}
                          >
                            {fmtMoney(upnl, 0)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                    {!positions.length && (
                      <TableRow>
                        <TableCell colSpan={4} className="text-sm text-muted-foreground">
                          No positions
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance */}
        <Card className="col-span-12 lg:col-span-7">
          <CardHeader>
            <div>
              <CardTitle>Daily Performance</CardTitle>
              <CardDescription>Equity curve + summary</CardDescription>
            </div>
            <Badge variant="outline">{fmtUtcTs(perf?.generated_at)}</Badge>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg bg-muted/40 p-4">
                <div className="text-xs text-muted-foreground">Latest equity</div>
                <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(perfSummary.equity)}</div>
              </div>
              <div className="rounded-lg bg-muted/40 p-4">
                <div className="text-xs text-muted-foreground">1D</div>
                <div
                  className={[
                    "mt-1 text-lg font-semibold tabular-nums",
                    (perfSummary.d1 || 0) > 0
                      ? "text-emerald-700"
                      : (perfSummary.d1 || 0) < 0
                        ? "text-rose-700"
                        : "text-foreground",
                  ].join(" ")}
                >
                  {fmtMoney(perfSummary.d1)}{" "}
                  <span className="text-sm font-medium text-muted-foreground">({fmtPct(perfSummary.d1Pct)})</span>
                </div>
              </div>
              <div className="rounded-lg bg-muted/40 p-4">
                <div className="text-xs text-muted-foreground">7D</div>
                <div
                  className={[
                    "mt-1 text-lg font-semibold tabular-nums",
                    (perfSummary.d7 || 0) > 0
                      ? "text-emerald-700"
                      : (perfSummary.d7 || 0) < 0
                        ? "text-rose-700"
                        : "text-foreground",
                  ].join(" ")}
                >
                  {fmtMoney(perfSummary.d7)}{" "}
                  <span className="text-sm font-medium text-muted-foreground">({fmtPct(perfSummary.d7Pct)})</span>
                </div>
              </div>
            </div>

            <div className="mt-5 rounded-lg bg-white p-4 ring-1 ring-border/60">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">Equity (sparkline)</div>
                <div className="text-xs text-muted-foreground">{perfSummary.lastDate ? fmtUtcTs(perfSummary.lastDate) : "-"}</div>
              </div>
              <div className="mt-3">
                {sparklinePath ? (
                  <svg viewBox="0 0 240 64" className="h-16 w-full">
                    <path
                      d={sparklinePath}
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className={(perfSummary.d7 || 0) >= 0 ? "text-emerald-600" : "text-rose-600"}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  </svg>
                ) : (
                  <div className="text-sm text-muted-foreground">No performance series yet</div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Purchase history */}
        <Card className="col-span-12 lg:col-span-5">
          <CardHeader>
            <div>
              <CardTitle>Purchase History</CardTitle>
              <CardDescription>Filled BUY orders (summary)</CardDescription>
            </div>
            <Badge variant="outline">{fmtUtcTs(orders?.generated_at)}</Badge>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Filled BUY</div>
              <div className="flex items-center gap-2">
                <Badge variant="outline">{filledBuys.length} fills</Badge>
                <Badge variant="outline">
                  Page {buyPage + 1}/{buyPageCount}
                </Badge>
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">Page size: {BUY_PAGE_SIZE}</div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setBuyPage((p) => Math.max(0, p - 1))}
                  disabled={buyPage <= 0}
                >
                  Prev
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setBuyPage((p) => Math.min(buyPageCount - 1, p + 1))}
                  disabled={buyPage >= buyPageCount - 1}
                >
                  Next
                </Button>
              </div>
            </div>

            <div className="mt-3 overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[140px]">Time</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead className="text-right">Qty</TableHead>
                    <TableHead className="text-right">Avg px</TableHead>
                    <TableHead className="text-right">Notional</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {buyPageItems.map((b) => (
                    <TableRow key={`${b.symbol}_${b.filled_at}`}>
                      <TableCell className="text-xs text-muted-foreground">{fmtUtcTs(b.filled_at)}</TableCell>
                      <TableCell className="font-medium">{b.symbol}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtNumber(b.qty, 0)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(b.px, 2)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(b.notional, 0)}</TableCell>
                    </TableRow>
                  ))}
                  {!filledBuys.length && (
                    <TableRow>
                      <TableCell colSpan={5} className="text-sm text-muted-foreground">
                        No filled BUY orders yet
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>

            <div className="mt-5 flex items-center justify-between">
              <div className="text-sm font-medium">Top bought (notional)</div>
              <a className="text-xs text-primary hover:underline" href="/data/orders.json" target="_blank" rel="noopener noreferrer">
                View raw
              </a>
            </div>

            <div className="mt-3 overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead className="text-right">Qty</TableHead>
                    <TableHead className="text-right">VWAP</TableHead>
                    <TableHead className="text-right">Notional</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {buySummary.map((x) => (
                    <TableRow key={x.symbol}>
                      <TableCell className="font-medium">{x.symbol}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtNumber(x.qty, 0)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(x.vwap, 2)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(x.notional, 0)}</TableCell>
                    </TableRow>
                  ))}
                  {!buySummary.length && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-sm text-muted-foreground">
                        No BUY aggregates yet
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

