import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";

type Broker = "alpaca" | "ibkr" | "unknown";
type Env = "paper" | "live" | "unknown";
type AccountStatus = "connected" | "degraded" | "disconnected" | "unknown";

type AccountSummary = {
  account_id: string;
  display_name: string;
  broker: Broker;
  env: Env;
  status: AccountStatus;
  last_sync_utc: string | null;

  equity_usd: number | null;
  cash_usd: number | null;
  buying_power_usd: number | null;

  positions_count: number;
  open_orders_count: number;
  today_pnl_usd: number | null;
  data_base_path: string;
};

function statusBadgeVariant(s: AccountStatus): "good" | "info" | "bad" | "outline" {
  if (s === "connected") return "good";
  if (s === "degraded") return "info";
  if (s === "disconnected") return "bad";
  return "outline";
}

function brokerBadge(b: Broker): string {
  if (b === "ibkr") return "IBKR";
  if (b === "alpaca") return "Alpaca";
  return "Unknown";
}

function envBadge(e: Env): string {
  if (e === "paper") return "Paper";
  if (e === "live") return "Live";
  return "Unknown";
}

function fmtMoney(v: number | null | undefined, maxFractionDigits: number = 0): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  return v.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: maxFractionDigits });
}

function fmtUtcTs(iso: string): string {
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

function fmtPctFromPnl(pnl: number | null, equity: number): string {
  if (pnl === null || !Number.isFinite(pnl) || !Number.isFinite(equity) || equity <= 0) return "-";
  return `${((pnl / equity) * 100).toFixed(2)}%`;
}

type AccountsResponse = {
  generated_at?: string;
  accounts?: AccountSummary[];
};

type OrdersSnapshot = {
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
    limit_price?: number | null;
    stop_price?: number | null;
    submitted_at?: string | null;
    filled_at?: string | null;
    canceled_at?: string | null;
  }>;
};

type PortfolioSnapshot = {
  generated_at?: string;
  positions?: Array<{ symbol: string; qty: number; market_value?: number; avg_entry?: number; unrealized_pnl?: number }>;
};

export function AccountsPage() {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>("all");

  const loadAccounts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchJson<AccountsResponse>("/api/accounts", { method: "GET" });
      const xs = Array.isArray(data?.accounts) ? data.accounts : [];
      setAccounts(xs);
      setGeneratedAt(String(data?.generated_at || "") || null);
    } catch (e: any) {
      setError(String(e?.message || e));
      setAccounts([]);
      setGeneratedAt(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAccounts();
  }, [loadAccounts]);

  const selected = useMemo(() => {
    if (selectedAccountId === "all") return null;
    return accounts.find((a) => a.account_id === selectedAccountId) || null;
  }, [accounts, selectedAccountId]);

  const [portfolio, setPortfolio] = useState<PortfolioSnapshot | null>(null);
  const [orders, setOrders] = useState<OrdersSnapshot | null>(null);

  const loadSelectedDetails = useCallback(async () => {
    if (!selected) {
      setPortfolio(null);
      setOrders(null);
      return;
    }
    try {
      const base = selected.data_base_path || "/data";
      const [pf, od] = await Promise.all([
        fetchJson<PortfolioSnapshot>(`${base}/portfolio.json`),
        fetchJson<OrdersSnapshot>(`${base}/orders.json`),
      ]);
      setPortfolio(pf);
      setOrders(od);
    } catch (_) {
      // best-effort: leave existing or null
      setPortfolio(null);
      setOrders(null);
    }
  }, [selected]);

  useEffect(() => {
    void loadSelectedDetails();
  }, [loadSelectedDetails]);

  const fills = useMemo(() => {
    const all = orders?.orders || [];
    return all
      .filter((o) => String(o?.status || "").toLowerCase() === "filled")
      .filter((o) => (Number(o?.filled_qty || 0) || 0) > 0)
      .slice(0, 50);
  }, [orders]);

  return (
    <div>
      <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xl font-semibold tracking-tight">Accounts</div>
          <div className="mt-1 text-sm text-muted-foreground">Multi-broker accounts (Alpaca + IBKR-ready)</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {loading ? <Badge variant="outline">loading</Badge> : null}
          <select
            value={selectedAccountId}
            onChange={(e) => setSelectedAccountId(e.target.value)}
            className="h-9 rounded-md border border-border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <option value="all">All accounts</option>
            {accounts.map((a) => (
              <option key={a.account_id} value={a.account_id}>
                {a.display_name}
              </option>
            ))}
          </select>
          <Button variant="outline" size="sm" onClick={() => setSelectedAccountId("all")} disabled={loading}>
            Reset
          </Button>
          <Button variant="outline" size="sm" onClick={() => void loadAccounts()} disabled={loading}>
            Refresh
          </Button>
        </div>
      </div>

      {error ? (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Failed to load accounts</CardTitle>
            <CardDescription className="break-all">{error}</CardDescription>
          </CardHeader>
        </Card>
      ) : null}

      {selectedAccountId === "all" ? (
        <Card>
          <CardHeader>
            <div>
              <CardTitle>All accounts</CardTitle>
              <CardDescription>Broker / env / status + daily snapshot summary</CardDescription>
            </div>
            <Badge variant="outline">{generatedAt ? fmtUtcTs(generatedAt) : "-"}</Badge>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Account</TableHead>
                    <TableHead>Broker</TableHead>
                    <TableHead>Env</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Equity</TableHead>
                    <TableHead className="text-right">Cash</TableHead>
                    <TableHead className="text-right">Buying power</TableHead>
                    <TableHead className="text-right">Pos</TableHead>
                    <TableHead className="text-right">Open orders</TableHead>
                    <TableHead className="text-right">Today PnL</TableHead>
                    <TableHead className="text-right">Last sync</TableHead>
                    <TableHead className="text-right">Details</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {accounts.map((a) => (
                    <TableRow key={a.account_id}>
                      <TableCell className="font-medium">
                        <div className="flex flex-col">
                          <span>{a.display_name}</span>
                          <span className="text-xs text-muted-foreground">{a.account_id}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{brokerBadge(a.broker)}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{envBadge(a.env)}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={statusBadgeVariant(a.status)}>{a.status}</Badge>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(a.equity_usd, 0)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(a.cash_usd, 0)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMoney(a.buying_power_usd, 0)}</TableCell>
                      <TableCell className="text-right tabular-nums">{a.positions_count}</TableCell>
                      <TableCell className="text-right tabular-nums">{a.open_orders_count}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {typeof a.today_pnl_usd === "number" ? (
                          <span className={a.today_pnl_usd >= 0 ? "text-emerald-700" : "text-rose-700"}>
                            {fmtMoney(a.today_pnl_usd, 0)}{" "}
                            <span className="text-xs text-muted-foreground">({fmtPctFromPnl(a.today_pnl_usd, a.equity_usd || 0)})</span>
                          </span>
                        ) : (
                          "-"
                        )}
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground">{a.last_sync_utc ? fmtUtcTs(a.last_sync_utc) : "-"}</TableCell>
                      <TableCell className="text-right">
                        <Button variant="outline" size="sm" onClick={() => setSelectedAccountId(a.account_id)} disabled={loading}>
                          View details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {!accounts.length && (
                    <TableRow>
                      <TableCell colSpan={12} className="text-sm text-muted-foreground">
                        No accounts
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-12 gap-6">
          <Card className="col-span-12">
            <CardHeader>
              <div>
                <CardTitle>{selected?.display_name || "-"}</CardTitle>
                <CardDescription>
                  <span className="mr-2">{selected?.account_id || "-"}</span>
                  <Badge variant="outline">{selected ? brokerBadge(selected.broker) : "-"}</Badge>{" "}
                  <Badge variant="outline">{selected ? envBadge(selected.env) : "-"}</Badge>{" "}
                  {selected ? <Badge variant={statusBadgeVariant(selected.status)}>{selected.status}</Badge> : null}
                </CardDescription>
              </div>
              <Badge variant="outline">{selected?.last_sync_utc ? fmtUtcTs(selected.last_sync_utc) : "-"}</Badge>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-4">
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Equity / NetLiq</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(selected?.equity_usd, 0)}</div>
                </div>
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Cash</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(selected?.cash_usd, 0)}</div>
                </div>
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Buying power / Available funds</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">{fmtMoney(selected?.buying_power_usd, 0)}</div>
                </div>
                <div className="rounded-lg bg-muted/40 p-4">
                  <div className="text-xs text-muted-foreground">Positions / Open orders</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums">
                    {(selected?.positions_count ?? 0).toString()} / {(selected?.open_orders_count ?? 0).toString()}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="col-span-12 lg:col-span-6">
            <CardHeader>
              <div>
                <CardTitle>Holdings</CardTitle>
                <CardDescription>Positions snapshot</CardDescription>
              </div>
              <Badge variant="outline">{portfolio?.generated_at ? fmtUtcTs(portfolio.generated_at) : "-"}</Badge>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead className="text-right">Qty</TableHead>
                      <TableHead className="text-right">Avg</TableHead>
                      <TableHead className="text-right">Mkt value</TableHead>
                      <TableHead className="text-right">uPnL</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(portfolio?.positions || []).slice(0, 20).map((p) => {
                      const upnl = typeof p.unrealized_pnl === "number" ? p.unrealized_pnl : 0;
                      return (
                        <TableRow key={p.symbol}>
                          <TableCell className="font-medium">{p.symbol}</TableCell>
                          <TableCell className="text-right tabular-nums">{Number(p.qty || 0).toLocaleString()}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmtMoney(p.avg_entry ?? null, 2)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmtMoney(p.market_value ?? null, 0)}</TableCell>
                          <TableCell className={["text-right tabular-nums", upnl >= 0 ? "text-emerald-700" : "text-rose-700"].join(" ")}>
                            {fmtMoney(upnl, 0)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                    {!(portfolio?.positions || []).length ? (
                      <TableRow>
                        <TableCell colSpan={5} className="text-sm text-muted-foreground">
                          No positions
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          <Card className="col-span-12 lg:col-span-6">
            <CardHeader>
              <div>
                <CardTitle>Orders</CardTitle>
                <CardDescription>Open + recent orders</CardDescription>
              </div>
              <Badge variant="outline">{orders?.generated_at ? fmtUtcTs(orders.generated_at) : "-"}</Badge>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead className="text-right">Qty</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Limit</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(orders?.orders || []).slice(0, 50).map((o, idx) => (
                      <TableRow key={`${String(o.filled_at || o.submitted_at || "")}_${idx}`}>
                        <TableCell className="text-xs text-muted-foreground">{fmtUtcTs(String(o.filled_at || o.submitted_at || ""))}</TableCell>
                        <TableCell className="font-medium">{String(o.symbol || "-")}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{String(o.side || "-").toUpperCase()}</Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{String(o.filled_qty ?? "-")}</TableCell>
                        <TableCell>
                          <Badge variant={String(o.status || "").toLowerCase() === "filled" ? "good" : "info"}>{String(o.status || "-")}</Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">-</TableCell>
                      </TableRow>
                    ))}
                    {!(orders?.orders || []).length ? (
                      <TableRow>
                        <TableCell colSpan={6} className="text-sm text-muted-foreground">
                          No orders
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          <Card className="col-span-12 lg:col-span-6">
            <CardHeader>
              <div>
                <CardTitle>Trades / Fills</CardTitle>
                <CardDescription>Fill-level records</CardDescription>
              </div>
              <Badge variant="outline">{fills.length ? "filled" : "-"}</Badge>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="overflow-hidden rounded-lg ring-1 ring-border/60">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead className="text-right">Qty</TableHead>
                      <TableHead className="text-right">Px</TableHead>
                      <TableHead className="text-right">Commission</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {fills.map((f, idx) => (
                      <TableRow key={`${String(f.filled_at || "")}_${idx}`}>
                        <TableCell className="text-xs text-muted-foreground">{fmtUtcTs(String(f.filled_at || ""))}</TableCell>
                        <TableCell className="font-medium">{String(f.symbol || "-")}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{String(f.side || "-").toUpperCase()}</Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{String(f.filled_qty ?? "-")}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmtMoney(f.filled_avg_price ?? null, 2)}</TableCell>
                        <TableCell className="text-right tabular-nums">-</TableCell>
                      </TableRow>
                    ))}
                    {!fills.length ? (
                      <TableRow>
                        <TableCell colSpan={6} className="text-sm text-muted-foreground">
                          No fills
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          <Card className="col-span-12 lg:col-span-6">
            <CardHeader>
              <div>
                <CardTitle>Performance</CardTitle>
                <CardDescription>Equity curve + KPIs (placeholder)</CardDescription>
              </div>
              <Badge variant="outline">{selected?.last_sync_utc ? fmtUtcTs(selected.last_sync_utc) : "-"}</Badge>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">
                Performance charting will be wired to `performance.json` equity_series next (IBKR-ready). Current page already pulls real
                portfolio/orders snapshots.
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

