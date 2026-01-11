import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { useInterval } from "../lib/interval";

type MarketStatus = Record<string, unknown> & {
  generated_at?: string;
  market?: string;
  next_open?: string | null;
  seconds_to_open?: number | null;
  early_hours?: unknown;
  after_hours?: unknown;
  error?: string;
};

function _pad2(v: number): string {
  return String(v).padStart(2, "0");
}

function _partsForZone(now: Date, timeZone: string): Record<string, string> {
  const dtf = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  const out: Record<string, string> = {};
  for (const p of dtf.formatToParts(now)) {
    if (p.type === "year" || p.type === "month" || p.type === "day" || p.type === "hour" || p.type === "minute" || p.type === "second") {
      out[p.type] = p.value;
    }
  }
  return out;
}

function _offsetMinutes(now: Date, timeZone: string): number {
  const parts = _partsForZone(now, timeZone);
  const year = Number(parts.year);
  const month = Number(parts.month);
  const day = Number(parts.day);
  const hour = Number(parts.hour);
  const minute = Number(parts.minute);
  const second = Number(parts.second);
  const asUtcMs = Date.UTC(year, month - 1, day, hour, minute, second);
  return Math.round((asUtcMs - now.getTime()) / 60000);
}

function _formatOffset(offsetMinutes: number): string {
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const hh = Math.floor(abs / 60);
  const mm = abs % 60;
  if (mm === 0) return `UTC${sign}${hh}`;
  return `UTC${sign}${hh}:${_pad2(mm)}`;
}

function _formatDateTime(now: Date, timeZone: string): string {
  const p = _partsForZone(now, timeZone);
  const yyyy = p.year || "0000";
  const mm = p.month || "00";
  const dd = p.day || "00";
  const hh = p.hour || "00";
  const mi = p.minute || "00";
  const ss = p.second || "00";
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
}

function _normalizeMarketLabel(raw: unknown): string {
  const s = String(raw || "").trim();
  if (!s) return "-";
  return s.replace(/[_-]+/g, " ");
}

export function MarketTimeStatus() {
  const [market, setMarket] = useState<MarketStatus | null>(null);
  const tz = "America/New_York";

  const load = useCallback(async () => {
    try {
      const ms = await fetchJson<MarketStatus>("/data/market_status.json", { method: "GET" });
      setMarket(ms);
    } catch (_) {
      // ignore transient; keep last data
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useInterval(() => {
    void load();
  }, 60_000);

  const marketLabel = useMemo(() => {
    if (market?.error) return "error";
    return _normalizeMarketLabel(market?.market);
  }, [market]);

  const marketOffset = useMemo(() => {
    return _formatOffset(_offsetMinutes(new Date(), tz));
  }, [market?.generated_at, tz]);

  const nextOpenEt = useMemo(() => {
    const iso = market?.next_open;
    if (!iso) return "-";
    const d = new Date(String(iso));
    if (!Number.isFinite(d.valueOf())) return String(iso);
    return _formatDateTime(d, tz);
  }, [market]);

  const nextOpenOffset = useMemo(() => {
    const iso = market?.next_open;
    if (!iso) return marketOffset;
    const d = new Date(String(iso));
    if (!Number.isFinite(d.valueOf())) return marketOffset;
    return _formatOffset(_offsetMinutes(d, tz));
  }, [market?.next_open, marketOffset, tz]);

  return (
    <div className="hidden items-center gap-2 lg:flex">
      <div className="rounded-md bg-white px-3 py-2 shadow-sm ring-1 ring-border/70">
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <span className="font-medium">Market status</span>
          <span className="tabular-nums">ET</span>
          <span className="tabular-nums">{marketOffset}</span>
        </div>
        <div className="mt-0.5 text-xs font-semibold tabular-nums">{marketLabel}</div>
      </div>

      <div className="rounded-md bg-white px-3 py-2 shadow-sm ring-1 ring-border/70">
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <span className="font-medium">Next open</span>
          <span className="tabular-nums">ET</span>
          <span className="tabular-nums">{nextOpenOffset}</span>
        </div>
        <div className="mt-0.5 text-xs font-semibold tabular-nums">{nextOpenEt}</div>
      </div>
    </div>
  );
}

