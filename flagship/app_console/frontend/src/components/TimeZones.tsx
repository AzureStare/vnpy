import React, { useMemo, useState } from "react";
import { useInterval } from "../lib/interval";

type ZoneSpec = {
  key: string;
  label: string;
  timeZone: string;
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

export function TimeZones() {
  const [now, setNow] = useState<Date>(() => new Date());

  useInterval(() => {
    setNow(new Date());
  }, 1000);

  const zones = useMemo<ZoneSpec[]>(
    () => [
      { key: "cn", label: "CN", timeZone: "Asia/Shanghai" },
      { key: "et", label: "ET", timeZone: "America/New_York" },
      { key: "utc", label: "UTC", timeZone: "UTC" },
    ],
    []
  );

  return (
    <div className="hidden items-center gap-2 lg:flex">
      {zones.map((z) => {
        const offset = _formatOffset(_offsetMinutes(now, z.timeZone));
        const ts = _formatDateTime(now, z.timeZone);
        return (
          <div key={z.key} className="rounded-md bg-white px-3 py-2 shadow-sm ring-1 ring-border/70">
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <span className="font-medium">{z.label}</span>
              <span className="tabular-nums">{offset}</span>
            </div>
            <div className="mt-0.5 text-xs font-semibold tabular-nums">{ts}</div>
          </div>
        );
      })}
    </div>
  );
}


