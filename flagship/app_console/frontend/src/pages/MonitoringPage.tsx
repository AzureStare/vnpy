import React, { useMemo } from "react";
import { Badge } from "../components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";

function inferSubdomainUrl(sub: string): string {
  const proto = window.location.protocol || "https:";
  const host = window.location.hostname || "";
  if (host.startsWith("app.")) {
    return `${proto}//${sub}.${host.slice(4)}`;
  }
  return `${proto}//${host}`;
}

export function MonitoringPage() {
  const grafana = useMemo(() => inferSubdomainUrl("monitor"), []);
  const prom = useMemo(() => inferSubdomainUrl("metrics"), []);
  const db = useMemo(() => inferSubdomainUrl("db"), []);

  return (
    <div className="grid grid-cols-12 gap-6">
      <Card className="col-span-12">
        <CardHeader>
          <div>
            <CardTitle>Monitoring</CardTitle>
            <CardDescription>Grafana / Prometheus / PGAdmin quick links</CardDescription>
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
          </div>

          <div className="mt-4 text-xs text-muted-foreground">
            If you are not under <code className="font-mono">app.*</code> domain (local/dev), links fall back to current
            host.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

