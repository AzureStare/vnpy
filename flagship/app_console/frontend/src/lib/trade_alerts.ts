export type TradeAlertsState = {
  enabled: boolean;
  muted: boolean;
  initialized: boolean;
  seen: Record<string, string>; // order_id -> filled_at
};

export const TRADE_ALERTS_KEY = "flagship_trade_alerts_state_v1";

export function loadTradeAlertsState(): TradeAlertsState {
  try {
    const raw = localStorage.getItem(TRADE_ALERTS_KEY);
    const obj = raw ? (JSON.parse(raw) as Partial<TradeAlertsState>) : {};
    return {
      enabled: !!obj.enabled,
      muted: !!obj.muted,
      initialized: !!obj.initialized,
      seen: obj.seen && typeof obj.seen === "object" ? (obj.seen as Record<string, string>) : {},
    };
  } catch (_) {
    return { enabled: false, muted: false, initialized: false, seen: {} };
  }
}

export function saveTradeAlertsState(state: TradeAlertsState): void {
  try {
    localStorage.setItem(TRADE_ALERTS_KEY, JSON.stringify(state));
  } catch (_) {
    // ignore
  }
}

