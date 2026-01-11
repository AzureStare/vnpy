"""
Generate a daily trade recap HTML for Ops Console.

Inputs (expected in Ops snapshot dir, default: logs/app):
- orders.json: fetched from Alpaca (via app_console_snapshot.snapshot_orders)
- portfolio.json: fetched from Alpaca (via app_console_snapshot.snapshot_portfolio)

Optional inputs:
- intraday_runner_YYYYMMDD.log: intraday exit reasons (default: logs/)
- portfolio_YYYYMMDD.json: archived EOD portfolio snapshots (for realized PnL cost basis fallback)

Outputs (written to Ops snapshot dir so Caddy can serve under /data):
- trade_recap_YYYYMMDD.html
- trade_recap_latest.html
- trade_recap_index.json
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

from vnpy.trader.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "app"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ExitEvent:
    ts_et: datetime
    vt_symbol: str
    symbol: str
    reason: str
    profit_pct: float | None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=".",
            suffix=".tmp",
        ) as f:
            tmp = Path(f.name)
            f.write(text)
        tmp.replace(path)
        try:
            path.chmod(0o644)
        except Exception:
            pass
    finally:
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=".",
            suffix=".tmp",
        ) as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        try:
            path.chmod(0o644)
        except Exception:
            pass
    finally:
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


def _parse_iso_datetime(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        s = text.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(EASTERN)


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(float(v))
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_prior_portfolio_archive(output_dir: Path, trade_date: date, lookback_days: int = 10) -> Path | None:
    """
    Find the newest portfolio_YYYYMMDD.json strictly before trade_date (ET),
    looking back up to lookback_days (calendar).
    """
    for i in range(1, lookback_days + 1):
        d = trade_date - timedelta(days=i)
        candidate = output_dir / f"portfolio_{d.strftime('%Y%m%d')}.json"
        if candidate.exists():
            return candidate
    return None


def _parse_exit_events(intraday_log_path: Path) -> List[ExitEvent]:
    if not intraday_log_path.exists():
        return []

    # Example:
    # 2026-01-05 11:16:01.089 | INFO | Logger | [IntradayRunner] ASTS.NASDAQ 盘中离场: EMA10 趋势止盈, 当前收益: 8.82%
    exit_re = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}).*?\[IntradayRunner\]\s+"
        r"(?P<vt_symbol>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)\s+盘中离场:\s+(?P<reason>.*?),\s+当前收益:\s+(?P<pct>[-+]?\d+(?:\.\d+)?)%"
    )

    events: List[ExitEvent] = []
    seen: set[tuple[str, str, str]] = set()
    for line in intraday_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = exit_re.search(line)
        if not m:
            continue
        ts_text = m.group("ts")
        vt_symbol = m.group("vt_symbol").strip()
        reason = m.group("reason").strip()
        pct = _safe_float(m.group("pct"))

        # Deduplicate dual logger lines by (ts, vt_symbol, reason)
        key = (ts_text, vt_symbol, reason)
        if key in seen:
            continue
        seen.add(key)

        try:
            ts_naive = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            continue
        ts_et = ts_naive.replace(tzinfo=EASTERN)
        symbol = vt_symbol.split(".", 1)[0].upper()
        events.append(
            ExitEvent(
                ts_et=ts_et,
                vt_symbol=vt_symbol,
                symbol=symbol,
                reason=reason,
                profit_pct=pct,
            )
        )

    events.sort(key=lambda e: e.ts_et)
    return events


def _match_exit_event(exit_events: List[ExitEvent], symbol: str, filled_at_et: datetime, max_delta_seconds: int = 300) -> ExitEvent | None:
    """
    Match the closest exit event for symbol around filled_at_et (ET).
    """
    best: ExitEvent | None = None
    best_delta = None
    for ev in exit_events:
        if ev.symbol != symbol:
            continue
        delta = abs((ev.ts_et - filled_at_et).total_seconds())
        if delta > max_delta_seconds:
            continue
        if best is None or (best_delta is not None and delta < best_delta):
            best = ev
            best_delta = delta
    return best


def _build_recap_index(output_dir: Path, keep_days: int = 30) -> Dict[str, Any]:
    recaps: List[Dict[str, str]] = []
    recap_re = re.compile(r"^trade_recap_(\d{8})\.html$")
    for p in output_dir.glob("trade_recap_*.html"):
        m = recap_re.match(p.name)
        if not m:
            continue
        ymd = m.group(1)
        try:
            d = datetime.strptime(ymd, "%Y%m%d").date()
        except Exception:
            continue
        recaps.append({"trade_date": d.isoformat(), "file": p.name})

    recaps.sort(key=lambda x: x["trade_date"], reverse=True)
    recaps = recaps[: int(keep_days)]

    latest = recaps[0] if recaps else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest": latest,
        "recaps": recaps,
    }


def generate_trade_recap(
    trade_date: date,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    log_dir: Path = DEFAULT_LOG_DIR,
    strategy_version: str = "v7",
) -> Path:
    """
    Generate and write recap HTML + latest + index.
    Returns the written recap path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    orders_path = output_dir / "orders.json"
    portfolio_path = output_dir / "portfolio.json"
    intraday_log_path = log_dir / f"intraday_runner_{trade_date.strftime('%Y%m%d')}.log"

    if not orders_path.exists():
        raise FileNotFoundError(f"missing orders snapshot: {orders_path}")
    if not portfolio_path.exists():
        raise FileNotFoundError(f"missing portfolio snapshot: {portfolio_path}")

    orders_payload = _load_json(orders_path)
    portfolio_payload = _load_json(portfolio_path)
    exit_events = _parse_exit_events(intraday_log_path)

    orders_generated_at = str(orders_payload.get("generated_at") or "")
    portfolio_generated_at = str(portfolio_payload.get("generated_at") or "")

    orders: List[Dict[str, Any]] = list(orders_payload.get("orders") or [])
    positions: List[Dict[str, Any]] = list(portfolio_payload.get("positions") or [])

    # Filter filled orders for this trade_date (ET)
    filled_orders: List[Dict[str, Any]] = []
    for o in orders:
        filled_at = _parse_iso_datetime(o.get("filled_at"))
        if not filled_at:
            continue
        filled_at_et = _to_et(filled_at)
        if filled_at_et.date() != trade_date:
            continue
        status = (o.get("status") or "").lower()
        if status not in ("filled", "partially_filled"):
            continue
        filled_qty = _safe_float(o.get("filled_qty"))
        if not filled_qty or filled_qty <= 0:
            continue
        filled_orders.append(o)

    def _filled_at_et(o: Dict[str, Any]) -> datetime:
        dt = _parse_iso_datetime(o.get("filled_at")) or datetime.now(timezone.utc)
        return _to_et(dt)

    filled_orders.sort(key=_filled_at_et)

    buys: List[Dict[str, Any]] = []
    sells: List[Dict[str, Any]] = []
    for o in filled_orders:
        side = (o.get("side") or "").lower()
        if side == "buy":
            buys.append(o)
        elif side == "sell":
            sells.append(o)

    # Same-day buy avg price per symbol (for realized PnL best-effort)
    same_day_buy_avg: Dict[str, float] = {}
    for o in buys:
        symbol = str(o.get("symbol") or "").upper()
        qty = _safe_float(o.get("filled_qty")) or 0.0
        px = _safe_float(o.get("filled_avg_price"))
        if not symbol or qty <= 0 or px is None or px <= 0:
            continue
        cur = same_day_buy_avg.get(symbol)
        if cur is None:
            # store as weighted sum in temp dict
            same_day_buy_avg[symbol] = px * qty
            same_day_buy_avg[f"__qty__{symbol}"] = qty  # type: ignore[assignment]
        else:
            same_day_buy_avg[symbol] = float(cur) + px * qty
            same_day_buy_avg[f"__qty__{symbol}"] = float(same_day_buy_avg.get(f"__qty__{symbol}", 0.0)) + qty  # type: ignore[assignment]

    for symbol in list(same_day_buy_avg.keys()):
        if symbol.startswith("__qty__"):
            continue
        qty_key = f"__qty__{symbol}"
        total_qty = float(same_day_buy_avg.get(qty_key, 0.0))
        if total_qty > 0:
            same_day_buy_avg[symbol] = float(same_day_buy_avg[symbol]) / total_qty
        same_day_buy_avg.pop(qty_key, None)

    prior_portfolio_path = _find_latest_prior_portfolio_archive(output_dir, trade_date, lookback_days=14)
    prior_avg_entry: Dict[str, float] = {}
    prior_portfolio_date: str | None = None
    if prior_portfolio_path:
        try:
            prior_payload = _load_json(prior_portfolio_path)
            prior_portfolio_date = prior_portfolio_path.stem.replace("portfolio_", "")
            for p in list(prior_payload.get("positions") or []):
                sym = str(p.get("symbol") or "").upper()
                avg_entry = _safe_float(p.get("avg_entry"))
                qty = _safe_float(p.get("qty")) or 0.0
                if sym and qty > 0 and avg_entry and avg_entry > 0:
                    prior_avg_entry[sym] = float(avg_entry)
        except Exception as exc:
            logger.warning(f"[TradeRecap] failed to read prior portfolio archive: {prior_portfolio_path}: {exc}")

    # Build realized rows
    realized_rows: List[Dict[str, Any]] = []
    for o in sells:
        symbol = str(o.get("symbol") or "").upper()
        qty = _safe_float(o.get("filled_qty")) or 0.0
        sell_px = _safe_float(o.get("filled_avg_price"))
        filled_et = _filled_at_et(o)

        exit_ev = _match_exit_event(exit_events, symbol, filled_et) if exit_events else None

        cost_px = None
        cost_src = "unknown"
        if symbol in same_day_buy_avg:
            cost_px = same_day_buy_avg[symbol]
            cost_src = "same_day_buy"
        elif symbol in prior_avg_entry:
            cost_px = prior_avg_entry[symbol]
            cost_src = f"prior_portfolio_{prior_portfolio_date or ''}".strip("_")

        realized_pnl = None
        realized_pct = None
        if sell_px and cost_px and qty > 0:
            realized_pnl = (float(sell_px) - float(cost_px)) * float(qty)
            if cost_px > 0:
                realized_pct = float(sell_px) / float(cost_px) - 1.0

        realized_rows.append(
            {
                "symbol": symbol,
                "filled_at_et": filled_et.strftime("%H:%M:%S"),
                "qty": float(qty),
                "sell_px": sell_px,
                "cost_px": cost_px,
                "cost_source": cost_src,
                "realized_pnl": realized_pnl,
                "realized_pct": realized_pct,
                "exit_reason": exit_ev.reason if exit_ev else None,
                "exit_profit_pct": exit_ev.profit_pct if exit_ev else None,
            }
        )

    # Holdings at snapshot time (post-market)
    holding_rows: List[Dict[str, Any]] = []
    for p in positions:
        sym = str(p.get("symbol") or "").upper()
        qty = _safe_float(p.get("qty")) or 0.0
        if not sym or qty <= 0:
            continue
        avg_entry = _safe_float(p.get("avg_entry"))
        market_value = _safe_float(p.get("market_value"))
        unreal_pnl = _safe_float(p.get("unrealized_pnl"))
        unreal_pct = None
        if avg_entry and avg_entry > 0 and market_value is not None:
            cost_value = float(avg_entry) * float(qty)
            if cost_value > 0:
                unreal_pct = float(market_value) / cost_value - 1.0
        holding_rows.append(
            {
                "symbol": sym,
                "qty": float(qty),
                "avg_entry": avg_entry,
                "market_value": market_value,
                "unrealized_pnl": unreal_pnl,
                "unrealized_pct": unreal_pct,
            }
        )
    holding_rows.sort(key=lambda r: float(r.get("market_value") or 0.0), reverse=True)

    # Buy rows (for timeline)
    buy_rows: List[Dict[str, Any]] = []
    for o in buys:
        sym = str(o.get("symbol") or "").upper()
        qty = _safe_float(o.get("filled_qty")) or 0.0
        px = _safe_float(o.get("filled_avg_price"))
        filled_et = _filled_at_et(o)
        buy_rows.append(
            {
                "symbol": sym,
                "filled_at_et": filled_et.strftime("%H:%M:%S"),
                "qty": float(qty),
                "buy_px": px,
            }
        )

    # HTML generation (keep it self-contained)
    def fmt_money(x: float | None) -> str:
        if x is None:
            return "unknown"
        return f"{x:,.2f}"

    def fmt_px(x: float | None) -> str:
        if x is None:
            return "unknown"
        return f"{x:.4f}"

    def fmt_pct(x: float | None) -> str:
        if x is None:
            return "unknown"
        return f"{x*100:.2f}%"

    realized_total = sum(float(r["realized_pnl"]) for r in realized_rows if r.get("realized_pnl") is not None)
    unreal_total = sum(float(r["unrealized_pnl"]) for r in holding_rows if r.get("unrealized_pnl") is not None)

    html = f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Flagship {strategy_version.upper()} 交易复盘（{trade_date.isoformat()}）</title>
    <style>
      :root {{
        /* Match Ops Console (Tailwind-ish light theme) */
        --bg: #f8fafc;         /* slate-50 */
        --card: #ffffff;
        --border: #e2e8f0;     /* slate-200 */
        --text: #0f172a;       /* slate-900 */
        --muted: #64748b;      /* slate-500 */
        --muted2: #94a3b8;     /* slate-400 */
        --primary: #2563eb;    /* blue-600 */
        --ok: #16a34a;         /* green-600 */
        --bad: #dc2626;        /* red-600 */
      }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans";
        line-height: 1.55;
      }}
      .container {{ max-width: 1100px; margin: 28px auto 60px; padding: 0 18px; }}
      h1 {{ margin: 0 0 8px; font-size: 22px; }}
      .sub {{ margin: 4px 0 16px; color: var(--muted); font-size: 13px; }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        margin-top: 12px;
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
      }}
      h2 {{ margin: 0 0 10px; font-size: 16px; }}
      h3 {{ margin: 12px 0 8px; font-size: 14px; }}
      p {{ margin: 8px 0; }}
      ul {{ margin: 6px 0 10px 18px; }}
      li {{ margin: 4px 0; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
      th, td {{ border: 1px solid var(--border); padding: 8px 10px; vertical-align: top; font-size: 13px; }}
      th {{ text-align: left; color: var(--muted); background: #f1f5f9; }}
      .pill {{
        display: inline-flex;
        gap: 6px;
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        font-size: 12px;
        color: var(--muted);
        background: #ffffff;
      }}
      .pill.ok {{ color: var(--ok); border-color: rgba(22, 163, 74, 0.25); background: rgba(22, 163, 74, 0.06); }}
      .pill.bad {{ color: var(--bad); border-color: rgba(220, 38, 38, 0.25); background: rgba(220, 38, 38, 0.06); }}
      code {{ color: var(--text); background: #f1f5f9; border: 1px solid var(--border); border-radius: 8px; padding: 1px 6px; }}
      @media print {{
        body {{ background: #fff; color: #111; }}
        .card {{ background: #fff; border-color: #ddd; }}
        .pill {{ background: #fff; border-color: #bbb; }}
        th, td {{ border-color: #ddd; }}
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Flagship {strategy_version.upper()} 交易复盘（{trade_date.isoformat()}）</h1>
      <p class="sub">
        数据来源：<span class="pill">{orders_path.name}</span>、<span class="pill">{portfolio_path.name}</span>、
        <span class="pill">{intraday_log_path.name}</span>
      </p>
      <p class="sub">
        快照时间（UTC）：orders=<code>{orders_generated_at or "unknown"}</code> · portfolio=<code>{portfolio_generated_at or "unknown"}</code>
      </p>

      <section class="card">
        <h2>1. 概览</h2>
        <ul>
          <li>当日成交买单数：<b>{len(buy_rows)}</b></li>
          <li>当日成交卖单数：<b>{len(realized_rows)}</b></li>
          <li>Realized PnL (Est.)：<span class="pill {'ok' if realized_total >= 0 else 'bad'}">${fmt_money(realized_total)}</span></li>
          <li>Unrealized PnL (Snapshot)：<span class="pill {'ok' if unreal_total >= 0 else 'bad'}">${fmt_money(unreal_total)}</span></li>
        </ul>
        <p class="sub">说明：Realized PnL 为 best-effort 估算，成本优先同日 BUY，其次前一交易日 portfolio 归档，否则标记 unknown。</p>
      </section>

      <section class="card">
        <h2>2. 开仓/加仓（当日成交 BUY）</h2>
        <table>
          <thead>
            <tr><th>时间(ET)</th><th>Symbol</th><th>Qty</th><th>Filled Avg Px</th></tr>
          </thead>
          <tbody>
"""
    if buy_rows:
        for r in buy_rows:
            html += f"            <tr><td>{r['filled_at_et']}</td><td>{r['symbol']}</td><td>{int(r['qty'])}</td><td>{fmt_px(r['buy_px'])}</td></tr>\n"
    else:
        html += "            <tr><td colspan=\"4\">无</td></tr>\n"

    html += """          </tbody>
        </table>
      </section>

      <section class="card">
        <h2>3. 平仓/减仓（当日成交 SELL）</h2>
        <table>
          <thead>
            <tr>
              <th>时间(ET)</th><th>Symbol</th><th>Qty</th><th>Sell Px</th>
              <th>Cost Px</th><th>Cost Source</th><th>Realized PnL (Est.)</th><th>Exit Reason</th>
            </tr>
          </thead>
          <tbody>
"""
    if realized_rows:
        for r in realized_rows:
            pnl = r.get("realized_pnl")
            pnl_str = fmt_money(pnl) if pnl is not None else "unknown"
            pill_cls = "ok" if (pnl is not None and float(pnl) >= 0) else ("bad" if pnl is not None else "")
            pill_open = f"<span class=\"pill {pill_cls}\">" if pill_cls else "<span class=\"pill\">"
            exit_reason = r.get("exit_reason") or "unknown"
            if r.get("exit_profit_pct") is not None:
                exit_reason = f"{exit_reason} ({float(r['exit_profit_pct']):.2f}%)"
            html += (
                "            <tr>"
                f"<td>{r['filled_at_et']}</td>"
                f"<td>{r['symbol']}</td>"
                f"<td>{int(r['qty'])}</td>"
                f"<td>{fmt_px(r.get('sell_px'))}</td>"
                f"<td>{fmt_px(r.get('cost_px'))}</td>"
                f"<td>{r.get('cost_source') or 'unknown'}</td>"
                f"<td>{pill_open}${pnl_str}</span></td>"
                f"<td>{exit_reason}</td>"
                "</tr>\n"
            )
    else:
        html += "            <tr><td colspan=\"8\">无</td></tr>\n"

    html += """          </tbody>
        </table>
      </section>

      <section class="card">
        <h2>4. 收盘持仓（Snapshot）</h2>
        <table>
          <thead>
            <tr><th>Symbol</th><th>Qty</th><th>Avg Entry</th><th>Market Value</th><th>Unrealized PnL</th><th>Unrealized %</th></tr>
          </thead>
          <tbody>
"""
    if holding_rows:
        for r in holding_rows:
            upnl = r.get("unrealized_pnl")
            pill_cls = "ok" if (upnl is not None and float(upnl) >= 0) else ("bad" if upnl is not None else "")
            pill_open = f"<span class=\"pill {pill_cls}\">" if pill_cls else "<span class=\"pill\">"
            html += (
                "            <tr>"
                f"<td>{r['symbol']}</td>"
                f"<td>{int(r['qty'])}</td>"
                f"<td>{fmt_px(r.get('avg_entry'))}</td>"
                f"<td>${fmt_money(r.get('market_value'))}</td>"
                f"<td>{pill_open}${fmt_money(upnl)}</span></td>"
                f"<td>{fmt_pct(r.get('unrealized_pct'))}</td>"
                "</tr>\n"
            )
    else:
        html += "            <tr><td colspan=\"6\">无</td></tr>\n"

    html += f"""          </tbody>
        </table>
      </section>

      <section class="card">
        <h2>5. 盘中离场触发（来自 intraday_runner 日志）</h2>
        <table>
          <thead>
            <tr><th>时间(ET)</th><th>VT Symbol</th><th>原因</th><th>收益(估)</th></tr>
          </thead>
          <tbody>
"""
    if exit_events:
        for ev in exit_events:
            pct = f"{ev.profit_pct:.2f}%" if ev.profit_pct is not None else "unknown"
            pill_cls = "ok" if (ev.profit_pct is not None and ev.profit_pct >= 0) else ("bad" if ev.profit_pct is not None else "")
            pill_open = f"<span class=\"pill {pill_cls}\">" if pill_cls else "<span class=\"pill\">"
            html += (
                "            <tr>"
                f"<td>{ev.ts_et.strftime('%H:%M:%S')}</td>"
                f"<td>{ev.vt_symbol}</td>"
                f"<td>{ev.reason}</td>"
                f"<td>{pill_open}{pct}</span></td>"
                "</tr>\n"
            )
    else:
        html += "            <tr><td colspan=\"4\">无（可能未启动 intraday_runner 或当日无盘中离场）</td></tr>\n"

    html += f"""          </tbody>
        </table>
        <p class="sub">日志文件：{intraday_log_path}</p>
      </section>

      <section class="card">
        <h2>6. 文件落地位置</h2>
        <ul>
          <li><b>复盘 HTML</b>：{output_dir}/trade_recap_{trade_date.strftime('%Y%m%d')}.html</li>
          <li><b>Latest</b>：{output_dir}/trade_recap_latest.html</li>
          <li><b>Index</b>：{output_dir}/trade_recap_index.json</li>
          <li><b>订单快照</b>：{orders_path}</li>
          <li><b>组合快照</b>：{portfolio_path}</li>
        </ul>
      </section>
    </div>
  </body>
</html>
"""

    recap_path = output_dir / f"trade_recap_{trade_date.strftime('%Y%m%d')}.html"
    _atomic_write_text(recap_path, html)
    _atomic_write_text(output_dir / "trade_recap_latest.html", html)

    index_payload = _build_recap_index(output_dir)
    _atomic_write_json(output_dir / "trade_recap_index.json", index_payload)

    logger.info(f"[TradeRecap] wrote {recap_path}")
    return recap_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily trade recap HTML (published under /data).")
    parser.add_argument("--trade-date", type=str, help="YYYY-MM-DD (ET). Defaults to today ET.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Ops snapshot output dir (default: logs/app).")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Logs dir containing intraday_runner logs.")
    parser.add_argument("--strategy", type=str, default="v7")
    args = parser.parse_args()

    if args.trade_date:
        trade_date = datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    else:
        trade_date = datetime.now(tz=EASTERN).date()

    out_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    log_dir = args.log_dir if args.log_dir.is_absolute() else (PROJECT_ROOT / args.log_dir)

    generate_trade_recap(
        trade_date=trade_date,
        output_dir=out_dir,
        log_dir=log_dir,
        strategy_version=str(args.strategy),
    )


if __name__ == "__main__":
    main()


