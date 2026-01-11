"""
Generate daily performance report for Ops Console (with GPT summary).

Outputs:
- logs/app/report_latest.html
- logs/app/report_YYYYMMDD.html (archived)

Design:
- Reads performance.json and orders.json from snapshot
- Computes KPIs (return, max drawdown, Sharpe, win rate, etc.)
- Calls GPT API to generate summary (falls back to template if key missing)
- Generates HTML report similar to backtest report style
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "app"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_LLM_MODEL = os.getenv("FLAGSHIP_APP_REPORT_LLM_MODEL", "gpt-5.2")
DEFAULT_LLM_TIMEOUT_SECONDS = int(os.getenv("FLAGSHIP_APP_REPORT_LLM_TIMEOUT_SECONDS", "30"))
DEFAULT_LLM_MAX_COMPLETION_TOKENS = int(os.getenv("FLAGSHIP_APP_REPORT_LLM_MAX_COMPLETION_TOKENS", "1200"))


def _load_json_snapshot(output_dir: Path, filename: str) -> Dict[str, Any] | None:
    path = output_dir / filename
    if not path.exists():
        logger.warning(f"[app_console_report] {filename} not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"[app_console_report] failed to load {filename}: {exc}")
        return None


def _compute_kpis(performance_data: Dict[str, Any] | None, orders_data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Compute performance KPIs from snapshot data"""
    kpis: Dict[str, Any] = {
        "total_return": None,
        "max_drawdown": None,
        "sharpe_ratio": None,
        "win_rate": None,
        "total_trades": 0,
        "total_orders": 0,
        "start_date": None,
        "end_date": None,
        "start_equity": None,
        "end_equity": None,
    }

    if not performance_data or not performance_data.get("equity_series"):
        return kpis

    series = performance_data["equity_series"]
    if len(series) < 2:
        return kpis

    equities = [s["equity"] for s in series if s.get("equity") is not None]
    if len(equities) < 2:
        return kpis

    first_equity = equities[0]
    last_equity = equities[-1]
    kpis["start_equity"] = first_equity
    kpis["end_equity"] = last_equity
    kpis["total_return"] = (last_equity - first_equity) / first_equity if first_equity > 0 else 0.0

    # Parse dates
    try:
        kpis["start_date"] = series[0]["date"]
        kpis["end_date"] = series[-1]["date"]
    except Exception:
        pass

    # Max Drawdown
    peak = first_equity
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    kpis["max_drawdown"] = max_dd

    # Sharpe Ratio (simplified: assume 252 trading days, risk-free = 0.02)
    returns = []
    for i in range(1, len(equities)):
        ret = (equities[i] - equities[i - 1]) / equities[i - 1] if equities[i - 1] > 0 else 0.0
        returns.append(ret)

    if len(returns) > 1:
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = variance ** 0.5
        if std_ret > 0:
            # Annualized Sharpe
            kpis["sharpe_ratio"] = (mean_ret * 252 - 0.02) / (std_ret * (252 ** 0.5))
        else:
            kpis["sharpe_ratio"] = 0.0

    # Win Rate & Trades (from orders)
    if orders_data and orders_data.get("orders"):
        orders = orders_data["orders"]
        kpis["total_orders"] = len(orders)
        filled = [o for o in orders if o.get("status") in ("filled", "partially_filled")]
        kpis["total_trades"] = len(filled)

        # Simplified win rate: assume sell orders are wins (would need trade PnL for real calculation)
        wins = [o for o in filled if o.get("side") == "sell"]
        if len(filled) > 0:
            kpis["win_rate"] = len(wins) / len(filled)
        else:
            kpis["win_rate"] = None
    else:
        kpis["win_rate"] = None

    return kpis


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _get_openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()

    # Prefer project root vt_setting.json
    try:
        vt_setting_path = PROJECT_ROOT / "vt_setting.json"
        if vt_setting_path.exists():
            data = json.loads(vt_setting_path.read_text(encoding="utf-8"))
            key_local = data.get("open-ai.api_key")
            if isinstance(key_local, str) and key_local.strip():
                return key_local.strip()
    except Exception:
        pass

    try:
        from vnpy.trader.setting import SETTINGS
    except Exception:
        return None

    key2 = SETTINGS.get("open-ai.api_key")
    if isinstance(key2, str) and key2.strip():
        return key2.strip()
    return None


def _summarize_with_openai(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    max_completion_tokens: int,
) -> str:
    """Call OpenAI Chat Completions API via stdlib (no extra deps)."""
    import urllib.request

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        # gpt-5.x: chat/completions uses max_completion_tokens (max_tokens is rejected)
        "max_completion_tokens": int(max_completion_tokens),
    }

    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return str(data["choices"][0]["message"]["content"] or "").strip()


def _build_summary_prompt(kpis: Dict[str, Any]) -> str:
    win_rate = kpis.get("win_rate")
    win_rate_text = f"{float(win_rate):.2%}" if isinstance(win_rate, (int, float)) else "N/A"

    total_return = kpis.get("total_return")
    total_return_text = f"{float(total_return):.2%}" if isinstance(total_return, (int, float)) else "N/A"

    max_dd = kpis.get("max_drawdown")
    max_dd_text = f"{float(max_dd):.2%}" if isinstance(max_dd, (int, float)) else "N/A"

    sharpe = kpis.get("sharpe_ratio")
    sharpe_text = f"{float(sharpe):.2f}" if isinstance(sharpe, (int, float)) else "N/A"

    start_eq = kpis.get("start_equity")
    end_eq = kpis.get("end_equity")
    start_eq_text = f"${float(start_eq):,.2f}" if isinstance(start_eq, (int, float)) else "N/A"
    end_eq_text = f"${float(end_eq):,.2f}" if isinstance(end_eq, (int, float)) else "N/A"

    return (
        "请用中文总结以下交易账户的表现数据，并给出简洁的分析与建议（200字以内）：\n\n"
        f"时间范围：{kpis.get('start_date', 'N/A')} 至 {kpis.get('end_date', 'N/A')}\n"
        f"初始权益：{start_eq_text}\n"
        f"最终权益：{end_eq_text}\n"
        f"总收益率：{total_return_text}\n"
        f"最大回撤：{max_dd_text}\n"
        f"夏普比率：{sharpe_text}\n"
        f"胜率：{win_rate_text}\n"
        f"总交易次数：{kpis.get('total_trades', 0)}\n"
        f"总订单数：{kpis.get('total_orders', 0)}\n\n"
        "请提供：\n"
        "1. 简要表现评价\n"
        "2. 主要风险点\n"
        "3. 改进建议"
    )


def _call_gpt_summary(kpis: Dict[str, Any]) -> str:
    api_key = _get_openai_api_key()
    if not api_key:
        logger.warning("[app_console_report] OpenAI API key not found, using template summary")
        return _template_summary(kpis)

    system_prompt = "你是一位专业的量化交易分析师，擅长用简洁的中文总结交易表现。"
    prompt = _build_summary_prompt(kpis)

    try:
        out = _summarize_with_openai(
            api_key=api_key,
            model=DEFAULT_LLM_MODEL,
            system_prompt=system_prompt,
            user_prompt=prompt,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            max_completion_tokens=DEFAULT_LLM_MAX_COMPLETION_TOKENS,
        )
        if not out:
            retry_tokens = min(max(DEFAULT_LLM_MAX_COMPLETION_TOKENS * 2, DEFAULT_LLM_MAX_COMPLETION_TOKENS + 1000), 5000)
            out = _summarize_with_openai(
                api_key=api_key,
                model=DEFAULT_LLM_MODEL,
                system_prompt=system_prompt,
                user_prompt=prompt,
                timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
                max_completion_tokens=retry_tokens,
            )
        return out or _template_summary(kpis)
    except Exception as exc:
        logger.warning(f"[app_console_report] GPT API call failed: {exc}, using template summary")
        return _template_summary(kpis)


def _template_summary(kpis: Dict[str, Any]) -> str:
    """Fallback template summary when GPT is unavailable"""
    total_return = kpis.get("total_return", 0.0)
    max_dd = kpis.get("max_drawdown", 0.0)
    sharpe = kpis.get("sharpe_ratio", 0.0)
    win_rate = kpis.get("win_rate")

    summary_parts = []
    summary_parts.append(f"账户表现：总收益率 {total_return:.2%}，最大回撤 {max_dd:.2%}。")
    if sharpe > 1.0:
        summary_parts.append("夏普比率表现良好，风险调整后收益较优。")
    elif sharpe < 0.5:
        summary_parts.append("夏普比率偏低，建议优化风险控制。")
    if win_rate is not None:
        if win_rate > 0.6:
            summary_parts.append(f"胜率 {win_rate:.1%} 较高，交易策略表现稳定。")
        elif win_rate < 0.4:
            summary_parts.append(f"胜率 {win_rate:.1%} 偏低，建议优化入场/出场逻辑。")
    summary_parts.append("建议持续监控回撤水平，适时调整仓位与止损策略。")

    return " ".join(summary_parts)


def _generate_html_report(
    kpis: Dict[str, Any],
    gpt_summary: str,
    performance_data: Dict[str, Any] | None,
    orders_data: Dict[str, Any] | None,
    output_path: Path,
) -> None:
    """Generate HTML report similar to backtest report style"""
    now = datetime.now(timezone.utc)

    # Format KPIs
    def fmt_pct(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v:.2%}"

    def fmt_num(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v:,.2f}"

    # Build equity chart data (simple SVG/Canvas, or use inline plotly)
    chart_html = ""
    if performance_data and performance_data.get("equity_series"):
        series = performance_data["equity_series"]
        if len(series) >= 2:
            # Simple inline chart using SVG
            equities = [s["equity"] for s in series]
            min_eq = min(equities)
            max_eq = max(equities)
            range_eq = max_eq - min_eq or 1

            points = []
            for i, s in enumerate(series):
                x = 50 + (i / (len(series) - 1)) * 700 if len(series) > 1 else 50
                y = 250 - ((s["equity"] - min_eq) / range_eq) * 200
                points.append(f"{x},{y}")

            chart_html = f"""
            <section class="card">
                <h2>Equity Curve</h2>
                <svg width="800" height="300" class="chart">
                    <polyline points="{' '.join(points)}" fill="none" stroke="var(--primary)" stroke-width="2"/>
                    <text x="400" y="290" text-anchor="middle" font-size="12" fill="var(--muted)">Date</text>
                    <text x="20" y="150" text-anchor="middle" font-size="12" fill="var(--muted)" transform="rotate(-90 20 150)">Equity</text>
                </svg>
            </section>
            """

    # Orders summary table
    orders_html = ""
    if orders_data and orders_data.get("orders"):
        orders = orders_data["orders"][:50]  # Top 50 most recent
        orders_rows = ""
        for o in orders:
            dt = o.get("submitted_at", "-")
            if dt and dt != "-":
                try:
                    dt = datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            orders_rows += f"""
            <tr>
                <td>{dt}</td>
                <td>{o.get('symbol', '-')}</td>
                <td>{o.get('side', '-')}</td>
                <td>{o.get('qty', 0):.0f}</td>
                <td>{o.get('filled_qty', 0):.0f}</td>
                <td>{o.get('status', '-')}</td>
            </tr>
            """
        orders_html = f"""
        <section class="card">
            <h2>Recent Orders (Top 50)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th class="tr">Qty</th>
                        <th class="tr">Filled</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {orders_rows}
                </tbody>
            </table>
        </section>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flagship Ops Console - Performance Report</title>
    <style>
        :root {{
            --bg: #f8fafc;         /* slate-50 */
            --card: #ffffff;
            --border: #e2e8f0;     /* slate-200 */
            --text: #0f172a;       /* slate-900 */
            --muted: #64748b;      /* slate-500 */
            --muted2: #94a3b8;     /* slate-400 */
            --primary: #2563eb;    /* blue-600 */
            --ok: #16a34a;         /* green-600 */
            --bad: #dc2626;        /* red-600 */
            --radius: 14px;
        }}

        body {{
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans";
            line-height: 1.55;
        }}

        .container {{
            max-width: 1100px;
            margin: 28px auto 60px;
            padding: 0 18px;
        }}

        .header {{
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
        }}

        h1 {{
            margin: 0;
            font-size: 22px;
        }}

        h2 {{
            margin: 0 0 10px;
            font-size: 16px;
        }}

        .meta {{
            color: var(--muted);
            font-size: 13px;
            margin: 0;
            line-height: 1.6;
        }}

        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            margin-top: 12px;
            box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }}
        .kpi-card {{
            background: rgba(241, 245, 249, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
        }}
        .kpi-label {{
            font-size: 12px;
            color: var(--muted);
        }}
        .kpi-value {{
            font-size: 20px;
            font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            margin-top: 6px;
        }}
        .summary-box {{
            background: rgba(37, 99, 235, 0.06);
            border: 1px solid rgba(37, 99, 235, 0.25);
            border-radius: 12px;
            padding: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 8px 10px;
            border: 1px solid var(--border);
            text-align: left;
            font-size: 13px;
        }}
        th {{
            background: #f1f5f9;
            color: var(--muted);
            font-weight: 600;
        }}

        .tr {{
            text-align: right;
        }}

        .chart {{
            border: 1px solid var(--border);
            background: var(--card);
            border-radius: 12px;
            width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Flagship Ops Console - Performance Report</h1>
            <p class="meta">
                Generated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                Period: {kpis.get('start_date', 'N/A')} to {kpis.get('end_date', 'N/A')}
            </p>
        </div>

        <section class="card">
            <h2>Performance Metrics</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Return</div>
                    <div class="kpi-value" style="color: {'var(--ok)' if (kpis.get('total_return', 0) or 0) >= 0 else 'var(--bad)'}">
                        {fmt_pct(kpis.get('total_return'))}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Max Drawdown</div>
                    <div class="kpi-value" style="color: var(--bad)">
                        {fmt_pct(kpis.get('max_drawdown'))}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Sharpe Ratio</div>
                    <div class="kpi-value">
                        {fmt_num(kpis.get('sharpe_ratio'))}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Win Rate</div>
                    <div class="kpi-value">
                        {fmt_pct(kpis.get('win_rate'))}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Trades</div>
                    <div class="kpi-value">
                        {kpis.get('total_trades', 0)}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Start Equity</div>
                    <div class="kpi-value">
                        ${fmt_num(kpis.get('start_equity'))}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">End Equity</div>
                    <div class="kpi-value">
                        ${fmt_num(kpis.get('end_equity'))}
                    </div>
                </div>
            </div>
        </section>

        <section class="card">
            <h2>Summary</h2>
            <div class="summary-box">
                {_escape_html(gpt_summary).replace(chr(10), '<br>')}
            </div>
        </section>

        {chart_html}

        {orders_html}
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"[app_console_report] wrote {output_path}")


def generate_report(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Generate performance report from snapshots"""
    performance_data = _load_json_snapshot(output_dir, "performance.json")
    orders_data = _load_json_snapshot(output_dir, "orders.json")

    kpis = _compute_kpis(performance_data, orders_data)
    gpt_summary = _call_gpt_summary(kpis)

    # Generate latest report
    latest_path = output_dir / "report_latest.html"
    _generate_html_report(kpis, gpt_summary, performance_data, orders_data, latest_path)

    # Generate dated archive
    today = date.today()
    archive_path = output_dir / f"report_{today.strftime('%Y%m%d')}.html"
    _generate_html_report(kpis, gpt_summary, performance_data, orders_data, archive_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ops Console performance report")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for report")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir

    generate_report(args.output_dir)


if __name__ == "__main__":
    main()

