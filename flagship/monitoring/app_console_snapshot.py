"""
Generate JSON snapshots for the Ops Console static page.

Outputs:
- logs/app/portfolio.json
- logs/app/selection.json
- logs/app/orders.json
- logs/app/performance.json
- logs/app/accounts.json

Design:
- Portfolio is fetched from the single Alpaca account configured in vt_setting.json.
- Selection uses the latest daily_signal.parquet (by datetime max) joined with
  Postgres daily_selection on the same trade_date, sorted by signal desc.
- Orders: fetched from Alpaca order history.
- Performance: account equity time series from Alpaca portfolio history.
- Accounts: best-effort aggregation for Ops Console multi-account UI (single-account compatible).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

import polars as pl

from flagship.trading.execution.broker_alpaca import AlpacaAdapter
from flagship.config import PROJECT_ROOT
from flagship.trading.execution.broker_ibkr import IbkrAdapter, IbkrConnection, IbkrImportError
from flagship.trading.config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, LAB_PATH, SETTINGS as VT_SETTINGS
from flagship.trading.config import load_ibkr_accounts
from flagship.universe.pg_ticker_db import get_pg_connection
from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "app"
DEFAULT_SELECTION_TOP_N = int(os.getenv("FLAGSHIP_APP_SELECTION_TOPN", "10"))
DEFAULT_SELECTION_DATE_CANDIDATES = int(os.getenv("FLAGSHIP_APP_SELECTION_DATE_CANDIDATES", "30"))
DEFAULT_MASSIVE_TIMEOUT_SECONDS = int(os.getenv("FLAGSHIP_APP_MASSIVE_TIMEOUT", "10"))
DEFAULT_MARKET_OPEN_ET = dtime(9, 30)
DEFAULT_MONITOR_TOP_K = int(os.getenv("FLAGSHIP_APP_MONITOR_TOPK", "200"))
DEFAULT_MONITOR_WINDOWS = (1, 3, 5)
EASTERN = ZoneInfo("America/New_York")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=".", suffix=".tmp"
        ) as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        try:
            # Make snapshots readable by static file servers and ops users.
            path.chmod(0o644)
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] write {path} failed: {exc}")
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_float(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except Exception:
        return None

def _count_open_orders(orders_payload: dict | None) -> int:
    if not orders_payload:
        return 0
    items = orders_payload.get("orders")
    if not isinstance(items, list):
        return 0
    cnt = 0
    for o in items:
        if not isinstance(o, dict):
            continue
        status = str(o.get("status") or "").lower()
        if status in ("new", "accepted", "pending_new", "submitted", "held", "partially_filled"):
            cnt += 1
    return cnt

def _load_json_file(path: Path) -> dict | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def snapshot_accounts(output_dir: Path) -> Dict[str, Any]:
    """
    Generate logs/app/accounts.json for the Accounts page.

    Convention:
    - Default account uses output_dir/{portfolio,orders,performance}.json and data_base_path=/data
    - Additional accounts can be written to output_dir/accounts/<account_id>/*.json and will be included automatically.
    """
    default_id = os.getenv("FLAGSHIP_DEFAULT_ACCOUNT_ID") or "alpaca_paper_main"
    default_name = os.getenv("FLAGSHIP_DEFAULT_ACCOUNT_NAME") or "Alpaca Paper (Main)"

    accounts: list[dict[str, Any]] = []

    # Map IBKR accounts (configured in vt_setting.json) so Accounts page can show broker/name correctly.
    # NOTE: per-account snapshots for IBKR are generated under logs/app/accounts/<account_id>/ by snapshot_ibkr_account().
    ibkr_name_by_account_id: dict[str, str] = {}
    try:
        for cfg in load_ibkr_accounts():
            ibkr_name_by_account_id[str(cfg.account_id)] = str(cfg.display_name or cfg.account_id)
    except Exception:
        ibkr_name_by_account_id = {}

    def _build_one(account_id: str, display_name: str, broker: str, env: str, base_dir: Path, data_base_path: str) -> dict[str, Any] | None:
        pf = _load_json_file(base_dir / "portfolio.json") or {}
        od = _load_json_file(base_dir / "orders.json") or {}
        pr = _load_json_file(base_dir / "performance.json") or {}

        acct = pf.get("account") if isinstance(pf.get("account"), dict) else {}
        positions = pf.get("positions") if isinstance(pf.get("positions"), list) else []
        status = str(acct.get("status") or "unknown") if isinstance(acct, dict) else "unknown"
        last_sync = pf.get("generated_at") or od.get("generated_at") or pr.get("generated_at")

        equity = _to_float(acct.get("equity") if isinstance(acct, dict) else None)
        cash = _to_float(acct.get("cash") if isinstance(acct, dict) else None)
        buying_power = _to_float(acct.get("buying_power") if isinstance(acct, dict) else None)

        today_pnl: float | None = None
        series = pr.get("equity_series")
        if isinstance(series, list) and len(series) >= 2:
            try:
                last = series[-1]
                prev = series[-2]
                if isinstance(last, dict) and isinstance(prev, dict):
                    last_eq = _to_float(last.get("equity"))
                    prev_eq = _to_float(prev.get("equity"))
                    if last_eq is not None and prev_eq is not None:
                        today_pnl = last_eq - prev_eq
            except Exception:
                today_pnl = None

        return {
            "account_id": account_id,
            "display_name": display_name,
            "broker": broker,
            "env": env,
            "status": status,
            "last_sync_utc": str(last_sync) if last_sync else None,
            "equity_usd": equity,
            "cash_usd": cash,
            "buying_power_usd": buying_power,
            "positions_count": len(positions) if isinstance(positions, list) else 0,
            "open_orders_count": _count_open_orders(od),
            "today_pnl_usd": today_pnl,
            "data_base_path": data_base_path,
        }

    # Default account
    one = _build_one(str(default_id), str(default_name), "alpaca", "paper", output_dir, "/data")
    if one:
        accounts.append(one)

    # Additional accounts (if present)
    accounts_root = output_dir / "accounts"
    if accounts_root.exists() and accounts_root.is_dir():
        for child in sorted(accounts_root.iterdir()):
            if not child.is_dir():
                continue
            account_id = child.name
            if account_id == str(default_id):
                continue
            is_ibkr = account_id in ibkr_name_by_account_id
            broker = "ibkr" if is_ibkr else "unknown"
            display_name = ibkr_name_by_account_id.get(account_id, account_id)
            one = _build_one(account_id, display_name, broker, "paper", child, f"/data/accounts/{account_id}")
            if one:
                accounts.append(one)

    data = {
        "generated_at": _now_iso(),
        "accounts": accounts,
    }
    target = output_dir / "accounts.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target} ({len(accounts)} accounts)")
    return data


def snapshot_ibkr_account(output_dir: Path, *, account_id: str, display_name: str, host: str, port: int, client_id: int) -> Dict[str, Any] | None:
    """
    Generate per-account snapshots under:
      logs/app/accounts/<account_id>/{portfolio,orders,performance}.json
    """
    try:
        adapter = IbkrAdapter(
            IbkrConnection(
                host=host,
                port=int(port),
                client_id=int(client_id),
                account_id=account_id,
                display_name=display_name,
                paper=True,
            )
        )
    except IbkrImportError as exc:
        logger.warning(f"[app_console_snapshot] IBKR disabled (missing deps): {exc}")
        return None
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] IBKR connect failed for {account_id}: {exc}")
        return None

    base = output_dir / "accounts" / account_id
    base.mkdir(parents=True, exist_ok=True)

    # Portfolio
    try:
        info = adapter.get_account_info()
        positions = []
        pos = adapter.get_positions()
        for sym, qty in sorted(pos.items()):
            positions.append({"symbol": sym, "qty": int(qty)})
        pf = {
            "generated_at": _now_iso(),
            "account": {"cash": float(info.cash), "equity": float(info.equity), "buying_power": float(info.buying_power), "status": "connected"},
            "positions": positions,
        }
        _atomic_write_json(base / "portfolio.json", pf)
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] IBKR portfolio failed for {account_id}: {exc}")

    # Orders/Performance: placeholder best-effort (ib_insync order/perf APIs vary by account perms)
    try:
        # Open trades/orders
        orders = []
        for tr in adapter.ib.trades():
            try:
                o = tr.order
                c = tr.contract
                orders.append(
                    {
                        "id": str(getattr(o, "orderId", "")),
                        "symbol": getattr(c, "symbol", None),
                        "side": getattr(o, "action", None),
                        "qty": float(getattr(o, "totalQuantity", 0) or 0),
                        "filled_qty": float(getattr(o, "filledQuantity", 0) or 0),
                        "filled_avg_price": None,
                        "status": str(getattr(tr, "orderStatus", None) or ""),
                        "submitted_at": None,
                        "filled_at": None,
                        "canceled_at": None,
                    }
                )
            except Exception:
                continue
        od = {"generated_at": _now_iso(), "orders": orders}
        _atomic_write_json(base / "orders.json", od)
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] IBKR orders failed for {account_id}: {exc}")

    try:
        # Performance: we don't have a unified equity history source from IBKR in this repo yet.
        pr = {"generated_at": _now_iso(), "equity_series": []}
        _atomic_write_json(base / "performance.json", pr)
    except Exception:
        pass

    try:
        adapter.disconnect()
    except Exception:
        pass

    return {"account_id": account_id, "base_dir": str(base)}


def _parse_iso_datetime(text: str | None) -> datetime | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        # Support common ISO format with trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _parse_hhmm(text: str | None) -> dtime | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    return None


def _get_massive_api_key() -> str | None:
    api_key = (
        os.getenv("MASSIVE_API_KEY")
        or os.getenv("POLYGON_API_KEY")
        or str(VT_SETTINGS.get("datafeed.password") or "").strip()
    )
    api_key = str(api_key or "").strip()
    return api_key or None


def _massive_get_json(path: str, api_key: str, timeout_seconds: int) -> tuple[Any, str | None]:
    """
    Minimal Massive (Polygon rebrand) REST helper using stdlib urllib.
    Returns: (payload_json, date_header)
    """
    import urllib.parse
    import urllib.request

    base = "https://api.massive.com"
    url = f"{base}{path}"
    url = f"{url}?{urllib.parse.urlencode({'apiKey': api_key})}"

    req = urllib.request.Request(url=url, method="GET", headers={"User-Agent": "flagship-app-console/1.0"})
    with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw), resp.headers.get("Date")


def _normalize_holiday_status(text: str | None) -> str:
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "_")


def _fetch_massive_upcoming_holidays(api_key: str, timeout_seconds: int) -> dict[date, dict[str, Any]]:
    """
    Fetch upcoming holidays/early closes from Massive.
    Returns mapping: date -> raw holiday dict.
    """
    out: dict[date, dict[str, Any]] = {}
    try:
        payload, _ = _massive_get_json("/v1/marketstatus/upcoming", api_key=api_key, timeout_seconds=timeout_seconds)
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] massive upcoming fetch failed: {exc}")
        return out

    items: list[dict[str, Any]] = []
    if isinstance(payload, list):
        items = [it for it in payload if isinstance(it, dict)]
    elif isinstance(payload, dict):
        raw_items = payload.get("results")
        if isinstance(raw_items, list):
            items = [it for it in raw_items if isinstance(it, dict)]

    for it in items:
        try:
            d = date.fromisoformat(str(it.get("date") or ""))
        except Exception:
            continue
        out[d] = it
    return out


def _is_closed_trading_day(d: date, holiday_by_date: dict[date, dict[str, Any]] | None = None) -> bool:
    """
    True if market is closed on date d:
    - weekend
    - holiday status=closed (when available)
    """
    if d.weekday() >= 5:
        return True
    it = (holiday_by_date or {}).get(d)
    if not it:
        return False
    return _normalize_holiday_status(it.get("status")) == "closed"


def _next_trading_day(after_date: date, holiday_by_date: dict[date, dict[str, Any]] | None = None) -> date:
    """
    Return next trading day AFTER after_date, skipping weekend and known 'closed' holidays.
    """
    d = after_date + timedelta(days=1)
    for _ in range(14):  # safety bound (~2 weeks)
        if _is_closed_trading_day(d, holiday_by_date=holiday_by_date):
            d += timedelta(days=1)
            continue
        return d

    # Fallback: weekday-only
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ---------------- Alpaca REST (fallback) ----------------
def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": str(ALPACA_API_KEY),
        "APCA-API-SECRET-KEY": str(ALPACA_SECRET_KEY),
        "Content-Type": "application/json",
    }


def _alpaca_get_json(path: str, query: dict[str, str] | None = None, timeout_seconds: int = 20) -> Any:
    """
    Minimal Alpaca REST GET helper (stdlib only), used as a fallback when alpaca-py
    client methods are unavailable/incompatible.
    """
    import urllib.parse
    import urllib.request

    base = str(ALPACA_BASE_URL).rstrip("/")
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    req = urllib.request.Request(url=url, method="GET", headers=_alpaca_headers())
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


# ---------------- Portfolio ----------------
def snapshot_portfolio(output_dir: Path) -> Dict[str, Any]:
    adapter = AlpacaAdapter()
    account = adapter.client.get_account()
    positions: List[Dict[str, Any]] = []
    try:
        alpaca_positions = adapter.client.get_all_positions()
        for p in alpaca_positions:
            try:
                positions.append(
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "market_value": float(p.market_value),
                        "avg_entry": float(p.avg_entry_price),
                        "unrealized_pnl": float(p.unrealized_pl),
                    }
                )
            except Exception:
                continue
    except Exception as exc:
        logger.error(f"[app_console_snapshot] fetch positions failed: {exc}")

    data = {
        "generated_at": _now_iso(),
        "account": {
            "cash": float(account.cash),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "status": getattr(account, "status", ""),
        },
        "positions": positions,
    }
    target = output_dir / "portfolio.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target}")
    return data


# ---------------- Orders ----------------
def snapshot_orders(output_dir: Path, limit: int = 1000) -> Dict[str, Any]:
    """Fetch order history from Alpaca"""
    adapter = AlpacaAdapter()
    orders: List[Dict[str, Any]] = []
    try:
        # Preferred: alpaca-py request objects
        alpaca_orders = None
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            status_all = getattr(QueryOrderStatus, "ALL", None) or QueryOrderStatus.CLOSED
            request_params = GetOrdersRequest(status=status_all, limit=int(limit), nested=False)
            alpaca_orders = adapter.client.get_orders(filter=request_params)
        except Exception:
            alpaca_orders = None

        # Fallback: raw REST
        if alpaca_orders is None:
            alpaca_orders = _alpaca_get_json(
                "/v2/orders",
                query={
                    "status": "all",
                    "limit": str(int(limit)),
                    "nested": "false",
                    "direction": "desc",
                },
            )

        for o in alpaca_orders:
            try:
                # REST returns dict; alpaca-py returns model objects
                if isinstance(o, dict):
                    orders.append(
                        {
                            "id": str(o.get("id", "")),
                            "symbol": o.get("symbol"),
                            "side": o.get("side"),
                            "qty": float(o.get("qty") or 0),
                            "filled_qty": float(o.get("filled_qty") or 0),
                            "filled_avg_price": float(o["filled_avg_price"]) if o.get("filled_avg_price") is not None else None,
                            "status": o.get("status"),
                            "order_type": o.get("order_type"),
                            "limit_price": float(o["limit_price"]) if o.get("limit_price") is not None else None,
                            "stop_price": float(o["stop_price"]) if o.get("stop_price") is not None else None,
                            "submitted_at": o.get("submitted_at"),
                            "filled_at": o.get("filled_at"),
                            "canceled_at": o.get("canceled_at"),
                        }
                    )
                    continue

                orders.append(
                    {
                        "id": str(o.id),
                        "symbol": o.symbol,
                        "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                        "qty": float(o.qty) if o.qty else 0,
                        "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                        "filled_avg_price": float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) is not None else None,
                        "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                        "order_type": o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type),
                        "limit_price": float(o.limit_price) if o.limit_price else None,
                        "stop_price": float(o.stop_price) if o.stop_price else None,
                        "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                        "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                        "canceled_at": o.canceled_at.isoformat() if o.canceled_at else None,
                    }
                )
            except Exception:
                continue
    except Exception as exc:
        logger.error(f"[app_console_snapshot] fetch orders failed: {exc}")

    data = {
        "generated_at": _now_iso(),
        "orders": orders,
    }
    target = output_dir / "orders.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target} ({len(orders)} orders)")
    return data


# ---------------- Performance ----------------
def snapshot_performance(output_dir: Path, lookback_days: int = 365) -> Dict[str, Any]:
    """Fetch account equity time series from Alpaca portfolio history"""
    adapter = AlpacaAdapter()
    equity_series: List[Dict[str, Any]] = []

    try:
        # Map lookback_days to Alpaca supported period strings
        if lookback_days <= 7:
            period = "1W"
        elif lookback_days <= 31:
            period = "1M"
        elif lookback_days <= 93:
            period = "3M"
        elif lookback_days <= 186:
            period = "6M"
        elif lookback_days <= 366:
            period = "1A"
        else:
            period = "all"

        portfolio_history: Any | None = None

        # Preferred: alpaca-py client method (if available)
        if hasattr(adapter.client, "get_portfolio_history"):
            try:
                from alpaca.trading.requests import GetPortfolioHistoryRequest

                req = GetPortfolioHistoryRequest(period=period, timeframe="1D")
                portfolio_history = adapter.client.get_portfolio_history(history_filter=req)  # type: ignore[arg-type]
            except TypeError:
                # Some alpaca-py versions use 'filter' kwarg naming
                try:
                    from alpaca.trading.requests import GetPortfolioHistoryRequest

                    req = GetPortfolioHistoryRequest(period=period, timeframe="1D")
                    portfolio_history = adapter.client.get_portfolio_history(filter=req)  # type: ignore[arg-type]
                except Exception:
                    portfolio_history = None
            except Exception:
                portfolio_history = None

        # Fallback: raw REST
        if portfolio_history is None:
            portfolio_history = _alpaca_get_json(
                "/v2/account/portfolio/history",
                query={
                    "period": period,
                    "timeframe": "1D",
                },
            )

        # Normalize response
        if isinstance(portfolio_history, dict):
            equities = portfolio_history.get("equity") or []
            timestamps = portfolio_history.get("timestamp") or []
        else:
            equities = getattr(portfolio_history, "equity", None) or []
            timestamps = getattr(portfolio_history, "timestamp", None) or []

        for i, ts in enumerate(timestamps):
            if i >= len(equities):
                break
            try:
                ts_val = float(ts)
                # Alpaca sometimes returns seconds; some SDKs return milliseconds
                ts_sec = ts_val / 1000.0 if ts_val > 10_000_000_000 else ts_val
                equity_series.append(
                    {
                        "date": datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat(),
                        "equity": float(equities[i]) if equities[i] is not None else 0.0,
                    }
                )
            except Exception:
                continue
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] fetch portfolio history failed: {exc}, using current equity as fallback")
        # Fallback: use current equity as single point
        try:
            account = adapter.client.get_account()
            equity_series.append(
                {
                    "date": _now_iso(),
                    "equity": float(account.equity),
                }
            )
        except Exception:
            pass

    data = {
        "generated_at": _now_iso(),
        "equity_series": equity_series,
    }
    target = output_dir / "performance.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target} ({len(equity_series)} data points)")
    return data


# ---------------- Market Status (Massive) ----------------
def snapshot_market_status(output_dir: Path, timeout_seconds: int = DEFAULT_MASSIVE_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """
    Fetch market status from Massive (Polygon rebrand) and write logs/app/market_status.json.

    API:
    - GET /v1/marketstatus/now?apiKey=...
    """
    api_key = _get_massive_api_key()
    if not api_key:
        data = {
            "generated_at": _now_iso(),
            "source": "massive",
            "error": "missing_api_key",
        }
        target = output_dir / "market_status.json"
        _atomic_write_json(target, data)
        logger.warning("[app_console_snapshot] market_status: api key missing (set vt_setting.json datafeed.password or MASSIVE_API_KEY)")
        return data

    try:
        payload, date_header = _massive_get_json(
            "/v1/marketstatus/now", api_key=api_key, timeout_seconds=timeout_seconds
        )
    except Exception as exc:
        data = {
            "generated_at": _now_iso(),
            "source": "massive",
            "error": str(exc),
        }
        target = output_dir / "market_status.json"
        _atomic_write_json(target, data)
        logger.warning(f"[app_console_snapshot] market_status fetch failed: {exc}")
        return data

    # Parse server time (prefer payload.serverTime, fallback to Date header, then local utc)
    server_dt = _parse_iso_datetime(payload.get("serverTime"))
    if server_dt is None and date_header:
        try:
            server_dt = parsedate_to_datetime(date_header)
        except Exception:
            server_dt = None
    if server_dt is None:
        server_dt = datetime.now(timezone.utc)
    if server_dt.tzinfo is None:
        server_dt = server_dt.replace(tzinfo=timezone.utc)
    server_et = server_dt.astimezone(EASTERN)

    holiday_by_date = _fetch_massive_upcoming_holidays(api_key=api_key, timeout_seconds=timeout_seconds)

    def _session_open_time(d: date) -> dtime:
        it = holiday_by_date.get(d)
        if it:
            t = _parse_hhmm(it.get("open"))
            if t:
                return t
        return DEFAULT_MARKET_OPEN_ET

    next_open_dt: datetime | None = None
    cursor_date = server_et.date()
    for _ in range(30):  # safety bound (~6 weeks)
        if _is_closed_trading_day(cursor_date, holiday_by_date=holiday_by_date):
            cursor_date += timedelta(days=1)
            continue
        open_dt = datetime.combine(cursor_date, _session_open_time(cursor_date), tzinfo=EASTERN)
        if server_et < open_dt:
            next_open_dt = open_dt
            break
        cursor_date += timedelta(days=1)

    seconds_to_open: int | None = None
    if next_open_dt is not None:
        seconds_to_open = int(max(0.0, (next_open_dt - server_et).total_seconds()))

    data = {
        "generated_at": _now_iso(),
        "source": "massive",
        "server_time": payload.get("serverTime"),
        "date_header": date_header,
        "market": payload.get("market"),
        "exchanges": payload.get("exchanges"),
        "early_hours": payload.get("earlyHours"),
        "after_hours": payload.get("afterHours"),
        "next_open": next_open_dt.isoformat() if next_open_dt else None,
        "seconds_to_open": seconds_to_open,
    }
    target = output_dir / "market_status.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target}")
    return data


# ---------------- Selection ----------------
def _load_signal_parquet(lab_path: Path) -> pl.DataFrame:
    signal_path = lab_path / "signal" / "daily_signal.parquet"
    if not signal_path.exists():
        raise FileNotFoundError(f"signal file not found: {signal_path}")
    df = pl.read_parquet(signal_path)
    if df.is_empty():
        raise ValueError("signal parquet is empty")
    return df


def _pick_price_column(df: pl.DataFrame) -> str:
    if "close_price" in df.columns:
        return "close_price"
    if "close" in df.columns:
        return "close"
    raise ValueError("price column not found (expected close_price or close)")


def _add_return_columns(df: pl.DataFrame, price_col: str, windows: Iterable[int]) -> pl.DataFrame:
    out = df
    for k in windows:
        window = int(k)
        if window <= 0:
            continue
        out = out.with_columns(
            [
                (pl.col(price_col) / pl.col(price_col).shift(window).over("vt_symbol") - 1.0).alias(
                    f"trail_ret_{window}d"
                ),
                (pl.col(price_col).shift(-window).over("vt_symbol") / pl.col(price_col) - 1.0).alias(
                    f"fwd_ret_{window}d"
                ),
            ]
        )
    return out


def _load_recent_selection_dates(max_date: date, limit: int) -> List[date]:
    """
    Return recent trade_date candidates from Postgres daily_selection, bounded by max_date.
    """
    if limit <= 0:
        return []
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT trade_date
                FROM daily_selection
                WHERE trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT %s
                """,
                (max_date, limit),
            )
            return [row[0] for row in cur.fetchall() if row and row[0]]


def _slice_signal_for_date(df: pl.DataFrame, as_of_date: date) -> pl.DataFrame:
    out = df.filter(pl.col("datetime").dt.date() == as_of_date)
    return out.with_columns(pl.lit(as_of_date).alias("as_of_date"))


def _load_daily_selection(trade_date: date) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT vt_symbol, close_price, adv_usd, med_volume, market_cap
                    FROM daily_selection
                    WHERE trade_date = %s
                    """,
                    (trade_date,),
                )
                rows = cur.fetchall()
                for row in rows:
                    vt_symbol, close_price, adv_usd, med_volume, market_cap = row
                    out[str(vt_symbol)] = {
                        "close_price": float(close_price) if close_price is not None else None,
                        "adv_usd": float(adv_usd) if adv_usd is not None else None,
                        "med_volume": int(med_volume) if med_volume is not None else None,
                        "market_cap": float(market_cap) if market_cap is not None else None,
                    }
            except Exception:
                # Backward-compatible schema: daily_selection may not have market_cap column.
                conn.rollback()
                cur.execute(
                    """
                    SELECT vt_symbol, close_price, adv_usd, med_volume
                    FROM daily_selection
                    WHERE trade_date = %s
                    """,
                    (trade_date,),
                )
                for row in cur.fetchall():
                    vt_symbol, close_price, adv_usd, med_volume = row
                    out[str(vt_symbol)] = {
                        "close_price": float(close_price) if close_price is not None else None,
                        "adv_usd": float(adv_usd) if adv_usd is not None else None,
                        "med_volume": int(med_volume) if med_volume is not None else None,
                        "market_cap": None,
                    }
    return out


def build_signal_only_rows(
    signal_df: pl.DataFrame,
    top_n: int,
    lab_path: Path | None = None,
    as_of_date: date | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in signal_df.iter_rows(named=True):
        raw_close = row.get("close_price")
        close_price: float | None = float(raw_close) if raw_close is not None else None
        adv_usd: float | None = None
        med_volume: int | None = None
        market_cap: float | None = None

        # Even when Postgres daily_selection is missing, we can enrich with AlphaLab daily parquet.
        if lab_path and as_of_date:
            lab_close, lab_medvol, lab_adv = _compute_adv_medvol_from_lab(lab_path, row["vt_symbol"], as_of_date)
            if lab_close is not None:
                close_price = lab_close
            if lab_medvol is not None:
                med_volume = lab_medvol
            if lab_adv is not None:
                adv_usd = lab_adv

        rows.append(
            {
                "vt_symbol": row["vt_symbol"],
                "signal": float(row["signal"]) if row.get("signal") is not None else None,
                "close_price": close_price,
                "adv_usd": adv_usd,
                "med_volume": med_volume,
                "market_cap": market_cap,
            }
        )
    rows = sorted(rows, key=lambda r: (r["signal"] is None, -(r["signal"] or 0.0)))
    return rows[:max(0, top_n)]


def _compute_adv_medvol_from_lab(
    lab_path: Path, vt_symbol: str, as_of_date: date, lookback_days: int = 60
) -> tuple[float | None, int | None, float | None]:
    """
    Compute ADV (USD) and med_volume from AlphaLab daily parquet.
    Returns: (close_price, med_volume, adv_usd)
    """
    try:
        lab = AlphaLab(lab_path)
        daily_path = lab.daily_path
        file_path = daily_path / f"{vt_symbol}.parquet"

        if not file_path.exists():
            # Try symbol root matching
            symbol_root = vt_symbol.split(".")[0]
            candidates = sorted(daily_path.glob(f"{symbol_root}.*.parquet"))
            if len(candidates) == 1:
                file_path = candidates[0]
            else:
                return (None, None, None)

        df = pl.read_parquet(file_path)
        if df.is_empty():
            return (None, None, None)

        # Filter to lookback window
        start_date = as_of_date - timedelta(days=lookback_days)
        df = df.filter(
            (pl.col("datetime") >= datetime.combine(start_date, datetime.min.time()))
            & (pl.col("datetime") <= datetime.combine(as_of_date, datetime.min.time()))
        ).sort("datetime")

        if df.is_empty():
            return (None, None, None)

        # Get close price for as_of_date
        as_of_rows = df.filter(pl.col("datetime").dt.date() == as_of_date)
        if as_of_rows.is_empty():
            # Use last available close
            close_price = df["close"].tail(1).item() if df.height > 0 else None
        else:
            close_price = as_of_rows["close"].item()

        if close_price is None:
            return (None, None, None)

        # Compute med_volume (last 30 days, or all if < 30)
        volumes = df["volume"].to_list()
        if len(volumes) >= 30:
            recent_volumes = volumes[-30:]
        else:
            recent_volumes = volumes

        if not recent_volumes:
            return (float(close_price), None, None)

        import statistics

        med_volume = int(statistics.median(recent_volumes))
        adv_usd = med_volume * float(close_price)

        return (float(close_price), med_volume, adv_usd)
    except Exception as exc:
        logger.warning(f"[app_console_snapshot] compute ADV/MedVol for {vt_symbol} failed: {exc}")
        return (None, None, None)


def build_selection_rows(
    signal_df: pl.DataFrame,
    selection_map: Dict[str, Dict[str, Any]],
    top_n: int,
    lab_path: Path | None = None,
    as_of_date: date | None = None,
) -> List[Dict[str, Any]]:
    def _iter_rows() -> Iterable[Dict[str, Any]]:
        for row in signal_df.iter_rows(named=True):
            vt_symbol = row["vt_symbol"]
            sel = selection_map.get(vt_symbol)
            if sel is None:
                # Only rank symbols that are actually in daily_selection (U_t).
                continue

            # Prefer values from Postgres, but compute from lab if missing
            close_price = sel.get("close_price")
            adv_usd = sel.get("adv_usd")
            med_volume = sel.get("med_volume")
            market_cap = sel.get("market_cap")

            # If ADV/MedVol missing, compute from lab
            if (
                (adv_usd is None or float(adv_usd or 0.0) <= 0.0)
                or (med_volume is None or int(med_volume or 0) <= 0)
                or (close_price is None or float(close_price or 0.0) <= 0.0)
            ) and lab_path and as_of_date:
                lab_close, lab_medvol, lab_adv = _compute_adv_medvol_from_lab(lab_path, vt_symbol, as_of_date)
                if lab_close is not None:
                    close_price = lab_close
                if lab_medvol is not None:
                    med_volume = lab_medvol
                if lab_adv is not None:
                    adv_usd = lab_adv

            yield {
                "vt_symbol": vt_symbol,
                "signal": float(row["signal"]) if row.get("signal") is not None else None,
                "close_price": close_price,
                "adv_usd": adv_usd,
                "med_volume": med_volume,
                "market_cap": market_cap,
            }

    rows = sorted(
        list(_iter_rows()),
        key=lambda r: (r["signal"] is None, -(r["signal"] or 0.0)),
    )
    return rows[:max(0, top_n)]


def snapshot_selection(output_dir: Path, lab_path: Path, top_n: int = DEFAULT_SELECTION_TOP_N) -> Dict[str, Any]:
    signal_all = _load_signal_parquet(lab_path)
    max_dt = signal_all.select(pl.col("datetime").max()).item()
    if not max_dt:
        raise ValueError("cannot find max datetime in signal parquet")
    max_signal_date = max_dt.date()

    # Prefer a date that exists in BOTH signal parquet and daily_selection.
    signal_dates = set(
        signal_all.select(pl.col("datetime").dt.date().unique()).to_series().to_list()  # type: ignore[attr-defined]
    )
    candidates = _load_recent_selection_dates(max_signal_date, DEFAULT_SELECTION_DATE_CANDIDATES)
    signal_date = next((d for d in candidates if d in signal_dates), max_signal_date)

    # Live semantics:
    # - signal_date: the close date used to compute factors/signals (DATA_DATE)
    # - trade_date: next trading day when these signals are intended to be executed at open
    api_key = _get_massive_api_key()
    holiday_by_date: dict[date, dict[str, Any]] = {}
    if api_key:
        holiday_by_date = _fetch_massive_upcoming_holidays(api_key=api_key, timeout_seconds=DEFAULT_MASSIVE_TIMEOUT_SECONDS)
    trade_date = _next_trading_day(signal_date, holiday_by_date=holiday_by_date)

    signal_df = _slice_signal_for_date(signal_all, signal_date)
    selection_map = _load_daily_selection(signal_date)

    rows: List[Dict[str, Any]] = []
    if selection_map:
        rows = build_selection_rows(signal_df, selection_map, top_n, lab_path=lab_path, as_of_date=signal_date)
    if not rows:
        rows = build_signal_only_rows(signal_df, top_n, lab_path=lab_path, as_of_date=signal_date)

    data = {
        "generated_at": _now_iso(),
        "as_of_date": trade_date.isoformat(),
        "signal_date": signal_date.isoformat(),
        "rows": rows,
    }
    target = output_dir / "selection.json"
    _atomic_write_json(target, data)
    
    # Save a historical copy
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_target = history_dir / f"selection_{trade_date.strftime('%Y%m%d')}.json"
    _atomic_write_json(history_target, data)
    
    logger.info(f"[app_console_snapshot] wrote {target} and history {history_target}")
    return data


# ---------------- Monitor Returns ----------------
def snapshot_monitor_returns(
    output_dir: Path,
    lab_path: Path,
    *,
    windows: Iterable[int] = DEFAULT_MONITOR_WINDOWS,
    universe_top_k: int = DEFAULT_MONITOR_TOP_K,
    spy_symbol: str = "SPY.NASDAQ",
) -> Dict[str, Any]:
    """
    Generate logs/app/monitor_returns.json for Ops Console.

    Returns are computed on close-close basis:
    - trailing: close_t / close_{t-k} - 1
    - forward : close_{t+k} / close_t - 1
    """
    signal_all = _load_signal_parquet(lab_path)
    max_dt = signal_all.select(pl.col("datetime").max()).item()
    if not max_dt:
        raise ValueError("cannot find max datetime in signal parquet")
    signal_date = max_dt.date()

    signal_df = _slice_signal_for_date(signal_all, signal_date)
    if signal_df.is_empty():
        raise ValueError("signal parquet has no rows for signal_date")

    if "signal" in signal_df.columns:
        signal_df = signal_df.sort("signal", descending=True)

    if universe_top_k > 0:
        signal_df = signal_df.head(int(universe_top_k))

    vt_symbols = sorted(set(signal_df["vt_symbol"].to_list()))
    if spy_symbol not in vt_symbols:
        vt_symbols.append(spy_symbol)

    max_window = max(int(w) for w in windows if int(w) > 0) if windows else 0
    buffer_days = max(7, max_window * 2)
    start_date = signal_date - timedelta(days=buffer_days)
    end_date = signal_date + timedelta(days=buffer_days)

    lab = AlphaLab(str(lab_path))
    bar_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        extended_days=0,
    )
    if bar_df is None or bar_df.is_empty():
        raise ValueError("no bar data for monitor returns")

    price_col = _pick_price_column(bar_df)
    bar_df = bar_df.sort(["vt_symbol", "datetime"])
    bar_df = _add_return_columns(bar_df, price_col, windows)

    signal_rows = signal_df.select(
        [
            "vt_symbol",
            "datetime",
            "signal",
            *[c for c in ["lgb_signal", "p_up"] if c in signal_df.columns],
        ]
    )

    returns_rows = (
        bar_df.filter(pl.col("vt_symbol").is_in(vt_symbols))
        .filter(pl.col("datetime").dt.date() == signal_date)
        .select(
            [
                "vt_symbol",
                "datetime",
                pl.col(price_col).alias("close_t"),
                *[f"trail_ret_{int(w)}d" for w in windows if int(w) > 0],
                *[f"fwd_ret_{int(w)}d" for w in windows if int(w) > 0],
            ]
        )
    )

    spy_row = (
        bar_df.filter(pl.col("vt_symbol") == spy_symbol)
        .filter(pl.col("datetime").dt.date() == signal_date)
        .select(
            [
                "datetime",
                pl.col(price_col).alias("spy_close_t"),
                *[
                    pl.col(f"trail_ret_{int(w)}d").alias(f"spy_trail_ret_{int(w)}d")
                    for w in windows
                    if int(w) > 0
                ],
                *[
                    pl.col(f"fwd_ret_{int(w)}d").alias(f"spy_fwd_ret_{int(w)}d")
                    for w in windows
                    if int(w) > 0
                ],
            ]
        )
    )

    merged = signal_rows.join(returns_rows, on=["vt_symbol", "datetime"], how="left")
    merged = merged.join(spy_row, on="datetime", how="left")

    for w in windows:
        window = int(w)
        if window <= 0:
            continue
        merged = merged.with_columns(
            [
                (pl.col(f"trail_ret_{window}d") - pl.col(f"spy_trail_ret_{window}d")).alias(
                    f"trail_excess_{window}d"
                ),
                (pl.col(f"fwd_ret_{window}d") - pl.col(f"spy_fwd_ret_{window}d")).alias(
                    f"fwd_excess_{window}d"
                ),
            ]
        )

    merged = merged.filter(pl.col("vt_symbol") != spy_symbol)

    data = {
        "generated_at": _now_iso(),
        "signal_date": signal_date.isoformat(),
        "windows": [int(w) for w in windows if int(w) > 0],
        "top_k": int(universe_top_k),
        "rows": merged.to_dicts(),
    }
    target = output_dir / "monitor_returns.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target}")
    return data


# ---------------- CLI ----------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate app console snapshots")
    parser.add_argument(
        "mode",
        nargs="+",
        choices=["portfolio", "selection", "orders", "performance", "market_status", "accounts", "monitor", "all"],
        help="Snapshot type(s) to generate (can specify multiple)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for JSON files")
    parser.add_argument("--lab-path", type=Path, default=LAB_PATH, help="AlphaLab path")
    parser.add_argument("--top-n", type=int, default=DEFAULT_SELECTION_TOP_N, help="Top N selection rows by signal")
    parser.add_argument("--monitor-top-k", type=int, default=DEFAULT_MONITOR_TOP_K, help="Top K for monitor universe")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir

    if not args.lab_path.is_absolute():
        args.lab_path = PROJECT_ROOT / args.lab_path

    modes = set(args.mode)
    if "all" in modes:
        modes = {"portfolio", "selection", "orders", "performance", "market_status", "accounts", "monitor"}

    if "portfolio" in modes:
        snapshot_portfolio(args.output_dir)

    if "selection" in modes:
        snapshot_selection(args.output_dir, args.lab_path, args.top_n)

    if "orders" in modes:
        snapshot_orders(args.output_dir)

    if "performance" in modes:
        snapshot_performance(args.output_dir)

    if "market_status" in modes:
        snapshot_market_status(args.output_dir)

    if "accounts" in modes:
        # Optional: generate IBKR per-account snapshots first, then aggregate accounts.json.
        ibkr_cfgs = load_ibkr_accounts()
        for cfg in ibkr_cfgs:
            snapshot_ibkr_account(
                args.output_dir,
                account_id=cfg.account_id,
                display_name=cfg.display_name,
                host=cfg.host,
                port=int(cfg.port),
                client_id=int(cfg.client_id),
            )
        snapshot_accounts(args.output_dir)

    if "monitor" in modes:
        snapshot_monitor_returns(args.output_dir, args.lab_path, universe_top_k=args.monitor_top_k)


if __name__ == "__main__":
    main()

