"""
Generate JSON snapshots for the Ops Console static page.

Outputs:
- logs/app/portfolio.json
- logs/app/selection.json
- logs/app/orders.json
- logs/app/performance.json

Design:
- Portfolio is fetched from the single Alpaca account configured in vt_setting.json.
- Selection uses the latest daily_signal.parquet (by datetime max) joined with
  Postgres daily_selection on the same trade_date, sorted by signal desc.
- Orders: fetched from Alpaca order history.
- Performance: account equity time series from Alpaca portfolio history.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Ensure `import flagship` works when running as a script (e.g. `python flagship/...py`)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flagship.paper_trading.broker_alpaca import AlpacaAdapter
from flagship.paper_trading.config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, LAB_PATH
from flagship.scripts.pg_ticker_db import get_pg_connection
from vnpy.alpha.lab import AlphaLab
from vnpy.trader.logger import logger

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "app"
DEFAULT_SELECTION_TOP_N = int(os.getenv("FLAGSHIP_APP_SELECTION_TOPN", "10"))
DEFAULT_SELECTION_DATE_CANDIDATES = int(os.getenv("FLAGSHIP_APP_SELECTION_DATE_CANDIDATES", "30"))


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


# ---------------- Selection ----------------
def _load_signal_parquet(lab_path: Path) -> pl.DataFrame:
    signal_path = lab_path / "signal" / "daily_signal.parquet"
    if not signal_path.exists():
        raise FileNotFoundError(f"signal file not found: {signal_path}")
    df = pl.read_parquet(signal_path)
    if df.is_empty():
        raise ValueError("signal parquet is empty")
    return df


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
    as_of_date = next((d for d in candidates if d in signal_dates), max_signal_date)

    signal_df = _slice_signal_for_date(signal_all, as_of_date)
    selection_map = _load_daily_selection(as_of_date)

    rows: List[Dict[str, Any]] = []
    if selection_map:
        rows = build_selection_rows(signal_df, selection_map, top_n, lab_path=lab_path, as_of_date=as_of_date)
    if not rows:
        rows = build_signal_only_rows(signal_df, top_n, lab_path=lab_path, as_of_date=as_of_date)

    data = {
        "generated_at": _now_iso(),
        "as_of_date": as_of_date.isoformat(),
        "rows": rows,
    }
    target = output_dir / "selection.json"
    _atomic_write_json(target, data)
    logger.info(f"[app_console_snapshot] wrote {target}")
    return data


# ---------------- CLI ----------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate app console snapshots")
    parser.add_argument(
        "mode",
        nargs="+",
        choices=["portfolio", "selection", "orders", "performance", "all"],
        help="Snapshot type(s) to generate (can specify multiple)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for JSON files")
    parser.add_argument("--lab-path", type=Path, default=LAB_PATH, help="AlphaLab path")
    parser.add_argument("--top-n", type=int, default=DEFAULT_SELECTION_TOP_N, help="Top N selection rows by signal")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir

    if not args.lab_path.is_absolute():
        args.lab_path = PROJECT_ROOT / args.lab_path

    modes = set(args.mode)
    if "all" in modes:
        modes = {"portfolio", "selection", "orders", "performance"}

    if "portfolio" in modes:
        snapshot_portfolio(args.output_dir)

    if "selection" in modes:
        snapshot_selection(args.output_dir, args.lab_path, args.top_n)

    if "orders" in modes:
        snapshot_orders(args.output_dir)

    if "performance" in modes:
        snapshot_performance(args.output_dir)


if __name__ == "__main__":
    main()

