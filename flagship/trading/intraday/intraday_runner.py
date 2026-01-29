"""
Intraday runner (Plan B):
- Subscribe Polygon WebSocket minute aggregates for watched symbols
- Feed minute bars into FlagshipAlphaMomentumStrategy to trigger intraday exits
- Execute exits on Alpaca

Default is **exit-only** (no new entries intraday) and **RTH-only** (09:30-16:00 ET).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Direction, Exchange, Interval, Offset
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData
from vnpy.alpha.strategy import AlphaStrategy

from flagship.config.polygon_config import get_polygon_api_key
from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter
from flagship.monitoring.order_metrics import classify_order_error
from flagship.trading.execution.broker_alpaca import AlpacaAdapter
from flagship.trading.config import DAILY_SIGNAL_FILE, LAB_PATH
from flagship.trading.controls import get_disabled_vt_symbols
from flagship.trading.realtime.polygon_ws import (
    Market,
    POLYGON_WS_AVAILABLE,
    WebSocketClient,
    polygon_ws_import_error,
)
from flagship.trading.calendar import get_holiday_info, is_market_closed_day
from flagship.strategy.registry import STRATEGY_REGISTRY, build_strategy_instance

try:
    # alpaca-py
    from alpaca.trading.enums import OrderSide
except Exception:  # pragma: no cover
    OrderSide = None  # type: ignore


EASTERN = ZoneInfo("America/New_York")
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)
EXIT_ORDER_COOLDOWN_SECONDS = int(os.getenv("FLAGSHIP_INTRADAY_EXIT_COOLDOWN_SECONDS", "300"))
OPEN_ORDERS_CACHE_SECONDS = int(os.getenv("FLAGSHIP_INTRADAY_OPEN_ORDERS_CACHE_SECONDS", "5"))


class SymbolsSource(str):
    POSITIONS = "positions"
    SIGNAL = "signal"
    BOTH = "both"


@dataclass
class WatchedSymbol:
    root: str
    vt_symbol: str
    exchange: Exchange


def _is_rth(dt_eastern_naive: datetime, *, close_time: dtime = RTH_CLOSE) -> bool:
    """RTH window in America/New_York (close_time supports early close)."""
    t = dt_eastern_naive.time()
    return RTH_OPEN <= t <= close_time


def _utc_ms_to_eastern_naive(ms: int) -> datetime:
    utc_dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return utc_dt.astimezone(EASTERN).replace(tzinfo=None)


def _infer_vt_symbol_from_lab(lab: AlphaLab, root: str) -> str:
    candidates = sorted(lab.daily_path.glob(f"{root}.*.parquet"))
    if len(candidates) == 1:
        return candidates[0].stem
    # fallback
    return f"{root}.NASDAQ"


def _load_signal_df(signal_path: Path) -> pl.DataFrame:
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")
    df = pl.read_parquet(signal_path)
    if "vt_symbol" not in df.columns:
        raise RuntimeError(f"Signal parquet missing vt_symbol: {signal_path}")
    return df


def _build_watchlist(
    lab: AlphaLab,
    adapter: AlpacaAdapter,
    signal_df: pl.DataFrame | None,
    source: str,
    signal_top_n: int,
) -> list[WatchedSymbol]:
    disabled_roots: set[str] = set()
    try:
        disabled_vt = get_disabled_vt_symbols()
        disabled_roots = {str(v).split(".")[0] for v in disabled_vt if str(v).strip()}
    except Exception:
        disabled_roots = set()

    positions_roots: set[str] = set()
    if source in (SymbolsSource.POSITIONS, SymbolsSource.BOTH):
        positions_roots = set(adapter.get_positions().keys())

    root_to_vt: dict[str, str] = {}
    if signal_df is not None:
        df = signal_df

        # Reduce signal-based subscriptions to Top-N by score (positions are always included).
        if signal_top_n > 0 and "signal" in df.columns:
            try:
                df = df.sort("signal", descending=True).head(int(signal_top_n))
            except Exception:
                df = signal_df

        # Always keep SPY for macro risk checks if present (even if not in Top-N)
        try:
            spy = signal_df.filter(pl.col("vt_symbol") == "SPY.NASDAQ")
            if not spy.is_empty():
                df = pl.concat([df, spy]).unique(subset=["vt_symbol"], keep="first")
        except Exception:
            pass

        for vt in df["vt_symbol"].to_list():
            vt_str = str(vt)
            root = vt_str.split(".")[0]
            # Disabled symbols: do not subscribe from signal list, unless it's already held.
            if root in disabled_roots and root not in positions_roots and vt_str != "SPY.NASDAQ":
                continue
            root_to_vt[root] = vt_str

    roots: set[str] = set()
    if source in (SymbolsSource.POSITIONS, SymbolsSource.BOTH):
        roots.update(positions_roots)
    if signal_df is not None and source in (SymbolsSource.SIGNAL, SymbolsSource.BOTH):
        roots.update(root_to_vt.keys())

    watched: list[WatchedSymbol] = []
    for root in sorted(roots):
        vt_symbol = root_to_vt.get(root) or _infer_vt_symbol_from_lab(lab, root)
        parts = vt_symbol.split(".")
        exchange = Exchange(parts[1]) if len(parts) == 2 else Exchange.NASDAQ
        watched.append(WatchedSymbol(root=root, vt_symbol=vt_symbol, exchange=exchange))
    return watched


class LiveEngine:
    """
    盘中执行使用的最小策略引擎：
    - get_signal(): 返回完整 signal_df（每日快照）
    - send_order(): 发送 Alpaca 订单（默认 exit-only）
    """

    def __init__(
        self,
        adapter: AlpacaAdapter,
        signal_df: pl.DataFrame,
        mode: str,
    ) -> None:
        self.adapter = adapter
        self.signal_df = signal_df
        self.mode = mode
        self.root_close: dict[str, float] = {}
        self.orders: dict[str, Any] = {}
        self._order_count = 0
        self.order_submitted_total = 0
        self.order_failed_total = 0
        self.order_rejected_total = 0
        self.order_skipped_total = 0
        self.order_error_reasons: dict[str, int] = {}
        self.pending_exit_roots: dict[str, datetime] = {}
        self.open_sell_roots_cache: set[str] = set()
        self.open_sell_roots_cache_ts: datetime | None = None
        self.open_orders_warned = False

    def get_signal(self) -> pl.DataFrame:
        return self.signal_df

    def get_cash_available(self) -> float:
        try:
            info = self.adapter.get_account_info()
            return min(info.cash, info.buying_power)
        except Exception:
            return self.adapter.get_cash()

    def get_holding_value(self) -> float:
        total = 0.0
        pos = self.adapter.get_positions()
        for root, qty in pos.items():
            px = self.root_close.get(root)
            if px is None:
                continue
            total += float(qty) * float(px)
        return total

    def write_log(self, msg: str, strategy=None) -> None:
        logger.info(f"[IntradayRunner] {msg}")

    def get_pricetick(self, vt_symbol: str) -> float:
        return 0.01

    def get_size(self, vt_symbol: str) -> float:
        return 1.0

    def send_order(
        self,
        strategy: AlphaStrategy,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> list[str]:
        """
        将 vnpy 订单转换为 Alpaca 下单。

        安全策略：
        - exit-only 模式忽略 OPEN 订单。
        - CLOSE 订单按 Alpaca 当前持仓量做截断，避免超卖。
        """
        self._order_count += 1
        oid = f"intraday_{self._order_count}"

        # Cache order info for debugging
        self.orders[oid] = {
            "vt_symbol": vt_symbol,
            "direction": direction,
            "offset": offset,
            "price": price,
            "volume": volume,
        }

        if self.mode == "exit-only" and offset != Offset.CLOSE:
            logger.warning(
                f"[IntradayRunner] exit-only: ignore OPEN order {oid} {vt_symbol} {direction} {volume}"
            )
            self.order_skipped_total += 1
            return []

        root = vt_symbol.split(".")[0]
        current_qty = self.adapter.get_positions().get(root, 0)
        now = datetime.now(timezone.utc)

        # 已有未完成卖单或处于冷却期时，跳过重复离场
        last_exit_at = self.pending_exit_roots.get(root)
        if last_exit_at and (now - last_exit_at).total_seconds() < EXIT_ORDER_COOLDOWN_SECONDS:
            logger.info(f"[IntradayRunner] {root} 离场冷却中，跳过重复卖单")
            self.order_skipped_total += 1
            return []
        if root in self._get_open_sell_roots():
            logger.info(f"[IntradayRunner] {root} 已有未完成卖单，跳过重复卖单")
            self.pending_exit_roots.setdefault(root, now)
            self.order_skipped_total += 1
            return []

        # 平多仓 -> Alpaca SELL
        if offset == Offset.CLOSE and direction == Direction.SHORT:
            sell_qty = min(int(volume), int(current_qty))
            if sell_qty <= 0:
                self.order_skipped_total += 1
                return []
            if OrderSide is None:
                logger.error("[IntradayRunner] alpaca-py OrderSide not available, cannot send orders.")
                self.order_failed_total += 1
                return []
            try:
                self.adapter.place_order(root, sell_qty, OrderSide.SELL)
                self.order_submitted_total += 1
            except Exception as exc:
                self.order_failed_total += 1
                reason, rejected = classify_order_error(exc)
                self.order_error_reasons[reason] = self.order_error_reasons.get(reason, 0) + 1
                if rejected:
                    self.order_rejected_total += 1
            self.pending_exit_roots[root] = now
            return []

        # 平空仓 -> Alpaca BUY（本策略不期望出现）
        if offset == Offset.CLOSE and direction == Direction.LONG:
            buy_qty = int(volume)
            if buy_qty <= 0:
                self.order_skipped_total += 1
                return []
            if OrderSide is None:
                logger.error("[IntradayRunner] alpaca-py OrderSide not available, cannot send orders.")
                self.order_failed_total += 1
                return []
            try:
                self.adapter.place_order(root, buy_qty, OrderSide.BUY)
                self.order_submitted_total += 1
            except Exception as exc:
                self.order_failed_total += 1
                reason, rejected = classify_order_error(exc)
                self.order_error_reasons[reason] = self.order_error_reasons.get(reason, 0) + 1
                if rejected:
                    self.order_rejected_total += 1
            return []

        # full 模式允许开仓下单
        if offset == Offset.OPEN and direction == Direction.LONG:
            buy_qty = int(volume)
            if buy_qty <= 0:
                self.order_skipped_total += 1
                return []
            if OrderSide is None:
                logger.error("[IntradayRunner] alpaca-py OrderSide not available, cannot send orders.")
                self.order_failed_total += 1
                return []
            try:
                self.adapter.place_order(root, buy_qty, OrderSide.BUY)
                self.order_submitted_total += 1
            except Exception as exc:
                self.order_failed_total += 1
                reason, rejected = classify_order_error(exc)
                self.order_error_reasons[reason] = self.order_error_reasons.get(reason, 0) + 1
                if rejected:
                    self.order_rejected_total += 1
            return []

        if offset == Offset.OPEN and direction == Direction.SHORT:
            # 不支持
            logger.warning(f"[IntradayRunner] ignore SHORT OPEN {vt_symbol} volume={volume}")
            self.order_skipped_total += 1
            return []

        return []

    def _get_open_sell_roots(self) -> set[str]:
        now = datetime.now(timezone.utc)
        if self.open_sell_roots_cache_ts and (now - self.open_sell_roots_cache_ts).total_seconds() < OPEN_ORDERS_CACHE_SECONDS:
            return self.open_sell_roots_cache

        client = getattr(self.adapter, "client", None)
        if client is None:
            return set()

        orders = None
        if hasattr(client, "get_orders"):
            try:
                orders = client.get_orders(status="open")
            except TypeError:
                try:
                    orders = client.get_orders()
                except Exception:
                    orders = None
            except Exception:
                orders = None
        elif hasattr(client, "list_orders"):
            try:
                orders = client.list_orders(status="open")
            except TypeError:
                try:
                    orders = client.list_orders()
                except Exception:
                    orders = None
            except Exception:
                orders = None
        else:
            if not self.open_orders_warned:
                logger.warning("[IntradayRunner] Alpaca client 无法获取 open orders，可能导致重复离场下单。")
                self.open_orders_warned = True
            return set()

        roots: set[str] = set()
        terminal_status = {"filled", "canceled", "cancelled", "rejected", "expired"}
        if orders:
            for order in orders:
                symbol = str(getattr(order, "symbol", "") or "").strip()
                if not symbol:
                    continue
                side = str(getattr(order, "side", "") or "").lower()
                status = str(getattr(order, "status", "") or "").lower()
                if side != "sell":
                    continue
                if status in terminal_status:
                    continue
                roots.add(symbol)

        self.open_sell_roots_cache = roots
        self.open_sell_roots_cache_ts = now
        return roots

    def clear_pending_exit(self, root: str) -> None:
        self.pending_exit_roots.pop(root, None)

    def cancel_order(self, strategy, order_id: str) -> None:
        # not implemented for now
        return

    def cancel_all(self, strategy) -> None:
        # We rely on AlpacaAdapter.cancel_all_open_orders before start; not on every minute.
        return


def _init_strategy_state_from_alpaca(
    strategy: AlphaStrategy,
    adapter: AlpacaAdapter,
    watched: list[WatchedSymbol],
    last_prices: dict[str, float],
    today: date,
) -> None:
    # Prevent daily rebalance logic: we only want intraday exits.
    strategy.current_trade_date = today
    strategy.daily_rebalance_done = True

    # Sync positions and seed entry price/high from Alpaca positions if available
    root_to_vt = {w.root: w.vt_symbol for w in watched}
    try:
        alpaca_positions = adapter.client.get_all_positions()
    except Exception:
        alpaca_positions = []

    for pos in alpaca_positions:
        root = str(getattr(pos, "symbol", ""))
        if not root:
            continue
        vt_symbol = root_to_vt.get(root, f"{root}.NASDAQ")
        qty = int(float(getattr(pos, "qty", 0) or 0))
        if qty <= 0:
            continue
        strategy.pos_data[vt_symbol] = qty
        strategy.target_data[vt_symbol] = qty

        avg_entry = float(getattr(pos, "avg_entry_price", 0) or 0)
        if avg_entry > 0:
            strategy.entry_prices.setdefault(vt_symbol, avg_entry)
            px = last_prices.get(root, avg_entry)
            strategy.entry_highs.setdefault(vt_symbol, max(avg_entry, px))


def _build_bar_from_agg(
    agg: Any,
    vt_symbol: str,
    exchange: Exchange,
    dt: datetime,
) -> BarData | None:
    close = getattr(agg, "close", None)
    if close is None:
        return None
    open_ = float(getattr(agg, "open", close) or close)
    high = float(getattr(agg, "high", close) or close)
    low = float(getattr(agg, "low", close) or close)
    close_f = float(close)
    vol = float(getattr(agg, "volume", 0) or 0)

    root = vt_symbol.split(".")[0]
    return BarData(
        gateway_name="POLYGON",
        symbol=root,
        exchange=exchange,
        datetime=dt,
        interval=Interval.MINUTE,
        volume=vol,
        turnover=0,
        open_interest=0,
        open_price=open_,
        high_price=high,
        low_price=low,
        close_price=close_f,
    )


def _fill_bar(
    vt_symbol: str,
    exchange: Exchange,
    dt: datetime,
    last_close: float,
) -> BarData:
    root = vt_symbol.split(".")[0]
    px = float(last_close)
    return BarData(
        gateway_name="FILL",
        symbol=root,
        exchange=exchange,
        datetime=dt,
        interval=Interval.MINUTE,
        volume=0,
        turnover=0,
        open_interest=0,
        open_price=px,
        high_price=px,
        low_price=px,
        close_price=px,
    )


def _load_index_close(lab: AlphaLab, vt_symbol: str, d: date) -> float | None:
    try:
        bars = lab.load_bar_data(vt_symbol=vt_symbol, interval=Interval.DAILY, start=d.isoformat(), end=d.isoformat())
        if not bars:
            return None
        return float(bars[-1].close_price)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship Alpha-Momentum 盘中执行器（Plan B）。")
    parser.add_argument(
        "--mode",
        choices=["exit-only", "full"],
        default="exit-only",
        help="exit-only: 仅允许卖出离场；full: 允许盘中开仓（不推荐）。",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_REGISTRY.keys()),
        default="v7",
        help="策略版本（例如 v5/v7），默认 v7。",
    )
    parser.add_argument(
        "--strategy-name",
        default="IntradayRunner",
        help="策略实例名称（用于日志标识）。",
    )
    parser.add_argument(
        "--use-polygon-ws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 Polygon WS 分钟聚合（默认 true）。",
    )
    parser.add_argument(
        "--symbols-source",
        choices=[SymbolsSource.POSITIONS, SymbolsSource.SIGNAL, SymbolsSource.BOTH],
        default=SymbolsSource.BOTH,
        help="订阅/跟踪的标的来源。",
    )
    parser.add_argument(
        "--signal-top-n",
        type=int,
        default=int(os.getenv("FLAGSHIP_INTRADAY_SIGNAL_TOPN", "10")),
        help="信号订阅限制为 Top-N（默认 10），持仓标的始终包含。",
    )
    parser.add_argument(
        "--rth-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅在 RTH（09:30-16:00 ET）处理与交易。",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=0,
        help=">0 时启用轮询兜底（按 N 秒轮询最新成交价）。",
    )
    parser.add_argument(
        "--stop-after-close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="收盘/提前收盘后退出，避免隔夜挂起（默认 true）。",
    )
    parser.add_argument(
        "--close-grace-seconds",
        type=int,
        default=300,
        help="收盘后延迟退出秒数（默认 300）。",
    )
    parser.add_argument(
        "--ws-reconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="WS 断开/报错时自动重连（默认 true）。",
    )
    parser.add_argument(
        "--ws-max-backoff",
        type=int,
        default=30,
        help="重连最大退避秒数（默认 30）。",
    )
    parser.add_argument(
        "--ws-max-errors-before-poll",
        type=int,
        default=3,
        help="当连续 WS 错误达到 N 次时改用轮询（默认 3）。",
    )
    args = parser.parse_args()

    log_path = Path("logs") / f"intraday_runner_{datetime.now().strftime('%Y%m%d')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(sink=str(log_path), rotation="1 day")

    adapter = AlpacaAdapter()
    lab = AlphaLab(str(LAB_PATH))

    # 读取每日信号快照（若不存在则继续运行）
    signal_df: pl.DataFrame | None = None
    try:
        signal_df = _load_signal_df(Path(DAILY_SIGNAL_FILE))
    except Exception as exc:
        logger.warning(f"[IntradayRunner] Signal not loaded: {exc}")
        signal_df = None

    watched = _build_watchlist(lab, adapter, signal_df, args.symbols_source, int(args.signal_top_n))
    if not watched:
        raise RuntimeError("No symbols to track (empty watchlist).")

    roots = [w.root for w in watched]
    logger.info(f"[IntradayRunner] watchlist={len(watched)} symbols, mode={args.mode}, rth_only={args.rth_only}")

    # Seed last prices from Alpaca positions if available
    last_prices: dict[str, float] = {}
    try:
        for pos in adapter.client.get_all_positions():
            root = str(getattr(pos, "symbol", ""))
            px = float(getattr(pos, "current_price", 0) or 0)
            if root and px > 0:
                last_prices[root] = px
    except Exception:
        pass

    # 信号缺失时，用空 DF 保底（依赖 ATR/价格回落的退出逻辑）
    engine = LiveEngine(
        adapter=adapter,
        signal_df=signal_df if signal_df is not None else pl.DataFrame(),
        mode=args.mode,
    )
    engine.root_close.update(last_prices)

    strategy = build_strategy_instance(
        args.strategy,
        engine=engine,
        vt_symbols=[w.vt_symbol for w in watched],
        strategy_name_override=args.strategy_name,
    )

    today_et = datetime.now(EASTERN).date()
    if args.rth_only and is_market_closed_day(today_et):
        logger.warning(f"[IntradayRunner] Market closed today ({today_et}), exit.")
        return

    # Simple heartbeat marker (updated on every processed minute)
    heartbeat_path = Path("logs") / f"intraday_runner_heartbeat_{today_et.strftime('%Y%m%d')}.txt"
    metrics_writer = TextfileMetricsWriter("flagship_intraday_runner.prom")
    ws_errors_total: int = 0
    ws_consecutive_errors: int = 0

    def _emit_metrics(*, heartbeat_dt: datetime | None = None) -> None:
        ts = time.time()
        samples = [
            Sample(
                name="flagship_intraday_runner_heartbeat_timestamp_seconds",
                value=float(ts),
                labels={"date": today_et.isoformat()},
            ),
            Sample(
                name="flagship_intraday_runner_session_close_timestamp_seconds",
                value=float(session_close_at.timestamp()),
                labels={"date": today_et.isoformat()},
            ),
            Sample(
                name="flagship_intraday_runner_watchlist_size",
                value=float(len(watched)),
            ),
            Sample(
                name="flagship_intraday_runner_ws_errors_total",
                value=float(ws_errors_total),
            ),
            Sample(
                name="flagship_intraday_runner_ws_consecutive_errors",
                value=float(ws_consecutive_errors),
            ),
            Sample(
                name="flagship_intraday_runner_order_submitted_total",
                value=float(engine.order_submitted_total),
            ),
            Sample(
                name="flagship_intraday_runner_order_failed_total",
                value=float(engine.order_failed_total),
            ),
            Sample(
                name="flagship_intraday_runner_order_rejected_total",
                value=float(engine.order_rejected_total),
            ),
            Sample(
                name="flagship_intraday_runner_order_skipped_total",
                value=float(engine.order_skipped_total),
            ),
        ]
        for reason, count in sorted(engine.order_error_reasons.items()):
            samples.append(
                Sample(
                    name="flagship_intraday_runner_order_errors_total",
                    value=float(count),
                    labels={"reason": reason},
                )
            )
        if heartbeat_dt is not None:
            samples.append(
                Sample(
                    name="flagship_intraday_runner_last_processed_minute",
                    value=float(heartbeat_dt.replace(tzinfo=EASTERN).timestamp()),
                )
            )

        metrics_writer.write(
            samples,
            help_map={
                "flagship_intraday_runner_heartbeat_timestamp_seconds": "Intraday runner heartbeat (wall clock).",
                "flagship_intraday_runner_session_close_timestamp_seconds": "Session close timestamp (ET) + grace.",
                "flagship_intraday_runner_watchlist_size": "Number of symbols tracked by intraday runner.",
                "flagship_intraday_runner_ws_errors_total": "Total WS errors observed in current process.",
                "flagship_intraday_runner_ws_consecutive_errors": "Consecutive WS errors (for fallback/backoff).",
                "flagship_intraday_runner_last_processed_minute": "Last processed minute timestamp (ET).",
                "flagship_intraday_runner_order_submitted_total": "Orders submitted by intraday runner (process lifetime).",
                "flagship_intraday_runner_order_failed_total": "Orders failed by intraday runner (process lifetime).",
                "flagship_intraday_runner_order_rejected_total": "Orders rejected by broker (process lifetime).",
                "flagship_intraday_runner_order_skipped_total": "Orders skipped by runner (process lifetime).",
                "flagship_intraday_runner_order_errors_total": "Order errors by reason (process lifetime).",
            },
            type_map={
                "flagship_intraday_runner_heartbeat_timestamp_seconds": "gauge",
                "flagship_intraday_runner_session_close_timestamp_seconds": "gauge",
                "flagship_intraday_runner_watchlist_size": "gauge",
                "flagship_intraday_runner_ws_errors_total": "gauge",
                "flagship_intraday_runner_ws_consecutive_errors": "gauge",
                "flagship_intraday_runner_last_processed_minute": "gauge",
                "flagship_intraday_runner_order_submitted_total": "gauge",
                "flagship_intraday_runner_order_failed_total": "gauge",
                "flagship_intraday_runner_order_rejected_total": "gauge",
                "flagship_intraday_runner_order_skipped_total": "gauge",
                "flagship_intraday_runner_order_errors_total": "gauge",
            },
        )

    # Determine session close time (Polygon holiday early close overrides default 16:00 ET)
    session_close_time: dtime = RTH_CLOSE
    holiday_info = get_holiday_info(today_et)
    if holiday_info and holiday_info.is_early_close and holiday_info.close_time:
        session_close_time = holiday_info.close_time
        logger.info(
            f"[IntradayRunner] early_close detected: date={today_et}, close={session_close_time.strftime('%H:%M')}"
        )

    session_close_at = (
        datetime.combine(today_et, session_close_time, tzinfo=EASTERN)
        + timedelta(seconds=max(0, int(args.close_grace_seconds)))
    )
    _init_strategy_state_from_alpaca(strategy, adapter, watched, last_prices, today=today_et)

    # Load last daily VIX/VIX3M close to adjust stop-loss multiplier (optional)
    data_date = today_et - timedelta(days=1)
    vix_close = _load_index_close(lab, "VIX.CBOE", data_date)
    vix3m_close = _load_index_close(lab, "VIX3M.CBOE", data_date)

    current_minute: datetime | None = None
    pending: dict[str, Any] = {}
    last_close_by_root: dict[str, float] = dict(last_prices)

    def _process_minute(dt_minute: datetime, pending_aggs: dict[str, Any]) -> None:
        if args.rth_only and not _is_rth(dt_minute, close_time=session_close_time):
            return
        try:
            heartbeat_path.write_text(dt_minute.isoformat(), encoding="utf-8")
        except Exception:
            pass
        _emit_metrics(heartbeat_dt=dt_minute)

        bars: dict[str, BarData] = {}
        # Build full bars set for strategy (all same datetime)
        for w in watched:
            agg = pending_aggs.get(w.root)
            bar = _build_bar_from_agg(agg, w.vt_symbol, w.exchange, dt_minute) if agg else None
            if bar is None:
                last_close = last_close_by_root.get(w.root)
                if last_close is None:
                    continue
                bar = _fill_bar(w.vt_symbol, w.exchange, dt_minute, last_close)
            bars[w.vt_symbol] = bar
            last_close_by_root[w.root] = float(bar.close_price)

        # Add constant VIX bars if available
        if vix_close and vix3m_close:
            bars["VIX.CBOE"] = _fill_bar("VIX.CBOE", Exchange.CBOE, dt_minute, float(vix_close))
            bars["VIX3M.CBOE"] = _fill_bar("VIX3M.CBOE", Exchange.CBOE, dt_minute, float(vix3m_close))

        # Update portfolio price map for holding value estimation
        engine.root_close.update(last_close_by_root)

        # Run strategy (exit-only paths will return early)
        strategy.on_bars(bars)

        # 刷新实际持仓，保持策略状态同步，避免重复离场
        pos = adapter.get_positions()
        root_to_vt = {w.root: w.vt_symbol for w in watched}
        for w in watched:
            qty = int(pos.get(w.root, 0))
            vt = root_to_vt.get(w.root, w.vt_symbol)
            strategy.pos_data[vt] = qty
            strategy.target_data[vt] = qty
            if qty <= 0:
                engine.clear_pending_exit(w.root)
                strategy.entry_prices.pop(vt, None)
                strategy.entry_highs.pop(vt, None)

    def _run_poll_loop() -> None:
        if args.poll_seconds <= 0:
            raise ValueError("poll mode requires --poll-seconds > 0")
        logger.warning(f"[IntradayRunner] Polling every {args.poll_seconds}s (fallback mode).")
        last_processed: datetime | None = None
        try:
            while True:
                if args.stop_after_close and datetime.now(EASTERN) >= session_close_at:
                    logger.info(f"[IntradayRunner] stop-after-close reached ({session_close_at}), exit poll loop.")
                    return
                now_et = datetime.now(EASTERN).replace(tzinfo=None)
                dt_minute = now_et.replace(second=0, microsecond=0)
                if last_processed != dt_minute:
                    for root in roots:
                        px = adapter.get_last_trade_price(root)
                        if px and px > 0:
                            last_close_by_root[root] = float(px)
                    _process_minute(dt_minute, {})
                    last_processed = dt_minute
                time.sleep(max(1, int(args.poll_seconds)))
        except KeyboardInterrupt:
            logger.info("[IntradayRunner] stopped by user")

    # --- Polling fallback (no WS) ---
    if not args.use_polygon_ws:
        _run_poll_loop()
        return

    # --- Polygon WS ---
    if (not POLYGON_WS_AVAILABLE) or WebSocketClient is None or Market is None:
        if args.poll_seconds > 0:
            logger.warning(
                f"[IntradayRunner] Polygon WS not available ({polygon_ws_import_error()}), fallback to polling."
            )
            _run_poll_loop()
            return
        raise RuntimeError(f"Polygon WebSocket not available: {polygon_ws_import_error()}")

    subs = [f"AM.{root}" for root in roots]

    def _handle_msg(batch: list[Any]) -> None:
        nonlocal current_minute, pending
        for m in batch:
            if getattr(m, "event_type", None) != "AM":
                continue
            root = getattr(m, "symbol", None)
            if not root:
                continue
            start_ts = getattr(m, "start_timestamp", None)
            if start_ts is None:
                continue
            dt_minute = _utc_ms_to_eastern_naive(int(start_ts))

            if current_minute is None:
                current_minute = dt_minute
            if dt_minute < current_minute:
                continue
            if dt_minute > current_minute:
                _process_minute(current_minute, pending)
                pending = {}
                current_minute = dt_minute

            pending[root] = m

    def _run_ws_loop() -> None:
        """
        Keep WS connection alive:
        - reconnect on transient errors/disconnects
        - optional fallback to polling after repeated WS errors (if --poll-seconds>0)
        """
        async def _connect_until_close(ws: Any) -> None:
            """
            Run websocket connect and ensure we exit after close by scheduling an async close.
            """
            async def _handle_msg_async(msgs: list[Any]) -> None:
                _handle_msg(msgs)

            async def _close_when_due() -> None:
                while True:
                    if datetime.now(EASTERN) >= session_close_at:
                        # Closing the websocket will cause connect() to return (ConnectionClosedOK).
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        return
                    await asyncio.sleep(1)

            close_task = asyncio.create_task(_close_when_due())
            try:
                await ws.connect(_handle_msg_async, close_timeout=1)
            finally:
                close_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await close_task

        backoff = 1
        consecutive_errors = 0
        while True:
            if args.stop_after_close and datetime.now(EASTERN) >= session_close_at:
                logger.info(f"[IntradayRunner] stop-after-close reached ({session_close_at}), exit WS loop.")
                return
            ws = WebSocketClient(api_key=get_polygon_api_key(), market=Market.Stocks, subscriptions=[])
            ws.subscribe(*subs)
            logger.info(f"[IntradayRunner] Subscribed {len(subs)} tickers (minute aggs).")
            try:
                if args.stop_after_close:
                    asyncio.run(_connect_until_close(ws))
                    # If we got here and time is past close, treat as normal stop.
                    if datetime.now(EASTERN) >= session_close_at:
                        logger.info(f"[IntradayRunner] Session ended ({session_close_at}), exit.")
                        return
                    raise RuntimeError("Polygon WS stopped (connect returned early)")
                else:
                    ws.run(_handle_msg)
                    # ws.run() returning usually means disconnected
                    raise RuntimeError("Polygon WS stopped (run() returned)")
            except KeyboardInterrupt:
                logger.info("[IntradayRunner] stopped by user")
                return
            except Exception as exc:
                consecutive_errors += 1
                ws_errors_total = int(ws_errors_total) + 1
                ws_consecutive_errors = int(consecutive_errors)
                _emit_metrics()
                logger.warning(f"[IntradayRunner] Polygon WS error: {exc} (errors={consecutive_errors})")

            if args.poll_seconds > 0 and consecutive_errors >= max(1, int(args.ws_max_errors_before_poll)):
                logger.warning(
                    f"[IntradayRunner] Too many WS errors, fallback to polling (poll_seconds={args.poll_seconds})."
                )
                _run_poll_loop()
                return

            if not args.ws_reconnect:
                raise RuntimeError("Polygon WS disconnected and ws_reconnect is disabled")

            sleep_s = min(int(args.ws_max_backoff), backoff)
            logger.warning(f"[IntradayRunner] WS reconnect in {sleep_s}s ...")
            time.sleep(max(1, sleep_s))
            backoff = min(backoff * 2, max(2, int(args.ws_max_backoff)))

    _run_ws_loop()


if __name__ == "__main__":
    main()


