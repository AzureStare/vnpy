"""
Open rebalance core logic (library module).

This module contains the reusable parts previously embedded in `alpaca_executor.py`:
- StrategyRunner: compute target positions from daily signals + AlphaLab daily bars
- execute_rebalance: compare targets vs actual positions and submit Alpaca market orders

NOTE:
- This module is designed to be imported by both:
  - `flagship/trading/execution/alpaca_executor.py` (CLI)
  - `flagship/trading/execution/executor_daemon.py` (always-on daemon)
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import polars as pl
from alpaca.trading.enums import OrderSide

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData

from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter
from flagship.trading.execution.broker_alpaca import AlpacaAdapter
from flagship.trading.config import LAB_PATH
from flagship.strategy.flagship_alpha_momentum_strategy import FlagshipAlphaMomentumStrategy
from flagship.strategy.flagship_alpha_momentum_strategy_v7 import (
    FlagshipAlphaMomentumStrategy as FlagshipAlphaMomentumStrategyV7,
)

class _MockEngine:
    """
    Minimal engine wrapper to satisfy Strategy requirements for target generation.
    """

    def __init__(self, adapter: AlpacaAdapter):
        self.adapter = adapter
        self.orders: dict[str, Any] = {}
        self.trades: dict[str, Any] = {}
        self.signal_df: pl.DataFrame = pl.DataFrame()
        # root ticker -> last close price
        self.root_close: dict[str, float] = {}

    def get_signal(self) -> pl.DataFrame:
        return self.signal_df

    def get_cash_available(self) -> float:
        try:
            return float(self.adapter.get_cash())
        except Exception:
            return 0.0

    def get_holding_value(self) -> float:
        if not self.root_close:
            return 0.0
        total = 0.0
        for sym, qty in self.adapter.get_positions().items():
            px = self.root_close.get(sym)
            if px is None:
                continue
            total += float(qty) * float(px)
        return total

    def get_position(self, vt_symbol: str) -> float:
        root = vt_symbol.split(".")[0]
        positions = self.adapter.get_positions()
        return float(positions.get(root, 0.0))

    def send_order(self, strategy, vt_symbol, direction, offset, price, volume):
        oid = f"mock_{len(self.orders) + 1}"
        self.orders[oid] = {
            "vt_symbol": vt_symbol,
            "direction": direction,
            "offset": offset,
            "price": price,
            "volume": volume,
        }
        return [oid]

    def cancel_order(self, strategy, order_id):
        return

    def cancel_all(self, strategy):
        return

    def write_log(self, msg: str, strategy=None):
        logger.info(f"[StrategyLog] {msg}")

    def get_pricetick(self, vt_symbol):
        return 0.01

    def get_size(self, vt_symbol):
        return 1


class StrategyRunner:
    """
    Runs the vnpy strategy logic using daily bars and the latest signal snapshot to determine target positions.
    """

    def __init__(self, adapter: AlpacaAdapter, *, strategy_version: str = "v7", lab_path: Path = LAB_PATH):
        self.adapter = adapter
        self.engine = _MockEngine(adapter)
        self.lab = AlphaLab(str(lab_path))

        if strategy_version == "v5":
            StrategyClass = FlagshipAlphaMomentumStrategy
            strategy_name = "Live_Flagship_V5"
            settings = {"top_n": 5}
        else:
            StrategyClass = FlagshipAlphaMomentumStrategyV7
            strategy_name = "Live_Flagship_V7_Aggressive"
            settings = {"top_n": 8}

        self.strategy = StrategyClass(
            strategy_engine=self.engine,  # type: ignore[arg-type]
            strategy_name=strategy_name,
            vt_symbols=[],
            setting=settings,
        )
        self.strategy.on_init()

    def inject_signal(self, signal_path: Path) -> None:
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
        logger.info(f"[OpenRebalance] loading signals from {signal_path}")
        self.engine.signal_df = pl.read_parquet(signal_path)

    def run_daily_logic(self) -> Dict[str, float]:
        signal_df = self.engine.signal_df
        if signal_df.is_empty():
            logger.warning("[OpenRebalance] signal_df is empty, no targets.")
            return {}

        # Determine DATA_DATE from signal parquet
        sig_dt = signal_df.select(pl.col("datetime").max()).item()
        if isinstance(sig_dt, datetime):
            data_date = sig_dt.date()
        else:
            data_date = date.today() - timedelta(days=1)

        # Build vt_symbol universe to load bars for:
        # - all vt_symbols in signal file
        # - all currently held alpaca symbols mapped to vt_symbol
        vt_symbols_from_signal = sorted(set(signal_df["vt_symbol"].to_list()))

        current_positions = self.adapter.get_positions()
        root_to_vt: dict[str, str] = {v.split(".")[0]: str(v) for v in vt_symbols_from_signal}

        vt_symbols_from_positions: list[str] = []
        for sym in current_positions.keys():
            vt = root_to_vt.get(sym)
            if vt:
                vt_symbols_from_positions.append(vt)
                continue

            # fallback: try find unique match in lab files
            candidates = sorted(self.lab.daily_path.glob(f"{sym}.*.parquet"))
            if len(candidates) == 1:
                vt_symbols_from_positions.append(candidates[0].stem)
            else:
                vt_symbols_from_positions.append(f"{sym}.NASDAQ")

        vt_symbols_to_load = sorted(
            set(vt_symbols_from_signal + vt_symbols_from_positions + ["VIX.CBOE", "VIX3M.CBOE"])
        )

        bars: dict[str, BarData] = {}
        root_close: dict[str, float] = {}
        for vt_symbol in vt_symbols_to_load:
            try:
                bar_list = self.lab.load_bar_data(
                    vt_symbol=vt_symbol,
                    interval=Interval.DAILY,
                    start=data_date.isoformat(),
                    end=data_date.isoformat(),
                )
                if not bar_list:
                    continue
                bars[vt_symbol] = bar_list[-1]
                root = vt_symbol.split(".")[0]
                root_close[root] = float(bars[vt_symbol].close_price)
            except Exception as exc:
                logger.warning(f"[OpenRebalance] failed to load daily bar for {vt_symbol} {data_date}: {exc}")
                continue

        self.engine.root_close = root_close
        logger.info(f"[OpenRebalance] constructed {len(bars)} bars for strategy execution.")

        self.strategy.on_bars(bars)
        targets = self.strategy.target_data
        logger.info(f"[OpenRebalance] strategy generated {len(targets)} target positions.")
        return targets


def execute_rebalance(adapter: AlpacaAdapter, targets: Dict[str, float]) -> None:
    """
    Compare targets with actual positions and execute orders (market orders).
    Also writes `flagship_executor.prom` metrics for last run summary.
    """

    metrics_writer = TextfileMetricsWriter("flagship_executor.prom")
    sell_orders_submitted = 0
    buy_orders_submitted = 0
    buy_scale_ratio = 1.0

    current_positions = adapter.get_positions()
    buying_power = adapter.get_buying_power()
    budget = buying_power * 0.98  # safety buffer
    logger.info(f"[Alpaca] buying_power={buying_power:.2f}, budget={budget:.2f}")

    # 1) Sell first
    for vt_symbol, target_qty in targets.items():
        symbol = vt_symbol.split(".")[0]
        current_qty = current_positions.get(symbol, 0)
        if target_qty < current_qty:
            sell_qty = int(current_qty - target_qty)
            if sell_qty > 0:
                logger.info(f"[OpenRebalance] selling {sell_qty} of {symbol}")
                adapter.place_order(symbol, sell_qty, OrderSide.SELL)
                sell_orders_submitted += 1

    # Liquidate positions not in targets
    target_roots = {vt.split(".")[0] for vt in targets.keys()}
    for symbol, current_qty in current_positions.items():
        if symbol not in target_roots and current_qty > 0:
            logger.info(f"[OpenRebalance] liquidating {current_qty} of {symbol} (not in targets)")
            adapter.place_order(symbol, int(current_qty), OrderSide.SELL)
            sell_orders_submitted += 1

    time.sleep(2.0)

    # 2) Recompute buying power and plan buys
    buying_power = adapter.get_buying_power()
    budget = buying_power * 0.98
    logger.info(f"[Alpaca] buying_power_for_buys={buying_power:.2f}, budget={budget:.2f}")

    buy_orders: list[dict[str, float | int | str]] = []
    total_estimated_cost = 0.0

    for vt_symbol, target_qty in targets.items():
        symbol = vt_symbol.split(".")[0]
        current_qty = current_positions.get(symbol, 0)
        if target_qty > current_qty:
            buy_qty = int(target_qty - current_qty)
            if buy_qty <= 0:
                continue
            price = float(adapter.get_last_trade_price(symbol) or 0.0)
            if price <= 0:
                logger.warning(f"[OpenRebalance] cannot get price for {symbol}, skip buy.")
                continue
            cost = float(buy_qty) * price
            buy_orders.append({"symbol": symbol, "qty": buy_qty, "price": price, "cost": cost})
            total_estimated_cost += cost

    # 3) Risk control scaling
    if total_estimated_cost > budget and total_estimated_cost > 0:
        ratio = budget / total_estimated_cost
        buy_scale_ratio = float(ratio)
        logger.warning(
            f"[OpenRebalance] total_cost {total_estimated_cost:.2f} > budget {budget:.2f}. "
            f"scale buys by {ratio:.4f}"
        )
        for order in buy_orders:
            q = int(float(order["qty"]) * ratio)
            order["qty"] = q
            order["cost"] = float(q) * float(order["price"])
    else:
        logger.info(f"[OpenRebalance] total_estimated_cost {total_estimated_cost:.2f} within budget {budget:.2f}")

    # 4) Execute buys
    for order in buy_orders:
        qty = int(order["qty"])
        symbol = str(order["symbol"])
        if qty <= 0:
            continue
        logger.info(
            f"[OpenRebalance] buying {qty} of {symbol} @ ~{float(order['price']):.2f} "
            f"(est_cost={float(order['cost']):.2f})"
        )
        adapter.place_order(symbol, qty, OrderSide.BUY)
        buy_orders_submitted += 1

    metrics_writer.write(
        samples=[
            Sample("flagship_executor_last_rebalance_timestamp_seconds", float(time.time())),
            Sample("flagship_executor_sell_orders_submitted", float(sell_orders_submitted)),
            Sample("flagship_executor_buy_orders_submitted", float(buy_orders_submitted)),
            Sample("flagship_executor_buy_scale_ratio", float(buy_scale_ratio)),
            Sample("flagship_executor_buying_power", float(buying_power)),
            Sample("flagship_executor_budget", float(budget)),
            Sample("flagship_executor_total_estimated_buy_cost", float(total_estimated_cost)),
        ],
        help_map={
            "flagship_executor_last_rebalance_timestamp_seconds": "Last rebalance timestamp (epoch seconds).",
            "flagship_executor_sell_orders_submitted": "Number of sell orders submitted in last rebalance run.",
            "flagship_executor_buy_orders_submitted": "Number of buy orders submitted in last rebalance run.",
            "flagship_executor_buy_scale_ratio": "Buy scaling ratio applied due to buying power constraint (1.0=no scale).",
            "flagship_executor_buying_power": "Buying power observed at the start of buy planning.",
            "flagship_executor_budget": "Budget used for buy planning (buying_power * buffer).",
            "flagship_executor_total_estimated_buy_cost": "Estimated total buy cost before scaling.",
        },
        type_map={
            "flagship_executor_last_rebalance_timestamp_seconds": "gauge",
            "flagship_executor_sell_orders_submitted": "gauge",
            "flagship_executor_buy_orders_submitted": "gauge",
            "flagship_executor_buy_scale_ratio": "gauge",
            "flagship_executor_buying_power": "gauge",
            "flagship_executor_budget": "gauge",
            "flagship_executor_total_estimated_buy_cost": "gauge",
        },
    )


