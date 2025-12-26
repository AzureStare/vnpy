"""
Alpaca execution engine for Flagship Alpha-Momentum.
Interacts with Alpaca API to execute trades based on strategy target positions.
"""
from __future__ import annotations

import sys
import argparse
import time
import math
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import polars as pl
from datetime import date, datetime, timedelta, timezone

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.trading.enums import OrderSide

from vnpy.trader.logger import logger
from vnpy.trader.object import BarData, TickData, TradeData
from vnpy.trader.constant import Direction, Exchange, Interval
from vnpy.alpha.lab import AlphaLab

from flagship.paper_trading.config import DAILY_SIGNAL_FILE, TOP_N, LAB_PATH
from flagship.strategy.flagship_alpha_momentum_strategy import FlagshipAlphaMomentumStrategy
from flagship.config.polygon_config import get_polygon_api_key
from flagship.paper_trading.broker_alpaca import AlpacaAdapter
from flagship.paper_trading.polygon_ws import (
    Market,
    POLYGON_WS_AVAILABLE,
    WebSocketClient,
    polygon_ws_import_error,
)


class PolygonPriceCache:
    """
    Lightweight Polygon WS price cache for tickers we plan to trade.
    Keeps latest trade price per symbol (root ticker, e.g. 'AAPL').
    """

    def __init__(self, api_key: str, symbols: list[str]) -> None:
        self.api_key = api_key
        self.symbols = sorted(set(symbols))
        self.latest_trade: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._ws: Any | None = None

    def start(self) -> None:
        if not POLYGON_WS_AVAILABLE:
            err = polygon_ws_import_error()
            detail = f" ({err})" if err else ""
            logger.warning(f"[PolygonWS] polygon websocket client not available{detail}, skipping ws subscription.")
            return
        if not self.symbols:
            logger.warning("[PolygonWS] empty symbols, skipping ws subscription.")
            return

        self._running = True
        subs = [f"T.{sym}" for sym in self.symbols]
        self._ws = WebSocketClient(api_key=self.api_key, market=Market.Stocks, subscriptions=[])

        def _run() -> None:
            try:
                # subscribe uses varargs strings, not list
                self._ws.subscribe(*subs)
                logger.info(f"[PolygonWS] Subscribed {len(subs)} tickers (trades).")

                def _handle_msg(batch: list[Any]) -> None:
                    if not self._running:
                        return
                    try:
                        for m in batch:
                            if getattr(m, "event_type", None) == "T":
                                sym = getattr(m, "symbol", None)
                                px = getattr(m, "price", None)
                                if sym and px is not None:
                                    self.latest_trade[str(sym)] = float(px)
                    except Exception:
                        return

                # run blocks in this thread
                self._ws.run(_handle_msg)
            except Exception as exc:
                logger.warning(f"[PolygonWS] websocket stopped: {exc}")
            finally:
                logger.info("[PolygonWS] websocket closed.")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_price(self, symbol: str) -> float | None:
        return self.latest_trade.get(symbol)

class MockEngine:
    """
    Minimal mock of BacktestingEngine/CtaEngine to satisfy Strategy requirements.
    Used to run the strategy logic in a standalone script.
    """
    def __init__(self, adapter: AlpacaAdapter):
        self.adapter = adapter
        self.orders: dict[str, Any] = {}
        self.trades: dict[str, Any] = {}
        self.signal_df: pl.DataFrame = pl.DataFrame()
        # root ticker -> last close price (actual, not normalized)
        self.root_close: dict[str, float] = {}

    def get_signal(self) -> pl.DataFrame:
        return self.signal_df

    def get_cash_available(self) -> float:
        # Use min(cash, buying_power) to avoid rejection while not unintentionally using leverage.
        try:
            info = self.adapter.get_account_info()
            return min(info.cash, info.buying_power)
        except Exception:
            return self.adapter.get_cash()

    def get_holding_value(self) -> float:
        """
        Approx holding market value using latest close_price in signal_df.
        This is sufficient for risk checks in strategy init and sizing.
        """
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
        """Get position for a specific vt_symbol."""
        root = vt_symbol.split('.')[0]
        positions = self.adapter.get_positions()
        return float(positions.get(root, 0.0))

    def send_order(self, strategy, vt_symbol, direction, offset, price, volume):
        """Capture order instructions"""
        # In this mock, we don't simulate order lifecycles fully.
        # We assume the strategy sets targets using set_target logic usually,
        # but if it uses send_order directly, we log it.
        # Note: Flagship strategy uses set_target if it inherits from AlphaStrategy template?
        # Checking Strategy Code: It uses self.set_target(vt_symbol, 0) and execute_trading
        oid = f"mock_{len(self.orders)+1}"
        self.orders[oid] = {
            "vt_symbol": vt_symbol,
            "direction": direction,
            "offset": offset,
            "price": price,
            "volume": volume,
        }
        return [oid]
        
    def cancel_order(self, strategy, order_id):
        pass
    
    def cancel_all(self, strategy):
        pass
    
    def write_log(self, msg: str, strategy = None):
        logger.info(f"[StrategyLog] {msg}")
    
    def get_pricetick(self, vt_symbol):
        return 0.01
        
    def get_size(self, vt_symbol):
        return 1

class StrategyRunner:
    """
    Runs the vnpy strategy logic using live data to determine target positions.
    """
    def __init__(self, adapter: AlpacaAdapter):
        self.adapter = adapter
        self.engine = MockEngine(adapter)
        self.lab = AlphaLab(str(LAB_PATH))
        
        # Initialize Strategy
        # settings: top_n, etc. from config
        settings = {"top_n": TOP_N} 
        self.strategy = FlagshipAlphaMomentumStrategy(
            strategy_engine=self.engine,
            strategy_name="Live_Flagship_V5",
            vt_symbols=[], # Will be populated dynamically or ignored
            setting=settings
        )
        self.strategy.on_init()
        # AlphaStrategy 模板没有 on_start，这里保持兼容不调用
    
    def inject_signal(self, signal_path: Path):
        """Load signal parquet and inject into strategy"""
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
            
        logger.info(f"Loading signals from {signal_path}")
        df = pl.read_parquet(signal_path)
        
        # The strategy expects a get_signal() method or similar. 
        # AlphaStrategy template usually handles signal loading via engine.
        # We need to monkey-patch or manually set the signal DataFrame on the strategy/engine.
        
        # AlphaStrategy.get_signal() calls engine.get_signal()
        self.engine.signal_df = df
        logger.info("Signal injected.")

    def run_daily_logic(self):
        """
        Feed 'current' market data to the strategy to trigger rebalancing.
        Since we are running BEFORE market open, we feed YESTERDAY's close data
        as if it were the "on_bars" event.
        """
        # 1. Get positions from Alpaca to sync strategy state
        # (Optional: sync self.strategy.pos_data with actual alpaca positions?)
        current_positions = self.adapter.get_positions()
        signal_df = self.engine.signal_df
        if signal_df.is_empty() or "vt_symbol" not in signal_df.columns or "datetime" not in signal_df.columns:
            logger.error("Signal file is empty or missing required columns (vt_symbol/datetime).")
            return

        # Map Alpaca root symbol -> vt_symbol used in signal (to keep pos_data consistent)
        root_to_vt: dict[str, str] = {}
        try:
            for vt in signal_df["vt_symbol"].to_list():
                root = str(vt).split(".")[0]
                root_to_vt[root] = str(vt)
        except Exception:
            root_to_vt = {}

        for sym, qty in current_positions.items():
            vt_symbol = root_to_vt.get(sym, f"{sym}.NASDAQ")
            self.strategy.pos_data[vt_symbol] = qty
            
        logger.info(f"[Sync] Alpaca positions: {current_positions}")
        logger.info(f"[Sync] Mapped strategy positions: {dict(self.strategy.pos_data)}")
            
        # Determine signal date (DATA_DATE)
        sig_dt = signal_df.select(pl.col("datetime").max()).item()
        if isinstance(sig_dt, datetime):
            data_date = sig_dt.date()
        else:
            data_date = date.today() - timedelta(days=1)  # fallback, should not happen

        # Build vt_symbol universe to load bars for:
        # - all vt_symbols in signal file
        # - all currently held alpaca symbols mapped to vt_symbol
        vt_symbols_from_signal = sorted(set(signal_df["vt_symbol"].to_list()))

        vt_symbols_from_positions: list[str] = []
        for sym in current_positions.keys():
            vt = root_to_vt.get(sym)
            if vt:
                vt_symbols_from_positions.append(vt)
            else:
                # fallback: try find unique match in lab files
                candidates = sorted(self.lab.daily_path.glob(f"{sym}.*.parquet"))
                if len(candidates) == 1:
                    vt_symbols_from_positions.append(candidates[0].stem)
                else:
                    vt_symbols_from_positions.append(f"{sym}.NASDAQ")

        vt_symbols_to_load = sorted(set(vt_symbols_from_signal + vt_symbols_from_positions + ["VIX.CBOE", "VIX3M.CBOE"]))

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
                logger.warning(f"[AlpacaExecutor] Failed to load daily bar for {vt_symbol} {data_date}: {exc}")
                continue

        # Feed price map into engine for portfolio value estimation
        self.engine.root_close = root_close
            
        logger.info(f"Constructed {len(bars)} bars for strategy execution.")
        
        # 3. Trigger Strategy
        # This will calculate target positions
        self.strategy.on_bars(bars)
        
        # 4. Extract Targets
        # Strategy updates self.target_data
        targets = self.strategy.target_data
        logger.info(f"Strategy generated {len(targets)} target positions.")
        return targets

def execute_rebalance(adapter: AlpacaAdapter, targets: Dict[str, float]):
    """
    Compare targets with actual positions and execute orders.
    """
    current_positions = adapter.get_positions()
    # Use current buying power as budget guardrail
    buying_power = adapter.get_buying_power()
    budget = buying_power * 0.98  # safety buffer
    logger.info(f"[Alpaca] buying_power={buying_power:.2f}, budget={budget:.2f}")
    
    # 1. Sell first to free up cash
    for vt_symbol, target_qty in targets.items():
        symbol = vt_symbol.split('.')[0]
        current_qty = current_positions.get(symbol, 0)
        
        if target_qty < current_qty:
            sell_qty = int(current_qty - target_qty)
            if sell_qty > 0:
                logger.info(f"Selling {sell_qty} of {symbol}")
                adapter.place_order(symbol, sell_qty, OrderSide.SELL)
                
    # Check for full liquidations (targets that are 0 but held)
    for symbol, current_qty in current_positions.items():
        # Reconstruct vt_symbol might be tricky without map, iterate targets
        # Assuming simple mapping for now.
        found = False
        for vt in targets:
            if vt.split('.')[0] == symbol:
                found = True
                break
        if not found and current_qty > 0:
             logger.info(f"Liquidating {current_qty} of {symbol} (No longer in target)")
             adapter.place_order(symbol, int(current_qty), OrderSide.SELL)

    # Wait for sells to process
    time.sleep(2.0)
    
    # Refresh Buying Power after sells
    buying_power = adapter.get_buying_power()
    budget = buying_power * 0.98
    logger.info(f"[Alpaca] Buying Power for buys: {buying_power:.2f}, Budget: {budget:.2f}")

    # 2. Buy Planning
    buy_orders = [] # (symbol, qty, price)
    total_estimated_cost = 0.0
    
    for vt_symbol, target_qty in targets.items():
        symbol = vt_symbol.split('.')[0]
        current_qty = current_positions.get(symbol, 0)
        
        if target_qty > current_qty:
            buy_qty = int(target_qty - current_qty)
            if buy_qty > 0:
                # Estimate cost
                price = adapter.get_last_trade_price(symbol)
                if price <= 0:
                    logger.warning(f"Could not get price for {symbol}, skipping buy.")
                    continue
                cost = buy_qty * price
                buy_orders.append({"symbol": symbol, "qty": buy_qty, "price": price, "cost": cost})
                total_estimated_cost += cost

    # 3. Risk Control & Scaling
    if total_estimated_cost > budget:
        ratio = budget / total_estimated_cost
        logger.warning(f"Total cost {total_estimated_cost:.2f} > Budget {budget:.2f}. Scaling buys by {ratio:.4f}")
        for order in buy_orders:
            order["qty"] = int(order["qty"] * ratio)
            # update cost for log
            order["cost"] = order["qty"] * order["price"]
    else:
        logger.info(f"Total estimated cost {total_estimated_cost:.2f} within budget {budget:.2f}")

    # 4. Execute Buys
    for order in buy_orders:
        qty = order["qty"]
        symbol = order["symbol"]
        if qty > 0:
            logger.info(f"Buying {qty} of {symbol} @ ~{order['price']:.2f} (Est Cost: {order['cost']:.2f})")
            adapter.place_order(symbol, qty, OrderSide.BUY)


def main():
    parser = argparse.ArgumentParser(description="Execute Flagship Alpha-Momentum trades on Alpaca Paper.")
    parser.add_argument(
        "--wait-for-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait until market open before placing orders (default: true).",
    )
    parser.add_argument("--max-wait-seconds", type=int, default=10 * 3600, help="Max seconds to wait for open.")
    parser.add_argument(
        "--cancel-open-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cancel all open orders before execution (default: true).",
    )
    parser.add_argument(
        "--use-polygon-ws",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Subscribe Polygon WS for tickers (pre-open price monitoring).",
    )
    args = parser.parse_args()

    logger.info("Starting Alpaca Executor...")
    
    # 1. Setup Adapter
    try:
        adapter = AlpacaAdapter()
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return

    if args.cancel_open_orders:
        adapter.cancel_all_open_orders()

    # 2. Setup Strategy Runner
    runner = StrategyRunner(adapter)
    
    # 3. Inject Signals
    runner.inject_signal(DAILY_SIGNAL_FILE)
    
    # 4. Run Logic
    targets = runner.run_daily_logic()
    
    if not targets:
        logger.info("No targets generated.")
        return

    # Optional: subscribe Polygon WS (pre-open monitoring)
    ws_cache: PolygonPriceCache | None = None
    if args.use_polygon_ws:
        try:
            sig_df = runner.engine.signal_df
            roots = sorted({str(v).split(".")[0] for v in sig_df["vt_symbol"].to_list()})
            ws_cache = PolygonPriceCache(get_polygon_api_key(), roots)
            ws_cache.start()
        except Exception as exc:
            logger.warning(f"[PolygonWS] failed to start: {exc}")
            ws_cache = None

    # 5. Wait for market open if needed
    if args.wait_for_open:
        ok = adapter.wait_for_open(max_wait_seconds=args.max_wait_seconds)
        if not ok:
            logger.warning("[AlpacaExecutor] Market did not open within max wait, skip execution.")
            return
    else:
        # Safety: never place orders when market is closed unless user explicitly enables waiting.
        if not adapter.is_market_open():
            logger.warning("[AlpacaExecutor] Market is CLOSED and --no-wait-for-open is set. Skip execution.")
            return

    # 6. Refresh account/buying power right before execution
    info = adapter.get_account_info()
    logger.info(f"[Alpaca] Pre-trade Account: cash={info.cash:.2f}, equity={info.equity:.2f}, buying_power={info.buying_power:.2f}")

    # 7. Execute
    execute_rebalance(adapter, targets)
    logger.info("Execution cycle complete.")

    if ws_cache:
        ws_cache.stop()

if __name__ == "__main__":
    main()

