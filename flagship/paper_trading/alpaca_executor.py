"""
Alpaca execution engine for Flagship Alpha-Momentum.
Interacts with Alpaca API to execute trades based on strategy target positions.
"""
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Any
import polars as pl
from datetime import date

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from vnpy.trader.logger import logger
from vnpy.trader.object import BarData, TickData, TradeData
from vnpy.trader.constant import Direction, Exchange, Interval

from flagship.paper_trading.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, 
    DAILY_SIGNAL_FILE, TOP_N
)
from flagship.strategy.flagship_alpha_momentum_strategy import FlagshipAlphaMomentumStrategy

class AlpacaAdapter:
    """Wrapper for Alpaca API interaction"""
    def __init__(self):
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        account = self.client.get_account()
        logger.info(f"[Alpaca] Connected. Status: {account.status}, Equity: {account.equity}, Buying Power: {account.buying_power}")

    def get_positions(self) -> Dict[str, int]:
        """Get current positions {symbol: qty}"""
        positions = {}
        try:
            alpaca_positions = self.client.get_all_positions()
            for pos in alpaca_positions:
                positions[pos.symbol] = int(pos.qty)
        except Exception as e:
            logger.error(f"[Alpaca] Error fetching positions: {e}")
        return positions

    def get_cash(self) -> float:
        """Get available cash"""
        try:
            acct = self.client.get_account()
            return float(acct.cash)
        except Exception as e:
            logger.error(f"[Alpaca] Error fetching cash: {e}")
            return 0.0

    def place_order(self, symbol: str, qty: int, side: OrderSide, type: str = "market") -> None:
        """Place an order"""
        if qty <= 0:
            return
            
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        try:
            self.client.submit_order(order_data=req)
            logger.info(f"[Alpaca] Submitted {side} order for {qty} shares of {symbol}")
        except APIError as e:
            logger.error(f"[Alpaca] Order failed for {symbol}: {e}")

class MockEngine:
    """
    Minimal mock of BacktestingEngine/CtaEngine to satisfy Strategy requirements.
    Used to run the strategy logic in a standalone script.
    """
    def __init__(self):
        self.orders = {}
        self.trades = {}
        # We need to capture target positions from the strategy
        self.target_positions = {} 

    def send_order(self, strategy, vt_symbol, direction, offset, price, volume, stop, lock):
        """Capture order instructions"""
        # In this mock, we don't simulate order lifecycles fully.
        # We assume the strategy sets targets using set_target logic usually,
        # but if it uses send_order directly, we log it.
        # Note: Flagship strategy uses set_target if it inherits from AlphaStrategy template?
        # Checking Strategy Code: It uses self.set_target(vt_symbol, 0) and execute_trading
        pass
        
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
        self.engine = MockEngine()
        
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
        self.strategy.on_start()
    
    def inject_signal(self, signal_path: Path):
        """Load signal parquet and inject into strategy"""
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
            
        logger.info(f"Loading signals from {signal_path}")
        df = pl.read_parquet(signal_path)
        
        # The strategy expects a get_signal() method or similar. 
        # AlphaStrategy template usually handles signal loading via engine.
        # We need to monkey-patch or manually set the signal DataFrame on the strategy/engine.
        
        # In standard AlphaStrategy, self.get_signal() calls engine.get_signal().
        # Let's mock that on the engine.
        self.engine.signal_df = df
        
        # Monkey patch strategy.get_signal
        def get_signal_mock():
            # Strategy logic filters by date/time usually.
            # For live execution, we return the WHOLE signal df (assuming it's just today's)
            # or let the strategy logic filter it.
            # Strategy.on_bars calls get_signal().
            return df
            
        self.strategy.get_signal = get_signal_mock
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
        for sym, qty in current_positions.items():
            vt_symbol = f"{sym}.SMART" # Assuming SMART exchange
            self.strategy.pos_data[vt_symbol] = qty
            
        # 2. Construct BarData for Yesterday (trigger)
        # We need yesterday's prices for ALL symbols in our signal file + held positions
        # Load from the downloaded parquet files
        # Efficient way: Scan the 'daily_signal.parquet' which likely has 'close_price'
        # if we exported it in run_live_inference.
        
        signal_df = self.engine.signal_df
        if "close_price" not in signal_df.columns:
            logger.error("Signal file missing 'close_price'. Cannot execute strategy logic.")
            return

        bars = {}
        for row in signal_df.iter_rows(named=True):
            sym = row['vt_symbol']
            # strip exchange suffix if present for Alpaca
            # vnpy symbol: AAPL.NASDAQ -> Alpaca: AAPL
            # But strategy uses vt_symbol internally.
            
            # Construct Bar
            # We use the close price from signal file (yesterday's close)
            bar = BarData(
                symbol=sym.split('.')[0],
                exchange=Exchange.SMART,
                datetime=row['datetime'], # Signal date
                interval=Interval.DAILY,
                open_price=row['close_price'], # Approximation if we lack full OHLC
                high_price=row['close_price'],
                low_price=row['close_price'],
                close_price=row['close_price'],
                volume=0,
                gateway_name="ALPACA"
            )
            bars[sym] = bar
            
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

    # Wait for sells to process? (Paper trading is fast, but async)
    # Ideally wait or use buying power check.
    
    # 2. Buy
    for vt_symbol, target_qty in targets.items():
        symbol = vt_symbol.split('.')[0]
        current_qty = current_positions.get(symbol, 0)
        
        if target_qty > current_qty:
            buy_qty = int(target_qty - current_qty)
            if buy_qty > 0:
                logger.info(f"Buying {buy_qty} of {symbol}")
                adapter.place_order(symbol, buy_qty, OrderSide.BUY)

def main():
    logger.info("Starting Alpaca Executor...")
    
    # 1. Setup Adapter
    try:
        adapter = AlpacaAdapter()
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return

    # 2. Setup Strategy Runner
    runner = StrategyRunner(adapter)
    
    # 3. Inject Signals
    runner.inject_signal(DAILY_SIGNAL_FILE)
    
    # 4. Run Logic
    targets = runner.run_daily_logic()
    
    if not targets:
        logger.info("No targets generated.")
        return

    # 5. Execute
    execute_rebalance(adapter, targets)
    logger.info("Execution cycle complete.")

if __name__ == "__main__":
    main()

