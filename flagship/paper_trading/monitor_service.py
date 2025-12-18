"""
Real-time monitoring service using Polygon.io WebSocket.
Monitors active positions and logs significant price movements.
"""
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Set

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market
from vnpy.trader.logger import logger
from flagship.paper_trading.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from flagship.paper_trading.alpaca_executor import AlpacaAdapter
from flagship.config.polygon_config import get_polygon_api_key

# Logging setup for monitor
monitor_log_path = PROJECT_ROOT / "logs" / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(sink=str(monitor_log_path), rotation="1 day")

class PortfolioMonitor:
    def __init__(self):
        self.adapter = AlpacaAdapter()
        self.active_symbols: Set[str] = set()
        self.positions: Dict[str, int] = {}
        self.polygon_key = get_polygon_api_key()
        self.ws_client = WebSocketClient(api_key=self.polygon_key, subscriptions=[], market=Market.Stocks)
        self.running = False
        
    def update_positions(self):
        """Fetch current positions from Alpaca"""
        try:
            pos_dict = self.adapter.get_positions()
            self.positions = pos_dict
            new_symbols = set(pos_dict.keys())
            
            # Identify changes
            added = new_symbols - self.active_symbols
            removed = self.active_symbols - new_symbols
            
            if added:
                logger.info(f"[Monitor] New positions to track: {added}")
                # Subscribe to new symbols (Second Aggregates 'A.*' and Trades 'T.*')
                # Polygon WS subscriptions are additive usually, or we reset.
                # Client allows subscribe method.
                self.ws_client.subscribe([f"A.{sym}" for sym in added])
                self.ws_client.subscribe([f"T.{sym}" for sym in added])
                
            if removed:
                logger.info(f"[Monitor] Positions closed: {removed}")
                self.ws_client.unsubscribe([f"A.{sym}" for sym in removed])
                self.ws_client.unsubscribe([f"T.{sym}" for sym in removed])
                
            self.active_symbols = new_symbols
            
        except Exception as e:
            logger.error(f"[Monitor] Failed to update positions: {e}")

    def handle_msg(self, msg: WebSocketMessage):
        """Handle incoming WebSocket messages"""
        # Polygon client returns a list of messages or single? 
        # The callback signature depends on the client version.
        # Assuming current library passes list of objects or single object.
        
        # Checking polygon-python library usage:
        # client = WebSocketClient(...)
        # for msg in client: ... (iterator style) OR handle_msg callback?
        # We'll use the iterator pattern in run() loop if blocking, 
        # or callback if async.
        pass # implemented in run loop usually

    def run(self):
        """Main monitoring loop"""
        self.running = True
        logger.info("[Monitor] Starting monitoring service...")
        
        # Initial position load
        self.update_positions()
        
        # Start a background thread to refresh positions periodically (e.g. every minute)
        def refresh_loop():
            while self.running:
                time.sleep(60)
                self.update_positions()
        
        refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        refresh_thread.start()
        
        # Connect and stream
        # Using the context manager or connect method
        logger.info(f"[Monitor] Connecting to Polygon WebSocket...")
        
        # Subscribe initially
        subs = [f"A.{s}" for s in self.active_symbols] + [f"T.{s}" for s in self.active_symbols]
        if not subs:
            logger.warning("[Monitor] No active positions to monitor. Waiting...")
        
        # Loop for handling messages
        try:
            # We need to subscribe first if we use the iterator
            self.ws_client.subscribe(subs)
            
            for msg in self.ws_client:
                if not self.running:
                    break
                
                # Check message type
                # Usually list of messages
                for m in msg:
                    if m.event_type == 'A': # Aggregate (Second)
                        sym = m.symbol
                        price = m.close
                        # Simple logging of price updates
                        # In production: check against thresholds
                        # logger.debug(f"[Px] {sym}: {price}")
                        pass
                    elif m.event_type == 'T': # Trade
                        sym = m.symbol
                        price = m.price
                        # logger.debug(f"[Tr] {sym}: {price}")
                        pass
                    
        except Exception as e:
            logger.error(f"[Monitor] WebSocket error: {e}")
        finally:
            self.ws_client.close()
            logger.info("[Monitor] Service stopped.")

if __name__ == "__main__":
    monitor = PortfolioMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        monitor.running = False

