"""
Shared Alpaca adapter for paper trading scripts.

Keep all Alpaca client wiring in one place to reduce duplication across:
- alpaca_executor.py (open rebalance)
- intraday_runner.py (intraday exits)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from vnpy.trader.logger import logger

from flagship.paper_trading.config import ALPACA_API_KEY, ALPACA_PAPER, ALPACA_SECRET_KEY


@dataclass(frozen=True)
class AccountInfo:
    cash: float
    equity: float
    buying_power: float


class AlpacaAdapter:
    """Wrapper for Alpaca API interaction"""

    def __init__(self) -> None:
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        account = self.client.get_account()
        logger.info(
            f"[Alpaca] Connected. Status: {account.status}, Equity: {account.equity}, "
            f"Cash: {account.cash}, Buying Power: {account.buying_power}"
        )

    def get_account_info(self) -> AccountInfo:
        acct = self.client.get_account()
        return AccountInfo(
            cash=float(acct.cash),
            equity=float(acct.equity),
            buying_power=float(acct.buying_power),
        )

    def get_buying_power(self) -> float:
        return self.get_account_info().buying_power

    def is_market_open(self) -> bool:
        clock = self.client.get_clock()
        return bool(getattr(clock, "is_open", False))

    def wait_for_open(self, max_wait_seconds: int = 10 * 3600) -> bool:
        """
        Wait until market open (Alpaca clock). Returns True if market opens within max_wait_seconds.
        """
        import time

        start_ts = time.time()
        while True:
            clock = self.client.get_clock()
            if bool(getattr(clock, "is_open", False)):
                logger.info("[Alpaca] Market is OPEN.")
                return True

            now = getattr(clock, "timestamp", None) or datetime.now(timezone.utc)
            # Ensure timezone-aware timestamps for safe subtraction
            if isinstance(now, datetime) and now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)

            next_open = getattr(clock, "next_open", None)
            if isinstance(next_open, datetime) and next_open.tzinfo is None:
                next_open = next_open.replace(tzinfo=timezone.utc)

            if next_open:
                wait_sec = max(0.0, (next_open - now).total_seconds())
                logger.info(f"[Alpaca] Market CLOSED. next_open={next_open}, wait≈{int(wait_sec)}s")
            else:
                wait_sec = 60.0
                logger.info("[Alpaca] Market CLOSED. next_open unknown, sleep 60s")

            if time.time() - start_ts > max_wait_seconds:
                logger.warning("[Alpaca] wait_for_open timeout, giving up.")
                return False

            time.sleep(min(60.0, max(5.0, wait_sec)))

    def cancel_all_open_orders(self) -> None:
        try:
            self.client.cancel_orders()
            logger.info("[Alpaca] Requested cancel of all open orders.")
        except Exception as exc:
            logger.warning(f"[Alpaca] Failed to cancel open orders: {exc}")

    def get_positions(self) -> Dict[str, int]:
        """Get current positions {symbol: qty}"""
        positions: Dict[str, int] = {}
        try:
            alpaca_positions = self.client.get_all_positions()
            for pos in alpaca_positions:
                positions[pos.symbol] = int(pos.qty)
        except Exception as exc:
            logger.error(f"[Alpaca] Error fetching positions: {exc}")
        return positions

    def get_cash(self) -> float:
        """Get available cash"""
        try:
            acct = self.client.get_account()
            return float(acct.cash)
        except Exception as exc:
            logger.error(f"[Alpaca] Error fetching cash: {exc}")
            return 0.0

    def get_last_trade_price(self, symbol: str) -> float:
        """Get last trade price for a symbol"""
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.data_client.get_stock_latest_trade(req)
            return float(trade[symbol].price)
        except Exception as exc:
            logger.warning(f"[Alpaca] Failed to get last trade for {symbol}: {exc}")
            return 0.0

    def place_order(self, symbol: str, qty: int, side: OrderSide) -> None:
        """Place a market order"""
        if qty <= 0:
            return

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            self.client.submit_order(order_data=req)
            logger.info(f"[Alpaca] Submitted {side} order for {qty} shares of {symbol}")
        except APIError as exc:
            logger.error(f"[Alpaca] Order failed for {symbol}: {exc}")


