from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pytz

from vnpy.alpha.strategy import AlphaStrategy
from vnpy.trader.constant import Direction, Interval, Offset, OrderType, Status
from vnpy.trader.object import BarData, OrderData, TickData, TradeData
from vnpy.trader.utility import BarGenerator

class FlagshipAlphaMomentumStrategy(AlphaStrategy):
    """
    Flagship Alpha-Momentum Strategy V5 (Restored)
    
    Logic:
    1. Select Top N stocks based on Alpha Score (LightGBM/Composite).
    2. Buy on Open (or close of previous day).
    3. Hold for max 5 days.
    4. Exits:
       - Stop Loss: 3.0 ATR (tighten to 1.5 ATR if VIX panic).
       - Take Profit: Max(2.0 ATR, 5%).
       - Trailing Stop: 1.5 ATR from High (if new high reached).
       - Time Stop: 5 days.
       - Rank Decay: If score drops out of Top N * Buffer.
    """

    author = "Flagship Capital"
    
    # Strategy parameters
    top_n: int = 5
    min_score_threshold: float = 0.5
    min_quantile_threshold: float = 80.0 # Top 20%
    
    # Risk parameters
    max_holding_days: int = 5
    stop_loss_atr_multiplier: float = 3.0
    panic_stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.0
    take_profit_min_pct: float = 0.05
    trailing_stop_atr_multiplier: float = 1.5
    
    # Position sizing
    max_pos_weight: float = 0.20 # 5 stocks = 20% each (theoretical max)
    
    # VIX Filter
    vix_panic_threshold: float = 1.1 # VIX / VIX3M > 1.1 => Panic

    parameters = [
        "top_n",
        "min_score_threshold",
        "min_quantile_threshold",
        "max_holding_days",
        "stop_loss_atr_multiplier",
        "panic_stop_loss_atr_multiplier",
        "take_profit_atr_multiplier",
        "take_profit_min_pct",
        "trailing_stop_atr_multiplier",
        "max_pos_weight",
        "vix_panic_threshold",
    ]

    def __init__(
        self,
        alpha_engine: AlphaEngine,
        strategy_name: str,
        vt_symbols: List[str],
        setting: Dict[str, Any],
    ) -> None:
        super().__init__(alpha_engine, strategy_name, vt_symbols, setting)
        
        self.entry_prices: Dict[str, float] = {}
        self.entry_times: Dict[str, datetime] = {}
        self.high_prices: Dict[str, float] = {} # For trailing stop
        self.holding_days: Dict[str, int] = {}
        
        self.atr_map: Dict[str, float] = {}
        
        # State
        self.target_portfolio: Set[str] = set()
        self.panic_mode: bool = False

    def on_init(self) -> None:
        self.write_log("Flagship Alpha-Momentum V5 Strategy Initialized")
        self.load_bars(10)

    def on_start(self) -> None:
        self.write_log("Flagship Alpha-Momentum V5 Strategy Started")

    def on_stop(self) -> None:
        self.write_log("Flagship Alpha-Momentum V5 Strategy Stopped")

    def on_daily_bar(self, bar: BarData) -> None:
        """Daily logic (Portfolio Rebalancing)"""
        # Usually called by engine at end of day or explicitly via timer
        pass

    def on_min_bar(self, bar: BarData) -> None:
        """Intraday logic (Risk Management & Execution)"""
        vt_symbol = bar.vt_symbol
        
        # Update ATR (simplified, assumes daily ATR available or calculated elsewhere)
        # Here we rely on pre-calculated ATR passed via signal or calculated on fly
        # For simplicity, we assume we have daily ATR from external source or bar manager
        atr = self.atr_map.get(vt_symbol, bar.close_price * 0.02) # Fallback 2%
        
        if self.trading:
            self._check_risk_management(bar, atr)

    def _check_risk_management(self, bar: BarData, atr: float) -> None:
        """Check Stop Loss, Take Profit, Time Stop"""
        vt_symbol = bar.vt_symbol
        pos = self.get_pos(vt_symbol)
        
        if pos == 0:
            return
        
        entry_price = self.entry_prices.get(vt_symbol, bar.close_price)
        high_price = self.high_prices.get(vt_symbol, entry_price)
        
        # Update High Price
        if bar.high_price > high_price:
            self.high_prices[vt_symbol] = bar.high_price
            high_price = bar.high_price
            
        # 1. Take Profit
        tp_price = entry_price + max(self.take_profit_atr_multiplier * atr, entry_price * self.take_profit_min_pct)
        if bar.high_price >= tp_price:
            self.sell(vt_symbol, bar.close_price, pos, "Take Profit")
            return
        
        # 2. Trailing Stop
        trailing_stop_price = high_price - self.trailing_stop_atr_multiplier * atr
        if bar.low_price <= trailing_stop_price:
             self.sell(vt_symbol, bar.close_price, pos, "Trailing Stop")
             return

        # 3. Stop Loss (Adaptive)
        sl_multiplier = self.panic_stop_loss_atr_multiplier if self.panic_mode else self.stop_loss_atr_multiplier
        sl_price = entry_price - sl_multiplier * atr
        if bar.low_price <= sl_price:
            self.sell(vt_symbol, bar.close_price, pos, "Stop Loss")
            return
            
        # 4. Time Stop (Checked at daily level usually, but can check here if we track timestamps)
        entry_time = self.entry_times.get(vt_symbol)
        if entry_time:
            days_held = (bar.datetime.replace(tzinfo=None) - entry_time.replace(tzinfo=None)).days
            if days_held >= self.max_holding_days:
                 self.sell(vt_symbol, bar.close_price, pos, "Time Stop")
            return
        
    def update_daily_signal(self, trade_date: datetime) -> None:
        """
        Called externally or by timer to update target portfolio based on new signals.
        """
        self.write_log(f"Updating Daily Signal for {trade_date}")
        # Load signal data ... (implementation detail omitted for brevity, assumes data feed)
        
        # 1. Check VIX
        self._update_market_regime()
        
        # 2. Rebalance
        # Logic to read parquet/DB and set target_portfolio
        pass

    def _update_market_regime(self):
        # Placeholder for VIX check
        pass

