"""
Flagship Alpha-Momentum 策略实现。

基于策略文档 v7.0 实现：
- 选股机制：Top 50 候选，配合 Setup A/B 结构过滤
- 持仓集中：5-8 只
- 止盈机制：Profit Ladder (阶梯止盈)
- 杠杆模式：大盘走强时允许 130%-160% 杠杆
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import math
from datetime import date, datetime

import numpy as np
import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction, Interval
from vnpy.trader.utility import round_to, ArrayManager
from vnpy.trader.logger import logger

from vnpy.alpha.strategy import AlphaStrategy


class FlagshipAlphaMomentumStrategy(AlphaStrategy):
    """
    Flagship Alpha-Momentum V7.0 Aggressive 策略实现类。
    
    策略特性：
    - 选股机制：Top 50 候选，配合 Setup A/B 结构过滤
    - 持仓集中：5-8 只
    - 止盈机制：Profit Ladder (阶梯止盈)
    - 杠杆模式：大盘走强时允许 130%-160% 杠杆
    """

    top_n: int = 8
    min_score_threshold: float = 0.0  # V7 以后主要靠排名和结构，Score 只作为初选
    min_quantile_threshold: float = 90.0
    min_holding_days: int = 1
    max_holding_days: int = 1000  # V7 不设固定持有期
    cash_ratio: float = 0.95
    min_volume: int = 1
    open_rate: float = 0.0005
    close_rate: float = 0.0015
    min_commission: int = 1
    price_add: float = 0.0005
    
    # 风险管理参数
    hard_stop_loss_pct: float = 0.07  # 硬止损 7%
    stop_loss_atr_multiplier: float = 2.5  # ATR 止损倍数
    max_daily_drawdown: float = 0.05
    
    # 仓位控制
    base_pos_size: float = 0.15  # 基础仓位 15%
    max_leverage: float = 1.6
    
    # VIX 阈值
    vix_threshold_1: float = 1.0
    vix_threshold_2: float = 1.1

    def on_init(self) -> None:
        """策略初始化回调"""
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        self.entry_prices: dict[str, float] = {}
        self.entry_highs: dict[str, float] = {}
        self.previous_net_value: float = 0.0
        
        # 记录当日最低价（用于 Spike Check）
        self.daily_lows: dict[str, float] = {}
        
        self.bar_managers: dict[str, ArrayManager] = {}
        for vt_symbol in self.vt_symbols:
            self.bar_managers[vt_symbol] = ArrayManager(size=100)
        
        self.current_vix: float | None = None
        self.current_vix3m: float | None = None
        
        self.trade_entry_reasons: dict[str, str] = {}
        self.trade_exit_reasons: dict[str, str] = {}
        self.trade_entry_reasons_by_tradeid: dict[str, str] = {}
        self.trade_exit_reasons_by_tradeid: dict[str, str] = {}
        
        self.daily_indicators: dict[str, dict[str, float]] = {}  # 缓存信号文件中的指标 (ema, atr etc)
        
        self.current_trade_date: date | None = None
        self.daily_rebalance_done: bool = False
        
        self.write_log("Flagship Alpha-Momentum V7.0 Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """交易执行回调"""
        if trade.direction == Direction.LONG:
            if trade.vt_symbol not in self.entry_prices:
                self.entry_prices[trade.vt_symbol] = trade.price
                self.entry_highs[trade.vt_symbol] = trade.price
                if trade.vt_symbol in self.trade_entry_reasons:
                    self.trade_entry_reasons_by_tradeid[trade.vt_tradeid] = self.trade_entry_reasons[trade.vt_symbol]
        else:
            if trade.vt_symbol in self.trade_exit_reasons:
                self.trade_exit_reasons_by_tradeid[trade.vt_tradeid] = self.trade_exit_reasons[trade.vt_symbol]
            self.holding_days.pop(trade.vt_symbol, None)
            self.entry_prices.pop(trade.vt_symbol, None)
            self.entry_highs.pop(trade.vt_symbol, None)
            self.daily_lows.pop(trade.vt_symbol, None)
            self.trade_entry_reasons.pop(trade.vt_symbol, None)
            self.trade_exit_reasons.pop(trade.vt_symbol, None)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        if not bars:
            return
            
        first_bar = next(iter(bars.values()))
        current_datetime = first_bar.datetime.replace(tzinfo=None)
        current_date = current_datetime.date()
        
        is_new_trading_day = False
        if self.current_trade_date != current_date:
            is_new_trading_day = True
            self.current_trade_date = current_date
            self.daily_rebalance_done = False
            for vt_symbol in [s for s, p in self.pos_data.items() if p > 0]:
                self.holding_days[vt_symbol] += 1
            if self.previous_net_value == 0:
                self.previous_net_value = self.get_cash_available() + self.get_holding_value()

        # 1. 更新基础数据
        self._update_bar_managers(bars)
        
        # 更新入场后最高价和当日最低价
        for vt_symbol, bar in bars.items():
            if self.get_pos(vt_symbol) > 0:
                if vt_symbol in self.entry_highs:
                    self.entry_highs[vt_symbol] = max(self.entry_highs[vt_symbol], bar.high_price)
                if vt_symbol not in self.daily_lows or is_new_trading_day:
                    self.daily_lows[vt_symbol] = bar.low_price
                else:
                    self.daily_lows[vt_symbol] = min(self.daily_lows[vt_symbol], bar.low_price)

        # 2. 止盈止损检查 (每分钟)
        self._check_profit_ladder_exit(bars)
        
        # 3. 每日调仓 (开盘第一个K线)
        if is_new_trading_day and not self.daily_rebalance_done:
            self._run_daily_rebalance(bars)
            self.daily_rebalance_done = True
            # 更新前一日净值，用于次日回撤计算
            self.previous_net_value = self.get_cash_available() + self.get_holding_value()

    def _check_profit_ladder_exit(self, bars: dict[str, BarData]) -> None:
        """阶梯止盈检查逻辑"""
        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            if pos <= 0:
                continue
                
            entry_price = self.entry_prices.get(vt_symbol)
            entry_high = self.entry_highs.get(vt_symbol)
            if not entry_price or not entry_high:
                continue
                
            # 获取日级指标 (EMA, ATR)
            indicators = self.daily_indicators.get(vt_symbol, {})
            atr = indicators.get("atr_14")
            ema10 = indicators.get("ema10")
            ema5 = indicators.get("ema5")
            
            # 计算当前浮盈
            profit_pct = (bar.close_price / entry_price) - 1
            days = self.holding_days.get(vt_symbol, 0)
            
            exit_reason = None
            
            # A. 硬止损 (通用)
            hard_stop = entry_price * (1 - self.hard_stop_loss_pct)
            if atr:
                hard_stop = min(hard_stop, entry_price - 2.5 * atr)
            
            if bar.close_price < hard_stop:
                exit_reason = f"硬止损触发 (Price < {hard_stop:.2f})"
            
            # B. 阶梯止盈 (Profit Ladder)
            if not exit_reason:
                if profit_pct < 0.05 and days < 3:
                    # 早期阶段：仅硬止损，忽略 EMA
                    pass
                elif 0.05 <= profit_pct < 0.15:
                    # 趋势阶段：收盘跌破 EMA10
                    if ema10 and bar.close_price < ema10:
                        exit_reason = "EMA10 趋势离场"
                elif profit_pct >= 0.15:
                    # 获利丰厚阶段：收盘跌破 EMA5 或 从最高点回撤 10%
                    trailing_stop = entry_high * 0.90
                    if ema5:
                        stop_line = max(ema5, trailing_stop)
                    else:
                        stop_line = trailing_stop
                        
                    if bar.close_price < stop_line:
                        exit_reason = f"利润锁定离场 (Price < {stop_line:.2f})"
            
            if exit_reason:
                self.write_log(f"{vt_symbol} 离场: {exit_reason}, 收益: {profit_pct:.2%}")
                self.trade_exit_reasons[vt_symbol] = exit_reason
                self.set_target(vt_symbol, 0)
                self.execute_trading(bars, price_add=self.price_add)

    def _run_daily_rebalance(self, bars: dict[str, BarData]) -> None:
        """每日选股与调仓"""
        last_signal = self.get_signal()
        if last_signal.is_empty():
            return
            
        # 缓存指标
        for row in last_signal.iter_rows(named=True):
            self.daily_indicators[row["vt_symbol"]] = row
            
        # 1. 结构过滤器 (Setup A/B)
        candidates = []
        top_50 = last_signal.sort("signal", descending=True).head(50)
        
        for row in top_50.iter_rows(named=True):
            symbol = row["vt_symbol"]
            if symbol not in bars: continue
            if self.get_pos(symbol) > 0: continue
            
            p = row["close_price"]
            ema10 = row.get("ema10")
            ema20 = row.get("ema20")
            ema50 = row.get("ema50")
            
            setup_a = False
            # Setup A: Breakout (P > EMA20)
            if ema20 and p > ema20:
                setup_a = True
                
            setup_b = False
            # Setup B: Pullback (P > EMA50, P < EMA10, P > EMA20)
            if ema50 and ema10 and ema20 and p > ema50 and p < ema10 and p > ema20:
                setup_b = True
                
            if setup_a or setup_b:
                candidates.append(row)
                
        # 2. 最终选股 (Top N)
        final_buy = sorted(candidates, key=lambda x: x["signal"], reverse=True)[:self.top_n]
        
        # 3. 杠杆计算 (SPY > MA50)
        spy_row = last_signal.filter(pl.col("vt_symbol") == "SPY.NASDAQ")
        leverage = 1.0
        if not spy_row.is_empty():
            spy_close = spy_row["close_price"][0]
            # 注意：ma50 列如果不在信号文件中，这里会报错，所以我们在 run_live_inference 中输出了 ema50 暂用
            spy_trend_ma = spy_row.get("ema50", [0])[0]
            if spy_close > spy_trend_ma:
                leverage = self.max_leverage 
                
        # 4. 执行买入
        if final_buy:
            total_equity = self.get_cash_available() + self.get_holding_value()
            target_value_per_stock = (total_equity * leverage) / self.top_n
            
            for row in final_buy:
                symbol = row["vt_symbol"]
                p = row["close_price"]
                volume = round_to(target_value_per_stock / p, self.min_volume)
                if volume > 0:
                    self.set_target(symbol, volume)
                    self.trade_entry_reasons[symbol] = f"V7 Setup, Score={row['signal']:.3f}"
                    
        self.execute_trading(bars, price_add=self.price_add)

    def _update_bar_managers(self, bars: dict[str, BarData]) -> None:
        """更新历史K线数据管理器"""
        for vt_symbol, bar in bars.items():
            if vt_symbol in self.bar_managers:
                self.bar_managers[vt_symbol].update_bar(bar)

    def _get_vix_ratio(self, bars: dict[str, BarData]) -> float | None:
        """获取VIX期限结构比率"""
        vix_bar = bars.get("VIX.CBOE")
        vix3m_bar = bars.get("VIX3M.CBOE")
        if vix_bar and vix3m_bar and vix_bar.close_price > 0 and vix3m_bar.close_price > 0:
            return vix_bar.close_price / vix3m_bar.close_price
        return None
