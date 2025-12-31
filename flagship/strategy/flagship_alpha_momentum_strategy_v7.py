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
    
    # 阶梯止盈阈值
    profit_threshold_trend: float = 0.05  # 5% 进入趋势阶段
    profit_threshold_win: float = 0.15    # 15% 进入获利丰厚阶段
    spike_threshold: float = 0.10         # 10% 暴涨日阈值
    
    # 仓位控制
    base_pos_size: float = 0.15  # 基础仓位 15%
    max_leverage: float = 1.6
    
    def on_init(self) -> None:
        """策略初始化回调"""
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        self.entry_prices: dict[str, float] = {}
        self.entry_highs: dict[str, float] = {}
        self.previous_net_value: float = 0.0
        
        # 记录当日最低价（用于 Spike Check）
        self.daily_lows: dict[str, float] = {}
        # 记录当日是否触发过暴涨（10%+）
        self.spike_triggered: dict[str, bool] = defaultdict(bool)
        
        self.bar_managers: dict[str, ArrayManager] = {}
        for vt_symbol in self.vt_symbols:
            self.bar_managers[vt_symbol] = ArrayManager(size=100)
        
        # 大盘指标
        self.spy_ema10: float | None = None
        self.spy_ema20: float | None = None
        
        self.trade_entry_reasons: dict[str, str] = {}
        self.trade_exit_reasons: dict[str, str] = {}
        self.trade_entry_reasons_by_tradeid: dict[str, str] = {}
        self.trade_exit_reasons_by_tradeid: dict[str, str] = {}
        
        self.daily_indicators: dict[str, dict[str, float]] = {}  # 缓存信号文件中的指标 (ema, atr etc)
        
        self.current_trade_date: date | None = None
        self.daily_rebalance_done: bool = False
        
        self.write_log("Flagship Alpha-Momentum V7.0 Strategy initialized (Intraday Exit Mode)")

    def on_trade(self, trade: TradeData) -> None:
        """交易执行回调"""
        if trade.direction == Direction.LONG:
            if trade.vt_symbol not in self.entry_prices:
                self.entry_prices[trade.vt_symbol] = trade.price
                self.entry_highs[trade.vt_symbol] = trade.price
                self.spike_triggered[trade.vt_symbol] = False # 重置暴涨状态
                if trade.vt_symbol in self.trade_entry_reasons:
                    self.trade_entry_reasons_by_tradeid[trade.vt_tradeid] = self.trade_entry_reasons[trade.vt_symbol]
        else:
            if trade.vt_symbol in self.trade_exit_reasons:
                self.trade_exit_reasons_by_tradeid[trade.vt_tradeid] = self.trade_exit_reasons[trade.vt_symbol]
            self.holding_days.pop(trade.vt_symbol, None)
            self.entry_prices.pop(trade.vt_symbol, None)
            self.entry_highs.pop(trade.vt_symbol, None)
            self.daily_lows.pop(trade.vt_symbol, None)
            self.spike_triggered.pop(trade.vt_symbol, None)
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
                self.spike_triggered[vt_symbol] = False # 新的一天重置暴涨记录
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

    def _update_bar_managers(self, bars: dict[str, BarData]) -> None:
        """更新历史K线数据管理器"""
        for vt_symbol, bar in bars.items():
            if vt_symbol in self.bar_managers:
                am = self.bar_managers[vt_symbol]
                am.update_bar(bar)
                
                # 特殊处理 SPY 指标
                if vt_symbol == "SPY.NASDAQ" and am.inited:
                    self.spy_ema10 = am.ema(10)
                    self.spy_ema20 = am.ema(20)

    def _check_profit_ladder_exit(self, bars: dict[str, BarData]) -> None:
        """阶梯止盈检查逻辑 (分钟级)"""
        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            if pos <= 0:
                continue
                
            entry_price = self.entry_prices.get(vt_symbol)
            entry_high = self.entry_highs.get(vt_symbol)
            if not entry_price or not entry_high:
                continue
                
            # 获取日级指标 (来自信号文件)
            indicators = self.daily_indicators.get(vt_symbol, {})
            atr = indicators.get("atr_14")
            
            # 使用 ArrayManager 计算实时 EMA (更灵敏)
            am = self.bar_managers.get(vt_symbol)
            ema5 = am.ema(5) if am and am.inited else indicators.get("ema5")
            ema10 = am.ema(10) if am and am.inited else indicators.get("ema10")
            
            # 计算当前浮盈 (基于分钟价)
            profit_pct = (bar.close_price / entry_price) - 1
            days = self.holding_days.get(vt_symbol, 0)
            
            exit_reason = None
            
            # 0. 暴涨日实时保护 (Intraday Spike Guard)
            # 如果盘中涨幅超过 10%，记录触发
            day_return = (bar.close_price / bar.open_price - 1) if bar.open_price > 0 else 0
            if day_return > self.spike_threshold:
                self.spike_triggered[vt_symbol] = True
            
            if self.spike_triggered[vt_symbol]:
                # 触发暴涨后，若价格回落跌破当日最低价（收盘价附近的快速回落），立即锁定
                if bar.close_price <= self.daily_lows.get(vt_symbol, 0):
                    exit_reason = "暴涨日回落止盈 (Price <= Day Low)"

            # 1. 宏观极速刹车 (SPY 盘中跌破 EMA10)
            quick_brake = False
            spy_bar = bars.get("SPY.NASDAQ")
            if spy_bar and self.spy_ema10 and spy_bar.close_price < self.spy_ema10:
                quick_brake = True

            # 2. 阶梯止盈逻辑 (Profit Ladder)
            if not exit_reason:
                if profit_pct < self.profit_threshold_trend and days < 3:
                    # A. 早期阶段：仅硬止损
                    hard_stop = entry_price * (1 - self.hard_stop_loss_pct)
                    if atr:
                        hard_stop = min(hard_stop, entry_price - self.stop_loss_atr_multiplier * atr)
                    if bar.close_price < hard_stop:
                        exit_reason = f"早期硬止损 (Price < {hard_stop:.2f})"
                
                elif (self.profit_threshold_trend <= profit_pct < self.profit_threshold_win) and not quick_brake:
                    # B. 趋势阶段：跌破 EMA10
                    if ema10 and bar.close_price < ema10:
                        exit_reason = "EMA10 趋势止盈"
                
                else:
                    # C. 获利丰厚阶段 (>15%) 或 大盘刹车模式：使用 EMA5 或 10% 回撤
                    trailing_stop = entry_high * 0.90
                    stop_line = trailing_stop
                    if ema5:
                        stop_line = max(ema5, trailing_stop)
                    
                    if bar.close_price < stop_line:
                        exit_reason = f"利润锁定/大盘避险 (Price < {stop_line:.2f})"
            
            if exit_reason:
                self.write_log(f"{vt_symbol} 盘中离场: {exit_reason}, 当前收益: {profit_pct:.2%}")
                self.trade_exit_reasons[vt_symbol] = exit_reason
                self.set_target(vt_symbol, 0)
                self.execute_trading(bars, price_add=self.price_add)

    def _get_vix_ratio(self, bars: dict[str, BarData]) -> float | None:
        """获取VIX期限结构比率"""
        vix_bar = bars.get("VIX.CBOE")
        vix3m_bar = bars.get("VIX3M.CBOE")
        if vix_bar and vix3m_bar and vix_bar.close_price > 0 and vix3m_bar.close_price > 0:
            return vix_bar.close_price / vix3m_bar.close_price
        return None
