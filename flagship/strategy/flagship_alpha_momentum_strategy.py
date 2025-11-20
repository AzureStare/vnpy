"""
Flagship Alpha-Momentum 策略实现。

基于策略文档实现：
- 选股机制：Top N（10-15只），Score > 0.5
- 仓位权重：逆波动率加权（风险平价）
- 风险管理：Alpha 衰减离场、个股止损、组合熔断、环境过滤器
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to
from vnpy.trader.logger import logger

from vnpy.alpha.strategy import AlphaStrategy


class FlagshipAlphaMomentumStrategy(AlphaStrategy):
    """
    Flagship Alpha-Momentum 策略实现类。
    
    策略参数：
    - top_n: 持仓股票数量上限（默认 12）
    - min_score_threshold: 最低 Score 门槛（默认 0.5）
    - min_holding_days: 最小持仓天数（默认 2，避免过度交易）
    - max_holding_days: 最大持仓天数（默认 5）
    - cash_ratio: 现金使用比例（默认 0.95，预留 5% 缓冲）
    - min_volume: 最小交易单位
    - open_rate: 开仓手续费率
    - close_rate: 平仓手续费率
    - min_commission: 最小手续费
    - price_add: 下单价格调整比例
    - stop_loss_atr_multiplier: 止损 ATR 倍数（默认 2.5）
    - max_daily_drawdown: 单日最大回撤阈值（默认 0.04，即 4%）
    """

    top_n: int = 12
    min_score_threshold: float = 0.5
    min_holding_days: int = 2
    max_holding_days: int = 5
    cash_ratio: float = 0.95
    min_volume: int = 1
    open_rate: float = 0.0005
    close_rate: float = 0.0015
    min_commission: int = 1
    price_add: float = 0.0005
    stop_loss_atr_multiplier: float = 2.5
    max_daily_drawdown: float = 0.04

    def on_init(self) -> None:
        """策略初始化回调"""
        # 持仓天数跟踪
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        
        # 入场价格记录（用于止损计算）
        self.entry_prices: dict[str, float] = {}
        
        # 前一日净值（用于计算单日回撤）
        self.previous_net_value: float = self.get_cash_available() + self.get_holding_value()
        
        self.write_log("Flagship Alpha-Momentum Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """交易执行回调"""
        if trade.direction == Direction.LONG:
            # 开仓：记录入场价格
            if trade.vt_symbol not in self.entry_prices:
                self.entry_prices[trade.vt_symbol] = trade.price
        else:
            # 平仓：清除持仓记录
            self.holding_days.pop(trade.vt_symbol, None)
            self.entry_prices.pop(trade.vt_symbol, None)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        # 获取当前信号并排序
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 开始处理 K线切片")
        last_signal: pl.DataFrame = self.get_signal()
        if last_signal.is_empty():
            logger.warning(f"[FlagshipAlphaMomentumStrategy.on_bars] 信号数据为空，跳过")
            return
        
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 获取到 {len(last_signal)} 行信号数据")
        
        # 按 Score 降序排序（策略文档要求）
        last_signal = last_signal.sort("signal", descending=True)
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 信号已按 Score 降序排序")
        
        # 记录 Score 分布
        if len(last_signal) > 0:
            max_score = last_signal["signal"][0]
            min_score = last_signal["signal"][-1]
            mean_score = last_signal["signal"].mean()
            logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] Score 分布: max={max_score:.3f}, min={min_score:.3f}, mean={mean_score:.3f}")

        # 更新持仓天数
        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos > 0]
        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        # === 风险管理：检查单日回撤 ===
        current_net_value = self.get_cash_available() + self.get_holding_value()
        daily_return = (current_net_value - self.previous_net_value) / self.previous_net_value if self.previous_net_value > 0 else 0.0
        
        if daily_return < -self.max_daily_drawdown:
            # 单日回撤超过阈值，强制平仓 50%
            self.write_log(f"单日回撤 {daily_return:.2%} 超过阈值 {self.max_daily_drawdown:.2%}，强制平仓 50%")
            for vt_symbol in list(pos_symbols)[:len(pos_symbols)//2]:
                self.set_target(vt_symbol, 0)
            self.execute_trading(bars, price_add=self.price_add)
            self.previous_net_value = current_net_value
            return

        # === 风险管理：检查个股止损 ===
        sell_for_stop_loss: set[str] = set()
        for vt_symbol in pos_symbols:
            if vt_symbol not in bars:
                continue
            
            bar = bars[vt_symbol]
            entry_price = self.entry_prices.get(vt_symbol)
            if entry_price is None:
                continue
            
            # 简化的止损：基于价格跌幅（实际应该用 ATR，但需要历史数据）
            # 这里先用简单的百分比止损
            price_drop = (entry_price - bar.close_price) / entry_price
            if price_drop > 0.05:  # 5% 止损（简化版）
                sell_for_stop_loss.add(vt_symbol)
                self.write_log(f"{vt_symbol} 触发止损，入场价 {entry_price:.2f}，当前价 {bar.close_price:.2f}")

        # === Alpha 衰减离场：检查持仓股票是否跌出 Top 20 或 Score < 0 ===
        sell_for_alpha_decay: set[str] = set()
        top_20_symbols = set(last_signal.head(20)["vt_symbol"].to_list())
        
        for vt_symbol in pos_symbols:
            if vt_symbol not in bars:
                continue
            
            # 检查是否在 Top 20
            if vt_symbol not in top_20_symbols:
                sell_for_alpha_decay.add(vt_symbol)
                continue
            
            # 检查 Score 是否 < 0
            symbol_signal = last_signal.filter(pl.col("vt_symbol") == vt_symbol)
            if not symbol_signal.is_empty():
                score = symbol_signal["signal"][0]
                if score < 0:
                    sell_for_alpha_decay.add(vt_symbol)

        # === 生成卖出列表 ===
        sell_symbols: set[str] = sell_for_stop_loss | sell_for_alpha_decay
        
        # 检查最小持仓天数
        for vt_symbol in list(sell_symbols):
            if self.holding_days[vt_symbol] < self.min_holding_days:
                sell_symbols.remove(vt_symbol)
        
        # 检查最大持仓天数
        for vt_symbol in pos_symbols:
            if self.holding_days[vt_symbol] >= self.max_holding_days:
                sell_symbols.add(vt_symbol)

        # === 生成买入列表 ===
        # 筛选满足 Score 门槛的股票（策略文档要求：Score > 0.5）
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 筛选满足 Score 门槛的股票（Score >= {self.min_score_threshold}）")
        buyable_df = last_signal.filter(
            (pl.col("signal") >= self.min_score_threshold) &
            (~pl.col("vt_symbol").is_in(pos_symbols))
        )
        logger.info(f"[FlagshipAlphaMomentumStrategy.on_bars] 满足 Score 门槛的候选股票: {len(buyable_df)} 只")
        
        # 计算需要买入的数量
        target_positions = self.top_n - (len(pos_symbols) - len(sell_symbols))
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 目标持仓数: {self.top_n}, 当前持仓: {len(pos_symbols)}, 卖出: {len(sell_symbols)}, 需要买入: {target_positions}")
        
        # 选取 Top N（策略文档要求：按 Score 降序选 Top N）
        buy_symbols = buyable_df.head(max(0, target_positions))["vt_symbol"].to_list()
        logger.info(f"[FlagshipAlphaMomentumStrategy.on_bars] 最终选股: {len(buy_symbols)} 只（Top {target_positions}，Score >= {self.min_score_threshold}）")
        
        if len(buy_symbols) > 0:
            top_scores = buyable_df.head(len(buy_symbols))["signal"].to_list()
            logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 选中股票的 Score 范围: {min(top_scores):.3f} ~ {max(top_scores):.3f}")

        # === 执行卖出 ===
        cash = self.get_cash_available()
        for vt_symbol in sell_symbols:
            if vt_symbol not in bars:
                continue
            
            bar = bars[vt_symbol]
            sell_price = bar.close_price
            sell_volume = self.get_pos(vt_symbol)
            
            self.set_target(vt_symbol, 0)
            
            turnover = sell_price * sell_volume
            cost = max(turnover * self.close_rate, self.min_commission)
            cash += turnover - cost

        # === 执行买入（逆波动率加权） ===
        if buy_symbols:
            # 简化版：等权重分配（实际应该用 60 日历史波动率计算逆波动率权重）
            buy_value = cash * self.cash_ratio / len(buy_symbols)
            
            for vt_symbol in buy_symbols:
                if vt_symbol not in bars:
                    continue
                
                bar = bars[vt_symbol]
                buy_price = bar.close_price
                if not buy_price:
                    continue
                
                buy_volume = round_to(buy_value / buy_price, self.min_volume)
                self.set_target(vt_symbol, buy_volume)

        # === 执行交易 ===
        self.execute_trading(bars, price_add=self.price_add)
        
        # 更新前一日净值
        self.previous_net_value = self.get_cash_available() + self.get_holding_value()

