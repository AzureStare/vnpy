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
import math

import numpy as np
import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to, ArrayManager
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
    stop_loss_atr_multiplier: float = 3.0  # Chandelier Exit: 3*ATR
    max_daily_drawdown: float = 0.04
    max_single_position_weight: float = 0.03  # 单一个股权重上限 3%
    vix_threshold_1: float = 1.0  # VIX/VIX3M 阈值1
    vix_threshold_2: float = 1.1  # VIX/VIX3M 阈值2
    volatility_window: int = 60  # 历史波动率计算窗口（60日）

    def on_init(self) -> None:
        """策略初始化回调"""
        # 持仓天数跟踪
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        
        # 入场价格记录（用于止损计算）
        self.entry_prices: dict[str, float] = {}
        
        # 入场后最高价记录（用于 Chandelier Exit）
        self.entry_highs: dict[str, float] = {}
        
        # 前一日净值（用于计算单日回撤）
        self.previous_net_value: float = self.get_cash_available() + self.get_holding_value()
        
        # 历史K线数据管理器（用于计算ATR和波动率）
        self.bar_managers: dict[str, ArrayManager] = {}
        for vt_symbol in self.vt_symbols:
            self.bar_managers[vt_symbol] = ArrayManager(size=100)  # 保留100根K线
        
        # VIX数据缓存
        self.current_vix: float | None = None
        self.current_vix3m: float | None = None
        
        # 交易信号记录（用于交易清单显示）
        self.trade_signals: dict[str, str] = {}  # vt_symbol -> 信号说明
        self.trade_entry_reasons: dict[str, str] = {}  # vt_symbol -> 开仓原因
        self.trade_exit_reasons: dict[str, str] = {}  # vt_symbol -> 平仓原因
        
        self.write_log("Flagship Alpha-Momentum Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """交易执行回调"""
        if trade.direction == Direction.LONG:
            # 开仓：记录入场价格和最高价
            if trade.vt_symbol not in self.entry_prices:
                self.entry_prices[trade.vt_symbol] = trade.price
                self.entry_highs[trade.vt_symbol] = trade.price
        else:
            # 平仓：清除持仓记录和信号
            self.holding_days.pop(trade.vt_symbol, None)
            self.entry_prices.pop(trade.vt_symbol, None)
            self.entry_highs.pop(trade.vt_symbol, None)
            self.trade_entry_reasons.pop(trade.vt_symbol, None)
            self.trade_exit_reasons.pop(trade.vt_symbol, None)

    def _update_bar_managers(self, bars: dict[str, BarData]) -> None:
        """更新历史K线数据管理器"""
        for vt_symbol, bar in bars.items():
            if vt_symbol in self.bar_managers:
                self.bar_managers[vt_symbol].update_bar(bar)
    
    def _calculate_historical_volatility(self, vt_symbol: str, window: int = 60) -> float | None:
        """
        计算历史波动率（60日收益率标准差年化）。
        
        Returns:
            历史波动率，如果数据不足返回 None
        """
        if vt_symbol not in self.bar_managers:
            return None
        
        am = self.bar_managers[vt_symbol]
        if not am.inited or am.count < window:
            return None
        
        # 计算收益率
        closes = am.close_array[-window:]
        returns = np.diff(closes) / closes[:-1]
        
        # 计算年化波动率（假设252个交易日）
        volatility = np.std(returns) * math.sqrt(252)
        return float(volatility)
    
    def _get_atr(self, vt_symbol: str, window: int = 14) -> float | None:
        """获取ATR值"""
        if vt_symbol not in self.bar_managers:
            return None
        
        am = self.bar_managers[vt_symbol]
        if not am.inited:
            return None
        
        return am.atr(window)
    
    def _get_vix_ratio(self, bars: dict[str, BarData]) -> float | None:
        """
        获取VIX期限结构比率（VIX/VIX3M）。
        
        Returns:
            VIX比率，如果数据不可用返回 None
        """
        vix_bar = bars.get("VIX.CBOE")
        vix3m_bar = bars.get("VIX3M.CBOE")
        
        if vix_bar and vix3m_bar and vix_bar.close_price > 0 and vix3m_bar.close_price > 0:
            ratio = vix_bar.close_price / vix3m_bar.close_price
            self.current_vix = vix_bar.close_price
            self.current_vix3m = vix3m_bar.close_price
            return float(ratio)
        
        return None
    
    def _get_target_leverage(self, vix_ratio: float | None) -> float:
        """
        根据VIX期限结构计算目标杠杆。
        
        策略文档公式：
        - Ratio <= 1.0: Leverage = 1.0
        - 1.0 < Ratio <= 1.1: Leverage = 0.5
        - Ratio > 1.1: Leverage = 0.3
        """
        if vix_ratio is None:
            return 1.0  # 默认杠杆
        
        if vix_ratio <= self.vix_threshold_1:
            return 1.0
        elif vix_ratio <= self.vix_threshold_2:
            return 0.5
        else:
            return 0.3

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        # 更新历史K线数据
        self._update_bar_managers(bars)
        
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
        
        # 风险管理：VIX期限结构过滤器
        vix_ratio = self._get_vix_ratio(bars)
        target_leverage = self._get_target_leverage(vix_ratio)
        if vix_ratio is not None:
            logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] VIX比率: {vix_ratio:.3f}, 目标杠杆: {target_leverage:.2f}")
        
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
            
            # 更新入场后最高价（用于 Chandelier Exit）
            if vt_symbol in bars and vt_symbol in self.entry_highs:
                current_high = bars[vt_symbol].high_price
                if current_high > self.entry_highs[vt_symbol]:
                    self.entry_highs[vt_symbol] = current_high

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

        # === 风险管理：Chandelier Exit 止损（策略文档要求：max(H) - 3*ATR）===
        sell_for_stop_loss: set[str] = set()
        for vt_symbol in pos_symbols:
            if vt_symbol not in bars:
                continue
            
            bar = bars[vt_symbol]
            entry_high = self.entry_highs.get(vt_symbol)
            if entry_high is None:
                continue
            
            # 获取ATR
            atr = self._get_atr(vt_symbol, window=14)
            if atr is None or atr <= 0:
                # 如果ATR不可用，使用简化的百分比止损作为后备
                entry_price = self.entry_prices.get(vt_symbol)
                if entry_price:
                    price_drop = (entry_price - bar.close_price) / entry_price
                    if price_drop > 0.05:  # 5% 止损（后备方案）
                        sell_for_stop_loss.add(vt_symbol)
                        self.write_log(f"{vt_symbol} 触发止损（后备），入场价 {entry_price:.2f}，当前价 {bar.close_price:.2f}")
                continue
            
            # Chandelier Exit: P_exit = max(H) - 3*ATR
            exit_price = entry_high - self.stop_loss_atr_multiplier * atr
            if bar.close_price <= exit_price:
                sell_for_stop_loss.add(vt_symbol)
                self.write_log(
                    f"{vt_symbol} 触发 Chandelier Exit 止损："
                    f"最高价={entry_high:.2f}, ATR={atr:.2f}, "
                    f"止损价={exit_price:.2f}, 当前价={bar.close_price:.2f}"
                )

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

        # === 记录卖出信号原因 ===
        for vt_symbol in sell_for_stop_loss:
            self.trade_exit_reasons[vt_symbol] = "Chandelier Exit止损"
        for vt_symbol in sell_for_alpha_decay:
            self.trade_exit_reasons[vt_symbol] = "Alpha衰减离场"
        for vt_symbol in pos_symbols:
            if self.holding_days[vt_symbol] >= self.max_holding_days:
                self.trade_exit_reasons[vt_symbol] = "最大持仓天数"
        
        # === 记录卖出信号原因 ===
        for vt_symbol in sell_for_stop_loss:
            self.trade_exit_reasons[vt_symbol] = "Chandelier Exit止损"
        for vt_symbol in sell_for_alpha_decay:
            self.trade_exit_reasons[vt_symbol] = "Alpha衰减离场"
        
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
                self.trade_exit_reasons[vt_symbol] = "最大持仓天数"

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

        # === 记录买入信号原因 ===
        for vt_symbol in buy_symbols:
            if vt_symbol in bars:
                symbol_signal = last_signal.filter(pl.col("vt_symbol") == vt_symbol)
                if not symbol_signal.is_empty():
                    score = symbol_signal["signal"][0]
                    # 计算排名（Score降序）
                    ranked_signal = last_signal.with_row_index("rank").sort("signal", descending=True)
                    rank_row = ranked_signal.filter(pl.col("vt_symbol") == vt_symbol)
                    if not rank_row.is_empty():
                        rank = rank_row["rank"][0] + 1  # 从1开始
                        self.trade_entry_reasons[vt_symbol] = f"Score排名#{rank}, Score={score:.3f}"
        
        # === 执行买入（逆波动率加权，策略文档要求）===
        if buy_symbols:
            # 计算每个股票的历史波动率
            volatilities: dict[str, float] = {}
            for vt_symbol in buy_symbols:
                vol = self._calculate_historical_volatility(vt_symbol, window=self.volatility_window)
                if vol is not None and vol > 0:
                    volatilities[vt_symbol] = vol
                else:
                    # 如果无法计算波动率，使用默认值（避免除零）
                    volatilities[vt_symbol] = 0.3  # 默认30%波动率
            
            # 计算逆波动率权重：W_i = (1/σ_i) / Σ(1/σ_j)
            inv_vols: dict[str, float] = {}
            total_inv_vol = 0.0
            for vt_symbol, vol in volatilities.items():
                inv_vol = 1.0 / vol if vol > 0 else 0.0
                inv_vols[vt_symbol] = inv_vol
                total_inv_vol += inv_vol
            
            # 应用VIX杠杆调整
            total_buy_value = cash * self.cash_ratio * target_leverage
            
            # 第一步：计算原始权重
            raw_weights: dict[str, float] = {}
            for vt_symbol in buy_symbols:
                if total_inv_vol > 0:
                    raw_weights[vt_symbol] = inv_vols[vt_symbol] / total_inv_vol
                else:
                    # 如果所有波动率都无效，使用等权重
                    raw_weights[vt_symbol] = 1.0 / len(buy_symbols)
            
            # 第二步：应用单一个股权重上限（策略文档要求：3%）
            capped_weights: dict[str, float] = {}
            for vt_symbol, raw_weight in raw_weights.items():
                capped_weights[vt_symbol] = min(raw_weight, self.max_single_position_weight)
            
            # 第三步：重新归一化权重（确保总权重为1）
            total_capped_weight = sum(capped_weights.values())
            if total_capped_weight > 0:
                normalized_weights: dict[str, float] = {}
                for vt_symbol, capped_weight in capped_weights.items():
                    normalized_weights[vt_symbol] = capped_weight / total_capped_weight
            else:
                # 如果所有权重都被上限截断，使用等权重
                normalized_weights = {vt_symbol: 1.0 / len(buy_symbols) for vt_symbol in buy_symbols}
            
            # 计算每个股票的买入金额和数量
            for vt_symbol in buy_symbols:
                if vt_symbol not in bars:
                    continue
                
                bar = bars[vt_symbol]
                buy_price = bar.close_price
                if not buy_price or buy_price <= 0:
                    continue
                
                # 获取归一化后的权重
                weight = normalized_weights.get(vt_symbol, 0.0)
                
                # 计算买入金额
                buy_value = total_buy_value * weight
                buy_volume = round_to(buy_value / buy_price, self.min_volume)
                
                if buy_volume > 0:
                    self.set_target(vt_symbol, buy_volume)
                    logger.debug(
                        f"{vt_symbol} 买入：波动率={volatilities.get(vt_symbol, 0):.3f}, "
                        f"原始权重={raw_weights.get(vt_symbol, 0):.4f}, "
                        f"最终权重={weight:.4f}, 金额={buy_value:.2f}, 数量={buy_volume}"
                    )

        # === 执行交易 ===
        self.execute_trading(bars, price_add=self.price_add)
        
        # 更新前一日净值
        self.previous_net_value = self.get_cash_available() + self.get_holding_value()

