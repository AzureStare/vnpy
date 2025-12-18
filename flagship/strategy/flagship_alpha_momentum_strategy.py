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
    Flagship Alpha-Momentum 策略实现类。
    
    策略参数：
    - top_n: 持仓股票数量上限（默认 5）
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

    top_n: int = 5
    min_score_threshold: float = 0.5
    min_quantile_threshold: float = 90.0  # Percentile (0-100); only buy when in Top percentile
    min_holding_days: int = 2
    max_holding_days: int = 5
    cash_ratio: float = 0.95
    min_volume: int = 1
    open_rate: float = 0.0005
    close_rate: float = 0.0015
    min_commission: int = 1
    price_add: float = 0.0005
    stop_loss_atr_multiplier: float = 3.0  # Chandelier Exit: 3*ATR
    trailing_stop_atr_multiplier: float = 1.5  # 跟踪止盈：1.5*ATR
    take_profit_atr_multiplier: float = 2.0  # 固定止盈：2*ATR
    take_profit_pct: float = 0.05  # 固定止盈百分比：5%
    max_daily_drawdown: float = 0.04
    conviction_lambda: float = 2.0  # 动态信念加权放大系数（策略文档要求：λ=2.0）
    base_position_cap: float = 0.05  # 基础上限 5%（策略文档要求）
    conviction_position_cap: float = 0.08  # 极品上限 8%（策略文档要求：Z_score > 2.0 时触发）
    vix_threshold_1: float = 1.0  # VIX/VIX3M 阈值1
    vix_threshold_2: float = 1.1  # VIX/VIX3M 阈值2
    volatility_window: int = 60  # 历史波动率计算窗口（60日）

    def on_init(self) -> None:
        """策略初始化回调"""
        # 持仓天数跟踪（按交易日计算）
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
        # 以trade_id为key保存entry_reason，避免平仓时被清除
        self.trade_entry_reasons_by_tradeid: dict[str, str] = {}
        # 以trade_id为key保存exit_reason，避免平仓时被清除
        self.trade_exit_reasons_by_tradeid: dict[str, str] = {}
        
        # 日级ATR缓存（从信号文件中读取，用于止盈止损计算）
        self.daily_atr: dict[str, float] = {}  # vt_symbol -> 日级ATR
        
        # 分钟线回测专用：跟踪当前交易日和是否已执行当日选股
        self.current_trade_date: date | None = None
        self.daily_rebalance_done: bool = False
        
        self.write_log("Flagship Alpha-Momentum Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """交易执行回调"""
        if trade.direction == Direction.LONG:
            # 开仓：记录入场价格和最高价
            if trade.vt_symbol not in self.entry_prices:
                self.entry_prices[trade.vt_symbol] = trade.price
                self.entry_highs[trade.vt_symbol] = trade.price
                # 保存entry_reason到trade_id映射，避免平仓时被清除
                if trade.vt_symbol in self.trade_entry_reasons:
                    self.trade_entry_reasons_by_tradeid[trade.vt_tradeid] = self.trade_entry_reasons[trade.vt_symbol]
        else:
            # 平仓：保存exit_reason到trade_id映射，避免被清除
            if trade.vt_symbol in self.trade_exit_reasons:
                self.trade_exit_reasons_by_tradeid[trade.vt_tradeid] = self.trade_exit_reasons[trade.vt_symbol]
            # 清除持仓记录和信号
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
        """
        获取ATR值（绝对价格，美元）
        
        优先使用信号文件中的日级ATR（用于分钟级回测），
        如果不可用则回退到实时计算（仅作为备用）。
        
        注意：如果信号文件中的ATR值小于1，则视为百分比形式，需要乘以当前价格转换为绝对价格。
        """
        # 优先使用信号文件中的日级ATR
        if vt_symbol in self.daily_atr:
            atr_value = self.daily_atr[vt_symbol]
            # 如果ATR值小于1，可能是百分比形式，需要转换为绝对价格
            if atr_value < 1.0:
                # 获取当前价格（用于转换百分比ATR为绝对价格）
                if vt_symbol in self.bar_managers:
                    am = self.bar_managers[vt_symbol]
                    if am.inited and len(am.close_array) > 0:
                        current_price = am.close_array[-1]
                        # 将百分比ATR转换为绝对价格
                        return current_price * atr_value
            # 否则直接返回（已经是绝对价格）
            return atr_value
        
        # 回退到实时计算（仅作为备用）
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
        
        # 获取当前K线时间（用于判断是否是新交易日）
        if not bars:
            return
        
        # 获取第一个K线的时间（所有K线应该在同一时间）
        first_bar = next(iter(bars.values()))
        current_datetime = first_bar.datetime.replace(tzinfo=None)
        current_date = current_datetime.date()
        
        # 判断是否是新交易日（分钟线回测时）
        is_new_trading_day = False
        if self.current_trade_date is None or self.current_trade_date != current_date:
            is_new_trading_day = True
            self.current_trade_date = current_date
            self.daily_rebalance_done = False
            # 更新持仓天数（按交易日计算）
            pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos > 0]
            for vt_symbol in pos_symbols:
                self.holding_days[vt_symbol] += 1
        
        # 获取当前信号并排序
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 开始处理 K线切片")
        last_signal: pl.DataFrame = self.get_signal()
        if last_signal.is_empty():
            logger.warning(f"[FlagshipAlphaMomentumStrategy.on_bars] 信号数据为空，跳过")
            return
        
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 获取到 {len(last_signal)} 行信号数据")
        
        # LightGBM信号文件已经包含预测的Score，直接使用
        # 如果信号文件包含因子值和权重信息（IC-IR方案），则基于当日universe重新计算Score
        # 否则直接使用预计算的signal（LightGBM方案）
        if "z_mom" in last_signal.columns and "z_vwap_residual" in last_signal.columns:
            # IC-IR方案：需要重新计算Score
            if "weight_mom" in last_signal.columns and "weight_vwap" in last_signal.columns:
                weight_mom_col = pl.col("weight_mom").fill_null(0.0)
                weight_vwap_col = pl.col("weight_vwap").fill_null(0.0)
                
                weight_mom_nonzero = last_signal.filter(pl.col("weight_mom") != 0).height
                weight_vwap_nonzero = last_signal.filter(pl.col("weight_vwap") != 0).height
                
                if weight_mom_nonzero == 0 and weight_vwap_nonzero == 0:
                    logger.warning(f"[FlagshipAlphaMomentumStrategy.on_bars] 警告：权重列全为0，使用静态权重0.6/0.4")
                    last_signal = last_signal.with_columns(
                        (0.6 * pl.col("z_mom") + 0.4 * pl.col("z_vwap_residual")).alias("signal")
                    )
                else:
                    last_signal = last_signal.with_columns(
                        (weight_mom_col * pl.col("z_mom") + weight_vwap_col * pl.col("z_vwap_residual")).alias("signal")
                    )
                    logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] IC-IR方案：重新计算Score，weight_mom非零: {weight_mom_nonzero}, weight_vwap非零: {weight_vwap_nonzero}")
            else:
                last_signal = last_signal.with_columns(
                    (0.6 * pl.col("z_mom") + 0.4 * pl.col("z_vwap_residual")).alias("signal")
                )
                logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] IC-IR方案：使用静态权重0.6/0.4")
        else:
            # LightGBM方案：直接使用预计算的signal
            logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] LightGBM方案：使用预计算的Score")
        
        # 按 Score 降序排序（策略文档要求）
        last_signal = last_signal.sort("signal", descending=True)
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 信号已按 Score 降序排序")
        
        # 更新日级ATR缓存（从信号文件中读取）
        if "atr_14" in last_signal.columns:
            for row in last_signal.iter_rows(named=True):
                vt_symbol = row["vt_symbol"]
                atr_value = row.get("atr_14")
                if atr_value is not None and (isinstance(atr_value, (int, float)) and not (isinstance(atr_value, float) and atr_value != atr_value)):
                    self.daily_atr[vt_symbol] = float(atr_value)
        
        # 风险管理：VIX期限结构过滤器
        vix_ratio = self._get_vix_ratio(bars)
        target_leverage = self._get_target_leverage(vix_ratio)
        if vix_ratio is not None:
            logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] VIX比率: {vix_ratio:.3f}, 目标杠杆: {target_leverage:.2f}")
        
        # 记录 Score 分布
        if len(last_signal) > 0:
            signal_values = last_signal["signal"]
            # 过滤掉None和NaN值
            valid_signals = [s for s in signal_values if s is not None and not (isinstance(s, float) and (s != s))]
            if valid_signals:
                max_score = max(valid_signals)
                min_score = min(valid_signals)
                mean_score = sum(valid_signals) / len(valid_signals)
            logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] Score 分布: max={max_score:.3f}, min={min_score:.3f}, mean={mean_score:.3f}")
            else:
                logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] Score 分布: 无有效信号值")

        # 更新持仓天数（已在交易日开始时更新）
        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos > 0]
        
        # 更新入场后最高价（用于 Chandelier Exit）- 每分钟都更新
        for vt_symbol in pos_symbols:
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

        # === 风险管理：5种退出条件检查（每分钟检查，按优先级执行）===
        # 优先级1：固定止盈 (Target Profit)
        sell_for_take_profit: set[str] = set()
        # 优先级2：跟踪止盈 (Trailing Stop)
        sell_for_trailing_stop: set[str] = set()
        # 优先级3：波动率自适应止损 (Stop Loss)
        sell_for_stop_loss: set[str] = set()
        
        for vt_symbol in pos_symbols:
            if vt_symbol not in bars:
                continue
            
            bar = bars[vt_symbol]
            entry_price = self.entry_prices.get(vt_symbol)
            entry_high = self.entry_highs.get(vt_symbol)
            
            if entry_price is None or entry_high is None:
                continue
            
            # 获取ATR
            atr = self._get_atr(vt_symbol, window=14)
            if atr is None or atr <= 0:
                # 如果ATR不可用，使用简化的百分比止损作为后备
                    price_drop = (entry_price - bar.close_price) / entry_price
                    if price_drop > 0.05:  # 5% 止损（后备方案）
                        sell_for_stop_loss.add(vt_symbol)
                        self.write_log(f"{vt_symbol} 触发止损（后备），入场价 {entry_price:.2f}，当前价 {bar.close_price:.2f}")
                continue
            
            # 优先级1：固定止盈检查
            # P_current >= P_entry + max(2×ATR, 5%×P_entry)
            take_profit_atr_target = entry_price + self.take_profit_atr_multiplier * atr
            take_profit_pct_target = entry_price * (1 + self.take_profit_pct)
            take_profit_target = max(take_profit_atr_target, take_profit_pct_target)
            
            if bar.close_price >= take_profit_target:
                sell_for_take_profit.add(vt_symbol)
                self.write_log(
                    f"{vt_symbol} 触发固定止盈："
                    f"入场价={entry_price:.2f}, 目标价={take_profit_target:.2f}, "
                    f"当前价={bar.close_price:.2f}, 盈利={(bar.close_price/entry_price-1)*100:.2f}%"
                )
                continue  # 已触发止盈，不再检查其他条件
            
            # 优先级2：跟踪止盈检查
            # P_current <= max(H_entry) - 1.5×ATR（仅在价格创出新高后检查）
            # 只有当价格曾经创出新高（entry_high > entry_price）时，才启用跟踪止盈
            if entry_high > entry_price:
                trailing_stop_price = entry_high - self.trailing_stop_atr_multiplier * atr
                if bar.close_price <= trailing_stop_price:
                    sell_for_trailing_stop.add(vt_symbol)
                    self.write_log(
                        f"{vt_symbol} 触发跟踪止盈："
                        f"入场价={entry_price:.2f}, 最高价={entry_high:.2f}, ATR={atr:.2f}, "
                        f"跟踪止损价={trailing_stop_price:.2f}, 当前价={bar.close_price:.2f}"
                    )
                    continue  # 已触发跟踪止盈，不再检查止损
            
            # 优先级3：波动率自适应止损检查
            # 根据VIX调整止损倍数
            vix_ratio = self._get_vix_ratio(bars)
            if vix_ratio is not None and vix_ratio > self.vix_threshold_2:
                # 恐慌市场：止损收紧至1.5×ATR
                stop_loss_multiplier = 1.5
            else:
                # 正常市场：止损 = 3.0×ATR
                stop_loss_multiplier = self.stop_loss_atr_multiplier
            
            exit_price = entry_high - stop_loss_multiplier * atr
            if bar.close_price <= exit_price:
                sell_for_stop_loss.add(vt_symbol)
                self.write_log(
                    f"{vt_symbol} 触发 Chandelier Exit 止损："
                    f"最高价={entry_high:.2f}, ATR={atr:.2f}, "
                    f"止损倍数={stop_loss_multiplier:.1f}, "
                    f"止损价={exit_price:.2f}, 当前价={bar.close_price:.2f}"
                )

        # === 执行退出：按优先级立即执行（每分钟检查）===
        # 执行固定止盈卖出
        if sell_for_take_profit:
            for vt_symbol in sell_for_take_profit:
                if vt_symbol not in bars:
                    continue
                if self.get_pos(vt_symbol) > 0:
                    self.set_target(vt_symbol, 0)
                    self.trade_exit_reasons[vt_symbol] = "固定止盈"
            self.execute_trading(bars, price_add=self.price_add)
            pos_symbols = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos > 0]
        
        # 执行跟踪止盈卖出
        if sell_for_trailing_stop:
            for vt_symbol in sell_for_trailing_stop:
                if vt_symbol not in bars:
                    continue
                if self.get_pos(vt_symbol) > 0:
                    self.set_target(vt_symbol, 0)
                    self.trade_exit_reasons[vt_symbol] = "跟踪止盈"
            self.execute_trading(bars, price_add=self.price_add)
            pos_symbols = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos > 0]
        
        # 执行止损卖出
        if sell_for_stop_loss:
            for vt_symbol in sell_for_stop_loss:
                if vt_symbol not in bars:
                    continue
                if self.get_pos(vt_symbol) > 0:
                    self.set_target(vt_symbol, 0)
                    self.trade_exit_reasons[vt_symbol] = "Chandelier Exit止损"
            self.execute_trading(bars, price_add=self.price_add)
            pos_symbols = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos > 0]
        
        # === 日线选股和调仓：只在每天的第一个K线执行 ===
        # 判断是否应该执行选股和调仓
        # 简化逻辑：如果是新交易日且当日还未执行选股调仓，就执行
        should_rebalance = False
        if is_new_trading_day and not self.daily_rebalance_done:
            should_rebalance = True
            logger.info(f"[FlagshipAlphaMomentumStrategy.on_bars] 新交易日 {current_date}，执行选股调仓（时间: {current_datetime}）")
        
        if not should_rebalance:
            # 不是选股调仓时间，只执行止损（已执行），然后返回
            return
        
        # 标记当日已执行选股调仓
        self.daily_rebalance_done = True

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

        # === 记录卖出信号原因（优先级4和5：仅在每日选股调仓时检查）===
        for vt_symbol in sell_for_alpha_decay:
            # 如果之前没有记录退出原因（说明不是止盈/止损），则记录为信号衰减
            if vt_symbol not in self.trade_exit_reasons:
                self.trade_exit_reasons[vt_symbol] = "信号衰减离场"
        
        for vt_symbol in pos_symbols:
            if self.holding_days[vt_symbol] >= self.max_holding_days:
                # 如果之前没有记录退出原因（说明不是止盈/止损/信号衰减），则记录为时间止损
                if vt_symbol not in self.trade_exit_reasons:
                    self.trade_exit_reasons[vt_symbol] = "时间止损"
        
        # === 生成卖出列表（排除已止损的）===
        sell_symbols: set[str] = sell_for_alpha_decay
        
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
        # 计算当日分位数 (Percentile)
        if not last_signal.is_empty():
            last_signal = last_signal.with_columns(
                (pl.col("signal").rank("min") / pl.len() * 100).alias("quantile")
            )

        # 筛选满足 Score 门槛 和 分位数门槛 的股票
        logger.debug(f"[FlagshipAlphaMomentumStrategy.on_bars] 筛选标准: Score >= {self.min_score_threshold} 且 Quantile >= {self.min_quantile_threshold}")
        
        buyable_df = last_signal.filter(
            (pl.col("signal") >= self.min_score_threshold) &
            (pl.col("quantile") >= self.min_quantile_threshold) &
            (~pl.col("vt_symbol").is_in(pos_symbols))
        )
        logger.info(f"[FlagshipAlphaMomentumStrategy.on_bars] 满足门槛的候选股票: {len(buyable_df)} 只")
        
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
                    quantile = symbol_signal["quantile"][0]
                    # 计算排名（Score降序）
                    ranked_signal = last_signal.with_row_index("rank").sort("signal", descending=True)
                    rank_row = ranked_signal.filter(pl.col("vt_symbol") == vt_symbol)
                    if not rank_row.is_empty():
                        rank = rank_row["rank"][0] + 1  # 从1开始
                        self.trade_entry_reasons[vt_symbol] = f"Top {100-quantile:.1f}%, Rank#{rank}, Score={score:.3f}"
        
        # === 执行买入（动态信念加权，策略文档要求）===
        # 策略文档 4.1 节：动态信念加权 (Dynamic Conviction Weighting)
        # W_{i,t} = exp(S_{i,t} * λ) / Σ_j exp(S_{j,t} * λ)
        # 其中 λ = 2.0（放大系数）
        if buy_symbols:
            import math
            
            # 获取买入股票的 Score 值
            buy_scores: dict[str, float] = {}
            buy_z_scores: dict[str, float] = {}  # 用于判断是否触发极品上限
            for vt_symbol in buy_symbols:
                symbol_signal = last_signal.filter(pl.col("vt_symbol") == vt_symbol)
                if not symbol_signal.is_empty():
                    score = symbol_signal["signal"][0]
                    buy_scores[vt_symbol] = score
                    # 计算 Z-Score（用于判断是否触发极品上限）
                    # Z-Score = (Score - Mean) / Std
                    mean_score = last_signal["signal"].mean()
                    std_score = last_signal["signal"].std()
                    if std_score and std_score > 0:
                        z_score = (score - mean_score) / std_score
                        buy_z_scores[vt_symbol] = z_score
                    else:
                        buy_z_scores[vt_symbol] = 0.0
                else:
                    buy_z_scores[vt_symbol] = 0.0
            
            # 第一步：计算动态信念权重（Softmax）
            # exp_scores = exp(S_i * λ)
            exp_scores: dict[str, float] = {}
            for vt_symbol, score in buy_scores.items():
                exp_scores[vt_symbol] = math.exp(score * self.conviction_lambda)
            
            # 归一化：W_i = exp(S_i * λ) / Σ_j exp(S_j * λ)
            total_exp_score = sum(exp_scores.values())
            raw_weights: dict[str, float] = {}
            for vt_symbol in buy_symbols:
                if total_exp_score > 0:
                    raw_weights[vt_symbol] = exp_scores[vt_symbol] / total_exp_score
                else:
                    # 如果所有 Score 都无效，使用等权重
                    raw_weights[vt_symbol] = 1.0 / len(buy_symbols)
            
            # 第二步：应用双层动态上限（策略文档 4.2 节）
            # - 基础上限：5%（适用于普通入选标的）
            # - 极品上限：8%（触发条件：Z_score > 2.0）
            capped_weights: dict[str, float] = {}
            for vt_symbol, raw_weight in raw_weights.items():
                z_score = buy_z_scores.get(vt_symbol, 0.0)
                # 判断是否触发极品上限
                if z_score > 2.0:
                    # 触发极品上限：8%
                    cap = self.conviction_position_cap
                else:
                    # 基础上限：5%
                    cap = self.base_position_cap
                
                # 应用上限（策略文档要求：基础上限5%，极品上限8%）
                capped_weights[vt_symbol] = min(raw_weight, cap)
            
            # 第三步：重新归一化权重（确保总权重为1）
            total_capped_weight = sum(capped_weights.values())
            if total_capped_weight > 0:
                normalized_weights: dict[str, float] = {}
                for vt_symbol, capped_weight in capped_weights.items():
                    normalized_weights[vt_symbol] = capped_weight / total_capped_weight
            else:
                # 如果所有权重都被上限截断，使用等权重
                normalized_weights = {vt_symbol: 1.0 / len(buy_symbols) for vt_symbol in buy_symbols}
            
            # 应用VIX杠杆调整
            total_buy_value = cash * self.cash_ratio * target_leverage
            
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
                    z_score = buy_z_scores.get(vt_symbol, 0.0)
                    cap_type = "极品上限(8%)" if z_score > 2.0 else "基础上限(5%)"
                    logger.debug(
                        f"{vt_symbol} 买入：Score={buy_scores.get(vt_symbol, 0):.3f}, "
                        f"Z-Score={z_score:.2f}, {cap_type}, "
                        f"原始权重={raw_weights.get(vt_symbol, 0):.4f}, "
                        f"最终权重={weight:.4f}, 金额={buy_value:.2f}, 数量={buy_volume}"
                    )

        # === 执行交易 ===
        self.execute_trading(bars, price_add=self.price_add)
        
        # 更新前一日净值
        self.previous_net_value = self.get_cash_available() + self.get_holding_value()
