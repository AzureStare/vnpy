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


class AlphaMomentumV7(AlphaStrategy):
    """
    Flagship Alpha-Momentum V7.0 Aggressive 策略实现类。

    策略特性：
    - 选股机制：Top 50 候选，配合 Setup A/B 结构过滤
    - 持仓集中：5-8 只
    - 止盈机制：Profit Ladder (阶梯止盈)
    - 杠杆模式：大盘走强时允许 130%-160% 杠杆
    """

    STRATEGY_VERSION = "v7"
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

    # 趋势阶段止盈线（分钟级）：支持线性加权（默认纯 EMA20）
    trend_exit_ema_period: int = 20
    trend_exit_boll_mid_period: int = 5  # 5分钟布林带中轨（等价于 SMA(5)）
    trend_exit_weight_ema: float = 1.0
    trend_exit_weight_boll_mid: float = 0.0

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
            ema20 = am.ema(int(self.trend_exit_ema_period)) if am and am.inited else indicators.get("ema20")
            boll_mid_5 = am.sma(int(self.trend_exit_boll_mid_period)) if am and am.inited else None

            # 线性加权趋势止盈线（缺失时自动降级）
            w_ema = float(self.trend_exit_weight_ema)
            w_mid = float(self.trend_exit_weight_boll_mid)
            if w_ema < 0:
                w_ema = 0.0
            if w_mid < 0:
                w_mid = 0.0

            trend_line = None
            w_sum = 0.0
            if ema20 is not None and w_ema > 0:
                trend_line = float(ema20) * w_ema
                w_sum += w_ema
            if boll_mid_5 is not None and w_mid > 0:
                trend_line = (float(trend_line or 0.0) + float(boll_mid_5) * w_mid)
                w_sum += w_mid
            if trend_line is not None and w_sum > 0:
                trend_line = float(trend_line) / float(w_sum)

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
                    # B. 趋势阶段：跌破趋势止盈线（默认 EMA20）
                    if trend_line and bar.close_price < float(trend_line):
                        if self.trend_exit_weight_boll_mid > 0:
                            exit_reason = f"趋势止盈(加权线) (Price < {float(trend_line):.2f})"
                        else:
                            exit_reason = "EMA20 趋势止盈"
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

    def _cache_daily_indicators(self, signal_df: pl.DataFrame) -> None:
        """
        缓存信号文件中的日级指标（用于分钟级止盈止损）。

        说明：
        - 策略分钟级 exit 会优先用 ArrayManager 的实时 EMA；若未 inited，则回退到这里缓存的日级 EMA。
        - adv_usd 用于 1% ADV 流动性硬约束。
        """
        if signal_df.is_empty() or "vt_symbol" not in signal_df.columns:
            return

        candidate_cols = [
            "atr_14",
            "ema5",
            "ema10",
            "ema20",
            "ema50",
            "atr_percent",
            "close_price",
            "adv_usd",
        ]
        indicator_cols = [c for c in candidate_cols if c in signal_df.columns]
        if not indicator_cols:
            return

        for row in signal_df.select(["vt_symbol", *indicator_cols]).iter_rows(named=True):
            vt_symbol = row["vt_symbol"]
            indicators: dict[str, float] = {}
            for c in indicator_cols:
                v = row.get(c)
                if v is None:
                    continue
                try:
                    indicators[c] = float(v)
                except Exception:
                    continue
            if indicators:
                self.daily_indicators[vt_symbol] = indicators

    def _is_risk_off(self, signal_df: pl.DataFrame, bars: dict[str, BarData]) -> bool:
        """
        风险关闭模式：
        - 触发条件：SPY < EMA20
        - 动作：停止新开仓（止盈止损仍按分钟执行）
        """
        try:
            spy_df = signal_df.filter(pl.col("vt_symbol") == "SPY.NASDAQ")
            if not spy_df.is_empty() and "close_price" in spy_df.columns and "ema20" in spy_df.columns:
                spy_close = spy_df["close_price"][0]
                spy_ema20 = spy_df["ema20"][0]
                if spy_close is not None and spy_ema20 is not None and float(spy_close) < float(spy_ema20):
                    return True
        except Exception:
            pass

        # Fallback to intraday EMA20 if daily signal missing
        spy_bar = bars.get("SPY.NASDAQ")
        if spy_bar and self.spy_ema20 is not None and spy_bar.close_price < self.spy_ema20:
            return True

        return False

    def _get_target_leverage(self, signal_df: pl.DataFrame, bars: dict[str, BarData]) -> float:
        """
        杠杆模式：
        - 条件：SPY > EMA50 时，允许上调至 max_leverage；否则保持 1.0。
        """
        leverage: float = 1.0
        try:
            spy_df = signal_df.filter(pl.col("vt_symbol") == "SPY.NASDAQ")
            if not spy_df.is_empty() and "close_price" in spy_df.columns and "ema50" in spy_df.columns:
                spy_close = spy_df["close_price"][0]
                spy_ema50 = spy_df["ema50"][0]
                if spy_close is not None and spy_ema50 is not None and float(spy_close) > float(spy_ema50):
                    leverage = float(self.max_leverage)
        except Exception:
            pass

        return leverage

    def _build_candidate_df(
        self,
        signal_df: pl.DataFrame,
        *,
        exclude_symbols: set[str],
        top_k: int = 50,
    ) -> pl.DataFrame:
        """从信号中构建候选列表（按 signal 降序，取 Top K）。"""
        if signal_df.is_empty() or "signal" not in signal_df.columns:
            return pl.DataFrame()

        excluded = set(exclude_symbols)
        excluded.update({"SPY.NASDAQ", "VIX.CBOE", "VIX3M.CBOE"})

        df = signal_df
        if excluded:
            df = df.filter(~pl.col("vt_symbol").is_in(list(excluded)))

        df = df.filter(pl.col("signal") >= float(self.min_score_threshold))
        df = df.sort("signal", descending=True)
        return df.head(top_k)

    def _compute_softmax_weights(
        self,
        *,
        signal_df: pl.DataFrame,
        target_symbols: list[str],
        lambda_: float = 2.0,
    ) -> dict[str, float]:
        """
        Softmax 权重（Dynamic Conviction Weighting）：
        - 先对当日截面 signal 做 z-score
        - 再做 softmax(exp(z * lambda))
        """
        if not target_symbols:
            return {}

        # Missing signal column => equal weight
        if signal_df.is_empty() or "signal" not in signal_df.columns:
            equal = 1.0 / float(len(target_symbols))
            return {s: equal for s in target_symbols}

        # Cross-sectional mean/std
        try:
            series = signal_df.select(pl.col("signal").cast(pl.Float64)).to_series()
            mean_signal = float(series.mean())
            std_signal = float(series.std())
        except Exception:
            mean_signal = 0.0
            std_signal = 1.0

        if std_signal <= 1e-12:
            std_signal = 1.0

        subset = (
            signal_df
            .filter(pl.col("vt_symbol").is_in(target_symbols))
            .select(["vt_symbol", "signal"])
        )

        if subset.is_empty():
            equal = 1.0 / float(len(target_symbols))
            return {s: equal for s in target_symbols}

        symbols: list[str] = []
        zscores: list[float] = []
        for row in subset.iter_rows(named=True):
            sym = row["vt_symbol"]
            val = row.get("signal")
            if val is None:
                continue
            symbols.append(sym)
            zscores.append((float(val) - mean_signal) / std_signal)

        if not symbols:
            equal = 1.0 / float(len(target_symbols))
            return {s: equal for s in target_symbols}

        z_arr = np.asarray(zscores, dtype=float) * float(lambda_)
        z_arr = z_arr - float(np.max(z_arr))  # stable softmax
        exp_scores = np.exp(z_arr)
        denom = float(exp_scores.sum())
        if denom <= 0:
            equal = 1.0 / float(len(symbols))
            weights = {s: equal for s in symbols}
        else:
            weights = {symbols[i]: float(exp_scores[i] / denom) for i in range(len(symbols))}

        # Ensure all targets have a key
        for s in target_symbols:
            weights.setdefault(s, 0.0)
        return weights

    def _run_daily_rebalance(self, bars: dict[str, BarData]) -> None:
        """
        每日调仓：
        - 候选：当日信号 Top 50
        - 组合：最多 top_n，优先保留已有持仓，不强制因排名掉出而卖出（让利润奔跑）
        - 新开仓：risk-off（SPY < EMA20）时禁止新开仓
        - 仓位：Softmax 权重 + 杠杆模式（SPY > EMA50） + 1% ADV 约束
        """
        current_date: date
        if self.current_trade_date is not None:
            current_date = self.current_trade_date
        else:
            first_bar = next(iter(bars.values()))
            current_date = first_bar.datetime.replace(tzinfo=None).date()

        # SignalResolver 会将 minute dt 映射到 trade_date snapshot
        signal_df = self.get_signal()
        if signal_df is None or signal_df.is_empty():
            self.write_log(f"[rebalance] {current_date} no signal, skip")
            return

        # Tradeable subset for this backtest run (avoid targeting symbols without bars)
        tradable_set = set(self.vt_symbols)
        tradable_df = (
            signal_df.filter(pl.col("vt_symbol").is_in(list(tradable_set)))
            if tradable_set
            else signal_df
        )

        # Cache daily indicators (for exits & adv cap)
        self._cache_daily_indicators(tradable_df)

        # Risk-Off mode: stop new buys
        if self._is_risk_off(signal_df, bars):
            self.write_log(f"[rebalance] {current_date} Risk-Off (SPY < EMA20): skip new entries")
            return

        # Current holdings (exclude indices)
        index_symbols = {"SPY.NASDAQ", "VIX.CBOE", "VIX3M.CBOE"}
        held_symbols = [s for s, p in self.pos_data.items() if p > 0 and s not in index_symbols]

        # Trading controls: disabled symbols (block new entries only).
        # This list is injected by live runner; backtests may leave it empty.
        disabled_symbols: set[str] = set()
        try:
            raw = getattr(self, "disabled_vt_symbols", None)
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, (list, tuple, set)):
                disabled_symbols = {str(s).strip() for s in raw if str(s).strip()}
        except Exception:
            disabled_symbols = set()

        # Candidate list: Top 50 by model signal, excluding holdings
        exclude_for_candidates = set(held_symbols) | set(disabled_symbols)
        candidate_df = self._build_candidate_df(tradable_df, exclude_symbols=exclude_for_candidates, top_k=50)

        # Add new positions up to top_n
        slots = max(0, int(self.top_n) - len(held_symbols))
        new_symbols: list[str] = []
        if slots > 0 and not candidate_df.is_empty():
            new_symbols = candidate_df.head(slots)["vt_symbol"].to_list()

        target_symbols: list[str] = list(held_symbols) + list(new_symbols)
        if not target_symbols:
            self.write_log(f"[rebalance] {current_date} no target symbols, skip")
            return

        # Softmax weights (lambda=2.0) on daily cross-section z-scored signal
        weights = self._compute_softmax_weights(signal_df=signal_df, target_symbols=target_symbols, lambda_=2.0)

        # Leverage regime
        target_leverage = self._get_target_leverage(signal_df, bars)
        gross_exposure = min(float(self.max_leverage), float(target_leverage)) * float(self.cash_ratio)

        portfolio_value = self.get_portfolio_value()
        for vt_symbol in target_symbols:
            bar = bars.get(vt_symbol)
            if not bar or bar.close_price <= 0:
                continue

            w = float(weights.get(vt_symbol, 0.0))
            if w <= 0:
                continue

            target_value = portfolio_value * gross_exposure * w

            # Liquidity hard constraint: position value <= 1% ADV
            adv_usd = self.daily_indicators.get(vt_symbol, {}).get("adv_usd")
            if adv_usd is not None and adv_usd > 0:
                max_position_value = adv_usd * 0.01
                target_value = min(target_value, max_position_value)

            target_volume = int(target_value / bar.close_price)
            if target_volume <= 0:
                continue

            self.set_target(vt_symbol, target_volume)

        self.execute_trading(bars, price_add=self.price_add)
        self.write_log(
            f"[rebalance] {current_date} held={len(held_symbols)} new={len(new_symbols)} "
            f"targets={len(target_symbols)} gross_exposure={gross_exposure:.2f}"
        )
