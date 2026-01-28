from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set, TYPE_CHECKING

from vnpy.alpha.strategy import AlphaStrategy
from vnpy.trader.object import BarData, TradeData

if TYPE_CHECKING:
    from vnpy.alpha.strategy.backtesting import BacktestingEngine


class AlphaMomentumV5(AlphaStrategy):
    """
    Flagship Alpha-Momentum 策略 V5 (复刻版)

    逻辑:
    1. 基于 Alpha 得分 (LightGBM/Composite) 选择 Top N 股票。
    2. 开盘买入 (或前一日收盘)。
    3. 最多持有 5 天。
    4. 退出机制:
       - 止损: 3.0 ATR (若 VIX 恐慌则收紧至 1.5 ATR)。
       - 止盈: Max(2.0 ATR, 5%)。
       - 跟踪止损: 距高点 1.5 ATR (若创新高)。
       - 时间止损: 5 天。
       - 排名衰减: 若得分跌出 Top N * Buffer。
    """

    # 策略参数
    top_n: int = 5
    min_score_threshold: float = 0.5
    min_quantile_threshold: float = 80.0 # 前 20%

    # 风控参数
    max_holding_days: int = 5
    stop_loss_atr_multiplier: float = 3.0
    panic_stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.0
    take_profit_min_pct: float = 0.05
    trailing_stop_atr_multiplier: float = 1.5

    # 仓位管理
    max_pos_weight: float = 0.20 # 5 只股票 = 每只 20% (理论最大值)

    # VIX 过滤
    vix_panic_threshold: float = 1.1 # VIX / VIX3M > 1.1 => 恐慌

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
        alpha_engine: "BacktestingEngine",
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
        self.write_log("Flagship Alpha-Momentum V5 策略初始化")
        self.load_bars(10)

    def on_start(self) -> None:
        self.write_log("Flagship Alpha-Momentum V5 策略启动")

    def on_stop(self) -> None:
        self.write_log("Flagship Alpha-Momentum V5 策略停止")

    def on_daily_bar(self, bar: BarData) -> None:
        """每日逻辑 (投资组合再平衡)"""
        # 通常由引擎在日终调用或通过定时器显式调用
        pass

    def on_min_bar(self, bar: BarData) -> None:
        """盘中逻辑 (风控与执行)"""
        vt_symbol = bar.vt_symbol

        # 更新 ATR (简化处理，假设每日 ATR 可用或在别处计算)
        # 这里我们依赖通过信号传递或动态计算的预计算 ATR
        # 为简单起见，我们假设从外部源或 bar 管理器获取每日 ATR
        atr = self.atr_map.get(vt_symbol, bar.close_price * 0.02) # 兜底 2%

        if self.trading:
            self._check_risk_management(bar, atr)

    def _check_risk_management(self, bar: BarData, atr: float) -> None:
        """检查止损、止盈、时间止损"""
        vt_symbol = bar.vt_symbol
        pos = self.get_pos(vt_symbol)

        if pos == 0:
            return

        entry_price = self.entry_prices.get(vt_symbol, bar.close_price)
        high_price = self.high_prices.get(vt_symbol, entry_price)

        # 更新最高价
        if bar.high_price > high_price:
            self.high_prices[vt_symbol] = bar.high_price
            high_price = bar.high_price

        # 1. 止盈
        tp_price = entry_price + max(self.take_profit_atr_multiplier * atr, entry_price * self.take_profit_min_pct)
        if bar.high_price >= tp_price:
            self.sell(vt_symbol, bar.close_price, pos, "Take Profit")
            return

        # 2. 跟踪止损
        trailing_stop_price = high_price - self.trailing_stop_atr_multiplier * atr
        if bar.low_price <= trailing_stop_price:
             self.sell(vt_symbol, bar.close_price, pos, "Trailing Stop")
             return

        # 3. 止损 (自适应)
        sl_multiplier = self.panic_stop_loss_atr_multiplier if self.panic_mode else self.stop_loss_atr_multiplier
        sl_price = entry_price - sl_multiplier * atr
        if bar.low_price <= sl_price:
            self.sell(vt_symbol, bar.close_price, pos, "Stop Loss")
            return

        # 4. 时间止损 (通常在每日级别检查，但如果跟踪时间戳也可在此检查)
        entry_time = self.entry_times.get(vt_symbol)
        if entry_time:
            days_held = (bar.datetime.replace(tzinfo=None) - entry_time.replace(tzinfo=None)).days
            if days_held >= self.max_holding_days:
                 self.sell(vt_symbol, bar.close_price, pos, "Time Stop")
            return
