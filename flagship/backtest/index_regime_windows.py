"""
大盘行情分段配置，用于批量回测。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RegimeWindow:
    id: int
    label: str
    start: date
    end: date
    feature: str
    open_price: float
    close_price: float
    pct_change: float


REGIME_WINDOWS: list[RegimeWindow] = [
    RegimeWindow(1, "2024Q1震荡上涨", date(2024, 1, 2), date(2024, 4, 12),
                 "小幅震荡上涨，2月23日主要高位小幅横盘震荡",
                 14873.7, 15885.02, 6.799384),
    RegimeWindow(2, "2024春季回落", date(2024, 3, 21), date(2024, 8, 5),
                 "小周期震荡回落，有十多天快速回落行情",
                 16571.24, 16200.08, -1.920176),
    RegimeWindow(3, "2024夏季短反", date(2024, 8, 1), date(2024, 8, 21),
                 "短周期快速V型反转",
                 17647.03, 17918.99, 1.541109),
    RegimeWindow(4, "2024秋季震荡上行", date(2024, 8, 5), date(2024, 12, 17),
                 "中等幅度震荡上涨，震荡行情较多",
                 15712.53, 20109.06, 27.981044),
    RegimeWindow(5, "2024-2025高位箱体", date(2024, 11, 29), date(2025, 2, 26),
                 "高位小箱体震荡行情",
                 19087.47, 19109.32, 0.114473),
    RegimeWindow(6, "2025政策利空急跌", date(2025, 2, 21), date(2025, 4, 7),
                 "政策利空事件，行情快速冲高回踩，有加速大幅杀跌行情",
                 20000.69, 15603.26, -22.009788),
    RegimeWindow(7, "2025春季筑底", date(2025, 3, 26), date(2025, 4, 25),
                 "市场利空消化，市场小幅波动筑底",
                 18217.33, 17382.94, -4.580199),
    RegimeWindow(8, "2025春季V反", date(2025, 2, 21), date(2025, 5, 16),
                 "市场大震荡V型反转，上涨斜率较低",
                 20000.69, 19211.11, -3.976262),
    RegimeWindow(9, "2025夏季稳步上涨", date(2025, 4, 21), date(2025, 7, 31),
                 "长时间段稳步震荡上涨，小幅上涨周期明显",
                 16052.76, 21122.45, 31.581423),
    RegimeWindow(10, "2025秋季高位回落", date(2025, 9, 22), date(2025, 11, 21),
                 "高位震荡快速回落，小头部形态",
                 22606.59, 22273.08, -1.475278),
]


def get_regime_window(idx: int) -> RegimeWindow:
    for window in REGIME_WINDOWS:
        if window.id == idx:
            return window
    raise ValueError(f"Regime window id={idx} not found")


