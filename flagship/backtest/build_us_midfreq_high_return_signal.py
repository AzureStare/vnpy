from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from flagship.config import PROJECT_ROOT

from vnpy.trader.constant import Interval

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.alpha import AlphaLab  # noqa: E402
from vnpy.alpha.dataset.datasets.us_midfreq_high_return import (  # noqa: E402
    UsMidfreqHighReturnDataset,
)
from vnpy.alpha.dataset import Segment  # noqa: E402


def _infer_periods(df: pl.DataFrame) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    """
    根据可用交易日自动切分 Train/Valid/Test 时间段，比例约 6/2/2。
    """
    dates = (
        df.select("datetime")
        .unique()
        .sort("datetime")["datetime"]
        .to_list()
    )
    if len(dates) < 40:
        raise RuntimeError("可用交易日不足 40 天，暂不构建中频因子数据集。")

    n = len(dates)
    i_train_end = int(n * 0.6)
    i_valid_end = int(n * 0.8)

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    train_period = (fmt(dates[0]), fmt(dates[i_train_end]))
    valid_period = (fmt(dates[i_train_end + 1]), fmt(dates[i_valid_end]))
    test_period = (fmt(dates[i_valid_end + 1]), fmt(dates[-1]))
    return train_period, valid_period, test_period


def build_signal() -> None:
    """
    基于日线数据构建 Flagship Alpha-Momentum 因子与综合 Score，并落地为 AlphaLab 信号文件。

    流程：
        1) 从 lab/us_midfreq_high_return/daily 读取所有可用标的的日线数据；
        2) 使用 UsMidfreqHighReturnDataset 计算因子 A/B/C、截面 Z-Score 与综合 Score；
        3) 将全时段的 Score_t 作为 signal 列写入 lab/us_midfreq_high_return/signal/us_midfreq_high_return.parquet；
        4) 同时将完整数据集以 pickle 形式保存，便于后续做模型扩展或因子分析。
    """
    lab_path = PROJECT_ROOT.joinpath("lab/us_midfreq_high_return")
    lab = AlphaLab(str(lab_path))

    # 自动发现已有日线合约
    daily_files = sorted(lab.daily_path.glob("*.parquet"))
    vt_symbols = [p.stem for p in daily_files]
    if not vt_symbols:
        raise RuntimeError(f"在 {lab.daily_path} 下未发现任何日线 parquet 文件。")

    # 读取原始日线数据（不额外扩展窗口，因子内部自带回溯）
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start="2000-01-01",
        end="2100-01-01",
        extended_days=0,
    )
    if raw_df is None or raw_df.is_empty():
        raise RuntimeError("AlphaLab.load_bar_df 返回空数据，无法构建因子数据集。")

    # 自动推断训练/验证/测试时间段
    train_period, valid_period, test_period = _infer_periods(raw_df)

    # 如存在指数成分数据，则按照成分股过滤，避免前视和幸存者偏差
    component_filters: dict[str, list[tuple[datetime, datetime]]] | None = None
    index_symbol = "I:NDX"
    try:
        component_filters = lab.load_component_filters(
            index_symbol=index_symbol,
            start=train_period[0],
            end=test_period[1],
        )
        if not component_filters:
            component_filters = None
    except Exception:
        component_filters = None

    # 构建数据集并计算因子 + Score
    dataset = UsMidfreqHighReturnDataset(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )
    dataset.prepare_data(filters=component_filters)
    dataset.process_data()

    # 将全时段的 Score_t 作为信号输出
    infer_df = dataset.fetch_infer(Segment.TEST)
    # 也可以改成 dataset.infer_df 取全时段；这里先对测试段输出交易用信号
    signal_df = (
        infer_df.select(["datetime", "vt_symbol", "score"])
        .sort(["datetime", "vt_symbol"])
        .rename({"score": "signal"})
    )
    lab.save_signal("us_midfreq_high_return", signal_df)

    # 同步保存数据集对象，便于后续做模型扩展或因子分析
    lab.save_dataset("us_midfreq_high_return", dataset)

    print(
        f"[INFO] Saved signal for {len(signal_df)} rows, "
        f"{signal_df['vt_symbol'].n_unique()} symbols, "
        f"to {lab.signal_path.joinpath('us_midfreq_high_return.parquet')}"
    )


if __name__ == "__main__":
    build_signal()


