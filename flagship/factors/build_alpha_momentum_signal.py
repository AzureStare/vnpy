"""
构建 Flagship Alpha-Momentum 策略信号。

基于策略文档计算三个因子（A/B/C），进行截面标准化，生成综合 Score。
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from datetime import date, datetime, timedelta

from flagship.config import PROJECT_ROOT

from vnpy.trader.constant import Interval

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset.datasets.us_midfreq_high_return import UsMidfreqHighReturnDataset
from vnpy.alpha.dataset import Segment


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
        raise RuntimeError("可用交易日不足 40 天，暂不构建因子数据集。")

    n = len(dates)
    i_train_end = int(n * 0.6)
    i_valid_end = int(n * 0.8)

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    train_period = (fmt(dates[0]), fmt(dates[i_train_end]))
    valid_period = (fmt(dates[i_train_end + 1]), fmt(dates[i_valid_end]))
    test_period = (fmt(dates[i_valid_end + 1]), fmt(dates[-1]))
    return train_period, valid_period, test_period


def build_signal(
    lab_path: str | Path,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """
    基于日线数据构建 Flagship Alpha-Momentum 因子与综合 Score，并落地为 AlphaLab 信号文件。

    Args:
        lab_path: AlphaLab 数据目录路径
        start: 起始日期（YYYY-MM-DD），如果为 None 则使用所有可用数据
        end: 结束日期（YYYY-MM-DD），如果为 None 则使用所有可用数据
    """
    logger.info(f"[build_signal] 开始构建 Flagship Alpha-Momentum 信号")
    logger.info(f"[build_signal] Lab 路径: {lab_path}")
    logger.info(f"[build_signal] 日期范围: {start} 到 {end}")
    
    lab = AlphaLab(str(lab_path))
    logger.debug(f"[build_signal] AlphaLab 初始化完成")

    # 自动发现已有日线合约
    logger.info(f"[build_signal] 扫描日线数据目录: {lab.daily_path}")
    daily_files = sorted(lab.daily_path.glob("*.parquet"))
    vt_symbols = [p.stem for p in daily_files]
    logger.info(f"[build_signal] 发现 {len(vt_symbols)} 个日线数据文件")
    
    if not vt_symbols:
        logger.error(f"[build_signal] 在 {lab.daily_path} 下未发现任何日线 parquet 文件")
        raise RuntimeError(f"在 {lab.daily_path} 下未发现任何日线 parquet 文件。")
    
    logger.debug(f"[build_signal] 前5个合约: {vt_symbols[:5]}")

    # 读取原始日线数据
    # 因子计算需要至少60天的历史数据，所以需要扩展起始日期
    logger.info(f"[build_signal] 开始读取日线数据...")
    
    # 解析日期范围
    if start:
        start_dt = datetime.fromisoformat(start).date()
    else:
        start_dt = date(2020, 1, 1)
    
    if end:
        end_dt = datetime.fromisoformat(end).date()
    else:
        end_dt = date(2100, 1, 1)
    
    # 扩展起始日期以获取足够的历史数据（至少90天，用于计算60日因子）
    extended_start = start_dt - timedelta(days=90)
    logger.info(f"[build_signal] 扩展日期范围: {extended_start} 到 {end_dt}（因子计算需要历史数据）")
    
    logger.debug(f"[build_signal] 读取参数: vt_symbols={len(vt_symbols)}, start={extended_start}, end={end_dt}")
    
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=extended_start.isoformat(),
        end=end_dt.isoformat(),
        extended_days=0,
    )
    
    if raw_df is None or raw_df.is_empty():
        logger.error(f"[build_signal] AlphaLab.load_bar_df 返回空数据")
        raise RuntimeError("AlphaLab.load_bar_df 返回空数据，无法构建因子数据集。")
    
    logger.info(f"[build_signal] 日线数据读取完成: {len(raw_df)} 行")
    logger.debug(f"[build_signal] 数据列: {raw_df.columns}")

    # 自动推断训练/验证/测试时间段
    logger.info(f"[build_signal] 推断训练/验证/测试时间段...")
    train_period, valid_period, test_period = _infer_periods(raw_df)
    logger.info(f"[build_signal] 时间段划分:")
    logger.info(f"[build_signal]   - 训练期: {train_period[0]} 到 {train_period[1]}")
    logger.info(f"[build_signal]   - 验证期: {valid_period[0]} 到 {valid_period[1]}")
    logger.info(f"[build_signal]   - 测试期: {test_period[0]} 到 {test_period[1]}")

    # 构建数据集并计算因子 + Score
    logger.info(f"[build_signal] 初始化 UsMidfreqHighReturnDataset...")
    dataset = UsMidfreqHighReturnDataset(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )
    
    logger.info(f"[build_signal] 准备数据 (prepare_data)...")
    dataset.prepare_data(filters=None)
    
    logger.info(f"[build_signal] 处理数据 (process_data)...")
    dataset.process_data()
    logger.info(f"[build_signal] 因子计算完成")

    # 将全时段的 Score_t 作为信号输出
    logger.info(f"[build_signal] 提取测试期信号数据...")
    infer_df = dataset.fetch_infer(Segment.TEST)
    logger.info(f"[build_signal] 测试期数据: {len(infer_df)} 行")
    
    signal_df = (
        infer_df.select(["datetime", "vt_symbol", "score"])
        .sort(["datetime", "vt_symbol"])
        .rename({"score": "signal"})
    )
    logger.info(f"[build_signal] 信号数据准备完成: {len(signal_df)} 行, {signal_df['vt_symbol'].n_unique()} 个合约")
    
    logger.info(f"[build_signal] 保存信号文件...")
    signal_path = lab.signal_path.joinpath("flagship_alpha_momentum.parquet")
    lab.save_signal("flagship_alpha_momentum", signal_df)
    logger.info(f"[build_signal] 信号文件已保存: {signal_path}")

    # 同步保存数据集对象，便于后续做模型扩展或因子分析
    logger.info(f"[build_signal] 保存数据集对象...")
    lab.save_dataset("flagship_alpha_momentum", dataset)
    logger.info(f"[build_signal] 数据集对象已保存")

    logger.info(f"[build_signal] 信号构建完成:")
    logger.info(f"[build_signal]   - 信号行数: {len(signal_df)}")
    logger.info(f"[build_signal]   - 合约数量: {signal_df['vt_symbol'].n_unique()}")
    logger.info(f"[build_signal]   - 信号文件: {signal_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Flagship Alpha-Momentum signal from daily bar data."
    )
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab 数据目录路径（默认 lab/flagship_alpha_momentum）",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="起始日期 (YYYY-MM-DD)，如果为 None 则使用所有可用数据",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="结束日期 (YYYY-MM-DD)，如果为 None 则使用所有可用数据",
    )
    args = parser.parse_args()

    build_signal(
        lab_path=args.lab_path,
        start=args.start,
        end=args.end,
    )

