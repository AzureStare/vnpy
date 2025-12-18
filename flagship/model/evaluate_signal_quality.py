"""
因子有效性评估模块

用于评估信号/因子的预测能力，包括：
- IC (Information Coefficient): 因子值与未来收益的相关性
- Rank IC: 因子排序与收益排序的相关性
- 分位数收益分析: 验证高分组是否跑赢低分组
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.trader.logger import logger


def compute_ic_stats(ic_series: pl.Series | list[float]) -> dict[str, float]:
    """
    计算IC统计量
    
    Parameters
    ----------
    ic_series : pl.Series | list[float]
        IC序列（每日IC值）
    
    Returns
    -------
    dict[str, float]
        包含均值、标准差、t值、ICIR的字典
    """
    if isinstance(ic_series, pl.Series):
        # 直接使用Series的is_finite方法
        ic_values = ic_series.filter(ic_series.is_finite()).to_list()
    elif isinstance(ic_series, list):
        ic_values = [x for x in ic_series if x is not None and not (isinstance(x, float) and (x != x))]
    else:
        ic_values = []
    
    if not ic_values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "t_value": 0.0,
            "icir": 0.0,
        }
    
    ic_array = np.array(ic_values)
    mean = float(np.mean(ic_array))
    std = float(np.std(ic_array, ddof=1))  # 样本标准差
    
    n = len(ic_values)
    if std > 0 and n > 1:
        t_value = mean / (std / math.sqrt(n))
        icir = mean / std if std > 0 else 0.0
    else:
        t_value = 0.0
        icir = 0.0
    
    return {
        "mean": mean,
        "std": std,
        "t_value": t_value,
        "icir": icir,
    }


def evaluate_metrics(
    dataset: AlphaDataset,
    signal_df: pl.DataFrame,
    segment: Segment = Segment.TEST,
    quantiles: int = 5,
) -> dict[str, Any]:
    """
    评估因子有效性指标
    
    Parameters
    ----------
    dataset : AlphaDataset
        数据集对象，包含label列
    signal_df : pl.DataFrame
        信号DataFrame，必须包含 datetime, vt_symbol, signal 列
    segment : Segment
        评估的数据段（TRAIN/VALID/TEST）
    quantiles : int
        分位数数量（默认5，即分为5组）
    
    Returns
    -------
    dict[str, Any]
        包含以下键的字典：
        - sample_size: 样本数
        - coverage: 覆盖率（有信号的数据占比）
        - ic: IC统计量 {mean, std, t_value, icir}
        - rank_ic: Rank IC统计量 {mean, std, t_value, icir}
        - quantile_returns: 分位数收益列表
        - top_quantile_turnover: Top分位换手率
    """
    logger.info(f"[evaluate_metrics] 开始评估因子有效性，segment={segment}, quantiles={quantiles}")
    
    # 1. 获取数据集
    # 优先使用infer_df（包含所有处理后的数据），如果不存在则尝试fetch_infer
    df_data = None
    if hasattr(dataset, 'infer_df') and dataset.infer_df is not None and not dataset.infer_df.is_empty():
        df_data = dataset.infer_df
        logger.debug("[evaluate_metrics] 使用dataset.infer_df")
    elif hasattr(dataset, 'raw_df') and dataset.raw_df is not None and not dataset.raw_df.is_empty():
        df_data = dataset.raw_df
        logger.debug("[evaluate_metrics] 使用dataset.raw_df")
    else:
        # 回退到fetch_infer
        try:
            df_data = dataset.fetch_infer(segment)
            logger.debug(f"[evaluate_metrics] 使用fetch_infer({segment})")
        except Exception as exc:
            logger.debug(f"[evaluate_metrics] fetch_infer失败: {exc}，尝试使用fetch_learn")
            try:
                df_data = dataset.fetch_learn(segment)
                logger.debug(f"[evaluate_metrics] 使用fetch_learn({segment})")
            except Exception as exc2:
                logger.warning(f"[evaluate_metrics] 无法获取数据集: {exc2}")
    
    if df_data is None or df_data.is_empty():
        logger.warning(f"[evaluate_metrics] 数据集为空")
        return {
            "sample_size": 0,
            "coverage": 0.0,
            "ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "rank_ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "quantile_returns": [],
            "top_quantile_turnover": {"average": 0.0, "count": 0},
        }
    
    # 根据信号文件的日期范围过滤数据集
    if not signal_df.is_empty():
        signal_start = signal_df["datetime"].min()
        signal_end = signal_df["datetime"].max()
        logger.info(f"[evaluate_metrics] 信号文件日期范围: {signal_start} 到 {signal_end}")
        logger.info(f"[evaluate_metrics] 过滤前数据集: {len(df_data)} 行")
        
        df_data = df_data.filter(
            (pl.col("datetime") >= signal_start) & 
            (pl.col("datetime") <= signal_end)
        )
        logger.info(f"[evaluate_metrics] 过滤后数据集: {len(df_data)} 行")
    
    # 2. 检查必需的列
    required_cols = ["datetime", "vt_symbol"]
    if "label" not in df_data.columns:
        logger.warning("[evaluate_metrics] 数据集缺少label列")
        return {
            "sample_size": 0,
            "coverage": 0.0,
            "ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "rank_ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "quantile_returns": [],
            "top_quantile_turnover": {"average": 0.0, "count": 0},
        }
    
    # 3. 检查signal_df的列
    signal_col = None
    for col in ["signal", "score"]:
        if col in signal_df.columns:
            signal_col = col
            break
    
    if signal_col is None:
        logger.warning("[evaluate_metrics] signal_df缺少signal或score列")
        return {
            "sample_size": 0,
            "coverage": 0.0,
            "ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "rank_ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "quantile_returns": [],
            "top_quantile_turnover": {"average": 0.0, "count": 0},
        }
    
    # 4. 合并数据
    signal_subset = signal_df.select(["datetime", "vt_symbol", signal_col])
    
    # 调试信息
    logger.debug(f"[evaluate_metrics] df_data: {len(df_data)} 行, 列: {df_data.columns[:10]}")
    logger.debug(f"[evaluate_metrics] signal_subset: {len(signal_subset)} 行, 列: {signal_subset.columns}")
    if not df_data.is_empty():
        logger.debug(f"[evaluate_metrics] df_data日期范围: {df_data['datetime'].min()} 到 {df_data['datetime'].max()}")
        logger.debug(f"[evaluate_metrics] df_data vt_symbol示例: {df_data['vt_symbol'].head(5).to_list()}")
    if not signal_subset.is_empty():
        logger.debug(f"[evaluate_metrics] signal_subset日期范围: {signal_subset['datetime'].min()} 到 {signal_subset['datetime'].max()}")
        logger.debug(f"[evaluate_metrics] signal_subset vt_symbol示例: {signal_subset['vt_symbol'].head(5).to_list()}")
    
    merged_df = df_data.join(
        signal_subset,
        on=["datetime", "vt_symbol"],
        how="inner",
    )
    
    if merged_df.is_empty():
        logger.warning("[evaluate_metrics] 合并后的数据为空，可能原因：")
        logger.warning(f"  - 日期范围不匹配: df_data={df_data['datetime'].min() if not df_data.is_empty() else 'N/A'}~{df_data['datetime'].max() if not df_data.is_empty() else 'N/A'}, signal={signal_subset['datetime'].min() if not signal_subset.is_empty() else 'N/A'}~{signal_subset['datetime'].max() if not signal_subset.is_empty() else 'N/A'}")
        if not df_data.is_empty() and not signal_subset.is_empty():
            df_symbols = set(df_data['vt_symbol'].unique().to_list())
            signal_symbols = set(signal_subset['vt_symbol'].unique().to_list())
            common_symbols = df_symbols & signal_symbols
            logger.warning(f"  - 共同股票数: {len(common_symbols)} / df_data={len(df_symbols)}, signal={len(signal_symbols)}")
        return {
            "sample_size": 0,
            "coverage": 0.0,
            "ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "rank_ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "quantile_returns": [],
            "top_quantile_turnover": {"average": 0.0, "count": 0},
        }
    
    # 5. 过滤有效数据（label和signal都不为NaN）
    valid_df = merged_df.filter(
        pl.col("label").is_finite() & pl.col(signal_col).is_finite()
    )
    
    total_samples = len(df_data)
    valid_samples = len(valid_df)
    coverage = valid_samples / total_samples if total_samples > 0 else 0.0
    
    logger.info(f"[evaluate_metrics] 总样本数: {total_samples}, 有效样本数: {valid_samples}, 覆盖率: {coverage:.2%}")
    
    if valid_df.is_empty():
        logger.warning("[evaluate_metrics] 有效数据为空")
        return {
            "sample_size": valid_samples,
            "coverage": coverage,
            "ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "rank_ic": {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0},
            "quantile_returns": [],
            "top_quantile_turnover": {"average": 0.0, "count": 0},
        }
    
    # 6. 计算每日截面IC
    ic_daily = (
        valid_df
        .group_by("datetime")
        .agg([
            pl.corr(signal_col, "label").alias("ic"),
        ])
        .filter(pl.col("ic").is_finite())
        .sort("datetime")
    )
    
    ic_stats = {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0}
    if not ic_daily.is_empty():
        ic_series = ic_daily["ic"]
        ic_stats = compute_ic_stats(ic_series)
        logger.info(f"[evaluate_metrics] IC统计: mean={ic_stats['mean']:.4f}, std={ic_stats['std']:.4f}, ICIR={ic_stats['icir']:.4f}")
    
    # 7. 计算Rank IC
    rank_ic_daily = (
        valid_df
        .with_columns([
            pl.col(signal_col).rank().over("datetime").alias("signal_rank"),
            pl.col("label").rank().over("datetime").alias("label_rank"),
        ])
        .group_by("datetime")
        .agg([
            pl.corr("signal_rank", "label_rank").alias("rank_ic"),
        ])
        .filter(pl.col("rank_ic").is_finite())
        .sort("datetime")
    )
    
    rank_ic_stats = {"mean": 0.0, "std": 0.0, "t_value": 0.0, "icir": 0.0}
    if not rank_ic_daily.is_empty():
        rank_ic_series = rank_ic_daily["rank_ic"]
        rank_ic_stats = compute_ic_stats(rank_ic_series)
        logger.info(f"[evaluate_metrics] Rank IC统计: mean={rank_ic_stats['mean']:.4f}, std={rank_ic_stats['std']:.4f}, Rank ICIR={rank_ic_stats['icir']:.4f}")
    
    # 8. 计算分位数收益
    quantile_returns = []
    if not valid_df.is_empty():
        # 按日期分组，计算每日的分位数收益
        # 使用rank和cut来实现分位数分组
        quantile_df = (
            valid_df
            .with_columns([
                # 计算每日截面rank
                pl.col(signal_col).rank().over("datetime").alias("signal_rank"),
                pl.col("datetime").count().over("datetime").alias("daily_count"),
            ])
            .with_columns([
                # 计算分位数：rank / count * quantiles，然后向下取整
                ((pl.col("signal_rank") / pl.col("daily_count") * quantiles)
                 .floor()
                 .clip(0, quantiles - 1)
                 .cast(pl.Int32)
                 .add(1))
                .alias("quantile_num")
            ])
            .with_columns([
                # 转换为Q1, Q2, ...格式
                pl.format("Q{}", pl.col("quantile_num")).alias("quantile_label")
            ])
            .group_by(["datetime", "quantile_label"])
            .agg([
                pl.col("label").mean().alias("avg_return"),
                pl.col("label").count().alias("count"),
            ])
            .group_by("quantile_label")
            .agg([
                pl.col("avg_return").mean().alias("avg_return"),
                pl.col("count").sum().alias("count"),
            ])
            .sort("quantile_label")
        )
        
        for row in quantile_df.iter_rows(named=True):
            quantile_returns.append({
                "quantile": row["quantile_label"],
                "avg_return": float(row["avg_return"]),
                "count": int(row["count"]),
            })
        
        logger.info(f"[evaluate_metrics] 分位数收益: {quantile_returns}")
    
    # 9. 计算Top分位换手率（简化版本，计算每日Top分位的股票变化率）
    turnover_stats = {"average": 0.0, "count": 0}
    if not valid_df.is_empty() and quantiles >= 2:
        # 获取每日Top分位的股票列表，计算换手率
        top_quantile_label = f"Q{quantiles}"  # 最高分位
        
        daily_top_stocks = (
            valid_df
            .with_columns([
                # 计算每日截面rank
                pl.col(signal_col).rank().over("datetime").alias("signal_rank"),
                pl.col("datetime").count().over("datetime").alias("daily_count"),
            ])
            .with_columns([
                # 计算分位数
                ((pl.col("signal_rank") / pl.col("daily_count") * quantiles)
                 .floor()
                 .clip(0, quantiles - 1)
                 .cast(pl.Int32)
                 .add(1))
                .alias("quantile_num")
            ])
            .with_columns([
                pl.format("Q{}", pl.col("quantile_num")).alias("quantile_label")
            ])
            .filter(pl.col("quantile_label") == top_quantile_label)
            .select(["datetime", "vt_symbol"])
            .sort("datetime")
        )
        
        if not daily_top_stocks.is_empty():
            dates = daily_top_stocks["datetime"].unique().sort()
            turnovers = []
            
            prev_stocks = set()
            for date in dates:
                current_stocks = set(
                    daily_top_stocks
                    .filter(pl.col("datetime") == date)
                    .select("vt_symbol")
                    .to_series()
                    .to_list()
                )
                
                if prev_stocks:
                    # 计算换手率：新进入的股票数 / 总股票数
                    new_stocks = current_stocks - prev_stocks
                    exit_stocks = prev_stocks - current_stocks
                    turnover = (len(new_stocks) + len(exit_stocks)) / (len(prev_stocks) + len(current_stocks)) if (prev_stocks or current_stocks) else 0.0
                    turnovers.append(turnover)
                
                prev_stocks = current_stocks
            
            if turnovers:
                turnover_stats = {
                    "average": float(np.mean(turnovers)),
                    "count": len(turnovers),
                }
                logger.info(f"[evaluate_metrics] Top分位换手率: {turnover_stats['average']:.4f}, 统计天数: {turnover_stats['count']}")
    
    # 10. 返回结果
    result = {
        "sample_size": valid_samples,
        "coverage": coverage,
        "ic": ic_stats,
        "rank_ic": rank_ic_stats,
        "quantile_returns": quantile_returns,
        "top_quantile_turnover": turnover_stats,
    }
    
    logger.info(f"[evaluate_metrics] 因子有效性评估完成")
    return result

