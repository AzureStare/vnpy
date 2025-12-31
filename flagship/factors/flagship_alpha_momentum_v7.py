"""
Flagship Alpha-Momentum v7.0 (Aggressive) 因子计算模块。

基于策略文档 v7.0 实现：
- 核心因子：alpha_mom, alpha_vwap, alpha_trend (继承 v5.0)
- 新增因子：
    - rs_score: 相对强度 (vs SPY 60d)
    - beta: 相对大盘 Beta (60d)
    - atr_percent: 波动率百分比 (ATR/Price)
    - sector_rank: 板块热度 (暂用 ETF 代理，本版本简化为全局 RS 排名)
- 过滤条件：ADV > 40M, Price > 10, Market Cap 2B-100B, Price > MA50
- 止盈机制：Profit Ladder (在策略层实现)
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import polars as pl
import numpy as np

from flagship.factors.flagship_alpha_momentum_v5 import FlagshipAlphaMomentumV5Dataset


class FlagshipAlphaMomentumV7Dataset(FlagshipAlphaMomentumV5Dataset):
    """
    Flagship Alpha-Momentum v7.0 Aggressive 数据集
    """

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        process_type: str = "append",
        spy_symbol: str = "SPY.NASDAQ",
    ) -> None:
        # V5 已经定义了 alpha_mom, alpha_vwap, atr_14, close_price, ma50, ret_10d
        # 我们需要在 post_process 中增加 V7 的新因子
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            process_type=process_type,
            spy_symbol=spy_symbol,
        )
        
        # V7 标签保持 rank_5d 逻辑，但增加对 5 日收益的计算
        self.add_feature("ret_5d", "ts_delay(close, -5) / close - 1")
        
        # 覆盖 V5 的处理，使用 V7 的 post_process
        self.processors["infer"] = self._post_process_v7

    def _post_process_v7(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V7 后处理：增加 RS, Beta, ATR% 因子
        """
        # 首先调用 V5 的基础处理（计算了 alpha_mom, alpha_vwap, alpha_trend, z_score 等）
        # 注意：V5 的 _post_process 会进行过滤，我们需要在过滤前计算 RS 和 Beta
        # 所以我们不直接调用 super()._post_process，而是手动合并逻辑
        
        if df.is_empty():
            return df

        eps: float = 1e-8
        
        # 1. 计算基础趋势特征 (EMA 等) 和 ADV
        df = df.sort(["vt_symbol", "datetime"])
        df = df.with_columns([
            pl.col("close_price").ewm_mean(span=5, adjust=False, min_samples=1).over("vt_symbol").alias("ema5"),
            pl.col("close_price").ewm_mean(span=10, adjust=False, min_samples=1).over("vt_symbol").alias("ema10"),
            pl.col("close_price").ewm_mean(span=20, adjust=False, min_samples=1).over("vt_symbol").alias("ema20"),
            pl.col("close_price").ewm_mean(span=50, adjust=False, min_samples=1).over("vt_symbol").alias("ema50"),
        ])
        
        # 计算 ADV (30d median volume * close) 用于仓位管理
        if "volume" in df.columns and "close_price" in df.columns:
            df = df.with_columns([
                 (pl.col("volume").rolling_median(window_size=30, min_periods=1).over("vt_symbol") * pl.col("close_price")).alias("adv_usd")
            ])
        
        # atr_percent: ATR / Price
        if "atr_14" in df.columns:
            df = df.with_columns([
                (pl.col("atr_14") / (pl.col("close_price") + eps)).alias("atr_percent")
            ])
        
        # 2. 计算相对强度 (RS) 和 Beta (需要 SPY 数据)
        # 假设 df 中包含 SPY 数据
        spy_df = df.filter(pl.col("vt_symbol") == self.spy_symbol).select([
            "datetime", "close_price"
        ]).rename({"close_price": "spy_close"})
        
        if not spy_df.is_empty():
            df = df.join(spy_df, on="datetime", how="left")
            # rs_60d: (P / P_60) / (SPY / SPY_60) - 1
            df = df.with_columns([
                (
                    (pl.col("close_price") / (pl.col("close_price").shift(60).over("vt_symbol") + eps)) /
                    (pl.col("spy_close") / (pl.col("spy_close").shift(60).over("vt_symbol") + eps) + eps) - 1
                ).alias("rs_60d")
            ])
            
            # Beta 60d: 使用简化的 rolling correlation * std_ratio
            df = df.with_columns([
                pl.col("close_price").pct_change().over("vt_symbol").alias("ret_i"),
                pl.col("spy_close").pct_change().over("vt_symbol").alias("ret_spy"),
            ])
            
            # 使用 polars 的滚动计算
            df = df.with_columns([
                (
                    pl.rolling_cov(pl.col("ret_i"), pl.col("ret_spy"), window_size=60) /
                    (pl.rolling_var(pl.col("ret_spy"), window_size=60) + eps)
                ).over("vt_symbol").alias("beta")
            ])
        else:
            # Fallback if SPY missing
            df = df.with_columns([
                pl.lit(0.0).alias("rs_60d"),
                pl.lit(1.0).alias("beta")
            ])

        # 3. 计算 V5 的核心因子 (alpha_mom, alpha_vwap, alpha_trend)
        # alpha_trend = ema_distance * I[EMA5 > EMA20]
        if "ema20" in df.columns and "atr_14" in df.columns:
            df = df.with_columns([
                ((pl.col("close_price") - pl.col("ema20")) / (pl.col("atr_14") + eps)).alias("ema_distance")
            ])
            df = df.with_columns([
                (pl.col("ema_distance") * (pl.col("ema5") > pl.col("ema20")).cast(pl.Float64)).alias("alpha_trend")
            ])

        # 4. 过滤 (V7 要求已经在 SQL 选股中做了 ADV/Price/Cap 过滤，这里做 MA50 和 ATR% 过滤)
        # MA50 过滤
        if "ma50" in df.columns:
            df = df.filter(pl.col("close_price") > pl.col("ma50"))
        
        # ATR% 过滤：若 ATR% < 3%，直接剔除（V7 激进版不碰低波动）
        if "atr_percent" in df.columns:
            df = df.filter(pl.col("atr_percent") >= 0.03)

        # 5. 因子合成与标准化 (LGB 模型会处理非线性，但我们仍需准备干净的特征)
        # 我们对所有核心特征进行截面 Z-Score
        core_features = ["alpha_mom", "alpha_vwap", "alpha_trend", "rs_60d", "beta", "atr_percent"]
        for feat in core_features:
            if feat in df.columns:
                df = df.with_columns([
                    ((pl.col(feat) - pl.col(feat).mean().over("datetime")) / 
                     (pl.col(feat).std().over("datetime") + eps)).alias(feat)
                ])

        # 6. 计算最终 Score (作为 fallback)
        # V7 以后主要靠 LGB 模型分数，这里的 score 仅用于没有模型时的基础排序
        df = df.with_columns([
            (pl.col("alpha_mom") * 0.3 + 
             pl.col("alpha_vwap") * 0.3 + 
             pl.col("alpha_trend") * 0.2 + 
             pl.col("rs_60d") * 0.2).alias("score")
        ])

        return df.drop(["ret_i", "ret_spy", "spy_close"])

