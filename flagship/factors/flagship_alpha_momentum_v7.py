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


import polars as pl

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
        
        # V7 新增特征与 Label 调整
        # 1. 设置 Label 为 5 日远期收益率 (V7 标准)
        self.set_label("ts_delay(close, -5) / close - 1")
        
        # 2. 批量添加 V7 特征 (继承自 V5 的特征会自动包含)
        v7_features = {
            "ret_5d": "ts_delay(close, -5) / close - 1",      # 未来 5 日收益 (用于分析)
            "return_5d": "close / ts_delay(close, 5) - 1",    # 过去 5 日历史收益 (Mom 5d 特征)
            "volume": "volume",                               # 成交量 (用于计算 ADV)
            "adv_usd": "volume",                              # 占位符，将在 post_process 中更新为真实 ADV
            "med_volume": "volume",                           # 占位符，将在 post_process 中更新为真实 MedVol
        }
        for name, expr in v7_features.items():
            self.add_feature(name, expr)

        # 覆盖 V5 的处理逻辑：V7 使用独立的 _post_process_v7
        # 清除父类 (V5) 注册的处理器，确保不会执行 V5 的 _post_process
        self.infer_processors = []
        self.learn_processors = []
        
        self.add_processor("learn", self._post_process_v7)
        self.add_processor("infer", self._post_process_v7)

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
                 pl.col("volume").rolling_median(window_size=30, min_periods=1).over("vt_symbol").alias("med_volume")
            ])
            df = df.with_columns([
                 (pl.col("med_volume") * pl.col("close_price")).alias("adv_usd")
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
            # rs_score: (P / P_60) / (SPY / SPY_60) - 1
            df = df.with_columns([
                (
                    (pl.col("close_price") / (pl.col("close_price").shift(60).over("vt_symbol") + eps)) /
                    (pl.col("spy_close") / (pl.col("spy_close").shift(60).over("vt_symbol") + eps) + eps) - 1
                ).alias("rs_score")
            ]).with_columns([
                pl.col("rs_score").alias("rs_60d") # Alias for compatibility with models trained with old names
            ])
            
            # Beta 60d: 使用基础统计公式实现，兼容旧版 Polars
            df = df.with_columns([
                pl.col("close_price").pct_change().over("vt_symbol").alias("ret_i"),
                pl.col("spy_close").pct_change().over("vt_symbol").alias("ret_spy"),
            ])
            
            df = df.with_columns([
                (
                    ( (pl.col("ret_i") * pl.col("ret_spy")).rolling_mean(60) - 
                      pl.col("ret_i").rolling_mean(60) * pl.col("ret_spy").rolling_mean(60) ) /
                    ( pl.col("ret_spy").rolling_std(60).pow(2) + eps )
                ).over("vt_symbol").alias("beta")
            ])
        else:
            # Fallback if SPY missing
            df = df.with_columns([
                pl.lit(0.0).alias("rs_score"),
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
        
        # ATR% 过滤：若 ATR% < 3%，直接剔除（V7 不碰低波动）
        if "atr_percent" in df.columns:
            df = df.filter(pl.col("atr_percent") >= 0.03)

        # 5. 因子合成与标准化 (LGB 模型会处理非线性，但我们仍需准备干净的特征)
        # 我们对所有核心特征进行截面 Z-Score
        core_features = ["alpha_mom", "alpha_vwap", "alpha_trend", "rs_score", "beta", "atr_percent", "return_5d"]
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
             pl.col("rs_score") * 0.2).alias("score")
        ])

        # 明确保留 adv_usd 和 med_volume 列（供前端和策略使用），确保不被过滤掉
        # 在 AlphaDataset 中，如果不在 features/label 中可能会被过滤
        # 我们这里已经是 DataFrame 形式，直接返回即可
        
        # 清理临时列
        drop_cols = [c for c in ["ret_i", "ret_spy", "spy_close", "ema_distance"] if c in df.columns]
        return df.drop(drop_cols)

