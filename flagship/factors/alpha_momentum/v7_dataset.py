"""
Flagship Alpha-Momentum v7.0 (Aggressive) 因子计算模块。

策略逻辑与因子体系详解：

1. 核心因子体系 (Core Factors):
   - alpha_mom (波动率调整突破): 衡量价格相对于过去20日最高价的突破程度，经ATR标准化。
     公式: (Close - Max(High, 20d)) / ATR(14)
   - alpha_vwap (VWAP确认强度): 衡量价格相对于VWAP的偏离度，并由相对成交量(RelVol)加权。
     公式: (Close - VWAP) / (IntradayVol * Close) * ln(1 + RelVol)
   - alpha_trend (趋势强度): 基于EMA均线距离的趋势因子，仅在短均线(EMA5) > 长均线(EMA20)时激活。
     公式: (Close - EMA20) / ATR(14) * I(EMA5 > EMA20)

2. V7 新增增强因子 (V7 Enhancements):
   - rs_score (相对强度): 个股相对于大盘(SPY)的60日相对强弱。
   - beta (市场敏感度): 个股相对于大盘(SPY)的60日Beta系数。
   - atr_percent (波动率占比): ATR(14) / Price，用于衡量绝对波动水平。
   - return_5d (短期动量): 过去5日收益率。

3. 过滤条件 (Filters):
   - 基础过滤: ADV > 40M, Price > 10, Market Cap 2B-100B (在SQL选股阶段完成)。
   - 趋势过滤: Price > MA50。
   - 波动过滤: ATR% >= 3% (剔除低波动个股)。

4. 标签 (Label):
   - 目标变量: 未来5日收益率 (Forward 5d Return)。

5. 因子合成:
   - 采用 LightGBM LambdaRank 进行非线性合成排序，或者使用加权线性合成作为 Fallback。
"""
from __future__ import annotations

import polars as pl

from vnpy.alpha import AlphaDataset


class FlagshipAlphaMomentumV7Dataset(AlphaDataset):
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
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            process_type=process_type,
        )

        self.spy_symbol = spy_symbol
        eps: float = 1e-8

        # alpha_mom = (P_t - max(H_{t-20:t-1})) / ATR_14
        self.add_feature(
            "alpha_mom",
            f"(close - ts_max(ts_delay(high, 1), 20)) / (ta_atr(high, low, close, 14) + {eps})",
        )

        # alpha_vwap = (P_t - VWAP_t) / σ_intra * ln(1 + RelVol_t)
        has_turnover = "turnover" in df.columns
        if has_turnover:
            vwap_expr = f"(turnover / (volume + {eps}))"
        else:
            vwap_expr = f"((high + low + close) / 3)"

        atr_expr = f"ta_atr(high, low, close, 14)"
        sigma_intra_expr = f"{atr_expr} / (close + {eps})"
        rel_vol_expr = f"volume / (ts_mean(ts_delay(volume, 1), 20) + {eps})"
        vwap_divergence_expr = f"(close - {vwap_expr}) / ({sigma_intra_expr} * close + {eps})"

        self.add_feature(
            "alpha_vwap",
            f"{vwap_divergence_expr} * ts_log({rel_vol_expr} + 1)",
        )

        # ATR（用于策略和波动度特征）
        self.add_feature("atr_14", f"ta_atr(high, low, close, 14)")

        # 基础列用于过滤与后处理
        self.add_feature("close_price", "close")
        self.add_feature("ma50", f"ts_mean(close, 50)")
        self.add_feature("ret_10d", f"close / (ts_delay(close, 10) + {eps}) - 1")

        # ===== V7 标签与新增特征 =====
        # Label：5 日远期收益率（V7 标准）
        self.set_label("ts_delay(close, -5) / close - 1")

        v7_features = {
            "ret_5d": "ts_delay(close, -5) / close - 1",      # 未来 5 日收益 (用于分析)
            "return_5d": "close / ts_delay(close, 5) - 1",    # 过去 5 日历史收益 (Mom 5d 特征)
            "volume": "volume",                               # 成交量 (用于计算 ADV)
            "adv_usd": "volume",                              # 占位符，将在 post_process 中更新为真实 ADV
            "med_volume": "volume",                           # 占位符，将在 post_process 中更新为真实 MedVol
        }
        for name, expr in v7_features.items():
            self.add_feature(name, expr)

        # 只在 infer 阶段运行一次 V7 后处理，避免重复覆盖列
        self.add_processor("infer", self._post_process_v7)

    def _post_process_v7(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V7 后处理：增加 RS, Beta, ATR% 因子
        """
        if df.is_empty():
            return df

        eps: float = 1e-8

        # 1. 计算基础趋势特征 (EMA 等) 和 ADV
        df = df.sort(["vt_symbol", "datetime"])
        df = df.with_columns(
            [
                pl.col("close_price").ewm_mean(span=5, adjust=False, min_samples=1).over("vt_symbol").alias("ema5"),
                pl.col("close_price").ewm_mean(span=10, adjust=False, min_samples=1).over("vt_symbol").alias("ema10"),
                pl.col("close_price").ewm_mean(span=20, adjust=False, min_samples=1).over("vt_symbol").alias("ema20"),
                pl.col("close_price").ewm_mean(span=50, adjust=False, min_samples=1).over("vt_symbol").alias("ema50"),
            ]
        )

        # 计算 ADV (30d median volume * close) 用于仓位管理
        if "volume" in df.columns and "close_price" in df.columns:
            df = df.with_columns(
                [
                    pl.col("volume").rolling_median(window_size=30, min_periods=1).over("vt_symbol").alias("med_volume")
                ]
            )
            df = df.with_columns(
                [
                    (pl.col("med_volume") * pl.col("close_price")).alias("adv_usd")
                ]
            )

        # atr_percent: ATR / Price
        if "atr_14" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("atr_14") / (pl.col("close_price") + eps)).alias("atr_percent")
                ]
            )

        # 2. 计算相对强度 (RS) 和 Beta (需要 SPY 数据)
        spy_df = (
            df.filter(pl.col("vt_symbol") == self.spy_symbol)
            .select(["datetime", "close_price"])
            .rename({"close_price": "spy_close"})
            .sort("datetime")
        )

        if not spy_df.is_empty():
            df = df.join(spy_df, on="datetime", how="left")

            # label_excess_5d: forward 5d return minus SPY forward 5d return
            df = df.with_columns(
                [
                    (pl.col("spy_close").shift(-5).over("vt_symbol") / (pl.col("spy_close") + eps) - 1).alias(
                        "spy_ret_5d"
                    )
                ]
            )
            ret_base = "ret_5d" if "ret_5d" in df.columns else ("label" if "label" in df.columns else None)
            if ret_base is not None:
                df = df.with_columns([(pl.col(ret_base) - pl.col("spy_ret_5d")).alias("label_excess_5d")])

            # rs_score: (P / P_60) / (SPY / SPY_60) - 1
            df = df.with_columns(
                [
                    (
                        (pl.col("close_price") / (pl.col("close_price").shift(60).over("vt_symbol") + eps))
                        / (pl.col("spy_close") / (pl.col("spy_close").shift(60).over("vt_symbol") + eps) + eps)
                        - 1
                    ).alias("rs_score")
                ]
            ).with_columns(
                [
                    pl.col("rs_score").alias("rs_60d")  # Alias for compatibility with models trained with old names
                ]
            )

            # Beta 60d
            df = df.with_columns(
                [
                    pl.col("close_price").pct_change().over("vt_symbol").alias("ret_i"),
                    pl.col("spy_close").pct_change().over("vt_symbol").alias("ret_spy"),
                ]
            )

            df = df.with_columns(
                [
                    (
                        ((pl.col("ret_i") * pl.col("ret_spy")).rolling_mean(60)
                         - pl.col("ret_i").rolling_mean(60) * pl.col("ret_spy").rolling_mean(60))
                        / (pl.col("ret_spy").rolling_std(60).pow(2) + eps)
                    ).over("vt_symbol").alias("beta")
                ]
            )
        else:
            df = df.with_columns(
                [
                    pl.lit(0.0).alias("rs_score"),
                    pl.lit(1.0).alias("beta"),
                    pl.lit(None).cast(pl.Float64).alias("label_excess_5d"),
                ]
            )

        # 3. 计算核心趋势因子
        if "ema20" in df.columns and "atr_14" in df.columns:
            df = df.with_columns(
                [
                    ((pl.col("close_price") - pl.col("ema20")) / (pl.col("atr_14") + eps)).alias("ema_distance")
                ]
            )
            df = df.with_columns(
                [
                    (pl.col("ema_distance") * (pl.col("ema5") > pl.col("ema20")).cast(pl.Float64)).alias("alpha_trend")
                ]
            )

        # 4. 过滤（MA50 + ATR%）
        if "ma50" in df.columns:
            df = df.filter(pl.col("close_price") > pl.col("ma50"))

        if "atr_percent" in df.columns:
            df = df.filter(pl.col("atr_percent") >= 0.03)

        # 5. 因子合成与标准化
        core_features = ["alpha_mom", "alpha_vwap", "alpha_trend", "rs_score", "beta", "atr_percent", "return_5d"]
        for feat in core_features:
            if feat in df.columns:
                df = df.with_columns(
                    [
                        ((pl.col(feat) - pl.col(feat).mean().over("datetime"))
                         / (pl.col(feat).std().over("datetime") + eps)).alias(feat)
                    ]
                )

        # 6. 计算最终 Score (作为 fallback)
        df = df.with_columns(
            [
                (pl.col("alpha_mom") * 0.3
                 + pl.col("alpha_vwap") * 0.3
                 + pl.col("alpha_trend") * 0.2
                 + pl.col("rs_score") * 0.2).alias("score")
            ]
        )

        # 清理临时列
        drop_cols = [c for c in ["ret_i", "ret_spy", "spy_close", "spy_ret_5d", "ema_distance"] if c in df.columns]
        return df.drop(drop_cols)
