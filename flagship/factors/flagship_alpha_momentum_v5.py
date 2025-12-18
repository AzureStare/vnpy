"""
Flagship Alpha-Momentum v5.0 因子计算模块。

基于策略文档v5.0实现：
- 因子A：波动率调整突破（ATR标准化）
- 因子B：VWAP确认强度
- 因子C：趋势强度（EMA多周期，v5.0新增）
- 三重过滤：MA50趋势 + EMA短期趋势 + 相对强度（RS vs SPY）
- 分层回归因子合成
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import polars as pl

from vnpy.alpha import AlphaDataset


class FlagshipAlphaMomentumV5Dataset(AlphaDataset):
    """
    Flagship Alpha-Momentum v5.0 数据集
    
    因子体系：
    - 因子A：波动率调整突破 alpha_mom = (P_t - max(H_{t-20:t-1})) / ATR_14
    - 因子B：VWAP确认强度 alpha_vwap = (P_t - VWAP_t) / σ_intra * ln(1 + RelVol_t)
    - 因子C：趋势强度 alpha_trend = (P_t - EMA_20) / ATR_14 * I[EMA_5 > EMA_20] (v5.0新增)
    
    过滤条件：
    - MA50趋势过滤：P_t > MA_50
    - EMA短期趋势过滤：P_t > EMA_5 且 EMA_5 > EMA_20 (v5.0新增)
    - 相对强度过滤：R_{t-10:t} > R_{SPY, t-10:t}
    
    因子合成：
    - 分层回归：以alpha_mom为主轴，对alpha_vwap回归取残差
    - 加权合成：Score = w1 * Z(alpha_mom) + w2 * Z(residual_vwap) + w3 * Z(alpha_trend)
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

        # ========== 因子A：波动率调整突破 ==========
        # alpha_mom = (P_t - max(H_{t-20:t-1})) / ATR_14
        self.add_feature(
            "alpha_mom",
            f"(close - ts_max(ts_delay(high, 1), 20)) / (ta_atr(high, low, close, 14) + {eps})",
        )

        # ========== 因子B：VWAP确认强度 ==========
        # alpha_vwap = (P_t - VWAP_t) / σ_intra * ln(1 + RelVol_t)
        # 将所有表达式内联到一个特征中，避免依赖其他因子（因为因子计算是并行的）
        
        # 检查是否有turnover列
        has_turnover = "turnover" in df.columns
        
        if has_turnover:
            # 使用真实的VWAP = turnover / volume
            vwap_expr = f"(turnover / (volume + {eps}))"
        else:
            # 使用典型价格作为VWAP的近似
            vwap_expr = f"((high + low + close) / 3)"
        
        # 内联所有计算
        atr_expr = f"ta_atr(high, low, close, 14)"
        sigma_intra_expr = f"{atr_expr} / (close + {eps})"
        rel_vol_expr = f"volume / (ts_mean(ts_delay(volume, 1), 20) + {eps})"
        vwap_divergence_expr = f"(close - {vwap_expr}) / ({sigma_intra_expr} * close + {eps})"
        
        self.add_feature(
            "alpha_vwap",
            f"{vwap_divergence_expr} * ts_log({rel_vol_expr} + 1)",
        )

        # ========== 因子C：趋势强度（v5.0新增）==========
        # alpha_trend = (P_t - EMA_20) / ATR_14 * I[EMA_5 > EMA_20]
        # EMA/ema_distance 将在后处理阶段用 polars ewm_mean 计算，
        # 避免依赖 vnpy 表达式层新增 ta_ema（遵守“不改 vnpy/ 框架库代码”的仓库约束）。
        
        # 日级ATR特征（用于策略中的止盈止损计算）
        self.add_feature("atr_14", f"ta_atr(high, low, close, 14)")
        
        # 注意：trend_state 和 alpha_trend 将在后处理阶段计算
        # 因为需要用到逻辑操作符（&），这在DataProxy表达式中不支持

        # ========== 过滤条件 ==========
        # 保留close列用于过滤（需要作为feature才能在后处理中使用）
        self.add_feature(
            "close_price",
            "close",
        )
        
        # MA50趋势过滤（在后处理阶段应用，这里只计算MA50）
        self.add_feature(
            "ma50",
            f"ts_mean(close, 50)",
        )
        
        # 相对强度过滤（需要SPY数据）
        # 先计算10日收益率
        self.add_feature(
            "ret_10d",
            f"close / (ts_delay(close, 10) + {eps}) - 1",
        )

        # Label：3 日前瞻收益率（匹配 2–5 天中频持仓）
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")

        # 在推断（infer）阶段做过滤、分层回归和综合 Score
        self.add_processor("infer", self._post_process)

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        后处理：三重过滤 + 分层回归 + 综合Score
        
        步骤：
        1. 应用三重过滤（MA50 + EMA短期趋势 + RS）
        2. 分层回归：alpha_vwap = β * alpha_mom + ε
        3. Winsorization + Z-Score
        4. 加权合成 Score（包含alpha_trend）
        """
        # 先检查基础因子是否存在
        if "alpha_mom" not in df.columns or "alpha_vwap" not in df.columns:
            return df

        eps: float = 1e-8

        # ========== 步骤0：计算趋势相关特征（v5.0新增）==========
        # 0.0 计算 EMA5/20/50 与 ema_distance（用 polars，避免依赖 vnpy 表达式函数）
        if "close_price" in df.columns:
            # 需要保证每个 vt_symbol 内按时间有序
            df = df.sort(["vt_symbol", "datetime"])

            if "ema5" not in df.columns:
                df = df.with_columns([
                    pl.col("close_price").ewm_mean(span=5, adjust=False, min_samples=1).over("vt_symbol").alias("ema5"),
                    pl.col("close_price").ewm_mean(span=20, adjust=False, min_samples=1).over("vt_symbol").alias("ema20"),
                    pl.col("close_price").ewm_mean(span=50, adjust=False, min_samples=1).over("vt_symbol").alias("ema50"),
                ])

            if "atr_14" in df.columns and "ema20" in df.columns and "ema_distance" not in df.columns:
                df = df.with_columns([
                    ((pl.col("close_price") - pl.col("ema20")) / (pl.col("atr_14") + eps)).alias("ema_distance")
                ])

        # 趋势状态：多头排列（EMA5 > EMA20 > EMA50）
        if "ema5" in df.columns and "ema20" in df.columns and "ema50" in df.columns:
            df = df.with_columns([
                (
                    (pl.col("ema5") > pl.col("ema20")) &
                    (pl.col("ema20") > pl.col("ema50"))
                ).cast(pl.Int32).alias("trend_state")
            ])
        
        # 趋势强度因子：只有当EMA5 > EMA20时，因子值才为正
        # alpha_trend = ema_distance * I[EMA5 > EMA20]
        if "ema_distance" in df.columns and "ema5" in df.columns and "ema20" in df.columns:
            df = df.with_columns([
                (
                    pl.col("ema_distance") *
                    (pl.col("ema5") > pl.col("ema20")).cast(pl.Float64)
                ).alias("alpha_trend")
            ])
        elif "ema20" in df.columns and "ema5" in df.columns and "close_price" in df.columns:
            # 如果没有ema_distance，使用close_price和ema20计算
            # 需要ATR，但这里简化处理：使用close_price的滚动标准差作为ATR的近似
            # 注意：这只是一个fallback，正常情况下应该使用ema_distance
            df = df.with_columns([
                (
                    (pl.col("close_price") - pl.col("ema20")) / 
                    (pl.col("close_price").std().over(["vt_symbol", "datetime"]) + eps) *
                    (pl.col("ema5") > pl.col("ema20")).cast(pl.Float64)
                ).alias("alpha_trend")
            ])
        
        # 如果alpha_trend仍未计算出来，设置为0
        if "alpha_trend" not in df.columns:
            df = df.with_columns([pl.lit(0.0).alias("alpha_trend")])

        # ========== 步骤1：三重过滤 ==========
        # MA50过滤
        if "ma50" in df.columns and "close_price" in df.columns:
            df = df.filter(pl.col("close_price") > pl.col("ma50"))
        
        # EMA短期趋势过滤（v5.0新增）
        if "ema5" in df.columns and "ema20" in df.columns and "close_price" in df.columns:
            df = df.filter(
                (pl.col("close_price") > pl.col("ema5")) &
                (pl.col("ema5") > pl.col("ema20"))
            )
        
        # 相对强度过滤（需要SPY数据）
        # 这里先跳过RS过滤，因为需要合并SPY数据
        # TODO: 在prepare_data阶段合并SPY数据并计算RS
        
        if df.is_empty():
            return df

        # ========== 步骤2：分层回归 ==========
        # 对每个日期，将alpha_vwap对alpha_mom回归，取残差
        # alpha_vwap = β * alpha_mom + ε
        # residual_vwap = ε
        
        # 计算回归系数β（截面回归）
        df = df.with_columns([
            # 计算协方差和方差
            (pl.col("alpha_mom") * pl.col("alpha_vwap")).mean().over("datetime").alias("cov_mom_vwap"),
            (pl.col("alpha_mom") * pl.col("alpha_mom")).mean().over("datetime").alias("var_mom"),
        ])
        
        # β = cov(alpha_mom, alpha_vwap) / var(alpha_mom)
        df = df.with_columns([
            (pl.col("cov_mom_vwap") / (pl.col("var_mom") + eps)).alias("beta")
        ])
        
        # 残差 = alpha_vwap - β * alpha_mom
        df = df.with_columns([
            (pl.col("alpha_vwap") - pl.col("beta") * pl.col("alpha_mom")).alias("residual_vwap")
        ])

        # ========== 步骤3：Winsorization + Z-Score ==========
        # 对alpha_mom、residual_vwap和alpha_trend做截面Winsorization
        quantile_exprs: list[pl.Expr] = []
        processed_factors = ["alpha_mom", "residual_vwap", "alpha_trend"]
        
        for col in processed_factors:
            quantile_exprs.extend([
                pl.col(col).quantile(0.01).over("datetime").alias(f"{col}_q01"),
                pl.col(col).quantile(0.99).over("datetime").alias(f"{col}_q99"),
            ])

        df_q = df.with_columns(quantile_exprs)

        clipped_exprs: list[pl.Expr] = []
        for col in processed_factors:
            clipped_exprs.append(
                pl.col(col).clip(pl.col(f"{col}_q01"), pl.col(f"{col}_q99")).alias(col)
            )

        df_w = df_q.with_columns(clipped_exprs)

        # 截面Z-Score
        z_exprs: list[pl.Expr] = [
            (
                (pl.col("alpha_mom") - pl.col("alpha_mom").mean().over("datetime"))
                / (pl.col("alpha_mom").std().over("datetime") + eps)
            ).alias("z_mom"),
            (
                (pl.col("residual_vwap") - pl.col("residual_vwap").mean().over("datetime"))
                / (pl.col("residual_vwap").std().over("datetime") + eps)
            ).alias("z_vwap_residual"),
            (
                (pl.col("alpha_trend") - pl.col("alpha_trend").mean().over("datetime"))
                / (pl.col("alpha_trend").std().over("datetime") + eps)
            ).alias("z_trend"),
        ]

        df_z = df_w.with_columns(z_exprs)

        # ========== 步骤4：基于滚动 IC-IR 的动态权重 ==========
        # 若缺少label（无法计算IC），回退静态权重
        use_dynamic = "label" in df_z.columns
        if not use_dynamic:
            raise ValueError("动态权重需要label列以计算IC/IR，当前数据缺少label，停止执行")
        dynamic_weights = None

        def _ewm_stats(values: Iterable[float], alpha: float) -> Tuple[list[float], list[float]]:
            """返回 (ewm_mean, ewm_std) 列表"""
            means: list[float] = []
            vars_: list[float] = []
            initialized = False
            mean_val = 0.0
            var_val = 0.0
            for x in values:
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    means.append(None)
                    vars_.append(None)
                    continue
                if not initialized:
                    mean_val = float(x)
                    var_val = 0.0
                    initialized = True
                else:
                    diff = float(x) - mean_val
                    mean_val = mean_val + alpha * diff
                    var_val = (1 - alpha) * (var_val + alpha * diff * diff)
                means.append(mean_val)
                vars_.append(var_val)
            stds = [math.sqrt(v) if v is not None else None for v in vars_]
            return means, stds

        if use_dynamic:
            # 1) 计算每日截面 IC（包含alpha_trend）
            # 过滤掉label、z_mom、z_vwap_residual、z_trend为NaN/Inf的行，避免IC计算失败
            df_z_for_ic = df_z.filter(
                pl.col("label").is_finite()
                & pl.col("z_mom").is_finite()
                & pl.col("z_vwap_residual").is_finite()
                & pl.col("z_trend").is_finite()
            )
            ic_daily = (
                df_z_for_ic
                .group_by("datetime")
                .agg([
                    pl.corr("z_mom", "label").alias("ic_mom"),
                    pl.corr("z_vwap_residual", "label").alias("ic_vwap"),
                    pl.corr("z_trend", "label").alias("ic_trend"),
                ])
                .sort("datetime")
            )

            if not ic_daily.is_empty():
                half_life = 20  # 可调参数：指数半衰期（日）
                alpha_decay = 1 - math.exp(-math.log(2) / half_life)

                ic_mom_list = ic_daily["ic_mom"].to_list()
                ic_vwap_list = ic_daily["ic_vwap"].to_list()
                ic_trend_list = ic_daily["ic_trend"].to_list()

                mom_mean, mom_std = _ewm_stats(ic_mom_list, alpha_decay)
                vwap_mean, vwap_std = _ewm_stats(ic_vwap_list, alpha_decay)
                trend_mean, trend_std = _ewm_stats(ic_trend_list, alpha_decay)

                ir_mom: list[float] = []
                ir_vwap: list[float] = []
                ir_trend: list[float] = []
                for m, s in zip(mom_mean, mom_std):
                    if m is None or s is None:
                        ir_mom.append(None)
                    else:
                        ir_mom.append(m / (s + eps))
                for m, s in zip(vwap_mean, vwap_std):
                    if m is None or s is None:
                        ir_vwap.append(None)
                    else:
                        ir_vwap.append(m / (s + eps))
                for m, s in zip(trend_mean, trend_std):
                    if m is None or s is None:
                        ir_trend.append(None)
                    else:
                        ir_trend.append(m / (s + eps))

                # 2) IR → 权重（非负、cap、归一化；三因子同时失效则权重全零）
                cap = 0.8  # 单因子上限

                weights_mom: list[float] = []
                weights_vwap: list[float] = []
                weights_trend: list[float] = []
                for irm, irv, irt in zip(ir_mom, ir_vwap, ir_trend):
                    raw_m = max(0.0, irm) if irm is not None else 0.0
                    raw_v = max(0.0, irv) if irv is not None else 0.0
                    raw_t = max(0.0, irt) if irt is not None else 0.0
                    total_raw = raw_m + raw_v + raw_t
                    if total_raw <= 0:
                        weights_mom.append(0.0)
                        weights_vwap.append(0.0)
                        weights_trend.append(0.0)
                        continue
                    norm_m = raw_m / total_raw
                    norm_v = raw_v / total_raw
                    norm_t = raw_t / total_raw
                    capped_m = min(norm_m, cap)
                    capped_v = min(norm_v, cap)
                    capped_t = min(norm_t, cap)
                    total_cap = capped_m + capped_v + capped_t
                    if total_cap <= 0:
                        weights_mom.append(0.0)
                        weights_vwap.append(0.0)
                        weights_trend.append(0.0)
                    else:
                        weights_mom.append(capped_m / total_cap)
                        weights_vwap.append(capped_v / total_cap)
                        weights_trend.append(capped_t / total_cap)

                dynamic_weights = pl.DataFrame(
                    {
                        "datetime": ic_daily["datetime"],
                        "weight_mom": weights_mom,
                        "weight_vwap": weights_vwap,
                        "weight_trend": weights_trend,
                        "ir_mom": ir_mom,
                        "ir_vwap": ir_vwap,
                        "ir_trend": ir_trend,
                    }
                )

        if dynamic_weights is None or dynamic_weights.is_empty():
            raise ValueError("动态权重计算失败：IC/IR 结果为空，请检查数据/label 可用性")

        # Join weights by datetime, then forward-fill for dates where label is unavailable (e.g. live inference last N days)
        df_z = df_z.join(dynamic_weights, on="datetime", how="left").sort(["datetime", "vt_symbol"])
        df_z = df_z.with_columns(
            [
                # forward fill then fallback to 0
                pl.col("weight_mom").fill_null(strategy="forward").fill_null(0.0).alias("weight_mom"),
                pl.col("weight_vwap").fill_null(strategy="forward").fill_null(0.0).alias("weight_vwap"),
                pl.col("weight_trend").fill_null(strategy="forward").fill_null(0.0).alias("weight_trend"),
                # IR fields are optional for reporting; forward-fill only
                pl.col("ir_mom").fill_null(strategy="forward").alias("ir_mom"),
                pl.col("ir_vwap").fill_null(strategy="forward").alias("ir_vwap"),
                pl.col("ir_trend").fill_null(strategy="forward").alias("ir_trend"),
            ]
        )
        df_z = df_z.with_columns(
            (
                pl.col("weight_mom") * pl.col("z_mom")
                + pl.col("weight_vwap") * pl.col("z_vwap_residual")
                + pl.col("weight_trend") * pl.col("z_trend")
            ).alias("score")
        )

        # 清理中间列
        drop_cols = (
            [f"{c}_q01" for c in processed_factors]
            + [f"{c}_q99" for c in processed_factors]
            + ["cov_mom_vwap", "var_mom", "beta"]
        )
        return df_z.drop(drop_cols)

