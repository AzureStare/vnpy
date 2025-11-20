from __future__ import annotations

import polars as pl

from vnpy.alpha import AlphaDataset


class UsMidfreqHighReturnDataset(AlphaDataset):
    """
    日度因子数据集：Flagship Capital Alpha-Momentum（美股中频高收益版）

    - 因子 A：波动率调整突破强度 alpha_mom
    - 因子 B：相对异常成交量 alpha_vol
    - 因子 C：聪明资金流向代理 alpha_flow
    - 截面 Winsorization(1%, 99%) + Z-Score
    - 综合得分：Score_t = 0.4*Z_mom + 0.4*Z_flow + 0.2*Z_vol
    """

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        process_type: str = "append",
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            process_type=process_type,
        )

        eps: float = 1e-8

        # 因子 A：波动率调整突破强度
        # alpha_mom = (P_t - max(H_{t-20:t-1})) / ATR_14
        # 这里通过 ts_delay(high, 1) 实现 “过去 20 日不含当日”的窗口
        self.add_feature(
            "alpha_mom",
            f"(close - ts_max(ts_delay(high, 1), 20)) / (ta_atr(high, low, close, 14) + {eps})",
        )

        # 因子 B：相对异常成交量
        # alpha_vol = V_t / mean(V_{t-20:t-1})
        self.add_feature(
            "alpha_vol",
            f"volume / (ts_mean(ts_delay(volume, 1), 20) + {eps})",
        )

        # 因子 C：聪明资金流向代理
        # alpha_flow = CLV_t * ln(1 + alpha_vol_t)
        # 其中 CLV_t = (P_t - L_t) / (H_t - L_t)
        self.add_feature(
            "alpha_flow",
            f"((close - low) / (high - low + {eps}))"
            f" * ts_log(volume / (ts_mean(ts_delay(volume, 1), 20) + {eps}) + 1)",
        )

        # Label：3 日前瞻收益率（与 Alpha158 示例保持一致，匹配 2–5 天中频持仓）
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")

        # 在推断（infer）阶段做截面 Winsorization + Z-Score + 综合 Score
        self.add_processor("infer", self._post_process)

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        """对因子做截面 Winsorize + Z-Score，并生成综合 Score"""
        factor_cols = ["alpha_mom", "alpha_vol", "alpha_flow"]
        missing = [c for c in factor_cols if c not in df.columns]
        if missing:
            # 因子列缺失时直接返回原始数据，避免中断流水线
            return df

        eps: float = 1e-8

        # 1) 按日期对每个因子做截面分位数裁剪（Winsorization 1% / 99%）
        quantile_exprs: list[pl.Expr] = []
        for col in factor_cols:
            quantile_exprs.append(
                pl.col(col)
                .quantile(0.01)
                .over("datetime")
                .alias(f"{col}_q01")
            )
            quantile_exprs.append(
                pl.col(col)
                .quantile(0.99)
                .over("datetime")
                .alias(f"{col}_q99")
            )

        df_q = df.with_columns(quantile_exprs)

        clipped_exprs: list[pl.Expr] = []
        for col in factor_cols:
            clipped_exprs.append(
                pl.col(col)
                .clip(pl.col(f"{col}_q01"), pl.col(f"{col}_q99"))
                .alias(col)
            )

        df_w = df_q.with_columns(clipped_exprs)

        # 2) 按日期做截面 Z-Score
        z_exprs: list[pl.Expr] = [
            (
                (pl.col("alpha_mom") - pl.col("alpha_mom").mean().over("datetime"))
                / (pl.col("alpha_mom").std().over("datetime") + eps)
            ).alias("z_mom"),
            (
                (pl.col("alpha_vol") - pl.col("alpha_vol").mean().over("datetime"))
                / (pl.col("alpha_vol").std().over("datetime") + eps)
            ).alias("z_vol"),
            (
                (pl.col("alpha_flow") - pl.col("alpha_flow").mean().over("datetime"))
                / (pl.col("alpha_flow").std().over("datetime") + eps)
            ).alias("z_flow"),
        ]

        df_z = df_w.with_columns(z_exprs)

        # 3) 线性加权综合得分 Score_t
        df_z = df_z.with_columns(
            (
                0.4 * pl.col("z_mom")
                + 0.4 * pl.col("z_flow")
                + 0.2 * pl.col("z_vol")
            ).alias("score")
        )

        # 4) 清理中间量
        drop_cols = [f"{c}_q01" for c in factor_cols] + [f"{c}_q99" for c in factor_cols]
        return df_z.drop(drop_cols)


