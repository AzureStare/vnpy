"""
旗艦 Alpha-Momentum LightGBM 訓練/信號導出腳本。

對標 `examples/alpha_research/research_workflow_lgb.ipynb` 的流程：
1. 從 AlphaLab 加載預先準備好的 AlphaDataset（由 prepare_alpha_momentum_dataset.py 生成）。
2. 使用 vn.py 內建的 LgbModel 完成訓練（含早停、驗證集監控）。
3. 可選地導出指定數據段（默認 TEST）的模型信號，寫入 AlphaLab.signal 供回測使用。
4. 將訓練好的模型通過 AlphaLab.save_model 保存，方便後續推理或再訓練。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl
import pandas as pd
import lightgbm as lgb

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import Segment, AlphaDataset
from vnpy.trader.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM model for Flagship Alpha-Momentum strategy."
    )
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab 數據根目錄（默認 lab/flagship_alpha_momentum）",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="flagship_alpha_momentum",
        help="AlphaLab.dataset 中的數據集名稱（需先由 prepare_alpha_momentum_dataset.py 生成）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="flagship_alpha_momentum_lgb",
        help="保存至 AlphaLab.model 的模型名稱（默認 flagship_alpha_momentum_lgb）",
    )
    parser.add_argument(
        "--signal-name",
        type=str,
        default="flagship_alpha_momentum",
        help="導出信號時使用的名稱（寫入 AlphaLab.signal/<name>.parquet）",
    )
    parser.add_argument(
        "--signal-segment",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="導出信號所使用的數據段（默認 test）",
    )
    parser.add_argument(
        "--no-signal",
        action="store_true",
        help="僅訓練並保存模型，不導出信號",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--num-boost-round", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--label-column",
        type=str,
        default="rank_5d",
        help="用作排序的標籤列（默認 rank_5d，若不存在將回退到 label_excess_5d 或 label）",
    )
    parser.add_argument(
        "--ndcg-at",
        type=int,
        nargs="+",
        default=[5, 10],
        help="NDCG 評估階梯（默認 5 10）",
    )
    return parser.parse_args()


def load_dataset(lab: AlphaLab, dataset_name: str) -> AlphaDataset:
    dataset = lab.load_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(
            f"Dataset '{dataset_name}' 不存在。請先運行 "
            f"'flagship/factors/prepare_alpha_momentum_dataset.py' 生成數據集。"
        )
    return dataset


def log_feature_importance(model: lgb.Booster, top_k: int = 20) -> None:
    feature_names = model.feature_name()
    if not feature_names:
        return

    gain_importance = model.feature_importance(importance_type="gain")
    split_importance = model.feature_importance(importance_type="split")

    def format_top(info: list[tuple[str, float]]) -> str:
        lines = [f"{name}: {value:.6f}" for name, value in info[:top_k]]
        return "\n    ".join(lines)

    gain_pairs = sorted(zip(feature_names, gain_importance), key=lambda x: x[1], reverse=True)
    split_pairs = sorted(zip(feature_names, split_importance), key=lambda x: x[1], reverse=True)

    gain_text = format_top(gain_pairs)
    split_text = format_top(split_pairs)
    logger.info(f"[train_flagship_lgb] Top feature importance (gain):\n    {gain_text}")
    logger.info(f"[train_flagship_lgb] Top feature importance (split):\n    {split_text}")


def build_lgb_dataset(df: pl.DataFrame, label_col: str, feature_cols: list[str]) -> lgb.Dataset:
    """
    构建 LightGBM Lambdarank 数据集。
    
    关键：
    1. 确保 group 数组的顺序与数据的行顺序完全一致。
    2. 标签值归一化到 0~30 范围，避免超过 LightGBM 的 label mappings 限制。
       使用线性缩放将原始排名 [0, group_size-1] 映射到 [0, 30]。
    """
    # 确保数据按 datetime 排序
    df_sorted = df.sort(["datetime", "vt_symbol"])
    
    # 使用 polars 计算每个组的排名（按 datetime 分组）
    # 关键：先按 datetime 和标签值排序，分配排名后不再排序，保持标签值的顺序
    # 这样既能确保 group 顺序一致，又能确保标签值顺序正确
    df_with_rank = df_sorted.sort(["datetime", label_col], descending=[False, True]).with_columns([
        # 使用 int_range 确保每个组内的排名是连续的整数（0, 1, 2, ...）
        (pl.int_range(pl.len()).over("datetime")).cast(pl.Int32).alias("rank_raw")
    ])
    # 注意：不再排序，保持分配排名后的顺序，这样 group 顺序和标签值顺序都正确
    
    # 按 datetime 分组，计算每组大小（使用 maintain_order=True 保持原始顺序）
    grouped = df_with_rank.group_by("datetime", maintain_order=True).agg([
        pl.len().alias("group_size"),
    ])
    
    # 提取 group 数组（按数据的实际顺序）
    groups = grouped["group_size"].to_list()
    
    # 将标签排名归一化到 0~30 的范围，避免超过 LightGBM 的 label mappings 限制
    # 使用线性缩放：label_normalized = int(30 * rank / (group_size - 1))
    MAX_LABEL_VALUE = 30  # LightGBM lambdarank 的标签值上限
    
    label_ranks = []
    idx = 0
    for group_idx, group_size in enumerate(groups):
        # 获取当前组的原始排名（0, 1, 2, ..., group_size-1）
        group_ranks_raw = df_with_rank["rank_raw"][idx:idx + group_size].to_list()
        
        # 归一化到 0~MAX_LABEL_VALUE 范围
        if group_size > 1:
            # 线性缩放：将 [0, group_size-1] 映射到 [0, MAX_LABEL_VALUE]
            group_labels = [
                int(MAX_LABEL_VALUE * rank / (group_size - 1))
                for rank in group_ranks_raw
            ]
        else:
            # 如果组大小只有 1，标签值就是 0
            group_labels = [0]
        
        # 确保标签值在 [0, MAX_LABEL_VALUE] 范围内
        group_labels = [min(max(label, 0), MAX_LABEL_VALUE) for label in group_labels]
        
        label_ranks.extend(group_labels)
        idx += group_size
    
    # 转换为 pandas（使用 df_with_rank 而不是 df_sorted，因为顺序已经改变了）
    df_pd = df_with_rank.select(feature_cols).to_pandas()
    labels = pd.Series(label_ranks, dtype=int)
    
    return lgb.Dataset(data=df_pd, label=labels, group=groups, free_raw_data=False)


def export_signal(
    lab: AlphaLab,
    dataset: AlphaDataset,
    booster: lgb.Booster,
    segment: Segment,
    signal_name: str,
    label_col: str,
) -> None:
    infer_df = dataset.fetch_infer(segment)
    if infer_df.is_empty():
        logger.warning("[train_flagship_lgb] %s 數據段為空，跳過信號導出", segment.name)
        return

    infer_df = infer_df.sort(["datetime", "vt_symbol"])
    feature_cols = [c for c in infer_df.columns if c not in ("datetime", "vt_symbol", label_col, "label")]
    preds = booster.predict(infer_df.select(feature_cols).to_pandas())
    
    # 选择要导出的列：包含所有因子列和特征列，用于回测和因子有效性评估
    # 基础列
    select_cols = ["datetime", "vt_symbol"]
    
    # 因子列（用于策略和因子有效性评估）
    factor_cols = [
        "alpha_mom", "alpha_vwap", "alpha_trend",
        "residual_vwap",
        "z_mom", "z_vwap_residual", "z_trend",
        "weight_mom", "weight_vwap", "weight_trend",
        "ir_mom", "ir_vwap", "ir_trend",
        "score",
    ]
    
    # 特征列
    feature_cols_to_export = [
        "ema5", "ema20", "ema50", "ema_distance",
        "atr_14", "close_price", "ma50", "ret_10d",
        "trend_state",
    ]
    
    # 标签列（用于因子有效性评估）
    label_cols_to_export = ["ret_5d", "rank_5d", "label"]
    
    # 添加所有存在的列
    all_cols_to_export = factor_cols + feature_cols_to_export + label_cols_to_export
    for col in all_cols_to_export:
        if col in infer_df.columns and col not in select_cols:
            select_cols.append(col)
    
    # 记录导出的列
    logger.info(f"[train_flagship_lgb] 信号导出包含 {len(select_cols)} 列: {select_cols[:10]}...")
    if "atr_14" in select_cols:
        logger.info("[train_flagship_lgb] ✓ atr_14 列已包含")
    else:
        logger.warning("[train_flagship_lgb] ✗ 数据集中没有 atr_14 列")
    
    signal_df = (
        infer_df.select(select_cols)
        .with_columns(pl.Series(name="signal", values=preds))
        .drop_nulls("signal")
    )
    lab.save_signal(signal_name, signal_df)
    logger.info(f"[train_flagship_lgb] 信號已導出：{signal_name}（{signal_df.height} 行）")


def main() -> None:
    args = parse_args()

    lab = AlphaLab(str(args.lab_path))
    dataset = load_dataset(lab, args.dataset_name)

    sample_train = dataset.fetch_learn(Segment.TRAIN)
    
    # 如果训练数据为空，使用 VALID 作为训练数据
    if sample_train.is_empty():
        logger.warning("[train_flagship_lgb] 訓練數據為空，使用 VALID 作為訓練數據")
        sample_train = dataset.fetch_learn(Segment.VALID)
        if sample_train.is_empty():
            raise RuntimeError("訓練數據和驗證數據都為空，無法訓練")
    
    # 尝试多个标签列：优先使用用户指定的列，如果不存在则尝试其他候选列
    label_candidates = [args.label_column, "label_excess_5d", "rank_5d", "label"]
    label_col = None
    for candidate in label_candidates:
        if candidate in sample_train.columns:
            label_col = candidate
            logger.info(f"[train_flagship_lgb] 使用標籤列: {label_col}")
            break
    
    if label_col is None:
        raise RuntimeError(f"找不到任何可用的標籤列，嘗試過的列: {label_candidates}")

    train_df = sample_train.sort(["datetime", "vt_symbol"])
    valid_df = dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"])
    
    # 如果验证数据为空，使用 TEST 作为验证数据
    if valid_df.is_empty():
        logger.warning("[train_flagship_lgb] 驗證數據為空，使用 TEST 作為驗證數據")
        valid_df = dataset.fetch_learn(Segment.TEST).sort(["datetime", "vt_symbol"])

    feature_cols = [c for c in train_df.columns if c not in ("datetime", "vt_symbol", label_col, "label")]
    train_set = build_lgb_dataset(train_df, label_col, feature_cols)
    valid_set = build_lgb_dataset(valid_df, label_col, feature_cols)
    
    # 记录数据集信息
    train_groups = train_set.get_group()
    valid_groups = valid_set.get_group()
    train_labels = train_set.get_label()
    valid_labels = valid_set.get_label()
    
    logger.info(
        f"[train_flagship_lgb] Train: {len(train_groups)} groups, {len(train_labels)} labels, "
        f"label range: [{min(train_labels)}, {max(train_labels)}]"
    )
    logger.info(
        f"[train_flagship_lgb] Valid: {len(valid_groups)} groups, {len(valid_labels)} labels, "
        f"label range: [{min(valid_labels)}, {max(valid_labels)}]"
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": args.ndcg_at,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "seed": args.seed,
    }

    logger.info(
        f"[train_flagship_lgb] 開始訓練 LightGBM Lambdarank，label={label_col}, ndcg_at={args.ndcg_at}"
    )
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ],
    )
    log_feature_importance(booster)

    lab.save_model(args.model_name, booster)
    logger.info(
        f"[train_flagship_lgb] 模型已保存：lab.model/{args.model_name}.pkl"
    )

    if not args.no_signal:
        segment = Segment[args.signal_segment.upper()]
        export_signal(lab, dataset, booster, segment, args.signal_name, label_col)


if __name__ == "__main__":
    main()

