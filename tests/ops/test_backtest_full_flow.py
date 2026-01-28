from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import Segment
from vnpy.trader.constant import Interval

from flagship.backtest.flagship_alpha_momentum_backtest import run_backtest
from flagship.model.train_flagship_lgb import build_lgb_dataset


def _pick_label_column(df: pl.DataFrame) -> str | None:
    for candidate in ("rank_5d", "label_excess_5d", "label"):
        if candidate in df.columns:
            return candidate
    return None


def _sample_df(df: pl.DataFrame, *, max_dates: int = 3, max_symbols: int = 5) -> pl.DataFrame:
    if df.is_empty():
        return df
    dates = (
        df.select(pl.col("datetime").dt.date().unique().sort())
        .to_series()
        .to_list()
    )
    keep_dates = dates[:max_dates]
    out = df.filter(pl.col("datetime").dt.date().is_in(keep_dates))
    symbols = (
        out.select(pl.col("vt_symbol").unique().sort())
        .to_series()
        .to_list()
    )
    keep_symbols = symbols[:max_symbols]
    return out.filter(pl.col("vt_symbol").is_in(keep_symbols)).sort(["datetime", "vt_symbol"])


@pytest.mark.parametrize("strategy_version", ["v5", "v7"])
def test_backtest_full_flow_train_signal_backtest(strategy_version: str) -> None:
    # 测试场景：基于现有 lab parquet 做最小样本训练 -> 产信号 -> 回测
    # 输入：lab/flagship_alpha_momentum 下已存在的数据集 parquet
    # 期望：能生成信号并完成回测，返回 stats（包含 total_trades）
    lab_path = Path("lab/flagship_alpha_momentum")
    if not lab_path.exists():
        pytest.skip("lab/flagship_alpha_momentum 不存在，跳过回测全流程单测")

    lab = AlphaLab(str(lab_path))
    dataset = lab.load_dataset("flagship_alpha_momentum")
    if dataset is None:
        pytest.skip("flagship_alpha_momentum 数据集不存在，跳过回测全流程单测")

    train_df = _sample_df(dataset.fetch_learn(Segment.TRAIN))
    valid_df = _sample_df(dataset.fetch_learn(Segment.VALID))
    test_df = _sample_df(dataset.fetch_learn(Segment.TEST))

    if train_df.is_empty() or valid_df.is_empty() or test_df.is_empty():
        pytest.skip("训练/验证/测试数据为空，跳过回测全流程单测")

    label_col = _pick_label_column(train_df)
    if label_col is None:
        pytest.skip("找不到可用标签列，跳过回测全流程单测")

    feature_cols = [c for c in train_df.columns if c not in ("datetime", "vt_symbol", label_col, "label")]

    try:
        import lightgbm as lgb
    except Exception:
        pytest.skip("lightgbm 不可用，跳过回测全流程单测")

    train_set = build_lgb_dataset(train_df, label_col, feature_cols)
    valid_set = build_lgb_dataset(valid_df, label_col, feature_cols)

    booster = lgb.train(
        params={
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.1,
            "num_leaves": 16,
            "seed": 2024,
        },
        train_set=train_set,
        num_boost_round=30,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(5),
            lgb.log_evaluation(0),
        ],
    )

    infer_df = test_df.sort(["datetime", "vt_symbol"])
    infer_features = [c for c in infer_df.columns if c not in ("datetime", "vt_symbol", label_col, "label")]
    preds = booster.predict(infer_df.select(infer_features).to_pandas())
    signal_df = infer_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series(name="signal", values=preds)
    )

    signal_name = f"unit_test_backtest_{strategy_version}"
    lab.save_signal(signal_name, signal_df)

    start = infer_df.select(pl.col("datetime").min()).item()
    end = infer_df.select(pl.col("datetime").max()).item()
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)

    stats = run_backtest(
        lab_path=lab_path,
        start=start,
        end=end,
        interval=Interval.DAILY,
        signal_name=signal_name,
        use_postgres_selection=False,
        vt_symbols=sorted(signal_df.get_column("vt_symbol").unique().to_list()),
        strategy_version=strategy_version,
        dataset_name=None,
    )

    assert isinstance(stats, dict)
    assert "total_trades" in stats
