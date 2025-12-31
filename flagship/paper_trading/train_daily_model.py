"""
Script to retrain the LightGBM model daily for paper trading.
Uses the latest available data to train a fresh model.
"""
import sys
import argparse
import joblib
import polars as pl
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from flagship.paper_trading.config import (
    LAB_PATH, LIVE_MODEL_PATH, CURRENT_REGIME_ID
)
from flagship.factors.flagship_alpha_momentum_v7 import FlagshipAlphaMomentumV7Dataset
from flagship.factors.flagship_alpha_momentum_v5 import FlagshipAlphaMomentumV5Dataset
from flagship.scripts.pg_ticker_db import get_selected_symbols_in_range
from flagship.model.train_flagship_lgb import build_lgb_dataset, log_feature_importance

import lightgbm as lgb

def train_daily_model(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    output_model_path: Path = LIVE_MODEL_PATH,
    regime_id: int = CURRENT_REGIME_ID,
    strategy_version: str = "v7"
) -> None:
    """
    Retrain the model using data up to the target date.
    
    Args:
        target_date: The current trading date. Training will use data before this.
        lab_path: Path to AlphaLab.
        output_model_path: Path to save the new model.
        regime_id: ID of the regime (for potential parameter lookup).
        strategy_version: "v5" or "v7".
    """
    if target_date is None:
        target_date = date.today()
        
    logger.info(f"[train_daily_model] Starting daily model retraining for {target_date} (Strategy: {strategy_version})")
    
    # 1. Define Data Segments
    # Dynamic split:
    # Test: The future (not used for training, but dataset needs it defined) -> T to T+5
    # Valid: Recent history (e.g., last 2 months) -> T-60 to T-1
    # Train: History before valid with Embargo (Gap) -> T-(60+Gap+Window) to T-(60+Gap)
    
    test_start = target_date
    test_end = target_date + timedelta(days=5) # Dummy future
    
    valid_end = target_date - timedelta(days=1)
    valid_start = valid_end - timedelta(days=60)
    
    # Embargo/Purging: label is rank_5d (uses T to T+5 return).
    # To prevent leakage where Train label overlaps with Valid features/label,
    # we need a gap of at least 5 days. We use 7 days for safety.
    train_end = valid_start - timedelta(days=7)
    train_start = train_end - timedelta(days=730) # 2 years training window
    
    # Check if we have enough history
    # Simple check: train_start should be reasonable
    
    # Format periods
    train_period = (train_start.isoformat(), train_end.isoformat())
    valid_period = (valid_start.isoformat(), valid_end.isoformat())
    test_period = (test_start.isoformat(), test_end.isoformat())
    
    logger.info(f"[train_daily_model] Train Period: {train_period}")
    logger.info(f"[train_daily_model] Valid Period: {valid_period}")
    
    # 2. Load Data
    lab = AlphaLab(str(lab_path))
    # 仅训练“静态过滤后的 universe”：从 daily_selection 中取训练窗口内出现过的标的集合
    # NOTE: daily_selection table should ideally match the strategy's universe criteria.
    # If mixed (v5 and v7 in same table), we might get a union. 
    # For now, we rely on the DB containing selection relevant to the strategy we run, 
    # or just assume the model can handle a broader universe.
    vt_symbols = get_selected_symbols_in_range(start_date=train_start, end_date=valid_end)
    if not vt_symbols:
        raise RuntimeError(
            f"[train_daily_model] daily_selection 在 {train_start}~{valid_end} 没有任何选股记录，"
            f"无法构建静态 universe 进行训练。请先补齐该区间的 daily_selection。"
        )
    
    # 确保包含 SPY 用于计算 RS/Beta
    if "SPY.NASDAQ" not in vt_symbols:
        vt_symbols.append("SPY.NASDAQ")
        
    logger.info(f"[train_daily_model] Training universe symbols: {len(vt_symbols)}")
        
    logger.info("[train_daily_model] Loading raw data...")
    # 我们需要加载足够久的数据来计算 60d RS/Beta，即便在 train_start 的第一天也需要
    data_load_start = train_start - timedelta(days=120) 
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=data_load_start.isoformat(),
        end=valid_end.isoformat(), # Load up to validation end
        extended_days=0
    )
    
    if raw_df is None or raw_df.is_empty():
        raise RuntimeError("Failed to load bar data for training.")

    # 3. Compute Factors
    logger.info(f"[train_daily_model] Computing factors ({strategy_version})...")
    
    if strategy_version == "v5":
        DatasetClass = FlagshipAlphaMomentumV5Dataset
        # V5 Feature Columns
        feature_cols = ["alpha_mom", "alpha_vwap", "residual_vwap", "alpha_trend", "rs_10d"]
    else:
        DatasetClass = FlagshipAlphaMomentumV7Dataset
        # V7 Feature Columns
        feature_cols = ["alpha_mom", "alpha_vwap", "alpha_trend", "rs_60d", "beta", "atr_percent", "return_5d"]

    dataset = DatasetClass(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    # Apply filters if we have selection logic available here, 
    # but for model training, using a broader universe (all available) is often better for robustness.
    # We'll skip strict selection filtering for training to have more samples.
    dataset.prepare_data(filters=None)
    dataset.process_data()
    
    # 4. Train Model
    sample_train = dataset.fetch_learn(Segment.TRAIN)
    if sample_train.is_empty():
        raise RuntimeError("[train_daily_model] TRAIN segment is empty, cannot train.")

    # 训练标签：优先使用 rank_5d
    label_candidates = ["rank_5d", "label"]
    label_col = None
    for c in label_candidates:
        if c in sample_train.columns:
            label_col = c
            break
    
    missing_feats = [c for c in feature_cols if c not in sample_train.columns]
    if missing_feats:
        logger.warning(f"[train_daily_model] Missing some features: {missing_feats}. Using available columns.")
        feature_cols = [c for c in sample_train.columns if c in feature_cols]

    train_df = sample_train.sort(["datetime", "vt_symbol"])
    valid_df = dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"])
    if valid_df.is_empty():
        logger.warning("[train_daily_model] VALID segment is empty, using TRAIN as VALID for early stopping.")
        valid_df = train_df

    train_set = build_lgb_dataset(train_df, label_col, feature_cols)
    valid_set = build_lgb_dataset(valid_df, label_col, feature_cols)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "seed": 2024,
    }

    logger.info(f"[train_daily_model] Training LightGBM LambdaRank ({strategy_version}), label={label_col}, features={len(feature_cols)}")
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=800,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(50),
        ],
    )
    log_feature_importance(booster)
    
    # 5. Validation Check
    # train_lambdarank logs validation scores.
    # We can inspect booster.best_score if needed.
    best_score = booster.best_score['valid']['ndcg@5']
    logger.info(f"[train_daily_model] Best Validation NDCG@5: {best_score:.4f}")
    
    if best_score < 0.85: # Threshold from experience/backtest
        logger.warning(f"[train_daily_model] Validation score {best_score:.4f} is low. Model might be unstable.")
        # We could choose to abort saving here, but for now we proceed with warning.
    
    # 6. Save Model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, output_model_path)
    logger.info(f"[train_daily_model] Live model saved to: {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7", help="Strategy version")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    train_daily_model(target_date=target_dt, strategy_version=args.strategy)

