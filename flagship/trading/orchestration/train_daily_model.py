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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from flagship.trading.config import (
    LAB_PATH, LIVE_MODEL_PATH, LIVE_LR_MODEL_PATH
)
from flagship.factors.flagship_alpha_momentum_v7 import FlagshipAlphaMomentumV7Dataset
from flagship.factors.flagship_alpha_momentum_v5 import FlagshipAlphaMomentumV5Dataset
from flagship.universe.pg_ticker_db import get_selected_symbols_in_range
from flagship.model.train_flagship_lgb import build_lgb_dataset, log_feature_importance

import lightgbm as lgb

def train_daily_model(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    output_model_path: Path = LIVE_MODEL_PATH,
    output_lr_model_path: Path = LIVE_LR_MODEL_PATH,
    strategy_version: str = "v7"
) -> None:
    """
    Retrain the model using data up to the target date.
    
    Args:
        target_date: The current trading date. Training will use data before this.
        lab_path: Path to AlphaLab.
        output_model_path: Path to save the new model.
        output_lr_model_path: Path to save the Logistic Regression meta model (v7 only).
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
    train_start = train_end - timedelta(days=1095) # 3 years training window (Increased from 2y)
    
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
    
    # Save a history copy for auditing
    history_dir = output_model_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"booster_{target_date.strftime('%Y%m%d')}_{strategy_version}.joblib"
    joblib.dump(booster, history_path)
    logger.info(f"[train_daily_model] History model saved to: {history_path}")

    # 6.5 Train Logistic Regression meta model (v7 only): p_up = P(label_excess_5d > 0)
    lr_info: dict[str, object] | None = None
    if strategy_version == "v7":
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
            import numpy as np  # type: ignore

            if "label_excess_5d" not in valid_df.columns:
                raise RuntimeError("valid_df missing label_excess_5d (check v7 dataset post_process)")

            lgb_feature_cols = list(booster.feature_name() or [])
            if not lgb_feature_cols:
                lgb_feature_cols = list(feature_cols)

            missing_for_pred = [c for c in lgb_feature_cols if c not in valid_df.columns]
            if missing_for_pred:
                raise RuntimeError(f"valid_df missing lgb features: {missing_for_pred[:10]}")

            valid_pd = valid_df.select(["vt_symbol", "label_excess_5d", *lgb_feature_cols]).to_pandas()
            X_lgb = valid_pd[lgb_feature_cols]
            lgb_signal = booster.predict(X_lgb)
            valid_pd["lgb_signal"] = lgb_signal

            # Filter: exclude SPY and rows without labels
            valid_pd = valid_pd[valid_pd["vt_symbol"] != "SPY.NASDAQ"]
            valid_pd = valid_pd.replace([np.inf, -np.inf], np.nan)

            lr_feature_cols = ["lgb_signal", *lgb_feature_cols]
            valid_pd = valid_pd.dropna(subset=["label_excess_5d", *lr_feature_cols])
            if valid_pd.empty:
                raise RuntimeError("no valid rows for LR training after dropna")

            y = (valid_pd["label_excess_5d"].astype(float) > 0.0).astype(int).to_numpy()
            if int(np.min(y)) == int(np.max(y)):
                raise RuntimeError("LR training requires both classes; got a single class in y")

            X = valid_pd[lr_feature_cols].to_numpy(dtype=float)

            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
                ]
            )
            pipe.fit(X, y)

            # Save artifact
            output_lr_model_path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {
                "pipeline": pipe,
                "feature_cols": lr_feature_cols,
                "label": "label_excess_5d>0",
                "strategy_version": strategy_version,
                "trained_at": datetime.now().isoformat(),
                "n_samples": int(len(valid_pd)),
                "pos_rate": float(np.mean(y)),
            }
            joblib.dump(artifact, output_lr_model_path)
            logger.info(f"[train_daily_model] LR model saved to: {output_lr_model_path}")
            lr_info = artifact
        except Exception as exc:
            logger.warning(f"[train_daily_model] LR meta model skipped/failed: {exc}")
    
    # Save training metrics for app console
    metrics = {
        "target_date": target_date.isoformat(),
        "strategy_version": strategy_version,
        "best_ndcg_5": float(best_score),
        "train_period": train_period,
        "valid_period": valid_period,
        "feature_importance": dict(zip(feature_cols, booster.feature_importance().tolist())),
        "lr_meta_model": lr_info,
        "generated_at": datetime.now().isoformat()
    }
    metrics_path = PROJECT_ROOT / "logs" / "app" / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    logger.info(f"[train_daily_model] Training metrics saved to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7", help="Strategy version")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    train_daily_model(target_date=target_dt, strategy_version=args.strategy)

