"""
Flagship Alpha-Momentum V7 Backtest Script (Rolling Walk-Forward)
================================================================

This script implements a Rolling Walk-Forward Validation backtest for the V7 strategy.
Instead of using fixed regimes, it simulates a realistic periodic retraining process.

Backtest Logic:
1.  **Initial Train**: Train on [Start - 2 Years, Start - Gap].
2.  **Validation**: Used for early stopping within the training window.
3.  **Test (Simulation)**: Run strategy on [Start, Start + Step].
4.  **Roll**: Move Start forward by Step, Retrain, Repeat.

Config:
- Step Size: 30 Days (Retrain monthly).
- Train Window: 730 Days (2 Years).
- Embargo: 7 Days.
- Strategy: V7 Aggressive.
"""

import sys
import argparse
import joblib
import polars as pl
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from flagship.factors.flagship_alpha_momentum_v7 import FlagshipAlphaMomentumV7Dataset
from flagship.model.train_flagship_lgb import build_lgb_dataset
import lightgbm as lgb
from flagship.universe.pg_ticker_db import get_selected_symbols_in_range
from flagship.backtest.flagship_alpha_momentum_backtest import run_backtest

# Backtest Configuration
LAB_PATH = Path("lab/flagship_alpha_momentum")
MODEL_DIR = Path("flagship/models/backtest_v7_rolling")
SIGNAL_DIR = Path("flagship/signals/backtest_v7_rolling")
STEP_DAYS = 30 # Retrain every 30 days
TRAIN_WINDOW = 1095 # 3 Years
EMBARGO_DAYS = 7
BACKTEST_UNIVERSE_TOP_N = 8  # Limit minute-bar backtest universe to daily Top-N union for performance

def train_model_for_window(
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    lab: AlphaLab,
    model_path: Path
) -> lgb.Booster:
    """Train LightGBM model for a specific window."""
    
    logger.info(f"Training Model: Train[{train_start} ~ {train_end}] Valid[{valid_start} ~ {valid_end}]")
    
    # Load Data (Universe from daily_selection)
    # Note: For backtest speed, we might want to cache this or load in chunks, 
    # but loading specific range is safer for memory.
    vt_symbols = get_selected_symbols_in_range(start_date=train_start, end_date=valid_end)
    if "SPY.NASDAQ" not in vt_symbols:
        vt_symbols.append("SPY.NASDAQ")
        
    if not vt_symbols:
        logger.warning("No symbols found for training window.")
        return None

    # Load Bars
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=(train_start - timedelta(days=120)).isoformat(), # Buffer for factors
        end=valid_end.isoformat(),
        extended_days=0
    )
    
    if raw_df is None or raw_df.is_empty():
        return None
        
    # Prepare Dataset
    # We define a dummy TEST period because Dataset init requires it
    dataset = FlagshipAlphaMomentumV7Dataset(
        df=raw_df,
        train_period=(train_start.isoformat(), train_end.isoformat()),
        valid_period=(valid_start.isoformat(), valid_end.isoformat()),
        test_period=(valid_end.isoformat(), valid_end.isoformat()) # Dummy
    )
    
    dataset.prepare_data(filters=None)
    dataset.process_data()
    
    sample_train = dataset.fetch_learn(Segment.TRAIN)
    sample_valid = dataset.fetch_learn(Segment.VALID)
    
    if sample_train.is_empty() or sample_valid.is_empty():
        logger.warning("Empty train or valid set.")
        return None

    # Features & Label
    label_col = "rank_5d" if "rank_5d" in sample_train.columns else "label"
    feature_cols = ["alpha_mom", "alpha_vwap", "alpha_trend", "rs_score", "beta", "atr_percent", "return_5d"]
    # Filter available features
    feature_cols = [c for c in feature_cols if c in sample_train.columns]
    
    train_set = build_lgb_dataset(sample_train, label_col, feature_cols)
    valid_set = build_lgb_dataset(sample_valid, label_col, feature_cols)
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "seed": 2024,
        "verbose": -1
    }
    
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=500,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)] # Quiet training
    )
    
    # Save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, model_path)
    return booster

def generate_signals_for_window(
    test_start: date,
    test_end: date,
    booster: lgb.Booster,
    lab: AlphaLab,
    output_dir: Path
):
    """Generate signals for the test window using the trained model."""
    logger.info(f"Generating Signals: {test_start} ~ {test_end}")
    
    vt_symbols = get_selected_symbols_in_range(start_date=test_start, end_date=test_end)
    if "SPY.NASDAQ" not in vt_symbols:
        vt_symbols.append("SPY.NASDAQ")
        
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=(test_start - timedelta(days=120)).isoformat(),
        end=test_end.isoformat(),
        extended_days=0
    )
    
    if raw_df is None or raw_df.is_empty():
        return

    dataset = FlagshipAlphaMomentumV7Dataset(
        df=raw_df,
        train_period=(test_start.isoformat(), test_start.isoformat()), # Dummy
        valid_period=(test_start.isoformat(), test_start.isoformat()), # Dummy
        test_period=(test_start.isoformat(), test_end.isoformat())
    )
    dataset.prepare_data(filters=None)
    dataset.process_data()
    
    infer_df = dataset.fetch_infer(Segment.TEST)
    if infer_df.is_empty():
        return
        
    feature_cols = booster.feature_name()
    # Check missing
    missing = [c for c in feature_cols if c not in infer_df.columns]
    if missing:
        logger.warning(f"Missing features in inference: {missing}")
        return

    X = infer_df.select(feature_cols).to_pandas()
    scores = booster.predict(X)
    
    # Export
    # Include cols needed for strategy
    # Ensure adv_usd exists, if not fallback to 0 (though factor script should provide it)
    if "adv_usd" not in infer_df.columns:
        logger.warning("adv_usd missing in infer_df, filling with 0")
        infer_df = infer_df.with_columns(pl.lit(0.0).alias("adv_usd"))
        
    cols_to_save = ["datetime", "vt_symbol", "atr_14", "close_price", "adv_usd"]
    opt_cols = ["ema5", "ema10", "ema20", "ema50", "atr_percent", "alpha_mom", "alpha_vwap", "alpha_trend", "rs_score", "return_5d"]
    for c in opt_cols:
        if c in infer_df.columns:
            cols_to_save.append(c)
            
    signal_df = infer_df.select(cols_to_save).with_columns(pl.Series("signal", scores))
    
    # Save per day or one big file? 
    # For backtest strategy engine, one big file is easier, or partitioned.
    # We will save one file per window, then maybe merge later or strategy loads them all.
    # Strategy usually loads one big parquet or daily files. 
    # Let's save one file per window.
    out_file = output_dir / f"signal_{test_start}_{test_end}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(out_file)
    logger.info(f"Saved signals to {out_file}")


def run_rolling_backtest(start_date: str, end_date: str):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    lab = AlphaLab(str(LAB_PATH))
    current_dt = start_dt
    
    all_signal_files = []
    
    while current_dt < end_dt:
        # Define Windows
        # Test Window: [current, current + step)
        window_test_end = min(current_dt + timedelta(days=STEP_DAYS - 1), end_dt)
        
        # Validation Window: [current - 60, current - 1]
        valid_end = current_dt - timedelta(days=1)
        valid_start = valid_end - timedelta(days=60)
        
        # Train Window: [valid_start - gap - train_window, valid_start - gap]
        train_end = valid_start - timedelta(days=EMBARGO_DAYS)
        train_start = train_end - timedelta(days=TRAIN_WINDOW)
        
        # Check data availability
        # If train_start is too early (before 2016?), might need to skip or shrink
        # Assuming we have data.
        
        # 1. Train Model
        model_name = f"model_{current_dt}.pkl"
        model_path = MODEL_DIR / model_name
        
        if not model_path.exists():
            booster = train_model_for_window(train_start, train_end, valid_start, valid_end, lab, model_path)
        else:
            logger.info(f"Loading cached model: {model_path}")
            booster = joblib.load(model_path)
            
        if booster:
            # 2. Generate Signals for Test Window
            generate_signals_for_window(current_dt, window_test_end, booster, lab, SIGNAL_DIR)
        
        # Move to next window
        current_dt += timedelta(days=STEP_DAYS)
        
    logger.info("Rolling Backtest Signal Generation Complete.")
    
    # Merge all signals
    logger.info("Merging all signals...")
    signal_files = sorted(SIGNAL_DIR.glob("signal_*.parquet"))
    if not signal_files:
        logger.error("No signal files generated.")
        return

    # NOTE:
    # We may have schema drift across historical runs (e.g. rs_60d renamed to rs_score).
    # Normalize columns before concatenation to avoid polars ShapeError.
    dfs: list[pl.DataFrame] = []
    all_cols: set[str] = set()

    for p in signal_files:
        df = pl.read_parquet(p)

        # Normalize RS column naming
        if "rs_60d" in df.columns and "rs_score" not in df.columns:
            df = df.rename({"rs_60d": "rs_score"})
        if "rs_score" in df.columns and "rs_60d" not in df.columns:
            df = df.with_columns(pl.col("rs_score").alias("rs_60d"))

        all_cols.update(df.columns)
        dfs.append(df)

    cols_order: list[str] = ["datetime", "vt_symbol"] + [
        c for c in sorted(all_cols) if c not in ("datetime", "vt_symbol")
    ]

    aligned: list[pl.DataFrame] = []
    for df in dfs:
        missing = [c for c in cols_order if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).alias(c) for c in missing])
        aligned.append(df.select(cols_order))

    full_df = pl.concat(aligned)
        
    full_df = full_df.sort(["datetime", "vt_symbol"]).unique(subset=["datetime", "vt_symbol"], keep="last")
    
    merged_path = SIGNAL_DIR / "v7_rolling_backtest_signals.parquet"
    full_df.write_parquet(merged_path)
    logger.info(f"Merged signals saved to {merged_path}")

    # Also save to AlphaLab signal store so run_backtest can load it by name
    try:
        lab.save_signal("v7_rolling_backtest_signals", full_df)
        logger.info("Saved merged signals to AlphaLab: v7_rolling_backtest_signals")
    except Exception as exc:
        logger.warning(f"Failed to save merged signals to AlphaLab: {exc}")

    # Reduce minute-level backtest universe:
    # BacktestingEngine loads ALL minute bars for vt_symbols upfront (very expensive).
    # So we only keep the union of daily Top-N symbols across the backtest period.
    backtest_vt_symbols: list[str] | None = None
    try:
        top_n = int(BACKTEST_UNIVERSE_TOP_N)
        universe_df = (
            full_df
            .with_columns(pl.col("datetime").dt.date().alias("_trade_date"))
            .filter(~pl.col("vt_symbol").is_in(["SPY.NASDAQ"]))
            .sort(["_trade_date", "signal"], descending=[False, True])
            .group_by("_trade_date", maintain_order=True)
            .agg(pl.col("vt_symbol").head(top_n).alias("_top_symbols"))
            .explode("_top_symbols")
            .select(pl.col("_top_symbols").alias("vt_symbol"))
            .unique()
            .sort("vt_symbol")
        )
        backtest_vt_symbols = universe_df["vt_symbol"].to_list()
        for s in ["SPY.NASDAQ", "VIX.CBOE", "VIX3M.CBOE"]:
            if s not in backtest_vt_symbols:
                backtest_vt_symbols.append(s)
        backtest_vt_symbols = sorted(set(backtest_vt_symbols))
        logger.info(
            f"Minute backtest universe reduced: top_n={top_n}, vt_symbols={len(backtest_vt_symbols)}"
        )
    except Exception as exc:
        logger.warning(f"Failed to build reduced minute backtest universe, will use full signal universe: {exc}")
        backtest_vt_symbols = None
    
    # Run Strategy Backtest Engine automatically
    logger.info("Step 3: Running Strategy Backtest Engine...")
    run_backtest(
        lab_path=LAB_PATH,
        start=datetime.combine(start_dt, datetime.min.time()),
        end=datetime.combine(end_dt, datetime.max.time()),
        interval=Interval.MINUTE,
        signal_name="v7_rolling_backtest_signals",
        strategy_version="v7",
        strategy_setting={},  # keep V7 class defaults; avoid V5-style fallback defaults in run_backtest
        vt_symbols=backtest_vt_symbols,
        use_postgres_selection=False # Already filtered in signal gen
    )
    
    logger.info("Rolling Backtest Workflow Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="Backtest Start Date (Test Period Start)")
    parser.add_argument("--end", type=str, required=True, help="Backtest End Date")
    args = parser.parse_args()
    
    run_rolling_backtest(args.start, args.end)

