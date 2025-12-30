"""
Script to generate trading signals for the current day using the pre-trained LightGBM model.
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
    LAB_PATH, LIVE_MODEL_PATH, DAILY_SIGNAL_FILE, CURRENT_REGIME_ID
)
from flagship.paper_trading.ensure_data_completeness import get_daily_selection_from_postgres
from flagship.factors.flagship_alpha_momentum_v7 import FlagshipAlphaMomentumV7Dataset
from flagship.backtest.index_regime_windows import get_regime_window

def run_live_inference(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    model_path: Path = LIVE_MODEL_PATH,
    output_file: Path = DAILY_SIGNAL_FILE,
    regime_id: int = CURRENT_REGIME_ID
) -> None:
    """
    Generate signals for the specific target date.
    
    Args:
        target_date: The date to generate signals for (usually 'today' or 'yesterday's close' for next open).
                     If None, defaults to yesterday (assuming running before market open).
        lab_path: Path to AlphaLab.
        model_path: Path to the trained .pkl model.
        output_file: Path to save the resulting signal parquet.
        regime_id: ID of the regime to use for config/dataset prep.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    
    logger.info(f"[run_live_inference] Generating signals for target date: {target_date} (V7.0 Aggressive)")
    logger.info(f"[run_live_inference] Using model: {model_path}")
    logger.info(f"[run_live_inference] Regime ID: {regime_id}")

    # 1. Initialize Lab and Load Model
    lab = AlphaLab(str(lab_path))
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Did you run training?")
    
    logger.info(f"[run_live_inference] Loading model...")
    booster = joblib.load(model_path)
    
    # 2. Prepare Dataset for Inference
    # We need to construct the dataset object to calculate factors.
    # We need enough history for factors (e.g., ~120 days).
    start_date = target_date - timedelta(days=180) 
    
    # Define periods (train/valid not used for inference but required by init)
    train_period = (start_date.isoformat(), (start_date + timedelta(days=30)).isoformat())
    valid_period = ((start_date + timedelta(days=31)).isoformat(), (start_date + timedelta(days=60)).isoformat())
    # For inference, ensure TEST covers the whole window including target_date
    test_period = (start_date.isoformat(), target_date.isoformat())

    logger.info(f"[run_live_inference] Loading bar data from {start_date}...")
    
    # Preferred: use Postgres daily_selection universe for target_date (static filtered universe U_t)
    vt_symbols = get_daily_selection_from_postgres(target_date)
    if not vt_symbols:
        logger.warning(f"[run_live_inference] No daily_selection for {target_date}, falling back to lab symbols.")
        daily_files = sorted(lab.daily_path.glob("*.parquet"))
        vt_symbols = [p.stem for p in daily_files]
    
    # 确保包含 SPY 用于计算 RS/Beta
    if "SPY.NASDAQ" not in vt_symbols:
        vt_symbols.append("SPY.NASDAQ")

    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start_date.isoformat(),
        end=target_date.isoformat(),
        extended_days=0
    )
    
    if raw_df is None or raw_df.is_empty():
        raise RuntimeError("Failed to load bar data for inference.")
        
    logger.info(f"[run_live_inference] Loaded {len(raw_df)} rows of bar data.")

    # 3. Compute Factors
    logger.info("[run_live_inference] Calculating factors (V7)...")
    dataset = FlagshipAlphaMomentumV7Dataset(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    logger.info("[run_live_inference] Calculating factors (prepare_data)...")
    dataset.prepare_data(filters=None) 
    
    logger.info("[run_live_inference] Processing data (process_data)...")
    dataset.process_data()
    
    # 4. Extract Features and Predict
    logger.info("[run_live_inference] Fetching inference data...")
    infer_df = dataset.fetch_infer(Segment.TEST)
    
    # Filter for the specific target date
    target_df = infer_df.filter(pl.col("datetime").dt.date() == target_date)
    
    if target_df.is_empty():
        logger.warning(f"[run_live_inference] No data found for date {target_date}. Market might be closed or data missing.")
        # Try finding the latest available date if target date is missing
        last_date = infer_df.select(pl.col("datetime").max()).item()
        if last_date:
            logger.warning(f"[run_live_inference] Falling back to latest available date: {last_date}")
            target_df = infer_df.filter(pl.col("datetime") == last_date)
        else:
             raise RuntimeError("No inference data available.")

    # Feature columns expected by the model
    feature_cols = booster.feature_name()
    missing_features = [c for c in feature_cols if c not in target_df.columns]
    if missing_features:
        raise RuntimeError(f"[run_live_inference] Missing features in inference df: {missing_features[:20]}")
    
    logger.info(f"[run_live_inference] Predicting with {len(feature_cols)} features...")
    
    X = target_df.select(feature_cols).to_pandas()
    scores = booster.predict(X)
    
    # 5. Export Signal
    # We need to include 'atr_14' and other cols for the strategy
    select_cols = ["datetime", "vt_symbol", "atr_14", "close_price", "ema5", "ema10", "ema20", "ema50", "atr_percent"] 
    # Add other potentially useful columns if they exist
    for col in ["alpha_mom", "alpha_vwap", "alpha_trend", "rs_60d", "beta"]:
        if col in target_df.columns:
            select_cols.append(col)
            
    signal_df = (
        target_df.select(select_cols)
        .with_columns(pl.Series(name="signal", values=scores)) 
    )
    
    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(output_file)
    
    logger.info(f"[run_live_inference] Saved {len(signal_df)} signals to {output_file}")
    logger.info("[run_live_inference] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    run_live_inference(target_date=target_dt)
