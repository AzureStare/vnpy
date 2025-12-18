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
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from flagship.paper_trading.config import (
    LAB_PATH, LIVE_MODEL_PATH, DAILY_SIGNAL_FILE, CURRENT_REGIME_ID
)
from flagship.factors.flagship_alpha_momentum_v5 import FlagshipAlphaMomentumV5Dataset
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
    
    logger.info(f"[run_live_inference] Generating signals for target date: {target_date}")
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
    
    # Define a "dummy" test period that covers our target date
    # The dataset class uses these periods to slice data.
    # For inference, we want the "test" segment to include our target date.
    test_period = (
        (target_date - timedelta(days=5)).isoformat(), # small buffer before
        (target_date + timedelta(days=5)).isoformat()  # small buffer after
    )
    
    # Dummy train/valid periods (not used for inference but required by init)
    train_period = (
        (start_date).isoformat(),
        (start_date + timedelta(days=30)).isoformat()
    )
    valid_period = (
        (start_date + timedelta(days=31)).isoformat(),
        (start_date + timedelta(days=60)).isoformat()
    )

    logger.info(f"[run_live_inference] Loading bar data from {start_date}...")
    
    # Detect symbols (load all available or use selection)
    daily_files = sorted(lab.daily_path.glob("*.parquet"))
    vt_symbols = [p.stem for p in daily_files]
    
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=None, # Daily by default in load_bar_df if interval is None/Daily
        start=start_date.isoformat(),
        end=test_period[1],
        extended_days=0
    )
    
    if raw_df is None or raw_df.is_empty():
        raise RuntimeError("Failed to load bar data for inference.")
        
    logger.info(f"[run_live_inference] Loaded {len(raw_df)} rows of bar data.")

    # 3. Compute Factors
    dataset = FlagshipAlphaMomentumV5Dataset(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    logger.info("[run_live_inference] Calculating factors (prepare_data)...")
    dataset.prepare_data(filters=None) # We can apply filters if we have a live daily selection source
    
    logger.info("[run_live_inference] Processing data (process_data)...")
    dataset.process_data()
    
    # 4. Extract Features and Predict
    logger.info("[run_live_inference] Fetching inference data...")
    infer_df = dataset.fetch_infer(Segment.TEST)
    
    # Filter for the specific target date
    # Note: 'datetime' col in polars is usually datetime, so we compare dates
    target_df = infer_df.filter(pl.col("datetime").dt.date() == target_date)
    
    if target_df.is_empty():
        logger.warning(f"[run_live_inference] No data found for date {target_date}. Market might be closed or data missing.")
        # Try finding the latest available date if target date is missing (e.g. if run on weekend)
        last_date = infer_df.select(pl.col("datetime").max()).item()
        if last_date:
            logger.warning(f"[run_live_inference] Falling back to latest available date: {last_date}")
            target_df = infer_df.filter(pl.col("datetime") == last_date)
        else:
             raise RuntimeError("No inference data available.")

    # Feature columns expected by the model
    # We need to get these from the dataset or define them matching training
    # FlagshipAlphaMomentumV5Dataset.feature_cols should be populated after process_data
    feature_cols = dataset.feature_cols
    
    logger.info(f"[run_live_inference] Predicting with {len(feature_cols)} features...")
    
    X = target_df.select(feature_cols).to_pandas()
    scores = booster.predict(X)
    
    # 5. Export Signal
    # We need to include 'atr_14' and other cols for the strategy
    select_cols = ["datetime", "vt_symbol", "atr_14", "close_price"] 
    # Add other potentially useful columns if they exist
    for col in ["z_mom", "z_vwap_residual", "z_trend", "weight_mom", "weight_vwap", "weight_trend"]:
        if col in target_df.columns:
            select_cols.append(col)
            
    signal_df = (
        target_df.select(select_cols)
        .with_columns(pl.Series(name="signal", values=scores)) # 'signal' is the score
        .rename({"signal": "score"}) # Rename to score to match some conventions, or keep as signal. 
                                     # The strategy uses 'signal' column as score usually.
                                     # Let's keep it as 'signal' to match training output
        .rename({"score": "signal"}) 
    )
    
    # Ensure 'atr_14' is present
    if "atr_14" not in signal_df.columns:
        logger.error("[run_live_inference] atr_14 missing from signal dataframe!")
        # Attempt to recover or fail?
        # It should be there if factor calculation worked.
    
    logger.info(f"[run_live_inference] Saving {len(signal_df)} signals to {output_file}")
    
    # Save to parquet
    # Ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(output_file)
    
    logger.info("[run_live_inference] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    run_live_inference(target_date=target_dt)
