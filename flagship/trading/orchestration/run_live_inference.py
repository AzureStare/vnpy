"""
Script to generate trading signals for the current day using the pre-trained LightGBM model.
"""
import argparse
import joblib
import polars as pl
from datetime import date, datetime, timedelta
from pathlib import Path

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from flagship.trading.config import (
    LAB_PATH, LIVE_MODEL_PATH, LIVE_LR_MODEL_PATH, DAILY_SIGNAL_FILE
)
from flagship.trading.signal_blend import blend_lgb_with_lr
from flagship.trading.orchestration.ensure_data_completeness import get_daily_selection_from_postgres
from flagship.factors.alpha_momentum.v7_dataset import FlagshipAlphaMomentumV7Dataset
from flagship.factors.alpha_momentum.v5_dataset import FlagshipAlphaMomentumV5Dataset

def run_live_inference(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    model_path: Path = LIVE_MODEL_PATH,
    lr_model_path: Path = LIVE_LR_MODEL_PATH,
    output_file: Path = DAILY_SIGNAL_FILE,
    strategy_version: str = "v7"
) -> None:
    """
    Generate signals for the specific target date.
    
    Args:
        target_date: The date to generate signals for (usually 'today' or 'yesterday's close' for next open).
                     If None, defaults to yesterday (assuming running before market open).
        lab_path: Path to AlphaLab.
        model_path: Path to the trained .pkl model.
        output_file: Path to save the resulting signal parquet.
        strategy_version: "v5" or "v7".
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    
    logger.info(f"[run_live_inference] Generating signals for target date: {target_date} (Strategy: {strategy_version})")
    logger.info(f"[run_live_inference] Using model: {model_path}")

    # 1. Initialize Lab and Load Model
    lab = AlphaLab(str(lab_path))
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Did you run training?")
    
    logger.info(f"[run_live_inference] Loading model...")
    booster = joblib.load(model_path)
    lr_artifact = None
    if lr_model_path.exists():
        try:
            lr_artifact = joblib.load(lr_model_path)
            logger.info(f"[run_live_inference] Loaded LR meta model: {lr_model_path}")
        except Exception as exc:
            logger.warning(f"[run_live_inference] Failed to load LR meta model: {exc}")
            lr_artifact = None
    
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
    logger.info(f"[run_live_inference] Calculating factors ({strategy_version})...")
    
    if strategy_version == "v5":
        DatasetClass = FlagshipAlphaMomentumV5Dataset
    else:
        DatasetClass = FlagshipAlphaMomentumV7Dataset

    dataset = DatasetClass(
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
    
    import numpy as np  # type: ignore
    
    X = target_df.select(feature_cols).to_pandas()
    lgb_signal = booster.predict(X)

    # Optional: Logistic meta model -> p_up = P(label_excess_5d > 0)
    p_up: np.ndarray | None = None
    if isinstance(lr_artifact, dict):
        pipe = lr_artifact.get("pipeline")
        lr_cols = lr_artifact.get("feature_cols")
        if pipe is not None and isinstance(lr_cols, list) and lr_cols:
            try:
                import pandas as pd  # type: ignore

                lr_df = X.copy()
                lr_df.insert(0, "lgb_signal", lgb_signal)
                missing_lr = [c for c in lr_cols if c not in lr_df.columns]
                if missing_lr:
                    raise RuntimeError(f"missing LR features: {missing_lr[:10]}")
                proba = pipe.predict_proba(lr_df[lr_cols])  # type: ignore[attr-defined]
                if proba is not None and len(proba) == len(lgb_signal):
                    p_up = np.asarray(proba)[:, 1].astype(float)
            except Exception as exc:
                logger.warning(f"[run_live_inference] LR predict_proba failed, skip: {exc}")
                p_up = None

    # Final signal: soft blend inside Top-M of lgb_signal
    final_signal = np.asarray(lgb_signal, dtype=float)
    if p_up is not None and len(p_up) == len(final_signal):
        top_n = 5 if strategy_version == "v5" else 8
        final_signal = blend_lgb_with_lr(
            final_signal,
            p_up,
            top_n=int(top_n),
            top_m_multiplier=5,
            alpha=0.5,
            std_floor=1e-6,
        )
    
    # 5. Export Signal
    # We need to include 'atr_14' and other cols for the strategy
    select_cols = ["datetime", "vt_symbol", "atr_14", "close_price", "adv_usd"]
    
    # Add strategy-specific optional columns if they exist
    potential_cols = ["ema5", "ema10", "ema20", "ema50", "atr_percent", "alpha_mom", "alpha_vwap", "alpha_trend", "rs_score", "rs_10d", "beta", "return_5d"]
    for col in potential_cols:
        if col in target_df.columns:
            select_cols.append(col)
            
    p_up_list: list[float | None]
    if p_up is None:
        p_up_list = [None for _ in range(len(final_signal))]
    else:
        p_up_list = [float(x) if np.isfinite(x) else None for x in p_up.tolist()]

    signal_df = target_df.select(select_cols).with_columns(
        [
            pl.Series(name="lgb_signal", values=[float(x) for x in lgb_signal.tolist()]),
            pl.Series(name="p_up", values=p_up_list),
            pl.Series(name="signal", values=[float(x) for x in final_signal.tolist()]),
        ]
    )
    
    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(output_file)
    
    logger.info(f"[run_live_inference] Saved {len(signal_df)} signals to {output_file}")

    # 6. Save Top 50 Ranking to Postgres daily_ranking_history
    try:
        from flagship.universe.pg_ticker_db import get_pg_connection, get_ticker_market_caps_batch
        top_50 = signal_df.sort("signal", descending=True).head(50)
        
        # Batch fetch market caps for these 50 symbols
        symbols_only = [s.split(".")[0] for s in top_50["vt_symbol"].to_list()]
        market_caps = get_ticker_market_caps_batch(symbols_only, target_date)
        
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                # Clear existing for same date
                cur.execute("DELETE FROM daily_ranking_history WHERE trade_date = %s", (target_date,))
                
                rows = []
                for i, row in enumerate(top_50.iter_rows(named=True)):
                    raw_sym = row["vt_symbol"].split(".")[0]
                    mkt_cap = market_caps.get(raw_sym)
                    
                    rows.append((
                        target_date,
                        row["vt_symbol"],
                        float(row["signal"]),
                        i + 1,
                        float(row["close_price"]),
                        mkt_cap,
                        float(row.get("adv_usd") or 0.0)
                    ))
                
                from psycopg2.extras import execute_values
                execute_values(cur, """
                    INSERT INTO daily_ranking_history (trade_date, vt_symbol, signal_score, rank_pos, close_price, market_cap, adv_usd)
                    VALUES %s
                """, rows)
                conn.commit()
        logger.info(f"[run_live_inference] Saved Top 50 rankings to Postgres for {target_date} (incl. MarketCap)")
    except Exception as exc:
        logger.error(f"[run_live_inference] Failed to save rankings to Postgres: {exc}")
    
    logger.info("[run_live_inference] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7", help="Strategy version")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    run_live_inference(target_date=target_dt, strategy_version=args.strategy)
