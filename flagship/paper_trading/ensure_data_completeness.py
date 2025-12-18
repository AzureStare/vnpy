"""
Script to ensure data completeness for selected stocks.
Checks if required history exists in AlphaLab, and downloads missing data if needed.
"""
import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from flagship.paper_trading.config import LAB_PATH
from flagship.scripts.download_backtest_data import download_bars_for_symbols
from flagship.scripts.pg_ticker_db import get_pg_connection

def get_daily_selection_from_postgres(trade_date: date) -> list[str]:
    """Fetch symbols selected for a specific date from Postgres"""
    symbols = []
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT vt_symbol FROM daily_selection WHERE trade_date = %s",
                (trade_date,)
            )
            rows = cur.fetchall()
            symbols = [row[0] for row in rows]
    return symbols

def check_and_backfill_data(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    lookback_days: int = 180
) -> None:
    """
    Verify data existence for selected symbols.
    
    Args:
        target_date: The trading date for which we have a selection.
        lab_path: Path to AlphaLab.
        lookback_days: Required history length.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
        
    logger.info(f"[ensure_data_completeness] Checking data for {target_date}")
    
    # 1. Get Universe
    symbols = get_daily_selection_from_postgres(target_date)
    if not symbols:
        logger.warning(f"[ensure_data_completeness] No selection found in Postgres for {target_date}")
        return

    logger.info(f"[ensure_data_completeness] Checking {len(symbols)} symbols...")
    
    # 2. Check Lab Data
    lab = AlphaLab(str(lab_path))
    missing_symbols = []
    
    start_check_date = target_date - timedelta(days=lookback_days)
    
    for symbol in symbols:
        # Check if file exists
        # AlphaLab usually stores as {symbol}.parquet in daily_path
        # Note: symbol in postgres might differ slightly from file name? 
        # Usually vnpy uses vt_symbol (Symbol.Exchange). 
        # Lab files are usually just Symbol.parquet if from Polygon download script?
        # Let's check the file pattern.
        # download_backtest_data saves as BarData list, lab.save_bar_data saves to parquet.
        # AlphaLab.save_bar_data uses vt_symbol as filename usually? 
        # Actually standard AlphaLab saves by daily/minute folders or single file per symbol?
        # Standard AlphaLab typically: data/daily/AAPL.NASDAQ.parquet
        
        # We need to handle potential naming mismatches. 
        # Postgres stores 'vt_symbol'. File usually matches.
        file_path = lab.daily_path / f"{symbol}.parquet"
        
        if not file_path.exists():
            missing_symbols.append(symbol)
            continue
            
        # Optional: Check if file covers the required range
        # This is expensive (reading every parquet). 
        # Optimization: Only check modification time or rely on 'download' if we suspect gaps.
        # For robustness, let's read the last date.
        try:
            # Only read the last row to check freshness
            # Polars scan_parquet is lazy
            lf = pl.scan_parquet(file_path)
            last_dt = lf.select(pl.col("datetime").max()).collect().item()
            
            if last_dt is None:
                missing_symbols.append(symbol)
                continue
                
            last_date = last_dt.date()
            if last_date < target_date:
                # Data is stale
                missing_symbols.append(symbol)
        except Exception:
            missing_symbols.append(symbol)

    if missing_symbols:
        logger.warning(f"[ensure_data_completeness] Found {len(missing_symbols)} symbols with missing/stale data.")
        logger.info(f"[ensure_data_completeness] Triggering download for missing symbols...")
        
        # Remove exchange suffix for download script (it expects pure symbol usually?)
        # download_bars_for_symbols takes list[str]. 
        # Inside it calls get_exchange_for_symbol.
        # If we pass "AAPL.NASDAQ", get_exchange might fail if it expects "AAPL".
        # Let's strip suffix just in case.
        clean_symbols = [s.split('.')[0] for s in missing_symbols]
        
        download_bars_for_symbols(
            symbols=clean_symbols,
            start_date=start_check_date,
            end_date=target_date,
            interval=Interval.DAILY,
            lab_dir=lab_path
        )
        logger.info("[ensure_data_completeness] Backfill complete.")
    else:
        logger.info("[ensure_data_completeness] All data is present and up-to-date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure data completeness")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    check_and_backfill_data(target_date=target_dt)

