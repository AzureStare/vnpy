"""
Script to update live data for paper trading.
Downloads yesterday's daily bars from Polygon to update the dataset.
"""
import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from flagship.paper_trading.config import LAB_PATH
from flagship.scripts.download_backtest_data import download_bars_for_symbols, load_daily_universe

def get_last_trading_day() -> date:
    """
    Get the last trading day.
    Simple logic: Yesterday. 
    (In a real production system, this should check a trading calendar API)
    """
    return date.today() - timedelta(days=1)

def update_live_data(
    lab_path: Path = LAB_PATH,
    lookback_days: int = 5,
    universe_file: Path | None = None
) -> None:
    """
    Download recent data to ensure the lab is up to date for inference.
    
    Args:
        lab_path: Path to AlphaLab directory
        lookback_days: Number of days to look back and download (to cover weekends/holidays)
        universe_file: Optional path to a JSON universe file. If None, it might need to 
                       rely on existing symbols in the lab or Postgres (not fully implemented here).
    """
    end_date = get_last_trading_day()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"[update_live_data] Updating data from {start_date} to {end_date}")
    
    # 1. Determine Universe
    symbols = []
    if universe_file and universe_file.exists():
        symbols = load_daily_universe(universe_file)
        logger.info(f"[update_live_data] Loaded {len(symbols)} symbols from {universe_file}")
    else:
        # Fallback: Load symbols that already exist in the lab
        # This assumes we are trading a fixed set or the set already in the lab is sufficient
        import polars as pl
        # A simple way to get all symbols is to check the file list if structured by symbol,
        # but AlphaLab structure is usually by date/file.
        # Alternatively, query Postgres if available.
        try:
            from flagship.scripts.pg_ticker_db import get_ref_tickers
            tickers = get_ref_tickers(active=True)
            symbols = [t['symbol'] for t in tickers]
            logger.info(f"[update_live_data] Loaded {len(symbols)} active symbols from Postgres")
        except ImportError:
            logger.warning("[update_live_data] Postgres not available and no universe file provided.")
            logger.warning("[update_live_data] Scanning lab directory for existing symbols (this might be slow or inaccurate)...")
            # Minimal fallback: scan a recent parquet file if possible, or fail
            # For now, let's assume we need *some* source.
            pass

    if not symbols:
        logger.error("[update_live_data] No symbols found to update. Aborting.")
        return

    # 2. Download Data
    # Reuse the logic from download_backtest_data
    download_bars_for_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=Interval.DAILY,
        lab_dir=lab_path
    )
    
    logger.info("[update_live_data] Data update complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update live data for paper trading")
    parser.add_argument("--lookback", type=int, default=5, help="Days to look back for download")
    args = parser.parse_args()
    
    update_live_data(lookback_days=args.lookback)

