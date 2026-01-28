"""
Script to update broad market indices (SPY, QQQ) and VIX data daily.
Addresses the requirement for "updating market index data".
"""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from flagship.trading.config import LAB_PATH
from flagship.market_data.polygon_data_tools import download_bars_for_symbols, download_vix_indices

def update_market_indices(
    lab_path: Path = LAB_PATH,
    lookback_days: int = 5
) -> None:
    """
    Download recent data for SPY, QQQ and VIX indices.
    
    Args:
        lab_path: Path to AlphaLab directory
        lookback_days: Number of days to look back
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"[update_market_indices] Updating indices from {start_date} to {end_date}")
    
    # 1. Download VIX and VIX3M
    # Using existing script logic
    try:
        logger.info("[update_market_indices] Downloading VIX indices...")
        download_vix_indices(
            start_date=start_date,
            end_date=end_date,
            lab_dir=lab_path
        )
    except Exception as e:
        logger.error(f"[update_market_indices] Failed to download VIX data: {e}")

    # 2. Download SPY and QQQ
    indices = ["SPY", "QQQ"]
    logger.info(f"[update_market_indices] Downloading {indices}...")
    try:
        download_bars_for_symbols(
            symbols=indices,
            start_date=start_date,
            end_date=end_date,
            interval=Interval.DAILY,
            lab_dir=lab_path
        )
    except Exception as e:
        logger.error(f"[update_market_indices] Failed to download index bars: {e}")
        
    logger.info("[update_market_indices] Index update complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update market indices for paper trading")
    parser.add_argument("--lookback", type=int, default=5, help="Days to look back for download")
    args = parser.parse_args()
    
    update_market_indices(lookback_days=args.lookback)

