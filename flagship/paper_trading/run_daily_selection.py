"""
Script to automate daily stock selection ($U_t$) based on strategy rules.
Wraps the logic of `build_daily_selection_to_postgres.py` for a single day.
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
from vnpy.alpha import AlphaLab
from flagship.paper_trading.config import LAB_PATH
from flagship.scripts.build_daily_selection_to_postgres import (
    create_daily_selection_table,
    filter_universe_from_lab,
    save_selection_to_postgres
)

def run_daily_selection(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH
) -> None:
    """
    Run the universe selection logic for a specific date and save to Postgres.
    
    Args:
        target_date: The date to run selection for. Defaults to yesterday (last complete trading day).
        lab_path: Path to AlphaLab.
    """
    if target_date is None:
        # Default to yesterday because we need full daily bars to calculate ADV/MedianVolume correctly
        target_date = date.today() - timedelta(days=1)
        
    logger.info(f"[run_daily_selection] Running selection for {target_date}")
    
    # 1. Ensure Table Exists
    create_daily_selection_table()
    
    # 2. Init Lab
    lab = AlphaLab(str(lab_path))
    
    # 3. Filter Universe
    # Criteria: ADV >= $2.5M, Price $20-$600 (from build_daily_selection_to_postgres defaults)
    # Note: filter_universe_from_lab relies on what's currently in the lab.
    # It assumes data up to target_date is available (which should be handled by update_live_data or completeness check).
    try:
        selections = filter_universe_from_lab(
            lab=lab,
            trade_date=target_date,
            min_adv_usd=2.5e8,  # $250M ADV as per original script default
            min_price=20.0,
            max_price=600.0
        )
        
        if selections:
            # 4. Save to Postgres
            save_selection_to_postgres(target_date, selections)
            logger.info(f"[run_daily_selection] Saved {len(selections)} symbols to Postgres for {target_date}")
        else:
            logger.warning(f"[run_daily_selection] No symbols met the criteria for {target_date}")
            
    except Exception as e:
        logger.error(f"[run_daily_selection] Failed to run selection: {e}")
        # Don't exit with error if it's just "data missing" for a holiday, but log it clearly
        # For automation, we might want to raise if it's a critical failure
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock selection")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    run_daily_selection(target_date=target_dt)

